"""内置小模型 (builtin tiny model) — 车手反馈驱动的针对性调教优化。

随 EXE 打包、纯本地、零网络依赖、开箱即用。与 external LLM 后端
(``local``/``openai``) 的区别: 不做自由文本生成, 而是聚焦产品核心闭环 ——
**收集车手反馈 → 识别问题意图 → 给出针对性调教修正 → 随样本积累在线学习**。

结构:

- 意图识别: 复用 :mod:`f1opt.feedback.intent` 的 ``classify_intent`` /
  ``classify_sub_intent`` (understeer/oversteer/tyre_wear/brake/ers/
  traction/balance)。
- 调教映射: 每个子意图 → 有方向的字段修正表 (基准幅度), 经
  :class:`~f1opt.data.setup_schema.SetupField` 的 min/max/step 钳制。
- 在线学习: 按 ``(track_id, sub_intent)`` 维护字段权重 (有界 EMA 式更新,
  :meth:`learn_outcome` 提供闭环钩子, 遥测对比改善信号可接入), 样本
  (问题摘要 + 上下文) 持久化到 ``data_dir/builtin_llm_state.json``。

线程安全: 内部锁保护状态; 输出确定性 (同状态同输入 → 同输出)。
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from f1opt.data.setup_schema import ALL_SETUP_FIELDS
from f1opt.feedback.intent import classify_intent, classify_sub_intent
from f1opt.feedback.registry import (
    ISSUE_REGISTRY, get_adjustments_for_issue, get_sub_intent_for_issue,
    get_coupled_solutions_for_issue,
)

__all__ = ["BuiltinTinyModel", "get_builtin_model", "reset_builtin_model_cache"]

#: 输出标记 (UI 审计据此断言「内置小模型确实回答了」)。
BUILTIN_MARKER = "【内置小模型】"
BUILTIN_MODEL_VERSION = "builtin-tiny-v1"

_STATE_FILENAME = "builtin_llm_state.json"
_MAX_SAMPLES = 500
_WEIGHT_MIN, _WEIGHT_MAX = 0.5, 2.0

# ---------------------------------------------------------------------------
# 动态从 ISSUE_REGISTRY 加载调校表 (physics-grounded)
# ---------------------------------------------------------------------------
def _load_adjustments() -> dict[str, list[tuple[str, float, str]]]:
    """Load adjustment table from the unified physics-grounded registry."""
    table: dict[str, list[tuple[str, float, str]]] = {}
    for phase, issues in ISSUE_REGISTRY.items():
        for issue_id, data in issues.items():
            if data.get("adjustments"):
                table[issue_id] = list(data["adjustments"])
    return table

# 旧硬编码，已改为从 registry 动态加载
_ADJUSTMENT_TABLE: dict[str, list[tuple[str, float, str]]] = _load_adjustments()

_FIELD_LABELS: dict[str, str] = {
    "front_wing": "前翼", "rear_wing": "后翼", "front_arb": "前防倾杆",
    "rear_arb": "后防倾杆", "front_toe": "前束角", "rear_toe": "后束角",
    "on_throttle_diff": "油门差速锁", "off_throttle_diff": "滑行差速锁",
    "brake_pressure": "刹车压力", "front_brake_bias": "刹车平衡",
    "engine_braking": "发动机制动", "ers_deploy_mode": "ERS 部署模式",
    "front_tyre_pressure": "前胎压", "rear_tyre_pressure": "后胎压",
    "front_camber": "前外倾角", "rear_camber": "后外倾角",
    "rear_suspension": "后悬挂",
}

_SUB_INTENT_LABELS: dict[str, str] = {
    "understeer": "推头（转向不足）", "understeer_in": "推头（入弯）", "understeer_apex": "弯中推头", "understeer_out": "出弎推头",
    "oversteer": "甩尾（转向过度）", "oversteer_in": "甩尾（入弎）", "oversteer_apex": "弯中甩尾", "oversteer_out": "出弎甩尾",
    "brake": "制动问题（锁死/距离）", "brake_lock": "刹车锁死", "brake_lift": "刹车收油",
    "traction": "牵引力不足（打滑）", "acceleration": "加速不足",
    "tyre_wear": "轮胎磨损过快",
    "ers": "ERS 能量管理", "fuel_wear": "油耗",
    "balance": "前后平衡", "general": "整体表现",
    "steer_delay": "转向迟钝", "steer_sharp": "转向过于灵敏",
    "steer_delay_apex": "弯中转向迟钝", "steer_sharp_apex": "弯中转向过于灵敏",
    "steer_delay_out": "出弎转向迟钝", "steer_sharp_out": "出弎转向过于灵敏",
    "straight_slow": "直道慢", "lap_slow": "单圈慢",
    "rear_slip": "车尾易打滑", "rear_slip_out": "出弎车尾易打滑",
}

#: Opt-BUILTIN-02: 问题关键词直判 (优先于 classify_sub_intent)。
#: classify_sub_intent 按 intent 类别过滤模式 —— "怎么调" 触发 setup_advice
#: 时 understeer 等问题模式不参与匹配, 车手最典型的 "推头怎么调" 会被判
#: 成 general 而丢失针对性修正。直判层不受 intent 类别限制。
_PROBLEM_KEYWORDS: list[tuple[str, re.Pattern[str]]] = [
    ("understeer", re.compile(r"推头|转向不足|understeer|under.?steer", re.I)),
    ("oversteer", re.compile(r"甩尾|转向过度|车尾不稳|oversteer|over.?steer", re.I)),
    ("brake", re.compile(r"锁死|抱死|刹不住|制动距离|brake|lock.?up", re.I)),
    ("tyre_wear", re.compile(r"磨胎|磨损|胎耗|掉胎|tyre.?wear|tire.?wear|degradation", re.I)),
    ("traction", re.compile(r"打滑|牵引力|驱动轮|wheelspin|traction", re.I)),
    ("ers", re.compile(r"\bers\b|能量管理|电量|deploy", re.I)),
    ("balance", re.compile(r"前后平衡|整体平衡|balance", re.I)),
]


def _detect_sub_intent(question: str | None) -> str:
    """问题 → 子意图: 关键词直判优先, classify_sub_intent 兜底。"""
    q = question or ""
    for name, pat in _PROBLEM_KEYWORDS:
        if pat.search(q):
            return name
    try:
        intent = classify_intent(q)
        sub = classify_sub_intent(q, intent.intent)
        if sub.confidence >= 0.5:
            return sub.sub_intent
    except Exception:  # noqa: BLE001
        pass
    return "general"


def _field_spec(field: str) -> Any:
    for spec in ALL_SETUP_FIELDS():
        if spec.name == field:
            return spec
    return None


def _clamp_delta(field: str, delta: float) -> float:
    """把基准幅度按字段 step 取整, 并保证至少一个 step (为 0 则跳过)。"""
    spec = _field_spec(field)
    if spec is None or spec.step <= 0:
        return round(delta, 2)
    steps = delta / spec.step
    rounded = round(steps)
    if rounded == 0:
        rounded = 1 if steps > 0 else (-1 if steps < 0 else 0)
    value = rounded * spec.step
    return round(value, _step_decimals(spec.step))


def _step_decimals(step: float) -> int:
    if step >= 1.0:
        return 0
    return max(0, -int(__import__("math").floor(__import__("math").log10(step))))


class BuiltinTinyModel:
    """内置小模型: 反馈收集 + 针对性调教修正 + 在线学习。"""

    def __init__(self, data_dir: str | Path = "data_store") -> None:
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / _STATE_FILENAME
        self._lock = threading.Lock()
        self._samples: list[dict[str, Any]] = []
        self._weights: dict[str, dict[str, float]] = {}
        self._outcomes = 0
        self._load()

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._samples = list(raw.get("samples", []))[:_MAX_SAMPLES]
            self._weights = {
                k: {f: float(w) for f, w in v.items()}
                for k, v in raw.get("weights", {}).items()
            }
            self._outcomes = int(raw.get("outcomes", 0))
        except Exception:  # noqa: BLE001 — 损坏/缺失时从零开始
            self._samples, self._weights, self._outcomes = [], {}, 0

    def _save(self) -> None:
        try:
            self._path.write_text(
                json.dumps(
                    {
                        "version": BUILTIN_MODEL_VERSION,
                        "samples": self._samples,
                        "weights": self._weights,
                        "outcomes": self._outcomes,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001 — 持久化失败不影响本次建议
            pass

    # ------------------------------------------------------------------ #
    # 收集 + 学习
    # ------------------------------------------------------------------ #
    def collect(
        self,
        question: str | None,
        sub_intent: str,
        track_id: str,
        driver_style: str | None = None,
    ) -> None:
        """收集一条车手反馈样本 (每次针对性建议都会记录)。"""
        with self._lock:
            self._samples.append(
                {
                    "ts": round(time.time(), 3),
                    "track": track_id,
                    "sub_intent": sub_intent,
                    "driver": driver_style or "default",
                    "q": (question or "")[:80],
                }
            )
            if len(self._samples) > _MAX_SAMPLES:
                self._samples = self._samples[-_MAX_SAMPLES:]
            self._save()

    def learn_outcome(
        self, track_id: str, sub_intent: str, field: str, improved: bool
    ) -> None:
        """闭环钩子: 一次修正遥测对比改善 (True) 或变差 (False) 后更新权重。"""
        key = f"{track_id}|{sub_intent}"
        with self._lock:
            w = self._weights.setdefault(key, {}).setdefault(field, 1.0)
            w = w * (1.10 if improved else 0.90)
            self._weights[key][field] = min(_WEIGHT_MAX, max(_WEIGHT_MIN, w))
            self._outcomes += 1
            self._save()

    # ------------------------------------------------------------------ #
    # 针对性建议
    # ------------------------------------------------------------------ #
    def _weight(self, key: str, field: str) -> float:
        return float(self._weights.get(key, {}).get(field, 1.0))

    def suggest(
        self,
        question: str | None,
        track_id: str,
        setup: dict[str, Any] | None = None,
        driver_style: str | None = None,
        issue_ids: list[str] | None = None,
        corner_id: str | None = None,
    ) -> dict[str, Any]:
        """生成针对性调教建议。

        优先使用 issue_ids（来自赛道图点击），此时 question 参数可选。
        若无 issue_ids，则退化为文本意图识别（兼容旧路径）。

        Args:
            question: 文本反馈 (旧路径 fallback)。
            track_id: 当前赛道。
            setup: 当前调教字典。
            issue_ids: 前端赛道图点选的 issue_id 列表，如 ["understeer_in", "brake_lock"]。
            corner_id: 弯道编号，如 "T3"。
        """
        # 1. 确定输入 issue_ids
        if not issue_ids:
            issue_ids = []
        if not issue_ids and question:
            # 兼容旧文本路径：从文本推断 issue_ids
            # 文本模式没有 phase 信息 → 默认映射到 "入弯" (turn-in)
            # 这是合理的，因为 90% 的车手反馈集中在入弯阶段
            sub_intent = _detect_sub_intent(question)
            if sub_intent != "general":
                # 纯 sub_intent → 默认取 turn-in phase 的对应 issue
                default_phase_map = {
                    "understeer": "understeer_in",
                    "oversteer": "oversteer_in",
                    "brake": "brake_lock",
                    "traction": "acceleration",  # 出弎牵引差
                    "tyre_wear": "tire_wear",
                    "ers": "fuel_wear",
                    "balance": "balance",
                }
                issue_ids = [default_phase_map.get(sub_intent, sub_intent)]

        if not issue_ids:
            return {
                "summary": f"{BUILTIN_MARKER}未选择任何反馈问题，当前调教保持不变。",
                "adjustments": [],
                "sub_intent": "general",
                "samples_same_context": 0,
                "samples_total": len(self._samples),
                "confidence": 0.0,
            }

        # 2. 逐 issue 生成修正
        all_adjustments: list[dict[str, Any]] = []
        all_warnings: list[str] = []
        for issue_id in issue_ids:
            table = _ADJUSTMENT_TABLE.get(issue_id, [])
            key = f"{track_id}|{issue_id}"
            for field, base, reason in table:
                spec = _field_spec(field)
                if spec is None:
                    continue
                delta = _clamp_delta(field, base * self._weight(key, field))
                if delta == 0:
                    continue
                item: dict[str, Any] = {
                    "field": field,
                    "label": _FIELD_LABELS.get(field, field),
                    "delta": delta,
                    "unit": spec.unit,
                    "reason": reason,
                    "issue_id": issue_id,
                    "corner_id": corner_id,
                }
                if setup and field in setup:
                    try:
                        current = float(setup[field])
                        new_val = round(current + delta, _step_decimals(spec.step))
                        new_val = min(spec.max, max(spec.min, new_val))
                        item["from"] = current
                        item["to"] = new_val
                    except (TypeError, ValueError):
                        pass
                all_adjustments.append(item)

        # 3. 合并相同字段的修正 (同字段多次修正 → 取最后一个或警告)
        field_count: dict[str, int] = {}
        for adj in all_adjustments:
            f = adj["field"]
            field_count[f] = field_count.get(f, 0) + 1

        # 4. 统计与摘要
        with self._lock:
            n_same = sum(
                1 for s in self._samples
                if s.get("track") == track_id
                and any(i in (issue_ids or []) for i in [s.get("sub_intent", "")])
            )
            total = len(self._samples)

        coupled_solutions = get_coupled_solutions_for_issue(issue_ids[0]) if issue_ids else []
        summary = self._render_summary_items(issue_ids, track_id, all_adjustments, n_same, total, coupled_solutions)

        for iid in issue_ids:
            self.collect(question or iid, iid, track_id, driver_style)

        return {
            "summary": summary,
            "adjustments": all_adjustments,
            "sub_intent": get_sub_intent_for_issue(issue_ids[0]) or "general",
            "issue_ids": issue_ids,
            "phase": issue_ids[0].rsplit("_", 1)[0] if "_" in issue_ids[0] else issue_ids[0],
            "corner_id": corner_id,
            "samples_same_context": n_same,
            "samples_total": total,
            "confidence": round(min(0.95, 0.5 + 0.05 * n_same), 2),
            "warnings": all_warnings,
            "coupled_solutions": coupled_solutions,
        }

    def _render_summary_items(
        self,
        issue_ids: list[str],
        track_id: str,
        adjustments: list[dict[str, Any]],
        n_same: int,
        total: int,
        coupled_solutions: list[dict[str, Any]] | None = None,
    ) -> str:
        """渲染多 issue 修正摘要（赛道图点击模式）。"""
        if not adjustments:
            labels = "、".join(
                _SUB_INTENT_LABELS.get(i, i) for i in issue_ids
            )
            return (
                f"{BUILTIN_MARKER}已收集你的反馈（{labels}，赛道 {track_id}）。"
                f"该问题暂无可靠的定向修正规则，建议先跑 2-3 圈收集遥测。"
                f"目前经验池 {total} 条。"
            )
        # 按 issue_id 分组展示
        lines = [
            f"{BUILTIN_MARKER}针对你选择的反馈（赛道 {track_id}，"
            f"经验池 {total} 条），给出以下针对性调教修正："
        ]
        seen: dict[str, int] = {}
        for adj in adjustments:
            iid = adj.get("issue_id", "")
            corner = adj.get("corner_id", "")
            prefix = f"[{corner}] " if corner else ""
            label = _SUB_INTENT_LABELS.get(iid, iid)
            # 同一 issue 首次出现时加小标题
            if iid not in seen:
                lines.append(f"· {label}：")
                seen[iid] = 1
            if "from" in adj:
                change = f"{adj['from']:g} → {adj['to']:g}（{adj['delta']:+g} {adj['unit']}）"
            else:
                change = f"{adj['delta']:+g} {adj['unit']}"
            lines.append(f"  {prefix}{adj['label']}：{change} —— {adj['reason']}")
        # 整体调校方案（耦合组协同，而非单参数）
        if coupled_solutions:
            lines.append("")
            lines.append(
                f"▸ 以上为默认综合方案。针对该问题，提供 {len(coupled_solutions)} 套"
                f"跨参数整体方案（按耦合组协同调校，可择一或组合）："
            )
            for i, sol in enumerate(coupled_solutions, 1):
                groups = "、".join(sol.get("coupling_groups", []))
                params = sol.get("params", {})
                # 展平 params：兼容 (delta,reason) 元组与纯数值
                flat = []
                for k, v in params.items():
                    delta = v[0] if isinstance(v, (list, tuple)) else v
                    flat.append(f"{k}{'+' if delta >= 0 else ''}{delta:g}")
                lines.append(
                    f"  方案{i}【{sol.get('name','')}】(组:{groups}): {', '.join(flat)}"
                )
                lines.append(f"      {sol.get('summary','')}")
                lines.append(f"      代价: {sol.get('tradeoff','—')}")
        lines.append("")
        lines.append(
            "改完后请跑一圈对比；若该方向有效我会加权，无效则自动反向学习。"
        )
        return "\n".join(lines)

    def _render_summary(
        self,
        sub_intent: str,
        track_id: str,
        adjustments: list[dict[str, Any]],
        n_same: int,
        total: int,
    ) -> str:
        label = _SUB_INTENT_LABELS.get(sub_intent, sub_intent)
        if not adjustments:
            return (
                f"{BUILTIN_MARKER}已收集你的反馈（{label}，赛道 {track_id}）。"
                f"该问题暂无可靠的定向修正规则，建议先跑 2-3 圈收集遥测，"
                f"我会结合圈速变化继续学习。目前经验池 {total} 条。"
            )
        lines = [
            f"{BUILTIN_MARKER}针对{label}，赛道 {track_id}，"
            f"基于已积累的 {n_same} 条同类反馈（经验池共 {total} 条），"
            "给出以下针对性调教修正："
        ]
        for i, adj in enumerate(adjustments, 1):
            if "from" in adj:
                change = f"{adj['from']:g} → {adj['to']:g}（{adj['delta']:+g} {adj['unit']}）"
            else:
                change = f"{adj['delta']:+g} {adj['unit']}"
            lines.append(f"{i}. {adj['label']}：{change} —— {adj['reason']}")
        lines.append(
            "改完后请跑一圈对比；若该方向有效我会加权，无效则自动反向学习。"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": BUILTIN_MODEL_VERSION,
                "samples_total": len(self._samples),
                "outcomes": self._outcomes,
                "weight_keys": len(self._weights),
            }


# --------------------------------------------------------------------------- #
# 模块级单例 (引擎复用, 避免每次反馈重复读盘)
# --------------------------------------------------------------------------- #
_BUILTIN_MODEL: BuiltinTinyModel | None = None
_BUILTIN_LOCK = threading.Lock()


def get_builtin_model(data_dir: str | Path = "data_store") -> BuiltinTinyModel:
    global _BUILTIN_MODEL
    with _BUILTIN_LOCK:
        if _BUILTIN_MODEL is None:
            _BUILTIN_MODEL = BuiltinTinyModel(data_dir)
        return _BUILTIN_MODEL


def reset_builtin_model_cache() -> None:
    global _BUILTIN_MODEL
    with _BUILTIN_LOCK:
        _BUILTIN_MODEL = None
