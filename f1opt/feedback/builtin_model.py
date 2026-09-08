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

__all__ = ["BuiltinTinyModel", "get_builtin_model", "reset_builtin_model_cache"]

#: 输出标记 (UI 审计据此断言「内置小模型确实回答了」)。
BUILTIN_MARKER = "【内置小模型】"
BUILTIN_MODEL_VERSION = "builtin-tiny-v1"

_STATE_FILENAME = "builtin_llm_state.json"
_MAX_SAMPLES = 500
_WEIGHT_MIN, _WEIGHT_MAX = 0.5, 2.0

# 子意图 → 字段修正表: (field, 基准幅度, 中文理由)。方向遵循标准调校经验:
# 推头 → 减前翼/软前ARB/增后翼; 甩尾 → 增后翼/软后ARB; 打滑 → 低油门差速锁等。
_ADJUSTMENT_TABLE: dict[str, list[tuple[str, float, str]]] = {
    "understeer": [
        ("front_wing", -1.0, "减小前轴下压力, 改善入弯转向响应"),
        ("front_arb", -2.0, "调软前防倾杆, 让前轴更多机械抓地"),
        ("rear_wing", 1.0, "小幅增加后翼, 把平衡往后移"),
    ],
    "oversteer": [
        ("rear_wing", 1.0, "增加后轴下压力, 稳住车尾"),
        ("rear_arb", -2.0, "调软后防倾杆, 减少尾部的瞬间释放"),
        ("front_wing", 1.0, "微增前翼维持整体平衡"),
    ],
    "traction": [
        ("on_throttle_diff", -5.0, "降低油门差速锁, 出弯更友好"),
        ("off_throttle_diff", -5.0, "降低滑行差速锁, 入弯更从容"),
    ],
    "brake": [
        ("brake_pressure", -5.0, "降低刹车压力, 减少抱死倾向"),
        ("front_brake_bias", 1.0, "刹车平衡前移, 保住前轮抱死余量"),
    ],
    "tyre_wear": [
        ("front_tyre_pressure", -1.0, "略降前胎压, 缓解前轮磨损速率"),
        ("rear_tyre_pressure", -1.0, "略降后胎压, 平衡左右/前后磨损"),
    ],
    "balance": [
        ("front_arb", -1.0, "微调前防倾杆, 找回前后平衡"),
        ("front_wing", -1.0, "微调前翼, 配合整体平衡"),
    ],
    "ers": [],
    "general": [],
}

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
    "understeer": "推头（转向不足）", "oversteer": "甩尾（转向过度）",
    "traction": "牵引力不足（打滑）", "brake": "制动问题（锁死/距离）",
    "tyre_wear": "轮胎磨损过快", "ers": "ERS 能量管理",
    "balance": "前后平衡", "general": "整体表现",
}


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
    ) -> dict[str, Any]:
        """生成针对性调教建议 (自然语言 + 结构化修正列表)。"""
        try:
            intent = classify_intent(question or "")
            sub = classify_sub_intent(question or "", intent.intent)
            sub_intent = sub.sub_intent if sub.confidence >= 0.5 else "general"
        except Exception:  # noqa: BLE001
            sub_intent = "general"

        table = _ADJUSTMENT_TABLE.get(sub_intent, [])
        key = f"{track_id}|{sub_intent}"
        adjustments: list[dict[str, Any]] = []
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
            adjustments.append(item)

        with self._lock:
            n_same = sum(
                1 for s in self._samples
                if s.get("track") == track_id and s.get("sub_intent") == sub_intent
            )
            total = len(self._samples)

        summary = self._render_summary(sub_intent, track_id, adjustments, n_same, total)
        self.collect(question, sub_intent, track_id, driver_style)
        return {
            "summary": summary,
            "adjustments": adjustments,
            "sub_intent": sub_intent,
            "samples_same_context": n_same,
            "samples_total": total,
            "confidence": round(min(0.95, 0.5 + 0.05 * n_same), 2),
        }

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
