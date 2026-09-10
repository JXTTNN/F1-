"""会话分析器 (Opt-LOOP-01) — 车手反馈 × 实测遥测 → 针对性新调教。

串起产品主闭环::

    选赛道 → 游戏跑圈 (UDP 60Hz 遥测 + 调教自动采集) → 回应用输入反馈
    → [开始分析] → 弯道证据 + 实测残差 + 定向搜索 → 新调教 → 再跑再反馈

三个组件:

- :class:`ServerFrameBuffer` — 60Hz 统一帧环形缓存 (线程安全, ~1 分钟窗口),
  供弯道级证据切片 (速度/油门/刹车/转向)。
- :func:`analyze_session` — 主分析: 反馈意图 → 弯道证据 → 以玩家当前调教
  为起点的**约束式定向搜索** (只动反馈相关字段, 目标 = DNN 预测 + 实测
  残差修正 :func:`~f1opt.model.online_correction.corrected_lap_time`) →
  完整新调教 + 逐字段改动理由。
- :class:`LearningLoop` — 迭代闭环账本: 每次分析记录 pending 修正基线;
  下次分析时用之后实测最快圈对比, 自动回填
  :meth:`~f1opt.feedback.builtin_model.BuiltinTinyModel.learn_outcome`
  (改善→加权, 恶化→反向), 持久化随 builtin state。

全部纯本地、毫秒级, 与 builtin 小模型共享学习状态。
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from f1opt.data.setup_schema import ALL_SETUP_FIELDS, CarSetup
from f1opt.data.tracks import ALL_TRACKS
from f1opt.feedback.builtin_model import (
    BUILTIN_MARKER,
    _ADJUSTMENT_TABLE,
    _detect_sub_intent,
)
from f1opt.feedback.engine import _PROBLEM_TO_SETUP_HINT

__all__ = [
    "ServerFrameBuffer",
    "LearningLoop",
    "analyze_session",
    "extract_corner_numbers",
]

_STATE_FILENAME = "learning_loop_state.json"
_FRAME_BUFFER_CAPACITY = 4000  # ~60s @ 60Hz
_CANDIDATE_STEPS = (1, 2, 3)  # 每字段搜索半径 (x step)

#: 弯号提取: "T1/T5" "1号弯" "5 号弯" "turn 3" "弯 2" — 全部转成 int 列表。
_CORNER_PATTERNS = [
    re.compile(r"\bT(\d{1,2})\b", re.I),
    re.compile(r"(\d{1,2})\s*号弯"),
    re.compile(r"(\d{1,2})\s*号 ?弯"),
    re.compile(r"\bturn\s*(\d{1,2})\b", re.I),
    re.compile(r"弯\s*(\d{1,2})"),
]

_SUB_INTENT_LABELS = {
    "understeer": "推头（转向不足）", "oversteer": "甩尾（转向过度）",
    "traction": "牵引力不足（打滑）", "brake": "制动问题（锁死/距离）",
    "tyre_wear": "轮胎磨损过快", "ers": "ERS 能量管理",
    "balance": "前后平衡", "general": "整体表现",
}


def extract_corner_numbers(question: str | None) -> list[int]:
    """从反馈文本提取弯号 (T3 / 3号弯 / turn 3 …), 保持出现顺序去重。"""
    if not question:
        return []
    found: list[int] = []
    for pat in _CORNER_PATTERNS:
        for m in pat.finditer(question):
            n = int(m.group(1))
            if 1 <= n <= 40 and n not in found:
                found.append(n)
    return found


class ServerFrameBuffer:
    """60Hz 统一帧环形缓存 (线程安全)。

    每帧是 aligner 的 ``latest_unified_frame()`` 字典 (含 ``lap_distance`` /
    ``speed`` / ``throttle`` / ``brake`` / ``steer``)。容量 4000 帧 ≈ 60s
    @60Hz — 覆盖最近若干个弯, 足够做弯道切片证据; 更久远的历史由
    laps.parquet 承载。
    """

    def __init__(self, capacity: int = _FRAME_BUFFER_CAPACITY) -> None:
        self._frames: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def append(self, frame: dict[str, Any]) -> None:
        with self._lock:
            self._frames.append(dict(frame))

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._frames)

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)


class LearningLoop:
    """迭代闭环账本: pending 修正 → 实测圈速对比 → 自动回填学习。

    状态持久化到 ``data_dir/learning_loop_state.json`` (随 builtin state
    一起落盘), EXE 重启后闭环不断。
    """

    def __init__(self, data_dir: str | Path) -> None:
        self._path = Path(data_dir) / _STATE_FILENAME
        self._lock = threading.Lock()
        self.pending: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self.pending = raw.get("pending")
            self.history = list(raw.get("history", []))[-50:]
        except Exception:  # noqa: BLE001 — 缺失/损坏从零开始
            self.pending, self.history = None, []

    def _save(self) -> None:
        try:
            self._path.write_text(
                json.dumps(
                    {"pending": self.pending, "history": self.history},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass

    def set_pending(
        self,
        track_id: str,
        sub_intent: str,
        changes: list[dict[str, Any]],
        baseline_best_lap: float | None,
    ) -> None:
        with self._lock:
            self.pending = {
                "track_id": track_id,
                "sub_intent": sub_intent,
                "fields": [c["field"] for c in changes],
                "baseline_best_lap": baseline_best_lap,
                "ts": time.time(),
            }
            self._save()

    def settle(
        self,
        track_id: str,
        best_lap_since: float | None,
        on_outcome,
    ) -> dict[str, Any] | None:
        """有 pending 且已有新实测圈速时结算: 对比基线 → 回填学习。

        ``on_outcome(track_id, sub_intent, field, improved)`` 由调用方传入
        (通常是 :meth:`BuiltinTinyModel.learn_outcome`)。返回结算摘要;
        无 pending / 无新实测时返回 None。
        """
        with self._lock:
            pending = self.pending
        if not pending or pending.get("track_id") != track_id:
            return None
        baseline = pending.get("baseline_best_lap")
        if baseline is None or best_lap_since is None:
            return None
        improved = best_lap_since < baseline - 1e-6
        for field in pending.get("fields", []):
            try:
                on_outcome(track_id, pending["sub_intent"], field, improved)
            except Exception:  # noqa: BLE001
                pass
        entry = {
            "track_id": track_id,
            "sub_intent": pending.get("sub_intent"),
            "baseline": baseline,
            "best_lap_since": best_lap_since,
            "delta_s": round(best_lap_since - baseline, 3),
            "improved": improved,
            "ts": time.time(),
        }
        with self._lock:
            self.history.append(entry)
            self.pending = None
            self._save()
        return entry


def _track_by_id(track_id: str):
    for t in ALL_TRACKS:
        if t.track_id == track_id:
            return t
    return None


def _corner_evidence(
    frames: list[dict[str, Any]],
    track_id: str,
    corner_numbers: list[int],
) -> list[dict[str, Any]]:
    """按弯号切 lap_distance 窗口, 汇总该弯的速度/油门/刹车/转向证据。

    弯心近似: corner i (1-based) 位于赛道全长的 ``i/(corners+1)`` 处,
    窗口 = ±(length / (corners + 1) / 2)。线性近似对"哪一段慢/滑"的
    定性判断足够; 精确弯心坐标不在遥测数据里。
    """
    track = _track_by_id(track_id)
    if track is None or not frames:
        return []
    length = float(track.length_m)
    n = max(1, int(track.corners))
    window = length / (n + 1) / 2.0
    out: list[dict[str, Any]] = []
    for cn in corner_numbers:
        center = length * cn / (n + 1)
        lo, hi = center - window, center + window
        seg = [
            f for f in frames
            if isinstance(f.get("lap_distance"), (int, float))
            and lo <= f["lap_distance"] <= hi
        ]
        if len(seg) < 10:
            out.append({
                "corner": cn, "samples": len(seg),
                "note": "窗口内样本不足 (刚起步/帧缓存未覆盖该弯)",
            })
            continue
        speeds = [f["speed"] for f in seg if isinstance(f.get("speed"), (int, float))]
        thr = [f["throttle"] for f in seg if isinstance(f.get("throttle"), (int, float))]
        brk = [f["brake"] for f in seg if isinstance(f.get("brake"), (int, float))]
        st = [abs(f["steer"]) for f in seg if isinstance(f.get("steer"), (int, float))]
        if not speeds:
            out.append({"corner": cn, "samples": len(seg), "note": "无速度数据"})
            continue
        out.append({
            "corner": cn,
            "samples": len(seg),
            "min_speed_kph": round(min(speeds), 1),
            "avg_speed_kph": round(sum(speeds) / len(speeds), 1),
            "avg_throttle": round(sum(thr) / len(thr), 2) if thr else None,
            "avg_brake": round(sum(brk) / len(brk), 2) if brk else None,
            "max_steer_abs": round(max(st), 2) if st else None,
            "window_m": [round(lo), round(hi)],
        })
    return out


def _hint_fields(sub_intent: str) -> list[str]:
    """反馈子意图 → 允许修改的字段集 (engine hint 表 + builtin 修正表并集)。"""
    fields: list[str] = []
    for f in _PROBLEM_TO_SETUP_HINT.get(sub_intent, []):
        if f not in fields:
            fields.append(f)
    for f, _delta, _reason in _ADJUSTMENT_TABLE.get(sub_intent, []):
        if f not in fields:
            fields.append(f)
    return fields


def _target_lap(
    setup: CarSetup, track_id: str, obs_buffer: Any, driver_profile: Any
) -> float:
    """定向搜索目标: DNN 预测 + 实测残差修正 (体现实测学到的偏差)。"""
    try:
        from f1opt.model.online_correction import corrected_lap_time

        return float(
            corrected_lap_time(setup, track_id, driver_profile, obs_buffer)
        )
    except Exception:  # noqa: BLE001 — 无观测时纯 DNN
        from f1opt.model.surrogate import predict_lap_time

        return float(predict_lap_time(setup, track_id, driver_profile))


def _constrained_search(
    base_setup: CarSetup,
    fields: list[str],
    track_id: str,
    obs_buffer: Any,
    driver_profile: Any,
    weights: dict[str, float] | None = None,
) -> tuple[CarSetup, list[dict[str, Any]], float, float]:
    """坐标式定向搜索: 只动 ``fields``, 逐字段贪心 ±(1..3) step。

    返回 (新调教, changes[], 目标改善秒数, 基线目标值)。
    """
    specs = {s.name: s for s in ALL_SETUP_FIELDS()}
    current = base_setup
    best_target = _target_lap(current, track_id, obs_buffer, driver_profile)
    baseline = best_target
    changes: list[dict[str, Any]] = []
    weights = weights or {}
    for rounds in range(2):  # 两轮坐标下降
        improved_any = False
        for field in fields:
            spec = specs.get(field)
            if spec is None or spec.step <= 0:
                continue
            cur_val = float(getattr(current, field))
            w = float(weights.get(field, 1.0))
            best_delta, best_val, best_t = 0.0, cur_val, best_target
            for k in _CANDIDATE_STEPS:
                for sign in (1.0, -1.0):
                    delta = sign * k * spec.step * max(0.5, min(2.0, w))
                    steps = round(delta / spec.step)
                    if steps == 0:
                        continue
                    delta = steps * spec.step
                    new_val = min(spec.max, max(spec.min, cur_val + delta))
                    if new_val == cur_val:
                        continue
                    cand = current.model_copy(update={field: new_val})
                    t = _target_lap(cand, track_id, obs_buffer, driver_profile)
                    if t < best_t - 1e-4:
                        best_t, best_val, best_delta = t, new_val, new_val - cur_val
            if best_delta != 0.0:
                current = current.model_copy(update={field: best_val})
                best_target = best_t
                changes.append({
                    "field": field,
                    "from": cur_val,
                    "to": best_val,
                    "delta": round(best_delta, 3),
                })
                improved_any = True
        if not improved_any:
            break
    return current, changes, baseline, best_target


def analyze_session(
    question: str | None,
    track_id: str,
    current_setup: CarSetup | None,
    obs_buffer: Any | None,
    frame_buffer: ServerFrameBuffer | None,
    builtin_model: Any | None = None,
    driver_profile: Any = None,
) -> dict[str, Any]:
    """主分析入口 — 反馈 × 实测遥测 → 针对性新调教。

    任何数据缺失都优雅降级: 无帧→无弯道证据; 无观测→纯 DNN 目标;
    无当前调教→只给诊断与方向性修正, 不产新调教。
    """
    sub_intent = _detect_sub_intent(question)
    corners = extract_corner_numbers(question)

    frames = frame_buffer.snapshot() if frame_buffer is not None else []
    evidence = _corner_evidence(frames, track_id, corners) if corners else []

    n_obs = len(obs_buffer) if obs_buffer is not None else 0
    observations = None
    best_lap: float | None = None
    residual_mean: float | None = None
    if obs_buffer is not None:
        try:
            observations = obs_buffer.observations_for_track(track_id)
        except Exception:  # noqa: BLE001
            observations = None
        if observations:
            best_lap = min(o.observed_lap for o in observations)
            residual_mean = (
                sum(o.residual for o in observations) / len(observations)
            )

    builtin_summary: str | None = None
    weights: dict[str, float] = {}
    if builtin_model is not None:
        try:
            result = builtin_model.suggest(question, track_id)
            builtin_summary = result["summary"]
            weights = {
                a["field"]: 1.0 + 0.0 * a["delta"] for a in result["adjustments"]
            }
            # 权重从 builtin 的学习状态取 (suggest 未直接返回); 用内部表。
            try:
                key = f"{track_id}|{sub_intent}"
                weights = dict(builtin_model._weights.get(key, {}))
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            builtin_summary = None

    fields = _hint_fields(sub_intent)
    new_setup: CarSetup | None = None
    changes: list[dict[str, Any]] = []
    gain: float | None = None
    if current_setup is not None and fields:
        new_setup, changes, baseline_t, new_t = _constrained_search(
            current_setup, fields, track_id, obs_buffer, driver_profile, weights
        )
        gain = round(baseline_t - new_t, 3)

    lines = [f"{BUILTIN_MARKER}会话分析完成（赛道 {track_id}）。"]
    lines.append(f"问题识别: {_SUB_INTENT_LABELS.get(sub_intent, sub_intent)}")
    if corners:
        lines.append(f"涉及弯道: {', '.join(f'T{c}' for c in corners)}")
    if evidence:
        for ev in evidence:
            if "min_speed_kph" in ev:
                lines.append(
                    f"- T{ev['corner']}: 最低速 {ev['min_speed_kph']} km/h, "
                    f"平均油门 {ev.get('avg_throttle')}, "
                    f"平均刹车 {ev.get('avg_brake')}"
                )
            else:
                lines.append(f"- T{ev['corner']}: {ev.get('note', '样本不足')}")
    if residual_mean is not None:
        lines.append(
            f"实测 vs 模型偏差: {residual_mean:+.3f}s "
            f"(近 {n_obs} 圈, 最快 {best_lap:.3f}s)" if best_lap else
            f"实测 vs 模型偏差: {residual_mean:+.3f}s"
        )
    elif n_obs:
        lines.append(f"已有 {n_obs} 圈实测观测 (本赛道 {len(observations or [])} 圈)")
    else:
        lines.append("暂无实测圈速 —— 先连上游戏跑几圈, 分析会更准")
    if changes:
        lines.append(f"定向搜索给出 {len(changes)} 项修正, 预计收益 {gain:+.3f}s:")
        for c in changes:
            lines.append(f"- {c['field']}: {c['from']:g} → {c['to']:g} ({c['delta']:+g})")
    else:
        lines.append("当前调教在反馈方向上已接近局部最优, 建议先跑圈积累数据。")
    if builtin_summary:
        lines.append(builtin_summary.split("\n", 1)[-1] if "\n" in builtin_summary else builtin_summary)

    return {
        "summary": "\n".join(lines),
        "sub_intent": sub_intent,
        "corners": corners,
        "evidence": evidence,
        "changes": changes,
        "new_setup": new_setup.model_dump() if new_setup is not None else None,
        "expected_gain_s": gain,
        "learning": {
            "samples_total": n_obs,
            "best_lap": best_lap,
            "residual_mean": residual_mean,
        },
    }
