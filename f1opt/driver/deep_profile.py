"""车手深度画像 (deep driver profile) — 在 8 维基础画像之上叠加风格原型、
弯道相位分析、一致性/疲劳/适应建模.

本模块**不修改** :mod:`f1opt.driver.profile`, 而是基于其 ``extract_driver_profile``
输出与原始帧做进一步的高阶分析, 输出可解释的中文洞察与调教建议.

公开 API:

- :class:`DrivingStyleArchetype` — 7 种驾驶风格原型枚举.
- :func:`classify_archetype` — 由风格指标字典判定原型.
- :class:`CornerPhaseAnalysis` / :func:`analyze_corner_phases` — 弯道四相位分析.
- :class:`DriverConsistencyAnalyzer` — 圈速/输入一致性分析.
- :class:`AdaptationProfile` — 条件适应能力建模.
- :class:`FatigueModel` — stint 内疲劳曲线.
- :class:`DeepDriverProfiler` — 组合以上模块的综合画像器.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from f1opt.driver.profile import extract_driver_profile
from f1opt.numeric import clamp01 as _clamp01
from f1opt.numeric import coefficient_of_variation as _cv

# 弯道四相位固定顺序.
_PHASE_ORDER: tuple[str, ...] = (
    "braking",
    "trail_braking",
    "mid_corner",
    "exit",
)

# 一致性归一化饱和常数 (CV 量级 ~2%).
_CONS_K = 0.02
# 疲劳线性退化速率 (s/lap).
_FATIGUE_RATE = 0.05


# --- 通用工具 --------------------------------------------------------------
def _f(frame: dict[str, Any], key: str, default: float = 0.0) -> float:
    """安全取帧字段并转 float; None/不可转换返回 ``default``。"""
    v = frame.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# --- 驾驶风格原型 ----------------------------------------------------------
class DrivingStyleArchetype(Enum):
    """驾驶风格原型 (7 种).

    - ``SMOOTH_OPERATOR`` — 高平顺、低进攻、一致.
    - ``QUALIFIER`` — 高进攻、油门平顺、低一致.
    - ``RACE_CRAFT`` — 均衡、高一致、中等进攻.
    - ``TIRE_WHISPERER`` — 低胎耗、平顺输入.
    - ``AGGRESSIVE_OVERTAKER`` — 高进攻、高 ERS、低平顺.
    - ``DEVELOPMENT`` — 低一致 (新人发育中).
    - ``WET_SPECIALIST`` — 雨战专长 (由湿滑条件指标推导).
    """

    SMOOTH_OPERATOR = "smooth_operator"
    QUALIFIER = "qualifier"
    RACE_CRAFT = "race_craft"
    TIRE_WHISPERER = "tire_whisperer"
    AGGRESSIVE_OVERTAKER = "aggressive_overtaker"
    DEVELOPMENT = "development"
    WET_SPECIALIST = "wet_specialist"


def _opt_float(metrics: dict[str, Any], key: str) -> float | None:
    """从 metrics 安全取可选浮点; 缺失或不可转换返回 None。"""
    if key not in metrics or metrics[key] is None:
        return None
    try:
        return float(metrics[key])
    except (TypeError, ValueError):
        return None


def classify_archetype(metrics: dict[str, Any]) -> DrivingStyleArchetype:
    """由风格指标字典判定驾驶原型.

    ``metrics`` 推荐键: throttle_smoothness, steer_smoothness, aggression_score,
    consistency_score, ers_usage_intensity; 可选键: wet_performance,
    tire_wear_score (两者缺失则跳过对应分支).
    """
    wet = _opt_float(metrics, "wet_performance")
    if wet is not None and wet > 0.75:
        return DrivingStyleArchetype.WET_SPECIALIST

    thr_s = float(metrics.get("throttle_smoothness", 0.0) or 0.0)
    str_s = float(metrics.get("steer_smoothness", 0.0) or 0.0)
    smoothness = (thr_s + str_s) / 2.0
    aggression = float(metrics.get("aggression_score", 0.0) or 0.0)
    consistency = float(metrics.get("consistency_score", 0.0) or 0.0)
    ers = float(metrics.get("ers_usage_intensity", 0.0) or 0.0)

    # 高进攻 + 高 ERS + 低平顺 → 进攻型超车手.
    if aggression > 0.7 and ers > 0.6 and smoothness < 0.5:
        return DrivingStyleArchetype.AGGRESSIVE_OVERTAKER
    # 高进攻 + 低一致 → 排位赛型.
    if aggression > 0.7 and consistency < 0.6:
        return DrivingStyleArchetype.QUALIFIER
    # 高平顺 + 低进攻 + 高一致 → 平顺操控型.
    if smoothness > 0.7 and aggression < 0.4 and consistency > 0.6:
        return DrivingStyleArchetype.SMOOTH_OPERATOR
    # 高平顺 + 低进攻 → 保胎型.
    if smoothness > 0.7 and aggression < 0.5:
        return DrivingStyleArchetype.TIRE_WHISPERER
    tire_wear = _opt_float(metrics, "tire_wear_score")
    if tire_wear is not None and tire_wear < 0.3 and smoothness > 0.5:
        return DrivingStyleArchetype.TIRE_WHISPERER
    # 低一致 → 发育型新人.
    if consistency < 0.4:
        return DrivingStyleArchetype.DEVELOPMENT
    # 默认: 均衡比赛型.
    return DrivingStyleArchetype.RACE_CRAFT


# --- 弯道四相位分析 --------------------------------------------------------
def _detect_corners(frames: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """检测弯道片段 (|steer| > 0.1 或 brake > 0.3 的连续帧)."""
    corners: list[tuple[int, int]] = []
    in_corner = False
    start = 0
    for i, f in enumerate(frames):
        s = abs(_f(f, "steer"))
        b = _f(f, "brake")
        is_corner = s > 0.1 or b > 0.3
        if is_corner and not in_corner:
            start = i
            in_corner = True
        elif not is_corner and in_corner:
            corners.append((start, i))
            in_corner = False
    if in_corner:
        corners.append((start, len(frames)))
    return corners


def _classify_phase(frame: dict[str, Any]) -> str:
    """将单帧归类到弯道相位 (braking/trail_braking/mid_corner/exit/other)."""
    b = _f(frame, "brake")
    t = _f(frame, "throttle")
    s = abs(_f(frame, "steer"))
    if b > 0.3 and t < 0.2:
        return "braking"
    if t > 0.5:
        return "exit"
    if b > 0.05:
        return "trail_braking"
    if t < 0.2 and s > 0.1:
        return "mid_corner"
    return "other"


def _estimate_dt(frames: list[dict[str, Any]]) -> float:
    """由 session_time 估计帧步长; 不可用则回退 1/60s。"""
    times: list[float] = []
    for f in frames:
        t = f.get("session_time")
        if t is None:
            continue
        try:
            times.append(float(t))
        except (TypeError, ValueError):
            continue
    if len(times) >= 2:
        span = times[-1] - times[0]
        if span > 0.0:
            return span / (len(times) - 1)
    return 1.0 / 60.0


def _phase_signal(phase: str, frame: dict[str, Any]) -> float:
    """相位主控信号: braking/trail→brake, mid→|steer|, exit→throttle。"""
    if phase in ("braking", "trail_braking"):
        return _f(frame, "brake")
    if phase == "mid_corner":
        return abs(_f(frame, "steer"))
    if phase == "exit":
        return _f(frame, "throttle")
    return 0.0


def _phase_peak(phase: str, frames: list[dict[str, Any]]) -> float:
    if not frames:
        return 0.0
    return _clamp01(max(_phase_signal(phase, f) for f in frames))


def _phase_smoothness(phase: str, frames: list[dict[str, Any]]) -> float:
    if not frames:
        return 0.0
    if len(frames) < 2:
        return 1.0
    sig = np.array([_phase_signal(phase, f) for f in frames], dtype=np.float64)
    diff = np.diff(sig)
    if diff.size == 0:
        return 1.0
    return _clamp01(1.0 - float(np.std(diff)))


def _duration_consistency(durations: list[float]) -> float:
    """跨弯道持续时长一致性 = 1 - cv(durations), 钳位 [0,1]。"""
    if len(durations) < 2:
        return 1.0
    cv = _cv(durations)
    return _clamp01(1.0 - cv)


def _empty_phases() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for p in _PHASE_ORDER:
        result[p] = {
            "duration_s": 0.0,
            "peak_value": 0.0,
            "smoothness": 0.0,
            "consistency_across_corners": 0.0,
        }
    result["weak_phase"] = None
    result["corners_detected"] = 0
    return result


def _weak_phase(result: dict[str, Any]) -> str | None:
    """薄弱相位 = 一致性最低; 同分取持续时长最长。"""
    candidates = [p for p in _PHASE_ORDER if p in result]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda p: (result[p]["consistency_across_corners"], -result[p]["duration_s"]),
    )


def analyze_corner_phases(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """将每个弯道拆分为 4 相位并聚合统计.

    返回结构::

        {
            "braking":        {"duration_s", "peak_value", "smoothness",
                               "consistency_across_corners"},
            "trail_braking":  {...},
            "mid_corner":     {...},
            "exit":           {...},
            "weak_phase": str | None,
            "corners_detected": int,
        }

    空帧或无弯道 → 各相位零指标, ``weak_phase`` 为 None。
    """
    if not frames:
        return _empty_phases()
    corners = _detect_corners(frames)
    if not corners:
        return _empty_phases()
    dt = _estimate_dt(frames)

    # 每相位收集: 跨弯道持续时长列表 + 全部帧.
    phase_durations: dict[str, list[float]] = {p: [] for p in _PHASE_ORDER}
    phase_frames: dict[str, list[dict[str, Any]]] = {p: [] for p in _PHASE_ORDER}
    for s_idx, e_idx in corners:
        seg = frames[s_idx:e_idx]
        if not seg:
            continue
        per_phase: dict[str, list[dict[str, Any]]] = {p: [] for p in _PHASE_ORDER}
        for f in seg:
            p = _classify_phase(f)
            if p in per_phase:
                per_phase[p].append(f)
        for p in _PHASE_ORDER:
            phase_durations[p].append(len(per_phase[p]) * dt)
            phase_frames[p].extend(per_phase[p])

    result: dict[str, Any] = {}
    for p in _PHASE_ORDER:
        durs = phase_durations[p]
        duration_s = float(np.mean(durs)) if durs else 0.0
        result[p] = {
            "duration_s": duration_s,
            "peak_value": _phase_peak(p, phase_frames[p]),
            "smoothness": _phase_smoothness(p, phase_frames[p]),
            "consistency_across_corners": _duration_consistency(durs),
        }
    result["weak_phase"] = _weak_phase(result)
    result["corners_detected"] = len(corners)
    return result


@dataclass
class CornerPhaseAnalysis:
    """弯道四相位分析结果容器."""

    phases: dict[str, dict[str, float]]
    weak_phase: str | None
    corners_detected: int

    @classmethod
    def from_frames(cls, frames: list[dict[str, Any]]) -> CornerPhaseAnalysis:
        raw = analyze_corner_phases(frames)
        return cls(
            phases={p: dict(raw[p]) for p in _PHASE_ORDER},
            weak_phase=raw.get("weak_phase"),
            corners_detected=int(raw.get("corners_detected", 0)),
        )


# --- 一致性分析 ------------------------------------------------------------
class DriverConsistencyAnalyzer:
    """圈速/输入一致性分析器.

    Parameters
    ----------
    min_laps
        触发完整分析所需的最少圈数; 不足则返回 ``insufficient_data=True`` 的退化结果。
    """

    def __init__(self, min_laps: int = 3) -> None:
        self.min_laps = max(1, int(min_laps))

    def analyze(self, lap_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        """计算一致性指标.

        ``lap_metrics`` 每项推荐键: lap_time, sector_times (list[float]),
        throttle_smoothness, brake_aggression。
        """
        n = len(lap_metrics)
        if n < self.min_laps:
            return {
                "lap_time_cv": 0.0,
                "sector_consistency": {},
                "input_consistency": {},
                "overall_consistency_score": 0.0,
                "weak_sector": None,
                "trend": "stable",
                "insufficient_data": True,
                "n_laps": n,
            }

        lap_times = [_f(lm, "lap_time") for lm in lap_metrics]
        lap_cv = _cv(lap_times)

        sector_cvs: dict[int, float] = {}
        max_sectors = 0
        for lm in lap_metrics:
            st = lm.get("sector_times")
            if isinstance(st, (list, tuple)):
                max_sectors = max(max_sectors, len(st))
        for si in range(max_sectors):
            vals: list[float] = []
            for lm in lap_metrics:
                st = lm.get("sector_times")
                if isinstance(st, (list, tuple)) and si < len(st) and st[si] is not None:
                    vals.append(float(st[si]))
            sector_cvs[si] = _cv(vals) if vals else 0.0

        thr_vals = [_f(lm, "throttle_smoothness") for lm in lap_metrics]
        brk_vals = [_f(lm, "brake_aggression") for lm in lap_metrics]
        input_cvs = {
            "throttle_smoothness": _cv(thr_vals),
            "brake_aggression": _cv(brk_vals),
        }

        lap_cv_n = lap_cv / (lap_cv + _CONS_K) if (lap_cv + _CONS_K) > 0 else 0.0
        input_mean = float(np.mean(list(input_cvs.values()))) if input_cvs else 0.0
        input_cv_n = (
            input_mean / (input_mean + _CONS_K) if (input_mean + _CONS_K) > 0 else 0.0
        )
        overall = _clamp01(1.0 - 0.5 * (lap_cv_n + input_cv_n))

        weak_sector: int | None = None
        if sector_cvs:
            max_cv = max(sector_cvs.values())
            if max_cv > 0.0:
                weak_sector = max(sector_cvs, key=lambda k: sector_cvs[k])

        return {
            "lap_time_cv": float(lap_cv),
            "sector_consistency": {int(k): float(v) for k, v in sector_cvs.items()},
            "input_consistency": {k: float(v) for k, v in input_cvs.items()},
            "overall_consistency_score": float(overall),
            "weak_sector": weak_sector,
            "trend": self._trend(lap_times),
            "insufficient_data": False,
            "n_laps": n,
        }

    @staticmethod
    def _trend(lap_times: list[float]) -> str:
        """圈速趋势: improving(变快)/degrading(变慢)/stable。"""
        m = len(lap_times)
        if m < 2:
            return "stable"
        x = np.arange(m, dtype=np.float64)
        y = np.asarray(lap_times, dtype=np.float64)
        slope = float(np.polyfit(x, y, 1)[0])
        mean_y = float(np.mean(y))
        if mean_y == 0.0:
            return "stable"
        rel = slope / mean_y
        if rel < -1e-4:
            return "improving"
        if rel > 1e-4:
            return "degrading"
        return "stable"

    @staticmethod
    def consistency_label(score: float) -> str:
        """一致性分数 → 中文标签。"""
        if score >= 0.85:
            return "高度一致"
        if score >= 0.65:
            return "较为一致"
        if score >= 0.40:
            return "波动较大"
        return "不稳定"


# --- 条件适应建模 ----------------------------------------------------------
class AdaptationProfile:
    """车手条件适应能力建模 (dry/wet/intermediate/hot/cold/high_altitude).

    performance 以圈速计: 越小越好。``dry`` 为基线; 适应强度 =
    baseline_perf / condition_perf (1 = 无退化)。
    """

    BASELINE = "dry"

    def __init__(self) -> None:
        self._records: dict[str, list[float]] = {}

    def record_condition(self, condition: str, performance: float) -> None:
        """记录某条件下的表现 (圈速)。"""
        self._records.setdefault(condition, []).append(float(performance))

    def _mean(self, condition: str) -> float | None:
        recs = self._records.get(condition)
        if not recs:
            return None
        return float(np.mean(recs))

    def adaptation_strength(self, condition: str) -> float:
        """条件适应强度 in [0,1]; 1 = 无退化 vs 基线。"""
        if condition == self.BASELINE:
            return 1.0
        base = self._mean(self.BASELINE)
        cond = self._mean(condition)
        if base is None or cond is None or cond <= 0.0:
            return 1.0
        return _clamp01(base / cond)

    def weak_conditions(self) -> list[str]:
        """适应强度 < 0.7 的非基线条件。"""
        return [
            c
            for c in self._records
            if c != self.BASELINE and self.adaptation_strength(c) < 0.7
        ]

    def strong_conditions(self) -> list[str]:
        """适应强度 >= 0.7 的非基线条件。"""
        return [
            c
            for c in self._records
            if c != self.BASELINE and self.adaptation_strength(c) >= 0.7
        ]

    def recommendation(self) -> str:
        """基于强弱条件的中文建议。"""
        weak = self.weak_conditions()
        strong = self.strong_conditions()
        if not weak and not strong:
            return "暂无足够数据, 建议在多条件下采集更多圈速数据以评估适应能力。"
        parts: list[str] = []
        if weak:
            parts.append(f"建议针对 {', '.join(weak)} 条件加强练习与调教适配")
        if strong:
            parts.append(f"车手在 {', '.join(strong)} 条件下表现稳定, 可保持现有策略")
        return "。".join(parts) + "。"


# --- 疲劳模型 --------------------------------------------------------------
class FatigueModel:
    """stint 内疲劳曲线: 前 10% 平坦, 中段线性退化 ~0.05s/lap, 末 20% 加速。

    Parameters
    ----------
    base_lap_time
        新胎/满电状态下的基准圈速 (秒)。
    stint_length_laps
        stint 预计长度 (圈)。
    """

    def __init__(self, base_lap_time: float, stint_length_laps: int = 30) -> None:
        self.base_lap_time = float(base_lap_time)
        self.stint_length_laps = max(1, int(stint_length_laps))

    def _zones(self) -> tuple[float, float]:
        n = self.stint_length_laps
        return 0.1 * n, 0.8 * n  # flat_zone, accel_start

    def fatigue_index(self, lap_number: int) -> float:
        """疲劳指数 in [0,1]: 0=新鲜, 1=力竭。"""
        if lap_number <= 0:
            return 0.0
        n = self.stint_length_laps
        if lap_number >= n:
            return 1.0
        flat_zone, accel_start = self._zones()
        if lap_number <= flat_zone:
            return 0.0
        if lap_number <= accel_start:
            span = max(1e-9, accel_start - flat_zone)
            return _clamp01(0.7 * (lap_number - flat_zone) / span)
        span = max(1e-9, n - accel_start)
        progress = (lap_number - accel_start) / span
        return _clamp01(0.7 + 0.3 * (progress * progress))

    def lap_time_with_fatigue(self, lap_number: int) -> float:
        """含疲劳退化的圈速预测。"""
        if lap_number <= 0:
            return self.base_lap_time
        n = self.stint_length_laps
        lap = min(float(lap_number), float(n))
        flat_zone, accel_start = self._zones()
        if lap <= flat_zone:
            penalty = 0.0
        elif lap <= accel_start:
            penalty = _FATIGUE_RATE * (lap - flat_zone)
        else:
            linear_part = _FATIGUE_RATE * (accel_start - flat_zone)
            span = max(1e-9, n - accel_start)
            ap = (lap - accel_start) / span
            accel_penalty = _FATIGUE_RATE * (n - accel_start) * (ap + ap * ap)
            penalty = linear_part + accel_penalty
        return self.base_lap_time + penalty

    def recommended_pit_window(self, current_lap: int) -> dict[str, Any]:
        """进站建议; 疲劳 > 0.7 推荐进站。"""
        idx = self.fatigue_index(current_lap)
        if idx > 0.7:
            return {
                "pit_recommended": True,
                "reason": f"疲劳指数 {idx:.2f} 已超过 0.7, 建议尽快进站换胎",
                "fatigue_index": float(idx),
            }
        return {
            "pit_recommended": False,
            "reason": f"疲劳指数 {idx:.2f}, 可继续当前 stint",
            "fatigue_index": float(idx),
        }


# --- 综合深度画像器 --------------------------------------------------------
class DeepDriverProfiler:
    """组合原型/相位/一致性/疲劳的综合车手画像器."""

    def __init__(
        self,
        frames: list[dict[str, Any]],
        setup: dict[str, Any],
        track_id: str,
        lap_metrics: list[dict[str, Any]] | None = None,
    ) -> None:
        self.frames = list(frames)
        self.setup = dict(setup) if setup else {}
        self.track_id = track_id
        self.lap_metrics = list(lap_metrics) if lap_metrics else []

    def profile(self) -> dict[str, Any]:
        """返回综合画像字典。"""
        basic = extract_driver_profile(self.frames)
        metrics: dict[str, Any] = {
            "brake_point_norm": basic.brake_point_norm,
            "throttle_smoothness": basic.throttle_smoothness,
            "steer_smoothness": basic.steer_smoothness,
            "corner_balance_pref": basic.corner_balance_pref,
            "aggression_score": basic.aggression_score,
            "consistency_score": basic.consistency_score,
            "ers_usage_intensity": basic.ers_usage_intensity,
            "drs_usage_efficiency": basic.drs_usage_efficiency,
        }
        archetype = classify_archetype(metrics)
        corner_phases = analyze_corner_phases(self.frames)

        consistency: dict[str, Any] | None = None
        if self.lap_metrics:
            consistency = DriverConsistencyAnalyzer().analyze(self.lap_metrics)

        base_lap_time = 90.0
        lts = [lm.get("lap_time") for lm in self.lap_metrics]
        lts = [float(v) for v in lts if v is not None and float(v) > 0.0]
        if lts:
            base_lap_time = float(min(lts))

        fatigue = FatigueModel(base_lap_time=base_lap_time, stint_length_laps=30)
        fatigue_projection: dict[str, Any] = {
            "base_lap_time": base_lap_time,
            "stint_length_laps": 30,
            "projected_lap_times": [
                fatigue.lap_time_with_fatigue(i) for i in range(1, 31)
            ],
            "fatigue_indices": [fatigue.fatigue_index(i) for i in range(1, 31)],
            "pit_window_at_lap_20": fatigue.recommended_pit_window(20),
        }

        strengths, weaknesses = self._derive_strengths_weaknesses(
            metrics, consistency, corner_phases
        )
        setup_recs = self._setup_recommendations(archetype, weaknesses)

        return {
            "archetype": archetype,
            "corner_phases": corner_phases,
            "consistency": consistency,
            "fatigue_projection": fatigue_projection,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "setup_recommendations": setup_recs,
        }

    @staticmethod
    def _derive_strengths_weaknesses(
        metrics: dict[str, Any],
        consistency: dict[str, Any] | None,
        corner_phases: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        strengths: list[str] = []
        weaknesses: list[str] = []
        if float(metrics.get("throttle_smoothness", 0.0)) > 0.7:
            strengths.append("油门控制平顺")
        if float(metrics.get("steer_smoothness", 0.0)) > 0.7:
            strengths.append("转向输入细腻")
        if float(metrics.get("consistency_score", 0.0)) > 0.7:
            strengths.append("圈速一致性强")
        if float(metrics.get("ers_usage_intensity", 0.0)) > 0.7:
            strengths.append("ERS 部署积极")
        if float(metrics.get("drs_usage_efficiency", 0.0)) > 0.7:
            strengths.append("DRS 使用高效")
        if consistency and float(consistency.get("overall_consistency_score", 0.0)) > 0.7:
            strengths.append("单圈波动小")

        if corner_phases and corner_phases.get("weak_phase"):
            wp = corner_phases["weak_phase"]
            weaknesses.append(f"弯道 {wp} 相位为薄弱环节")
        aggression = float(metrics.get("aggression_score", 0.0))
        if aggression > 0.8:
            weaknesses.append("进攻性过强, 轮胎损耗大")
        if 0.0 < aggression < 0.3:
            weaknesses.append("进攻性不足, 制动距离偏长")
        if float(metrics.get("consistency_score", 0.0)) < 0.4:
            weaknesses.append("一致性偏低, 圈速波动大")
        if 0.0 < float(metrics.get("throttle_smoothness", 0.0)) < 0.4:
            weaknesses.append("油门输入粗糙")
        if consistency and float(consistency.get("overall_consistency_score", 0.0)) < 0.4:
            weaknesses.append("单圈波动较大")

        if not strengths:
            strengths.append("风格均衡, 无明显短板")
        if not weaknesses:
            weaknesses.append("暂无明显薄弱环节")
        return strengths, weaknesses

    @staticmethod
    def _setup_recommendations(
        archetype: DrivingStyleArchetype, weaknesses: list[str]
    ) -> list[dict[str, str]]:
        table: dict[DrivingStyleArchetype, list[dict[str, str]]] = {
            DrivingStyleArchetype.SMOOTH_OPERATOR: [
                {
                    "field": "differential",
                    "direction": "increase",
                    "reason": "平滑风格适合更锁定的差速器, 提升弯中稳定性",
                }
            ],
            DrivingStyleArchetype.QUALIFIER: [
                {
                    "field": "brake_bias",
                    "direction": "rear",
                    "reason": "排位赛激进风格可适度后移制动平衡, 提升入弯灵活性",
                }
            ],
            DrivingStyleArchetype.RACE_CRAFT: [
                {
                    "field": "suspension",
                    "direction": "stiffen",
                    "reason": "均衡风格可适度硬悬, 兼顾响应与稳定",
                }
            ],
            DrivingStyleArchetype.TIRE_WHISPERER: [
                {
                    "field": "tire_pressure",
                    "direction": "decrease",
                    "reason": "保胎风格可略降胎压, 增大接地面积",
                }
            ],
            DrivingStyleArchetype.AGGRESSIVE_OVERTAKER: [
                {
                    "field": "ers_deploy",
                    "direction": "aggressive",
                    "reason": "进攻型风格可激进部署 ERS, 配合超车",
                }
            ],
            DrivingStyleArchetype.DEVELOPMENT: [
                {
                    "field": "setup_complexity",
                    "direction": "simplify",
                    "reason": "新人车手建议简化调教, 优先可控性",
                }
            ],
            DrivingStyleArchetype.WET_SPECIALIST: [
                {
                    "field": "ride_height",
                    "direction": "increase",
                    "reason": "雨战专长可适当升高底盘, 防止触底水滑",
                }
            ],
        }
        recs: list[dict[str, str]] = list(table.get(archetype, []))
        for w in weaknesses:
            if "轮胎" in w:
                recs.append({
                    "field": "tire_pressure",
                    "direction": "increase",
                    "reason": "降低轮胎损耗, 适度提高胎压延长寿命",
                })
            elif "油门" in w:
                recs.append({
                    "field": "throttle_sensitivity",
                    "direction": "decrease",
                    "reason": "平滑油门映射, 降低突兀输入",
                })
            elif "制动" in w:
                recs.append({
                    "field": "brake_bias",
                    "direction": "front",
                    "reason": "适度前移制动平衡, 缩短制动距离",
                })
            elif "一致" in w or "波动" in w:
                recs.append({
                    "field": "suspension",
                    "direction": "soften",
                    "reason": "适度软悬, 提升容错与一致性",
                })
            elif "相位" in w:
                recs.append({
                    "field": "aero_balance",
                    "direction": "adjust",
                    "reason": "针对薄弱弯道相位调整空气平衡",
                })
        seen: set[tuple[str, str]] = set()
        out: list[dict[str, str]] = []
        for r in recs:
            key = (r["field"], r["direction"])
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out


__all__ = [
    "DrivingStyleArchetype",
    "classify_archetype",
    "CornerPhaseAnalysis",
    "analyze_corner_phases",
    "DriverConsistencyAnalyzer",
    "AdaptationProfile",
    "FatigueModel",
    "DeepDriverProfiler",
]
