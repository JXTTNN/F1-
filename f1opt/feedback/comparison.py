"""Comparative lap & sector analysis for the F1 setup optimizer.

Adds a *comparative* layer on top of the per-lap metrics produced by
:func:`f1opt.feedback.engine.extract_metrics` and the sector-level
predictions from :func:`f1opt.model.surrogate.predict_full` (whose
``sectors`` field is a ``[s1, s2, s3]`` triple):

- :class:`LapComparator` — compare a lap against a reference lap (teammate /
  best / target), identifying the strongest / weakest sector and rendering a
  Chinese verdict. Iter-180: added :meth:`ideal_lap` for theoretical best
  lap from sector stitching.
- :class:`SectorAnalyzer` — deep multi-lap sector analysis: per-sector
  averages / bests / consistency (coefficient of variation), the theoretical
  "perfect lap" (sum of best sectors), the weak / strong sector, and a
  corner-count-adjusted per-sector strength map.
- :class:`TeammateComparison` — driver vs teammate head-to-head (best / avg
  gaps, sectors won, consistency comparison, qualifying-gap prediction).
  Iter-180: :meth:`head_to_head` now includes ideal lap comparison.
- :class:`SetupChangeImpact` — before/after setup-change impact with a
  significance flag and a Chinese verdict.

Lap dicts use the shape ``{lap_time, sector_times: [s1, s2, s3], avg_speed,
max_speed, ...}`` (matching ``extract_metrics`` ``values`` / surrogate
``predict_full`` output). All classes degrade gracefully on empty / partial
inputs — they return zero / neutral values and never raise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "LapComparator",
    "SectorAnalyzer",
    "SetupChangeImpact",
    "TeammateComparison",
]

_N_SECTORS = 3
_VERDICT_EPS = 0.05
_SIGNIFICANT_LAP_DELTA = 0.1
_SIGNIFICANT_CONSISTENCY_IMPROV_PCT = 0.10


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _lap_time(lap: dict[str, Any]) -> float | None:
    v = lap.get("lap_time")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _sector_times(lap: dict[str, Any]) -> list[float]:
    raw = lap.get("sector_times") or []
    out: list[float] = []
    for i in range(_N_SECTORS):
        out.append(_as_float(raw[i]) if i < len(raw) else 0.0)
    return out


def _speed(lap: dict[str, Any], key: str) -> float | None:
    v = lap.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _cv(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    mean = float(np.mean(arr))
    if abs(mean) < 1e-12:
        return 0.0
    return float(np.std(arr) / abs(mean))


def _verdict_lap(delta: float, *, no_ref: bool = False) -> str:
    if no_ref:
        return "无参考圈速"
    if delta < -_VERDICT_EPS:
        return f"快{abs(delta):.3f}秒"
    if delta > _VERDICT_EPS:
        return f"慢{delta:.3f}秒"
    return "持平"


def _verdict_setup(delta_avg: float) -> str:
    if delta_avg < -_VERDICT_EPS:
        return f"调教改进{abs(delta_avg):.3f}秒"
    if delta_avg > _VERDICT_EPS:
        return f"调教退步{delta_avg:.3f}秒"
    return "无明显变化"


class LapComparator:
    """Compare laps against a reference lap (teammate / best / target)."""

    def __init__(self, reference_lap: dict[str, Any] | None = None) -> None:
        self.reference_lap = reference_lap

    def compare(self, lap: dict[str, Any]) -> dict[str, Any]:
        lap_sec = _sector_times(lap)
        lap_avg = _speed(lap, "avg_speed")
        lap_max = _speed(lap, "max_speed")
        lap_lt = _lap_time(lap)

        ref = self.reference_lap
        if ref is None:
            return {
                "lap_time_delta": 0.0,
                "sector_deltas": [0.0, 0.0, 0.0],
                "speed_deltas": {"avg_speed_delta": 0.0, "max_speed_delta": 0.0},
                "strength_sector": 0,
                "weakness_sector": 0,
                "verdict": _verdict_lap(0.0, no_ref=True),
            }

        ref_sec = _sector_times(ref)
        ref_avg = _speed(ref, "avg_speed")
        ref_max = _speed(ref, "max_speed")
        ref_lt = _lap_time(ref)

        lap_time_delta = (
            float(lap_lt - ref_lt)
            if (lap_lt is not None and ref_lt is not None)
            else 0.0
        )
        sector_deltas = [float(lap_sec[i] - ref_sec[i]) for i in range(_N_SECTORS)]
        speed_deltas = {
            "avg_speed_delta": (
                float(lap_avg - ref_avg)
                if (lap_avg is not None and ref_avg is not None)
                else 0.0
            ),
            "max_speed_delta": (
                float(lap_max - ref_max)
                if (lap_max is not None and ref_max is not None)
                else 0.0
            ),
        }

        strength_sector = int(np.argmin(sector_deltas)) + 1
        weakness_sector = int(np.argmax(sector_deltas)) + 1

        return {
            "lap_time_delta": lap_time_delta,
            "sector_deltas": sector_deltas,
            "speed_deltas": speed_deltas,
            "strength_sector": strength_sector,
            "weakness_sector": weakness_sector,
            "verdict": _verdict_lap(lap_time_delta),
        }

    def compare_multi(self, laps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.compare(lap) for lap in laps]

    def rank_laps(
        self, laps: list[dict[str, Any]]
    ) -> list[tuple[int, dict[str, Any]]]:
        indexed = list(enumerate(laps))

        def _key(pair: tuple[int, dict[str, Any]]) -> float:
            lt = _lap_time(pair[1])
            return lt if lt is not None else float("inf")

        indexed.sort(key=_key)
        return [(i, lap) for i, lap in indexed]

    @staticmethod
    def ideal_lap(laps: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute the theoretical best lap from best-sector stitching (Iter-180).

        Finds the fastest sector 1/2/3 across all supplied laps and sums them
        to produce the "ideal lap" — the lap the driver could achieve by
        stringing together their best sector performances. Also reports which
        lap each best sector came from and the potential gain vs the actual
        best full lap.

        Returns:
            ``{"lap_time": float, "sector_times": [s1,s2,s3],
            "best_s1_from_lap": int, "best_s2_from_lap": int,
            "best_s3_from_lap": int, "potential_gain_s": float}``
        """
        if not laps:
            return {
                "lap_time": 0.0,
                "sector_times": [0.0, 0.0, 0.0],
                "best_s1_from_lap": -1,
                "best_s2_from_lap": -1,
                "best_s3_from_lap": -1,
                "potential_gain_s": 0.0,
            }
        best_sectors: list[float] = [float("inf")] * _N_SECTORS
        best_from: list[int] = [-1] * _N_SECTORS
        actual_best = float("inf")
        for idx, lap in enumerate(laps):
            sec = _sector_times(lap)
            for s in range(_N_SECTORS):
                if sec[s] > 0 and sec[s] < best_sectors[s]:
                    best_sectors[s] = sec[s]
                    best_from[s] = idx
            lt = _lap_time(lap)
            if lt is not None and lt > 0 and lt < actual_best:
                actual_best = lt
        ideal_time = sum(best_sectors)
        gain = max(0.0, actual_best - ideal_time) if actual_best < float("inf") else 0.0
        return {
            "lap_time": round(ideal_time, 3),
            "sector_times": [round(v, 3) for v in best_sectors],
            "best_s1_from_lap": best_from[0],
            "best_s2_from_lap": best_from[1],
            "best_s3_from_lap": best_from[2],
            "potential_gain_s": round(gain, 3),
        }


class SectorAnalyzer:
    """Deep multi-lap sector analysis."""

    def __init__(self) -> None:
        pass

    def analyze(self, sector_times: list[list[float]]) -> dict[str, Any]:
        if not sector_times:
            return {
                "sector_averages": [],
                "sector_bests": [],
                "sector_consistency": [],
                "theoretical_best": 0.0,
                "theoretical_best_delta": 0.0,
                "weak_sector": 0,
                "strong_sector": 0,
                "potential_gain_s": 0.0,
            }
        per_sector: list[list[float]] = [[] for _ in range(_N_SECTORS)]
        lap_times: list[float] = []
        for row in sector_times:
            row_list = list(row) if row is not None else []
            lap_sum = 0.0
            for i in range(_N_SECTORS):
                v = _as_float(row_list[i]) if i < len(row_list) else 0.0
                per_sector[i].append(v)
                lap_sum += v
            lap_times.append(lap_sum)
        sector_averages = [float(np.mean(per_sector[i])) for i in range(_N_SECTORS)]
        sector_bests = [float(np.min(per_sector[i])) for i in range(_N_SECTORS)]
        sector_consistency = [_cv(per_sector[i]) for i in range(_N_SECTORS)]
        theoretical_best = float(sum(sector_bests))
        actual_best = float(min(lap_times)) if lap_times else 0.0
        theoretical_best_delta = max(0.0, actual_best - theoretical_best)
        weak_sector = int(np.argmax(sector_consistency)) + 1
        strong_sector = int(np.argmin(sector_consistency)) + 1
        potential_gain_s = float(
            sum(sector_averages[i] - sector_bests[i] for i in range(_N_SECTORS))
        )
        if potential_gain_s < 0:
            potential_gain_s = 0.0
        return {
            "sector_averages": sector_averages,
            "sector_bests": sector_bests,
            "sector_consistency": sector_consistency,
            "theoretical_best": theoretical_best,
            "theoretical_best_delta": theoretical_best_delta,
            "weak_sector": weak_sector,
            "strong_sector": strong_sector,
            "potential_gain_s": potential_gain_s,
        }

    def corner_strength_map(
        self,
        sector_times: list[list[float]],
        corner_counts: list[int],
    ) -> dict[int, float]:
        averages = self.analyze(sector_times)["sector_averages"]
        if not averages or not corner_counts:
            return {i + 1: 0.0 for i in range(_N_SECTORS)}
        counts = [
            (
                int(corner_counts[i])
                if i < len(corner_counts) and corner_counts[i]
                else 1
            )
            for i in range(_N_SECTORS)
        ]
        tpc = [float(averages[i]) / float(counts[i]) for i in range(_N_SECTORS)]
        min_tpc = min(tpc)
        max_tpc = max(tpc)
        if max_tpc - min_tpc < 1e-12:
            return {i + 1: 1.0 for i in range(_N_SECTORS)}
        span = max_tpc - min_tpc
        return {i + 1: float(1.0 - (tpc[i] - min_tpc) / span) for i in range(_N_SECTORS)}


class TeammateComparison:
    """Driver vs teammate head-to-head comparison."""

    def __init__(
        self,
        driver_laps: list[dict[str, Any]],
        teammate_laps: list[dict[str, Any]],
    ) -> None:
        self.driver_laps = driver_laps
        self.teammate_laps = teammate_laps

    @staticmethod
    def _lap_times(laps: list[dict[str, Any]]) -> list[float]:
        return [t for t in (_lap_time(lap) for lap in laps) if t is not None]

    @classmethod
    def _best_lap_time(cls, laps: list[dict[str, Any]]) -> float | None:
        times = cls._lap_times(laps)
        return float(min(times)) if times else None

    @classmethod
    def _avg_lap_time(cls, laps: list[dict[str, Any]]) -> float | None:
        times = cls._lap_times(laps)
        return float(np.mean(times)) if times else None

    @classmethod
    def _lap_time_cv(cls, laps: list[dict[str, Any]]) -> float:
        return _cv(cls._lap_times(laps))

    @staticmethod
    def _sector_bests(laps: list[dict[str, Any]]) -> list[float | None]:
        per_sector: list[list[float]] = [[] for _ in range(_N_SECTORS)]
        for lap in laps:
            sec = _sector_times(lap)
            for i in range(_N_SECTORS):
                per_sector[i].append(sec[i])
        return [
            (float(min(per_sector[i])) if per_sector[i] else None)
            for i in range(_N_SECTORS)
        ]

    def head_to_head(self) -> dict[str, Any]:
        driver_best = self._best_lap_time(self.driver_laps)
        teammate_best = self._best_lap_time(self.teammate_laps)
        driver_avg = self._avg_lap_time(self.driver_laps)
        teammate_avg = self._avg_lap_time(self.teammate_laps)

        gap_best_s = (
            float(driver_best - teammate_best)
            if (driver_best is not None and teammate_best is not None)
            else 0.0
        )
        gap_avg_s = (
            float(driver_avg - teammate_avg)
            if (driver_avg is not None and teammate_avg is not None)
            else 0.0
        )

        drv_sec = self._sector_bests(self.driver_laps)
        tea_sec = self._sector_bests(self.teammate_laps)
        sectors_won_driver = 0
        sectors_won_teammate = 0
        for i in range(_N_SECTORS):
            d = drv_sec[i]
            t = tea_sec[i]
            if d is None or t is None:
                continue
            if d < t:
                sectors_won_driver += 1
            elif t < d:
                sectors_won_teammate += 1

        driver_cv = self._lap_time_cv(self.driver_laps)
        teammate_cv = self._lap_time_cv(self.teammate_laps)
        better = "driver" if driver_cv <= teammate_cv else "teammate"

        if driver_best is not None and teammate_best is not None:
            if gap_best_s < -_VERDICT_EPS:
                verdict = f"车手比队友快{abs(gap_best_s):.3f}秒"
            elif gap_best_s > _VERDICT_EPS:
                verdict = f"车手比队友慢{gap_best_s:.3f}秒"
            else:
                verdict = "车手与队友圈速持平"
        else:
            verdict = "数据不足"

        # Iter-180: ideal lap comparison
        driver_ideal = LapComparator.ideal_lap(self.driver_laps)
        teammate_ideal = LapComparator.ideal_lap(self.teammate_laps)
        gap_ideal_s = (
            driver_ideal["lap_time"] - teammate_ideal["lap_time"]
            if driver_ideal["lap_time"] > 0 and teammate_ideal["lap_time"] > 0
            else 0.0
        )

        return {
            "driver_best": driver_best,
            "teammate_best": teammate_best,
            "driver_avg": driver_avg,
            "teammate_avg": teammate_avg,
            "gap_best_s": gap_best_s,
            "gap_avg_s": gap_avg_s,
            "sectors_won_driver": sectors_won_driver,
            "sectors_won_teammate": sectors_won_teammate,
            "consistency_comparison": {
                "driver_cv": driver_cv,
                "teammate_cv": teammate_cv,
                "better": better,
            },
            "verdict": verdict,
            "ideal_lap_comparison": {
                "driver_ideal": driver_ideal,
                "teammate_ideal": teammate_ideal,
                "gap_ideal_s": round(gap_ideal_s, 3),
            },
        }

    def qualifying_prediction(self) -> dict[str, Any]:
        h2h = self.head_to_head()
        gap_best = float(h2h["gap_best_s"])
        drv_cv = float(h2h["consistency_comparison"]["driver_cv"])
        tea_cv = float(h2h["consistency_comparison"]["teammate_cv"])
        n_drv = len(self.driver_laps)
        n_tea = len(self.teammate_laps)
        n_total = n_drv + n_tea

        consistency_penalty = max(0.0, drv_cv - tea_cv)
        predicted_pole_s = max(0.0, gap_best + consistency_penalty)

        sample_factor = min(1.0, n_total / 6.0)
        cv_spread = abs(drv_cv - tea_cv)
        confidence = max(
            0.1,
            min(0.95, 0.5 * sample_factor + 0.4 * (1.0 - min(1.0, cv_spread))),
        )

        if n_drv == 0 or n_tea == 0:
            reasoning = "样本不足, 无法可靠预测排位赛差距"
            confidence = 0.1
        elif predicted_pole_s < _VERDICT_EPS:
            reasoning = (
                f"车手最佳圈接近或快于队友 (gap={gap_best:+.3f}s), "
                f"一致性CV={drv_cv:.3f}, 有望争夺杆位"
            )
        else:
            reasoning = (
                f"基于最佳圈差距 {gap_best:+.3f}s 与一致性 "
                f"(车手CV={drv_cv:.3f}, 队友CV={tea_cv:.3f}) 预测排位赛"
                f"落后约 {predicted_pole_s:.3f}s"
            )

        return {
            "predicted_pole_s": float(predicted_pole_s),
            "confidence": float(confidence),
            "reasoning": reasoning,
        }


class SetupChangeImpact:
    """Compare before/after setup-change lap sets."""

    def __init__(
        self,
        before_laps: list[dict[str, Any]],
        after_laps: list[dict[str, Any]],
    ) -> None:
        self.before_laps = before_laps
        self.after_laps = after_laps

    def analyze(self) -> dict[str, Any]:
        before_lts = [t for t in (_lap_time(lap) for lap in self.before_laps) if t is not None]
        after_lts = [t for t in (_lap_time(lap) for lap in self.after_laps) if t is not None]

        if not before_lts or not after_lts:
            return {
                "before_avg": None,
                "after_avg": None,
                "delta_avg_s": 0.0,
                "before_cv": 0.0,
                "after_cv": 0.0,
                "significant": False,
                "verdict": "数据不足",
            }

        before_avg = float(np.mean(before_lts))
        after_avg = float(np.mean(after_lts))
        delta_avg = after_avg - before_avg
        before_cv = _cv(before_lts)
        after_cv = _cv(after_lts)
        significant = (
            abs(delta_avg) >= _SIGNIFICANT_LAP_DELTA
            or (before_cv > 0 and (before_cv - after_cv) / before_cv >= _SIGNIFICANT_CONSISTENCY_IMPROV_PCT)
        )

        return {
            "before_avg": before_avg,
            "after_avg": after_avg,
            "delta_avg_s": delta_avg,
            "before_cv": before_cv,
            "after_cv": after_cv,
            "significant": significant,
            "verdict": _verdict_setup(delta_avg),
        }
