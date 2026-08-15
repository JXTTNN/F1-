"""Higher-order analytics derived from aligned F1 telemetry frames.

Three public classes:

* :class:`TelemetryAnalytics` — derives per-lap metrics (speed/throttle/brake/
  steering/g-force/ERS/DRS/tire-load traces, plus smoothing & racing-line
  proxies) from a list of unified frame dicts. The frames are expected to
  follow the layout produced by
  :class:`f1opt.telemetry.aligner.TelemetryAligner` (``speed``, ``throttle``,
  ``brake``, ``steer``, ``g_lat``, ``g_long``, ``rpm`` …) but the module is
  defensive: missing fields, ``None`` values and empty / single-frame inputs
  return sensible defaults rather than raise.

* :class:`PerformanceBenchmark` — compares a metrics dict (as produced by
  :meth:`TelemetryAnalytics.compute_all`) against per-track-type references
  and produces a graded scorecard with strengths / weaknesses.

* :class:`AnomalyDetector` — flags per-frame anomalies (sudden deceleration,
  extreme g, sustained redline, simultaneous brake+throttle, extreme steering,
  ERS overdeploy, sensor-stuck on physical channels, speed outliers, lap-time
  jumps) and reports a per-severity distribution.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from f1opt.data.tracks import TRACKS_BY_ID

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
#: Default frame interval when frames lack ``session_time`` / ``lap_time``.
_DEFAULT_DT: float = 1.0 / 60.0

#: F1 minimum car+driver mass (kg) — used for tire load estimation.
_F1_MASS_KG: float = 798.0
#: Gravity (m/s^2).
_GRAVITY: float = 9.81
#: Reference geometry for tire load transfer estimation.
_WHEELBASE_M: float = 3.5
_TRACK_WIDTH_M: float = 2.0
_CG_HEIGHT_M: float = 0.30

#: Histogram bin count for speed / throttle / brake distributions.
_N_BINS: int = 10


# --------------------------------------------------------------------------- #
# Per-track-type reference targets (used by PerformanceBenchmark).
# --------------------------------------------------------------------------- #
TRACK_REFERENCES: dict[str, dict[str, float]] = {
    "high_speed_low_downforce": {
        "v_max_target": 340.0,
        "v_avg_target": 245.0,
        "corner_count": 13.0,
        "drs_zones": 3.0,
        "full_throttle_pct_target": 0.65,
        "lap_smoothing_target": 0.80,
    },
    "street": {
        "v_max_target": 290.0,
        "v_avg_target": 195.0,
        "corner_count": 19.0,
        "drs_zones": 1.0,
        "full_throttle_pct_target": 0.45,
        "lap_smoothing_target": 0.75,
    },
    "high_downforce": {
        "v_max_target": 310.0,
        "v_avg_target": 220.0,
        "corner_count": 14.0,
        "drs_zones": 2.0,
        "full_throttle_pct_target": 0.50,
        "lap_smoothing_target": 0.78,
    },
    "medium": {
        "v_max_target": 325.0,
        "v_avg_target": 230.0,
        "corner_count": 15.0,
        "drs_zones": 2.0,
        "full_throttle_pct_target": 0.55,
        "lap_smoothing_target": 0.80,
    },
    "mixed": {
        "v_max_target": 330.0,
        "v_avg_target": 240.0,
        "corner_count": 18.0,
        "drs_zones": 2.0,
        "full_throttle_pct_target": 0.55,
        "lap_smoothing_target": 0.78,
    },
}

#: Fallback reference when ``track_id`` is unknown.
_DEFAULT_REFERENCE: dict[str, float] = TRACK_REFERENCES["medium"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_float(v: Any, default: float = 0.0) -> float:
    """Coerce ``v`` to a finite float; ``None`` / non-numeric → ``default``."""
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f


def _field(frames: list[dict], key: str, default: float = 0.0) -> np.ndarray:
    """Extract a numeric field from frames as a numpy array (defaults if missing)."""
    return np.array(
        [_to_float(f.get(key), default) for f in frames], dtype=np.float64
    )


def _field_multi(
    frames: list[dict], keys: tuple[str, ...], default: float = 0.0
) -> np.ndarray:
    """Extract the first available field from ``keys`` per frame."""
    if not frames:
        return np.array([], dtype=np.float64)
    out: list[float] = []
    for f in frames:
        v: float | None = None
        for k in keys:
            if k in f and f[k] is not None:
                v = _to_float(f[k], default)
                break
        out.append(default if v is None else v)
    return np.array(out, dtype=np.float64)


def _times(frames: list[dict]) -> np.ndarray:
    """Per-frame time stamps; falls back to 60Hz spacing when missing."""
    if not frames:
        return np.array([], dtype=np.float64)
    ts: list[float] = []
    for i, f in enumerate(frames):
        t = f.get("session_time")
        if t is None:
            t = f.get("lap_time")
        if t is None:
            t = i * _DEFAULT_DT
        ts.append(_to_float(t, i * _DEFAULT_DT))
    return np.array(ts, dtype=np.float64)


def _deltas(times: np.ndarray) -> np.ndarray:
    """Forward time deltas (length = len(times)); last repeats prior."""
    if times.size < 2:
        return np.full_like(times, _DEFAULT_DT)
    d = np.diff(times)
    d = np.where(d <= 0, _DEFAULT_DT, d)
    return np.append(d, d[-1])


def _histogram(
    values: np.ndarray, bins: int = _N_BINS, lo: float = 0.0, hi: float | None = None
) -> list[int]:
    """Histogram of ``values`` as a list of int counts; empty input → zeros."""
    if values.size == 0:
        return [0] * bins
    if hi is None:
        hi = float(values.max())
    if hi <= lo:
        hi = lo + 1.0
    counts, _ = np.histogram(values, bins=bins, range=(lo, hi))
    return counts.astype(int).tolist()


def _count_runs(mask: np.ndarray) -> int:
    """Count contiguous True runs in a boolean mask."""
    if mask.size == 0:
        return 0
    # A run starts where mask is True and the previous frame is False.
    starts = mask & ~np.roll(mask, 1)
    starts[0] = mask[0]  # first frame: starts a run iff True (roll wraps)
    return int(starts.sum())


def _run_durations(
    mask: np.ndarray, dt: np.ndarray
) -> list[float]:
    """Duration (seconds) of each contiguous True run in ``mask``."""
    if mask.size == 0 or not mask.any():
        return []
    durations: list[float] = []
    in_run = False
    run_t = 0.0
    for i, m in enumerate(mask):
        if m:
            if not in_run:
                in_run = True
                run_t = float(dt[i])
            else:
                run_t += float(dt[i])
        else:
            if in_run:
                durations.append(run_t)
                in_run = False
                run_t = 0.0
    if in_run:
        durations.append(run_t)
    return durations


# --------------------------------------------------------------------------- #
# TelemetryAnalytics
# --------------------------------------------------------------------------- #
class TelemetryAnalytics:
    """Compute advanced per-lap metrics from aligned telemetry frames.

    ``frames`` is a list of dicts with keys such as ``speed`` (km/h),
    ``throttle`` / ``brake`` / ``steer`` (each in ``[-1, 1]`` or ``[0, 1]``),
    ``g_lat`` / ``g_long`` (g), ``rpm``, ``drs`` (0/1) and ``ers_deploy``
    (0..1). All keys are optional; missing fields default to 0.

    ``track_length_m`` is used only for the racing-line deviation estimate.
    """

    def __init__(
        self, frames: list[dict], track_length_m: float = 5000.0
    ) -> None:
        self.frames: list[dict] = list(frames)
        self.track_length_m: float = float(track_length_m)

    # ------------------------------------------------------------------ #
    # Aggregate
    # ------------------------------------------------------------------ #
    def compute_all(self) -> dict[str, Any]:
        """Return a dict with every sub-analysis keyed by short name."""
        return {
            "speed": self.speed_trace_analysis(),
            "throttle": self.throttle_trace_analysis(),
            "brake": self.brake_trace_analysis(),
            "steering": self.steering_trace_analysis(),
            "gforce": self.gforce_analysis(),
            "ers": self.ers_analysis(),
            "drs": self.drs_analysis(),
            "tire_load": self.tire_load_analysis(),
            "lap_smoothing_score": self.lap_smoothing_score(),
            "racing_line_deviation": self.racing_line_deviation(),
            "corner_exit_speed": self.corner_exit_speed_analysis(),
            "sector_timing": self.sector_timing_analysis(),
            "ers_deploy_mode": self.ers_deploy_mode_analysis(),
            "tyre_degradation": self.tyre_degradation_analysis(),
            "throttle_brake_overlap": self.throttle_brake_overlap_analysis(),
            "corner_entry_exit_ratio": self.corner_entry_exit_ratio_analysis(),
            "gear_usage": self.gear_usage_analysis(),
            "sector_throttle_smoothness": self.sector_throttle_smoothness(),
            "engine_braking": self.engine_braking_analysis(),
            "grip_utilization": self.grip_utilization_analysis(),
            "downforce_balance": self.downforce_balance_analysis(),
            "active_aero": self.active_aero_usage_analysis(),
            "mechanical_grip_trend": self.mechanical_grip_trend_analysis(),
            "brake_temp_balance": self.brake_temp_balance_analysis(),
            "tyre_temp_gradient": self.tyre_temp_gradient_analysis(),
            "fuel_per_sector": self.fuel_per_sector_analysis(),
            "ers_sector": self.ers_sector_analysis(),
            "grip_consistency": self.grip_consistency_analysis(),
            "ers_recovery_efficiency": self.ers_recovery_efficiency_analysis(),
        }

    # ------------------------------------------------------------------ #
    # Speed
    # ------------------------------------------------------------------ #
    def speed_trace_analysis(self) -> dict[str, Any]:
        speeds = _field(self.frames, "speed")
        if speeds.size == 0:
            return {
                "v_max": 0.0,
                "v_min": 0.0,
                "v_avg": 0.0,
                "v_std": 0.0,
                "speed_histogram": [0] * _N_BINS,
                "corner_speed_distribution": {"fast": [], "medium": [], "slow": []},
            }
        v_max = float(speeds.max())
        v_min = float(speeds.min())
        v_avg = float(speeds.mean())
        v_std = float(speeds.std())
        hi = max(v_max, 1.0)
        speed_hist = _histogram(speeds, bins=_N_BINS, lo=0.0, hi=hi)

        # Cornering frames: |steer| > 0.1 — bucket their apex speeds.
        steer = _field(self.frames, "steer")
        corner_mask = np.abs(steer) > 0.1
        corner_speeds = speeds[corner_mask]
        fast = corner_speeds[corner_speeds > 200.0].tolist()
        medium = corner_speeds[
            (corner_speeds >= 120.0) & (corner_speeds <= 200.0)
        ].tolist()
        slow = corner_speeds[corner_speeds < 120.0].tolist()
        return {
            "v_max": v_max,
            "v_min": v_min,
            "v_avg": v_avg,
            "v_std": v_std,
            "speed_histogram": speed_hist,
            "corner_speed_distribution": {
                "fast": fast,
                "medium": medium,
                "slow": slow,
            },
        }

    # ------------------------------------------------------------------ #
    # Throttle
    # ------------------------------------------------------------------ #
    def throttle_trace_analysis(self) -> dict[str, Any]:
        thr = _field(self.frames, "throttle")
        times = _times(self.frames)
        dt = _deltas(times)
        if thr.size == 0:
            return {
                "full_throttle_pct": 0.0,
                "zero_throttle_pct": 0.0,
                "throttle_histogram": [0] * _N_BINS,
                "lift_and_coast_events": 0,
            }
        total_t = float(dt.sum())
        if total_t <= 0:
            total_t = thr.size * _DEFAULT_DT
        full_t = float(dt[thr > 0.95].sum())
        zero_t = float(dt[thr < 0.05].sum())
        full_pct = full_t / total_t if total_t > 0 else 0.0
        zero_pct = zero_t / total_t if total_t > 0 else 0.0
        thr_hist = _histogram(thr, bins=_N_BINS, lo=0.0, hi=1.0)

        # Lift-and-coast: throttle drops from >0.8 to <0.3 within 0.5 s.
        events = 0
        if thr.size >= 2:
            # Find rising-time of each frame; for each frame, look ahead up to
            # 0.5 s for a frame with throttle < 0.3, starting from one with >0.8.
            for i in range(thr.size - 1):
                if thr[i] <= 0.8:
                    continue
                # Walk forward until cumulative time > 0.5 s.
                t_budget = 0.5
                j = i
                while j + 1 < thr.size and t_budget > 0:
                    t_budget -= float(dt[j])
                    j += 1
                    if thr[j] < 0.3:
                        events += 1
                        break
                    if t_budget <= 0:
                        break
        return {
            "full_throttle_pct": full_pct,
            "zero_throttle_pct": zero_pct,
            "throttle_histogram": thr_hist,
            "lift_and_coast_events": events,
        }

    # ------------------------------------------------------------------ #
    # Brake
    # ------------------------------------------------------------------ #
    def brake_trace_analysis(self) -> dict[str, Any]:
        brake = _field(self.frames, "brake")
        steer = _field(self.frames, "steer")
        if brake.size == 0:
            return {
                "brake_intensity_hist": [0] * _N_BINS,
                "peak_brake_pressure": 0.0,
                "brake_release_smoothness": 1.0,
                "trail_brake_events": 0,
            }
        brake_hist = _histogram(brake, bins=_N_BINS, lo=0.0, hi=1.0)
        peak = float(brake.max())

        # Release smoothness: 1 - normalized std of brake derivative during
        # release phases (where brake is decreasing).
        if brake.size >= 2:
            d_brake = np.diff(brake)
            release_mask = d_brake < 0.0
            if release_mask.any():
                release_std = float(d_brake[release_mask].std())
                # Normalize: a typical aggressive release moves ~0.5/s.
                brake_release_smoothness = max(0.0, 1.0 - release_std / 0.5)
            else:
                brake_release_smoothness = 1.0
        else:
            brake_release_smoothness = 1.0

        # Trail-braking: brake decreasing while |steer| increasing.
        trail_events = 0
        if brake.size >= 2 and steer.size >= 2:
            d_brake = np.diff(brake)
            d_steer_abs = np.diff(np.abs(steer))
            # Trail-brake frames: brake decreasing AND |steer| increasing.
            trail_mask = (d_brake < -0.01) & (d_steer_abs > 0.01)
            # Count contiguous runs as single events.
            trail_events = _count_runs(trail_mask)
        return {
            "brake_intensity_hist": brake_hist,
            "peak_brake_pressure": peak,
            "brake_release_smoothness": brake_release_smoothness,
            "trail_brake_events": trail_events,
        }

    # ------------------------------------------------------------------ #
    # Steering
    # ------------------------------------------------------------------ #
    def steering_trace_analysis(self) -> dict[str, Any]:
        steer = _field(self.frames, "steer")
        times = _times(self.frames)
        dt = _deltas(times)
        if steer.size == 0:
            return {
                "steer_reversals": 0,
                "corner_count_estimate": 0,
                "avg_corner_duration": 0.0,
                "steering_aggression": 0.0,
            }
        # Steer reversals: sign changes (ignore zero crossings through 0).
        if steer.size >= 2:
            signs = np.sign(steer)
            # Treat 0 as not-yet-a-sign change.
            nonzero = signs != 0
            reversals = 0
            last_sign = 0
            for s in signs:
                if s == 0:
                    continue
                if last_sign != 0 and s != last_sign:
                    reversals += 1
                last_sign = int(s)
            _ = nonzero  # reserved for future use
        else:
            reversals = 0

        # Corner count: contiguous runs of |steer| > 0.3.
        corner_mask = np.abs(steer) > 0.3
        corner_count = _count_runs(corner_mask)
        durations = _run_durations(corner_mask, dt)
        avg_dur = float(np.mean(durations)) if durations else 0.0

        # Steering aggression: peak |d(steer)/dt| (per second).
        if steer.size >= 2:
            d_steer = np.diff(steer)
            d_t = np.diff(times)
            d_t = np.where(d_t <= 0, _DEFAULT_DT, d_t)
            rate = np.abs(d_steer / d_t)
            aggression = float(rate.max()) if rate.size else 0.0
        else:
            aggression = 0.0

        return {
            "steer_reversals": reversals,
            "corner_count_estimate": corner_count,
            "avg_corner_duration": avg_dur,
            "steering_aggression": aggression,
        }

    # ------------------------------------------------------------------ #
    # G-force
    # ------------------------------------------------------------------ #
    def gforce_analysis(self) -> dict[str, Any]:
        g_lat = _field(self.frames, "g_lat")
        g_long = _field(self.frames, "g_long")
        steer = _field(self.frames, "steer")
        if g_lat.size == 0:
            return {
                "g_lat_max": 0.0,
                "g_lat_avg": 0.0,
                "g_long_max": 0.0,
                "g_long_min": 0.0,
                "traction_circle_area": 0.0,
                "understeer_indicator": 0.0,
            }
        g_lat_max = float(np.abs(g_lat).max())
        g_lat_avg = float(np.abs(g_lat).mean())
        g_long_max = float(g_long.max())
        g_long_min = float(g_long.min())

        # Traction circle area: ellipse area = π · a · b where a = max|g_lat|,
        # b = max|g_long|. Always positive for any non-zero input.
        a = float(np.abs(g_lat).max())
        b = float(np.abs(g_long).max())
        area = math.pi * a * b

        # Understeer indicator: mean(|g_lat| / |steer|) for |steer| > 0.1.
        # Higher = more cornering force per unit steering = less understeer.
        understeer = 0.0
        active = np.abs(steer) > 0.1
        if active.any():
            ratios = np.abs(g_lat[active]) / np.clip(
                np.abs(steer[active]), 0.1, None
            )
            understeer = float(ratios.mean())

        return {
            "g_lat_max": g_lat_max,
            "g_lat_avg": g_lat_avg,
            "g_long_max": g_long_max,
            "g_long_min": g_long_min,
            "traction_circle_area": area,
            "understeer_indicator": understeer,
        }

    # ------------------------------------------------------------------ #
    # ERS
    # ------------------------------------------------------------------ #
    def ers_analysis(self) -> dict[str, Any]:
        deploy = _field_multi(self.frames, ("ers_deployed_this_lap", "ers_deploy", "ers_deployed"))
        brake = _field(self.frames, "brake")
        times = _times(self.frames)
        dt = _deltas(times)
        if deploy.size == 0:
            return {
                "ers_deploy_total": 0.0,
                "ers_recover_total": 0.0,
                "deploy_events": 0,
                "recover_events": 0,
                "ers_efficiency": 0.0,
            }
        # Iter-259: ers_deployed_this_lap 是累计能量 (单调递增), 而非速率。
        # Deploy total = 末值 - 首值 (本圈累计部署 MJ); 事件数用差分上升沿。
        deploy_total = float(deploy[-1] - deploy[0]) if deploy.size >= 2 else 0.0
        # Recovery: derive from braking (brake > 0.3 → MGU-K harvest).
        recover_signal = np.clip(brake - 0.3, 0.0, 1.0)
        recover_total = float((recover_signal * dt).sum())

        # Deploy events: rising edges of cumulative deploy (> 0.01 MJ).
        deploy_events = (
            int(np.sum(np.diff(deploy) > 0.01)) if deploy.size >= 2 else 0
        )
        # Recover events: rising edges of braking above 0.3.
        recover_events = 0
        if brake.size >= 2:
            high = brake > 0.3
            recover_events = _count_runs(high)

        # Efficiency: deploy / recover ratio (0 when recover is 0).
        if recover_total > 1e-9:
            efficiency = deploy_total / recover_total
        else:
            efficiency = 0.0

        return {
            "ers_deploy_total": deploy_total,
            "ers_recover_total": recover_total,
            "deploy_events": deploy_events,
            "recover_events": recover_events,
            "ers_efficiency": efficiency,
        }

    # ------------------------------------------------------------------ #
    # DRS
    # ------------------------------------------------------------------ #
    def drs_analysis(self) -> dict[str, Any]:
        drs = _field_multi(self.frames, ("drs", "drs_allowed", "drs_active"))
        speed = _field(self.frames, "speed")
        times = _times(self.frames)
        dt = _deltas(times)
        if drs.size == 0:
            return {
                "drs_activations": 0,
                "drs_duration_total": 0.0,
                "drs_speed_gain_avg": 0.0,
            }
        active = drs > 0.5
        activations = _count_runs(active)
        duration_total = float(dt[active].sum()) if active.any() else 0.0

        # Speed gain: avg speed with DRS vs without.
        gain = 0.0
        if active.any() and (~active).any() and speed.size == drs.size:
            with_drs = float(speed[active].mean())
            without_drs = float(speed[~active].mean())
            gain = with_drs - without_drs

        return {
            "drs_activations": activations,
            "drs_duration_total": duration_total,
            "drs_speed_gain_avg": gain,
        }

    # ------------------------------------------------------------------ #
    # Tire load estimation
    # ------------------------------------------------------------------ #
    def tire_load_analysis(self) -> dict[str, Any]:
        g_lat = _field(self.frames, "g_lat")
        g_long = _field(self.frames, "g_long")
        if g_lat.size == 0:
            return {
                "fl_load_n": 0.0,
                "fr_load_n": 0.0,
                "rl_load_n": 0.0,
                "rr_load_n": 0.0,
                "load_transfer_pct": 0.0,
                "imbalance_pct": 0.0,
            }
        base = _F1_MASS_KG * _GRAVITY / 4.0  # static load per tire
        # Longitudinal transfer (per axle): m · g_long · g · cg_h / wheelbase.
        # Positive g_long (acceleration) shifts load rearward; negative
        # (braking) shifts forward. Split equally across the two tires/axle.
        long_factor = _F1_MASS_KG * g_long * _GRAVITY * _CG_HEIGHT_M / _WHEELBASE_M / 2.0
        # Lateral transfer (per side): m · g_lat · g · cg_h / track_width.
        # Positive g_lat (right turn) shifts load to the left (outer) tires.
        lat_factor = _F1_MASS_KG * g_lat * _GRAVITY * _CG_HEIGHT_M / _TRACK_WIDTH_M / 2.0

        # Per-tire load per frame (front-left, front-right, rear-left, rear-right).
        # Front axle unloads under acceleration (g_long > 0); rear loads up.
        fl = base - long_factor + lat_factor
        fr = base - long_factor - lat_factor
        rl = base + long_factor + lat_factor
        rr = base + long_factor - lat_factor

        # Report mean load per tire over the lap.
        fl_load = float(fl.mean())
        fr_load = float(fr.mean())
        rl_load = float(rl.mean())
        rr_load = float(rr.mean())

        # Load transfer %: mean of |longitudinal + lateral transfer| / base.
        total_transfer = np.abs(long_factor * 2.0) + np.abs(lat_factor * 2.0)
        load_transfer_pct = float((total_transfer / (base * 4.0)).mean() * 100.0)

        # Imbalance %: mean abs deviation of per-tire loads from the static.
        per_frame = np.stack([fl, fr, rl, rr], axis=1)
        imbalance_pct = float(
            (np.abs(per_frame - base).mean(axis=1) / base).mean() * 100.0
        )

        return {
            "fl_load_n": fl_load,
            "fr_load_n": fr_load,
            "rl_load_n": rl_load,
            "rr_load_n": rr_load,
            "load_transfer_pct": load_transfer_pct,
            "imbalance_pct": imbalance_pct,
        }

    # ------------------------------------------------------------------ #
    # Smoothness & racing line
    # ------------------------------------------------------------------ #
    def lap_smoothing_score(self) -> float:
        """Overall input smoothness in ``[0, 1]`` (1 = perfectly smooth).

        Combines throttle / brake / steering smoothness, each measured as
        ``1 - normalized jerk RMS`` of the input signal.
        """
        if not self.frames:
            return 0.0
        thr = _field(self.frames, "throttle")
        brake = _field(self.frames, "brake")
        steer = _field(self.frames, "steer")
        times = _times(self.frames)
        thr_s = self._smoothness(thr, times, scale=2.0)
        brk_s = self._smoothness(brake, times, scale=2.0)
        str_s = self._smoothness(steer, times, scale=3.0)
        score = (thr_s + brk_s + str_s) / 3.0
        return float(max(0.0, min(1.0, score)))

    @staticmethod
    def _smoothness(
        signal: np.ndarray, times: np.ndarray, scale: float
    ) -> float:
        """``1 - normalized_jerk_rms`` for a single input signal."""
        if signal.size < 3:
            return 1.0
        d = np.diff(times)
        d = np.where(d <= 0, _DEFAULT_DT, d)
        # First derivative (per second).
        v = np.diff(signal) / d
        # Second derivative (jerk rate per second^2 — using constant dt approx).
        a = np.diff(v) / d[1:]
        jerk_rms = float(np.sqrt(np.mean(a**2))) if a.size else 0.0
        return max(0.0, 1.0 - jerk_rms / scale)

    def racing_line_deviation(self) -> float:
        """Estimated deviation from the optimal racing line, in meters.

        Proxy: standard deviation of the steering angle while cornering
        (``|steer| > 0.3``), scaled to a meters estimate. Lower = closer to
        the optimal line. Returns 0.0 when no cornering frames are present.
        """
        steer = _field(self.frames, "steer")
        if steer.size == 0:
            return 0.0
        corner = steer[np.abs(steer) > 0.3]
        if corner.size < 2:
            return 0.0
        # Scale: ~10 m of track-width-equivalent deviation per unit steering std.
        return float(min(corner.std() * 10.0, 20.0))

    # ------------------------------------------------------------------ #
    # Corner exit speed analysis (Iter-184)
    # ------------------------------------------------------------------ #
    def corner_exit_speed_analysis(self) -> dict[str, Any]:
        """Identify corner exit speeds and flag slowest exits.

        Iter-184: detects frames where |steer| transitions from >0.3 to <0.1
        (corner exit), records the speed at that point, and reports the
        minimum / average exit speeds + the N slowest exit indices.

        Returns a dict with ``exit_speeds`` (list), ``min_exit_speed``,
        ``avg_exit_speed``, ``exit_count``, and ``slowest_exits`` (top-5).
        """
        steer = _field(self.frames, "steer")
        speed = _field(self.frames, "speed")
        if steer.size < 2 or speed.size < 2:
            return {
                "exit_speeds": [],
                "min_exit_speed": 0.0,
                "avg_exit_speed": 0.0,
                "exit_count": 0,
                "slowest_exits": [],
            }
        exit_speeds: list[float] = []
        in_corner = False
        for i in range(1, len(steer)):
            if np.abs(steer[i]) > 0.3:
                in_corner = True
            elif in_corner and np.abs(steer[i]) < 0.1:
                # Transition: corner -> straight = exit point.
                exit_speeds.append(float(speed[i]))
                in_corner = False
        if not exit_speeds:
            return {
                "exit_speeds": [],
                "min_exit_speed": 0.0,
                "avg_exit_speed": 0.0,
                "exit_count": 0,
                "slowest_exits": [],
            }
        # Sort ascending; slowest first.
        sorted_exits = sorted(exit_speeds)
        n_slowest = min(5, len(sorted_exits))
        return {
            "exit_speeds": exit_speeds,
            "min_exit_speed": float(sorted_exits[0]),
            "avg_exit_speed": float(np.mean(exit_speeds)),
            "exit_count": len(exit_speeds),
            "slowest_exits": sorted_exits[:n_slowest],
        }

    # ------------------------------------------------------------------ #
    # Sector timing analysis (Iter-185)
    # ------------------------------------------------------------------ #
    def sector_timing_analysis(self, track_length_m: float | None = None) -> dict[str, Any]:
        """Compute per-sector timing metrics from lap_distance.

        Iter-185: splits the track into 3 sectors (by equal distance), computes
        the time spent in each sector, and returns per-sector speed/steer/brake
        aggregates. Uses ``track_length_m`` if supplied; otherwise falls back to
        ``self.track_length_m``.

        Returns a dict with ``sectors`` (list of 3 dicts) and ``total_time``.
        """
        track_len = track_length_m if track_length_m is not None else self.track_length_m
        lap_dist = _field(self.frames, "lap_distance")
        times = _times(self.frames)
        speed = _field(self.frames, "speed")
        steer = _field(self.frames, "steer")
        brake = _field(self.frames, "brake")
        throttle = _field(self.frames, "throttle")

        if lap_dist.size < 2 or track_len <= 0:
            return {"sectors": [], "total_time": 0.0}

        # Sort by lap_distance for monotonic interpolation.
        order = np.argsort(lap_dist)
        ld_sorted = lap_dist[order]
        t_sorted = times[order]
        sp_sorted = speed[order]
        st_sorted = steer[order]
        b_sorted = brake[order]
        thr_sorted = throttle[order]

        # 3 equal-distance sectors.
        boundaries = [track_len / 3.0, 2.0 * track_len / 3.0, track_len]
        sectors: list[dict[str, Any]] = []
        prev_bound = 0.0
        total_time = 0.0
        for bound in boundaries:
            mask = (ld_sorted >= prev_bound) & (ld_sorted < bound + 1e-6)
            if not mask.any():
                sectors.append({
                    "time_s": 0.0,
                    "avg_speed": 0.0,
                    "max_speed": 0.0,
                    "avg_steer_abs": 0.0,
                    "avg_brake": 0.0,
                    "full_throttle_pct": 0.0,
                })
                prev_bound = bound
                continue
            t_sec = t_sorted[mask]
            sector_time = float(t_sec[-1] - t_sec[0]) if len(t_sec) >= 2 else 0.0
            total_time += sector_time
            sectors.append({
                "time_s": sector_time,
                "avg_speed": float(np.mean(sp_sorted[mask])) if len(sp_sorted[mask]) else 0.0,
                "max_speed": float(np.max(sp_sorted[mask])) if len(sp_sorted[mask]) else 0.0,
                "avg_steer_abs": float(np.mean(np.abs(st_sorted[mask]))) if len(st_sorted[mask]) else 0.0,
                "avg_brake": float(np.mean(b_sorted[mask])) if len(b_sorted[mask]) else 0.0,
                "full_throttle_pct": float(np.mean(thr_sorted[mask] > 0.95)) if len(thr_sorted[mask]) else 0.0,
            })
            prev_bound = bound
        return {"sectors": sectors, "total_time": total_time}

    # ------------------------------------------------------------------ #
    # ERS deploy mode distribution (Iter-186)
    # ------------------------------------------------------------------ #
    def ers_deploy_mode_analysis(self) -> dict[str, Any]:
        """Analyse ERS deploy mode distribution across a lap.

        Iter-186: classifies every frame into deploy mode buckets (0=off,
        1=Medium, 2=Hotlap, 3=Overtake) and returns the fraction of time
        spent in each mode plus the mode with the highest utilisation.

        Returns a dict with ``mode_fractions``, ``dominant_mode``, and
        ``n_frames``.
        """
        deploy_mode = _field_multi(self.frames, ("ers_deploy_mode", "deploy_mode"))
        if deploy_mode.size == 0:
            return {
                "mode_fractions": {},
                "dominant_mode": "unknown",
                "n_frames": 0,
            }
        modes = np.round(deploy_mode).astype(int)
        mode_names = {0: "off", 1: "Medium", 2: "Hotlap", 3: "Overtake"}
        fractions: dict[str, float] = {}
        for mode_val, name in mode_names.items():
            fractions[name] = float(np.mean(modes == mode_val))
        dominant = max(fractions.items(), key=lambda kv: kv[1])[0] if fractions else "unknown"
        return {
            "mode_fractions": fractions,
            "dominant_mode": dominant,
            "n_frames": int(modes.size),
        }

    # ------------------------------------------------------------------ #
    # Tyre degradation rate estimation (Iter-187)
    # ------------------------------------------------------------------ #
    def tyre_degradation_analysis(self) -> dict[str, Any]:
        """Estimate tyre degradation rate from wear trends.

        Iter-187: fits a linear regression to tyre_wear_* fields over
        lap_distance to estimate % wear per lap. Returns per-tyre
        degradation rates and the fastest-wearing tyre.

        Returns a dict with ``per_tyre``, ``fastest_wearing``, and ``r_squared``.
        """
        wear_fields = (
            ("fl", "tyre_wear_fl"),
            ("fr", "tyre_wear_fr"),
            ("rl", "tyre_wear_rl"),
            ("rr", "tyre_wear_rr"),
        )
        lap_dist = _field(self.frames, "lap_distance")
        if lap_dist.size < 2:
            return {"per_tyre": {}, "fastest_wearing": "", "r_squared": {}}

        per_tyre: dict[str, dict[str, float]] = {}
        best_r2 = -1.0
        fastest = ""
        for label, field in wear_fields:
            wear = _field(self.frames, field)
            if wear.size < 2:
                per_tyre[label] = {"rate_pct_per_lap": 0.0, "r_squared": 0.0}
                continue
            # Linear regression: wear ~ lap_distance.
            x = lap_dist
            y = wear
            x_mean = float(np.mean(x))
            y_mean = float(np.mean(y))
            num = float(np.sum((x - x_mean) * (y - y_mean)))
            den = float(np.sum((x - x_mean) ** 2))
            slope = num / den if den > 0 else 0.0
            # R²
            y_pred = y_mean + slope * (x - x_mean)
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - y_mean) ** 2))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            per_tyre[label] = {
                "rate_pct_per_lap": slope * self.track_length_m / 100.0,
                "r_squared": r2,
            }
            if slope > best_r2:
                best_r2 = slope
                fastest = label
        return {"per_tyre": per_tyre, "fastest_wearing": fastest, "r_squared": {k: v["r_squared"] for k, v in per_tyre.items()}}

    # ------------------------------------------------------------------ #
    # Throttle/brake overlap detection (Iter-188)
    # ------------------------------------------------------------------ #
    def throttle_brake_overlap_analysis(self) -> dict[str, Any]:
        """Detect throttle/brake overlap — simultaneous input > 0.3.

        Iter-188: counts frames where both throttle and brake > 0.3
        (inefficient driving — wastes fuel and brakes). Returns
        overlap count, fraction, and total duration.

        Returns a dict with ``overlap_count``, ``overlap_fraction``,
        ``overlap_duration_s``, and ``overlap_frames``.
        """
        throttle = _field(self.frames, "throttle")
        brake = _field(self.frames, "brake")
        if throttle.size < 1 or brake.size < 1:
            return {"overlap_count": 0, "overlap_fraction": 0.0, "overlap_duration_s": 0.0, "overlap_frames": []}
        n = min(len(throttle), len(brake))
        overlap_mask = (throttle[:n] > 0.3) & (brake[:n] > 0.3)
        overlap_count = int(np.sum(overlap_mask))
        dt = _deltas(_times(self.frames))
        overlap_duration = float(dt[:n][overlap_mask].sum()) if overlap_count > 0 else 0.0
        overlap_frames = [int(i) for i in np.where(overlap_mask)[0][:10]]
        return {
            "overlap_count": overlap_count,
            "overlap_fraction": float(overlap_count / n) if n > 0 else 0.0,
            "overlap_duration_s": overlap_duration,
            "overlap_frames": overlap_frames,
        }

    # ------------------------------------------------------------------ #
    # Corner entry/exit speed ratio (Iter-189)
    # ------------------------------------------------------------------ #
    def corner_entry_exit_ratio_analysis(self) -> dict[str, Any]:
        """Compute corner entry-vs-exit speed ratio.

        Iter-189: for each corner (|steer| > 0.3 run), find the minimum
        speed (apex) and compare entry speed (start of corner) vs exit
        speed (end of corner). Lower ratio = better exit drive.

        Returns a dict with ``ratios`` (list), ``avg_ratio``, and ``min_ratio``.
        """
        steer = _field(self.frames, "steer")
        speed = _field(self.frames, "speed")
        if steer.size < 3 or speed.size < 3:
            return {"ratios": [], "avg_ratio": 0.0, "min_ratio": 0.0}
        ratios: list[float] = []
        in_corner = False
        corner_start = 0
        for i in range(1, len(steer)):
            if not in_corner and np.abs(steer[i]) > 0.3:
                in_corner = True
                corner_start = i
            elif in_corner and np.abs(steer[i]) < 0.1:
                in_corner = False
                if i - corner_start >= 3:
                    entry_speed = float(speed[corner_start])
                    exit_speed = float(speed[i])
                    if entry_speed > 0:
                        ratios.append(exit_speed / entry_speed)
        if not ratios:
            return {"ratios": [], "avg_ratio": 0.0, "min_ratio": 0.0}
        return {
            "ratios": ratios,
            "avg_ratio": float(np.mean(ratios)),
            "min_ratio": float(np.min(ratios)),
        }

    # ------------------------------------------------------------------ #
    # Gear usage efficiency analysis (Iter-197)
    # ------------------------------------------------------------------ #
    def gear_usage_analysis(self) -> dict[str, Any]:
        """Analyse gear usage distribution and upshift RPM behaviour.

        Iter-197: computes time spent in each gear, average upshift RPM,
        and the fraction of shifts that occur near the redline (optimal
        power band). A driver who short-shifts loses acceleration.

        Returns a dict with ``gear_time_fractions``, ``avg_upshift_rpm``,
        ``redline_shift_pct``, and ``gear_counts``.
        """
        gear = _field(self.frames, "gear")
        rpm = _field(self.frames, "rpm")
        if gear.size < 2 or rpm.size < 2:
            return {
                "gear_time_fractions": {},
                "avg_upshift_rpm": 0.0,
                "redline_shift_pct": 0.0,
                "gear_counts": {},
            }
        # Time fraction per gear (gear 1-8).
        gear_int = np.round(gear).astype(int)
        gear_int = np.clip(gear_int, 0, 8)
        gear_counts: dict[int, int] = {}
        for g in gear_int:
            gear_counts[g] = gear_counts.get(g, 0) + 1
        total = gear.size
        fractions = {str(g): c / total for g, c in sorted(gear_counts.items())}
        # Upshift RPM: detect gear increases and record the RPM at shift point.
        upshift_rpms: list[float] = []
        for i in range(1, len(gear_int)):
            if gear_int[i] > gear_int[i - 1]:
                upshift_rpms.append(float(rpm[i - 1]))
        avg_upshift = float(np.mean(upshift_rpms)) if upshift_rpms else 0.0
        # Redline shift fraction: shifts at > 90% of max RPM observed.
        max_rpm = float(np.max(rpm)) if rpm.size > 0 else 13000.0
        redline_threshold = max_rpm * 0.90
        redline_count = sum(1 for r in upshift_rpms if r >= redline_threshold)
        redline_pct = redline_count / len(upshift_rpms) if upshift_rpms else 0.0
        gear_counts_serializable = {str(k): v for k, v in sorted(gear_counts.items())}
        return {
            "gear_time_fractions": fractions,
            "avg_upshift_rpm": avg_upshift,
            "redline_shift_pct": redline_pct,
            "gear_counts": gear_counts_serializable,
        }

    # ------------------------------------------------------------------ #
    # Sector-level throttle smoothness (Iter-205)
    # ------------------------------------------------------------------ #
    def sector_throttle_smoothness(self, track_length_m: float | None = None) -> dict[str, Any]:
        """Compute per-sector throttle application smoothness.

        Iter-205: splits the lap into 3 sectors and computes throttle jerk
        RMS per sector. A higher jerk means rougher throttle application —
        causes traction loss on corner exit.

        Returns a dict with ``sector_smoothness`` (list of 3 floats, 0-1)
        and ``worst_sector``.
        """
        track_len = track_length_m if track_length_m is not None else self.track_length_m
        throttle = _field(self.frames, "throttle")
        lap_dist = _field(self.frames, "lap_distance")
        if throttle.size < 3 or lap_dist.size < 2:
            return {"sector_smoothness": [1.0, 1.0, 1.0], "worst_sector": -1}
        # Sort by lap_distance.
        order = np.argsort(lap_dist)
        ld_sorted = lap_dist[order]
        thr_sorted = throttle[order]
        boundaries = [track_len / 3.0, 2.0 * track_len / 3.0, track_len]
        smoothness: list[float] = []
        prev = 0.0
        for bound in boundaries:
            mask = (ld_sorted >= prev) & (ld_sorted < bound + 1e-6)
            prev = bound
            sector_thr = thr_sorted[mask]
            if sector_thr.size < 3:
                smoothness.append(1.0)
                continue
            # Throttle jerk: second difference of throttle.
            jerk = np.diff(np.diff(sector_thr))
            jerk_rms = float(np.sqrt(np.mean(jerk**2))) if jerk.size else 0.0
            s = max(0.0, 1.0 - jerk_rms / 2.0)
            smoothness.append(s)
        worst = int(np.argmin(smoothness)) if smoothness else -1
        return {"sector_smoothness": smoothness, "worst_sector": worst}

    # ------------------------------------------------------------------ #
    # Engine braking / lift-off deceleration (Iter-199)
    # ------------------------------------------------------------------ #
    def engine_braking_analysis(self) -> dict[str, Any]:
        """Analyse lift-off deceleration (engine braking) on throttle release.

        Iter-199: measures the average deceleration rate (km/h per s) when
        throttle transitions from >0.8 to <0.1 without brake application.
        Strong engine braking helps turn-in without wearing physical brakes.

        Returns a dict with ``avg_lift_off_decel_kmh_s``, ``lift_off_count``,
        and ``max_decel_kmh_s``.
        """
        throttle = _field(self.frames, "throttle")
        brake = _field(self.frames, "brake")
        speed = _field(self.frames, "speed")
        times = _times(self.frames)
        if throttle.size < 3 or speed.size < 3:
            return {"avg_lift_off_decel_kmh_s": 0.0, "lift_off_count": 0, "max_decel_kmh_s": 0.0}
        decel_rates: list[float] = []
        for i in range(1, len(throttle) - 1):
            if throttle[i - 1] > 0.8 and throttle[i] < 0.1 and brake[i] < 0.1:
                # Found lift-off event. Measure deceleration over next 0.5s.
                dt_total = 0.0
                j = i
                v_start = float(speed[i])
                while j + 1 < len(speed) and dt_total < 0.5:
                    dt_total += float(times[j + 1] - times[j]) if j + 1 < len(times) else 1.0 / 60.0
                    j += 1
                if dt_total > 0.05:
                    v_end = float(speed[j])
                    decel = (v_start - v_end) / dt_total
                    if decel > 0:
                        decel_rates.append(decel)
        return {
            "avg_lift_off_decel_kmh_s": float(np.mean(decel_rates)) if decel_rates else 0.0,
            "lift_off_count": len(decel_rates),
            "max_decel_kmh_s": float(max(decel_rates)) if decel_rates else 0.0,
        }

    # ------------------------------------------------------------------ #
    # Grip utilization ratio (Iter-211)
    # ------------------------------------------------------------------ #
    def grip_utilization_analysis(self) -> dict[str, Any]:
        """Compute front vs rear grip utilization ratio.

        Iter-211: compares max g_lat at high speed vs low speed to determine
        if the car is aero-limited or mechanical-grip-limited. Also computes
        the ratio of lateral g to longitudinal g (braking) to assess if the
        driver is using the full traction circle.

        Returns a dict with ``front_rear_ratio``, ``aero_mech_ratio``, and
        ``traction_circle_utilization``.
        """
        g_lat = _field(self.frames, "g_lat")
        g_long = _field(self.frames, "g_long")
        speed = _field(self.frames, "speed")
        if g_lat.size < 2 or g_long.size < 2 or speed.size < 2:
            return {"front_rear_ratio": 0.0, "aero_mech_ratio": 0.0, "traction_circle_utilization": 0.0}
        # Front/rear g_lat balance proxy: compare high-g (>3G) vs low-g (<2G) frequency.
        high_g = np.sum(np.abs(g_lat) > 3.0)
        low_g = np.sum(np.abs(g_lat) < 2.0)
        front_rear_ratio = float(high_g) / max(float(low_g), 1.0)
        # Aero vs mechanical: max g_lat at high speed / max g_lat at low speed.
        high_spd_mask = speed > 250.0
        low_spd_mask = speed < 150.0
        high_g_max = float(np.max(np.abs(g_lat[high_spd_mask]))) if high_spd_mask.any() else 0.0
        low_g_max = float(np.max(np.abs(g_lat[low_spd_mask]))) if low_spd_mask.any() else 0.0
        aero_mech = high_g_max / low_g_max if low_g_max > 0 else 0.0
        # Traction circle utilization: RMS of (g_lat, g_long) vs max possible.
        combined = np.sqrt(g_lat**2 + g_long**2)
        max_combined = float(np.max(combined))
        traction_util = max_combined / 5.0  # typical F1 peak ~5G
        return {
            "front_rear_ratio": front_rear_ratio,
            "aero_mech_ratio": aero_mech,
            "traction_circle_utilization": min(traction_util, 1.0),
        }

    # ------------------------------------------------------------------ #
    # Downforce balance estimation (Iter-190)
    # ------------------------------------------------------------------ #
    def downforce_balance_analysis(self) -> dict[str, Any]:
        """Estimate aero balance from g_lat distribution.

        Iter-190: compares g_lat at high speed (>250 km/h) corners vs
        low speed (<150 km/h) corners. If low-speed g_lat is low relative
        to high-speed, the car lacks mechanical grip (suspension/tyres).
        If high-speed g_lat is low, the car lacks downforce.

        Returns a dict with ``high_speed_g_lat_avg``, ``low_speed_g_lat_avg``,
        ``aero_balance_ratio``, and ``diagnosis``.
        """
        speed = _field(self.frames, "speed")
        g_lat = _field(self.frames, "g_lat")
        steer = _field(self.frames, "steer")
        if speed.size < 2 or g_lat.size < 2:
            return {"high_speed_g_lat_avg": 0.0, "low_speed_g_lat_avg": 0.0, "aero_balance_ratio": 0.0, "diagnosis": "insufficient data"}
        corner_mask = np.abs(steer) > 0.3
        if not corner_mask.any():
            return {"high_speed_g_lat_avg": 0.0, "low_speed_g_lat_avg": 0.0, "aero_balance_ratio": 0.0, "diagnosis": "no cornering data"}
        high_speed_mask = corner_mask & (speed > 250.0)
        low_speed_mask = corner_mask & (speed < 150.0)
        high_g = float(np.mean(np.abs(g_lat[high_speed_mask]))) if high_speed_mask.any() else 0.0
        low_g = float(np.mean(np.abs(g_lat[low_speed_mask]))) if low_speed_mask.any() else 0.0
        ratio = high_g / low_g if low_g > 0 else 0.0
        if ratio > 1.8:
            diag = "aero-dominant: strong downforce, may need more mechanical grip"
        elif ratio < 1.1:
            diag = "mechanical-dominant: good mechanical grip, may need more downforce"
        else:
            diag = "balanced aero/mechanical grip"
        return {
            "high_speed_g_lat_avg": high_g,
            "low_speed_g_lat_avg": low_g,
            "aero_balance_ratio": ratio,
            "diagnosis": diag,
        }

    # ------------------------------------------------------------------ #
    # Active aero usage (F1 2026 X-Mode / Z-Mode)
    # ------------------------------------------------------------------ #
    def active_aero_usage_analysis(self) -> dict[str, Any]:
        """Analyze F1 2026 active aero (X-Mode / Z-Mode) usage across frames.

        X-mode = low-drag (straight-line speed), Z-mode = high-downforce
        (cornering grip). Reports the fraction of frames where each mode is
        deployed (aero position > 0.5) plus the mean aero position.
        """
        x = _field(self.frames, "active_aero_x")
        z = _field(self.frames, "active_aero_z")
        n = int(x.size)
        x_active = int(np.count_nonzero(x > 0.5)) if n else 0
        z_active = int(np.count_nonzero(z > 0.5)) if n else 0
        return {
            "total_frames": n,
            "x_mode_frames": x_active,
            "z_mode_frames": z_active,
            "x_mode_fraction": x_active / n if n else 0.0,
            "z_mode_fraction": z_active / n if n else 0.0,
            "mean_aero_x": float(np.mean(x)) if n else 0.0,
            "mean_aero_z": float(np.mean(z)) if n else 0.0,
        }

    # ------------------------------------------------------------------ #
    # Mechanical grip trend (Iter-216)
    # ------------------------------------------------------------------ #
    def mechanical_grip_trend_analysis(self) -> dict[str, Any]:
        """Track mechanical grip degradation over a lap.

        Iter-216: computes the slope of g_lat vs lap_distance for low-speed
        corners (<150 km/h). A negative slope means mechanical grip is
        fading (tyre overheating / graining). A positive slope means the
        tyres are coming into their window.

        Returns a dict with ``low_speed_g_lat_slope``, ``low_speed_g_lat_r2``,
        and ``trend`` (``"decaying"`` / ``"stable"`` / ``"improving"``).
        """
        speed = _field(self.frames, "speed")
        g_lat = _field(self.frames, "g_lat")
        steer = _field(self.frames, "steer")
        lap_dist = _field(self.frames, "lap_distance")
        if speed.size < 2 or g_lat.size < 2 or steer.size < 2 or lap_dist.size < 2:
            return {"low_speed_g_lat_slope": 0.0, "low_speed_g_lat_r2": 0.0, "trend": "insufficient data"}
        corner_mask = np.abs(steer) > 0.3
        low_speed_mask = corner_mask & (speed < 150.0)
        if not low_speed_mask.any() or low_speed_mask.sum() < 3:
            return {"low_speed_g_lat_slope": 0.0, "low_speed_g_lat_r2": 0.0, "trend": "insufficient low-speed cornering data"}
        g_low = np.abs(g_lat[low_speed_mask])
        d_low = lap_dist[low_speed_mask]
        x_mean = float(np.mean(d_low))
        y_mean = float(np.mean(g_low))
        num = float(np.sum((d_low - x_mean) * (g_low - y_mean)))
        den = float(np.sum((d_low - x_mean) ** 2))
        slope = num / den if den > 0 else 0.0
        y_pred = y_mean + slope * (d_low - x_mean)
        ss_res = float(np.sum((g_low - y_pred) ** 2))
        ss_tot = float(np.sum((g_low - y_mean) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        if slope < -0.001:
            trend = "decaying"
        elif slope > 0.001:
            trend = "improving"
        else:
            trend = "stable"
        return {
            "low_speed_g_lat_slope": slope,
            "low_speed_g_lat_r2": r2,
            "trend": trend,
        }

    # ------------------------------------------------------------------ #
    # Brake temperature balance (Iter-221)
    # ------------------------------------------------------------------ #
    def brake_temp_balance_analysis(self) -> dict[str, Any]:
        """Compute front/rear brake temperature balance.

        Iter-221: compares front vs rear brake temperatures to detect bias
        issues. A large front-rear temp difference indicates the brake bias
        is too far forward or rearward, causing uneven thermal load.

        Returns a dict with ``front_avg``, ``rear_avg``, ``f_r_ratio``,
        ``diagnosis``, and ``front_peak`` / ``rear_peak``.
        """
        f_temp_fl = _field(self.frames, "brake_temp_fl")
        f_temp_fr = _field(self.frames, "brake_temp_fr")
        f_temp_rl = _field(self.frames, "brake_temp_rl")
        f_temp_rr = _field(self.frames, "brake_temp_rr")
        if f_temp_fl.size < 2 or f_temp_rl.size < 2:
            return {"front_avg": 0.0, "rear_avg": 0.0, "f_r_ratio": 0.0, "diagnosis": "insufficient data", "front_peak": 0.0, "rear_peak": 0.0}
        front_avg = float((np.mean(f_temp_fl) + np.mean(f_temp_fr)) / 2.0)
        rear_avg = float((np.mean(f_temp_rl) + np.mean(f_temp_rr)) / 2.0)
        ratio = front_avg / rear_avg if rear_avg > 0 else 0.0
        if ratio > 1.3:
            diag = "front-bias: 前刹车温度显著高于后刹, 刹车偏置偏前"
        elif ratio < 0.7:
            diag = "rear-bias: 后刹车温度显著高于前刹, 刹车偏置偏后"
        else:
            diag = "balanced: 前后刹车温度分布均匀"
        front_peak = float(max(np.max(f_temp_fl), np.max(f_temp_fr)))
        rear_peak = float(max(np.max(f_temp_rl), np.max(f_temp_rr)))
        return {
            "front_avg": front_avg,
            "rear_avg": rear_avg,
            "f_r_ratio": ratio,
            "diagnosis": diag,
            "front_peak": front_peak,
            "rear_peak": rear_peak,
        }

    # ------------------------------------------------------------------ #
    # Tyre temperature gradient (Iter-224)
    # ------------------------------------------------------------------ #
    def tyre_temp_gradient_analysis(self) -> dict[str, Any]:
        """Compute tyre surface temperature gradients across axles and sides.

        Iter-224: analyses left-right and front-rear tyre surface temperature
        differences to detect setup asymmetry, overheating, or pressure issues.
        Large side-to-side differences indicate camber or pressure imbalance;
        large front-rear differences indicate aero or brake balance issues.

        Returns a dict with ``left_right_delta``, ``front_rear_delta``,
        ``max_inner_surface_delta``, and ``diagnosis``.
        """
        ts_fl = _field(self.frames, "tyre_temp_fl")
        ts_fr = _field(self.frames, "tyre_temp_fr")
        ts_rl = _field(self.frames, "tyre_temp_rl")
        ts_rr = _field(self.frames, "tyre_temp_rr")
        ti_fl = _field(self.frames, "tyre_inner_temp_fl")
        ti_fr = _field(self.frames, "tyre_inner_temp_fr")
        ti_rl = _field(self.frames, "tyre_inner_temp_rl")
        ti_rr = _field(self.frames, "tyre_inner_temp_rr")
        if ts_fl.size < 2 or ts_fr.size < 2:
            return {
                "left_right_delta": 0.0, "front_rear_delta": 0.0,
                "max_inner_surface_delta": 0.0, "diagnosis": "insufficient data",
            }
        # Left-right delta: average of (FL - FR) and (RL - RR).
        lr_front = float(np.mean(ts_fl - ts_fr))
        lr_rear = float(np.mean(ts_rl - ts_rr))
        left_right_delta = (lr_front + lr_rear) / 2.0
        # Front-rear delta: average of (FL - RL) and (FR - RR).
        fr_left = float(np.mean(ts_fl - ts_rl))
        fr_right = float(np.mean(ts_fr - ts_rr))
        front_rear_delta = (fr_left + fr_right) / 2.0
        # Inner-surface delta: how much hotter the inner carcass is vs surface.
        inner_surface_deltas: list[float] = []
        for surf, inner in [(ts_fl, ti_fl), (ts_fr, ti_fr), (ts_rl, ti_rl), (ts_rr, ti_rr)]:
            if inner.size >= 2:
                inner_surface_deltas.append(float(np.mean(inner - surf)))
        max_is_delta = max(inner_surface_deltas) if inner_surface_deltas else 0.0
        # Diagnosis
        diag_parts: list[str] = []
        if abs(left_right_delta) > 3.0:
            side = "左" if left_right_delta > 0 else "右"
            diag_parts.append(f"{side}侧轮胎温度偏高 {abs(left_right_delta):.1f}°C")
        if abs(front_rear_delta) > 5.0:
            axle = "前" if front_rear_delta > 0 else "后"
            diag_parts.append(f"{axle}轴轮胎温度偏高 {abs(front_rear_delta):.1f}°C")
        if max_is_delta > 5.0:
            diag_parts.append(f"胎内温度过高 (inner-surface delta={max_is_delta:.1f}°C)")
        return {
            "left_right_delta": left_right_delta,
            "front_rear_delta": front_rear_delta,
            "max_inner_surface_delta": max_is_delta,
            "diagnosis": "; ".join(diag_parts) if diag_parts else "balanced",
        }

    # ------------------------------------------------------------------ #
    # Fuel consumption per sector (Iter-230, Iter-243)
    # ------------------------------------------------------------------ #
    def fuel_per_sector_analysis(self, track_length_m: float | None = None) -> dict[str, Any]:
        """Compute fuel consumption per sector.

        Iter-230: splits the lap into 3 sectors and estimates fuel consumed
        per sector using throttle × speed as a proxy for fuel flow. Returns
        per-sector fuel estimates and the sector with highest consumption.

        Iter-243: adds total lap fuel proxy, per-sector fuel fractions, and
        estimated fuel consumption rate per km for fuel economy monitoring.
        """
        track_len = track_length_m if track_length_m is not None else self.track_length_m
        throttle = _field(self.frames, "throttle")
        speed = _field(self.frames, "speed")
        lap_dist = _field(self.frames, "lap_distance")
        if throttle.size < 2 or speed.size < 2 or lap_dist.size < 2:
            return {"sectors": [], "highest_consumption_sector": -1, "total_fuel_proxy": 0.0, "fuel_proxy_per_km": 0.0}
        n = min(len(throttle), len(speed), len(lap_dist))
        order = np.argsort(lap_dist[:n])
        thr_s = throttle[:n][order]
        spd_s = speed[:n][order]
        ld_s = lap_dist[:n][order]
        boundaries = [track_len / 3.0, 2.0 * track_len / 3.0, track_len]
        sectors = []
        prev = 0.0
        highest = -1
        highest_val = 0.0
        total_fuel = 0.0
        for si, bound in enumerate(boundaries):
            mask = (ld_s >= prev) & (ld_s < bound + 1e-6)
            prev = bound
            if not mask.any():
                sectors.append({"sector": si + 1, "fuel_estimate": 0.0, "fuel_fraction": 0.0, "avg_throttle": 0.0})
                continue
            # Fuel proxy: throttle × speed × distance-step (higher = more fuel).
            fuel_proxy = float(np.sum(thr_s[mask] * spd_s[mask]))
            avg_thr = float(np.mean(thr_s[mask]))
            total_fuel += fuel_proxy
            sectors.append({"sector": si + 1, "fuel_estimate": fuel_proxy, "fuel_fraction": 0.0, "avg_throttle": avg_thr})
            if fuel_proxy > highest_val:
                highest_val = fuel_proxy
                highest = si
        # Compute per-sector fuel fractions and per-km rate.
        if total_fuel > 0:
            for se in sectors:
                se["fuel_fraction"] = se["fuel_estimate"] / total_fuel
        fuel_per_km = total_fuel / (track_len / 1000.0) if track_len > 0 else 0.0
        return {
            "sectors": sectors,
            "highest_consumption_sector": highest,
            "total_fuel_proxy": total_fuel,
            "fuel_proxy_per_km": fuel_per_km,
        }

    # ------------------------------------------------------------------ #
    # ERS deploy per sector (Iter-236)
    # ------------------------------------------------------------------ #
    def ers_sector_analysis(self, track_length_m: float | None = None) -> dict[str, Any]:
        """Compute ERS deploy efficiency per sector.

        Iter-236: splits the lap into 3 sectors and computes ERS deploy vs
        recover ratio per sector. Identifies sectors where ERS is over/under
        deployed relative to recovery opportunities.
        """
        track_len = track_length_m if track_length_m is not None else self.track_length_m
        deploy = _field_multi(self.frames, ("ers_deployed_this_lap", "ers_deploy", "ers_deployed"))
        brake = _field(self.frames, "brake")
        lap_dist = _field(self.frames, "lap_distance")
        if deploy.size < 2 or brake.size < 2 or lap_dist.size < 2:
            return {"sectors": [], "worst_sector": -1}
        n = min(len(deploy), len(brake), len(lap_dist))
        order = np.argsort(lap_dist[:n])
        dep_s = deploy[:n][order]
        brk_s = brake[:n][order]
        ld_s = lap_dist[:n][order]
        boundaries = [track_len / 3.0, 2.0 * track_len / 3.0, track_len]
        sectors = []
        prev = 0.0
        worst = -1
        worst_eff = 1.0
        for si, bound in enumerate(boundaries):
            mask = (ld_s >= prev) & (ld_s < bound + 1e-6)
            prev = bound
            if not mask.any():
                sectors.append({"sector": si + 1, "deploy_total": 0.0, "recover_total": 0.0, "efficiency": 0.0})
                continue
            dep_total = float(np.sum(dep_s[mask]))
            rec_total = float(np.sum(np.clip(brk_s[mask] - 0.3, 0.0, 1.0)))
            eff = dep_total / rec_total if rec_total > 0 else 0.0
            sectors.append({"sector": si + 1, "deploy_total": dep_total, "recover_total": rec_total, "efficiency": eff})
            if eff < worst_eff:
                worst_eff = eff
                worst = si
        return {"sectors": sectors, "worst_sector": worst}

    # ------------------------------------------------------------------ #
    # Grip consistency across sectors (Iter-239)
    # ------------------------------------------------------------------ #
    def grip_consistency_analysis(self, track_length_m: float | None = None) -> dict[str, Any]:
        """Compute grip consistency (g_lat std) per sector.

        Iter-239: measures how consistent the lateral g is within each sector.
        High std means the driver is struggling to maintain consistent cornering
        force — could indicate tyre graining, setup imbalance, or driver error.
        """
        track_len = track_length_m if track_length_m is not None else self.track_length_m
        g_lat = _field(self.frames, "g_lat")
        steer = _field(self.frames, "steer")
        lap_dist = _field(self.frames, "lap_distance")
        if g_lat.size < 2 or steer.size < 2 or lap_dist.size < 2:
            return {"sectors": [], "worst_consistency_sector": -1, "overall_std": 0.0}
        n = min(len(g_lat), len(steer), len(lap_dist))
        corner_mask = np.abs(steer[:n]) > 0.3
        order = np.argsort(lap_dist[:n])
        g_s = np.abs(g_lat[:n][order])
        ld_s = lap_dist[:n][order]
        cm_s = corner_mask[order]
        boundaries = [track_len / 3.0, 2.0 * track_len / 3.0, track_len]
        sectors = []
        prev = 0.0
        worst = -1
        worst_std = 0.0
        for si, bound in enumerate(boundaries):
            mask = (ld_s >= prev) & (ld_s < bound + 1e-6)
            prev = bound
            corner_in_sector = mask & cm_s
            if not corner_in_sector.any():
                sectors.append({"sector": si + 1, "g_lat_std": 0.0, "g_lat_mean": 0.0, "n_corner_frames": 0})
                continue
            g_sec = g_s[corner_in_sector]
            g_std = float(np.std(g_sec))
            g_mean = float(np.mean(g_sec))
            n_frames = int(np.sum(corner_in_sector))
            sectors.append({"sector": si + 1, "g_lat_std": g_std, "g_lat_mean": g_mean, "n_corner_frames": n_frames})
            if g_std > worst_std:
                worst_std = g_std
                worst = si
        overall_std = float(np.std(np.abs(g_lat[:n][corner_mask]))) if corner_mask.any() else 0.0
        return {"sectors": sectors, "worst_consistency_sector": worst, "overall_std": overall_std}

    # ------------------------------------------------------------------ #
    # ERS recovery efficiency analysis (Iter-245)
    # ------------------------------------------------------------------ #
    def ers_recovery_efficiency_analysis(self) -> dict[str, Any]:
        """Analyse MGU-K recovery efficiency during braking zones.

        Iter-245: identifies individual braking events (brake > 0.3 runs),
        computes the average ERS deploy during each braking event (recovery
        is proportional to brake pressure), and reports per-event recovery
        scores and overall recovery efficiency.

        Returns a dict with ``recovery_events``, ``avg_recovery_per_event``,
        ``total_recovery_score``, ``recovery_event_count``, and
        ``braking_zone_count``.
        """
        brake = _field(self.frames, "brake")
        deploy = _field_multi(self.frames, ("ers_deployed_this_lap", "ers_deploy", "ers_deployed"))
        if brake.size < 2:
            return {
                "recovery_events": [],
                "avg_recovery_per_event": 0.0,
                "total_recovery_score": 0.0,
                "recovery_event_count": 0,
                "braking_zone_count": 0,
            }
        # Find braking zones: contiguous runs of brake > 0.3.
        brake_mask = brake > 0.3
        # Build runs manually.
        brake_runs: list[tuple[int, int]] = []
        in_run = False
        run_start = 0
        for i, m in enumerate(brake_mask):
            if m and not in_run:
                in_run = True
                run_start = i
            elif not m and in_run:
                in_run = False
                brake_runs.append((run_start, i))
        if in_run:
            brake_runs.append((run_start, len(brake_mask)))
        n = min(len(brake), len(deploy))
        recovery_events: list[dict[str, float]] = []
        total_recovery = 0.0
        for start, end in brake_runs:
            if start >= n:
                continue
            end = min(end, n)
            if end <= start + 1:
                continue
            # Recovery is proportional to brake pressure; deploy during
            # braking is typically 0 (MGU-K harvests, doesn't deploy).
            brake_intensity = float(np.sum(brake[start:end]))
            deploy_during = float(np.sum(deploy[start:end]))
            # Recovery efficiency: 1.0 means no deploy during braking
            # (pure recovery), 0.0 means full deploy during braking.
            rec_eff = 1.0 - (deploy_during / max(brake_intensity, 1.0))
            rec_eff = max(0.0, min(1.0, rec_eff))
            recovery_events.append({
                "start_frame": start,
                "end_frame": end,
                "brake_intensity": brake_intensity,
                "deploy_during_brake": deploy_during,
                "recovery_efficiency": rec_eff,
            })
            total_recovery += rec_eff
        recovery_count = len(recovery_events)
        avg_rec = total_recovery / recovery_count if recovery_count > 0 else 0.0
        return {
            "recovery_events": recovery_events,
            "avg_recovery_per_event": avg_rec,
            "total_recovery_score": total_recovery,
            "recovery_event_count": recovery_count,
            "braking_zone_count": len(brake_runs),
        }


# --------------------------------------------------------------------------- #
# PerformanceBenchmark
# --------------------------------------------------------------------------- #
class PerformanceBenchmark:
    """Benchmark a lap's metrics against track-specific references.

    ``track_id`` selects a reference set via :data:`TRACK_REFERENCES`; if the
    id is unknown the ``"medium"`` reference is used.
    """

    def __init__(self, track_id: str) -> None:
        self.track_id = track_id
        track = TRACKS_BY_ID.get(track_id)
        self.track_type: str = track.track_type if track is not None else "medium"
        self.reference: dict[str, float] = TRACK_REFERENCES.get(
            self.track_type, _DEFAULT_REFERENCE
        )

    # ------------------------------------------------------------------ #
    def benchmark(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Compare ``metrics`` (output of :meth:`TelemetryAnalytics.compute_all`)
        to the track references.

        Returns a dict with ``speed_score``, ``consistency_score``,
        ``efficiency_score``, ``overall_score`` (each in ``[0, 1]``),
        ``grade`` (one of S/A/B/C/D), plus ``strengths`` and ``weaknesses``
        lists of strings.
        """
        ref = self.reference
        speed_m = metrics.get("speed", {}) or {}
        throttle_m = metrics.get("throttle", {}) or {}
        smoothing = _to_float(metrics.get("lap_smoothing_score"), 0.0)
        deviation = _to_float(metrics.get("racing_line_deviation"), 0.0)
        ers_m = metrics.get("ers", {}) or {}
        drs_m = metrics.get("drs", {}) or {}

        # ----- Speed score: v_max and v_avg vs reference targets. --------
        v_max = _to_float(speed_m.get("v_max"), 0.0)
        v_avg = _to_float(speed_m.get("v_avg"), 0.0)
        speed_score = self._ratio_score(
            v_max, ref["v_max_target"]
        ) * 0.5 + self._ratio_score(v_avg, ref["v_avg_target"]) * 0.5

        # ----- Consistency: smoothing + low racing-line deviation. -------
        smoothing_score = max(0.0, min(1.0, smoothing))
        # Deviation: 0 m → 1.0; 5 m+ → 0.0.
        deviation_score = max(0.0, 1.0 - deviation / 5.0)
        consistency_score = smoothing_score * 0.6 + deviation_score * 0.4

        # ----- Efficiency: full_throttle + ERS efficiency + DRS usage. ---
        full_thr = _to_float(throttle_m.get("full_throttle_pct"), 0.0)
        thr_score = self._ratio_score(
            full_thr, ref["full_throttle_pct_target"]
        )
        ers_eff = _to_float(ers_m.get("ers_efficiency"), 0.0)
        # ERS efficiency is "good" in a moderate band; cap at 1.0 and floor at 0.
        ers_score = max(0.0, min(1.0, ers_eff / 2.0))
        drs_acts = _to_float(drs_m.get("drs_activations"), 0.0)
        drs_score = self._ratio_score(drs_acts, ref["drs_zones"])
        efficiency_score = (thr_score + ers_score + drs_score) / 3.0

        # Clamp all sub-scores to [0, 1].
        speed_score = max(0.0, min(1.0, speed_score))
        consistency_score = max(0.0, min(1.0, consistency_score))
        efficiency_score = max(0.0, min(1.0, efficiency_score))

        overall = (
            speed_score * 0.4 + consistency_score * 0.35 + efficiency_score * 0.25
        )
        overall = max(0.0, min(1.0, overall))
        grade = self._grade(overall)

        # Strengths / weaknesses: compare each sub-metric against a 0.7 / 0.4
        # threshold to label it a strength or weakness respectively.
        strengths: list[str] = []
        weaknesses: list[str] = []
        if speed_score >= 0.7:
            strengths.append("top speed")
        elif speed_score < 0.4:
            weaknesses.append("top speed")
        if consistency_score >= 0.7:
            strengths.append("input smoothness")
        elif consistency_score < 0.4:
            weaknesses.append("input smoothness")
        if efficiency_score >= 0.7:
            strengths.append("energy efficiency")
        elif efficiency_score < 0.4:
            weaknesses.append("energy efficiency")
        if thr_score >= 0.7:
            strengths.append("full-throttle commitment")
        elif thr_score < 0.4:
            weaknesses.append("full-throttle commitment")
        if drs_score >= 0.7:
            strengths.append("DRS usage")
        elif drs_score < 0.4:
            weaknesses.append("DRS usage")

        return {
            "speed_score": speed_score,
            "consistency_score": consistency_score,
            "efficiency_score": efficiency_score,
            "overall_score": overall,
            "grade": grade,
            "strengths": strengths,
            "weaknesses": weaknesses,
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _ratio_score(value: float, target: float) -> float:
        """``value / target`` clamped to ``[0, 1]`` (1.0 = met/exceeded)."""
        if target <= 0:
            return 0.0
        return max(0.0, min(1.0, value / target))

    @staticmethod
    def _grade(overall: float) -> str:
        """Letter grade from an overall score in ``[0, 1]``."""
        if overall >= 0.9:
            return "S"
        if overall >= 0.8:
            return "A"
        if overall >= 0.65:
            return "B"
        if overall >= 0.5:
            return "C"
        return "D"


# --------------------------------------------------------------------------- #
# AnomalyDetector
# --------------------------------------------------------------------------- #
class AnomalyDetector:
    """Detect anomalies in a list of telemetry frames.

    Each :meth:`detect` call returns a list of anomaly dicts of the form
    ``{"frame_t": float, "type": str, "severity": str, "description": str}``.
    """

    def __init__(self) -> None:
        # Thresholds (kept as instance attributes so callers may override).
        self.sudden_decel_kmh: float = 50.0
        self.sudden_decel_window_s: float = 0.1
        self.g_lat_threshold: float = 5.0
        self.g_long_threshold: float = 4.0
        self.redline_rpm: float = 13000.0
        self.redline_window_s: float = 1.0
        self.brake_throttle_threshold: float = 0.5
        self.steer_threshold: float = 0.95
        self.steer_window_s: float = 0.5
        self.ers_overdeploy_threshold: float = 0.9
        self.ers_overdeploy_window_s: float = 2.0
        # Iter-129: sensor-stuck / outlier / lap-time-jump thresholds.
        # sensor_stuck applies to *physical* channels only (speed/rpm/g_lat/
        # g_long) — driver inputs (throttle/brake/steer) can legitimately be
        # held constant for many frames. A frozen physical signal almost
        # always indicates a sensor or stream fault.
        self.sensor_stuck_channels: tuple[str, ...] = (
            "speed", "rpm", "g_lat", "g_long",
        )
        self.sensor_stuck_min_frames: int = 60  # 1.0 s at 60 Hz.
        # Speed outliers: real F1 cars never exceed ~375 km/h (Monza trap).
        # Negative speed is non-physical.
        self.speed_outlier_max_kmh: float = 380.0
        self.speed_outlier_min_kmh: float = 0.0
        # Lap-time jumps: lap_time should be monotonic and advance by ~dt per
        # frame. A negative delta or a > 5 s jump signals a timestamp glitch
        # or session reset mid-lap.
        self.lap_time_jump_max_s: float = 5.0

    # ------------------------------------------------------------------ #
    def detect(self, frames: list[dict]) -> list[dict[str, Any]]:
        """Return a list of anomaly dicts (one per detected event)."""
        anomalies: list[dict[str, Any]] = []
        if not frames:
            return anomalies
        times = _times(frames)
        dt = _deltas(times)
        speed = _field(frames, "speed")
        g_lat = _field(frames, "g_lat")
        g_long = _field(frames, "g_long")
        rpm = _field(frames, "rpm")
        throttle = _field(frames, "throttle")
        brake = _field(frames, "brake")
        steer = _field(frames, "steer")
        ers = _field_multi(frames, ("ers_deployed_this_lap", "ers_deploy", "ers_deployed"))

        n = len(frames)

        # 1) Sudden deceleration: speed drop > threshold within window.
        for i in range(n):
            # Walk forward from i until cumulative time exceeds window.
            t_budget = self.sudden_decel_window_s
            j = i
            while j + 1 < n and t_budget > 0:
                t_budget -= float(dt[j])
                j += 1
                drop = float(speed[i] - speed[j])
                if drop > self.sudden_decel_kmh:
                    sev = "high" if drop > 100.0 else "medium"
                    anomalies.append({
                        "frame_t": float(times[i]),
                        "type": "sudden_deceleration",
                        "severity": sev,
                        "description": (
                            f"speed dropped {drop:.1f} km/h in "
                            f"{float(times[j] - times[i]):.2f}s"
                        ),
                    })
                    break
                if t_budget <= 0:
                    break

        # 2) Extreme g: |g_lat| > thr OR |g_long| > thr.
        for i in range(n):
            gl = float(g_lat[i])
            glong = float(g_long[i])
            if abs(gl) > self.g_lat_threshold or abs(glong) > self.g_long_threshold:
                which = "lateral" if abs(gl) > abs(glong) else "longitudinal"
                sev = "high"
                anomalies.append({
                    "frame_t": float(times[i]),
                    "type": "extreme_g",
                    "severity": sev,
                    "description": (
                        f"extreme {which} g: g_lat={gl:.2f}, g_long={glong:.2f}"
                    ),
                })

        # 3) Sustained redline: rpm > redline for > window.
        redline_mask = rpm > self.redline_rpm
        for start, end in self._runs(redline_mask):
            duration = float(times[end - 1] - times[start]) + float(dt[end - 1])
            if duration > self.redline_window_s:
                anomalies.append({
                    "frame_t": float(times[start]),
                    "type": "sustained_redline",
                    "severity": "medium",
                    "description": (
                        f"rpm > {self.redline_rpm:.0f} for {duration:.2f}s"
                    ),
                })

        # 4) Brake and throttle: both > threshold simultaneously.
        bt_mask = (throttle > self.brake_throttle_threshold) & (
            brake > self.brake_throttle_threshold
        )
        for start, end in self._runs(bt_mask):
            anomalies.append({
                "frame_t": float(times[start]),
                "type": "brake_and_throttle",
                "severity": "low",
                "description": (
                    f"throttle and brake both > {self.brake_throttle_threshold:.1f}"
                    f" for {float(times[end - 1] - times[start]) + float(dt[end - 1]):.2f}s"
                ),
            })

        # 5) Extreme steering: |steer| > threshold for > window.
        steer_mask = np.abs(steer) > self.steer_threshold
        for start, end in self._runs(steer_mask):
            duration = float(times[end - 1] - times[start]) + float(dt[end - 1])
            if duration > self.steer_window_s:
                anomalies.append({
                    "frame_t": float(times[start]),
                    "type": "extreme_steering",
                    "severity": "medium",
                    "description": (
                        f"|steer| > {self.steer_threshold:.2f} for {duration:.2f}s"
                    ),
                })

        # 6) ERS overdeploy: deploy > threshold sustained > window.
        ers_mask = ers > self.ers_overdeploy_threshold
        for start, end in self._runs(ers_mask):
            duration = float(times[end - 1] - times[start]) + float(dt[end - 1])
            if duration > self.ers_overdeploy_window_s:
                anomalies.append({
                    "frame_t": float(times[start]),
                    "type": "ers_overdeploy",
                    "severity": "medium",
                    "description": (
                        f"ers_deploy > {self.ers_overdeploy_threshold:.1f} for"
                        f" {duration:.2f}s"
                    ),
                })

        # 7) Iter-129: sensor stuck — a *physical* channel holds an identical
        # value for >= sensor_stuck_min_frames consecutive samples. Driver
        # inputs (throttle/brake/steer) are excluded because they can
        # legitimately be held constant for long stretches (e.g. flat-out on
        # a straight). A frozen speed/rpm/g_lat/g_long almost always means a
        # sensor or stream fault.
        for channel in self.sensor_stuck_channels:
            sig = _field(frames, channel)
            if sig.size < self.sensor_stuck_min_frames:
                continue
            for start, end in self._runs_stuck(sig):
                if end - start >= self.sensor_stuck_min_frames:
                    duration = (
                        float(times[end - 1] - times[start]) + float(dt[end - 1])
                        if end - 1 < n
                        else float(times[min(end - 1, n - 1)] - times[start])
                    )
                    anomalies.append({
                        "frame_t": float(times[start]),
                        "type": "sensor_stuck",
                        "severity": "medium",
                        "description": (
                            f"{channel} frozen at {float(sig[start]):.4f} for"
                            f" {end - start} frames (~{duration:.2f}s)"
                        ),
                    })

        # 8) Iter-129: speed outlier — speed outside plausible F1 range.
        # Negative speed is non-physical; > 380 km/h exceeds the Monza speed
        # trap by a wide margin (real F1 cars top out around 360-375 km/h).
        for i in range(n):
            v = float(speed[i])
            if v < self.speed_outlier_min_kmh or v > self.speed_outlier_max_kmh:
                anomalies.append({
                    "frame_t": float(times[i]),
                    "type": "speed_outlier",
                    "severity": "high",
                    "description": (
                        f"speed={v:.1f} km/h outside ["
                        f"{self.speed_outlier_min_kmh:.0f}, "
                        f"{self.speed_outlier_max_kmh:.0f}]"
                    ),
                })

        # 9) Iter-129: lap-time jump — lap_time delta > threshold or < 0
        # between consecutive frames. Real lap timers advance by ~dt per
        # frame; a > 5 s jump in a single frame step signals a timestamp
        # glitch, session reset, or packet reordering mid-lap.
        lap_t = _field_multi(frames, ("lap_time", "current_lap_time"))
        if lap_t.size >= 2:
            for i in range(1, lap_t.size):
                delta = float(lap_t[i] - lap_t[i - 1])
                if delta < 0.0 or delta > self.lap_time_jump_max_s:
                    sev = "high" if delta < 0.0 else "medium"
                    anomalies.append({
                        "frame_t": float(times[i]),
                        "type": "lap_time_jump",
                        "severity": sev,
                        "description": (
                            f"lap_time delta={delta:.3f}s at frame {i} "
                            f"({float(lap_t[i - 1]):.3f} → {float(lap_t[i]):.3f})"
                        ),
                    })

        # Stable order: sort by frame_t then type for deterministic output.
        anomalies.sort(key=lambda a: (a["frame_t"], a["type"]))
        return anomalies

    # ------------------------------------------------------------------ #
    def severity_distribution(
        self, anomalies: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Return ``{"low": N, "medium": N, "high": N}`` for ``anomalies``."""
        dist: dict[str, int] = {"low": 0, "medium": 0, "high": 0}
        for a in anomalies:
            sev = a.get("severity", "low")
            if sev in dist:
                dist[sev] += 1
            else:
                dist["low"] += 1
        return dist

    # ------------------------------------------------------------------ #
    @staticmethod
    def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
        """Return ``(start, end)`` (end exclusive) for each True run in ``mask``."""
        runs: list[tuple[int, int]] = []
        if mask.size == 0:
            return runs
        in_run = False
        start = 0
        for i, m in enumerate(mask):
            if m and not in_run:
                in_run = True
                start = i
            elif not m and in_run:
                in_run = False
                runs.append((start, i))
        if in_run:
            runs.append((start, mask.size))
        return runs

    @staticmethod
    def _runs_stuck(signal: np.ndarray) -> list[tuple[int, int]]:
        """Iter-129: Return ``(start, end)`` (end exclusive) for each maximal run
        of *identical* consecutive values in ``signal``.

        Used by the sensor-stuck detector: a frozen sensor produces a run of
        identical values spanning many frames. NaN values are treated as
        distinct (never equal to anything, including themselves), so a stream
        of NaNs does NOT count as stuck.
        """
        runs: list[tuple[int, int]] = []
        if signal.size == 0:
            return runs
        in_run = False
        start = 0
        prev: float | None = None
        for i, v in enumerate(signal):
            v_f = float(v)
            # NaN-aware comparison: NaN != NaN, so a NaN never continues a run.
            same = prev is not None and not (
                math.isnan(v_f) or math.isnan(prev)
            ) and v_f == prev
            if same and not in_run:
                in_run = True
                start = i - 1  # run began at the previous frame
            elif same and in_run:
                # Continue the run.
                pass
            elif not same and in_run:
                in_run = False
                runs.append((start, i))
            prev = v_f
        if in_run:
            runs.append((start, signal.size))
        return runs


__all__ = [
    "TelemetryAnalytics",
    "PerformanceBenchmark",
    "AnomalyDetector",
    "TRACK_REFERENCES",
]
