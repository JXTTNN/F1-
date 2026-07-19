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
        deploy = _field_multi(self.frames, ("ers_deploy", "ers_deployed"))
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
        # Deploy total: integral of deploy rate over time.
        deploy_total = float((deploy * dt).sum())
        # Recovery: derive from braking (brake > 0.3 → MGU-K harvest).
        recover_signal = np.clip(brake - 0.3, 0.0, 1.0)
        recover_total = float((recover_signal * dt).sum())

        # Deploy events: rising edges crossing 0.5.
        deploy_events = 0
        if deploy.size >= 2:
            high = deploy > 0.5
            deploy_events = _count_runs(high)
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
        ers = _field_multi(frames, ("ers_deploy", "ers_deployed"))

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
