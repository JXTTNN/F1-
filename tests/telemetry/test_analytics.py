"""Unit tests for :mod:`f1opt.telemetry.analytics`.

Synthetic F1 telemetry frames are generated with realistic ranges
(speed 0-340 km/h, throttle/brake 0-1, steer -1 to 1, g_lat -3 to 3,
g_long -1 to 1, rpm 0-13000, drs 0/1, ers_deploy 0-1) and fed to the three
public classes (:class:`TelemetryAnalytics`, :class:`PerformanceBenchmark`,
:class:`AnomalyDetector`).

The tests cover:

* :class:`TelemetryAnalytics` — every sub-analysis returns the expected keys
  and finite, in-range values.
* Edge cases — empty frames and single-frame inputs do not crash.
* :class:`PerformanceBenchmark` — scorecard shape, grade alphabet, and
  strengths/weaknesses typing.
* :class:`AnomalyDetector` — per-type detection, clean-data invariance,
  determinism, and the severity distribution summary.
"""

from __future__ import annotations

import math

import pytest

from f1opt.telemetry.analytics import (
    TRACK_REFERENCES,
    AnomalyDetector,
    PerformanceBenchmark,
    TelemetryAnalytics,
)

_DT = 1.0 / 60.0  # 60 Hz frame interval.


# --------------------------------------------------------------------------- #
# Synthetic frame builders
# --------------------------------------------------------------------------- #
def make_frame(
    t: float,
    *,
    speed: float = 200.0,
    throttle: float = 0.7,
    brake: float = 0.0,
    steer: float = 0.0,
    g_lat: float = 0.0,
    g_long: float = 0.0,
    rpm: float = 8000.0,
    drs: int = 0,
    ers_deploy: float = 0.3,
    ers_deploy_mode: float = 0.0,
) -> dict:
    """Build a synthetic frame dict with realistic F1 defaults."""
    return {
        "session_time": float(t),
        "speed": float(speed),
        "throttle": float(throttle),
        "brake": float(brake),
        "steer": float(steer),
        "g_lat": float(g_lat),
        "g_long": float(g_long),
        "rpm": float(rpm),
        "drs": int(drs),
        "ers_deploy": float(ers_deploy),
        "ers_deploy_mode": float(ers_deploy_mode),
    }


def constant_lap(
    n: int = 600,
    **overrides,
) -> list[dict]:
    """Build a constant-value lap of ``n`` frames at 60 Hz."""
    return [make_frame(i * _DT, **overrides) for i in range(n)]


def realistic_lap(n: int = 600) -> list[dict]:
    """Build a lap with smooth sinusoidal variation; no anomalies expected."""
    frames: list[dict] = []
    for i in range(n):
        t = i * _DT
        speed = 250.0 + 50.0 * math.sin(t * 2.0)
        steer = 0.5 * math.sin(t * 3.0)
        if 100 <= i < 150:
            throttle, brake = 0.0, 0.4
        else:
            throttle, brake = 0.8, 0.0
        g_lat = 2.0 * math.sin(t * 3.0)
        g_long = 0.5 * math.sin(t * 2.0)
        rpm = 9000.0 + 2000.0 * math.sin(t * 2.0)
        drs = 1 if 300 <= i < 400 else 0
        frames.append(
            make_frame(
                t,
                speed=speed,
                throttle=throttle,
                brake=brake,
                steer=steer,
                g_lat=g_lat,
                g_long=g_long,
                rpm=rpm,
                drs=drs,
                ers_deploy=0.5,
            )
        )
    return frames


def cornering_lap(n: int = 600) -> list[dict]:
    """A lap with multiple clear cornering events (|steer| > 0.3)."""
    frames: list[dict] = []
    for i in range(n):
        t = i * _DT
        # Square-wave-ish steering via sign of a fast sine.
        steer = 0.7 * math.copysign(1.0, math.sin(t * 4.0)) * abs(
            math.sin(t * 4.0)
        )
        speed = 200.0 - 80.0 * abs(steer)
        throttle = 0.9 if abs(steer) < 0.2 else 0.4
        frames.append(
            make_frame(
                t,
                speed=speed,
                throttle=throttle,
                brake=0.0,
                steer=steer,
                g_lat=2.5 * steer,
                g_long=0.2,
                rpm=10000.0,
                drs=0,
                ers_deploy=0.4,
            )
        )
    return frames


# --------------------------------------------------------------------------- #
# TelemetryAnalytics — sub-analysis shape and value sanity
# --------------------------------------------------------------------------- #
class TestTelemetryAnalyticsShape:
    def test_compute_all_returns_all_sub_analyses(self) -> None:
        out = TelemetryAnalytics(realistic_lap()).compute_all()
        expected = {
            "speed", "throttle", "brake", "steering", "gforce",
            "ers", "drs", "tire_load", "lap_smoothing_score",
            "racing_line_deviation",
        }
        assert expected <= set(out)
        assert isinstance(out["speed"], dict)
        assert isinstance(out["tire_load"], dict)

    def test_active_aero_usage_detects_x_z_modes(self) -> None:
        """active_aero_usage_analysis 应正确检测 F1 2026 X/Z 模式。"""
        frames = [{"active_aero_x": 0.2, "active_aero_z": 0.9} for _ in range(60)]
        frames += [{"active_aero_x": 0.9, "active_aero_z": 0.2} for _ in range(40)]
        out = TelemetryAnalytics(frames).active_aero_usage_analysis()
        assert out["total_frames"] == 100
        assert out["x_mode_frames"] == 40
        assert out["z_mode_frames"] == 60
        assert out["x_mode_fraction"] == pytest.approx(0.4)
        assert out["z_mode_fraction"] == pytest.approx(0.6)

    def test_speed_trace_has_v_max_v_min_v_avg(self) -> None:
        s = TelemetryAnalytics(realistic_lap()).speed_trace_analysis()
        for k in ("v_max", "v_min", "v_avg", "v_std",
                  "speed_histogram", "corner_speed_distribution"):
            assert k in s
        assert math.isfinite(s["v_max"])
        assert math.isfinite(s["v_min"])
        assert math.isfinite(s["v_avg"])
        assert s["v_max"] >= s["v_min"]
        assert len(s["speed_histogram"]) == 10
        assert set(s["corner_speed_distribution"]) == {"fast", "medium", "slow"}

    def test_throttle_full_throttle_pct_in_range(self) -> None:
        t = TelemetryAnalytics(realistic_lap()).throttle_trace_analysis()
        assert 0.0 <= t["full_throttle_pct"] <= 1.0
        assert 0.0 <= t["zero_throttle_pct"] <= 1.0
        assert len(t["throttle_histogram"]) == 10
        assert isinstance(t["lift_and_coast_events"], int)

    def test_brake_peak_brake_pressure_finite(self) -> None:
        b = TelemetryAnalytics(realistic_lap()).brake_trace_analysis()
        assert "brake_intensity_hist" in b
        assert math.isfinite(b["peak_brake_pressure"])
        assert 0.0 <= b["brake_release_smoothness"] <= 1.0
        assert isinstance(b["trail_brake_events"], int)

    def test_steering_corner_count_positive_for_cornering_data(self) -> None:
        s = TelemetryAnalytics(cornering_lap()).steering_trace_analysis()
        assert s["corner_count_estimate"] > 0
        assert isinstance(s["steer_reversals"], int)
        assert s["avg_corner_duration"] >= 0.0
        assert s["steering_aggression"] >= 0.0

    def test_gforce_g_lat_max_finite_and_traction_circle_positive(self) -> None:
        g = TelemetryAnalytics(realistic_lap()).gforce_analysis()
        assert math.isfinite(g["g_lat_max"])
        assert g["traction_circle_area"] > 0.0
        assert "g_long_max" in g and "g_long_min" in g
        assert "understeer_indicator" in g

    def test_ers_analysis_deploy_events_count(self) -> None:
        e = TelemetryAnalytics(realistic_lap()).ers_analysis()
        for k in ("ers_deploy_total", "ers_recover_total",
                  "deploy_events", "recover_events", "ers_efficiency"):
            assert k in e
        assert isinstance(e["deploy_events"], int)
        assert isinstance(e["recover_events"], int)
        assert e["ers_deploy_total"] >= 0.0

    def test_drs_analysis_activations_count(self) -> None:
        d = TelemetryAnalytics(realistic_lap()).drs_analysis()
        for k in ("drs_activations", "drs_duration_total", "drs_speed_gain_avg"):
            assert k in d
        assert isinstance(d["drs_activations"], int)
        assert d["drs_duration_total"] >= 0.0
        # DRS is active for a chunk of realistic_lap → at least 1 activation.
        assert d["drs_activations"] >= 1

    def test_tire_load_analysis_has_four_loads(self) -> None:
        tl = TelemetryAnalytics(realistic_lap()).tire_load_analysis()
        for k in ("fl_load_n", "fr_load_n", "rl_load_n", "rr_load_n",
                  "load_transfer_pct", "imbalance_pct"):
            assert k in tl
        # Static load per tire ~ m·g/4 = 798·9.81/4 ≈ 1955 N. Mean load
        # should be near that (well within ±50%).
        for k in ("fl_load_n", "fr_load_n", "rl_load_n", "rr_load_n"):
            assert math.isfinite(tl[k])
            assert tl[k] > 0.0

    def test_lap_smoothing_score_in_range(self) -> None:
        score = TelemetryAnalytics(realistic_lap()).lap_smoothing_score()
        assert 0.0 <= score <= 1.0

    def test_racing_line_deviation_non_negative(self) -> None:
        dev = TelemetryAnalytics(cornering_lap()).racing_line_deviation()
        assert dev >= 0.0


# --------------------------------------------------------------------------- #
# TelemetryAnalytics — edge cases
# --------------------------------------------------------------------------- #
class TestTelemetryAnalyticsEdgeCases:
    def test_empty_frames_no_crash(self) -> None:
        a = TelemetryAnalytics([])
        # All methods must return sensible defaults, not raise.
        speed = a.speed_trace_analysis()
        assert speed["v_max"] == 0.0
        assert speed["corner_speed_distribution"]["fast"] == []
        thr = a.throttle_trace_analysis()
        assert thr["full_throttle_pct"] == 0.0
        brk = a.brake_trace_analysis()
        assert brk["peak_brake_pressure"] == 0.0
        st = a.steering_trace_analysis()
        assert st["corner_count_estimate"] == 0
        gf = a.gforce_analysis()
        assert gf["g_lat_max"] == 0.0
        ers = a.ers_analysis()
        assert ers["deploy_events"] == 0
        drs = a.drs_analysis()
        assert drs["drs_activations"] == 0
        tl = a.tire_load_analysis()
        assert tl["fl_load_n"] == 0.0
        assert a.lap_smoothing_score() == 0.0
        assert a.racing_line_deviation() == 0.0
        out = a.compute_all()
        assert isinstance(out, dict)
        assert "speed" in out

    def test_single_frame_no_crash(self) -> None:
        a = TelemetryAnalytics([make_frame(0.0)])
        speed = a.speed_trace_analysis()
        assert speed["v_max"] == 200.0
        # Smoothness on a single frame: defaults to 1.0 (no jerk measurable).
        assert 0.0 <= a.lap_smoothing_score() <= 1.0
        out = a.compute_all()
        assert "gforce" in out


# --------------------------------------------------------------------------- #
# PerformanceBenchmark
# --------------------------------------------------------------------------- #
class TestPerformanceBenchmark:
    def test_benchmark_returns_all_score_keys_and_grade(self) -> None:
        metrics = TelemetryAnalytics(realistic_lap()).compute_all()
        out = PerformanceBenchmark("monza").benchmark(metrics)
        for k in ("speed_score", "consistency_score", "efficiency_score",
                  "overall_score", "grade", "strengths", "weaknesses"):
            assert k in out
        for k in ("speed_score", "consistency_score",
                  "efficiency_score", "overall_score"):
            assert 0.0 <= out[k] <= 1.0

    def test_grade_is_one_of_valid_letters(self) -> None:
        metrics = TelemetryAnalytics(realistic_lap()).compute_all()
        for track_id in ("monza", "monaco", "suzuka", "silverstone", "bahrain"):
            grade = PerformanceBenchmark(track_id).benchmark(metrics)["grade"]
            assert grade in {"S", "A", "B", "C", "D"}

    def test_strengths_and_weaknesses_are_lists(self) -> None:
        metrics = TelemetryAnalytics(realistic_lap()).compute_all()
        out = PerformanceBenchmark("monza").benchmark(metrics)
        assert isinstance(out["strengths"], list)
        assert isinstance(out["weaknesses"], list)
        # All entries are strings.
        assert all(isinstance(s, str) for s in out["strengths"])
        assert all(isinstance(w, str) for w in out["weaknesses"])

    def test_grade_thresholds(self) -> None:
        # Craft a deliberately weak lap: low speed, no DRS, no smoothness.
        weak = constant_lap(n=300, speed=80.0, throttle=0.2, brake=0.5,
                            steer=0.0, g_lat=0.0, g_long=0.0, rpm=4000.0,
                            drs=0, ers_deploy=0.0)
        metrics = TelemetryAnalytics(weak).compute_all()
        out = PerformanceBenchmark("monza").benchmark(metrics)
        assert out["grade"] in {"D", "C"}
        assert out["overall_score"] < 0.65

    def test_unknown_track_falls_back_to_medium_reference(self) -> None:
        bm = PerformanceBenchmark("__nonexistent_track__")
        assert bm.reference is TRACK_REFERENCES["medium"]
        out = bm.benchmark(TelemetryAnalytics(realistic_lap()).compute_all())
        assert "grade" in out


# --------------------------------------------------------------------------- #
# AnomalyDetector — basic shape & invariants
# --------------------------------------------------------------------------- #
class TestAnomalyDetectorBasic:
    def test_detect_returns_list(self) -> None:
        det = AnomalyDetector()
        out = det.detect(realistic_lap())
        assert isinstance(out, list)
        for a in out:
            assert {"frame_t", "type", "severity", "description"} <= set(a)

    def test_clean_data_returns_empty_list(self) -> None:
        det = AnomalyDetector()
        assert det.detect(realistic_lap()) == []

    def test_severity_distribution_has_low_medium_high(self) -> None:
        det = AnomalyDetector()
        dist = det.severity_distribution([])
        assert set(dist) == {"low", "medium", "high"}
        assert sum(dist.values()) == 0
        # With some synthetic anomalies.
        sample = [
            {"frame_t": 0.0, "type": "x", "severity": "low", "description": ""},
            {"frame_t": 1.0, "type": "y", "severity": "high", "description": ""},
            {"frame_t": 2.0, "type": "z", "severity": "medium", "description": ""},
            {"frame_t": 3.0, "type": "w", "severity": "high", "description": ""},
        ]
        dist = det.severity_distribution(sample)
        assert dist == {"low": 1, "medium": 1, "high": 2}

    def test_detect_is_deterministic(self) -> None:
        det = AnomalyDetector()
        lap = realistic_lap()
        a1 = det.detect(lap)
        a2 = det.detect(lap)
        assert a1 == a2


# --------------------------------------------------------------------------- #
# AnomalyDetector — per-type detection
# --------------------------------------------------------------------------- #
class TestAnomalyDetectorTypes:
    def test_detects_sudden_deceleration(self) -> None:
        # Build 12 frames where speed drops 100 km/h between frame 0 and 6
        # (6 frames at 60 Hz = 0.1 s, well within the window).
        frames: list[dict] = []
        for i in range(12):
            speed = 300.0 if i < 6 else 200.0
            frames.append(make_frame(i * _DT, speed=speed))
        det = AnomalyDetector()
        out = det.detect(frames)
        decel = [a for a in out if a["type"] == "sudden_deceleration"]
        assert len(decel) >= 1
        assert decel[0]["severity"] in {"medium", "high"}

    def test_detects_extreme_g(self) -> None:
        # A few frames with |g_lat| > 5.
        frames = [make_frame(i * _DT, g_lat=6.0) for i in range(5)]
        det = AnomalyDetector()
        out = det.detect(frames)
        extreme = [a for a in out if a["type"] == "extreme_g"]
        assert len(extreme) >= 1
        assert extreme[0]["severity"] == "high"

    def test_detects_brake_and_throttle(self) -> None:
        # Frames where both brake and throttle exceed 0.5.
        frames = [
            make_frame(i * _DT, throttle=0.8, brake=0.7) for i in range(10)
        ]
        det = AnomalyDetector()
        out = det.detect(frames)
        bt = [a for a in out if a["type"] == "brake_and_throttle"]
        assert len(bt) >= 1
        assert bt[0]["severity"] == "low"

    def test_detects_extreme_steering(self) -> None:
        # 35 frames (>0.5 s at 60 Hz) at |steer| = 1.0.
        frames = [
            make_frame(i * _DT, steer=1.0) for i in range(35)
        ]
        det = AnomalyDetector()
        out = det.detect(frames)
        es = [a for a in out if a["type"] == "extreme_steering"]
        assert len(es) >= 1
        assert es[0]["severity"] == "medium"

    def test_detects_sustained_redline(self) -> None:
        # 70 frames (>1 s at 60 Hz) at rpm = 13500 (> 13000).
        frames = [
            make_frame(i * _DT, rpm=13500.0) for i in range(70)
        ]
        det = AnomalyDetector()
        out = det.detect(frames)
        sr = [a for a in out if a["type"] == "sustained_redline"]
        assert len(sr) >= 1
        assert sr[0]["severity"] == "medium"

    def test_detects_ers_overdeploy(self) -> None:
        # Iter-260: 过部署用 ERS 模式 (hotlap=2/overtake=3) 判定.
        # 130 frames (>2 s at 60 Hz) at ers_deploy_mode=2 (hotlap).
        frames = [
            make_frame(i * _DT, ers_deploy_mode=2.0) for i in range(130)
        ]
        det = AnomalyDetector()
        out = det.detect(frames)
        eo = [a for a in out if a["type"] == "ers_overdeploy"]
        assert len(eo) >= 1
        assert eo[0]["severity"] == "medium"


# --------------------------------------------------------------------------- #
# AnomalyDetector — realistic lap integration
# --------------------------------------------------------------------------- #
class TestAnomalyDetectorRealistic:
    def test_realistic_lap_has_few_or_no_anomalies(self) -> None:
        det = AnomalyDetector()
        out = det.detect(realistic_lap())
        # A clean, smooth lap should produce zero anomalies.
        assert len(out) == 0

    def test_realistic_lap_with_one_anomaly_isolated(self) -> None:
        # Take a clean lap and inject a single extreme-g frame.
        lap = realistic_lap()
        lap[100] = {**lap[100], "g_lat": 6.0}
        det = AnomalyDetector()
        out = det.detect(lap)
        assert any(a["type"] == "extreme_g" for a in out)
        # The realistic lap otherwise produces no extra anomalies.
        assert all(a["type"] == "extreme_g" for a in out)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
