"""End-to-end scenario tests for the F1 setup optimizer.

Organized into 8 test classes covering realistic multi-module workflows:

- ``TestSetupOptimizationWorkflow`` — predict -> search -> validate,
  bayesian -> validate, pareto tradeoffs, preset -> optimize.
- ``TestFeedbackWorkflow`` — telemetry -> feedback -> narration ->
  setup tuning -> comparison after a setup change.
- ``TestTelemetryPipeline`` — packets -> aggregator, replay, analytics,
  anomaly detection, export/import roundtrip.
- ``TestRaceStrategyWorkflow`` — strategy per track, weather-aware strategy,
  stint simulation, strategy comparison.
- ``TestPhysicsIntegration`` — tire -> vehicle, suspension -> setup harmony,
  weather -> physics.
- ``TestAPIServerWorkflow`` — full search / bayesian / strategy / health via
  the FastAPI extended app over ``httpx`` ``ASGITransport``.
- ``TestDriverWorkflow`` — frames -> deep profile, profile -> coaching,
  skill assessment.
- ``TestPerformanceWorkflow`` — cache warmup, profiler measures search,
  latency budget check.

All tests use deterministic seeds and synthetic-but-realistic data. Async
tests rely on ``asyncio_mode=auto`` (no ``@pytest.mark.asyncio`` decorator).
"""

from __future__ import annotations

import math
import struct
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from f1opt.api.app import create_app
from f1opt.api.extended_app import create_extended_app
from f1opt.data.setup_schema import DEFAULT_SETUP, SETUP_FIELDS, CarSetup
from f1opt.data.track_evolution import (
    WeatherCondition,
    WeatherForecast,
    WeatherImpactModel,
)
from f1opt.data.tracks import TRACKS_BY_ID
from f1opt.driver.deep_profile import (
    AdaptationProfile,
    DeepDriverProfiler,
    DriverConsistencyAnalyzer,
    DrivingStyleArchetype,
    classify_archetype,
)
from f1opt.driver.profile import (
    AGGRESSIVE_PROFILE,
    DEFAULT_PROFILE,
    DriverProfile,
    extract_driver_profile,
)
from f1opt.feedback.comparison import (
    LapComparator,
    SetupChangeImpact,
)
from f1opt.feedback.engine import generate_feedback
from f1opt.feedback.nlg import FeedbackNarrator
from f1opt.model.bayesian import bayesian_search_setup
from f1opt.model.cache import WarmupCache
from f1opt.model.optimizer import search_setup
from f1opt.model.pareto import MultiObjectiveOptimizer, ParetoFront
from f1opt.model.physics import (
    AeroModel,
    PowertrainModel,
    TireDegradationModel,
    TireThermalModel,
    TireThermalState,
    TireWearState,
)
from f1opt.model.presets import SetupAutoTuner, SetupPresets
from f1opt.model.strategy import (
    RaceStrategyPlanner,
    StintSimulator,
    StrategyComparator,
)
from f1opt.model.surrogate import MODEL_VERSION, predict_lap_time
from f1opt.model.suspension import SetupHarmonics, SuspensionModel, VehicleDynamicsModel
from f1opt.model.tire_dynamics import MagicFormulaTire, TireSet
from f1opt.model.validation import SetupSanityChecker
from f1opt.observability.profiler import (
    DEFAULT_LATENCY_BUDGETS_MS,
    LatencyBudget,
    PerformanceProfiler,
)
from f1opt.telemetry.aggregator import LapAggregator
from f1opt.telemetry.analytics import AnomalyDetector, TelemetryAnalytics
from f1opt.telemetry.packets import (
    HEADER_FORMAT,
    NUM_CARS,
    PacketHeader,
)
from f1opt.telemetry.replay import (
    SessionExporter,
    SessionImporter,
    SessionRecorder,
    TelemetryReplay,
)

SESSION_UID = 0x0123456789ABCDEF


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module", autouse=True)
def _ensure_model_ready() -> None:
    try:
        from f1opt.model.surrogate import reset_default_model_cache

        reset_default_model_cache()
    except Exception:
        pass


@pytest.fixture(scope="module")
def extended_app() -> FastAPI:
    return create_extended_app()


@pytest.fixture(scope="module")
def core_app() -> FastAPI:
    return create_app(start_listener=False)


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# --------------------------------------------------------------------------- #
# Synthetic unified-frame factory (aligner frame keys)
# --------------------------------------------------------------------------- #
def _frame(i: int, **overrides: float) -> dict[str, Any]:
    t = i / 60.0
    f: dict[str, Any] = {
        "session_time": t,
        "speed": 250.0 + 5.0 * (i % 60),
        "throttle": 0.8,
        "brake": 0.0,
        "steer": 0.0,
        "gear": 6,
        "rpm": 9000,
        "tyre_temp_fl": 90.0,
        "tyre_temp_fr": 91.0,
        "tyre_temp_rl": 92.0,
        "tyre_temp_rr": 93.0,
        "g_lat": 0.0,
        "g_long": 0.0,
        "g_vert": 1.0,
        "world_x": 0.0,
        "world_y": 0.0,
        "world_z": 0.0,
        "velocity_x": 0.0,
        "velocity_y": 0.0,
        "velocity_z": 0.0,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "ers_store": 1_000_000.0,
        "ers_deploy_mode": 0,
        "drs_allowed": 0,
        "fuel_in_tank": 30.0,
        "fuel_remaining_laps": 5.0,
        "lap_time": 86.5 + t,
        "lap_distance": float(i),
        "tyre_wear_fl": 5.0,
        "tyre_wear_fr": 5.0,
        "tyre_wear_rl": 15.0,
        "tyre_wear_rr": 16.0,
    }
    f.update(overrides)
    return f


def _scripted_understeer_frames(n: int = 600) -> list[dict[str, Any]]:
    """Frames whose middle section shows high steer + low g_lat (understeer)."""
    frames: list[dict[str, Any]] = []
    for i in range(n):
        if n // 3 <= i < 2 * n // 3:
            frames.append(_frame(i, steer=0.8, g_lat=1.0, brake=0.0, throttle=0.5))
        else:
            frames.append(_frame(i))
    return frames


def _scripted_aggressive_frames(n: int = 600) -> list[dict[str, Any]]:
    """Frames with sharp brake onsets + high g_lat (aggressive style)."""
    frames: list[dict[str, Any]] = []
    for i in range(n):
        brake = 1.0 if (i % 60) < 5 else 0.0
        g_lat = 4.5 + 0.5 * math.sin(i * 0.3)
        frames.append(_frame(i, brake=brake, g_lat=g_lat, throttle=0.9))
    return frames


def _lap(lap_time: float, s1: float, s2: float, s3: float) -> dict[str, Any]:
    return {"lap_time": lap_time, "sector_times": [s1, s2, s3]}


# --------------------------------------------------------------------------- #
# F1 25 byte-packet builders (mirror f1opt.telemetry.packets layouts)
# --------------------------------------------------------------------------- #
_MOTION_PER = struct.Struct("<" + "fff" + "h" * 9 + "f" * 6)
_TELEM_PER = struct.Struct("<HfffBbHBBH4H4B4BB4f4B")  # Iter-278
_STATUS_PER = struct.Struct("<BBBBBfffHHBBHBBBbfffBffffB")  # Iter-278
_MOTION_TRAILER = struct.Struct("<30f")
_TELEM_TRAILER = struct.Struct("<BBB")


def _header(
    packet_id: int,
    *,
    session_time: float,
    frame: int,
    overall_frame: int | None = None,
    player_car: int = 0,
) -> bytes:
    return struct.pack(
        HEADER_FORMAT,
        2025, 25, 1, 0, 1, packet_id,
        SESSION_UID, session_time, frame,
        overall_frame if overall_frame is not None else frame,
        player_car, 255,
    )


def _motion_packet(session_time: float, frame: int, g_lat: float) -> bytes:
    car0 = (
        0.0, 0.0, 0.0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        g_lat, 0.0, 1.0, 0.0, 0.0, 0.0,
    )
    body = _MOTION_PER.pack(*car0)
    full = NUM_CARS * _MOTION_PER.size + _MOTION_TRAILER.size
    body += b"\x00" * (full - len(body))
    return _header(0, session_time=session_time, frame=frame) + body


def _telemetry_packet(
    session_time: float, frame: int, speed: float, throttle: float, *,
    player_car: int = 0, overall_frame: int | None = None,
) -> bytes:
    car0 = (
        int(speed), float(throttle), 0.0, 0.0,
        0, 6, 9000, 0, 0, 0,
        100, 110, 105, 120,
        90, 91, 92, 93,
        95, 96, 97, 98,
        100,
        21.0, 21.0, 21.0, 21.0,
        0, 0, 0, 0,
    )
    body = _TELEM_PER.pack(*car0)
    full = NUM_CARS * _TELEM_PER.size + _TELEM_TRAILER.size
    body += b"\x00" * (full - len(body))
    return _header(
        6, session_time=session_time, frame=frame,
        overall_frame=overall_frame, player_car=player_car,
    ) + body


def _status_packet(
    session_time: float, frame: int, *,
    ers_store: float = 1_000_000.0, fuel_in_tank: float = 30.0,
    drs_allowed: int = 0,
) -> bytes:
    car0 = (
        1, 1, 0, 50, 0,
        float(fuel_in_tank), 100.0, 5.0,
        12000, 4000,
        8, int(drs_allowed), 0,
        16, 16, 0, 0,
        float(ers_store), 0,
        0.0, 0.0, 0.0, 0,
    )
    body = _STATUS_PER.pack(*car0)
    full = NUM_CARS * _STATUS_PER.size
    body += b"\x00" * (full - len(body))
    return _header(7, session_time=session_time, frame=frame) + body


def _make_header(packet_id: int, *, frame: int, session_time: float) -> PacketHeader:
    return PacketHeader(
        packet_format=2025, game_year=25, game_major_version=1,
        game_minor_version=0, packet_version=1, packet_id=packet_id,
        session_uid=SESSION_UID, session_time=session_time,
        frame_identifier=frame, overall_frame_identifier=frame,
        player_car_index=0, secondary_player_car_index=255,
    )


def _empty_lap() -> dict[str, Any]:
    return {
        "m_lastLapTimeInMS": 0, "m_currentLapTimeInMS": 0,
        "m_sector1TimeInMS": 0, "m_sector2TimeInMS": 0,
        "m_lapDistance": 0.0, "m_totalDistance": 0.0, "m_safetyCarDelta": 0.0,
        "m_carPosition": 0, "m_currentLapNum": 0, "m_pitStatus": 0,
        "m_numPitStops": 0, "m_sector": 0, "m_currentLapInvalid": 0,
        "m_penalties": 0, "m_totalWarnings": 0, "m_cornerCuttingWarnings": 0,
        "m_numUnservedDriveThroughPens": 0, "m_numUnservedStopGoPens": 0,
        "m_gridPosition": 0, "m_driverStatus": 0, "m_resultStatus": 0,
        "m_pitLaneTimerActive": 0, "m_pitLaneTimeInLaneInMS": 0,
        "m_pitStopTimerInMS": 0, "m_pitStopShouldServePen": 0,
    }


def _lap_data(car0: dict[str, Any]) -> dict[str, Any]:
    base = _empty_lap()
    base.update(car0)
    return {"m_lapData": [base] + [_empty_lap() for _ in range(NUM_CARS - 1)]}


def _empty_telem() -> dict[str, Any]:
    return {
        "m_speed": 0, "m_throttle": 0.0, "m_steer": 0.0, "m_brake": 0.0,
        "m_clutch": 0, "m_gear": 0, "m_engineRPM": 0, "m_drs": 0,
        "m_revLightsPercent": 0, "m_revLightsBitValue": 0,
        "m_brakesTemperature": [0, 0, 0, 0],
        "m_tyresSurfaceTemperature": [0, 0, 0, 0],
        "m_tyresInnerTemperature": [0, 0, 0, 0],
        "m_engineTemperature": 0,
        "m_tyresPressure": [0.0, 0.0, 0.0, 0.0],
        "m_surfaceType": [0, 0, 0, 0],
    }


def _telemetry(car0: dict[str, Any]) -> dict[str, Any]:
    base = _empty_telem()
    base.update(car0)
    return {"m_carTelemetryData": [base] + [_empty_telem() for _ in range(NUM_CARS - 1)]}


# =========================================================================== #
# 1. TestSetupOptimizationWorkflow
# =========================================================================== #
class TestSetupOptimizationWorkflow:
    """Realistic setup-optimization pipelines across surrogate / optimizer /
    bayesian / pareto / presets / validation."""

    def test_predict_then_search_pipeline(self) -> None:
        baseline_lt = predict_lap_time(DEFAULT_SETUP, "silverstone")
        assert isinstance(baseline_lt, float) and baseline_lt > 0.0
        result = search_setup(
            "silverstone",
            driver_profile=DEFAULT_PROFILE,
            baseline=DEFAULT_SETUP,
            iterations=6,
            seed=42,
        )
        assert result.recommended_lap_time > 0.0
        assert result.baseline_lap_time == pytest.approx(baseline_lt, rel=0.05)
        assert result.predicted_gain_s >= 0.0
        assert len(result.search_trace) >= 1
        assert result.algorithm in {"scipy-de", "scipy-de-vec", "numpy-local"}

    def test_search_then_validate(self) -> None:
        result = search_setup(
            "monza", driver_profile=AGGRESSIVE_PROFILE, iterations=5, seed=7,
        )
        rec = CarSetup(**result.recommended)
        checker = SetupSanityChecker(rec, "monza")
        assert isinstance(checker.is_sane(), bool)
        range_warnings = checker.check_range_compliance()
        assert isinstance(range_warnings, list)
        track_report = checker.check_track_appropriateness()
        assert isinstance(track_report, list)
        rec_diff = DEFAULT_SETUP.diff(rec)
        assert isinstance(rec_diff, list)

    def test_bayesian_then_validate(self) -> None:
        result = bayesian_search_setup(
            "suzuka", baseline=DEFAULT_SETUP, n_iterations=8, seed=11,
        )
        rec = result["recommended_setup"]
        assert isinstance(rec, CarSetup)
        assert result["recommended_lap_time"] > 0.0
        assert result["iterations"] == 8
        assert len(result["history"]) == 8
        checker = SetupSanityChecker(rec, "suzuka")
        assert isinstance(checker.is_sane(), bool)
        assert isinstance(checker.check_range_compliance(), list)
        overall = checker.overall_warnings()
        assert isinstance(overall, list)

    def test_pareto_tradeoffs(self) -> None:
        opt = MultiObjectiveOptimizer(
            [[0.0, 1.0]] * 21,  # Iter-250: 21 dims (CarSetup has 21 fields incl active_aero)
            objectives=["lap_time", "tire_wear"],
            n_iterations=6,
            seed=3,
        )
        res = opt.search("spa")
        front = res["pareto_front"]
        assert isinstance(front, ParetoFront)
        front_points = front.compute_front()
        assert len(front_points) >= 1
        meta_front = front.compute_front_with_metadata()
        assert len(meta_front) >= 1
        best_lt = res["best_lap_time_setup"]
        best_tw = res["best_tire_wear_setup"]
        knee = res["knee_setup"]
        assert isinstance(best_lt, CarSetup)
        assert isinstance(best_tw, CarSetup)
        assert isinstance(knee, CarSetup)
        crowding = front.crowding_distance(front_points)
        assert isinstance(crowding, dict)
        knee_idx = front.knee_point(front_points)
        assert isinstance(knee_idx, int)
        meta_front = front.compute_front_with_metadata()
        if meta_front:
            ref = [max(m["values"][k] for m in meta_front) + 1.0
                   for k in range(len(meta_front[0]["values"]))]
            hv = front.hypervolume(ref)
            assert hv >= 0.0

    def test_preset_to_optimize(self) -> None:
        presets = SetupPresets()
        preset = presets.for_track("monaco")
        assert isinstance(preset, CarSetup)
        # Iter-94 修复: 旧版用 DEFAULT_PROFILE (全零 = 最差车手, +0.75s 慢) 调用
        # search_setup, 但 preset_lt/default_lt 用无 driver (None, Iter-93 后中性)
        # 计算, driver 不一致导致 recommended_lap 比 preset_lt 慢 1.4s. 现统一用
        # None (中性车手) 保持 driver 一致, 测试 optimizer 端到端 preset->推荐.
        preset_lt = predict_lap_time(preset, "monaco")
        default_lt = predict_lap_time(DEFAULT_SETUP, "monaco")
        result = search_setup(
            "monaco",
            driver_profile=None,
            baseline=preset,
            iterations=5,
            seed=99,
        )
        assert result.baseline_lap_time == pytest.approx(preset_lt, rel=0.05)
        assert result.recommended_lap_time <= preset_lt + 0.5
        assert result.recommended_lap_time <= default_lt + 0.5


# =========================================================================== #
# 2. TestFeedbackWorkflow
# =========================================================================== #
class TestFeedbackWorkflow:
    """Telemetry -> feedback -> narration -> setup tuning -> comparison."""

    def test_telemetry_to_feedback(self) -> None:
        frames = _scripted_understeer_frames(600)
        out = generate_feedback(
            frames, DEFAULT_SETUP.model_dump(), "melbourne",
            question="为什么推头",
        )
        assert set(out.keys()) == {"summary", "dimensions", "setup_suggestions", "sources", "_quality"}
        assert isinstance(out["summary"], str) and out["summary"]
        names = [d["name"] for d in out["dimensions"]]
        assert "lap_time_potential" in names
        assert len(out["setup_suggestions"]) >= 1
        for s in out["setup_suggestions"]:
            spec = SETUP_FIELDS[s["name"]]
            assert spec.min <= s["after"] <= spec.max
        assert len(out["sources"]) > 0

    def test_feedback_to_narration(self) -> None:
        frames = _scripted_understeer_frames(400)
        out = generate_feedback(frames, DEFAULT_SETUP.model_dump(), "silverstone")
        narrator = FeedbackNarrator(language="zh")
        narration = narrator.narrate_all(out["dimensions"])
        assert isinstance(narration, str) and narration != ""
        summary = narrator.summarize_session(out)
        assert isinstance(summary, str) and summary != ""
        for dim in out["dimensions"][:2]:
            single = narrator.narrate_dimension(dim)
            assert isinstance(single, str)

    def test_feedback_setup_tuning(self) -> None:
        frames = _scripted_understeer_frames(600)
        out = generate_feedback(frames, DEFAULT_SETUP.model_dump(), "monza")
        tuner = SetupAutoTuner(DEFAULT_SETUP, "monza", out["dimensions"])
        tuned = tuner.tune()
        assert isinstance(tuned, CarSetup)
        diff = tuner.tune_diff()
        assert isinstance(diff, list)
        confidence = tuner.confidence_score()
        assert 0.0 <= confidence <= 1.0
        constrained = tuner.apply_constraints(tuned)
        assert isinstance(constrained, CarSetup)
        checker = SetupSanityChecker(tuned, "monza")
        assert isinstance(checker.check_range_compliance(), list)

    def test_comparison_after_setup_change(self) -> None:
        before_laps = [
            _lap(91.0, 30.5, 30.0, 30.5),
            _lap(91.2, 30.6, 30.0, 30.6),
            _lap(90.9, 30.4, 30.0, 30.5),
        ]
        after_laps = [
            _lap(90.1, 30.0, 29.8, 30.3),
            _lap(90.3, 30.1, 29.9, 30.3),
            _lap(89.8, 29.9, 29.7, 30.2),
        ]
        impact = SetupChangeImpact(before_laps, after_laps).analyze()
        assert "delta_avg_s" in impact
        assert "before_avg" in impact
        assert "after_avg" in impact
        assert "verdict" in impact
        assert "significant" in impact
        assert impact["delta_avg_s"] < 0.0
        comparator = LapComparator(before_laps[0])
        comp = comparator.compare(after_laps[0])
        assert isinstance(comp, dict)


# =========================================================================== #
# 3. TestTelemetryPipeline
# =========================================================================== #
class TestTelemetryPipeline:
    """Packets -> aggregator, replay, analytics, anomalies, export/import."""

    async def test_packet_to_aggregator(self) -> None:
        agg = LapAggregator("/tmp/e2e_scenarios_lap.parquet")
        track = TRACKS_BY_ID["melbourne"]

        async def feed(pid: int, parsed: dict[str, Any], t: float, frame: int) -> None:
            await agg(_make_header(pid, frame=frame, session_time=t), parsed, b"")

        await feed(1, {"m_trackId": 1, "m_weather": 0}, 0.0, 100)
        await feed(
            2,
            _lap_data({
                "m_currentLapNum": 1, "m_lastLapTimeInMS": 0,
                "m_currentLapInvalid": 0, "m_lapDistance": 0.0,
            }),
            0.1, 101,
        )
        for i in range(30):
            await feed(
                6,
                _telemetry({
                    "m_speed": 200 + i, "m_throttle": 0.8, "m_brake": 0.1,
                }),
                0.2 + i * 0.05, 102 + i,
            )
        await feed(
            2,
            _lap_data({
                "m_currentLapNum": 2, "m_lastLapTimeInMS": 90000,
                "m_currentLapInvalid": 0, "m_lapDistance": track.length_m,
            }),
            2.0, 132,
        )
        rows = agg.rows
        assert len(rows) == 1
        row = rows[0]
        assert row["clean"] is True
        assert row["lap_number"] == 1
        assert row["lap_time_ms"] == 90000
        assert row["num_samples"] == 30
        pq = agg.to_parquet_bytes()
        assert isinstance(pq, bytes) and len(pq) > 0

    def test_replay_session(self) -> None:
        recorder = SessionRecorder("silverstone", DEFAULT_SETUP.model_dump(), "Test Driver")
        for i in range(120):
            recorder.record_frame(_frame(i, speed=200.0 + i))
        recorder.record_lap(1, 88.5, [29.0, 29.5, 30.0])
        recorder.set_metadata("weather", "dry")
        assert recorder.duration() > 0.0
        assert len(recorder.frames) == 120
        assert len(recorder.laps) == 1
        session_dict = recorder.to_dict()
        assert session_dict["track_id"] == "silverstone"

        replay = TelemetryReplay(recorder.frames, speed=10.0)
        replay.start()
        assert replay.is_finished() is False
        assert replay.frames_remaining() == 120
        first = replay.next_frame()
        assert first is not None
        assert "session_time" in first
        replay.seek(1.0)
        assert replay.current_session_time() >= 1.0 - 0.1
        summary = recorder.summary()
        assert isinstance(summary, dict)
        assert "track_id" in summary

    def test_analytics_from_frames(self) -> None:
        frames = _scripted_understeer_frames(600)
        track = TRACKS_BY_ID["melbourne"]
        analytics = TelemetryAnalytics(frames, track_length_m=track.length_m)
        metrics = analytics.compute_all()
        expected_keys = {
            "speed", "throttle", "brake", "steering", "gforce",
            "ers", "drs", "tire_load", "lap_smoothing_score",
            "racing_line_deviation",
        }
        assert expected_keys.issubset(metrics.keys())
        assert isinstance(metrics["speed"], dict)
        assert metrics["speed"]["v_avg"] > 0.0
        assert metrics["speed"]["v_max"] >= metrics["speed"]["v_min"]

    def test_anomaly_detection(self) -> None:
        frames = _scripted_understeer_frames(600)
        # Inject a sudden deceleration anomaly: speed drops 60 km/h in 1 frame.
        frames[300] = _frame(300, speed=120.0, brake=1.0)
        frames[301] = _frame(301, speed=60.0, brake=1.0)
        detector = AnomalyDetector()
        anomalies = detector.detect(frames)
        assert isinstance(anomalies, list)
        if anomalies:
            # Iter-129 extended the detector with three new types:
            # sensor_stuck (frozen physical channel), speed_outlier (outside
            # plausible F1 range), lap_time_jump (timestamp glitch). The
            # _frame() fixture has constant rpm/g_lat/g_long, so sensor_stuck
            # correctly fires here.
            valid_types = {
                "sudden_deceleration", "extreme_g", "sustained_redline",
                "brake_and_throttle", "extreme_steering", "ers_overdeploy",
                "sensor_stuck", "speed_outlier", "lap_time_jump",
            }
            for a in anomalies:
                assert a["type"] in valid_types
                assert "severity" in a and "frame_t" in a
            dist = detector.severity_distribution(anomalies)
            assert isinstance(dist, dict)

    def test_export_import_roundtrip(self) -> None:
        recorder = SessionRecorder("monza", DEFAULT_SETUP.model_dump(), "Roundtrip")
        for i in range(60):
            recorder.record_frame(_frame(i, speed=220.0 + i))
        recorder.record_lap(1, 82.0, [27.0, 27.5, 27.5])
        exporter = SessionExporter(recorder)
        csv_text = exporter.to_csv()
        assert isinstance(csv_text, str) and len(csv_text) > 0
        json_text = exporter.to_json()
        assert isinstance(json_text, str) and len(json_text) > 0
        md_text = exporter.to_summary_markdown()
        assert isinstance(md_text, str) and len(md_text) > 0

        importer = SessionImporter()
        fmt = importer.detect_format(csv_text)
        assert fmt == "csv"
        imported = importer.from_csv(csv_text)
        assert isinstance(imported, SessionRecorder)
        assert len(imported.frames) == len(recorder.frames)
        # JSON roundtrip via SessionRecorder.
        reconstructed = SessionRecorder.from_json(json_text)
        assert isinstance(reconstructed, SessionRecorder)
        assert reconstructed.track_id == "monza"


# =========================================================================== #
# 4. TestRaceStrategyWorkflow
# =========================================================================== #
class TestRaceStrategyWorkflow:
    """Strategy planning, weather-aware strategy, stint simulation, comparison."""

    def test_strategy_per_track(self) -> None:
        for track_id in ("monaco", "monza", "spa", "silverstone"):
            planner = RaceStrategyPlanner(track_id, total_laps=30, fuel_load_kg=80.0)
            strat = planner.optimal_strategy(["soft", "medium", "hard"])
            assert strat["strategy_type"] in ("0-stop", "1-stop", "2-stop")
            assert strat["total_time_est"] > 0.0
            assert isinstance(strat["recommendation_reason"], str)
            assert strat["recommendation_reason"]
            assert isinstance(strat["plan"], dict)

    def test_strategy_with_weather(self) -> None:
        forecast = WeatherForecast(WeatherCondition(precipitation_mm=0.0))
        assert forecast.forecast_at(0).is_dry() is True
        forecast.add_change(15, WeatherCondition(precipitation_mm=8.0))
        assert forecast.will_change_dry_to_wet() is True
        rec = forecast.strategy_recommendation()
        assert isinstance(rec, str) and rec
        planner = RaceStrategyPlanner("silverstone", total_laps=30, fuel_load_kg=80.0)
        wet_plan = planner.optimal_strategy(["medium", "hard"])
        assert wet_plan["total_time_est"] > 0.0

    def test_stint_simulation(self) -> None:
        stint = StintSimulator("soft", 15, "monza", base_lap_time=82.0)
        curve = stint.degradation_curve()
        assert len(curve) == 15
        # Degradation curve must be monotonically increasing (wear grows lap time).
        assert all(curve[i + 1] >= curve[i] for i in range(len(curve) - 1))
        records = stint.simulate()
        assert len(records) == 15
        assert records[0]["lap"] == 1
        assert records[-1]["lap"] == 15
        assert records[-1]["tire_wear_pct"] > records[0]["tire_wear_pct"]
        assert stint.total_time() > 0.0
        assert stint.avg_lap_time() > 0.0
        assert stint.avg_lap_time() == pytest.approx(
            stint.total_time() / 15, rel=1e-6,
        )

    def test_strategy_comparison(self) -> None:
        planner = RaceStrategyPlanner("spa", total_laps=25, fuel_load_kg=75.0)
        plans = [
            planner.plan_no_stop("hard"),
            planner.plan_one_stop(["medium", "hard"]),
            planner.plan_two_stop(["soft", "medium", "hard"]),
        ]
        comparator = StrategyComparator(plans)
        ranked = comparator.rank()
        assert len(ranked) == 3
        best_plan = comparator.best()
        assert best_plan in plans
        for idx in range(len(plans)):
            gap = comparator.gap_to_best(idx)
            assert gap >= 0.0
        rec = comparator.recommendation()
        assert isinstance(rec, str) and rec


# =========================================================================== #
# 5. TestPhysicsIntegration
# =========================================================================== #
class TestPhysicsIntegration:
    """Tire -> vehicle dynamics, suspension -> setup harmony, weather -> physics."""

    def test_tire_to_vehicle(self) -> None:
        tire = MagicFormulaTire(
            slip_ratio=0.1, slip_angle=6.0, load_n=4000.0,
            camber_deg=-3.5, temp_c=90.0, compound="soft",
        )
        fx = tire.pure_longitudinal()
        fy = tire.pure_lateral()
        assert fx > 0.0
        assert fy > 0.0
        fx_c, fy_c = tire.combined_force(0.1, 6.0)
        assert fx_c <= fx + 1e-6
        assert fy_c <= fy + 1e-6
        opt_sr = tire.optimal_slip_ratio()
        opt_sa = tire.optimal_slip_angle()
        assert 0.0 < opt_sr < 0.5
        assert 0.0 < opt_sa < 20.0
        mz = tire.self_aligning_torque(6.0)
        assert isinstance(mz, float)
        tire_set = TireSet(compound="soft", track_temp_c=30.0)
        assert tire_set.total_longitudinal() >= 0.0
        assert tire_set.total_lateral() >= 0.0
        balance = tire_set.lateral_balance()
        assert -1.0 <= balance <= 1.0
        diagnosis = tire_set.grip_balance_diagnosis()
        assert isinstance(diagnosis, str)

        veh = VehicleDynamicsModel(DEFAULT_SETUP.model_dump())
        k = veh.steady_state_balance(1.0, 50.0)
        assert isinstance(k, float)
        assert veh.yaw_inertia() > 0.0
        assert veh.response_time(1.0) > 0.0
        stab = veh.stability_factor()
        assert stab in {"understeer", "oversteer", "neutral"}
        tb = veh.trail_brake_balance(0.55, 1.5)
        assert "front_share" in tb and "rear_share" in tb and "trail_brake_safe" in tb
        aero = veh.aero_sensitivity(80.0)
        assert aero["downforce_total_n"] >= 0.0

    def test_suspension_to_setup_harmony(self) -> None:
        setup_dict = DEFAULT_SETUP.model_dump()
        susp = SuspensionModel(setup_dict)
        assert susp.spring_rate("front") > 0.0
        assert susp.spring_rate("rear") > 0.0
        assert susp.arb_stiffness("front") > 0.0
        assert susp.ride_height("front") > 0.0
        rake = susp.rake_angle()
        assert isinstance(rake, float)
        weights = susp.corner_weights(798.0)
        assert set(weights.keys()) == {"fl", "fr", "rl", "rr"}
        assert sum(weights.values()) == pytest.approx(798.0 * 9.81, rel=1e-4)
        roll_dist = susp.roll_stiffness_distribution()
        assert 0.0 <= roll_dist <= 1.0
        nf = susp.natural_frequency("front", 200.0)
        assert nf > 0.0
        ltd = susp.load_transfer_distribution(2.0)
        assert 0.0 <= ltd <= 1.0

        harmonics = SetupHarmonics(setup_dict)
        spring_check = harmonics.check_spring_arb_harmony()
        assert "ok" in spring_check and "warnings" in spring_check
        rh_check = harmonics.check_ride_height_rake()
        assert "rake_mm" in rh_check
        camber_check = harmonics.check_camber_alignment()
        assert "ok" in camber_check
        all_check = harmonics.all_checks()
        assert "total_warnings" in all_check
        diagnosis = VehicleDynamicsModel(setup_dict).setup_balance_diagnosis()
        assert "mechanical_balance" in diagnosis
        assert "aero_balance" in diagnosis
        assert "overall" in diagnosis
        assert "recommendation" in diagnosis

    def test_weather_to_physics(self) -> None:
        dry = WeatherCondition(track_temp_c=30.0, precipitation_mm=0.0)
        wet = WeatherCondition(track_temp_c=20.0, precipitation_mm=8.0)
        model = WeatherImpactModel()
        assert model.grip_multiplier(dry) > model.grip_multiplier(wet)
        assert model.lap_time_delta(dry, 90.0) == 0.0
        assert model.lap_time_delta(wet, 90.0) > 0.0
        assert model.tire_temp_impact(wet) < model.tire_temp_impact(dry)
        thermal = TireThermalModel()
        state = TireThermalState()
        temps = thermal.temperature(state, ambient_track_temp_c=30.0, slip_work=5000.0, duration_s=10.0)
        assert set(temps.keys()) == {"FL", "FR", "RL", "RR"}
        for tid in ("FL", "FR", "RL", "RR"):
            assert temps[tid]["surface"] > 0.0
        grip_at_peak = thermal.grip_factor(90.0)
        grip_at_cold = thermal.grip_factor(40.0)
        assert grip_at_peak > grip_at_cold
        deg_model = TireDegradationModel()
        wear_state = TireWearState()
        wear_inc = deg_model.wear_lap(wear_state, slip_angle_deg=5.0, tyre_load_g=1.5, track_temp_c=40.0)
        assert wear_inc > 0.0
        penalty = deg_model.wear_to_laptime_penalty(50.0)
        assert penalty > 0.0
        aero = AeroModel()
        df = aero.downforce(front_wing=25, rear_wing=27, ride_height_f=20, ride_height_r=40, speed_ms=80.0)
        assert df >= 0.0
        drag = aero.drag(front_wing=25, rear_wing=27, ride_height_avg=30.0, speed_ms=80.0)
        assert drag >= 0.0
        pt = PowertrainModel()
        ers = pt.ers_deploy_per_lap(2, "high_speed_low_downforce")
        assert ers > 0.0
        benefit = pt.laptime_benefit_kj_to_s(ers, 5000.0)
        assert benefit > 0.0
        fuel_pen = pt.fuel_effect_laptime(80.0)
        assert fuel_pen > 0.0


# =========================================================================== #
# 6. TestAPIServerWorkflow
# =========================================================================== #
class TestAPIServerWorkflow:
    """FastAPI extended app exercised via httpx ASGITransport."""

    async def test_full_search_via_api(self, core_app: FastAPI) -> None:
        async with _client(core_app) as client:
            r = await client.post(
                "/api/search",
                json={
                    "track_id": "silverstone",
                    "iterations": 5,
                    "seed": 42,
                    "driver_style": "default",
                },
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "recommended" in body
        assert "baseline" in body
        assert "predicted_gain_s" in body
        assert body["recommended_lap_time"] > 0.0
        assert body["iterations"] == 5
        assert isinstance(body["model_version"], str)

    async def test_bayesian_via_api(self, extended_app: FastAPI) -> None:
        async with _client(extended_app) as client:
            r = await client.post(
                "/api/bayesian-search",
                json={
                    "track_id": "monaco",
                    "n_iterations": 5,
                    "seed": 7,
                    "acquisition": "ei",
                },
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "recommended_setup" in body
        assert isinstance(body["recommended_setup"], dict)
        CarSetup(**body["recommended_setup"])
        assert body["iterations"] == 5
        assert isinstance(body["predicted_gain_s"], float)
        assert isinstance(body["history"], list)

    async def test_strategy_via_api(self, extended_app: FastAPI) -> None:
        async with _client(extended_app) as client:
            r = await client.post(
                "/api/strategy/plan",
                json={
                    "track_id": "monza",
                    "total_laps": 25,
                    "fuel_load_kg": 75.0,
                    "available_compounds": ["soft", "medium", "hard"],
                },
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["strategy_type"] in ("0-stop", "1-stop", "2-stop")
        assert body["total_time_est"] > 0.0
        assert isinstance(body["recommendation_reason"], str)
        assert body["recommendation_reason"]

    async def test_health_extended(self, extended_app: FastAPI) -> None:
        async with _client(extended_app) as client:
            r = await client.get("/api/health/extended")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "ok"
        assert isinstance(body["version"], str) and body["version"]
        assert isinstance(body["modules_loaded"], list)
        assert len(body["modules_loaded"]) > 0
        assert isinstance(body["model_available"], bool)
        assert isinstance(body["test_count_estimate"], int)
        # Cross-check surrogate model_version if model is available.
        assert isinstance(MODEL_VERSION, str)


# =========================================================================== #
# 7. TestDriverWorkflow
# =========================================================================== #
class TestDriverWorkflow:
    """Frames -> deep profile, profile -> coaching, skill assessment."""

    def test_frames_to_deep_profile(self) -> None:
        frames = _scripted_aggressive_frames(600)
        lap_metrics = [
            {"lap_time": 90.0 + 0.1 * i, "sector_times": [30.0, 30.0, 30.0 + 0.1 * i],
             "throttle_smoothness": 0.7, "brake_aggression": 0.5}
            for i in range(5)
        ]
        profiler = DeepDriverProfiler(
            frames, DEFAULT_SETUP.model_dump(), "silverstone", lap_metrics,
        )
        profile = profiler.profile()
        assert "archetype" in profile
        assert isinstance(profile["archetype"], DrivingStyleArchetype)
        assert "corner_phases" in profile
        assert "consistency" in profile
        assert "fatigue_projection" in profile
        assert "strengths" in profile
        assert "weaknesses" in profile
        assert "setup_recommendations" in profile
        fp = profile["fatigue_projection"]
        assert "base_lap_time" in fp
        assert len(fp["projected_lap_times"]) == 30
        assert len(fp["fatigue_indices"]) == 30
        basic = extract_driver_profile(frames)
        assert isinstance(basic, DriverProfile)
        vec = basic.to_vector()
        assert len(vec) == 8
        roundtrip = DriverProfile.from_vector(vec)
        assert roundtrip.to_vector() == pytest.approx(vec, abs=1e-6)

    def test_profile_to_coaching(self) -> None:
        frames = _scripted_aggressive_frames(600)
        profiler = DeepDriverProfiler(
            frames, DEFAULT_SETUP.model_dump(), "monza",
        )
        profile = profiler.profile()
        archetype = profile["archetype"]
        recs = profile["setup_recommendations"]
        assert isinstance(recs, list)
        for rec in recs:
            assert "field" in rec and "direction" in rec and "reason" in rec
        # Archetype classification consistency.
        metrics = {
            "throttle_smoothness": 0.2,
            "steer_smoothness": 0.3,
            "aggression_score": 0.9,
            "consistency_score": 0.3,
            "ers_usage_intensity": 0.8,
        }
        classified = classify_archetype(metrics)
        assert classified == DrivingStyleArchetype.AGGRESSIVE_OVERTAKER
        assert isinstance(archetype, DrivingStyleArchetype)

    def test_skill_assessment(self) -> None:
        lap_metrics = [
            {"lap_time": 90.0 - 0.05 * i, "sector_times": [30.0, 30.0, 30.0 - 0.05 * i],
             "throttle_smoothness": 0.8, "brake_aggression": 0.4}
            for i in range(6)
        ]
        analyzer = DriverConsistencyAnalyzer(min_laps=3)
        result = analyzer.analyze(lap_metrics)
        assert result["insufficient_data"] is False
        assert result["n_laps"] == 6
        assert "lap_time_cv" in result
        assert "sector_consistency" in result
        assert "overall_consistency_score" in result
        assert "trend" in result
        assert result["trend"] in {"improving", "degrading", "stable"}
        label = DriverConsistencyAnalyzer.consistency_label(result["overall_consistency_score"])
        assert isinstance(label, str)

        adaptation = AdaptationProfile()
        adaptation.record_condition("dry", 90.0)
        adaptation.record_condition("wet", 95.0)
        adaptation.record_condition("intermediate", 92.0)
        assert adaptation.adaptation_strength("dry") == 1.0
        assert 0.0 <= adaptation.adaptation_strength("wet") <= 1.0
        rec = adaptation.recommendation()
        assert isinstance(rec, str)


# =========================================================================== #
# 8. TestPerformanceWorkflow
# =========================================================================== #
class TestPerformanceWorkflow:
    """Cache warmup, profiler measures search, latency budget check."""

    def test_cache_warmup(self) -> None:
        warmer = WarmupCache()
        result = warmer.warmup_all()
        assert "surrogate" in result
        assert result["lookups_loaded"] is True
        assert result["track_refs_count"] > 0
        assert result["setup_fields_count"] > 0
        assert result["compounds_count"] > 0
        assert result["surrogate"]["tracks_warmed"] > 0
        assert result["surrogate"]["time_s"] >= 0.0

    def test_profiler_measures_search(self) -> None:
        profiler = PerformanceProfiler()
        with profiler.measure("search_setup"):
            search_setup(
                "silverstone", driver_profile=DEFAULT_PROFILE,
                iterations=3, seed=42,
            )
        report = profiler.report()
        assert "search_setup" in report
        assert report["search_setup"]["count"] >= 1.0
        assert report["search_setup"]["avg_ms"] > 0.0
        assert report["search_setup"]["p50_ms"] > 0.0
        text = profiler.summary_text()
        assert isinstance(text, str) and "search_setup" in text

        # Benchmark variant.
        bench = profiler.benchmark(
            predict_lap_time,
            [(DEFAULT_SETUP, "monaco"), (DEFAULT_SETUP, "monza")],
            warmup=1,
            repeats=2,
        )
        assert bench["name"] == "predict_lap_time"
        assert bench["n_calls"] == 4
        assert bench["avg_time_ms"] > 0.0

    def test_latency_budget_check(self) -> None:
        profiler = PerformanceProfiler()
        timings: dict[str, float] = {}
        with profiler.measure("predict_lap_time"):
            predict_lap_time(DEFAULT_SETUP, "silverstone")
        timings["predict_lap_time"] = profiler.report()["predict_lap_time"]["p50_ms"]

        with profiler.measure("search_setup"):
            search_setup(
                "monaco", driver_profile=DEFAULT_PROFILE,
                iterations=3, seed=1,
            )
        timings["search_setup"] = profiler.report()["search_setup"]["p50_ms"]

        budget = LatencyBudget()
        assert "predict_lap_time" in budget.budgets
        assert "search_setup" in budget.budgets
        results = budget.check_all(timings)
        assert len(results) == 2
        for r in results:
            assert "within_budget" in r
            assert "headroom_ms" in r
        # search_setup budget is generous (500ms default); 3-iter search is fast.
        search_check = budget.check("search_setup", timings["search_setup"])
        assert search_check["budget_ms"] == DEFAULT_LATENCY_BUDGETS_MS["search_setup"]
        # Sanity: measured time and budget are both positive numbers.
        assert search_check["elapsed_ms"] > 0.0
        assert search_check["budget_ms"] > 0.0
        violations = budget.violations(timings)
        assert isinstance(violations, list)
