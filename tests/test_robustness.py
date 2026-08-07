"""Robustness / edge-case test suite for the F1 setup optimizer.

Probes every public API with malformed, boundary, and extreme inputs using a
hypothesis-style property-testing mindset implemented with plain pytest
parameterized fuzz (no hypothesis dependency). The existing code is expected to
handle all of these gracefully; any crash would surface as a failing test.

Covers 10 test classes:
    1. TestSetupSchemaEdgeCases     — CarSetup boundary / validation / roundtrip
    2. TestTrackEdgeCases           — track DB lookups + invariants
    3. TestPacketParserEdgeCases    — UDP packet parsers on truncated/extreme bytes
    4. TestAggregatorEdgeCases      — LapAggregator empty / dedup / threshold
    5. TestSurrogateEdgeCases       — surrogate model on extreme/unknown inputs
    6. TestOptimizerEdgeCases       — search_setup on unknown track + bad weights
    7. TestFeedbackEdgeCases        — feedback engine on empty / zero / single frame
    8. TestPhysicsModuleEdgeCases   — physics sub-models on out-of-range inputs
    9. TestBatchPredictEdgeCases    — batch / sensitivity on empty / single / full
   10. TestApiAppEdgeCases          — REST endpoints on missing/bad/empty inputs
"""

from __future__ import annotations

import math
import struct
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from f1opt.data.setup_schema import (
    ALL_SETUP_FIELDS,
    DEFAULT_SETUP,
    SETUP_FIELDS,
    CarSetup,
)
from f1opt.data.tracks import (
    ALL_TRACKS,
    all_tracks,
    get_track,
    sprint_tracks,
)
from f1opt.feedback.engine import (
    extract_metrics,
    generate_feedback,
    rule_based_feedback,
)
from f1opt.model.batch import batch_predict_lap_times, sensitivity_analysis
from f1opt.model.optimizer import SearchOptimizer, SearchResult, search_setup
from f1opt.model.physics import (
    AeroModel,
    MassModel,
    PowertrainModel,
    TireThermalModel,
)
from f1opt.model.surrogate import (
    SurrogateModel,
    predict_full,
    predict_lap_time,
)
from f1opt.telemetry.aggregator import LapAggregator
from f1opt.telemetry.packets import (
    HEADER_FORMAT,
    PACKET_PARSERS,
    PacketHeader,
    parse_car_status,
    parse_header,
    parse_motion_ex,
    parse_time_trial,
)


# --------------------------------------------------------------------------- #
# Module-scoped fixture: ensure the default surrogate model is trained.
# Mirrors the pattern in tests/api/test_app.py and tests/model/test_optimizer.py.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module", autouse=True)
def _trained_default_model() -> None:
    """Load the cached trained model (or train if absent) + reset the LRU cache."""
    from f1opt.model.surrogate import (
        default_model_path,
        reset_default_model_cache,
    )
    from f1opt.model.train import train

    path = default_model_path()
    if path.exists():
        reset_default_model_cache()
        return
    train(iterations=300, log=False)


# --------------------------------------------------------------------------- #
# App fixture for API tests (fresh app, UDP listener disabled).
# --------------------------------------------------------------------------- #
@pytest.fixture
def app() -> Any:
    """A fresh FastAPI app with the UDP listener disabled (no port binding)."""
    from f1opt.api.app import create_app

    return create_app(start_listener=False)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_header_bytes(packet_id: int) -> bytes:
    """Build a 29-byte F1 25 packet header for the given packet_id."""
    return struct.pack(
        HEADER_FORMAT,
        2025,                 # m_packetFormat
        25,                   # m_gameYear
        1,                    # m_gameMajorVersion
        0,                    # m_gameMinorVersion
        1,                    # m_packetVersion
        packet_id,            # m_packetId
        0x123456789ABCDEF0,  # m_sessionUID
        10.5,                 # m_sessionTime
        100,                  # m_frameIdentifier
        200,                  # m_overallFrameIdentifier
        0,                    # m_playerCarIndex
        255,                  # m_secondaryPlayerCarIndex
    )


def _min_setup_kwargs() -> dict[str, Any]:
    """Build kwargs with every setup field at its min boundary."""
    return {name: spec.min for name, spec in SETUP_FIELDS.items()}


def _max_setup_kwargs() -> dict[str, Any]:
    """Build kwargs with every setup field at its max boundary."""
    return {name: spec.max for name, spec in SETUP_FIELDS.items()}


# --- Aggregator helpers (PacketHeader objects + parsed payloads) ------------- #
def _agg_header(
    packet_id: int,
    *,
    overall_frame: int = 100,
    session_time: float = 10.0,
    session_uid: int = 0xABCDEF1234567890,
) -> PacketHeader:
    """Build a PacketHeader object for the aggregator subscriber."""
    return PacketHeader(
        packet_format=2025,
        game_year=25,
        game_major_version=1,
        game_minor_version=0,
        packet_version=1,
        packet_id=packet_id,
        session_uid=session_uid,
        session_time=session_time,
        frame_identifier=overall_frame,
        overall_frame_identifier=overall_frame,
        player_car_index=0,
        secondary_player_car_index=255,
    )


def _empty_lap() -> dict[str, Any]:
    return {
        "m_lastLapTimeInMS": 0,
        "m_currentLapTimeInMS": 0,
        "m_sector1TimeInMS": 0,
        "m_sector2TimeInMS": 0,
        "m_lapDistance": 0.0,
        "m_totalDistance": 0.0,
        "m_safetyCarDelta": 0.0,
        "m_carPosition": 0,
        "m_currentLapNum": 0,
        "m_pitStatus": 0,
        "m_numPitStops": 0,
        "m_sector": 0,
        "m_currentLapInvalid": 0,
        "m_penalties": 0,
        "m_totalWarnings": 0,
        "m_cornerCuttingWarnings": 0,
        "m_numUnservedDriveThroughPens": 0,
        "m_numUnservedStopGoPens": 0,
        "m_gridPosition": 0,
        "m_driverStatus": 0,
        "m_resultStatus": 0,
        "m_pitLaneTimerActive": 0,
        "m_pitLaneTimeInLaneInMS": 0,
        "m_pitStopTimerInMS": 0,
        "m_pitStopShouldServePen": 0,
    }


def _lap_data(car0: dict[str, Any]) -> dict[str, Any]:
    return {"m_lapData": [car0] + [_empty_lap()] * 21}


def _empty_telem() -> dict[str, Any]:
    return {
        "m_speed": 0,
        "m_throttle": 0.0,
        "m_steer": 0.0,
        "m_brake": 0.0,
        "m_clutch": 0,
        "m_gear": 0,
        "m_engineRPM": 0,
        "m_drs": 0,
        "m_revLightsPercent": 0,
        "m_revLightsBitValue": 0,
        "m_brakesTemperature": [0, 0, 0, 0],
        "m_tyresSurfaceTemperature": [0, 0, 0, 0],
        "m_tyresInnerTemperature": [0, 0, 0, 0],
        "m_engineTemperature": 0,
        "m_tyresPressure": [0.0, 0.0, 0.0, 0.0],
        "m_surfaceType": [0, 0, 0, 0],
    }


def _telemetry(car0: dict[str, Any]) -> dict[str, Any]:
    return {"m_carTelemetryData": [car0] + [_empty_telem()] * 21}


def _full_lap_row(
    *,
    session_uid: int = 123,
    car_index: int = 0,
    lap_number: int = 1,
    overall_frame_start: int = 100,
    clean: bool = True,
    quality_flag: str = "OK",
) -> dict[str, Any]:
    """Build a complete aggregator row matching the Parquet schema."""
    return {
        "session_uid": session_uid,
        "car_index": car_index,
        "lap_number": lap_number,
        "lap_time_ms": 90000,
        "overall_frame_start": overall_frame_start,
        "overall_frame_end": 200,
        "session_time_start": 0.0,
        "session_time_end": 90.0,
        "num_samples": 60,
        "avg_speed": 250.0,
        "avg_throttle": 0.8,
        "avg_brake": 0.2,
        "avg_ers_deploy": 0.0,
        "max_tyre_wear": 0.0,
        "track_id": 3,
        "weather": 0,
        "clean": clean,
        "invalid_reason": None,
        "quality_flag": quality_flag,
    }


# --------------------------------------------------------------------------- #
# 1. CarSetup schema edge cases
# --------------------------------------------------------------------------- #
class TestSetupSchemaEdgeCases:
    """Probe CarSetup validation at boundaries, out-of-range, and roundtrip."""

    def test_all_fields_at_min_boundary_is_valid(self) -> None:
        setup = CarSetup(**_min_setup_kwargs())
        for spec in ALL_SETUP_FIELDS():
            assert getattr(setup, spec.name) == spec.min

    def test_all_fields_at_max_boundary_is_valid(self) -> None:
        setup = CarSetup(**_max_setup_kwargs())
        for spec in ALL_SETUP_FIELDS():
            assert getattr(setup, spec.name) == spec.max

    def test_front_wing_below_min_raises(self) -> None:
        kwargs = DEFAULT_SETUP.model_dump()
        kwargs["front_wing"] = -1
        with pytest.raises(ValidationError):
            CarSetup(**kwargs)

    def test_fuel_load_above_max_raises(self) -> None:
        kwargs = DEFAULT_SETUP.model_dump()
        kwargs["fuel_load"] = 999
        with pytest.raises(ValidationError):
            CarSetup(**kwargs)

    def test_front_wing_non_aligned_to_step_raises(self) -> None:
        """front_wing=5.5 is non-integer → pydantic int validation rejects it."""
        kwargs = DEFAULT_SETUP.model_dump()
        kwargs["front_wing"] = 5.5
        with pytest.raises(ValidationError):
            CarSetup(**kwargs)

    def test_float_field_non_aligned_to_step_raises(self) -> None:
        """front_camber=-3.505 is not aligned to step 0.01 → ValidationError."""
        kwargs = DEFAULT_SETUP.model_dump()
        kwargs["front_camber"] = -3.505
        with pytest.raises(ValidationError):
            CarSetup(**kwargs)

    def test_from_vector_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError):
            CarSetup.from_vector([0.0] * 18)
        with pytest.raises(ValueError):
            CarSetup.from_vector([0.0] * 20)

    def test_to_vector_from_vector_roundtrip_preserves_values(self) -> None:
        for setup in (
            DEFAULT_SETUP,
            CarSetup(**_min_setup_kwargs()),
            CarSetup(**_max_setup_kwargs()),
        ):
            rebuilt = CarSetup.from_vector(setup.to_vector())
            for spec in ALL_SETUP_FIELDS():
                assert getattr(rebuilt, spec.name) == getattr(setup, spec.name)

    def test_default_setup_is_valid(self) -> None:
        # Re-constructing DEFAULT_SETUP from its dump should not raise.
        rebuilt = CarSetup(**DEFAULT_SETUP.model_dump())
        assert rebuilt == DEFAULT_SETUP
        assert len(rebuilt.to_vector()) == 21


# --------------------------------------------------------------------------- #
# 2. Track DB edge cases
# --------------------------------------------------------------------------- #
class TestTrackEdgeCases:
    """Probe the track database for completeness and lookup robustness."""

    @pytest.mark.parametrize("track_id", [t.track_id for t in ALL_TRACKS])
    def test_get_track_returns_valid_track_for_each_id(self, track_id: str) -> None:
        # Resolve the module fresh at call time. tests/data/test_tracks.py
        # calls importlib.reload(f1opt.data.tracks), which re-executes the
        # module in its existing __dict__ — swapping the Track class and
        # TRACKS_BY_ID in place. A Track captured at collection time would
        # then be a *different* class object than the instances returned by
        # get_track (which resolves TRACKS_BY_ID dynamically), breaking
        # isinstance. Resolving both symbols from the live module keeps them
        # consistent regardless of any reload.
        import f1opt.data.tracks as _tracks

        track = _tracks.get_track(track_id)
        assert isinstance(track, _tracks.Track)
        assert track.track_id == track_id
        assert track.length_m > 0
        assert track.corners > 0

    def test_get_track_nonexistent_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            get_track("nonexistent")
        with pytest.raises(ValueError):
            get_track("")

    def test_all_tracks_has_exactly_24_entries(self) -> None:
        assert len(ALL_TRACKS) == 24
        assert len(all_tracks()) == 24

    def test_every_track_has_positive_length_and_corners(self) -> None:
        for track in ALL_TRACKS:
            assert track.length_m > 0, track.track_id
            assert track.corners > 0, track.track_id

    def test_sprint_tracks_returns_six(self) -> None:
        sprints = sprint_tracks()
        assert len(sprints) == 6
        assert all(t.is_sprint for t in sprints)


# --------------------------------------------------------------------------- #
# 3. Packet parser edge cases
# --------------------------------------------------------------------------- #
class TestPacketParserEdgeCases:
    """Probe UDP packet parsers with empty / truncated / extreme byte inputs."""

    def test_parse_header_empty_bytes_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_header(b"")

    def test_parse_header_too_short_bytes_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_header(b"\x00" * 10)
        with pytest.raises(ValueError):
            parse_header(b"\x00" * 28)

    def test_parse_motion_ex_truncated_body_does_not_crash(self) -> None:
        """A truncated MotionEx body (1 byte) is zero-padded; no crash."""
        data = _make_header_bytes(13) + b"\x00"
        result = parse_motion_ex(data)
        assert isinstance(result, dict)
        assert "m_suspensionPosition" in result
        assert len(result["m_suspensionPosition"]) == 4

    def test_parse_time_trial_on_101_byte_packet_returns_fields(self) -> None:
        """EA PDF total = 101B (29 header + 72 body). Feed exactly 72B body."""
        body = bytes(range(72))
        data = _make_header_bytes(14) + body
        result = parse_time_trial(data)
        assert isinstance(result, dict)
        assert "m_timeTrialDataSet" in result
        tts = result["m_timeTrialDataSet"]
        assert tts["m_carIdx"] == body[0]
        assert tts["m_teamId"] == body[1]
        assert isinstance(tts["m_lapTimeInMS"], int)
        assert result["_expected_body_size"] == 72

    @pytest.mark.parametrize("pid", list(range(16)))
    def test_all_parsers_on_minimal_bytes_return_dict(self, pid: int) -> None:
        """Each of the 16 parsers handles a minimal (truncated) body gracefully."""
        parser = PACKET_PARSERS[pid]
        data = _make_header_bytes(pid) + b"\x00" * 10
        result = parser(data)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_parse_car_status_invalid_ers_deploy_mode_handled(self) -> None:
        """ersDeployMode=255 (invalid semantically) is parsed without crashing."""
        status_per_fmt = "<BBBBBfffHHBBHBBbbfBfffB"
        per_size = struct.calcsize(status_per_fmt)
        car0_vals = (
            1, 1, 2, 50, 0,        # BBBBB: traction, abs, fuelMix, frontBrakeBias, pitLimiter
            100.0, 110.0, 5.0,     # fff: fuelInTank, fuelCapacity, fuelRemainingLaps
            12000, 5000,           # HH: maxRPM, idleRPM
            8, 1,                  # BB: maxGears, drsAllowed
            500,                   # H: drsActivationDistance
            22, 22,                # BB: actualTyreCompound, visualTyreCompound
            3, 0,                  # bb: tyresAgeLaps, vehicleFiaFlags
            50.0,                  # f: ersStoreEnergy
            255,                   # B: ersDeployMode = 255 (out-of-range semantically)
            10.0, 20.0, 30.0,      # fff: ersHarvestedMGUK, MGUH, deployed
            0,                     # B: networkPaused
        )
        car0_bytes = struct.pack(status_per_fmt, *car0_vals)
        body = car0_bytes + b"\x00" * (per_size * 21)
        data = _make_header_bytes(7) + body
        result = parse_car_status(data)
        assert isinstance(result, dict)
        assert len(result["m_carStatusData"]) == 22
        # The parser returns the raw uint8 value (255); it does not crash.
        assert result["m_carStatusData"][0]["m_ersDeployMode"] == 255


# --------------------------------------------------------------------------- #
# 4. LapAggregator edge cases
# --------------------------------------------------------------------------- #
class TestAggregatorEdgeCases:
    """Probe LapAggregator with no frames, dedup, and threshold behavior."""

    def test_no_frames_all_rows_returns_empty(self, tmp_path: Any) -> None:
        agg = LapAggregator(tmp_path / "empty.parquet")
        assert agg.all_rows() == []
        assert agg.rows == []

    def test_to_parquet_bytes_on_empty_returns_valid_parquet(self, tmp_path: Any) -> None:
        agg = LapAggregator(tmp_path / "empty_pq.parquet")
        data = agg.to_parquet_bytes()
        assert isinstance(data, bytes)
        assert data[:4] == b"PAR1"

    def test_all_rows_dedup_identical_keys(self, tmp_path: Any) -> None:
        """all_rows() dedupes a memory row that also exists on disk.

        The documented dedup contract is "disk + pending-in-memory": a row
        pending in memory that also exists on disk (e.g. flushed twice) is
        not double-counted. We flush one row to disk, then re-inject the
        same row into the in-memory buffer and assert a single row back.
        """
        agg = LapAggregator(tmp_path / "dedup.parquet")
        row = _full_lap_row()
        agg._rows.append(dict(row))  # noqa: SLF001
        assert agg.flush() == 1  # writes the row to disk, clears _rows
        # Re-inject the identical row in memory; all_rows() must dedupe.
        agg._rows.append(dict(row))  # noqa: SLF001 — same dedup key as on disk
        rows = agg.all_rows()
        assert len(rows) == 1

    async def test_clean_threshold_ok_vs_invalid_behavior(self, tmp_path: Any) -> None:
        """Same SUSPECT-flagged lap: 'OK' threshold → clean=False;
        'INVALID' threshold → clean=True."""
        async def _run_lap(agg: LapAggregator) -> None:
            sid = 42
            # Session packet (provides track_id / weather).
            await agg(
                _agg_header(1, overall_frame=100, session_time=10.0, session_uid=sid),
                {"m_trackId": 7, "m_weather": 0},
                b"",
            )
            # LapData: car 0 starts lap 1.
            await agg(
                _agg_header(2, overall_frame=101, session_time=10.1, session_uid=sid),
                _lap_data(
                    {"m_currentLapNum": 1, "m_lastLapTimeInMS": 0, "m_currentLapInvalid": 0}
                ),
                b"",
            )
            # CarTelemetry frame flagged SUSPECT_RANGE during the lap.
            tel = _telemetry(
                {
                    "m_speed": 200,
                    "m_throttle": 0.5,
                    "m_steer": 0.0,
                    "m_brake": 0.0,
                    "m_clutch": 0,
                    "m_gear": 6,
                    "m_engineRPM": 9000,
                    "m_drs": 0,
                    "m_revLightsPercent": 80,
                    "m_revLightsBitValue": 0,
                    "m_brakesTemperature": [400, 410, 420, 430],
                    "m_tyresSurfaceTemperature": [90, 91, 92, 93],
                    "m_tyresInnerTemperature": [85, 86, 87, 88],
                    "m_engineTemperature": 105,
                    "m_tyresPressure": [21.5, 21.6, 21.7, 21.8],
                    "m_surfaceType": [0, 1, 2, 3],
                }
            )
            tel["_flag"] = "SUSPECT_RANGE"
            await agg(
                _agg_header(6, overall_frame=102, session_time=10.2, session_uid=sid),
                tel,
                b"",
            )
            # LapData: car 0 completes lap 1 (lap number increments).
            await agg(
                _agg_header(2, overall_frame=103, session_time=10.3, session_uid=sid),
                _lap_data(
                    {"m_currentLapNum": 2, "m_lastLapTimeInMS": 90000, "m_currentLapInvalid": 0}
                ),
                b"",
            )

        agg_ok = LapAggregator(tmp_path / "thr_ok.parquet", clean_threshold="OK")
        await _run_lap(agg_ok)
        agg_invalid = LapAggregator(tmp_path / "thr_invalid.parquet", clean_threshold="INVALID")
        await _run_lap(agg_invalid)

        rows_ok = agg_ok.rows
        rows_invalid = agg_invalid.rows
        assert len(rows_ok) == 1
        assert len(rows_invalid) == 1
        # SUSPECT_RANGE severity = 1.
        # 'OK' threshold severity = 0 → 1 > 0 → clean = False.
        assert rows_ok[0]["clean"] is False
        # 'INVALID' threshold severity = 2 → 1 <= 2 → clean = True.
        assert rows_invalid[0]["clean"] is True
        # Both rows carry the SUSPECT_RANGE quality flag.
        assert rows_ok[0]["quality_flag"] == "SUSPECT_RANGE"
        assert rows_invalid[0]["quality_flag"] == "SUSPECT_RANGE"


# --------------------------------------------------------------------------- #
# 5. Surrogate model edge cases
# --------------------------------------------------------------------------- #
class TestSurrogateEdgeCases:
    """Probe the surrogate model on default / unknown / extreme / untrained."""

    def test_predict_lap_time_returns_finite_float(self) -> None:
        lap = predict_lap_time(DEFAULT_SETUP, "melbourne")
        assert isinstance(lap, float)
        assert math.isfinite(lap)
        assert lap > 0.0

    def test_predict_full_returns_required_keys(self) -> None:
        out = predict_full(DEFAULT_SETUP, "melbourne")
        assert isinstance(out, dict)
        assert "lap_time" in out
        assert "sectors" in out
        assert "responses" in out
        assert isinstance(out["lap_time"], float)
        assert isinstance(out["sectors"], list)
        assert len(out["sectors"]) == 3
        assert isinstance(out["responses"], dict)
        assert len(out["responses"]) >= 7

    def test_predict_lap_time_unknown_track_returns_finite(self) -> None:
        lap = predict_lap_time(DEFAULT_SETUP, "definitely_not_a_track_id")
        assert isinstance(lap, float)
        assert math.isfinite(lap)
        assert lap > 0.0

    def test_untrained_model_returns_finite_prior(self) -> None:
        model = SurrogateModel()
        lap = model.predict_lap_time(DEFAULT_SETUP, "melbourne")
        assert isinstance(lap, float)
        assert math.isfinite(lap)
        assert 60.0 <= lap <= 200.0

    def test_predict_all_max_setup_finite(self) -> None:
        setup = CarSetup(**_max_setup_kwargs())
        lap = predict_lap_time(setup, "melbourne")
        assert isinstance(lap, float)
        assert math.isfinite(lap)
        assert lap > 0.0

    def test_predict_all_min_setup_finite(self) -> None:
        setup = CarSetup(**_min_setup_kwargs())
        lap = predict_lap_time(setup, "melbourne")
        assert isinstance(lap, float)
        assert math.isfinite(lap)
        assert lap > 0.0


# --------------------------------------------------------------------------- #
# 6. Optimizer edge cases
# --------------------------------------------------------------------------- #
class TestOptimizerEdgeCases:
    """Probe search_setup on unknown tracks and with bad weight inputs."""

    def test_search_setup_unknown_track_returns_valid_result(self) -> None:
        result = search_setup("definitely_not_a_track", iterations=25, seed=0)
        assert isinstance(result, SearchResult)
        assert isinstance(result.predicted_gain_s, float)
        assert math.isfinite(result.predicted_gain_s)
        assert result.recommended_lap_time > 0.0
        # Recommended setup reconstructs to a valid CarSetup.
        CarSetup(**result.recommended)

    def test_negative_tire_wear_weight_raises(self) -> None:
        """SearchOptimizer rejects negative tire_wear_weight with ValueError.

        (search_setup itself delegates to _search without validation; the
        public SearchOptimizer class is where the guard lives.)
        """
        with pytest.raises(ValueError):
            SearchOptimizer(iterations=10, tire_wear_weight=-1.0)

    def test_tire_wear_weight_zero_backward_compatible(self) -> None:
        result = search_setup("melbourne", iterations=25, seed=0, tire_wear_weight=0.0)
        assert result.tire_wear_weight == 0.0
        # Iter-164.03: tire_wear 现在始终报告真实胎耗代理 (即使 weight=0),
        # 用于透明性. 旧版 weight=0 时 tire_wear=0.0 的行为已改变.
        assert result.tire_wear > 0.0
        assert isinstance(result.predicted_gain_s, float)

    def test_search_setup_diff_is_list(self) -> None:
        result = search_setup("melbourne", iterations=25, seed=0)
        assert isinstance(result.diff, list)
        # diff entries (if any) carry the required keys.
        for entry in result.diff:
            assert {"name", "before", "after", "delta"} <= set(entry.keys())


# --------------------------------------------------------------------------- #
# 7. Feedback engine edge cases
# --------------------------------------------------------------------------- #
class TestFeedbackEdgeCases:
    """Probe the feedback engine on empty / single / all-zero frames."""

    def test_generate_feedback_empty_frames_returns_dimensions(self) -> None:
        out = generate_feedback([], DEFAULT_SETUP.model_dump(), "melbourne")
        assert isinstance(out, dict)
        assert "dimensions" in out
        assert len(out["dimensions"]) == 12  # Iter-164.14: +corner_analysis → 12 dims
        # Data-dependent dimensions report 数据不足.
        data_dep_dims = {
            "balance",
            "grip",
            "tyres",
            "braking",
            "ers_drs",
            "throttle_brake_smoothness",
            "confidence",
        }
        for dim in out["dimensions"]:
            if dim["name"] in data_dep_dims:
                assert dim["value"] == "数据不足", dim["name"]

    def test_generate_feedback_single_frame_does_not_crash(self) -> None:
        frame = {
            "session_time": 0.0,
            "speed": 200.0,
            "throttle": 0.5,
            "brake": 0.0,
            "steer": 0.0,
            "g_lat": 0.0,
            "lap_time": 90.0,
            "lap_distance": 0.0,
        }
        out = generate_feedback([frame], DEFAULT_SETUP.model_dump(), "melbourne")
        assert isinstance(out, dict)
        assert len(out["dimensions"]) == 12  # Iter-164.14: +corner_analysis → 12 dims

    def test_rule_based_feedback_all_zero_frames_does_not_crash(self) -> None:
        frames = [
            {
                "session_time": 0.0,
                "speed": 0.0,
                "throttle": 0.0,
                "brake": 0.0,
                "steer": 0.0,
                "g_lat": 0.0,
                "lap_time": 0.0,
                "lap_distance": 0.0,
                "ers_store": 0.0,
                "drs_allowed": 0,
                "fuel_in_tank": 0.0,
            }
            for _ in range(10)
        ]
        metrics = extract_metrics(frames, DEFAULT_SETUP.model_dump(), "melbourne")
        out = rule_based_feedback(metrics, DEFAULT_SETUP.model_dump(), "melbourne")
        assert isinstance(out, dict)
        assert len(out["dimensions"]) == 12  # Iter-164.14: +corner_analysis → 12 dims

    def test_extract_metrics_empty_list_returns_n_frames_zero(self) -> None:
        metrics = extract_metrics([], {}, "melbourne")
        assert isinstance(metrics, dict)
        assert metrics["n_frames"] == 0
        assert metrics["values"] == {}
        assert metrics["sources"] == []


# --------------------------------------------------------------------------- #
# 8. Physics sub-model edge cases
# --------------------------------------------------------------------------- #
class TestPhysicsModuleEdgeCases:
    """Probe physics sub-models with out-of-range / extreme inputs."""

    @pytest.mark.parametrize("temp", [-50.0, 200.0])
    def test_grip_factor_extreme_temps_clamped_in_unit_range(self, temp: float) -> None:
        model = TireThermalModel()
        g = model.grip_factor(temp)
        assert isinstance(g, float)
        assert 0.0 <= g <= 1.0

    def test_aero_downforce_speed_zero_returns_zero(self) -> None:
        model = AeroModel()
        f = model.downforce(
            front_wing=25, rear_wing=27, ride_height_f=10.0, ride_height_r=12.0, speed_ms=0.0
        )
        assert isinstance(f, float)
        assert f == 0.0

    def test_aero_downforce_negative_speed_clamped_to_zero(self) -> None:
        model = AeroModel()
        f = model.downforce(
            front_wing=25, rear_wing=27, ride_height_f=10.0, ride_height_r=12.0, speed_ms=-10.0
        )
        assert isinstance(f, float)
        assert f == 0.0

    def test_powertrain_boost_mode_laptime_out_of_range_clamped(self) -> None:
        """boost_mode_laptime(99) clamps mode to [0,3] → returns 0.20 (mode 3)."""
        model = PowertrainModel()
        val = model.boost_mode_laptime(99)
        assert isinstance(val, float)
        assert val == 0.20

    def test_mass_model_total_mass_fuel_above_max_clamped(self) -> None:
        """fuel_load=999 clamps to 110 kg → total = 798 + 110 = 908."""
        model = MassModel()
        m = model.total_mass(fuel_load_kg=999.0)
        assert isinstance(m, float)
        assert m == 908.0


# --------------------------------------------------------------------------- #
# 9. Batch predict edge cases
# --------------------------------------------------------------------------- #
class TestBatchPredictEdgeCases:
    """Probe batch predict / sensitivity on empty / single / full inputs."""

    def test_batch_predict_empty_list_returns_empty(self) -> None:
        result = batch_predict_lap_times([], "melbourne")
        assert result == []

    def test_batch_predict_single_element_returns_float_list(self) -> None:
        result = batch_predict_lap_times([DEFAULT_SETUP], "melbourne")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], float)
        assert math.isfinite(result[0])

    def test_sensitivity_analysis_returns_21_keys(self) -> None:
        result = sensitivity_analysis(DEFAULT_SETUP, "melbourne")
        assert isinstance(result, dict)
        assert len(result) == 21
        assert set(result.keys()) == set(SETUP_FIELDS.keys())
        for v in result.values():
            assert isinstance(v, float)
            assert v >= 0.0


# --------------------------------------------------------------------------- #
# 10. API app edge cases
# --------------------------------------------------------------------------- #
class TestApiAppEdgeCases:
    """Probe REST endpoints on missing / bad / empty inputs (httpx ASGI)."""

    async def test_health_returns_200(self, app: Any) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/health")
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "ok"

    async def test_tracks_nonexistent_returns_404(self, app: Any) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/tracks/nonexistent")
            assert r.status_code == 404

    async def test_predict_missing_setup_returns_422(self, app: Any) -> None:
        """POST /api/predict without the required `setup` field → 422."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post("/api/predict", json={"track_id": "melbourne"})
            assert r.status_code == 422

    async def test_samples_empty_returns_zero_count(self, app: Any) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/samples")
            assert r.status_code == 200
            body = r.json()
            assert body["count"] == 0
            assert body["samples"] == []
