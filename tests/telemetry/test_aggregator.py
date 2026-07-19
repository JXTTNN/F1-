"""Unit tests for :mod:`f1opt.telemetry.aggregator`.

Calls the :class:`LapAggregator` subscriber directly with crafted
``(header, parsed, raw)`` tuples (no UDP socket needed). Verifies:
- A clean lap (no flashback, no invalid flag) is aggregated into a row.
- A lap spanning a flashback (overall_frame regression) is discarded.
- Parquet flush writes the correct schema and rows.
- Multiple cars and laps are tracked independently.
"""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from f1opt.telemetry.aggregator import _SCHEMA, LapAggregator
from f1opt.telemetry.packets import PacketHeader

SESSION_UID = 0xABCDEF1234567890


def make_header(
    packet_id: int,
    *,
    overall_frame: int = 100,
    session_time: float = 10.0,
    session_uid: int = SESSION_UID,
) -> PacketHeader:
    """Build a :class:`PacketHeader` directly (no byte packing needed)."""
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


def lap_data(car0: dict, **kw) -> dict:
    """Build a LapData parsed payload with car 0 set and others zeroed."""
    return {"m_lapData": [car0] + [_empty_lap()] * 21, **kw}


def _empty_lap() -> dict:
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


def telemetry(car0: dict) -> dict:
    """Build a CarTelemetry parsed payload with car 0 set."""
    return {"m_carTelemetryData": [car0] + [_empty_telem()] * 21}


def _empty_telem() -> dict:
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


# --------------------------------------------------------------------------- #
# Clean lap
# --------------------------------------------------------------------------- #
class TestCleanLap:
    async def test_clean_lap_aggregated(self) -> None:
        agg = LapAggregator("/tmp/test_clean_lap.parquet")

        # Session packet (provides track_id / weather).
        sess = {"m_trackId": 7, "m_weather": 2}
        await agg(make_header(1, overall_frame=100, session_time=10.0), sess, b"")

        # LapData: car 0 starts lap 1.
        await agg(
            make_header(2, overall_frame=101, session_time=10.1),
            lap_data({"m_currentLapNum": 1, "m_lastLapTimeInMS": 0, "m_currentLapInvalid": 0}),
            b"",
        )

        # CarTelemetry: 5 frames during the lap.
        for i in range(5):
            await agg(
                make_header(6, overall_frame=102 + i, session_time=10.2 + i * 0.02),
                telemetry({"m_speed": 300, "m_throttle": 0.9, "m_brake": 0.1}),
                b"",
            )

        # LapData: car 0 completes lap 1 (lap number → 2, lastLapTime set).
        await agg(
            make_header(2, overall_frame=110, session_time=11.0),
            lap_data({"m_currentLapNum": 2, "m_lastLapTimeInMS": 95000, "m_currentLapInvalid": 0}),
            b"",
        )

        rows = agg.rows
        assert len(rows) == 1
        row = rows[0]
        assert row["lap_number"] == 1
        assert row["lap_time_ms"] == 95000
        assert row["num_samples"] == 5
        assert row["avg_speed"] == pytest.approx(300.0)
        assert row["avg_throttle"] == pytest.approx(0.9)
        assert row["avg_brake"] == pytest.approx(0.1)
        assert row["clean"] is True
        assert row["track_id"] == 7
        assert row["weather"] == 2
        assert row["session_uid"] == SESSION_UID


# --------------------------------------------------------------------------- #
# Flashback discards lap
# --------------------------------------------------------------------------- #
class TestFlashbackDiscard:
    async def test_flashback_lap_discarded(self) -> None:
        agg = LapAggregator("/tmp/test_flashback.parquet")

        # LapData: car 0 starts lap 1.
        await agg(
            make_header(2, overall_frame=100, session_time=10.0),
            lap_data({"m_currentLapNum": 1, "m_lastLapTimeInMS": 0, "m_currentLapInvalid": 0}),
            b"",
        )

        # 3 telemetry frames (frames 101, 102, 103).
        for i in range(3):
            await agg(
                make_header(6, overall_frame=101 + i, session_time=10.1 + i * 0.02),
                telemetry({"m_speed": 300, "m_throttle": 0.9, "m_brake": 0.0}),
                b"",
            )

        # Flashback: overall_frame regresses 103 → 50.
        await agg(
            make_header(6, overall_frame=50, session_time=8.0),
            telemetry({"m_speed": 200, "m_throttle": 0.5, "m_brake": 0.0}),
            b"",
        )

        # LapData: car 0 completes lap 1 (lap → 2, lastLapTime=95000).
        await agg(
            make_header(2, overall_frame=51, session_time=8.1),
            lap_data({"m_currentLapNum": 2, "m_lastLapTimeInMS": 95000, "m_currentLapInvalid": 0}),
            b"",
        )

        # The lap should have been discarded (dirty due to flashback).
        assert len(agg.rows) == 0

    async def test_invalid_lap_discarded(self) -> None:
        """A lap with m_currentLapInvalid=1 at completion is discarded."""
        agg = LapAggregator("/tmp/test_invalid.parquet")

        # Start lap 1.
        await agg(
            make_header(2, overall_frame=100),
            lap_data({"m_currentLapNum": 1, "m_lastLapTimeInMS": 0, "m_currentLapInvalid": 0}),
            b"",
        )

        # Telemetry.
        await agg(
            make_header(6, overall_frame=101),
            telemetry({"m_speed": 300, "m_throttle": 1.0, "m_brake": 0.0}),
            b"",
        )

        # Complete lap 1 but with invalid flag set.
        await agg(
            make_header(2, overall_frame=110),
            lap_data({"m_currentLapNum": 2, "m_lastLapTimeInMS": 95000, "m_currentLapInvalid": 1}),
            b"",
        )

        assert len(agg.rows) == 0


# --------------------------------------------------------------------------- #
# Parquet flush
# --------------------------------------------------------------------------- #
class TestParquetFlush:
    async def test_flush_writes_parquet(self, tmp_path) -> None:
        output = tmp_path / "laps.parquet"
        agg = LapAggregator(output)

        await agg(make_header(1, overall_frame=100), {"m_trackId": 3, "m_weather": 0}, b"")
        await agg(
            make_header(2, overall_frame=101),
            lap_data({"m_currentLapNum": 1, "m_lastLapTimeInMS": 0, "m_currentLapInvalid": 0}),
            b"",
        )
        await agg(
            make_header(6, overall_frame=102),
            telemetry({"m_speed": 250, "m_throttle": 0.8, "m_brake": 0.2}),
            b"",
        )
        await agg(
            make_header(2, overall_frame=110),
            lap_data({"m_currentLapNum": 2, "m_lastLapTimeInMS": 90000, "m_currentLapInvalid": 0}),
            b"",
        )

        n = agg.flush()
        assert n == 1
        assert output.exists()

        table = pq.read_table(output)
        assert table.num_rows == 1
        assert table.column("lap_time_ms")[0].as_py() == 90000
        assert table.column("clean")[0].as_py() is True
        assert table.column("track_id")[0].as_py() == 3
        assert table.column("avg_speed")[0].as_py() == pytest.approx(250.0)

    async def test_flush_appends_to_existing(self, tmp_path) -> None:
        output = tmp_path / "laps.parquet"
        agg = LapAggregator(output)

        # First lap.
        await agg(
            make_header(2, overall_frame=100),
            lap_data({"m_currentLapNum": 1, "m_lastLapTimeInMS": 0, "m_currentLapInvalid": 0}),
            b"",
        )
        await agg(
            make_header(2, overall_frame=110),
            lap_data({"m_currentLapNum": 2, "m_lastLapTimeInMS": 90000, "m_currentLapInvalid": 0}),
            b"",
        )
        assert agg.flush() == 1

        # Second lap (same aggregator, new lap).
        await agg(
            make_header(2, overall_frame=120),
            lap_data({"m_currentLapNum": 2, "m_lastLapTimeInMS": 0, "m_currentLapInvalid": 0}),
            b"",
        )
        await agg(
            make_header(2, overall_frame=130),
            lap_data({"m_currentLapNum": 3, "m_lastLapTimeInMS": 88000, "m_currentLapInvalid": 0}),
            b"",
        )
        assert agg.flush() == 1

        # Both laps should be in the file.
        table = pq.read_table(output)
        assert table.num_rows == 2

    async def test_flush_empty_returns_zero(self, tmp_path) -> None:
        agg = LapAggregator(tmp_path / "empty.parquet")
        assert agg.flush() == 0


# --------------------------------------------------------------------------- #
# Iter-13: all_rows / to_parquet_bytes (export API backing)
# --------------------------------------------------------------------------- #
class TestExportHelpers:
    async def _produce_one_lap(self, agg: LapAggregator, frame_base: int = 100) -> None:
        await agg(
            make_header(1, overall_frame=frame_base),
            {"m_trackId": 3, "m_weather": 0},
            b"",
        )
        await agg(
            make_header(2, overall_frame=frame_base + 1),
            lap_data({"m_currentLapNum": 1, "m_lastLapTimeInMS": 0, "m_currentLapInvalid": 0}),
            b"",
        )
        await agg(
            make_header(6, overall_frame=frame_base + 2),
            telemetry({"m_speed": 250, "m_throttle": 0.8, "m_brake": 0.2}),
            b"",
        )
        await agg(
            make_header(2, overall_frame=frame_base + 10),
            lap_data({"m_currentLapNum": 2, "m_lastLapTimeInMS": 90000, "m_currentLapInvalid": 0}),
            b"",
        )

    async def test_all_rows_merges_disk_and_pending(self, tmp_path) -> None:
        output = tmp_path / "laps.parquet"
        agg = LapAggregator(output)
        await self._produce_one_lap(agg, frame_base=100)
        assert agg.flush() == 1  # 1 row on disk, pending cleared

        # Produce a second pending lap (not yet flushed) with a distinct frame
        # base so its dedup key differs from the flushed row.
        await self._produce_one_lap(agg, frame_base=500)
        # all_rows merges disk (1) + pending (1) = 2, deduped.
        merged = agg.all_rows()
        assert len(merged) == 2
        assert all(r["lap_time_ms"] == 90000 for r in merged)

    async def test_all_rows_empty_when_nothing(self, tmp_path) -> None:
        agg = LapAggregator(tmp_path / "none.parquet")
        assert agg.all_rows() == []

    async def test_to_parquet_bytes_roundtrip(self, tmp_path) -> None:
        agg = LapAggregator(tmp_path / "laps.parquet")
        await self._produce_one_lap(agg)
        data = agg.to_parquet_bytes()
        assert isinstance(data, bytes) and len(data) > 0
        # Round-trip: bytes -> table.
        import io

        table = pq.read_table(io.BytesIO(data))
        assert table.num_rows == 1
        assert table.column("lap_time_ms")[0].as_py() == 90000

    async def test_to_parquet_bytes_empty_valid(self, tmp_path) -> None:
        agg = LapAggregator(tmp_path / "none.parquet")
        data = agg.to_parquet_bytes()
        import io

        table = pq.read_table(io.BytesIO(data))
        assert table.num_rows == 0
        assert table.num_columns == len(_SCHEMA)


# --------------------------------------------------------------------------- #
# Validation (smoke test of validate_sample via the listener path)
# --------------------------------------------------------------------------- #
class TestValidation:
    def test_validate_clean_telemetry(self) -> None:
        from f1opt.telemetry.validation import validate_sample

        parsed = {
            "m_carTelemetryData": [
                {"m_throttle": 0.5, "m_brake": 0.3, "m_steer": 0.0,
                 "m_tyresPressure": [21.0, 21.0, 21.0, 21.0],
                 "m_brakesTemperature": [100, 110, 105, 120],
                 "m_tyresSurfaceTemperature": [80, 82, 81, 83],
                 "m_tyresInnerTemperature": [85, 86, 87, 88],
                 "m_engineTemperature": 100}
            ]
        }
        ok, reason = validate_sample(6, parsed)
        assert ok is True
        assert reason is None

    def test_validate_bad_throttle(self) -> None:
        from f1opt.telemetry.validation import validate_sample

        parsed = {"m_carTelemetryData": [{"m_throttle": 1.5, "m_brake": 0.0, "m_steer": 0.0}]}
        ok, reason = validate_sample(6, parsed)
        assert ok is False
        assert "m_throttle" in reason

    def test_validate_negative_pressure(self) -> None:
        from f1opt.telemetry.validation import validate_sample

        parsed = {"m_carTelemetryData": [
            {"m_throttle": 0.5, "m_brake": 0.0, "m_steer": 0.0, "m_tyresPressure": [-1.0, 21.0, 21.0, 21.0]}
        ]}
        ok, reason = validate_sample(6, parsed)
        assert ok is False
        assert "m_tyresPressure" in reason

    def test_validate_wear_out_of_range(self) -> None:
        from f1opt.telemetry.validation import validate_sample

        parsed = {"m_carDamageData": [{"m_tyresWear": [150.0, 0.0, 0.0, 0.0], "m_tyresDamage": [0.0, 0.0, 0.0, 0.0]}]}
        ok, reason = validate_sample(10, parsed)
        assert ok is False
        assert "m_tyresWear" in reason


# --------------------------------------------------------------------------- #
# FrameTracker
# --------------------------------------------------------------------------- #
class TestFrameTracker:
    def test_first_observation(self) -> None:
        from f1opt.telemetry.validation import FrameTracker

        ft = FrameTracker()
        regressed, gap, delta = ft.observe(1, 100)
        assert regressed is False
        assert gap is False
        assert delta == 0

    def test_normal_progression(self) -> None:
        from f1opt.telemetry.validation import FrameTracker

        ft = FrameTracker()
        ft.observe(1, 100)
        regressed, gap, delta = ft.observe(1, 101)
        assert regressed is False
        assert gap is False
        assert delta == 1

    def test_gap_detected(self) -> None:
        from f1opt.telemetry.validation import FrameTracker

        ft = FrameTracker()
        ft.observe(1, 100)
        regressed, gap, delta = ft.observe(1, 105)
        assert regressed is False
        assert gap is True
        assert delta == 5

    def test_regression_detected(self) -> None:
        from f1opt.telemetry.validation import FrameTracker

        ft = FrameTracker()
        ft.observe(1, 100)
        regressed, gap, delta = ft.observe(1, 50)
        assert regressed is True
        assert gap is False
        assert delta == -50

    def test_per_session_isolation(self) -> None:
        from f1opt.telemetry.validation import FrameTracker

        ft = FrameTracker()
        ft.observe(1, 100)
        ft.observe(2, 200)
        # Session 1 and 2 are independent.
        assert ft.last(1) == 100
        assert ft.last(2) == 200
        # Session 3 is unknown.
        assert ft.last(3) is None
