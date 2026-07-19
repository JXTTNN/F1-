"""Tests for the field-level validation closure loop.

Closes the gap: ``validation.py`` validates fields, but the listener only
``log.debug``'d failures and the aggregator did not mark samples as suspect.
These tests verify the closed loop from datagram validation → sample flag →
per-lap ``quality_flag`` column → ``clean`` decision:

- :class:`SampleFlag` / :func:`flag_severity` / :func:`merge_flags` /
  :func:`flag_sample` classify each frame into a severity-ordered flag.
- :class:`TelemetryListener` stamps a ``_flag`` key on non-OK frames (instead
  of only ``log.debug``), increments ``flagged_samples``, and exposes
  ``flag_counts()``.
- :class:`LapAggregator` tracks the per-lap worst flag, exposes it as a
  ``quality_flag`` Parquet column, and sets ``clean`` from both the
  flashback-dirty check and the worst flag severity against ``clean_threshold``.

The listener tests craft real CarTelemetry byte streams (29-byte header +
per-car body) and send them over UDP to an ephemeral-port listener, mirroring
``tests/telemetry/test_listener.py``. The aggregator tests call the subscriber
directly with dict-level payloads, mirroring ``tests/telemetry/test_aggregator.py``.
"""

from __future__ import annotations

import asyncio
import io
import socket
import struct

import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import FastAPI
from fastapi.testclient import TestClient

from f1opt.api.app import create_app
from f1opt.telemetry.aggregator import LapAggregator
from f1opt.telemetry.listener import TelemetryListener
from f1opt.telemetry.packets import HEADER_FORMAT, NUM_CARS, PacketHeader
from f1opt.telemetry.validation import (
    SampleFlag,
    flag_sample,
    flag_severity,
    merge_flags,
)

SESSION_UID = 0x123456789ABCDEF0

# CarTelemetry per-car struct format (must match f1opt/telemetry/packets.py).
_TELEM_PER_FMT = "<HfffBbHBBH4H4B4BH4f4B"
_TELEM_PER_SIZE = struct.calcsize(_TELEM_PER_FMT)


# --------------------------------------------------------------------------- #
# Helpers: byte-level CarTelemetry construction (for listener UDP tests)
# --------------------------------------------------------------------------- #
def make_header_bytes(
    packet_id: int,
    *,
    session_uid: int = SESSION_UID,
    session_time: float = 10.5,
    frame: int = 100,
    overall_frame: int = 200,
    player_car: int = 0,
) -> bytes:
    """Build a 29-byte F1 25 packet header."""
    return struct.pack(
        HEADER_FORMAT,
        2025, 25, 1, 0, 1, packet_id,
        session_uid, session_time, frame, overall_frame,
        player_car, 255,
    )


def make_car_telemetry_body(throttle: float = 0.5) -> bytes:
    """Build a CarTelemetry body with car 0 set, other 21 cars zeroed.

    Car 0 gets valid tyre pressures/temps so only ``throttle`` can trigger a
    validation failure (throttle > 1.0 or < 0.0).
    """
    car0 = struct.pack(
        _TELEM_PER_FMT,
        300,                                    # m_speed (H)
        throttle,                               # m_throttle (f)
        0.0,                                    # m_steer (f)
        0.0,                                    # m_brake (f)
        0,                                      # m_clutch (B)
        0,                                      # m_gear (b)
        0,                                      # m_engineRPM (H)
        0,                                      # m_drs (B)
        0,                                      # m_revLightsPercent (B)
        0,                                      # m_revLightsBitValue (H)
        100, 100, 100, 100,                     # m_brakesTemperature (4H)
        80, 80, 80, 80,                         # m_tyresSurfaceTemperature (4B)
        85, 85, 85, 85,                         # m_tyresInnerTemperature (4B)
        100,                                    # m_engineTemperature (H)
        21.0, 21.0, 21.0, 21.0,                 # m_tyresPressure (4f)
        0, 0, 0, 0,                             # m_surfaceType (4B)
    )
    return car0 + b"\x00" * (_TELEM_PER_SIZE * (NUM_CARS - 1)) + b"\x00\x00\x00"


def make_car_telemetry_packet(throttle: float = 0.5, **kw) -> bytes:
    return make_header_bytes(6, **kw) + make_car_telemetry_body(throttle=throttle)


async def _send_burst(port: int, packets: list[bytes]) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for pkt in packets:
            sock.sendto(pkt, ("127.0.0.1", port))
    finally:
        sock.close()


# --------------------------------------------------------------------------- #
# Helpers: dict-level construction (for aggregator direct-call tests)
# --------------------------------------------------------------------------- #
def make_header(
    packet_id: int,
    *,
    overall_frame: int = 100,
    session_time: float = 10.0,
    session_uid: int = SESSION_UID,
) -> PacketHeader:
    """Build a PacketHeader directly (no byte packing needed)."""
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


def _empty_lap() -> dict:
    return {
        "m_lastLapTimeInMS": 0, "m_currentLapTimeInMS": 0,
        "m_sector1TimeInMS": 0, "m_sector2TimeInMS": 0,
        "m_lapDistance": 0.0, "m_totalDistance": 0.0,
        "m_safetyCarDelta": 0.0, "m_carPosition": 0,
        "m_currentLapNum": 0, "m_pitStatus": 0, "m_numPitStops": 0,
        "m_sector": 0, "m_currentLapInvalid": 0, "m_penalties": 0,
        "m_totalWarnings": 0, "m_cornerCuttingWarnings": 0,
        "m_numUnservedDriveThroughPens": 0, "m_numUnservedStopGoPens": 0,
        "m_gridPosition": 0, "m_driverStatus": 0, "m_resultStatus": 0,
        "m_pitLaneTimerActive": 0, "m_pitLaneTimeInLaneInMS": 0,
        "m_pitStopTimerInMS": 0, "m_pitStopShouldServePen": 0,
    }


def telemetry_dict(
    throttle: float = 0.5,
    *,
    flag: str | None = None,
    validation_ok: bool | None = None,
    validation_reason: str | None = None,
) -> dict:
    """Build a CarTelemetry parsed payload for car 0 + 21 zeroed cars.

    ``flag`` attaches a ``_flag`` key (mimicking what
    ``TelemetryListener._on_datagram`` stamps via :func:`flag_sample`).
    ``validation_ok`` attaches a ``__validation__`` marker.
    """
    car0 = _empty_telem()
    car0["m_speed"] = 300
    car0["m_throttle"] = throttle
    car0["m_tyresPressure"] = [21.0, 21.0, 21.0, 21.0]
    car0["m_brakesTemperature"] = [100, 100, 100, 100]
    car0["m_tyresSurfaceTemperature"] = [80, 80, 80, 80]
    car0["m_tyresInnerTemperature"] = [85, 85, 85, 85]
    car0["m_engineTemperature"] = 100
    out: dict = {"m_carTelemetryData": [car0] + [_empty_telem()] * (NUM_CARS - 1)}
    if validation_ok is not None:
        out["__validation__"] = {"ok": validation_ok, "reason": validation_reason}
    if flag is not None:
        out["_flag"] = flag
    return out


def lap_data(car0: dict) -> dict:
    return {"m_lapData": [car0] + [_empty_lap()] * (NUM_CARS - 1)}


def _start_lap() -> dict:
    return lap_data({"m_currentLapNum": 1, "m_lastLapTimeInMS": 0, "m_currentLapInvalid": 0})


def _end_lap() -> dict:
    return lap_data({"m_currentLapNum": 2, "m_lastLapTimeInMS": 95000, "m_currentLapInvalid": 0})


async def _run_one_lap(
    agg: LapAggregator, frame_flags: list[str | None] | None = None
) -> list[dict]:
    """Drive a single complete lap through the aggregator and return its rows.

    ``frame_flags`` is a list of ``_flag`` values per CarTelemetry frame
    (``None`` → unflagged / OK frame). Defaults to two OK frames.
    """
    await agg(make_header(1, overall_frame=100), {"m_trackId": 7, "m_weather": 2}, b"")
    await agg(make_header(2, overall_frame=101), _start_lap(), b"")
    for i, flag in enumerate(frame_flags or [None, None]):
        await agg(make_header(6, overall_frame=102 + i), telemetry_dict(flag=flag), b"")
    await agg(make_header(2, overall_frame=110), _end_lap(), b"")
    return agg.rows


# --------------------------------------------------------------------------- #
# SampleFlag + flag_severity + merge_flags + flag_sample (pure, sync)
# --------------------------------------------------------------------------- #
class TestSampleFlag:
    def test_sample_flag_values_exist(self) -> None:
        """All five documented flags are present on the enum."""
        expected = {"OK", "SUSPECT_RANGE", "SUSPECT_STALE", "SUSPECT_FRAME_REGRESS", "INVALID"}
        assert {f.value for f in SampleFlag} == expected

    def test_flag_severity_ordering_ok_lt_suspect_lt_invalid(self) -> None:
        """flag_severity: OK=0, every SUSPECT_*=1, INVALID=2 (higher = worse)."""
        assert flag_severity("OK") == 0
        assert flag_severity(SampleFlag.SUSPECT_RANGE) == 1
        assert flag_severity(SampleFlag.SUSPECT_STALE) == 1
        assert flag_severity(SampleFlag.SUSPECT_FRAME_REGRESS) == 1
        assert flag_severity("INVALID") == 2
        # Ordering invariant: OK < any SUSPECT_* < INVALID.
        assert flag_severity("OK") < flag_severity("SUSPECT_RANGE")
        assert flag_severity("SUSPECT_FRAME_REGRESS") < flag_severity("INVALID")


class TestFlagSample:
    def test_flag_sample_valid_frame_returns_ok(self) -> None:
        """A valid frame with ok validation and no regression → 'OK'."""
        flag = flag_sample({}, {"ok": True, "reason": None}, (False, False, 1))
        assert flag == "OK"

    def test_flag_sample_range_violation(self) -> None:
        """A validation failure with a range reason → 'SUSPECT_RANGE'."""
        flag = flag_sample(
            {},
            {"ok": False, "reason": "car[0].m_throttle=1.5 out of range [0.0, 1.0]"},
            (False, False, 1),
        )
        assert flag == "SUSPECT_RANGE"

    def test_flag_sample_frame_regression(self) -> None:
        """A frame-tracker regression (first tuple element True) → SUSPECT_FRAME_REGRESS."""
        flag = flag_sample({}, {"ok": True, "reason": None}, (True, False, -5))
        assert flag == "SUSPECT_FRAME_REGRESS"

    def test_flag_sample_both_range_and_regress_returns_regress(self) -> None:
        """When a frame both fails a range check and regresses, the worst flag
        wins — SUSPECT_FRAME_REGRESS (higher detail rank than SUSPECT_RANGE)."""
        flag = flag_sample(
            {},
            {"ok": False, "reason": "car[0].m_throttle=1.5 out of range [0.0, 1.0]"},
            (True, False, -5),
        )
        assert flag == "SUSPECT_FRAME_REGRESS"

    def test_merge_flags_returns_worst(self) -> None:
        """merge_flags picks the highest-severity flag."""
        assert merge_flags(["OK", "SUSPECT_RANGE"]) == "SUSPECT_RANGE"
        assert merge_flags(["SUSPECT_RANGE", "INVALID"]) == "INVALID"
        # Tie within SUSPECT_* → detail rank breaks it (regress > range).
        assert merge_flags(["SUSPECT_RANGE", "SUSPECT_FRAME_REGRESS"]) == "SUSPECT_FRAME_REGRESS"

    def test_merge_flags_empty_returns_ok(self) -> None:
        """Edge case: an empty flag list collapses to 'OK'."""
        assert merge_flags([]) == "OK"


# --------------------------------------------------------------------------- #
# Listener: _flag stamping + flagged_samples + flag_counts (async, UDP)
# --------------------------------------------------------------------------- #
class TestListenerFlagging:
    async def test_flagged_samples_increments_on_invalid_frame(self) -> None:
        """A throttle=1.5 packet is flagged (SUSPECT_RANGE): flagged_samples==1
        and the parsed frame carries a ``_flag`` key instead of only log.debug."""
        listener = TelemetryListener("127.0.0.1", 0, queue_size=64)
        seen: list[dict] = []

        async def sub(header, parsed, raw):
            seen.append(parsed)

        listener.subscribe(sub)
        await listener.start()
        port = listener.bound_port
        assert port is not None
        try:
            await _send_burst(port, [make_car_telemetry_packet(throttle=1.5)])
            await asyncio.sleep(0.2)
            assert listener.received >= 1
            assert listener.validation_failures == 1
            assert listener.flagged_samples == 1
            assert len(seen) >= 1
            assert seen[0].get("_flag") == "SUSPECT_RANGE"
        finally:
            await listener.stop()

    async def test_flag_counts_returns_ok_and_suspect_tallies(self) -> None:
        """flag_counts() returns a dict with all flag keys and correct tallies
        after a mix of one invalid (SUSPECT_RANGE) and two valid (OK) frames."""
        listener = TelemetryListener("127.0.0.1", 0, queue_size=64)

        async def noop(header, parsed, raw):
            pass

        listener.subscribe(noop)
        await listener.start()
        port = listener.bound_port
        assert port is not None
        try:
            await _send_burst(
                port,
                [
                    make_car_telemetry_packet(throttle=1.5),  # SUSPECT_RANGE
                    make_car_telemetry_packet(throttle=0.5),  # OK
                    make_car_telemetry_packet(throttle=0.5),  # OK
                ],
            )
            await asyncio.sleep(0.25)
            assert listener.received >= 3
            counts = listener.flag_counts()
            # All five flag keys are always present.
            assert set(counts.keys()) == {
                "OK", "SUSPECT_RANGE", "SUSPECT_STALE", "SUSPECT_FRAME_REGRESS", "INVALID"
            }
            assert counts["SUSPECT_RANGE"] >= 1
            assert counts["OK"] >= 2
            assert listener.flagged_samples >= 1
        finally:
            await listener.stop()


# --------------------------------------------------------------------------- #
# Aggregator: quality_flag column + clean threshold (async, direct call)
# --------------------------------------------------------------------------- #
class TestAggregatorQualityFlag:
    async def test_lap_all_ok_frames_quality_ok_clean_true(self) -> None:
        """A lap with only unflagged (OK) frames → quality_flag='OK', clean=True."""
        agg = LapAggregator("/tmp/test_closure_ok.parquet")
        rows = await _run_one_lap(agg, frame_flags=[None, None])
        assert len(rows) == 1
        assert rows[0]["quality_flag"] == "OK"
        assert rows[0]["clean"] is True

    async def test_lap_with_suspect_range_frame_clean_true(self) -> None:
        """A lap with one SUSPECT_RANGE frame → quality_flag='SUSPECT_RANGE',
        clean=True under the default threshold ('SUSPECT')."""
        agg = LapAggregator("/tmp/test_closure_suspect.parquet")
        rows = await _run_one_lap(agg, frame_flags=["SUSPECT_RANGE", None])
        assert len(rows) == 1
        assert rows[0]["quality_flag"] == "SUSPECT_RANGE"
        assert rows[0]["clean"] is True

    async def test_lap_with_invalid_frame_clean_false(self) -> None:
        """A lap with an INVALID frame → quality_flag='INVALID', clean=False
        (severity 2 > default threshold severity 1)."""
        agg = LapAggregator("/tmp/test_closure_invalid.parquet")
        rows = await _run_one_lap(agg, frame_flags=["INVALID", None])
        assert len(rows) == 1
        assert rows[0]["quality_flag"] == "INVALID"
        assert rows[0]["clean"] is False

    async def test_clean_threshold_ok_makes_suspect_unclean(self) -> None:
        """With clean_threshold='OK' (severity 0), even a SUSPECT_RANGE lap is
        clean=False (severity 1 > 0)."""
        agg = LapAggregator("/tmp/test_closure_threshold.parquet", clean_threshold="OK")
        rows = await _run_one_lap(agg, frame_flags=["SUSPECT_RANGE", None])
        assert len(rows) == 1
        assert rows[0]["quality_flag"] == "SUSPECT_RANGE"
        assert rows[0]["clean"] is False

    async def test_to_parquet_bytes_roundtrip_includes_quality_flag(self, tmp_path) -> None:
        """to_parquet_bytes (existing export API) still works and includes the
        new quality_flag column with the worst flag value."""
        agg = LapAggregator(tmp_path / "laps.parquet")
        await _run_one_lap(agg, frame_flags=["SUSPECT_RANGE", None])
        data = agg.to_parquet_bytes()
        assert isinstance(data, bytes) and len(data) > 0
        table = pq.read_table(io.BytesIO(data))
        assert table.num_rows == 1
        assert "quality_flag" in table.column_names
        assert table.column("quality_flag")[0].as_py() == "SUSPECT_RANGE"


# --------------------------------------------------------------------------- #
# Backward compat: old Parquet without quality_flag column
# --------------------------------------------------------------------------- #
class TestBackwardCompat:
    def test_old_parquet_without_quality_flag_reads_ok(self, tmp_path) -> None:
        """A Parquet file written before the quality_flag column existed is
        read back with quality_flag defaulted to 'OK' (not a KeyError / null)."""
        old_schema = pa.schema([
            pa.field("session_uid", pa.uint64()),
            pa.field("car_index", pa.uint8()),
            pa.field("lap_number", pa.uint8()),
            pa.field("lap_time_ms", pa.uint32()),
            pa.field("clean", pa.bool_()),
            pa.field("invalid_reason", pa.string()),
        ])
        old_row = {
            "session_uid": 42,
            "car_index": 0,
            "lap_number": 1,
            "lap_time_ms": 90000,
            "clean": True,
            "invalid_reason": None,
        }
        path = tmp_path / "old.parquet"
        pq.write_table(pa.Table.from_pylist([old_row], schema=old_schema), path)

        agg = LapAggregator(path)
        rows = agg.all_rows()
        assert len(rows) == 1
        assert rows[0]["quality_flag"] == "OK"
        assert rows[0]["clean"] is True
        assert rows[0]["lap_time_ms"] == 90000


# --------------------------------------------------------------------------- #
# /api/health still exposes validation_failures (preserved observability)
# --------------------------------------------------------------------------- #
class TestHealthValidationFailures:
    def test_health_exposes_validation_failures_int(self) -> None:
        """GET /api/health returns 200 with a validation_failures int field."""
        app: FastAPI = create_app(start_listener=False)
        with TestClient(app) as client:
            r = client.get("/api/health")
            assert r.status_code == 200
            body = r.json()
            assert "validation_failures" in body
            assert isinstance(body["validation_failures"], int)
            assert body["validation_failures"] == 0

    def test_health_still_returns_core_fields(self) -> None:
        """The existing /api/health contract (status/version/udp_listening) is
        preserved alongside validation_failures."""
        app: FastAPI = create_app(start_listener=False)
        with TestClient(app) as client:
            r = client.get("/api/health")
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "ok"
            assert body["version"]
            assert body["udp_listening"] is False
            assert "validation_failures" in body
