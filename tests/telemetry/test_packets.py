"""Unit tests for :mod:`f1opt.telemetry.packets`.

Covers:
- ``PacketHeader`` size (29 bytes) and field round-trip.
- All 16 parsers run without crashing on a zero body (size-tolerant).
- All 16 parsers return the expected top-level structure.
- Specific field round-trip for Session / LapData / CarTelemetry / CarStatus.
- EA-verified total sizes for HIGH-confidence packets (LapPositions=1131,
  FinalClassification=1042).
"""

from __future__ import annotations

import struct

import pytest

from f1opt.telemetry.packets import (
    CONFIDENCE,
    HEADER_FORMAT,
    HEADER_SIZE,
    NUM_CARS,
    PACKET_NAMES,
    PACKET_PARSERS,
    packet_name,
    parse_header,
    parse_packet,
)

NUM_PACKETS = 17  # Iter-278: +CarTelemetryData2 (packet 16)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_header(
    packet_id: int,
    *,
    session_uid: int = 0x123456789ABCDEF0,
    session_time: float = 10.5,
    frame: int = 100,
    overall_frame: int = 200,
    player_car: int = 0,
    secondary: int = 255,
    game_year: int = 25,
) -> bytes:
    """Build a 29-byte F1 25 packet header."""
    return struct.pack(
        HEADER_FORMAT,
        2025,
        game_year,
        1,
        0,
        1,
        packet_id,
        session_uid,
        session_time,
        frame,
        overall_frame,
        player_car,
        secondary,
    )


def make_packet(packet_id: int, body: bytes = b"", **kw) -> bytes:
    return make_header(packet_id, **kw) + body


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
class TestHeader:
    def test_header_size_is_29(self) -> None:
        assert HEADER_SIZE == 29
        assert struct.calcsize(HEADER_FORMAT) == 29

    def test_header_field_round_trip(self) -> None:
        data = make_header(
            6,
            session_uid=42,
            session_time=1.25,
            frame=10,
            overall_frame=20,
            player_car=3,
            secondary=7,
        )
        h = parse_header(data)
        assert h.packet_format == 2025
        assert h.game_year == 25
        assert h.game_major_version == 1
        assert h.game_minor_version == 0
        assert h.packet_version == 1
        assert h.packet_id == 6
        assert h.session_uid == 42
        assert h.session_time == pytest.approx(1.25)
        assert h.frame_identifier == 10
        assert h.overall_frame_identifier == 20
        assert h.player_car_index == 3
        assert h.secondary_player_car_index == 7

    def test_header_name_property(self) -> None:
        assert parse_header(make_header(0)).name == "Motion"
        assert parse_header(make_header(15)).name == "LapPositions"

    def test_header_too_short_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_header(b"\x00" * 10)

    def test_num_cars_is_22(self) -> None:
        assert NUM_CARS == 22


# --------------------------------------------------------------------------- #
# All 16 parsers — smoke + structure
# --------------------------------------------------------------------------- #
class TestAllParsersSmoke:
    @pytest.mark.parametrize("pid", list(range(NUM_PACKETS)))
    def test_parser_returns_dict_on_zero_body(self, pid: int) -> None:
        parser = PACKET_PARSERS[pid]
        # Generous zero body; parsers are size-tolerant (pad/ignore).
        data = make_header(pid) + b"\x00" * 2048
        result = parser(data)
        assert isinstance(result, dict)
        assert len(result) > 0

    @pytest.mark.parametrize("pid", list(range(NUM_PACKETS)))
    def test_parse_packet_dispatches(self, pid: int) -> None:
        data = make_header(pid) + b"\x00" * 2048
        header, parsed = parse_packet(data)
        assert header.packet_id == pid
        assert isinstance(parsed, dict)

    @pytest.mark.parametrize("pid", list(range(NUM_PACKETS)))
    def test_parser_tolerates_truncated_body(self, pid: int) -> None:
        """A truncated body (just 1 byte) must not crash the parser."""
        parser = PACKET_PARSERS[pid]
        data = make_header(pid) + b"\x00"
        result = parser(data)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("pid", list(range(NUM_PACKETS)))
    def test_confidence_note_exists(self, pid: int) -> None:
        assert pid in CONFIDENCE
        assert isinstance(CONFIDENCE[pid], str)
        assert len(CONFIDENCE[pid]) > 0

    def test_packet_names_complete(self) -> None:
        assert len(PACKET_NAMES) == NUM_PACKETS
        for i in range(NUM_PACKETS):
            assert PACKET_NAMES[i] == packet_name(i)

    def test_unknown_packet_name(self) -> None:
        assert "Unknown" in packet_name(99)


# --------------------------------------------------------------------------- #
# Motion (id 0)
# --------------------------------------------------------------------------- #
class TestMotion:
    def test_structure(self) -> None:
        _, p = parse_packet(make_packet(0, b"\x00" * 1400))
        assert len(p["m_carMotionData"]) == NUM_CARS
        car0 = p["m_carMotionData"][0]
        assert "m_worldPositionX" in car0
        assert "m_yaw" in car0
        assert "m_frontWheelsAngle" in p
        assert len(p["m_suspensionPosition"]) == 4
        assert len(p["m_wheelSpeed"]) == 4


# --------------------------------------------------------------------------- #
# Session (id 1)
# --------------------------------------------------------------------------- #
class TestSession:
    def test_structure_zero_body(self) -> None:
        _, p = parse_packet(make_packet(1, b"\x00" * 400))
        assert p["m_weather"] == 0
        assert p["m_totalLaps"] == 0
        assert p["m_numMarshalZones"] == 0
        assert len(p["m_marshalZones"]) == 21
        assert len(p["m_weatherForecastSamples"]) == 20
        assert "m_pitStopWindowIdealLap" in p
        assert "m_dynamicRacingLineType" in p

    def test_leading_field_round_trip(self) -> None:
        # Leading 16 fields: BbbBHBbBHHBBBBBBB (19 bytes)
        body = struct.pack(
            "<BbbBHBbBHHBBBBBBB",
            2,     # weather
            30,    # trackTemp
            25,    # airTemp
            58,    # totalLaps
            5243,  # trackLength
            5,     # sessionType
            7,     # trackId
            1,     # formula
            1800,  # sessionTimeLeft
            3600,  # sessionDuration
            80,    # pitSpeedLimit
            0, 0, 0, 0, 0, 0,  # gamePaused..numMarshalZones
        )
        body += b"\x00" * 400
        _, p = parse_packet(make_packet(1, body))
        assert p["m_weather"] == 2
        assert p["m_trackTemperature"] == 30
        assert p["m_airTemperature"] == 25
        assert p["m_totalLaps"] == 58
        assert p["m_trackLength"] == 5243
        assert p["m_sessionType"] == 5
        assert p["m_trackId"] == 7
        assert p["m_formula"] == 1
        assert p["m_sessionTimeLeft"] == 1800
        assert p["m_sessionDuration"] == 3600
        assert p["m_pitSpeedLimit"] == 80


# --------------------------------------------------------------------------- #
# LapData (id 2)
# --------------------------------------------------------------------------- #
_LAP_PER_FMT = "<IIHBHBHBHBfff" + "B" * 15 + "HHBfB"  # Iter-281
_LAP_PER_SIZE = struct.calcsize(_LAP_PER_FMT)


class TestLapData:
    def test_structure(self) -> None:
        _, p = parse_packet(make_packet(2, b"\x00" * 1000))
        assert len(p["m_lapData"]) == NUM_CARS
        car0 = p["m_lapData"][0]
        assert "m_lastLapTimeInMS" in car0
        assert "m_currentLapNum" in car0
        assert "m_currentLapInvalid" in car0

    def test_car0_field_round_trip(self) -> None:
        car0_vals = (
            95000,   # lastLapTimeInMS (I)
            30000,   # currentLapTimeInMS (I)
            30000, 0,  # sector1TimeInMSPart (H) + MinutesPart (B)
            30000, 0,  # sector2TimeInMSPart (H) + MinutesPart (B)
            0, 0,      # deltaToCarInFrontInMSPart + MinutesPart
            0, 0,      # deltaToRaceLeaderInMSPart + MinutesPart
            1500.0,  # lapDistance (f)
            1500.0,  # totalDistance (f)
            0.0,     # safetyCarDelta (f)
            # 15 B fields: carPosition, currentLapNum, pitStatus, ...
            1,       # carPosition
            3,       # currentLapNum
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 13 remaining B fields
            # HHB: pitLaneTimeInLaneInMS, pitStopTimerInMS, pitStopShouldServePen
            0, 0, 0,
            # fB: speedTrapFastestSpeed, speedTrapFastestLap
            0.0, 0,
        )
        assert len(car0_vals) == 33
        car0_bytes = struct.pack(_LAP_PER_FMT, *car0_vals)
        body = car0_bytes + b"\x00" * (_LAP_PER_SIZE * (NUM_CARS - 1))
        _, p = parse_packet(make_packet(2, body))
        car0 = p["m_lapData"][0]
        assert car0["m_lastLapTimeInMS"] == 95000
        assert car0["m_currentLapTimeInMS"] == 30000
        assert car0["m_currentLapNum"] == 3
        assert car0["m_carPosition"] == 1
        assert car0["m_lapDistance"] == pytest.approx(1500.0)


# --------------------------------------------------------------------------- #
# CarTelemetry (id 6)
# --------------------------------------------------------------------------- #
_TELEM_PER_FMT = "<HfffBbHBBH4H4B4BB4f4B"  # Iter-278: engineTemperature 为 uint8
_TELEM_PER_SIZE = struct.calcsize(_TELEM_PER_FMT)


class TestCarTelemetry:
    def test_structure(self) -> None:
        _, p = parse_packet(make_packet(6, b"\x00" * 1400))
        assert len(p["m_carTelemetryData"]) == NUM_CARS
        car0 = p["m_carTelemetryData"][0]
        assert "m_speed" in car0
        assert "m_throttle" in car0
        assert "m_tyresPressure" in car0
        assert len(car0["m_brakesTemperature"]) == 4
        assert len(car0["m_tyresPressure"]) == 4
        assert "m_suggestedGear" in p

    def test_car0_field_round_trip(self) -> None:
        car0_vals = (
            310,       # speed (H)
            0.85,      # throttle (f)
            -0.2,      # steer (f)
            0.0,       # brake (f)
            0,         # clutch (B)
            7,         # gear (b)
            11000,     # engineRPM (H)
            1,         # drs (B)
            80,        # revLightsPercent (B)
            0,         # revLightsBitValue (H)
            400, 410, 420, 430,    # brakesTemperature (4H)
            90, 91, 92, 93,        # tyresSurfaceTemperature (4B)
            85, 86, 87, 88,        # tyresInnerTemperature (4B)
            105,                  # engineTemperature (H)
            21.5, 21.6, 21.7, 21.8,  # tyresPressure (4f)
            0, 1, 2, 3,            # surfaceType (4B)
        )
        car0_bytes = struct.pack(_TELEM_PER_FMT, *car0_vals)
        body = car0_bytes + b"\x00" * (_TELEM_PER_SIZE * (NUM_CARS - 1)) + b"\x00\x00\x00"
        _, p = parse_packet(make_packet(6, body))
        car0 = p["m_carTelemetryData"][0]
        assert car0["m_speed"] == 310
        assert car0["m_throttle"] == pytest.approx(0.85)
        assert car0["m_steer"] == pytest.approx(-0.2)
        assert car0["m_gear"] == 7
        assert car0["m_engineRPM"] == 11000
        assert car0["m_brakesTemperature"] == [400, 410, 420, 430]
        assert car0["m_tyresPressure"][0] == pytest.approx(21.5)


# --------------------------------------------------------------------------- #
# CarStatus (id 7)
# --------------------------------------------------------------------------- #
# Iter-278: 权威规范 26 字段 (含 enginePowerICE/MGUK + ersHarvestLimitPerLap)
_STATUS_PER_FMT = "<BBBBBfffHHBBHBBBbfffBffffB"
_STATUS_PER_SIZE = struct.calcsize(_STATUS_PER_FMT)


class TestCarStatus:
    def test_structure(self) -> None:
        _, p = parse_packet(make_packet(7, b"\x00" * 1100))
        assert len(p["m_carStatusData"]) == NUM_CARS
        car0 = p["m_carStatusData"][0]
        assert "m_ersDeployMode" in car0
        assert "m_fuelInTank" in car0
        assert "m_actualTyreCompound" in car0

    def test_car0_field_round_trip(self) -> None:
        car0_vals = (
            1, 1, 2, 50, 0,        # BBBBB: traction, abs, fuelMix, frontBrakeBias, pitLimiter
            100.5, 110.0, 5.5,     # fff: fuelInTank, fuelCapacity, fuelRemainingLaps
            12000, 5000,           # HH: maxRPM, idleRPM
            8, 1,                  # BB: maxGears, drsAllowed
            500,                   # H: drsActivationDistance
            22, 22,                # BB: actualTyreCompound (C6=22), visualTyreCompound
            3, 0,                  # Bb: tyresAgeLaps, vehicleFiaFlags
            400000.0, 350000.0,    # ff: enginePowerICE (W), enginePowerMGUK (W)
            50.0,                  # f: ersStoreEnergy (J)
            2,                     # B: ersDeployMode (hotlap)
            10.0, 20.0, 8.5, 30.0,  # ffff: ersHarvestedMGUK, MGUH, harvestLimit, deployed
            0,                     # B: networkPaused
        )
        car0_bytes = struct.pack(_STATUS_PER_FMT, *car0_vals)
        body = car0_bytes + b"\x00" * (_STATUS_PER_SIZE * (NUM_CARS - 1))
        _, p = parse_packet(make_packet(7, body))
        car0 = p["m_carStatusData"][0]
        assert car0["m_tractionControl"] == 1
        assert car0["m_fuelMix"] == 2
        assert car0["m_fuelInTank"] == pytest.approx(100.5)
        assert car0["m_maxRPM"] == 12000
        assert car0["m_actualTyreCompound"] == 22  # C6
        assert car0["m_ersDeployMode"] == 2
        assert car0["m_ersDeployedThisLap"] == pytest.approx(30.0)


# --------------------------------------------------------------------------- #
# CarSetups (id 5)
# --------------------------------------------------------------------------- #
class TestCarSetups:
    def test_structure(self) -> None:
        _, p = parse_packet(make_packet(5, b"\x00" * 900))
        assert len(p["m_carSetups"]) == NUM_CARS
        car0 = p["m_carSetups"][0]
        assert "m_frontWing" in car0
        assert "m_frontCamber" in car0
        assert "m_fuelLoad" in car0


# --------------------------------------------------------------------------- #
# Event (id 3)
# --------------------------------------------------------------------------- #
class TestEvent:
    def test_event_string_code(self) -> None:
        body = b"BTST" + b"\x00" * 8  # "BTST" event code + payload
        _, p = parse_packet(make_packet(3, body))
        assert p["m_eventStringCode"] == "BTST"
        assert isinstance(p["m_eventDetails"], bytes)


# --------------------------------------------------------------------------- #
# Participants (id 4)
# --------------------------------------------------------------------------- #
class TestParticipants:
    def test_structure(self) -> None:
        _, p = parse_packet(make_packet(4, b"\x00" * 1100))
        assert "m_numActiveCars" in p
        assert len(p["m_participants"]) == NUM_CARS
        assert "m_name" in p["m_participants"][0]


# --------------------------------------------------------------------------- #
# FinalClassification (id 8) — HIGH confidence, size verified
# --------------------------------------------------------------------------- #
class TestFinalClassification:
    def test_structure(self) -> None:
        _, p = parse_packet(make_packet(8, b"\x00" * 1100))
        assert "m_numCars" in p
        assert len(p["m_classificationData"]) == NUM_CARS
        car0 = p["m_classificationData"][0]
        assert "m_position" in car0
        assert "m_resultReason" in car0  # F1 25 field


# --------------------------------------------------------------------------- #
# LobbyInfo (id 9)
# --------------------------------------------------------------------------- #
class TestLobbyInfo:
    def test_structure(self) -> None:
        _, p = parse_packet(make_packet(9, b"\x00" * 1000))
        assert "m_numPlayers" in p
        assert len(p["m_lobbyPlayers"]) == NUM_CARS


# --------------------------------------------------------------------------- #
# CarDamage (id 10)
# --------------------------------------------------------------------------- #
class TestCarDamage:
    def test_structure(self) -> None:
        _, p = parse_packet(make_packet(10, b"\x00" * 1100))
        assert len(p["m_carDamageData"]) == NUM_CARS
        car0 = p["m_carDamageData"][0]
        assert "m_tyresWear" in car0
        assert "m_tyreBlisters" in car0  # F1 25 field
        assert len(car0["m_tyresWear"]) == 4
        assert len(car0["m_tyreBlisters"]) == 4


# --------------------------------------------------------------------------- #
# SessionHistory (id 11)
# --------------------------------------------------------------------------- #
class TestSessionHistory:
    def test_structure(self) -> None:
        _, p = parse_packet(make_packet(11, b"\x00" * 1200))
        assert "m_carIdx" in p
        assert "m_numLaps" in p
        assert len(p["m_lapHistoryData"]) == 100
        assert len(p["m_tyreStintsHistoryData"]) == 8


# --------------------------------------------------------------------------- #
# TyreSets (id 12)
# --------------------------------------------------------------------------- #
class TestTyreSets:
    def test_structure(self) -> None:
        _, p = parse_packet(make_packet(12, b"\x00" * 200))
        assert "m_carIdx" in p
        assert "m_fittedIdx" in p
        assert len(p["m_tyreSetData"]) == 13


# --------------------------------------------------------------------------- #
# MotionEx (id 13)
# --------------------------------------------------------------------------- #
class TestMotionEx:
    def test_structure(self) -> None:
        _, p = parse_packet(make_packet(13, b"\x00" * 300))
        assert "m_suspensionPosition" in p
        assert len(p["m_suspensionPosition"]) == 4
        assert "m_chassisPitch" in p  # F1 25 field
        assert "m_wheelCamber" in p   # F1 25 field
        assert len(p["m_wheelCamber"]) == 4


# --------------------------------------------------------------------------- #
# TimeTrial (id 14) — MEDIUM-LOW confidence (Iter-11 cross-check)
# --------------------------------------------------------------------------- #
class TestTimeTrial:
    def test_structure(self) -> None:
        _, p = parse_packet(make_packet(14, b"\x00" * 100))
        assert "m_timeTrialDataSet" in p
        assert "m_carIdx" in p["m_timeTrialDataSet"]

    def test_full_101b_body_padded(self) -> None:
        # EA PDF total = 101B (29 header + 72 body). Feed exactly 72B body.
        body = bytes(range(72))
        _, p = parse_packet(make_packet(14, body))
        assert p["_expected_body_size"] == 72
        assert p["_body_size"] == 72
        # First TimeTrialDataSet (BB + IIII = 2B + 16B = 18B) parsed from body head.
        assert p["m_timeTrialDataSet"]["m_carIdx"] == body[0]
        assert p["m_timeTrialDataSet"]["m_teamId"] == body[1]
        assert len(p["_remaining"]) == 72 - 18

    def test_truncated_body_padded(self) -> None:
        # Truncated body (only 5B) is zero-padded so parsing does not crash.
        _, p = parse_packet(make_packet(14, b"\x01\x02\x03\x04\x05"))
        assert p["m_timeTrialDataSet"]["m_carIdx"] == 1
        assert p["m_timeTrialDataSet"]["m_teamId"] == 2

    def test_confidence_notes_2026_season_pack(self) -> None:
        from f1opt.telemetry.packets import (
            CONFIDENCE_CARSTATUS,
            CONFIDENCE_CARTELEMETRY,
            CONFIDENCE_TIMETRIAL,
        )

        assert "2026 Season Pack" in CONFIDENCE_CARSTATUS
        assert "m_ersDeployMode" in CONFIDENCE_CARSTATUS
        assert "2026 Season Pack" in CONFIDENCE_CARTELEMETRY
        assert "MEDIUM-LOW" in CONFIDENCE_TIMETRIAL


# --------------------------------------------------------------------------- #
# LapPositions (id 15) — HIGH confidence, size verified
# --------------------------------------------------------------------------- #
class TestLapPositions:
    def test_structure(self) -> None:
        _, p = parse_packet(make_packet(15, b"\x00" * 1200))
        assert "m_numLaps" in p
        assert "m_lapStart" in p
        assert len(p["m_positionForVehicleIdx"]) == 50
        assert len(p["m_positionForVehicleIdx"][0]) == NUM_CARS


# --------------------------------------------------------------------------- #
# EA-verified total packet sizes (HIGH confidence)
# --------------------------------------------------------------------------- #
class TestVerifiedSizes:
    """Verify the EA-PDF-documented total packet sizes round-trip."""

    def test_lap_positions_size(self) -> None:
        # EA PDF: 1131 = 29 header + 2 + 50*22
        body = b"\x00" * (1131 - HEADER_SIZE)
        h, p = parse_packet(make_packet(15, body))
        assert h.packet_id == 15
        assert len(p["m_positionForVehicleIdx"]) == 50

    def test_final_classification_size(self) -> None:
        # EA PDF: 1042 = 29 header + 1 + 22*46
        body = b"\x00" * (1042 - HEADER_SIZE)
        h, p = parse_packet(make_packet(8, body))
        assert h.packet_id == 8
        assert len(p["m_classificationData"]) == NUM_CARS

    def test_motion_ex_size(self) -> None:
        # EA PDF: 273 total
        body = b"\x00" * (273 - HEADER_SIZE)
        h, p = parse_packet(make_packet(13, body))
        assert h.packet_id == 13
        assert "m_chassisPitch" in p
