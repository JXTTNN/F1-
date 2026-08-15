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
    validate_packet_size,
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

    def test_num_cars_is_24(self) -> None:
        assert NUM_CARS == 24  # F1 26 权威规范: 24 车位


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
        assert "m_gForceLateral" in car0
        assert "m_yaw" in car0


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
        assert len(p["m_weatherForecastSamples"]) == 64  # Iter-286: F1 26 64 样本
        assert "m_pitStopWindowIdealLap" in p
        assert "m_dynamicRacingLineType" in p

    def test_leading_field_round_trip(self) -> None:
        # Iter-286: 前 16 字段: BbbBHBbBHHBBBBBB (19 bytes)
        body = struct.pack(
            "<BbbBHBbBHHBBBBBB",
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
            0, 0, 0, 0, 0,  # gamePaused..numMarshalZones
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
_PART_PER_FMT = "<BHHHBBB32sBBHBB" + "BBB" * 4
_PART_PER_SIZE = struct.calcsize(_PART_PER_FMT)


class TestParticipants:
    def test_structure(self) -> None:
        _, p = parse_packet(make_packet(4, b"\x00" * 1400))
        assert "m_numActiveCars" in p
        assert len(p["m_participants"]) == NUM_CARS
        car0 = p["m_participants"][0]
        assert "m_name" in car0
        assert "m_platform" in car0

    def test_car0_field_round_trip(self) -> None:
        car0_vals = (
            0, 255, 100, 5, 0, 44, 1, b"LEWIS", 1, 1, 150, 3, 2,
            255, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0,
        )
        car0_bytes = struct.pack(_PART_PER_FMT, *car0_vals)
        body = b"\x01" + car0_bytes + b"\x00" * (_PART_PER_SIZE * (NUM_CARS - 1))
        _, p = parse_packet(make_packet(4, body))
        assert p["m_numActiveCars"] == 1
        car0 = p["m_participants"][0]
        assert car0["m_driverId"] == 255
        assert car0["m_networkId"] == 100
        assert car0["m_teamId"] == 5
        assert car0["m_name"] == "LEWIS"
        assert car0["m_techLevel"] == 150
        assert car0["m_platform"] == 3
        assert car0["m_numColours"] == 2
        assert car0["m_liveryColours0_r"] == 255
        assert car0["m_liveryColours1_g"] == 255


# --------------------------------------------------------------------------- #
# FinalClassification (id 8) — HIGH confidence, size verified
# --------------------------------------------------------------------------- #
_FC_PER_FMT = "<BBBBBBBIdBBB" + "B" * 24


class TestFinalClassification:
    def test_structure(self) -> None:
        _, p = parse_packet(make_packet(8, b"\x00" * 1100))
        assert "m_numCars" in p
        assert len(p["m_classificationData"]) == NUM_CARS
        car0 = p["m_classificationData"][0]
        assert "m_position" in car0
        assert "m_resultReason" in car0
        assert "m_penaltiesTime" in car0

    def test_car0_field_round_trip(self) -> None:
        car0_vals = (
            1, 53, 2, 25, 2, 3, 7,
            95500,
            5413.25,
            5, 2, 3,
            *([9] * 8), *([10] * 8), *([11] * 8),
        )
        car0_bytes = struct.pack(_FC_PER_FMT, *car0_vals)
        body = b"\x01" + car0_bytes + b"\x00" * (struct.calcsize(_FC_PER_FMT) * (NUM_CARS - 1))
        _, p = parse_packet(make_packet(8, body))
        car0 = p["m_classificationData"][0]
        assert car0["m_position"] == 1
        assert car0["m_resultReason"] == 7
        assert car0["m_bestLapTimeInMS"] == 95500
        assert car0["m_totalRaceTime"] == pytest.approx(5413.25)
        assert car0["m_penaltiesTime"] == 5
        assert car0["m_tyreStintsActual"] == [9] * 8


# --------------------------------------------------------------------------- #
# LobbyInfo (id 9)
# --------------------------------------------------------------------------- #
_LOBBY_PER_FMT = "<BHBB32sBBBHB"


class TestLobbyInfo:
    def test_structure(self) -> None:
        _, p = parse_packet(make_packet(9, b"\x00" * 1000))
        assert "m_numPlayers" in p
        assert len(p["m_lobbyPlayers"]) == NUM_CARS
        p0 = p["m_lobbyPlayers"][0]
        assert "m_platform" in p0
        assert "m_readyStatus" in p0

    def test_player0_field_round_trip(self) -> None:
        p0_vals = (0, 300, 1, 3, b"NICK", 5, 1, 1, 99, 1)
        p0_bytes = struct.pack(_LOBBY_PER_FMT, *p0_vals)
        body = b"\x01" + p0_bytes + b"\x00" * (struct.calcsize(_LOBBY_PER_FMT) * (NUM_CARS - 1))
        _, p = parse_packet(make_packet(9, body))
        p0 = p["m_lobbyPlayers"][0]
        assert p0["m_teamId"] == 300
        assert p0["m_platform"] == 3
        assert p0["m_name"] == "NICK"
        assert p0["m_techLevel"] == 99
        assert p0["m_readyStatus"] == 1


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
        _, p = parse_packet(make_packet(11, b"\x00" * 1500))
        assert "m_carIdx" in p
        assert "m_numLaps" in p
        assert "m_numTyreStints" in p
        assert "m_bestLapTimeLapNum" in p
        assert len(p["m_lapHistoryData"]) == 100
        assert len(p["m_tyreStintsHistoryData"]) == 8
        lap0 = p["m_lapHistoryData"][0]
        assert "m_sector1TimeMinutes" in lap0
        assert "m_lapValidBitFlags" in lap0
        stint0 = p["m_tyreStintsHistoryData"][0]
        assert "m_endLap" in stint0


# --------------------------------------------------------------------------- #
# TyreSets (id 12)
# --------------------------------------------------------------------------- #
class TestTyreSets:
    def test_structure(self) -> None:
        _, p = parse_packet(make_packet(12, b"\x00" * 250))
        assert "m_carIdx" in p
        assert "m_fittedIdx" in p
        assert len(p["m_tyreSetData"]) == 20  # 13 dry + 7 wet (Iter-283)
        s0 = p["m_tyreSetData"][0]
        assert "m_fitted" in s0
        assert "m_lapDeltaTime" in s0


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
        assert "m_playerSessionBestDataSet" in p
        assert "m_personalBestDataSet" in p
        assert "m_rivalDataSet" in p
        assert "m_carIdx" in p["m_playerSessionBestDataSet"]

    def test_three_datasets_round_trip(self) -> None:
        # 权威规范: 3 × 25B = 75B body (Iter-283)
        body = bytes(range(75))
        _, p = parse_packet(make_packet(14, body))
        player = p["m_playerSessionBestDataSet"]
        assert player["m_carIdx"] == 0
        assert player["m_teamId"] == 1 | (2 << 8)  # uint16 LE
        assert player["m_lapTimeInMS"] == int.from_bytes(body[3:7], "little")
        assert player["m_sector3TimeInMS"] == int.from_bytes(body[15:19], "little")
        assert player["m_tractionControl"] == body[19]
        assert player["m_valid"] == body[24]
        assert p["m_personalBestDataSet"]["m_carIdx"] == body[25]
        assert p["m_rivalDataSet"]["m_carIdx"] == body[50]

    def test_truncated_body_padded(self) -> None:
        # Truncated body (only 5B) is zero-padded so parsing does not crash.
        _, p = parse_packet(make_packet(14, b"\x01\x02\x03\x04\x05"))
        assert p["m_playerSessionBestDataSet"]["m_carIdx"] == 1
        assert p["m_playerSessionBestDataSet"]["m_teamId"] == 2 | (3 << 8)

    def test_confidence_notes_2026_season_pack(self) -> None:
        from f1opt.telemetry.packets import (
            CONFIDENCE_CARSTATUS,
            CONFIDENCE_CARTELEMETRY,
            CONFIDENCE_TIMETRIAL,
        )

        assert "2026 Season Pack" in CONFIDENCE_CARSTATUS
        assert "m_ersDeployMode" in CONFIDENCE_CARSTATUS
        assert "2026 Season Pack" in CONFIDENCE_CARTELEMETRY
        assert "HIGH" in CONFIDENCE_TIMETRIAL


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
# F1 26 权威规范总包大小 (含 29B 头) — MacManley/f1-26-udp README (Iter-286)
_AUTHORITATIVE_TOTAL_SIZES: dict[int, int] = {
    0: 1325,   # Motion (24 × 54B)
    1: 926,    # Session
    2: 1399,   # LapData (24 × 57 + 2 timeTrial)
    3: 45,     # Event (4 code + 12 details)
    4: 1470,   # Participants (1 + 24 × 60)
    5: 1233,   # CarSetups (24 × 50 + 4 nextFrontWing)
    6: 1448,   # CarTelemetry (24 × 59 + 3 trailer)
    7: 1445,   # CarStatus (24 × 59)
    8: 1134,   # FinalClassification (1 + 24 × 46)
    9: 1062,   # LobbyInfo (1 + 24 × 43)
    10: 1133,  # CarDamage (24 × 46)
    11: 1460,  # SessionHistory (7 + 100×14 + 8×3)
    12: 231,   # TyreSets (1 + 20×10 + 1)
    13: 273,   # MotionEx (61 float)
    14: 104,   # TimeTrial (3 × 25)
    15: 1231,  # LapPositions (2 + 50 × 24)
    16: 269,   # CarTelemetry2 (24 × 10)
}


class TestVerifiedSizes:
    """Verify F1 26 authoritative total packet sizes (MacManley/f1-26-udp)."""

    @pytest.mark.parametrize("pid", sorted(_AUTHORITATIVE_TOTAL_SIZES))
    def test_authoritative_size_round_trips(self, pid: int) -> None:
        total = _AUTHORITATIVE_TOTAL_SIZES[pid]
        body_size = total - HEADER_SIZE
        check = validate_packet_size(pid, body_size)
        assert check["ok"] is True, f"packet {pid}: {check}"
        data = make_packet(pid, b"\x00" * body_size)
        h, parsed = parse_packet(data)
        assert h.packet_id == pid
        assert isinstance(parsed, dict)

    def test_all_packet_ids_have_sizes(self) -> None:
        assert set(_AUTHORITATIVE_TOTAL_SIZES) == set(range(NUM_PACKETS))
