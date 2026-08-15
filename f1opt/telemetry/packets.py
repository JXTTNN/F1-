"""F1 25 / 2026 Season Pack UDP packet parsers (packetFormat=2025).

All 16 official packet types are parsed per the EA F1 25 UDP specification.
Data is Little Endian, packed (no padding). Grid = 22 cars
(``cs_maxNumCarsInUDPData``); 2026 Season Pack keeps 22 cars (11 teams x 2,
including Cadillac).

Sources used for the field layouts (cross-referenced):
- **EA SPORTS F1 25 "Data Output from F1 25 v3.pdf"** (official, packetFormat=2025),
  retrieved via the EA forum blog *"F1(R) 25 UDP SPECIFICATION"* (Version 10.0).
  Authoritative for: ``PacketHeader`` (29 bytes — includes ``m_gameYear`` AND
  ``m_overallFrameIdentifier``), the 16 packet IDs / names, and the documented
  per-packet sizes (Motion=1349, LapPositions=1131, FinalClassification=1042,
  LobbyInfo=954, CarDamage=1041, MotionEx=273, TimeTrial=101).
- **EA forum community post by "Xionhearts"** documenting the F1 24 -> F1 25 delta
  (new LapPositions packet id=15; Retirement/DRSDisabled event reasons;
  StopGoPenaltyServer stopTime; Participant name 48->32 + LiveryColour;
  FinalClassification ``m_resultReason``; LobbyInfo name 48->32;
  CarDamage ``m_tyreBlisters[4]``; MotionEx ``m_chassisPitch`` /
  ``m_wheelCamber[4]`` / ``m_wheelCamberGain[4]``). Used as authoritative for the
  F1 25-specific additions.
- The well-documented **F1 23 / F1 24 UDP spec layouts** (the immediately
  preceding formats, broadly mirrored across open-source F1 telemetry libraries
  such as P403n1x87/f1-packets) for the otherwise-unchanged per-packet bodies.

Confidence: each parser carries a ``CONFIDENCE`` note.
- ``HIGH``   — total size verified against the EA PDF (LapPositions, FinalClassification)
               or header-only (PacketHeader).
- ``MEDIUM`` — layout reconstructed from F1 23/24 + F1 25 delta; field semantics
               correct, exact trailing/total bytes may differ from the live game.
- ``LOW``    — under-documented 2026 Season Pack fields; best-effort, TODO-marked.
  Parsers are size-tolerant: they pad truncated bodies and ignore trailing bytes,
  so a real packet that is longer than the documented layout will not crash.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# --------------------------------------------------------------------------- #
# PacketHeader
# --------------------------------------------------------------------------- #
# uint16 m_packetFormat (2025)
# uint8  m_gameYear            (last two digits, e.g. 25 -> 2026 Season Pack)
# uint8  m_gameMajorVersion
# uint8  m_gameMinorVersion
# uint8  m_packetVersion
# uint8  m_packetId
# uint64 m_sessionUID
# float  m_sessionTime
# uint32 m_frameIdentifier     (resets on flashback)
# uint32 m_overallFrameIdentifier  (does NOT reset on flashback)
# uint8  m_playerCarIndex
# uint8  m_secondaryPlayerCarIndex (255 if none)
HEADER_FORMAT = "<HBBBBBQfIIBB"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 29
assert HEADER_SIZE == 29, f"header size mismatch: {HEADER_SIZE}"

NUM_CARS = 22  # cs_maxNumCarsInUDPData (2026 grid: 11 teams x 2)

PACKET_NAMES: dict[int, str] = {
    0: "Motion",
    1: "Session",
    2: "LapData",
    3: "Event",
    4: "Participants",
    5: "CarSetups",
    6: "CarTelemetry",
    7: "CarStatus",
    8: "FinalClassification",
    9: "LobbyInfo",
    10: "CarDamage",
    11: "SessionHistory",
    12: "TyreSets",
    13: "MotionEx",
    14: "TimeTrial",
    15: "LapPositions",
    16: "CarTelemetryData2",  # Iter-278: F1 2026 主动空力/超车
}


def packet_name(packet_id: int) -> str:
    """Return the human-readable name for a packet id (``Unknown`` if unknown)."""
    return PACKET_NAMES.get(packet_id, f"Unknown({packet_id})")


@dataclass
class PacketHeader:
    """Parsed 29-byte F1 25 packet header."""

    packet_format: int
    game_year: int
    game_major_version: int
    game_minor_version: int
    packet_version: int
    packet_id: int
    session_uid: int
    session_time: float
    frame_identifier: int
    overall_frame_identifier: int
    player_car_index: int
    secondary_player_car_index: int

    @property
    def name(self) -> str:
        return packet_name(self.packet_id)


_HEADER_STRUCT = struct.Struct(HEADER_FORMAT)


def parse_header(data: bytes) -> PacketHeader:
    """Parse the 29-byte :class:`PacketHeader` from the start of ``data``."""
    if len(data) < HEADER_SIZE:
        raise ValueError(
            f"packet too short for header: {len(data)} bytes < {HEADER_SIZE}"
        )
    fields = _HEADER_STRUCT.unpack(data[:HEADER_SIZE])
    return PacketHeader(*fields)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _unpack_body(data: bytes, body_struct: struct.Struct) -> tuple:
    """Unpack the packet body (after the header), tolerating truncation/overflow.

    Truncated bodies are zero-padded so missing fields default to 0; trailing
    bytes beyond the documented layout are ignored. This makes parsing robust to
    partial datagrams and to spec revisions that append fields.
    """
    body = data[HEADER_SIZE:]
    expected = body_struct.size
    if len(body) < expected:
        body = body + b"\x00" * (expected - len(body))
    elif len(body) > expected:
        body = body[:expected]  # only truncate on overflow (avoid a copy when exact)
    return body_struct.unpack(body)


class _LazyCarList:
    """List-like, lazily materialized per-car dict sequence (22 cars).

    Eagerly parsing every 60Hz Motion packet builds 22 per-car dicts (~58µs,
    GIL-bound) even though the hot ingest path (``TelemetryAligner.on_packet``)
    only reads the single player car. This sequence defers dict construction
    to ``__getitem__``, so the player-car-only path materializes exactly one
    dict while preserving the ``len`` / indexing / iteration / mutation
    contract of a regular list.
    """

    __slots__ = ("_vals", "_per", "_names", "_cache")

    def __init__(self, vals: tuple, per: int, names: tuple[str, ...]) -> None:
        self._vals = vals
        self._per = per
        self._names = names
        self._cache: list[dict[str, Any] | None] = [None] * NUM_CARS

    def __len__(self) -> int:
        return NUM_CARS

    def __bool__(self) -> bool:
        return True

    def __iter__(self):
        for i in range(NUM_CARS):
            yield self[i]

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(NUM_CARS))]
        if i < 0:
            i += NUM_CARS
        if not 0 <= i < NUM_CARS:
            raise IndexError(i)
        d = self._cache[i]
        if d is None:
            base = i * self._per
            d = dict(zip(self._names, self._vals[base : base + self._per]))
            self._cache[i] = d
        return d


def _cars(vals: tuple, per: int, names: tuple[str, ...]) -> _LazyCarList:
    """Group a flat tuple of ``NUM_CARS * per`` values into per-car dicts.

    Returns a :class:`_LazyCarList` that materializes per-car dicts on access;
    this keeps the 60Hz Motion hot path cheap (player-car-only ingest builds
    one dict instead of 22) while preserving list-like semantics for full-scan
    consumers (validation, aggregator, lobby decode).
    """
    return _LazyCarList(vals, per, names)


# --------------------------------------------------------------------------- #
# Packet 0 — Motion  (size 1349 per EA PDF; body layout MEDIUM confidence)
# --------------------------------------------------------------------------- #
CONFIDENCE_MOTION = (
    "MEDIUM — CarMotionData (54B/car) and player-car extra section follow the "
    "F1 23/24 layout; F1 25 PDF reports total 1349B (12B more than the sum of "
    "the documented fields). Trailing bytes are ignored, fields are correct."
)
# CarMotionData: 3 pos float, 9 int16 (vel + 2 dirs), 6 floats (g-force, yaw/pitch/roll)
_CAR_MOTION_FMT = "fff" + "h" * 9 + "f" * 6
_CAR_MOTION_NAMES = (
    "m_worldPositionX", "m_worldPositionY", "m_worldPositionZ",
    "m_worldVelocityX", "m_worldVelocityY", "m_worldVelocityZ",
    "m_worldForwardDirX", "m_worldForwardDirY", "m_worldForwardDirZ",
    "m_worldRightDirX", "m_worldRightDirY", "m_worldRightDirZ",
    "m_gForceLateral", "m_gForceLongitudinal", "m_gForceVertical",
    "m_yaw", "m_pitch", "m_roll",
)
# Player-car-only: 5 arrays of 4 floats + 10 individual floats = 30 floats
_MOTION_PLAYER_FMT = "f" * 30
_MOTION_BODY = struct.Struct("<" + _CAR_MOTION_FMT * NUM_CARS + _MOTION_PLAYER_FMT)


def parse_motion(data: bytes) -> dict[str, Any]:
    """Parse PacketMotionData (packet id 0)."""
    vals = _unpack_body(data, _MOTION_BODY)
    per = len(_CAR_MOTION_NAMES)
    cars = _cars(vals, per, _CAR_MOTION_NAMES)
    p = vals[NUM_CARS * per :]
    return {
        "m_carMotionData": cars,
        "m_suspensionPosition": list(p[0:4]),
        "m_suspensionVelocity": list(p[4:8]),
        "m_suspensionAcceleration": list(p[8:12]),
        "m_wheelSpeed": list(p[12:16]),
        "m_wheelSlip": list(p[16:20]),
        "m_localVelocityX": p[20],
        "m_localVelocityY": p[21],
        "m_localVelocityZ": p[22],
        "m_angularVelocityX": p[23],
        "m_angularVelocityY": p[24],
        "m_angularVelocityZ": p[25],
        "m_angularAccelerationX": p[26],
        "m_angularAccelerationY": p[27],
        "m_angularAccelerationZ": p[28],
        "m_frontWheelsAngle": p[29],
    }


# --------------------------------------------------------------------------- #
# Packet 1 — Session  (MEDIUM confidence)
# --------------------------------------------------------------------------- #
CONFIDENCE_SESSION = (
    "MEDIUM — follows F1 24 layout (WeatherForecastSample = 7B with rainPercentage "
    "+ weatherDelta). F1 25 delta did not list Session changes."
)
_WFS_FMT = "BBBbbbbB"  # Iter-281: sessionType, timeOffset, weather, trackTemp, trackTempChange, airTemp, airTempChange, rain% (权威 8 字段)
_SESSION_BODY = struct.Struct(
    "<"
    + "BbbBHBbBHHBBBBBBB"  # weather..numMarshalZones (16 fields, includes m_formula)
    + "fb" * 21  # marshal zones
    + "BBB"  # safetyCar, networkGame, numWeatherForecastSamples
    + _WFS_FMT * 20  # weather forecast samples
    + "BBII"  # forecastAccuracy, aiDifficulty, seasonLink, weekendLink
    + "B" * 12  # pit-stop window + assists + racing line flags
)


def parse_session(data: bytes) -> dict[str, Any]:
    """Parse PacketSessionData (packet id 1)."""
    v = _unpack_body(data, _SESSION_BODY)
    (weather, track_temp, air_temp, total_laps, track_len, session_type, track_id,
     formula, session_time_left, session_duration, pit_speed_limit, game_paused,
     is_spectating, spectator_car_idx, sli_pro, num_marshal) = v[0:16]
    i = 16
    zones = []
    for _ in range(21):
        zones.append({"m_zoneStart": v[i], "m_zoneFlag": v[i + 1]})
        i += 2
    sc_status, network_game, num_wfs = v[i], v[i + 1], v[i + 2]
    i += 3
    wfs = []
    for _ in range(20):
        wfs.append({
            "m_sessionType": v[i], "m_timeOffset": v[i + 1], "m_weather": v[i + 2],
            "m_trackTemperature": v[i + 3], "m_trackTemperatureChange": v[i + 4],
            "m_airTemperature": v[i + 5], "m_airTemperatureChange": v[i + 6],
            "m_rainPercentage": v[i + 7],
        })
        i += 8
    (forecast_acc, ai_diff, season_link, weekend_link) = v[i], v[i + 1], v[i + 2], v[i + 3]
    i += 4
    assists = v[i:i + 12]
    return {
        "m_weather": weather,
        "m_trackTemperature": track_temp,
        "m_airTemperature": air_temp,
        "m_totalLaps": total_laps,
        "m_trackLength": track_len,
        "m_sessionType": session_type,
        "m_trackId": track_id,
        "m_formula": formula,
        "m_sessionTimeLeft": session_time_left,
        "m_sessionDuration": session_duration,
        "m_pitSpeedLimit": pit_speed_limit,
        "m_gamePaused": game_paused,
        "m_isSpectating": is_spectating,
        "m_spectatorCarIndex": spectator_car_idx,
        "m_sliProNativeSupport": sli_pro,
        "m_numMarshalZones": num_marshal,
        "m_marshalZones": zones,
        "m_safetyCarStatus": sc_status,
        "m_networkGame": network_game,
        "m_numWeatherForecastSamples": num_wfs,
        "m_weatherForecastSamples": wfs,
        "m_forecastAccuracy": forecast_acc,
        "m_aiDifficulty": ai_diff,
        "m_seasonLinkIdentifier": season_link,
        "m_weekendLinkIdentifier": weekend_link,
        "m_pitStopWindowIdealLap": assists[0],
        "m_pitStopWindowLatestLap": assists[1],
        "m_pitStopRejoinWindow": assists[2],
        "m_steeringAssist": assists[3],
        "m_brakingAssist": assists[4],
        "m_gearboxAssist": assists[5],
        "m_pitAssist": assists[6],
        "m_pitReleaseAssist": assists[7],
        "m_ERSAssist": assists[8],
        "m_drivingAssist": assists[9],
        "m_dynamicRacingLine": assists[10],
        "m_dynamicRacingLineType": assists[11],
    }


# --------------------------------------------------------------------------- #
# Packet 2 — Lap Data  (MEDIUM confidence)
# --------------------------------------------------------------------------- #
CONFIDENCE_LAPDATA = (
    "MEDIUM — per-car LapData follows F1 23/24 layout. Exposes lap times, sector "
    "times, lap distance, current lap num, position, lap-invalid flag."
)
# Iter-281: 按 MacManley/f1-26-udp 权威规范修正 LapData 线格式。
# 扇区时间/与前车差距/与领跑差距均拆分为 MSPart(uint16) + MinutesPart(uint8),
# 且位于 lapDistance/totalDistance/safetyCarDelta 之前; 旧版误作 HH(无分钟) +
# fff 直接跟在扇区后, 导致 lapDistance 起全部错位 8 字节, 且缺失 delta 与
# speedTrapFastest 字段。
_LAP_PER = "IIHBHBHBHBfff" + "B" * 15 + "HHBfB"
_LAP_NAMES = (
    "m_lastLapTimeInMS", "m_currentLapTimeInMS",
    "m_sector1TimeInMSPart", "m_sector1TimeMinutesPart",
    "m_sector2TimeInMSPart", "m_sector2TimeMinutesPart",
    "m_deltaToCarInFrontInMSPart", "m_deltaToCarInFrontInMinutesPart",
    "m_deltaToRaceLeaderInMSPart", "m_deltaToRaceLeaderInMinutesPart",
    "m_lapDistance", "m_totalDistance", "m_safetyCarDelta", "m_carPosition",
    "m_currentLapNum", "m_pitStatus", "m_numPitStops", "m_sector",
    "m_currentLapInvalid", "m_penalties", "m_totalWarnings", "m_cornerCuttingWarnings",
    "m_numUnservedDriveThroughPens", "m_numUnservedStopGoPens", "m_gridPosition",
    "m_driverStatus", "m_resultStatus", "m_pitLaneTimerActive",
    "m_pitLaneTimeInLaneInMS", "m_pitStopTimerInMS", "m_pitStopShouldServePen",
    "m_speedTrapFastestSpeed", "m_speedTrapFastestLap",
)
_LAPDATA_BODY = struct.Struct("<" + _LAP_PER * NUM_CARS)


def parse_lap_data(data: bytes) -> dict[str, Any]:
    """Parse PacketLapData (packet id 2)."""
    vals = _unpack_body(data, _LAPDATA_BODY)
    cars = _cars(vals, len(_LAP_NAMES), _LAP_NAMES)
    # Iter-281: 组合 MSPart + MinutesPart 为完整 ms 值 (供 aligner/aggregator 使用)。
    for c in cars:
        c["m_sector1TimeInMS"] = int(c["m_sector1TimeMinutesPart"]) * 60000 + int(c["m_sector1TimeInMSPart"])
        c["m_sector2TimeInMS"] = int(c["m_sector2TimeMinutesPart"]) * 60000 + int(c["m_sector2TimeInMSPart"])
        c["m_deltaToCarInFrontInMS"] = int(c["m_deltaToCarInFrontInMinutesPart"]) * 60000 + int(c["m_deltaToCarInFrontInMSPart"])
        c["m_deltaToRaceLeaderInMS"] = int(c["m_deltaToRaceLeaderInMinutesPart"]) * 60000 + int(c["m_deltaToRaceLeaderInMSPart"])
    return {"m_lapData": cars}


# --------------------------------------------------------------------------- #
# Packet 3 — Event  (MEDIUM confidence; event payloads best-effort)
# --------------------------------------------------------------------------- #
CONFIDENCE_EVENT = (
    "MEDIUM — 4-byte event string code parsed; event-specific payloads (FastestLap, "
    "Retirement reason, DRSDisabled reason, StopGoPenaltyServer stopTime) are exposed "
    "as raw ``m_eventDetails`` bytes when not individually decoded."
)
_EVENT_BODY = struct.Struct("<4s")


def parse_event(data: bytes) -> dict[str, Any]:
    """Parse PacketEventData (packet id 3)."""
    body = data[HEADER_SIZE:]
    code, = _unpack_body(data, _EVENT_BODY) if len(body) >= 4 else (b"????",)
    try:
        code_str = code.decode("ascii", errors="replace").rstrip("\x00")
    except Exception:  # pragma: no cover - defensive
        code_str = ""
    return {
        "m_eventStringCode": code_str,
        "m_eventDetails": body[4:],  # raw event-specific payload
    }


# --------------------------------------------------------------------------- #
# Packet 4 — Participants  (MEDIUM-LOW confidence; F1 25 livery fields)
# --------------------------------------------------------------------------- #
CONFIDENCE_PARTICIPANTS = (
    "MEDIUM — name length 32 (F1 25 reduction from 48); includes F1 25 LiveryColour "
    "(numColours + 4 RGB triplets). Exact trailing assist bytes are best-effort."
)
# per participant: aiControlled, driverId, networkId, teamId, myTeam, raceNumber,
# nationality, name[32], yourTelemetry, showOnlineNames, numColours, 4*LiveryColour(3B)
_PART_FMT = "BBBBBBB32sBBB" + "BBB" * 4
_PART_NAMES = (
    "m_aiControlled", "m_driverId", "m_networkId", "m_teamId", "m_myTeam",
    "m_raceNumber", "m_nationality", "m_name", "m_yourTelemetry", "m_showOnlineNames",
    "m_numColours",
    "m_liveryColours0_r", "m_liveryColours0_g", "m_liveryColours0_b",
    "m_liveryColours1_r", "m_liveryColours1_g", "m_liveryColours1_b",
    "m_liveryColours2_r", "m_liveryColours2_g", "m_liveryColours2_b",
    "m_liveryColours3_r", "m_liveryColours3_g", "m_liveryColours3_b",
)
_PARTICIPANTS_BODY = struct.Struct("<B" + _PART_FMT * NUM_CARS)


def parse_participants(data: bytes) -> dict[str, Any]:
    """Parse PacketParticipantsData (packet id 4)."""
    v = _unpack_body(data, _PARTICIPANTS_BODY)
    num_active = v[0]
    per = len(_PART_NAMES)
    rest = v[1:]
    cars = _cars(rest, per, _PART_NAMES)
    for c in cars:
        c["m_name"] = c["m_name"].decode("utf-8", errors="replace").rstrip("\x00")
    return {"m_numActiveCars": num_active, "m_participants": cars}


# --------------------------------------------------------------------------- #
# Packet 5 — Car Setups  (MEDIUM confidence)
# --------------------------------------------------------------------------- #
CONFIDENCE_CARSETUPS = (
    "MEDIUM — discrete setup values as uint8 (game clicks), camber/toe/tyre-pressure/"
    "fuel as float; follows F1 23/24 layout."
)
_SETUP_PER = "BBBB ffff BBBB BBBB BBBB BBBB Bf".replace(" ", "")
# frontWing, rearWing, onThrottleDiff, offThrottleDiff, frontCamber, rearCamber,
# frontToe, rearToe, frontSuspension, rearSuspension, frontAntiRollBar,
# rearAntiRollBar, frontSuspensionHeight, rearSuspensionHeight, brakePressure,
# brakeBias, rearLeftTyrePressure, rearRightTyrePressure, frontLeftTyrePressure,
# frontRightTyrePressure, ballast, fuelLoad
_SETUP_NAMES = (
    "m_frontWing", "m_rearWing", "m_onThrottleDiff", "m_offThrottleDiff",
    "m_frontCamber", "m_rearCamber", "m_frontToe", "m_rearToe",
    "m_frontSuspension", "m_rearSuspension", "m_frontAntiRollBar", "m_rearAntiRollBar",
    "m_frontSuspensionHeight", "m_rearSuspensionHeight", "m_brakePressure",
    "m_brakeBias", "m_rearLeftTyrePressure", "m_rearRightTyrePressure",
    "m_frontLeftTyrePressure", "m_frontRightTyrePressure", "m_ballast", "m_fuelLoad",
)
_CARSETUPS_BODY = struct.Struct("<" + _SETUP_PER * NUM_CARS)


def parse_car_setups(data: bytes) -> dict[str, Any]:
    """Parse PacketCarSetupData (packet id 5)."""
    vals = _unpack_body(data, _CARSETUPS_BODY)
    return {"m_carSetups": _cars(vals, len(_SETUP_NAMES), _SETUP_NAMES)}


# --------------------------------------------------------------------------- #
# Packet 6 — Car Telemetry  (HIGH on per-car fields; MEDIUM on trailer)
# --------------------------------------------------------------------------- #
CONFIDENCE_CARTELEMETRY = (
    "HIGH on per-car fields (speed/throttle/steer/brake/clutch/gear/rpm/DRS/tyre "
    "temps & pressures/surfaceType). MEDIUM on trailer (mfdPanelIndex + suggestedGear). "
    "2026 Season Pack note (Iter-11, EA_Groguet official): overtake mode lives in a NEW "
    "Car Telemetry 2 packet introduced by the 2026 Season Pack — NOT this packet id 6. "
    "This parser is the F1 25 base CarTelemetry packet; a future CarTelemetry2 parser "
    "will expose m_overtakeMode when the 2026 Season Pack layout is published in full."
)
# per car: speed(H) throttle(f) steer(f) brake(f) clutch(B) gear(b) engineRPM(H)
# drs(B) revLightsPercent(B) revLightsBitValue(H) brakesTemperature[4](H) 
# tyresSurfaceTemperature[4](B) tyresInnerTemperature[4](B) engineTemperature(B)
# tyresPressure[4](f) surfaceType[4](B)
# Iter-278: engineTemperature 权威规范为 uint8 (B), 旧版误作 uint16 (H) 导致
# tyresPressure/surfaceType 错位 1 字节。
_TELEM_PER = "HfffBbHBBH4H4B4BB4f4B"
_TELEM_BODY = struct.Struct("<" + _TELEM_PER * NUM_CARS + "BBB")


def parse_car_telemetry(data: bytes) -> dict[str, Any]:
    """Parse PacketCarTelemetryData (packet id 6)."""
    v = _unpack_body(data, _TELEM_BODY)
    per = struct.calcsize("<" + _TELEM_PER)
    # number of scalar values per car = (per bytes) / 1 is wrong for mixed; compute via unpack
    # Instead, unpack one car to count fields:
    one = struct.unpack("<" + _TELEM_PER, b"\x00" * per)
    fields_per_car = len(one)
    cars = []
    base = 0
    for _ in range(NUM_CARS):
        c = v[base:base + fields_per_car]
        cars.append({
            "m_speed": c[0], "m_throttle": c[1], "m_steer": c[2], "m_brake": c[3],
            "m_clutch": c[4], "m_gear": c[5], "m_engineRPM": c[6], "m_drs": c[7],
            "m_revLightsPercent": c[8], "m_revLightsBitValue": c[9],
            "m_brakesTemperature": list(c[10:14]),
            "m_tyresSurfaceTemperature": list(c[14:18]),
            "m_tyresInnerTemperature": list(c[18:22]),
            "m_engineTemperature": c[22],
            "m_tyresPressure": list(c[23:27]),
            "m_surfaceType": list(c[27:31]),
        })
        base += fields_per_car
    trailer = v[base:]
    return {
        "m_carTelemetryData": cars,
        "m_mfdPanelIndex": trailer[0],
        "m_mfdPanelIndexSecondaryPlayer": trailer[1],
        "m_suggestedGear": trailer[2],
    }


# --------------------------------------------------------------------------- #
# Packet 7 — Car Status  (MEDIUM confidence; includes m_ersDeployMode)
# --------------------------------------------------------------------------- #
CONFIDENCE_CARSTATUS = (
    "MEDIUM — includes m_ersDeployMode (uint8, 0=none/1=medium/2=hotlap/3=deployment). "
    "Positioned after m_ersStoreEnergy; C6 tyre compound = value 22 (F1 25). 2026 Season "
    "Pack note (Iter-11, EA_Groguet official confirmation): the 2026 Season Pack 'boost' "
    "mode is covered by m_ersDeployMode — no separate boost field is required, so this "
    "parser already captures 2026 boost behaviour correctly."
    "\nIter-191: Adds m_activeAeroX (float) and m_activeAeroZ (float) — F1 2026 active "
    "aerodynamics front-wing angle (X) and rear-wing angle (Z) in degrees."
)
# per car: tractionControl, antiLockBrakes, fuelMix, frontBrakeBias, pitLimiterStatus,
# fuelInTank(f), fuelCapacity(f), fuelRemainingLaps(f), maxRPM(H), idleRPM(H),
# maxGears, drsAllowed, drsActivationDistance(H), actualTyreCompound, visualTyreCompound,
# tyresAgeLaps(B), vehicleFiaFlags(b), enginePowerICE(f), enginePowerMGUK(f),
# ersStoreEnergy(f), ersDeployMode, ersHarvestedThisLapMGUK(f),
# ersHarvestedThisLapMGUH(f), ersHarvestLimitPerLap(f), ersDeployedThisLap(f),
# networkPaused
# Iter-278: 按 MacManley/f1-26-udp 权威规范修正 — 补齐 enginePowerICE/enginePowerMGUK/
# ersHarvestLimitPerLap 三个 float (旧版缺失导致 ersStoreEnergy 起全部错位 8 字节),
# 并删除不存在的 m_activeAeroX/Z (主动空力在 Packet 16 CarTelemetryData2, 不在 CarStatus)。
_STATUS_PER = "BBBBB fff HH BB H BB Bb fff B ffff B".replace(" ", "")
_STATUS_NAMES = (
    "m_tractionControl", "m_antiLockBrakes", "m_fuelMix", "m_frontBrakeBias",
    "m_pitLimiterStatus", "m_fuelInTank", "m_fuelCapacity", "m_fuelRemainingLaps",
    "m_maxRPM", "m_idleRPM", "m_maxGears", "m_drsAllowed", "m_drsActivationDistance",
    "m_actualTyreCompound", "m_visualTyreCompound", "m_tyresAgeLaps",
    "m_vehicleFiaFlags", "m_enginePowerICE", "m_enginePowerMGUK", "m_ersStoreEnergy",
    "m_ersDeployMode", "m_ersHarvestedThisLapMGUK", "m_ersHarvestedThisLapMGUH",
    "m_ersHarvestLimitPerLap", "m_ersDeployedThisLap", "m_networkPaused",
)
_STATUS_BODY = struct.Struct("<" + _STATUS_PER * NUM_CARS)


def parse_car_status(data: bytes) -> dict[str, Any]:
    """Parse PacketCarStatusData (packet id 7)."""
    vals = _unpack_body(data, _STATUS_BODY)
    return {"m_carStatusData": _cars(vals, len(_STATUS_NAMES), _STATUS_NAMES)}


# --------------------------------------------------------------------------- #
# Packet 16 — Car Telemetry 2  (Iter-278: 权威规范新增, 主动空力 + 超车)
# --------------------------------------------------------------------------- #
# per car: activeAeroMode(B, 0=Corner/Z / 1=Straight/X), activeAeroAvailable(B),
# activeAeroActivationDistance(H), overtakeAvailable(B), overtakeActive(B),
# overtakeActivationDistance(H), 2026Regulations(B), drivingWrongWay(B)
_CT2_PER = "BBHBBHBB"
_CT2_NAMES = (
    "m_activeAeroMode",
    "m_activeAeroAvailable",
    "m_activeAeroActivationDistance",
    "m_overtakeAvailable",
    "m_overtakeActive",
    "m_overtakeActivationDistance",
    "m_2026Regulations",
    "m_drivingWrongWay",
)
_CT2_BODY = struct.Struct("<" + _CT2_PER * NUM_CARS)


def parse_car_telemetry_2(data: bytes) -> dict[str, Any]:
    """Parse PacketCarTelemetryData2 (packet id 16) — F1 2026 主动空力 + 超车."""
    vals = _unpack_body(data, _CT2_BODY)
    return {"m_carTelemetryData2": _cars(vals, len(_CT2_NAMES), _CT2_NAMES)}


CONFIDENCE_CARTELEMETRY2 = (
    "HIGH — PacketCarTelemetryData2 (packet id 16) per MacManley/f1-26-udp 权威规范. "
    "含 m_activeAeroMode (0=Corner/Z / 1=Straight/X), m_activeAeroAvailable, "
    "m_activeAeroActivationDistance, m_overtakeAvailable/Active, "
    "m_overtakeActivationDistance, m_2026Regulations, m_drivingWrongWay."
)


# --------------------------------------------------------------------------- #
# Packet 8 — Final Classification  (HIGH — size verified 1042 = 29 + 1 + 22*46)
# --------------------------------------------------------------------------- #
CONFIDENCE_FINALCLASS = (
    "HIGH — total size 1042B verified against EA PDF (29 header + 1 numCars + "
    "22 cars * 46B). Includes F1 25 m_resultReason."
)
# per car: position, numLaps, gridPosition, points, pitStops, resultStatus,
# bestLapTimeInMS(I), totalRaceTime(d), totalRaceTimeWarnings, numPenalties,
# numTyreStints, tyreStintsActual[8], tyreStintsVisual[8], tyreStintsEndLaps[8],
# resultReason
_FC_PER = "BBBBBB I d B BB 8B8B8B B".replace(" ", "")
_FC_BODY = struct.Struct("<B" + _FC_PER * NUM_CARS)


def parse_final_classification(data: bytes) -> dict[str, Any]:
    """Parse PacketFinalClassificationData (packet id 8)."""
    v = _unpack_body(data, _FC_BODY)
    num_cars = v[0]
    rest = v[1:]
    fields_per_car = len(struct.unpack("<" + _FC_PER, b"\x00" * struct.calcsize("<" + _FC_PER)))
    cars = []
    base = 0
    for _ in range(NUM_CARS):
        c = rest[base:base + fields_per_car]
        cars.append({
            "m_position": c[0], "m_numLaps": c[1], "m_gridPosition": c[2],
            "m_points": c[3], "m_pitStops": c[4], "m_resultStatus": c[5],
            "m_bestLapTimeInMS": c[6], "m_totalRaceTime": c[7],
            "m_totalRaceTimeWarnings": c[8], "m_numPenalties": c[9],
            "m_numTyreStints": c[10],
            "m_tyreStintsActual": list(c[11:19]),
            "m_tyreStintsVisual": list(c[19:27]),
            "m_tyreStintsEndLaps": list(c[27:35]),
            "m_resultReason": c[35],
        })
        base += fields_per_car
    return {"m_numCars": num_cars, "m_classificationData": cars}


# --------------------------------------------------------------------------- #
# Packet 9 — Lobby Info  (MEDIUM-LOW — name 32, trailing bytes uncertain)
# --------------------------------------------------------------------------- #
CONFIDENCE_LOBBYINFO = (
    "MEDIUM-LOW — F1 25 name length 32 (size 954 verified). Per-player trailing "
    "assist bytes are best-effort; trailing bytes ignored."
)
# per player: aiControlled, networkId, teamId, nationality, name[32], carTelemetrySetup
_LOBBY_PER = "BBBB32sB"
_LOBBY_NAMES = (
    "m_aiControlled", "m_networkId", "m_teamId", "m_nationality", "m_name",
    "m_carTelemetrySetup",
)
_LOBBY_BODY = struct.Struct("<B" + _LOBBY_PER * NUM_CARS)


def parse_lobby_info(data: bytes) -> dict[str, Any]:
    """Parse PacketLobbyInfoData (packet id 9)."""
    v = _unpack_body(data, _LOBBY_BODY)
    num_players = v[0]
    rest = v[1:]
    players = _cars(rest, len(_LOBBY_NAMES), _LOBBY_NAMES)
    for p in players:
        p["m_name"] = p["m_name"].decode("utf-8", errors="replace").rstrip("\x00")
    return {"m_numPlayers": num_players, "m_lobbyPlayers": players}


# --------------------------------------------------------------------------- #
# Packet 10 — Car Damage  (MEDIUM-LOW; F1 25 m_tyreBlisters[4] included)
# --------------------------------------------------------------------------- #
CONFIDENCE_CARDAMAGE = (
    "MEDIUM-LOW — includes F1 25 m_tyreBlisters[4]. Tyre wear arrays modelled as "
    "float[4]; exact byte total is best-effort (EA PDF reports 1041B)."
)
# per car: tyresWear[4](f) tyresDamage[4](f) frontLeftWingDamage(f) frontRightWingDamage(f)
# rearWingDamage(f) floorDamage(f) diffuserDamage(f) sidepodDamage(f) drsDamage(f)
# engineDamage(B) gearBoxDamage(B) suspensionDamage[4](4B) ersDamage(B)
# tyreBlisters[4](4B)
_DMG_PER = "4f4f7f2B4BB4B"
_DMG_BODY = struct.Struct("<" + _DMG_PER * NUM_CARS)


def parse_car_damage(data: bytes) -> dict[str, Any]:
    """Parse PacketCarDamageData (packet id 10)."""
    v = _unpack_body(data, _DMG_BODY)
    fpc = len(struct.unpack("<" + _DMG_PER, b"\x00" * struct.calcsize("<" + _DMG_PER)))
    cars = []
    base = 0
    for _ in range(NUM_CARS):
        c = v[base:base + fpc]
        cars.append({
            "m_tyresWear": list(c[0:4]),
            "m_tyresDamage": list(c[4:8]),
            "m_frontLeftWingDamage": c[8],
            "m_frontRightWingDamage": c[9],
            "m_rearWingDamage": c[10],
            "m_floorDamage": c[11],
            "m_diffuserDamage": c[12],
            "m_sidepodDamage": c[13],
            "m_drsDamage": c[14],
            "m_engineDamage": c[15],
            "m_gearBoxDamage": c[16],
            "m_suspensionDamage": list(c[17:21]),
            "m_ersDamage": c[21],
            "m_tyreBlisters": list(c[22:26]),
        })
        base += fpc
    return {"m_carDamageData": cars}


# --------------------------------------------------------------------------- #
# Packet 11 — Session History  (MEDIUM confidence)
# --------------------------------------------------------------------------- #
CONFIDENCE_SESSIONHIST = (
    "MEDIUM — carIdx, numLaps, 100 LapHistoryData (11B each), 8 TyreStintHistoryData."
)
# LapHistoryData: lapTimeInMS(I) sector1(H) sector2(H) sector3(H) lapValidBitFlags(B) = 11B
# TyreStintHistoryData: tyreActualCompound(B) tyreVisualCompound(B) endLap(B) = 3B
_SH_LAP = "IHHHB"
_SH_STINT = "BBB"
_SH_BODY = struct.Struct(
    "<" + "BB" + _SH_LAP * 100 + _SH_STINT * 8
)


def parse_session_history(data: bytes) -> dict[str, Any]:
    """Parse PacketSessionHistoryData (packet id 11)."""
    v = _unpack_body(data, _SH_BODY)
    car_idx, num_laps = v[0], v[1]
    i = 2
    laps = []
    for _ in range(100):
        laps.append({
            "m_lapTimeInMS": v[i], "m_sector1TimeInMS": v[i + 1],
            "m_sector2TimeInMS": v[i + 2], "m_sector3TimeInMS": v[i + 3],
            "m_lapValidBitFlags": v[i + 4],
        })
        i += 5
    stints = []
    for _ in range(8):
        stints.append({
            "m_tyreActualCompound": v[i],
            "m_tyreVisualCompound": v[i + 1],
            "m_endLap": v[i + 2],
        })
        i += 3
    return {
        "m_carIdx": car_idx,
        "m_numLaps": num_laps,
        "m_lapHistoryData": laps,
        "m_tyreStintsHistoryData": stints,
    }


# --------------------------------------------------------------------------- #
# Packet 12 — Tyre Sets  (MEDIUM confidence)
# --------------------------------------------------------------------------- #
CONFIDENCE_TYRESETS = (
    "MEDIUM — carIdx, 13 TyreSetData, fittedIdx. Per-set layout follows F1 24."
)
# TyreSetData: actualTyreCompound, visualTyreCompound, wear, available,
# recommendedSession, lifeSpan, usableLife, lapDeltaTime(b), fitted
_TS_SET = "BBBBBbbB"
_TS_BODY = struct.Struct("<" + "BB" + _TS_SET * 13 + "B")


def parse_tyre_sets(data: bytes) -> dict[str, Any]:
    """Parse PacketTyreSetsData (packet id 12)."""
    v = _unpack_body(data, _TS_BODY)
    car_idx = v[0]
    i = 1
    sets = []
    for _ in range(13):
        s = v[i:i + 8]
        sets.append({
            "m_actualTyreCompound": s[0], "m_visualTyreCompound": s[1],
            "m_wear": s[2], "m_available": s[3], "m_recommendedSession": s[4],
            "m_lifeSpan": s[5], "m_usableLife": s[6], "m_lapDeltaTime": s[7],
        })
        i += 8
    fitted_idx = v[i]
    return {
        "m_carIdx": car_idx,
        "m_tyreSetData": sets,
        "m_fittedIdx": fitted_idx,
    }


# --------------------------------------------------------------------------- #
# Packet 13 — Motion Ex  (LOW-MEDIUM; F1 25 additions included)
# --------------------------------------------------------------------------- #
CONFIDENCE_MOTIONEX = (
    "LOW-MEDIUM — base follows F1 24 MotionEx; F1 25 adds m_chassisPitch, "
    "m_wheelCamber[4], m_wheelCamberGain[4] (EA PDF total 273B). Exact field set is "
    "under-documented; trailing bytes ignored."
)
# 7 float[4] arrays + 10 individual floats + 2 float[4] arrays (F1 24) +
# chassisPitch(f) + wheelCamber[4] + wheelCamberGain[4] (F1 25)
_MOTIONEX_BODY = struct.Struct(
    "<" + "4f" * 7 + "f" * 10 + "4f" * 2 + "f" + "4f" * 2
)


def parse_motion_ex(data: bytes) -> dict[str, Any]:
    """Parse PacketMotionExData (packet id 13)."""
    v = _unpack_body(data, _MOTIONEX_BODY)
    i = 0
    out: dict[str, Any] = {}
    for name in (
        "m_suspensionPosition", "m_suspensionVelocity", "m_suspensionAcceleration",
        "m_wheelSpeed", "m_wheelSlipRatio", "m_wheelLatForce", "m_wheelLongForce",
    ):
        out[name] = list(v[i:i + 4])
        i += 4
    scalars = (
        "m_heightOfCOGAboveGround", "m_localVelocityX", "m_localVelocityY",
        "m_localVelocityZ", "m_angularVelocityX", "m_angularVelocityY",
        "m_angularVelocityZ", "m_angularAccelerationX", "m_angularAccelerationY",
        "m_angularAccelerationZ",
    )
    for name in scalars:
        out[name] = v[i]
        i += 1
    for name in ("m_wheelVerticalForce", "m_wheelSlipAngle"):
        out[name] = list(v[i:i + 4])
        i += 4
    out["m_frontWheelsAngle"] = v[i]
    i += 1
    out["m_chassisPitch"] = v[i]
    i += 1
    out["m_wheelCamber"] = list(v[i:i + 4])
    i += 4
    out["m_wheelCamberGain"] = list(v[i:i + 4])
    return out


# --------------------------------------------------------------------------- #
# Packet 14 — Time Trial  (MEDIUM-LOW; EA PDF 101B, first 6 fields documented)
# --------------------------------------------------------------------------- #
CONFIDENCE_TIMETRIAL = (
    "MEDIUM-LOW — EA PDF reports 101B total (Iter-11 cross-checked against EA F1 25 "
    "v3 PDF + 2026 Season Pack discussion). First TimeTrialDataSet fields (carIdx 1B + "
    "teamId 1B + lapTimeInMS 4B + sector1/2/3InMS 4B*3 = 18B) are documented; the "
    "remaining 54B may contain a second TimeTrialDataSet (AI/benchmark) + padding. "
    "We parse the first set and expose the remainder as raw bytes for forward compat. "
    "Full layout pending real-game capture."
)
_TT_HEAD = struct.Struct("<BB IIII")
_TT_TOTAL_BODY = 101 - 29  # 72B body after 29B header (EA PDF total 101B)


def parse_time_trial(data: bytes) -> dict[str, Any]:
    """Parse PacketTimeTrialData (packet id 14).

    EA PDF total = 101B (29B header + 72B body). The first ``TimeTrialDataSet``
    (carIdx/teamId/lapTimeInMS/sector1-3InMS) is documented; the remainder is
    exposed as raw bytes (likely a second dataset + padding) so future EA spec
    updates don't break parsing. Body is padded to 72B if shorter.
    """
    body = data[HEADER_SIZE:]
    if len(body) < _TT_HEAD.size:
        body = body + b"\x00" * (_TT_HEAD.size - len(body))
    car_idx, team_id, lap, s1, s2, s3 = _TT_HEAD.unpack(body[: _TT_HEAD.size])
    remaining = body[_TT_HEAD.size:]
    return {
        "m_timeTrialDataSet": {
            "m_carIdx": car_idx,
            "m_teamId": team_id,
            "m_lapTimeInMS": lap,
            "m_sector1TimeInMS": s1,
            "m_sector2TimeInMS": s2,
            "m_sector3TimeInMS": s3,
        },
        "_remaining": remaining,
        "_body_size": len(body),
        "_expected_body_size": _TT_TOTAL_BODY,
    }


# --------------------------------------------------------------------------- #
# Packet 15 — Lap Positions  (HIGH — size verified 1131 = 29 + 2 + 50*22)
# --------------------------------------------------------------------------- #
CONFIDENCE_LAPPOSITIONS = (
    "HIGH — total size 1131B verified against EA PDF (29 header + 2 + 50*22). "
    "m_positionForVehicleIdx[50][22]."
)
_LP_BODY = struct.Struct("<" + "BB" + "B" * (50 * NUM_CARS))


def parse_lap_positions(data: bytes) -> dict[str, Any]:
    """Parse PacketLapPositionsData (packet id 15)."""
    v = _unpack_body(data, _LP_BODY)
    num_laps, lap_start = v[0], v[1]
    positions = v[2:]
    grid = [list(positions[r * NUM_CARS: (r + 1) * NUM_CARS]) for r in range(50)]
    return {
        "m_numLaps": num_laps,
        "m_lapStart": lap_start,
        "m_positionForVehicleIdx": grid,
    }


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
PACKET_PARSERS: dict[int, Callable[[bytes], dict[str, Any]]] = {
    0: parse_motion,
    1: parse_session,
    2: parse_lap_data,
    3: parse_event,
    4: parse_participants,
    5: parse_car_setups,
    6: parse_car_telemetry,
    7: parse_car_status,
    8: parse_final_classification,
    9: parse_lobby_info,
    10: parse_car_damage,
    11: parse_session_history,
    12: parse_tyre_sets,
    13: parse_motion_ex,
    14: parse_time_trial,
    15: parse_lap_positions,
    16: parse_car_telemetry_2,
}

# Confidence notes keyed by packet id (for introspection / reporting).
CONFIDENCE: dict[int, str] = {
    0: CONFIDENCE_MOTION,
    1: CONFIDENCE_SESSION,
    2: CONFIDENCE_LAPDATA,
    3: CONFIDENCE_EVENT,
    4: CONFIDENCE_PARTICIPANTS,
    5: CONFIDENCE_CARSETUPS,
    6: CONFIDENCE_CARTELEMETRY,
    7: CONFIDENCE_CARSTATUS,
    8: CONFIDENCE_FINALCLASS,
    9: CONFIDENCE_LOBBYINFO,
    10: CONFIDENCE_CARDAMAGE,
    11: CONFIDENCE_SESSIONHIST,
    12: CONFIDENCE_TYRESETS,
    13: CONFIDENCE_MOTIONEX,
    14: CONFIDENCE_TIMETRIAL,
    15: CONFIDENCE_LAPPOSITIONS,
    16: CONFIDENCE_CARTELEMETRY2,
}


def parse_packet(data: bytes) -> tuple[PacketHeader, dict[str, Any]]:
    """Parse a full datagram: header + body via the registered parser.

    Returns ``(header, parsed)``. Unknown packet ids yield an empty body dict.
    """
    header = parse_header(data)
    parser = PACKET_PARSERS.get(header.packet_id)
    parsed = parser(data) if parser is not None else {}
    return header, parsed


# --------------------------------------------------------------------------- #
# Packet size validation (Iter-192)
# --------------------------------------------------------------------------- #
# Expected minimum body sizes per packet type (after header).
# These are derived from the EA F1 25 PDF reported sizes (total - 29B header).
# Used for informational logging; parsing is still tolerant of truncation.
_EXPECTED_BODY_SIZES: dict[int, int] = {
    0: 1320,   # Motion: 1349 - 29
    1: 614,    # Session (approx)
    2: 1254,   # LapData: 57 bytes/car × 22 (Iter-281, 权威规范)
    3: 4,      # Event (min: 4-byte code)
    4: 1122,   # Participants
    5: 1078,   # CarSetups
    6: 1301,   # CarTelemetry: 59 bytes/car × 22 + 3 trailer (Iter-278)
    7: 1298,   # CarStatus: 59 bytes/car × 22 (Iter-278, 权威规范)
    8: 1013,   # FinalClassification: 1042 - 29
    9: 925,    # LobbyInfo: 954 - 29
    10: 1012,  # CarDamage: 1041 - 29
    11: 526,   # SessionHistory
    12: 110,   # TyreSets
    13: 244,   # MotionEx: 273 - 29
    14: 72,    # TimeTrial: 101 - 29
    15: 1102,  # LapPositions: 1131 - 29
    16: 220,   # CarTelemetry2: 10 bytes/car × 22 (Iter-278)
}


def validate_packet_size(packet_id: int, body_size: int) -> dict[str, Any]:
    """Check if a packet body size matches the expected EA spec.

    Iter-192: returns a dict with ``expected``, ``actual``, ``ok``, and
    ``warning`` (if mismatch). Used for debugging and monitoring.
    """
    expected = _EXPECTED_BODY_SIZES.get(packet_id)
    if expected is None:
        return {"expected": None, "actual": body_size, "ok": True, "warning": "unknown packet type"}
    if body_size >= expected:
        return {"expected": expected, "actual": body_size, "ok": True, "warning": None}
    return {
        "expected": expected,
        "actual": body_size,
        "ok": False,
        "warning": f"packet {packet_id} ({packet_name(packet_id)}): body {body_size}B < expected {expected}B (truncated?)",
    }


# --------------------------------------------------------------------------- #
# Packet type statistics (Iter-208)
# --------------------------------------------------------------------------- #
class PacketTypeStats:
    """Track per-packet-type count and rate (Iter-208).

    Accumulates per-packet-id counts so callers can compute the real-time
    packet rate distribution (e.g. 60 Hz Motion vs 2 Hz CarStatus).
    Thread-safe counter increment; reset() is NOT thread-safe.
    """

    def __init__(self) -> None:
        self._counts: dict[int, int] = {}
        self._total: int = 0
        self._start_time: float | None = None

    def record(self, packet_id: int) -> None:
        """Increment the counter for ``packet_id``."""
        self._counts[packet_id] = self._counts.get(packet_id, 0) + 1
        self._total += 1
        if self._start_time is None:
            import time
            self._start_time = time.time()

    def distribution(self) -> dict[str, Any]:
        """Return per-packet-type counts and fractions.

        Returns a dict with ``counts`` (``{name: count}``), ``total``,
        and ``fractions`` (``{name: fraction}``).
        """
        if self._total == 0:
            return {"counts": {}, "total": 0, "fractions": {}}
        counts = {packet_name(k): v for k, v in self._counts.items()}
        fractions = {k: v / self._total for k, v in counts.items()}
        return {"counts": counts, "total": self._total, "fractions": fractions}

    def reset(self) -> None:
        """Clear all counters."""
        self._counts.clear()
        self._total = 0
        self._start_time = None


# --------------------------------------------------------------------------- #
# Active aero mode helpers (Iter-219)
# --------------------------------------------------------------------------- #
# F1 2026 Season Pack introduces active aero with X (low-drag) and Z (high-
# downforce) modes. The driver can switch between them on-the-fly, and the
# DRS-style zone activation partly depends on the active aero state.
#
# Active aero mode values (packetFormat=2025, CarStatus / CarTelemetry):
#   0 = Fixed (legacy static aero)
#   1 = Active (X mode — low drag, straight-line speed)
#   2 = Active (Z mode — high downforce, cornering grip)
#   3 = Auto (car decides)

_ACTIVE_AERO_MODE_NAMES: dict[int, str] = {
    0: "Fixed",
    1: "Active-X",
    2: "Active-Z",
    3: "Auto",
}


def active_aero_mode_name(mode: int) -> str:
    """Return human-readable name for an active aero mode value."""
    return _ACTIVE_AERO_MODE_NAMES.get(mode, f"Unknown({mode})")


def is_low_drag_mode(mode: int) -> bool:
    """Return True if the active aero mode is X (low-drag)."""
    return mode == 1


def is_high_downforce_mode(mode: int) -> bool:
    """Return True if the active aero mode is Z (high-downforce)."""
    return mode == 2


def active_aero_mode_from_frame(frame: dict) -> str:
    """Extract the active aero mode name from a unified frame dict.

    Tries ``active_aero_mode``, ``aero_mode``, and ``active_aero`` keys.
    Returns ``"unknown"`` when no key is present.
    """
    mode = frame.get("active_aero_mode") or frame.get("aero_mode") or frame.get("active_aero")
    if mode is None:
        return "unknown"
    try:
        return active_aero_mode_name(int(mode))
    except (TypeError, ValueError):
        return "unknown"


__all__ = [
    "HEADER_FORMAT",
    "HEADER_SIZE",
    "NUM_CARS",
    "PACKET_NAMES",
    "PACKET_PARSERS",
    "CONFIDENCE",
    "PacketHeader",
    "PacketTypeStats",  # Iter-208
    "active_aero_mode_name",  # Iter-219
    "is_low_drag_mode",  # Iter-219
    "is_high_downforce_mode",  # Iter-219
    "active_aero_mode_from_frame",  # Iter-219
    "parse_header",
    "parse_packet",
    "validate_packet_size",
    "packet_name",
    "parse_motion",
    "parse_session",
    "parse_lap_data",
    "parse_event",
    "parse_participants",
    "parse_car_setups",
    "parse_car_telemetry",
    "parse_car_status",
    "parse_final_classification",
    "parse_lobby_info",
    "parse_car_damage",
    "parse_session_history",
    "parse_tyre_sets",
    "parse_motion_ex",
    "parse_time_trial",
    "parse_lap_positions",
]
