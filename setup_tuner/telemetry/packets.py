"""F1 2026 UDP 遥测包解析（纯 struct，零 numpy/torch）。

本模块解析 EA F1 2026（packetFormat=2026）UDP 遥测协议的 6 类核心包，
**仅解析玩家车辆数据**（通过 ``playerCarIndex`` 索引），剥离全部 ML 依赖。

协议特征：
- 小端（little-endian）、紧凑（无填充）。
- Header 固定 29 字节。
- 按车分包（LapData/CarSetups/CarTelemetry/CarStatus/CarTelemetry2）含 24 个
  车位固定数组（``NUM_CARS = 24``）；本版只解包玩家那一段，避免 60Hz 全量解包开销。
- 容错：短包抛 :class:`PacketTooShortError`；未知 packetId 跳过不崩溃。

官方规范出处（每条字段映射均在行内注释中标注）：
- **EA F1 2026 UDP Telemetry Specification**（MacManley/f1-26-udp 权威规范，
  F1 2026 Season Pack）。本模块字段偏移/类型/大小端均对照该规范。
- 旧版 ``legacy/f1opt/telemetry/packets.py`` 仅用于**字段偏移核对**，
  逻辑全部重写（对齐 C-06：拒绝复用旧 bug 代码）。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
# Source: EA F1 2026 UDP Telemetry Specification, PacketHeader
# 29 字节包头，小端无填充：
#   uint16 m_packetFormat
#   uint8  m_gameYear / m_gameMajorVersion / m_gameMinorVersion / m_packetVersion / m_packetId
#   uint64 m_sessionUID
#   float  m_sessionTime
#   uint32 m_frameIdentifier           (resets on flashback)
#   uint32 m_overallFrameIdentifier    (does NOT reset on flashback)
#   uint8  m_playerCarIndex
#   uint8  m_secondaryPlayerCarIndex   (255 if none)
HEADER_FORMAT = "<HBBBBBQfIIBB"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 29
assert HEADER_SIZE == 29, f"header size mismatch: {HEADER_SIZE}"

# Source: EA F1 2026 UDP Telemetry Specification — cs_maxNumCarsInUDPData
# F1 2026 固定数组 24 车位（标准网格 22 车，协议保留 24 槽）。
NUM_CARS = 24

# Source: EA F1 2026 UDP Telemetry Specification — Packet IDs
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
    16: "CarTelemetryData2",  # F1 2026 主动空力/超车
}


def packet_name(packet_id: int) -> str:
    """返回 packet_id 的人类可读名称（未知返回 ``Unknown(<id>)``）。"""
    return PACKET_NAMES.get(packet_id, f"Unknown({packet_id})")


# --------------------------------------------------------------------------- #
# 异常
# --------------------------------------------------------------------------- #
class PacketTooShortError(ValueError):
    """包数据不足以解析所需字段时抛出。

    由 ``parse_header`` / 各 ``parse_*`` 函数抛出；``listener`` 层捕获后跳过，
    保证丢包/截断不崩溃（对齐 FR-TEL-04）。
    """


class UnknownPacketError(ValueError):
    """包头 packetId 不在本版支持的 6 类包范围内时抛出。"""


# --------------------------------------------------------------------------- #
# PacketHeader
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PacketHeader:
    """解析后的 29 字节 F1 2026 包头。

    所有字段名与官方规范一致（``m_`` 前缀去除，转为 snake_case）。
    """

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
    """解析 29 字节 :class:`PacketHeader`。

    Raises:
        PacketTooShortError: ``len(data) < 29``。
    """
    if len(data) < HEADER_SIZE:
        raise PacketTooShortError(
            f"packet too short for header: {len(data)} bytes < {HEADER_SIZE}",
        )
    fields = _HEADER_STRUCT.unpack(data[:HEADER_SIZE])
    return PacketHeader(*fields)


# --------------------------------------------------------------------------- #
# 辅助：只解包玩家车辆那一段
# --------------------------------------------------------------------------- #
def _slice_player_car(
    data: bytes, per_car_struct: struct.Struct, player_car_index: int,
) -> tuple:
    """从按车分组的包体中只切出 ``player_car_index`` 对应的那一辆车并解包。

    跳过前 ``player_car_index`` 辆车，只解包目标车，避免 60Hz 全量解包 24 车。

    Raises:
        PacketTooShortError: 包体不足以覆盖到玩家车辆末尾。
    """
    if not 0 <= player_car_index < NUM_CARS:
        # player_car_index 越界（如 255 表示无第二玩家）——视为无效包跳过。
        raise PacketTooShortError(
            f"player_car_index {player_car_index} out of range [0,{NUM_CARS})",
        )
    per_size = per_car_struct.size
    offset = HEADER_SIZE + player_car_index * per_size
    end = offset + per_size
    if len(data) < end:
        raise PacketTooShortError(
            f"packet too short for player car {player_car_index}: "
            f"{len(data)} bytes < {end}",
        )
    return per_car_struct.unpack(data[offset:end])


# --------------------------------------------------------------------------- #
# Packet 1 — Session（全局会话数据，非按车分组）
# --------------------------------------------------------------------------- #
# Source: EA F1 2026 UDP Telemetry Specification, Packet 1 (Session)
# 包体开头 16 字段（小端）：
#   uint8  m_weather
#   int8   m_trackTemperature
#   int8   m_airTemperature
#   uint8  m_totalLaps
#   uint16 m_trackLength
#   int8   m_sessionType
#   uint8  m_trackId
#   uint8  m_formula
#   uint16 m_sessionTimeLeft
#   uint16 m_sessionDuration
#   uint8  m_pitSpeedLimit
#   uint8  m_gamePaused
#   uint8  m_isSpectating
#   uint8  m_spectatorCarIndex
#   uint8  m_sliProNativeSupport
#   uint8  m_numMarshalZones
# 紧随 21 个 marshal zones（各 float m_zoneStart + uint8 m_zoneFlag = 42 字节），
# 再接 uint8 m_safetyCarStatus / uint8 m_networkGame / uint8 m_numWeatherForecastSamples，
# 之后是 64 个天气预报样本（各 8 字节）。
# 注意：含 "fb" * 21 表达式，必须用 + 显式拼接（隐式拼接仅限纯字面量）
_SESSION_PREFIX_FMT = (
    "<"                  # 小端
    + "BbbBHBbBHHBBBBBB"  # 16 字段
    + "fb" * 21           # 21 marshal zones (各 float + uint8)
    + "BB"                # safetyCarStatus, networkGame
    + "B"                 # numWeatherForecastSamples
)
_SESSION_PREFIX_STRUCT = struct.Struct(_SESSION_PREFIX_FMT)

# Source: EA F1 2026 UDP Telemetry Specification, Packet 1, WeatherForecastSample
# 每个 8 字节：sessionType(B) timeOffset(B) weather(B) trackTemp(b)
#   trackTempChange(b) airTemp(b) airTempChange(b) rainPercentage(B)
_WFS_STRUCT = struct.Struct("<BBBbbbbB")


def parse_session(data: bytes, player_car_index: int = 0) -> dict[str, Any]:
    """解析 Packet 1 (Session) — 赛道/天气/气温/天气预报样本。

    本包为全局会话数据，``player_car_index`` 不使用（保留参数以统一签名）。

    Raises:
        PacketTooShortError: 包体不足以解析前缀或天气样本。
    """
    body_start = HEADER_SIZE
    prefix_end = body_start + _SESSION_PREFIX_STRUCT.size
    if len(data) < prefix_end:
        raise PacketTooShortError(
            f"Session prefix too short: {len(data)} < {prefix_end}",
        )
    v = _SESSION_PREFIX_STRUCT.unpack(data[body_start:prefix_end])
    # Source: EA F1 2026 UDP Telemetry Specification, Packet 1, Fields 1-16
    (weather, track_temp, air_temp, total_laps, track_len, session_type, track_id,
     formula, session_time_left, session_duration, pit_speed_limit, game_paused,
     is_spectating, spectator_car_idx, sli_pro, _num_marshal,
     *_marshal_flat, sc_status, network_game, num_wfs) = v

    # 解析天气预测样本（最多 64 个，按 num_wfs 实际数量）
    wfs: list[dict[str, Any]] = []
    wfs_start = prefix_end
    for i in range(num_wfs):
        s_off = wfs_start + i * _WFS_STRUCT.size
        s_end = s_off + _WFS_STRUCT.size
        if len(data) < s_end:
            # 样本数声明超过实际字节——已解到的保留，剩余跳过（容错）。
            break
        (st, to, w, tt, ttc, at, atc, rain) = _WFS_STRUCT.unpack(
            data[s_off:s_end],
        )
        wfs.append({
            # Source: EA F1 2026 UDP Telemetry Specification, Packet 1, WeatherForecastSample
            "m_sessionType": st,
            "m_timeOffset": to,
            "m_weather": w,
            "m_trackTemperature": tt,
            "m_trackTemperatureChange": ttc,
            "m_airTemperature": at,
            "m_airTemperatureChange": atc,
            "m_rainPercentage": rain,
        })

    return {
        # Source: EA F1 2026 UDP Telemetry Specification, Packet 1, Session Fields
        "m_trackId": track_id,                # uint8 — 赛道 ID（用于自动识别）
        "m_weather": weather,                 # uint8 — 当前天气
        "m_trackTemperature": track_temp,     # int8  — 赛道温度 (℃)
        "m_airTemperature": air_temp,         # int8  — 气温 (℃)
        "m_totalLaps": total_laps,            # uint8 — 总圈数
        "m_trackLength": track_len,           # uint16 — 赛道长度 (m)
        "m_sessionType": session_type,        # int8  — 会话类型
        "m_formula": formula,                 # uint8 — 方程式类别
        "m_sessionTimeLeft": session_time_left,
        "m_sessionDuration": session_duration,
        "m_pitSpeedLimit": pit_speed_limit,
        "m_gamePaused": game_paused,
        "m_isSpectating": is_spectating,
        "m_spectatorCarIndex": spectator_car_idx,
        "m_sliProNativeSupport": sli_pro,
        "m_safetyCarStatus": sc_status,
        "m_networkGame": network_game,
        "m_numWeatherForecastSamples": num_wfs,
        "m_weatherForecastSamples": wfs,
    }


# --------------------------------------------------------------------------- #
# Packet 2 — LapData（按车分组，只解玩家车）
# --------------------------------------------------------------------------- #
# Source: EA F1 2026 UDP Telemetry Specification, Packet 2 (LapData)
# 每车 58 字节：
#   uint32 m_lastLapTimeInMS
#   uint32 m_currentLapTimeInMS
#   uint16 m_sector1TimeInMSPart + uint8 m_sector1TimeMinutesPart
#   uint16 m_sector2TimeInMSPart + uint8 m_sector2TimeMinutesPart
#   uint16 m_deltaToCarInFrontInMSPart + uint8 m_deltaToCarInFrontInMinutesPart
#   uint16 m_deltaToRaceLeaderInMSPart + uint8 m_deltaToRaceLeaderInMinutesPart
#   float  m_lapDistance / m_totalDistance / m_safetyCarDelta
#   uint8 ×15: carPosition/currentLapNum/pitStatus/numPitStops/sector/
#              currentLapInvalid/penalties/totalWarnings/cornerCuttingWarnings/
#              numUnservedDriveThroughPens/numUnservedStopGoPens/gridPosition/
#              driverStatus/resultStatus/pitLaneTimerActive
#   uint16 m_pitLaneTimeInLaneInMS + uint16 m_pitStopTimerInMS
#   uint8  m_pitStopShouldServePen
#   float  m_speedTrapFastestSpeed
#   uint8  m_speedTrapFastestLap
_LAP_PER_FMT = (
    "<"
    "II"          # m_lastLapTimeInMS, m_currentLapTimeInMS
    "HBHBHBHB"    # sector1/deltaFront/deltaLeader (MSPart+MinutesPart)
    "fff"         # lapDistance, totalDistance, safetyCarDelta
    "BBBBBBBBBBBBBBB"  # 15 × uint8
    "HHB"         # pitLaneTimeInLaneInMS, pitStopTimerInMS, pitStopShouldServePen
    "fB"          # speedTrapFastestSpeed, speedTrapFastestLap
)
_LAP_PER_STRUCT = struct.Struct(_LAP_PER_FMT)


def parse_lap_data(data: bytes, player_car_index: int) -> dict[str, Any]:
    """解析 Packet 2 (LapData) — 玩家车圈速/扇区/距离/圈数/无效标志。

    Raises:
        PacketTooShortError: 包体不足以覆盖玩家车辆字段。
    """
    c = _slice_player_car(data, _LAP_PER_STRUCT, player_car_index)
    # 扇区时间组合：MSPart + MinutesPart * 60000 → 完整毫秒值
    # Source: EA F1 2026 UDP Telemetry Specification, Packet 2, Sector Time Encoding
    sector1_ms = int(c[3]) * 60000 + int(c[2])
    sector2_ms = int(c[5]) * 60000 + int(c[4])
    delta_front_ms = int(c[7]) * 60000 + int(c[6])
    delta_leader_ms = int(c[9]) * 60000 + int(c[8])

    return {
        # Source: EA F1 2026 UDP Telemetry Specification, Packet 2, LapData Fields
        "m_lastLapTimeInMS": c[0],           # uint32 — 上一圈用时 (ms)
        "m_currentLapTimeInMS": c[1],        # uint32 — 当前圈用时 (ms)
        "m_sector1TimeInMS": sector1_ms,     # 组合 — 扇区 1 用时 (ms)
        "m_sector2TimeInMS": sector2_ms,     # 组合 — 扇区 2 用时 (ms)
        "m_deltaToCarInFrontInMS": delta_front_ms,
        "m_deltaToRaceLeaderInMS": delta_leader_ms,
        "m_lapDistance": c[10],              # float — 圈内距离 (m)
        "m_totalDistance": c[11],            # float — 总距离 (m)
        "m_safetyCarDelta": c[12],           # float — 安全车差 (s)
        "m_carPosition": c[13],             # uint8 — 名次
        "m_currentLapNum": c[14],           # uint8 — 当前圈号
        "m_pitStatus": c[15],               # uint8 — 进站状态
        "m_numPitStops": c[16],             # uint8 — 进站次数
        "m_sector": c[17],                  # uint8 — 当前扇区 (0/1/2)
        "m_currentLapInvalid": c[18],       # uint8 — 当前圈无效标志 (1=无效)
        "m_penalties": c[19],               # uint8 — 罚秒
        "m_totalWarnings": c[20],
        "m_cornerCuttingWarnings": c[21],
        "m_numUnservedDriveThroughPens": c[22],
        "m_numUnservedStopGoPens": c[23],
        "m_gridPosition": c[24],
        "m_driverStatus": c[25],
        "m_resultStatus": c[26],
        "m_pitLaneTimerActive": c[27],
        "m_pitLaneTimeInLaneInMS": c[28],   # uint16
        "m_pitStopTimerInMS": c[29],         # uint16
        "m_pitStopShouldServePen": c[30],
        "m_speedTrapFastestSpeed": c[31],   # float
        "m_speedTrapFastestLap": c[32],
    }


# --------------------------------------------------------------------------- #
# Packet 5 — CarSetups（按车分组，只解玩家车）
# --------------------------------------------------------------------------- #
# Source: EA F1 2026 UDP Telemetry Specification, Packet 5 (CarSetups)
# 每车 50 字节：
#   uint8 m_frontWing / m_rearWing / m_onThrottleDiff / m_offThrottleDiff  (game clicks)
#   float m_frontCamber / m_rearCamber / m_frontToe / m_rearToe
#   uint8 m_frontSuspension / m_rearSuspension / m_frontAntiRollBar /
#         m_rearAntiRollBar / m_frontSuspensionHeight / m_rearSuspensionHeight /
#         m_brakePressure / m_brakeBias / m_engineBraking
#   float m_rearLeftTyrePressure / m_rearRightTyrePressure /
#         m_frontLeftTyrePressure / m_frontRightTyrePressure
#   uint8 m_ballast
#   float m_fuelLoad
_SETUP_PER_FMT = (
    "<"
    "BBBB"        # frontWing, rearWing, onThrottleDiff, offThrottleDiff
    "ffff"        # frontCamber, rearCamber, frontToe, rearToe
    "BBBBBBBBB"   # 9 × uint8 (悬挂/防倾杆/行驶高度/刹车压力/配比/发动机制动)
    "ffff"        # 4 × float 胎压
    "B"           # ballast
    "f"           # fuelLoad
)
_SETUP_PER_STRUCT = struct.Struct(_SETUP_PER_FMT)


def parse_car_setups(data: bytes, player_car_index: int) -> dict[str, Any]:
    """解析 Packet 5 (CarSetups) — 玩家车调教参数（前翼/后翼/差速/外倾/束角/悬挂/胎压等）。

    Raises:
        PacketTooShortError: 包体不足以覆盖玩家车辆字段。
    """
    c = _slice_player_car(data, _SETUP_PER_STRUCT, player_car_index)
    return {
        # Source: EA F1 2026 UDP Telemetry Specification, Packet 5, CarSetup Fields
        "m_frontWing": c[0],                 # uint8 — 前翼 (clicks)
        "m_rearWing": c[1],                  # uint8 — 后翼 (clicks)
        "m_onThrottleDiff": c[2],            # uint8 — 油门差速器 (clicks)
        "m_offThrottleDiff": c[3],           # uint8 — 收油差速器 (clicks)
        "m_frontCamber": c[4],               # float — 前外倾角
        "m_rearCamber": c[5],                # float — 后外倾角
        "m_frontToe": c[6],                  # float — 前束角
        "m_rearToe": c[7],                   # float — 后束角
        "m_frontSuspension": c[8],           # uint8 — 前悬挂
        "m_rearSuspension": c[9],            # uint8 — 后悬挂
        "m_frontAntiRollBar": c[10],         # uint8 — 前防倾杆
        "m_rearAntiRollBar": c[11],          # uint8 — 后防倾杆
        "m_frontSuspensionHeight": c[12],    # uint8 — 前行驶高度
        "m_rearSuspensionHeight": c[13],     # uint8 — 后行驶高度
        "m_brakePressure": c[14],            # uint8 — 刹车压力
        "m_brakeBias": c[15],                # uint8 — 刹车配比
        "m_engineBraking": c[16],            # uint8 — 发动机制动
        "m_rearLeftTyrePressure": c[17],     # float — 后左胎压 (PSI)
        "m_rearRightTyrePressure": c[18],    # float — 后右胎压
        "m_frontLeftTyrePressure": c[19],    # float — 前左胎压
        "m_frontRightTyrePressure": c[20],   # float — 前右胎压
        "m_ballast": c[21],                  # uint8 — 配重
        "m_fuelLoad": c[22],                 # float — 燃油量 (kg)
    }


# --------------------------------------------------------------------------- #
# Packet 6 — CarTelemetry（按车分组，只解玩家车）
# --------------------------------------------------------------------------- #
# Source: EA F1 2026 UDP Telemetry Specification, Packet 6 (CarTelemetry)
# 每车 59 字节：
#   uint16 m_speed
#   float  m_throttle / m_steer / m_brake
#   uint8  m_clutch
#   int8   m_gear
#   uint16 m_engineRPM
#   uint8  m_drs
#   uint8  m_revLightsPercent
#   uint16 m_revLightsBitValue
#   uint16 m_brakesTemperature[4]
#   uint8  m_tyresSurfaceTemperature[4]
#   uint8  m_tyresInnerTemperature[4]
#   uint8  m_engineTemperature
#   float  m_tyresPressure[4]
#   uint8  m_surfaceType[4]
_TELEM_PER_FMT = (
    "<"
    "H"           # speed
    "fff"         # throttle, steer, brake
    "B"           # clutch
    "b"           # gear (int8)
    "H"           # engineRPM
    "BBH"         # drs, revLightsPercent, revLightsBitValue
    "4H"          # brakesTemperature[4]
    "4B"          # tyresSurfaceTemperature[4]
    "4B"          # tyresInnerTemperature[4]
    "B"           # engineTemperature
    "4f"          # tyresPressure[4]
    "4B"          # surfaceType[4]
)
_TELEM_PER_STRUCT = struct.Struct(_TELEM_PER_FMT)


def parse_car_telemetry(data: bytes, player_car_index: int) -> dict[str, Any]:
    """解析 Packet 6 (CarTelemetry) — 玩家车实时遥测（速度/油门/刹车/档位/转速/DRS/胎温胎压）。

    Raises:
        PacketTooShortError: 包体不足以覆盖玩家车辆字段。
    """
    c = _slice_player_car(data, _TELEM_PER_STRUCT, player_car_index)
    return {
        # Source: EA F1 2026 UDP Telemetry Specification, Packet 6, CarTelemetry Fields
        "m_speed": c[0],                     # uint16 — 速度 (km/h)
        "m_throttle": c[1],                  # float  — 油门 (0~1)
        "m_steer": c[2],                     # float  — 方向 (-1~1)
        "m_brake": c[3],                     # float  — 刹车 (0~1)
        "m_clutch": c[4],                    # uint8  — 离合
        "m_gear": c[5],                      # int8   — 档位 (-1=倒挡)
        "m_engineRPM": c[6],                 # uint16 — 转速
        "m_drs": c[7],                       # uint8  — DRS 状态
        "m_revLightsPercent": c[8],
        "m_revLightsBitValue": c[9],
        "m_brakesTemperature": list(c[10:14]),       # uint16[4] — 制动温度
        "m_tyresSurfaceTemperature": list(c[14:18]), # uint8[4]  — 胎面温度
        "m_tyresInnerTemperature": list(c[18:22]),   # uint8[4]  — 胎内温度
        "m_engineTemperature": c[22],                 # uint8     — 引擎温度
        "m_tyresPressure": list(c[23:27]),            # float[4]  — 胎压 (PSI)
        "m_surfaceType": list(c[27:31]),              # uint8[4]  — 路面类型
    }


# --------------------------------------------------------------------------- #
# Packet 7 — CarStatus（按车分组，只解玩家车）
# --------------------------------------------------------------------------- #
# Source: EA F1 2026 UDP Telemetry Specification, Packet 7 (CarStatus)
# 每车 59 字节：
#   uint8 m_tractionControl / m_antiLockBrakes / m_fuelMix / m_frontBrakeBias /
#         m_pitLimiterStatus
#   float m_fuelInTank / m_fuelCapacity / m_fuelRemainingLaps
#   uint16 m_maxRPM / m_idleRPM
#   uint8  m_maxGears / m_drsAllowed
#   uint16 m_drsActivationDistance
#   uint8  m_actualTyreCompound / m_visualTyreCompound / m_tyresAgeLaps
#   int8   m_vehicleFiaFlags
#   float  m_enginePowerICE / m_enginePowerMGUK / m_ersStoreEnergy
#   uint8  m_ersDeployMode
#   float  m_ersHarvestedThisLapMGUK / m_ersHarvestedThisLapMGUH /
#          m_ersHarvestLimitPerLap / m_ersDeployedThisLap
#   uint8  m_networkPaused
_STATUS_PER_FMT = (
    "<"
    "BBBBB"       # tractionControl, antiLockBrakes, fuelMix, frontBrakeBias, pitLimiterStatus
    "fff"         # fuelInTank, fuelCapacity, fuelRemainingLaps
    "HH"          # maxRPM, idleRPM
    "BB"          # maxGears, drsAllowed
    "H"           # drsActivationDistance
    "BB"          # actualTyreCompound, visualTyreCompound
    "B"           # tyresAgeLaps
    "b"           # vehicleFiaFlags (int8)
    "fff"         # enginePowerICE, enginePowerMGUK, ersStoreEnergy
    "B"           # ersDeployMode
    "ffff"        # ersHarvestedThisLapMGUK, ...MGUH, ersHarvestLimitPerLap, ersDeployedThisLap
    "B"           # networkPaused
)
_STATUS_PER_STRUCT = struct.Struct(_STATUS_PER_FMT)


def parse_car_status(data: bytes, player_car_index: int) -> dict[str, Any]:
    """解析 Packet 7 (CarStatus) — 玩家车轮胎配方/胎龄/燃油/ERS 状态。

    Raises:
        PacketTooShortError: 包体不足以覆盖玩家车辆字段。
    """
    c = _slice_player_car(data, _STATUS_PER_STRUCT, player_car_index)
    return {
        # Source: EA F1 2026 UDP Telemetry Specification, Packet 7, CarStatus Fields
        "m_tractionControl": c[0],
        "m_antiLockBrakes": c[1],
        "m_fuelMix": c[2],                   # uint8 — 燃油混合模式
        "m_frontBrakeBias": c[3],
        "m_pitLimiterStatus": c[4],
        "m_fuelInTank": c[5],                # float — 油箱燃油 (kg)
        "m_fuelCapacity": c[6],              # float — 油箱容量
        "m_fuelRemainingLaps": c[7],         # float — 剩余燃油圈数
        "m_maxRPM": c[8],
        "m_idleRPM": c[9],
        "m_maxGears": c[10],
        "m_drsAllowed": c[11],
        "m_drsActivationDistance": c[12],
        "m_actualTyreCompound": c[13],       # uint8 — 实际轮胎配方
        "m_visualTyreCompound": c[14],       # uint8 — 视觉轮胎配方
        "m_tyresAgeLaps": c[15],             # uint8 — 胎龄 (圈)
        "m_vehicleFiaFlags": c[16],          # int8  — FIA 旗语
        "m_enginePowerICE": c[17],
        "m_enginePowerMGUK": c[18],
        "m_ersStoreEnergy": c[19],           # float — ERS 储能
        "m_ersDeployMode": c[20],            # uint8 — ERS 部署模式
        "m_ersHarvestedThisLapMGUK": c[21],
        "m_ersHarvestedThisLapMGUH": c[22],
        "m_ersHarvestLimitPerLap": c[23],
        "m_ersDeployedThisLap": c[24],
        "m_networkPaused": c[25],
    }


# --------------------------------------------------------------------------- #
# Packet 16 — CarTelemetryData2（按车分组，只解玩家车）
# --------------------------------------------------------------------------- #
# Source: EA F1 2026 UDP Telemetry Specification, Packet 16 (CarTelemetryData2)
# 每车 10 字节：
#   uint8  m_activeAeroMode (0=Corner/Z, 1=Straight/X)
#   uint8  m_activeAeroAvailable
#   uint16 m_activeAeroActivationDistance
#   uint8  m_overtakeAvailable / m_overtakeActive
#   uint16 m_overtakeActivationDistance
#   uint8  m_2026Regulations / m_drivingWrongWay
_CT2_PER_FMT = "<BBHBBHBB"
_CT2_PER_STRUCT = struct.Struct(_CT2_PER_FMT)


def parse_car_telemetry_2(data: bytes, player_car_index: int) -> dict[str, Any]:
    """解析 Packet 16 (CarTelemetryData2) — 玩家车主动空力模式/超车状态。

    ``m_activeAeroMode``: 0 = Corner/Z（弯道 Z 轴），1 = Straight/X（直道 X 轴）。

    Raises:
        PacketTooShortError: 包体不足以覆盖玩家车辆字段。
    """
    c = _slice_player_car(data, _CT2_PER_STRUCT, player_car_index)
    return {
        # Source: EA F1 2026 UDP Telemetry Specification, Packet 16, CarTelemetryData2 Fields
        "m_activeAeroMode": c[0],                  # uint8 — 主动空力模式 (0=Z/弯, 1=X/直)
        "m_activeAeroAvailable": c[1],
        "m_activeAeroActivationDistance": c[2],
        "m_overtakeAvailable": c[3],
        "m_overtakeActive": c[4],
        "m_overtakeActivationDistance": c[5],
        "m_2026Regulations": c[6],
        "m_drivingWrongWay": c[7],
    }


# --------------------------------------------------------------------------- #
# 主入口：按 packetId 分发
# --------------------------------------------------------------------------- #
# 本版支持的 6 类包及其解析函数
_PARSERS: dict[int, Any] = {
    1: parse_session,
    2: parse_lap_data,
    5: parse_car_setups,
    6: parse_car_telemetry,
    7: parse_car_status,
    16: parse_car_telemetry_2,
}

# 支持的 packetId 集合（供外部查询）
SUPPORTED_PACKET_IDS: frozenset[int] = frozenset(_PARSERS.keys())


def parse_packet(data: bytes) -> dict[str, Any] | None:
    """解析一个完整 UDP 包，返回 ``{packet_id, name, header, **body}`` 或 ``None``。

    - 短包（< 29 字节或包体不足）抛 :class:`PacketTooShortError`。
    - 未知 packetId（不在 6 类支持范围内）返回 ``None``，不抛错（跳过不崩溃）。
    - ``player_car_index`` 从包头提取，自动传入各解析函数。

    返回结构::

        {
            "packet_id": int,
            "name": str,
            "header": PacketHeader,
            ...各包特有字段...
        }
    """
    header = parse_header(data)
    parser = _PARSERS.get(header.packet_id)
    if parser is None:
        # 未知/不支持的 packetId —— 跳过不崩溃
        return None
    body = parser(data, header.player_car_index)
    result: dict[str, Any] = {
        "packet_id": header.packet_id,
        "name": header.name,
        "header": header,
    }
    result.update(body)
    return result