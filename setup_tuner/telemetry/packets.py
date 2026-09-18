"""F1 26 UDP 遥测包解析（纯 struct，零 numpy/torch）。

本模块解析 EA F1 26（packetFormat=2026，即 F1 25 + 2026 Season Pack）
UDP 遥测协议的 **14 类包**（除 LobbyInfo(9) 与 TimeTrial(14) 两个与调教无关者），
**仅解析玩家车辆数据**（通过 ``playerCarIndex`` 索引，或如 MotionEx 般本就
只含玩家车），剥离全部 ML 依赖。

task-82 扩展：此前只解 6 类（1/2/5/6/7/13），其余 9 类**只存原始字节无法使用**。
本次补齐 0 Motion / 3 Event / 4 Participants / 8 FinalClassification /
10 CarDamage / 11 SessionHistory / 12 TyreSets / 15 LapPositions /
16 CarTelemetry2，全部结构对照 EA 官方《2026 Season Pack Telemetry Output
Structures》逐个核对（含包体大小实测一致性校验）。

协议特征：
- 小端（little-endian）、紧凑（无填充）。
- Header 固定 29 字节。
- 按车分包（LapData/CarSetups/CarTelemetry/CarStatus/Motion/CarDamage/
  CarTelemetry2）含 24 个车位固定数组（``NUM_CARS = 24``）；只解包玩家那一段，
  避免 60Hz 全量解包开销。名单类（Participants/FinalClassification）低频，
  故解全部车位。
- MotionEx（13）/SessionHistory（11）/TyreSets（12）非按车分组，包体直接为
  玩家数据。
- 容错：短包抛 :class:`PacketTooShortError`；未知 packetId 跳过不崩溃。

官方规范出处（每条字段映射均在行内注释中标注）：
- **EA《2026 Season Pack Telemetry Output Structures》**（F1 26 官方结构定义，
  ``.ref/f1_2026_structures.txt`` 本地参考，版权属 EA，不入库）。
- 包体大小逐项与真实录制实测值一致（Motion 1325 / Session 926 / Lap 1399 /
  Event 45 / Participants 1470 / CarSetups 1233 / CarTelemetry 1448 /
  CarStatus 1445 / FinalClassification 1134 / CarDamage 1133 /
  SessionHistory 1460 / TyreSets 231 / MotionEx 273 / LapPositions 1231 /
  CarTelemetry2 269）。
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
# Source: EA F1 25 UDP Telemetry Specification, PacketHeader
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

# Source: EA F1 25 UDP Telemetry Specification — cs_maxNumCarsInUDPData
# F1 25 固定数组 24 车位（标准网格 22 车，协议保留 24 槽）。
NUM_CARS = 24

# Source: EA F1 25 UDP Telemetry Specification — Packet IDs
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
    16: "CarTelemetry2",  # 2026 赛季包新增（主动空力 + 超车模式）

}


def packet_name(packet_id: int) -> str:
    """返回 packet_id 的人类可读名称（未知返回 ``Unknown(<id>)``）。"""
    return PACKET_NAMES.get(packet_id, f"Unknown({packet_id})")


def to_sector_1based(raw: Any) -> int | None:
    """把 UDP 的 0 基 ``m_sector`` 归一化为 1 基（0/1/2 → 1/2/3）。

    规范中 ``m_sector`` 取值 0=S1、1=S2、2=S3；而本项目的前端（``S1/S2/S3`` 展示与
    CSS 类名）与规则引擎（``engine._apply_corner_rules`` 判断 ``sector == 3``）
    都按 1 基理解。此前两处各自直读原始值，导致：
        - 前端显示 "S0/S1/S2"、``sector-0`` 样式类不存在；
        - 规则 5 判断 ``sector == 3`` 永不成立（真实值域上限是 2）。
    统一在此转换，避免每个消费方各自 +1 而再次错位。
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return max(1, min(3, int(raw) + 1))
    return None


# --------------------------------------------------------------------------- #
# 异常
# --------------------------------------------------------------------------- #
class PacketTooShortError(ValueError):
    """包数据不足以解析所需字段时抛出。

    由 ``parse_header`` / 各 ``parse_*`` 函数抛出；``listener`` 层捕获后跳过，
    保证丢包/截断不崩溃（对齐 FR-TEL-04）。
    """


class UnknownPacketError(ValueError):
    """包头 packetId 不在本版支持的 7 类包范围内时抛出。"""


# --------------------------------------------------------------------------- #
# PacketHeader
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PacketHeader:
    """解析后的 29 字节 F1 25 包头。

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
# ── m_weather 官方枚举（Packet 1 / WeatherForecastSample 共用）─────────────
# Source: EA F1 25 UDP Telemetry Specification, Packet 1 (Session)
#   uint8 m_weather — 0 = clear, 1 = light cloud, 2 = overcast,
#                     3 = light rain, 4 = heavy rain, 5 = storm
# **只有 3/4/5 是湿地**（需中性胎/雨胎）；1 = 轻云、2 = 阴天仍属干地。
# 此处曾把 weather >= 1 判为湿地，导致轻云干地被套上"湿地保守系数"。
WEATHER_CLEAR = 0
WEATHER_LIGHT_CLOUD = 1
WEATHER_OVERCAST = 2
WEATHER_LIGHT_RAIN = 3
WEATHER_HEAVY_RAIN = 4
WEATHER_STORM = 5
#: 判为湿地的最小 m_weather 代码（官方：小雨及以上）。
WET_WEATHER_MIN = WEATHER_LIGHT_RAIN
#: m_weather 代码 → 中文名（未知代码回落为 ``未知(n)``）。
WEATHER_NAMES: dict[int, str] = {
    WEATHER_CLEAR: "晴",
    WEATHER_LIGHT_CLOUD: "轻云",
    WEATHER_OVERCAST: "阴",
    WEATHER_LIGHT_RAIN: "小雨",
    WEATHER_HEAVY_RAIN: "大雨",
    WEATHER_STORM: "暴雨",
}


def weather_label(code: Any) -> str:
    """m_weather 代码 → 中文名（非法值返回 ``未知``）。"""
    if isinstance(code, bool) or not isinstance(code, int):
        return "未知"
    return WEATHER_NAMES.get(code, f"未知({code})")


def is_wet_weather_code(code: Any) -> bool:
    """m_weather 代码是否属于湿地（官方规范：**≥3** = 降雨/暴雨）。

    1（轻云）与 2（阴天）**不是**湿地——这是被实际数据验证过的边界：
    车手在 Hungaroring ``m_weather=1`` 时用的是 C5/C4 干胎。
    """
    return (
        isinstance(code, int)
        and not isinstance(code, bool)
        and code >= WET_WEATHER_MIN
    )


# Source: EA F1 25 UDP Telemetry Specification, Packet 1 (Session)
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

# Source: EA F1 25 UDP Telemetry Specification, Packet 1, WeatherForecastSample
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
    # Source: EA F1 25 UDP Telemetry Specification, Packet 1, Fields 1-16
    (weather, track_temp, air_temp, total_laps, track_len, session_type, track_id,
     formula, session_time_left, session_duration, pit_speed_limit, game_paused,
     is_spectating, spectator_car_idx, sli_pro, _num_marshal,
     *_marshal_flat, sc_status, network_game, num_wfs) = v

    wfs = _parse_weather_forecast_samples(data, prefix_end, num_wfs)
    return _build_session_dict(
        weather, track_temp, air_temp, total_laps, track_len, session_type, track_id,
        formula, session_time_left, session_duration, pit_speed_limit, game_paused,
        is_spectating, spectator_car_idx, sli_pro, sc_status, network_game, num_wfs, wfs,
    )


def _parse_weather_forecast_samples(
    data: bytes, wfs_start: int, num_wfs: int,
) -> list[dict[str, Any]]:
    """解析天气预测样本（最多 64 个，按 num_wfs 实际数量，容错截断）。"""
    wfs: list[dict[str, Any]] = []
    for i in range(num_wfs):
        s_off = wfs_start + i * _WFS_STRUCT.size
        s_end = s_off + _WFS_STRUCT.size
        if len(data) < s_end:
            # 样本数声明超过实际字节——已解到的保留，剩余跳过（容错）。
            break
        (st, to, w, tt, ttc, at, atc, rain) = _WFS_STRUCT.unpack(data[s_off:s_end])
        wfs.append({
            # Source: EA F1 25 UDP Telemetry Specification, Packet 1, WeatherForecastSample
            "m_sessionType": st,
            "m_timeOffset": to,
            "m_weather": w,
            "m_trackTemperature": tt,
            "m_trackTemperatureChange": ttc,
            "m_airTemperature": at,
            "m_airTemperatureChange": atc,
            "m_rainPercentage": rain,
        })
    return wfs


def _build_session_dict(
    weather: int, track_temp: int, air_temp: int, total_laps: int,
    track_len: int, session_type: int, track_id: int, formula: int,
    session_time_left: int, session_duration: int, pit_speed_limit: int,
    game_paused: int, is_spectating: int, spectator_car_idx: int,
    sli_pro: int, sc_status: int, network_game: int,
    num_wfs: int, wfs: list[dict[str, Any]],
) -> dict[str, Any]:
    """组装 Packet 1 Session 返回字典。"""
    return {
        # Source: EA F1 25 UDP Telemetry Specification, Packet 1, Session Fields
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
# Source: EA F1 25 UDP Telemetry Specification, Packet 2 (LapData)
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
    # Source: EA F1 25 UDP Telemetry Specification, Packet 2, Sector Time Encoding
    sector1_ms = int(c[3]) * 60000 + int(c[2])
    sector2_ms = int(c[5]) * 60000 + int(c[4])
    delta_front_ms = int(c[7]) * 60000 + int(c[6])
    delta_leader_ms = int(c[9]) * 60000 + int(c[8])

    return {
        # Source: EA F1 25 UDP Telemetry Specification, Packet 2, LapData Fields
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
# Source: EA F1 25 UDP Telemetry Specification, Packet 5 (CarSetups)
# 每车 50 字节：
#   uint8 m_frontWing / m_rearWing / m_onThrottleDiff / m_offThrottleDiff  (game clicks)
#   float m_frontCamber / m_rearCamber / m_frontToe / m_rearToe
#   uint8 m_frontSuspension / m_rearSuspension / m_frontAntiRollBar /
#         m_rearAntiRollBar / m_frontSuspensionHeight / m_rearSuspensionHeight /
#         m_brakePressure / m_brakeBias
#   uint8 m_engineBraking                                          (F1 24/25 新增)
#   float m_rearLeftTyrePressure / m_rearRightTyrePressure /
#         m_frontLeftTyrePressure / m_frontRightTyrePressure
#   uint8 m_ballast
#   float m_fuelLoad
_SETUP_PER_FMT = (
    "<"
    "BBBB"        # frontWing, rearWing, onThrottleDiff, offThrottleDiff
    "ffff"        # frontCamber, rearCamber, frontToe, rearToe
    "BBBBBBBB"    # 8 × uint8 (悬挂/防倾杆/行驶高度/刹车压力/刹车偏置)
    "B"           # engineBraking (F1 24/25 新增字段)
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
        # Source: EA F1 25 UDP Telemetry Specification, Packet 5, CarSetup Fields
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
        "m_engineBraking": c[16],            # uint8 — 引擎制动 (F1 24/25 新增)
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
# Source: EA F1 25 UDP Telemetry Specification, Packet 6 (CarTelemetry)
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
        # Source: EA F1 25 UDP Telemetry Specification, Packet 6, CarTelemetry Fields
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
# Source: EA F1 25 UDP Telemetry Specification, Packet 7 (CarStatus)
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
        # Source: EA F1 25 UDP Telemetry Specification, Packet 7, CarStatus Fields
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
# Packet 13 — MotionEx（玩家车专用，非按车分组）
# --------------------------------------------------------------------------- #
# Source: EA F1 25 UDP Telemetry Specification, Packet 13 (MotionEx)
# 包体 244 字节 = 61 × float32，小端无填充。字段顺序（官方规范原文）：
#   8 × float[4] 数组（车轮顺序统一 RL, RR, FL, FR）：
#     m_suspensionPosition / m_suspensionVelocity / m_suspensionAcceleration /
#     m_wheelSpeed / m_wheelSlipRatio / m_wheelSlipAngle /
#     m_wheelLatForce / m_wheelLongForce
#   11 × 标量：
#     m_heightOfCOGAboveGround / m_localVelocityX/Y/Z /
#     m_angularVelocityX/Y/Z / m_angularAccelerationX/Y/Z / m_frontWheelsAngle
#   float[4]：m_wheelVertForce
#   6 × 标量：
#     m_frontAeroHeight(底板前缘离地高) / m_rearAeroHeight(底板后缘离地高) /
#     m_frontRollAngle / m_rearRollAngle / m_chassisYaw / m_chassisPitch
#   float[4] ×2：m_wheelCamber / m_wheelCamberGain
#
# 本包是「规则9 刮底检测」唯一的真实信号源：规范明确
# ``m_frontAeroHeight`` / ``m_rearAeroHeight`` 为 "plank edge height above road
# surface"（底板前后缘离地高度，单位米），采样频率与菜单设置一致（最高 60Hz）。
# ``m_suspensionPosition`` 亦可用于识别悬挂触底（行程被压到下限）。
_MOTIONEX_BODY = struct.Struct("<" + "4f" * 8 + "f" * 11 + "4f" + "f" * 6 + "4f" * 2)
assert _MOTIONEX_BODY.size == 244, f"motionex body size mismatch: {_MOTIONEX_BODY.size}"

# Packet 13 的 8 个车轮数组字段（顺序即规范顺序）
_MOTIONEX_WHEEL_ARRAYS_HEAD = (
    "m_suspensionPosition",
    "m_suspensionVelocity",
    "m_suspensionAcceleration",
    "m_wheelSpeed",
    "m_wheelSlipRatio",
    "m_wheelSlipAngle",
    "m_wheelLatForce",
    "m_wheelLongForce",
)
# 中段 11 个标量字段（官方规范顺序，紧随 8 个车轮数组之后）
_MOTIONEX_SCALARS_HEAD = (
    "m_heightOfCOGAboveGround",
    "m_localVelocityX",
    "m_localVelocityY",
    "m_localVelocityZ",
    "m_angularVelocityX",
    "m_angularVelocityY",
    "m_angularVelocityZ",
    "m_angularAccelerationX",
    "m_angularAccelerationY",
    "m_angularAccelerationZ",
    "m_frontWheelsAngle",
)
# 夹在 m_wheelVertForce 之后的车轮数组字段
_MOTIONEX_WHEEL_ARRAYS_TAIL = ("m_wheelCamber", "m_wheelCamberGain")
# 中段标量（m_frontWheelsAngle 之后的 6 个）
_MOTIONEX_SCALARS_TAIL = (
    "m_frontAeroHeight",
    "m_rearAeroHeight",
    "m_frontRollAngle",
    "m_rearRollAngle",
    "m_chassisYaw",
    "m_chassisPitch",
)


def parse_motion_ex(data: bytes, player_car_index: int = 0) -> dict[str, Any]:
    """解析 Packet 13 (MotionEx) — 玩家车悬挂/姿态/离地高度。

    **非按车分组**：包体只含玩家车一份数据，紧随 29 字节包头之后。
    ``player_car_index`` 仅为与 :data:`_PARSERS` 统一调用签名而保留，
    本包解析不使用该参数（与 ``parse_session`` 同模式）。

    Raises:
        PacketTooShortError: 包体不足以覆盖 244 字节 MotionEx 结构。
    """
    size = _MOTIONEX_BODY.size
    end = HEADER_SIZE + size
    if len(data) < end:
        raise PacketTooShortError(
            f"packet too short for MotionEx: {len(data)} bytes < {end}",
        )
    v = _MOTIONEX_BODY.unpack(data[HEADER_SIZE:end])

    out: dict[str, Any] = {}
    i = 0
    for name in _MOTIONEX_WHEEL_ARRAYS_HEAD:
        out[name] = list(v[i:i + 4])
        i += 4
    for name in _MOTIONEX_SCALARS_HEAD:
        out[name] = v[i]
        i += 1
    out["m_wheelVertForce"] = list(v[i:i + 4])
    i += 4
    for name in _MOTIONEX_SCALARS_TAIL:
        out[name] = v[i]
        i += 1
    for name in _MOTIONEX_WHEEL_ARRAYS_TAIL:
        out[name] = list(v[i:i + 4])
        i += 4
    return out


# --------------------------------------------------------------------------- #
# Packet 0 — Motion（按车分组：世界坐标/速度/方向/G 值/姿态）
# --------------------------------------------------------------------------- #
# Source: 2026 Season Pack Telemetry Output Structures (EA, F1 26 UDP spec)
# CarMotionData = 54 字节（24 车 × 54 + 29 = 1325 ✓ 与实测一致）：
#   float ×6  m_worldPosition{X,Y,Z} / m_worldVelocity{X,Y,Z}（米、米/秒）
#   int16 ×6  m_worldForwardDir{X,Y,Z} / m_worldRightDir{X,Y,Z}（归一化，/32767）
#   int16 ×3  m_gForce{Lateral,Longitudinal,Vertical}  ← **量化值，除以 1000.0**
#   float ×3  m_yaw / m_pitch / m_roll（弧度）
# 注意：G 值量化除数官方为 **1000.0**（部分第三方解析器误用 100）。
_MOTION_CAR = struct.Struct("<ffffffhhhhhhhhhfff")
assert _MOTION_CAR.size == 54, _MOTION_CAR.size
#: Motion G 值量化除数（官方规范原文：divide by 1000.0f）。
MOTION_G_DIVISOR = 1000.0


def parse_motion(data: bytes, player_car_index: int = 0) -> dict[str, Any]:
    """解析 Packet 0 (Motion) — 玩家车世界坐标/速度/G 值/姿态。

    用途：世界坐标可还原理想线（与赛道 SVG 几何对齐），三轴 G 值可做
    纵向/横向加速度分析（刹车点、最大侧向 G、路肩冲击）。

    Raises:
        PacketTooShortError: 包体不足以覆盖玩家车辆结构。
    """
    v = _slice_player_car(data, _MOTION_CAR, player_car_index)
    return {
        "m_worldPositionX": v[0], "m_worldPositionY": v[1], "m_worldPositionZ": v[2],
        "m_worldVelocityX": v[3], "m_worldVelocityY": v[4], "m_worldVelocityZ": v[5],
        "m_worldForwardDirX": v[6], "m_worldForwardDirY": v[7], "m_worldForwardDirZ": v[8],
        "m_worldRightDirX": v[9], "m_worldRightDirY": v[10], "m_worldRightDirZ": v[11],
        # 量化 G 值 → 实际 G（官方：float(m_gForceLateral) / 1000.0f）
        "m_gForceLateral": v[12] / MOTION_G_DIVISOR,
        "m_gForceLongitudinal": v[13] / MOTION_G_DIVISOR,
        "m_gForceVertical": v[14] / MOTION_G_DIVISOR,
        "m_yaw": v[15], "m_pitch": v[16], "m_roll": v[17],
    }


# --------------------------------------------------------------------------- #
# Packet 3 — Event（会话事件）
# --------------------------------------------------------------------------- #
# Source: 2026 Season Pack Telemetry Output Structures — PacketEventData
# header(29) + uint8 m_eventStringCode[4] + union EventDataDetails(12) = 45 ✓
_EVENT_DETAIL = struct.Struct("<12s")
_EVENT_CODES: dict[str, str] = {
    "SSTA": "会话开始", "SEND": "会话结束", "FTLP": "最快圈",
    "RTMT": "退赛", "DRSE": "DRS 启用", "DRSD": "DRS 禁用",
    "TMPT": "队友进站", "CHQF": "挥方格旗", "RCWN": "比赛冠军",
    "PENA": "判罚", "SPTP": "测速点", "STLG": "起步灯亮",
    "LGOT": "起步灯灭", "DTSV": "通过处罚已执行", "SGSV": "停走处罚已执行",
    "FLBK": "回放", "BUTN": "按钮状态", "RDFL": "红旗",
    "OVTK": "超车", "SCAR": "安全车", "COLL": "碰撞",
}
_EVENT_SAFETY_CAR_TYPE = {0: "无安全车", 1: "实体安全车", 2: "虚拟安全车", 3: "编队圈安全车"}
_EVENT_SAFETY_CAR_ACTION = {0: "部署", 1: "回站中", 2: "已回站", 3: "恢复比赛"}
_EVENT_COLLISION_SEVERITY = {0: "轻微", 1: "中等", 2: "严重"}


def parse_event(data: bytes, player_car_index: int = 0) -> dict[str, Any] | None:
    """解析 Packet 3 (Event) — 会话事件（最快圈/判罚/安全车/碰撞…）。

    事件明细是 **union**（同一 12 字节按事件类型解释），故先取事件码再按码分支；
    未知事件码保留 ``m_eventDetailsRaw``（hex）以便后续补解析，不臆测字段。

    ``player_car_index`` 仅为与 :data:`_PARSERS` 统一签名保留。
    """
    end = HEADER_SIZE + 4 + _EVENT_DETAIL.size
    if len(data) < end:
        raise PacketTooShortError(
            f"packet too short for Event: {len(data)} bytes < {end}",
        )
    code = data[HEADER_SIZE:HEADER_SIZE + 4].decode("ascii", errors="replace")
    detail = data[HEADER_SIZE + 4:end]
    out: dict[str, Any] = {
        "m_eventStringCode": code,
        "event_label": _EVENT_CODES.get(code, f"未知({code})"),
        "m_eventDetailsRaw": detail.hex(),
    }
    if code == "FTLP":
        veh, lap_time = struct.unpack_from("<Bf", detail, 0)
        out.update(m_vehicleIdx=veh, m_lapTime=lap_time)
    elif code == "RTMT":
        veh, reason = struct.unpack_from("<BB", detail, 0)
        out.update(m_vehicleIdx=veh, m_reason=reason)
    elif code == "SPTP":
        veh, speed, overall, driver, fast_veh, fast_speed = struct.unpack_from(
            "<BfBBfB", detail, 0,
        )
        out.update(
            m_vehicleIdx=veh, m_speed=speed,
            m_isOverallFastestInSession=overall,
            m_isDriverFastestInSession=driver,
            m_fastestVehicleIdxInSession=fast_veh,
            m_fastestSpeedInSession=fast_speed,
        )
    elif code == "SCAR":
        sc_type, ev_type = struct.unpack_from("<BB", detail, 0)
        out.update(
            m_safetyCarType=sc_type,
            m_eventType=ev_type,
            safety_car_label=(
                f"{_EVENT_SAFETY_CAR_TYPE.get(sc_type, '?')}"
                f"·{_EVENT_SAFETY_CAR_ACTION.get(ev_type, '?')}"
            ),
        )
    elif code == "COLL":
        v1, v2, sev = struct.unpack_from("<BBB", detail, 0)
        out.update(
            m_vehicle1Idx=v1, m_vehicle2Idx=v2, m_severity=sev,
            collision_label=_EVENT_COLLISION_SEVERITY.get(sev, "?"),
        )
    elif code == "PENA":
        vals = struct.unpack_from("<BBBBBBB", detail, 0)
        out.update(
            m_penaltyType=vals[0], m_infringementType=vals[1],
            m_vehicleIdx=vals[2], m_otherVehicleIdx=vals[3],
            m_time=vals[4], m_lapNum=vals[5], m_placesGained=vals[6],
        )
    elif code == "STLG":
        out["m_numLights"] = struct.unpack_from("<B", detail, 0)[0]
    elif code == "OVTK":
        a, b = struct.unpack_from("<BB", detail, 0)
        out.update(m_overtakingVehicleIdx=a, m_beingOvertakenVehicleIdx=b)
    elif code == "DRSD":
        out["m_reason"] = struct.unpack_from("<B", detail, 0)[0]
    elif code in ("TMPT", "RCWN", "DTSV"):
        out["m_vehicleIdx"] = struct.unpack_from("<B", detail, 0)[0]
    elif code == "SGSV":
        veh, stop_time = struct.unpack_from("<Bf", detail, 0)
        out.update(m_vehicleIdx=veh, m_stopTime=stop_time)
    elif code == "FLBK":
        frame, session_time = struct.unpack_from("<If", detail, 0)
        out.update(m_flashbackFrameIdentifier=frame, m_flashbackSessionTime=session_time)
    elif code == "BUTN":
        out["m_buttonStatus"] = struct.unpack_from("<I", detail, 0)[0]
    return out


# --------------------------------------------------------------------------- #
# Packet 4 — Participants（参赛者名单）
# --------------------------------------------------------------------------- #
# Source: 2026 Season Pack Telemetry Output Structures — ParticipantData
# 60 字节/车（24 车 × 60 + 29 + 1 = 1470 ✓）：
#   uint8 m_aiControlled; uint16 m_driverId; uint16 m_networkId; uint16 m_teamId;
#   uint8 m_myTeam; uint8 m_raceNumber; uint8 m_nationality;
#   char[32] m_name; uint8 m_yourTelemetry; uint8 m_showOnlineNames;
#   uint16 m_techLevel; uint8 m_platform; uint8 m_numColours;
#   LiveryColour[4] m_liveryColours (uint8 rgb ×3 ×4)
_PARTICIPANT = struct.Struct("<BHHHBBB32sBBHBB" + "B" * 12)
assert _PARTICIPANT.size == 60, _PARTICIPANT.size


def parse_participants(data: bytes, player_car_index: int = 0) -> dict[str, Any]:
    """解析 Packet 4 (Participants) — 参赛者名单（含玩家标记/车队/名字）。

    与既有解析器不同，本包**解全部车位**：名单低频（实测 358 包/会话）且
    "谁在场上"是分析多车数据的前提。名字为 UTF-8 定长字段，尾部 NUL 截断。
    """
    need = HEADER_SIZE + 1 + _PARTICIPANT.size * NUM_CARS
    if len(data) < HEADER_SIZE + 1:
        raise PacketTooShortError(f"packet too short for Participants: {len(data)}")
    num_active = data[HEADER_SIZE]
    # 包体可能不含全部 24 槽（游戏中通常补齐）——按实际长度解到哪算哪
    available = max(0, min(NUM_CARS, (len(data) - HEADER_SIZE - 1) // _PARTICIPANT.size))
    cars: list[dict[str, Any]] = []
    for i in range(available):
        off = HEADER_SIZE + 1 + i * _PARTICIPANT.size
        v = _PARTICIPANT.unpack_from(data, off)
        name = v[7].split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        cars.append({
            "car_index": i,
            "m_aiControlled": v[0],
            "m_driverId": v[1],
            "m_networkId": v[2],
            "m_teamId": v[3],
            "m_myTeam": v[4],
            "m_raceNumber": v[5],
            "m_nationality": v[6],
            "m_name": name,
            "m_yourTelemetry": v[8],
            "m_showOnlineNames": v[9],
            "m_techLevel": v[10],
            "m_platform": v[11],
            "m_numColours": v[12],
        })
    player = cars[player_car_index] if 0 <= player_car_index < len(cars) else None
    return {
        "m_numActiveCars": num_active,
        "participants": cars,
        "player": player,
        # 便于落库/报告：玩家名字与车队
        "m_playerName": player["m_name"] if player else None,
        "m_playerTeamId": player["m_teamId"] if player else None,
        "_needFull": need,
    }


# --------------------------------------------------------------------------- #
# Packet 8 — FinalClassification（最终名次）
# --------------------------------------------------------------------------- #
# Source: 2026 Season Pack Telemetry Output Structures — FinalClassificationData
# 46 字节/车（24 车 × 46 + 29 + 1 = 1134 ✓）：
#   uint8 ×7 position/numLaps/gridPosition/points/numPitStops/resultStatus/resultReason
#   uint32 m_bestLapTimeInMS; double m_totalRaceTime;
#   uint8 penaltiesTime/numPenalties/numTyreStints;
#   uint8[8] m_tyreStintsActual / m_tyreStintsVisual / m_tyreStintsEndLaps
_FINAL_CLASS = struct.Struct("<" + "B" * 7 + "Id" + "B" * 3 + "B" * 24)
assert _FINAL_CLASS.size == 46, _FINAL_CLASS.size


def parse_final_classification(data: bytes, player_car_index: int = 0) -> dict[str, Any]:
    """解析 Packet 8 (FinalClassification) — 最终名次/最佳圈/轮胎分段。"""
    if len(data) < HEADER_SIZE + 1:
        raise PacketTooShortError(f"packet too short for FinalClassification: {len(data)}")
    num_cars = data[HEADER_SIZE]
    available = max(0, min(NUM_CARS, (len(data) - HEADER_SIZE - 1) // _FINAL_CLASS.size))
    cars: list[dict[str, Any]] = []
    for i in range(available):
        off = HEADER_SIZE + 1 + i * _FINAL_CLASS.size
        v = _FINAL_CLASS.unpack_from(data, off)
        cars.append({
            "car_index": i,
            "m_position": v[0], "m_numLaps": v[1], "m_gridPosition": v[2],
            "m_points": v[3], "m_numPitStops": v[4],
            "m_resultStatus": v[5], "m_resultReason": v[6],
            "m_bestLapTimeInMS": v[7], "m_totalRaceTime": v[8],
            "m_penaltiesTime": v[9], "m_numPenalties": v[10],
            "m_numTyreStints": v[11],
            "m_tyreStintsActual": list(v[12:20]),
            "m_tyreStintsVisual": list(v[20:28]),
            "m_tyreStintsEndLaps": list(v[28:36]),
        })
    return {"m_numCars": num_cars, "classification": cars}


# --------------------------------------------------------------------------- #
# Packet 10 — CarDamage（车辆损伤）
# --------------------------------------------------------------------------- #
# Source: 2026 Season Pack Telemetry Output Structures — CarDamageData
# 46 字节/车（24 车 × 46 + 29 = 1133 ✓）：
#   float[4] m_tyresWear（胎耗 %）
#   uint8[4] m_tyresDamage / m_brakesDamage / m_tyreBlisters
#   uint8 ×18 前左翼/前右翼/尾翼/底板/扩散器/侧箱/DRS 故障/ERS 故障/
#             变速箱/引擎/MGU-H/ES/CE/ICE/MGU-K/TC/引擎爆缸/引擎卡死
_CAR_DAMAGE = struct.Struct("<ffff" + "B" * 30)
assert _CAR_DAMAGE.size == 46, _CAR_DAMAGE.size
#: 除胎耗外的 uint8 损伤字段名（顺序严格按官方结构）。
_DAMAGE_BYTE_FIELDS = (
    "m_tyresDamage", "m_brakesDamage", "m_tyreBlisters",
    "m_frontLeftWingDamage", "m_frontRightWingDamage", "m_rearWingDamage",
    "m_floorDamage", "m_diffuserDamage", "m_sidepodDamage",
    "m_drsFault", "m_ersFault", "m_gearBoxDamage", "m_engineDamage",
    "m_engineMGUHWear", "m_engineESWear", "m_engineCEWear", "m_engineICEWear",
    "m_engineMGUKWear", "m_engineTCWear", "m_engineBlown", "m_engineSeized",
)


def parse_car_damage(data: bytes, player_car_index: int = 0) -> dict[str, Any]:
    """解析 Packet 10 (CarDamage) — 玩家车损伤/胎耗/胎泡。

    关键用途：**损伤会拖慢圈速**。此前未解析，训练样本里"圈速慢"无法区分
    "调教不好"还是"车撞坏了"——损伤标签是训练数据质量的必要混淆控制。
    """
    v = _slice_player_car(data, _CAR_DAMAGE, player_car_index)
    out: dict[str, Any] = {"m_tyresWear": [v[0], v[1], v[2], v[3]]}
    for i, name in enumerate(_DAMAGE_BYTE_FIELDS):
        if i < 3:
            # 前 3 项（胎损/刹车损/胎泡）是 4 元素数组
            out[name] = [v[4 + i * 4 + k] for k in range(4)]
        else:
            out[name] = v[16 + (i - 3)]
    # 是否需要"明显损伤"快捷标志（供引擎/报告直接判污染）
    wing = max(out["m_frontLeftWingDamage"], out["m_frontRightWingDamage"])
    out["damage_severe"] = bool(
        wing >= 20 or out["m_rearWingDamage"] >= 20 or out["m_floorDamage"] >= 20
        or out["m_diffuserDamage"] >= 20 or out["m_engineBlown"] or out["m_engineSeized"]
    )
    return out


# --------------------------------------------------------------------------- #
# Packet 11 — SessionHistory（逐圈历史 + 轮胎分段）
# --------------------------------------------------------------------------- #
# Source: 2026 Season Pack Telemetry Output Structures — PacketSessionHistoryData
# LapHistoryData = 14 字节（含扇区分/秒两段编码 + 有效位标志）
# 100 圈 × 14 + 8 段 × 3 + 29 + 7 = 1460 ✓
_LAP_HISTORY = struct.Struct("<IHBHBHBB")
_TYRE_STINT_HISTORY = struct.Struct("<BBB")
_CS_MAX_LAPS_HISTORY = 100
_CS_MAX_TYRE_STINTS = 8


def parse_session_history(data: bytes, player_car_index: int = 0) -> dict[str, Any]:
    """解析 Packet 11 (SessionHistory) — 逐圈用时/扇区/有效位 + 轮胎分段。

    这是**唯一带官方"圈有效位标志"**的来源（``m_lapValidBitFlags``：
    0x01 圈有效、0x02/0x04/0x08 对应三个扇区有效），比 LapData 的
    单帧 ``m_currentLapInvalid`` 更适合训练数据标注。
    """
    need = HEADER_SIZE + 7
    if len(data) < need:
        raise PacketTooShortError(f"packet too short for SessionHistory: {len(data)}")
    off = HEADER_SIZE
    car_idx = data[off]
    num_laps = data[off + 1]
    num_stints = data[off + 2]
    best_lap = data[off + 3]
    best_s1, best_s2, best_s3 = data[off + 4], data[off + 5], data[off + 6]
    off += 7

    laps: list[dict[str, Any]] = []
    max_parsable = max(0, min(_CS_MAX_LAPS_HISTORY, (len(data) - off) // _LAP_HISTORY.size))
    for i in range(max_parsable):
        v = _LAP_HISTORY.unpack_from(data, off + i * _LAP_HISTORY.size)
        s1 = v[1] + v[2] * 60000
        s2 = v[3] + v[4] * 60000
        s3 = v[5] + v[6] * 60000
        flags = v[7]
        laps.append({
            "lap_number": i + 1,
            "m_lapTimeInMS": v[0],
            "sector1_ms": s1, "sector2_ms": s2, "sector3_ms": s3,
            "m_lapValidBitFlags": flags,
            "lap_valid": bool(flags & 0x01),
            "sector1_valid": bool(flags & 0x02),
            "sector2_valid": bool(flags & 0x04),
            "sector3_valid": bool(flags & 0x08),
        })
    off += max_parsable * _LAP_HISTORY.size

    stints: list[dict[str, Any]] = []
    max_stints = max(0, min(_CS_MAX_TYRE_STINTS, (len(data) - off) // _TYRE_STINT_HISTORY.size))
    for i in range(max_stints):
        v = _TYRE_STINT_HISTORY.unpack_from(data, off + i * _TYRE_STINT_HISTORY.size)
        stints.append({
            "m_endLap": v[0],
            "m_tyreActualCompound": v[1],
            "m_tyreVisualCompound": v[2],
            "is_current": v[0] == 255,
        })
    return {
        "m_carIdx": car_idx,
        "m_numLaps": num_laps,
        "m_numTyreStints": num_stints,
        "m_bestLapTimeLapNum": best_lap,
        "m_bestSector1LapNum": best_s1,
        "m_bestSector2LapNum": best_s2,
        "m_bestSector3LapNum": best_s3,
        "lap_history": laps,
        "tyre_stints": stints,
    }


# --------------------------------------------------------------------------- #
# Packet 12 — TyreSets（轮胎组）
# --------------------------------------------------------------------------- #
# Source: 2026 Season Pack Telemetry Output Structures — PacketTyreSetsData
# 10 字节/套（20 套 × 10 + 29 + 1 + 1 = 231 ✓）：
#   uint8 actual/visual compound, wear, available, recommendedSession,
#         lifeSpan, usableLife; int16 m_lapDeltaTime; uint8 m_fitted
_CS_MAX_TYRE_SETS = 20
_TYRE_SET = struct.Struct("<BBBBBBBhB")
assert _TYRE_SET.size == 10, _TYRE_SET.size


def parse_tyre_sets(data: bytes, player_car_index: int = 0) -> dict[str, Any]:
    """解析 Packet 12 (TyreSets) — 可用轮胎组/磨损/寿命/圈速差。

    ``m_lapDeltaTime``（与已装配组的圈速差，毫秒）是**配方选择的直接依据**；
    ``m_lifeSpan`` / ``m_usableLife`` 支撑"这套胎还能跑几圈"的策略判断。
    """
    if len(data) < HEADER_SIZE + 2:
        raise PacketTooShortError(f"packet too short for TyreSets: {len(data)}")
    car_idx = data[HEADER_SIZE]
    sets: list[dict[str, Any]] = []
    off = HEADER_SIZE + 1
    available = max(0, min(_CS_MAX_TYRE_SETS, (len(data) - off) // _TYRE_SET.size))
    for i in range(available):
        v = _TYRE_SET.unpack_from(data, off + i * _TYRE_SET.size)
        sets.append({
            "index": i,
            "m_actualTyreCompound": v[0],
            "m_visualTyreCompound": v[1],
            "m_wear": v[2],
            "m_available": v[3],
            "m_recommendedSession": v[4],
            "m_lifeSpan": v[5],
            "m_usableLife": v[6],
            "m_lapDeltaTime": v[7],
            "m_fitted": v[8],
        })
    fitted_idx = None
    tail = off + available * _TYRE_SET.size
    if len(data) > tail:
        fitted_idx = data[tail]
    fitted = next((s for s in sets if s["m_fitted"]), None)
    return {
        "m_carIdx": car_idx,
        "tyre_sets": sets,
        "m_fittedIdx": fitted_idx,
        "fitted": fitted,
    }


# --------------------------------------------------------------------------- #
# Packet 15 — LapPositions（逐圈位置矩阵）
# --------------------------------------------------------------------------- #
# Source: 2026 Season Pack Telemetry Output Structures — PacketLapPositionsData
# uint8 m_numLaps + uint8 m_lapStart + uint8[50][24] → 29 + 2 + 1200 = 1231 ✓
_CS_MAX_LAPS_LAP_POSITIONS = 50


def parse_lap_positions(data: bytes, player_car_index: int = 0) -> dict[str, Any]:
    """解析 Packet 15 (LapPositions) — 每圈起点的全部车位置（可画位置图）。

    位置矩阵为 50 圈 × 24 车；``0`` 表示该圈无记录。
    """
    if len(data) < HEADER_SIZE + 2:
        raise PacketTooShortError(f"packet too short for LapPositions: {len(data)}")
    num_laps = data[HEADER_SIZE]
    lap_start = data[HEADER_SIZE + 1]
    base = HEADER_SIZE + 2
    laps: list[list[int]] = []
    for lap in range(min(num_laps, _CS_MAX_LAPS_LAP_POSITIONS)):
        off = base + lap * NUM_CARS
        if off + NUM_CARS > len(data):
            break
        laps.append(list(data[off:off + NUM_CARS]))
    return {
        "m_numLaps": num_laps,
        "m_lapStart": lap_start,
        "positions": laps,
    }


# --------------------------------------------------------------------------- #
# Packet 16 — CarTelemetry2（2026 赛季包新增：主动空力 + 超车模式）
# --------------------------------------------------------------------------- #
# Source: 2026 Season Pack Telemetry Output Structures — CarTelemetry2Data
# 10 字节/车（24 车 × 10 + 29 = 269 ✓）：
#   uint8  m_activeAeroMode                // 0 = Corner mode, 1 = Straight mode
#   uint8  m_activeAeroAvailable           // 0/1
#   uint16 m_activeAeroActivationDistance  // 0=不可用，非 0=还有 X 米可用
#   uint8  m_overtakeAvailable
#   uint8  m_overtakeActive
#   uint16 m_overtakeActivationDistance
#   uint8  m_2026Regulations               // 1 = 适用 2026 规则
#   uint8  m_drivingWrongWay
# 实测校验：玩家数据 offset8 恒为 1（m_2026Regulations）→ 结构与实车行为吻合。
_CAR_TELEMETRY2 = struct.Struct("<BBHBBHBB")
assert _CAR_TELEMETRY2.size == 10, _CAR_TELEMETRY2.size
ACTIVE_AERO_CORNER = 0
ACTIVE_AERO_STRAIGHT = 1


def parse_car_telemetry_2(data: bytes, player_car_index: int = 0) -> dict[str, Any]:
    """解析 Packet 16 (CarTelemetry2) — 2026 主动空力模式 + 超车模式状态。

    F1 2026 引入主动空力（Corner/Straight 两种模式）与超车模式，本包是其
    **唯一遥测出口**；此前未解析 → 工具看不到空力模式切换与超车模式可用性
    （而空力模式直接决定直道阻力与弯中下压力）。
    """
    v = _slice_player_car(data, _CAR_TELEMETRY2, player_car_index)
    return {
        "m_activeAeroMode": v[0],
        "active_aero_mode_label": (
            "直道模式" if v[0] == ACTIVE_AERO_STRAIGHT else "弯道模式"
        ),
        "m_activeAeroAvailable": v[1],
        "m_activeAeroActivationDistance": v[2],
        "m_overtakeAvailable": v[3],
        "m_overtakeActive": v[4],
        "m_overtakeActivationDistance": v[5],
        "m_2026Regulations": v[6],
        "m_drivingWrongWay": v[7],
    }


# --------------------------------------------------------------------------- #

# 主入口：按 packetId 分发
# --------------------------------------------------------------------------- #
# 本版支持的 7 类包及其解析函数
_PARSERS: dict[int, Any] = {
    0: parse_motion,
    1: parse_session,
    2: parse_lap_data,
    3: parse_event,
    4: parse_participants,
    5: parse_car_setups,
    6: parse_car_telemetry,
    7: parse_car_status,
    8: parse_final_classification,
    10: parse_car_damage,
    11: parse_session_history,
    12: parse_tyre_sets,
    13: parse_motion_ex,
    15: parse_lap_positions,
    16: parse_car_telemetry_2,
}

# 支持的 packetId 集合（供外部查询）
SUPPORTED_PACKET_IDS: frozenset[int] = frozenset(_PARSERS.keys())


def parse_packet(data: bytes) -> dict[str, Any] | None:
    """解析一个完整 UDP 包，返回 ``{packet_id, name, header, **body}`` 或 ``None``。

    - 短包（< 29 字节或包体不足）抛 :class:`PacketTooShortError`。
    - 未知 packetId（不在 7 类支持范围内）返回 ``None``，不抛错（跳过不崩溃）。
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