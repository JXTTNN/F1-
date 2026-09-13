"""生成模拟 F1 遥测数据并填充数据库。

为 24 条赛道生成真实的模拟遥测数据：
- 每赛道 10-20 圈
- 每圈包含：速度/油门/刹车/档位/转速/圈速/扇区/弯道标记
- 模拟 EA F1 2026 UDP 数据包格式（Packet 1/2/5/6）
- 填充 SQLite 数据库（setup / feedback / suggestion / iteration 表）
- 生成训练数据集 data/training_dataset.json 供神经网络训练用

使用方式::

    python scripts/generate_sim_telemetry.py

输出：
- data/f1opt.db                            — 填充后的 SQLite 数据库
- data/training_dataset.json               — 神经网络训练数据集
- data/sim_telemetry/sim_packets_<track>.jsonl — 每赛道的模拟 UDP 包流（可选审计）

数据来源：
- 赛道/弯道静态数据来自 :mod:`setup_tuner.domain.track`（24 条 F1 2026 赛历）
- 调教参数 schema 来自 :mod:`setup_tuner.domain.setup`（23 项 7 大类）
- 症状枚举来自 :mod:`setup_tuner.domain.symptoms`（12 项 4 类）
- 规则引擎 :mod:`setup_tuner.engine.engine` 用于生成 expected_delta
"""

from __future__ import annotations

import json
import math
import random
import struct
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# 确保项目根目录在 sys.path 中（脚本可独立运行）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from setup_tuner.db.store import Store  # noqa: E402
from setup_tuner.domain.setup import CarSetup  # noqa: E402
from setup_tuner.domain.symptoms import (  # noqa: E402
    Symptom,
    get_symptom_category,
)
from setup_tuner.domain.track import Track, get_all_tracks  # noqa: E402
from setup_tuner.engine.engine import generate_suggestion  # noqa: E402
from setup_tuner.telemetry.packets import (  # noqa: E402
    HEADER_FORMAT,
    NUM_CARS,
)

# --------------------------------------------------------------------------- #
# 常量与配置
# --------------------------------------------------------------------------- #
# 用固定随机种子保证可复现（对齐 FR-ENG-05 确定性要求）
_RANDOM_SEED = 20260911

# 数据库路径
_DB_PATH = _ROOT / "data" / "f1opt.db"

# 训练数据集路径
_TRAINING_DATASET_PATH = _ROOT / "data" / "training_dataset.json"

# 模拟 UDP 包流审计输出目录
_SIM_PACKETS_DIR = _ROOT / "data" / "sim_telemetry"

# EA F1 2026 UDP 包头常量
_PACKET_FORMAT = 2026
_GAME_YEAR = 26
_GAME_MAJOR_VERSION = 1
_GAME_MINOR_VERSION = 0
_PACKET_VERSION = 1

# 模拟物理参数（基于 F1 真实量级）
_MAX_SPEED_KMH = 340           # 直道极速上限
_MIN_SPEED_KMH = 45            # Monaco 大酒店弯最低速度
_IDLE_RPM = 2500               # 怠速转速
_MAX_RPM = 12500               # 红线转速
_GEAR_MAX = 8                  # 8 档
_FRAMES_PER_SEC = 60           # UDP 输出频率 60Hz
_FRAMES_PER_LAP = 600          # 每圈约 10 秒 × 60fps = 600 帧（可变）

# 赛道类型 → 圈速基准（秒）与圈数范围
# 量级核对自 F1 2026 真实圈速（Monaco ~72s, Suzuka ~95s, Monza ~83s）
_TRACK_TYPE_LAP_TIME: dict[str, tuple[float, float]] = {
    # track_type → (base_lap_time_sec, jitter_sec)
    "high_speed_low_downforce": (85.0, 2.0),   # Monza-like
    "street":                   (76.0, 3.0),   # Monaco/Miami-like
    "high_downforce":           (78.0, 2.5),   # Hungaroring-like
    "medium":                   (90.0, 3.0),   # Melbourne-like
    "mixed":                    (92.0, 3.5),   # Suzuka/Silverstone-like
}

# 赛道长度修正：长赛道圈速按比例放大
_LENGTH_REF_M = 5000.0  # 参考长度


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO8601 字符串。"""
    return datetime.now(UTC).isoformat()


def _clamp(value: float, lo: float, hi: float) -> float:
    """裁剪到 [lo, hi]。"""
    return max(lo, min(hi, value))


def _speed_to_gear(speed_kmh: float) -> int:
    """速度 → 档位映射（F1 8 档）。"""
    if speed_kmh < 30:
        return 1
    if speed_kmh < 80:
        return 2
    if speed_kmh < 130:
        return 3
    if speed_kmh < 180:
        return 4
    if speed_kmh < 220:
        return 5
    if speed_kmh < 260:
        return 6
    if speed_kmh < 300:
        return 7
    return 8


def _speed_to_rpm(speed_kmh: float, gear: int) -> int:
    """速度 + 档位 → 转速（简化模型）。"""
    if speed_kmh <= 0:
        return _IDLE_RPM
    # 每档对应的速度区间中点对应 ~11000rpm
    gear_speed_min = [0, 30, 80, 130, 180, 220, 260, 300, 340]
    lo = gear_speed_min[max(0, gear - 1)]
    hi = gear_speed_min[min(_GEAR_MAX, gear)]
    if hi <= lo:
        return _IDLE_RPM
    ratio = (speed_kmh - lo) / (hi - lo)
    rpm = _IDLE_RPM + ratio * (_MAX_RPM - _IDLE_RPM)
    return int(_clamp(rpm, _IDLE_RPM, _MAX_RPM))


def _estimate_lap_time(track: Track) -> float:
    """根据赛道类型与长度估算圈速基准（秒）。"""
    base, jitter = _TRACK_TYPE_LAP_TIME.get(track.track_type, (90.0, 3.0))
    # 按长度修正
    length_factor = track.length_m / _LENGTH_REF_M
    return base * math.sqrt(length_factor) + random.uniform(-jitter, jitter)


# --------------------------------------------------------------------------- #
# EA F1 2026 UDP 包构造
# --------------------------------------------------------------------------- #
def _build_header(
    packet_id: int,
    session_uid: int,
    session_time: float,
    frame_identifier: int,
    player_car_index: int = 0,
) -> bytes:
    """构造 29 字节 EA F1 2026 UDP 包头。

    对齐 :mod:`setup_tuner.telemetry.packets` 的 HEADER_FORMAT。
    """
    return struct.pack(
        HEADER_FORMAT,
        _PACKET_FORMAT,
        _GAME_YEAR,
        _GAME_MAJOR_VERSION,
        _GAME_MINOR_VERSION,
        _PACKET_VERSION,
        packet_id,
        session_uid,
        session_time,
        frame_identifier,
        frame_identifier,  # overall_frame_identifier
        player_car_index,
        255,  # secondary_player_car_index（无第二玩家）
    )


def _build_session_packet(
    track: Track,
    session_uid: int,
    session_time: float,
    frame_id: int,
    total_laps: int,
) -> bytes:
    """构造 Packet 1 (Session) — 赛道/天气/气温。

    仅填充关键字段，其余用合理默认值；包体长度对齐
    :mod:`setup_tuner.telemetry.packets` 的 _SESSION_PREFIX_STRUCT。
    """
    header = _build_header(1, session_uid, session_time, frame_id)
    # 16 字段前缀（对齐 packets.py _SESSION_PREFIX_FMT 的 "BbbBHBbBHHBBBBBB"）
    # 1. m_weather (B)        2. m_trackTemperature (b)  3. m_airTemperature (b)
    # 4. m_totalLaps (B)      5. m_trackLength (H)       6. m_sessionType (B)
    # 7. m_trackId (b)        8. m_formula (B)           9. m_sessionTimeLeft (H)
    # 10. m_sessionDuration (H) 11. m_pitSpeedLimit (B)  12. m_gamePaused (B)
    # 13. m_isSpectating (B)  14. m_spectatorCarIndex (B) 15. m_sliProNativeSupport (B)
    # 16. m_numMarshalZones (B)
    weather = 0  # clear
    track_temp = 25 + random.randint(-5, 10)
    air_temp = track_temp - 5
    track_len = int(track.length_m)
    session_type = 5  # race
    formula = 1  # F1
    session_time_left = 0
    session_duration = 0
    pit_speed_limit = 80
    prefix = struct.pack(
        "<BbbBHBbBHHBBBBBB",
        weather, track_temp, air_temp, total_laps, track_len,
        session_type, track.udp_track_id, formula,
        session_time_left, session_duration, pit_speed_limit,
        0, 0, 0, 0, 0,  # paused/spectating/spectatorIdx/sliPro/numMarshal
    )
    # 21 marshal zones (各 float + uint8 = 5 字节)
    marshal_zones = b"".join(struct.pack("<fb", 0.0, 0) for _ in range(21))
    # safetyCarStatus, networkGame, numWeatherForecastSamples
    suffix = struct.pack("<BBB", 0, 0, 0)
    return header + prefix + marshal_zones + suffix


def _build_lap_packet(
    session_uid: int,
    session_time: float,
    frame_id: int,
    last_lap_ms: int,
    current_lap_ms: int,
    sector1_ms: int,
    sector2_ms: int,
    lap_distance: float,
    total_distance: float,
    current_lap_num: int,
    sector: int,
    speed_trap_speed: float,
) -> bytes:
    """构造 Packet 2 (LapData) — 玩家车圈速/扇区/距离。

    只填充玩家车（player_car_index=0）那一段，其余 23 车位用零填充。
    """
    header = _build_header(2, session_uid, session_time, frame_id)
    # 玩家车 58 字节（对齐 packets.py _LAP_PER_FMT）
    # II(2) + HBHBHBHB(8) + fff(3) + BBBBBBBBBBBBBBB(15) + HHB(3) + fB(2) = 33 字段
    # sector 编码：MSPart (uint16) + MinutesPart (uint8)
    s1_ms_part = sector1_ms % 60000
    s1_min_part = sector1_ms // 60000
    s2_ms_part = sector2_ms % 60000
    s2_min_part = sector2_ms // 60000
    df_ms_part = 0
    df_min_part = 0
    dl_ms_part = 0
    dl_min_part = 0
    player = struct.pack(
        "<" + "II" + "HBHBHBHB" + "fff" + "BBBBBBBBBBBBBBB" + "HHB" + "fB",
        last_lap_ms, current_lap_ms,  # II (2)
        s1_ms_part, s1_min_part, s2_ms_part, s2_min_part,  # HBHB (4)
        df_ms_part, df_min_part, dl_ms_part, dl_min_part,  # HBHB (4)
        lap_distance, total_distance, 0.0,  # fff (3)
        # 15 × uint8: carPosition/currentLapNum/pitStatus/numPitStops/sector/
        #   currentLapInvalid/penalties/totalWarnings/cornerCuttingWarnings/
        #   numUnservedDriveThroughPens/numUnservedStopGoPens/gridPosition/
        #   driverStatus/resultStatus/pitLaneTimerActive
        1, current_lap_num, 0, 0, sector, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  # 15
        0, 0, 0,  # HHB (3): pitLaneTimeInLaneInMS, pitStopTimerInMS, pitStopShouldServePen
        speed_trap_speed, 0,  # fB (2)
    )
    # 其余 23 车位用零填充（避免解析越界）
    other_car_size = struct.calcsize(
        "<" + "II" + "HBHBHBHB" + "fff" + "BBBBBBBBBBBBBBB" + "HHB" + "fB",
    )
    others = b"\x00" * (other_car_size * (NUM_CARS - 1))
    return header + player + others


def _build_telemetry_packet(
    session_uid: int,
    session_time: float,
    frame_id: int,
    speed: float,
    throttle: float,
    brake: float,
    steer: float,
    gear: int,
    rpm: int,
    drs: int = 0,
) -> bytes:
    """构造 Packet 6 (CarTelemetry) — 玩家车实时遥测。"""
    header = _build_header(6, session_uid, session_time, frame_id)
    # 玩家车 59 字节（参考 packets.py _TELEM_PER_FMT）
    # brakesTemp[4] uint16, tyresSurfaceTemp[4] uint8, tyresInnerTemp[4] uint8,
    # engineTemp uint8, tyresPressure[4] float, surfaceType[4] uint8
    brake_temps = [random.randint(100, 600) for _ in range(4)]
    tyre_surface_temps = [random.randint(80, 110) for _ in range(4)]
    tyre_inner_temps = [random.randint(85, 115) for _ in range(4)]
    engine_temp = random.randint(95, 115)
    tyre_pressures = [25.5 + random.uniform(-1, 1) for _ in range(4)]
    surface_types = [0, 0, 0, 0]
    player = struct.pack(
        "<" + "H" + "fff" + "B" + "b" + "H" + "BBH" + "4H" + "4B" + "4B" + "B" + "4f" + "4B",
        int(speed), throttle, steer, brake,
        0,  # clutch
        gear, rpm,
        drs, 0, 0,  # drs, revLightsPercent, revLightsBitValue
        *brake_temps,
        *tyre_surface_temps,
        *tyre_inner_temps,
        engine_temp,
        *tyre_pressures,
        *surface_types,
    )
    # 其余 23 车位零填充
    other_car_size = struct.calcsize(
        "<" + "H" + "fff" + "B" + "b" + "H" + "BBH" + "4H" + "4B" + "4B" + "B" + "4f" + "4B",
    )
    others = b"\x00" * (other_car_size * (NUM_CARS - 1))
    return header + player + others


def _build_car_setups_packet(
    session_uid: int,
    session_time: float,
    frame_id: int,
    setup: CarSetup,
) -> bytes:
    """构造 Packet 5 (CarSetups) — 玩家车调教参数。"""
    header = _build_header(5, session_uid, session_time, frame_id)
    # 玩家车 50 字节（参考 packets.py _SETUP_PER_FMT）
    player = struct.pack(
        "<" + "BBBB" + "ffff" + "BBBBBBBBB" + "ffff" + "B" + "f",
        int(setup.front_wing), int(setup.rear_wing),
        int(setup.on_throttle_diff), int(setup.off_throttle_diff),
        setup.front_camber, setup.rear_camber,
        setup.front_toe, setup.rear_toe,
        int(setup.front_spring), int(setup.rear_spring),
        int(setup.front_anti_roll_bar), int(setup.rear_anti_roll_bar),
        int(setup.front_ride_height), int(setup.rear_ride_height),
        int(setup.brake_pressure), int(setup.brake_bias),
        int(setup.engine_braking),
        setup.rear_tyre_pressure, setup.rear_tyre_pressure,
        setup.front_tyre_pressure, setup.front_tyre_pressure,
        int(setup.ballast),
        30.0,  # fuelLoad
    )
    # 其余 23 车位零填充
    other_car_size = struct.calcsize(
        "<" + "BBBB" + "ffff" + "BBBBBBBBB" + "ffff" + "B" + "f",
    )
    others = b"\x00" * (other_car_size * (NUM_CARS - 1))
    return header + player + others


# --------------------------------------------------------------------------- #
# 单圈遥测生成
# --------------------------------------------------------------------------- #
def _generate_lap_telemetry(
    track: Track,
    lap_time_sec: float,
    setup: CarSetup,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """为单圈生成 60Hz 遥测帧序列。

    按赛道弯道分布生成速度/油门/刹车/档位/转速曲线：
    - 直道：高速(280-340km/h)、油门100%、刹车0%、高档位(7-8)
    - 弯道：低速(80-250km/h)、油门0-50%、刹车50-100%、低档位(2-5)
    - 添加随机噪声模拟真实驾驶

    Returns:
        遥测帧列表，每帧含 speed/throttle/brake/gear/rpm/lap_distance/sector/corner 等字段。
    """
    n_frames = max(60, int(lap_time_sec * _FRAMES_PER_SEC))
    frames: list[dict[str, Any]] = []
    corners = track.corners
    n_corners = len(corners)
    track_length = track.length_m

    # 弯道位置（按编号均匀分布近似）
    corner_positions = [
        (i + 0.5) / n_corners for i in range(n_corners)
    ]
    # 弯道影响半径（占圈长比例）
    corner_radius = 0.6 / n_corners

    for i in range(n_frames):
        progress = i / n_frames  # 0~1 圈进度
        lap_distance = progress * track_length

        # 计算最近弯道影响
        in_corner = False
        corner_number: int | None = None
        corner_speed_target = _MAX_SPEED_KMH
        for idx, cp in enumerate(corner_positions):
            dist = min(abs(progress - cp), 1.0 - abs(progress - cp))
            if dist < corner_radius:
                # 在弯道影响范围内
                corner = corners[idx]
                # 距离弯心越近速度越低
                factor = 1.0 - (1.0 - dist / corner_radius) ** 2
                target = corner.speed_kmh + (_MAX_SPEED_KMH - corner.speed_kmh) * factor
                if target < corner_speed_target:
                    corner_speed_target = target
                    in_corner = True
                    corner_number = corner.number

        # 速度：目标 + 噪声
        noise = rng.uniform(-5, 5)
        speed = _clamp(corner_speed_target + noise, _MIN_SPEED_KMH, _MAX_SPEED_KMH)

        # 油门/刹车
        if in_corner:
            # 弯道：低油门，可能刹车
            throttle = rng.uniform(0.2, 0.7)
            brake = rng.uniform(0.0, 0.3)
        else:
            # 直道：全油门
            throttle = rng.uniform(0.85, 1.0)
            brake = 0.0

        # 档位/转速
        gear = _speed_to_gear(speed)
        rpm = _speed_to_rpm(speed, gear)

        # DRS（直道且速度 > 250）
        drs = 1 if (not in_corner and speed > 250) else 0

        # 方向盘（弯道有转向）
        steer = rng.uniform(-0.3, 0.3) if in_corner else rng.uniform(-0.05, 0.05)

        # 扇区（0/1/2）
        sector = int(progress * 3) if progress < 1.0 else 2

        frames.append({
            "frame_id": i,
            "progress": progress,
            "speed_kmh": round(speed, 2),
            "throttle": round(throttle, 3),
            "brake": round(brake, 3),
            "steer": round(steer, 3),
            "gear": gear,
            "engine_rpm": rpm,
            "drs": drs,
            "lap_distance_m": round(lap_distance, 2),
            "sector": sector,
            "corner_number": corner_number,
            "in_corner": in_corner,
        })

    return frames


# --------------------------------------------------------------------------- #
# 单赛道遥测生成
# --------------------------------------------------------------------------- #
def _generate_track_telemetry(
    track: Track,
    rng: random.Random,
) -> dict[str, Any]:
    """为单条赛道生成 10-20 圈完整遥测数据。

    Returns:
        含 track_id / laps / session_info / setups 的字典。
    """
    n_laps = rng.randint(10, 20)
    base_lap_time = _estimate_lap_time(track)

    # 生成 3-5 组不同调教快照
    setups = _generate_setups_for_track(track, rng)

    # 选一组作为本次 session 的调教
    active_setup = rng.choice(setups)

    # session UID（伪随机 64 位）
    session_uid = rng.getrandbits(64)
    session_time_start = time.time()

    laps: list[dict[str, Any]] = []
    total_distance = 0.0
    for lap_idx in range(n_laps):
        # 圈速：基准 + 噪声 + 趋势（前几圈慢，后几圈快）
        trend = -0.5 * (1.0 - lap_idx / max(1, n_laps - 1))
        lap_time = base_lap_time + trend + rng.uniform(-1.0, 1.0)
        lap_time = max(60.0, lap_time)

        # 扇区时间
        sector1 = lap_time * 0.33 + rng.uniform(-0.5, 0.5)
        sector2 = lap_time * 0.33 + rng.uniform(-0.5, 0.5)
        sector3 = lap_time - sector1 - sector2

        # 生成 60Hz 遥测帧
        frames = _generate_lap_telemetry(track, lap_time, active_setup, rng)

        # 最高速度
        max_speed = max(f["speed_kmh"] for f in frames)

        lap_data = {
            "lap_number": lap_idx + 1,
            "lap_time_sec": round(lap_time, 3),
            "sector1_sec": round(sector1, 3),
            "sector2_sec": round(sector2, 3),
            "sector3_sec": round(sector3, 3),
            "max_speed_kmh": round(max_speed, 2),
            "frames": frames,
            "is_valid": True,
        }
        laps.append(lap_data)
        total_distance += track.length_m

    # 构造 UDP 包流（每圈首帧 + 末帧的 4 类包）
    udp_packets = _build_udp_packet_stream(
        track, session_uid, session_time_start, laps, active_setup,
    )

    return {
        "track_id": track.track_id,
        "track_name": track.official_name,
        "session_uid": session_uid,
        "n_laps": n_laps,
        "active_setup": active_setup.to_dict(),
        "laps": laps,
        "udp_packets": udp_packets,
        "setups": [s.to_dict() for s in setups],
    }


def _generate_setups_for_track(track: Track, rng: random.Random) -> list[CarSetup]:
    """为赛道生成 3-5 组不同调教快照。

    基于赛道类型调整参数偏向（如高下压力赛道 → 高翼）。
    """
    n_setups = rng.randint(3, 5)
    setups: list[CarSetup] = []

    # 赛道类型 → 翼面偏向
    wing_bias = {
        "high_speed_low_downforce": (2, 4),  # 低翼
        "street":                   (6, 9),  # 高翼
        "high_downforce":           (7, 10), # 高翼
        "medium":                   (4, 7),
        "mixed":                    (5, 8),
    }.get(track.track_type, (4, 7))

    for _ in range(n_setups):
        defaults = CarSetup.default().to_dict()
        # 翼面随机
        defaults["front_wing"] = float(rng.randint(*wing_bias))
        defaults["rear_wing"] = float(rng.randint(*wing_bias))
        # 差速器
        defaults["on_throttle_diff"] = float(rng.randint(40, 70))
        defaults["off_throttle_diff"] = float(rng.randint(40, 70))
        # 刹车
        defaults["brake_bias"] = float(rng.randint(55, 70))
        defaults["brake_pressure"] = float(rng.randint(70, 85))
        # 悬挂高度
        defaults["front_ride_height"] = float(rng.randint(15, 25))
        defaults["rear_ride_height"] = float(rng.randint(20, 30))
        # 胎压
        defaults["front_tyre_pressure"] = round(rng.uniform(23.5, 26.5), 1)
        defaults["rear_tyre_pressure"] = round(rng.uniform(23.5, 26.5), 1)
        # 阻尼
        defaults["damping"] = float(rng.randint(200, 300))
        setups.append(CarSetup.from_dict(defaults))

    return setups


def _build_udp_packet_stream(
    track: Track,
    session_uid: int,
    session_time_start: float,
    laps: list[dict[str, Any]],
    setup: CarSetup,
) -> list[dict[str, Any]]:
    """为赛道构造 UDP 包流审计记录（每圈首帧 + 末帧的 4 类包）。

    仅记录包元数据 + 解析后字段（不存储原始 bytes，避免审计文件过大）。
    """
    packets: list[dict[str, Any]] = []
    frame_counter = 0

    # Session 包（每赛道 1 个）
    session_pkt = _build_session_packet(
        track, session_uid, 0.0, 0, len(laps),
    )
    packets.append({
        "packet_id": 1,
        "name": "Session",
        "frame_id": 0,
        "size_bytes": len(session_pkt),
        "m_trackId": track.udp_track_id,
        "m_trackLength": int(track.length_m),
        "m_totalLaps": len(laps),
    })

    for lap in laps:
        lap_time_ms = int(lap["lap_time_sec"] * 1000)
        sector1_ms = int(lap["sector1_sec"] * 1000)
        sector2_ms = int(lap["sector2_sec"] * 1000)
        max_speed = lap["max_speed_kmh"]

        # LapData 包（每圈 1 个，圈末）
        lap_pkt = _build_lap_packet(
            session_uid, lap["lap_time_sec"], frame_counter,
            last_lap_ms=lap_time_ms,
            current_lap_ms=lap_time_ms,
            sector1_ms=sector1_ms, sector2_ms=sector2_ms,
            lap_distance=track.length_m,
            total_distance=track.length_m * lap["lap_number"],
            current_lap_num=lap["lap_number"],
            sector=2,
            speed_trap_speed=max_speed,
        )
        packets.append({
            "packet_id": 2,
            "name": "LapData",
            "frame_id": frame_counter,
            "lap_number": lap["lap_number"],
            "size_bytes": len(lap_pkt),
            "m_lastLapTimeInMS": lap_time_ms,
            "m_sector1TimeInMS": sector1_ms,
            "m_sector2TimeInMS": sector2_ms,
            "m_speedTrapFastestSpeed": max_speed,
        })
        frame_counter += 1

        # Telemetry 包（取圈中 3 个采样点）
        for sample_idx in (0, len(lap["frames"]) // 2, len(lap["frames"]) - 1):
            f = lap["frames"][sample_idx]
            telem_pkt = _build_telemetry_packet(
                session_uid, f["progress"] * lap["lap_time_sec"], frame_counter,
                speed=f["speed_kmh"],
                throttle=f["throttle"],
                brake=f["brake"],
                steer=f["steer"],
                gear=f["gear"],
                rpm=f["engine_rpm"],
                drs=f["drs"],
            )
            packets.append({
                "packet_id": 6,
                "name": "CarTelemetry",
                "frame_id": frame_counter,
                "lap_number": lap["lap_number"],
                "size_bytes": len(telem_pkt),
                "m_speed": f["speed_kmh"],
                "m_throttle": f["throttle"],
                "m_brake": f["brake"],
                "m_gear": f["gear"],
                "m_engineRPM": f["engine_rpm"],
                "m_drs": f["drs"],
            })
            frame_counter += 1

    # CarSetups 包（每赛道 1 个）
    setup_pkt = _build_car_setups_packet(session_uid, 0.0, frame_counter, setup)
    packets.append({
        "packet_id": 5,
        "name": "CarSetups",
        "frame_id": frame_counter,
        "size_bytes": len(setup_pkt),
        "m_frontWing": int(setup.front_wing),
        "m_rearWing": int(setup.rear_wing),
        "m_brakeBias": int(setup.brake_bias),
        "m_brakePressure": int(setup.brake_pressure),
    })

    return packets


# --------------------------------------------------------------------------- #
# 数据库填充
# --------------------------------------------------------------------------- #
def _fill_database(store: Store, all_track_data: list[dict[str, Any]]) -> None:
    """将模拟数据写入 SQLite 数据库。

    填充表：
    - track / corner：赛道与弯道静态数据（24 条赛道）
    - setup：为每赛道写入 3-5 组调教快照
    - feedback：为每赛道生成 5-10 条模拟反馈
    - suggestion：为每赛道生成 1-3 条模拟建议报告
    - iteration：生成迭代历史
    """
    rng = random.Random(_RANDOM_SEED + 1)
    print("[DB] 开始填充数据库...")

    # ① 写入赛道与弯道静态数据
    for track in get_all_tracks():
        store.upsert_track(
            track_id=track.track_id,
            official_name=track.official_name,
            circuit_name=track.circuit_name,
            track_type=track.track_type,
            length_m=track.length_m,
            corners=len(track.corners),
            svg_path=track.svg_path,
            udp_track_id=track.udp_track_id,
        )
        for corner in track.corners:
            store.upsert_corner(
                track_id=track.track_id,
                corner_number=corner.number,
                corner_type=corner.corner_type,
                anchor_x=corner.anchor.anchor_x,
                anchor_y=corner.anchor.anchor_y,
                name=corner.name,
                speed_kmh=corner.speed_kmh,
            )
    print(f"[DB] 写入 {len(get_all_tracks())} 条赛道 + 弯道静态数据")

    for track_data in all_track_data:
        track_id = track_data["track_id"]
        setups_dicts = track_data["setups"]


        # ① 写入调教快照
        setup_ids: list[int] = []
        for setup_dict in setups_dicts:
            setup_id = store.import_setup(track_id, setup_dict)
            setup_ids.append(setup_id)
        print(f"  [{track_id}] 写入 {len(setup_ids)} 组调教快照")

        # ② 写入反馈（5-10 条）
        n_feedbacks = rng.randint(5, 10)
        track_obj = next(
            t for t in get_all_tracks() if t.track_id == track_id
        )
        for _ in range(n_feedbacks):
            # 随机选弯道（或全局）
            corner_number = rng.choice(
                [None] + [c.number for c in track_obj.corners],
            )
            # 随机选症状
            symptom = rng.choice(list(Symptom))
            category = get_symptom_category(symptom)
            strength = rng.randint(1, 5)
            setup_id = rng.choice(setup_ids)
            store.add_feedback(
                track_id=track_id,
                corner_number=corner_number,
                symptom=symptom.value,
                category=category,
                strength=strength,
                setup_id=setup_id,
            )
        print(f"  [{track_id}] 写入 {n_feedbacks} 条反馈")

        # ③ 写入建议报告（1-3 条）
        n_suggestions = rng.randint(1, 3)
        for _ in range(n_suggestions):
            # 用规则引擎生成建议
            symptoms = _pick_symptoms(rng)
            current_setup = rng.choice(setups_dicts)
            try:
                report = generate_suggestion(
                    symptoms=symptoms,
                    current_setup=current_setup,
                    track_id=track_id,
                    telemetry=None,
                )
                report_json = json.dumps(report, ensure_ascii=False)
                setup_id = rng.choice(setup_ids)
                store.save_suggestion(
                    track_id=track_id,
                    report_json=report_json,
                    setup_id=setup_id,
                )
            except Exception as exc:
                print(f"  [{track_id}] 生成建议失败: {exc}")
                continue
        print(f"  [{track_id}] 写入 {n_suggestions} 条建议")

        # ④ 写入迭代历史（2-4 轮）
        n_rounds = rng.randint(2, 4)
        for round_no in range(1, n_rounds + 1):
            before_id = rng.choice(setup_ids)
            after_id = rng.choice(setup_ids)
            store.save_iteration(
                track_id=track_id,
                round_no=round_no,
                before_setup_id=before_id,
                after_setup_id=after_id,
                suggestion_id=None,
            )
        print(f"  [{track_id}] 写入 {n_rounds} 轮迭代")


def _pick_symptoms(rng: random.Random) -> list[tuple[str, int]]:
    """随机选 1-3 个症状 + 强度，用于生成建议。"""
    n = rng.randint(1, 3)
    chosen = rng.sample(list(Symptom), n)
    return [(s.value, rng.randint(2, 5)) for s in chosen]


# --------------------------------------------------------------------------- #
# 训练数据集生成
# --------------------------------------------------------------------------- #
def _generate_training_dataset(
    all_track_data: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """生成训练数据集 data/training_dataset.json。

    每条样本：
        {
            "track_id": "suzuka",
            "symptoms": [{"symptom": "understeer", "strength": 3}],
            "current_setup": {...},
            "expected_delta": {...}  # 规则引擎输出
        }

    用规则引擎生成 expected_delta（因为暂时没有真实 F1 数据）。
    """
    print("[TRAIN] 生成训练数据集...")
    dataset: list[dict[str, Any]] = []

    for track_data in all_track_data:
        track_id = track_data["track_id"]
        setups = track_data["setups"]

        # 每赛道生成 5-10 条训练样本
        n_samples = rng.randint(5, 10)
        for _ in range(n_samples):
            symptoms = _pick_symptoms(rng)
            current_setup = rng.choice(setups)
            try:
                suggestion = generate_suggestion(
                    symptoms=symptoms,
                    current_setup=current_setup,
                    track_id=track_id,
                    telemetry=None,
                )
                expected_delta = suggestion["setup_delta"]
            except Exception as exc:
                print(f"  [{track_id}] 训练样本生成失败: {exc}")
                continue

            dataset.append({
                "track_id": track_id,
                "symptoms": [
                    {"symptom": s, "strength": st} for s, st in symptoms
                ],
                "current_setup": current_setup,
                "expected_delta": expected_delta,
            })

    print(f"[TRAIN] 共生成 {len(dataset)} 条训练样本")
    return dataset


# --------------------------------------------------------------------------- #
# 审计输出
# --------------------------------------------------------------------------- #
def _write_audit_packets(all_track_data: list[dict[str, Any]]) -> None:
    """将模拟 UDP 包流写入 data/sim_telemetry/ 供审计。"""
    _SIM_PACKETS_DIR.mkdir(parents=True, exist_ok=True)
    for track_data in all_track_data:
        track_id = track_data["track_id"]
        out_path = _SIM_PACKETS_DIR / f"sim_packets_{track_id}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for pkt in track_data["udp_packets"]:
                f.write(json.dumps(pkt, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def main() -> int:
    """主入口：生成模拟遥测数据 + 填充数据库 + 生成训练集。"""
    random.seed(_RANDOM_SEED)
    rng = random.Random(_RANDOM_SEED)

    print("=" * 70)
    print("F1OPT 模拟遥测数据生成器")
    print(f"随机种子: {_RANDOM_SEED}")
    print(f"数据库: {_DB_PATH}")
    print(f"训练集: {_TRAINING_DATASET_PATH}")
    print("=" * 70)

    # ① 加载 24 条赛道
    tracks = get_all_tracks()
    print(f"[INFO] 加载 {len(tracks)} 条赛道")

    # ② 为每条赛道生成遥测数据
    all_track_data: list[dict[str, Any]] = []
    for track in tracks:
        print(f"[GEN] 生成 {track.track_id} ({track.official_name})...")
        track_data = _generate_track_telemetry(track, rng)
        all_track_data.append(track_data)
        n_laps = track_data["n_laps"]
        n_frames = sum(len(lap["frames"]) for lap in track_data["laps"])
        n_pkts = len(track_data["udp_packets"])
        print(f"       圈数={n_laps}, 帧数={n_frames}, UDP包={n_pkts}")

    # ③ 填充数据库
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with Store(str(_DB_PATH)) as store:
        _fill_database(store, all_track_data)

    # ④ 生成训练数据集
    dataset = _generate_training_dataset(all_track_data, rng)
    _TRAINING_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _TRAINING_DATASET_PATH.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"[TRAIN] 训练集已写入: {_TRAINING_DATASET_PATH}")

    # ⑤ 审计输出（UDP 包流）
    _write_audit_packets(all_track_data)
    print(f"[AUDIT] UDP 包流已写入: {_SIM_PACKETS_DIR}")

    # ⑥ 汇总
    total_laps = sum(td["n_laps"] for td in all_track_data)
    total_frames = sum(
        sum(len(lap["frames"]) for lap in td["laps"])
        for td in all_track_data
    )
    total_pkts = sum(len(td["udp_packets"]) for td in all_track_data)
    db_size = _DB_PATH.stat().st_size

    print("=" * 70)
    print("生成完成！汇总：")
    print(f"  赛道数:     {len(all_track_data)}")
    print(f"  总圈数:     {total_laps}")
    print(f"  总帧数:     {total_frames}")
    print(f"  UDP 包数:   {total_pkts}")
    print(f"  训练样本:   {len(dataset)}")
    print(f"  数据库大小: {db_size} 字节 ({db_size / 1024:.1f} KB)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())