"""切片级深度测试共享 fixtures。

提供构造合法 UDP 包字节数据的 helper（参考 tests/test_t2_telemetry.py 的构造方式），
以及领域模型、引擎计算所需的常用 fixtures。所有 helper 均用 struct.pack 构造
真实字节数据，不 mock 被测模块本身。
"""

from __future__ import annotations

import struct
from collections.abc import Callable

import pytest

from setup_tuner.telemetry.packets import HEADER_FORMAT, NUM_CARS


# ===========================================================================
# UDP 包头构造
# ===========================================================================
def build_header(
    packet_id: int,
    player_car_index: int = 0,
    *,
    packet_format: int = 2026,
    game_year: int = 26,
    game_major: int = 1,
    game_minor: int = 0,
    packet_version: int = 1,
    session_uid: int = 0x1234_5678_9ABC_DEF0,
    session_time: float = 12.5,
    frame_id: int = 100,
    overall_frame_id: int = 200,
    secondary_player: int = 255,
) -> bytes:
    """构造 29 字节 F1 2026 UDP 包头（小端无填充）。

    Args:
        packet_id: 包类型 ID（0-16 等）。
        player_car_index: 玩家车在 24 车位数组中的下标。
        packet_format: 协议版本（默认 2026）。
        其余字段默认取易识别的魔术数。

    Returns:
        29 字节包头。
    """
    return struct.pack(
        HEADER_FORMAT,
        packet_format,
        game_year,
        game_major,
        game_minor,
        packet_version,
        packet_id,
        session_uid,
        session_time,
        frame_id,
        overall_frame_id,
        player_car_index,
        secondary_player,
    )


# ===========================================================================
# Packet 1 (Session) 包体构造
# ===========================================================================
def build_session_body(
    *,
    weather: int = 0,
    track_temp: int = 25,
    air_temp: int = 22,
    total_laps: int = 58,
    track_len: int = 5807,
    session_type: int = 0,
    track_id: int = 2,
    formula: int = 1,
    session_time_left: int = 1800,
    session_duration: int = 3600,
    pit_speed_limit: int = 60,
    num_wfs: int = 0,
) -> bytes:
    """构造 Packet 1 (Session) 包体（prefix + num_wfs 个天气样本）。"""
    prefix_core = struct.pack(
        "<BbbBHBbBHHBBBBBB",
        weather, track_temp, air_temp, total_laps, track_len,
        session_type, track_id, formula, session_time_left, session_duration,
        pit_speed_limit, 0, 0, 0, 0, 21,
    )
    marshal_zones = b"".join(struct.pack("<fb", 0.0, 0) for _ in range(21))
    tail = struct.pack("<BBB", 0, 0, num_wfs)
    body = prefix_core + marshal_zones + tail
    for _ in range(num_wfs):
        body += struct.pack("<BBBbbbbB", 0, 5, 1, 20, 0, 18, 0, 30)
    return body


# ===========================================================================
# Packet 2 (LapData) 单车段构造
# ===========================================================================
def build_lap_per_car(
    *,
    last_lap_ms: int = 90000,
    current_lap_ms: int = 45000,
    s1_ms_part: int = 30000,
    s1_min_part: int = 0,
    s2_ms_part: int = 25000,
    s2_min_part: int = 0,
    delta_front_ms_part: int = 1234,
    delta_front_min_part: int = 0,
    delta_leader_ms_part: int = 5678,
    delta_leader_min_part: int = 0,
    lap_dist: float = 1500.0,
    total_dist: float = 3000.0,
    sc_delta: float = 0.0,
    car_pos: int = 5,
    current_lap: int = 3,
    pit_status: int = 0,
    num_pits: int = 0,
    sector: int = 1,
    lap_invalid: int = 0,
    penalties: int = 0,
    pit_lane_time_ms: int = 0,
    pit_stop_timer_ms: int = 0,
    pit_should_serve: int = 0,
    speed_trap_speed: float = 320.0,
    speed_trap_lap: int = 0,
) -> bytes:
    """构造单辆车的 LapData 段（58 字节）。"""
    return struct.pack(
        "<IIHBHBHBHBfffBBBBBBBBBBBBBBBHHBfB",
        last_lap_ms, current_lap_ms,
        s1_ms_part, s1_min_part,
        s2_ms_part, s2_min_part,
        delta_front_ms_part, delta_front_min_part,
        delta_leader_ms_part, delta_leader_min_part,
        lap_dist, total_dist, sc_delta,
        car_pos, current_lap, pit_status, num_pits, sector,
        lap_invalid, penalties, 0, 0, 0, 0, 0, 0, 0, 0,
        pit_lane_time_ms, pit_stop_timer_ms, pit_should_serve,
        speed_trap_speed, speed_trap_lap,
    )


# ===========================================================================
# Packet 5 (CarSetups) 单车段构造
# ===========================================================================
def build_setup_per_car(
    *,
    front_wing: int = 5,
    rear_wing: int = 5,
    on_throttle: int = 50,
    off_throttle: int = 50,
    front_camber: float = -2.5,
    rear_camber: float = -2.5,
    front_toe: float = 0.25,
    rear_toe: float = 0.25,
    brake_pressure: int = 75,
    brake_bias: int = 65,
    front_left_press: float = 25.5,
    rear_left_press: float = 25.5,
    ballast: int = 50,
    fuel_load: float = 100.0,
) -> bytes:
    """构造单辆车的 CarSetups 段（50 字节）。"""
    return struct.pack(
        "<BBBBffffBBBBBBBBBffffBf",
        front_wing, rear_wing, on_throttle, off_throttle,
        front_camber, rear_camber, front_toe, rear_toe,
        1, 1, 1, 1, 1, 1, brake_pressure, brake_bias, 50,
        rear_left_press, 25.5, front_left_press, 25.5,
        ballast, fuel_load,
    )


# ===========================================================================
# Packet 6 (CarTelemetry) 单车段构造
# ===========================================================================
def build_telem_per_car(
    *,
    speed: int = 300,
    throttle: float = 0.8,
    steer: float = 0.0,
    brake: float = 0.2,
    clutch: int = 0,
    gear: int = 5,
    rpm: int = 11000,
    drs: int = 0,
    brakes_temp: tuple = (400, 450, 420, 430),
    tyres_surface_temp: tuple = (90, 92, 88, 89),
    tyres_inner_temp: tuple = (85, 87, 83, 84),
    engine_temp: int = 105,
    tyres_press: tuple = (25.5, 25.5, 26.0, 26.0),
    surface_type: tuple = (0, 0, 0, 0),
) -> bytes:
    """构造单辆车的 CarTelemetry 段（59 字节）。"""
    return struct.pack(
        "<HfffBbHBBH4H4B4BB4f4B",
        speed, throttle, steer, brake,
        clutch, gear, rpm, drs, 50, 0x00FF,
        *brakes_temp,
        *tyres_surface_temp,
        *tyres_inner_temp,
        engine_temp,
        *tyres_press,
        *surface_type,
    )


# ===========================================================================
# Packet 7 (CarStatus) 单车段构造
# ===========================================================================
def build_status_per_car(
    *,
    traction: int = 1,
    abs_on: int = 1,
    fuel_mix: int = 2,
    front_brake_bias: int = 55,
    pit_limiter: int = 0,
    fuel_in_tank: float = 50.0,
    fuel_capacity: float = 110.0,
    fuel_remaining_laps: float = 25.0,
    max_rpm: int = 13000,
    idle_rpm: int = 4000,
    max_gears: int = 8,
    drs_allowed: int = 1,
    actual_tyre: int = 5,
    visual_tyre: int = 5,
    tyres_age: int = 3,
    fia_flags: int = 0,
    ers_store: float = 1000000.0,
    ers_deploy_mode: int = 1,
    network_paused: int = 0,
) -> bytes:
    """构造单辆车的 CarStatus 段（59 字节）。"""
    return struct.pack(
        "<BBBBBfffHHBBHBBBbfffBffffB",
        traction, abs_on, fuel_mix, front_brake_bias, pit_limiter,
        fuel_in_tank, fuel_capacity, fuel_remaining_laps,
        max_rpm, idle_rpm,
        max_gears, drs_allowed,
        500,
        actual_tyre, visual_tyre,
        tyres_age,
        fia_flags,
        800.0, 60.0, ers_store,
        ers_deploy_mode,
        100000.0, 200000.0, 2000000.0, 500000.0,
        network_paused,
    )


# ===========================================================================
# Packet 16 (CarTelemetryData2) 单车段构造
# ===========================================================================
def build_ct2_per_car(
    *,
    aero_mode: int = 0,
    aero_available: int = 1,
    aero_distance: int = 100,
    overtake_available: int = 1,
    overtake_active: int = 0,
    overtake_distance: int = 200,
    reg_2026: int = 1,
    wrong_way: int = 0,
) -> bytes:
    """构造单辆车的 CarTelemetryData2 段（10 字节）。"""
    return struct.pack(
        "<BBHBBHBB",
        aero_mode, aero_available, aero_distance,
        overtake_available, overtake_active, overtake_distance,
        reg_2026, wrong_way,
    )


# ===========================================================================
# 完整包构造 fixtures
# ===========================================================================
@pytest.fixture
def make_session_packet() -> Callable[..., bytes]:
    """返回构造完整 Session 包（header + body）的工厂函数。"""
    def _make(
        *,
        packet_id: int = 1,
        player_car_index: int = 0,
        weather: int = 0,
        track_id: int = 2,
        num_wfs: int = 0,
    ) -> bytes:
        body = build_session_body(
            weather=weather, track_id=track_id, num_wfs=num_wfs,
        )
        return build_header(packet_id=packet_id, player_car_index=player_car_index) + body
    return _make


@pytest.fixture
def make_lap_data_packet() -> Callable[..., bytes]:
    """返回构造完整 LapData 包的工厂函数（仅含玩家车一段）。"""
    def _make(
        *,
        player_car_index: int = 0,
        last_lap_ms: int = 90000,
        car_pos: int = 5,
    ) -> bytes:
        per = build_lap_per_car(last_lap_ms=last_lap_ms, car_pos=car_pos)
        return build_header(packet_id=2, player_car_index=player_car_index) + per
    return _make


@pytest.fixture
def make_car_setups_packet() -> Callable[..., bytes]:
    """返回构造完整 CarSetups 包的工厂函数。"""
    def _make(
        *,
        player_car_index: int = 0,
        front_wing: int = 5,
    ) -> bytes:
        per = build_setup_per_car(front_wing=front_wing)
        return build_header(packet_id=5, player_car_index=player_car_index) + per
    return _make


@pytest.fixture
def make_car_telemetry_packet() -> Callable[..., bytes]:
    """返回构造完整 CarTelemetry 包的工厂函数。"""
    def _make(
        *,
        player_car_index: int = 0,
        speed: int = 300,
        gear: int = 5,
    ) -> bytes:
        per = build_telem_per_car(speed=speed, gear=gear)
        return build_header(packet_id=6, player_car_index=player_car_index) + per
    return _make


@pytest.fixture
def make_car_status_packet() -> Callable[..., bytes]:
    """返回构造完整 CarStatus 包的工厂函数。"""
    def _make(
        *,
        player_car_index: int = 0,
        tyres_age: int = 3,
    ) -> bytes:
        per = build_status_per_car(tyres_age=tyres_age)
        return build_header(packet_id=7, player_car_index=player_car_index) + per
    return _make


@pytest.fixture
def make_ct2_packet() -> Callable[..., bytes]:
    """返回构造完整 CarTelemetryData2 包的工厂函数。"""
    def _make(
        *,
        player_car_index: int = 0,
        aero_mode: int = 0,
    ) -> bytes:
        per = build_ct2_per_car(aero_mode=aero_mode)
        return build_header(packet_id=16, player_car_index=player_car_index) + per
    return _make


# ===========================================================================
# 引擎测试常用 fixtures
# ===========================================================================
@pytest.fixture
def default_setup_dict() -> dict[str, float]:
    """返回全部 23 参数取缺省值的调教字典。"""
    from setup_tuner.domain.setup import CarSetup
    return CarSetup.default().to_dict()


@pytest.fixture
def all_symptoms_with_strength() -> list[tuple[str, int]]:
    """返回全部 12 症状（强度 3）的列表。"""
    from setup_tuner.domain.symptoms import Symptom
    return [(s.value, 3) for s in Symptom]


# ===========================================================================
# 切片 6-9 共享夹具：Store / 业务服务 / 真实 API 客户端
# ===========================================================================
from collections.abc import Iterator  # noqa: E402  (放此处便于阅读)


@pytest.fixture
def store() -> Iterator[Store]:  # type: ignore[name-defined]  # noqa: F821
    """每个测试用独立的内存 SQLite 库（快速隔离）。

    用 yield 确保连接关闭；用 ``:memory:`` 避免磁盘 I/O。
    """
    from setup_tuner.db.store import Store

    s = Store(":memory:")
    yield s
    s.close()


@pytest.fixture
def store_file(tmp_path: Path) -> Iterator[Store]:  # type: ignore[name-defined]  # noqa: F821
    """基于临时文件的 SQLite Store（验证真实文件持久化与 WAL 模式）。

    用 tmp_path fixture 确保每个测试用例独立 db 文件，自动清理。
    """
    from setup_tuner.db.store import Store

    db_path = tmp_path / "test_deep_slice.db"
    s = Store(str(db_path))
    yield s
    s.close()


@pytest.fixture
def feedback_service(store: Store) -> FeedbackService:  # type: ignore[name-defined]  # noqa: F821
    """反馈服务（注入内存 store）。"""
    from setup_tuner.feedback.service import FeedbackService

    return FeedbackService(store)


@pytest.fixture
def iteration_service(store: Store) -> IterationService:  # type: ignore[name-defined]  # noqa: F821
    """迭代服务（注入内存 store）。"""
    from setup_tuner.feedback.iteration import IterationService

    return IterationService(store)


@pytest.fixture
def telemetry_stream() -> TelemetryStream:  # type: ignore[name-defined]  # noqa: F821
    """遥测帧缓存（空）。"""
    from setup_tuner.telemetry.stream import TelemetryStream

    return TelemetryStream()


@pytest.fixture
def sample_params() -> dict[str, float]:
    """23 项调教参数样例（取 CarSetup.default）。"""
    from setup_tuner.domain.setup import CarSetup

    return CarSetup.default().to_dict()


@pytest.fixture
def sample_packet5() -> dict[str, float | int | list[float]]:
    """模拟 UDP Packet 5（CarSetups）解析后字典。"""
    return {
        "m_frontWing": 6.0,
        "m_rearWing": 4.0,
        "m_onThrottleDiff": 55.0,
        "m_offThrottleDiff": 45.0,
        "m_frontCamber": -2.0,
        "m_rearCamber": -2.2,
        "m_frontToe": 0.20,
        "m_rearToe": 0.30,
        "m_frontSuspension": 260.0,
        "m_rearSuspension": 240.0,
        "m_frontAntiRollBar": 240.0,
        "m_rearAntiRollBar": 260.0,
        "m_frontSuspensionHeight": 21.0,
        "m_rearSuspensionHeight": 19.0,
        "m_brakePressure": 80.0,
        "m_brakeBias": 63.0,
        "m_engineBraking": 48.0,
        "m_ballast": 52.0,
        "m_activeAeroMode": 0,  # 0=Z/弯道模式
        "tyresPressure": [25.5, 25.6, 25.4, 25.5],
    }


@pytest.fixture
def app_client(tmp_path: Path) -> Iterator[TestClient]:  # type: ignore[name-defined]  # noqa: F821
    """真实 FastAPI TestClient（启动 lifespan，初始化 Store/服务）。

    数据目录指向 tmp_path 子目录，确保测试隔离且不污染工作区。
    with 语句触发 lifespan startup/shutdown，确保 Store/Listener 正确初始化。
    """
    from fastapi.testclient import TestClient

    from setup_tuner.app import create_app
    from setup_tuner.config import Config

    config = Config(data_dir=str(tmp_path / "data_deep_slice"))
    app = create_app(config)
    with TestClient(app) as client:
        yield client


# ===========================================================================
# 常量暴露（供测试模块引用）
# ===========================================================================
__all__ = [
    "NUM_CARS",
    "build_header",
    "build_session_body",
    "build_lap_per_car",
    "build_setup_per_car",
    "build_telem_per_car",
    "build_status_per_car",
    "build_ct2_per_car",
    # 切片 6-9 夹具
    "store",
    "store_file",
    "feedback_service",
    "iteration_service",
    "telemetry_stream",
    "sample_params",
    "sample_packet5",
    "app_client",
]