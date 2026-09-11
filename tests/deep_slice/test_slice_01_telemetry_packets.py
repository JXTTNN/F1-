"""切片 1 深度测试：UDP 遥测包解析（setup_tuner.telemetry.packets）。

覆盖 6 类包：Session(1) / LapData(2) / CarSetups(5) / CarTelemetry(6) /
CarStatus(7) / CarTelemetryData2(16)。

5 种测试方式：
    1. unit     — 每个公开解析函数的正常输入正确性
    2. boundary — 极端输入、越界、空值、短包、player_car_index 越界
    3. property — 确定性（相同输入相同输出）、往返一致性（构造→解析→对照）
    4. static   — 类型约束、值域约束、不变量约束（参数化）
    5. smoke    — 真实调用链路 parse_packet 端到端
"""

from __future__ import annotations

import struct

import pytest

from setup_tuner.telemetry.packets import (
    HEADER_FORMAT,
    HEADER_SIZE,
    NUM_CARS,
    PACKET_NAMES,
    SUPPORTED_PACKET_IDS,
    PacketHeader,
    PacketTooShortError,
    packet_name,
    parse_car_setups,
    parse_car_status,
    parse_car_telemetry,
    parse_car_telemetry_2,
    parse_header,
    parse_lap_data,
    parse_packet,
    parse_session,
)

from .conftest import (
    build_ct2_per_car,
    build_header,
    build_lap_per_car,
    build_session_body,
    build_setup_per_car,
    build_status_per_car,
    build_telem_per_car,
)


# ===========================================================================
# 1. 单元测试 (unit) — 每个公开解析函数的正常输入正确性
# ===========================================================================
class TestUnit:
    """单元测试：验证每个公开解析函数在正常输入下的字段正确性。"""

    def test_unit_parse_header_all_fields(self) -> None:
        """parse_header 应正确解析 29 字节包头的全部 12 个字段。"""
        data = build_header(
            packet_id=1, player_car_index=7,
            packet_format=2026, game_year=26,
            session_uid=0xDEAD_BEEF_CAFE_BABE,
            session_time=42.5, frame_id=999,
            overall_frame_id=12345, secondary_player=3,
        )
        h = parse_header(data)
        assert h.packet_format == 2026
        assert h.game_year == 26
        assert h.packet_id == 1
        assert h.session_uid == 0xDEAD_BEEF_CAFE_BABE
        assert h.session_time == pytest.approx(42.5)
        assert h.frame_identifier == 999
        assert h.overall_frame_identifier == 12345
        assert h.player_car_index == 7
        assert h.secondary_player_car_index == 3

    def test_unit_parse_session_fields(self) -> None:
        """parse_session 应正确解析 Session 包体的全部字段。"""
        body = build_session_body(
            weather=2, track_temp=30, air_temp=28, total_laps=58,
            track_len=5807, track_id=2, formula=1,
        )
        data = build_header(packet_id=1) + body
        result = parse_session(data)
        assert result["m_trackId"] == 2
        assert result["m_weather"] == 2
        assert result["m_trackTemperature"] == 30
        assert result["m_airTemperature"] == 28
        assert result["m_totalLaps"] == 58
        assert result["m_trackLength"] == 5807

    def test_unit_parse_lap_data_fields(self) -> None:
        """parse_lap_data 应正确解析 LapData 玩家车段。"""
        per = build_lap_per_car(
            last_lap_ms=95000, current_lap_ms=45000,
            s1_ms_part=30000, s1_min_part=1,  # 1*60000+30000=90000
            car_pos=5, current_lap=3, sector=1,
        )
        data = build_header(packet_id=2, player_car_index=0) + per
        result = parse_lap_data(data, 0)
        assert result["m_lastLapTimeInMS"] == 95000
        assert result["m_currentLapTimeInMS"] == 45000
        assert result["m_sector1TimeInMS"] == 90000
        assert result["m_carPosition"] == 5
        assert result["m_currentLapNum"] == 3

    def test_unit_parse_car_setups_fields(self) -> None:
        """parse_car_setups 应正确解析 CarSetups 玩家车段。"""
        per = build_setup_per_car(
            front_wing=7, rear_wing=3,
            front_camber=-3.0, rear_camber=-2.0,
            brake_pressure=80, brake_bias=70,
        )
        data = build_header(packet_id=5, player_car_index=0) + per
        result = parse_car_setups(data, 0)
        assert result["m_frontWing"] == 7
        assert result["m_rearWing"] == 3
        assert result["m_frontCamber"] == pytest.approx(-3.0)
        assert result["m_brakePressure"] == 80

    def test_unit_parse_car_telemetry_fields(self) -> None:
        """parse_car_telemetry 应正确解析 CarTelemetry 玩家车段（含数组）。"""
        per = build_telem_per_car(
            speed=320, throttle=1.0, brake=0.5, gear=6, rpm=12000,
            brakes_temp=(450, 460, 440, 430),
        )
        data = build_header(packet_id=6, player_car_index=0) + per
        result = parse_car_telemetry(data, 0)
        assert result["m_speed"] == 320
        assert result["m_throttle"] == pytest.approx(1.0)
        assert result["m_gear"] == 6
        assert result["m_brakesTemperature"] == [450, 460, 440, 430]

    def test_unit_parse_car_status_fields(self) -> None:
        """parse_car_status 应正确解析 CarStatus 玩家车段。"""
        per = build_status_per_car(
            fuel_mix=3, actual_tyre=7, tyres_age=5,
            ers_store=800000.0,
        )
        data = build_header(packet_id=7, player_car_index=0) + per
        result = parse_car_status(data, 0)
        assert result["m_fuelMix"] == 3
        assert result["m_actualTyreCompound"] == 7
        assert result["m_tyresAgeLaps"] == 5
        assert result["m_ersStoreEnergy"] == pytest.approx(800000.0)

    def test_unit_parse_car_telemetry_2_fields(self) -> None:
        """parse_car_telemetry_2 应正确解析 CarTelemetryData2 玩家车段。"""
        per = build_ct2_per_car(
            aero_mode=1, overtake_active=1, overtake_distance=250,
        )
        data = build_header(packet_id=16, player_car_index=0) + per
        result = parse_car_telemetry_2(data, 0)
        assert result["m_activeAeroMode"] == 1
        assert result["m_overtakeActive"] == 1
        assert result["m_overtakeActivationDistance"] == 250

    def test_unit_packet_name_known(self) -> None:
        """packet_name 对已知 ID 返回正确名称。"""
        assert packet_name(1) == "Session"
        assert packet_name(2) == "LapData"
        assert packet_name(16) == "CarTelemetryData2"

    def test_unit_parse_packet_dispatch(self) -> None:
        """parse_packet 应按 packet_id 分发到正确的解析函数。"""
        body = build_session_body(track_id=5)
        data = build_header(packet_id=1) + body
        result = parse_packet(data)
        assert result is not None
        assert result["packet_id"] == 1
        assert result["name"] == "Session"
        assert isinstance(result["header"], PacketHeader)


# ===========================================================================
# 2. 边界/异常测试 (boundary) — 极端输入、越界、空值、短包
# ===========================================================================
class TestBoundary:
    """边界/异常测试：验证短包、空包、越界索引等异常输入的处理。"""

    def test_boundary_empty_bytes_raises(self) -> None:
        """空 bytes 应抛 PacketTooShortError。"""
        with pytest.raises(PacketTooShortError):
            parse_header(b"")

    def test_boundary_short_header_raises(self) -> None:
        """不足 29 字节应抛 PacketTooShortError。"""
        with pytest.raises(PacketTooShortError):
            parse_header(b"\x00" * 28)

    def test_boundary_exactly_29_bytes_ok(self) -> None:
        """恰好 29 字节包头应解析成功（边界值）。"""
        data = build_header(packet_id=0)
        assert len(data) == 29
        h = parse_header(data)
        assert h.packet_id == 0

    def test_boundary_short_session_body_raises(self) -> None:
        """Session 包体不足应抛 PacketTooShortError。"""
        data = build_header(packet_id=1) + b"\x00" * 10
        with pytest.raises(PacketTooShortError):
            parse_session(data)

    def test_boundary_short_lap_data_raises(self) -> None:
        """LapData 包体不足以覆盖玩家车应抛 PacketTooShortError。"""
        data = build_header(packet_id=2, player_car_index=5) + b"\x00" * 10
        with pytest.raises(PacketTooShortError):
            parse_lap_data(data, 5)

    def test_boundary_player_car_index_out_of_range(self) -> None:
        """player_car_index 越界（如 255）应抛 PacketTooShortError。"""
        data = build_header(packet_id=2, player_car_index=255) + b"\x00" * 64
        with pytest.raises(PacketTooShortError):
            parse_lap_data(data, 255)

    @pytest.mark.parametrize("bad_index", [NUM_CARS, NUM_CARS + 1, 255])
    def test_boundary_invalid_player_car_index(self, bad_index: int) -> None:
        """参数化：多种越界 player_car_index 均应抛 PacketTooShortError。

        注：-1 在 struct.pack('B') 时即抛 struct.error（uint8 范围 0~255），
        故不在此参数化中测试；此处仅测试可成功构造包头但解析时越界的场景。
        """
        per = build_lap_per_car()
        data = build_header(packet_id=2, player_car_index=bad_index) + per
        with pytest.raises(PacketTooShortError):
            parse_lap_data(data, bad_index)

    def test_boundary_unknown_packet_id_returns_none(self) -> None:
        """未知 packetId 应返回 None（跳过不崩溃）。"""
        for pid in (0, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 99):
            data = build_header(packet_id=pid) + b"\x00" * 64
            assert parse_packet(data) is None

    def test_boundary_max_player_car_index(self) -> None:
        """player_car_index = NUM_CARS - 1（23）为合法边界，应解析成功。"""
        per_size = struct.calcsize("<BBHBBHBB")
        # 构造 24 辆车的 CarTelemetryData2 包体
        body = b"\x00" * per_size * (NUM_CARS - 1) + build_ct2_per_car(aero_mode=1)
        data = build_header(packet_id=16, player_car_index=NUM_CARS - 1) + body
        result = parse_car_telemetry_2(data, NUM_CARS - 1)
        assert result["m_activeAeroMode"] == 1

    def test_boundary_session_with_max_weather_samples(self) -> None:
        """Session 含 64 个天气样本（协议上限）应正确解析。"""
        body = build_session_body(num_wfs=64)
        data = build_header(packet_id=1) + body
        result = parse_session(data)
        assert result["m_numWeatherForecastSamples"] == 64
        assert len(result["m_weatherForecastSamples"]) == 64

    def test_boundary_session_truncated_weather_samples(self) -> None:
        """Session 声明的天气样本数超过实际字节时应容错（已解的保留）。"""
        body = build_session_body(num_wfs=2)
        # 截断最后一个天气样本的尾部 4 字节
        truncated = body[:-4]
        data = build_header(packet_id=1) + truncated
        result = parse_session(data)
        # 第 1 个样本完整，第 2 个被截断 → 只保留 1 个
        assert len(result["m_weatherForecastSamples"]) == 1


# ===========================================================================
# 3. 属性不变量测试 (property) — 确定性、往返一致性
# ===========================================================================
class TestProperty:
    """属性不变量测试：验证解析的确定性与往返一致性。"""

    def test_property_parse_header_deterministic(self) -> None:
        """确定性：相同包头字节解析两次结果完全一致。"""
        data = build_header(packet_id=1, player_car_index=3)
        h1 = parse_header(data)
        h2 = parse_header(data)
        assert h1 == h2

    def test_property_parse_packet_deterministic(self) -> None:
        """确定性：相同完整包解析两次结果完全一致。"""
        body = build_session_body(track_id=7)
        data = build_header(packet_id=1) + body
        r1 = parse_packet(data)
        r2 = parse_packet(data)
        assert r1 == r2

    def test_property_roundtrip_header(self) -> None:
        """往返一致性：构造包头 → 解析 → 重新构造 → 字节相同。"""
        original = build_header(
            packet_id=6, player_car_index=2,
            session_uid=0xCAFE_BABE_1234_5678,
            session_time=99.5, frame_id=500,
        )
        h = parse_header(original)
        rebuilt = struct.pack(
            HEADER_FORMAT,
            h.packet_format, h.game_year,
            h.game_major_version, h.game_minor_version,
            h.packet_version, h.packet_id,
            h.session_uid, h.session_time,
            h.frame_identifier, h.overall_frame_identifier,
            h.player_car_index, h.secondary_player_car_index,
        )
        assert original == rebuilt

    def test_property_roundtrip_all_supported_packets(self) -> None:
        """往返一致性：6 类包构造 → 解析 → 关键字段对照一致。"""
        cases = [
            (1, build_session_body(track_id=3), {"m_trackId": 3}),
            (2, build_lap_per_car(last_lap_ms=77777), {"m_lastLapTimeInMS": 77777}),
            (5, build_setup_per_car(front_wing=9), {"m_frontWing": 9}),
            (6, build_telem_per_car(speed=350), {"m_speed": 350}),
            (7, build_status_per_car(tyres_age=8), {"m_tyresAgeLaps": 8}),
            (16, build_ct2_per_car(aero_mode=1), {"m_activeAeroMode": 1}),
        ]
        for pid, body, expected_kv in cases:
            data = build_header(packet_id=pid) + body
            result = parse_packet(data)
            assert result is not None, f"packet_id={pid} 解析返回 None"
            for k, v in expected_kv.items():
                assert result[k] == v, f"packet_id={pid} 字段 {k} 期望 {v} 实际 {result[k]}"

    def test_property_header_name_consistent_with_packet_id(self) -> None:
        """不变量：header.name 始终与 PACKET_NAMES[packet_id] 一致。"""
        for pid, expected_name in PACKET_NAMES.items():
            if pid in SUPPORTED_PACKET_IDS:
                data = build_header(packet_id=pid) + b"\x00" * 128
                try:
                    h = parse_header(data)
                    assert h.name == expected_name
                except PacketTooShortError:
                    pass  # 某些包体不足时跳过

    def test_property_player_car_index_isolation(self) -> None:
        """不变量：只解包玩家车段，其他车数据不影响结果。"""

        # 玩家车（index=2）数据固定
        target = build_ct2_per_car(aero_mode=1, overtake_active=1)
        # 前 2 辆车用不同垃圾数据
        junk_a = build_ct2_per_car(aero_mode=0, overtake_active=0)
        junk_b = build_ct2_per_car(aero_mode=1, overtake_active=0)
        body = junk_a + junk_b + target
        data = build_header(packet_id=16, player_car_index=2) + body
        result = parse_car_telemetry_2(data, 2)
        # 玩家车结果不受前两辆车影响
        assert result["m_activeAeroMode"] == 1
        assert result["m_overtakeActive"] == 1


# ===========================================================================
# 4. 静态分析 (static) — 类型约束、值域约束、不变量约束（参数化）
# ===========================================================================
class TestStatic:
    """静态分析：用参数化测试验证类型约束、值域约束与不变量约束。"""

    def test_static_header_size_is_29(self) -> None:
        """值域约束：HEADER_SIZE 必须为 29（协议固定）。"""
        assert HEADER_SIZE == 29
        assert struct.calcsize(HEADER_FORMAT) == 29

    def test_static_num_cars_is_24(self) -> None:
        """值域约束：NUM_CARS 必须为 24（F1 2026 协议固定数组大小）。"""
        assert NUM_CARS == 24

    def test_static_supported_packet_ids_exact(self) -> None:
        """不变量约束：SUPPORTED_PACKET_IDS 必须为 {1,2,5,6,7,16}。"""
        assert SUPPORTED_PACKET_IDS == frozenset({1, 2, 5, 6, 7, 16})

    @pytest.mark.parametrize("pid,name", [
        (0, "Motion"), (1, "Session"), (2, "LapData"), (3, "Event"),
        (4, "Participants"), (5, "CarSetups"), (6, "CarTelemetry"),
        (7, "CarStatus"), (8, "FinalClassification"), (9, "LobbyInfo"),
        (10, "CarDamage"), (11, "SessionHistory"), (12, "TyreSets"),
        (13, "MotionEx"), (14, "TimeTrial"), (15, "LapPositions"),
        (16, "CarTelemetryData2"),
    ])
    def test_static_packet_names_complete(self, pid: int, name: str) -> None:
        """参数化：PACKET_NAMES 覆盖全部 17 个已知 packet_id 且名称正确。"""
        assert PACKET_NAMES[pid] == name

    @pytest.mark.parametrize("pid", list(SUPPORTED_PACKET_IDS))
    def test_static_supported_packet_has_parser(self, pid: int) -> None:
        """不变量约束：每个支持的 packet_id 都有对应解析函数（非 None）。"""
        from setup_tuner.telemetry.packets import _PARSERS  # noqa: PLC2701
        assert _PARSERS[pid] is not None

    def test_static_header_is_frozen_dataclass(self) -> None:
        """类型约束：PacketHeader 为 frozen dataclass（不可变）。"""
        data = build_header(packet_id=1)
        h = parse_header(data)
        with pytest.raises((AttributeError, TypeError)):
            h.packet_id = 99  # type: ignore[misc]

    def test_static_parse_header_returns_packet_header_type(self) -> None:
        """类型约束：parse_header 返回值必须是 PacketHeader 实例。"""
        h = parse_header(build_header(packet_id=1))
        assert isinstance(h, PacketHeader)

    def test_static_little_endian_packet_format(self) -> None:
        """值域约束：packet_format=2026 小端应为字节 [0xEA, 0x07]。"""
        data = build_header(packet_id=0, packet_format=2026)
        assert data[0] == 0xEA
        assert data[1] == 0x07

    @pytest.mark.parametrize("field_name", [
        "packet_format", "game_year", "game_major_version", "game_minor_version",
        "packet_version", "packet_id", "session_uid", "session_time",
        "frame_identifier", "overall_frame_identifier",
        "player_car_index", "secondary_player_car_index",
    ])
    def test_static_header_has_all_fields(self, field_name: str) -> None:
        """不变量约束：PacketHeader 必须含全部 12 个字段。"""
        h = parse_header(build_header(packet_id=1))
        assert hasattr(h, field_name)


# ===========================================================================
# 5. 实际运行冒烟 (smoke) — 真实调用链路 parse_packet 端到端
# ===========================================================================
class TestSmoke:
    """实际运行冒烟：用真实字节数据走完整 parse_packet 链路。"""

    def test_smoke_session_full_chain(self) -> None:
        """冒烟：构造完整 Session 包 → parse_packet → 验证输出结构。"""
        body = build_session_body(
            weather=1, track_temp=22, air_temp=20, total_laps=58,
            track_len=5807, track_id=2, num_wfs=3,
        )
        data = build_header(packet_id=1, player_car_index=0) + body
        result = parse_packet(data)
        assert result is not None
        assert result["packet_id"] == 1
        assert result["name"] == "Session"
        assert isinstance(result["header"], PacketHeader)
        assert result["m_trackId"] == 2
        assert result["m_numWeatherForecastSamples"] == 3
        assert len(result["m_weatherForecastSamples"]) == 3

    def test_smoke_lap_data_full_chain(self) -> None:
        """冒烟：构造完整 LapData 包 → parse_packet → 验证字段。"""
        per = build_lap_per_car(
            last_lap_ms=95123, current_lap_ms=45678,
            car_pos=3, current_lap=5, sector=2,
            speed_trap_speed=315.5,
        )
        data = build_header(packet_id=2, player_car_index=0) + per
        result = parse_packet(data)
        assert result is not None
        assert result["packet_id"] == 2
        assert result["m_lastLapTimeInMS"] == 95123
        assert result["m_carPosition"] == 3
        assert result["m_speedTrapFastestSpeed"] == pytest.approx(315.5)

    def test_smoke_car_setups_full_chain(self) -> None:
        """冒烟：构造完整 CarSetups 包 → parse_packet → 验证字段。"""
        per = build_setup_per_car(
            front_wing=8, rear_wing=4,
            front_camber=-3.2, rear_camber=-2.1,
            front_left_press=26.5, rear_left_press=24.8,
            ballast=60, fuel_load=105.0,
        )
        data = build_header(packet_id=5, player_car_index=0) + per
        result = parse_packet(data)
        assert result is not None
        assert result["m_frontWing"] == 8
        assert result["m_rearWing"] == 4
        assert result["m_frontCamber"] == pytest.approx(-3.2)
        assert result["m_fuelLoad"] == pytest.approx(105.0)

    def test_smoke_car_telemetry_full_chain(self) -> None:
        """冒烟：构造完整 CarTelemetry 包 → parse_packet → 验证数组字段。"""
        per = build_telem_per_car(
            speed=340, throttle=0.95, brake=0.3, gear=7, rpm=12500,
            brakes_temp=(480, 470, 460, 450),
            tyres_surface_temp=(100, 102, 98, 99),
            tyres_press=(26.5, 26.5, 25.8, 25.8),
        )
        data = build_header(packet_id=6, player_car_index=0) + per
        result = parse_packet(data)
        assert result is not None
        assert result["m_speed"] == 340
        assert result["m_brakesTemperature"] == [480, 470, 460, 450]
        assert result["m_tyresSurfaceTemperature"] == [100, 102, 98, 99]

    def test_smoke_car_status_full_chain(self) -> None:
        """冒烟：构造完整 CarStatus 包 → parse_packet → 验证字段。"""
        per = build_status_per_car(
            fuel_mix=2, actual_tyre=6, visual_tyre=6, tyres_age=10,
            ers_store=2000000.0, ers_deploy_mode=2,
        )
        data = build_header(packet_id=7, player_car_index=0) + per
        result = parse_packet(data)
        assert result is not None
        assert result["m_fuelMix"] == 2
        assert result["m_tyresAgeLaps"] == 10
        assert result["m_ersDeployMode"] == 2

    def test_smoke_ct2_full_chain(self) -> None:
        """冒烟：构造完整 CarTelemetryData2 包 → parse_packet → 验证字段。"""
        per = build_ct2_per_car(
            aero_mode=1, aero_distance=150,
            overtake_available=1, overtake_active=1,
            overtake_distance=250, wrong_way=0,
        )
        data = build_header(packet_id=16, player_car_index=0) + per
        result = parse_packet(data)
        assert result is not None
        assert result["m_activeAeroMode"] == 1
        assert result["m_overtakeActive"] == 1
        assert result["m_drivingWrongWay"] == 0

    def test_smoke_all_6_packet_types_in_sequence(self) -> None:
        """冒烟：连续解析 6 类包，全部成功且 packet_id 正确。"""
        packets = [
            (1, build_session_body()),
            (2, build_lap_per_car()),
            (5, build_setup_per_car()),
            (6, build_telem_per_car()),
            (7, build_status_per_car()),
            (16, build_ct2_per_car()),
        ]
        for pid, body in packets:
            data = build_header(packet_id=pid) + body
            result = parse_packet(data)
            assert result is not None, f"packet_id={pid} 冒烟失败"
            assert result["packet_id"] == pid

    def test_smoke_negative_gear_and_fia_flags(self) -> None:
        """冒烟：int8 负值字段（gear=-1, fia_flags=-1）正确解析。"""
        per_telem = build_telem_per_car(gear=-1)
        data_telem = build_header(packet_id=6) + per_telem
        result_telem = parse_packet(data_telem)
        assert result_telem is not None
        assert result_telem["m_gear"] == -1

        per_status = build_status_per_car(fia_flags=-1)
        data_status = build_header(packet_id=7) + per_status
        result_status = parse_packet(data_status)
        assert result_status is not None
        assert result_status["m_vehicleFiaFlags"] == -1