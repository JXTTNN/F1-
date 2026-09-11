"""T2 遥测解析单元测试。

覆盖验收标准：
1. 29 字节 header 解析（字段偏移/大小端正确）
2. 6 类包解析（Session/LapData/CarSetups/CarTelemetry/CarStatus/CarTelemetry2）
3. 短包抛 PacketTooShortError
4. 未知 packetId 返回 None
5. playerCarIndex 索引正确（只解包玩家车那一段）
6. 用构造的 bytes 样本逐字段断言

所有样本均由 struct.pack 构造，字段值选取易识别的魔术数便于断言。
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
    parse_header,
    parse_packet,
)


# ===========================================================================
# 辅助：构造包头与包体
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
    """构造 29 字节包头（小端）。"""
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
# 1. Header 解析
# ===========================================================================
class TestHeader:
    """29 字节包头解析（字段偏移/大小端）。"""

    def test_header_size_is_29(self) -> None:
        """HEADER_SIZE 必须为 29（协议固定）。"""
        assert HEADER_SIZE == 29
        assert struct.calcsize(HEADER_FORMAT) == 29

    def test_parse_header_all_fields(self) -> None:
        """逐字段断言 header 解析与构造值一致（含大小端）。"""
        data = build_header(
            packet_id=1,
            player_car_index=7,
            packet_format=2026,
            game_year=26,
            game_major=1,
            game_minor=0,
            packet_version=1,
            session_uid=0xDEAD_BEEF_CAFE_BABE,
            session_time=42.5,
            frame_id=999,
            overall_frame_id=12345,
            secondary_player=3,
        )
        h = parse_header(data)
        assert isinstance(h, PacketHeader)
        assert h.packet_format == 2026
        assert h.game_year == 26
        assert h.game_major_version == 1
        assert h.game_minor_version == 0
        assert h.packet_version == 1
        assert h.packet_id == 1
        assert h.session_uid == 0xDEAD_BEEF_CAFE_BABE
        assert h.session_time == pytest.approx(42.5)
        assert h.frame_identifier == 999
        assert h.overall_frame_identifier == 12345
        assert h.player_car_index == 7
        assert h.secondary_player_car_index == 3

    def test_parse_header_little_endian(self) -> None:
        """uint16 packet_format=2026 小端应为字节 [0xEA, 0x07]。"""
        data = build_header(packet_id=0, packet_format=2026)
        # 2026 = 0x07EA → 小端低字节在前
        assert data[0] == 0xEA
        assert data[1] == 0x07

    def test_parse_header_session_uid_offset(self) -> None:
        """session_uid 起始于偏移 7（H+B*5=2+5=7），8 字节小端。"""
        uid = 0x0102_0304_0506_0708
        data = build_header(packet_id=0, session_uid=uid)
        # 偏移 7~14 为 session_uid 小端
        uid_bytes = data[7:15]
        assert uid_bytes == struct.pack("<Q", uid)

    def test_header_name_property(self) -> None:
        """header.name 应返回 packet_id 对应的人类可读名称。"""
        assert parse_header(build_header(packet_id=1)).name == "Session"
        assert parse_header(build_header(packet_id=2)).name == "LapData"
        assert parse_header(build_header(packet_id=5)).name == "CarSetups"
        assert parse_header(build_header(packet_id=6)).name == "CarTelemetry"
        assert parse_header(build_header(packet_id=7)).name == "CarStatus"
        assert parse_header(build_header(packet_id=16)).name == "CarTelemetryData2"

    def test_short_header_raises(self) -> None:
        """不足 29 字节应抛 PacketTooShortError。"""
        with pytest.raises(PacketTooShortError):
            parse_header(b"\x00" * 28)
        with pytest.raises(PacketTooShortError):
            parse_header(b"")
        # 恰好 29 字节不抛
        parse_header(build_header(packet_id=0))


# ===========================================================================
# 2. 未知 packetId / packet_name
# ===========================================================================
class TestPacketDispatch:
    """parse_packet 分发逻辑。"""

    def test_unknown_packet_id_returns_none(self) -> None:
        """未知 packetId 应返回 None（跳过不崩溃）。"""
        # 0 (Motion) 不在本版 6 类支持范围内
        data = build_header(packet_id=0) + b"\x00" * 64
        assert parse_packet(data) is None
        # 3 (Event) 不支持
        data = build_header(packet_id=3) + b"\x00" * 64
        assert parse_packet(data) is None
        # 99 完全未知
        data = build_header(packet_id=99) + b"\x00" * 64
        assert parse_packet(data) is None

    def test_supported_packet_ids(self) -> None:
        """SUPPORTED_PACKET_IDS 应为 {1,2,5,6,7,16}。"""
        assert SUPPORTED_PACKET_IDS == frozenset({1, 2, 5, 6, 7, 16})

    def test_packet_name_known_and_unknown(self) -> None:
        """packet_name 已知返回名称，未知返回 Unknown(<id>)。"""
        assert packet_name(1) == "Session"
        assert packet_name(99) == "Unknown(99)"
        assert PACKET_NAMES[16] == "CarTelemetryData2"


# ===========================================================================
# 3. Packet 1 — Session
# ===========================================================================
class TestSessionPacket:
    """Packet 1 (Session) 解析。"""

    @staticmethod
    def _build_session_body(
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
        """构造 Session 包体（prefix + num_wfs 个天气样本）。"""
        # 16 字段
        prefix_core = struct.pack(
            "<BbbBHBbBHHBBBBBB",
            weather, track_temp, air_temp, total_laps, track_len,
            session_type, track_id, formula, session_time_left, session_duration,
            pit_speed_limit, 0, 0, 0, 0, 21,
        )
        # 21 marshal zones (float zoneStart + uint8 zoneFlag)
        marshal_zones = b"".join(struct.pack("<fb", 0.0, 0) for _ in range(21))
        # safetyCarStatus, networkGame, numWeatherForecastSamples
        tail = struct.pack("<BBB", 0, 0, num_wfs)
        body = prefix_core + marshal_zones + tail
        # 天气样本（各 8 字节）
        for _ in range(num_wfs):
            body += struct.pack("<BBBbbbbB", 0, 5, 1, 20, 0, 18, 0, 30)
        return body

    def test_session_basic_fields(self) -> None:
        """逐字段断言 Session 解析。"""
        body = self._build_session_body(
            weather=0, track_temp=25, air_temp=22, total_laps=58,
            track_len=5807, track_id=2, formula=1,
            session_time_left=1800, session_duration=3600,
            pit_speed_limit=60, num_wfs=0,
        )
        data = build_header(packet_id=1) + body
        result = parse_packet(data)
        assert result is not None
        assert result["packet_id"] == 1
        assert result["name"] == "Session"
        assert result["m_trackId"] == 2
        assert result["m_weather"] == 0
        assert result["m_trackTemperature"] == 25
        assert result["m_airTemperature"] == 22
        assert result["m_totalLaps"] == 58
        assert result["m_trackLength"] == 5807
        assert result["m_sessionType"] == 0
        assert result["m_formula"] == 1
        assert result["m_sessionTimeLeft"] == 1800
        assert result["m_sessionDuration"] == 3600
        assert result["m_pitSpeedLimit"] == 60
        assert result["m_numWeatherForecastSamples"] == 0
        assert result["m_weatherForecastSamples"] == []

    def test_session_with_weather_forecast_samples(self) -> None:
        """含 2 个天气样本时应正确解析样本数组。"""
        body = self._build_session_body(num_wfs=2)
        data = build_header(packet_id=1) + body
        result = parse_packet(data)
        assert result is not None
        assert result["m_numWeatherForecastSamples"] == 2
        wfs = result["m_weatherForecastSamples"]
        assert len(wfs) == 2
        assert wfs[0]["m_weather"] == 1
        assert wfs[0]["m_rainPercentage"] == 30
        assert wfs[1]["m_timeOffset"] == 5

    def test_session_short_body_raises(self) -> None:
        """Session 包体不足应抛 PacketTooShortError。"""
        data = build_header(packet_id=1) + b"\x00" * 10  # 远不足
        with pytest.raises(PacketTooShortError):
            parse_packet(data)


# ===========================================================================
# 4. Packet 2 — LapData
# ===========================================================================
class TestLapDataPacket:
    """Packet 2 (LapData) 解析（按车分组，只解玩家车）。"""

    @staticmethod
    def _build_lap_per_car(
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
        """构造单辆车的 LapData 段（57 字节）。"""
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

    def test_lap_data_player_car_index_0(self) -> None:
        """playerCarIndex=0 时解析第一辆车。"""
        per = self._build_lap_per_car(
            last_lap_ms=95000, current_lap_ms=45000,
            s1_ms_part=30000, s1_min_part=0,
            s2_ms_part=25000, s2_min_part=0,
            delta_front_ms_part=1234, delta_front_min_part=0,
            delta_leader_ms_part=5678, delta_leader_min_part=0,
            lap_dist=1500.0, total_dist=3000.0,
            car_pos=5, current_lap=3, sector=1,
            speed_trap_speed=320.0,
        )
        data = build_header(packet_id=2, player_car_index=0) + per
        result = parse_packet(data)
        assert result is not None
        assert result["m_lastLapTimeInMS"] == 95000
        assert result["m_currentLapTimeInMS"] == 45000
        # 扇区组合：minutes * 60000 + ms_part
        assert result["m_sector1TimeInMS"] == 0 * 60000 + 30000
        assert result["m_sector2TimeInMS"] == 0 * 60000 + 25000
        assert result["m_deltaToCarInFrontInMS"] == 0 * 60000 + 1234
        assert result["m_deltaToRaceLeaderInMS"] == 0 * 60000 + 5678
        assert result["m_lapDistance"] == pytest.approx(1500.0)
        assert result["m_totalDistance"] == pytest.approx(3000.0)
        assert result["m_carPosition"] == 5
        assert result["m_currentLapNum"] == 3
        assert result["m_sector"] == 1
        assert result["m_speedTrapFastestSpeed"] == pytest.approx(320.0)

    def test_lap_data_sector_time_with_minutes(self) -> None:
        """扇区时间组合应正确：minutes * 60000 + ms_part。"""
        per = self._build_lap_per_car(
            s1_ms_part=15000, s1_min_part=1,  # 1*60000+15000 = 75000
            s2_ms_part=20000, s2_min_part=2,  # 2*60000+20000 = 140000
        )
        data = build_header(packet_id=2, player_car_index=0) + per
        result = parse_packet(data)
        assert result is not None
        assert result["m_sector1TimeInMS"] == 75000
        assert result["m_sector2TimeInMS"] == 140000

    def test_lap_data_player_car_index_nonzero(self) -> None:
        """playerCarIndex=3 时应跳过前 3 辆车，解析第 4 辆车。"""
        per_size = struct.calcsize("<IIHBHBHBHBfffBBBBBBBBBBBBBBBHHBfB")
        junk = b"\x00" * per_size
        target = self._build_lap_per_car(last_lap_ms=77777, car_pos=10)
        # 前 3 辆填 0，第 4 辆（index=3）填 target
        body = junk * 3 + target
        # 补齐到 24 辆车避免越界（虽不必要，但稳妥）
        body += junk * (NUM_CARS - 4)
        data = build_header(packet_id=2, player_car_index=3) + body
        result = parse_packet(data)
        assert result is not None
        assert result["m_lastLapTimeInMS"] == 77777
        assert result["m_carPosition"] == 10

    def test_lap_data_short_body_raises(self) -> None:
        """包体不足以覆盖玩家车应抛 PacketTooShortError。"""
        data = build_header(packet_id=2, player_car_index=5) + b"\x00" * 10
        with pytest.raises(PacketTooShortError):
            parse_packet(data)

    def test_lap_data_player_index_out_of_range(self) -> None:
        """playerCarIndex 越界（如 255）应抛 PacketTooShortError。"""
        data = build_header(packet_id=2, player_car_index=255) + b"\x00" * 64
        with pytest.raises(PacketTooShortError):
            parse_packet(data)


# ===========================================================================
# 5. Packet 5 — CarSetups
# ===========================================================================
class TestCarSetupsPacket:
    """Packet 5 (CarSetups) 解析。"""

    @staticmethod
    def _build_setup_per_car(
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
        return struct.pack(
            "<BBBBffffBBBBBBBBBffffBf",
            front_wing, rear_wing, on_throttle, off_throttle,
            front_camber, rear_camber, front_toe, rear_toe,
            1, 1, 1, 1, 1, 1, brake_pressure, brake_bias, 50,  # 9 × uint8
            rear_left_press, 25.5, front_left_press, 25.5,  # 4 × float 胎压
            ballast, fuel_load,
        )

    def test_car_setups_fields(self) -> None:
        """逐字段断言 CarSetups 解析。"""
        per = self._build_setup_per_car(
            front_wing=7, rear_wing=3, on_throttle=60, off_throttle=40,
            front_camber=-3.0, rear_camber=-2.0,
            front_toe=0.20, rear_toe=0.30,
            brake_pressure=80, brake_bias=70,
            front_left_press=26.0, rear_left_press=25.0,
            ballast=55, fuel_load=110.0,
        )
        data = build_header(packet_id=5, player_car_index=0) + per
        result = parse_packet(data)
        assert result is not None
        assert result["m_frontWing"] == 7
        assert result["m_rearWing"] == 3
        assert result["m_onThrottleDiff"] == 60
        assert result["m_offThrottleDiff"] == 40
        assert result["m_frontCamber"] == pytest.approx(-3.0)
        assert result["m_rearCamber"] == pytest.approx(-2.0)
        assert result["m_frontToe"] == pytest.approx(0.20)
        assert result["m_rearToe"] == pytest.approx(0.30)
        assert result["m_brakePressure"] == 80
        assert result["m_brakeBias"] == 70
        assert result["m_rearLeftTyrePressure"] == pytest.approx(25.0)
        assert result["m_frontLeftTyrePressure"] == pytest.approx(26.0)
        assert result["m_ballast"] == 55
        assert result["m_fuelLoad"] == pytest.approx(110.0)

    def test_car_setups_player_car_index_2(self) -> None:
        """playerCarIndex=2 时应解析第 3 辆车。"""
        per_size = struct.calcsize("<BBBBffffBBBBBBBBBffffBf")
        target = self._build_setup_per_car(front_wing=9)
        body = b"\x00" * per_size * 2 + target
        data = build_header(packet_id=5, player_car_index=2) + body
        result = parse_packet(data)
        assert result is not None
        assert result["m_frontWing"] == 9


# ===========================================================================
# 6. Packet 6 — CarTelemetry
# ===========================================================================
class TestCarTelemetryPacket:
    """Packet 6 (CarTelemetry) 解析。"""

    @staticmethod
    def _build_telem_per_car(
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

    def test_car_telemetry_fields(self) -> None:
        """逐字段断言 CarTelemetry 解析（含数组字段）。"""
        per = self._build_telem_per_car(
            speed=320, throttle=1.0, brake=0.5,
            gear=6, rpm=12000,
            brakes_temp=(450, 460, 440, 430),
            tyres_surface_temp=(95, 97, 93, 94),
            engine_temp=110,
            tyres_press=(26.0, 26.0, 25.5, 25.5),
        )
        data = build_header(packet_id=6, player_car_index=0) + per
        result = parse_packet(data)
        assert result is not None
        assert result["m_speed"] == 320
        assert result["m_throttle"] == pytest.approx(1.0)
        assert result["m_steer"] == pytest.approx(0.0)
        assert result["m_brake"] == pytest.approx(0.5)
        assert result["m_gear"] == 6
        assert result["m_engineRPM"] == 12000
        assert result["m_brakesTemperature"] == [450, 460, 440, 430]
        assert result["m_tyresSurfaceTemperature"] == [95, 97, 93, 94]
        assert result["m_engineTemperature"] == 110
        assert result["m_tyresPressure"] == [
            pytest.approx(26.0), pytest.approx(26.0),
            pytest.approx(25.5), pytest.approx(25.5),
        ]

    def test_car_telemetry_negative_gear(self) -> None:
        """档位 int8 应支持负值（-1=倒挡）。"""
        per = self._build_telem_per_car(gear=-1)
        data = build_header(packet_id=6, player_car_index=0) + per
        result = parse_packet(data)
        assert result is not None
        assert result["m_gear"] == -1


# ===========================================================================
# 7. Packet 7 — CarStatus
# ===========================================================================
class TestCarStatusPacket:
    """Packet 7 (CarStatus) 解析。"""

    @staticmethod
    def _build_status_per_car(
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
        return struct.pack(
            "<BBBBBfffHHBBHBBBbfffBffffB",
            traction, abs_on, fuel_mix, front_brake_bias, pit_limiter,
            fuel_in_tank, fuel_capacity, fuel_remaining_laps,
            max_rpm, idle_rpm,
            max_gears, drs_allowed,
            500,  # drs_activation_distance
            actual_tyre, visual_tyre,
            tyres_age,
            fia_flags,
            800.0, 60.0, ers_store,  # enginePowerICE, MGUK, ersStoreEnergy
            ers_deploy_mode,
            100000.0, 200000.0, 2000000.0, 500000.0,  # 4 × float ERS
            network_paused,
        )

    def test_car_status_fields(self) -> None:
        """逐字段断言 CarStatus 解析。"""
        per = self._build_status_per_car(
            fuel_mix=3, front_brake_bias=58,
            fuel_in_tank=45.0, fuel_capacity=110.0,
            actual_tyre=7, visual_tyre=7, tyres_age=5,
            ers_store=800000.0, ers_deploy_mode=2,
        )
        data = build_header(packet_id=7, player_car_index=0) + per
        result = parse_packet(data)
        assert result is not None
        assert result["m_tractionControl"] == 1
        assert result["m_antiLockBrakes"] == 1
        assert result["m_fuelMix"] == 3
        assert result["m_frontBrakeBias"] == 58
        assert result["m_fuelInTank"] == pytest.approx(45.0)
        assert result["m_fuelCapacity"] == pytest.approx(110.0)
        assert result["m_actualTyreCompound"] == 7
        assert result["m_visualTyreCompound"] == 7
        assert result["m_tyresAgeLaps"] == 5
        assert result["m_ersStoreEnergy"] == pytest.approx(800000.0)
        assert result["m_ersDeployMode"] == 2

    def test_car_status_negative_fia_flags(self) -> None:
        """FIA 旗语 int8 应支持负值。"""
        per = self._build_status_per_car(fia_flags=-1)
        data = build_header(packet_id=7, player_car_index=0) + per
        result = parse_packet(data)
        assert result is not None
        assert result["m_vehicleFiaFlags"] == -1


# ===========================================================================
# 8. Packet 16 — CarTelemetryData2
# ===========================================================================
class TestCarTelemetry2Packet:
    """Packet 16 (CarTelemetryData2) 解析。"""

    @staticmethod
    def _build_ct2_per_car(
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
        return struct.pack(
            "<BBHBBHBB",
            aero_mode, aero_available, aero_distance,
            overtake_available, overtake_active, overtake_distance,
            reg_2026, wrong_way,
        )

    def test_ct2_fields(self) -> None:
        """逐字段断言 CarTelemetryData2 解析。"""
        per = self._build_ct2_per_car(
            aero_mode=1, aero_distance=150,
            overtake_available=1, overtake_active=1,
            overtake_distance=250, wrong_way=1,
        )
        data = build_header(packet_id=16, player_car_index=0) + per
        result = parse_packet(data)
        assert result is not None
        assert result["m_activeAeroMode"] == 1
        assert result["m_activeAeroAvailable"] == 1
        assert result["m_activeAeroActivationDistance"] == 150
        assert result["m_overtakeAvailable"] == 1
        assert result["m_overtakeActive"] == 1
        assert result["m_overtakeActivationDistance"] == 250
        assert result["m_2026Regulations"] == 1
        assert result["m_drivingWrongWay"] == 1

    def test_ct2_player_car_index_1(self) -> None:
        """playerCarIndex=1 时应解析第 2 辆车。"""
        per_size = struct.calcsize("<BBHBBHBB")
        target = self._build_ct2_per_car(aero_mode=0, overtake_active=1)
        body = b"\x00" * per_size + target
        data = build_header(packet_id=16, player_car_index=1) + body
        result = parse_packet(data)
        assert result is not None
        assert result["m_activeAeroMode"] == 0
        assert result["m_overtakeActive"] == 1


# ===========================================================================
# 9. parse_packet 端到端
# ===========================================================================
class TestParsePacketEndToEnd:
    """parse_packet 端到端：header + body 组合。"""

    def test_result_contains_header_object(self) -> None:
        """parse_packet 结果应含 header PacketHeader 对象。"""
        body = TestSessionPacket._build_session_body()
        data = build_header(packet_id=1, player_car_index=0) + body
        result = parse_packet(data)
        assert result is not None
        assert isinstance(result["header"], PacketHeader)
        assert result["header"].packet_id == 1

    def test_short_packet_at_parse_packet(self) -> None:
        """parse_packet 对短包应抛 PacketTooShortError（不返回 None）。"""
        with pytest.raises(PacketTooShortError):
            parse_packet(b"\x00" * 10)