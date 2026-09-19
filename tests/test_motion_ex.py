"""Packet 13 (MotionEx) 解析 + 规则9 刮底检测 端到端测试。

背景：
    此前 ``setup_tuner`` 只解析 5 类包（Session/LapData/CarSetups/CarTelemetry/
    CarStatus），而 ``engine._apply_ride_height_rules``（规则9 刮底检测）是空实现
    （``pass`` + 注释「遥测中无直接刮底信号」）。

    但 EA F1 25 规范中 Packet 13 (MotionEx) 明确提供
    ``m_frontAeroHeight`` / ``m_rearAeroHeight``——
    "plank edge height above road surface"（底板前/后缘离地高度），
    正是刮底检测所需的信号。本测试锁定：
        1. MotionEx 244 字节包体按官方字段序正确解析；
        2. 短包抛 ``PacketTooShortError``，不静默返回脏数据；
        3. ``parse_packet`` 能分发 packet_id 13（SUPPORTED_PACKET_IDS 含 13）；
        4. ``LapAggregator.on_motion_ex`` 正确累积极值/均值/触底比例；
        5. 规则9 在有触底信号时抬高 ``ride_height_req``，无信号时不误触发。

所有样本由 ``struct.pack`` 按官方字段序构造，字段值选取易识别魔术数。
"""

from __future__ import annotations

import struct

import pytest

from setup_tuner.engine.engine import _apply_ride_height_rules
from setup_tuner.telemetry.lap_aggregator import LapAggregator
from setup_tuner.telemetry.packets import (
    HEADER_FORMAT,
    HEADER_SIZE,
    SUPPORTED_PACKET_IDS,
    PacketTooShortError,
    packet_name,
    parse_motion_ex,
    parse_packet,
)

# ===========================================================================
# 辅助：按官方字段序构造 MotionEx 包
# ===========================================================================
# Source: EA F1 25 UDP Telemetry Specification, Packet 13 (MotionEx), size 273 bytes
# body = 244 字节 = 61 × float32：
#   8 × float[4] 数组 + 11 标量 + float[4] + 6 标量 + 2 × float[4]
_MOTIONEX_FMT = "<" + "4f" * 8 + "f" * 11 + "4f" + "f" * 6 + "4f" * 2


def build_header(packet_id: int = 13, player_car_index: int = 0) -> bytes:
    """构造 29 字节包头。"""
    return struct.pack(
        HEADER_FORMAT,
        2026,           # packet_format
        26,             # game_year
        1,              # game_major
        0,              # game_minor
        1,              # packet_version
        packet_id,
        0x1234_5678_9ABC_DEF0,  # session_uid
        12.5,           # session_time
        100,            # frame_id
        200,            # overall_frame_id
        player_car_index,
        255,            # secondary_player
    )


def build_motion_ex(
    *,
    suspension_position: tuple[float, float, float, float] = (0.1, 0.2, 0.3, 0.4),
    front_aero_height: float = 0.025,
    rear_aero_height: float = 0.030,
    cog_height: float = 0.30,
    chassis_pitch: float = 0.01,
    wheel_camber: tuple[float, float, float, float] = (-0.03, -0.03, -0.02, -0.02),
    wheel_camber_gain: tuple[float, float, float, float] = (0.001, 0.001, 0.0, 0.0),
    packet_id: int = 13,
) -> bytes:
    """构造一个完整的 MotionEx UDP 包（header + 244 字节包体）。"""
    values: list[float] = []
    # 8 个车轮数组（车轮顺序 RL, RR, FL, FR）
    values += list(suspension_position)          # m_suspensionPosition
    values += [0.01, 0.02, 0.03, 0.04]           # m_suspensionVelocity
    values += [0.1, 0.2, 0.3, 0.4]               # m_suspensionAcceleration
    values += [50.0, 51.0, 52.0, 53.0]           # m_wheelSpeed
    values += [0.01, 0.02, 0.03, 0.04]           # m_wheelSlipRatio
    values += [0.05, 0.06, 0.07, 0.08]           # m_wheelSlipAngle
    values += [100.0, 200.0, 300.0, 400.0]       # m_wheelLatForce
    values += [500.0, 600.0, 700.0, 800.0]       # m_wheelLongForce
    # 11 个标量
    values += [
        cog_height,      # m_heightOfCOGAboveGround
        1.0, 2.0, 3.0,   # m_localVelocityX/Y/Z
        0.1, 0.2, 0.3,   # m_angularVelocityX/Y/Z
        0.01, 0.02, 0.03,  # m_angularAccelerationX/Y/Z
        0.15,            # m_frontWheelsAngle
    ]
    # float[4]
    values += [1000.0, 2000.0, 3000.0, 4000.0]   # m_wheelVertForce
    # 6 个标量
    values += [
        front_aero_height,  # m_frontAeroHeight
        rear_aero_height,   # m_rearAeroHeight
        0.02,               # m_frontRollAngle
        0.03,               # m_rearRollAngle
        0.07,               # m_chassisYaw
        chassis_pitch,      # m_chassisPitch
    ]
    # 2 × float[4]
    values += list(wheel_camber)        # m_wheelCamber
    values += list(wheel_camber_gain)   # m_wheelCamberGain

    assert len(values) == 61, f"字段数应为 61，实际 {len(values)}"
    body = struct.pack(_MOTIONEX_FMT, *values)
    assert len(body) == 244, f"包体应为 244 字节，实际 {len(body)}"
    return build_header(packet_id=packet_id) + body


# ===========================================================================
# 1. 结构体与协议一致性
# ===========================================================================
class TestMotionExStructure:
    """MotionEx 包体结构与官方规范一致。"""

    def test_body_size_is_244(self) -> None:
        """官方规范：Packet 13 总长 273 字节，减去 29 字节 header = 244 字节包体。"""
        assert struct.calcsize(_MOTIONEX_FMT) == 244
        assert HEADER_SIZE + 244 == 273

    def test_packet_id_13_in_supported_ids(self) -> None:
        """packet_id 13 必须在 SUPPORTED_PACKET_IDS 内（否则 stream 缓存会丢弃）。"""
        assert 13 in SUPPORTED_PACKET_IDS

    def test_packet_name_is_motion_ex(self) -> None:
        """包名映射正确。"""
        assert packet_name(13) == "MotionEx"


# ===========================================================================
# 2. 字段解析
# ===========================================================================
class TestMotionExParsing:
    """逐字段断言 MotionEx 解析结果与构造值一致。"""

    def test_parse_all_key_fields(self) -> None:
        """关键字段（悬挂位置/底板离地高度/姿态）逐字段对照。"""
        data = build_motion_ex(
            suspension_position=(0.11, 0.22, 0.33, 0.44),
            front_aero_height=0.0123,
            rear_aero_height=0.0456,
            cog_height=0.321,
            chassis_pitch=0.078,
        )
        parsed = parse_motion_ex(data)

        # 规范为 float32，断言须用 float32 容差而非精确相等
        assert parsed["m_suspensionPosition"] == pytest.approx(
            [0.11, 0.22, 0.33, 0.44], abs=1e-6,
        )
        assert parsed["m_frontAeroHeight"] == pytest.approx(0.0123, abs=1e-6)
        assert parsed["m_rearAeroHeight"] == pytest.approx(0.0456, abs=1e-6)
        assert parsed["m_heightOfCOGAboveGround"] == pytest.approx(0.321, abs=1e-6)
        assert parsed["m_chassisPitch"] == pytest.approx(0.078, abs=1e-6)

    def test_wheel_arrays_are_length_four(self) -> None:
        """所有车轮数组必须为长度 4（RL, RR, FL, FR）。"""
        parsed = parse_motion_ex(build_motion_ex())
        for key in (
            "m_suspensionPosition",
            "m_suspensionVelocity",
            "m_suspensionAcceleration",
            "m_wheelSpeed",
            "m_wheelSlipRatio",
            "m_wheelSlipAngle",
            "m_wheelLatForce",
            "m_wheelLongForce",
            "m_wheelVertForce",
            "m_wheelCamber",
            "m_wheelCamberGain",
        ):
            assert key in parsed, f"缺少字段 {key}"
            assert len(parsed[key]) == 4, f"{key} 长度应为 4"

    def test_official_field_order(self) -> None:
        """字段顺序必须与官方规范一致（用可区分的递增魔术数锁定顺序）。

        若把 ``m_chassisPitch`` 与 ``m_wheelCamber`` 顺序写反，
        或把 ``m_wheelSlipAngle`` 挪到 ``m_wheelVertForce`` 之后，
        本测试即会失败。
        """
        data = build_motion_ex(wheel_camber=(-0.031, -0.032, -0.021, -0.022))
        parsed = parse_motion_ex(data)

        # 8 个数组后的第一个标量必须是 COG 高度
        assert parsed["m_heightOfCOGAboveGround"] == pytest.approx(0.30, abs=1e-6)
        # m_wheelVertForce 在 m_frontWheelsAngle 之后
        assert parsed["m_wheelVertForce"] == [1000.0, 2000.0, 3000.0, 4000.0]
        # 6 个标量之后依次是 camber / camberGain（float32 容差）
        assert parsed["m_wheelCamber"] == pytest.approx(
            [-0.031, -0.032, -0.021, -0.022], abs=1e-6,
        )
        assert parsed["m_wheelCamberGain"] == pytest.approx(
            [0.001, 0.001, 0.0, 0.0], abs=1e-6,
        )


# ===========================================================================
# 3. 容错
# ===========================================================================
class TestMotionExRobustness:
    """短包必须显式抛错，不得静默返回脏数据。"""

    def test_short_packet_raises(self) -> None:
        """包体不足 244 字节时抛 PacketTooShortError。"""
        data = build_header(packet_id=13) + b"\x00" * 200
        with pytest.raises(PacketTooShortError):
            parse_motion_ex(data)

    def test_header_only_raises(self) -> None:
        """只有包头时抛 PacketTooShortError。"""
        with pytest.raises(PacketTooShortError):
            parse_motion_ex(build_header(packet_id=13))

    def test_exact_size_ok(self) -> None:
        """恰好 273 字节应成功（无多余字节依赖）。"""
        parsed = parse_motion_ex(build_motion_ex())
        assert isinstance(parsed, dict)

    def test_trailing_bytes_tolerated(self) -> None:
        """包尾有多余字节时仍能解析（前向兼容新版协议）。"""
        data = build_motion_ex() + b"\xff" * 16
        parsed = parse_motion_ex(data)
        assert parsed["m_frontAeroHeight"] == pytest.approx(0.025, abs=1e-6)


# ===========================================================================
# 4. parse_packet 分发
# ===========================================================================
class TestParsePacketDispatch:
    """``parse_packet`` 主入口正确分发 packet_id 13 且无需 player_car_index。"""

    def test_dispatch_packet_13(self) -> None:
        parsed = parse_packet(build_motion_ex())
        assert parsed is not None
        assert parsed["packet_id"] == 13
        assert parsed["name"] == "MotionEx"
        assert "m_frontAeroHeight" in parsed

    def test_nonzero_player_car_index_does_not_shift(self) -> None:
        """MotionEx 非按车分组：player_car_index 非 0 时解析结果不变。"""
        base = build_motion_ex()
        # 用 player_car_index=10 重建包头
        body = base[HEADER_SIZE:]
        shifted = build_header(packet_id=13, player_car_index=10) + body
        parsed = parse_packet(shifted)
        assert parsed is not None
        assert parsed["m_frontAeroHeight"] == pytest.approx(0.025, abs=1e-6)


# ===========================================================================
# 5. LapAggregator 累积
# ===========================================================================
class TestLapAggregatorMotionEx:
    """整圈聚合器对 MotionEx 的累积统计。"""

    def test_no_motion_ex_no_keys(self) -> None:
        """未收到 MotionEx 时，快照不应出现刮底相关键（避免下游误判）。"""
        agg = LapAggregator()
        agg.on_telemetry({"m_speed": 200.0, "m_throttle": 1.0})
        snap = agg.snapshot()
        assert "plank_bottoming" not in snap
        assert "motion_ex_frames" not in snap

    def test_single_frame_stats(self) -> None:
        """单帧极值/均值/比例正确。"""
        agg = LapAggregator()
        agg.on_motion_ex({
            "m_frontAeroHeight": 0.008,
            "m_rearAeroHeight": 0.030,
            "m_suspensionPosition": [0.05, 0.06, 0.07, 0.08],
        })
        snap = agg.snapshot()
        assert snap["motion_ex_frames"] == 1
        assert snap["plank_front_height_min"] == pytest.approx(0.008, abs=1e-6)
        assert snap["plank_rear_height_min"] == pytest.approx(0.030, abs=1e-6)
        assert snap["plank_bottoming_ratio"] == pytest.approx(1.0, abs=1e-6)
        assert snap["plank_bottoming"] is True
        assert snap["suspension_height_min"] == pytest.approx(0.05, abs=1e-6)

    def test_bottoming_detected_from_front_plank(self) -> None:
        """前缘进入触地带 → plank_bottoming 为真。"""
        agg = LapAggregator()
        agg.on_motion_ex({
            "m_frontAeroHeight": 0.005,
            "m_rearAeroHeight": 0.035,
            "m_suspensionPosition": [0.02] * 4,
        })
        assert agg.snapshot()["plank_bottoming"] is True

    def test_bottoming_detected_from_rear_plank(self) -> None:
        """后缘进入触地带同样触发（不应只看前缘）。"""
        agg = LapAggregator()
        agg.on_motion_ex({
            "m_frontAeroHeight": 0.040,
            "m_rearAeroHeight": 0.009,
            "m_suspensionPosition": [0.02] * 4,
        })
        assert agg.snapshot()["plank_bottoming"] is True

    def test_no_false_positive_when_clear(self) -> None:
        """离地高度充足时不得误报。"""
        agg = LapAggregator()
        for _ in range(5):
            agg.on_motion_ex({
                "m_frontAeroHeight": 0.028,
                "m_rearAeroHeight": 0.033,
                "m_suspensionPosition": [0.06] * 4,
            })
        snap = agg.snapshot()
        assert snap["plank_bottoming"] is False
        assert snap["plank_bottoming_ratio"] == pytest.approx(0.0, abs=1e-6)

    def test_ratio_reflects_partial_bottoming(self) -> None:
        """10 帧中 3 帧触底 → 比例 0.3。"""
        agg = LapAggregator()
        for i in range(10):
            agg.on_motion_ex({
                "m_frontAeroHeight": 0.006 if i < 3 else 0.030,
                "m_rearAeroHeight": 0.035,
                "m_suspensionPosition": [0.02] * 4,
            })
        snap = agg.snapshot()
        assert snap["plank_bottoming_ratio"] == pytest.approx(0.3, abs=1e-6)
        assert snap["plank_bottoming"] is True

    def test_frames_reset_on_lap_change(self) -> None:
        """圈号变化时 MotionEx 累积量一起重置（不跨圈污染）。"""
        agg = LapAggregator()
        agg.on_lap_data({"m_currentLapNum": 1, "m_sector": 0})
        agg.on_telemetry({"m_speed": 100.0})
        agg.on_motion_ex({
            "m_frontAeroHeight": 0.006,
            "m_rearAeroHeight": 0.035,
            "m_suspensionPosition": [0.02] * 4,
        })
        assert agg.snapshot()["plank_bottoming"] is True

        # 切换到第 2 圈
        agg.on_lap_data({"m_currentLapNum": 2, "m_sector": 0})
        snap = agg.snapshot()
        assert "motion_ex_frames" not in snap, "换圈后 MotionEx 累积量应清空"

    def test_malformed_frame_ignored(self) -> None:
        """缺字段/类型错误的帧不得让聚合器崩溃或计数。"""
        agg = LapAggregator()
        agg.on_motion_ex({})
        agg.on_motion_ex({"m_frontAeroHeight": "bad", "m_rearAeroHeight": None})
        agg.on_motion_ex({"m_suspensionPosition": "not-a-list"})
        assert "motion_ex_frames" not in agg.snapshot()

    def test_suspension_only_frame_counts(self) -> None:
        """仅有悬挂数据（无 aero height）时也计入帧数并记录悬挂极小值。

        本帧悬挂未压到下限（0.003 m 仅 RL 一路偏低，仍高于触底阈值区
        的独立判据），因此不应仅凭悬挂判定刮底——但数据的极值必须被记录。
        """
        agg = LapAggregator()
        agg.on_motion_ex({"m_suspensionPosition": [0.05, 0.06, 0.07, 0.08]})
        snap = agg.snapshot()
        assert snap["motion_ex_frames"] == 1
        assert snap["suspension_height_min"] == pytest.approx(0.05, abs=1e-6)
        # 无 aero height 且悬挂未触底 → 不判定刮底
        assert snap["plank_bottoming"] is False

    def test_suspension_only_frame_can_flag_bottoming(self) -> None:
        """仅有悬挂数据且已压到行程下限时，应作为触底迹象计入。"""
        agg = LapAggregator()
        agg.on_motion_ex({"m_suspensionPosition": [0.003, 0.02, 0.02, 0.02]})
        snap = agg.snapshot()
        assert snap["suspension_height_min"] == pytest.approx(0.003, abs=1e-6)
        assert snap["plank_bottoming"] is True


# ===========================================================================
# 6. 规则9 刮底检测
# ===========================================================================
class TestRideHeightRule:
    """规则9：底板触地 → ride_height_req 抬高。"""

    def test_triggers_on_bottoming_flag(self) -> None:
        dx = {"ride_height_req": 0.0}
        _apply_ride_height_rules({"plank_bottoming": True}, dx)
        assert dx["ride_height_req"] == pytest.approx(0.4, abs=1e-6)

    def test_triggers_on_low_min_height_without_flag(self) -> None:
        """未提供布尔标志时，用最小离地高度兜底判断。"""
        dx = {"ride_height_req": 0.0}
        _apply_ride_height_rules({"plank_front_height_min": 0.007}, dx)
        assert dx["ride_height_req"] == pytest.approx(0.4, abs=1e-6)

    def test_triggers_on_low_rear_min_height(self) -> None:
        """后缘最小高度低同样触发（不只看前缘）。"""
        dx = {"ride_height_req": 0.0}
        _apply_ride_height_rules({"plank_rear_height_min": 0.009}, dx)
        assert dx["ride_height_req"] == pytest.approx(0.4, abs=1e-6)

    def test_no_trigger_without_data(self) -> None:
        """无任何刮底信号时不触发（保持既有行为）。"""
        dx = {"ride_height_req": 0.0}
        _apply_ride_height_rules({}, dx)
        assert dx["ride_height_req"] == pytest.approx(0.0, abs=1e-6)

    def test_no_trigger_when_height_sufficient(self) -> None:
        """离地高度充足时不触发。"""
        dx = {"ride_height_req": 0.0}
        _apply_ride_height_rules({
            "plank_bottoming": False,
            "plank_front_height_min": 0.025,
            "plank_rear_height_min": 0.030,
        }, dx)
        assert dx["ride_height_req"] == pytest.approx(0.0, abs=1e-6)

    def test_bool_true_flag_not_treated_as_number(self) -> None:
        """``True`` 作为数值兜底时不得被当成 1.0 米之类的无效值误判。"""
        dx = {"ride_height_req": 0.0}
        _apply_ride_height_rules({"plank_front_height_min": True}, dx)
        assert dx["ride_height_req"] == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# 7. 端到端：UDP bytes → 解析 → 聚合 → 规则9
# ===========================================================================
class TestEndToEnd:
    """从原始 UDP 字节到 DX 贡献的完整链路。"""

    def test_full_chain_triggers_ride_height(self) -> None:
        """构造触底遥测包 → 聚合 → 规则9 抬高 ride_height_req。"""
        agg = LapAggregator()
        for _ in range(4):
            parsed = parse_packet(build_motion_ex(
                front_aero_height=0.006,
                rear_aero_height=0.035,
                suspension_position=(0.02, 0.02, 0.02, 0.02),
            ))
            assert parsed is not None
            agg.on_motion_ex(parsed)
        # 需要至少一帧 CarTelemetry 才有快照内的帧计数语义
        agg.on_telemetry({"m_speed": 180.0, "m_throttle": 0.8})

        snap = agg.snapshot()
        dx = {"ride_height_req": 0.0}
        _apply_ride_height_rules(snap, dx)
        assert dx["ride_height_req"] == pytest.approx(0.4, abs=1e-6)

    def test_full_chain_no_trigger_when_clear(self) -> None:
        """干净的遥测不应产生刮底修正。"""
        agg = LapAggregator()
        for _ in range(4):
            parsed = parse_packet(build_motion_ex(
                front_aero_height=0.030,
                rear_aero_height=0.035,
            ))
            assert parsed is not None
            agg.on_motion_ex(parsed)

        snap = agg.snapshot()
        dx = {"ride_height_req": 0.0}
        _apply_ride_height_rules(snap, dx)
        assert dx["ride_height_req"] == pytest.approx(0.0, abs=1e-6)
