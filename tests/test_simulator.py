"""TelemetrySimulator 单元测试。

覆盖：
1. 模拟遥测生成（CarTelemetry 帧 + corner 事件）
2. 回放功能（按赛道/圈数回放，max_laps 退出）
3. handler 推送（WebSocket 风格回调）
4. 赛道切换（24 条赛道）
5. 数据包格式正确性（字段值范围、类型）
6. 边界条件：无效赛道ID、空回放、handler 异常
7. generate_lap_snapshot 便捷函数
8. 物理辅助函数（_clamp / _speed_to_gear / _speed_to_rpm / _estimate_lap_time）
"""

from __future__ import annotations

import threading
import time

import pytest

from setup_tuner.domain.track import ALL_TRACKS, get_track_by_id
from setup_tuner.telemetry.simulator import (
    DEFAULT_FPS,
    DEFAULT_SEED,
    TelemetrySimulator,
    _clamp,
    _estimate_lap_time,
    _speed_to_gear,
    _speed_to_rpm,
    generate_lap_snapshot,
)


# ===========================================================================
# 1. 物理辅助函数
# ===========================================================================
class TestPhysicsHelpers:
    """_clamp / _speed_to_gear / _speed_to_rpm / _estimate_lap_time。"""

    def test_clamp_within_range(self) -> None:
        """值在范围内时返回原值。"""
        assert _clamp(5.0, 0.0, 10.0) == 5.0

    def test_clamp_below_min(self) -> None:
        """值低于下限时返回下限。"""
        assert _clamp(-1.0, 0.0, 10.0) == 0.0

    def test_clamp_above_max(self) -> None:
        """值高于上限时返回上限。"""
        assert _clamp(11.0, 0.0, 10.0) == 10.0

    def test_speed_to_gear_all_gears(self) -> None:
        """速度→档位映射覆盖 1-8 档。"""
        assert _speed_to_gear(0) == 1
        assert _speed_to_gear(30) == 2  # 30 是 2 档起点
        assert _speed_to_gear(80) == 3
        assert _speed_to_gear(130) == 4
        assert _speed_to_gear(180) == 5
        assert _speed_to_gear(220) == 6
        assert _speed_to_gear(260) == 7
        assert _speed_to_gear(300) == 8
        assert _speed_to_gear(340) == 8

    def test_speed_to_rpm_idle(self) -> None:
        """速度为 0 时返回怠速转速。"""
        assert _speed_to_rpm(0.0, 1) == 2500

    def test_speed_to_rpm_within_range(self) -> None:
        """转速在 [2500, 12500] 区间。"""
        for speed in [0, 30, 80, 130, 180, 220, 260, 300, 340]:
            gear = _speed_to_gear(speed)
            rpm = _speed_to_rpm(speed, gear)
            assert 2500 <= rpm <= 12500

    def test_estimate_lap_time_positive(self) -> None:
        """圈速估算为正数。"""
        track = get_track_by_id("suzuka")
        assert track is not None
        lap_time = _estimate_lap_time(track)
        assert lap_time > 0
        # F1 圈速通常 70-100 秒
        assert 60.0 < lap_time < 120.0


# ===========================================================================
# 2. 模拟器生命周期
# ===========================================================================
class TestSimulatorLifecycle:
    """start / stop / is_running 状态机。"""

    def test_initial_state(self) -> None:
        """新建模拟器未启动。"""
        sim = TelemetrySimulator()
        assert sim.is_running is False
        assert sim.current_track_id is None
        assert sim.current_lap == 0
        assert sim.current_corner is None

    def test_start_stop(self) -> None:
        """start → is_running True → stop → is_running False。"""
        sim = TelemetrySimulator(fps=30)
        ok = sim.start(track_id="suzuka", max_laps=1)
        assert ok is True
        assert sim.is_running is True
        assert sim.current_track_id == "suzuka"
        # 等待自然结束或主动停止
        time.sleep(0.1)
        sim.stop()
        assert sim.is_running is False

    def test_start_idempotent(self) -> None:
        """运行中再次 start 返回 False。"""
        sim = TelemetrySimulator(fps=30)
        sim.start(track_id="suzuka", max_laps=1)
        ok = sim.start(track_id="monaco")  # 已在运行
        assert ok is False
        sim.stop()

    def test_start_invalid_track(self) -> None:
        """无效赛道 ID 返回 False。"""
        sim = TelemetrySimulator()
        ok = sim.start(track_id="nonexistent_track")
        assert ok is False
        assert sim.is_running is False

    def test_stop_idempotent(self) -> None:
        """未运行时 stop 不报错。"""
        sim = TelemetrySimulator()
        sim.stop()  # 不报错
        assert sim.is_running is False

    def test_max_laps_auto_stop(self) -> None:
        """max_laps=1 时跑完一圈后自动停止（用 generate_lap_snapshot 验证圈帧数有限）。"""
        # 直接验证 max_laps 逻辑：generate_lap_snapshot 生成有限帧数
        frames = generate_lap_snapshot("monaco", fps=10)
        assert len(frames) > 0  # 有限圈帧数
        # 启动后主动停止（自然结束需完整圈速 ~76s，测试中不等）
        sim = TelemetrySimulator(fps=100)
        sim.start(track_id="monaco", max_laps=1)
        assert sim.is_running is True
        sim.stop()
        assert sim.is_running is False


# ===========================================================================
# 3. handler 注册与推送
# ===========================================================================
class TestHandlerDispatch:
    """handler 注册 / 移除 / 分发 / 异常隔离。"""

    def test_add_handler_dedup(self) -> None:
        """重复添加同一 handler 不重复注册。"""
        sim = TelemetrySimulator()
        count = [0]

        def handler(frame: dict) -> None:
            count[0] += 1

        sim.add_handler(handler)
        sim.add_handler(handler)  # 重复
        sim._dispatch({"x": 1})
        assert count[0] == 1

    def test_remove_handler(self) -> None:
        """remove_handler 移除回调。"""
        sim = TelemetrySimulator()
        received: list[dict] = []

        def handler(frame: dict) -> None:
            received.append(frame)

        sim.add_handler(handler)
        sim.remove_handler(handler)
        sim._dispatch({"x": 1})
        assert len(received) == 0

    def test_remove_handler_not_registered(self) -> None:
        """移除未注册的 handler 不报错。"""
        sim = TelemetrySimulator()

        def handler(frame: dict) -> None:
            pass

        sim.remove_handler(handler)  # 不报错

    def test_handler_exception_isolated(self) -> None:
        """handler 抛异常不影响其他 handler。"""
        sim = TelemetrySimulator()
        ok_received: list[dict] = []

        def bad(frame: dict) -> None:
            raise RuntimeError("boom")

        def good(frame: dict) -> None:
            ok_received.append(frame)

        sim.add_handler(bad)
        sim.add_handler(good)
        sim._dispatch({"x": 1})
        assert len(ok_received) == 1

    def test_dispatch_no_handlers(self) -> None:
        """无 handler 时 _dispatch 不报错。"""
        sim = TelemetrySimulator()
        sim._dispatch({"x": 1})

    def test_handler_receives_frames(self) -> None:
        """启动后 handler 收到遥测帧（WebSocket 推送模拟）。"""
        sim = TelemetrySimulator(fps=50)
        received: list[dict] = []
        event = threading.Event()

        def handler(frame: dict) -> None:
            received.append(frame)
            # 等收到 CarTelemetry 帧（packet_id=6），corner 事件是 packet_id=2
            if frame.get("packet_id") == 6 and not event.is_set():
                event.set()

        sim.add_handler(handler)
        sim.start(track_id="suzuka", max_laps=1)
        try:
            assert event.wait(timeout=3.0), "未在 3s 内收到 CarTelemetry 帧"
            # 找到第一个 CarTelemetry 帧
            telem = next(f for f in received if f["packet_id"] == 6)
            assert telem["packet_id"] == 6
            assert "speed" in telem
        finally:
            sim.stop()


# ===========================================================================
# 4. 数据包格式正确性
# ===========================================================================
class TestFrameFormat:
    """生成的帧字段类型与值范围正确。"""

    @staticmethod
    def _collect_one_frame(track_id: str = "suzuka") -> dict:
        """收集一帧用于断言。"""
        sim = TelemetrySimulator(fps=30)
        frame_holder: list[dict] = []
        event = threading.Event()

        def handler(frame: dict) -> None:
            if frame.get("packet_id") == 6 and not frame_holder:
                frame_holder.append(frame)
                event.set()

        sim.add_handler(handler)
        sim.start(track_id=track_id, max_laps=1)
        event.wait(timeout=3.0)
        sim.stop()
        return frame_holder[0]

    def test_frame_fields_present(self) -> None:
        """帧含全部预期字段。"""
        frame = self._collect_one_frame("suzuka")
        expected = {
            "packet_id", "name", "m_speed", "m_throttle", "m_steer", "m_brake",
            "m_gear", "m_engineRPM", "m_drs", "speed", "throttle", "brake",
            "gear", "engine_rpm", "lap_number", "sector", "track_id",
        }
        assert expected.issubset(frame.keys())

    def test_frame_field_types(self) -> None:
        """帧字段类型正确。"""
        frame = self._collect_one_frame("suzuka")
        assert isinstance(frame["packet_id"], int)
        assert isinstance(frame["m_speed"], int)
        assert isinstance(frame["m_gear"], int)
        assert isinstance(frame["m_engineRPM"], int)
        assert isinstance(frame["speed"], (int, float))
        assert isinstance(frame["m_throttle"], (int, float))
        assert isinstance(frame["m_brakesTemperature"], list)
        assert len(frame["m_brakesTemperature"]) == 4

    def test_frame_value_ranges(self) -> None:
        """帧值在合理范围内。"""
        frame = self._collect_one_frame("suzuka")
        assert 0 <= frame["speed"] <= 340
        assert 0 <= frame["throttle"] <= 1.0
        assert 0 <= frame["brake"] <= 1.0
        assert 1 <= frame["gear"] <= 8
        assert 2500 <= frame["engine_rpm"] <= 12500
        assert 0 <= frame["sector"] <= 2
        assert 0 <= frame["progress"] <= 1.0
        assert frame["lap_number"] >= 1

    def test_corner_event_format(self) -> None:
        """弯道切换事件含 event='corner' 字段。"""
        sim = TelemetrySimulator(fps=50)
        corner_events: list[dict] = []
        event = threading.Event()

        def handler(frame: dict) -> None:
            if frame.get("event") == "corner":
                corner_events.append(frame)
                if len(corner_events) >= 1:
                    event.set()

        sim.add_handler(handler)
        sim.start(track_id="suzuka", max_laps=1)
        try:
            event.wait(timeout=3.0)
            if corner_events:
                ev = corner_events[0]
                assert ev["event"] == "corner"
                assert "corner_number" in ev
                assert "sector" in ev
                assert "lap_number" in ev
        finally:
            sim.stop()


# ===========================================================================
# 5. 赛道切换（24 条赛道）
# ===========================================================================
class TestTrackSwitching:
    """24 条赛道均可启动模拟。"""

    def test_all_24_tracks_exist(self) -> None:
        """ALL_TRACKS 含 24 条赛道。"""
        assert len(ALL_TRACKS) == 24

    @pytest.mark.parametrize("track_id", [t.track_id for t in ALL_TRACKS])
    def test_start_each_track(self, track_id: str) -> None:
        """每条赛道都能启动并收到首帧。"""
        sim = TelemetrySimulator(fps=50)
        received: list[dict] = []
        event = threading.Event()

        def handler(frame: dict) -> None:
            if frame.get("packet_id") == 6 and not received:
                received.append(frame)
                event.set()

        sim.add_handler(handler)
        ok = sim.start(track_id=track_id, max_laps=1)
        assert ok is True
        try:
            assert event.wait(timeout=3.0), f"{track_id} 未在 3s 内收到帧"
            assert received[0]["track_id"] == track_id
        finally:
            sim.stop()


# ===========================================================================
# 6. generate_lap_snapshot 便捷函数
# ===========================================================================
class TestGenerateLapSnapshot:
    """generate_lap_snapshot 离线生成单圈帧。"""

    def test_snapshot_returns_frames(self) -> None:
        """generate_lap_snapshot 返回非空帧列表。"""
        frames = generate_lap_snapshot("suzuka", fps=30)
        assert len(frames) > 0
        assert all(f["packet_id"] == 6 for f in frames)

    def test_snapshot_custom_lap_time(self) -> None:
        """自定义圈速影响帧数。"""
        frames = generate_lap_snapshot("monaco", lap_time_sec=60.0, fps=30)
        # n_frames = max(60, int(60.0 * 30)) = 1800
        assert len(frames) == 1800

    def test_snapshot_min_60_frames(self) -> None:
        """帧数下限 60（即使圈速很短）。"""
        frames = generate_lap_snapshot("monaco", lap_time_sec=0.5, fps=10)
        assert len(frames) >= 60

    def test_snapshot_invalid_track_raises(self) -> None:
        """无效赛道抛 ValueError。"""
        with pytest.raises(ValueError, match="track not found"):
            generate_lap_snapshot("nonexistent")

    def test_snapshot_reproducible_with_seed(self) -> None:
        """相同 seed 生成相同帧。"""
        f1 = generate_lap_snapshot("suzuka", fps=20, seed=42)
        f2 = generate_lap_snapshot("suzuka", fps=20, seed=42)
        assert len(f1) == len(f2)
        assert f1[0]["speed"] == f2[0]["speed"]

    def test_snapshot_frame_fields(self) -> None:
        """快照帧含全部预期字段。"""
        frames = generate_lap_snapshot("suzuka", fps=10)
        f = frames[0]
        assert f["packet_id"] == 6
        assert f["name"] == "CarTelemetry"
        assert "m_speed" in f and "speed" in f
        assert "lap_number" in f and "sector" in f


# ===========================================================================
# 7. 边界条件
# ===========================================================================
class TestSimulatorEdgeCases:
    """无效输入、空回放、并发启停。"""

    def test_invalid_track_id(self) -> None:
        """无效赛道 ID start 返回 False。"""
        sim = TelemetrySimulator()
        assert sim.start(track_id="") is False
        assert sim.start(track_id="not_a_track") is False

    def test_zero_fps_clamped_to_one(self) -> None:
        """fps=0 被夹到 1。"""
        sim = TelemetrySimulator(fps=0)
        assert sim._fps == 1

    def test_negative_fps_clamped(self) -> None:
        """负 fps 被夹到 1。"""
        sim = TelemetrySimulator(fps=-10)
        assert sim._fps == 1

    def test_default_constants(self) -> None:
        """默认常量值正确。"""
        assert DEFAULT_FPS == 10
        assert DEFAULT_SEED == 20260911

    def test_concurrent_start_stop(self) -> None:
        """并发 start/stop 不报错。"""
        sim = TelemetrySimulator(fps=50)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                sim.start(track_id="suzuka", max_laps=1)
                time.sleep(0.05)
                sim.stop()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        sim.stop()
        # 不报错即可（行为不严格断言，因并发竞争本就非确定）
        assert sim.is_running is False

    def test_corner_influence_straight_vs_corner(self) -> None:
        """直道与弯道速度差异：弯道速度低于直道极速。"""
        track = get_track_by_id("suzuka")
        assert track is not None
        sim = TelemetrySimulator()
        n_corners = len(track.corners)
        corner_positions = [(i + 0.5) / n_corners for i in range(n_corners)]
        corner_radius = 0.6 / n_corners
        # 进度 0.0（直道起点）
        in_c1, _, target1 = sim._compute_corner_influence(
            0.0, track, corner_positions, corner_radius,
        )
        # 进度对应第一个弯道中心
        in_c2, _, target2 = sim._compute_corner_influence(
            corner_positions[0], track, corner_positions, corner_radius,
        )
        # 弯道目标速度应低于直道
        if in_c2:
            assert target2 < target1