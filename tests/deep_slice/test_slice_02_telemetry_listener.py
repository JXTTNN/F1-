"""切片 2 深度测试：UDP 监听器与遥测流（listener + stream）。

覆盖 setup_tuner.telemetry.listener.TelemetryListener（后台线程接收 + 解析 + 回调）
与 setup_tuner.telemetry.stream.TelemetryStream（最新帧内存缓存）。

5 种测试方式：
    1. unit     — handler 注册/移除、stream update/get 正确性
    2. boundary — 短包/未知包/异常 handler 容错、空 stream、未运行 stop
    3. property — 幂等性（start/stop 多次）、确定性、线程安全快照一致性
    4. static   — 类型约束、常量约束、SUPPORTED_PACKET_IDS 一致性
    5. smoke    — 真实 UDP socket 收发链路、真实 handler 回调
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

from setup_tuner.telemetry.listener import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    MAX_PACKET_SIZE,
    TelemetryListener,
)
from setup_tuner.telemetry.packets import (
    SUPPORTED_PACKET_IDS,
    PacketTooShortError,
)
from setup_tuner.telemetry.stream import TelemetryStream

from .conftest import build_header, build_session_body


# ===========================================================================
# 1. 单元测试 (unit) — handler 注册/移除、stream update/get 正确性
# ===========================================================================
class TestUnit:
    """单元测试：验证 listener handler 管理与 stream 缓存的正常行为。"""

    def test_unit_listener_add_handler(self) -> None:
        """add_handler 注册后 handler 应在列表中。"""
        listener = TelemetryListener()
        handler = lambda parsed: None  # noqa: E731
        listener.add_handler(handler)
        assert handler in listener._handlers  # noqa: SLF001

    def test_unit_listener_add_handler_dedup(self) -> None:
        """add_handler 对同一 handler 多次注册应去重。"""
        listener = TelemetryListener()
        handler = lambda parsed: None  # noqa: E731
        listener.add_handler(handler)
        listener.add_handler(handler)
        assert len(listener._handlers) == 1  # noqa: SLF001

    def test_unit_listener_remove_handler(self) -> None:
        """remove_handler 应正确移除已注册 handler。"""
        listener = TelemetryListener()
        handler = lambda parsed: None  # noqa: E731
        listener.add_handler(handler)
        listener.remove_handler(handler)
        assert handler not in listener._handlers  # noqa: SLF001

    def test_unit_listener_remove_unregistered_no_error(self) -> None:
        """remove_handler 对未注册 handler 不应报错。"""
        listener = TelemetryListener()
        handler = lambda parsed: None  # noqa: E731
        listener.remove_handler(handler)  # 不抛异常

    def test_unit_stream_update_and_get(self) -> None:
        """stream.update 后 get_latest 应返回更新数据。"""
        stream = TelemetryStream()
        data = {"m_trackId": 5, "m_weather": 1}
        stream.update(1, data)
        result = stream.get_latest(1)
        assert result is not None
        assert result["m_trackId"] == 5

    def test_unit_stream_get_latest_none_for_unreceived(self) -> None:
        """stream.get_latest 对未收到的 packet_id 应返回 None。"""
        stream = TelemetryStream()
        assert stream.get_latest(6) is None

    def test_unit_stream_get_all_latest(self) -> None:
        """stream.get_all_latest 应返回所有已缓存且有数据的包。"""
        stream = TelemetryStream()
        stream.update(1, {"a": 1})
        stream.update(6, {"b": 2})
        result = stream.get_all_latest()
        assert 1 in result and 6 in result
        assert result[1]["a"] == 1
        assert result[6]["b"] == 2

    def test_unit_stream_clear(self) -> None:
        """stream.clear 应清空所有缓存。"""
        stream = TelemetryStream()
        stream.update(1, {"a": 1})
        stream.clear()
        assert stream.get_latest(1) is None
        assert stream.get_all_latest() == {}

    def test_unit_stream_returns_copy(self) -> None:
        """stream.get_latest 返回的是副本，修改不影响缓存。"""
        stream = TelemetryStream()
        stream.update(1, {"x": 10})
        result = stream.get_latest(1)
        assert result is not None
        result["x"] = 999
        # 缓存不受影响
        again = stream.get_latest(1)
        assert again is not None
        assert again["x"] == 10


# ===========================================================================
# 2. 边界/异常测试 (boundary) — 短包/未知包/异常 handler 容错
# ===========================================================================
class TestBoundary:
    """边界/异常测试：验证容错降级与边界场景。"""

    def test_boundary_stream_unsupported_packet_id_ignored(self) -> None:
        """stream.update 对不在 SUPPORTED_PACKET_IDS 的 packet_id 静默忽略。"""
        stream = TelemetryStream()
        stream.update(99, {"a": 1})  # 99 不在支持范围
        assert stream.get_latest(99) is None

    def test_boundary_stream_overwrite(self) -> None:
        """stream.update 同一 packet_id 多次应覆盖写（仅保留最新）。"""
        stream = TelemetryStream()
        stream.update(1, {"v": 1})
        stream.update(1, {"v": 2})
        stream.update(1, {"v": 3})
        result = stream.get_latest(1)
        assert result is not None
        assert result["v"] == 3

    def test_boundary_listener_process_short_packet_no_crash(self) -> None:
        """listener._process 对短包应容错跳过不崩溃。"""
        listener = TelemetryListener()
        # 短包（< 29 字节）应被捕获 PacketTooShortError 并跳过
        listener._process(b"\x00" * 10)  # 不抛异常

    def test_boundary_listener_process_unknown_packet_no_crash(self) -> None:
        """listener._process 对未知 packetId 应跳过不崩溃。"""
        listener = TelemetryListener()
        data = build_header(packet_id=99) + b"\x00" * 64
        listener._process(data)  # 不抛异常

    def test_boundary_listener_dispatch_handler_exception_no_crash(self) -> None:
        """handler 抛异常不应中断监听器。"""
        listener = TelemetryListener()
        call_count = {"n": 0}

        def bad_handler(_parsed: dict) -> None:
            call_count["n"] += 1
            raise RuntimeError("handler 故障")

        def good_handler(_parsed: dict) -> None:
            call_count["n"] += 100

        listener.add_handler(bad_handler)
        listener.add_handler(good_handler)
        # 分发一个解析后的 dict
        listener._dispatch({"packet_id": 1})  # 不抛异常
        # 两个 handler 都被调用（bad 抛异常后 good 仍执行）
        assert call_count["n"] == 101

    def test_boundary_listener_stop_when_not_running(self) -> None:
        """未启动时 stop 应安全返回（幂等）。"""
        listener = TelemetryListener()
        listener.stop()  # 不抛异常

    def test_boundary_listener_start_when_already_running(self) -> None:
        """已运行时 start 应安全返回（幂等）。"""
        listener = TelemetryListener(host="127.0.0.1", port=20790)
        listener.start()
        assert listener.is_running
        listener.start()  # 再次 start 不抛异常
        assert listener.is_running
        listener.stop()

    def test_boundary_stream_init_all_none(self) -> None:
        """stream 初始化后所有支持的 packet_id 槽位为 None。"""
        stream = TelemetryStream()
        for pid in SUPPORTED_PACKET_IDS:
            assert stream.get_latest(pid) is None


# ===========================================================================
# 3. 属性不变量测试 (property) — 幂等性、确定性、线程安全
# ===========================================================================
class TestProperty:
    """属性不变量测试：验证幂等性与确定性。"""

    def test_property_start_stop_idempotent(self) -> None:
        """幂等性：多次 start/stop 应保持状态一致。"""
        listener = TelemetryListener(host="127.0.0.1", port=20791)
        listener.start()
        listener.start()
        assert listener.is_running
        listener.stop()
        listener.stop()
        assert not listener.is_running

    def test_property_stream_update_idempotent_same_data(self) -> None:
        """幂等性：相同数据多次 update 后 get_latest 结果一致。"""
        stream = TelemetryStream()
        data = {"v": 1}
        stream.update(1, data)
        r1 = stream.get_latest(1)
        stream.update(1, data)
        r2 = stream.get_latest(1)
        assert r1 == r2

    def test_property_stream_get_all_latest_snapshot(self) -> None:
        """不变量：get_all_latest 返回快照，修改不影响后续读取。"""
        stream = TelemetryStream()
        stream.update(1, {"x": 1})
        stream.update(6, {"y": 2})
        snap = stream.get_all_latest()
        snap[1]["x"] = 999
        # 再次读取不受影响
        again = stream.get_all_latest()
        assert again[1]["x"] == 1

    def test_property_listener_is_running_reflects_state(self) -> None:
        """不变量：is_running 准确反映运行状态。"""
        listener = TelemetryListener(host="127.0.0.1", port=20792)
        assert not listener.is_running
        listener.start()
        assert listener.is_running
        listener.stop()
        assert not listener.is_running

    def test_property_stream_clear_idempotent(self) -> None:
        """幂等性：多次 clear 结果一致（全空）。"""
        stream = TelemetryStream()
        stream.update(1, {"a": 1})
        stream.clear()
        first = stream.get_all_latest()
        stream.clear()
        second = stream.get_all_latest()
        assert first == second == {}


# ===========================================================================
# 4. 静态分析 (static) — 类型约束、常量约束
# ===========================================================================
class TestStatic:
    """静态分析：验证常量与类型约束。"""

    def test_static_default_host(self) -> None:
        """常量约束：DEFAULT_HOST 为 127.0.0.1。"""
        assert DEFAULT_HOST == "127.0.0.1"

    def test_static_default_port(self) -> None:
        """常量约束：DEFAULT_PORT 为 20777（EA F1 默认 UDP 端口）。"""
        assert DEFAULT_PORT == 20777

    def test_static_max_packet_size(self) -> None:
        """常量约束：MAX_PACKET_SIZE >= 1400（F1 2026 最大包约 1.4KB）。"""
        assert MAX_PACKET_SIZE >= 1400

    def test_static_stream_only_caches_supported_ids(self) -> None:
        """不变量约束：stream 仅缓存 SUPPORTED_PACKET_IDS 中的 6 类包。"""
        stream = TelemetryStream()
        # 内部 cache 的 key 集合应等于 SUPPORTED_PACKET_IDS
        assert set(stream._cache.keys()) == set(SUPPORTED_PACKET_IDS)  # noqa: SLF001

    def test_static_listener_init_attributes(self) -> None:
        """类型约束：TelemetryListener 初始化后含必要属性。"""
        listener = TelemetryListener()
        assert hasattr(listener, "add_handler")
        assert hasattr(listener, "remove_handler")
        assert hasattr(listener, "start")
        assert hasattr(listener, "stop")
        assert hasattr(listener, "is_running")

    @pytest.mark.parametrize("pid", list(SUPPORTED_PACKET_IDS))
    def test_static_stream_has_slot_for_each_supported(self, pid: int) -> None:
        """参数化：stream 为每个支持的 packet_id 预留了槽位。"""
        stream = TelemetryStream()
        assert pid in stream._cache  # noqa: SLF001


# ===========================================================================
# 5. 实际运行冒烟 (smoke) — 真实 UDP socket 收发链路
# ===========================================================================
class TestSmoke:
    """实际运行冒烟：用真实 UDP socket 收发数据，验证完整链路。"""

    def test_smoke_real_udp_send_receive(self) -> None:
        """冒烟：真实 UDP socket 发送 Session 包 → listener 接收 → handler 回调。"""
        received: list[dict] = []
        ready = threading.Event()

        def handler(parsed: dict) -> None:
            received.append(parsed)
            ready.set()

        # 选一个不太可能冲突的端口
        port = 20793
        listener = TelemetryListener(host="127.0.0.1", port=port)
        listener.add_handler(handler)
        listener.start()

        try:
            # 构造真实 Session 包并发送
            body = build_session_body(track_id=7, weather=0)
            data = build_header(packet_id=1, player_car_index=0) + body
            time.sleep(0.05)  # 等待 listener bind 就绪
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                sender.sendto(data, ("127.0.0.1", port))
            # 等待接收（最多 2 秒）
            assert ready.wait(timeout=2.0), "未收到 UDP 包"
            assert len(received) == 1
            assert received[0]["packet_id"] == 1
            assert received[0]["m_trackId"] == 7
        finally:
            listener.stop()

    def test_smoke_real_udp_short_packet_skipped(self) -> None:
        """冒烟：真实发送短包 → listener 应跳过不崩溃，handler 不被调用。"""
        received: list[dict] = []

        def handler(parsed: dict) -> None:
            received.append(parsed)

        port = 20794
        listener = TelemetryListener(host="127.0.0.1", port=port)
        listener.add_handler(handler)
        listener.start()

        try:
            time.sleep(0.05)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                sender.sendto(b"\x00" * 10, ("127.0.0.1", port))
            time.sleep(0.3)  # 给 listener 时间处理
            assert len(received) == 0  # 短包被跳过
        finally:
            listener.stop()

    def test_smoke_stream_update_get_real_data(self) -> None:
        """冒烟：用真实解析数据走 stream update → get_latest 链路。"""
        from setup_tuner.telemetry.packets import parse_packet

        body = build_session_body(track_id=11, weather=2)
        data = build_header(packet_id=1) + body
        parsed = parse_packet(data)
        assert parsed is not None

        stream = TelemetryStream()
        stream.update(parsed["packet_id"], parsed)
        result = stream.get_latest(1)
        assert result is not None
        assert result["m_trackId"] == 11
        assert result["m_weather"] == 2

    def test_smoke_listener_to_stream_integration(self) -> None:
        """冒烟：listener handler 中调用 stream.update 的真实集成链路。"""

        stream = TelemetryStream()
        ready = threading.Event()

        def handler(parsed: dict) -> None:
            stream.update(parsed["packet_id"], parsed)
            ready.set()

        port = 20795
        listener = TelemetryListener(host="127.0.0.1", port=port)
        listener.add_handler(handler)
        listener.start()

        try:
            body = build_session_body(track_id=14)
            data = build_header(packet_id=1) + body
            time.sleep(0.05)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                sender.sendto(data, ("127.0.0.1", port))
            assert ready.wait(timeout=2.0)
            # stream 中应有 packet_id=1 的最新帧
            latest = stream.get_latest(1)
            assert latest is not None
            assert latest["m_trackId"] == 14
        finally:
            listener.stop()

    def test_smoke_multiple_packets_to_stream(self) -> None:
        """冒烟：连续发送多个不同类型包，stream 应缓存各自最新帧。"""

        stream = TelemetryStream()
        received_count = {"n": 0}
        ready = threading.Event()

        def handler(parsed: dict) -> None:
            stream.update(parsed["packet_id"], parsed)
            received_count["n"] += 1
            if received_count["n"] >= 2:
                ready.set()

        port = 20796
        listener = TelemetryListener(host="127.0.0.1", port=port)
        listener.add_handler(handler)
        listener.start()

        try:
            time.sleep(0.05)
            # 发送 Session 包
            session_data = build_header(packet_id=1) + build_session_body(track_id=3)
            # 发送 LapData 包
            from .conftest import build_lap_per_car
            lap_data = build_header(packet_id=2) + build_lap_per_car(last_lap_ms=88888)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                sender.sendto(session_data, ("127.0.0.1", port))
                sender.sendto(lap_data, ("127.0.0.1", port))
            assert ready.wait(timeout=2.0)
            all_latest = stream.get_all_latest()
            assert 1 in all_latest
            assert 2 in all_latest
            assert all_latest[1]["m_trackId"] == 3
            assert all_latest[2]["m_lastLapTimeInMS"] == 88888
        finally:
            listener.stop()

    def test_smoke_packet_too_short_error_is_value_error(self) -> None:
        """冒烟：PacketTooShortError 是 ValueError 子类（类型约束）。"""
        assert issubclass(PacketTooShortError, ValueError)