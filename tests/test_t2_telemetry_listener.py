"""T2 遥测监听器 + 最新帧缓存 单元测试。

覆盖验收标准：
1. TelemetryListener handler 注册/移除/分发（含 handler 异常不崩溃）
2. TelemetryListener 生命周期（start/stop 幂等、is_running 状态）
3. TelemetryListener._process 容错（短包/未知包/解析异常均不崩溃）
4. TelemetryStream update/get_latest/get_all_latest/clear（线程安全）
5. TelemetryStream 不支持的 packet_id 静默忽略
6. 边界条件：空包、截断包、错误包格式、最大值/最小值
"""

from __future__ import annotations

import socket
import struct
import threading
import time

from setup_tuner.telemetry.listener import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    MAX_PACKET_SIZE,
    TelemetryListener,
)
from setup_tuner.telemetry.packets import (
    HEADER_FORMAT,
    SUPPORTED_PACKET_IDS,
)
from setup_tuner.telemetry.stream import TelemetryStream


# ===========================================================================
# 辅助：构造合法的 Session 包（packet_id=1）
# ===========================================================================
def _build_session_packet(track_id: int = 2) -> bytes:
    """构造一个合法的 Packet 1 (Session) 用于 listener 测试。"""
    header = struct.pack(
        HEADER_FORMAT,
        2026, 26, 1, 0, 1, 1,  # packet_id=1
        0x1234_5678_9ABC_DEF0, 12.5, 100, 200, 0, 255,
    )
    # Session prefix: 16 fields + 21 marshal zones + 3 tail bytes
    prefix_core = struct.pack(
        "<BbbBHBbBHHBBBBBB",
        0, 25, 22, 58, 5807, 0, track_id, 1, 1800, 3600, 60, 0, 0, 0, 0, 21,
    )
    marshal_zones = b"".join(struct.pack("<fb", 0.0, 0) for _ in range(21))
    tail = struct.pack("<BBB", 0, 0, 0)  # num_wfs=0
    return header + prefix_core + marshal_zones + tail


def _build_header_only(packet_id: int = 0) -> bytes:
    """构造仅 29 字节包头（packet_id=0 未知，parse_packet 返回 None）。"""
    return struct.pack(
        HEADER_FORMAT,
        2026, 26, 1, 0, 1, packet_id,
        0, 0.0, 0, 0, 0, 255,
    )


# ===========================================================================
# 1. TelemetryListener — handler 注册/移除/分发
# ===========================================================================
class TestListenerHandlers:
    """handler 注册、移除、分发逻辑。"""

    def test_add_handler(self) -> None:
        """add_handler 注册回调。"""
        listener = TelemetryListener()
        received: list[dict] = []

        def handler(parsed: dict) -> None:
            received.append(parsed)

        listener.add_handler(handler)
        # 直接调用 _dispatch 测试分发
        listener._dispatch({"packet_id": 1, "test": True})
        assert len(received) == 1
        assert received[0]["test"] is True

    def test_add_handler_dedup(self) -> None:
        """重复添加同一 handler 不重复注册。"""
        listener = TelemetryListener()
        count = [0]

        def handler(parsed: dict) -> None:
            count[0] += 1

        listener.add_handler(handler)
        listener.add_handler(handler)  # 重复
        listener._dispatch({"x": 1})
        assert count[0] == 1

    def test_remove_handler(self) -> None:
        """remove_handler 移除已注册回调。"""
        listener = TelemetryListener()
        received: list[dict] = []

        def handler(parsed: dict) -> None:
            received.append(parsed)

        listener.add_handler(handler)
        listener.remove_handler(handler)
        listener._dispatch({"x": 1})
        assert len(received) == 0

    def test_remove_handler_not_registered(self) -> None:
        """移除未注册的 handler 不报错。"""
        listener = TelemetryListener()

        def handler(parsed: dict) -> None:
            pass

        listener.remove_handler(handler)  # 不报错

    def test_dispatch_handler_exception_does_not_crash(self) -> None:
        """handler 抛异常不影响其他 handler 和监听线程。"""
        listener = TelemetryListener()
        ok_received: list[dict] = []

        def bad_handler(parsed: dict) -> None:
            raise RuntimeError("handler boom")

        def good_handler(parsed: dict) -> None:
            ok_received.append(parsed)

        listener.add_handler(bad_handler)
        listener.add_handler(good_handler)
        # bad_handler 抛异常，good_handler 仍应被调用
        listener._dispatch({"x": 1})
        assert len(ok_received) == 1

    def test_dispatch_no_handlers(self) -> None:
        """无 handler 时 _dispatch 不报错。"""
        listener = TelemetryListener()
        listener._dispatch({"x": 1})  # 不报错


# ===========================================================================
# 2. TelemetryListener — 生命周期
# ===========================================================================
class TestListenerLifecycle:
    """start/stop 幂等性与 is_running 状态。"""

    def test_is_running_false_before_start(self) -> None:
        """未启动时 is_running 为 False。"""
        listener = TelemetryListener()
        assert listener.is_running is False

    def test_start_stop_lifecycle(self) -> None:
        """start → is_running True → stop → is_running False。"""
        # 使用可用端口（0 = OS 分配）
        _listener = TelemetryListener(host="127.0.0.1", port=0)
        # port=0 时 bind 会分配端口，但我们的代码直接 bind，需改用实际端口
        # 找一个可用端口
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        listener2 = TelemetryListener(host="127.0.0.1", port=port)
        listener2.start()
        assert listener2.is_running is True
        listener2.stop()
        assert listener2.is_running is False

    def test_start_idempotent(self) -> None:
        """多次 start 不报错（幂等）。"""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        listener = TelemetryListener(host="127.0.0.1", port=port)
        listener.start()
        listener.start()  # 幂等，不报错
        assert listener.is_running is True
        listener.stop()

    def test_stop_idempotent(self) -> None:
        """未启动时 stop 不报错（幂等）。"""
        listener = TelemetryListener()
        listener.stop()  # 不报错
        assert listener.is_running is False

    def test_stop_after_start_idempotent(self) -> None:
        """多次 stop 不报错。"""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        listener = TelemetryListener(host="127.0.0.1", port=port)
        listener.start()
        listener.stop()
        listener.stop()  # 幂等
        assert listener.is_running is False


# ===========================================================================
# 3. TelemetryListener — _process 容错
# ===========================================================================
class TestListenerProcess:
    """_process 方法容错：短包/未知包/解析异常均不崩溃。"""

    def test_process_valid_packet(self) -> None:
        """合法包应分发到 handler。"""
        listener = TelemetryListener()
        received: list[dict] = []

        def handler(parsed: dict) -> None:
            received.append(parsed)

        listener.add_handler(handler)
        packet = _build_session_packet(track_id=7)
        listener._process(packet)
        assert len(received) == 1
        assert received[0]["packet_id"] == 1
        assert received[0]["m_trackId"] == 7

    def test_process_short_packet_skipped(self) -> None:
        """短包（< 29 字节）应跳过不崩溃。"""
        listener = TelemetryListener()
        received: list[dict] = []

        def handler(parsed: dict) -> None:
            received.append(parsed)

        listener.add_handler(handler)
        listener._process(b"\x00" * 10)  # 短包
        listener._process(b"")  # 空包
        assert len(received) == 0

    def test_process_unknown_packet_skipped(self) -> None:
        """未知 packetId 应跳过（parse_packet 返回 None）。"""
        listener = TelemetryListener()
        received: list[dict] = []

        def handler(parsed: dict) -> None:
            received.append(parsed)

        listener.add_handler(handler)
        # packet_id=0 (Motion) 不在支持的 6 类范围内
        data = _build_header_only(packet_id=0) + b"\x00" * 64
        listener._process(data)
        assert len(received) == 0

    def test_process_truncated_body_skipped(self) -> None:
        """包体截断（header 合法但 body 不足）应跳过不崩溃。"""
        listener = TelemetryListener()
        received: list[dict] = []

        def handler(parsed: dict) -> None:
            received.append(parsed)

        listener.add_handler(handler)
        # packet_id=1 (Session) 但 body 远不足
        data = _build_header_only(packet_id=1) + b"\x00" * 5
        listener._process(data)
        assert len(received) == 0


# ===========================================================================
# 4. TelemetryListener — 端到端 UDP 收包
# ===========================================================================
class TestListenerUdpReceive:
    """端到端 UDP 收包：发送 UDP 包 → listener 接收并分发。"""

    def test_receive_and_dispatch(self) -> None:
        """发送合法 UDP 包，listener 应接收并分发到 handler。"""
        # 找可用端口
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        listener = TelemetryListener(host="127.0.0.1", port=port)
        received: list[dict] = []
        event = threading.Event()

        def handler(parsed: dict) -> None:
            received.append(parsed)
            event.set()

        listener.add_handler(handler)
        listener.start()

        try:
            # 发送合法 Session 包
            packet = _build_session_packet(track_id=3)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                sender.sendto(packet, ("127.0.0.1", port))

            # 等待接收（最多 2 秒）
            assert event.wait(timeout=2.0), "listener 未在 2s 内收到包"
            assert len(received) == 1
            assert received[0]["packet_id"] == 1
            assert received[0]["m_trackId"] == 3
        finally:
            listener.stop()

    def test_receive_short_packet_no_crash(self) -> None:
        """发送短包，listener 不崩溃且继续运行。"""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        listener = TelemetryListener(host="127.0.0.1", port=port)
        listener.start()

        try:
            # 发送短包
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                sender.sendto(b"\x00" * 5, ("127.0.0.1", port))
                time.sleep(0.1)

            # listener 仍应运行
            assert listener.is_running is True

            # 发送合法包仍能接收
            received: list[dict] = []
            event = threading.Event()

            def handler(parsed: dict) -> None:
                received.append(parsed)
                event.set()

            listener.add_handler(handler)
            packet = _build_session_packet(track_id=5)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                sender.sendto(packet, ("127.0.0.1", port))
            assert event.wait(timeout=2.0)
            assert received[0]["m_trackId"] == 5
        finally:
            listener.stop()


# ===========================================================================
# 5. TelemetryStream — 基本操作
# ===========================================================================
class TestTelemetryStream:
    """最新帧内存缓存测试。"""

    def test_update_and_get_latest(self) -> None:
        """update 后 get_latest 返回该帧的副本。"""
        stream = TelemetryStream()
        data = {"m_speed": 300, "m_gear": 7}
        stream.update(6, data)
        latest = stream.get_latest(6)
        assert latest is not None
        assert latest["m_speed"] == 300
        assert latest["m_gear"] == 7

    def test_get_latest_returns_copy(self) -> None:
        """get_latest 返回的是副本，修改不影响缓存。"""
        stream = TelemetryStream()
        stream.update(6, {"m_speed": 200})
        latest = stream.get_latest(6)
        assert latest is not None
        latest["m_speed"] = 999
        # 缓存不受影响
        latest2 = stream.get_latest(6)
        assert latest2 is not None
        assert latest2["m_speed"] == 200

    def test_get_latest_no_data_returns_none(self) -> None:
        """未 update 的 packet_id 返回 None。"""
        stream = TelemetryStream()
        assert stream.get_latest(6) is None

    def test_update_overwrites(self) -> None:
        """同 packet_id 多次 update 覆盖最新帧。"""
        stream = TelemetryStream()
        stream.update(6, {"m_speed": 100})
        stream.update(6, {"m_speed": 200})
        latest = stream.get_latest(6)
        assert latest is not None
        assert latest["m_speed"] == 200

    def test_update_unsupported_packet_id_ignored(self) -> None:
        """不在 SUPPORTED_PACKET_IDS 中的 packet_id 静默忽略。"""
        stream = TelemetryStream()
        stream.update(99, {"x": 1})  # 99 不支持
        assert stream.get_latest(99) is None

    def test_get_all_latest(self) -> None:
        """get_all_latest 返回所有有数据的帧快照。"""
        stream = TelemetryStream()
        stream.update(1, {"m_trackId": 2})
        stream.update(6, {"m_speed": 300})
        all_latest = stream.get_all_latest()
        assert 1 in all_latest
        assert 6 in all_latest
        assert all_latest[1]["m_trackId"] == 2
        assert all_latest[6]["m_speed"] == 300
        # 未更新的包不在结果中
        assert 2 not in all_latest

    def test_get_all_latest_empty(self) -> None:
        """无数据时 get_all_latest 返回空字典。"""
        stream = TelemetryStream()
        assert stream.get_all_latest() == {}

    def test_get_all_latest_returns_deep_copy(self) -> None:
        """get_all_latest 返回深拷贝，修改不影响缓存。"""
        stream = TelemetryStream()
        stream.update(6, {"m_speed": 200})
        all_latest = stream.get_all_latest()
        all_latest[6]["m_speed"] = 999
        # 缓存不受影响
        latest = stream.get_latest(6)
        assert latest is not None
        assert latest["m_speed"] == 200

    def test_clear(self) -> None:
        """clear 清空所有缓存。"""
        stream = TelemetryStream()
        stream.update(1, {"x": 1})
        stream.update(6, {"y": 2})
        stream.clear()
        assert stream.get_latest(1) is None
        assert stream.get_latest(6) is None
        assert stream.get_all_latest() == {}

    def test_clear_empty(self) -> None:
        """对空缓存 clear 不报错。"""
        stream = TelemetryStream()
        stream.clear()
        assert stream.get_all_latest() == {}

    def test_all_supported_ids_have_slots(self) -> None:
        """所有 SUPPORTED_PACKET_IDS 都有缓存槽位。"""
        stream = TelemetryStream()
        for pid in SUPPORTED_PACKET_IDS:
            stream.update(pid, {"id": pid})
            latest = stream.get_latest(pid)
            assert latest is not None
            assert latest["id"] == pid


# ===========================================================================
# 6. TelemetryStream — 线程安全
# ===========================================================================
class TestTelemetryStreamThreadSafety:
    """多线程并发读写不报错。"""

    def test_concurrent_update_read(self) -> None:
        """多线程并发 update + get_latest 不报错。"""
        stream = TelemetryStream()
        errors: list[Exception] = []

        def writer():
            try:
                for i in range(100):
                    stream.update(6, {"m_speed": i})
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    stream.get_latest(6)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ===========================================================================
# 7. 常量校验
# ===========================================================================
class TestListenerConstants:
    """监听器常量校验。"""

    def test_default_host(self) -> None:
        assert DEFAULT_HOST == "127.0.0.1"

    def test_default_port(self) -> None:
        assert DEFAULT_PORT == 20777

    def test_max_packet_size(self) -> None:
        assert MAX_PACKET_SIZE == 2048