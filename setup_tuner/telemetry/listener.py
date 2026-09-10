"""UDP 遥测监听器（标准库 socket，后台线程，容错）。

在可配置端口（默认 ``127.0.0.1:20777``）监听 EA F1 2026 UDP 遥测数据包，
后台线程接收并解析，通过回调机制通知注册的 handler。

容错策略（对齐 FR-TEL-04）：
- **丢包**：UDP 本身允许丢包，不做重传，直接跳过。
- **乱序**：UDP 不保证顺序，按到达顺序处理，不重排。
- **短包**：捕获 :class:`PacketTooShortError`，记录日志后跳过，不崩溃。
- **未知包**：``parse_packet`` 返回 ``None``，跳过。
- **解析异常**：捕获所有 ``Exception``，记录日志后继续，保证线程不死。

使用方式::

    listener = TelemetryListener()
    listener.add_handler(my_handler)  # my_handler(parsed_dict) -> None
    listener.start()
    ...
    listener.stop()
"""

from __future__ import annotations

import contextlib
import logging
import socket
import threading
from collections.abc import Callable
from typing import Any

from .packets import PacketTooShortError, parse_packet

logger = logging.getLogger(__name__)

# 默认监听地址（EA F1 2026 默认 UDP 输出端口）
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 20777
# 单个 UDP 包最大接收字节数（F1 2026 最大包约 1.4KB，预留余量）
MAX_PACKET_SIZE = 2048
# 后台线程名
_THREAD_NAME = "f1opt-telemetry-listener"

# handler 类型：接收解析后的 dict
PacketHandler = Callable[[dict[str, Any]], None]


class TelemetryListener:
    """UDP 遥测监听器，后台线程接收 + 解析 + 回调。

    线程安全：handler 列表的增删通过内部锁保护；start/stop 可安全多次调用。
    socket 在 stop 时关闭以解除阻塞的 recvfrom。
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        max_packet_size: int = MAX_PACKET_SIZE,
    ) -> None:
        self._host = host
        self._port = port
        self._max_packet_size = max_packet_size
        self._handlers: list[PacketHandler] = []
        self._handlers_lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()

    # ------------------------------------------------------------------ #
    # handler 注册
    # ------------------------------------------------------------------ #
    def add_handler(self, handler: PacketHandler) -> None:
        """注册一个包处理回调。收到并成功解析的包会传入所有已注册 handler。"""
        with self._handlers_lock:
            if handler not in self._handlers:
                self._handlers.append(handler)

    def remove_handler(self, handler: PacketHandler) -> None:
        """移除一个已注册的回调。"""
        with self._handlers_lock:
            if handler in self._handlers:
                self._handlers.remove(handler)

    def _dispatch(self, parsed: dict[str, Any]) -> None:
        """将解析结果分发给所有已注册 handler（handler 异常不中断监听）。"""
        with self._handlers_lock:
            handlers = list(self._handlers)
        for h in handlers:
            try:
                h(parsed)
            except Exception:
                logger.exception("handler %r raised, continuing", h)

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """启动监听线程（幂等：已运行时直接返回）。"""
        if self._running.is_set():
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # SO_REUSEADDR 避免 TIME_WAIT 导致端口占用重启失败
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self._host, self._port))
        self._running.set()
        self._thread = threading.Thread(
            target=self._run_loop, name=_THREAD_NAME, daemon=True,
        )
        self._thread.start()
        logger.info("telemetry listener started on %s:%d", self._host, self._port)

    def stop(self) -> None:
        """停止监听线程（幂等：未运行时直接返回）。"""
        if not self._running.is_set():
            return
        self._running.clear()
        # 关闭 socket 以解除阻塞的 recvfrom
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
            self._sock = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        logger.info("telemetry listener stopped")

    @property
    def is_running(self) -> bool:
        """监听线程是否在运行。"""
        return self._running.is_set()

    # ------------------------------------------------------------------ #
    # 接收循环
    # ------------------------------------------------------------------ #
    def _run_loop(self) -> None:
        """后台接收循环：recvfrom → parse_packet → dispatch，全程容错。"""
        assert self._sock is not None
        while self._running.is_set():
            try:
                data, _addr = self._sock.recvfrom(self._max_packet_size)
            except OSError:
                # socket 关闭（stop 调用）或异常——检查 running 决定是否退出
                if not self._running.is_set():
                    break
                logger.warning("socket recv error, continuing", exc_info=True)
                continue
            self._process(data)

    def _process(self, data: bytes) -> None:
        """解析单个包并分发，容错：短包/未知包/解析异常均不崩溃。"""
        try:
            parsed = parse_packet(data)
        except PacketTooShortError:
            # 短包——记录并跳过（对齐 FR-TEL-04 容错降级）
            logger.debug("packet too short (%d bytes), skipped", len(data))
            return
        except Exception:
            # 其他解析异常——记录并跳过，保证线程不死
            logger.exception("parse_packet failed (%d bytes), skipped", len(data))
            return
        if parsed is None:
            # 未知 packetId——跳过不崩溃
            return
        self._dispatch(parsed)