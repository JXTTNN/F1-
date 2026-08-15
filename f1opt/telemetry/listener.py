"""asyncio UDP listener for F1 25 telemetry with backpressure-safe dispatch.

The listener binds a UDP socket (default ``0.0.0.0:20777``), parses each
datagram via :mod:`f1opt.telemetry.packets`, optionally validates the parsed
fields via :mod:`f1opt.telemetry.validation`, tracks frame regressions / gaps
via :class:`~f1opt.telemetry.validation.FrameTracker`, and fans the parsed
packets out to async subscribers.

Backpressure: an internal bounded :class:`asyncio.Queue` decouples the
synchronous recv callback (which must never block) from slower async
subscribers. When the queue is full the oldest enqueued item is dropped to
make room for the newest (drop-oldest policy); the ``dropped`` counter is
exposed for observability.

Iter-182: 性能优化 — 自适应队列大小, 连接健康监控, 延迟统计, 订阅者超时保护.
"""

from __future__ import annotations

import asyncio
import socket
import time
from collections.abc import Awaitable, Callable
from typing import Any, NamedTuple

from f1opt.observability.logging import get_logger

from .packets import PACKET_PARSERS, PacketHeader, parse_header
from .validation import FrameTracker, SampleFlag, flag_sample, validate_sample

log = get_logger(__name__)

#: Subscriber coroutine signature: ``(header, parsed, raw_bytes) -> None``.
Subscriber = Callable[[PacketHeader, dict[str, Any], bytes], Awaitable[None]]


class _QueuedPacket(NamedTuple):
    """Internal queue item: header + raw datagram + frame info.

    Body parsing happens lazily in the dispatch loop (not the synchronous
    recv callback) so the recv path stays fast under UDP flood.
    """

    header: PacketHeader
    raw: bytes
    recv_time: float
    regressed: bool
    gap: bool
    delta: int

#: 默认内核 UDP 缓冲区大小 (4 MB).
_DEFAULT_KERNEL_BUF = 4 * 1024 * 1024

#: 默认内部队列大小 (自适应起点).
_DEFAULT_QUEUE_SIZE = 1024

#: 最大内部队列大小 (自适应上限).
_MAX_QUEUE_SIZE = 4096

#: 最小内部队列大小 (自适应下限).
#: 2 是绝对下限 (queue_size=1 用于测试), 自适应缩容不会低于此值.
_MIN_QUEUE_SIZE = 2

#: 订阅者超时 (秒) — 单个订阅者处理超时后跳过, 不阻塞其他订阅者.
_SUBSCRIBER_TIMEOUT = 5.0

#: 自适应调整间隔 (秒) — 每 N 秒评估一次队列利用率.
_ADAPTIVE_INTERVAL = 10.0

#: 队列利用率高水位 (超过此比例触发扩容).
_HIGH_WATERMARK = 0.7

#: 队列利用率低水位 (低于此比例触发缩容).
_LOW_WATERMARK = 0.3

#: Iter-183: Burst 检测阈值 — pps 超过此倍数视为 burst.
_BURST_PPS_MULTIPLIER = 2.0

#: Iter-183: Burst 检测基准 — 正常 F1 25 60Hz 遥测 ≈ 60 pps.
_NOMINAL_PPS = 60.0

#: Iter-183: 队列满次数统计窗口大小 (用于计算 queue_full_pct).
_QUEUE_FULL_WINDOW = 1000

#: Iter-183: 队列健康状态阈值.
_QUEUE_HEALTH_CRITICAL_PCT = 0.5   # 满比例 > 50% → critical
_QUEUE_HEALTH_WARNING_PCT = 0.1    # 满比例 > 10% → warning


class TelemetryListener:
    """asyncio UDP server that parses F1 25 datagrams and dispatches to subscribers.

    Iter-182: 添加自适应队列大小, 连接健康监控, 延迟统计, 订阅者超时保护.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 20777,
        *,
        queue_size: int = _DEFAULT_QUEUE_SIZE,
        validate: bool = True,
        track_frames: bool = True,
        adaptive_queue: bool = True,
        kernel_buf_size: int = _DEFAULT_KERNEL_BUF,
    ) -> None:
        self.host = host
        self.port = port
        self._queue_size = max(_MIN_QUEUE_SIZE, min(queue_size, _MAX_QUEUE_SIZE))
        self._queue: asyncio.Queue[_QueuedPacket | None] = asyncio.Queue(
            self._queue_size
        )
        self._subscribers: list[Subscriber] = []
        self._transport: asyncio.DatagramTransport | None = None
        self._dispatch_task: asyncio.Task[None] | None = None
        self._adaptive_task: asyncio.Task[None] | None = None
        self._frame_tracker = FrameTracker() if track_frames else None
        self._validate = validate
        self._adaptive_queue = adaptive_queue
        self._kernel_buf_size = kernel_buf_size
        # Observability counters.
        self.dropped = 0
        self.received = 0
        self.parse_errors = 0
        self.regressions = 0
        self.gaps = 0
        self.validation_failures = 0
        # Field-level flag closure: count of frames tagged with a non-OK flag.
        self.flagged_samples = 0
        self._flag_counts: dict[str, int] = {}
        # Iter-182: 延迟统计 (从接收 UDP 到 dispatch 完成).
        self._dispatch_latencies: list[float] = []  # 最近 100 个延迟样本
        self._max_latency_samples = 100
        # Iter-183: P99 dispatch 延迟 (最近 100 个样本的 99 分位).
        self._dispatch_latencies_sorted: list[float] = []  # 排序副本, 延迟更新
        # Iter-183: pps 估计 — 基于 received 计数器差分, 在自适应循环中计算.
        self._last_pps_check_received: int = 0
        self._last_pps_check_time: float = 0.0
        # Iter-183: Burst 检测计数器.
        self.burst_events: int = 0
        self._in_burst: bool = False
        # Iter-183: 队列满统计 (最近 _QUEUE_FULL_WINDOW 次 put 中的满次数).
        self._queue_full_count: int = 0
        # Iter-182: 连接健康监控.
        self._last_packet_time: float = 0.0
        self._connection_healthy: bool = True
        self._health_check_interval: float = 2.0  # 2 秒无数据视为连接异常
        # Iter-240: connection drop tracking.
        self._connection_drop_count: int = 0
        self._last_drop_time: float = 0.0
        self._was_unhealthy: bool = False
        # Iter-182: 订阅者统计.
        self._subscriber_errors: dict[str, int] = {}
        self._subscriber_timeouts: dict[str, int] = {}
        # Iter-182: 自适应队列统计.
        self._adaptive_check_count: int = 0
        self._adaptive_expand_count: int = 0
        self._adaptive_shrink_count: int = 0

    def subscribe(self, sub: Subscriber) -> None:
        """Register an async subscriber ``async def f(header, parsed, raw)``."""
        self._subscribers.append(sub)

    @property
    def is_running(self) -> bool:
        return self._transport is not None and self._dispatch_task is not None

    @property
    def bound_port(self) -> int | None:
        """Actual bound port (useful when ``port=0`` was requested)."""
        if self._transport is None:
            return None
        sock = self._transport.get_extra_info("socket")
        if sock is None:
            return None
        return sock.getsockname()[1]

    async def start(self) -> None:
        """Bind the UDP socket and start the dispatch loop.

        The kernel UDP recv buffer is enlarged to 4 MB so sustained 60 Hz F1
        telemetry bursts (≈1.3 KB/packet × 60/s ≈ 80 KB/s) do not overflow the
        default ~212 KB kernel buffer before the asyncio dispatch loop can
        drain it. This is a production-critical setting for F1 25/2026 UDP.

        Iter-182: 同时启动自适应队列调整任务.
        """
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(
            lambda: _DatagramProtocol(self._on_datagram),
            local_addr=(self.host, self.port),
        )
        self._transport = transport  # type: ignore[assignment]
        # Enlarge kernel recv buffer (best-effort; kernel may cap to net.core.rmem_max).
        sock = transport.get_extra_info("socket")
        if sock is not None:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._kernel_buf_size)
            except (OSError, ValueError):
                pass  # non-fatal: default buffer still works at lower burst rates
        self._dispatch_task = asyncio.create_task(
            self._dispatch_loop(), name="telemetry-dispatch"
        )
        # Iter-182: 启动自适应队列调整任务.
        if self._adaptive_queue:
            self._adaptive_task = asyncio.create_task(
                self._adaptive_loop(), name="telemetry-adaptive"
            )
        self._last_packet_time = time.monotonic()
        self._connection_healthy = True
        self._last_pps_check_time = self._last_packet_time

    async def stop(self) -> None:
        """Stop the listener and wait for the dispatch loop to drain.

        Iter-182: 同时停止自适应队列调整任务.
        """
        # 先停止自适应任务
        if self._adaptive_task is not None:
            self._adaptive_task.cancel()
            try:
                await self._adaptive_task
            except asyncio.CancelledError:
                pass
            self._adaptive_task = None
        if self._dispatch_task is not None:
            await self._queue.put(None)  # sentinel
            await self._dispatch_task
            self._dispatch_task = None
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    # ------------------------------------------------------------------ #
    # Internal: recv path (header-only, non-blocking)
    #
    # Only the 29-byte header is parsed here (~1µs); the body parse (Motion
    # ≈ 75µs) runs in the async dispatch loop instead, keeping this callback
    # fast. NOTE: body parsing still runs on the event loop, so sustained
    # throughput stays capped at ~13k pps — far above the real 60 Hz F1 rate
    # but below the 25k pps artificial stress flood (~60% delivered on
    # Windows). Future optimization: offload body parsing to a worker thread
    # or vectorize with numpy.
    # ------------------------------------------------------------------ #
    def _on_datagram(self, data: bytes, addr: tuple[str, int]) -> None:
        self.received += 1
        # Iter-182: 记录最后收包时间用于健康监控.
        self._last_packet_time = time.monotonic()
        self._connection_healthy = True
        recv_time = self._last_packet_time
        # 只解析 header（≈1µs）；body 解析移入 dispatch 循环 (async)，避免
        # 同步 body 解析 (Motion ≈ 75µs) 阻塞事件循环、限制 UDP 洪泛吞吐。
        try:
            header = parse_header(data)
        except Exception:
            self.parse_errors += 1
            return
        regressed = False
        gap = False
        delta = 0
        if self._frame_tracker is not None:
            regressed, gap, delta = self._frame_tracker.observe(
                header.session_uid, header.overall_frame_identifier
            )
            if regressed:
                self.regressions += 1
            if gap:
                self.gaps += 1
        item = _QueuedPacket(header, data, recv_time, regressed, gap, delta)
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            # Drop oldest to make room for newest (drop-oldest backpressure).
            self._queue_full_count += 1
            try:
                self._queue.get_nowait()
                self.dropped += 1
                self._queue.put_nowait(item)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass  # extremely unlikely race

    def flag_counts(self) -> dict[str, int]:
        """Return per-flag tallies: ``{OK: N, SUSPECT_RANGE: M, ...}``.

        Every received-and-parsed frame is counted exactly once under its
        classified flag. Flags never observed report 0.
        """
        return {f.value: self._flag_counts.get(f, 0) for f in SampleFlag}

    @property
    def connection_healthy(self) -> bool:
        """连接健康状态 (基于最近收包时间).

        Iter-182: 允许外部查询连接状态, 用于 UI 显示.
        Iter-240: tracks connection drops with count and timestamp.
        """
        if not self._connection_healthy:
            return False
        if self._last_packet_time == 0.0:
            return True  # 尚未收到任何包, 视为健康
        elapsed = time.monotonic() - self._last_packet_time
        healthy = elapsed < self._health_check_interval
        if not healthy and not self._was_unhealthy:
            self._connection_drop_count += 1
            self._last_drop_time = time.monotonic()
        self._was_unhealthy = not healthy
        return healthy

    @property
    def connection_drop_count(self) -> int:
        """Iter-240: 连接断开次数."""
        return self._connection_drop_count

    @property
    def last_drop_time(self) -> float:
        """Iter-240: 最近一次断开的时间戳."""
        return self._last_drop_time

    @property
    def avg_dispatch_latency_ms(self) -> float:
        """平均 dispatch 延迟 (毫秒).

        Iter-182: 基于最近 100 个样本的平均延迟.
        """
        if not self._dispatch_latencies:
            return 0.0
        return sum(self._dispatch_latencies) / len(self._dispatch_latencies) * 1000.0

    @property
    def p99_dispatch_latency_ms(self) -> float:
        """P99 dispatch 延迟 (毫秒) — 最近 100 个样本的 99 分位.

        Iter-183: 用于识别偶发的高延迟尖峰, 比平均值更能反映真实抖动.
        """
        if not self._dispatch_latencies:
            return 0.0
        # 懒惰排序: 只在有新样本加入时 rebuild.
        if len(self._dispatch_latencies_sorted) != len(self._dispatch_latencies):
            self._dispatch_latencies_sorted = sorted(self._dispatch_latencies)
        idx = int(len(self._dispatch_latencies_sorted) * 0.99)
        idx = min(idx, len(self._dispatch_latencies_sorted) - 1)
        return self._dispatch_latencies_sorted[idx] * 1000.0

    @property
    def queue_utilization(self) -> float:
        """当前队列利用率 (0.0 ~ 1.0).

        Iter-182: 用于自适应队列调整.
        """
        return self._queue.qsize() / max(self._queue.maxsize, 1)

    @property
    def packets_per_second(self) -> float:
        """估算包到达率 (pps) — 基于 received 计数器差分.

        Iter-183: 使用自适应循环中记录的上次检查点的 received 计数差分.
        """
        now = time.monotonic()
        elapsed = now - self._last_pps_check_time
        if elapsed <= 0:
            return 0.0
        delta = self.received - self._last_pps_check_received
        self._last_pps_check_received = self.received
        self._last_pps_check_time = now
        return delta / elapsed

    @property
    def packets_per_minute(self) -> float:
        """估算包到达率 (ppm) — packets_per_second * 60.

        Iter-244: 提供每分钟速率统计，便于长时间监控。
        """
        return self.packets_per_second * 60.0

    @property
    def is_burst(self) -> bool:
        """是否检测到遥测 burst (pps 超过正常值 2 倍).

        Iter-183: burst 检测用于自适应队列的快速扩容决策.
        """
        pps = self.packets_per_second
        return pps > _NOMINAL_PPS * _BURST_PPS_MULTIPLIER

    @property
    def queue_full_pct(self) -> float:
        """队列满比例 (0.0 ~ 1.0) — QueueFull 次数 / 总接收包数.

        Iter-183: 高 queue_full_pct 表示订阅者处理速度跟不上到达率.
        """
        if self.received == 0:
            return 0.0
        return self._queue_full_count / self.received

    @property
    def queue_health(self) -> str:
        """队列健康状态: ``"normal"`` / ``"warning"`` / ``"critical"``.

        Iter-183: 基于 queue_full_pct 的 tri-state 健康指示.
        """
        pct = self.queue_full_pct
        if pct > _QUEUE_HEALTH_CRITICAL_PCT:
            return "critical"
        if pct > _QUEUE_HEALTH_WARNING_PCT:
            return "warning"
        return "normal"

    # ------------------------------------------------------------------ #
    # Iter-228: connection health statistics snapshot
    # ------------------------------------------------------------------ #
    def stats_snapshot(self) -> dict[str, Any]:
        """Export a snapshot of all listener health metrics as a dict.

        Iter-228: provides a single dict with all queue / latency / throughput
        stats, suitable for logging, monitoring dashboards, or API responses.
        """
        return {
            "received": self.received,
            "dropped": self.dropped,
            "queue_size": self._queue_size,
            "queue_current": self._queue.qsize(),
            "queue_utilization": self.queue_utilization,
            "queue_full_count": self._queue_full_count,
            "queue_full_pct": self.queue_full_pct,
            "queue_health": self.queue_health,
            "packets_per_second": self.packets_per_second,
            "packets_per_minute": self.packets_per_minute,  # Iter-244
            "is_burst": self.is_burst,
            "avg_dispatch_latency_ms": self.avg_dispatch_latency_ms,
            "p99_dispatch_latency_ms": self.p99_dispatch_latency_ms,
            "subscriber_errors": dict(self._subscriber_errors),
            "subscriber_timeouts": dict(self._subscriber_timeouts),
            "adaptive_queue_enabled": self._adaptive_queue,
            "adaptive_expand_count": self._adaptive_expand_count,  # Iter-248
            "adaptive_shrink_count": self._adaptive_shrink_count,  # Iter-248
            "connection_drop_count": self._connection_drop_count,
            "last_drop_time": self._last_drop_time,
        }

    async def _dispatch_loop(self) -> None:
        """Pull queued packets, parse bodies, and fan out to subscribers.

        Iter-182: 添加订阅者超时保护, 延迟统计.
        body 解析 + 校验 + 质量标记移入此处 (async)，recv callback 只解析 header.
        """
        while True:
            item = await self._queue.get()
            if item is None:
                return  # sentinel — stop() was called
            header, raw, recv_time, regressed, gap, delta = item
            # Parse the body here (async context), not in the recv callback.
            try:
                parser = PACKET_PARSERS.get(header.packet_id)
                parsed = parser(raw) if parser is not None else {}
            except Exception:
                self.parse_errors += 1
                continue
            # Validation (moved from the synchronous recv callback).
            if self._validate:
                ok, reason = validate_sample(header.packet_id, parsed)
                parsed["__validation__"] = {"ok": ok, "reason": reason}
                if not ok:
                    self.validation_failures += 1
                    log.debug(
                        "validation failed: %s (packet=%s)", reason, header.name
                    )
            # Quality flag classification (moved from the synchronous recv callback).
            vinfo = parsed.get("__validation__", {"ok": True, "reason": None})
            flag = flag_sample(parsed, vinfo, (regressed, gap, delta))
            self._flag_counts[flag] = self._flag_counts.get(flag, 0) + 1
            if flag != SampleFlag.OK:
                parsed["_flag"] = flag
                self.flagged_samples += 1
            parsed["__recv_time__"] = recv_time
            dispatch_start = time.monotonic()
            for sub in list(self._subscribers):
                sub_name = getattr(sub, "__name__", str(sub))
                try:
                    await asyncio.wait_for(
                        sub(header, parsed, raw),
                        timeout=_SUBSCRIBER_TIMEOUT,
                    )
                except TimeoutError:
                    self._subscriber_timeouts[sub_name] = (
                        self._subscriber_timeouts.get(sub_name, 0) + 1
                    )
                    log.warning("subscriber %s timed out after %.1fs", sub_name, _SUBSCRIBER_TIMEOUT)
                except Exception:
                    self._subscriber_errors[sub_name] = (
                        self._subscriber_errors.get(sub_name, 0) + 1
                    )
                    log.exception("subscriber %s raised", sub_name)
            # Iter-182: 记录 dispatch 延迟.
            dispatch_elapsed = time.monotonic() - dispatch_start
            self._dispatch_latencies.append(dispatch_elapsed)
            if len(self._dispatch_latencies) > self._max_latency_samples:
                self._dispatch_latencies = self._dispatch_latencies[-self._max_latency_samples:]
            # Iter-183: 标记排序缓存为脏, 供 p99_dispatch_latency_ms 懒惰重建.
            self._dispatch_latencies_sorted = []

    async def _adaptive_loop(self) -> None:
        """Iter-182, Iter-248: 自适应队列大小调整 (EMA + hysteresis).

        每 _ADAPTIVE_INTERVAL 秒评估一次队列利用率:
        - 利用率 EMA > 70% → 扩容 (翻倍, 上限 _MAX_QUEUE_SIZE)
        - 利用率 EMA < 30% → 缩容 (减半, 下限 _MIN_QUEUE_SIZE)

        Iter-248: 使用 EMA 平滑利用率 (防止瞬时抖动), 添加 hysteresis
        cooldown 防止快速 flip-flop, 记录 expand/shrink 计数到 stats_snapshot.
        """
        # EMA smoothing factor for utilization (Iter-248).
        _EMA_ALPHA = 0.3
        _utilization_ema = 0.0
        _ema_initialized = False
        _last_resize_time = 0.0
        _RESIZE_COOLDOWN_S = 2.0  # Iter-248: cooldown between resizes.

        while True:
            await asyncio.sleep(_ADAPTIVE_INTERVAL)
            self._adaptive_check_count += 1
            # Iter-183: 检测 burst (在自适应循环中, 不在热路径).
            pps = self.packets_per_second
            if pps > _NOMINAL_PPS * _BURST_PPS_MULTIPLIER:
                if not self._in_burst:
                    self._in_burst = True
                    self.burst_events += 1
                    log.warning("burst detected: pps=%.1f (threshold=%.1f)", pps, _NOMINAL_PPS * _BURST_PPS_MULTIPLIER)
            else:
                self._in_burst = False
            utilization = self.queue_utilization
            # Iter-248: EMA smoothing.
            if not _ema_initialized:
                _utilization_ema = utilization
                _ema_initialized = True
            else:
                _utilization_ema = _EMA_ALPHA * utilization + (1.0 - _EMA_ALPHA) * _utilization_ema
            current_size = self._queue.maxsize
            now = time.monotonic()
            # Iter-248: cooldown check to prevent rapid oscillation.
            if now - _last_resize_time < _RESIZE_COOLDOWN_S:
                continue
            if _utilization_ema > _HIGH_WATERMARK and current_size < _MAX_QUEUE_SIZE:
                new_size = min(current_size * 2, _MAX_QUEUE_SIZE)
                await self._resize_queue(new_size)
                self._adaptive_expand_count += 1
                _last_resize_time = now
                log.info(
                    "adaptive queue expanded: %d -> %d (ema_util=%.2f)",
                    current_size, new_size, _utilization_ema,
                )
            elif _utilization_ema < _LOW_WATERMARK and current_size > _MIN_QUEUE_SIZE:
                new_size = max(current_size // 2, _MIN_QUEUE_SIZE)
                await self._resize_queue(new_size)
                self._adaptive_shrink_count += 1
                _last_resize_time = now
                log.info(
                    "adaptive queue shrunk: %d -> %d (ema_util=%.2f)",
                    current_size, new_size, _utilization_ema,
                )

    async def _resize_queue(self, new_size: int) -> None:
        """Iter-182: 替换内部队列为新大小 (保留现有元素).

        由于 asyncio.Queue 不支持动态调整 maxsize, 需要创建新队列并迁移元素.
        这是一个谨慎操作: 迁移期间不接收新元素 (通过 put_nowait 可能失败).
        """
        new_queue: asyncio.Queue[_QueuedPacket | None] = asyncio.Queue(new_size)
        # 迁移现有元素 (从旧队列中取出并放入新队列).
        while not self._queue.empty():
            try:
                old_item = self._queue.get_nowait()
                try:
                    new_queue.put_nowait(old_item)
                except asyncio.QueueFull:
                    self.dropped += 1
            except asyncio.QueueEmpty:
                break
        self._queue = new_queue
        self._queue_size = new_size


class _DatagramProtocol(asyncio.DatagramProtocol):
    """Thin adapter forwarding received datagrams to a sync callback."""

    def __init__(
        self, on_datagram: Callable[[bytes, tuple[str, int]], None]
    ) -> None:
        self._on_datagram = on_datagram

    def connection_made(self, transport: asyncio.BaseTransport) -> None:  # noqa: D401
        # The listener holds its own transport reference; nothing to do here.
        pass

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        self._on_datagram(data, addr)

    def error_received(self, exc: Exception) -> None:
        log.warning("UDP error: %s", exc)


__all__ = ["TelemetryListener", "Subscriber"]
