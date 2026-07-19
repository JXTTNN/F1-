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
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import Awaitable, Callable
from typing import Any

from f1opt.observability.logging import get_logger

from .packets import PacketHeader, parse_packet
from .validation import FrameTracker, SampleFlag, flag_sample, validate_sample

log = get_logger(__name__)

#: Subscriber coroutine signature: ``(header, parsed, raw_bytes) -> None``.
Subscriber = Callable[[PacketHeader, dict[str, Any], bytes], Awaitable[None]]


class TelemetryListener:
    """asyncio UDP server that parses F1 25 datagrams and dispatches to subscribers."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 20777,
        *,
        queue_size: int = 1024,
        validate: bool = True,
        track_frames: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self._queue: asyncio.Queue[
            tuple[PacketHeader, dict[str, Any], bytes] | None
        ] = asyncio.Queue(queue_size)
        self._subscribers: list[Subscriber] = []
        self._transport: asyncio.DatagramTransport | None = None
        self._dispatch_task: asyncio.Task[None] | None = None
        self._frame_tracker = FrameTracker() if track_frames else None
        self._validate = validate
        # Observability counters.
        self.dropped = 0
        self.received = 0
        self.parse_errors = 0
        self.regressions = 0
        self.gaps = 0
        self.validation_failures = 0
        # Field-level flag closure: count of frames tagged with a non-OK flag
        # (range violation, frame regression, ...). Exposed via /api/metrics.
        self.flagged_samples = 0
        self._flag_counts: dict[str, int] = {}

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
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            except (OSError, ValueError):
                pass  # non-fatal: default buffer still works at lower burst rates
        self._dispatch_task = asyncio.create_task(
            self._dispatch_loop(), name="telemetry-dispatch"
        )

    async def stop(self) -> None:
        """Stop the listener and wait for the dispatch loop to drain."""
        if self._dispatch_task is not None:
            await self._queue.put(None)  # sentinel
            await self._dispatch_task
            self._dispatch_task = None
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    # ------------------------------------------------------------------ #
    # Internal: recv path (synchronous — never blocks the event loop)
    # ------------------------------------------------------------------ #
    def _on_datagram(self, data: bytes, addr: tuple[str, int]) -> None:
        self.received += 1
        try:
            header, parsed = parse_packet(data)
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
        if self._validate:
            ok, reason = validate_sample(header.packet_id, parsed)
            parsed["__validation__"] = {"ok": ok, "reason": reason}
            if not ok:
                self.validation_failures += 1
                log.debug(
                    "validation failed: %s (packet=%s)", reason, header.name
                )
        # Classify the sample into a quality flag (considers validation +
        # frame regression). Counted for every frame; non-OK frames are
        # tagged with ``_flag`` so the aggregator can track per-lap worst flag
        # instead of silently writing anomalous samples.
        vinfo = parsed.get("__validation__", {"ok": True, "reason": None})
        flag = flag_sample(parsed, vinfo, (regressed, gap, delta))
        self._flag_counts[flag] = self._flag_counts.get(flag, 0) + 1
        if flag != SampleFlag.OK:
            parsed["_flag"] = flag
            self.flagged_samples += 1
        item: tuple[PacketHeader, dict[str, Any], bytes] | None = (
            header,
            parsed,
            data,
        )
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            # Drop oldest to make room for newest (drop-oldest backpressure).
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

    async def _dispatch_loop(self) -> None:
        """Pull parsed packets from the queue and fan them out to subscribers."""
        while True:
            item = await self._queue.get()
            if item is None:
                return  # sentinel — stop() was called
            header, parsed, raw = item
            for sub in list(self._subscribers):
                try:
                    await sub(header, parsed, raw)
                except Exception:
                    log.exception("subscriber raised")


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
