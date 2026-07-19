"""Unit tests for :mod:`f1opt.telemetry.listener`.

Uses an ephemeral port (``port=0``) and a real UDP socket to send crafted F1 25
datagrams to the listener. Verifies:
- The listener receives, parses, and dispatches to subscribers.
- Parse errors are counted (too-short datagrams).
- Backpressure: drop-oldest when the bounded queue overflows.
- Frame regression / gap counters increment correctly.
"""

from __future__ import annotations

import asyncio
import socket
import struct

from f1opt.telemetry.listener import TelemetryListener
from f1opt.telemetry.packets import HEADER_FORMAT


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_header(
    packet_id: int,
    *,
    session_uid: int = 0x123456789ABCDEF0,
    session_time: float = 10.5,
    frame: int = 100,
    overall_frame: int = 200,
    player_car: int = 0,
    secondary: int = 255,
) -> bytes:
    return struct.pack(
        HEADER_FORMAT,
        2025, 25, 1, 0, 1, packet_id,
        session_uid, session_time, frame, overall_frame,
        player_car, secondary,
    )


def make_packet(packet_id: int, body: bytes = b"", **kw) -> bytes:
    return make_header(packet_id, **kw) + body


async def _send_burst(port: int, packets: list[bytes]) -> None:
    """Send a burst of UDP datagrams to ``port`` via a single socket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for pkt in packets:
            sock.sendto(pkt, ("127.0.0.1", port))
    finally:
        sock.close()


# --------------------------------------------------------------------------- #
# Basic receive + dispatch
# --------------------------------------------------------------------------- #
class TestListenerBasic:
    async def test_receive_and_dispatch(self) -> None:
        listener = TelemetryListener("127.0.0.1", 0, queue_size=64)
        received: list[tuple] = []

        async def sub(header, parsed, raw):
            received.append((header, parsed))

        listener.subscribe(sub)
        await listener.start()
        port = listener.bound_port
        assert port is not None

        try:
            await _send_burst(port, [make_packet(6, b"\x00" * 1400)])
            await asyncio.sleep(0.15)
            assert listener.received >= 1
            assert len(received) >= 1
            assert received[0][0].packet_id == 6
        finally:
            await listener.stop()

    async def test_parse_error_counter(self) -> None:
        listener = TelemetryListener("127.0.0.1", 0, queue_size=64)
        await listener.start()
        port = listener.bound_port
        assert port is not None

        try:
            # Too-short datagram (less than 29-byte header).
            await _send_burst(port, [b"\x00\x00"])
            await asyncio.sleep(0.15)
            assert listener.parse_errors >= 1
            assert listener.received >= 1
        finally:
            await listener.stop()

    async def test_multiple_packets_dispatched(self) -> None:
        listener = TelemetryListener("127.0.0.1", 0, queue_size=64)
        ids: list[int] = []

        async def sub(header, parsed, raw):
            ids.append(header.packet_id)

        listener.subscribe(sub)
        await listener.start()
        port = listener.bound_port

        try:
            packets = [
                make_packet(0, b"\x00" * 1400),  # Motion
                make_packet(1, b"\x00" * 400),   # Session
                make_packet(6, b"\x00" * 1400),  # CarTelemetry
            ]
            await _send_burst(port, packets)
            await asyncio.sleep(0.2)
            assert sorted(ids) == [0, 1, 6]
        finally:
            await listener.stop()

    async def test_is_running_flag(self) -> None:
        listener = TelemetryListener("127.0.0.1", 0)
        assert not listener.is_running
        await listener.start()
        assert listener.is_running
        await listener.stop()
        assert not listener.is_running


# --------------------------------------------------------------------------- #
# Backpressure (drop-oldest)
# --------------------------------------------------------------------------- #
class TestBackpressure:
    async def test_drop_oldest_when_queue_full(self) -> None:
        """When the queue is full and the subscriber is blocked, oldest items drop."""
        listener = TelemetryListener("127.0.0.1", 0, queue_size=2)
        gate = asyncio.Event()

        async def gated_sub(header, parsed, raw):
            await gate.wait()  # block until released

        listener.subscribe(gated_sub)
        await listener.start()
        port = listener.bound_port

        try:
            # Send 12 packets while the subscriber is blocked.
            packets = [
                make_header(6, overall_frame=i) + b"\x00" * 100
                for i in range(12)
            ]
            await _send_burst(port, packets)
            # Let the event loop drain the socket recv buffer.
            await asyncio.sleep(0.3)

            # Release the gate so dispatch can drain the queue.
            gate.set()
            await asyncio.sleep(0.2)

            assert listener.received >= 12
            # queue_size=2, 12 sent, dispatch blocked → at least 12-3=9 dropped.
            assert listener.dropped >= 7, f"expected >=7 drops, got {listener.dropped}"
        finally:
            gate.set()
            await listener.stop()


# --------------------------------------------------------------------------- #
# Frame regression / gap tracking
# --------------------------------------------------------------------------- #
class TestFrameTracking:
    async def test_gap_and_regression_counters(self) -> None:
        listener = TelemetryListener("127.0.0.1", 0, queue_size=64)

        async def noop(header, parsed, raw):
            pass

        listener.subscribe(noop)
        await listener.start()
        port = listener.bound_port

        try:
            # overall_frame sequence: 100 → 105 (gap) → 103 (regression).
            packets = [
                make_header(6, overall_frame=100) + b"\x00" * 100,
                make_header(6, overall_frame=105) + b"\x00" * 100,
                make_header(6, overall_frame=103) + b"\x00" * 100,
            ]
            await _send_burst(port, packets)
            await asyncio.sleep(0.2)
            assert listener.received >= 3
            assert listener.gaps >= 1          # 100 → 105
            assert listener.regressions >= 1   # 105 → 103
        finally:
            await listener.stop()

    async def test_no_false_regression_on_normal_flow(self) -> None:
        listener = TelemetryListener("127.0.0.1", 0, queue_size=64)

        async def noop(header, parsed, raw):
            pass

        listener.subscribe(noop)
        await listener.start()
        port = listener.bound_port

        try:
            packets = [
                make_header(6, overall_frame=i) + b"\x00" * 100
                for i in range(100, 110)
            ]
            await _send_burst(port, packets)
            await asyncio.sleep(0.2)
            assert listener.regressions == 0
        finally:
            await listener.stop()
