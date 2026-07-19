"""Tests for :mod:`f1opt.telemetry.rate_monitor` (Iter-147)."""
from __future__ import annotations

import time

from f1opt.telemetry.rate_monitor import (
    RateStatus,
    TelemetryRateMonitor,
)


class TestTelemetryRateMonitor:
    def test_initial_state(self) -> None:
        m = TelemetryRateMonitor()
        assert m.rate(6) == 0.0
        status = m.status(6)
        assert isinstance(status, RateStatus)
        assert status.current_rate_hz == 0.0
        assert status.total_packets == 0

    def test_record_and_rate(self) -> None:
        m = TelemetryRateMonitor(window_s=10.0)
        now = time.monotonic()
        # Simulate 10 packets over 1 second = ~10 Hz
        for i in range(10):
            m._timestamps.setdefault(6, __import__("collections").deque()).append(
                now + i * 0.1
            )
        rate = m.rate(6)
        assert 8.0 <= rate <= 12.0, f"Expected ~10 Hz, got {rate}"

    def test_rate_zero_for_single_packet(self) -> None:
        m = TelemetryRateMonitor()
        m.record(6)
        assert m.rate(6) == 0.0

    def test_multiple_packet_types(self) -> None:
        m = TelemetryRateMonitor()
        now = time.monotonic()
        for i in range(5):
            m._timestamps.setdefault(6, __import__("collections").deque()).append(
                now + i * 0.3
            )
            m._timestamps.setdefault(0, __import__("collections").deque()).append(
                now + i * 0.3
            )
        assert m.rate(6) > 0
        assert m.rate(0) > 0
        # Unknown type returns 0
        assert m.rate(99) == 0.0

    def test_status_all(self) -> None:
        m = TelemetryRateMonitor(min_hz=1.0)
        now = time.monotonic()
        for i in range(10):
            m._timestamps.setdefault(6, __import__("collections").deque()).append(
                now + i * 0.1
            )
            m._timestamps.setdefault(0, __import__("collections").deque()).append(
                now + i * 0.5
            )
        statuses = m.status()
        assert isinstance(statuses, list)
        assert len(statuses) == 2
        assert all(isinstance(s, RateStatus) for s in statuses)

    def test_overall_health(self) -> None:
        m = TelemetryRateMonitor(min_hz=5.0)
        now = time.monotonic()
        # Simulate healthy CarTelemetry at ~10 Hz
        for i in range(20):
            m._timestamps.setdefault(6, __import__("collections").deque()).append(
                now + i * 0.1
            )
        health = m.overall_health()
        assert health["healthy"] is True
        assert health["total_packets"] == 20
        assert health["tracked_types"] == 1
        assert health["avg_rate_hz"] >= 5.0

    def test_overall_health_unhealthy(self) -> None:
        m = TelemetryRateMonitor(min_hz=10.0)
        now = time.monotonic()
        # Simulate slow CarTelemetry at ~2 Hz
        for i in range(5):
            m._timestamps.setdefault(6, __import__("collections").deque()).append(
                now + i * 0.5
            )
        health = m.overall_health()
        assert health["healthy"] is False
        assert "CarTelemetry" in health["unhealthy_types"]

    def test_reset(self) -> None:
        m = TelemetryRateMonitor()
        m.record(6)
        m.record(6)
        assert m.rate(6) > 0  # Two rapid calls produce a high rate
        m.reset()
        assert m.rate(6) == 0.0

    def test_status_name_unknown(self) -> None:
        m = TelemetryRateMonitor()
        m.record(99)
        s = m.status(99)
        assert "Unknown" in s.packet_name

    def test_async_subscriber_signature(self) -> None:
        """Verify on_packet accepts the correct subscriber signature."""
        m = TelemetryRateMonitor()

        class FakeHeader:
            packet_id = 6

        import asyncio

        async def run():
            await m.on_packet(FakeHeader(), {}, b"")
            await m.on_packet(FakeHeader(), {}, b"")

        asyncio.run(run())
        assert m.status(6).total_packets == 2

    def test_negative_packet_id_ignored(self) -> None:
        m = TelemetryRateMonitor()

        class FakeHeader:
            packet_id = -1

        import asyncio

        async def run():
            await m.on_packet(FakeHeader(), {}, b"")

        asyncio.run(run())
        assert m.status(0).total_packets == 0

    def test_invalid_window_raises(self) -> None:
        import pytest
        with pytest.raises(ValueError):
            TelemetryRateMonitor(window_s=0)
        with pytest.raises(ValueError):
            TelemetryRateMonitor(window_s=-1)

    def test_invalid_min_hz_raises(self) -> None:
        import pytest
        with pytest.raises(ValueError):
            TelemetryRateMonitor(min_hz=-1)