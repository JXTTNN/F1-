"""Real-time telemetry rate monitor (Iter-147).

EA F1 2026 professional standard: every telemetry data stream must be monitored
for rate stability. The :class:`TelemetryRateMonitor` provides a lightweight
subscriber-compatible monitor that tracks incoming packet rates, detects rate
drops, and exposes a sliding-window history for dashboard integration.

Key features:

- **Per-packet-type rate tracking**: Monitors Hz for each F1 packet type
  (Motion=0, Session=1, LapData=2, Event=3, Participants=4, CarSetups=5,
  CarTelemetry=6, CarStatus=7, FinalClassification=8, LobbyInfo=9,
  CarDamage=10, SessionHistory=11).
- **Sliding window**: Configurable window (default 5s) of rate samples.
- **Rate drop detection**: Alerts when rate drops below ``min_hz`` threshold.
- **Subscriber-compatible**: Implements the async subscriber signature
  ``(header, parsed, raw_bytes) -> None`` for direct use with
  :class:`~f1opt.telemetry.listener.TelemetryListener`.
- **Thread-safe**: Lock-protected state for concurrent access from async
  subscribers.

Usage::

    from f1opt.telemetry.rate_monitor import TelemetryRateMonitor
    from f1opt.telemetry.listener import TelemetryListener

    monitor = TelemetryRateMonitor(window_s=5.0, min_hz=10.0)
    listener = TelemetryListener()
    listener.subscribe(monitor.on_packet)
    # ... later:
    print(monitor.rate(6))  # CarTelemetry rate in Hz
    print(monitor.status())  # Overall rate health
"""

from __future__ import annotations

import time as _time
from collections import deque as _deque
from dataclasses import dataclass as _dataclass
from threading import Lock as _Lock
from typing import Any


@_dataclass
class RateStatus:
    """Per-packet-type rate status snapshot."""

    packet_id: int
    """F1 25 packet type identifier (0-11)."""

    packet_name: str
    """Human-readable packet type name."""

    current_rate_hz: float
    """Current packets-per-second rate over the sliding window."""

    min_rate_hz: float
    """Configured minimum acceptable rate."""

    healthy: bool
    """True if ``current_rate_hz >= min_rate_hz``."""

    total_packets: int
    """Total packets received since monitor start."""

    window_duration_s: float
    """Actual duration of the sliding window (seconds)."""


# F1 25 packet type names (indexed by packet_id).
_PACKET_NAMES: dict[int, str] = {
    0: "Motion",
    1: "Session",
    2: "LapData",
    3: "Event",
    4: "Participants",
    5: "CarSetups",
    6: "CarTelemetry",
    7: "CarStatus",
    8: "FinalClassification",
    9: "LobbyInfo",
    10: "CarDamage",
    11: "SessionHistory",
}


class TelemetryRateMonitor:
    """Real-time packet rate monitor with sliding window (Iter-147).

    Tracks incoming packet timestamps per packet type, computes current Hz
    rate over a sliding window, and detects rate drops below a configurable
    threshold.

    Thread-safe: all public methods are protected by an internal lock.
    """

    def __init__(
        self,
        window_s: float = 5.0,
        min_hz: float = 1.0,
        *,
        max_packets_per_type: int = 5000,
    ) -> None:
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        if min_hz < 0:
            raise ValueError("min_hz must be non-negative")
        self._window_s = window_s
        self._min_hz = min_hz
        self._lock = _Lock()
        # per-packet-type timestamp deques
        self._timestamps: dict[int, _deque[float]] = {}
        self._max_packets = max_packets_per_type

    # ------------------------------------------------------------------ #
    # Subscriber interface
    # ------------------------------------------------------------------ #
    async def on_packet(
        self,
        header: Any,
        parsed: dict[str, Any] | None,
        raw_bytes: bytes | None,
    ) -> None:
        """Async subscriber callback for :class:`TelemetryListener`.

        Records the current timestamp for the packet's type. Signature
        matches :data:`~f1opt.telemetry.listener.Subscriber`.
        """
        packet_id = int(getattr(header, "packet_id", -1))
        if packet_id < 0:
            return
        self.record(packet_id)

    def record(self, packet_id: int) -> None:
        """Record a packet arrival for the given packet type (sync).

        Thread-safe. Call from any context (sync subscriber, batch replay,
        unit tests).
        """
        now = _time.monotonic()
        with self._lock:
            if packet_id not in self._timestamps:
                self._timestamps[packet_id] = _deque(maxlen=self._max_packets)
            self._timestamps[packet_id].append(now)

    # ------------------------------------------------------------------ #
    # Rate queries
    # ------------------------------------------------------------------ #
    def rate(self, packet_id: int) -> float:
        """Current packets-per-second rate for ``packet_id``.

        Returns 0.0 if no packets have been recorded.
        """
        with self._lock:
            ts = self._timestamps.get(packet_id)
            if not ts:
                return 0.0
            return self._compute_rate(ts)

    def status(self, packet_id: int | None = None) -> RateStatus | list[RateStatus]:
        """Get rate status for one or all packet types.

        Args:
            packet_id: Specific packet type to query, or ``None`` for all.

        Returns:
            A single :class:`RateStatus` when ``packet_id`` is specified,
            or a list of :class:`RateStatus` for all tracked types when
            ``None``.
        """
        with self._lock:
            if packet_id is not None:
                return self._status_for(packet_id)
            return [
                self._status_for(pid)
                for pid in sorted(self._timestamps.keys())
            ]

    def overall_health(self) -> dict[str, Any]:
        """Overall rate health summary for dashboard integration.

        Returns:
            ``{"healthy": bool, "total_packets": int, "tracked_types": int,
            "unhealthy_types": list[str], "avg_rate_hz": float}``
        """
        with self._lock:
            statuses = [
                self._status_for(pid)
                for pid in sorted(self._timestamps.keys())
            ]
            unhealthy = [s.packet_name for s in statuses if not s.healthy]
            total_pkts = sum(s.total_packets for s in statuses)
            rates = [s.current_rate_hz for s in statuses if s.current_rate_hz > 0]
            avg_rate = sum(rates) / len(rates) if rates else 0.0
            return {
                "healthy": len(unhealthy) == 0 and len(statuses) > 0,
                "total_packets": total_pkts,
                "tracked_types": len(statuses),
                "unhealthy_types": unhealthy,
                "avg_rate_hz": round(avg_rate, 2),
            }

    def reset(self) -> None:
        """Clear all recorded timestamps."""
        with self._lock:
            self._timestamps.clear()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _compute_rate(self, ts: _deque[float]) -> float:
        """Compute Hz from a deque of monotonic timestamps."""
        if len(ts) < 2:
            return 0.0
        start = ts[0]
        end = ts[-1]
        duration = end - start
        if duration <= 0:
            return 0.0
        return (len(ts) - 1) / duration

    def _status_for(self, packet_id: int) -> RateStatus:
        ts = self._timestamps.get(packet_id)
        if not ts:
            return RateStatus(
                packet_id=packet_id,
                packet_name=_PACKET_NAMES.get(packet_id, f"Unknown({packet_id})"),
                current_rate_hz=0.0,
                min_rate_hz=self._min_hz,
                healthy=True,
                total_packets=0,
                window_duration_s=0.0,
            )
        rate = self._compute_rate(ts)
        start = ts[0]
        end = ts[-1]
        return RateStatus(
            packet_id=packet_id,
            packet_name=_PACKET_NAMES.get(packet_id, f"Unknown({packet_id})"),
            current_rate_hz=round(rate, 2),
            min_rate_hz=self._min_hz,
            healthy=rate >= self._min_hz,
            total_packets=len(ts),
            window_duration_s=round(end - start, 3),
        )


__all__ = [
    "RateStatus",
    "TelemetryRateMonitor",
]