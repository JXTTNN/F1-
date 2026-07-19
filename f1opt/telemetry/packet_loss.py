"""Packet loss detection for F1-25 UDP telemetry (Iter-139).

Monitors the ``overall_frame_identifier`` from each incoming packet header
to detect, classify, and report packet loss. Extends the existing
:class:`~f1opt.telemetry.validation.FrameTracker` (gap/regression) with:

- Per-packet-type loss tracking (which packet types are being dropped most).
- Sliding-window loss rate (burst detection with configurable window).
- Loss pattern classification: isolated vs burst vs systematic.
- Comprehensive loss report suitable for the analytics API and dashboard.

Usage::

    from f1opt.telemetry.packet_loss import PacketLossDetector

    detector = PacketLossDetector()
    for header, parsed in stream:
        detector.observe(header, parsed)

    print(detector.loss_rate())
    print(detector.report())
"""
from __future__ import annotations

from collections import deque as _deque
from dataclasses import dataclass as _dataclass
from typing import Any

from f1opt.telemetry.packets import PACKET_NAMES, PacketHeader

__all__ = [
    "LossReport",
    "PacketLossDetector",
]


@_dataclass
class LossReport:
    """Comprehensive packet loss summary for a monitoring window.

    Attributes:
        total_expected: Total datagrams expected (based on frame deltas).
        total_received: Total datagrams actually received.
        total_lost: Total lost datagrams.
        loss_rate: ``total_lost / total_expected`` as a float in [0, 1].
        per_type: Dict mapping packet type name to (received, lost) tuple.
        burst_count: Number of loss bursts detected (consecutive losses).
        max_burst: Largest consecutive loss burst size.
        avg_gap: Average gap size when loss occurred (>= 2 means multiple
            consecutive frames lost).
        pattern: Loss pattern classification (``"none"``, ``"isolated"``,
            ``"burst"``, ``"systematic"``).
        window_seconds: Length of the sliding window in seconds.
    """

    total_expected: int
    total_received: int
    total_lost: int
    loss_rate: float
    per_type: dict[str, tuple[int, int]]
    burst_count: int
    max_burst: int
    avg_gap: float
    pattern: str
    window_seconds: float


class _LossWindow:
    """Sliding window of frame deltas with second-granularity.""" ""

    def __init__(self, window_seconds: float = 5.0) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.window_seconds = window_seconds
        self._entries: _deque[tuple[float, float, int, int]] = _deque()
        # Each entry: (abs_time, delta, received, lost)

    def add(self, abs_time: float, delta: int, received: int, lost: int) -> None:
        self._entries.append((abs_time, delta, received, lost))
        self._prune(abs_time)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()

    def totals(self) -> dict[str, int]:
        if not self._entries:
            return {"received": 0, "lost": 0, "expected": 0}
        recv = sum(r for _, _, r, _ in self._entries)
        lost = sum(li for _, _, _, li in self._entries)
        return {"received": recv, "lost": lost, "expected": recv + lost}

    def loss_rate(self) -> float:
        t = self.totals()
        if t["expected"] == 0:
            return 0.0
        return t["lost"] / t["expected"]

    def reset(self) -> None:
        self._entries.clear()


class PacketLossDetector:
    """Detect and classify packet loss in F1-25 UDP telemetry streams.

    The F1 game sends packets at a fixed rate (typically 60 Hz for CarTelemetry,
    lower for other types). Each packet header carries an
    ``overall_frame_identifier`` that increments by 1 per datagram sent. By
    tracking the delta between consecutive frames, we can:

    - Count exactly how many datagrams were lost between two received packets.
    - Attribute losses to specific packet types.
    - Detect loss bursts (consecutive losses) and classify the pattern.

    The detector is designed to be lightweight: it keeps only the last frame
    identifier and a sliding window of recent deltas, so it can run in the
    hot path of the UDP recv callback without blocking.

    Public API:

    * :meth:`observe` — register a received packet (header + parsed body).
    * :meth:`loss_rate` — fraction of lost datagrams in the sliding window.
    * :meth:`per_type_summary` — per-packet-type breakdown.
    * :meth:`report` — comprehensive :class:`LossReport` for the current window.
    * :meth:`reset` — clear all state (for session restart).
    """

    def __init__(self, window_seconds: float = 5.0) -> None:
        self._window = _LossWindow(window_seconds)
        self._last_frame: dict[int, int] = {}  # session_uid -> last_frame
        self._last_time: dict[int, float] = {}  # session_uid -> last_session_time
        # Per-packet-type counters: {packet_id: [received, lost]}
        self._type_counts: dict[int, list[int]] = {}
        # Burst tracking.
        self._current_burst = 0
        self._bursts: list[int] = []
        self._total_expected = 0
        self._total_received = 0
        self._total_lost = 0

    def observe(
        self,
        header: PacketHeader,
        parsed: dict[str, Any] | None = None,
        *,
        session_time: float | None = None,
    ) -> tuple[int, int]:
        """Register a received packet.

        Args:
            header: Parsed packet header (must have ``session_uid`` and
                ``overall_frame_identifier``).
            parsed: Parsed packet body (unused; reserved for future per-type
                analysis).
            session_time: The ``m_sessionTime`` from the header (used for
                time-based windowing). If None, uses ``header.session_time``.

        Returns:
            ``(lost, delta)`` — how many datagrams were lost since the last
            received packet, and the frame delta. ``(0, 1)`` means no loss.
        """
        sid = header.session_uid
        frame = header.overall_frame_identifier
        pid = header.packet_id
        st = session_time if session_time is not None else header.session_time

        last = self._last_frame.get(sid)
        self._last_frame[sid] = frame
        self._last_time[sid] = st

        if last is None:
            # First packet for this session — no loss to detect.
            self._total_received += 1
            self._total_expected += 1
            self._inc_type(pid, received=1, lost=0)
            return 0, 0

        delta = frame - last
        if delta < 0:
            # Regression (flashback / reorder), not a gap.
            self._total_received += 1
            self._total_expected += 1
            self._inc_type(pid, received=1, lost=0)
            self._current_burst = 0
            return 0, delta

        lost = delta - 1  # e.g. delta=1 -> no loss; delta=3 -> 2 lost
        self._total_received += 1
        self._total_expected += delta
        self._total_lost += lost

        # Track per-packet-type: attribute lost packets to the current type
        # (best-effort — we don't know which types were lost, but by attributing
        # to the type we did receive, we get a useful proxy for "loss rate near
        # this packet type").
        self._inc_type(pid, received=1, lost=lost)

        # Update sliding window.
        self._window.add(st, delta, 1, lost)

        # Burst tracking.
        if lost > 0:
            self._current_burst += lost
        else:
            if self._current_burst > 0:
                self._bursts.append(self._current_burst)
            self._current_burst = 0

        return lost, delta

    def _inc_type(self, pid: int, *, received: int = 0, lost: int = 0) -> None:
        if pid not in self._type_counts:
            self._type_counts[pid] = [0, 0]
        self._type_counts[pid][0] += received
        self._type_counts[pid][1] += lost

    def loss_rate(self) -> float:
        """Fraction of lost datagrams in the current sliding window."""
        return self._window.loss_rate()

    def total_loss_rate(self) -> float:
        """Fraction of lost datagrams over the entire observation period."""
        if self._total_expected == 0:
            return 0.0
        return self._total_lost / self._total_expected

    def per_type_summary(self) -> dict[str, dict[str, int]]:
        """Per-packet-type breakdown: ``{name: {"received": N, "lost": M}}``."""
        return {
            PACKET_NAMES.get(pid, f"Unknown({pid})"): {
                "received": counts[0],
                "lost": counts[1],
            }
            for pid, counts in sorted(self._type_counts.items())
        }

    def _classify_pattern(self) -> str:
        """Classify the loss pattern from burst history."""
        if self._total_lost == 0:
            return "none"
        if not self._bursts:
            # All losses are isolated single-frame losses.
            return "isolated"
        avg_burst = sum(self._bursts) / len(self._bursts)
        burst_fraction = sum(self._bursts) / max(self._total_lost, 1)
        if avg_burst >= 5 and burst_fraction > 0.5:
            return "burst"
        if self.total_loss_rate() > 0.05:
            return "systematic"
        return "isolated"

    def report(self) -> LossReport:
        """Generate a comprehensive :class:`LossReport` for the current window."""
        if self._current_burst > 0:
            self._bursts.append(self._current_burst)
            self._current_burst = 0

        wt = self._window.totals()
        total_lost = self._total_lost if self._total_lost > 0 else wt["lost"]
        total_exp = self._total_expected if self._total_expected > 0 else wt["expected"]
        total_recv = self._total_received if self._total_received > 0 else wt["received"]

        pattern = self._classify_pattern()

        max_burst = max(self._bursts) if self._bursts else 0
        avg_gap = 0.0
        if self._bursts:
            avg_gap = sum(self._bursts) / len(self._bursts)

        per_type: dict[str, tuple[int, int]] = {
            name: (d["received"], d["lost"])
            for name, d in self.per_type_summary().items()
        }

        return LossReport(
            total_expected=total_exp,
            total_received=total_recv,
            total_lost=total_lost,
            loss_rate=self.loss_rate(),
            per_type=per_type,
            burst_count=len(self._bursts),
            max_burst=max_burst,
            avg_gap=avg_gap,
            pattern=pattern,
            window_seconds=self._window.window_seconds,
        )

    def reset(self) -> None:
        """Clear all state (for session restart)."""
        self._last_frame.clear()
        self._last_time.clear()
        self._type_counts.clear()
        self._current_burst = 0
        self._bursts.clear()
        self._total_expected = 0
        self._total_received = 0
        self._total_lost = 0
        self._window.reset()