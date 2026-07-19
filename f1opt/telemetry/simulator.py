"""Synthetic F1-25 telemetry stream simulator (Iter-163).

Generates :class:`~f1opt.telemetry.packets.PacketHeader` + parsed-body tuples
for stress testing the telemetry pipeline (packet loss detection, frame
tracking, validation, analytics). Supports configurable packet rates, session
durations, and loss injection patterns.

The simulator does NOT produce raw UDP bytes — it produces already-parsed
:class:`PacketHeader` objects (matching what :func:`parse_header` would return
from a real datagram) plus a minimal parsed-body dict. This lets the stress
test focus on the **downstream pipeline** (detectors, trackers, analytics)
without paying the binary-packing cost.

Loss patterns:

- ``"none"``       — no loss (every frame delivered).
- ``"isolated"``   — random per-packet drop with probability ``loss_rate``.
- ``"burst"``      — drop ``burst_size`` consecutive packets every
                     ``burst_interval`` packets.
- ``"systematic"`` — drop every ``systematic_k``-th packet.

Usage::

    from f1opt.telemetry.simulator import TelemetrySimulator

    sim = TelemetrySimulator(
        duration_s=120.0, packet_rate_hz=60, packet_types=(0, 6, 7),
        loss_pattern="isolated", loss_rate=0.05, seed=42,
    )
    for header, body in sim.generate_stream():
        detector.observe(header, body)
        frame_tracker.observe(header.session_uid, header.overall_frame_identifier)
"""
from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from f1opt.telemetry.packets import PacketHeader

__all__ = ["TelemetrySimulator"]

# Default packet types: the most common F1-25 telemetry streams at 60 Hz.
_DEFAULT_PACKET_TYPES: tuple[int, ...] = (0, 6, 7)  # Motion, CarTelemetry, CarStatus

_VALID_LOSS_PATTERNS: frozenset[str] = frozenset(
    {"none", "isolated", "burst", "systematic"}
)


class TelemetrySimulator:
    """Generate synthetic F1-25 telemetry packet streams.

    Args:
        session_uid: The session UID to stamp on every header (random if 0).
        packet_rate_hz: Packets per second per packet type (default 60).
        duration_s: Length of the simulated session in seconds (default 60).
        packet_types: Tuple of F1-25 packet IDs to emit (default Motion +
            CarTelemetry + CarStatus). Each type emits at ``packet_rate_hz``.
        loss_pattern: One of ``"none"``, ``"isolated"``, ``"burst"``,
            ``"systematic"`` (default ``"none"``).
        loss_rate: Per-packet drop probability for ``"isolated"`` pattern,
            in ``[0, 1)`` (default 0.0).
        burst_size: Number of consecutive packets to drop per burst for the
            ``"burst"`` pattern (default 5). Must be >= 1.
        burst_interval: Emit a burst every N packets for the ``"burst"``
            pattern (default 200). Must be > burst_size.
        systematic_k: Drop every K-th packet for the ``"systematic"`` pattern
            (default 5). Must be >= 2.
        seed: RNG seed for reproducible loss injection (default 0).
        start_frame: Initial ``overall_frame_identifier`` (default 0).
    """

    def __init__(
        self,
        *,
        session_uid: int = 0,
        packet_rate_hz: int = 60,
        duration_s: float = 60.0,
        packet_types: tuple[int, ...] = _DEFAULT_PACKET_TYPES,
        loss_pattern: str = "none",
        loss_rate: float = 0.0,
        burst_size: int = 5,
        burst_interval: int = 200,
        systematic_k: int = 5,
        seed: int = 0,
        start_frame: int = 0,
    ) -> None:
        if duration_s <= 0:
            raise ValueError(f"duration_s must be positive, got {duration_s}")
        if packet_rate_hz <= 0:
            raise ValueError(
                f"packet_rate_hz must be positive, got {packet_rate_hz}"
            )
        if not packet_types:
            raise ValueError("packet_types must be non-empty")
        if loss_pattern not in _VALID_LOSS_PATTERNS:
            raise ValueError(
                f"loss_pattern must be one of {_VALID_LOSS_PATTERNS}, "
                f"got {loss_pattern!r}"
            )
        if not 0.0 <= loss_rate < 1.0:
            raise ValueError(
                f"loss_rate must be in [0, 1), got {loss_rate}"
            )
        if loss_pattern == "burst" and burst_size < 1:
            raise ValueError(f"burst_size must be >= 1, got {burst_size}")
        if loss_pattern == "burst" and burst_interval <= burst_size:
            raise ValueError(
                f"burst_interval ({burst_interval}) must be > burst_size "
                f"({burst_size})"
            )
        if loss_pattern == "systematic" and systematic_k < 2:
            raise ValueError(
                f"systematic_k must be >= 2, got {systematic_k}"
            )

        self.session_uid = session_uid if session_uid != 0 else random.getrandbits(63) | 1
        self.packet_rate_hz = packet_rate_hz
        self.duration_s = duration_s
        self.packet_types = tuple(packet_types)
        self.loss_pattern = loss_pattern
        self.loss_rate = loss_rate
        self.burst_size = burst_size
        self.burst_interval = burst_interval
        self.systematic_k = systematic_k
        self.seed = seed
        self.start_frame = start_frame

    def generate_stream(self) -> Iterator[tuple[PacketHeader, dict[str, Any]]]:
        """Yield ``(header, body)`` tuples for the configured session.

        The stream interleaves all ``packet_types`` at ``packet_rate_hz`` each.
        ``overall_frame_identifier`` increments by 1 for EVERY emitted datagram
        (matching the F1-25 spec where it is a per-datagram counter), so a
        no-loss stream yields a strictly-incrementing frame sequence and
        downstream :class:`~f1opt.telemetry.packet_loss.PacketLossDetector`
        sees ``delta=1`` between consecutive packets.

        Loss is injected by skipping the frame (the ``overall_frame_identifier``
        still increments for the dropped datagram, so downstream detectors see
        a gap of size equal to the number of dropped datagrams).
        """
        rng = random.Random(self.seed)
        dt = 1.0 / self.packet_rate_hz
        total_frames = int(self.duration_s * self.packet_rate_hz)
        n_types = len(self.packet_types)

        # Burst state
        burst_remaining = 0
        # Systematic counter
        sys_counter = 0
        # Per-datagram overall_frame counter (increments on every emitted OR
        # dropped datagram, matching real F1-25 m_overallFrameIdentifier).
        overall_frame = self.start_frame

        for frame_idx in range(total_frames):
            session_time = frame_idx * dt
            # Interleave packet types in round-robin within each frame
            for type_idx, pid in enumerate(self.packet_types):
                # Determine if this packet is dropped
                dropped = False
                if self.loss_pattern == "isolated":
                    if rng.random() < self.loss_rate:
                        dropped = True
                elif self.loss_pattern == "burst":
                    if burst_remaining > 0:
                        dropped = True
                        burst_remaining -= 1
                    elif (frame_idx * n_types + type_idx) % self.burst_interval == 0:
                        # Start a new burst
                        burst_remaining = self.burst_size - 1
                        dropped = True
                elif self.loss_pattern == "systematic":
                    sys_counter += 1
                    if sys_counter % self.systematic_k == 0:
                        dropped = True

                # Always increment overall_frame (dropped datagrams still
                # consume a frame identifier in the real game's counter).
                this_frame = overall_frame
                overall_frame += 1

                if dropped:
                    continue

                header = PacketHeader(
                    packet_format=2025,
                    game_year=25,
                    game_major_version=1,
                    game_minor_version=0,
                    packet_version=1,
                    packet_id=pid,
                    session_uid=self.session_uid,
                    session_time=session_time,
                    frame_identifier=frame_idx,
                    overall_frame_identifier=this_frame,
                    player_car_index=0,
                    secondary_player_car_index=255,
                )
                body = {"_packet_id": pid, "_frame": this_frame}
                yield header, body
