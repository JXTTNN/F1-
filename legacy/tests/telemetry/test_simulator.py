"""Tests for :class:`f1opt.telemetry.simulator.TelemetrySimulator` (Iter-163).

The simulator generates synthetic F1-25 packet streams for stress testing the
telemetry pipeline (packet loss detection, frame tracking, validation,
analytics). It supports configurable packet rates, session durations, and
loss injection patterns.
"""
from __future__ import annotations

import pytest

from f1opt.telemetry.packets import PacketHeader


class TestSimulatorConstruction:
    def test_construct_with_defaults(self) -> None:
        """Simulator constructs with default parameters."""
        from f1opt.telemetry.simulator import TelemetrySimulator
        sim = TelemetrySimulator()
        assert sim.session_uid > 0
        assert sim.packet_rate_hz > 0
        assert sim.duration_s > 0

    def test_construct_with_custom_params(self) -> None:
        """Simulator accepts custom session_uid, rate, duration."""
        from f1opt.telemetry.simulator import TelemetrySimulator
        sim = TelemetrySimulator(
            session_uid=12345, packet_rate_hz=60, duration_s=10.0
        )
        assert sim.session_uid == 12345
        assert sim.packet_rate_hz == 60
        assert sim.duration_s == 10.0


class TestStreamGeneration:
    def test_stream_yields_packet_headers(self) -> None:
        """generate_stream yields PacketHeader objects."""
        from f1opt.telemetry.simulator import TelemetrySimulator
        sim = TelemetrySimulator(duration_s=0.1, packet_rate_hz=60)
        packets = list(sim.generate_stream())
        assert len(packets) > 0
        header, _body = packets[0]
        assert isinstance(header, PacketHeader)

    def test_stream_packet_count_matches_rate_times_duration(self) -> None:
        """Total packets ≈ rate × duration (per packet type)."""
        from f1opt.telemetry.simulator import TelemetrySimulator
        # 1 second at 60 Hz, single packet type → ~60 packets
        sim = TelemetrySimulator(
            duration_s=1.0, packet_rate_hz=60, packet_types=(6,)
        )
        packets = list(sim.generate_stream())
        assert 55 <= len(packets) <= 65  # ~60 ± tolerance

    def test_session_uid_propagated_to_headers(self) -> None:
        """All generated headers carry the simulator's session_uid."""
        from f1opt.telemetry.simulator import TelemetrySimulator
        sim = TelemetrySimulator(
            session_uid=999, duration_s=0.1, packet_rate_hz=60
        )
        for header, _ in sim.generate_stream():
            assert header.session_uid == 999

    def test_frame_identifiers_monotonic(self) -> None:
        """overall_frame_identifier increments by 1 per packet (no loss)."""
        from f1opt.telemetry.simulator import TelemetrySimulator
        sim = TelemetrySimulator(
            duration_s=0.5, packet_rate_hz=60, packet_types=(6,),
            loss_pattern="none",
        )
        packets = list(sim.generate_stream())
        frames = [h.overall_frame_identifier for h, _ in packets]
        for i in range(1, len(frames)):
            assert frames[i] == frames[i - 1] + 1

    def test_session_time_increases(self) -> None:
        """session_time increases monotonically with frame index."""
        from f1opt.telemetry.simulator import TelemetrySimulator
        sim = TelemetrySimulator(
            duration_s=0.5, packet_rate_hz=60, packet_types=(6,)
        )
        packets = list(sim.generate_stream())
        times = [h.session_time for h, _ in packets]
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1]

    def test_packet_id_matches_requested_type(self) -> None:
        """Generated packets carry the requested packet_id."""
        from f1opt.telemetry.simulator import TelemetrySimulator
        sim = TelemetrySimulator(
            duration_s=0.1, packet_rate_hz=60, packet_types=(6,)
        )
        for header, _ in sim.generate_stream():
            assert header.packet_id == 6


class TestLossInjection:
    def test_no_loss_yields_all_packets(self) -> None:
        """loss_pattern='none' keeps all packets."""
        from f1opt.telemetry.simulator import TelemetrySimulator
        sim = TelemetrySimulator(
            duration_s=0.5, packet_rate_hz=60, packet_types=(6,),
            loss_pattern="none",
        )
        packets = list(sim.generate_stream())
        frames = [h.overall_frame_identifier for h, _ in packets]
        # No gaps in frame sequence
        for i in range(1, len(frames)):
            assert frames[i] == frames[i - 1] + 1

    def test_isolated_loss_creates_gaps(self) -> None:
        """loss_pattern='isolated' with p=0.1 creates gaps."""
        from f1opt.telemetry.simulator import TelemetrySimulator
        sim = TelemetrySimulator(
            duration_s=2.0, packet_rate_hz=60, packet_types=(6,),
            loss_pattern="isolated", loss_rate=0.1, seed=42,
        )
        packets = list(sim.generate_stream())
        frames = [h.overall_frame_identifier for h, _ in packets]
        # At least one gap (delta > 1) should exist
        gaps = [frames[i] - frames[i - 1] for i in range(1, len(frames))]
        assert any(g > 1 for g in gaps)
        # Loss rate roughly 10% (allow tolerance)
        expected_total = 120  # 2s × 60Hz
        actual_received = len(packets)
        loss_fraction = 1 - actual_received / expected_total
        assert 0.03 < loss_fraction < 0.20

    def test_burst_loss_creates_large_gaps(self) -> None:
        """loss_pattern='burst' creates gaps of size >= burst_size."""
        from f1opt.telemetry.simulator import TelemetrySimulator
        sim = TelemetrySimulator(
            duration_s=5.0, packet_rate_hz=60, packet_types=(6,),
            loss_pattern="burst", burst_size=10, burst_interval=100,
            seed=42,
        )
        packets = list(sim.generate_stream())
        frames = [h.overall_frame_identifier for h, _ in packets]
        gaps = [frames[i] - frames[i - 1] for i in range(1, len(frames))]
        # At least one gap >= 10 (burst)
        assert any(g >= 10 for g in gaps)

    def test_systematic_loss_drops_every_kth(self) -> None:
        """loss_pattern='systematic' drops every K-th packet."""
        from f1opt.telemetry.simulator import TelemetrySimulator
        sim = TelemetrySimulator(
            duration_s=1.0, packet_rate_hz=60, packet_types=(6,),
            loss_pattern="systematic", systematic_k=5, seed=42,
        )
        packets = list(sim.generate_stream())
        frames = [h.overall_frame_identifier for h, _ in packets]
        # Every 5th frame should be missing → gaps of exactly 2
        gaps = [frames[i] - frames[i - 1] for i in range(1, len(frames))]
        assert all(g in (1, 2) for g in gaps)
        # Some gaps should be exactly 2 (the systematic drop)
        assert any(g == 2 for g in gaps)

    def test_reproducible_with_seed(self) -> None:
        """Same seed produces same packet stream."""
        from f1opt.telemetry.simulator import TelemetrySimulator
        sim1 = TelemetrySimulator(
            duration_s=0.5, packet_rate_hz=60, packet_types=(6,),
            loss_pattern="isolated", loss_rate=0.2, seed=123,
        )
        sim2 = TelemetrySimulator(
            duration_s=0.5, packet_rate_hz=60, packet_types=(6,),
            loss_pattern="isolated", loss_rate=0.2, seed=123,
        )
        p1 = [(h.overall_frame_identifier, h.session_time)
              for h, _ in sim1.generate_stream()]
        p2 = [(h.overall_frame_identifier, h.session_time)
              for h, _ in sim2.generate_stream()]
        assert p1 == p2


class TestMultiplePacketTypes:
    def test_multiple_packet_types_interleave(self) -> None:
        """Multiple packet types are interleaved in the stream."""
        from f1opt.telemetry.simulator import TelemetrySimulator
        sim = TelemetrySimulator(
            duration_s=0.2, packet_rate_hz=60,
            packet_types=(0, 6, 7),  # Motion, CarTelemetry, CarStatus
        )
        packets = list(sim.generate_stream())
        packet_ids = {h.packet_id for h, _ in packets}
        assert packet_ids == {0, 6, 7}

    def test_multi_type_total_count(self) -> None:
        """Total count = rate × duration × num_types (approx)."""
        from f1opt.telemetry.simulator import TelemetrySimulator
        sim = TelemetrySimulator(
            duration_s=1.0, packet_rate_hz=60,
            packet_types=(0, 6, 7),
        )
        packets = list(sim.generate_stream())
        # 3 types × 60 Hz × 1s = 180 ± tolerance
        assert 165 <= len(packets) <= 195

    def test_multi_type_overall_frame_increments_per_datagram(self) -> None:
        """Iter-163 RED: overall_frame_identifier increments by 1 for EVERY
        emitted datagram, regardless of packet type.

        F1-25 spec: m_overallFrameIdentifier is a per-datagram counter that
        increments on every UDP packet the game sends. With multiple packet
        types interleaved at the same frame_idx, each emitted datagram must
        still carry a unique, strictly-incrementing overall_frame_identifier.
        Otherwise downstream PacketLossDetector sees delta=0 between
        same-frame different-type packets and computes negative loss.
        """
        from f1opt.telemetry.simulator import TelemetrySimulator
        sim = TelemetrySimulator(
            duration_s=0.5, packet_rate_hz=60,
            packet_types=(0, 6, 7),  # 3 types × 60 Hz × 0.5s = 90 packets
            loss_pattern="none",
        )
        packets = list(sim.generate_stream())
        frames = [h.overall_frame_identifier for h, _ in packets]
        # Every emitted datagram must have a unique overall_frame_identifier
        # that increments by exactly 1 (no loss scenario).
        assert len(set(frames)) == len(frames), (
            "overall_frame_identifier must be unique per datagram"
        )
        for i in range(1, len(frames)):
            assert frames[i] == frames[i - 1] + 1, (
                f"overall_frame_identifier must increment by 1 per datagram; "
                f"got delta={frames[i] - frames[i - 1]} at index {i}"
            )

    def test_multi_type_no_loss_yields_zero_detected_loss(self) -> None:
        """Iter-163 RED: a clean multi-type stream must produce ZERO detected
        loss by PacketLossDetector. Catches the bug where same-frame
        different-type packets share overall_frame_identifier, causing
        delta=0 and negative loss.
        """
        from f1opt.telemetry.simulator import TelemetrySimulator
        from f1opt.telemetry.packet_loss import PacketLossDetector
        sim = TelemetrySimulator(
            duration_s=0.5, packet_rate_hz=60,
            packet_types=(0, 6, 7),
            loss_pattern="none",
        )
        detector = PacketLossDetector(window_seconds=5.0)
        for header, body in sim.generate_stream():
            detector.observe(header, body, session_time=header.session_time)
        report = detector.report()
        assert report.total_lost == 0, (
            f"Clean stream should have 0 detected loss, "
            f"got {report.total_lost} (rate={report.loss_rate:.4f})"
        )
        assert report.total_expected == report.total_received


class TestInvalidParams:
    def test_negative_duration_raises(self) -> None:
        """Negative duration raises ValueError."""
        from f1opt.telemetry.simulator import TelemetrySimulator
        with pytest.raises(ValueError, match="duration"):
            TelemetrySimulator(duration_s=-1.0)

    def test_zero_rate_raises(self) -> None:
        """Zero packet rate raises ValueError."""
        from f1opt.telemetry.simulator import TelemetrySimulator
        with pytest.raises(ValueError, match="rate"):
            TelemetrySimulator(packet_rate_hz=0)

    def test_negative_loss_rate_raises(self) -> None:
        """Negative loss_rate raises ValueError."""
        from f1opt.telemetry.simulator import TelemetrySimulator
        with pytest.raises(ValueError, match="loss_rate"):
            TelemetrySimulator(loss_pattern="isolated", loss_rate=-0.1)

    def test_loss_rate_above_one_raises(self) -> None:
        """loss_rate > 1 raises ValueError."""
        from f1opt.telemetry.simulator import TelemetrySimulator
        with pytest.raises(ValueError, match="loss_rate"):
            TelemetrySimulator(loss_pattern="isolated", loss_rate=1.5)

    def test_burst_pattern_requires_burst_size(self) -> None:
        """burst loss requires burst_size > 0."""
        from f1opt.telemetry.simulator import TelemetrySimulator
        with pytest.raises(ValueError, match="burst_size"):
            TelemetrySimulator(loss_pattern="burst", burst_size=0)

    def test_systematic_pattern_requires_k(self) -> None:
        """systematic loss requires systematic_k >= 2."""
        from f1opt.telemetry.simulator import TelemetrySimulator
        with pytest.raises(ValueError, match="systematic_k"):
            TelemetrySimulator(loss_pattern="systematic", systematic_k=1)
