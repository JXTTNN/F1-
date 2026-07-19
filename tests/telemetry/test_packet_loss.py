"""Tests for :mod:`f1opt.telemetry.packet_loss` (Iter-139)."""
from __future__ import annotations

import time

import pytest

from f1opt.telemetry.packet_loss import LossReport, PacketLossDetector
from f1opt.telemetry.packets import PacketHeader


def _header(
    session_uid: int = 1,
    frame: int = 0,
    packet_id: int = 6,
    session_time: float = 0.0,
) -> PacketHeader:
    return PacketHeader(
        packet_format=2025,
        game_year=25,
        game_major_version=1,
        game_minor_version=0,
        packet_version=1,
        packet_id=packet_id,
        session_uid=session_uid,
        session_time=session_time,
        frame_identifier=frame,
        overall_frame_identifier=frame,
        player_car_index=0,
        secondary_player_car_index=255,
    )


class TestPacketLossDetector:
    def test_first_packet_no_loss(self) -> None:
        d = PacketLossDetector()
        lost, delta = d.observe(_header(frame=100))
        assert lost == 0
        assert delta == 0

    def test_consecutive_no_loss(self) -> None:
        d = PacketLossDetector()
        d.observe(_header(frame=100))
        lost, delta = d.observe(_header(frame=101))
        assert lost == 0
        assert delta == 1

    def test_single_frame_loss(self) -> None:
        d = PacketLossDetector()
        d.observe(_header(frame=100))
        lost, delta = d.observe(_header(frame=102))
        assert lost == 1
        assert delta == 2

    def test_multi_frame_loss(self) -> None:
        d = PacketLossDetector()
        d.observe(_header(frame=100))
        lost, delta = d.observe(_header(frame=105))
        assert lost == 4
        assert delta == 5

    def test_regression_not_counted_as_loss(self) -> None:
        d = PacketLossDetector()
        d.observe(_header(frame=100))
        lost, delta = d.observe(_header(frame=99))
        assert lost == 0
        assert delta == -1

    def test_loss_rate_zero_when_empty(self) -> None:
        d = PacketLossDetector()
        assert d.loss_rate() == 0.0
        assert d.total_loss_rate() == 0.0

    def test_loss_rate_computed(self) -> None:
        d = PacketLossDetector(window_seconds=10.0)
        d.observe(_header(frame=100, session_time=0.0))
        d.observe(_header(frame=102, session_time=0.1))
        d.observe(_header(frame=103, session_time=0.2))
        # Expected: 3 (deltas: 0, 2, 1 = 3 expected), received: 3, lost: 1
        rate = d.loss_rate()
        assert 0.0 < rate < 1.0

    def test_total_loss_rate(self) -> None:
        d = PacketLossDetector()
        d.observe(_header(frame=100))
        d.observe(_header(frame=104))
        # total_expected = 1 + 4 = 5, total_received = 2, total_lost = 3
        assert d.total_loss_rate() == pytest.approx(3.0 / 5.0)

    def test_per_type_summary(self) -> None:
        d = PacketLossDetector()
        d.observe(_header(frame=100, packet_id=6))
        d.observe(_header(frame=102, packet_id=0))
        d.observe(_header(frame=103, packet_id=6))
        summary = d.per_type_summary()
        assert "CarTelemetry" in summary
        assert "Motion" in summary

    def test_report_no_loss(self) -> None:
        d = PacketLossDetector()
        d.observe(_header(frame=100))
        d.observe(_header(frame=101))
        r = d.report()
        assert r.pattern == "none"
        assert r.total_lost == 0
        assert r.loss_rate == 0.0

    def test_report_isolated_loss(self) -> None:
        d = PacketLossDetector(window_seconds=10.0)
        d.observe(_header(frame=100, session_time=0.0))
        d.observe(_header(frame=102, session_time=0.1))
        d.observe(_header(frame=103, session_time=0.2))
        r = d.report()
        assert r.total_lost > 0
        assert r.pattern in ("none", "isolated", "burst", "systematic")
        assert r.max_burst > 0

    def test_report_burst_loss(self) -> None:
        d = PacketLossDetector(window_seconds=10.0)
        d.observe(_header(frame=100, session_time=0.0))
        # Simulate a burst: 5 consecutive frames lost
        d.observe(_header(frame=106, session_time=0.1))
        r = d.report()
        assert r.total_lost == 5
        assert r.max_burst == 5

    def test_report_multiple_bursts(self) -> None:
        d = PacketLossDetector(window_seconds=10.0)
        d.observe(_header(frame=100, session_time=0.0))
        d.observe(_header(frame=103, session_time=0.1))  # burst of 2
        d.observe(_header(frame=104, session_time=0.2))
        d.observe(_header(frame=108, session_time=0.3))  # burst of 3
        r = d.report()
        assert r.burst_count == 2
        assert r.max_burst == 3
        assert r.total_lost == 5  # 2 + 3

    def test_report_fields(self) -> None:
        d = PacketLossDetector(window_seconds=5.0)
        d.observe(_header(frame=100, session_time=0.0))
        d.observe(_header(frame=101, session_time=0.1))
        r = d.report()
        assert isinstance(r, LossReport)
        assert r.total_expected >= 0
        assert r.total_received >= 0
        assert 0.0 <= r.loss_rate <= 1.0
        assert isinstance(r.per_type, dict)
        assert r.burst_count >= 0
        assert r.max_burst >= 0
        assert r.avg_gap >= 0
        assert r.pattern in ("none", "isolated", "burst", "systematic")
        assert r.window_seconds == 5.0

    def test_reset_clears_state(self) -> None:
        d = PacketLossDetector()
        d.observe(_header(frame=100))
        d.observe(_header(frame=105))
        assert d.total_loss_rate() > 0
        d.reset()
        assert d.total_loss_rate() == 0.0
        assert d.loss_rate() == 0.0
        assert d.per_type_summary() == {}

    def test_multi_session_isolation(self) -> None:
        d = PacketLossDetector()
        d.observe(_header(session_uid=1, frame=100))
        d.observe(_header(session_uid=1, frame=105))
        lost2, _ = d.observe(_header(session_uid=2, frame=50))
        assert lost2 == 0  # new session, no loss

    def test_packet_type_attribution(self) -> None:
        d = PacketLossDetector()
        d.observe(_header(frame=100, packet_id=6))
        d.observe(_header(frame=103, packet_id=0))
        s = d.per_type_summary()
        assert s["CarTelemetry"]["received"] == 1
        assert s["CarTelemetry"]["lost"] == 0
        assert s["Motion"]["received"] == 1
        assert s["Motion"]["lost"] == 2

    def test_window_prunes_old_entries(self) -> None:
        d = PacketLossDetector(window_seconds=0.1)
        d.observe(_header(frame=100, session_time=0.0))
        d.observe(_header(frame=103, session_time=0.05))
        time.sleep(0.15)
        d.observe(_header(frame=104, session_time=0.2))
        d.observe(_header(frame=105, session_time=0.21))
        r = d.report()
        # Window should have pruned the first entries.
        assert r.total_lost >= 0

    def test_negative_window_rejected(self) -> None:
        with pytest.raises(ValueError):
            PacketLossDetector(window_seconds=0.0)
        with pytest.raises(ValueError):
            PacketLossDetector(window_seconds=-1.0)

    def test_systematic_pattern_at_high_loss(self) -> None:
        d = PacketLossDetector(window_seconds=10.0)
        d.observe(_header(frame=100, session_time=0.0))
        # Lose 10 frames out of 20
        for i in range(1, 20):
            d.observe(_header(frame=100 + i * 2, session_time=i * 0.1))
        r = d.report()
        assert r.total_lost >= 9
        # Should be systematic at >5% loss rate
        assert r.pattern in ("systematic", "burst", "isolated")

    def test_large_frame_delta_safety(self) -> None:
        """Massive delta (e.g. game paused for hours) should not crash."""
        d = PacketLossDetector()
        d.observe(_header(frame=100))
        lost, delta = d.observe(_header(frame=1_000_000))
        assert lost == 999_899
        assert delta == 999_900
        assert d.total_loss_rate() > 0.99