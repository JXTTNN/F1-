"""Tests for drs_2026 (Iter-31)."""

from __future__ import annotations

from f1opt.model.drs_2026 import DRS2026State, drs_for_lap


class TestBasic:
    def test_returns_state(self) -> None:
        s = drs_for_lap("monza", lap=5, n_drs_zones=2, gap_ahead_s=0.5)
        assert isinstance(s, DRS2026State)

    def test_drs_active_when_close(self) -> None:
        s = drs_for_lap("monza", lap=5, n_drs_zones=2, gap_ahead_s=0.5)
        assert s.drs_available is True
        assert s.drs_active_zones == 2
        assert s.lap_time_gain_s > 0

    def test_drs_inactive_when_far(self) -> None:
        s = drs_for_lap("monza", lap=5, n_drs_zones=2, gap_ahead_s=2.0)
        assert s.drs_available is False
        assert s.drs_active_zones == 0
        assert s.lap_time_gain_s == 0.0

    def test_drs_inactive_no_car_ahead(self) -> None:
        s = drs_for_lap("monza", lap=5, n_drs_zones=2, gap_ahead_s=None)
        assert s.drs_available is False


class TestFIA2026Rules:
    def test_drs_disabled_lap_1(self) -> None:
        s = drs_for_lap("monza", lap=1, n_drs_zones=2, gap_ahead_s=0.3)
        assert s.drs_available is False
        assert "lap 1" in s.reason.lower()

    def test_drs_disabled_after_sc(self) -> None:
        """SC ended lap 10, DRS disabled laps 11 & 12, enabled lap 13."""
        s1 = drs_for_lap("monza", lap=11, gap_ahead_s=0.3, sc_just_ended_lap=10)
        s2 = drs_for_lap("monza", lap=12, gap_ahead_s=0.3, sc_just_ended_lap=10)
        s3 = drs_for_lap("monza", lap=13, gap_ahead_s=0.3, sc_just_ended_lap=10)
        assert s1.drs_available is False
        assert s2.drs_available is False
        assert s3.drs_available is True

    def test_drs_disabled_wet(self) -> None:
        s = drs_for_lap("monza", lap=10, gap_ahead_s=0.3, track_wetness=0.5)
        assert s.drs_available is False
        assert "wet" in s.reason.lower()


class TestGainScaling:
    def test_gain_scales_with_zones(self) -> None:
        s1 = drs_for_lap("monza", lap=5, n_drs_zones=1, gap_ahead_s=0.5)
        s3 = drs_for_lap("monza", lap=5, n_drs_zones=3, gap_ahead_s=0.5)
        assert s3.lap_time_gain_s > s1.lap_time_gain_s

    def test_threshold_is_1s(self) -> None:
        """DRS threshold is 1.0s exactly."""
        s_at = drs_for_lap("monza", lap=5, n_drs_zones=2, gap_ahead_s=1.0)
        s_over = drs_for_lap("monza", lap=5, n_drs_zones=2, gap_ahead_s=1.1)
        assert s_at.drs_available is True
        assert s_over.drs_available is False
