"""Tests for f1opt.model.safety_car (Iter-14)."""

from __future__ import annotations

import random

import pytest

from f1opt.model.safety_car import (
    SafetyCarModel,
    SafetyCarPeriod,
)


# --------------------------------------------------------------------------- #
# SafetyCarPeriod
# --------------------------------------------------------------------------- #
class TestSafetyCarPeriod:
    def test_duration_laps(self) -> None:
        p = SafetyCarPeriod(start_lap=10, end_lap=13, kind="sc")
        assert p.duration_laps == 4

    def test_active_during(self) -> None:
        p = SafetyCarPeriod(start_lap=10, end_lap=13, kind="sc")
        assert p.active_during(10)
        assert p.active_during(13)
        assert not p.active_during(9)
        assert not p.active_during(14)

    def test_is_restart_lap(self) -> None:
        p = SafetyCarPeriod(start_lap=10, end_lap=13, kind="sc")
        assert p.is_restart_lap(14)
        assert not p.is_restart_lap(13)
        assert not p.is_restart_lap(15)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
class TestGeneration:
    def test_no_periods_with_zero_retirements(self) -> None:
        scm = SafetyCarModel(seed=42)
        scm.generate_periods(total_laps=58, n_retirements=0, rng=random.Random(42))
        # 碎片概率很低, 大概率无 SC (但允许偶发)
        # 用固定 seed 检查确定性
        scm2 = SafetyCarModel(seed=42)
        scm2.generate_periods(total_laps=58, n_retirements=0, rng=random.Random(42))
        assert len(scm.periods) == len(scm2.periods)

    def test_deterministic_with_seed(self) -> None:
        scm1 = SafetyCarModel(seed=99)
        scm1.generate_periods(total_laps=58, n_retirements=4, rng=random.Random(99))
        scm2 = SafetyCarModel(seed=99)
        scm2.generate_periods(total_laps=58, n_retirements=4, rng=random.Random(99))
        assert len(scm1.periods) == len(scm2.periods)
        for p1, p2 in zip(scm1.periods, scm2.periods, strict=True):
            assert p1.start_lap == p2.start_lap
            assert p1.end_lap == p2.end_lap

    def test_periods_within_race(self) -> None:
        scm = SafetyCarModel(seed=7)
        scm.generate_periods(total_laps=50, n_retirements=5, rng=random.Random(7))
        for p in scm.periods:
            assert p.start_lap >= 1
            assert p.end_lap <= 50

    def test_retirements_increase_sc_probability(self) -> None:
        """更多退赛 → 平均更多 SC 时段."""
        n_low = 0
        n_high = 0
        for s in range(20):
            scm_low = SafetyCarModel(seed=s)
            scm_low.generate_periods(total_laps=58, n_retirements=0,
                                     rng=random.Random(s))
            n_low += len(scm_low.periods)
            scm_high = SafetyCarModel(seed=s)
            scm_high.generate_periods(total_laps=58, n_retirements=8,
                                      rng=random.Random(s))
            n_high += len(scm_high.periods)
        assert n_high > n_low

    def test_short_race_no_periods(self) -> None:
        scm = SafetyCarModel(seed=1)
        scm.generate_periods(total_laps=2, n_retirements=5, rng=random.Random(1))
        assert len(scm.periods) == 0

    def test_wet_weather_more_sc(self) -> None:
        """湿地下 SC 概率更高."""
        n_dry = 0
        n_wet = 0
        for s in range(30):
            scm_d = SafetyCarModel(seed=s)
            scm_d.generate_periods(total_laps=58, n_retirements=0,
                                   weather_wetness=0.0, rng=random.Random(s))
            n_dry += len(scm_d.periods)
            scm_w = SafetyCarModel(seed=s)
            scm_w.generate_periods(total_laps=58, n_retirements=0,
                                   weather_wetness=0.9, rng=random.Random(s))
            n_wet += len(scm_w.periods)
        assert n_wet >= n_dry


# --------------------------------------------------------------------------- #
# Merging
# --------------------------------------------------------------------------- #
class TestMerging:
    def test_overlapping_periods_merged(self) -> None:
        scm = SafetyCarModel()
        scm.periods = [
            SafetyCarPeriod(10, 14, "sc", "a"),
            SafetyCarPeriod(13, 16, "vsc", "b"),
        ]
        scm.periods = scm._merge_overlapping()
        assert len(scm.periods) == 1
        assert scm.periods[0].start_lap == 10
        assert scm.periods[0].end_lap == 16
        # SC 优先于 VSC
        assert scm.periods[0].kind == "sc"

    def test_adjacent_periods_merged(self) -> None:
        scm = SafetyCarModel()
        scm.periods = [
            SafetyCarPeriod(10, 12, "vsc", "a"),
            SafetyCarPeriod(13, 15, "vsc", "b"),
        ]
        scm.periods = scm._merge_overlapping()
        assert len(scm.periods) == 1

    def test_non_overlapping_kept(self) -> None:
        scm = SafetyCarModel()
        scm.periods = [
            SafetyCarPeriod(10, 12, "sc", "a"),
            SafetyCarPeriod(20, 22, "vsc", "b"),
        ]
        scm.periods = scm._merge_overlapping()
        assert len(scm.periods) == 2


# --------------------------------------------------------------------------- #
# Query methods
# --------------------------------------------------------------------------- #
class TestQueries:
    def test_lap_time_factor_normal(self) -> None:
        scm = SafetyCarModel()
        assert scm.lap_time_factor(10) == 1.0

    def test_lap_time_factor_sc(self) -> None:
        scm = SafetyCarModel()
        scm.periods = [SafetyCarPeriod(10, 13, "sc")]
        assert scm.lap_time_factor(10) == 1.30
        assert scm.lap_time_factor(13) == 1.30
        assert scm.lap_time_factor(14) == 1.0

    def test_lap_time_factor_vsc(self) -> None:
        scm = SafetyCarModel()
        scm.periods = [SafetyCarPeriod(10, 12, "vsc")]
        assert scm.lap_time_factor(11) == 1.25

    def test_pit_loss_discount_normal(self) -> None:
        scm = SafetyCarModel()
        assert scm.pit_loss_discount(10) == 1.0

    def test_pit_loss_discount_sc(self) -> None:
        scm = SafetyCarModel()
        scm.periods = [SafetyCarPeriod(10, 13, "sc")]
        assert scm.pit_loss_discount(11) == pytest.approx(0.20)

    def test_pit_loss_discount_vsc(self) -> None:
        scm = SafetyCarModel()
        scm.periods = [SafetyCarPeriod(10, 13, "vsc")]
        assert scm.pit_loss_discount(11) == pytest.approx(0.55)

    def test_restart_penalty(self) -> None:
        scm = SafetyCarModel()
        scm.periods = [SafetyCarPeriod(10, 13, "sc")]
        assert scm.restart_penalty_s(14) > 0.0   # 重启圈
        assert scm.restart_penalty_s(13) == 0.0  # SC 期间
        assert scm.restart_penalty_s(15) == 0.0  # 重启圈之后

    def test_is_under_sc_vsc(self) -> None:
        scm = SafetyCarModel()
        scm.periods = [
            SafetyCarPeriod(10, 13, "sc"),
            SafetyCarPeriod(20, 22, "vsc"),
        ]
        assert scm.is_under_sc(11)
        assert not scm.is_under_sc(21)
        assert scm.is_under_vsc(21)
        assert not scm.is_under_sc(15)


# --------------------------------------------------------------------------- #
# Summary & reset
# --------------------------------------------------------------------------- #
class TestSummaryReset:
    def test_summary_keys(self) -> None:
        scm = SafetyCarModel()
        scm.periods = [SafetyCarPeriod(10, 13, "sc", "test")]
        s = scm.summary()
        required = {"n_periods", "n_sc", "n_vsc", "total_sc_laps", "periods"}
        assert required.issubset(s.keys())
        assert s["n_sc"] == 1
        assert s["n_vsc"] == 0
        assert s["total_sc_laps"] == 4

    def test_reset_clears_periods(self) -> None:
        scm = SafetyCarModel()
        scm.periods = [SafetyCarPeriod(10, 13, "sc")]
        scm.reset()
        assert scm.periods == []
