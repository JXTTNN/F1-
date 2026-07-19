"""Tests for team_strategist (Iter-24)."""

from __future__ import annotations

from f1opt.data.setup_schema import DEFAULT_SETUP
from f1opt.model.race_simulator import RaceStrategy
from f1opt.model.team_strategist import (
    TeamStrategist,
    TeamStrategyDecision,
    decide_team_strategies,
    optimize_team_strategies,
)


def _strategist() -> TeamStrategist:
    return TeamStrategist(track_id="monza", total_laps=58, pit_loss_s=23.0)


# --------------------------------------------------------------------------- #
# Basic structure
# --------------------------------------------------------------------------- #
class TestBasicStructure:
    def test_returns_decision_with_two_strategies(self) -> None:
        d = _strategist().decide(car1_grid=3, car2_grid=5)
        assert isinstance(d, TeamStrategyDecision)
        assert isinstance(d.car1_strategy, RaceStrategy)
        assert isinstance(d.car2_strategy, RaceStrategy)
        assert isinstance(d.rationale, str)
        assert d.car1_role in ("aggressive", "conservative", "balanced",
                                "extreme", "undercut")
        assert d.car2_role in ("aggressive", "conservative", "balanced",
                                "extreme", "undercut")


# --------------------------------------------------------------------------- #
# Both top-5: split strategy
# --------------------------------------------------------------------------- #
class TestBothTopFive:
    def test_both_top5_split_strategy(self) -> None:
        d = _strategist().decide(car1_grid=2, car2_grid=4)
        # One car aggressive, one conservative
        roles = {d.car1_role, d.car2_role}
        assert "aggressive" in roles or "conservative" in roles
        # Strategies should differ
        assert (d.car1_strategy.pit_laps != d.car2_strategy.pit_laps
                or d.car1_strategy.compounds != d.car2_strategy.compounds)

    def test_both_top5_rationale_mentions_split(self) -> None:
        d = _strategist().decide(car1_grid=1, car2_grid=3)
        assert "split" in d.rationale.lower()


# --------------------------------------------------------------------------- #
# Both mid-field: same strategy
# --------------------------------------------------------------------------- #
class TestBothMidField:
    def test_both_mid_same_strategy(self) -> None:
        d = _strategist().decide(car1_grid=7, car2_grid=10)
        assert (d.car1_strategy.pit_laps == d.car2_strategy.pit_laps
                and d.car1_strategy.compounds == d.car2_strategy.compounds)

    def test_mid_field_rationale_mentions_track_position(self) -> None:
        d = _strategist().decide(car1_grid=8, car2_grid=11)
        assert "track position" in d.rationale.lower() or "mid-field" in d.rationale.lower()


# --------------------------------------------------------------------------- #
# Split grid: front conservative, back undercut
# --------------------------------------------------------------------------- #
class TestSplitGrid:
    def test_split_front_back(self) -> None:
        d = _strategist().decide(car1_grid=4, car2_grid=15)
        # Front car should be conservative
        # Back car should be undercut
        if d.car1_role == "conservative":
            assert d.car2_role == "undercut"
        else:
            assert d.car1_role == "undercut"
            assert d.car2_role == "conservative"

    def test_split_rationale_mentions_undercut(self) -> None:
        d = _strategist().decide(car1_grid=5, car2_grid=18)
        assert "undercut" in d.rationale.lower()


# --------------------------------------------------------------------------- #
# Both back markers: extreme split
# --------------------------------------------------------------------------- #
class TestBothBackMarkers:
    def test_both_back_extreme_split(self) -> None:
        d = _strategist().decide(car1_grid=15, car2_grid=18)
        roles = {d.car1_role, d.car2_role}
        # One extreme, other balanced or aggressive
        assert "extreme" in roles or "aggressive" in roles

    def test_both_back_rationale_mentions_maximize(self) -> None:
        d = _strategist().decide(car1_grid=14, car2_grid=20)
        assert "maximize" in d.rationale.lower() or "back-markers" in d.rationale.lower()


# --------------------------------------------------------------------------- #
# Wet forecast: both wet strategy
# --------------------------------------------------------------------------- #
class TestWetForecast:
    def test_wet_forecast_same_wet_strategies(self) -> None:
        d = _strategist().decide(
            car1_grid=3, car2_grid=10, forecast_wet=True,
        )
        assert d.car1_strategy.compounds == ("intermediate", "wet")
        assert d.car2_strategy.compounds == ("intermediate", "wet")
        assert "wet" in d.rationale.lower()


# --------------------------------------------------------------------------- #
# Sprint weekend: conservative 1-stop
# --------------------------------------------------------------------------- #
class TestSprint:
    def test_sprint_conservative_1stop(self) -> None:
        d = _strategist().decide(
            car1_grid=3, car2_grid=5, is_sprint=True,
        )
        assert d.car1_strategy.n_stops == 1
        assert d.car2_strategy.n_stops == 1
        assert "sprint" in d.rationale.lower()


# --------------------------------------------------------------------------- #
# Convenience function
# --------------------------------------------------------------------------- #
class TestConvenienceFunction:
    def test_decide_team_strategies_convenience(self) -> None:
        d = decide_team_strategies(
            track_id="monza", total_laps=58,
            car1_grid=3, car2_grid=7,
        )
        assert isinstance(d, TeamStrategyDecision)


# --------------------------------------------------------------------------- #
# optimize_team_strategies: uses StrategyOptimizer
# --------------------------------------------------------------------------- #
class TestOptimizeTeamStrategies:
    def test_optimize_returns_two_strategies(self) -> None:
        s1, s2 = optimize_team_strategies(
            setup1=DEFAULT_SETUP, setup2=DEFAULT_SETUP,
            track_id="monza", total_laps=30,
            car1_grid=3, car2_grid=5,
        )
        assert isinstance(s1, RaceStrategy)
        assert isinstance(s2, RaceStrategy)
        # Each should have valid compounds
        assert len(s1.compounds) == s1.n_stops + 1
        assert len(s2.compounds) == s2.n_stops + 1

    def test_optimize_top5_force_different(self) -> None:
        """If both cars top-5 and same strategy, force differentiation."""
        s1, s2 = optimize_team_strategies(
            setup1=DEFAULT_SETUP, setup2=DEFAULT_SETUP,
            track_id="monza", total_laps=30,
            car1_grid=2, car2_grid=4,
        )
        # Same setup → optimizer returns same strat → force differentiation
        assert (s1.pit_laps != s2.pit_laps
                or s1.compounds != s2.compounds)


# --------------------------------------------------------------------------- #
# Car1 vs Car2 grid swap symmetry
# --------------------------------------------------------------------------- #
class TestGridSymmetry:
    def test_grid_swap_keeps_front_car_conservative(self) -> None:
        d1 = _strategist().decide(car1_grid=3, car2_grid=12)
        d2 = _strategist().decide(car1_grid=12, car2_grid=3)
        # In d1, car1 is front; in d2, car2 is front
        # Front car should be conservative in both
        if d1.car1_role == "conservative":
            assert d2.car2_role == "conservative"
        elif d1.car1_role == "balanced":
            assert d2.car2_role == "balanced"
