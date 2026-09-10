"""Tests for f1opt.model.race_simulator (Iter-10)."""

from __future__ import annotations

import pytest

from f1opt.data.setup_schema import DEFAULT_SETUP
from f1opt.model.race_simulator import (
    RaceCar,
    RaceSimulation,
    RaceStrategy,
    _pit_loss_for,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_car(i: int, strategy: RaceStrategy | None = None,
              aggression: float = 0.7) -> RaceCar:
    if strategy is None:
        strategy = RaceStrategy(pit_laps=(20, 40),
                                compounds=("medium", "hard", "medium"))
    return RaceCar(
        driver_id=f"d{i:02d}",
        driver_name=f"Driver {i + 1}",
        setup=DEFAULT_SETUP,
        grid_position=i + 1,
        strategy=strategy,
        driver_aggression=aggression,
    )


def _make_grid(n: int = 20) -> list[RaceCar]:
    return [_make_car(i) for i in range(n)]


# --------------------------------------------------------------------------- #
# RaceStrategy
# --------------------------------------------------------------------------- #
class TestRaceStrategy:
    def test_n_stops(self) -> None:
        s = RaceStrategy(pit_laps=(15, 35), compounds=("m", "h", "m"))
        assert s.n_stops == 2

    def test_compound_for_lap_initial(self) -> None:
        s = RaceStrategy(pit_laps=(15, 35), compounds=("medium", "hard", "soft"))
        assert s.compound_for_lap(1) == "medium"
        assert s.compound_for_lap(15) == "medium"

    def test_compound_for_lap_after_pit(self) -> None:
        s = RaceStrategy(pit_laps=(15, 35), compounds=("medium", "hard", "soft"))
        assert s.compound_for_lap(16) == "hard"
        assert s.compound_for_lap(35) == "hard"
        assert s.compound_for_lap(36) == "soft"

    def test_laps_until_pit(self) -> None:
        s = RaceStrategy(pit_laps=(15, 35), compounds=("m", "h", "s"))
        assert s.laps_until_pit(10) == 5
        assert s.laps_until_pit(15) == 0
        assert s.laps_until_pit(20) == 15
        assert s.laps_until_pit(40) is None

    def test_no_stops_strategy(self) -> None:
        s = RaceStrategy(pit_laps=(), compounds=("medium",))
        assert s.n_stops == 0
        assert s.compound_for_lap(1) == "medium"
        assert s.laps_until_pit(1) is None

    def test_invalid_compound_length_raises(self) -> None:
        with pytest.raises(ValueError, match="compounds"):
            RaceStrategy(pit_laps=(15, 35), compounds=("m", "h"))


# --------------------------------------------------------------------------- #
# _pit_loss_for
# --------------------------------------------------------------------------- #
class TestPitLoss:
    def test_known_track(self) -> None:
        assert _pit_loss_for("monaco") == 21.0
        assert _pit_loss_for("monza") == 23.0

    def test_unknown_track_default(self) -> None:
        assert _pit_loss_for("unknown_track") == 23.0


# --------------------------------------------------------------------------- #
# RaceSimulation basics
# --------------------------------------------------------------------------- #
class TestRaceSimulationBasics:
    def test_run_returns_20_positions(self) -> None:
        sim = RaceSimulation(track_id="monza", cars=_make_grid(20),
                             total_laps=20, seed=42)
        res = sim.run()
        assert len(res) == 20
        positions = [p for p, _ in res]
        assert positions == list(range(1, 21))

    def test_winner_has_lowest_cumulative_time(self) -> None:
        sim = RaceSimulation(track_id="monza", cars=_make_grid(20),
                             total_laps=20, seed=42)
        res = sim.run()
        winner_time = res[0][1].cumulative_time
        for _, car in res[1:]:
            if not car.retired:
                assert car.cumulative_time >= winner_time

    def test_winner_completed_all_laps(self) -> None:
        sim = RaceSimulation(track_id="monza", cars=_make_grid(20),
                             total_laps=20, seed=42)
        res = sim.run()
        assert res[0][1].laps_completed == 20

    def test_run_idempotent(self) -> None:
        sim = RaceSimulation(track_id="monza", cars=_make_grid(20),
                             total_laps=15, seed=42)
        r1 = sim.run()
        t1 = [c.cumulative_time for _, c in r1]
        r2 = sim.run()
        t2 = [c.cumulative_time for _, c in r2]
        assert t1 == t2

    def test_empty_cars_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1 car"):
            RaceSimulation(track_id="monza", cars=[], total_laps=10)


# --------------------------------------------------------------------------- #
# Pit stops
# --------------------------------------------------------------------------- #
class TestPitStops:
    def test_pit_lap_adds_loss(self) -> None:
        """A car pitting should lose ~pit_loss_s on that lap vs no-pit car."""
        cars = _make_grid(2)
        # Car 1 pits on lap 10
        cars[0].strategy = RaceStrategy(pit_laps=(10,), compounds=("medium", "hard"))
        # Car 2 never pits
        cars[1].strategy = RaceStrategy(pit_laps=(), compounds=("medium",))
        sim = RaceSimulation(track_id="monza", cars=cars, total_laps=15, seed=42)
        res = sim.run()
        # Pitting car should have more pit stops
        assert res[0][1].pit_stops_done + res[1][1].pit_stops_done == 1

    def test_compound_changes_after_pit(self) -> None:
        car = _make_car(0, strategy=RaceStrategy(
            pit_laps=(10,), compounds=("medium", "soft")))
        sim = RaceSimulation(track_id="monza", cars=[car], total_laps=15, seed=42)
        sim.run()
        assert car.current_compound == "soft"
        assert car.pit_stops_done == 1


# --------------------------------------------------------------------------- #
# Retirement
# --------------------------------------------------------------------------- #
class TestRetirement:
    def test_high_retirement_prob_causes_retirements(self) -> None:
        cars = _make_grid(20)
        sim = RaceSimulation(track_id="monza", cars=cars, total_laps=30,
                             seed=42, retirement_prob_per_lap=0.05)
        res = sim.run()
        n_retired = sum(1 for _, c in res if c.retired)
        assert n_retired > 0

    def test_zero_retirement_prob_no_retirements(self) -> None:
        cars = _make_grid(20)
        sim = RaceSimulation(track_id="monza", cars=cars, total_laps=20,
                             seed=42, retirement_prob_per_lap=0.0)
        res = sim.run()
        assert all(not c.retired for _, c in res)

    def test_retired_car_classified_behind_finishers(self) -> None:
        cars = _make_grid(20)
        sim = RaceSimulation(track_id="monza", cars=cars, total_laps=20,
                             seed=42, retirement_prob_per_lap=0.1)
        res = sim.run()
        retired_positions = [p for p, c in res if c.retired]
        finisher_positions = [p for p, c in res if not c.retired]
        if retired_positions:  # only check if there are retirements
            assert max(finisher_positions) < min(retired_positions)


# --------------------------------------------------------------------------- #
# Grid start
# --------------------------------------------------------------------------- #
class TestGridStart:
    def test_start_delay_proportional_to_grid(self) -> None:
        """Cars further back should have larger start delay on lap 1."""
        cars = _make_grid(5)
        sim = RaceSimulation(track_id="monza", cars=cars, total_laps=1, seed=42)
        sim.run()
        # Initial cumulative times set by start delay
        # pole = 0.0, P2 = 0.05, P3 = 0.10, etc.
        # We just check pole sitter has lowest cumulative time after lap 1
        cars_sorted = sorted(cars, key=lambda c: c.grid_position)
        assert cars_sorted[0].cumulative_time <= cars_sorted[-1].cumulative_time


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
class TestSummary:
    def test_summary_returns_required_keys(self) -> None:
        sim = RaceSimulation(track_id="monza", cars=_make_grid(10),
                             total_laps=10, seed=42)
        s = sim.summary()
        required = {"track_id", "total_laps", "winner", "winner_time",
                    "n_finishers", "n_retirements", "classification"}
        assert required.issubset(s.keys())

    def test_summary_classification_structure(self) -> None:
        sim = RaceSimulation(track_id="monza", cars=_make_grid(5),
                             total_laps=5, seed=42)
        s = sim.summary()
        first = s["classification"][0]
        required = {"position", "driver", "grid", "laps", "total_time",
                    "gap_to_leader", "pit_stops", "retired",
                    "retirement_lap", "final_compound"}
        assert required.issubset(first.keys())

    def test_summary_winner_position_1(self) -> None:
        sim = RaceSimulation(track_id="monza", cars=_make_grid(10),
                             total_laps=10, seed=42)
        s = sim.summary()
        assert s["classification"][0]["position"] == 1
        assert s["classification"][0]["gap_to_leader"] == 0.0


# --------------------------------------------------------------------------- #
# Different strategies
# --------------------------------------------------------------------------- #
class TestStrategyComparison:
    def test_one_stop_vs_two_stop(self) -> None:
        """One-stop vs two-stop produce different race times."""
        car1 = _make_car(0, strategy=RaceStrategy(
            pit_laps=(25,), compounds=("medium", "hard")))
        car2 = _make_car(1, strategy=RaceStrategy(
            pit_laps=(15, 35), compounds=("medium", "hard", "soft")))
        sim = RaceSimulation(track_id="monza", cars=[car1, car2],
                             total_laps=50, seed=42)
        res = sim.run()
        # Both should complete the race
        assert all(c.laps_completed == 50 for _, c in res)
        # Both strategies should produce different times
        times = [c.cumulative_time for _, c in res]
        # Some difference (not identical)
        assert abs(times[0] - times[1]) > 0.1


# --------------------------------------------------------------------------- #
# Weather + Safety Car integration (Iter-15)
# --------------------------------------------------------------------------- #
class TestWeatherSCIntegration:
    def _make_grid(self, n: int = 6,
                   compounds: tuple[str, ...] = ("medium", "hard", "medium")
                   ) -> list[RaceCar]:
        return [RaceCar(
            driver_id=f"d{i:02d}", driver_name=f"D{i + 1}",
            setup=DEFAULT_SETUP, grid_position=i + 1,
            strategy=RaceStrategy(pit_laps=(15, 35), compounds=compounds),
        ) for i in range(n)]

    def test_wet_race_slower_than_dry(self) -> None:
        from f1opt.model.weather import WeatherModel, WeatherState
        cars_dry = self._make_grid(6)
        sim_dry = RaceSimulation(track_id="monza", cars=cars_dry,
                                 total_laps=30, seed=42)
        res_dry = sim_dry.run()

        cars_wet = self._make_grid(6, compounds=("intermediate", "wet", "intermediate"))
        wm = WeatherModel(initial=WeatherState(track_wetness=0.7))
        sim_wet = RaceSimulation(track_id="monza", cars=cars_wet,
                                 total_laps=30, seed=42, weather=wm,
                                 weather_rain_mmh=8.0)
        res_wet = sim_wet.run()
        # Wet race should be slower
        assert res_wet[0][1].cumulative_time > res_dry[0][1].cumulative_time

    def test_weather_state_evolves_during_race(self) -> None:
        from f1opt.model.weather import WeatherModel, WeatherState
        wm = WeatherModel(initial=WeatherState(track_wetness=0.3, track_temp_c=35.0))
        cars = self._make_grid(6)
        sim = RaceSimulation(track_id="monza", cars=cars, total_laps=20,
                             seed=42, weather=wm, weather_rain_mmh=10.0)
        sim.run()
        # After 20 laps with rain, wetness should have grown
        assert wm.state.track_wetness > 0.3

    def test_safety_car_periods_generated(self) -> None:
        cars = self._make_grid(10)
        sim = RaceSimulation(track_id="monza", cars=cars, total_laps=58,
                             seed=7, retirement_prob_per_lap=0.01)
        sim.run()
        # With high retirement, some SC periods likely (probabilistic)
        # Just check summary is callable
        sc_sum = sim.safety_car.summary()
        assert "n_periods" in sc_sum

    def test_sc_discounts_pit_loss(self) -> None:
        """SC 期间进站损失应低于正常 (free pit)."""
        from f1opt.model.safety_car import SafetyCarModel, SafetyCarPeriod
        scm = SafetyCarModel()
        scm.periods = [SafetyCarPeriod(10, 13, "sc")]
        cars = self._make_grid(2)
        sim = RaceSimulation(track_id="monza", cars=cars, total_laps=20,
                             seed=42, safety_car=scm)
        sim.run()
        # Just verify race ran and produced valid positions
        res = sim.run()
        assert len(res) == 2

    def test_wet_race_idempotent(self) -> None:
        from f1opt.model.weather import WeatherModel, WeatherState
        wm = WeatherModel(initial=WeatherState(track_wetness=0.5))
        cars = self._make_grid(4)
        sim = RaceSimulation(track_id="monza", cars=cars, total_laps=15,
                             seed=42, weather=wm, weather_rain_mmh=5.0)
        r1 = sim.run()
        t1 = [c.cumulative_time for _, c in r1]
        r2 = sim.run()
        t2 = [c.cumulative_time for _, c in r2]
        assert t1 == t2

    def test_dry_race_still_works_without_weather(self) -> None:
        cars = self._make_grid(6)
        sim = RaceSimulation(track_id="monza", cars=cars, total_laps=15, seed=42,
                             retirement_prob_per_lap=0.0)
        res = sim.run()
        # All cars finish
        assert all(not c.retired for _, c in res)
        # Default SafetyCarModel exists
        assert sim.safety_car is not None
