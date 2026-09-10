"""Tests for driver tire management style (Iter-22)."""

from __future__ import annotations

import pytest

from f1opt.data.setup_schema import DEFAULT_SETUP
from f1opt.model.tire_stint import TireStintPhysics


class TestTireManagementEffect:
    """Verify driver_tire_management scales wear rate."""

    def _stint(self, mgmt: float) -> TireStintPhysics:
        return TireStintPhysics(
            compound="soft",
            track_id="suzuka",
            base_lap_time=95.0,
            stint_length=20,
            driver_tire_management=mgmt,
        )

    def test_default_is_neutral(self) -> None:
        """Default 0.5 should give factor 1.025 (between 0.75 and 1.30)."""
        sim = self._stint(0.5)
        # factor = 1.30 - 0.5 * 0.55 = 1.025
        # soft base 5.2, suzuka abrasiveness 1.30
        expected = 5.2 * 1.30 * 1.025
        assert sim._wear_rate_per_lap(0) == pytest.approx(expected, rel=1e-3)

    def test_gentle_driver_lower_wear(self) -> None:
        """management=1.0 should reduce wear vs default 0.5."""
        gentle = self._stint(1.0)
        default = self._stint(0.5)
        assert gentle._wear_rate_per_lap(0) < default._wear_rate_per_lap(0)

    def test_aggressive_driver_higher_wear(self) -> None:
        """management=0.0 should increase wear vs default 0.5."""
        aggr = self._stint(0.0)
        default = self._stint(0.5)
        assert aggr._wear_rate_per_lap(0) > default._wear_rate_per_lap(0)

    def test_clamped_to_unit_interval(self) -> None:
        """Out-of-range values clamp to [0, 1]."""
        sim_high = self._stint(2.0)
        sim_low = self._stint(-1.0)
        assert sim_high.driver_tire_management == 1.0
        assert sim_low.driver_tire_management == 0.0

    def test_gentle_driver_extends_stint_length(self) -> None:
        """Gentle driver (1.0) should reach cliff later than aggressive (0.0)."""
        gentle_laps = self._stint(1.0).simulate()
        aggr_laps = self._stint(0.0).simulate()
        # Find first lap in cliff phase
        gentle_cliff = next(
            (i for i, lp in enumerate(gentle_laps) if lp["phase"] == "cliff"),
            len(gentle_laps),
        )
        aggr_cliff = next(
            (i for i, lp in enumerate(aggr_laps) if lp["phase"] == "cliff"),
            len(aggr_laps),
        )
        assert gentle_cliff > aggr_cliff

    def test_gentle_driver_lower_final_wear(self) -> None:
        """After 15 laps, gentle driver should have less total wear."""
        gentle_laps = self._stint(1.0).simulate()[:15]
        aggr_laps = self._stint(0.0).simulate()[:15]
        assert gentle_laps[-1]["wear_pct"] < aggr_laps[-1]["wear_pct"]

    def test_management_factor_range(self) -> None:
        """Verify the factor formula: 1.30 - mgmt * 0.55, range [0.75, 1.30]."""
        for mgmt in (0.0, 0.25, 0.5, 0.75, 1.0):
            sim = self._stint(mgmt)
            # base soft 5.2, suzuka 1.30
            expected_factor = 1.30 - mgmt * 0.55
            expected_wear = 5.2 * 1.30 * expected_factor
            assert sim._wear_rate_per_lap(0) == pytest.approx(expected_wear, rel=1e-4)


class TestLapSimulatorPassThrough:
    """LapTimeSimulator must pass driver_tire_management to TireStintPhysics."""

    def test_lap_simulator_creates_tire_with_management(self) -> None:
        from f1opt.model.lap_simulator import LapTimeSimulator

        sim = LapTimeSimulator(
            setup=DEFAULT_SETUP,
            track_id="monza",
            compound="medium",
            driver_tire_management=0.9,
        )
        assert sim._tire.driver_tire_management == 0.9

    def test_lap_simulator_default_management_neutral(self) -> None:
        from f1opt.model.lap_simulator import LapTimeSimulator

        sim = LapTimeSimulator(
            setup=DEFAULT_SETUP,
            track_id="monza",
        )
        assert sim._tire.driver_tire_management == 0.5


class TestRaceCarPassThrough:
    """RaceCar.driver_tire_management flows to LapTimeSimulator."""

    def test_race_car_field_exists(self) -> None:
        from f1opt.model.race_simulator import RaceCar, RaceStrategy

        car = RaceCar(
            driver_id="x",
            driver_name="X",
            setup=DEFAULT_SETUP,
            grid_position=1,
            strategy=RaceStrategy(pit_laps=(), compounds=("medium",)),
            driver_tire_management=0.8,
        )
        assert car.driver_tire_management == 0.8

    def test_race_car_default_management(self) -> None:
        from f1opt.model.race_simulator import RaceCar, RaceStrategy

        car = RaceCar(
            driver_id="x",
            driver_name="X",
            setup=DEFAULT_SETUP,
            grid_position=1,
            strategy=RaceStrategy(pit_laps=(), compounds=("medium",)),
        )
        assert car.driver_tire_management == 0.5
