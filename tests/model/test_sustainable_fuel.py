"""Tests for sustainable_fuel (Iter-27)."""

from __future__ import annotations

import pytest

from f1opt.model.sustainable_fuel import (
    FuelLapResult,
    fuel_consumption_lap,
)


class TestBasic:
    def test_returns_result(self) -> None:
        r = fuel_consumption_lap("monza", "race")
        assert isinstance(r, FuelLapResult)

    def test_fuel_under_30_kg_per_h_limit(self) -> None:
        """FIA 2026: max 30 kg/h fuel flow."""
        for mode in ("quali", "race", "save", "attack"):
            r = fuel_consumption_lap("monza", mode)
            assert r.fuel_flow_kg_per_h <= 30.0 + 1e-6

    def test_quali_more_fuel_than_save(self) -> None:
        r_q = fuel_consumption_lap("monza", "quali")
        r_s = fuel_consumption_lap("monza", "save")
        assert r_q.fuel_used_kg > r_s.fuel_used_kg


class TestEnergyDensity:
    def test_energy_density_0_97(self) -> None:
        """Sustainable fuel 3% lower energy density."""
        r = fuel_consumption_lap("monza", "race")
        assert r.energy_density_factor == pytest.approx(0.97)


class TestTemperatureDerating:
    def test_normal_temp_no_derating(self) -> None:
        r = fuel_consumption_lap("monza", "race", track_temp_c=35.0)
        assert r.derating_reason == "none"
        assert r.effective_power_factor < 1.0  # still reduced by alt+energy

    def test_high_temp_derating(self) -> None:
        r = fuel_consumption_lap("singapore", "race", track_temp_c=55.0,
                                  altitude_m=10.0)
        assert r.derating_reason in ("high_temp", "both")
        # Significant power drop
        assert r.effective_power_factor < 0.95

    def test_extreme_temp_heavy_derating(self) -> None:
        r = fuel_consumption_lap("singapore", "race", track_temp_c=70.0,
                                  altitude_m=10.0)
        assert r.effective_power_factor < 0.85

    def test_bigger_radiator_mitigates(self) -> None:
        r_small = fuel_consumption_lap("singapore", "race", track_temp_c=60.0,
                                        cooling_duct_size=0.8)
        r_big = fuel_consumption_lap("singapore", "race", track_temp_c=60.0,
                                      cooling_duct_size=1.3)
        assert r_big.effective_power_factor >= r_small.effective_power_factor


class TestAltitude:
    def test_mexico_city_altitude_derating(self) -> None:
        """Mexico City 2286m altitude → significant power drop."""
        r = fuel_consumption_lap("mexico_city", "race",
                                  altitude_m=2286.0, track_temp_c=30.0)
        assert r.derating_reason == "altitude"
        # Air density ~0.77, energy 0.97 → 0.75 effective
        assert r.effective_power_factor < 0.80

    def test_sea_level_no_altitude_derating(self) -> None:
        r = fuel_consumption_lap("monaco", "race", altitude_m=20.0,
                                  track_temp_c=30.0)
        assert r.derating_reason == "none"
