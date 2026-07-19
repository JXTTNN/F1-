"""Tests for power_unit_2026 (Iter-26)."""

from __future__ import annotations

import pytest

from f1opt.model.power_unit_2026 import (
    PowerUnit2026,
    PULapResult,
    simulate_pu_lap,
    total_power_for_mode,
)


class TestBasicStructure:
    def test_returns_result(self) -> None:
        r = simulate_pu_lap("monza", 0, "race")
        assert isinstance(r, PULapResult)
        assert r.deploy_mode == "race"

    def test_total_power_around_750kw(self) -> None:
        """Total power ~750 kW (400 ICE + 350 MGU-K)."""
        r = simulate_pu_lap("monza", 0, "quali")
        assert r.ice_power_kw == 400.0
        assert r.mguk_power_kw == 350.0
        assert r.total_power_kw == 750.0


class TestDeployModes:
    def test_quali_max_mguk(self) -> None:
        r = simulate_pu_lap("monza", 0, "quali")
        assert r.mguk_power_kw == 350.0
        assert r.electric_fraction == pytest.approx(350/750, rel=1e-3)

    def test_race_60_percent_mguk(self) -> None:
        r = simulate_pu_lap("monza", 0, "race")
        assert r.mguk_power_kw == 350 * 0.6
        assert r.electric_fraction < 0.5  # <50% since race mode 0.6

    def test_save_mode_lowest_mguk(self) -> None:
        r = simulate_pu_lap("monza", 0, "save")
        assert r.mguk_power_kw == 350 * 0.3

    def test_attack_mode_burst(self) -> None:
        r = simulate_pu_lap("monza", 0, "race", attack_mode=True)
        # 4s/90s × 100% + 86s/90s × 60% = 0.0444 + 0.5733 = 0.6178
        expected = 350 * (4/90 + (86/90) * 0.6)
        assert r.mguk_power_kw == pytest.approx(expected, rel=1e-3)
        assert r.attack_mode_activated is True

    def test_attack_mode_disabled_in_quali(self) -> None:
        """Quali already 100%, no attack mode benefit."""
        r = simulate_pu_lap("monza", 0, "quali", attack_mode=True)
        assert r.attack_mode_activated is False  # quali already at 100%
        assert r.mguk_power_kw == 350.0


class TestEnergyFlow:
    def test_deployed_energy_under_9_mj(self) -> None:
        """FIA 2026: max 9 MJ deployment per lap."""
        for mode in ("quali", "race", "save", "attack"):
            r = simulate_pu_lap("monza", 0, mode, attack_mode=(mode == "attack"))
            assert r.energy_deployed_mj <= 9.0 + 1e-6

    def test_recovered_energy_under_9_mj(self) -> None:
        r = simulate_pu_lap("monza", 0, "race", recovery_intensity=1.0)
        assert r.energy_recovered_mj <= 9.0 + 1e-6

    def test_quali_deploys_more_than_save(self) -> None:
        r_q = simulate_pu_lap("monza", 0, "quali")
        r_s = simulate_pu_lap("monza", 0, "save")
        assert r_q.energy_deployed_mj > r_s.energy_deployed_mj

    def test_soc_decreases_in_quali(self) -> None:
        """Quali deploys more than recovers → SoC drops."""
        pu = PowerUnit2026(track_id="monza", initial_soc=0.8)
        r = pu.simulate_lap(lap_idx=0, deploy_mode="quali",
                            recovery_intensity=0.5)
        assert r.battery_soc < 0.8


class TestFuelFlow:
    def test_fuel_under_30_kg_per_h_limit(self) -> None:
        """FIA 2026: 30 kg/h fuel flow, lap 90s = 0.75 kg/lap."""
        r = simulate_pu_lap("monza", 0, "race")
        # 0.75 kg/lap approximate
        assert 0.7 <= r.fuel_used_kg <= 0.8

    def test_ice_power_constant_400kw(self) -> None:
        """ICE always 400 kW regardless of mode."""
        for mode in ("quali", "race", "save", "attack"):
            r = simulate_pu_lap("monza", 0, mode, attack_mode=(mode == "attack"))
            assert r.ice_power_kw == 400.0


class TestWetConditions:
    def test_wet_reduces_mguk(self) -> None:
        r_dry = simulate_pu_lap("monza", 0, "race", track_wetness=0.0)
        r_wet = simulate_pu_lap("monza", 0, "race", track_wetness=0.5)
        assert r_wet.mguk_power_kw < r_dry.mguk_power_kw


class TestConvenienceFunction:
    def test_total_power_for_mode(self) -> None:
        assert total_power_for_mode("quali") == 750.0
        assert total_power_for_mode("race") == 400 + 350 * 0.6
        assert total_power_for_mode("save") == 400 + 350 * 0.3
        assert total_power_for_mode("attack") == 400 + 350 * 0.9

    def test_total_power_attack_burst(self) -> None:
        p_normal = total_power_for_mode("race", attack=False)
        p_attack = total_power_for_mode("race", attack=True)
        assert p_attack > p_normal
