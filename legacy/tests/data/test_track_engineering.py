"""Tests for f1opt.data.track_engineering (Iter-19)."""

from __future__ import annotations

import pytest

from f1opt.data.track_engineering import (
    TrackEngineering,
    air_density_factor,
    all_track_engineering,
    downforce_effective,
    get_track_engineering,
)
from f1opt.data.tracks import ALL_TRACKS


# --------------------------------------------------------------------------- #
# Registry completeness
# --------------------------------------------------------------------------- #
class TestRegistryCompleteness:
    def test_all_24_tracks_have_engineering_data(self) -> None:
        eng = all_track_engineering()
        assert len(eng) == 24

    def test_every_track_id_in_calendar_has_entry(self) -> None:
        for t in ALL_TRACKS:
            eng = get_track_engineering(t.track_id)
            assert eng.track_id == t.track_id

    def test_unknown_track_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown track_id"):
            get_track_engineering("does_not_exist")

    def test_registry_sorted_by_round_implicitly(self) -> None:
        """Engineering dict insertion order matches calendar order."""
        eng_ids = [e.track_id for e in all_track_engineering()]
        cal_ids = [t.track_id for t in ALL_TRACKS]
        assert eng_ids == cal_ids


# --------------------------------------------------------------------------- #
# Field validation (FIA / Pirelli level bounds)
# --------------------------------------------------------------------------- #
class TestFieldBounds:
    @pytest.mark.parametrize("eng", all_track_engineering())
    def test_tire_wear_factor_range(self, eng: TrackEngineering) -> None:
        assert 0.5 <= eng.tire_wear_factor <= 1.5

    @pytest.mark.parametrize("eng", all_track_engineering())
    def test_fuel_consumption_range(self, eng: TrackEngineering) -> None:
        assert 1.1 <= eng.fuel_consumption_kg_per_lap <= 2.2

    @pytest.mark.parametrize("eng", all_track_engineering())
    def test_top_speed_range(self, eng: TrackEngineering) -> None:
        assert 280.0 <= eng.top_speed_kmh <= 365.0

    @pytest.mark.parametrize("eng", all_track_engineering())
    def test_downforce_level_range(self, eng: TrackEngineering) -> None:
        assert 0.05 <= eng.downforce_level <= 1.05

    @pytest.mark.parametrize("eng", all_track_engineering())
    def test_drs_zones_range(self, eng: TrackEngineering) -> None:
        assert 1 <= eng.drs_zones <= 4

    @pytest.mark.parametrize("eng", all_track_engineering())
    def test_drs_total_length_positive(self, eng: TrackEngineering) -> None:
        assert eng.drs_total_length_m > 0

    @pytest.mark.parametrize("eng", all_track_engineering())
    def test_pit_loss_range(self, eng: TrackEngineering) -> None:
        assert 20.0 <= eng.pit_loss_s <= 24.0

    @pytest.mark.parametrize("eng", all_track_engineering())
    def test_pit_lane_speed_valid(self, eng: TrackEngineering) -> None:
        assert eng.pit_lane_speed_kmh in (60.0, 80.0)

    @pytest.mark.parametrize("eng", all_track_engineering())
    def test_overtaking_difficulty_range(self, eng: TrackEngineering) -> None:
        assert 0.0 <= eng.overtaking_difficulty <= 1.0

    @pytest.mark.parametrize("eng", all_track_engineering())
    def test_brake_wear_level_range(self, eng: TrackEngineering) -> None:
        assert 0.4 <= eng.brake_wear_level <= 1.1

    @pytest.mark.parametrize("eng", all_track_engineering())
    def test_ers_deployment_range(self, eng: TrackEngineering) -> None:
        assert 4.5 <= eng.ers_deployment_mj_per_lap <= 9.0

    @pytest.mark.parametrize("eng", all_track_engineering())
    def test_sector_count_is_3(self, eng: TrackEngineering) -> None:
        assert eng.sector_count == 3


# --------------------------------------------------------------------------- #
# Authoritative spot-checks (known F1 engineering values)
# --------------------------------------------------------------------------- #
class TestAuthoritativeValues:
    def test_monaco_has_highest_downforce(self) -> None:
        eng = get_track_engineering("monaco")
        assert eng.downforce_level == 1.0
        assert eng.overtaking_difficulty >= 0.95
        assert eng.top_speed_kmh < 300.0
        assert eng.pit_lane_speed_kmh == 60.0  # Monaco special limit

    def test_monza_has_lowest_downforce(self) -> None:
        eng = get_track_engineering("monza")
        assert eng.downforce_level <= 0.15
        assert eng.top_speed_kmh >= 355.0
        assert eng.overtaking_difficulty <= 0.25

    def test_mexico_city_extreme_altitude(self) -> None:
        eng = get_track_engineering("mexico_city")
        assert eng.altitude_m >= 2200.0
        # Despite thin air, top speed very high (low drag)
        assert eng.top_speed_kmh >= 355.0

    def test_spa_longest_track_highest_fuel(self) -> None:
        eng = get_track_engineering("spa")
        # Spa is longest (~7km) so should have highest fuel burn
        all_eng = all_track_engineering()
        max_fuel = max(e.fuel_consumption_kg_per_lap for e in all_eng)
        assert eng.fuel_consumption_kg_per_lap == max_fuel

    def test_silverstone_high_tire_wear(self) -> None:
        eng = get_track_engineering("silverstone")
        assert eng.tire_wear_factor >= 1.3  # high-energy circuit

    def test_monaco_low_tire_wear(self) -> None:
        eng = get_track_engineering("monaco")
        assert eng.tire_wear_factor <= 0.6  # low-energy street circuit

    def test_jeddah_3_drs_zones(self) -> None:
        eng = get_track_engineering("jeddah")
        assert eng.drs_zones == 3

    def test_baku_long_drs_total_length(self) -> None:
        eng = get_track_engineering("baku")
        # Baku has 2.2km main straight — longest DRS
        assert eng.drs_total_length_m >= 2000.0


# --------------------------------------------------------------------------- #
# Air density physics
# --------------------------------------------------------------------------- #
class TestAirDensity:
    def test_sea_level_density_is_unity(self) -> None:
        assert air_density_factor(0.0) == pytest.approx(1.0, abs=1e-6)

    def test_mexico_city_density_drop(self) -> None:
        # Mexico City altitude 2286m → ~0.77 air density
        rho = air_density_factor(2286.0)
        assert 0.74 <= rho <= 0.80

    def test_spielberg_moderate_altitude(self) -> None:
        # Spielberg ~670m → ~0.92
        rho = air_density_factor(670.0)
        assert 0.90 <= rho <= 0.94

    def test_density_monotonically_decreasing(self) -> None:
        prev = 1.0
        for h in [0, 500, 1000, 1500, 2000, 2286, 3000]:
            rho = air_density_factor(h)
            assert rho <= prev
            prev = rho


# --------------------------------------------------------------------------- #
# Effective downforce (combines nominal + air density)
# --------------------------------------------------------------------------- #
class TestEffectiveDownforce:
    def test_mexico_city_effective_much_lower_than_nominal(self) -> None:
        eng = get_track_engineering("mexico_city")
        nominal = eng.downforce_level
        eff = downforce_effective(eng)
        assert eff < nominal * 0.80  # >20% reduction expected

    def test_monaco_effective_close_to_nominal(self) -> None:
        eng = get_track_engineering("monaco")
        nominal = eng.downforce_level
        eff = downforce_effective(eng)
        assert eff >= nominal * 0.97  # Monaco ~20m altitude

    def test_monza_still_lowest_effective_downforce(self) -> None:
        monza_eff = downforce_effective(get_track_engineering("monza"))
        for t in ALL_TRACKS:
            if t.track_id == "monza":
                continue
            other_eff = downforce_effective(get_track_engineering(t.track_id))
            assert monza_eff < other_eff, (
                f"Monza effective downforce {monza_eff:.3f} should be lowest "
                f"but {t.track_id} is {other_eff:.3f}"
            )

    def test_monaco_highest_effective_downforce(self) -> None:
        monaco_eff = downforce_effective(get_track_engineering("monaco"))
        for t in ALL_TRACKS:
            if t.track_id == "monaco":
                continue
            other_eff = downforce_effective(get_track_engineering(t.track_id))
            assert monaco_eff > other_eff


# --------------------------------------------------------------------------- #
# Frozen / hashable
# --------------------------------------------------------------------------- #
class TestFrozen:
    def test_engineering_is_frozen(self) -> None:
        eng = get_track_engineering("monza")
        with pytest.raises((AttributeError, TypeError)):
            eng.downforce_level = 0.5  # type: ignore[misc]

    def test_engineering_is_hashable(self) -> None:
        eng = get_track_engineering("monza")
        assert hash(eng) == hash(eng)
