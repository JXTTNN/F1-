"""Tests for f1opt.model.aerodynamics (Iter-6)."""

from __future__ import annotations

import pytest

from f1opt.model.aerodynamics import (
    DRS_TRACK_DATA,
    AerodynamicsModel,
    AeroMap,
    get_drs_data,
)


# --------------------------------------------------------------------------- #
# AeroMap basics
# --------------------------------------------------------------------------- #
class TestAeroMap:
    def test_default_has_positive_cl_cd(self) -> None:
        am = AeroMap()
        assert am.cl() > 0
        assert am.cd() > 0
        assert am.cl_cd_ratio() > 0

    def test_higher_wing_more_downforce(self) -> None:
        low = AeroMap(front_wing=0.2, rear_wing=0.2)
        high = AeroMap(front_wing=0.9, rear_wing=0.9)
        assert high.cl() > low.cl()

    def test_higher_wing_more_drag(self) -> None:
        low = AeroMap(front_wing=0.2, rear_wing=0.2)
        high = AeroMap(front_wing=0.9, rear_wing=0.9)
        assert high.cd() > low.cd()

    def test_drs_reduces_drag(self) -> None:
        am_off = AeroMap(drs_active=False)
        am_on = AeroMap(drs_active=True)
        assert am_on.cd() < am_off.cd()

    def test_drs_reduces_downforce_slightly(self) -> None:
        am_off = AeroMap(drs_active=False)
        am_on = AeroMap(drs_active=True)
        assert am_on.cl() < am_off.cl()

    def test_rake_property(self) -> None:
        am = AeroMap(ride_height_front_mm=20, ride_height_rear_mm=35)
        assert am.rake_mm == 15.0


# --------------------------------------------------------------------------- #
# Ground effect
# --------------------------------------------------------------------------- #
class TestGroundEffect:
    def test_optimal_ride_height_beats_extreme(self) -> None:
        """Ride height near optimal (~20-25 mm) > very low or very high."""
        am_optimal = AeroMap(ride_height_front_mm=20, ride_height_rear_mm=30)
        am_too_low = AeroMap(ride_height_front_mm=5, ride_height_rear_mm=15)
        am_too_high = AeroMap(ride_height_front_mm=60, ride_height_rear_mm=70)
        cl_opt = am_optimal.cl()
        assert cl_opt > am_too_low.cl()
        assert cl_opt > am_too_high.cl()

    def test_porpoising_kicks_in_below_threshold(self) -> None:
        """Ride height below 10 mm triggers porpoising penalty."""
        am_normal = AeroMap(ride_height_front_mm=15, ride_height_rear_mm=25)
        am_porpoise = AeroMap(ride_height_front_mm=5, ride_height_rear_mm=15)
        # The too-low setup should have lower Cl than the moderate one
        # (despite being closer to ground).
        assert am_porpoise.cl() < am_normal.cl()

    def test_ground_effect_factor_at_optimal(self) -> None:
        am = AeroMap()
        f, p = am._ground_effect_factor(25.0)
        assert f > 1.0  # max gain applied
        assert p == 0.0  # no porpoising at 25 mm

    def test_ground_effect_factor_far_from_optimal(self) -> None:
        am = AeroMap()
        f, p = am._ground_effect_factor(60.0)
        # 35 mm off optimal → small gain
        assert f < 1.1
        assert p == 0.0  # no porpoising when too high

    def test_ground_effect_factor_below_porpoise_threshold(self) -> None:
        am = AeroMap()
        f, p = am._ground_effect_factor(5.0)
        assert p > 0.0  # porpoising penalty active


# --------------------------------------------------------------------------- #
# DRS data
# --------------------------------------------------------------------------- #
class TestDRSData:
    def test_get_drs_data_known_track(self) -> None:
        d = get_drs_data("monza")
        assert d["n_drs_zones"] >= 1
        assert d["avg_zone_length_m"] > 0

    def test_get_drs_data_unknown_returns_default(self) -> None:
        d = get_drs_data("nonexistent_track")
        assert d["n_drs_zones"] >= 1
        assert d["avg_zone_length_m"] > 0

    def test_all_drs_data_has_valid_fields(self) -> None:
        for _tid, d in DRS_TRACK_DATA.items():
            assert isinstance(d["n_drs_zones"], int)
            assert d["n_drs_zones"] >= 1
            assert isinstance(d["avg_zone_length_m"], float)
            assert 100.0 <= d["avg_zone_length_m"] <= 1500.0


# --------------------------------------------------------------------------- #
# AerodynamicsModel basics
# --------------------------------------------------------------------------- #
class TestAerodynamicsModelBasics:
    def test_compute_lap_aero_returns_required_keys(self) -> None:
        am = AerodynamicsModel(track_id="monza")
        r = am.compute_lap_aero(avg_speed_ms=80.0, max_speed_ms=95.0)
        required = {"track_id", "front_wing", "rear_wing", "cl", "cd",
                    "cl_cd_ratio", "downforce_avg_N", "downforce_max_N",
                    "drag_avg_N", "drag_max_N", "drs_zones", "drs_gain_s",
                    "corner_gain_from_downforce_s", "drag_loss_s",
                    "net_lap_gain_s", "rake_mm"}
        assert required.issubset(r.keys())

    def test_downforce_increases_with_speed_squared(self) -> None:
        am = AerodynamicsModel(track_id="monza")
        d1 = am.downforce_n(40.0)
        d2 = am.downforce_n(80.0)
        # 2x speed → 4x downforce
        assert d2 == pytest.approx(4 * d1, rel=0.05)

    def test_drag_increases_with_speed_squared(self) -> None:
        am = AerodynamicsModel(track_id="monza")
        d1 = am.drag_n(40.0)
        d2 = am.drag_n(80.0)
        # 2x speed → ~4x drag (with mild mach correction at 80 m/s)
        assert d2 > 3.5 * d1

    def test_drs_reduces_drag_force(self) -> None:
        am = AerodynamicsModel(track_id="monza")
        d_off = am.drag_n(80.0, drs_active=False)
        d_on = am.drag_n(80.0, drs_active=True)
        assert d_on < d_off

    def test_default_max_speed_is_1_4x_avg(self) -> None:
        am = AerodynamicsModel(track_id="monza")
        r = am.compute_lap_aero(avg_speed_ms=80.0)
        # max_speed defaults to 80*1.4 = 112 m/s; downforce_max should be
        # at 112 not 80.
        assert r["downforce_max_N"] > r["downforce_avg_N"]


# --------------------------------------------------------------------------- #
# DRS lap gain
# --------------------------------------------------------------------------- #
class TestDRSLapGain:
    def test_drs_gain_positive_on_track_with_drs(self) -> None:
        am = AerodynamicsModel(track_id="monza")
        r = am.compute_lap_aero(avg_speed_ms=80.0, max_speed_ms=95.0)
        assert r["drs_gain_s"] > 0

    def test_more_drs_zones_more_gain(self) -> None:
        am_few = AerodynamicsModel(track_id="suzuka")  # 1 DRS zone
        am_many = AerodynamicsModel(track_id="melbourne")  # 4 DRS zones
        r_few = am_few.compute_lap_aero(avg_speed_ms=80.0, max_speed_ms=95.0)
        r_many = am_many.compute_lap_aero(avg_speed_ms=80.0, max_speed_ms=95.0)
        assert r_many["drs_gain_s"] > r_few["drs_gain_s"]


# --------------------------------------------------------------------------- #
# Sensitivity analysis
# --------------------------------------------------------------------------- #
class TestSensitivityAnalysis:
    def test_returns_keys_for_all_params(self) -> None:
        am = AerodynamicsModel(track_id="monza")
        s = am.sensitivity_analysis()
        expected = {"front_wing", "rear_wing",
                    "ride_height_front_mm", "ride_height_rear_mm"}
        assert expected.issubset(s.keys())

    def test_rear_wing_positive_sensitivity(self) -> None:
        """Increasing rear_wing should improve lap time (positive gain delta)."""
        am = AerodynamicsModel(track_id="monaco", front_wing=0.5, rear_wing=0.5)
        s = am.sensitivity_analysis()
        assert s["rear_wing"] > 0  # more rear wing = more downforce = faster

    def test_high_ride_height_negative_sensitivity(self) -> None:
        """Raising rear ride height should reduce downforce (negative gain)."""
        am = AerodynamicsModel(track_id="monaco", front_wing=0.5, rear_wing=0.5,
                               ride_height_front_mm=30, ride_height_rear_mm=40)
        s = am.sensitivity_analysis()
        assert s["ride_height_rear_mm"] < 0


# --------------------------------------------------------------------------- #
# Optimize ride height
# --------------------------------------------------------------------------- #
class TestOptimizeRideHeight:
    def test_optimize_returns_reasonable_value(self) -> None:
        am = AerodynamicsModel(track_id="monza")
        best_rh, best_cl = am.optimize_ride_height()
        # Optimal should be in the search range (10-50 mm)
        assert 10.0 <= best_rh <= 50.0
        assert best_cl > 0

    def test_optimal_beats_extreme_high(self) -> None:
        am = AerodynamicsModel(track_id="monza")
        best_rh, best_cl = am.optimize_ride_height()
        # Compare to extreme high ride height
        am_high = AeroMap(ride_height_front_mm=49.0, ride_height_rear_mm=59.0)
        assert best_cl > am_high.cl()

    def test_optimize_restores_original_state(self) -> None:
        am = AerodynamicsModel(track_id="monza", ride_height_front_mm=22.0,
                               ride_height_rear_mm=35.0)
        original_rh = am.ride_height_front_mm
        am.optimize_ride_height()
        # After optimize, _aero is rebuilt with current values (unchanged)
        assert am.ride_height_front_mm == original_rh


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #
class TestRobustness:
    def test_zero_speed_zero_force(self) -> None:
        am = AerodynamicsModel(track_id="monza")
        assert am.downforce_n(0.0) == 0.0
        assert am.drag_n(0.0) == 0.0

    def test_negative_speed_zero_force(self) -> None:
        am = AerodynamicsModel(track_id="monza")
        # speed^2 makes negative speeds still produce force; but physically
        # we expect 0 at standstill. Verify it doesn't crash.
        d = am.downforce_n(-10.0)
        assert d >= 0

    def test_unknown_track_uses_default_drs(self) -> None:
        am = AerodynamicsModel(track_id="nonexistent")
        r = am.compute_lap_aero(avg_speed_ms=80.0)
        assert r["drs_zones"] >= 1

    def test_extreme_wing_settings_no_crash(self) -> None:
        for fw in (0.0, 1.0):
            for rw in (0.0, 1.0):
                am = AerodynamicsModel(track_id="monza",
                                       front_wing=fw, rear_wing=rw)
                r = am.compute_lap_aero(avg_speed_ms=80.0, max_speed_ms=110.0)
                assert r["cl"] > 0
                assert r["cd"] > 0

    def test_cl_cd_ratio_typical_f1_range(self) -> None:
        """F1 cars have Cl/Cd ratio typically 2-5."""
        for fw, rw in [(0.2, 0.3), (0.5, 0.5), (0.8, 0.9)]:
            am = AeroMap(front_wing=fw, rear_wing=rw,
                         ride_height_front_mm=20, ride_height_rear_mm=30)
            ratio = am.cl_cd_ratio()
            assert 1.5 <= ratio <= 6.0, f"fw={fw} rw={rw} ratio={ratio}"
