"""Tests for f1opt.model.brake_model (Iter-7)."""

from __future__ import annotations

import pytest

from f1opt.model.brake_model import (
    BRAKE_TRACK_LOAD,
    BrakeBias,
    BrakeModel,
    BrakeThermalModel,
    BrakeWearModel,
    get_brake_track_load,
)


# --------------------------------------------------------------------------- #
# Track load
# --------------------------------------------------------------------------- #
class TestTrackLoad:
    def test_known_track_load(self) -> None:
        assert get_brake_track_load("monaco") == 5
        assert get_brake_track_load("monza") == 2

    def test_unknown_track_default(self) -> None:
        assert 1 <= get_brake_track_load("nonexistent") <= 5

    def test_all_loads_in_range(self) -> None:
        for tid, load in BRAKE_TRACK_LOAD.items():
            assert 1 <= load <= 5, f"{tid} load {load} out of range"


# --------------------------------------------------------------------------- #
# BrakeBias
# --------------------------------------------------------------------------- #
class TestBrakeBias:
    def test_default_at_optimal(self) -> None:
        b = BrakeBias()
        assert b.front_fraction == pytest.approx(0.56)
        assert b.deviation_penalty_s() == pytest.approx(0.0, abs=1e-6)

    def test_front_fraction_clamped(self) -> None:
        b_low = BrakeBias(front_fraction=0.3)
        b_high = BrakeBias(front_fraction=0.9)
        assert b_low.front_fraction == 0.50
        assert b_high.front_fraction == 0.65

    def test_deviation_penalty_increases_with_offset(self) -> None:
        b_close = BrakeBias(front_fraction=0.57)
        b_far = BrakeBias(front_fraction=0.65)
        assert b_far.deviation_penalty_s() > b_close.deviation_penalty_s()

    def test_ers_migration_reduces_front(self) -> None:
        b = BrakeBias(front_fraction=0.58, ers_migration=0.03)
        assert b.effective_front_fraction < 0.58

    def test_rear_fraction_complement(self) -> None:
        b = BrakeBias(front_fraction=0.58)
        assert b.rear_fraction == pytest.approx(1.0 - 0.58)

    def test_lockup_risk_high_when_front_heavy(self) -> None:
        b_heavy = BrakeBias(front_fraction=0.65)
        b_light = BrakeBias(front_fraction=0.56)
        assert b_heavy.lockup_risk() > b_light.lockup_risk()

    def test_rear_instability_risk_high_when_rear_heavy(self) -> None:
        b_rear = BrakeBias(front_fraction=0.50)
        b_balanced = BrakeBias(front_fraction=0.56)
        assert b_rear.rear_instability_risk() > b_balanced.rear_instability_risk()


# --------------------------------------------------------------------------- #
# BrakeThermalModel
# --------------------------------------------------------------------------- #
class TestBrakeThermalModel:
    def test_temp_rises_from_cold_start(self) -> None:
        """Cold disc (200°C) should heat up after a lap."""
        tm = BrakeThermalModel()
        new_t = tm.temp_after_lap(200.0, track_load=3)
        assert new_t > 200.0

    def test_temp_stabilizes_over_many_laps(self) -> None:
        """Over many laps, temp converges to a stable equilibrium."""
        tm = BrakeThermalModel()
        t = 400.0
        for _ in range(100):
            t = tm.temp_after_lap(t, track_load=3)
        # Should be in physically reasonable F1 range (300-1000°C)
        assert 300.0 <= t <= 1000.0

    def test_bigger_duct_cools_more(self) -> None:
        """Larger cooling duct → lower equilibrium temperature."""
        tm_small = BrakeThermalModel(cooling_duct_area_mm2=60.0)
        tm_large = BrakeThermalModel(cooling_duct_area_mm2=200.0)
        t_small = 400.0
        t_large = 400.0
        for _ in range(50):
            t_small = tm_small.temp_after_lap(t_small, track_load=3)
            t_large = tm_large.temp_after_lap(t_large, track_load=3)
        assert t_large < t_small

    def test_heavier_track_load_heats_more(self) -> None:
        tm = BrakeThermalModel()
        t_light = tm.temp_after_lap(400.0, track_load=1)
        t_heavy = tm.temp_after_lap(400.0, track_load=5)
        assert t_heavy > t_light

    def test_thermal_penalty_below_window(self) -> None:
        tm = BrakeThermalModel()
        assert tm.thermal_penalty_s(300.0) > 0.0  # below 400

    def test_thermal_penalty_above_window(self) -> None:
        tm = BrakeThermalModel()
        assert tm.thermal_penalty_s(800.0) > 0.0  # above 700

    def test_thermal_penalty_zero_in_window(self) -> None:
        tm = BrakeThermalModel()
        assert tm.thermal_penalty_s(550.0) == 0.0
        assert tm.thermal_penalty_s(400.0) == 0.0
        assert tm.thermal_penalty_s(700.0) == 0.0

    def test_in_window_predicate(self) -> None:
        tm = BrakeThermalModel()
        assert tm.in_window(550.0)
        assert not tm.in_window(300.0)
        assert not tm.in_window(800.0)


# --------------------------------------------------------------------------- #
# BrakeWearModel
# --------------------------------------------------------------------------- #
class TestBrakeWearModel:
    def test_default_wear_positive(self) -> None:
        wm = BrakeWearModel()
        b = BrakeBias()
        w = wm.wear_per_lap(500.0, b, track_load=3, is_front=True)
        assert w > 0.0

    def test_high_temp_increases_wear(self) -> None:
        wm = BrakeWearModel()
        b = BrakeBias()
        w_cool = wm.wear_per_lap(450.0, b, track_load=3)
        w_hot = wm.wear_per_lap(900.0, b, track_load=3)
        assert w_hot > w_cool

    def test_high_track_load_increases_wear(self) -> None:
        wm = BrakeWearModel()
        b = BrakeBias()
        w_light = wm.wear_per_lap(500.0, b, track_load=1)
        w_heavy = wm.wear_per_lap(500.0, b, track_load=5)
        assert w_heavy > w_light

    def test_front_heavy_bias_wears_front_more(self) -> None:
        wm = BrakeWearModel()
        b_heavy = BrakeBias(front_fraction=0.65)
        b_light = BrakeBias(front_fraction=0.50)
        w_heavy = wm.wear_per_lap(500.0, b_heavy, track_load=3, is_front=True)
        w_light = wm.wear_per_lap(500.0, b_light, track_load=3, is_front=True)
        assert w_heavy > w_light

    def test_laps_remaining_positive_for_new_disc(self) -> None:
        wm = BrakeWearModel()
        b = BrakeBias()
        n = wm.laps_remaining(500.0, b, track_load=3)
        assert n > 0

    def test_laps_remaining_zero_when_at_min_thickness(self) -> None:
        wm = BrakeWearModel(current_thickness_mm=24.0)
        b = BrakeBias()
        n = wm.laps_remaining(500.0, b, track_load=3)
        assert n == 0


# --------------------------------------------------------------------------- #
# BrakeModel (综合)
# --------------------------------------------------------------------------- #
class TestBrakeModel:
    def test_simulate_lap_returns_required_keys(self) -> None:
        bm = BrakeModel(track_id="monaco")
        r = bm.simulate_lap(current_temp_c=500.0)
        required = {"track_id", "track_load", "front_fraction", "rear_fraction",
                    "temp_before_c", "temp_after_c", "in_window",
                    "bias_penalty_s", "thermal_penalty_s", "lockup_penalty_s",
                    "instability_penalty_s", "front_wear_mm", "rear_wear_mm",
                    "total_lap_penalty_s"}
        assert required.issubset(r.keys())

    def test_optimal_bias_zero_penalty(self) -> None:
        """At optimal bias (0.56) and in-window temp, bias penalty is 0."""
        bm = BrakeModel(track_id="monaco", front_fraction=0.56)
        r = bm.simulate_lap(current_temp_c=550.0)
        assert r["bias_penalty_s"] == pytest.approx(0.0, abs=1e-6)

    def test_suboptimal_bias_positive_penalty(self) -> None:
        bm = BrakeModel(track_id="monaco", front_fraction=0.65)
        r = bm.simulate_lap(current_temp_c=550.0)
        assert r["bias_penalty_s"] > 0.0

    def test_stint_returns_n_laps(self) -> None:
        bm = BrakeModel(track_id="monaco")
        laps = bm.simulate_stint(20)
        assert len(laps) == 20
        assert [lp["lap"] for lp in laps] == list(range(1, 21))

    def test_stint_thickness_decreases_monotonically(self) -> None:
        bm = BrakeModel(track_id="monaco")
        laps = bm.simulate_stint(15)
        for prev, cur in zip(laps, laps[1:], strict=False):
            assert cur["disc_thickness_front_mm"] < prev["disc_thickness_front_mm"]
            assert cur["disc_thickness_rear_mm"] < prev["disc_thickness_rear_mm"]

    def test_stint_temp_stays_under_1500(self) -> None:
        """Temperature must not diverge — physical bound at 1500°C."""
        bm = BrakeModel(track_id="monaco", cooling_duct_area_mm2=50.0)
        for lp in bm.simulate_stint(50):
            assert lp["temp_after_c"] < 1500.0


# --------------------------------------------------------------------------- #
# Optimization
# --------------------------------------------------------------------------- #
class TestBrakeOptimization:
    def test_optimize_bias_finds_optimal(self) -> None:
        """optimize_bias should return 0.56 (the optimal front_fraction)."""
        bm = BrakeModel(track_id="monaco", front_fraction=0.62)
        best_ff, best_pen = bm.optimize_bias(temp_c=550.0)
        assert best_ff == pytest.approx(0.56, abs=0.02)
        assert best_pen == pytest.approx(0.0, abs=0.01)

    def test_optimize_cooling_achieves_target_temp(self) -> None:
        """optimize_cooling should find a duct area that hits target temp."""
        bm = BrakeModel(track_id="monaco", cooling_duct_area_mm2=100.0)
        best_area, achieved_temp = bm.optimize_cooling(
            temp_c=500.0, target_temp_c=550.0
        )
        assert 50.0 <= best_area <= 200.0
        assert abs(achieved_temp - 550.0) < 30.0

    def test_optimize_does_not_mutate_state(self) -> None:
        bm = BrakeModel(track_id="monaco", front_fraction=0.58,
                        cooling_duct_area_mm2=110.0)
        original_ff = bm.front_fraction
        original_area = bm.cooling_duct_area_mm2
        bm.optimize_bias(temp_c=500.0)
        bm.optimize_cooling(temp_c=500.0)
        assert bm.front_fraction == original_ff
        assert bm.cooling_duct_area_mm2 == original_area


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #
class TestRobustness:
    def test_unknown_track_runs(self) -> None:
        bm = BrakeModel(track_id="nonexistent_track")
        r = bm.simulate_lap(current_temp_c=500.0)
        assert r["track_id"] == "nonexistent_track"

    def test_zero_lap_stint(self) -> None:
        bm = BrakeModel(track_id="monaco")
        assert bm.simulate_stint(0) == []

    def test_extreme_bias_clamped(self) -> None:
        bm = BrakeModel(track_id="monaco", front_fraction=0.99)
        r = bm.simulate_lap(current_temp_c=500.0)
        assert r["front_fraction"] <= 0.65

    def test_very_cold_start(self) -> None:
        bm = BrakeModel(track_id="monaco")
        r = bm.simulate_lap(current_temp_c=20.0)
        # Should still produce valid output and heat up
        assert r["temp_after_c"] > 20.0

    def test_very_hot_start_does_not_explode(self) -> None:
        bm = BrakeModel(track_id="monaco", cooling_duct_area_mm2=200.0)
        r = bm.simulate_lap(current_temp_c=900.0)
        # With big cooling, temp should DECREASE from very hot start
        assert r["temp_after_c"] < 900.0
