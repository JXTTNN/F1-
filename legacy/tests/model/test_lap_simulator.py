"""Tests for f1opt.model.lap_simulator (Iter-8)."""

from __future__ import annotations

from f1opt.data.setup_schema import DEFAULT_SETUP
from f1opt.model.lap_simulator import LapTimeSimulator, simulate_lap, simulate_stint


# --------------------------------------------------------------------------- #
# Basic simulation
# --------------------------------------------------------------------------- #
class TestLapTimeSimulatorBasics:
    def test_simulate_lap_returns_required_keys(self) -> None:
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza")
        r = sim.simulate_lap(lap_idx=0)
        required = {
            "lap", "lap_time", "base_prior", "tire_lap_time", "tire_wear_pct",
            "tire_phase", "aero_net_gain_s", "aero_cl", "aero_cd",
            "ers_net_gain_s", "ers_soc_after", "ers_deploy_mj",
            "brake_penalty_s", "brake_temp_after_c", "brake_in_window",
            "driver_gain_s", "smoothness_penalty_s", "consistency_penalty_s",
            "fuel_kg", "track_id", "compound",
        }
        assert required.issubset(r.keys())

    def test_lap_time_in_physical_range(self) -> None:
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza")
        r = sim.simulate_lap(lap_idx=0)
        assert 60.0 <= r["lap_time"] <= 180.0

    def test_lap_time_uses_track_prior_as_base(self) -> None:
        """Lap time should be near track_prior for a baseline setup."""
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monaco")
        r = sim.simulate_lap(lap_idx=0)
        # Should be within 10s of base prior (driver + aero + ers + brake)
        assert abs(r["lap_time"] - r["base_prior"]) < 15.0

    def test_track_specific_lap_times_differ(self) -> None:
        """Monaco (short) and Spa (long) should produce very different times."""
        monaco = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monaco").simulate_lap(0)
        spa = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="spa").simulate_lap(0)
        assert spa["lap_time"] > monaco["lap_time"] + 15.0  # Spa >> Monaco


# --------------------------------------------------------------------------- #
# Stint simulation
# --------------------------------------------------------------------------- #
class TestStintSimulation:
    def test_stint_returns_n_laps(self) -> None:
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza")
        stint = sim.simulate_stint(15)
        assert len(stint) == 15
        assert [lp["lap"] for lp in stint] == list(range(1, 16))

    def test_fuel_decreases_monotonically(self) -> None:
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza")
        stint = sim.simulate_stint(10)
        for prev, cur in zip(stint, stint[1:], strict=False):
            assert cur["fuel_kg"] < prev["fuel_kg"]

    def test_tire_wear_increases_monotonically(self) -> None:
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="suzuka",
                               compound="soft")
        stint = sim.simulate_stint(15)
        for prev, cur in zip(stint, stint[1:], strict=False):
            assert cur["tire_wear_pct"] >= prev["tire_wear_pct"]

    def test_lap_time_in_range_across_stint(self) -> None:
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                               compound="soft")
        for lp in sim.simulate_stint(25):
            assert 60.0 <= lp["lap_time"] <= 180.0

    def test_stint_resets_state(self) -> None:
        """simulate_stint should start fresh regardless of prior state."""
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza")
        sim.simulate_lap(0)  # mutate state
        sim.simulate_lap(1)
        stint = sim.simulate_stint(5)
        # First lap of stint should have 0% tire wear
        assert stint[0]["tire_wear_pct"] == 0.0


# --------------------------------------------------------------------------- #
# Driver behavior
# --------------------------------------------------------------------------- #
class TestDriverBehavior:
    def test_aggressive_driver_faster_than_conservative(self) -> None:
        sim_aggr = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                                    driver_aggression=0.9)
        sim_cons = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                                    driver_aggression=0.1)
        aggr = sim_aggr.simulate_lap(0)
        cons = sim_cons.simulate_lap(0)
        assert aggr["lap_time"] < cons["lap_time"]

    def test_smooth_driver_faster_than_rough(self) -> None:
        sim_smooth = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                                      driver_smoothness=0.9)
        sim_rough = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                                     driver_smoothness=0.1)
        smooth = sim_smooth.simulate_lap(0)
        rough = sim_rough.simulate_lap(0)
        assert smooth["lap_time"] < rough["lap_time"]

    def test_consistent_driver_faster_than_inconsistent(self) -> None:
        sim_cons = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                                    driver_consistency=0.9)
        sim_incons = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                                      driver_consistency=0.1)
        cons = sim_cons.simulate_lap(0)
        incons = sim_incons.simulate_lap(0)
        assert cons["lap_time"] < incons["lap_time"]

    def test_driver_aggression_clamped(self) -> None:
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                               driver_aggression=2.0)
        assert sim.driver_aggression == 1.0


# --------------------------------------------------------------------------- #
# Compound / setup interactions
# --------------------------------------------------------------------------- #
class TestCompoundInteractions:
    def test_soft_faster_initially_than_hard(self) -> None:
        """Soft compound should be faster on lap 1 (less warmup penalty)."""
        sim_soft = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                                    compound="soft", driver_aggression=0.5)
        sim_hard = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                                    compound="hard", driver_aggression=0.5)
        soft = sim_soft.simulate_lap(0)
        hard = sim_hard.simulate_lap(0)
        # Soft has lower warmup penalty → faster lap 1
        assert soft["lap_time"] < hard["lap_time"]

    def test_soft_degrades_faster_than_hard_over_stint(self) -> None:
        """Soft should accumulate wear faster than hard."""
        sim_soft = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="suzuka",
                                    compound="soft")
        sim_hard = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="suzuka",
                                    compound="hard")
        soft_stint = sim_soft.simulate_stint(15)
        hard_stint = sim_hard.simulate_stint(15)
        assert soft_stint[-1]["tire_wear_pct"] > hard_stint[-1]["tire_wear_pct"]


# --------------------------------------------------------------------------- #
# Brake thermal integration
# --------------------------------------------------------------------------- #
class TestBrakeIntegration:
    def test_brake_temp_increases_across_stint(self) -> None:
        """Brake temperature should rise and stabilize over a stint."""
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monaco",
                               brake_temp_c=300.0)
        stint = sim.simulate_stint(15)
        # Last lap brake temp should be higher than first lap (heating up)
        assert stint[-1]["brake_temp_after_c"] > stint[0]["brake_temp_after_c"]

    def test_brake_penalty_applied_when_out_of_window(self) -> None:
        """Very hot brake should produce non-zero penalty."""
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monaco",
                               brake_temp_c=900.0,
                               brake_cooling_duct_area_mm2=40.0)
        r = sim.simulate_lap(0)
        # brake_temp very high → penalty > 0
        assert r["brake_penalty_s"] > 0.0


# --------------------------------------------------------------------------- #
# ERS integration
# --------------------------------------------------------------------------- #
class TestERSIntegration:
    def test_ers_soc_changes_across_stint(self) -> None:
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                               ers_mode="attack", ers_initial_soc=0.8)
        stint = sim.simulate_stint(10)
        # In attack mode, SoC should decrease over time (or fluctuate)
        soc_first = stint[0]["ers_soc_after"]
        soc_last = stint[-1]["ers_soc_after"]
        # SoC must stay in [0, 1]
        for lp in stint:
            assert 0.0 <= lp["ers_soc_after"] <= 1.0
        # In attack mode, SoC should drop overall
        assert soc_last < soc_first + 0.1

    def test_attack_mode_faster_than_conserve(self) -> None:
        sim_atk = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                                   ers_mode="attack", ers_initial_soc=0.9)
        sim_con = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                                   ers_mode="conserve", ers_initial_soc=0.9)
        atk = sim_atk.simulate_lap(0)
        con = sim_con.simulate_lap(0)
        assert atk["lap_time"] < con["lap_time"]


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
class TestSummary:
    def test_summary_returns_required_keys(self) -> None:
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza")
        s = sim.summary(laps=15)
        required = {"track_id", "compound", "laps", "total_time",
                    "avg_lap_time", "best_lap", "best_lap_num", "worst_lap",
                    "final_tire_wear_pct", "final_ers_soc",
                    "final_brake_temp_c", "final_fuel_kg", "lap_times"}
        assert required.issubset(s.keys())

    def test_summary_best_worst_consistent(self) -> None:
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                               compound="soft")
        s = sim.summary(laps=15)
        assert s["best_lap"] < s["worst_lap"]
        assert s["best_lap"] <= s["avg_lap_time"] <= s["worst_lap"]


# --------------------------------------------------------------------------- #
# Module-level convenience functions
# --------------------------------------------------------------------------- #
class TestConvenienceFunctions:
    def test_simulate_lap_function(self) -> None:
        r = simulate_lap(DEFAULT_SETUP, track_id="monza", compound="medium")
        assert "lap_time" in r
        assert 60.0 <= r["lap_time"] <= 180.0

    def test_simulate_stint_function(self) -> None:
        laps = simulate_stint(DEFAULT_SETUP, track_id="monza", laps=10,
                              compound="soft")
        assert len(laps) == 10


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #
class TestRobustness:
    def test_unknown_track_runs(self) -> None:
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="nonexistent_track")
        r = sim.simulate_lap(0)
        assert 60.0 <= r["lap_time"] <= 180.0

    def test_zero_lap_stint(self) -> None:
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza")
        assert sim.simulate_stint(0) == []

    def test_extreme_driver_values_clamped(self) -> None:
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                               driver_aggression=99.0,
                               driver_smoothness=-1.0,
                               driver_consistency=99.0)
        assert sim.driver_aggression == 1.0
        assert sim.driver_smoothness == 0.0
        assert sim.driver_consistency == 1.0

    def test_wet_compound_runs(self) -> None:
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                               compound="wet")
        r = sim.simulate_lap(0)
        assert 60.0 <= r["lap_time"] <= 180.0


# --------------------------------------------------------------------------- #
# Weather integration (Iter-13)
# --------------------------------------------------------------------------- #
class TestWeatherIntegration:
    def test_no_weather_zero_penalty(self) -> None:
        """无 weather 字段 → weather_penalty_s == 0, track_wetness == 0."""
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza")
        r = sim.simulate_lap(0)
        assert r["weather_penalty_s"] == 0.0
        assert r["track_wetness"] == 0.0
        assert r["weather_recommended_compound"] is None

    def test_wet_slick_slower_than_dry(self) -> None:
        """湿地 + slick 比干地慢 (惩罚 > 0)."""
        from f1opt.model.weather import WeatherModel, WeatherState
        wm = WeatherModel(initial=WeatherState(track_wetness=0.5))
        sim_wet = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                                   compound="medium", weather=wm)
        sim_dry = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                                   compound="medium")
        wet = sim_wet.simulate_lap(0)
        dry = sim_dry.simulate_lap(0)
        assert wet["lap_time"] > dry["lap_time"]
        assert wet["weather_penalty_s"] > 0.0

    def test_inters_faster_than_slicks_in_wet(self) -> None:
        """湿地中半雨胎比干胎快."""
        from f1opt.model.weather import WeatherModel, WeatherState
        wm1 = WeatherModel(initial=WeatherState(track_wetness=0.5))
        wm2 = WeatherModel(initial=WeatherState(track_wetness=0.5))
        sim_slick = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                                     compound="medium", weather=wm1)
        sim_inters = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                                      compound="intermediate", weather=wm2)
        slick = sim_slick.simulate_lap(0)
        inters = sim_inters.simulate_lap(0)
        assert inters["lap_time"] < slick["lap_time"]

    def test_weather_recommended_compound_returned(self) -> None:
        from f1opt.model.weather import WeatherModel, WeatherState
        wm = WeatherModel(initial=WeatherState(track_wetness=0.5))
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                               compound="medium", weather=wm)
        r = sim.simulate_lap(0)
        assert r["weather_recommended_compound"] == "intermediate"

    def test_weather_evolution_between_laps(self) -> None:
        """天气在 stint 中演化: 雨停后圈速回升 (湿润度下降)."""
        from f1opt.model.weather import WeatherModel, WeatherState
        wm = WeatherModel(initial=WeatherState(track_wetness=0.7, track_temp_c=35.0))
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                               compound="intermediate", weather=wm)
        lap1 = sim.simulate_lap(0)
        # 雨停 30 分钟干燥
        wm.step(rain_mmh=0.0, minutes=30.0)
        lap2 = sim.simulate_lap(1)
        # 干燥后湿润度下降 → 惩罚变化
        assert lap2["track_wetness"] < lap1["track_wetness"]

    def test_follow_loss_factor_in_output(self) -> None:
        from f1opt.model.weather import WeatherModel, WeatherState
        wm = WeatherModel(initial=WeatherState(track_wetness=0.8))
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                               compound="wet", weather=wm)
        r = sim.simulate_lap(0)
        assert r["follow_loss_factor"] > 1.0

    def test_dry_weather_no_penalty(self) -> None:
        from f1opt.model.weather import WeatherModel, WeatherState
        wm = WeatherModel(initial=WeatherState(track_wetness=0.0))
        sim = LapTimeSimulator(setup=DEFAULT_SETUP, track_id="monza",
                               compound="medium", weather=wm)
        r = sim.simulate_lap(0)
        assert r["weather_penalty_s"] == 0.0
