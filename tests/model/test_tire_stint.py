"""Tests for f1opt.model.tire_stint (Iter-4)."""

from __future__ import annotations

from f1opt.model.tire_stint import (
    COMPOUND_STINT_PARAMS,
    TireStintPhysics,
    compound_work_window,
    track_abrasiveness,
)


# --------------------------------------------------------------------------- #
# compound_work_window / track_abrasiveness
# --------------------------------------------------------------------------- #
def test_work_window_soft_vs_hard() -> None:
    """Soft window cooler than hard; both are tuples of 2 floats."""
    soft_lo, soft_hi = compound_work_window("soft")
    hard_lo, hard_hi = compound_work_window("hard")
    assert soft_lo < soft_hi
    assert hard_lo < hard_hi
    assert soft_hi < hard_hi  # soft peaks cooler than hard


def test_work_window_unknown_compound_falls_back() -> None:
    lo, hi = compound_work_window("nonexistent")
    assert lo < hi


def test_track_abrasiveness_range() -> None:
    """abrasiveness falls in [0.6, 1.4]; monaco < 1.0 < suzuka."""
    assert 0.6 <= track_abrasiveness("monaco") < 1.0
    assert track_abrasiveness("suzuka") > 1.0
    assert 0.6 <= track_abrasiveness("unknown_track") <= 1.4


def test_compound_params_present_for_all_5_compounds() -> None:
    for name in ("soft", "medium", "hard", "intermediate", "wet"):
        assert name in COMPOUND_STINT_PARAMS
        p = COMPOUND_STINT_PARAMS[name]
        assert 0.5 <= p.warmup_laps <= 3.5
        assert p.steady_rate_s > 0
        assert 50.0 <= p.cliff_threshold_pct <= 90.0
        assert p.cliff_rate_s > 0


# --------------------------------------------------------------------------- #
# TireStintPhysics basic simulation
# --------------------------------------------------------------------------- #
class TestTireStintPhysicsBasics:
    def _sim(self, **kw) -> TireStintPhysics:
        defaults = dict(
            compound="medium", track_id="melbourne", base_lap_time=90.0,
            stint_length=20, initial_fuel_kg=110.0, track_temp_c=35.0,
        )
        defaults.update(kw)
        return TireStintPhysics(**defaults)

    def test_simulate_returns_n_laps(self) -> None:
        laps = self._sim(stint_length=15).simulate()
        assert len(laps) == 15
        assert [lp["lap"] for lp in laps] == list(range(1, 16))

    def test_lap_record_keys(self) -> None:
        laps = self._sim(stint_length=5).simulate()
        required = {"lap", "phase", "lap_time", "wear_pct", "front_wear_pct",
                    "rear_wear_pct", "tyre_temp_c", "fuel_kg", "cumulative_time"}
        assert required.issubset(laps[0].keys())

    def test_cumulative_time_monotone(self) -> None:
        laps = self._sim(stint_length=20).simulate()
        for prev, cur in zip(laps, laps[1:], strict=False):
            assert cur["cumulative_time"] > prev["cumulative_time"]

    def test_fuel_decreases_monotonically(self) -> None:
        laps = self._sim(stint_length=10).simulate()
        for prev, cur in zip(laps, laps[1:], strict=False):
            assert cur["fuel_kg"] < prev["fuel_kg"]
        assert laps[-1]["fuel_kg"] >= 0.0

    def test_wear_pct_monotone_and_bounded(self) -> None:
        laps = self._sim(stint_length=25).simulate()
        for prev, cur in zip(laps, laps[1:], strict=False):
            assert cur["wear_pct"] >= prev["wear_pct"]
        assert laps[-1]["wear_pct"] <= 100.0


# --------------------------------------------------------------------------- #
# Phase structure
# --------------------------------------------------------------------------- #
class TestPhaseStructure:
    def test_warmup_first_lap_is_slower_than_steady(self) -> None:
        """Lap 1 has warmup penalty → slower than a steady-phase lap."""
        sim = TireStintPhysics(compound="soft", track_id="suzuka",
                               base_lap_time=91.5, stint_length=12,
                               initial_fuel_kg=110.0, track_temp_c=35.0)
        laps = sim.simulate()
        # Lap 1 is warmup; lap 4 should be steady (warmer fuel advantage but
        # lower warmup penalty). Net: lap 4 < lap 1 (warmup penalty dominates).
        assert laps[3]["lap_time"] < laps[0]["lap_time"]

    def test_long_soft_stint_reaches_cliff(self) -> None:
        sim = TireStintPhysics(compound="soft", track_id="suzuka",
                               base_lap_time=91.5, stint_length=20,
                               track_temp_c=42.0)
        laps = sim.simulate()
        phases = [lp["phase"] for lp in laps]
        assert "cliff" in phases or "dead" in phases

    def test_hard_lasts_longer_than_soft_before_cliff(self) -> None:
        """Hard compound reaches cliff later than soft at same track."""
        sim_soft = TireStintPhysics(compound="soft", track_id="suzuka",
                                    base_lap_time=91.5, stint_length=25,
                                    track_temp_c=42.0)
        sim_hard = TireStintPhysics(compound="hard", track_id="suzuka",
                                    base_lap_time=92.0, stint_length=25,
                                    track_temp_c=42.0)
        soft_laps = sim_soft.simulate()
        hard_laps = sim_hard.simulate()

        def first_cliff(laps):
            for _i, lp in enumerate(laps):
                if lp["phase"] in ("cliff", "dead"):
                    return lp["lap"]
            return len(laps) + 1

        assert first_cliff(hard_laps) > first_cliff(soft_laps)

    def test_dead_phase_is_slowest(self) -> None:
        """If dead phase occurs, it must be slower than steady phase."""
        sim = TireStintPhysics(compound="soft", track_id="suzuka",
                               base_lap_time=91.5, stint_length=25,
                               track_temp_c=42.0)
        laps = sim.simulate()
        steady_times = [lp["lap_time"] for lp in laps if lp["phase"] == "steady"]
        dead_times = [lp["lap_time"] for lp in laps if lp["phase"] == "dead"]
        if dead_times and steady_times:
            assert min(dead_times) > max(steady_times)

    def test_warmup_phase_present_at_start(self) -> None:
        sim = TireStintPhysics(compound="hard", track_id="monaco",
                               base_lap_time=75.0, stint_length=10)
        laps = sim.simulate()
        assert laps[0]["phase"] == "warmup"


# --------------------------------------------------------------------------- #
# Track abrasiveness interaction
# --------------------------------------------------------------------------- #
class TestTrackAbrasivenessInteraction:
    def test_monaco_wears_slower_than_suzuka(self) -> None:
        sim_monaco = TireStintPhysics(compound="medium", track_id="monaco",
                                      base_lap_time=75.0, stint_length=15)
        sim_suzuka = TireStintPhysics(compound="medium", track_id="suzuka",
                                      base_lap_time=91.5, stint_length=15)
        m = sim_monaco.simulate()
        s = sim_suzuka.simulate()
        assert m[-1]["wear_pct"] < s[-1]["wear_pct"]


# --------------------------------------------------------------------------- #
# Balance asymmetry
# --------------------------------------------------------------------------- #
class TestBalanceAsymmetry:
    def test_understeer_wears_front_more(self) -> None:
        sim = TireStintPhysics(compound="medium", track_id="melbourne",
                               base_lap_time=90.0, stint_length=15,
                               balance_tendency="understeer")
        laps = sim.simulate()
        assert laps[-1]["front_wear_pct"] > laps[-1]["rear_wear_pct"]

    def test_oversteer_wears_rear_more(self) -> None:
        sim = TireStintPhysics(compound="medium", track_id="melbourne",
                               base_lap_time=90.0, stint_length=15,
                               balance_tendency="oversteer")
        laps = sim.simulate()
        assert laps[-1]["rear_wear_pct"] > laps[-1]["front_wear_pct"]

    def test_neutral_balance_keeps_front_close_to_rear(self) -> None:
        sim = TireStintPhysics(compound="medium", track_id="melbourne",
                               base_lap_time=90.0, stint_length=15,
                               balance_tendency="neutral")
        laps = sim.simulate()
        # Front bias is ~0.51 for medium → front slightly higher.
        assert abs(laps[-1]["front_wear_pct"] - laps[-1]["rear_wear_pct"]) < 10.0

    def test_invalid_balance_falls_back_to_neutral(self) -> None:
        sim = TireStintPhysics(compound="medium", track_id="melbourne",
                               base_lap_time=90.0, stint_length=5,
                               balance_tendency="weird")
        assert sim.balance_tendency == "neutral"


# --------------------------------------------------------------------------- #
# Pit window & summary
# --------------------------------------------------------------------------- #
class TestPitWindowAndSummary:
    def test_pit_window_is_valid_range(self) -> None:
        sim = TireStintPhysics(compound="soft", track_id="suzuka",
                               base_lap_time=91.5, stint_length=18)
        earliest, latest = sim.optimal_pit_window()
        assert 1 <= earliest < latest <= 18

    def test_summary_keys_present(self) -> None:
        sim = TireStintPhysics(compound="medium", track_id="melbourne",
                               base_lap_time=90.0, stint_length=12)
        s = sim.summary()
        required = {"compound", "track_id", "stint_length", "total_time",
                    "avg_lap_time", "best_lap", "worst_lap", "best_lap_num",
                    "pit_window", "ends_in_cliff"}
        assert required.issubset(s.keys())

    def test_summary_best_worst_consistent(self) -> None:
        sim = TireStintPhysics(compound="soft", track_id="suzuka",
                               base_lap_time=91.5, stint_length=15)
        s = sim.summary()
        assert s["best_lap"] < s["worst_lap"]
        assert s["best_lap"] <= s["avg_lap_time"] <= s["worst_lap"]

    def test_summary_empty_for_zero_length(self) -> None:
        sim = TireStintPhysics(compound="medium", track_id="melbourne",
                               base_lap_time=90.0, stint_length=0)
        assert sim.summary() == {}


# --------------------------------------------------------------------------- #
# Thermal
# --------------------------------------------------------------------------- #
class TestThermalPenalty:
    def test_hot_track_makes_stint_slower(self) -> None:
        """Extreme heat pushes tyres above window → slower than in-window stint."""
        # Soft window: 83-107°C. Heat buildup ~0.6°C/lap → at lap 10 tyre temp
        # ≈ track_temp + 6. Use track_temp=70 (in window) vs track_temp=140
        # (way above window at lap 10: 146°C, +39°C above high bound).
        sim_in_window = TireStintPhysics(compound="soft", track_id="suzuka",
                                         base_lap_time=91.5, stint_length=10,
                                         track_temp_c=70.0)
        sim_overheating = TireStintPhysics(compound="soft", track_id="suzuka",
                                           base_lap_time=91.5, stint_length=10,
                                           track_temp_c=140.0)
        in_w = sim_in_window.simulate()
        over = sim_overheating.simulate()
        # Lap 10: overheating should be slower than in-window.
        assert over[-1]["lap_time"] > in_w[-1]["lap_time"]

    def test_cold_track_also_penalises(self) -> None:
        """Track below window → under-temperature penalty too."""
        sim_cold = TireStintPhysics(compound="soft", track_id="suzuka",
                                    base_lap_time=91.5, stint_length=5,
                                    track_temp_c=20.0)  # well below 83
        sim_warm = TireStintPhysics(compound="soft", track_id="suzuka",
                                    base_lap_time=91.5, stint_length=5,
                                    track_temp_c=95.0)  # at optimal
        cold = sim_cold.simulate()
        warm = sim_warm.simulate()
        # Lap 1 in cold should be slower (warmup + under-temp).
        assert cold[0]["lap_time"] > warm[0]["lap_time"]


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #
class TestRobustness:
    def test_unknown_compound_uses_default(self) -> None:
        sim = TireStintPhysics(compound="nonexistent", track_id="melbourne",
                               base_lap_time=90.0, stint_length=5)
        laps = sim.simulate()
        assert len(laps) == 5

    def test_unknown_track_uses_default_abrasiveness(self) -> None:
        sim = TireStintPhysics(compound="soft", track_id="nonexistent_track",
                               base_lap_time=90.0, stint_length=5)
        laps = sim.simulate()
        # Should still produce valid output (no crash, valid time range).
        for lp in laps:
            assert 60.0 <= lp["lap_time"] <= 180.0

    def test_zero_fuel_no_crash(self) -> None:
        sim = TireStintPhysics(compound="medium", track_id="melbourne",
                               base_lap_time=90.0, stint_length=5,
                               initial_fuel_kg=0.0)
        laps = sim.simulate()
        assert laps[-1]["fuel_kg"] >= 0.0

    def test_lap_time_in_physically_reasonable_range(self) -> None:
        """F1 lap times: 60-180 s regardless of conditions."""
        for compound in ("soft", "medium", "hard", "intermediate", "wet"):
            sim = TireStintPhysics(compound=compound, track_id="suzuka",
                                   base_lap_time=91.5, stint_length=30,
                                   track_temp_c=45.0)
            for lp in sim.simulate():
                assert 60.0 <= lp["lap_time"] <= 180.0
