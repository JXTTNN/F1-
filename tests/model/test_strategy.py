"""Tests for race strategy planning (RaceStrategyPlanner, StintSimulator, StrategyComparator).

Covers: 0/1/2-stop planning dicts, optimal strategy selection, tire wear / fuel
projections, track-dependent pit loss, per-lap stint simulation with
degradation, strategy ranking / recommendation (Chinese), edge cases (0-lap
race), and determinism.
"""

from __future__ import annotations

import pytest

from f1opt.model.strategy import (
    RaceStrategyPlanner,
    StintSimulator,
    StrategyComparator,
)


# --------------------------------------------------------------------------- #
# RaceStrategyPlanner
# --------------------------------------------------------------------------- #
class TestRaceStrategyPlanner:
    def test_plan_one_stop_returns_dict_with_stops_and_projections(self) -> None:
        planner = RaceStrategyPlanner("melbourne", 58, 110.0)
        plan = planner.plan_one_stop()
        assert isinstance(plan, dict)
        assert {"stops", "total_time_est", "tire_wear_projection", "fuel_projection"} <= set(
            plan.keys()
        )
        assert isinstance(plan["stops"], list)
        # A 1-stop on a 58-lap race yields exactly one pit entry.
        assert len(plan["stops"]) == 1
        assert plan["total_time_est"] > 0.0
        assert isinstance(plan["tire_wear_projection"], list)
        assert len(plan["tire_wear_projection"]) == 58
        assert isinstance(plan["fuel_projection"], list)

    def test_plan_two_stop_returns_two_stops(self) -> None:
        planner = RaceStrategyPlanner("melbourne", 60, 110.0)
        plan = planner.plan_two_stop()
        assert len(plan["stops"]) == 2
        # Each stop has a lap + compound in/out + Chinese reason.
        for stop in plan["stops"]:
            assert {"lap", "compound_in", "compound_out", "reason"} <= set(stop.keys())
            assert isinstance(stop["reason"], str) and stop["reason"]

    def test_plan_no_stop_returns_zero_stops(self) -> None:
        planner = RaceStrategyPlanner("melbourne", 58, 110.0)
        plan = planner.plan_no_stop()
        assert plan["stops"] == []
        assert plan["total_time_est"] > 0.0

    def test_optimal_strategy_returns_valid_strategy_type(self) -> None:
        planner = RaceStrategyPlanner("melbourne", 58, 110.0)
        opt = planner.optimal_strategy(["soft", "medium", "hard"])
        assert opt["strategy_type"] in {"0-stop", "1-stop", "2-stop"}
        assert isinstance(opt["plan"], dict)
        assert opt["total_time_est"] > 0.0
        # Chinese recommendation reason is a non-empty string.
        assert isinstance(opt["recommendation_reason"], str)
        assert len(opt["recommendation_reason"]) > 0

    def test_tire_wear_projection_soft_wears_faster_than_hard(self) -> None:
        planner = RaceStrategyPlanner("melbourne", 58, 110.0)
        soft = planner.tire_wear_projection("soft", 10)
        hard = planner.tire_wear_projection("hard", 10)
        assert len(soft) == 10 and len(hard) == 10
        # Cumulative wear is monotonically increasing per lap.
        assert soft[-1] > hard[-1]
        assert soft[0] > hard[0]

    def test_fuel_projection_decreases_over_laps(self) -> None:
        planner = RaceStrategyPlanner("melbourne", 58, 110.0)
        fuel = planner.fuel_projection()
        # Length is total_laps + 1 (boundary at start + each lap).
        assert len(fuel) == 59
        # Monotonically non-increasing, starts at the fuel load.
        assert fuel[0] == pytest.approx(110.0)
        for k in range(1, len(fuel)):
            assert fuel[k] <= fuel[k - 1] + 1e-9
        assert fuel[-1] < fuel[0]

    def test_pit_loss_time_monaco_above_twenty_seconds(self) -> None:
        planner = RaceStrategyPlanner("monaco", 78, 110.0)
        assert planner.pit_loss_time("monaco") > 20.0
        # Monza is the highest-speed track -> largest pit loss.
        assert planner.pit_loss_time("monza") >= 24.0

    def test_zero_lap_race_handled(self) -> None:
        """A 0-lap race must not crash and produces empty/zero plans."""
        planner = RaceStrategyPlanner("melbourne", 0, 110.0)
        no_plan = planner.plan_no_stop()
        assert no_plan["stops"] == []
        assert no_plan["total_time_est"] == pytest.approx(0.0)
        one_plan = planner.plan_one_stop()
        assert one_plan["stops"] == []
        two_plan = planner.plan_two_stop()
        assert two_plan["stops"] == []
        # fuel_projection has length 1 (just the start boundary).
        assert len(planner.fuel_projection()) == 1

    def test_planner_determinism_same_inputs_same_plan(self) -> None:
        p1 = RaceStrategyPlanner("melbourne", 58, 110.0)
        p2 = RaceStrategyPlanner("melbourne", 58, 110.0)
        a = p1.plan_one_stop()
        b = p2.plan_one_stop()
        assert a == b
        assert p1.plan_two_stop() == p2.plan_two_stop()
        assert p1.optimal_strategy(["soft", "medium", "hard"]) == p2.optimal_strategy(
            ["soft", "medium", "hard"]
        )


# --------------------------------------------------------------------------- #
# StintSimulator
# --------------------------------------------------------------------------- #
class TestStintSimulator:
    def test_simulate_returns_list_of_well_formed_dicts(self) -> None:
        stint = StintSimulator("medium", 10, "melbourne", 75.0)
        laps = stint.simulate()
        assert isinstance(laps, list)
        assert len(laps) == 10
        for k, rec in enumerate(laps):
            assert {"lap", "lap_time", "tire_wear_pct", "fuel_kg", "cumulative_time"} <= set(
                rec.keys()
            )
            assert rec["lap"] == k + 1
            assert rec["lap_time"] > 0.0
            assert rec["fuel_kg"] >= 0.0
            assert rec["cumulative_time"] > 0.0

    def test_total_time_is_positive(self) -> None:
        stint = StintSimulator("soft", 12, "melbourne", 75.0)
        assert stint.total_time() > 0.0
        # total_time equals the sum of the degradation curve.
        assert stint.total_time() == pytest.approx(sum(stint.degradation_curve()))

    def test_avg_lap_time_in_reasonable_range(self) -> None:
        stint = StintSimulator("medium", 15, "melbourne", 75.0)
        avg = stint.avg_lap_time()
        # Average lap is near the base, lifted slightly by wear penalty.
        assert 70.0 < avg < 90.0
        # avg == total / length for a non-empty stint.
        assert avg == pytest.approx(stint.total_time() / 15)

    def test_degradation_curve_later_laps_slower(self) -> None:
        stint = StintSimulator("soft", 12, "melbourne", 75.0)
        curve = stint.degradation_curve()
        assert len(curve) == 12
        # Lap times strictly increase as cumulative wear grows.
        for k in range(1, len(curve)):
            assert curve[k] > curve[k - 1]

    def test_simulate_lap_time_increases_with_lap_number(self) -> None:
        """Per-lap lap_time in simulate() increases due to tire degradation."""
        stint = StintSimulator("soft", 10, "melbourne", 75.0)
        laps = stint.simulate()
        times = [rec["lap_time"] for rec in laps]
        for k in range(1, len(times)):
            assert times[k] > times[k - 1]
        # Cumulative time is strictly increasing too.
        cum = [rec["cumulative_time"] for rec in laps]
        for k in range(1, len(cum)):
            assert cum[k] > cum[k - 1]

    def test_compound_wear_order_soft_gt_medium_gt_hard(self) -> None:
        """Soft wears fastest, then medium, then hard."""
        n = 8
        soft = StintSimulator("soft", n, "melbourne", 75.0)
        medium = StintSimulator("medium", n, "melbourne", 75.0)
        hard = StintSimulator("hard", n, "melbourne", 75.0)
        # Final-lap wear (and total time) follows the compound ordering.
        assert soft.simulate()[-1]["tire_wear_pct"] > medium.simulate()[-1][
            "tire_wear_pct"
        ]
        assert medium.simulate()[-1]["tire_wear_pct"] > hard.simulate()[-1][
            "tire_wear_pct"
        ]
        # Faster-wearing compound -> slower total stint time.
        assert soft.total_time() > medium.total_time() > hard.total_time()

    def test_empty_stint_avg_lap_time_zero(self) -> None:
        stint = StintSimulator("hard", 0, "melbourne", 75.0)
        assert stint.avg_lap_time() == 0.0
        assert stint.simulate() == []
        assert stint.degradation_curve() == []


# --------------------------------------------------------------------------- #
# StrategyComparator
# --------------------------------------------------------------------------- #
class TestStrategyComparator:
    def _strategies(self) -> list[dict]:
        return [
            {"name": "A", "total_time_est": 5400.0},
            {"name": "B", "total_time_est": 5350.0},
            {"name": "C", "total_time_est": 5420.0},
        ]

    def test_rank_returns_ascending_by_total_time(self) -> None:
        comp = StrategyComparator(self._strategies())
        ranked = comp.rank()
        assert len(ranked) == 3
        times = [t for _, t in ranked]
        assert times == sorted(times)
        # Best (lowest time) is strategy B at index 1.
        assert ranked[0][0] == 1
        assert ranked[0][1] == pytest.approx(5350.0)

    def test_best_returns_lowest_total_time_strategy(self) -> None:
        comp = StrategyComparator(self._strategies())
        best = comp.best()
        assert best["name"] == "B"
        assert best["total_time_est"] == pytest.approx(5350.0)

    def test_gap_to_best_zero_for_best_strategy(self) -> None:
        comp = StrategyComparator(self._strategies())
        # Index 1 is the best -> gap is 0.
        assert comp.gap_to_best(1) == pytest.approx(0.0)
        # Index 0 (5400) vs best (5350) -> gap 50s.
        assert comp.gap_to_best(0) == pytest.approx(50.0)
        # Gaps are always non-negative.
        for i in range(len(self._strategies())):
            assert comp.gap_to_best(i) >= 0.0

    def test_recommendation_is_non_empty_chinese(self) -> None:
        comp = StrategyComparator(self._strategies())
        rec = comp.recommendation()
        assert isinstance(rec, str)
        assert len(rec) > 0
        # Contains Chinese ranking marker.
        assert "排名" in rec

    def test_recommendation_from_real_plans(self) -> None:
        """Comparator over real planner outputs produces a Chinese ranking."""
        planner = RaceStrategyPlanner("melbourne", 58, 110.0)
        plans = [
            planner.plan_no_stop(),
            planner.plan_one_stop(),
            planner.plan_two_stop(),
        ]
        comp = StrategyComparator(plans)
        ranked = comp.rank()
        assert len(ranked) == 3
        # All three plan total times are positive and finite.
        for _, t in ranked:
            assert t > 0.0
        rec = comp.recommendation()
        assert "排名" in rec and "策略" in rec
