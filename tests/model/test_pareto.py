"""Tests for multi-objective Pareto optimization (ParetoFront, MultiObjectiveOptimizer)."""

from __future__ import annotations

import numpy as np
import pytest

from f1opt.data.setup_schema import DEFAULT_SETUP, CarSetup
from f1opt.model.pareto import (
    MultiObjectiveOptimizer,
    ParetoFront,
    compound_objective,
    constraint_penalty,
    lap_time_objective,
    tire_wear_objective,
)


# --------------------------------------------------------------------------- #
# ParetoFront
# --------------------------------------------------------------------------- #
class TestParetoFront:
    def test_compute_front_returns_nondominated_indices(self) -> None:
        pf = ParetoFront(["a", "b"])
        pf.add_sample([1.0, 5.0])
        pf.add_sample([2.0, 2.0])
        pf.add_sample([5.0, 1.0])
        front = pf.compute_front()
        assert isinstance(front, list)
        # All three are mutually non-dominated.
        assert set(front) == {0, 1, 2}

    def test_2d_minimization_excludes_dominated_points(self) -> None:
        pf = ParetoFront(["a", "b"])
        pf.add_sample([1.0, 5.0])  # 0
        pf.add_sample([2.0, 2.0])  # 1
        pf.add_sample([3.0, 3.0])  # 2 dominated by 1 (2<=3 and 2<3)
        pf.add_sample([5.0, 1.0])  # 3
        front = pf.compute_front()
        assert 2 not in front
        assert set(front) == {0, 1, 3}

    def test_crowding_distance_edge_points_inf(self) -> None:
        pf = ParetoFront(["a", "b"])
        pf.add_sample([1.0, 5.0])  # 0
        pf.add_sample([2.0, 3.0])  # 1
        pf.add_sample([3.0, 2.0])  # 2
        pf.add_sample([5.0, 1.0])  # 3
        cd = pf.crowding_distance([0, 1, 2, 3])
        assert set(cd.keys()) == {0, 1, 2, 3}
        assert all(isinstance(v, float) for v in cd.values())
        # Extremes in any objective get inf.
        assert cd[0] == float("inf")
        assert cd[3] == float("inf")
        # Interior points get finite distances.
        assert cd[1] != float("inf")
        assert cd[2] != float("inf")

    def test_knee_point_returns_valid_index(self) -> None:
        pf = ParetoFront(["a", "b"])
        pf.add_sample([1.0, 5.0])
        pf.add_sample([2.0, 3.0])
        pf.add_sample([3.0, 2.0])
        pf.add_sample([5.0, 1.0])
        front = pf.compute_front()
        knee = pf.knee_point(front)
        assert knee in front

    def test_hypervolume_2d_rectangle_returns_area(self) -> None:
        pf = ParetoFront(["a", "b"])
        pf.add_sample([1.0, 2.0])
        hv = pf.hypervolume([5.0, 5.0])
        # Rectangle [1,5] x [2,5] = 4 * 3 = 12.
        assert hv == pytest.approx(12.0)

    def test_hypervolume_3d_cube_returns_volume(self) -> None:
        pf = ParetoFront(["a", "b", "c"])
        pf.add_sample([1.0, 1.0, 1.0])
        hv = pf.hypervolume([5.0, 5.0, 5.0])
        # Cube 4 * 4 * 4 = 64.
        assert hv == pytest.approx(64.0)

    def test_summary_has_required_keys(self) -> None:
        pf = ParetoFront(["lap_time", "tire_wear"])
        pf.add_sample([90.0, 0.7])
        pf.add_sample([91.0, 0.6])
        s = pf.summary()
        assert {
            "front_size",
            "total_samples",
            "objectives",
            "knee_index",
            "spread",
        } <= set(s.keys())
        assert s["total_samples"] == 2
        assert s["objectives"] == ["lap_time", "tire_wear"]
        assert s["front_size"] >= 1
        assert "lap_time" in s["spread"] and "tire_wear" in s["spread"]

    def test_compute_front_with_metadata_structure(self) -> None:
        pf = ParetoFront(["a", "b"])
        pf.add_sample([1.0, 5.0], metadata={"name": "x"})
        pf.add_sample([3.0, 3.0])
        pf.add_sample([5.0, 1.0])
        pf.add_sample([3.0, 4.0])  # dominated by [3,3]
        meta = pf.compute_front_with_metadata()
        # Only front members are returned; the dominated sample is excluded.
        assert {m["index"] for m in meta} == {0, 1, 2}
        for m in meta:
            assert set(m.keys()) == {"index", "values", "metadata", "dominated_by_count"}
            assert m["dominated_by_count"] == 0

    def test_maximize_direction(self) -> None:
        # Both objectives maximized: [1,5] dominates [1,3] (equal on a, strict on b).
        pf = ParetoFront(["a", "b"], maximize=[True, True])
        pf.add_sample([1.0, 5.0])  # 0
        pf.add_sample([1.0, 3.0])  # 1 dominated by 0
        pf.add_sample([5.0, 1.0])  # 2
        front = pf.compute_front()
        assert 1 not in front
        assert set(front) == {0, 2}

    def test_empty_pareto_front(self) -> None:
        pf = ParetoFront(["a", "b"])
        assert pf.compute_front() == []
        assert pf.crowding_distance([]) == {}

    def test_single_sample_front(self) -> None:
        pf = ParetoFront(["a", "b"])
        pf.add_sample([1.0, 2.0])
        assert pf.compute_front() == [0]
        cd = pf.crowding_distance([0])
        assert cd[0] == float("inf")


# --------------------------------------------------------------------------- #
# MultiObjectiveOptimizer
# --------------------------------------------------------------------------- #
def _bounds(n: int = 21) -> np.ndarray:
    return np.array([[0.0, 1.0]] * n)


class TestMultiObjectiveOptimizer:
    def test_search_returns_required_keys(self) -> None:
        opt = MultiObjectiveOptimizer(
            _bounds(), ["lap_time", "tire_wear"], n_iterations=3, seed=42
        )
        res = opt.search("melbourne")
        assert {
            "pareto_front",
            "best_lap_time_setup",
            "best_tire_wear_setup",
            "knee_setup",
            "history",
            "iterations",
        } <= set(res.keys())
        assert res["iterations"] == 3
        assert isinstance(res["history"], list)
        assert len(res["history"]) == 3
        assert isinstance(res["pareto_front"], ParetoFront)

    def test_best_lap_time_setup_is_carsetup(self) -> None:
        opt = MultiObjectiveOptimizer(
            _bounds(), ["lap_time", "tire_wear"], n_iterations=2, seed=42
        )
        res = opt.search("melbourne")
        assert isinstance(res["best_lap_time_setup"], CarSetup)

    def test_knee_setup_is_carsetup(self) -> None:
        opt = MultiObjectiveOptimizer(
            _bounds(), ["lap_time", "tire_wear"], n_iterations=2, seed=42
        )
        res = opt.search("melbourne")
        assert isinstance(res["knee_setup"], CarSetup)
        assert isinstance(res["best_tire_wear_setup"], CarSetup)

    def test_determinism_same_seed_same_best_lap_time(self) -> None:
        opt1 = MultiObjectiveOptimizer(
            _bounds(), ["lap_time", "tire_wear"], n_iterations=3, seed=42
        )
        opt2 = MultiObjectiveOptimizer(
            _bounds(), ["lap_time", "tire_wear"], n_iterations=3, seed=42
        )
        r1 = opt1.search("melbourne")
        r2 = opt2.search("melbourne")
        assert r1["best_lap_time_setup"].to_vector() == r2["best_lap_time_setup"].to_vector()
        v1 = opt1.evaluate(np.array(r1["best_lap_time_setup"].to_vector()), "melbourne")
        v2 = opt2.evaluate(np.array(r2["best_lap_time_setup"].to_vector()), "melbourne")
        assert v1[0] == pytest.approx(v2[0])

    def test_evaluate_returns_list_of_two_floats(self) -> None:
        opt = MultiObjectiveOptimizer(_bounds(), ["lap_time", "tire_wear"], seed=1)
        vals = opt.evaluate(np.full(21, 0.5), "melbourne")
        assert isinstance(vals, list)
        assert len(vals) == 2
        assert all(isinstance(v, float) for v in vals)
        assert all(bool(np.isfinite(v)) for v in vals)

    def test_crossover_returns_child_within_bounds(self) -> None:
        opt = MultiObjectiveOptimizer(_bounds(3), ["a", "b"], seed=1)
        p1 = np.array([0.2, 0.5, 0.8])
        p2 = np.array([0.7, 0.3, 0.1])
        child = opt._crossover(p1, p2)
        assert child.shape == (3,)
        assert bool(np.all(child >= 0.0)) and bool(np.all(child <= 1.0))

    def test_mutate_changes_subset_of_entries(self) -> None:
        opt = MultiObjectiveOptimizer(_bounds(5), ["a", "b"], seed=2)
        vec = np.full(5, 0.5)
        mutated = opt._mutate(vec, prob=0.1)
        changed = int(np.sum(np.abs(mutated - vec) > 1e-9))
        # At least one entry changes, but not all ("at most some").
        assert 0 < changed < 5
        assert bool(np.all(mutated >= 0.0)) and bool(np.all(mutated <= 1.0))

    def test_tournament_select_returns_valid_index(self) -> None:
        opt = MultiObjectiveOptimizer(_bounds(4), ["a", "b"], seed=3)
        pop = [
            {"vec": np.zeros(4), "values": [0.0, 0.0], "rank": 0, "crowding": 1.0},
            {"vec": np.ones(4), "values": [1.0, 1.0], "rank": 1, "crowding": 0.0},
        ]
        idx = opt._tournament_select(ParetoFront(["a", "b"]), pop, k=2)
        assert idx in (0, 1)

    def test_invalid_bounds_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            MultiObjectiveOptimizer(np.array([0.0, 1.0]), ["a", "b"])


# --------------------------------------------------------------------------- #
# Objective function helpers
# --------------------------------------------------------------------------- #
class TestObjectiveFunctions:
    def test_lap_time_objective_returns_finite_float(self) -> None:
        val = lap_time_objective(DEFAULT_SETUP, "melbourne", None)
        assert isinstance(val, float)
        assert bool(np.isfinite(val))
        assert val > 0.0

    def test_tire_wear_objective_returns_finite_float(self) -> None:
        val = tire_wear_objective(DEFAULT_SETUP, "melbourne", None)
        assert isinstance(val, float)
        assert bool(np.isfinite(val))

    def test_compound_objective_returns_finite_float(self) -> None:
        val = compound_objective(DEFAULT_SETUP, "melbourne", None, [1.0, 1.0])
        assert isinstance(val, float)
        assert bool(np.isfinite(val))

    def test_constraint_penalty_satisfied_is_zero(self) -> None:
        # tyre_temp < 1000 is always satisfied (tyre_temp ~90).
        pen = constraint_penalty(DEFAULT_SETUP, "melbourne", {"tyre_temp": 1000})
        assert pen == 0.0

    def test_constraint_penalty_violated_is_positive(self) -> None:
        # tyre_temp < 0 is always violated (tyre_temp > 0).
        pen = constraint_penalty(DEFAULT_SETUP, "melbourne", {"tyre_temp": 0})
        assert pen > 0.0

    def test_constraint_penalty_tuple_op(self) -> None:
        # Explicit (op, limit) form: tyre_temp > 0 always satisfied -> 0.
        pen = constraint_penalty(
            DEFAULT_SETUP, "melbourne", {"tyre_temp": (">", 0)}
        )
        assert pen == 0.0

    def test_constraint_penalty_empty_constraints_zero(self) -> None:
        assert constraint_penalty(DEFAULT_SETUP, "melbourne", {}) == 0.0
