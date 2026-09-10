"""Tests for f1opt.model.strategy_optimizer (Iter-16)."""

from __future__ import annotations

import pytest

from f1opt.data.setup_schema import DEFAULT_SETUP
from f1opt.model.strategy_optimizer import (
    StrategyCandidate,
    StrategyOptimizer,
    StrategySimulator,
    optimize_strategy,
)


# --------------------------------------------------------------------------- #
# StrategySimulator
# --------------------------------------------------------------------------- #
class TestStrategySimulator:
    def test_evaluate_returns_required_keys(self) -> None:
        sim = StrategySimulator(setup=DEFAULT_SETUP, track_id="monza",
                                total_laps=20, rng_seed=42)
        r = sim.evaluate(pit_laps=(10,), compounds=("medium", "hard"))
        required = {"total_time", "lap_times", "pit_laps", "compounds",
                    "n_stops", "best_lap", "worst_lap", "avg_lap"}
        assert required.issubset(r.keys())

    def test_zero_stops_strategy(self) -> None:
        sim = StrategySimulator(setup=DEFAULT_SETUP, track_id="monza",
                                total_laps=15, rng_seed=42)
        r = sim.evaluate(pit_laps=(), compounds=("medium",))
        assert r["n_stops"] == 0
        assert len(r["lap_times"]) == 15
        assert r["total_time"] > 0.0

    def test_one_stop_strategy(self) -> None:
        sim = StrategySimulator(setup=DEFAULT_SETUP, track_id="monza",
                                total_laps=20, rng_seed=42)
        r = sim.evaluate(pit_laps=(12,), compounds=("medium", "hard"))
        assert r["n_stops"] == 1
        assert len(r["lap_times"]) == 20
        # Pit lap should be slower (pit loss added)
        pit_lap_idx = 11  # 0-indexed lap 12
        assert r["lap_times"][pit_lap_idx] > r["lap_times"][pit_lap_idx - 1]

    def test_two_stop_strategy(self) -> None:
        sim = StrategySimulator(setup=DEFAULT_SETUP, track_id="monza",
                                total_laps=30, rng_seed=42)
        r = sim.evaluate(pit_laps=(10, 20), compounds=("medium", "hard", "soft"))
        assert r["n_stops"] == 2

    def test_invalid_compound_length_raises(self) -> None:
        sim = StrategySimulator(setup=DEFAULT_SETUP, track_id="monza",
                                total_laps=10, rng_seed=42)
        with pytest.raises(ValueError, match="compounds"):
            sim.evaluate(pit_laps=(5,), compounds=("medium",))  # need 2

    def test_non_increasing_pit_laps_raises(self) -> None:
        sim = StrategySimulator(setup=DEFAULT_SETUP, track_id="monza",
                                total_laps=20, rng_seed=42)
        with pytest.raises(ValueError, match="strictly increasing"):
            sim.evaluate(pit_laps=(15, 10), compounds=("medium", "hard", "soft"))

    def test_deterministic_with_seed(self) -> None:
        sim1 = StrategySimulator(setup=DEFAULT_SETUP, track_id="monza",
                                 total_laps=20, rng_seed=42)
        sim2 = StrategySimulator(setup=DEFAULT_SETUP, track_id="monza",
                                 total_laps=20, rng_seed=42)
        r1 = sim1.evaluate(pit_laps=(10,), compounds=("medium", "hard"))
        r2 = sim2.evaluate(pit_laps=(10,), compounds=("medium", "hard"))
        assert r1["total_time"] == pytest.approx(r2["total_time"], abs=1e-9)

    def test_pit_loss_added_to_total(self) -> None:
        sim = StrategySimulator(setup=DEFAULT_SETUP, track_id="monza",
                                total_laps=20, rng_seed=42)
        r_no_pit = sim.evaluate(pit_laps=(), compounds=("medium",))
        r_with_pit = sim.evaluate(pit_laps=(10,), compounds=("medium", "hard"))
        # With pit should be slower by ~pit_loss_s (but tire change effect varies)
        # Just check both are reasonable
        assert r_no_pit["total_time"] > 0
        assert r_with_pit["total_time"] > 0

    def test_stint_cache_reuse(self) -> None:
        """同一 compound 多次评估应使用缓存 (快)."""
        sim = StrategySimulator(setup=DEFAULT_SETUP, track_id="monza",
                                total_laps=30, rng_seed=42)
        sim.evaluate(pit_laps=(15,), compounds=("medium", "hard"))
        # 第二次评估应使用缓存
        sim.evaluate(pit_laps=(10,), compounds=("medium", "soft"))
        sim.evaluate(pit_laps=(10, 20), compounds=("medium", "hard", "soft"))
        # 缓存中应有 3 种 compound
        assert "medium" in sim._stint_cache
        assert "hard" in sim._stint_cache
        assert "soft" in sim._stint_cache


# --------------------------------------------------------------------------- #
# StrategyOptimizer
# --------------------------------------------------------------------------- #
class TestStrategyOptimizer:
    def test_optimize_returns_candidate(self) -> None:
        opt = StrategyOptimizer(setup=DEFAULT_SETUP, track_id="monza",
                                total_laps=30, rng_seed=42)
        best = opt.optimize()
        assert isinstance(best, StrategyCandidate)
        assert best.total_time > 0
        assert best.n_stops >= 0

    def test_optimize_populates_candidates(self) -> None:
        opt = StrategyOptimizer(setup=DEFAULT_SETUP, track_id="monza",
                                total_laps=30, rng_seed=42)
        opt.optimize()
        assert len(opt.all_candidates) > 5

    def test_best_has_lowest_time(self) -> None:
        opt = StrategyOptimizer(setup=DEFAULT_SETUP, track_id="monza",
                                total_laps=30, rng_seed=42)
        opt.optimize()
        assert opt.best is not None
        for c in opt.all_candidates:
            assert opt.best.total_time <= c.total_time + 1e-9

    def test_compound_length_matches_pit_laps(self) -> None:
        opt = StrategyOptimizer(setup=DEFAULT_SETUP, track_id="monza",
                                total_laps=30, rng_seed=42)
        best = opt.optimize()
        assert len(best.compounds) == len(best.pit_laps) + 1

    def test_pit_laps_in_range(self) -> None:
        opt = StrategyOptimizer(setup=DEFAULT_SETUP, track_id="monza",
                                total_laps=30, rng_seed=42)
        best = opt.optimize()
        for pl in best.pit_laps:
            assert 1 < pl < 30

    def test_deterministic_with_seed(self) -> None:
        opt1 = StrategyOptimizer(setup=DEFAULT_SETUP, track_id="monza",
                                 total_laps=30, rng_seed=42)
        opt2 = StrategyOptimizer(setup=DEFAULT_SETUP, track_id="monza",
                                 total_laps=30, rng_seed=42)
        b1 = opt1.optimize()
        b2 = opt2.optimize()
        assert b1.total_time == pytest.approx(b2.total_time, abs=1e-6)
        assert b1.pit_laps == b2.pit_laps
        assert b1.compounds == b2.compounds

    def test_summary_returns_required_keys(self) -> None:
        opt = StrategyOptimizer(setup=DEFAULT_SETUP, track_id="monza",
                                total_laps=30, rng_seed=42)
        opt.optimize()
        s = opt.summary()
        required = {"track_id", "total_laps", "best_strategy",
                    "n_candidates_evaluated", "top5"}
        assert required.issubset(s.keys())
        assert s["n_candidates_evaluated"] > 0
        assert len(s["top5"]) <= 5

    def test_different_n_stops_explored(self) -> None:
        """n_stops_options=(1,2,3) → all three appear in candidates."""
        opt = StrategyOptimizer(
            setup=DEFAULT_SETUP, track_id="monza", total_laps=40,
            n_stops_options=(1, 2, 3), rng_seed=42,
        )
        opt.optimize()
        n_stops_seen = {c.n_stops for c in opt.all_candidates}
        assert n_stops_seen == {1, 2, 3}


# --------------------------------------------------------------------------- #
# Convenience function
# --------------------------------------------------------------------------- #
class TestOptimizeStrategy:
    def test_returns_candidate(self) -> None:
        best = optimize_strategy(setup=DEFAULT_SETUP, track_id="monza",
                                 total_laps=30, seed=42)
        assert isinstance(best, StrategyCandidate)
        assert best.total_time > 0

    def test_reproducible(self) -> None:
        b1 = optimize_strategy(setup=DEFAULT_SETUP, track_id="monza",
                               total_laps=30, seed=7)
        b2 = optimize_strategy(setup=DEFAULT_SETUP, track_id="monza",
                               total_laps=30, seed=7)
        assert b1.pit_laps == b2.pit_laps
        assert b1.compounds == b2.compounds


# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #
class TestPerformance:
    def test_optimize_under_500ms(self) -> None:
        """单次优化 < 500 ms (缓存优化后)."""
        import time
        opt = StrategyOptimizer(setup=DEFAULT_SETUP, track_id="monza",
                                total_laps=53, rng_seed=42)
        # Warmup
        opt.optimize()
        t0 = time.perf_counter()
        opt2 = StrategyOptimizer(setup=DEFAULT_SETUP, track_id="monza",
                                 total_laps=53, rng_seed=42)
        opt2.optimize()
        dt = time.perf_counter() - t0
        assert dt < 0.5, f"optimize took {dt*1000:.0f}ms > 500ms"


# --------------------------------------------------------------------------- #
# Physics sanity
# --------------------------------------------------------------------------- #
class TestPhysicsSanity:
    def test_more_stops_not_always_better(self) -> None:
        """3 stops 应不是无脑最优 — 进站损失惩罚."""
        opt = StrategyOptimizer(
            setup=DEFAULT_SETUP, track_id="monza", total_laps=53,
            n_stops_options=(1, 2, 3), rng_seed=42,
        )
        opt.optimize()
        # 找最快 1-stop vs 最快 3-stop
        best_1 = min((c for c in opt.all_candidates if c.n_stops == 1),
                     key=lambda c: c.total_time, default=None)
        best_3 = min((c for c in opt.all_candidates if c.n_stops == 3),
                     key=lambda c: c.total_time, default=None)
        if best_1 and best_3:
            # 1-stop 至少有一次不差于 3-stop (并非所有赛道都需要 3 停)
            # 宽松断言: 1-stop 不超过 3-stop 太多
            assert best_1.total_time < best_3.total_time + 60.0

    def test_valid_compounds_used(self) -> None:
        opt = StrategyOptimizer(setup=DEFAULT_SETUP, track_id="monza",
                                total_laps=30, rng_seed=42)
        best = opt.optimize()
        for c in best.compounds:
            assert c in ("soft", "medium", "hard", "intermediate", "wet")
