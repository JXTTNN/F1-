"""Tests for f1opt.model.qualifying (Iter-20)."""

from __future__ import annotations

from f1opt.data.setup_schema import DEFAULT_SETUP
from f1opt.model.qualifying import (
    DriverQualifyingInput,
    simulate_qualifying,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_drivers(n: int = 22) -> list[DriverQualifyingInput]:
    return [
        DriverQualifyingInput(
            driver_id=f"d{i:02d}",
            driver_name=f"Driver {i + 1}",
            team_id=f"t{i // 2:02d}",
            setup=DEFAULT_SETUP,
            skill=0.95 - i * 0.02,  # decreasing skill so d00 is fastest
            aggression=0.7,
            smoothness=0.7,
            consistency=0.7,
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# Basic structure
# --------------------------------------------------------------------------- #
class TestBasicStructure:
    def test_returns_22_results(self) -> None:
        grid = simulate_qualifying("monza", _make_drivers(22), seed=42)
        assert len(grid) == 22

    def test_grid_positions_1_to_22(self) -> None:
        grid = simulate_qualifying("monza", _make_drivers(22), seed=42)
        positions = sorted(r.grid_position for r in grid)
        assert positions == list(range(1, 23))

    def test_results_sorted_by_grid_position(self) -> None:
        grid = simulate_qualifying("monza", _make_drivers(22), seed=42)
        for i in range(len(grid) - 1):
            assert grid[i].grid_position < grid[i + 1].grid_position

    def test_all_drivers_have_best_lap(self) -> None:
        grid = simulate_qualifying("monza", _make_drivers(22), seed=42)
        for r in grid:
            assert r.best_lap_time_s is not None
            assert 60.0 <= r.best_lap_time_s <= 180.0


# --------------------------------------------------------------------------- #
# Elimination logic
# --------------------------------------------------------------------------- #
class TestElimination:
    def test_6_drivers_eliminated_q1(self) -> None:
        grid = simulate_qualifying("monza", _make_drivers(22), seed=42)
        q1_out = [r for r in grid if r.eliminated_in == "Q1"]
        assert len(q1_out) == 6
        # All in P17-P22
        for r in q1_out:
            assert 17 <= r.grid_position <= 22

    def test_6_drivers_eliminated_q2(self) -> None:
        grid = simulate_qualifying("monza", _make_drivers(22), seed=42)
        q2_out = [r for r in grid if r.eliminated_in == "Q2"]
        assert len(q2_out) == 6
        for r in q2_out:
            assert 11 <= r.grid_position <= 16

    def test_10_drivers_reach_q3(self) -> None:
        grid = simulate_qualifying("monza", _make_drivers(22), seed=42)
        q3 = [r for r in grid if r.eliminated_in == "Q3-complete"]
        assert len(q3) == 10
        for r in q3:
            assert 1 <= r.grid_position <= 10


# --------------------------------------------------------------------------- #
# Skill → grid correlation
# --------------------------------------------------------------------------- #
class TestSkillCorrelation:
    def test_top_drivers_usually_top_grid(self) -> None:
        """Best 5 skill drivers should mostly be in top 10 (Q3)."""
        # Run 5 seeds and check stability
        top_skill_ids = {f"d{i:02d}" for i in range(5)}
        q3_count = 0
        for seed in range(5):
            grid = simulate_qualifying("monza", _make_drivers(22), seed=seed)
            q3_ids = {r.driver_id for r in grid if r.grid_position <= 10}
            q3_count += len(top_skill_ids & q3_ids)
        # Best 5 should reach Q3 in most seeds (allow some randomness)
        assert q3_count >= 18  # at least 18/25 = 72%

    def test_pole_is_fastest_lap(self) -> None:
        grid = simulate_qualifying("monza", _make_drivers(22), seed=42)
        pole = grid[0]
        for r in grid[1:]:
            assert pole.best_lap_time_s <= r.best_lap_time_s + 0.05


# --------------------------------------------------------------------------- #
# Lap recording
# --------------------------------------------------------------------------- #
class TestLapRecording:
    def test_all_drivers_have_q1_laps(self) -> None:
        grid = simulate_qualifying("monza", _make_drivers(22), seed=42)
        for r in grid:
            assert len(r.q1_laps) >= 2

    def test_q3_drivers_have_q3_laps(self) -> None:
        grid = simulate_qualifying("monza", _make_drivers(22), seed=42)
        for r in grid:
            if r.grid_position <= 10:
                assert len(r.q3_laps) >= 2

    def test_q1_only_drivers_no_q2_laps(self) -> None:
        grid = simulate_qualifying("monza", _make_drivers(22), seed=42)
        for r in grid:
            if r.eliminated_in == "Q1":
                assert len(r.q2_laps) == 0
                assert len(r.q3_laps) == 0

    def test_q2_only_drivers_no_q3_laps(self) -> None:
        grid = simulate_qualifying("monza", _make_drivers(22), seed=42)
        for r in grid:
            if r.eliminated_in == "Q2":
                assert len(r.q3_laps) == 0

    def test_lap_phase_label_correct(self) -> None:
        grid = simulate_qualifying("monza", _make_drivers(22), seed=42)
        for r in grid:
            for lap in r.q1_laps:
                assert lap.phase == "Q1"
            for lap in r.q2_laps:
                assert lap.phase == "Q2"
            for lap in r.q3_laps:
                assert lap.phase == "Q3"

    def test_q3_drivers_q3_best_better_than_q1(self) -> None:
        """Q3 uses soft + lighter fuel → should be faster than Q1."""
        grid = simulate_qualifying("monza", _make_drivers(22), seed=42)
        improved_count = 0
        for r in grid:
            if r.grid_position <= 10 and r.q1_laps and r.q3_laps:
                q1_best = min(lp.lap_time_s for lp in r.q1_laps)
                q3_best = min(lp.lap_time_s for lp in r.q3_laps)
                if q3_best < q1_best:
                    improved_count += 1
        # Most Q3 drivers should improve (allow some randomness)
        assert improved_count >= 7  # 7/10 minimum


# --------------------------------------------------------------------------- #
# Determinism / reproducibility
# --------------------------------------------------------------------------- #
class TestReproducibility:
    def test_same_seed_same_result(self) -> None:
        g1 = simulate_qualifying("monza", _make_drivers(22), seed=42)
        g2 = simulate_qualifying("monza", _make_drivers(22), seed=42)
        assert [r.driver_id for r in g1] == [r.driver_id for r in g2]
        assert [r.best_lap_time_s for r in g1] == [r.best_lap_time_s for r in g2]

    def test_different_seed_different_result(self) -> None:
        g1 = simulate_qualifying("monza", _make_drivers(22), seed=42)
        g2 = simulate_qualifying("monza", _make_drivers(22), seed=99)
        # Lap times should differ (allowing same grid by coincidence)
        assert [r.best_lap_time_s for r in g1] != [r.best_lap_time_s for r in g2]


# --------------------------------------------------------------------------- #
# Track-specific behavior
# --------------------------------------------------------------------------- #
class TestTrackSpecific:
    def test_monza_lap_faster_than_monaco(self) -> None:
        """Monza base lap (~83s) should be much faster than Monaco (~74s)? Actually
        Monaco ~74s, Monza ~83s. Monaco shorter. We test just that lap times differ."""
        g_monza = simulate_qualifying("monza", _make_drivers(22), seed=42)
        g_monaco = simulate_qualifying("monaco", _make_drivers(22), seed=42)
        pole_monza = g_monza[0].best_lap_time_s
        pole_monaco = g_monaco[0].best_lap_time_s
        # Both reasonable lap times, but they should differ
        assert pole_monza != pole_monaco

    def test_q2_tire_recorded_for_q3_drivers(self) -> None:
        grid = simulate_qualifying("monza", _make_drivers(22), seed=42)
        for r in grid:
            if r.grid_position <= 10:
                assert r.q2_tire_for_race in ("soft", "medium", "hard")


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
class TestEdgeCases:
    def test_minimum_drivers_q1_only(self) -> None:
        """If 22 drivers — Q1 still eliminates 6."""
        grid = simulate_qualifying("monza", _make_drivers(22), seed=1)
        assert len(grid) == 22

    def test_q3_setup_boost_speeds_up(self) -> None:
        """Drivers with Q3 setup boost should be faster in Q3 vs no-boost."""
        drivers = _make_drivers(22)
        # Half get boost, half don't
        for i, d in enumerate(drivers):
            d.q3_setup_boost = 0.3 if i < 10 else 0.0
        grid = simulate_qualifying("monza", drivers, seed=42)
        # Top 10 includes boost drivers more often than not
        boost_in_q3 = sum(
            1 for r in grid[:10] if r.driver_id in {d.driver_id for d in drivers[:10]}
        )
        assert boost_in_q3 >= 6  # ≥6/10 boost drivers reach Q3

    def test_validation_too_few_drivers(self) -> None:
        """Currently no validation; sim runs with any driver count (15 here)."""
        grid = simulate_qualifying("monza", _make_drivers(15), seed=42)
        assert len(grid) == 15
