"""Tests for sprint_weekend (Iter-32)."""

from __future__ import annotations

from f1opt.data.setup_schema import DEFAULT_SETUP
from f1opt.model.qualifying import DriverQualifyingInput
from f1opt.model.sprint_weekend import (
    SprintResult,
    SprintWeekendSimulator,
    simulate_sprint_weekend,
)


def _drivers(n: int = 20) -> list[DriverQualifyingInput]:
    return [
        DriverQualifyingInput(
            driver_id=f"d{i:02d}", driver_name=f"Driver {i+1}",
            team_id=f"t{i//2:02d}",
            setup=DEFAULT_SETUP,
            skill=0.95 - i * 0.02,
        )
        for i in range(n)
    ]


class TestSprintRace:
    def test_sprint_returns_result(self) -> None:
        sim = SprintWeekendSimulator(
            track_id="monza", drivers=_drivers(),
            race_setups={d.driver_id: d.setup for d in _drivers()},
            sprint_total_laps=10, seed=42,
        )
        r = sim.run_sprint()
        assert isinstance(r, SprintResult)
        assert r.track_id == "monza"

    def test_sprint_grid_has_20_drivers(self) -> None:
        sim = SprintWeekendSimulator(
            track_id="monza", drivers=_drivers(),
            race_setups={d.driver_id: d.setup for d in _drivers()},
            sprint_total_laps=10, seed=42,
        )
        r = sim.run_sprint()
        assert len(r.sprint_grid) == 20

    def test_sprint_classification_20_cars(self) -> None:
        sim = SprintWeekendSimulator(
            track_id="monza", drivers=_drivers(),
            race_setups={d.driver_id: d.setup for d in _drivers()},
            sprint_total_laps=10, seed=42,
        )
        r = sim.run_sprint()
        assert len(r.sprint_classification) == 20

    def test_sprint_points_top_8_only(self) -> None:
        sim = SprintWeekendSimulator(
            track_id="monza", drivers=_drivers(),
            race_setups={d.driver_id: d.setup for d in _drivers()},
            sprint_total_laps=10, seed=42,
        )
        r = sim.run_sprint()
        # 8-7-6-5-4-3-2-1 for positions 1-8
        nonzero = {d_id: p for d_id, p in r.sprint_points.items() if p > 0}
        assert len(nonzero) <= 8
        # Winner has 8
        winner_id = r.sprint_classification[0][1]
        assert r.sprint_points[winner_id] == 8

    def test_sprint_no_points_for_p9_plus(self) -> None:
        sim = SprintWeekendSimulator(
            track_id="monza", drivers=_drivers(),
            race_setups={d.driver_id: d.setup for d in _drivers()},
            sprint_total_laps=10, seed=42,
        )
        r = sim.run_sprint()
        # Position 9+ (assuming no DNFs) should have 0
        for pos, d_id, _ in r.sprint_classification[8:]:
            if pos <= 20:  # finished
                assert r.sprint_points[d_id] == 0


class TestFullWeekend:
    def test_full_weekend_returns_dict(self) -> None:
        r = simulate_sprint_weekend(
            track_id="monza", drivers=_drivers(),
            seed=42, sprint_total_laps=10, race_total_laps=15,
        )
        assert isinstance(r, dict)
        assert "sprint" in r
        assert "race_qualifying_grid" in r
        assert "race_classification" in r
        assert "sprint_points" in r
        assert "race_points" in r
        assert "total_points" in r
        assert "weekend_winner" in r

    def test_total_points_sprint_plus_race(self) -> None:
        r = simulate_sprint_weekend(
            track_id="monza", drivers=_drivers(),
            seed=42, sprint_total_laps=10, race_total_laps=15,
        )
        for d_id in r["total_points"]:
            expected = r["sprint_points"].get(d_id, 0) + r["race_points"].get(d_id, 0)
            assert r["total_points"][d_id] == expected

    def test_weekend_winner_has_max_points(self) -> None:
        r = simulate_sprint_weekend(
            track_id="monza", drivers=_drivers(),
            seed=42, sprint_total_laps=10, race_total_laps=15,
        )
        winner = r["weekend_winner"]
        max_pts = max(r["total_points"].values())
        assert r["total_points"][winner] == max_pts


class TestReproducibility:
    def test_same_seed_same_result(self) -> None:
        r1 = simulate_sprint_weekend(
            track_id="monza", drivers=_drivers(),
            seed=42, sprint_total_laps=10, race_total_laps=15,
        )
        r2 = simulate_sprint_weekend(
            track_id="monza", drivers=_drivers(),
            seed=42, sprint_total_laps=10, race_total_laps=15,
        )
        assert r1["sprint"].sprint_grid == r2["sprint"].sprint_grid
        assert r1["race_qualifying_grid"] == r2["race_qualifying_grid"]
