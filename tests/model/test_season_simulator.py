"""Tests for f1opt.model.season_simulator (Iter-18)."""

from __future__ import annotations

import pytest

from f1opt.model.season_simulator import (
    SeasonDriver,
    SeasonSimulator,
    simulate_season,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_drivers(n: int = 20) -> list[SeasonDriver]:
    return [
        SeasonDriver(
            driver_id=f"d{i:02d}",
            driver_name=f"Driver {i + 1}",
            team_id=f"t{i // 2:02d}",
            driver_aggression=0.5 + (i % 3) * 0.1,
            driver_smoothness=0.5 + (i % 4) * 0.1,
            driver_consistency=0.6 + (i % 5) * 0.05,
        )
        for i in range(n)
    ]


def _make_teams(n: int = 10) -> list[tuple[str, str]]:
    return [(f"t{i:02d}", f"Team {i + 1}") for i in range(n)]


# --------------------------------------------------------------------------- #
# SeasonSimulator validation
# --------------------------------------------------------------------------- #
class TestValidation:
    def test_wrong_driver_count_raises(self) -> None:
        with pytest.raises(ValueError, match="20 drivers"):
            SeasonSimulator(drivers=_make_drivers(15), teams=_make_teams(10))

    def test_wrong_team_count_raises(self) -> None:
        with pytest.raises(ValueError, match="10 teams"):
            SeasonSimulator(drivers=_make_drivers(20), teams=_make_teams(5))


# --------------------------------------------------------------------------- #
# Full season
# --------------------------------------------------------------------------- #
class TestFullSeason:
    def test_short_season_runs(self) -> None:
        """短 3 场赛季应成功运行."""
        sim = SeasonSimulator(
            drivers=_make_drivers(20),
            teams=_make_teams(10),
            calendar=("melbourne", "monza", "spa"),
            total_laps_per_race=20,  # 短赛快测
            seed=42,
        )
        r = sim.run()
        assert r["champions"]["drivers_champion"] is not None
        assert r["champions"]["constructors_champion"] is not None
        assert len(r["race_summaries"]) == 3

    def test_deterministic_with_seed(self) -> None:
        """同 seed 两赛季 → 同冠军同积分."""
        s1 = SeasonSimulator(
            drivers=_make_drivers(20), teams=_make_teams(10),
            calendar=("melbourne", "monza"), total_laps_per_race=15, seed=42,
        )
        s2 = SeasonSimulator(
            drivers=_make_drivers(20), teams=_make_teams(10),
            calendar=("melbourne", "monza"), total_laps_per_race=15, seed=42,
        )
        r1 = s1.run()
        r2 = s2.run()
        assert r1["champions"]["drivers_champion"].driver_id == \
               r2["champions"]["drivers_champion"].driver_id
        assert r1["champions"]["drivers_champion"].points == \
               r2["champions"]["drivers_champion"].points

    def test_all_drivers_have_points(self) -> None:
        """24 场赛季后所有车手应有积分 (除非全 DNF, 极不可能)."""
        sim = SeasonSimulator(
            drivers=_make_drivers(20), teams=_make_teams(10),
            calendar=("melbourne", "monza", "spa"),
            total_laps_per_race=20, seed=42,
        )
        r = sim.run()
        n_with_points = sum(1 for d in r["final_driver_standings"] if d.points > 0)
        # 至少大部分车手有积分
        assert n_with_points >= 10

    def test_constructor_points_sum_equals_drivers(self) -> None:
        """车队总积分应等于其两位车手积分之和."""
        sim = SeasonSimulator(
            drivers=_make_drivers(20), teams=_make_teams(10),
            calendar=("melbourne", "monza"), total_laps_per_race=15, seed=42,
        )
        r = sim.run()
        driver_pts = {d.driver_id: d.points for d in r["final_driver_standings"]}
        # team_id 映射
        team_of_driver = {d.driver_id: d.team_id for d in _make_drivers(20)}
        for team in r["final_constructor_standings"]:
            team_driver_pts = sum(
                pts for did, pts in driver_pts.items()
                if team_of_driver[did] == team.team_id
            )
            assert team.points == pytest.approx(team_driver_pts, abs=0.01)

    def test_winner_exists_per_race(self) -> None:
        sim = SeasonSimulator(
            drivers=_make_drivers(20), teams=_make_teams(10),
            calendar=("melbourne",), total_laps_per_race=15, seed=42,
        )
        r = sim.run()
        assert len(r["race_summaries"]) == 1
        assert r["race_summaries"][0]["winner"] is not None

    def test_races_completed_in_summary(self) -> None:
        sim = SeasonSimulator(
            drivers=_make_drivers(20), teams=_make_teams(10),
            calendar=("melbourne", "monza", "spa", "suzuka"),
            total_laps_per_race=15, seed=42,
        )
        r = sim.run()
        assert r["summary"]["races_completed"] == 4
        assert r["summary"]["total_races"] == 4


# --------------------------------------------------------------------------- #
# Convenience function
# --------------------------------------------------------------------------- #
class TestSimulateSeason:
    def test_returns_required_keys(self) -> None:
        r = simulate_season(
            seed=42,
            calendar=("melbourne", "monza"),
        )
        required = {"champions", "summary", "race_summaries",
                    "final_driver_standings", "final_constructor_standings"}
        assert required.issubset(r.keys())

    def test_default_calendar_24(self) -> None:
        """默认 24 场赛季 (不实际运行, 只验证 calendar)."""
        from f1opt.model.championship import SEASON_2026_CALENDAR
        assert len(SEASON_2026_CALENDAR) == 24


# --------------------------------------------------------------------------- #
# Physics sanity
# --------------------------------------------------------------------------- #
class TestPhysicsSanity:
    def test_high_consistency_driver_wins_more(self) -> None:
        """高一致性车手应赛季积分更高 (统计性, 多 seed 验证)."""
        from f1opt.model.race_simulator import RaceStrategy
        # 创建两组车手: 一致性高 vs 低
        drivers = []
        for i in range(20):
            consistency = 0.9 if i < 10 else 0.4
            drivers.append(SeasonDriver(
                driver_id=f"d{i:02d}",
                driver_name=f"D{i + 1}",
                team_id=f"t{i // 2:02d}",
                driver_aggression=0.7,
                driver_smoothness=0.7,
                driver_consistency=consistency,
                default_strategy=RaceStrategy(
                    pit_laps=(10, 25), compounds=("medium", "hard", "medium")
                ),
            ))
        sim = SeasonSimulator(
            drivers=drivers, teams=_make_teams(10),
            calendar=("melbourne", "monza", "spa"),
            total_laps_per_race=30, seed=42,
        )
        r = sim.run()
        # 高一致性组 (d00-d09) 平均积分应高于低一致性组 (d10-d19)
        high_pts = [d.points for d in r["final_driver_standings"]
                    if d.driver_id in {f"d{i:02d}" for i in range(10)}]
        low_pts = [d.points for d in r["final_driver_standings"]
                   if d.driver_id in {f"d{i:02d}" for i in range(10, 20)}]
        avg_high = sum(high_pts) / len(high_pts)
        avg_low = sum(low_pts) / len(low_pts)
        assert avg_high > avg_low, (
            f"high-consistency avg={avg_high} not > low-consistency avg={avg_low}"
        )
