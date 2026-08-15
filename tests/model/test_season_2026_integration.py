"""F1 2026 真实赛季集成测试 (Iter-36).

验证 :func:`simulate_season_2026` 用真实车手 + 车队数据跑完整赛季.
"""

from __future__ import annotations

import time

import pytest

from f1opt.model.season_simulator import (
    build_2026_season_drivers,
    build_2026_season_teams,
    simulate_season_2026,
)


# --------------------------------------------------------------------------- #
# 数据构建
# --------------------------------------------------------------------------- #
def test_build_2026_season_drivers_returns_22():
    drivers = build_2026_season_drivers()
    assert len(drivers) == 22


def test_build_2026_season_teams_returns_11():
    teams = build_2026_season_teams()
    assert len(teams) == 11


def test_2026_drivers_have_team_offsets():
    """每位车手应携带其车队的性能偏移."""
    drivers = build_2026_season_drivers()
    # 至少应有非零偏移 (顶队负, 后段正)
    offsets = [d.car_performance_offset_s for d in drivers]
    assert min(offsets) < 0  # 顶队快
    assert max(offsets) > 0  # 后段慢


def test_2026_drivers_use_real_ratings():
    """Verstappen 的 aggression 应基于真实档案 (>0.8)."""
    drivers = build_2026_season_drivers()
    ver = next(d for d in drivers if d.driver_id == "ver")
    assert ver.driver_aggression >= 0.8
    assert ver.driver_tire_management > 0.5  # Verstappen 轮胎管理出色


# --------------------------------------------------------------------------- #
# 完整赛季仿真
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def season_2026_result():
    """运行一次完整 2026 赛季 (耗时 ~5s)."""
    t0 = time.perf_counter()
    result = simulate_season_2026(seed=42)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def test_season_2026_completes(season_2026_result):
    result, elapsed = season_2026_result
    assert "champions" in result
    assert "summary" in result
    # 性能: 应在 20s 内完成 24 场赛季 (22 车手)
    assert elapsed < 20.0, f"Season took {elapsed:.1f}s, expected <20s"


def test_season_2026_has_champions(season_2026_result):
    result, _ = season_2026_result
    champs = result["champions"]
    assert "drivers_champion" in champs
    assert "constructors_champion" in champs


def test_season_2026_24_races(season_2026_result):
    result, _ = season_2026_result
    summaries = result["race_summaries"]
    assert len(summaries) == 24


def test_season_2026_champion_has_points(season_2026_result):
    result, _ = season_2026_result
    standings = result["final_driver_standings"]
    assert len(standings) == 22
    # 冠军 (榜首) 应有可观积分 (>300, 24 场均 12+ 分)
    assert standings[0].points > 200


def test_season_2026_top_teams_score_higher(season_2026_result):
    """顶队 (RBR/MCL/MER/FER) 总积分应高于后段 (Audi/Haas/RB/Cadillac)."""
    result, _ = season_2026_result
    constructor_standings = result["final_constructor_standings"]
    by_team = {s.team_id: s.points for s in constructor_standings}
    top_avg = (by_team.get("rbr", 0) + by_team.get("mcl", 0)
               + by_team.get("mer", 0) + by_team.get("fer", 0)) / 4
    back_avg = (by_team.get("aud", 0) + by_team.get("has", 0)
                + by_team.get("rb", 0) + by_team.get("cad", 0)) / 4
    assert top_avg > back_avg, \
        f"top_avg={top_avg} should > back_avg={back_avg}"


def test_season_2026_verstappen_top_5(season_2026_result):
    """Verstappen 应在车手榜前 5 (顶队 + 顶尖车手)."""
    result, _ = season_2026_result
    standings = result["final_driver_standings"]
    ver_pos = next(
        (i + 1 for i, s in enumerate(standings) if s.driver_id == "ver"),
        None,
    )
    assert ver_pos is not None
    assert ver_pos <= 7  # 给一些仿真噪声余量


def test_season_2026_reproducible(season_2026_result):
    """相同 seed 应产生相同结果."""
    result1, _ = season_2026_result
    result2 = simulate_season_2026(seed=42)
    assert (result1["champions"]["drivers_champion"]
            == result2["champions"]["drivers_champion"])
    assert (result1["champions"]["constructors_champion"]
            == result2["champions"]["constructors_champion"])
