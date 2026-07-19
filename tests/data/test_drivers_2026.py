"""F1 2026 真实车手数据库测试 (Iter-35)."""

from __future__ import annotations

import pytest

from f1opt.data.drivers_2026 import (
    DriverProfile2026,
    all_drivers_2026,
    all_teams_2026,
    drivers_by_team,
    get_driver_2026,
)


# --------------------------------------------------------------------------- #
# 注册表完整性
# --------------------------------------------------------------------------- #
def test_20_drivers_registered():
    drivers = all_drivers_2026()
    assert len(drivers) == 20


def test_10_teams_registered():
    teams = all_teams_2026()
    assert len(teams) == 10
    team_ids = [t[0] for t in teams]
    assert len(set(team_ids)) == 10  # unique


def test_each_team_has_2_drivers():
    teams = all_teams_2026()
    for team_id, _ in teams:
        ds = drivers_by_team(team_id)
        assert len(ds) == 2, f"{team_id} has {len(ds)} drivers"


def test_driver_ids_unique():
    drivers = all_drivers_2026()
    ids = [d.driver_id for d in drivers]
    assert len(set(ids)) == 20


@pytest.mark.parametrize("team_id", ["rbr", "mer", "fer", "mcl", "amr",
                                     "alp", "wil", "rb", "kck", "has"])
def test_known_team_has_drivers(team_id):
    ds = drivers_by_team(team_id)
    assert len(ds) == 2


# --------------------------------------------------------------------------- #
# 字段验证
# --------------------------------------------------------------------------- #
def test_all_ratings_in_valid_range():
    for d in all_drivers_2026():
        for attr in ("pace", "race", "consistency", "tyre_management",
                     "wet", "defending"):
            v = getattr(d, attr)
            assert 70 <= v <= 99, f"{d.driver_id}.{attr}={v} out of [70,99]"


def test_top_teams_have_higher_pace():
    """Red Bull / Ferrari / Mercedes / McLaren 平均 pace 应高于后段车队."""
    top = (drivers_by_team("rbr") + drivers_by_team("mer")
           + drivers_by_team("fer") + drivers_by_team("mcl"))
    back = (drivers_by_team("kck") + drivers_by_team("has")
            + drivers_by_team("rb"))
    avg_top = sum(d.pace for d in top) / len(top)
    avg_back = sum(d.pace for d in back) / len(back)
    assert avg_top > avg_back


def test_verstappen_is_top_rated():
    ver = get_driver_2026("ver")
    assert ver.driver_name == "Max Verstappen"
    assert ver.pace >= 95
    assert ver.overall >= 95


# --------------------------------------------------------------------------- #
# overall 计算
# --------------------------------------------------------------------------- #
def test_overall_weighted_average():
    d = DriverProfile2026(
        driver_id="x", driver_name="X", team_id="t", team_name="T",
        country_code="zz",
        pace=90, race=90, consistency=90, tyre_management=90,
        wet=90, defending=90,
    )
    # all 90 → overall 90
    assert d.overall == 90


def test_overall_weights_pace_highest():
    """pace 占 25%, 是最大权重."""
    d_pace_high = DriverProfile2026(
        driver_id="x", driver_name="X", team_id="t", team_name="T",
        country_code="zz",
        pace=99, race=70, consistency=70, tyre_management=70,
        wet=70, defending=70,
    )
    d_pace_low = DriverProfile2026(
        driver_id="x", driver_name="X", team_id="t", team_name="T",
        country_code="zz",
        pace=70, race=99, consistency=99, tyre_management=99,
        wet=99, defending=99,
    )
    # pace 占 0.25, 其他共 0.75; high pace (99) vs low pace (70) — 差 29
    # high pace overall = 0.25*99 + 0.75*70 = 24.75 + 52.5 = 77.25 → 77
    # low pace overall  = 0.25*70 + 0.75*99 = 17.5 + 74.25 = 91.75 → 91
    assert d_pace_low.overall > d_pace_high.overall


# --------------------------------------------------------------------------- #
# 属性转换 (0..1)
# --------------------------------------------------------------------------- #
def test_aggression_in_range():
    for d in all_drivers_2026():
        assert 0.0 <= d.aggression <= 1.0


def test_smoothness_in_range():
    for d in all_drivers_2026():
        assert 0.0 <= d.smoothness <= 1.0


def test_consistency_in_range():
    for d in all_drivers_2026():
        assert 0.0 <= d.driver_consistency <= 1.0


def test_tire_management_in_range():
    for d in all_drivers_2026():
        assert 0.0 <= d.driver_tire_management <= 1.0


def test_alonso_high_defending():
    """Alonso 以防守著称, defending 应该是顶尖."""
    alo = get_driver_2026("alo")
    assert alo.defending >= 94


def test_hamilton_high_wet():
    """Hamilton 雨战能力顶尖."""
    ham = get_driver_2026("ham")
    assert ham.wet >= 93


def test_verstappen_high_aggression_and_smoothness():
    ver = get_driver_2026("ver")
    assert ver.aggression >= 0.8
    assert ver.smoothness >= 0.7


# --------------------------------------------------------------------------- #
# 查询函数
# --------------------------------------------------------------------------- #
def test_get_driver_returns_correct():
    d = get_driver_2026("lec")
    assert d.driver_name == "Charles Leclerc"
    assert d.team_id == "fer"


def test_get_driver_unknown_raises():
    with pytest.raises(ValueError, match="Unknown driver_id"):
        get_driver_2026("nonexistent")


def test_drivers_by_team_unknown_returns_empty():
    assert drivers_by_team("xxx") == []


def test_all_drivers_returns_copy():
    """all_drivers_2026() 应返回列表副本, 不能修改内部注册表."""
    lst1 = all_drivers_2026()
    lst1.append("hack")
    lst2 = all_drivers_2026()
    assert "hack" not in lst2
    assert len(lst2) == 20


# --------------------------------------------------------------------------- #
# team_id 与 driver_id 一致性
# --------------------------------------------------------------------------- #
def test_team_driver_id_consistency():
    """driver.team_id 应能在 all_teams_2026() 找到."""
    team_ids = {t[0] for t in all_teams_2026()}
    for d in all_drivers_2026():
        assert d.team_id in team_ids


def test_team_name_matches_team_id():
    """同一 team_id 的车手应有相同 team_name."""
    by_team: dict[str, set[str]] = {}
    for d in all_drivers_2026():
        by_team.setdefault(d.team_id, set()).add(d.team_name)
    for team_id, names in by_team.items():
        assert len(names) == 1, f"{team_id} has multiple names: {names}"
