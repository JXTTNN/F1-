"""F1 2026 车队性能数据库测试 (Iter-36)."""

from __future__ import annotations

import pytest

from f1opt.data.teams_2026 import (
    all_teams_2026_profiles,
    get_team_profile_2026,
    pace_offset_for_team,
    teams_by_pu_supplier,
)


# --------------------------------------------------------------------------- #
# 注册表完整性
# --------------------------------------------------------------------------- #
def test_10_teams_registered():
    teams = all_teams_2026_profiles()
    assert len(teams) == 10


def test_team_ids_unique():
    teams = all_teams_2026_profiles()
    ids = [t.team_id for t in teams]
    assert len(set(ids)) == 10


@pytest.mark.parametrize("team_id", ["rbr", "mer", "fer", "mcl", "amr",
                                     "alp", "wil", "rb", "kck", "has"])
def test_known_team_lookup(team_id):
    t = get_team_profile_2026(team_id)
    assert t.team_id == team_id


def test_unknown_team_raises():
    with pytest.raises(ValueError, match="Unknown team_id"):
        get_team_profile_2026("nonexistent")


# --------------------------------------------------------------------------- #
# 性能参数范围验证
# --------------------------------------------------------------------------- #
def test_pace_offset_in_reasonable_range():
    """顶队 -0.6s, 后段 +0.8s, 总范围 -0.8 .. +1.0."""
    for t in all_teams_2026_profiles():
        assert -0.8 <= t.pace_offset_s <= 1.0, \
            f"{t.team_id} pace_offset={t.pace_offset_s}"


def test_aero_efficiency_in_range():
    for t in all_teams_2026_profiles():
        assert 0.85 <= t.aero_efficiency <= 1.10


def test_power_unit_within_2026_regulation():
    """FIA 2026 规则: PU 峰值 ≤ 750 kW (ICE 400 + MGU-K 350).
    允许小范围超调 (车队实际可能略超)."""
    for t in all_teams_2026_profiles():
        assert 740 <= t.power_unit_kW <= 755, \
            f"{t.team_id} PU={t.power_unit_kW}kW out of range"


def test_tire_degradation_factor_in_range():
    for t in all_teams_2026_profiles():
        assert 0.90 <= t.tire_degradation_factor <= 1.15


def test_reliability_in_range():
    for t in all_teams_2026_profiles():
        assert 0.85 <= t.reliability <= 0.99


def test_drs_effectiveness_in_range():
    for t in all_teams_2026_profiles():
        assert 0.90 <= t.drs_effectiveness <= 1.10


def test_fuel_efficiency_in_range():
    for t in all_teams_2026_profiles():
        assert 0.95 <= t.fuel_efficiency <= 1.05


# --------------------------------------------------------------------------- #
# 性能层级合理性
# --------------------------------------------------------------------------- #
def test_top_teams_faster_than_backmarkers():
    """顶队 (RBR/MCL/MER/FER) 圈速偏移应显著低于后段 (Sauber/Haas)."""
    top = ["rbr", "mcl", "mer", "fer"]
    back = ["kck", "has"]
    avg_top = sum(pace_offset_for_team(t) for t in top) / len(top)
    avg_back = sum(pace_offset_for_team(t) for t in back) / len(back)
    assert avg_top < avg_back
    assert avg_top < 0  # 顶队快于基准
    assert avg_back > 0  # 后段慢于基准


def test_red_bull_among_fastest():
    rbr = get_team_profile_2026("rbr")
    assert rbr.pace_offset_s <= -0.4


def test_haas_among_slowest():
    has = get_team_profile_2026("has")
    assert has.pace_offset_s >= 0.5


def test_mclaren_top_aero_efficiency():
    """McLaren 2025 以空气动力学效率著称."""
    mcl = get_team_profile_2026("mcl")
    assert mcl.aero_efficiency >= 1.05


# --------------------------------------------------------------------------- #
# PU 供应商
# --------------------------------------------------------------------------- #
def test_valid_pu_suppliers():
    valid = {"mercedes", "ferrari", "honda_rbpt", "renault"}
    for t in all_teams_2026_profiles():
        assert t.power_unit_supplier in valid


def test_mercedes_pu_customers():
    """Mercedes HPP 供应 Mercedes + McLaren + Williams + Alpine (2026)."""
    merc_teams = teams_by_pu_supplier("mercedes")
    merc_ids = {t.team_id for t in merc_teams}
    assert "mer" in merc_ids
    assert "mcl" in merc_ids
    assert "wil" in merc_ids
    assert "alp" in merc_ids  # 2026 Alpine 切换到 Mercedes


def test_ferrari_pu_customers():
    """Ferrari 供应 Ferrari + Sauber + Haas."""
    fer_teams = teams_by_pu_supplier("ferrari")
    fer_ids = {t.team_id for t in fer_teams}
    assert "fer" in fer_ids
    assert "kck" in fer_ids
    assert "has" in fer_ids


def test_honda_rbpt_customers():
    """Honda RBPT 供应 Red Bull + Aston Martin + Racing Bulls (2026)."""
    honda_teams = teams_by_pu_supplier("honda_rbpt")
    honda_ids = {t.team_id for t in honda_teams}
    assert "rbr" in honda_ids
    assert "amr" in honda_ids  # 2026 AMR 切换到 Honda RBPT
    assert "rb" in honda_ids


def test_pu_suppliers_cover_all_teams():
    """所有 10 队都应能在 PU 供应商列表中找到."""
    all_teams = all_teams_2026_profiles()
    for t in all_teams:
        assert teams_by_pu_supplier(t.power_unit_supplier), \
            f"{t.team_id} supplier {t.power_unit_supplier} not found"


# --------------------------------------------------------------------------- #
# 退赛概率
# --------------------------------------------------------------------------- #
def test_retirement_probability_positive():
    for t in all_teams_2026_profiles():
        prob = t.retirement_probability_per_race
        assert 0.0 < prob < 0.10  # 0-10% per race


def test_top_teams_more_reliable():
    """顶队可靠性应高于后段."""
    top_rel = get_team_profile_2026("rbr").reliability
    back_rel = get_team_profile_2026("has").reliability
    assert top_rel >= back_rel


# --------------------------------------------------------------------------- #
# 便捷函数
# --------------------------------------------------------------------------- #
def test_pace_offset_for_team_returns_float():
    off = pace_offset_for_team("rbr")
    assert isinstance(off, float)
    assert off < 0  # RBR 快于基准


def test_pace_offset_for_unknown_raises():
    with pytest.raises(ValueError):
        pace_offset_for_team("xxx")
