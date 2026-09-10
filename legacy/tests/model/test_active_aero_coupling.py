"""F1 2026 主动空动与圈速精确耦合测试 (Iter-55)."""

from __future__ import annotations

from f1opt.model.active_aero_coupling import (
    active_aero_lap_gain_s,
    active_aero_total_lap_effect_s,
    aero_pu_synergy_gain_s,
    fuel_save_from_aero_pct,
    max_fuel_save_pct,
    transition_penalty_per_activation_s,
    wet_aero_penalty_s,
    x_mode_straight_gain_s,
    x_mode_top_speed_boost_kmh,
    z_mode_corner_baseline_s,
)


# --------------------------------------------------------------------------- #
# 单直道 X-mode 收益
# --------------------------------------------------------------------------- #
def test_x_mode_gain_zero_short_straight():
    assert x_mode_straight_gain_s(300) == 0.0


def test_x_mode_gain_positive_long_straight():
    assert x_mode_straight_gain_s(1000) > 0


def test_x_mode_gain_monza_main():
    """Monza 主直道 1200m ≈ 0.36s."""
    gain = x_mode_straight_gain_s(1200)
    assert 0.30 < gain < 0.45


def test_x_mode_gain_proportional():
    g1 = x_mode_straight_gain_s(500)
    g2 = x_mode_straight_gain_s(1000)
    assert abs(g2 - 2 * g1) < 1e-9


def test_x_mode_top_speed_boost():
    assert x_mode_top_speed_boost_kmh() == 10.0


# --------------------------------------------------------------------------- #
# 省油
# --------------------------------------------------------------------------- #
def test_fuel_save_zero_no_activation():
    assert fuel_save_from_aero_pct(0) == 0.0


def test_fuel_save_one_activation():
    assert fuel_save_from_aero_pct(1) == 2.0


def test_fuel_save_three_activations():
    assert fuel_save_from_aero_pct(3) == 6.0


def test_fuel_save_capped_at_6():
    """超过 3 次也上限 6%."""
    assert fuel_save_from_aero_pct(5) == 6.0


def test_max_fuel_save_6pct():
    assert max_fuel_save_pct() == 6.0


# --------------------------------------------------------------------------- #
# Z-mode 弯角基准
# --------------------------------------------------------------------------- #
def test_z_mode_baseline_positive():
    """Z-mode 弯角基准收益应为正."""
    assert z_mode_corner_baseline_s() > 0


def test_z_mode_baseline_value():
    assert z_mode_corner_baseline_s() == 0.15


# --------------------------------------------------------------------------- #
# 单圈主动空动总收益
# --------------------------------------------------------------------------- #
def test_lap_gain_dry_monza():
    """干地 Monza 主动空动收益应为正."""
    gain = active_aero_lap_gain_s("monza", gap_to_ahead_s=0.8)
    assert gain > 0


def test_lap_gain_wet_negative():
    """湿地禁用 X-mode, 圈速损失."""
    gain = active_aero_lap_gain_s("monza", wet_conditions=True)
    assert gain < 0
    assert gain == -wet_aero_penalty_s()


def test_lap_gain_far_gap_only_z_mode():
    """gap > 1s 无法激活 X-mode, 仅 Z-mode 基准."""
    gain = active_aero_lap_gain_s("monza", gap_to_ahead_s=2.0)
    assert abs(gain - z_mode_corner_baseline_s()) < 1e-9


def test_lap_gain_monaco_minimal():
    """Monaco 直道短, X-mode 收益少, 主要靠 Z-mode."""
    gain_monaco = active_aero_lap_gain_s("monaco", gap_to_ahead_s=0.8)
    gain_monza = active_aero_lap_gain_s("monza", gap_to_ahead_s=0.8)
    assert gain_monaco < gain_monza  # Monaco 收益少


def test_lap_gain_includes_z_mode():
    """干地收益应包含 Z-mode 弯角基准."""
    gain = active_aero_lap_gain_s("monza", gap_to_ahead_s=0.8)
    assert gain > z_mode_corner_baseline_s()  # X + Z > Z


# --------------------------------------------------------------------------- #
# 过渡损失
# --------------------------------------------------------------------------- #
def test_transition_penalty_per_activation():
    assert transition_penalty_per_activation_s() == 0.02


def test_wet_penalty():
    assert wet_aero_penalty_s() == 0.80


# --------------------------------------------------------------------------- #
# 主动空动 + PU 协同
# --------------------------------------------------------------------------- #
def test_synergy_zero_wet():
    """湿地无协同."""
    s = aero_pu_synergy_gain_s("monza", 6.0, wet_conditions=True)
    assert s == 0.0


def test_synergy_zero_far_gap():
    """gap > 1s 无激活, 无协同."""
    s = aero_pu_synergy_gain_s("monza", 6.0, gap_to_ahead_s=2.0)
    assert s == 0.0


def test_synergy_positive_normal():
    """正常条件协同收益为正."""
    s = aero_pu_synergy_gain_s("monza", 6.0, gap_to_ahead_s=0.8)
    assert s > 0


def test_synergy_increases_with_pu_deploy():
    """PU 部署越多, 协同收益越大."""
    s_low = aero_pu_synergy_gain_s("monza", 4.0, gap_to_ahead_s=0.8)
    s_high = aero_pu_synergy_gain_s("monza", 9.0, gap_to_ahead_s=0.8)
    assert s_high > s_low


# --------------------------------------------------------------------------- #
# 完整圈速影响
# --------------------------------------------------------------------------- #
def test_total_effect_structure():
    r = active_aero_total_lap_effect_s("monza", gap_to_ahead_s=0.8)
    assert "total_gain_s" in r
    assert "x_mode_gain_s" in r
    assert "z_mode_gain_s" in r
    assert "transition_penalty_s" in r
    assert "synergy_gain_s" in r
    assert "fuel_save_pct" in r
    assert "top_speed_boost_kmh" in r


def test_total_effect_dry_positive():
    r = active_aero_total_lap_effect_s("monza", gap_to_ahead_s=0.8)
    assert r["total_gain_s"] > 0


def test_total_effect_wet_negative():
    r = active_aero_total_lap_effect_s("monza", wet_conditions=True)
    assert r["total_gain_s"] < 0
    assert r["x_mode_gain_s"] == 0.0


def test_total_effect_far_gap_only_z():
    r = active_aero_total_lap_effect_s("monza", gap_to_ahead_s=2.0)
    # 仅 Z-mode, 无 X-mode
    assert r["x_mode_gain_s"] == 0.0
    assert r["z_mode_gain_s"] > 0


def test_total_effect_components_sum():
    """total = x + z + transition + synergy."""
    r = active_aero_total_lap_effect_s("monza", gap_to_ahead_s=0.8, pu_deploy_mj=6.0)
    expected = (r["x_mode_gain_s"] + r["z_mode_gain_s"]
                + r["transition_penalty_s"] + r["synergy_gain_s"])
    assert abs(r["total_gain_s"] - expected) < 1e-9


def test_total_effect_fuel_save_with_activations():
    r = active_aero_total_lap_effect_s("monza", gap_to_ahead_s=0.8)
    if r["x_mode_gain_s"] > 0:
        assert r["fuel_save_pct"] > 0


def test_total_effect_top_speed_boost():
    r = active_aero_total_lap_effect_s("monza", gap_to_ahead_s=0.8)
    if r["x_mode_gain_s"] > 0:
        assert r["top_speed_boost_kmh"] == 10.0


# --------------------------------------------------------------------------- #
# 赛道对比 (EA F1 2026 物理对标)
# --------------------------------------------------------------------------- #
def test_monza_high_aero_gain():
    """Monza 长直道多, 主动空动收益最大."""
    monza = active_aero_lap_gain_s("monza", gap_to_ahead_s=0.8)
    assert monza > 0.5  # 3 直道 + Z-mode


def test_monaco_low_aero_gain():
    """Monaco 直道短, 收益主要靠 Z-mode."""
    monaco = active_aero_lap_gain_s("monaco", gap_to_ahead_s=0.8)
    # Monaco 直道 < 400m, 无 X-mode, 仅 Z-mode
    assert abs(monaco - z_mode_corner_baseline_s()) < 1e-9


def test_baku_uses_long_straight():
    """Baku 2.2km 直道, X-mode 收益大."""
    baku = active_aero_lap_gain_s("baku", gap_to_ahead_s=0.8)
    assert baku > 0.5


def test_spa_two_straights():
    spa = active_aero_lap_gain_s("spa", gap_to_ahead_s=0.8)
    assert spa > 0.3


# --------------------------------------------------------------------------- #
# 确定性
# --------------------------------------------------------------------------- #
def test_deterministic():
    g1 = active_aero_lap_gain_s("monza", gap_to_ahead_s=0.8)
    g2 = active_aero_lap_gain_s("monza", gap_to_ahead_s=0.8)
    assert g1 == g2


# --------------------------------------------------------------------------- #
# 实战场景
# --------------------------------------------------------------------------- #
def test_qualifying_vs_race_aero():
    """排位 (无前车) vs 正赛 (有前车) 主动空动差异.

    排位无前车 → gap 大 → 无法激活 X-mode → 仅 Z-mode.
    正赛有前车 → 可激活 X-mode.
    """
    quali_gain = active_aero_lap_gain_s("monza", gap_to_ahead_s=10.0)
    race_gain = active_aero_lap_gain_s("monza", gap_to_ahead_s=0.8)
    assert race_gain > quali_gain  # 正赛有 X-mode


def test_wet_race_no_x_mode():
    """湿地比赛全程无 X-mode."""
    for tid in ["monza", "spa", "baku"]:
        gain = active_aero_lap_gain_s(tid, wet_conditions=True)
        assert gain < 0  # 湿地损失


def test_full_lap_aero_pu_integration():
    """完整圈: 主动空动 + PU 部署协同."""
    r = active_aero_total_lap_effect_s(
        "monza", gap_to_ahead_s=0.8, pu_deploy_mj=9.0,  # 全力 PU
    )
    # 总收益应显著 (X + Z + 协同)
    assert r["total_gain_s"] > 0.6
    assert r["synergy_gain_s"] > 0  # PU 协同


def test_progressive_gap_threshold():
    """gap 从 0.8 → 1.0 → 1.01, X-mode 在 1.0 后消失."""
    g_08 = active_aero_lap_gain_s("monza", gap_to_ahead_s=0.8)
    g_10 = active_aero_lap_gain_s("monza", gap_to_ahead_s=1.0)
    g_101 = active_aero_lap_gain_s("monza", gap_to_ahead_s=1.01)
    assert g_08 > g_101  # 0.8 有 X-mode, 1.01 无
    assert g_10 > g_101  # 1.0 仍有, 1.01 无
