"""F1 2026 主动空气动力学激活策略测试 (Iter-50)."""

from __future__ import annotations

from f1opt.model.active_aero import (
    ActivationOpportunity,
    ActiveAeroMode,
    ActiveAeroStrategy,
    active_aero_gain_s,
    can_activate_x_mode,
    max_activations_per_lap,
    optimal_activation_plan,
    optimal_plan_for_track,
    straights_for_track,
)


# --------------------------------------------------------------------------- #
# ActiveAeroMode 枚举
# --------------------------------------------------------------------------- #
def test_mode_values():
    assert ActiveAeroMode.Z_MODE.value == "z_mode"
    assert ActiveAeroMode.X_MODE.value == "x_mode"


def test_two_modes():
    assert len(list(ActiveAeroMode)) == 2


# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
def test_max_activations_per_lap():
    assert max_activations_per_lap() == 3


# --------------------------------------------------------------------------- #
# active_aero_gain_s
# --------------------------------------------------------------------------- #
def test_gain_zero_for_short_straight():
    assert active_aero_gain_s(300) == 0.0  # < 400m


def test_gain_positive_for_long_straight():
    assert active_aero_gain_s(1000) > 0


def test_gain_proportional():
    g1 = active_aero_gain_s(500)
    g2 = active_aero_gain_s(1000)
    assert abs(g2 - 2 * g1) < 1e-9


def test_gain_monza_main_straight():
    """Monza 主直道 1200m 收益约 0.36s."""
    gain = active_aero_gain_s(1200)
    assert 0.30 < gain < 0.45


# --------------------------------------------------------------------------- #
# can_activate_x_mode
# --------------------------------------------------------------------------- #
def test_can_activate_normal_conditions():
    assert can_activate_x_mode(0.8) is True


def test_cannot_activate_far_gap():
    assert can_activate_x_mode(1.5) is False


def test_cannot_activate_wet():
    assert can_activate_x_mode(0.8, wet_conditions=True) is False


def test_cannot_activate_over_limit():
    assert can_activate_x_mode(0.8, activations_this_lap=3) is False


def test_can_activate_at_boundary():
    """gap = 1.0 (阈值) 应可激活."""
    assert can_activate_x_mode(1.0) is True


def test_cannot_activate_just_over_boundary():
    assert can_activate_x_mode(1.01) is False


# --------------------------------------------------------------------------- #
# optimal_activation_plan
# --------------------------------------------------------------------------- #
def test_plan_returns_strategy():
    plan = optimal_activation_plan([800, 600, 500], [1, 2, 3])
    assert isinstance(plan, ActiveAeroStrategy)


def test_plan_no_wet():
    """湿地条件无激活."""
    plan = optimal_activation_plan([800, 600], [1, 2], wet_conditions=True)
    assert plan.n_activations == 0


def test_plan_no_far_gap():
    """gap > 1s 无激活."""
    plan = optimal_activation_plan([800, 600], [1, 2], gap_to_ahead_s=2.0)
    assert plan.n_activations == 0


def test_plan_selects_top_3():
    """4 个直道时只选 top 3."""
    plan = optimal_activation_plan(
        [1000, 800, 600, 500], [1, 2, 3, 1],
    )
    assert plan.n_activations == 3


def test_plan_skips_short_straights():
    """短直道 (< 400m) 跳过."""
    plan = optimal_activation_plan([300, 800, 200], [1, 2, 3])
    assert plan.n_activations == 1  # 只有 800m


def test_plan_max_3():
    """超过 3 个长直道也只选 3."""
    plan = optimal_activation_plan(
        [1000, 900, 800, 700, 600], [1, 2, 3, 1, 2],
    )
    assert plan.n_activations == 3


def test_plan_total_gain_positive():
    plan = optimal_activation_plan([800, 600], [1, 2])
    assert plan.total_gain_s > 0


def test_plan_sorted_by_sector():
    """激活按扇区顺序."""
    plan = optimal_activation_plan([800, 1000, 600], [2, 1, 3])
    sectors = [a.sector_idx for a in plan.activations]
    assert sectors == sorted(sectors)


def test_plan_priority_assigned():
    plan = optimal_activation_plan([800, 600, 500], [1, 2, 3])
    priorities = [a.priority for a in plan.activations]
    assert priorities == list(range(1, len(priorities) + 1))


# --------------------------------------------------------------------------- #
# ActiveAeroStrategy 属性
# --------------------------------------------------------------------------- #
def test_strategy_can_activate():
    s = ActiveAeroStrategy(lap=1, gap_to_ahead_s=0.8)
    assert s.can_activate is True


def test_strategy_cannot_when_wet():
    s = ActiveAeroStrategy(lap=1, wet_conditions=True, gap_to_ahead_s=0.5)
    assert s.can_activate is False


def test_strategy_cannot_when_far():
    s = ActiveAeroStrategy(lap=1, gap_to_ahead_s=2.0)
    assert s.can_activate is False


def test_strategy_cannot_when_full():
    s = ActiveAeroStrategy(lap=1, gap_to_ahead_s=0.8)
    s.activations = [
        ActivationOpportunity(800, 1, 0.24),
        ActivationOpportunity(700, 2, 0.21),
        ActivationOpportunity(600, 3, 0.18),
    ]
    assert s.can_activate is False


def test_strategy_total_gain():
    s = ActiveAeroStrategy(lap=1)
    s.activations = [
        ActivationOpportunity(800, 1, 0.24),
        ActivationOpportunity(600, 2, 0.18),
    ]
    assert abs(s.total_gain_s - 0.42) < 1e-9


# --------------------------------------------------------------------------- #
# 赛道查询
# --------------------------------------------------------------------------- #
def test_straights_for_monza():
    s = straights_for_track("monza")
    assert len(s) >= 1
    assert max(s) >= 1000  # 主直道长


def test_straights_for_baku_long():
    """Baku 有 2.2km 超长直道."""
    s = straights_for_track("baku")
    assert max(s) >= 2000


def test_straights_for_monaco_short():
    """Monaco 直道最短."""
    s = straights_for_track("monaco")
    assert max(s) < 500


def test_straights_unknown_track():
    s = straights_for_track("nonexistent")
    assert s == [600.0]


def test_all_24_tracks_have_straights():
    track_ids = [
        "melbourne", "shanghai", "suzuka", "bahrain", "jeddah", "miami",
        "montreal", "monaco", "barcelona", "spielberg", "silverstone", "spa",
        "hungaroring", "zandvoort", "monza", "madrid", "baku", "singapore",
        "austin", "mexico_city", "interlagos", "las_vegas", "losail", "yas_marina",
    ]
    for tid in track_ids:
        s = straights_for_track(tid)
        assert len(s) >= 1


# --------------------------------------------------------------------------- #
# optimal_plan_for_track
# --------------------------------------------------------------------------- #
def test_plan_for_monza():
    plan = optimal_plan_for_track("monza")
    assert plan.n_activations <= 3
    if plan.n_activations > 0:
        assert plan.total_gain_s > 0


def test_plan_for_monaco_minimal():
    """Monaco 直道短, 激活少."""
    plan = optimal_plan_for_track("monaco")
    # Monaco 直道 250m < 400m 阈值, 无激活
    assert plan.n_activations == 0


def test_plan_for_baku_uses_long_straight():
    """Baku 2.2km 直道应被选中."""
    plan = optimal_plan_for_track("baku")
    if plan.n_activations > 0:
        gains = [a.gain_s for a in plan.activations]
        assert max(gains) > 0.5  # 2200m × 0.0003 = 0.66s


def test_plan_wet_no_activations():
    plan = optimal_plan_for_track("monza", wet_conditions=True)
    assert plan.n_activations == 0


def test_plan_far_gap_no_activations():
    plan = optimal_plan_for_track("monza", gap_to_ahead_s=2.0)
    assert plan.n_activations == 0


# --------------------------------------------------------------------------- #
# 确定性
# --------------------------------------------------------------------------- #
def test_deterministic():
    p1 = optimal_plan_for_track("monza")
    p2 = optimal_plan_for_track("monza")
    assert p1.n_activations == p2.n_activations
    assert p1.total_gain_s == p2.total_gain_s


# --------------------------------------------------------------------------- #
# 实战场景
# --------------------------------------------------------------------------- #
def test_monza_optimal_strategy():
    """Monza 最优激活策略: 主直道 + 2 个次直道."""
    plan = optimal_plan_for_track("monza")
    # Monza 有 3 个长直道, 应全用
    assert plan.n_activations == 3
    assert plan.total_gain_s > 0.5


def test_bahrain_three_zones():
    """Bahrain 3 个 DRS 区, 应有 3 次激活."""
    plan = optimal_plan_for_track("bahrain")
    assert plan.n_activations <= 3


def test_spa_two_long_straights():
    """Spa 2 个长直道 (Kemmel + Stavelot)."""
    plan = optimal_plan_for_track("spa")
    assert plan.n_activations == 2
    assert plan.total_gain_s > 0.4


def test_championship_track_comparison():
    """不同赛道激活收益对比."""
    monza = optimal_plan_for_track("monza").total_gain_s
    monaco = optimal_plan_for_track("monaco").total_gain_s
    assert monza > monaco  # Monza 直道优势


def test_wet_race_disables_x_mode():
    """湿地比赛全程禁用 X-mode."""
    for tid in ["monza", "spa", "baku"]:
        plan = optimal_plan_for_track(tid, wet_conditions=True)
        assert plan.n_activations == 0
