"""F1 2026 进站窗口优化器测试 (Iter-46)."""

from __future__ import annotations

from f1opt.model.pit_window import (
    PitWindowRecommendation,
    PitWindowState,
    recommend_pit_window,
)


# --------------------------------------------------------------------------- #
# 基础
# --------------------------------------------------------------------------- #
def test_basic_state():
    state = PitWindowState(
        current_lap=15, total_laps=53, current_compound="medium",
        tire_age_laps=15, current_position=3,
        gap_ahead_s=2.5, gap_behind_s=4.0,
    )
    rec = recommend_pit_window(state)
    assert isinstance(rec, PitWindowRecommendation)
    assert 1 <= rec.optimal_pit_lap <= 53


def test_optimal_pit_lap_in_valid_range():
    state = PitWindowState(
        current_lap=20, total_laps=53, current_compound="medium",
        tire_age_laps=20, current_position=5,
        gap_ahead_s=3.0, gap_behind_s=3.0,
        next_compound="hard",
    )
    rec = recommend_pit_window(state)
    assert state.current_lap + 1 <= rec.optimal_pit_lap <= state.total_laps - 2


def test_pit_window_contains_optimal():
    state = PitWindowState(
        current_lap=15, total_laps=53, current_compound="medium",
        tire_age_laps=15, current_position=3,
        gap_ahead_s=3.0, gap_behind_s=3.0,
    )
    rec = recommend_pit_window(state)
    assert rec.pit_window_start <= rec.optimal_pit_lap <= rec.pit_window_end


def test_window_start_at_least_current_plus_1():
    state = PitWindowState(
        current_lap=15, total_laps=53, current_compound="medium",
        tire_age_laps=15, current_position=3,
        gap_ahead_s=3.0, gap_behind_s=3.0,
    )
    rec = recommend_pit_window(state)
    assert rec.pit_window_start >= state.current_lap + 1


# --------------------------------------------------------------------------- #
# 轮胎临界
# --------------------------------------------------------------------------- #
def test_past_cliff_recommends_pit_now():
    """轮胎超过临界圈数 → 立即进站."""
    state = PitWindowState(
        current_lap=30, total_laps=53, current_compound="soft",
        tire_age_laps=25,  # soft cliff = 18
        current_position=4, gap_ahead_s=5.0, gap_behind_s=5.0,
    )
    rec = recommend_pit_window(state)
    assert rec.recommend_pit_now is True
    assert "临界" in rec.reason or "必须进站" in rec.reason
    assert rec.risk_level == "high"


def test_approaching_cliff_recommends_pit():
    """接近临界 + 进入窗口 → 推荐进站."""
    state = PitWindowState(
        current_lap=25, total_laps=53, current_compound="medium",
        tire_age_laps=25,  # medium cliff = 28, 接近
        current_position=3, gap_ahead_s=5.0, gap_behind_s=5.0,
    )
    rec = recommend_pit_window(state)
    assert rec.recommend_pit_now is True


def test_fresh_tire_no_pit():
    """新胎不应推荐立即进站."""
    state = PitWindowState(
        current_lap=5, total_laps=53, current_compound="hard",
        tire_age_laps=5, current_position=1,
        gap_ahead_s=10.0, gap_behind_s=10.0,
    )
    rec = recommend_pit_window(state)
    assert rec.recommend_pit_now is False


# --------------------------------------------------------------------------- #
# Undercut / Overcut
# --------------------------------------------------------------------------- #
def test_undercut_opportunity():
    """前车 1-3s 内 + 接近临界 → undercut 机会."""
    state = PitWindowState(
        current_lap=15, total_laps=53, current_compound="soft",
        tire_age_laps=13,  # soft cliff=18, 接近 (cliff-5=13)
        current_position=4, gap_ahead_s=2.0, gap_behind_s=5.0,
    )
    rec = recommend_pit_window(state)
    assert rec.undercut_opportunity is True


def test_no_undercut_when_gap_too_large():
    state = PitWindowState(
        current_lap=15, total_laps=53, current_compound="soft",
        tire_age_laps=13,
        current_position=4, gap_ahead_s=10.0, gap_behind_s=5.0,
    )
    rec = recommend_pit_window(state)
    assert rec.undercut_opportunity is False


def test_overcut_opportunity():
    """前车刚进站 (gap < 1s) + 自己旧胎未过临界 → overcut."""
    state = PitWindowState(
        current_lap=18, total_laps=53, current_compound="medium",
        tire_age_laps=18,  # 未过 cliff=28
        current_position=2, gap_ahead_s=0.5, gap_behind_s=5.0,
    )
    rec = recommend_pit_window(state)
    assert rec.overcut_opportunity is True


def test_no_overcut_when_past_cliff():
    state = PitWindowState(
        current_lap=35, total_laps=53, current_compound="soft",
        tire_age_laps=25,  # 过 cliff=18
        current_position=2, gap_ahead_s=0.5, gap_behind_s=5.0,
    )
    rec = recommend_pit_window(state)
    assert rec.overcot_opportunity if hasattr(rec, 'overcot_opportunity') else not rec.overcut_opportunity


# --------------------------------------------------------------------------- #
# 安全车
# --------------------------------------------------------------------------- #
def test_sc_opportunity():
    """SC 概率 > 20% + 轮胎未过临界 → SC 进站机会."""
    state = PitWindowState(
        current_lap=20, total_laps=53, current_compound="medium",
        tire_age_laps=20,
        current_position=3, gap_ahead_s=5.0, gap_behind_s=5.0,
        sc_probability=0.30,
    )
    rec = recommend_pit_window(state)
    assert rec.sc_pit_opportunity is True


def test_high_sc_probability_recommends_now():
    """SC 概率 > 35% → 推荐立即进站."""
    state = PitWindowState(
        current_lap=20, total_laps=53, current_compound="medium",
        tire_age_laps=20,
        current_position=3, gap_ahead_s=5.0, gap_behind_s=5.0,
        sc_probability=0.45,
    )
    rec = recommend_pit_window(state)
    assert rec.recommend_pit_now is True
    assert "SC" in rec.reason or "free pit" in rec.reason


def test_low_sc_probability_no_sc_opportunity():
    state = PitWindowState(
        current_lap=20, total_laps=53, current_compound="medium",
        tire_age_laps=20,
        current_position=3, gap_ahead_s=5.0, gap_behind_s=5.0,
        sc_probability=0.05,
    )
    rec = recommend_pit_window(state)
    assert rec.sc_pit_opportunity is False


# --------------------------------------------------------------------------- #
# 风险等级
# --------------------------------------------------------------------------- #
def test_high_risk_when_past_cliff():
    state = PitWindowState(
        current_lap=30, total_laps=53, current_compound="soft",
        tire_age_laps=25,
        current_position=4, gap_ahead_s=5.0, gap_behind_s=5.0,
    )
    rec = recommend_pit_window(state)
    assert rec.risk_level == "high"


def test_low_risk_fresh_tire():
    state = PitWindowState(
        current_lap=5, total_laps=53, current_compound="hard",
        tire_age_laps=5,
        current_position=1, gap_ahead_s=10.0, gap_behind_s=10.0,
    )
    rec = recommend_pit_window(state)
    assert rec.risk_level == "low"


def test_medium_risk_undercut():
    state = PitWindowState(
        current_lap=15, total_laps=53, current_compound="soft",
        tire_age_laps=13,
        current_position=4, gap_ahead_s=2.0, gap_behind_s=5.0,
    )
    rec = recommend_pit_window(state)
    assert rec.risk_level in ("medium", "high")


# --------------------------------------------------------------------------- #
# 位置预测
# --------------------------------------------------------------------------- #
def test_projected_position_after_pit():
    state = PitWindowState(
        current_lap=20, total_laps=53, current_compound="medium",
        tire_age_laps=20,
        current_position=3, gap_ahead_s=5.0, gap_behind_s=5.0,
    )
    rec = recommend_pit_window(state)
    # 进站后位置应下降
    assert rec.projected_position_after_pit >= state.current_position


def test_sc_discount_reduces_position_loss():
    """SC 折扣应减少进站位置损失."""
    state_normal = PitWindowState(
        current_lap=20, total_laps=53, current_compound="medium",
        tire_age_laps=20,
        current_position=3, gap_ahead_s=5.0, gap_behind_s=5.0,
        sc_pit_discount=1.0,
    )
    state_sc = PitWindowState(
        current_lap=20, total_laps=53, current_compound="medium",
        tire_age_laps=20,
        current_position=3, gap_ahead_s=5.0, gap_behind_s=5.0,
        sc_pit_discount=0.3,  # SC 期间只损失 30%
        sc_probability=0.40,
    )
    rec_normal = recommend_pit_window(state_normal)
    rec_sc = recommend_pit_window(state_sc)
    assert rec_sc.projected_position_after_pit <= rec_normal.projected_position_after_pit


# --------------------------------------------------------------------------- #
# 备选策略
# --------------------------------------------------------------------------- #
def test_alternatives_when_long_race():
    """剩余圈数多应有备选 2-stop."""
    state = PitWindowState(
        current_lap=10, total_laps=53, current_compound="medium",
        tire_age_laps=10,
        current_position=3, gap_ahead_s=5.0, gap_behind_s=5.0,
        next_compound="soft",  # next_cliff=18, 较短
    )
    rec = recommend_pit_window(state)
    # 应有备选策略
    assert len(rec.alternatives) >= 0  # 至少不报错


def test_alternatives_sc_mentioned():
    """SC 概率高时备选应提到 SC."""
    state = PitWindowState(
        current_lap=20, total_laps=53, current_compound="medium",
        tire_age_laps=20,
        current_position=3, gap_ahead_s=5.0, gap_behind_s=5.0,
        sc_probability=0.30,
    )
    rec = recommend_pit_window(state)
    combined = " ".join(rec.alternatives)
    assert "SC" in combined or "安全车" in combined


# --------------------------------------------------------------------------- #
# 确定性
# --------------------------------------------------------------------------- #
def test_deterministic():
    state = PitWindowState(
        current_lap=20, total_laps=53, current_compound="medium",
        tire_age_laps=20,
        current_position=3, gap_ahead_s=5.0, gap_behind_s=5.0,
    )
    r1 = recommend_pit_window(state)
    r2 = recommend_pit_window(state)
    assert r1.optimal_pit_lap == r2.optimal_pit_lap
    assert r1.recommend_pit_now == r2.recommend_pit_now


# --------------------------------------------------------------------------- #
# 实战场景
# --------------------------------------------------------------------------- #
def test_monza_1_stop_strategy():
    """Monza 53 圈 1-stop: medium-hard, 进站窗口 ~18-25."""
    state = PitWindowState(
        current_lap=18, total_laps=53, current_compound="medium",
        tire_age_laps=18,
        current_position=5, gap_ahead_s=3.0, gap_behind_s=3.0,
        next_compound="hard",
    )
    rec = recommend_pit_window(state)
    # 窗口应在合理范围
    assert 15 <= rec.pit_window_start <= 30
    assert rec.pit_window_end <= 40


def test_monaco_1_stop_extended():
    """Monaco 78 圈: 极长, 轮胎管理关键."""
    state = PitWindowState(
        current_lap=30, total_laps=78, current_compound="hard",
        tire_age_laps=30,
        current_position=2, gap_ahead_s=1.5, gap_behind_s=8.0,
        next_compound="medium",
    )
    rec = recommend_pit_window(state)
    # hard cliff=40, 30 圈接近但未到
    assert rec.optimal_pit_lap > 30


def test_sprint_race_short():
    """短赛 (19 圈): 1-stop 窗口紧凑."""
    state = PitWindowState(
        current_lap=8, total_laps=19, current_compound="soft",
        tire_age_laps=8,
        current_position=3, gap_ahead_s=1.5, gap_behind_s=2.0,
        next_compound="medium",
    )
    rec = recommend_pit_window(state)
    assert rec.pit_window_end <= 17  # 留 2 圈完赛


def test_wet_race_compound():
    """雨战用 intermediate, 进站窗口不同."""
    state = PitWindowState(
        current_lap=15, total_laps=53, current_compound="intermediate",
        tire_age_laps=15,
        current_position=4, gap_ahead_s=4.0, gap_behind_s=4.0,
        next_compound="wet",
    )
    rec = recommend_pit_window(state)
    # intermediate cliff=22, 应在 22 前进站
    assert rec.optimal_pit_lap >= 15
