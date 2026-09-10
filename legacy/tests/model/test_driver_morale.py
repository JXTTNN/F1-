"""F1 2026 车手信心与车队士气模型测试 (Iter-52)."""

from __future__ import annotations

from f1opt.model.driver_morale import (
    ConfidenceEvent,
    DriverConfidence,
    MoraleEvent,
    TeamMorale,
    apply_qualifying_result,
    apply_race_result,
    confidence_lap_time_delta_s,
    morale_rd_efficiency_factor,
)


# --------------------------------------------------------------------------- #
# DriverConfidence 基础
# --------------------------------------------------------------------------- #
def test_confidence_default():
    c = DriverConfidence(driver_id="ver")
    assert c.value == 50.0
    assert c.level == "medium"


def test_confidence_clamped_high():
    c = DriverConfidence(driver_id="ver", value=150)
    assert c.value == 100.0


def test_confidence_clamped_low():
    c = DriverConfidence(driver_id="ver", value=-10)
    assert c.value == 0.0


def test_confidence_levels():
    assert DriverConfidence("a", 10).level == "low"
    assert DriverConfidence("a", 50).level == "medium"
    assert DriverConfidence("a", 60).level == "high"
    assert DriverConfidence("a", 90).level == "very_high"


# --------------------------------------------------------------------------- #
# apply_event
# --------------------------------------------------------------------------- #
def test_race_win_boosts_confidence():
    c = DriverConfidence("ver", 50)
    delta = c.apply_event(ConfidenceEvent.RACE_WIN)
    assert delta == 8.0
    assert c.value == 58.0


def test_retirement_lowers_confidence():
    c = DriverConfidence("ver", 50)
    delta = c.apply_event(ConfidenceEvent.RETIREMENT)
    assert delta == -6.0
    assert c.value == 44.0


def test_confidence_clamped_after_event():
    c = DriverConfidence("ver", 95)
    c.apply_event(ConfidenceEvent.RACE_WIN)  # +8
    assert c.value == 100.0  # 不超 100


def test_confidence_clamped_low_after_event():
    c = DriverConfidence("ver", 3)
    c.apply_event(ConfidenceEvent.RETIREMENT)  # -6
    assert c.value == 0.0


def test_history_recorded():
    c = DriverConfidence("ver", 50)
    c.apply_event(ConfidenceEvent.RACE_WIN)
    c.apply_event(ConfidenceEvent.PODIUM)
    assert len(c.history) == 2
    assert c.history[0][0] == "race_win"


# --------------------------------------------------------------------------- #
# decay_toward_baseline
# --------------------------------------------------------------------------- #
def test_decay_toward_baseline_high():
    c = DriverConfidence("ver", 80)
    c.decay_toward_baseline(50, 0.1)
    # 80 + (50-80)*0.1 = 77
    assert abs(c.value - 77) < 1e-9


def test_decay_toward_baseline_low():
    c = DriverConfidence("ver", 30)
    c.decay_toward_baseline(50, 0.1)
    # 30 + (50-30)*0.1 = 32
    assert abs(c.value - 32) < 1e-9


def test_decay_at_baseline_no_change():
    c = DriverConfidence("ver", 50)
    c.decay_toward_baseline(50, 0.1)
    assert c.value == 50.0


# --------------------------------------------------------------------------- #
# confidence_lap_time_delta_s
# --------------------------------------------------------------------------- #
def test_delta_at_50_is_zero():
    assert confidence_lap_time_delta_s(50) == 0.0


def test_delta_at_100_fastest():
    assert confidence_lap_time_delta_s(100) < 0
    assert abs(confidence_lap_time_delta_s(100) - (-0.30)) < 1e-9


def test_delta_at_0_slowest():
    assert confidence_lap_time_delta_s(0) > 0
    assert abs(confidence_lap_time_delta_s(0) - 0.40) < 1e-9


def test_delta_monotonic_decreasing():
    """信心越高, delta 越小 (越快)."""
    deltas = [confidence_lap_time_delta_s(c) for c in range(0, 101, 10)]
    for i in range(len(deltas) - 1):
        assert deltas[i] >= deltas[i + 1]


def test_delta_clamped():
    assert confidence_lap_time_delta_s(150) == confidence_lap_time_delta_s(100)
    assert confidence_lap_time_delta_s(-10) == confidence_lap_time_delta_s(0)


def test_confidence_lap_time_property():
    c = DriverConfidence("ver", 80)
    assert c.lap_time_delta_s < 0  # 高信心更快


# --------------------------------------------------------------------------- #
# TeamMorale
# --------------------------------------------------------------------------- #
def test_morale_default():
    m = TeamMorale(team_id="rbr")
    assert m.value == 60.0
    assert m.level == "good"


def test_morale_levels():
    assert TeamMorale("a", 10).level == "critical"
    assert TeamMorale("a", 30).level == "low"
    assert TeamMorale("a", 60).level == "good"
    assert TeamMorale("a", 90).level == "excellent"


def test_morale_clamped():
    assert TeamMorale("a", 150).value == 100.0
    assert TeamMorale("a", -10).value == 0.0


def test_morale_event_win():
    m = TeamMorale("rbr", 60)
    delta = m.apply_event(MoraleEvent.RACE_WIN)
    assert delta == 6.0
    assert m.value == 66.0


def test_morale_double_dnf_huge_penalty():
    m = TeamMorale("rbr", 60)
    m.apply_event(MoraleEvent.DOUBLE_DNF)
    assert m.value == 52.0  # -8


def test_morale_teammate_conflict():
    m = TeamMorale("rbr", 60)
    m.apply_event(MoraleEvent.TEAMMATE_CONFLICT)
    assert m.value == 55.0  # -5


# --------------------------------------------------------------------------- #
# morale_rd_efficiency_factor
# --------------------------------------------------------------------------- #
def test_rd_efficiency_at_60():
    assert morale_rd_efficiency_factor(60) == 1.0


def test_rd_efficiency_at_100():
    assert abs(morale_rd_efficiency_factor(100) - 1.20) < 1e-9


def test_rd_efficiency_at_0():
    assert abs(morale_rd_efficiency_factor(0) - 0.70) < 1e-9


def test_rd_efficiency_monotonic():
    factors = [morale_rd_efficiency_factor(m) for m in range(0, 101, 10)]
    for i in range(len(factors) - 1):
        assert factors[i] <= factors[i + 1]


def test_morale_rd_efficiency_property():
    m = TeamMorale("rbr", 90)
    assert m.rd_efficiency_factor > 1.0


def test_morale_reliability_factor():
    m_high = TeamMorale("rbr", 80)
    m_low = TeamMorale("has", 20)
    assert m_high.reliability_factor == 1.0
    assert m_low.reliability_factor < 1.0


# --------------------------------------------------------------------------- #
# apply_race_result
# --------------------------------------------------------------------------- #
def test_race_result_win():
    c = DriverConfidence("ver", 50)
    m = TeamMorale("rbr", 60)
    apply_race_result(c, m, position=1)
    assert c.value == 58.0  # +8
    assert m.value == 66.0  # +6


def test_race_result_podium():
    c = DriverConfidence("ver", 50)
    m = TeamMorale("rbr", 60)
    apply_race_result(c, m, position=3)
    assert c.value == 55.0  # +5
    assert m.value == 64.0  # +4


def test_race_result_points():
    c = DriverConfidence("ver", 50)
    m = TeamMorale("rbr", 60)
    apply_race_result(c, m, position=7)
    assert c.value == 52.0  # +2
    assert m.value == 62.0  # +2


def test_race_result_no_points():
    c = DriverConfidence("ver", 50)
    m = TeamMorale("rbr", 60)
    apply_race_result(c, m, position=15)
    assert c.value == 50.0  # 无变化
    assert m.value == 60.0


def test_race_result_retired():
    c = DriverConfidence("ver", 50)
    m = TeamMorale("rbr", 60)
    apply_race_result(c, m, position=20, retired=True)
    assert c.value == 44.0  # -6
    assert m.value == 56.0  # -4


# --------------------------------------------------------------------------- #
# apply_qualifying_result
# --------------------------------------------------------------------------- #
def test_qualifying_good():
    c = DriverConfidence("ver", 50)
    apply_qualifying_result(c, qualifying_position=3, total_drivers=20)
    assert c.value > 50.0  # 上升


def test_qualifying_bad():
    c = DriverConfidence("ver", 50)
    apply_qualifying_result(c, qualifying_position=18, total_drivers=20)
    assert c.value < 50.0  # 下降


def test_qualifying_middle_no_change():
    c = DriverConfidence("ver", 50)
    apply_qualifying_result(c, qualifying_position=10, total_drivers=20)
    assert c.value == 50.0  # 中间不变


# --------------------------------------------------------------------------- #
# 实战场景
# --------------------------------------------------------------------------- #
def test_season_confidence_progression():
    """赛季中信心随成绩波动."""
    c = DriverConfidence("ver", 60)
    # 模拟 10 场比赛
    results = [1, 2, 1, 5, 1, 3, 1, 2, 1, 1]  # 多胜
    for pos in results:
        apply_race_result(c, TeamMorale("rbr"), position=pos)
        c.decay_toward_baseline(60, 0.05)
    # 多次胜利, 信心应高
    assert c.value > 70


def test_bad_streak_lowers_confidence():
    """连败退赛信心大降."""
    c = DriverConfidence("ver", 60)
    for _ in range(5):
        apply_race_result(c, TeamMorale("rbr"), position=20, retired=True)
    assert c.value <= 30  # 5×-6 = -30, 60-30=30


def test_morale_affects_rd_efficiency():
    """士气影响 R&D 效率."""
    m_good = TeamMorale("rbr", 90)
    m_bad = TeamMorale("has", 20)
    assert m_good.rd_efficiency_factor > m_bad.rd_efficiency_factor


def test_double_dnf_devastating():
    """双退对士气打击最大."""
    m = TeamMorale("rbr", 80)
    m.apply_event(MoraleEvent.DOUBLE_DNF)
    assert m.value == 72.0  # -8
    # 但单退只 -4
    m2 = TeamMorale("rbr", 80)
    m2.apply_event(MoraleEvent.RETIREMENT)
    assert m2.value == 76.0
