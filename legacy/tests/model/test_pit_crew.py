"""F1 2026 Pit Crew Performance 测试 (Iter-39)."""

from __future__ import annotations

import pytest

from f1opt.model.pit_crew import (
    all_pit_crew_ratings,
    effective_pit_loss_s,
    expected_pit_stop_time_s,
    pit_crew_ranking,
    pit_crew_rating,
    pit_stop_time_s,
)


# --------------------------------------------------------------------------- #
# 评分
# --------------------------------------------------------------------------- #
def test_all_11_teams_have_ratings():
    ratings = all_pit_crew_ratings()
    assert len(ratings) == 11


def test_red_bull_top_rating():
    """Red Bull 应该有顶尖 pit crew (历史最快进站)."""
    assert pit_crew_rating("rbr") >= 95


def test_haas_among_lowest():
    """Haas 应该是后段."""
    assert pit_crew_rating("has") <= 80


def test_ratings_in_valid_range():
    for tid, rating in all_pit_crew_ratings().items():
        assert 70 <= rating <= 99, f"{tid}: {rating}"


def test_unknown_team_raises():
    with pytest.raises(ValueError):
        pit_crew_rating("nonexistent")


def test_ranking_returns_sorted_desc():
    ranking = pit_crew_ranking()
    ratings = [r for _, r in ranking]
    assert ratings == sorted(ratings, reverse=True)
    assert ranking[0][0] == "rbr"  # RBR 应排第一


# --------------------------------------------------------------------------- #
# 进站时间
# --------------------------------------------------------------------------- #
def test_red_bull_fastest_pit_stop():
    """RBR 平均进站应快于后段."""
    rbr_avg = sum(pit_stop_time_s("rbr", seed=s) for s in range(50)) / 50
    has_avg = sum(pit_stop_time_s("has", seed=s) for s in range(50)) / 50
    assert rbr_avg < has_avg
    assert rbr_avg < 2.5  # RBR 平均应在 2.5s 内


def test_pit_stop_time_in_reasonable_range():
    """进站停车换胎时间应在 1.5-9.0s (含慢停最大)."""
    for tid in ["rbr", "mer", "fer", "has", "aud", "cad"]:
        for seed in range(50):
            t = pit_stop_time_s(tid, seed=seed)
            assert 1.5 <= t <= 9.0, f"{tid} seed={seed}: {t}"


def test_pit_stop_at_least_1_5s():
    """物理下限: 进站停车换胎 ≥ 1.5s."""
    for tid in all_pit_crew_ratings():
        for seed in range(20):
            assert pit_stop_time_s(tid, seed=seed) >= 1.5


def test_slow_stops_can_occur():
    """慢停 (>4s) 应偶尔发生在后段车队."""
    n_slow = 0
    for seed in range(200):
        t = pit_stop_time_s("has", seed=seed)
        if t > 4.0:
            n_slow += 1
    assert n_slow > 0  # 后段应有慢停


def test_perfect_stops_only_for_top_teams():
    """完美进站 (<1.8s) 应只发生在 rating>=90 的车队."""
    # RBR 应有完美进站
    n_perfect_rbr = 0
    for seed in range(200):
        if pit_stop_time_s("rbr", seed=seed) < 1.8:
            n_perfect_rbr += 1
    assert n_perfect_rbr > 0

    # Haas 不应有完美进站 (rating 76 < 90)
    n_perfect_has = 0
    for seed in range(200):
        if pit_stop_time_s("has", seed=seed) < 1.8:
            n_perfect_has += 1
    assert n_perfect_has == 0


def test_reproducible_with_seed():
    t1 = pit_stop_time_s("rbr", seed=42)
    t2 = pit_stop_time_s("rbr", seed=42)
    assert t1 == t2


# --------------------------------------------------------------------------- #
# 期望进站时间 (无随机性)
# --------------------------------------------------------------------------- #
def test_expected_pit_stop_time_in_range():
    for tid in all_pit_crew_ratings():
        t = expected_pit_stop_time_s(tid)
        assert 1.8 <= t <= 5.0


def test_expected_time_correlates_with_rating():
    """评分越高, 期望时间越短."""
    rbr_t = expected_pit_stop_time_s("rbr")  # rating 96
    has_t = expected_pit_stop_time_s("has")  # rating 76
    assert rbr_t < has_t


def test_expected_close_to_average():
    """期望时间应接近多次仿真的平均值."""
    for tid in ["rbr", "mer", "has"]:
        expected = expected_pit_stop_time_s(tid)
        actual_avg = sum(pit_stop_time_s(tid, seed=s) for s in range(500)) / 500
        # 期望值应在实际平均值 ±0.5s 内
        assert abs(expected - actual_avg) < 0.5, \
            f"{tid}: expected={expected:.2f}, actual_avg={actual_avg:.2f}"


# --------------------------------------------------------------------------- #
# 有效进站总损失
# --------------------------------------------------------------------------- #
def test_effective_pit_loss_for_monza():
    """Monza 进站总损失 = 维修区 (~23s) + 停车换胎."""
    loss = effective_pit_loss_s("rbr", "monza")
    # Monza pit_loss_s ≈ 23s + RBR 停车 ~2s = ~25s
    assert 24.0 <= loss <= 28.0


def test_effective_pit_loss_team_difference():
    """RBR 有效进站损失应 < Haas."""
    rbr_loss = effective_pit_loss_s("rbr", "monza")
    has_loss = effective_pit_loss_s("has", "monza")
    assert rbr_loss < has_loss


def test_effective_pit_loss_unknown_track_uses_default():
    """未知赛道应 fallback 到默认维修区损失."""
    loss = effective_pit_loss_s("rbr", "nonexistent_track")
    # 默认 ~23 + RBR ~2 = ~25
    assert 24.0 <= loss <= 28.0


# --------------------------------------------------------------------------- #
# 策略影响
# --------------------------------------------------------------------------- #
def test_slow_team_loses_more_per_stop():
    """慢队每次进站比快队多损失的时间."""
    rbr_t = expected_pit_stop_time_s("rbr")
    has_t = expected_pit_stop_time_s("has")
    diff = has_t - rbr_t
    # Haas 比 RBR 每次进站多损失约 1-2s
    assert 0.5 <= diff <= 3.0


def test_2stop_strategy_costs_more_for_slow_team():
    """2-stop 策略对慢队的额外成本 (vs 快队)."""
    rbr_per_stop = expected_pit_stop_time_s("rbr")
    has_per_stop = expected_pit_stop_time_s("has")
    # 2-stop 多 2 次进站, 每次多损失 diff
    diff_per_stop = has_per_stop - rbr_per_stop
    total_extra_for_2stop = 2 * diff_per_stop
    # 2-stop 对慢队额外 1-4s 总损失
    assert 1.0 <= total_extra_for_2stop <= 6.0
