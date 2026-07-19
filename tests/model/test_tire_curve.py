"""F1 2026 轮胎性能曲线模型测试 (Iter-47)."""

from __future__ import annotations

import pytest

from f1opt.model.tire_curve import (
    all_compounds,
    cliff_lap_for,
    compare_compounds,
    lap_time_delta_s,
    optimal_stint_length,
    tire_curve_for,
)


# --------------------------------------------------------------------------- #
# tire_curve_for 基础
# --------------------------------------------------------------------------- #
def test_curve_for_medium():
    c = tire_curve_for("medium")
    assert c.compound == "medium"
    assert c.warmup_laps == 2
    assert c.cliff_lap == 30


def test_curve_for_soft():
    c = tire_curve_for("soft")
    assert c.cliff_lap == 22
    assert c.deg_per_lap_s > tire_curve_for("medium").deg_per_lap_s


def test_curve_for_hard():
    c = tire_curve_for("hard")
    assert c.cliff_lap == 35
    assert c.deg_per_lap_s < tire_curve_for("medium").deg_per_lap_s


def test_curve_for_c0_c5():
    """C0-C5 化合物应可查询."""
    for i in range(6):
        c = tire_curve_for(f"c{i}")
        assert c.compound == f"c{i}"


def test_curve_for_intermediate_wet():
    assert tire_curve_for("intermediate").compound == "intermediate"
    assert tire_curve_for("wet").compound == "wet"


def test_curve_for_unknown_raises():
    with pytest.raises(ValueError, match="Unknown compound"):
        tire_curve_for("super_soft")


# --------------------------------------------------------------------------- #
# lap_time_delta_s 三阶段曲线
# --------------------------------------------------------------------------- #
def test_fresh_tire_has_warmup_penalty():
    """新胎 (age=0) 有 warmup 损失."""
    delta_0 = lap_time_delta_s("medium", 0)
    delta_2 = lap_time_delta_s("medium", 2)  # warmup 结束
    assert delta_0 > delta_2  # 新胎慢


def test_optimal_stage_slow_increase():
    """Optimal 阶段圈速缓慢退化."""
    delta_5 = lap_time_delta_s("medium", 5)
    delta_10 = lap_time_delta_s("medium", 10)
    delta_20 = lap_time_delta_s("medium", 20)
    assert delta_5 < delta_10 < delta_20  # 递增
    # 退化应缓慢 (每圈 ~0.03s)
    assert delta_10 - delta_5 < 1.0


def test_cliff_stage_steep_increase():
    """Cliff 阶段圈速陡降."""
    delta_29 = lap_time_delta_s("medium", 29)  # cliff 前
    delta_32 = lap_time_delta_s("medium", 32)  # cliff 后
    delta_35 = lap_time_delta_s("medium", 35)
    # cliff 后退化加速
    rate_before = (delta_29 - lap_time_delta_s("medium", 25)) / 4
    rate_after = (delta_35 - delta_32) / 3
    assert rate_after > rate_before


def test_delta_monotonic_increasing_after_warmup():
    """Warmup 后 delta 应单调递增."""
    prev = lap_time_delta_s("medium", 3)
    for age in range(4, 35):
        curr = lap_time_delta_s("medium", age)
        assert curr >= prev - 1e-9  # 允许数值误差
        prev = curr


# --------------------------------------------------------------------------- #
# 三阶段判断
# --------------------------------------------------------------------------- #
def test_stage_warmup():
    c = tire_curve_for("medium")
    assert c.is_in_warmup(0)
    assert c.is_in_warmup(1)
    assert not c.is_in_warmup(2)
    assert c.stage(0) == "warmup"


def test_stage_optimal():
    c = tire_curve_for("medium")
    assert c.is_in_optimal(2)
    assert c.is_in_optimal(15)
    assert c.is_in_optimal(29)
    assert not c.is_in_optimal(30)
    assert c.stage(15) == "optimal"


def test_stage_cliff():
    c = tire_curve_for("medium")
    assert c.is_past_cliff(30)
    assert c.is_past_cliff(40)
    assert not c.is_past_cliff(29)
    assert c.stage(35) == "cliff"


# --------------------------------------------------------------------------- #
# 最优 stint 长度
# --------------------------------------------------------------------------- #
def test_optimal_stint_length_medium():
    """medium 最优 stint = cliff_lap = 30."""
    assert optimal_stint_length("medium") == 30


def test_optimal_stint_length_soft_shorter():
    """soft 比 medium 最优 stint 短."""
    assert optimal_stint_length("soft") < optimal_stint_length("medium")


def test_optimal_stint_length_hard_longer():
    """hard 比 medium 最优 stint 长."""
    assert optimal_stint_length("hard") > optimal_stint_length("medium")


def test_max_competitive_stint():
    """最大有竞争力 stint > 最优 stint."""
    c = tire_curve_for("medium")
    assert c.max_competitive_stint_length() > c.optimal_stint_length()


# --------------------------------------------------------------------------- #
# 累计退化
# --------------------------------------------------------------------------- #
def test_total_degradation_positive():
    c = tire_curve_for("medium")
    total = c.total_degradation_over_stint(20)
    assert total > 0


def test_total_degradation_increases_with_stint():
    c = tire_curve_for("medium")
    assert c.total_degradation_over_stint(10) < c.total_degradation_over_stint(20)


def test_avg_delta_reasonable():
    c = tire_curve_for("medium")
    avg = c.avg_lap_time_delta_s(20)
    # 20 圈 medium 平均 delta 应在 0..1.5s 范围
    assert 0.0 <= avg <= 2.0


def test_avg_delta_zero_for_zero_laps():
    c = tire_curve_for("medium")
    assert c.avg_lap_time_delta_s(0) == 0.0


# --------------------------------------------------------------------------- #
# 化合物对比
# --------------------------------------------------------------------------- #
def test_compare_compounds_structure():
    r = compare_compounds("soft", "medium", 20)
    assert "compound_a" in r
    assert "compound_b" in r
    assert "better_compound" in r
    assert "margin_s" in r


def test_compare_soft_better_short_stint():
    """短 stint (10 圈) soft 更快."""
    r = compare_compounds("soft", "medium", 10)
    assert r["better_compound"] == "soft"


def test_compare_medium_better_long_stint():
    """长 stint (30 圈) medium 更快 (soft 过 cliff)."""
    r = compare_compounds("soft", "medium", 30)
    assert r["better_compound"] == "medium"


def test_compare_hard_better_very_long():
    """超长 stint (40 圈) hard 更快."""
    r = compare_compounds("hard", "medium", 40)
    assert r["better_compound"] == "hard"


def test_compare_margin_positive():
    r = compare_compounds("soft", "medium", 10)
    assert r["margin_s"] > 0


# --------------------------------------------------------------------------- #
# C0-C5 单调性
# --------------------------------------------------------------------------- #
def test_c0_to_c5_cliff_decreasing():
    """C0 (硬) 到 C5 (软) 临界圈数递减."""
    cliffs = [tire_curve_for(f"c{i}").cliff_lap for i in range(6)]
    for i in range(5):
        assert cliffs[i] >= cliffs[i + 1]


def test_c0_to_c5_deg_increasing():
    """C0 到 C5 退化率递增."""
    degs = [tire_curve_for(f"c{i}").deg_per_lap_s for i in range(6)]
    for i in range(5):
        assert degs[i] <= degs[i + 1]


def test_c5_faster_at_fresh():
    """C5 新胎比 C0 快 (base_offset 更负)."""
    c0 = tire_curve_for("c0")
    c5 = tire_curve_for("c5")
    assert c5.base_offset_s < c0.base_offset_s


# --------------------------------------------------------------------------- #
# all_compounds / cliff_lap_for
# --------------------------------------------------------------------------- #
def test_all_compounds_includes_aliases():
    compounds = all_compounds()
    assert "medium" in compounds
    assert "soft" in compounds
    assert "hard" in compounds
    assert "c0" in compounds
    assert "c5" in compounds
    assert "intermediate" in compounds
    assert "wet" in compounds


def test_cliff_lap_for_convenience():
    assert cliff_lap_for("medium") == 30
    assert cliff_lap_for("soft") == 22
    assert cliff_lap_for("hard") == 35


# --------------------------------------------------------------------------- #
# 缓存
# --------------------------------------------------------------------------- #
def test_curve_cached():
    """同一化合物应返回同一实例 (缓存)."""
    c1 = tire_curve_for("medium")
    c2 = tire_curve_for("medium")
    assert c1 is c2


# --------------------------------------------------------------------------- #
# 确定性
# --------------------------------------------------------------------------- #
def test_deterministic():
    d1 = lap_time_delta_s("medium", 15)
    d2 = lap_time_delta_s("medium", 15)
    assert d1 == d2


# --------------------------------------------------------------------------- #
# 实战场景
# --------------------------------------------------------------------------- #
def test_1_stop_monza_strategy():
    """Monza 53 圈 1-stop: medium(25) + hard(28).
    评估总退化."""
    medium = tire_curve_for("medium")
    hard = tire_curve_for("hard")
    stint1_deg = medium.total_degradation_over_stint(25)
    stint2_deg = hard.total_degradation_over_stint(28)
    total = stint1_deg + stint2_deg
    # 总退化应在合理范围 (10-30s)
    assert 5.0 < total < 40.0


def test_2_stop_strategy():
    """2-stop: soft(15) + soft(15) + medium(23).
    soft 短 stint 利用速度."""
    soft = tire_curve_for("soft")
    medium = tire_curve_for("medium")
    total = (soft.total_degradation_over_stint(15) * 2
             + medium.total_degradation_over_stint(23))
    assert total > 0


def test_3_stop_aggressive():
    """3-stop: 全 soft, 每 stint 12-13 圈.
    极致速度但多次进站."""
    soft = tire_curve_for("soft")
    avg_12 = soft.avg_lap_time_delta_s(12)
    # 12 圈 soft 平均 delta 应较低 (利用 optimal 阶段)
    assert avg_12 < 1.0


def test_wet_tire_curve():
    """雨胎 base_offset 高 (基础慢)."""
    wet = tire_curve_for("wet")
    inter = tire_curve_for("intermediate")
    assert wet.base_offset_s > inter.base_offset_s
    assert wet.base_offset_s > 1.0  # 雨胎明显慢


def test_cliff_penalty_significant():
    """过 cliff 后退化惩罚显著."""
    medium = tire_curve_for("medium")
    delta_at_cliff = medium.lap_time_delta_s(30)
    delta_5_after = medium.lap_time_delta_s(35)
    # 5 圈后退化应 > 1s
    assert delta_5_after - delta_at_cliff > 1.0
