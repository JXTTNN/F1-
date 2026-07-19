"""Iter-64 测试: EA F1 2026 轮胎温度窗口模型 (纯函数)."""

from __future__ import annotations

import timeit

from f1opt.model.tire_temperature import (
    all_compounds,
    is_in_optimal_window,
    temp_state,
    tire_temp_at_lap,
    tire_temp_penalty_s,
    tire_temp_window,
)


# --------------------------------------------------------------------------- #
# 工作窗口查询
# --------------------------------------------------------------------------- #
class TestTempWindow:
    """化合物工作温度窗口查询."""

    def test_soft_window(self):
        low, high = tire_temp_window("soft")
        assert low == 85.0
        assert high == 100.0

    def test_medium_window(self):
        low, high = tire_temp_window("medium")
        assert low == 90.0
        assert high == 110.0

    def test_hard_window_wider_than_soft(self):
        """硬胎窗口比软胎宽 (热稳定)."""
        sl, sh = tire_temp_window("soft")
        hl, hh = tire_temp_window("hard")
        assert (hh - hl) > (sh - sl)

    def test_wet_window_lower_than_dry(self):
        """雨胎窗口温度低于干胎."""
        sl, _ = tire_temp_window("soft")
        wl, wh = tire_temp_window("wet")
        assert wh < sl
        assert wl < sl

    def test_c5_narrowest_window(self):
        """C5 最软, 窗口最窄."""
        c5_low, c5_high = tire_temp_window("c5")
        hard_low, hard_high = tire_temp_window("hard")
        assert (c5_high - c5_low) < (hard_high - hard_low)

    def test_unknown_compound_returns_wide_window(self):
        low, high = tire_temp_window("nonexistent")
        assert low == 0.0
        assert high == 200.0


# --------------------------------------------------------------------------- #
# 轮胎温度估算
# --------------------------------------------------------------------------- #
class TestTireTempEstimation:
    """轮胎温度估算物理."""

    def test_base_temp_at_warm_lap(self):
        """已暖胎 (lap=3+), 干地, 30°C 赛道 → 90°C 基线 (medium 窗口下界)."""
        t = tire_temp_at_lap("medium", 30.0, 25.0, 3, 0.0, False)
        assert abs(t - 90.0) < 0.01  # 30 + 60 = 90

    def test_cold_start_lap_0(self):
        """第 0 圈冷启动, 偏移 -25°C."""
        t = tire_temp_at_lap("medium", 30.0, 25.0, 0, 0.0, False)
        # 30 + 60 + (-25) * (1 - 0/3) = 90 - 25 = 65
        assert abs(t - 65.0) < 0.01

    def test_warmup_progresses_over_3_laps(self):
        """暖胎 3 圈渐进升温."""
        t0 = tire_temp_at_lap("medium", 30.0, 25.0, 0, 0.0, False)
        t1 = tire_temp_at_lap("medium", 30.0, 25.0, 1, 0.0, False)
        t2 = tire_temp_at_lap("medium", 30.0, 25.0, 2, 0.0, False)
        t3 = tire_temp_at_lap("medium", 30.0, 25.0, 3, 0.0, False)
        assert t0 < t1 < t2 < t3
        # 第 3 圈已达基线 (90)
        assert abs(t3 - 90.0) < 0.01

    def test_age_heat_accumulation(self):
        """高龄胎热积累 (+0.3°C/lap)."""
        t_young = tire_temp_at_lap("medium", 30.0, 25.0, 10, 5.0, False)
        t_old = tire_temp_at_lap("medium", 30.0, 25.0, 10, 30.0, False)
        # 差 25 圈 * 0.3 = 7.5°C
        assert abs((t_old - t_young) - 7.5) < 0.01

    def test_wet_cools_tire(self):
        """湿地降温 -15°C."""
        t_dry = tire_temp_at_lap("medium", 30.0, 25.0, 5, 0.0, False)
        t_wet = tire_temp_at_lap("medium", 30.0, 25.0, 5, 0.0, True)
        assert abs((t_dry - t_wet) - 15.0) < 0.01

    def test_high_track_temp_raises_tire_temp(self):
        """高赛道温度 → 高胎温."""
        t_cold_track = tire_temp_at_lap("soft", 20.0, 18.0, 5, 0.0, False)
        t_hot_track = tire_temp_at_lap("soft", 45.0, 35.0, 5, 0.0, False)
        assert t_hot_track > t_cold_track


# --------------------------------------------------------------------------- #
# 圈速惩罚
# --------------------------------------------------------------------------- #
class TestTempPenalty:
    """温度偏离窗口的圈速惩罚."""

    def test_no_penalty_in_window(self):
        """工作窗口内无惩罚."""
        # medium 窗口 90-110, 30°C 赛道 + 60 = 90 (lap=3, age=0)
        p = tire_temp_penalty_s("medium", 30.0, 25.0, 3, 0.0, False)
        assert p == 0.0

    def test_cold_start_lap_0_has_penalty(self):
        """冷启动 (lap=0) 在窗口下有惩罚."""
        p = tire_temp_penalty_s("medium", 30.0, 25.0, 0, 0.0, False)
        # temp = 65, opt_min = 90, diff = 25
        # penalty = 25 * 0.025 = 0.625s
        assert p > 0.5
        assert abs(p - 0.625) < 0.001

    def test_hot_old_tire_has_penalty(self):
        """高龄胎在热赛道下过热惩罚."""
        # 40°C 赛道 + 60 = 100, age=50 * 0.3 = 15, temp = 115
        # medium 窗口 90-110, over by 5
        # penalty = 5 * 0.040 = 0.20s
        p = tire_temp_penalty_s("medium", 40.0, 30.0, 50, 50.0, False)
        assert p > 0.15
        assert abs(p - 0.20) < 0.01

    def test_soft_hot_penalty_heavier_than_cold(self):
        """软胎过热惩罚重于冷胎 (per °C)."""
        # soft: cold_pen=0.030, hot_pen=0.060
        # 制造相同温差 10°C
        # cold: temp = opt_min - 10 = 75, need track_temp = 75 - 60 + 25 = 40, lap=0
        # 实际: lap=0 → temp = track + 60 - 25 = track + 35 = 75 → track = 40
        cold_p = tire_temp_penalty_s("soft", 40.0, 25.0, 0, 0.0, False)
        # hot: temp = opt_max + 10 = 110, need track + 60 + 0 = 110 → track = 50, lap=3
        hot_p = tire_temp_penalty_s("soft", 50.0, 25.0, 3, 0.0, False)
        assert hot_p > cold_p

    def test_wet_reduces_tire_temp_penalty_for_hot_track(self):
        """湿地降温, 减少热赛道下的过热惩罚."""
        # 40°C 赛道, age=30, lap=30 → 干地 temp = 100 + 9 = 109 (in window)
        # 湿地 temp = 109 - 15 = 94 (cold for soft 85-100? still in window)
        # 实际: soft 窗口 85-100
        p_dry = tire_temp_penalty_s("soft", 40.0, 30.0, 30, 30.0, False)
        p_wet = tire_temp_penalty_s("soft", 40.0, 30.0, 30, 30.0, True)
        # 湿地降温后离窗口更远 (cold) 或更近 (hot), 但温度变化必然改变惩罚
        # 这里干地高温 → 湿地降温 → 更接近窗口下界 (软胎), 惩罚降低
        assert p_wet < p_dry or p_wet == 0.0

    def test_unknown_compound_no_penalty(self):
        p = tire_temp_penalty_s("nonexistent", 30.0, 25.0, 0, 0.0, False)
        assert p == 0.0


# --------------------------------------------------------------------------- #
# 状态查询
# --------------------------------------------------------------------------- #
class TestTempState:
    """温度状态查询."""

    def test_state_cold_at_lap_0(self):
        s = temp_state("medium", 30.0, 25.0, 0, 0.0, False)
        assert s == "cold"

    def test_state_optimal_when_warm(self):
        s = temp_state("medium", 30.0, 25.0, 5, 0.0, False)
        assert s == "optimal"

    def test_state_hot_when_old_on_hot_track(self):
        s = temp_state("soft", 45.0, 35.0, 50, 50.0, False)
        assert s == "hot"

    def test_is_in_optimal_window(self):
        assert is_in_optimal_window("medium", 30.0, 25.0, 5, 0.0, False) is True
        assert is_in_optimal_window("medium", 30.0, 25.0, 0, 0.0, False) is False

    def test_all_compounds_contains_known(self):
        cs = all_compounds()
        for c in ("hard", "medium", "soft", "intermediate", "wet"):
            assert c in cs


# --------------------------------------------------------------------------- #
# 性能 (< 2μs/call, 不影响 lap_simulator 总性能)
# --------------------------------------------------------------------------- #
class TestPerformance:
    """tire_temp_penalty_s 必须 < 2μs (不影响 lap_simulator 10μs 总性能)."""

    def test_penalty_under_2us(self):
        # warmup
        for _ in range(100):
            tire_temp_penalty_s("medium", 30.0, 25.0, 5, 10.0, False)
        elapsed = timeit.timeit(
            lambda: tire_temp_penalty_s("medium", 30.0, 25.0, 5, 10.0, False),
            number=50000,
        )
        per_call_us = (elapsed / 50000) * 1e6
        assert per_call_us < 2.0, f"tire_temp_penalty_s {per_call_us:.3f}μs > 2μs 阈值"
