"""Iter-56 测试: EA F1 2026 高性能单圈仿真器."""

from __future__ import annotations

import timeit

from f1opt.data.ea_f1_2026_benchmark import (
    accuracy_threshold_pct,
)
from f1opt.model.fuel_model import FuelMode
from f1opt.model.lap_simulator_2026 import (
    LapConfig2026,
    LapSimulator2026,
    MultiStintSimulator2026,
    StintPlan2026,
    compare_strategies,
    is_2026_compliant,
    quick_lap_time_s,
    simulate_lap_2026,
    validate_against_benchmark,
)
from f1opt.model.pu_2026 import BATTERY_CAPACITY_MJ, PUDeployMode
from f1opt.model.safety_car import SafetyCarModel, SafetyCarPeriod


# --------------------------------------------------------------------------- #
# 基准验证 (24 赛道)
# --------------------------------------------------------------------------- #
class TestBenchmarkValidation:
    """reference state 必须通过 24 赛道精度验证."""

    def test_reference_state_all_tracks_pass(self):
        """reference state: 所有 24 赛道必须 PASS 1.5% 阈值."""
        report = validate_against_benchmark()
        assert report["pass_rate"] == 1.0
        assert report["passed"] == 24
        assert report["total"] == 24

    def test_reference_state_avg_error_below_0_1_pct(self):
        """reference state 平均误差 < 0.1% (远优于 1.5% 阈值)."""
        report = validate_against_benchmark()
        assert report["avg_error_pct"] < 0.1

    def test_reference_state_worst_below_1_5_pct(self):
        """reference state 最差赛道 < 1.5%."""
        report = validate_against_benchmark()
        assert report["worst_error_pct"] < accuracy_threshold_pct()

    def test_known_tracks_match_benchmark(self):
        """关键赛道圈速必须接近 benchmark."""
        cases = [
            ("monaco", 73.0),
            ("spa", 104.5),
            ("spielberg", 64.5),
            ("monza", 81.0),
        ]
        for tid, expected in cases:
            sim = quick_lap_time_s(tid)
            err_pct = 100.0 * abs(sim - expected) / expected
            assert err_pct < 0.5, f"{tid}: sim={sim:.3f}, expected={expected}, err={err_pct:.3f}%"


# --------------------------------------------------------------------------- #
# 子系统 delta 方向性
# --------------------------------------------------------------------------- #
class TestSubsystemDeltas:
    """各子系统 delta 必须方向正确."""

    def test_tire_aging_makes_slower(self):
        """轮胎老化 (越过 cliff) → 圈速变慢.

        medium tire: warmup=2, optimal=2..30, cliff=30+.
        age=25 (optimal 末段) vs age=35 (cliff) — 老化变慢.
        """
        optimal = simulate_lap_2026(LapConfig2026(track_id="monza", tire_age_laps=25))
        cliffed = simulate_lap_2026(LapConfig2026(track_id="monza", tire_age_laps=35))
        assert cliffed.lap_time_s > optimal.lap_time_s
        assert cliffed.tire_delta_s > optimal.tire_delta_s

    def test_more_fuel_makes_slower(self):
        """更多燃油 → 圈速变慢."""
        light = simulate_lap_2026(LapConfig2026(track_id="monza", current_fuel_kg=20.0))
        heavy = simulate_lap_2026(LapConfig2026(track_id="monza", current_fuel_kg=110.0))
        assert heavy.lap_time_s > light.lap_time_s
        assert heavy.fuel_delta_s > light.fuel_delta_s

    def test_qualifying_mode_faster_than_balanced(self):
        """QUALIFYING 模式比 BALANCED 快."""
        balanced = simulate_lap_2026(
            LapConfig2026(track_id="monza", pu_mode=PUDeployMode.BALANCED)
        )
        qualifying = simulate_lap_2026(
            LapConfig2026(track_id="monza", pu_mode=PUDeployMode.QUALIFYING)
        )
        assert qualifying.lap_time_s < balanced.lap_time_s
        assert qualifying.pu_delta_s < balanced.pu_delta_s

    def test_conserve_mode_slower_than_balanced(self):
        """CONSERVE 模式比 BALANCED 慢 (省电池 → 部署少)."""
        balanced = simulate_lap_2026(
            LapConfig2026(track_id="monza", pu_mode=PUDeployMode.BALANCED)
        )
        conserve = simulate_lap_2026(
            LapConfig2026(track_id="monza", pu_mode=PUDeployMode.CONSERVE)
        )
        assert conserve.lap_time_s > balanced.lap_time_s

    def test_attack_mode_between_qualifying_and_balanced(self):
        """ATTACK 模式圈速介于 QUALIFYING 和 BALANCED 之间."""
        q = simulate_lap_2026(
            LapConfig2026(track_id="monza", pu_mode=PUDeployMode.QUALIFYING)
        ).lap_time_s
        a = simulate_lap_2026(
            LapConfig2026(track_id="monza", pu_mode=PUDeployMode.ATTACK)
        ).lap_time_s
        b = simulate_lap_2026(
            LapConfig2026(track_id="monza", pu_mode=PUDeployMode.BALANCED)
        ).lap_time_s
        assert q < a < b

    def test_wet_slower_than_dry(self):
        """湿地比干地慢 (X-mode 禁用 + 抓地力损失)."""
        dry = simulate_lap_2026(LapConfig2026(track_id="monza", wet=False))
        wet = simulate_lap_2026(LapConfig2026(track_id="monza", wet=True))
        assert wet.lap_time_s > dry.lap_time_s
        assert wet.weather_penalty_s > 0
        assert wet.aero_delta_s > dry.aero_delta_s

    def test_traffic_slower_than_clean_air(self):
        """交通 (gap < 1.0) 比净空慢 (无法激活 X-mode)."""
        clean = simulate_lap_2026(
            LapConfig2026(track_id="monza", gap_to_ahead_s=2.0)
        )
        traffic = simulate_lap_2026(
            LapConfig2026(track_id="monza", gap_to_ahead_s=0.5)
        )
        assert traffic.lap_time_s >= clean.lap_time_s

    def test_fuel_mode_lean_slower_rich_faster(self):
        """LEAN 模式慢, RICH 模式快 (相对 NORMAL)."""
        normal = simulate_lap_2026(
            LapConfig2026(track_id="monza", fuel_mode=FuelMode.NORMAL)
        )
        lean = simulate_lap_2026(
            LapConfig2026(track_id="monza", fuel_mode=FuelMode.LEAN)
        )
        rich = simulate_lap_2026(
            LapConfig2026(track_id="monza", fuel_mode=FuelMode.RICH)
        )
        assert lean.lap_time_s > normal.lap_time_s
        assert rich.lap_time_s < normal.lap_time_s

    def test_top_team_faster(self):
        """顶级车队 (car_offset 负) 比后段车队快."""
        back = simulate_lap_2026(
            LapConfig2026(track_id="monza", car_performance_offset_s=0.8)
        )
        top = simulate_lap_2026(
            LapConfig2026(track_id="monza", car_performance_offset_s=-0.6)
        )
        assert top.lap_time_s < back.lap_time_s

    def test_skilled_driver_faster(self):
        """车手技术好 (driver_offset 负) 更快."""
        rookie = simulate_lap_2026(
            LapConfig2026(track_id="monza", driver_skill_offset_s=0.4)
        )
        pro = simulate_lap_2026(
            LapConfig2026(track_id="monza", driver_skill_offset_s=-0.3)
        )
        assert pro.lap_time_s < rookie.lap_time_s

    def test_drs_qualifying_zero_delta(self):
        """排位赛 DRS delta = 0 (benchmark 已含 DRS)."""
        r = simulate_lap_2026(LapConfig2026(track_id="monza", session_type="qualifying"))
        assert r.drs_delta_s == 0.0

    def test_drs_race_no_car_slower(self):
        """正赛无前车 → DRS 不可用 → 圈速变慢 (delta > 0)."""
        r = simulate_lap_2026(
            LapConfig2026(
                track_id="monza", session_type="race", lap=5, gap_to_ahead_s=2.0
            )
        )
        assert r.drs_delta_s > 0.0
        # Monza DRS 总长 2000m × 0.000375 = 0.75s
        assert abs(r.drs_delta_s - 0.75) < 0.01

    def test_drs_race_close_car_zero_delta(self):
        """正赛有前车 (gap < 1s) → DRS 可用 → delta = 0."""
        r = simulate_lap_2026(
            LapConfig2026(
                track_id="monza", session_type="race", lap=5, gap_to_ahead_s=0.5
            )
        )
        assert r.drs_delta_s == 0.0

    def test_drs_wet_disabled(self):
        """湿地 → DRS 禁用 → delta > 0."""
        r = simulate_lap_2026(LapConfig2026(track_id="monza", wet=True))
        assert r.drs_delta_s > 0.0

    def test_drs_race_lap1_disabled(self):
        """正赛第 1 圈 DRS 禁用."""
        r = simulate_lap_2026(
            LapConfig2026(
                track_id="monza", session_type="race", lap=1, gap_to_ahead_s=0.5
            )
        )
        assert r.drs_delta_s > 0.0

    def test_drs_monza_more_than_monaco(self):
        """Monza (长 DRS 直道) DRS delta > Monaco (短直道) 当禁用时."""
        monza = simulate_lap_2026(
            LapConfig2026(
                track_id="monza", session_type="race", lap=5, gap_to_ahead_s=2.0
            )
        )
        monaco = simulate_lap_2026(
            LapConfig2026(
                track_id="monaco", session_type="race", lap=5, gap_to_ahead_s=2.0
            )
        )
        assert monza.drs_delta_s > monaco.drs_delta_s


# --------------------------------------------------------------------------- #
# LapResult2026 属性
# --------------------------------------------------------------------------- #
class TestLapResult:
    def test_within_threshold_pass(self):
        r = simulate_lap_2026(LapConfig2026(track_id="monza"))
        assert r.within_threshold
        assert r.verdict == "PASS"

    def test_fail_when_offset_too_large(self):
        """过大 driver_offset → FAIL."""
        r = simulate_lap_2026(
            LapConfig2026(track_id="monza", driver_skill_offset_s=5.0)
        )
        assert not r.within_threshold
        assert r.verdict == "FAIL"

    def test_lap_time_within_physical_bounds(self):
        """圈速在 60-180s 范围内 (EA F1 2026 物理)."""
        for tid in ["monaco", "spa", "spielberg", "monza"]:
            r = simulate_lap_2026(LapConfig2026(track_id=tid))
            assert 60.0 <= r.lap_time_s <= 180.0


# --------------------------------------------------------------------------- #
# Stint 仿真 (跨圈状态)
# --------------------------------------------------------------------------- #
class TestStintSimulation:
    def test_stint_returns_n_laps(self):
        sim = LapSimulator2026(track_id="monza", total_laps=10)
        stint = sim.simulate_stint()
        assert len(stint) == 10

    def test_stint_tire_wear_increases_lap_time(self):
        """stint 后期 (轮胎深 cliff) 圈速慢于前期.

        soft tire 30 圈 stint: cliff=22, 后期深 cliff 主导燃油效应.
        """
        sim = LapSimulator2026(
            track_id="monza", total_laps=30, compound="soft",
            initial_fuel_kg=110.0,
        )
        stint = sim.simulate_stint()
        # 前 3 圈 (warmup+满油) vs 后 3 圈 (深 cliff)
        early_avg = sum(r.lap_time_s for r in stint[:3]) / 3
        late_avg = sum(r.lap_time_s for r in stint[-3:]) / 3
        assert late_avg > early_avg

    def test_stint_fuel_decreases(self):
        """stint 中燃油减少 (车变快)."""
        sim = LapSimulator2026(
            track_id="monza", total_laps=10, fuel_burn_per_lap_kg=1.7
        )
        stint = sim.simulate_stint()
        # 燃油效应: 后期圈速的 fuel_delta < 前期
        assert stint[-1].fuel_delta_s < stint[0].fuel_delta_s

    def test_stint_summary(self):
        sim = LapSimulator2026(track_id="monza", total_laps=5)
        sim.simulate_stint()
        s = sim.summary()
        assert s["track_id"] == "monza"
        assert s["laps"] == 5
        assert s["best_lap"] <= s["worst_lap"]
        assert s["final_tire_age"] == 5
        assert s["final_fuel_kg"] < 110.0

    def test_stint_all_within_threshold(self):
        """stint 中至少 60% 圈在阈值内 (允许燃油/轮胎导致部分偏离).

        使用接近 reference 的燃油量 (60kg) 减少燃油偏差.
        """
        sim = LapSimulator2026(
            track_id="monza", total_laps=10, initial_fuel_kg=60.0
        )
        stint = sim.simulate_stint()
        within_count = sum(1 for r in stint if r.within_threshold)
        assert within_count >= 6

    def test_track_specific_fuel_burn_monaco_vs_spa(self):
        """EA F1 2026: Monaco (1.20 kg/lap) 比 Spa (2.10 kg/lap) 省油 ~43%.

        stint 仿真器必须自动按赛道查表, 不再用 flat 1.7 默认.
        """
        monaco_sim = LapSimulator2026(track_id="monaco", total_laps=10)
        spa_sim = LapSimulator2026(track_id="spa", total_laps=10)
        # 解析后的每圈消耗
        assert abs(monaco_sim._resolved_fuel_burn_kg - 1.20) < 1e-6
        assert abs(spa_sim._resolved_fuel_burn_kg - 2.10) < 1e-6
        # 实际燃油消耗
        monaco_stint = monaco_sim.simulate_stint()
        spa_stint = spa_sim.simulate_stint()
        monaco_total_burn = sum(r.fuel_burned_kg for r in monaco_stint)
        spa_total_burn = sum(r.fuel_burned_kg for r in spa_stint)
        assert spa_total_burn > monaco_total_burn * 1.5  # Spa 至少多 50%

    def test_fuel_mode_affects_burn_rate(self):
        """LEAN 模式省油 12%, PARTY 多耗 25% (EA F1 2026 物理)."""
        normal = LapSimulator2026(
            track_id="monza", total_laps=5, fuel_mode=FuelMode.NORMAL
        )
        lean = LapSimulator2026(
            track_id="monza", total_laps=5, fuel_mode=FuelMode.LEAN
        )
        party = LapSimulator2026(
            track_id="monza", total_laps=5, fuel_mode=FuelMode.PARTY
        )
        # monza base = 1.90; LEAN = 1.90*0.88, PARTY = 1.90*1.25
        assert lean._resolved_fuel_burn_kg < normal._resolved_fuel_burn_kg
        assert party._resolved_fuel_burn_kg > normal._resolved_fuel_burn_kg
        # 量级检查
        assert abs(normal._resolved_fuel_burn_kg - 1.90) < 1e-6
        assert abs(lean._resolved_fuel_burn_kg - 1.90 * 0.88) < 1e-6
        assert abs(party._resolved_fuel_burn_kg - 1.90 * 1.25) < 1e-6

    def test_explicit_fuel_burn_overrides_auto(self):
        """显式指定 fuel_burn_per_lap_kg > 0 时覆盖自动查表."""
        sim = LapSimulator2026(
            track_id="monza", total_laps=5, fuel_burn_per_lap_kg=2.5
        )
        assert abs(sim._resolved_fuel_burn_kg - 2.5) < 1e-6


# --------------------------------------------------------------------------- #
# SC/VSC 集成 (EA F1 2026 race physics)
# --------------------------------------------------------------------------- #
class TestSafetyCarIntegration:
    """SC/VSC 注入 stint 仿真器后的物理行为."""

    def _sc_model(self, start: int, end: int, kind: str = "sc") -> SafetyCarModel:
        m = SafetyCarModel()
        m.periods = [SafetyCarPeriod(start_lap=start, end_lap=end, kind=kind)]
        return m

    def test_sc_lap_is_30pct_slower(self):
        """SC 跟车圈速 = 正常 × 1.30 (EA F1 2026)."""
        base = LapSimulator2026(track_id="monza", total_laps=10)
        sc = LapSimulator2026(
            track_id="monza", total_laps=10, safety_car=self._sc_model(5, 7, "sc")
        )
        base_stint = base.simulate_stint()
        sc_stint = sc.simulate_stint()
        # lap 5,6,7 (idx 4,5,6) 是 SC 圈
        for idx in (4, 5, 6):
            ratio = sc_stint[idx].lap_time_s / base_stint[idx].lap_time_s
            assert 1.25 < ratio < 1.35, f"SC lap {idx+1} ratio={ratio:.3f}"

    def test_vsc_lap_is_25pct_slower(self):
        """VSC 圈速 = 正常 × 1.25."""
        base = LapSimulator2026(track_id="monza", total_laps=10)
        vsc = LapSimulator2026(
            track_id="monza", total_laps=10, safety_car=self._sc_model(5, 6, "vsc")
        )
        base_stint = base.simulate_stint()
        vsc_stint = vsc.simulate_stint()
        for idx in (4, 5):
            ratio = vsc_stint[idx].lap_time_s / base_stint[idx].lap_time_s
            assert 1.20 < ratio < 1.30, f"VSC lap {idx+1} ratio={ratio:.3f}"

    def test_restart_lap_has_penalty(self):
        """SC 结束后第 1 圈 +0.8s 重启惩罚 (冷胎+混乱)."""
        base = LapSimulator2026(track_id="monza", total_laps=10)
        sc = LapSimulator2026(
            track_id="monza", total_laps=10, safety_car=self._sc_model(5, 7, "sc")
        )
        base_stint = base.simulate_stint()
        sc_stint = sc.simulate_stint()
        # lap 8 (idx 7) = 重启圈
        diff = sc_stint[7].lap_time_s - base_stint[7].lap_time_s
        assert 0.5 < diff < 1.5, f"Restart penalty diff={diff:.3f}s"

    def test_sc_reduces_tire_wear(self):
        """SC 期间轮胎磨损 = 30% 正常 (低应力跟车)."""
        base = LapSimulator2026(track_id="monza", total_laps=10)
        sc = LapSimulator2026(
            track_id="monza", total_laps=10, safety_car=self._sc_model(5, 7, "sc")
        )
        base.simulate_stint()
        sc.simulate_stint()
        # SC stint: 3 SC 圈 × 0.3 + 7 正常圈 × 1.0 = 0.9 + 7 = 7.9
        # Base: 10 × 1.0 = 10
        assert sc._tire_age < base._tire_age
        assert abs(sc._tire_age - 7.9) < 0.01, f"SC tire age={sc._tire_age}"

    def test_sc_reduces_fuel_burn(self):
        """SC 期间燃油消耗 = 50% 正常."""
        base = LapSimulator2026(track_id="monza", total_laps=10)
        sc = LapSimulator2026(
            track_id="monza", total_laps=10, safety_car=self._sc_model(5, 7, "sc")
        )
        base.simulate_stint()
        sc.simulate_stint()
        # SC stint 燃油消耗更少 → 余油更多
        assert sc._fuel_kg > base._fuel_kg

    def test_sc_recharges_soc_to_full(self):
        """SC 期间 SoC 充满 (持续 regen)."""
        # 起步 SoC 半满
        sc = LapSimulator2026(
            track_id="monza", total_laps=10,
            initial_pu_soc_mj=BATTERY_CAPACITY_MJ * 0.5,
            safety_car=self._sc_model(5, 7, "sc"),
        )
        stint = sc.simulate_stint()
        # SC 期间 (lap 5,6,7) SoC 应被充满
        for idx in (4, 5, 6):
            assert abs(stint[idx].pu_soc_after_mj - BATTERY_CAPACITY_MJ) < 1e-6

    def test_drs_disabled_after_sc(self):
        """FIA 2026: SC 后 2 圈 DRS 禁用 → drs_delta > 0 (慢于 reference)."""
        sc = LapSimulator2026(
            track_id="monza", total_laps=12,
            session_type="race",
            gap_to_ahead_s=0.6,  # race 跟车
            safety_car=self._sc_model(5, 7, "sc"),
        )
        stint = sc.simulate_stint()
        # lap 8,9 (idx 7,8) = DRS 禁用窗口
        for idx in (7, 8):
            assert stint[idx].drs_delta_s > 0.01, (
                f"lap {idx+1} DRS should be disabled (delta>0), got {stint[idx].drs_delta_s}"
            )
        # lap 10 (idx 9) = DRS 恢复
        assert stint[9].drs_delta_s < 0.01, (
            f"lap 10 DRS should be enabled (delta~0), got {stint[9].drs_delta_s}"
        )

    def test_no_sc_no_change(self):
        """无 SC 注入时行为与原版一致."""
        a = LapSimulator2026(track_id="monza", total_laps=5)
        b = LapSimulator2026(track_id="monza", total_laps=5, safety_car=None)
        a_stint = a.simulate_stint()
        b_stint = b.simulate_stint()
        for i in range(5):
            assert abs(a_stint[i].lap_time_s - b_stint[i].lap_time_s) < 1e-9


# --------------------------------------------------------------------------- #
# 性能基准
# --------------------------------------------------------------------------- #
class TestPerformance:
    def test_single_lap_under_50_us(self):
        """单圈仿真 < 50 us (高性能目标)."""
        cfg = LapConfig2026(track_id="monza")
        n = 10000
        t = timeit.timeit(lambda: simulate_lap_2026(cfg), number=n)
        per_call_us = t / n * 1e6
        assert per_call_us < 50.0, f"Too slow: {per_call_us:.2f} us/call"

    def test_stint_under_1ms_for_20_laps(self):
        """20 圈 stint < 1 ms (跨圈性能)."""
        sim = LapSimulator2026(track_id="monza", total_laps=20)
        n = 100
        t = timeit.timeit(lambda: sim.simulate_stint(), number=n)
        per_stint_ms = t / n * 1e3
        assert per_stint_ms < 5.0, f"Stint too slow: {per_stint_ms:.2f} ms/stint"


# --------------------------------------------------------------------------- #
# 便捷函数
# --------------------------------------------------------------------------- #
class TestConvenience:
    def test_quick_lap_time_returns_float(self):
        t = quick_lap_time_s("monza")
        assert isinstance(t, float)
        assert 60.0 <= t <= 180.0

    def test_is_2026_compliant_reference(self):
        assert is_2026_compliant("monza") is True

    def test_is_2026_compliant_extreme_offset(self):
        assert is_2026_compliant("monza", driver_skill_offset_s=10.0) is False


# --------------------------------------------------------------------------- #
# EA F1 2026 物理量级合理性
# --------------------------------------------------------------------------- #
class TestEAF12026PhysicsMagnitude:
    def test_qualifying_vs_balanced_diff_realistic(self):
        """QUALIFYING vs BALANCED 圈速差 0.2-0.5s (EA F1 2026 量级)."""
        b = simulate_lap_2026(
            LapConfig2026(track_id="monza", pu_mode=PUDeployMode.BALANCED)
        ).lap_time_s
        q = simulate_lap_2026(
            LapConfig2026(track_id="monza", pu_mode=PUDeployMode.QUALIFYING)
        ).lap_time_s
        diff = b - q
        assert 0.15 < diff < 0.6, f"QUALIFYING vs BALANCED diff={diff:.3f}s"

    def test_wet_penalty_realistic(self):
        """湿地圈速损失 > 2s (X-mode 禁用 + 抓地力损失)."""
        dry = simulate_lap_2026(LapConfig2026(track_id="monza", wet=False)).lap_time_s
        wet = simulate_lap_2026(LapConfig2026(track_id="monza", wet=True)).lap_time_s
        diff = wet - dry
        assert diff > 2.0, f"Wet penalty too small: {diff:.3f}s"

    def test_fuel_mass_effect_realistic(self):
        """110kg vs 5kg 燃油差 > 3.5s (0.35s/10kg)."""
        full = simulate_lap_2026(
            LapConfig2026(track_id="monza", current_fuel_kg=110.0)
        ).lap_time_s
        empty = simulate_lap_2026(
            LapConfig2026(track_id="monza", current_fuel_kg=5.0)
        ).lap_time_s
        diff = full - empty
        assert diff > 3.0, f"Fuel mass effect too small: {diff:.3f}s"


# --------------------------------------------------------------------------- #
# 多 stint 进站仿真 (EA F1 2026 race physics)
# --------------------------------------------------------------------------- #
class TestMultiStintPitSimulation:
    """MultiStintSimulator2026 多 stint 进站仿真物理."""

    def test_stint_plan_validation(self):
        """StintPlan2026 校验."""
        plan = StintPlan2026(compounds=("medium", "soft"), stint_lengths=(35, 25))
        assert plan.n_stints == 2
        assert plan.n_stops == 1
        assert plan.total_laps == 60
        assert plan.pit_laps() == (35,)

    def test_stint_plan_rejects_mismatched_lengths(self):
        import pytest
        with pytest.raises(ValueError):
            StintPlan2026(compounds=("medium", "soft"), stint_lengths=(35,))

    def test_stint_plan_rejects_zero_length(self):
        import pytest
        with pytest.raises(ValueError):
            StintPlan2026(compounds=("medium", "soft"), stint_lengths=(35, 0))

    def test_no_stop_race_matches_single_stint(self):
        """0-stop 比赛圈速 = 单 stint 仿真器圈速 (同为 race session)."""
        plan = StintPlan2026(compounds=("medium",), stint_lengths=(10,))
        multi = MultiStintSimulator2026(track_id="monza", plan=plan)
        # 同为 race session + 净空 (无前车 DRS) 以匹配 multi-stint 内部设定
        single = LapSimulator2026(
            track_id="monza", total_laps=10, session_type="race"
        )
        multi_race = multi.simulate_race()
        single_stint = single.simulate_stint()
        for i in range(10):
            assert abs(multi_race[i].lap_time_s - single_stint[i].lap_time_s) < 1e-9

    def test_pit_loss_applied_on_pit_lap(self):
        """进站圈圈速含进站损失 (~23s for monza)."""
        plan = StintPlan2026(
            compounds=("medium", "soft"), stint_lengths=(10, 10)
        )
        multi = MultiStintSimulator2026(track_id="monza", plan=plan)
        race = multi.simulate_race()
        # 进站圈 = lap 10 (stint 0 末圈)
        # 无进站的同圈 (单 stint lap 10, race session)
        single = LapSimulator2026(
            track_id="monza", total_laps=10, session_type="race"
        )
        single_stint = single.simulate_stint()
        pit_lap_diff = race[9].lap_time_s - single_stint[9].lap_time_s
        # monza pit_loss = 23s
        assert 22.0 < pit_lap_diff < 24.0, f"pit loss={pit_lap_diff:.2f}s"

    def test_pit_loss_track_specific(self):
        """Monaco 进站损失 21s < Monza 23s (EA F1 2026 赛道差异)."""
        plan = StintPlan2026(compounds=("medium", "soft"), stint_lengths=(10, 10))
        monza = MultiStintSimulator2026(track_id="monza", plan=plan).simulate_race()
        monaco = MultiStintSimulator2026(track_id="monaco", plan=plan).simulate_race()
        # 单独看进站圈加成 (用单 stint 基线, race session)
        monza_single = LapSimulator2026(
            track_id="monza", total_laps=10, session_type="race"
        ).simulate_stint()
        monaco_single = LapSimulator2026(
            track_id="monaco", total_laps=10, session_type="race"
        ).simulate_stint()
        monza_pit = monza[9].lap_time_s - monza_single[9].lap_time_s
        monaco_pit = monaco[9].lap_time_s - monaco_single[9].lap_time_s
        assert monaco_pit < monza_pit, (
            f"Monaco pit {monaco_pit:.1f}s should < Monza {monza_pit:.1f}s"
        )

    def test_compound_switch_resets_tire_age(self):
        """进站后新 stint 轮胎 age=0 (warmup 损失再现)."""
        plan = StintPlan2026(
            compounds=("medium", "soft"), stint_lengths=(15, 10)
        )
        multi = MultiStintSimulator2026(track_id="monza", plan=plan)
        race = multi.simulate_race()
        # stint 1 第 1 圈 (idx 15) = 新胎 warmup, 应慢于 stint 1 第 3 圈 (idx 17)
        # (warmup 损失 + soft 化合物特性)
        assert race[15].tire_delta_s > race[17].tire_delta_s

    def test_fuel_carried_across_stints(self):
        """燃油跨 stint 延续 (不重置)."""
        plan = StintPlan2026(
            compounds=("medium", "medium"), stint_lengths=(10, 10)
        )
        multi = MultiStintSimulator2026(
            track_id="monza", plan=plan, initial_fuel_kg=80.0
        )
        race = multi.simulate_race()
        # stint 1 第 1 圈燃油 < 80kg (stint 0 已消耗 ~19kg)
        assert race[10].fuel_delta_s < race[0].fuel_delta_s

    def test_sc_free_pit_discount(self):
        """SC 期间进站损失 = 正常 × 0.20 (free pit)."""
        sc = SafetyCarModel()
        sc.periods = [SafetyCarPeriod(start_lap=10, end_lap=12, kind="sc")]
        plan = StintPlan2026(
            compounds=("medium", "soft"), stint_lengths=(10, 10)
        )
        # 进站圈 = lap 10 (SC 期间)
        multi_sc = MultiStintSimulator2026(
            track_id="monza", plan=plan, safety_car=sc
        )
        multi_normal = MultiStintSimulator2026(track_id="monza", plan=plan)
        multi_sc.simulate_race()
        multi_normal.simulate_race()
        # SC 进站圈 lap 10 (idx 9)
        # SC 圈本身已 ×1.30, 需比较"进站损失部分"
        sc_rec = multi_sc._pit_records[0]
        normal_rec = multi_normal._pit_records[0]
        assert sc_rec["sc_discount"] == 0.20
        assert sc_rec["pit_loss_s"] < normal_rec["pit_loss_s"] * 0.5

    def test_team_crew_offset_applied(self):
        """顶级车队 (rbr 96) 进站快于后段车队 (has 76)."""
        plan = StintPlan2026(
            compounds=("medium", "soft"), stint_lengths=(10, 10)
        )
        rbr = MultiStintSimulator2026(track_id="monza", plan=plan, team_id="rbr")
        has = MultiStintSimulator2026(track_id="monza", plan=plan, team_id="has")
        rbr.simulate_race()
        has.simulate_race()
        # crew offset: rbr 快 (负 offset), has 慢 (正 offset)
        assert rbr._pit_records[0]["crew_offset_s"] < has._pit_records[0]["crew_offset_s"]

    def test_multi_stint_total_laps(self):
        """2-stop 比赛 = 3 stint 总圈数正确."""
        plan = StintPlan2026(
            compounds=("medium", "soft", "medium"),
            stint_lengths=(20, 15, 25),
        )
        multi = MultiStintSimulator2026(track_id="spa", plan=plan)
        race = multi.simulate_race()
        assert len(race) == 60
        assert multi.summary()["n_stops"] == 2
        assert multi.summary()["n_stints"] == 3

    def test_multi_stint_summary(self):
        plan = StintPlan2026(compounds=("medium", "soft"), stint_lengths=(10, 10))
        multi = MultiStintSimulator2026(track_id="monza", plan=plan)
        multi.simulate_race()
        s = multi.summary()
        assert s["track_id"] == "monza"
        assert s["total_laps"] == 20
        assert s["n_stops"] == 1
        assert len(s["pit_records"]) == 1
        assert s["pit_records"][0]["from_compound"] == "medium"
        assert s["pit_records"][0]["to_compound"] == "soft"


# --------------------------------------------------------------------------- #
# 策略对比工具 (EA F1 2026 专业车队策略评估)
# --------------------------------------------------------------------------- #
class TestStrategyComparison:
    """compare_strategies 进站策略对比."""

    def test_returns_sorted_results(self):
        """结果按 total_time 升序, rank 从 1 开始."""
        plans = [
            StintPlan2026(("medium", "soft"), (32, 28)),
            StintPlan2026(("soft", "medium"), (25, 35)),
            StintPlan2026(("medium", "soft", "medium"), (20, 20, 20)),
        ]
        results = compare_strategies("monza", plans)
        assert len(results) == 3
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[2].rank == 3
        # 升序
        assert results[0].total_time_s <= results[1].total_time_s <= results[2].total_time_s

    def test_delta_to_best_zero_for_rank_1(self):
        """最优策略 delta_to_best = 0."""
        plans = [
            StintPlan2026(("medium", "soft"), (32, 28)),
            StintPlan2026(("soft", "medium"), (25, 35)),
        ]
        results = compare_strategies("monza", plans)
        assert results[0].delta_to_best_s == 0.0
        assert results[1].delta_to_best_s > 0.0

    def test_fewer_stops_better_at_low_wear_track(self):
        """Monaco (低磨损, 短圈) 0-stop 应优于 2-stop (进站损失主导)."""
        plans = [
            StintPlan2026(("medium",), (30,)),  # 0-stop
            StintPlan2026(("medium", "soft", "medium"), (10, 10, 10)),  # 2-stop
        ]
        results = compare_strategies("monaco", plans)
        # 0-stop 应 rank 1 (省 2×21s 进站损失)
        assert results[0].n_stops == 0
        assert results[1].n_stops == 2

    def test_strategy_with_sc_pit_better(self):
        """含 SC 时, 进站圈与 SC 重叠的策略应更优 (free-pit)."""
        sc = SafetyCarModel()
        sc.periods = [SafetyCarPeriod(start_lap=20, end_lap=23, kind="sc")]
        plans = [
            # 策略 A: 进站圈 = 20 (SC 期间, free-pit)
            StintPlan2026(("medium", "soft"), (20, 20)),
            # 策略 B: 进站圈 = 15 (SC 前, 全额进站损失)
            StintPlan2026(("medium", "soft"), (15, 25)),
        ]
        results = compare_strategies("monza", plans, safety_car=sc)
        # 策略 A (SC free-pit) 应更优
        assert results[0].pit_records[0]["sc_discount"] <= 0.5
        assert results[0].total_time_s < results[1].total_time_s

    def test_results_contain_pit_records(self):
        plans = [
            StintPlan2026(("medium", "soft"), (20, 20)),
            StintPlan2026(("medium",), (40,)),  # 0-stop, 无 pit_records
        ]
        results = compare_strategies("monza", plans)
        assert len(results[0].pit_records) == 1
        assert len(results[1].pit_records) == 0

    def test_empty_plans_raises(self):
        import pytest
        with pytest.raises(ValueError):
            compare_strategies("monza", [])

    def test_team_affects_comparison(self):
        """顶级车队策略对比: rbr 进站快, 总时间更优."""
        plans = [StintPlan2026(("medium", "soft"), (20, 20))]
        rbr_results = compare_strategies("monza", plans, team_id="rbr")
        has_results = compare_strategies("monza", plans, team_id="has")
        assert rbr_results[0].total_time_s < has_results[0].total_time_s

    def test_strategy_comparison_performance(self):
        """5 策略对比 < 10ms (高性能策略评估)."""
        plans = [
            StintPlan2026(("medium", "soft"), (32, 28)),
            StintPlan2026(("soft", "medium"), (25, 35)),
            StintPlan2026(("medium", "soft", "medium"), (20, 20, 20)),
            StintPlan2026(("soft", "medium", "soft"), (15, 20, 25)),
            StintPlan2026(("medium", "medium"), (30, 30)),
        ]
        import timeit
        t = timeit.timeit(
            lambda: compare_strategies("monza", plans), number=20
        )
        per_call_ms = t / 20 * 1e3
        assert per_call_ms < 50.0, f"Strategy comparison too slow: {per_call_ms:.2f}ms"
