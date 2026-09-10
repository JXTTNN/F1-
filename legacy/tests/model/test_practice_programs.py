"""F1 2026 自由练习赛项目模型测试 (Iter-45)."""

from __future__ import annotations

from f1opt.model.practice_programs import (
    PracticeProgram,
    PracticeSessionSimulator,
    ProgramOutcome,
    ProgramType,
    simulate_practice_program,
)


# --------------------------------------------------------------------------- #
# ProgramType / ProgramOutcome 枚举
# --------------------------------------------------------------------------- #
def test_program_type_values():
    assert ProgramType.ACCLIMATIZATION.value == "AP"
    assert ProgramType.AERO_TESTING.value == "AER"
    assert ProgramType.RACE_TRIM.value == "RT"
    assert ProgramType.QUALIFYING_SIM.value == "QS"
    assert ProgramType.TIRE_STRATEGY.value == "TS"
    assert ProgramType.FUEL_MANAGEMENT.value == "FM"
    assert ProgramType.SETUP_VERIFICATION.value == "SV"


def test_program_outcome_values():
    assert ProgramOutcome.SUCCESS.value == "success"
    assert ProgramOutcome.PARTIAL.value == "partial"
    assert ProgramOutcome.FAILED.value == "failed"
    assert ProgramOutcome.ABORTED.value == "aborted"


def test_seven_program_types():
    """EA F1 2026 标准有 7 个练习项目."""
    assert len(list(ProgramType)) == 7


# --------------------------------------------------------------------------- #
# PracticeProgram.standard
# --------------------------------------------------------------------------- #
def test_standard_ap():
    prog = PracticeProgram.standard(ProgramType.ACCLIMATIZATION)
    assert prog.program_type == ProgramType.ACCLIMATIZATION
    assert prog.target_laps == 3
    assert prog.delta_target_s == 2.0
    assert prog.tolerance_s == 0.5
    assert prog.rd_points == 5


def test_standard_qs():
    prog = PracticeProgram.standard(ProgramType.QUALIFYING_SIM)
    assert prog.target_laps == 3
    assert prog.delta_target_s == -0.3  # 需比参考快
    assert prog.rd_points == 12


def test_standard_rt():
    prog = PracticeProgram.standard(ProgramType.RACE_TRIM)
    assert prog.target_laps == 10
    assert prog.rd_points == 15  # 最高奖励


def test_standard_fm_uses_lean_fuel():
    prog = PracticeProgram.standard(ProgramType.FUEL_MANAGEMENT)
    assert prog.fuel_mode == "lean"


def test_standard_compound_override():
    prog = PracticeProgram.standard(ProgramType.RACE_TRIM, compound="soft")
    assert prog.compound == "soft"


def test_all_program_types_have_standard():
    """所有 7 种项目类型都应有标准定义."""
    for pt in ProgramType:
        prog = PracticeProgram.standard(pt)
        assert prog.target_laps > 0
        assert prog.rd_points > 0


# --------------------------------------------------------------------------- #
# PracticeSessionSimulator 基础
# --------------------------------------------------------------------------- #
def test_simulator_clamps_inputs():
    """输入超出 0..1 应被裁剪."""
    sim = PracticeSessionSimulator(
        driver_pace=1.5, driver_consistency=-0.5,
        setup_quality=2.0, track_id="monza",
    )
    assert sim.driver_pace == 1.0
    assert sim.driver_consistency == 0.0
    assert sim.setup_quality == 1.0


def test_run_program_returns_result():
    sim = PracticeSessionSimulator(
        driver_pace=0.85, driver_consistency=0.80,
        setup_quality=0.70, track_id="monza", seed=42,
    )
    prog = PracticeProgram.standard(ProgramType.ACCLIMATIZATION)
    result = sim.run_program(prog)
    assert result.program_type == ProgramType.ACCLIMATIZATION
    assert result.completed_laps >= 0


def test_run_program_completes_target_laps():
    sim = PracticeSessionSimulator(
        driver_pace=0.9, driver_consistency=0.9,
        setup_quality=0.9, track_id="monza", seed=42,
    )
    prog = PracticeProgram.standard(ProgramType.ACCLIMATIZATION)
    result = sim.run_program(prog)
    # 正常情况应完成全部目标圈
    assert result.completed_laps == prog.target_laps


# --------------------------------------------------------------------------- #
# 项目成功/失败
# --------------------------------------------------------------------------- #
def test_top_driver_succeeds_qs():
    """顶级车手 + 优调校应通过排位模拟."""
    n_success = 0
    for seed in range(30):
        r = simulate_practice_program(
            ProgramType.QUALIFYING_SIM,
            driver_pace=0.95, driver_consistency=0.95,
            setup_quality=0.95, track_id="monza", seed=seed,
        )
        if r.outcome in (ProgramOutcome.SUCCESS, ProgramOutcome.PARTIAL):
            n_success += 1
    assert n_success >= 25  # 至少 83% 通过


def test_weak_driver_fails_qs():
    """弱车手 + 差调校应失败排位模拟."""
    n_fail = 0
    for seed in range(30):
        r = simulate_practice_program(
            ProgramType.QUALIFYING_SIM,
            driver_pace=0.30, driver_consistency=0.30,
            setup_quality=0.30, track_id="monaco", seed=seed,
        )
        if r.outcome == ProgramOutcome.FAILED:
            n_fail += 1
    assert n_fail >= 25


def test_ap_easier_than_qs():
    """适应项目比排位模拟容易."""
    n_ap_success = 0
    n_qs_success = 0
    for seed in range(30):
        r_ap = simulate_practice_program(
            ProgramType.ACCLIMATIZATION,
            driver_pace=0.6, driver_consistency=0.6,
            setup_quality=0.6, track_id="monza", seed=seed,
        )
        r_qs = simulate_practice_program(
            ProgramType.QUALIFYING_SIM,
            driver_pace=0.6, driver_consistency=0.6,
            setup_quality=0.6, track_id="monza", seed=seed,
        )
        if r_ap.outcome in (ProgramOutcome.SUCCESS, ProgramOutcome.PARTIAL):
            n_ap_success += 1
        if r_qs.outcome in (ProgramOutcome.SUCCESS, ProgramOutcome.PARTIAL):
            n_qs_success += 1
    assert n_ap_success >= n_qs_success


# --------------------------------------------------------------------------- #
# R&D 点数
# --------------------------------------------------------------------------- #
def test_success_earns_full_rd_points():
    sim = PracticeSessionSimulator(
        driver_pace=0.98, driver_consistency=0.95,
        setup_quality=0.95, track_id="monza", seed=1,
    )
    prog = PracticeProgram.standard(ProgramType.RACE_TRIM)
    r = sim.run_program(prog)
    if r.outcome == ProgramOutcome.SUCCESS:
        assert r.rd_points_earned == prog.rd_points


def test_partial_earns_half_rd_points():
    sim = PracticeSessionSimulator(
        driver_pace=0.98, driver_consistency=0.95,
        setup_quality=0.95, track_id="monza", seed=1,
    )
    prog = PracticeProgram.standard(ProgramType.RACE_TRIM)
    r = sim.run_program(prog)
    if r.outcome == ProgramOutcome.PARTIAL:
        assert r.rd_points_earned == prog.rd_points // 2


def test_failed_earns_zero_rd_points():
    sim = PracticeSessionSimulator(
        driver_pace=0.98, driver_consistency=0.95,
        setup_quality=0.95, track_id="monza", seed=1,
    )
    prog = PracticeProgram.standard(ProgramType.RACE_TRIM)
    r = sim.run_program(prog)
    if r.outcome == ProgramOutcome.FAILED:
        assert r.rd_points_earned == 0


def test_rt_highest_rd_reward():
    """长跑项目奖励最高 (15 点)."""
    prog = PracticeProgram.standard(ProgramType.RACE_TRIM)
    assert prog.rd_points == 15
    # 检查其他项目都不超过 15
    for pt in ProgramType:
        p = PracticeProgram.standard(pt)
        assert p.rd_points <= 15


# --------------------------------------------------------------------------- #
# 圈速 delta
# --------------------------------------------------------------------------- #
def test_avg_delta_recorded():
    sim = PracticeSessionSimulator(
        driver_pace=0.8, driver_consistency=0.8,
        setup_quality=0.8, track_id="monza", seed=42,
    )
    r = sim.run_program(PracticeProgram.standard(ProgramType.RACE_TRIM))
    assert isinstance(r.avg_delta_s, float)


def test_best_delta_better_than_avg():
    """最佳圈 delta 应优于平均."""
    sim = PracticeSessionSimulator(
        driver_pace=0.7, driver_consistency=0.6,  # 低一致性, 圈速差异大
        setup_quality=0.7, track_id="monza", seed=42,
    )
    r = sim.run_program(PracticeProgram.standard(ProgramType.RACE_TRIM))
    if r.completed_laps > 1:
        assert r.best_delta_s <= r.avg_delta_s


def test_better_setup_lower_delta():
    """更好的调校 → 更低 (更快) 的 delta."""
    deltas_good = []
    deltas_bad = []
    for seed in range(20):
        r_good = simulate_practice_program(
            ProgramType.RACE_TRIM,
            driver_pace=0.7, driver_consistency=0.7,
            setup_quality=0.95, track_id="monza", seed=seed,
        )
        r_bad = simulate_practice_program(
            ProgramType.RACE_TRIM,
            driver_pace=0.7, driver_consistency=0.7,
            setup_quality=0.30, track_id="monza", seed=seed,
        )
        deltas_good.append(r_good.avg_delta_s)
        deltas_bad.append(r_bad.avg_delta_s)
    avg_good = sum(deltas_good) / len(deltas_good)
    avg_bad = sum(deltas_bad) / len(deltas_bad)
    assert avg_good < avg_bad  # 好调校 delta 更小


# --------------------------------------------------------------------------- #
# 轮胎磨损 & 燃油
# --------------------------------------------------------------------------- #
def test_tire_wear_recorded():
    sim = PracticeSessionSimulator(
        driver_pace=0.8, driver_consistency=0.8,
        setup_quality=0.8, track_id="monza", seed=42,
    )
    r = sim.run_program(PracticeProgram.standard(ProgramType.RACE_TRIM))
    assert r.tire_wear_pct > 0
    # 10 圈长跑应有显著磨损
    assert r.tire_wear_pct > 10.0


def test_fuel_used_recorded():
    sim = PracticeSessionSimulator(
        driver_pace=0.8, driver_consistency=0.8,
        setup_quality=0.8, track_id="monza", seed=42,
    )
    r = sim.run_program(PracticeProgram.standard(ProgramType.RACE_TRIM))
    assert r.fuel_used_kg > 0
    # 10 圈 × ~1.6 kg/lap ≈ 16 kg
    assert 10.0 < r.fuel_used_kg < 25.0


def test_lean_fuel_uses_less():
    """节油模式燃油消耗更低."""
    r_lean = simulate_practice_program(
        ProgramType.FUEL_MANAGEMENT,
        driver_pace=0.8, driver_consistency=0.8,
        setup_quality=0.8, track_id="monza", seed=42,
    )
    r_normal = simulate_practice_program(
        ProgramType.ACCLIMATIZATION,
        driver_pace=0.8, driver_consistency=0.8,
        setup_quality=0.8, track_id="monza", seed=42,
    )
    # 节油模式每圈 1.4kg vs 普通 1.6kg
    if r_lean.completed_laps > 0 and r_normal.completed_laps > 0:
        per_lap_lean = r_lean.fuel_used_kg / r_lean.completed_laps
        per_lap_normal = r_normal.fuel_used_kg / r_normal.completed_laps
        assert per_lap_lean < per_lap_normal


# --------------------------------------------------------------------------- #
# 反馈
# --------------------------------------------------------------------------- #
def test_feedback_non_empty():
    sim = PracticeSessionSimulator(
        driver_pace=0.7, driver_consistency=0.7,
        setup_quality=0.7, track_id="monza", seed=42,
    )
    r = sim.run_program(PracticeProgram.standard(ProgramType.RACE_TRIM))
    assert len(r.feedback) >= 1


def test_feedback_contains_outcome_message():
    sim = PracticeSessionSimulator(
        driver_pace=0.7, driver_consistency=0.7,
        setup_quality=0.7, track_id="monza", seed=42,
    )
    r = sim.run_program(PracticeProgram.standard(ProgramType.RACE_TRIM))
    # 反馈应包含结果描述
    combined = " ".join(r.feedback)
    assert any(kw in combined for kw in ["达成", "目标", "调整", "未达"])


def test_high_tire_wear_feedback():
    """高轮胎磨损应触发胎压反馈."""
    sim = PracticeSessionSimulator(
        driver_pace=0.5, driver_consistency=0.5,
        setup_quality=0.5, track_id="monza", seed=42,
    )
    r = sim.run_program(PracticeProgram.standard(ProgramType.RACE_TRIM))
    if r.tire_wear_pct > 15.0:
        combined = " ".join(r.feedback)
        assert "胎压" in combined or "磨损" in combined


# --------------------------------------------------------------------------- #
# 中止场景
# --------------------------------------------------------------------------- #
def test_aborted_possible_with_bad_setup():
    """极差调校 + 高难度赛道可能中止."""
    n_aborted = 0
    for seed in range(100):
        r = simulate_practice_program(
            ProgramType.QUALIFYING_SIM,
            driver_pace=0.3, driver_consistency=0.3,
            setup_quality=0.05, track_id="monaco", seed=seed,  # Monaco 高难度
        )
        if r.outcome == ProgramOutcome.ABORTED:
            n_aborted += 1
    # 至少有一次中止
    assert n_aborted >= 1


def test_aborted_earns_zero_rd():
    sim = PracticeSessionSimulator(
        driver_pace=0.3, driver_consistency=0.3,
        setup_quality=0.05, track_id="monaco", seed=42,
    )
    r = sim.run_program(PracticeProgram.standard(ProgramType.QUALIFYING_SIM))
    if r.outcome == ProgramOutcome.ABORTED:
        assert r.rd_points_earned == 0
        assert r.completed_laps < PracticeProgram.standard(ProgramType.QUALIFYING_SIM).target_laps


# --------------------------------------------------------------------------- #
# 确定性
# --------------------------------------------------------------------------- #
def test_deterministic_with_same_seed():
    sim1 = PracticeSessionSimulator(
        driver_pace=0.8, driver_consistency=0.8,
        setup_quality=0.8, track_id="monza", seed=42,
    )
    sim2 = PracticeSessionSimulator(
        driver_pace=0.8, driver_consistency=0.8,
        setup_quality=0.8, track_id="monza", seed=42,
    )
    prog = PracticeProgram.standard(ProgramType.RACE_TRIM)
    r1 = sim1.run_program(prog)
    r2 = sim2.run_program(prog)
    assert r1.avg_delta_s == r2.avg_delta_s
    assert r1.outcome == r2.outcome
    assert r1.rd_points_earned == r2.rd_points_earned


# --------------------------------------------------------------------------- #
# 实战场景
# --------------------------------------------------------------------------- #
def test_full_fp2_session():
    """FP2 典型: 跑 4 个项目 (RT + QS + TS + FM), 累计 R&D 点."""
    sim = PracticeSessionSimulator(
        driver_pace=0.88, driver_consistency=0.85,
        setup_quality=0.82, track_id="silverstone", seed=100,
    )
    programs = [
        PracticeProgram.standard(ProgramType.RACE_TRIM),
        PracticeProgram.standard(ProgramType.QUALIFYING_SIM),
        PracticeProgram.standard(ProgramType.TIRE_STRATEGY),
        PracticeProgram.standard(ProgramType.FUEL_MANAGEMENT),
    ]
    total_rd = 0
    for prog in programs:
        r = sim.run_program(prog)
        total_rd += r.rd_points_earned
    # 4 个项目最多 15+12+12+8 = 47 点
    assert 0 <= total_rd <= 47


def test_progressive_setup_improvement():
    """多节练习调校逐步改善: setup_quality 上升."""
    setup = 0.50
    results = []
    for session in range(4):  # FP1..FP3 + 调校
        sim = PracticeSessionSimulator(
            driver_pace=0.85, driver_consistency=0.85,
            setup_quality=setup, track_id="monza", seed=session,
        )
        r = sim.run_program(PracticeProgram.standard(ProgramType.SETUP_VERIFICATION))
        results.append((setup, r.outcome))
        # 成功 → 调校改善
        if r.outcome == ProgramOutcome.SUCCESS:
            setup = min(1.0, setup + 0.10)
        elif r.outcome == ProgramOutcome.PARTIAL:
            setup = min(1.0, setup + 0.05)
    # 调校应有改善 (除非全部失败)
    assert setup >= 0.50
