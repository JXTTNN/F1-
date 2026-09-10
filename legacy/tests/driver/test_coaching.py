"""DriverCoach / LearningPath / SkillAssessment 单元测试 — 教练与学习模块.

覆盖: 短板评估、计划生成与结构、难度映射、进度追踪、计划迭代、激励话术、
学习路径阶段与达标、技能评估分解与等级、原型对比、空数据容错与确定性.
"""

from __future__ import annotations

import copy

from f1opt.driver.coaching import (
    LEVEL_THRESHOLDS,
    CoachingPlan,
    DriverCoach,
    LearningPath,
    SkillAssessment,
)
from f1opt.driver.profile import (
    AGGRESSIVE_PROFILE,
    CONSERVATIVE_PROFILE,
    DEFAULT_PROFILE,
    DriverProfile,
)


# --- 测试夹具 --------------------------------------------------------------
def _lap_metrics(n: int = 4, base: float = 90.0) -> list[dict]:
    """生成 n 圈单圈指标 (lap_time/sector_times/输入平滑度/制动进攻性)."""
    out: list[dict] = []
    for i in range(n):
        out.append({
            "lap_time": base + 0.2 * i,
            "sector_times": [30.0 + 0.05 * i, 28.5 + 0.1 * i, 31.5 + 0.05 * i],
            "throttle_smoothness": 0.55,
            "brake_aggression": 0.7,
            "tire_wear_score": 0.35,
        })
    return out


def _consistent_laps(n: int = 4, base: float = 90.0) -> list[dict]:
    """生成高度一致的 n 圈 (低 CV, 高平顺度)."""
    return [
        {
            "lap_time": base + 0.01 * i,
            "sector_times": [30.0, 28.5, 31.5],
            "throttle_smoothness": 0.85,
            "brake_aggression": 0.4,
            "tire_wear_score": 0.2,
        }
        for i in range(n)
    ]


# --- DriverCoach.assess_weaknesses ----------------------------------------
def test_assess_weaknesses_returns_at_most_three_strings() -> None:
    """assess_weaknesses 返回不超过 3 个字符串。"""
    coach = DriverCoach(DEFAULT_PROFILE, archetype="RACE_CRAFT")
    result = coach.assess_weaknesses(_lap_metrics())
    assert isinstance(result, list)
    assert len(result) <= 3
    assert all(isinstance(w, str) for w in result)


def test_assess_weaknesses_empty_laps_returns_defaults() -> None:
    """空圈数据返回固定默认短板 (不崩溃)."""
    coach = DriverCoach(DEFAULT_PROFILE)
    result = coach.assess_weaknesses([])
    assert isinstance(result, list)
    assert 1 <= len(result) <= 3
    assert all(isinstance(w, str) for w in result)


def test_assess_weaknesses_identifies_throttle_weakness() -> None:
    """throttle_smoothness 偏低时识别出弯短板。"""
    coach = DriverCoach(DEFAULT_PROFILE)
    laps = _lap_metrics()
    for lm in laps:
        lm["throttle_smoothness"] = 0.3  # 偏低 → corner_exit_speed
    result = coach.assess_weaknesses(laps)
    assert "corner_exit_speed" in result


# --- DriverCoach.generate_plan --------------------------------------------
def test_generate_plan_returns_coaching_plan_with_required_fields() -> None:
    """generate_plan 返回 CoachingPlan 且含 focus_areas/exercises/targets。"""
    coach = DriverCoach(DEFAULT_PROFILE, archetype="RACE_CRAFT")
    plan = coach.generate_plan(_lap_metrics())
    assert isinstance(plan, CoachingPlan)
    assert isinstance(plan.focus_areas, list)
    assert isinstance(plan.exercises, list)
    assert isinstance(plan.targets, dict)
    assert len(plan.focus_areas) > 0


def test_generate_plan_exercises_are_dicts_with_required_keys() -> None:
    """每项练习为字典, 含 name/description/target_metric/target_value/duration_laps。"""
    coach = DriverCoach(DEFAULT_PROFILE)
    plan = coach.generate_plan(_lap_metrics())
    required = {"name", "description", "target_metric", "target_value", "duration_laps"}
    assert len(plan.exercises) > 0
    for ex in plan.exercises:
        assert isinstance(ex, dict)
        assert required <= set(ex.keys()), ex
        assert isinstance(ex["name"], str)
        assert isinstance(ex["description"], str)
        # 描述为中文 (含 CJK 字符).
        assert any("\u4e00" <= ch <= "\u9fff" for ch in ex["description"])
        assert isinstance(ex["duration_laps"], int)
        assert ex["duration_laps"] > 0


def test_generate_plan_difficulty_is_valid() -> None:
    """difficulty 必为 easy/medium/hard 之一。"""
    coach = DriverCoach(DEFAULT_PROFILE)
    plan = coach.generate_plan(_lap_metrics())
    assert plan.difficulty in ("easy", "medium", "hard")


def test_generate_plan_development_archetype_is_easy() -> None:
    """DEVELOPMENT 原型 → easy 难度。"""
    coach = DriverCoach(DEFAULT_PROFILE, archetype="DEVELOPMENT")
    plan = coach.generate_plan(_lap_metrics())
    assert plan.difficulty == "easy"


def test_generate_plan_aggressive_archetype_is_hard() -> None:
    """AGGRESSIVE 原型 → hard 难度。"""
    coach = DriverCoach(DEFAULT_PROFILE, archetype="AGGRESSIVE")
    plan = coach.generate_plan(_lap_metrics())
    assert plan.difficulty == "hard"


def test_generate_plan_race_craft_archetype_is_medium() -> None:
    """RACE_CRAFT 原型 → medium 难度。"""
    coach = DriverCoach(DEFAULT_PROFILE, archetype="RACE_CRAFT")
    plan = coach.generate_plan(_lap_metrics())
    assert plan.difficulty == "medium"


def test_generate_plan_targets_match_weaknesses() -> None:
    """目标指标与短板练习的 target_metric 一致。"""
    coach = DriverCoach(DEFAULT_PROFILE, archetype="RACE_CRAFT")
    plan = coach.generate_plan(_lap_metrics())
    exercise_metrics = {ex["target_metric"] for ex in plan.exercises}
    target_metrics = set(plan.targets.keys())
    # 每个练习的 target_metric 应在 plan.targets 中.
    assert exercise_metrics <= target_metrics


# --- CoachingPlan.to_dict --------------------------------------------------
def test_coaching_plan_to_dict_roundtrip() -> None:
    """to_dict 返回纯字典且包含全部字段。"""
    coach = DriverCoach(DEFAULT_PROFILE)
    plan = coach.generate_plan(_lap_metrics())
    d = plan.to_dict()
    assert isinstance(d, dict)
    for key in ("focus_areas", "exercises", "targets", "duration_laps", "difficulty"):
        assert key in d


# --- DriverCoach.track_progress -------------------------------------------
def test_track_progress_returns_dict_with_required_keys() -> None:
    """track_progress 返回含 targets_met/targets_total/progress_pct 的字典。"""
    coach = DriverCoach(DEFAULT_PROFILE)
    plan = coach.generate_plan(_lap_metrics())
    progress = coach.track_progress(_lap_metrics(), plan)
    assert isinstance(progress, dict)
    for key in ("targets_met", "targets_total", "progress_pct"):
        assert key in progress
    assert isinstance(progress["targets_met"], int)
    assert isinstance(progress["targets_total"], int)
    assert isinstance(progress["progress_pct"], float)


def test_track_progress_pct_in_range() -> None:
    """progress_pct ∈ [0, 100]。"""
    coach = DriverCoach(DEFAULT_PROFILE)
    plan = coach.generate_plan(_lap_metrics())
    progress = coach.track_progress(_lap_metrics(), plan)
    assert 0.0 <= progress["progress_pct"] <= 100.0
    assert 0 <= progress["targets_met"] <= progress["targets_total"]


def test_track_progress_areas_are_lists_of_strings() -> None:
    """areas_improved / areas_regressed 为字符串列表。"""
    coach = DriverCoach(DEFAULT_PROFILE)
    plan = coach.generate_plan(_lap_metrics())
    progress = coach.track_progress(_lap_metrics(), plan)
    assert isinstance(progress["areas_improved"], list)
    assert isinstance(progress["areas_regressed"], list)
    assert all(isinstance(a, str) for a in progress["areas_improved"])
    assert all(isinstance(a, str) for a in progress["areas_regressed"])


def test_track_progress_recommendation_is_chinese() -> None:
    """recommendation 为非空中文串。"""
    coach = DriverCoach(DEFAULT_PROFILE)
    plan = coach.generate_plan(_lap_metrics())
    progress = coach.track_progress(_lap_metrics(), plan)
    rec = progress["recommendation"]
    assert isinstance(rec, str)
    assert len(rec) > 0
    assert any("\u4e00" <= ch <= "\u9fff" for ch in rec)


# --- DriverCoach.next_plan -------------------------------------------------
def test_next_plan_returns_new_coaching_plan() -> None:
    """next_plan 返回新的 CoachingPlan。"""
    coach = DriverCoach(DEFAULT_PROFILE, archetype="RACE_CRAFT")
    plan = coach.generate_plan(_lap_metrics())
    nxt = coach.next_plan(plan, _lap_metrics())
    assert isinstance(nxt, CoachingPlan)
    assert nxt.difficulty in ("easy", "medium", "hard")
    assert nxt is not plan


def test_next_plan_advances_on_full_completion() -> None:
    """全部目标达标 → 难度递进 (easy → medium)。"""
    coach = DriverCoach(DEFAULT_PROFILE, archetype="DEVELOPMENT")  # easy 起步
    # 手工构造全部可达标的计划 (基于 _consistent_laps 的实际指标).
    plan = CoachingPlan(
        focus_areas=["braking_consistency", "corner_exit_speed", "lap_time_consistency"],
        exercises=[],
        targets={
            "brake_aggression_cv": 0.05,   # 0 <= 0.05 ✓
            "throttle_smoothness": 0.75,   # 0.85 >= 0.75 ✓
            "lap_time_cv": 0.02,           # ~0.0001 <= 0.02 ✓
        },
        duration_laps=20,
        difficulty="easy",
    )
    nxt = coach.next_plan(plan, _consistent_laps())
    # 全部达标 → 难度由 easy 递进到 medium.
    assert nxt.difficulty == "medium"


def test_next_plan_regresses_on_no_completion() -> None:
    """完全未达标 → 难度下调 (medium → easy)。"""
    coach = DriverCoach(DEFAULT_PROFILE, archetype="RACE_CRAFT")  # medium 起步
    plan = CoachingPlan(
        focus_areas=["corner_exit_speed"],
        exercises=[],
        targets={
            "throttle_smoothness": 0.95,   # 0.85 < 0.95, 未达标
            "lap_time_cv": 0.001,          # 阈值极低, 难以达标
        },
        duration_laps=15,
        difficulty="medium",
    )
    # 空数据 → 全部目标未达标 → 难度下调到 easy.
    nxt = coach.next_plan(plan, [])
    assert nxt.difficulty == "easy"


# --- DriverCoach.motivational_message -------------------------------------
def test_motivational_message_returns_nonempty_chinese() -> None:
    """motivational_message 返回非空中文串。"""
    coach = DriverCoach(DEFAULT_PROFILE)
    plan = coach.generate_plan(_lap_metrics())
    progress = coach.track_progress(_lap_metrics(), plan)
    msg = coach.motivational_message(progress)
    assert isinstance(msg, str)
    assert len(msg) > 0
    assert any("\u4e00" <= ch <= "\u9fff" for ch in msg)


def test_motivational_message_handles_zero_progress() -> None:
    """0% 进度也返回非空中文激励。"""
    coach = DriverCoach(DEFAULT_PROFILE)
    msg = coach.motivational_message({"progress_pct": 0.0})
    assert isinstance(msg, str)
    assert any("\u4e00" <= ch <= "\u9fff" for ch in msg)


# --- LearningPath ----------------------------------------------------------
def test_learning_path_stages_count_in_range() -> None:
    """stages 返回 5-7 个阶段。"""
    path = LearningPath("RACE_CRAFT")
    stages = path.stages()
    assert isinstance(stages, list)
    assert 5 <= len(stages) <= 7


def test_learning_path_stages_have_required_keys() -> None:
    """每个阶段含 stage/name/focus/exercises 键。"""
    path = LearningPath("RACE_CRAFT")
    stages = path.stages()
    required = {"stage", "name", "focus", "exercises"}
    for s in stages:
        assert required <= set(s.keys()), s
        assert isinstance(s["stage"], int)
        assert isinstance(s["name"], str)
        assert isinstance(s["focus"], str)
        assert isinstance(s["exercises"], list)
        # 阶段名含中文.
        assert any("\u4e00" <= ch <= "\u9fff" for ch in s["name"])


def test_learning_path_current_stage_returns_next_incomplete() -> None:
    """current_stage 返回下一个未完成阶段。"""
    path = LearningPath("RACE_CRAFT")
    stages = path.stages()
    # 无已完成 → 第 1 阶段.
    first = path.current_stage([])
    assert first["stage"] == stages[0]["stage"]
    # 完成第 1 阶段 → 第 2 阶段.
    second = path.current_stage([stages[0]["stage"]])
    assert second["stage"] == stages[1]["stage"]
    # 完成全部 → 末阶段 (不崩溃).
    all_done = path.current_stage([s["stage"] for s in stages])
    assert all_done["stage"] == stages[-1]["stage"]


def test_learning_path_assess_stage_completion_returns_bool() -> None:
    """assess_stage_completion 返回布尔值。"""
    path = LearningPath("RACE_CRAFT")
    stage = path.stages()[0]
    result = path.assess_stage_completion(stage, _lap_metrics())
    assert isinstance(result, bool)
    # 空数据 → False.
    assert path.assess_stage_completion(stage, []) is False


def test_learning_path_path_summary_returns_nonempty_chinese() -> None:
    """path_summary 返回非空中文描述。"""
    path = LearningPath("RACE_CRAFT")
    summary = path.path_summary()
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert any("\u4e00" <= ch <= "\u9fff" for ch in summary)


# --- SkillAssessment -------------------------------------------------------
def test_skill_assessment_returns_dict_with_overall_and_breakdown() -> None:
    """assess 返回含 overall_skill + skill_breakdown 的字典。"""
    sa = SkillAssessment()
    result = sa.assess(_lap_metrics(), DEFAULT_PROFILE)
    assert isinstance(result, dict)
    assert "overall_skill" in result
    assert "skill_breakdown" in result


def test_skill_assessment_overall_in_range() -> None:
    """overall_skill ∈ [0, 1]。"""
    sa = SkillAssessment()
    result = sa.assess(_lap_metrics(), DEFAULT_PROFILE)
    assert 0.0 <= result["overall_skill"] <= 1.0


def test_skill_assessment_breakdown_has_all_five_areas() -> None:
    """skill_breakdown 含全部 5 个技能维度且 ∈ [0,1]。"""
    sa = SkillAssessment()
    result = sa.assess(_lap_metrics(), DEFAULT_PROFILE)
    bd = result["skill_breakdown"]
    expected = {"braking", "cornering", "consistency", "racecraft", "tire_mgmt"}
    assert set(bd.keys()) == expected
    for v in bd.values():
        assert 0.0 <= v <= 1.0


def test_skill_assessment_level_is_valid() -> None:
    """skill_level 为 5 个等级之一。"""
    sa = SkillAssessment()
    valid_levels = set(LEVEL_THRESHOLDS.keys())
    result = sa.assess(_lap_metrics(), DEFAULT_PROFILE)
    assert result["skill_level"] in valid_levels


def test_skill_assessment_expert_profile_high_level() -> None:
    """高质量画像 + 一致数据 → 等级不低于 INTERMEDIATE。"""
    sa = SkillAssessment()
    expert_profile = DriverProfile(
        throttle_smoothness=0.9,
        steer_smoothness=0.85,
        aggression_score=0.6,
        consistency_score=0.9,
        drs_usage_efficiency=0.85,
    )
    result = sa.assess(_consistent_laps(), expert_profile)
    assert result["skill_level"] in ("INTERMEDIATE", "ADVANCED", "EXPERT")


def test_skill_assessment_strengths_weaknesses_are_string_lists() -> None:
    """strengths / weaknesses 为字符串列表。"""
    sa = SkillAssessment()
    result = sa.assess(_lap_metrics(), DEFAULT_PROFILE)
    assert isinstance(result["strengths"], list)
    assert isinstance(result["weaknesses"], list)
    assert all(isinstance(s, str) for s in result["strengths"])
    assert all(isinstance(w, str) for w in result["weaknesses"])
    # 非空且有中文.
    assert len(result["strengths"]) > 0
    assert any("\u4e00" <= ch <= "\u9fff" for ch in result["strengths"][0])


def test_skill_assessment_compare_to_archetype_returns_dict() -> None:
    """compare_to_archetype 返回字典且含差距信息。"""
    sa = SkillAssessment()
    skill = sa.assess(_lap_metrics(), DEFAULT_PROFILE)
    cmp = sa.compare_to_archetype(skill, "RACE_CRAFT")
    assert isinstance(cmp, dict)
    assert "archetype" in cmp
    assert "gaps" in cmp
    assert isinstance(cmp["gaps"], dict)
    assert set(cmp["gaps"].keys()) == {
        "braking", "cornering", "consistency", "racecraft", "tire_mgmt"
    }


# --- 边界与确定性 ----------------------------------------------------------
def test_empty_lap_metrics_no_crash_sensible_defaults() -> None:
    """空 lap_metrics: 全部接口不崩溃并返回合理默认。"""
    coach = DriverCoach(DEFAULT_PROFILE, archetype="RACE_CRAFT")
    # assess_weaknesses.
    w = coach.assess_weaknesses([])
    assert isinstance(w, list) and len(w) <= 3
    # generate_plan.
    plan = coach.generate_plan([])
    assert isinstance(plan, CoachingPlan)
    assert plan.difficulty in ("easy", "medium", "hard")
    # track_progress.
    progress = coach.track_progress([], plan)
    assert progress["progress_pct"] == 0.0
    assert progress["targets_total"] == len(plan.targets)
    # next_plan.
    nxt = coach.next_plan(plan, [])
    assert isinstance(nxt, CoachingPlan)
    # motivational_message.
    msg = coach.motivational_message(progress)
    assert isinstance(msg, str) and len(msg) > 0
    # SkillAssessment on empty data.
    sa = SkillAssessment()
    res = sa.assess([], DEFAULT_PROFILE)
    assert 0.0 <= res["overall_skill"] <= 1.0


def test_determinism_same_inputs_same_plan() -> None:
    """相同输入 → 相同计划 (确定性)."""
    laps = _lap_metrics()
    coach_a = DriverCoach(DEFAULT_PROFILE, archetype="RACE_CRAFT", track_id="melbourne")
    coach_b = DriverCoach(DEFAULT_PROFILE, archetype="RACE_CRAFT", track_id="melbourne")
    plan_a = coach_a.generate_plan(laps)
    plan_b = coach_b.generate_plan(laps)
    assert plan_a.focus_areas == plan_b.focus_areas
    assert plan_a.targets == plan_b.targets
    assert plan_a.difficulty == plan_b.difficulty
    assert plan_a.exercises == plan_b.exercises
    # 进度也确定.
    pa = coach_a.track_progress(laps, plan_a)
    pb = coach_b.track_progress(laps, plan_b)
    assert pa == pb


def test_determinism_weaknesses_stable_across_calls() -> None:
    """assess_weaknesses 多次调用结果一致。"""
    coach = DriverCoach(AGGRESSIVE_PROFILE)
    laps = _lap_metrics()
    r1 = coach.assess_weaknesses(laps)
    r2 = coach.assess_weaknesses(copy.deepcopy(laps))
    assert r1 == r2


def test_different_profiles_yield_different_skill_levels() -> None:
    """保守画像 vs 进攻画像在轮胎管理上应有差异。"""
    sa = SkillAssessment()
    laps = _consistent_laps()
    cons = sa.assess(laps, CONSERVATIVE_PROFILE)
    aggr = sa.assess(laps, AGGRESSIVE_PROFILE)
    # 保守画像轮胎管理应优于进攻画像 (进攻高 → 胎耗大).
    assert cons["skill_breakdown"]["tire_mgmt"] >= aggr["skill_breakdown"]["tire_mgmt"]
