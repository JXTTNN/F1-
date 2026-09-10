"""DeepDriverProfiler 单元测试 — 风格原型/弯道相位/一致性/适应/疲劳/综合画像.

覆盖: 原型分类全分支、四相位识别与空帧容错、一致性 CV 与趋势、适应强度区间、
疲劳单调性、综合画像结构完整性与确定性.
"""

from __future__ import annotations

import copy

import pytest

from f1opt.driver.deep_profile import (
    AdaptationProfile,
    CornerPhaseAnalysis,
    DeepDriverProfiler,
    DriverConsistencyAnalyzer,
    DrivingStyleArchetype,
    FatigueModel,
    analyze_corner_phases,
    classify_archetype,
)

_DT = 1.0 / 60.0


# --- 测试夹具 --------------------------------------------------------------
def _make_corner_lap(
    n_straight: int = 20,
    n_braking: int = 15,
    n_trail: int = 10,
    n_mid: int = 8,
    n_exit: int = 12,
) -> list[dict]:
    """生成一段含 4 相位的单弯遥测帧 (60Hz)."""
    frames: list[dict] = []
    t = 0.0
    dist = 0.0

    def add(thr: float, brk: float, st: float, glat: float, n: int) -> None:
        nonlocal t, dist
        for _ in range(n):
            frames.append({
                "session_time": t,
                "lap_distance": dist,
                "throttle": thr,
                "brake": brk,
                "steer": st,
                "g_lat": glat,
                "ers_deploy_mode": 0,
                "ers_store": 0.5,
                "drs_allowed": 0,
            })
            t += _DT
            dist += 50.0 * _DT

    add(1.0, 0.0, 0.0, 0.0, n_straight)   # 直道
    add(0.0, 1.0, 0.0, 0.0, n_braking)    # 制动
    add(0.0, 0.2, 0.5, 3.0, n_trail)      # trail-braking
    add(0.0, 0.0, 0.8, 5.0, n_mid)        # 弯心
    add(0.8, 0.0, 0.4, 2.0, n_exit)       # 出弯
    add(1.0, 0.0, 0.0, 0.0, n_straight)   # 直道
    return frames


def _lap_metrics(n: int = 4, base: float = 90.0) -> list[dict]:
    """生成 n 圈单圈指标 (lap_time/sector_times/输入平滑度)."""
    out: list[dict] = []
    for i in range(n):
        out.append({
            "lap_time": base + 0.1 * i,
            "sector_times": [30.0, 30.0 + 0.05 * i, 30.0 + 0.05 * i],
            "throttle_smoothness": 0.8,
            "brake_aggression": 0.3,
        })
    return out


# --- DrivingStyleArchetype -------------------------------------------------
def test_archetype_all_values_accessible() -> None:
    """7 种原型成员均可访问。"""
    expected = {
        DrivingStyleArchetype.SMOOTH_OPERATOR,
        DrivingStyleArchetype.QUALIFIER,
        DrivingStyleArchetype.RACE_CRAFT,
        DrivingStyleArchetype.TIRE_WHISPERER,
        DrivingStyleArchetype.AGGRESSIVE_OVERTAKER,
        DrivingStyleArchetype.DEVELOPMENT,
        DrivingStyleArchetype.WET_SPECIALIST,
    }
    assert set(DrivingStyleArchetype) == expected
    assert len(expected) == 7


def test_archetype_qualifier_high_aggression_low_consistency() -> None:
    """高进攻 + 低一致 → QUALIFIER。"""
    m = {
        "aggression_score": 0.9,
        "consistency_score": 0.3,
        "throttle_smoothness": 0.7,
        "steer_smoothness": 0.5,
        "ers_usage_intensity": 0.3,
    }
    assert classify_archetype(m) is DrivingStyleArchetype.QUALIFIER


def test_archetype_smooth_operator() -> None:
    """高平顺 + 低进攻 + 高一致 → SMOOTH_OPERATOR。"""
    m = {
        "aggression_score": 0.3,
        "consistency_score": 0.8,
        "throttle_smoothness": 0.85,
        "steer_smoothness": 0.8,
        "ers_usage_intensity": 0.3,
    }
    assert classify_archetype(m) is DrivingStyleArchetype.SMOOTH_OPERATOR


def test_archetype_aggressive_overtaker() -> None:
    """高进攻 + 高 ERS + 低平顺 → AGGRESSIVE_OVERTAKER。"""
    m = {
        "aggression_score": 0.9,
        "consistency_score": 0.5,
        "throttle_smoothness": 0.3,
        "steer_smoothness": 0.3,
        "ers_usage_intensity": 0.8,
    }
    assert classify_archetype(m) is DrivingStyleArchetype.AGGRESSIVE_OVERTAKER


def test_archetype_tire_whisperer() -> None:
    """高平顺 + 低进攻 (但一致不足) → TIRE_WHISPERER。"""
    m = {
        "aggression_score": 0.4,
        "consistency_score": 0.5,
        "throttle_smoothness": 0.85,
        "steer_smoothness": 0.8,
        "ers_usage_intensity": 0.3,
    }
    assert classify_archetype(m) is DrivingStyleArchetype.TIRE_WHISPERER


def test_archetype_development_low_consistency() -> None:
    """低一致 (非高进攻) → DEVELOPMENT。"""
    m = {
        "aggression_score": 0.4,
        "consistency_score": 0.3,
        "throttle_smoothness": 0.5,
        "steer_smoothness": 0.5,
        "ers_usage_intensity": 0.3,
    }
    assert classify_archetype(m) is DrivingStyleArchetype.DEVELOPMENT


def test_archetype_race_craft_default() -> None:
    """均衡 + 高一致 → RACE_CRAFT。"""
    m = {
        "aggression_score": 0.5,
        "consistency_score": 0.8,
        "throttle_smoothness": 0.5,
        "steer_smoothness": 0.5,
        "ers_usage_intensity": 0.5,
    }
    assert classify_archetype(m) is DrivingStyleArchetype.RACE_CRAFT


def test_archetype_wet_specialist() -> None:
    """wet_performance > 0.75 → WET_SPECIALIST。"""
    m = {
        "aggression_score": 0.5,
        "consistency_score": 0.8,
        "throttle_smoothness": 0.5,
        "steer_smoothness": 0.5,
        "wet_performance": 0.9,
    }
    assert classify_archetype(m) is DrivingStyleArchetype.WET_SPECIALIST


# --- CornerPhaseAnalysis ---------------------------------------------------
def test_corner_phases_four_phases_identified() -> None:
    """4 相位键均存在且 corners_detected >= 1。"""
    res = analyze_corner_phases(_make_corner_lap())
    for p in ("braking", "trail_braking", "mid_corner", "exit"):
        assert p in res
    assert res["corners_detected"] >= 1


def test_corner_phases_required_keys() -> None:
    """每个相位含 duration_s/peak_value/smoothness/consistency_across_corners。"""
    res = analyze_corner_phases(_make_corner_lap())
    required = {"duration_s", "peak_value", "smoothness", "consistency_across_corners"}
    for p in ("braking", "trail_braking", "mid_corner", "exit"):
        assert required <= set(res[p].keys()), p
        assert res[p]["duration_s"] >= 0.0
        assert 0.0 <= res[p]["peak_value"] <= 1.0
        assert 0.0 <= res[p]["smoothness"] <= 1.0
        assert 0.0 <= res[p]["consistency_across_corners"] <= 1.0


def test_corner_phases_weak_phase_detected() -> None:
    """检测到薄弱相位 (非 None, 属于 4 相位之一)."""
    res = analyze_corner_phases(_make_corner_lap())
    wp = res["weak_phase"]
    assert wp is not None
    assert wp in ("braking", "trail_braking", "mid_corner", "exit")


def test_corner_phases_phase_durations_nonzero() -> None:
    """4 相位持续时长均 > 0 (测试夹具覆盖全部相位)."""
    res = analyze_corner_phases(_make_corner_lap())
    for p in ("braking", "trail_braking", "mid_corner", "exit"):
        assert res[p]["duration_s"] > 0.0, p


def test_corner_phases_empty_frames_graceful() -> None:
    """空帧不崩溃, weak_phase 为 None。"""
    res = analyze_corner_phases([])
    assert res["corners_detected"] == 0
    assert res["weak_phase"] is None
    for p in ("braking", "trail_braking", "mid_corner", "exit"):
        assert res[p]["duration_s"] == 0.0


def test_corner_phase_analysis_dataclass_from_frames() -> None:
    """CornerPhaseAnalysis.from_frames 正确封装结果。"""
    cpa = CornerPhaseAnalysis.from_frames(_make_corner_lap())
    assert cpa.corners_detected >= 1
    assert set(cpa.phases.keys()) == {"braking", "trail_braking", "mid_corner", "exit"}
    assert cpa.weak_phase is not None


# --- DriverConsistencyAnalyzer --------------------------------------------
def test_consistency_lap_time_cv_computed() -> None:
    """lap_time_cv 为非负有限浮点。"""
    res = DriverConsistencyAnalyzer().analyze(_lap_metrics(4))
    cv = res["lap_time_cv"]
    assert isinstance(cv, float)
    assert 0.0 <= cv < float("inf")


def test_consistency_score_in_range() -> None:
    """overall_consistency_score in [0,1]。"""
    res = DriverConsistencyAnalyzer().analyze(_lap_metrics(4))
    assert 0.0 <= res["overall_consistency_score"] <= 1.0


def test_consistency_label_chinese_ranges() -> None:
    """4 段中文标签均覆盖。"""
    label = DriverConsistencyAnalyzer.consistency_label
    assert label(0.9) == "高度一致"
    assert label(0.7) == "较为一致"
    assert label(0.5) == "波动较大"
    assert label(0.2) == "不稳定"


def test_consistency_weak_sector_identified() -> None:
    """sector_times 中某段方差大 → weak_sector 指向该段。"""
    laps = [
        {"lap_time": 90.0, "sector_times": [30.0, 30.0, 30.0],
         "throttle_smoothness": 0.8, "brake_aggression": 0.3},
        {"lap_time": 90.5, "sector_times": [30.0, 35.0, 30.0],
         "throttle_smoothness": 0.79, "brake_aggression": 0.31},
        {"lap_time": 90.3, "sector_times": [30.0, 32.0, 30.0],
         "throttle_smoothness": 0.81, "brake_aggression": 0.29},
        {"lap_time": 90.4, "sector_times": [30.0, 33.0, 30.0],
         "throttle_smoothness": 0.80, "brake_aggression": 0.30},
    ]
    res = DriverConsistencyAnalyzer().analyze(laps)
    assert res["weak_sector"] == 1


def test_consistency_trend_improving() -> None:
    """圈速递减 → improving。"""
    laps = [
        {"lap_time": lt, "sector_times": [30, 30, 30],
         "throttle_smoothness": 0.8, "brake_aggression": 0.3}
        for lt in (92.0, 91.0, 90.0, 89.0)
    ]
    assert DriverConsistencyAnalyzer().analyze(laps)["trend"] == "improving"


def test_consistency_trend_degrading() -> None:
    """圈速递增 → degrading。"""
    laps = [
        {"lap_time": lt, "sector_times": [30, 30, 30],
         "throttle_smoothness": 0.8, "brake_aggression": 0.3}
        for lt in (89.0, 90.0, 91.0, 92.0)
    ]
    assert DriverConsistencyAnalyzer().analyze(laps)["trend"] == "degrading"


def test_consistency_trend_stable() -> None:
    """圈速恒定 → stable。"""
    laps = [
        {"lap_time": 90.0, "sector_times": [30, 30, 30],
         "throttle_smoothness": 0.8, "brake_aggression": 0.3}
        for _ in range(4)
    ]
    assert DriverConsistencyAnalyzer().analyze(laps)["trend"] == "stable"


def test_consistency_insufficient_laps() -> None:
    """少于 min_laps → insufficient_data=True。"""
    res = DriverConsistencyAnalyzer(min_laps=3).analyze(_lap_metrics(2))
    assert res["insufficient_data"] is True
    assert res["overall_consistency_score"] == 0.0


# --- AdaptationProfile ----------------------------------------------------
def test_adaptation_record_and_strength_in_range() -> None:
    """记录后 adaptation_strength in [0,1], 基线为 1.0。"""
    ap = AdaptationProfile()
    ap.record_condition("dry", 90.0)
    ap.record_condition("dry", 90.2)
    ap.record_condition("wet", 95.0)
    s = ap.adaptation_strength("wet")
    assert 0.0 <= s <= 1.0
    assert ap.adaptation_strength("dry") == 1.0


def test_adaptation_weak_conditions() -> None:
    """退化严重 (>1/0.7 ≈ 1.43x 圈速) → weak。"""
    ap = AdaptationProfile()
    ap.record_condition("dry", 90.0)
    ap.record_condition("wet", 130.0)   # 90/130 ≈ 0.69 < 0.7
    ap.record_condition("hot", 95.0)    # 90/95 ≈ 0.95 >= 0.7
    weak = ap.weak_conditions()
    assert "wet" in weak
    assert "hot" not in weak
    assert "dry" not in weak


def test_adaptation_strong_conditions() -> None:
    """轻微退化 → strong。"""
    ap = AdaptationProfile()
    ap.record_condition("dry", 90.0)
    ap.record_condition("hot", 95.0)
    strong = ap.strong_conditions()
    assert "hot" in strong
    assert "dry" not in strong


def test_adaptation_recommendation_nonempty() -> None:
    """recommendation 返回非空中文字符串。"""
    ap = AdaptationProfile()
    ap.record_condition("dry", 90.0)
    ap.record_condition("wet", 130.0)
    rec = ap.recommendation()
    assert isinstance(rec, str)
    assert len(rec) > 0
    assert "wet" in rec


def test_adaptation_unknown_condition_returns_one() -> None:
    """未记录条件 → adaptation_strength = 1.0 (中性)."""
    ap = AdaptationProfile()
    ap.record_condition("dry", 90.0)
    assert ap.adaptation_strength("high_altitude") == 1.0


# --- FatigueModel ---------------------------------------------------------
def test_fatigue_lap_time_non_decreasing() -> None:
    """lap_time_with_fatigue 随圈号单调非减。"""
    fm = FatigueModel(base_lap_time=90.0, stint_length_laps=30)
    times = [fm.lap_time_with_fatigue(i) for i in range(1, 31)]
    assert all(times[i + 1] >= times[i] for i in range(len(times) - 1))
    assert times[-1] > times[0]


def test_fatigue_index_in_range_monotonic() -> None:
    """fatigue_index in [0,1] 且单调非减。"""
    fm = FatigueModel(base_lap_time=90.0, stint_length_laps=30)
    idxs = [fm.fatigue_index(i) for i in range(0, 31)]
    assert all(0.0 <= x <= 1.0 for x in idxs)
    assert all(idxs[i + 1] >= idxs[i] for i in range(len(idxs) - 1))
    assert idxs[0] == 0.0
    assert idxs[-1] == 1.0


def test_fatigue_pit_window_early_not_recommended() -> None:
    """早期圈 → 不推荐进站。"""
    fm = FatigueModel(base_lap_time=90.0, stint_length_laps=30)
    res = fm.recommended_pit_window(5)
    assert res["pit_recommended"] is False


def test_fatigue_pit_window_late_recommended() -> None:
    """末段圈 → 推荐进站。"""
    fm = FatigueModel(base_lap_time=90.0, stint_length_laps=30)
    res = fm.recommended_pit_window(28)
    assert res["pit_recommended"] is True
    assert isinstance(res["reason"], str)
    assert len(res["reason"]) > 0


def test_fatigue_flat_zone_first_laps() -> None:
    """前 10% stint 疲劳指数为 0。"""
    fm = FatigueModel(base_lap_time=90.0, stint_length_laps=30)
    assert fm.fatigue_index(1) == 0.0
    assert fm.fatigue_index(3) == 0.0
    assert fm.lap_time_with_fatigue(1) == pytest.approx(90.0)


# --- DeepDriverProfiler ---------------------------------------------------
def test_profiler_returns_all_required_keys() -> None:
    """profile() 返回所有必需键。"""
    profiler = DeepDriverProfiler(
        _make_corner_lap(), {"tire_pressure": 21.0}, "monaco", _lap_metrics(4)
    )
    res = profiler.profile()
    required = {
        "archetype", "corner_phases", "consistency", "fatigue_projection",
        "strengths", "weaknesses", "setup_recommendations",
    }
    assert required <= set(res.keys())


def test_profiler_archetype_valid() -> None:
    """archetype 为合法 DrivingStyleArchetype。"""
    profiler = DeepDriverProfiler(_make_corner_lap(), {}, "test", _lap_metrics(4))
    res = profiler.profile()
    assert isinstance(res["archetype"], DrivingStyleArchetype)


def test_profiler_strengths_weaknesses_lists_of_strings() -> None:
    """strengths/weaknesses 为非空字符串列表。"""
    profiler = DeepDriverProfiler(_make_corner_lap(), {}, "test", _lap_metrics(4))
    res = profiler.profile()
    assert isinstance(res["strengths"], list)
    assert isinstance(res["weaknesses"], list)
    assert len(res["strengths"]) >= 1
    assert all(isinstance(s, str) for s in res["strengths"])
    assert all(isinstance(w, str) for w in res["weaknesses"])


def test_profiler_setup_recommendations_structure() -> None:
    """setup_recommendations 为 dict 列表, 每项含 field/direction/reason。"""
    profiler = DeepDriverProfiler(_make_corner_lap(), {}, "test", _lap_metrics(4))
    recs = profiler.profile()["setup_recommendations"]
    assert isinstance(recs, list)
    assert len(recs) >= 1
    required = {"field", "direction", "reason"}
    for r in recs:
        assert isinstance(r, dict)
        assert required <= set(r.keys())
        assert isinstance(r["field"], str)
        assert isinstance(r["direction"], str)
        assert isinstance(r["reason"], str)


def test_profiler_determinism() -> None:
    """相同输入 → 相同画像 (确定性)."""
    frames = _make_corner_lap()
    laps = _lap_metrics(4)
    p1 = DeepDriverProfiler(copy.deepcopy(frames), {}, "test", copy.deepcopy(laps)).profile()
    p2 = DeepDriverProfiler(copy.deepcopy(frames), {}, "test", copy.deepcopy(laps)).profile()
    assert p1 == p2


def test_profiler_minimal_frames_no_crash() -> None:
    """单帧输入不崩溃。"""
    single = [{
        "session_time": 0.0, "lap_distance": 0.0, "throttle": 0.5,
        "brake": 0.0, "steer": 0.0, "g_lat": 0.0,
        "ers_deploy_mode": 0, "ers_store": 0.5, "drs_allowed": 0,
    }]
    res = DeepDriverProfiler(single, {}, "test").profile()
    assert isinstance(res["archetype"], DrivingStyleArchetype)
    assert res["fatigue_projection"] is not None


def test_profiler_empty_lap_metrics_consistency_none() -> None:
    """lap_metrics 为空 → consistency 为 None。"""
    profiler = DeepDriverProfiler(_make_corner_lap(), {}, "test", None)
    res = profiler.profile()
    assert res["consistency"] is None


def test_profiler_fatigue_projection_present() -> None:
    """fatigue_projection 始终存在且含 projected_lap_times。"""
    profiler = DeepDriverProfiler(_make_corner_lap(), {}, "test")
    fp = profiler.profile()["fatigue_projection"]
    assert isinstance(fp, dict)
    assert "projected_lap_times" in fp
    assert "fatigue_indices" in fp
    assert len(fp["projected_lap_times"]) == 30


def test_profiler_corner_phases_in_profile() -> None:
    """profile 内 corner_phases 为 analyze_corner_phases 输出。"""
    frames = _make_corner_lap()
    res = DeepDriverProfiler(frames, {}, "test").profile()
    cp = res["corner_phases"]
    assert cp["corners_detected"] >= 1
    assert cp["weak_phase"] is not None


def test_profiler_consistency_propagated() -> None:
    """提供 lap_metrics 时 consistency 非空且含 lap_time_cv。"""
    res = DeepDriverProfiler(
        _make_corner_lap(), {}, "test", _lap_metrics(4)
    ).profile()
    assert res["consistency"] is not None
    assert "lap_time_cv" in res["consistency"]
