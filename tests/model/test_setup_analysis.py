"""调教分析侧: 参数贡献分解测试 (Iter-75 可解释性).

覆盖:
- analyze_setup_contributions 返回 18 个参数 (fuel_load 除外), 按 sensitivity 降序.
- sensitivity 排序正确 (最敏感的在前).
- optimal_direction 物理合理 (V-shape 先验下, 偏离最优的方向 delta 为正).
- rank_parameter_sensitivity 返回 top_n.
- explain_setup_change 对比两个 setup, 解释方向语义与圈速贡献.
"""

from __future__ import annotations

import pytest

from f1opt.data.setup_schema import DEFAULT_SETUP, SETUP_FIELDS
from f1opt.driver.profile import AGGRESSIVE_PROFILE, CONSERVATIVE_PROFILE
from f1opt.model.setup_analysis import (
    ParameterContribution,
    analyze_setup_contributions,
    explain_setup_change,
    rank_parameter_sensitivity,
)
from f1opt.model.surrogate import predict_lap_time


def test_analyze_returns_20_params_excluding_fuel() -> None:
    """analyze_setup_contributions 返回 20 个参数 (fuel_load 除外)."""
    contribs = analyze_setup_contributions(DEFAULT_SETUP, "melbourne", None)
    assert len(contribs) == 20
    names = {c.field_name for c in contribs}
    assert "fuel_load" not in names
    # 所有 20 个非 fuel 参数都在
    expected = set(SETUP_FIELDS.keys()) - {"fuel_load"}
    assert names == expected


def test_contributions_sorted_by_sensitivity_desc() -> None:
    """贡献列表按 sensitivity 降序 (最敏感在前)."""
    contribs = analyze_setup_contributions(DEFAULT_SETUP, "hungaroring", None)
    for i in range(len(contribs) - 1):
        assert contribs[i].sensitivity >= contribs[i + 1].sensitivity


def test_sensitivity_nonnegative() -> None:
    """所有参数 sensitivity >= 0 (|delta| 之和)."""
    contribs = analyze_setup_contributions(DEFAULT_SETUP, "monza", None)
    for c in contribs:
        assert c.sensitivity >= 0.0
        assert c.delta_plus != 0.0 or c.delta_minus != 0.0  # 至少一个方向有响应


def test_optimal_direction_values() -> None:
    """optimal_direction 取值在 {-1, 0, +1}."""
    contribs = analyze_setup_contributions(DEFAULT_SETUP, "melbourne", None)
    for c in contribs:
        assert c.optimal_direction in (-1, 0, 1)


def test_default_setup_on_medium_track_near_optimal() -> None:
    """DEFAULT_SETUP 是 medium 最优, melbourne (medium) 上多数参数 optimal_direction=0.

    V-shape 先验: 偏离最优的方向 delta 为正 (变慢), 两个方向都变慢 -> direction=0
    (当前最优). 少数参数可能因 DNN 残差有偏向, 但多数应为 0.
    """
    contribs = analyze_setup_contributions(DEFAULT_SETUP, "melbourne", None)
    neutral_count = sum(1 for c in contribs if c.optimal_direction == 0)
    # 至少一半参数在最优附近 (medium 赛道 + medium 最优 setup).
    assert neutral_count >= 9, (
        f"DEFAULT_SETUP 在 melbourne 上仅 {neutral_count}/20 参数最优, "
        f"预期 ≥9 (medium 最优 setup)"
    )


def test_rank_parameter_sensitivity_top_n() -> None:
    """rank_parameter_sensitivity 返回 top_n 个 (name, sensitivity)."""
    top5 = rank_parameter_sensitivity(DEFAULT_SETUP, "monza", None, top_n=5)
    assert len(top5) == 5
    # 降序
    for i in range(4):
        assert top5[i][1] >= top5[i + 1][1]
    # 参数名合法
    for name, sens in top5:
        assert name in SETUP_FIELDS
        assert sens >= 0.0


def test_rank_top_n_clamped() -> None:
    """top_n > 20 时返回全部 20 个."""
    top = rank_parameter_sensitivity(DEFAULT_SETUP, "melbourne", None, top_n=100)
    assert len(top) == 20


def test_explain_setup_change_basic() -> None:
    """explain_setup_change 对比两个 setup, 返回变化参数解释."""
    baseline = DEFAULT_SETUP.model_copy(update={"front_wing": 33, "rear_wing": 35})
    recommended = DEFAULT_SETUP  # 推荐是默认 (medium 最优)
    explanations = explain_setup_change(baseline, recommended, "melbourne", None)

    # front_wing 和 rear_wing 应在变化列表中
    fields = [e["field"] for e in explanations]
    assert "front_wing" in fields
    assert "rear_wing" in fields

    # 每个解释含完整字段
    for e in explanations:
        assert "field" in e
        assert "baseline_value" in e
        assert "recommended_value" in e
        assert "delta_steps" in e
        assert "lap_contribution" in e
        assert "direction" in e
        assert e["direction"] in ("faster", "slower", "neutral")


def test_explain_setup_change_sorted_by_contribution() -> None:
    """解释列表按 |lap_contribution| 降序."""
    baseline = DEFAULT_SETUP.model_copy(
        update={"front_wing": 35, "rear_wing": 38, "front_arb": 5}
    )
    recommended = DEFAULT_SETUP
    explanations = explain_setup_change(baseline, recommended, "hungaroring", None)
    for i in range(len(explanations) - 1):
        assert abs(explanations[i]["lap_contribution"]) >= abs(explanations[i + 1]["lap_contribution"])


def test_explain_setup_change_no_changes() -> None:
    """两个相同 setup 的解释列表为空."""
    explanations = explain_setup_change(DEFAULT_SETUP, DEFAULT_SETUP, "melbourne", None)
    assert len(explanations) == 0


def test_explain_setup_change_direction_semantics() -> None:
    """方向语义物理正确: 从次优基线到最优推荐, 主要变化参数 direction=faster."""
    # 次优基线 (前后翼偏高), 推荐 = DEFAULT_SETUP (medium 最优)
    baseline = DEFAULT_SETUP.model_copy(update={"front_wing": 40, "rear_wing": 42})
    recommended = DEFAULT_SETUP
    explanations = explain_setup_change(baseline, recommended, "melbourne", None)

    # front_wing 从 40 降到 DEFAULT (25), 应该 faster (向最优移动)
    fw_explain = next(e for e in explanations if e["field"] == "front_wing")
    assert fw_explain["lap_contribution"] > 0.0, (
        f"front_wing 40->25 在 melbourne 应变快, contribution={fw_explain['lap_contribution']:.4f}"
    )
    assert fw_explain["direction"] == "faster"


def test_driver_profile_affects_contributions() -> None:
    """driver_profile 影响 setup 贡献分析 (激进 vs 保守车手灵敏度分布不同).

    Iter-93 修复: 旧版用 ``None`` 作为保守对照, 但 Iter-93 把
    ``_normalize_driver_vector(None)`` 改为返回中性 [0.5]*8 (修正 DNN baseline
    残差 +1.5s 的根因). 真正的中性 driver 与 AGGR 的对比中, cross-term 修正
    仅调节 sensitivity 大小, 几乎不翻转 ``optimal_direction`` (V-shape setup_penalty
    主导方向, cross-term 幅度不足以翻转). 现改用 ``CONSERVATIVE_PROFILE``
    (与 AGGR 大多数维度 *反向*), cross-term 既改 sensitivity 也偶尔翻转方向.
    同时增加 sensitivity 分布差异断言 (与测试名 "灵敏度分布不同" 对齐).
    """
    aggr_contribs = analyze_setup_contributions(
        DEFAULT_SETUP, "hungaroring", AGGRESSIVE_PROFILE
    )
    cons_contribs = analyze_setup_contributions(
        DEFAULT_SETUP, "hungaroring", CONSERVATIVE_PROFILE
    )
    # (1) 至少有一个参数的 optimal_direction 不同 (车手风格影响最优调教方向)
    aggr_dirs = {c.field_name: c.optimal_direction for c in aggr_contribs}
    cons_dirs = {c.field_name: c.optimal_direction for c in cons_contribs}
    dir_diff_count = sum(
        1 for name in aggr_dirs
        if aggr_dirs[name] != cons_dirs.get(name)
    )
    # (2) sensitivity 分布差异: 至少 3 个参数的相对差异 > 20%
    # (车手风格应显著改变各参数的边际敏感度, 即使方向不变)
    aggr_sens = {c.field_name: c.sensitivity for c in aggr_contribs}
    cons_sens = {c.field_name: c.sensitivity for c in cons_contribs}
    sens_diff_count = 0
    for name in aggr_sens:
        a, c = aggr_sens[name], cons_sens[name]
        if a + c > 0 and abs(a - c) / (a + c) > 0.20:
            sens_diff_count += 1
    assert dir_diff_count >= 1 or sens_diff_count >= 3, (
        f"AGGR vs CONS 贡献分析既无方向差异 (dir_diff={dir_diff_count}) "
        f"也无显著 sensitivity 差异 (sens_diff={sens_diff_count} >=3) "
        "— DNN 对 driver 不敏感"
    )
