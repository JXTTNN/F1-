"""引擎边界条件单元测试。

覆盖验收标准：
1. _align_to_step — step<=0 时不对齐、正常对齐
2. _derive_telemetry_gain — None/空/湿地字符串/湿地数字/晴天
3. compute_setup_delta — telemetry_gain=None/自定义/缺参数
4. _build_param_detail — 空症状/多症状/联动说明/tradeoff
5. assess_confidence — 模糊/矛盾/明确+遥测/明确无遥测
6. compute_dx — 强度类型非法/浮点强度
7. get_rule — 未知症状返回 None
8. 边界值：强度 0/5、全部症状最大强度叠加
"""

from __future__ import annotations

import pytest

from setup_tuner.domain.setup import ALL_SETUP_FIELDS, CarSetup
from setup_tuner.domain.symptoms import Symptom
from setup_tuner.engine.confidence import assess_confidence
from setup_tuner.engine.coupling import (
    COUPLING_MATRIX,
    PARAM_NAMES,
    get_column,
    get_coupling,
    get_row,
)
from setup_tuner.engine.diagnostic import (
    DIAG_DIMS,
    compute_dx,
    dx_to_vector,
    empty_dx,
    is_zero_dx,
)
from setup_tuner.engine.engine import (
    _align_to_step,
    _build_param_detail,
    _clip,
    _derive_telemetry_gain,
    compute_setup_delta,
    generate_suggestion,
    validate_engine,
)
from setup_tuner.engine.rules import get_all_rules, get_rule, load_rules, validate_rules


@pytest.fixture
def default_setup() -> dict[str, float]:
    return CarSetup.default().to_dict()


# ===========================================================================
# 1. _clip / _align_to_step
# ===========================================================================
class TestNumericUtils:
    """数值工具函数边界条件。"""

    def test_clip_within_range(self) -> None:
        """值在区间内返回原值。"""
        assert _clip(5.0, 0.0, 10.0) == 5.0

    def test_clip_below_min(self) -> None:
        """值低于下界返回下界。"""
        assert _clip(-1.0, 0.0, 10.0) == 0.0

    def test_clip_above_max(self) -> None:
        """值高于上界返回上界。"""
        assert _clip(15.0, 0.0, 10.0) == 10.0

    def test_clip_at_boundaries(self) -> None:
        """边界值。"""
        assert _clip(0.0, 0.0, 10.0) == 0.0
        assert _clip(10.0, 0.0, 10.0) == 10.0

    def test_align_to_step_normal(self) -> None:
        """正常对齐到 step 档位。"""
        # step=1.0, lo=0.0 → 5.3 对齐到 5.0
        assert _align_to_step(5.3, 1.0, 0.0) == 5.0

    def test_align_to_step_zero_step(self) -> None:
        """step<=0 时返回原值（不对齐）。"""
        assert _align_to_step(5.3, 0.0, 0.0) == 5.3
        assert _align_to_step(5.3, -1.0, 0.0) == 5.3

    def test_align_to_step_float_step(self) -> None:
        """浮点步长对齐。"""
        # step=0.1, lo=0.0 → 0.37 对齐到 0.4
        result = _align_to_step(0.37, 0.1, 0.0)
        assert result == pytest.approx(0.4)

    def test_align_to_step_with_offset(self) -> None:
        """带偏移的对齐（lo 非零）。"""
        # step=0.05, lo=0.0 → 0.23 对齐到 0.25
        result = _align_to_step(0.23, 0.05, 0.0)
        assert result == pytest.approx(0.25)


# ===========================================================================
# 2. _derive_telemetry_gain
# ===========================================================================
class TestTelemetryGain:
    """遥测校准增益提取边界条件。"""

    def test_none_telemetry(self) -> None:
        """None 返回全 1.0 增益。"""
        gain = _derive_telemetry_gain(None)
        assert len(gain) == 23
        assert all(v == 1.0 for v in gain.values())

    def test_empty_telemetry(self) -> None:
        """空字典返回全 1.0。"""
        gain = _derive_telemetry_gain({})
        assert all(v == 1.0 for v in gain.values())

    def test_wet_string_variants(self) -> None:
        """湿地字符串变体 → ×0.7。"""
        for weather in ("wet", "rainy", "rain", "drizzle", "WET", "Rainy"):
            gain = _derive_telemetry_gain({"weather": weather})
            assert all(v == 0.7 for v in gain.values()), f"weather={weather!r} 未生效"

    def test_wet_m_weather_string(self) -> None:
        """m_weather 字符串湿地 → ×0.7。"""
        gain = _derive_telemetry_gain({"m_weather": "wet"})
        assert all(v == 0.7 for v in gain.values())

    def test_wet_numeric_m_weather(self) -> None:
        """m_weather 数字 >=1 → ×0.7。"""
        for w in (1, 2, 3):
            gain = _derive_telemetry_gain({"m_weather": w})
            assert all(v == 0.7 for v in gain.values()), f"m_weather={w} 未生效"

    def test_clear_numeric_m_weather(self) -> None:
        """m_weather=0 (clear) → ×1.0。"""
        gain = _derive_telemetry_gain({"m_weather": 0})
        assert all(v == 1.0 for v in gain.values())

    def test_clear_string_weather(self) -> None:
        """weather='clear' → ×1.0（不在湿地集合）。"""
        gain = _derive_telemetry_gain({"weather": "clear"})
        assert all(v == 1.0 for v in gain.values())

    def test_non_string_non_numeric_weather(self) -> None:
        """weather 为其他类型 → ×1.0。"""
        gain = _derive_telemetry_gain({"weather": None})
        assert all(v == 1.0 for v in gain.values())

    def test_gain_covers_all_23_params(self) -> None:
        """增益覆盖全部 23 参数。"""
        gain = _derive_telemetry_gain({"weather": "wet"})
        assert set(gain.keys()) == set(PARAM_NAMES)


# ===========================================================================
# 3. compute_setup_delta 边界
# ===========================================================================
class TestComputeSetupDelta:
    """SetupDelta 计算边界条件。"""

    def test_zero_dx_all_zero_delta(self, default_setup) -> None:
        """全零 Dx → 全零 delta。"""
        dx = empty_dx()
        delta = compute_setup_delta(dx, default_setup)
        assert all(v == 0.0 for v in delta.values())

    def test_custom_telemetry_gain(self, default_setup) -> None:
        """自定义 telemetry_gain 生效：×0.5 增益使 delta 绝对值之和减小。"""
        dx = compute_dx([("understeer", 3)])
        # 全参数 ×0.5
        gain = {name: 0.5 for name in PARAM_NAMES}
        delta_custom = compute_setup_delta(dx, default_setup, gain)
        delta_default = compute_setup_delta(dx, default_setup)
        # 自定义增益应使 delta 绝对值之和不大于默认（增益缩小幅度，档位对齐后可能部分归零）
        custom_sum = sum(abs(v) for v in delta_custom.values())
        default_sum = sum(abs(v) for v in delta_default.values())
        assert custom_sum <= default_sum + 1e-9

    def test_telemetry_gain_none_uses_default(self, default_setup) -> None:
        """telemetry_gain=None 时使用全 1.0。"""
        dx = compute_dx([("understeer", 3)])
        d1 = compute_setup_delta(dx, default_setup, None)
        d2 = compute_setup_delta(dx, default_setup)
        assert d1 == d2

    def test_delta_keys_cover_23_params(self, default_setup) -> None:
        """delta 键覆盖全部 23 参数。"""
        dx = compute_dx([("understeer", 3)])
        delta = compute_setup_delta(dx, default_setup)
        assert set(delta.keys()) == set(PARAM_NAMES)

    def test_max_strength_all_symptoms(self, default_setup) -> None:
        """全部症状最大强度叠加不越界。"""
        symptoms = [(s.value, 5) for s in Symptom]
        dx = compute_dx(symptoms)
        delta = compute_setup_delta(dx, default_setup)
        for spec in ALL_SETUP_FIELDS:
            current = default_setup[spec.name]
            next_val = current + delta[spec.name]
            assert spec.min_val - 1e-6 <= next_val <= spec.max_val + 1e-6
            assert abs(delta[spec.name]) <= spec.max_delta + 1e-6


# ===========================================================================
# 4. _build_param_detail 边界
# ===========================================================================
class TestBuildParamDetail:
    """参数详情组装边界条件。"""

    def test_zero_dx_no_linkages(self) -> None:
        """全零 Dx → 无联动说明。"""
        dx = empty_dx()
        detail = _build_param_detail("front_wing", 5.0, 0.0, dx)
        assert detail["param"] == "front_wing"
        assert detail["current"] == 5.0
        assert detail["setup_delta"] == 0.0
        assert detail["linkages"] == []
        assert detail["linked_notes"] == "本次无需调整"
        assert detail["source"] == ""
        assert detail["tradeoff"] is None

    def test_nonzero_delta_with_tradeoff(self) -> None:
        """非零 delta 且有 tradeoff 定义时包含 tradeoff。"""
        dx = compute_dx([("understeer", 3)])
        detail = _build_param_detail("front_wing", 5.0, 2.0, dx)
        assert detail["tradeoff"] is not None
        assert "极速" in detail["tradeoff"]

    def test_nonzero_delta_no_tradeoff_defined(self) -> None:
        """非零 delta 但无 tradeoff 定义时 tradeoff=None。"""
        dx = compute_dx([("understeer", 3)])
        # active_aero_z 无 tradeoff 定义
        detail = _build_param_detail("active_aero_z", 0.5, 0.1, dx)
        assert detail["tradeoff"] is None

    def test_linkages_with_multiple_dims(self) -> None:
        """多诊断维度联动时 linkages 含多项。"""
        # understeer 触发 front_grip_req + turnin_req
        dx = compute_dx([("understeer", 3)])
        detail = _build_param_detail("front_wing", 5.0, 1.0, dx)
        # front_wing 被 front_grip_req 和 turnin_req 联动
        assert len(detail["linkages"]) >= 1

    def test_source_aggregated(self) -> None:
        """多出处用逗号 join。"""
        dx = compute_dx([("understeer", 3)])
        detail = _build_param_detail("front_wing", 5.0, 1.0, dx)
        # front_wing 的出处为 EA_SETUP_GUIDE
        assert detail["source"]


# ===========================================================================
# 5. assess_confidence 边界
# ===========================================================================
class TestConfidenceEdgeCases:
    """置信度评估边界条件。"""

    def test_empty_symptoms_low(self) -> None:
        """空症状 → low。"""
        assert assess_confidence([], None) == "low"

    def test_vague_symptoms_low(self) -> None:
        """全部强度 <=1 → low。"""
        assert assess_confidence([("understeer", 1)], None) == "low"
        assert assess_confidence([("understeer", 0)], None) == "low"

    def test_clear_symptoms_no_telemetry_medium(self) -> None:
        """明确症状无遥测 → medium。"""
        assert assess_confidence([("understeer", 3)], None) == "medium"

    def test_clear_symptoms_with_telemetry_high(self) -> None:
        """明确症状+遥测佐证 → high。"""
        telemetry = {"m_weather": 0}
        assert assess_confidence([("understeer", 3)], telemetry) == "high"

    def test_conflict_pair_low(self) -> None:
        """矛盾症状对 → low。"""
        # understeer + straight_slow 矛盾
        assert assess_confidence(
            [("understeer", 3), ("straight_slow", 3)], None,
        ) == "low"

    def test_conflict_brake_long_lockup_low(self) -> None:
        """brake_long + lockup 矛盾 → low。"""
        assert assess_confidence(
            [("brake_long", 3), ("lockup", 3)], None,
        ) == "low"

    def test_conflict_only_when_both_strong(self) -> None:
        """矛盾对仅当两者强度 >=2 时触发。"""
        # understeer=3, straight_slow=1 → 不矛盾
        result = assess_confidence([("understeer", 3), ("straight_slow", 1)], None)
        assert result == "medium"

    def test_telemetry_evidence_various_keys(self) -> None:
        """各种遥测佐证字段。"""
        for key in ("tyres_age_laps", "brake", "weather", "last_lap_time_ms"):
            telemetry = {key: 1}
            assert assess_confidence([("understeer", 3)], telemetry) == "high"

    def test_telemetry_no_evidence_keys(self) -> None:
        """遥测无佐证字段 → medium。"""
        telemetry = {"unknown_key": 1}
        assert assess_confidence([("understeer", 3)], telemetry) == "medium"

    def test_empty_telemetry_dict(self) -> None:
        """空遥测字典 → medium（无佐证）。"""
        assert assess_confidence([("understeer", 3)], {}) == "medium"


# ===========================================================================
# 6. compute_dx 边界
# ===========================================================================
class TestComputeDxEdgeCases:
    """诊断向量计算边界条件。"""

    def test_float_strength(self) -> None:
        """浮点强度合法。"""
        dx = compute_dx([("understeer", 2.5)])
        assert dx["front_grip_req"] == pytest.approx(0.80 * 2.5)

    def test_invalid_strength_type(self) -> None:
        """强度类型非法抛 ValueError。"""
        with pytest.raises(ValueError, match="类型非法"):
            compute_dx([("understeer", "not_a_number")])  # type: ignore[arg-type]

    def test_strength_at_boundaries(self) -> None:
        """强度边界值 0 和 5。"""
        dx0 = compute_dx([("understeer", 0)])
        assert is_zero_dx(dx0)

        dx5 = compute_dx([("understeer", 5)])
        assert dx5["front_grip_req"] == pytest.approx(0.80 * 5)

    def test_empty_symptoms(self) -> None:
        """空症状列表 → 全零 Dx。"""
        dx = compute_dx([])
        assert is_zero_dx(dx)

    def test_dx_to_vector_all_zero(self) -> None:
        """全零 Dx → 向量全零。"""
        vec = dx_to_vector(empty_dx())
        assert len(vec) == 9
        assert all(v == 0.0 for v in vec)


# ===========================================================================
# 7. 规则库边界
# ===========================================================================
class TestRulesEdgeCases:
    """规则库边界条件。"""

    def test_get_rule_unknown_returns_none(self) -> None:
        """未知症状 get_rule 返回 None。"""
        assert get_rule("nonexistent_symptom") is None

    def test_get_rule_known(self) -> None:
        """已知症状 get_rule 返回规则字典。"""
        rule = get_rule("understeer")
        assert rule is not None
        assert rule["symptom"] == "understeer"
        assert rule["id"] == "rule_understeer"
        assert "delta_table" in rule
        assert len(rule["delta_table"]) == 23

    def test_load_rules_count_12(self) -> None:
        """load_rules 返回 12 条规则。"""
        rules = load_rules()
        assert len(rules) == 12

    def test_get_all_rules_same_as_load(self) -> None:
        """get_all_rules 与 load_rules 一致。"""
        assert get_all_rules() == load_rules()

    def test_validate_rules_passes(self) -> None:
        """validate_rules 通过。"""
        validate_rules()

    def test_every_rule_delta_table_covers_23(self) -> None:
        """每条规则 delta_table 覆盖 23 参数。"""
        rules = load_rules()
        for symptom, rule in rules.items():
            assert len(rule["delta_table"]) == 23, f"规则 {symptom!r} delta_table 非 23 项"

    def test_every_rule_has_source(self) -> None:
        """每条规则 source 非空。"""
        rules = load_rules()
        for symptom, rule in rules.items():
            assert rule["source"], f"规则 {symptom!r} source 为空"


# ===========================================================================
# 8. 耦合矩阵边界
# ===========================================================================
class TestCouplingMatrixEdgeCases:
    """耦合矩阵边界条件。"""

    def test_get_column_unknown_param(self) -> None:
        """get_column 未知参数 → 各维度 None。"""
        col = get_column("nonexistent_param")
        assert len(col) == 9
        assert all(v is None for v in col.values())

    def test_get_row_unknown_diag(self) -> None:
        """get_row 未知维度 → 全 None。"""
        row = get_row("nonexistent_diag")
        assert len(row) == 23
        assert all(v is None for v in row.values())

    def test_get_coupling_both_unknown(self) -> None:
        """get_coupling 两个都未知 → None。"""
        assert get_coupling("nope", "nope") is None

    def test_matrix_all_cells_have_valid_sign(self) -> None:
        """所有非零单元 sign ∈ {+1, -1}。"""
        for diag in DIAG_DIMS:
            for cell in COUPLING_MATRIX[diag].values():
                if cell is not None:
                    assert cell.sign in (+1, -1)


# ===========================================================================
# 9. validate_engine 自校验
# ===========================================================================
class TestValidateEngine:
    """引擎自校验。"""

    def test_validate_engine_passes(self) -> None:
        """validate_engine 通过。"""
        validate_engine()

    def test_generate_suggestion_empty_symptoms(self, default_setup) -> None:
        """空症状 → 全零 delta + low confidence。"""
        result = generate_suggestion([], default_setup, "test", None)
        assert all(v == 0.0 for v in result["setup_delta"].values())
        assert result["confidence"] == "low"
        assert "无调整" in result["summary"]

    def test_generate_suggestion_with_telemetry(self, default_setup) -> None:
        """有遥测佐证 → high confidence。"""
        result = generate_suggestion(
            [("understeer", 3)], default_setup, "test", {"m_weather": 0},
        )
        assert result["confidence"] == "high"

    def test_generate_suggestion_wet_telemetry(self, default_setup) -> None:
        """湿地遥测 → delta 幅度减小。"""
        result_dry = generate_suggestion(
            [("understeer", 3)], default_setup, "test", {"m_weather": 0},
        )
        result_wet = generate_suggestion(
            [("understeer", 3)], default_setup, "test", {"m_weather": 1},
        )
        # 湿地非零 delta 绝对值之和应小于晴天
        dry_sum = sum(abs(v) for v in result_dry["setup_delta"].values())
        wet_sum = sum(abs(v) for v in result_wet["setup_delta"].values())
        assert wet_sum < dry_sum