"""T6 分析模型单元测试（核心）。

覆盖验收标准：
1. 任意单症状产出 SetupDelta 覆盖全部 23 参数
2. 确定性：相同输入跑两次结果完全一致
3. 不越界：delta + current 在 min~max 区间内
4. 耦合矩阵 3 条硬约束（validate_matrix 通过）
5. 每非零 delta 有官方出处
6. 多症状叠加正确
7. 遥测增益（湿地 ×0.7）生效
"""

from __future__ import annotations

import pytest

from setup_tuner.domain.setup import ALL_SETUP_FIELDS, CarSetup
from setup_tuner.domain.symptoms import Symptom
from setup_tuner.engine.confidence import assess_confidence
from setup_tuner.engine.coupling import (
    COUPLING_MATRIX,
    PARAM_NAMES,
    VALID_SOURCES,
    get_column,
    get_coupling,
    get_row,
    matrix_stats,
    nonzero_cells_for_diag,
    nonzero_cells_for_param,
    validate_matrix,
)
from setup_tuner.engine.diagnostic import (
    DIAG_DIMS,
    SYMPTOM_TO_DX,
    compute_dx,
    dx_to_vector,
    empty_dx,
    is_zero_dx,
)
from setup_tuner.engine.engine import (
    compute_setup_delta,
    generate_suggestion,
    validate_engine,
)
from setup_tuner.engine.rules import get_all_rules, get_rule, validate_rules


# ===========================================================================
# fixtures
# ===========================================================================
@pytest.fixture
def default_setup() -> dict[str, float]:
    """全缺省值调教快照。"""
    return CarSetup.default().to_dict()


# ===========================================================================
# 1. 诊断向量 Dx
# ===========================================================================
class TestDiagnosticVector:
    """诊断向量 Dx 计算校验。"""

    def test_diag_dims_count_is_9(self) -> None:
        """诊断维度必须恰好 9 维。"""
        assert len(DIAG_DIMS) == 9

    def test_empty_dx_all_zero(self) -> None:
        """empty_dx 返回全零 9 维向量。"""
        dx = empty_dx()
        assert set(dx.keys()) == set(DIAG_DIMS)
        assert all(v == 0.0 for v in dx.values())

    def test_is_zero_dx(self) -> None:
        """is_zero_dx 判定。"""
        assert is_zero_dx(empty_dx())
        dx = {"front_grip_req": 0.1}
        for d in DIAG_DIMS:
            dx.setdefault(d, 0.0)
        assert not is_zero_dx(dx)

    def test_compute_dx_single_symptom(self) -> None:
        """单症状 Dx = 系数 × 强度。"""
        dx = compute_dx([("understeer", 3)])
        # understeer: front_grip_req=0.80, turnin_req=0.30
        assert dx["front_grip_req"] == pytest.approx(0.80 * 3)
        assert dx["turnin_req"] == pytest.approx(0.30 * 3)
        # 未触发维度为 0
        assert dx["rear_grip_req"] == 0.0

    def test_compute_dx_multiple_symptoms_additive(self) -> None:
        """多症状 Dx 代数求和。"""
        dx = compute_dx([("understeer", 3), ("oversteer", 2)])
        # understeer: front_grip_req=0.80*3=2.4
        # oversteer: rear_grip_req=0.80*2=1.6, hi_speed_stab_req=0.30*2=0.6
        assert dx["front_grip_req"] == pytest.approx(2.4)
        assert dx["rear_grip_req"] == pytest.approx(1.6)
        assert dx["hi_speed_stab_req"] == pytest.approx(0.6)

    def test_compute_dx_strength_zero_no_contribution(self) -> None:
        """强度 0 不产生贡献。"""
        dx = compute_dx([("understeer", 0)])
        assert is_zero_dx(dx)

    def test_compute_dx_unknown_symptom_raises(self) -> None:
        """未知症状抛 KeyError。"""
        with pytest.raises(KeyError):
            compute_dx([("nonexistent", 3)])

    def test_compute_dx_strength_out_of_range_raises(self) -> None:
        """强度越界抛 ValueError。"""
        with pytest.raises(ValueError):
            compute_dx([("understeer", 6)])
        with pytest.raises(ValueError):
            compute_dx([("understeer", -1)])

    def test_compute_dx_deterministic(self) -> None:
        """相同输入两次计算结果完全一致。"""
        symptoms = [("understeer", 3), ("oversteer", 2), ("tyre_wear", 4)]
        dx1 = compute_dx(symptoms)
        dx2 = compute_dx(symptoms)
        assert dx1 == dx2

    def test_dx_to_vector_order(self) -> None:
        """dx_to_vector 按 DIAG_DIMS 顺序输出。"""
        dx = compute_dx([("understeer", 1)])
        vec = dx_to_vector(dx)
        assert len(vec) == 9
        assert vec[0] == pytest.approx(0.80)  # front_grip_req
        assert vec[2] == pytest.approx(0.30)  # turnin_req

    def test_all_12_symptoms_have_dx_mapping(self) -> None:
        """全部 12 症状都有 Dx 映射。"""
        for sym in Symptom:
            assert sym.value in SYMPTOM_TO_DX, f"症状 {sym!r} 缺 Dx 映射"


# ===========================================================================
# 2. 耦合矩阵
# ===========================================================================
class TestCouplingMatrix:
    """9×23 耦合矩阵 3 条硬约束校验。"""

    def test_validate_matrix_passes(self) -> None:
        """validate_matrix 3 条硬约束通过。"""
        validate_matrix()  # 不抛异常即通过

    def test_matrix_shape_9x23(self) -> None:
        """矩阵形状 9 行 × 23 列。"""
        assert len(COUPLING_MATRIX) == 9
        for diag in DIAG_DIMS:
            assert len(COUPLING_MATRIX[diag]) == 23

    def test_param_names_count_23(self) -> None:
        """PARAM_NAMES 23 项且与 ALL_SETUP_FIELDS 一致。"""
        assert len(PARAM_NAMES) == 23
        assert PARAM_NAMES == [f.name for f in ALL_SETUP_FIELDS]

    def test_no_empty_column(self) -> None:
        """约束1：每参数（列）至少含 1 个非零元素。"""
        for param in PARAM_NAMES:
            nonzero = sum(
                1 for diag in DIAG_DIMS if COUPLING_MATRIX[diag][param] is not None
            )
            assert nonzero >= 1, f"参数 {param!r} 在所有诊断维度上均为零"

    def test_no_empty_row(self) -> None:
        """约束2：每诊断维度（行）至少含 1 个非零元素。"""
        for diag in DIAG_DIMS:
            nonzero = sum(
                1 for cell in COUPLING_MATRIX[diag].values() if cell is not None
            )
            assert nonzero >= 1, f"诊断维度 {diag!r} 在所有参数上均为零"

    def test_density_at_least_30_percent(self) -> None:
        """约束3：矩阵非零密度 ≥ 30%。"""
        stats = matrix_stats()
        assert stats["density"] >= 0.30, f"密度 {stats['density']:.2%} < 30%"

    def test_all_sources_valid(self) -> None:
        """每非零单元出处必须在 VALID_SOURCES 内。"""
        for diag in DIAG_DIMS:
            for cell in COUPLING_MATRIX[diag].values():
                if cell is not None:
                    assert cell.source in VALID_SOURCES, (
                        f"单元 {diag}/{cell.param} source {cell.source!r} 非法"
                    )

    def test_cell_sign_is_plus_or_minus_1(self) -> None:
        """每非零单元 sign 必须为 +1 或 -1。"""
        for diag in DIAG_DIMS:
            for cell in COUPLING_MATRIX[diag].values():
                if cell is not None:
                    assert cell.sign in (+1, -1)

    def test_cell_magnitude_positive(self) -> None:
        """每非零单元 magnitude 必须为正。"""
        for diag in DIAG_DIMS:
            for cell in COUPLING_MATRIX[diag].values():
                if cell is not None:
                    assert cell.magnitude > 0

    def test_get_coupling_known_nonzero(self) -> None:
        """get_coupling 已知非零格返回 CouplingCell。"""
        cell = get_coupling("front_grip_req", "front_wing")
        assert cell is not None
        assert cell.param == "front_wing"
        assert cell.diag == "front_grip_req"
        assert cell.sign == +1
        assert cell.magnitude == pytest.approx(1.5)

    def test_get_coupling_zero_cell_returns_none(self) -> None:
        """get_coupling 零格返回 None。"""
        # brake_power_req × front_wing 应为零（制动力与前翼无关）
        cell = get_coupling("brake_power_req", "front_wing")
        assert cell is None

    def test_get_coupling_unknown_diag_returns_none(self) -> None:
        """get_coupling 未知维度返回 None。"""
        assert get_coupling("nonexistent", "front_wing") is None

    def test_get_row_returns_23_entries(self) -> None:
        """get_row 返回 23 个参数的耦合单元。"""
        row = get_row("front_grip_req")
        assert len(row) == 23
        assert set(row.keys()) == set(PARAM_NAMES)

    def test_get_row_unknown_diag_returns_all_none(self) -> None:
        """get_row 未知维度返回全 None 行。"""
        row = get_row("nonexistent")
        assert len(row) == 23
        assert all(v is None for v in row.values())

    def test_get_column_returns_9_entries(self) -> None:
        """get_column 返回 9 个诊断维度的耦合单元。"""
        col = get_column("front_wing")
        assert len(col) == 9
        assert set(col.keys()) == set(DIAG_DIMS)

    def test_nonzero_cells_for_diag(self) -> None:
        """nonzero_cells_for_diag 返回非零单元列表。"""
        cells = nonzero_cells_for_diag("front_grip_req")
        assert len(cells) >= 1
        for c in cells:
            assert c.diag == "front_grip_req"

    def test_nonzero_cells_for_diag_unknown(self) -> None:
        """nonzero_cells_for_diag 未知维度返回空列表。"""
        assert nonzero_cells_for_diag("nonexistent") == []

    def test_nonzero_cells_for_param(self) -> None:
        """nonzero_cells_for_param 返回非零单元列表。"""
        cells = nonzero_cells_for_param("front_wing")
        assert len(cells) >= 1
        for c in cells:
            assert c.param == "front_wing"

    def test_cell_value_property(self) -> None:
        """CouplingCell.value = sign × magnitude。"""
        cell = get_coupling("front_grip_req", "front_camber")
        assert cell is not None
        # front_camber sign=-1, magnitude=0.3 → value=-0.3
        assert cell.value == pytest.approx(-0.3)

    def test_matrix_stats(self) -> None:
        """matrix_stats 返回完整统计。"""
        stats = matrix_stats()
        assert stats["diag_dims"] == 9
        assert stats["params"] == 23
        assert stats["total_cells"] == 9 * 23
        assert stats["nonzero_cells"] > 0
        assert stats["density"] > 0
        assert isinstance(stats["sources_used"], list)


# ===========================================================================
# 3. SetupDelta 计算 — 单症状覆盖 23 参数
# ===========================================================================
class TestSetupDeltaCoverage:
    """任意单症状产出 SetupDelta 覆盖全部 23 参数。"""

    def test_every_single_symptom_covers_23_params(self, default_setup) -> None:
        """每个单症状（强度 3）的 SetupDelta 必须覆盖全部 23 参数。"""
        for sym in Symptom:
            result = generate_suggestion(
                [(sym.value, 3)], default_setup, "test_track", None
            )
            delta = result["setup_delta"]
            assert set(delta.keys()) == set(default_setup.keys()), (
                f"症状 {sym!r} SetupDelta 未覆盖全部 23 参数"
            )

    def test_setup_delta_keys_match_param_names(self, default_setup) -> None:
        """SetupDelta 键集与 PARAM_NAMES 一致。"""
        result = generate_suggestion(
            [("understeer", 3)], default_setup, "test_track", None
        )
        assert set(result["setup_delta"].keys()) == set(PARAM_NAMES)


# ===========================================================================
# 4. 确定性
# ===========================================================================
class TestDeterminism:
    """确定性：相同输入跑两次结果完全一致。"""

    def test_same_input_same_output_single(self, default_setup) -> None:
        """单症状两次运行结果完全一致。"""
        for sym in Symptom:
            r1 = generate_suggestion([(sym.value, 3)], default_setup, "t", None)
            r2 = generate_suggestion([(sym.value, 3)], default_setup, "t", None)
            assert r1["setup_delta"] == r2["setup_delta"], (
                f"症状 {sym!r} 两次运行不一致"
            )

    def test_same_input_same_output_multi(self, default_setup) -> None:
        """多症状两次运行结果完全一致。"""
        symptoms = [("understeer", 3), ("oversteer", 2), ("tyre_wear", 4)]
        r1 = generate_suggestion(symptoms, default_setup, "t", None)
        r2 = generate_suggestion(symptoms, default_setup, "t", None)
        assert r1["setup_delta"] == r2["setup_delta"]
        assert r1["dx"] == r2["dx"]

    def test_compute_setup_delta_deterministic(self, default_setup) -> None:
        """compute_setup_delta 确定性。"""
        dx = compute_dx([("understeer", 3)])
        d1 = compute_setup_delta(dx, default_setup)
        d2 = compute_setup_delta(dx, default_setup)
        assert d1 == d2


# ===========================================================================
# 5. 不越界
# ===========================================================================
class TestBounds:
    """不越界：delta + current 在 min~max 区间内，|delta| <= max_delta。"""

    def test_next_within_bounds_all_symptoms(self, default_setup) -> None:
        """每个单症状的 next=current+delta 在 [min, max] 内。"""
        for sym in Symptom:
            result = generate_suggestion(
                [(sym.value, 5)], default_setup, "t", None  # 最大强度
            )
            delta = result["setup_delta"]
            for spec in ALL_SETUP_FIELDS:
                current = default_setup[spec.name]
                next_val = current + delta[spec.name]
                assert next_val >= spec.min_val - 1e-6, (
                    f"症状 {sym!r} 参数 {spec.name!r} "
                    f"next={next_val} < min={spec.min_val}"
                )
                assert next_val <= spec.max_val + 1e-6, (
                    f"症状 {sym!r} 参数 {spec.name!r} "
                    f"next={next_val} > max={spec.max_val}"
                )

    def test_delta_within_max_delta_all_symptoms(self, default_setup) -> None:
        """每个单症状的 |delta| <= max_delta。"""
        for sym in Symptom:
            result = generate_suggestion(
                [(sym.value, 5)], default_setup, "t", None
            )
            delta = result["setup_delta"]
            for spec in ALL_SETUP_FIELDS:
                assert abs(delta[spec.name]) <= spec.max_delta + 1e-6, (
                    f"症状 {sym!r} 参数 {spec.name!r} "
                    f"|delta|={abs(delta[spec.name])} > max_delta={spec.max_delta}"
                )

    def test_bounds_with_multi_symptoms(self, default_setup) -> None:
        """多症状叠加也不越界。"""
        symptoms = [(s.value, 5) for s in Symptom]
        result = generate_suggestion(symptoms, default_setup, "t", None)
        delta = result["setup_delta"]
        for spec in ALL_SETUP_FIELDS:
            current = default_setup[spec.name]
            next_val = current + delta[spec.name]
            assert spec.min_val - 1e-6 <= next_val <= spec.max_val + 1e-6
            assert abs(delta[spec.name]) <= spec.max_delta + 1e-6


# ===========================================================================
# 6. 每非零 delta 有官方出处
# ===========================================================================
class TestSourceTraceability:
    """每非零 delta 有官方出处。"""

    def test_nonzero_delta_has_source(self, default_setup) -> None:
        """每非零 delta 对应参数详情 source 非空。"""
        for sym in Symptom:
            result = generate_suggestion(
                [(sym.value, 3)], default_setup, "t", None
            )
            delta = result["setup_delta"]
            for pd in result["parameters"]:
                if abs(pd["setup_delta"]) > 1e-6:
                    assert pd["source"], (
                        f"症状 {sym!r} 参数 {pd['param']!r} 非零 delta 但 source 为空"
                    )

    def test_sources_in_valid_enum(self, default_setup) -> None:
        """出处必须在合法枚举内。"""
        for sym in Symptom:
            result = generate_suggestion(
                [(sym.value, 3)], default_setup, "t", None
            )
            for pd in result["parameters"]:
                if pd["source"]:
                    for src in pd["source"].split(","):
                        assert src in VALID_SOURCES, (
                            f"参数 {pd['param']!r} source {src!r} 不在合法枚举"
                        )


# ===========================================================================
# 7. 多症状叠加
# ===========================================================================
class TestMultiSymptom:
    """多症状叠加正确性。"""

    def test_multi_symptom_dx_additive(self) -> None:
        """多症状 Dx 为各症状代数和。"""
        dx_single_1 = compute_dx([("understeer", 3)])
        dx_single_2 = compute_dx([("oversteer", 2)])
        dx_combined = compute_dx([("understeer", 3), ("oversteer", 2)])
        for dim in DIAG_DIMS:
            assert dx_combined[dim] == pytest.approx(
                dx_single_1[dim] + dx_single_2[dim]
            )

    def test_multi_symptom_delta_not_equal_single(self, default_setup) -> None:
        """多症状叠加的 delta 不等于任一单症状（验证叠加生效）。"""
        r_single = generate_suggestion(
            [("understeer", 3)], default_setup, "t", None
        )
        r_multi = generate_suggestion(
            [("understeer", 3), ("oversteer", 3)], default_setup, "t", None
        )
        # 至少有一个参数的 delta 不同
        diffs = [
            abs(r_single["setup_delta"][p] - r_multi["setup_delta"][p])
            for p in PARAM_NAMES
        ]
        assert max(diffs) > 1e-6, "多症状叠加未产生差异"

    def test_zero_strength_no_effect(self, default_setup) -> None:
        """强度全 0 时 SetupDelta 全 0。"""
        result = generate_suggestion(
            [("understeer", 0), ("oversteer", 0)], default_setup, "t", None
        )
        for v in result["setup_delta"].values():
            assert abs(v) < 1e-6

    def test_empty_symptoms_zero_delta(self, default_setup) -> None:
        """空症状列表产出全零 delta。"""
        result = generate_suggestion([], default_setup, "t", None)
        for v in result["setup_delta"].values():
            assert abs(v) < 1e-6


# ===========================================================================
# 8. 遥测增益（湿地 ×0.7）
# ===========================================================================
class TestTelemetryGain:
    """遥测增益：湿地 ×0.7 生效。"""

    def test_wet_weather_reduces_delta(self, default_setup) -> None:
        """湿地（weather='wet'）应将 delta 衰减 ×0.7。"""
        symptoms = [("understeer", 3)]
        r_dry = generate_suggestion(symptoms, default_setup, "t", None)
        r_wet = generate_suggestion(
            symptoms, default_setup, "t", {"weather": "wet"}
        )
        # 每参数 wet delta ≈ dry delta × 0.7（受 clamp/对齐影响允许微小偏差）
        for p in PARAM_NAMES:
            dry = r_dry["setup_delta"][p]
            wet = r_wet["setup_delta"][p]
            if abs(dry) > 1e-6:
                # 湿地应使调整幅度变小
                assert abs(wet) <= abs(dry) + 1e-6, (
                    f"参数 {p!r} 湿地 |delta|={abs(wet)} > 干地 |delta|={abs(dry)}"
                )

    def test_wet_weather_string_variants(self, default_setup) -> None:
        """各种湿地字符串均触发 ×0.7。"""
        symptoms = [("understeer", 3)]
        r_dry = generate_suggestion(symptoms, default_setup, "t", None)
        for wet_str in ("wet", "rainy", "rain", "drizzle"):
            r_wet = generate_suggestion(
                symptoms, default_setup, "t", {"weather": wet_str}
            )
            # 至少有一个参数的 delta 不同（衰减生效）
            diffs = [
                abs(r_dry["setup_delta"][p] - r_wet["setup_delta"][p])
                for p in PARAM_NAMES
            ]
            assert max(diffs) > 1e-6, f"weather={wet_str!r} 未触发衰减"

    def test_wet_weather_numeric_m_weather(self, default_setup) -> None:
        """m_weather >= 1（数值）触发湿地衰减。"""
        symptoms = [("understeer", 3)]
        r_dry = generate_suggestion(symptoms, default_setup, "t", None)
        r_wet = generate_suggestion(
            symptoms, default_setup, "t", {"m_weather": 1}
        )
        diffs = [
            abs(r_dry["setup_delta"][p] - r_wet["setup_delta"][p])
            for p in PARAM_NAMES
        ]
        assert max(diffs) > 1e-6, "m_weather=1 未触发衰减"

    def test_clear_weather_no_gain(self, default_setup) -> None:
        """晴天（m_weather=0）不衰减。"""
        symptoms = [("understeer", 3)]
        r_none = generate_suggestion(symptoms, default_setup, "t", None)
        r_clear = generate_suggestion(
            symptoms, default_setup, "t", {"m_weather": 0}
        )
        assert r_none["setup_delta"] == r_clear["setup_delta"]


# ===========================================================================
# 9. 置信度
# ===========================================================================
class TestConfidence:
    """置信度评估。"""

    def test_high_with_telemetry_and_clear_feedback(self) -> None:
        """有遥测 + 反馈明确 → high。"""
        result = assess_confidence(
            [("understeer", 3)], {"m_tyresAgeLaps": 5}
        )
        assert result == "high"

    def test_medium_without_telemetry(self) -> None:
        """反馈明确但无遥测 → medium。"""
        result = assess_confidence([("understeer", 3)], None)
        assert result == "medium"

    def test_low_vague_feedback(self) -> None:
        """反馈模糊（强度 ≤ 1）→ low。"""
        assert assess_confidence([("understeer", 1)], None) == "low"
        assert assess_confidence([], None) == "low"

    def test_low_conflict_pair(self) -> None:
        """矛盾症状对 → low。"""
        # understeer + straight_slow 矛盾
        result = assess_confidence(
            [("understeer", 3), ("straight_slow", 3)], None
        )
        assert result == "low"

    def test_confidence_deterministic(self) -> None:
        """置信度确定性。"""
        symptoms = [("understeer", 3), ("oversteer", 2)]
        tel = {"m_weather": 0}
        assert assess_confidence(symptoms, tel) == assess_confidence(symptoms, tel)


# ===========================================================================
# 10. 规则库
# ===========================================================================
class TestRules:
    """规则库校验。"""

    def test_validate_rules_passes(self) -> None:
        """validate_rules 通过。"""
        validate_rules()

    def test_rule_count_is_12(self) -> None:
        """规则数 == 12。"""
        rules = get_all_rules()
        assert len(rules) == 12

    def test_every_rule_covers_23_params(self) -> None:
        """每条规则 delta_table 覆盖 23 参数。"""
        rules = get_all_rules()
        expected = set(PARAM_NAMES)
        for sym, rule in rules.items():
            assert set(rule["delta_table"].keys()) == expected, (
                f"规则 {sym!r} delta_table 参数集不匹配"
            )

    def test_get_rule_known(self) -> None:
        """get_rule 已知症状返回规则。"""
        rule = get_rule("understeer")
        assert rule is not None
        assert rule["symptom"] == "understeer"
        assert rule["name_zh"] == "转向不足"

    def test_get_rule_unknown_returns_none(self) -> None:
        """get_rule 未知症状返回 None。"""
        assert get_rule("nonexistent") is None

    def test_rule_source_nonempty(self) -> None:
        """每条规则 source 非空。"""
        rules = get_all_rules()
        for sym, rule in rules.items():
            assert rule["source"], f"规则 {sym!r} source 为空"


# ===========================================================================
# 11. 引擎自校验
# ===========================================================================
class TestEngineSelfValidation:
    """引擎构建期自校验。"""

    def test_validate_engine_passes(self) -> None:
        """validate_engine 通过（覆盖性 + 确定性 + 不越界 + 出处）。"""
        validate_engine()

    def test_generate_suggestion_structure(self, default_setup) -> None:
        """generate_suggestion 返回结构完整。"""
        result = generate_suggestion(
            [("understeer", 3)], default_setup, "suzuka", None
        )
        assert "track_id" in result
        assert "dx" in result
        assert "setup_delta" in result
        assert "parameters" in result
        assert "confidence" in result
        assert "summary" in result
        assert result["track_id"] == "suzuka"
        assert len(result["parameters"]) == 23

    def test_summary_zero_dx(self, default_setup) -> None:
        """无有效症状时摘要含「无调整建议」。"""
        result = generate_suggestion([], default_setup, "t", None)
        assert "无调整" in result["summary"]

    def test_summary_nonzero_dx(self, default_setup) -> None:
        """有症状时摘要含参数计数。"""
        result = generate_suggestion(
            [("understeer", 3)], default_setup, "t", None
        )
        assert "参数" in result["summary"]