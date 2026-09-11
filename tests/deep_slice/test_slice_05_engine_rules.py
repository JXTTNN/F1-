"""切片 5 深度测试：规则引擎核心（rules + engine + confidence）。

覆盖：
    - setup_tuner.engine.rules（规则库物化，load_rules / get_rule / validate_rules）
    - setup_tuner.engine.engine（compute_setup_delta / generate_suggestion 6 步流水线）
    - setup_tuner.engine.confidence（assess_confidence 置信度评估）

5 种测试方式：
    1. unit     — load_rules / get_rule / compute_setup_delta / assess_confidence 正确性
    2. boundary — 未知症状、空症状、缺参数、强度越界、矛盾症状
    3. property — 确定性、覆盖性（23 参数）、不越界、置信度枚举
    4. static   — 12 规则 / 23 参数 / 置信度枚举 / source 非空约束
    5. smoke    — 真实症状 → generate_suggestion 完整链路
"""

from __future__ import annotations

import pytest

from setup_tuner.domain.setup import ALL_SETUP_FIELDS, CarSetup
from setup_tuner.domain.symptoms import Symptom
from setup_tuner.engine.confidence import assess_confidence
from setup_tuner.engine.engine import (
    compute_setup_delta,
    generate_suggestion,
    validate_engine,
)
from setup_tuner.engine.rules import (
    RULE_SOURCE,
    get_all_rules,
    get_rule,
    load_rules,
    validate_rules,
)


# ===========================================================================
# 1. 单元测试 (unit) — 每个公开函数的正常输入正确性
# ===========================================================================
class TestUnit:
    """单元测试：验证规则引擎公开函数的正常输入正确性。"""

    def test_unit_load_rules_returns_12(self) -> None:
        """load_rules 应返回 12 条规则。"""
        rules = load_rules()
        assert len(rules) == 12

    def test_unit_get_rule_returns_dict(self) -> None:
        """get_rule 应返回规则字典。"""
        rule = get_rule("understeer")
        assert rule is not None
        assert rule["id"] == "rule_understeer"
        assert rule["symptom"] == "understeer"
        assert rule["category"] == "entry"
        assert rule["name_zh"] == "转向不足"

    def test_unit_get_rule_delta_table_covers_23(self) -> None:
        """get_rule 的 delta_table 应覆盖全部 23 参数。"""
        rule = get_rule("understeer")
        assert rule is not None
        assert len(rule["delta_table"]) == 23

    def test_unit_get_all_rules_alias(self) -> None:
        """get_all_rules 应为 load_rules 的别名。"""
        assert get_all_rules() == load_rules()

    def test_unit_compute_setup_delta_covers_23(self) -> None:
        """compute_setup_delta 应覆盖全部 23 参数。"""
        from setup_tuner.engine.diagnostic import compute_dx
        dx = compute_dx([("understeer", 3)])
        setup = CarSetup.default().to_dict()
        delta = compute_setup_delta(dx, setup)
        assert len(delta) == 23

    def test_unit_assess_confidence_high(self) -> None:
        """assess_confidence 有遥测 + 明确反馈 → high。"""
        telemetry = {"m_tyresAgeLaps": 5, "m_weather": 0}
        result = assess_confidence([("understeer", 3)], telemetry)
        assert result == "high"

    def test_unit_assess_confidence_medium(self) -> None:
        """assess_confidence 明确反馈但无遥测 → medium。"""
        result = assess_confidence([("understeer", 3)], None)
        assert result == "medium"

    def test_unit_assess_confidence_low_vague(self) -> None:
        """assess_confidence 模糊反馈（强度 ≤ 1）→ low。"""
        result = assess_confidence([("understeer", 1)], None)
        assert result == "low"

    def test_unit_assess_confidence_low_empty(self) -> None:
        """assess_confidence 空症状 → low。"""
        result = assess_confidence([], None)
        assert result == "low"

    def test_unit_assess_confidence_low_conflict(self) -> None:
        """assess_confidence 矛盾症状对 → low。"""
        # understeer + straight_slow 为矛盾对
        result = assess_confidence([("understeer", 3), ("straight_slow", 3)], None)
        assert result == "low"

    def test_unit_generate_suggestion_structure(self) -> None:
        """generate_suggestion 应返回完整报告结构。"""
        setup = CarSetup.default().to_dict()
        result = generate_suggestion(
            [("understeer", 3)], setup, "suzuka", None,
        )
        assert result["track_id"] == "suzuka"
        assert "dx" in result
        assert "setup_delta" in result
        assert "parameters" in result
        assert "confidence" in result
        assert "summary" in result
        assert len(result["parameters"]) == 23

    def test_unit_generate_suggestion_no_symptoms(self) -> None:
        """generate_suggestion 空症状应返回无调整建议。"""
        setup = CarSetup.default().to_dict()
        result = generate_suggestion([], setup, "suzuka", None)
        assert "无调整" in result["summary"] or "无有效症状" in result["summary"]


# ===========================================================================
# 2. 边界/异常测试 (boundary) — 未知症状、空症状、缺参数、矛盾症状
# ===========================================================================
class TestBoundary:
    """边界/异常测试：验证异常输入的处理。"""

    def test_boundary_get_rule_unknown_returns_none(self) -> None:
        """get_rule 对未知症状应返回 None。"""
        assert get_rule("nonexistent") is None

    def test_boundary_compute_setup_delta_empty_dx(self) -> None:
        """compute_setup_delta 对全零 Dx 应返回全零 delta。"""
        from setup_tuner.engine.diagnostic import empty_dx
        setup = CarSetup.default().to_dict()
        delta = compute_setup_delta(empty_dx(), setup)
        assert all(abs(v) < 1e-9 for v in delta.values())

    def test_boundary_compute_setup_delta_missing_param_raises(self) -> None:
        """compute_setup_delta 缺失参数应抛 KeyError。"""
        from setup_tuner.engine.diagnostic import compute_dx
        dx = compute_dx([("understeer", 3)])
        incomplete = {"front_wing": 5.0}  # 缺其他 22 个参数
        with pytest.raises(KeyError):
            compute_setup_delta(dx, incomplete)

    def test_boundary_assess_confidence_empty_symptoms(self) -> None:
        """assess_confidence 空症状 → low。"""
        assert assess_confidence([], None) == "low"
        assert assess_confidence([], {"m_weather": 0}) == "low"

    def test_boundary_assess_confidence_all_strength_zero(self) -> None:
        """assess_confidence 全强度 0 → low（模糊）。"""
        result = assess_confidence([("understeer", 0), ("oversteer", 0)], None)
        assert result == "low"

    def test_boundary_assess_confidence_conflict_pairs(self) -> None:
        """assess_confidence 全部矛盾对均 → low。"""
        conflict_pairs = [
            [("understeer", 3), ("straight_slow", 3)],
            [("oversteer", 3), ("straight_slow", 3)],
            [("brake_long", 3), ("lockup", 3)],
        ]
        for symptoms in conflict_pairs:
            assert assess_confidence(symptoms, None) == "low"

    def test_boundary_generate_suggestion_zero_dx(self) -> None:
        """generate_suggestion 强度全 0 应返回无调整。"""
        setup = CarSetup.default().to_dict()
        result = generate_suggestion(
            [("understeer", 0)], setup, "suzuka", None,
        )
        # 全部 delta 应为 0
        assert all(abs(v) < 1e-9 for v in result["setup_delta"].values())

    def test_boundary_assess_confidence_strength_boundary(self) -> None:
        """assess_confidence 强度边界 1（模糊）和 2（明确）。"""
        assert assess_confidence([("understeer", 1)], None) == "low"
        assert assess_confidence([("understeer", 2)], None) == "medium"

    def test_boundary_generate_suggestion_with_wet_weather(self) -> None:
        """generate_suggestion 湿地天气应触发保守调整（×0.7 增益）。"""
        setup = CarSetup.default().to_dict()
        dry = generate_suggestion([("understeer", 3)], setup, "suzuka", None)
        wet = generate_suggestion(
            [("understeer", 3)], setup, "suzuka", {"m_weather": 2},
        )
        # 湿地下 |delta| 应普遍 <= 干地下（保守）
        dry_total = sum(abs(v) for v in dry["setup_delta"].values())
        wet_total = sum(abs(v) for v in wet["setup_delta"].values())
        assert wet_total <= dry_total + 1e-6


# ===========================================================================
# 3. 属性不变量测试 (property) — 确定性、覆盖性、不越界
# ===========================================================================
class TestProperty:
    """属性不变量测试：验证确定性与覆盖性。"""

    def test_property_load_rules_deterministic(self) -> None:
        """确定性：两次 load_rules 结果完全一致。"""
        r1 = load_rules()
        r2 = load_rules()
        assert r1 == r2

    def test_property_compute_setup_delta_deterministic(self) -> None:
        """确定性：相同输入两次 compute_setup_delta 结果一致。"""
        from setup_tuner.engine.diagnostic import compute_dx
        dx = compute_dx([("understeer", 3)])
        setup = CarSetup.default().to_dict()
        d1 = compute_setup_delta(dx, setup)
        d2 = compute_setup_delta(dx, setup)
        assert d1 == d2

    def test_property_generate_suggestion_deterministic(self) -> None:
        """确定性：相同输入两次 generate_suggestion 结果一致。"""
        setup = CarSetup.default().to_dict()
        r1 = generate_suggestion([("understeer", 3)], setup, "suzuka", None)
        r2 = generate_suggestion([("understeer", 3)], setup, "suzuka", None)
        assert r1["setup_delta"] == r2["setup_delta"]
        assert r1["dx"] == r2["dx"]
        assert r1["confidence"] == r2["confidence"]

    def test_property_delta_within_max_delta(self) -> None:
        """不越界：|delta[p]| <= max_delta[p]。"""
        setup = CarSetup.default().to_dict()
        result = generate_suggestion(
            [("understeer", 5), ("oversteer", 5), ("brake_long", 5)],
            setup, "suzuka", None,
        )
        for spec in ALL_SETUP_FIELDS:
            delta = result["setup_delta"][spec.name]
            assert abs(delta) <= spec.max_delta + 1e-6, (
                f"{spec.name}: |delta|={abs(delta)} > max_delta={spec.max_delta}"
            )

    def test_property_next_value_within_range(self) -> None:
        """不越界：current + delta ∈ [min, max]。"""
        setup = CarSetup.default().to_dict()
        result = generate_suggestion(
            [("understeer", 5), ("bottoming", 5)],
            setup, "suzuka", None,
        )
        for spec in ALL_SETUP_FIELDS:
            current = setup[spec.name]
            delta = result["setup_delta"][spec.name]
            next_val = current + delta
            assert next_val >= spec.min_val - 1e-6, (
                f"{spec.name}: next={next_val} < min={spec.min_val}"
            )
            assert next_val <= spec.max_val + 1e-6, (
                f"{spec.name}: next={next_val} > max={spec.max_val}"
            )

    def test_property_assess_confidence_returns_valid_enum(self) -> None:
        """不变量：assess_confidence 返回值在 {high, medium, low}。"""
        cases = [
            ([], None),
            ([("understeer", 3)], None),
            ([("understeer", 3)], {"m_weather": 0}),
            ([("understeer", 1)], None),
            ([("understeer", 3), ("straight_slow", 3)], None),
        ]
        for symptoms, telemetry in cases:
            result = assess_confidence(symptoms, telemetry)
            assert result in ("high", "medium", "low")

    def test_property_rules_validate_passes(self) -> None:
        """不变量：validate_rules 应通过。"""
        validate_rules()  # 不抛异常

    def test_property_engine_validate_passes(self) -> None:
        """不变量：validate_engine 应通过。"""
        validate_engine()  # 不抛异常

    def test_property_rule_source_consistent(self) -> None:
        """不变量：所有规则 source 为 RULE_SOURCE。"""
        rules = load_rules()
        for rule in rules.values():
            assert rule["source"] == RULE_SOURCE


# ===========================================================================
# 4. 静态分析 (static) — 12 规则 / 23 参数 / source 非空
# ===========================================================================
class TestStatic:
    """静态分析：验证规则库数量与约束。"""

    def test_static_rules_count_is_12(self) -> None:
        """数量约束：规则库恰好 12 条。"""
        assert len(load_rules()) == 12

    def test_static_rule_source_is_ea_setup_guide(self) -> None:
        """枚举约束：RULE_SOURCE 为 EA_SETUP_GUIDE。"""
        assert RULE_SOURCE == "EA_SETUP_GUIDE"

    @pytest.mark.parametrize("symptom", [s.value for s in Symptom])
    def test_static_each_symptom_has_rule(self, symptom: str) -> None:
        """参数化：每个症状都有对应规则。"""
        rule = get_rule(symptom)
        assert rule is not None
        assert rule["symptom"] == symptom

    @pytest.mark.parametrize("symptom", [s.value for s in Symptom])
    def test_static_each_rule_delta_table_23_params(self, symptom: str) -> None:
        """参数化：每条规则 delta_table 覆盖 23 参数。"""
        rule = get_rule(symptom)
        assert rule is not None
        expected = {f.name for f in ALL_SETUP_FIELDS}
        assert set(rule["delta_table"].keys()) == expected

    @pytest.mark.parametrize("symptom", [s.value for s in Symptom])
    def test_static_each_rule_source_nonempty(self, symptom: str) -> None:
        """参数化：每条规则 source 非空。"""
        rule = get_rule(symptom)
        assert rule is not None
        assert rule["source"]

    @pytest.mark.parametrize("symptom", [s.value for s in Symptom])
    def test_static_each_rule_nonzero_delta_has_sources(self, symptom: str) -> None:
        """参数化：非零 delta 的参数 sources 非空。"""
        rule = get_rule(symptom)
        assert rule is not None
        for param, entry in rule["delta_table"].items():
            if abs(entry["value"]) > 1e-12:
                assert entry["sources"], f"{symptom}/{param} 非零但 sources 为空"

    @pytest.mark.parametrize("symptom", [s.value for s in Symptom])
    def test_static_each_rule_has_id_and_name(self, symptom: str) -> None:
        """参数化：每条规则有 id 和 name_zh。"""
        rule = get_rule(symptom)
        assert rule is not None
        assert rule["id"] == f"rule_{symptom}"
        assert rule["name_zh"]

    @pytest.mark.parametrize("symptom", [s.value for s in Symptom])
    def test_static_each_rule_has_dx(self, symptom: str) -> None:
        """参数化：每条规则有 dx 字段。"""
        rule = get_rule(symptom)
        assert rule is not None
        assert "dx" in rule
        assert isinstance(rule["dx"], dict)

    def test_static_confidence_conflict_pairs_count(self) -> None:
        """数量约束：矛盾症状对至少 3 对。"""
        from setup_tuner.engine.confidence import _CONFLICT_PAIRS  # noqa: PLC2701
        assert len(_CONFLICT_PAIRS) >= 3


# ===========================================================================
# 5. 实际运行冒烟 (smoke) — 真实症状 → generate_suggestion 完整链路
# ===========================================================================
class TestSmoke:
    """实际运行冒烟：用真实症状数据走完整 generate_suggestion 链路。"""

    def test_smoke_full_suggestion_chain(self) -> None:
        """冒烟：多症状 → generate_suggestion → 完整报告。"""
        setup = CarSetup.default().to_dict()
        result = generate_suggestion(
            [("understeer", 3), ("oversteer", 2), ("brake_long", 4)],
            setup, "suzuka", {"m_tyresAgeLaps": 5, "m_weather": 0},
        )
        assert result["track_id"] == "suzuka"
        assert result["confidence"] == "high"
        assert len(result["setup_delta"]) == 23
        assert len(result["parameters"]) == 23
        # 至少有 1 个非零 delta
        nonzero = sum(1 for v in result["setup_delta"].values() if abs(v) > 1e-9)
        assert nonzero > 0

    def test_smoke_all_12_symptoms_suggestion(self) -> None:
        """冒烟：全部 12 症状各自 generate_suggestion 均成功。"""
        setup = CarSetup.default().to_dict()
        for symptom in Symptom:
            result = generate_suggestion(
                [(symptom.value, 3)], setup, "suzuka", None,
            )
            assert len(result["setup_delta"]) == 23
            assert result["confidence"] in ("high", "medium", "low")

    def test_smoke_suggestion_with_real_telemetry(self) -> None:
        """冒烟：用真实遥测数据（含天气/胎耗）走完整链路。"""
        setup = CarSetup.default().to_dict()
        telemetry = {
            "m_weather": 1,  # 小雨
            "m_tyresAgeLaps": 8,
            "m_lastLapTimeInMS": 95000,
            "m_sector1TimeInMS": 30000,
        }
        result = generate_suggestion(
            [("understeer", 3), ("tyre_wear", 4)],
            setup, "monaco", telemetry,
        )
        assert result["confidence"] == "high"
        assert result["track_id"] == "monaco"

    def test_smoke_suggestion_param_details_complete(self) -> None:
        """冒烟：每参数详情含 param/current/next/setup_delta/linkages/source。"""
        setup = CarSetup.default().to_dict()
        result = generate_suggestion([("understeer", 3)], setup, "suzuka", None)
        for pd in result["parameters"]:
            assert "param" in pd
            assert "current" in pd
            assert "next" in pd
            assert "setup_delta" in pd
            assert "linkages" in pd
            assert "source" in pd

    def test_smoke_suggestion_next_equals_current_plus_delta(self) -> None:
        """冒烟：每参数 next == current + setup_delta。"""
        setup = CarSetup.default().to_dict()
        result = generate_suggestion([("understeer", 3)], setup, "suzuka", None)
        for pd in result["parameters"]:
            assert pd["next"] == pytest.approx(pd["current"] + pd["setup_delta"])

    def test_smoke_wet_vs_dry_confidence(self) -> None:
        """冒烟：干地与湿地置信度均合法。"""
        setup = CarSetup.default().to_dict()
        dry = generate_suggestion(
            [("understeer", 3)], setup, "suzuka", {"m_weather": 0},
        )
        wet = generate_suggestion(
            [("understeer", 3)], setup, "suzuka", {"m_weather": 2},
        )
        assert dry["confidence"] in ("high", "medium", "low")
        assert wet["confidence"] in ("high", "medium", "low")

    def test_smoke_rules_to_engine_consistency(self) -> None:
        """冒烟：规则库 delta_table 与引擎 compute_setup_delta 一致性。"""

        # understeer 强度 1 时的规则 delta_table
        rule = get_rule("understeer")
        assert rule is not None
        # 引擎对 understeer 强度 1 的 Dx 计算
        from setup_tuner.engine.diagnostic import compute_dx
        dx = compute_dx([("understeer", 1)])
        # 规则的 dx 应与 compute_dx 强度 1 一致
        for dim, coef in rule["dx"].items():
            assert dx[dim] == pytest.approx(coef)

    def test_smoke_full_pipeline_all_tracks(self) -> None:
        """冒烟：对全部 24 赛道各跑一次 generate_suggestion。"""
        from setup_tuner.domain.track import ALL_TRACKS
        setup = CarSetup.default().to_dict()
        for track in ALL_TRACKS:
            result = generate_suggestion(
                [("understeer", 3)], setup, track.track_id, None,
            )
            assert result["track_id"] == track.track_id
            assert len(result["setup_delta"]) == 23