"""T4 调教 schema + 症状枚举单元测试。

覆盖验收标准：
1. ALL_SETUP_FIELDS 长度 == 23，7 大类
2. Symptom 枚举 == 12，4 类
3. tyre_wear 中文为「胎耗偏高」
4. validate_value 校验范围
"""

from __future__ import annotations

import pytest

from setup_tuner.domain.setup import (
    ALL_GROUPS,
    ALL_SETUP_FIELDS,
    CarSetup,
    SetupField,
    get_field,
    get_fields_by_group,
    validate_value,
)
from setup_tuner.domain.symptoms import (
    DEFAULT_INTENSITY,
    INTENSITY_MAX,
    INTENSITY_MIN,
    SYMPTOM_INFO,
    Symptom,
    SymptomCategory,
    get_symptom_category,
    get_symptom_label,
    get_symptoms_by_category,
    validate_intensity,
)


# ===========================================================================
# 1. 调教参数 schema
# ===========================================================================
class TestSetupSchema:
    """23 项调教参数 / 7 大类 schema 校验。"""

    def test_field_count_is_23(self) -> None:
        """ALL_SETUP_FIELDS 必须恰好 23 项。"""
        assert len(ALL_SETUP_FIELDS) == 23

    def test_group_count_is_7(self) -> None:
        """ALL_GROUPS 必须恰好 7 大类。"""
        assert len(ALL_GROUPS) == 7

    def test_groups_expected(self) -> None:
        """7 大类名称与顺序。"""
        assert ALL_GROUPS == [
            "Aerodynamics",
            "Differential",
            "Suspension Geometry",
            "Suspension",
            "Brakes",
            "Tyres",
            "New2026",
        ]

    def test_all_fields_are_setupfield(self) -> None:
        """每项均为 SetupField 实例。"""
        for f in ALL_SETUP_FIELDS:
            assert isinstance(f, SetupField)

    def test_field_names_unique(self) -> None:
        """参数名唯一。"""
        names = [f.name for f in ALL_SETUP_FIELDS]
        assert len(names) == len(set(names))

    def test_field_names_match_carsetup_attributes(self) -> None:
        """参数名与 CarSetup 属性一一对应。"""
        cs = CarSetup.default()
        for f in ALL_SETUP_FIELDS:
            assert hasattr(cs, f.name), f"CarSetup 缺少属性 {f.name!r}"

    def test_field_group_in_all_groups(self) -> None:
        """每参数 group 必须在 ALL_GROUPS 内。"""
        for f in ALL_SETUP_FIELDS:
            assert f.group in ALL_GROUPS, f"参数 {f.name!r} group {f.group!r} 不在 7 大类"

    def test_field_min_le_default_le_max(self) -> None:
        """每参数 min_val <= default <= max_val。"""
        for f in ALL_SETUP_FIELDS:
            assert f.min_val <= f.default <= f.max_val, (
                f"参数 {f.name!r} default {f.default} 不在 [{f.min_val}, {f.max_val}]"
            )

    def test_field_max_delta_positive(self) -> None:
        """每参数 max_delta 必须为正。"""
        for f in ALL_SETUP_FIELDS:
            assert f.max_delta > 0, f"参数 {f.name!r} max_delta 非正"

    def test_field_step_positive(self) -> None:
        """每参数 step 必须为正。"""
        for f in ALL_SETUP_FIELDS:
            assert f.step > 0, f"参数 {f.name!r} step 非正"

    def test_field_source_nonempty(self) -> None:
        """每参数 source 非空（官方出处）。"""
        for f in ALL_SETUP_FIELDS:
            assert f.source, f"参数 {f.name!r} source 为空"

    def test_group_field_counts(self) -> None:
        """7 大类参数数量分布：4/2/4/7/2/2/2 = 23。"""
        expected = {
            "Aerodynamics": 4,
            "Differential": 2,
            "Suspension Geometry": 4,
            "Suspension": 7,
            "Brakes": 2,
            "Tyres": 2,
            "New2026": 2,
        }
        for group, count in expected.items():
            fields = get_fields_by_group(group)
            assert len(fields) == count, (
                f"大类 {group!r} 参数数 {len(fields)} != 期望 {count}"
            )

    def test_get_field_known(self) -> None:
        """get_field 已知名返回定义。"""
        f = get_field("front_wing")
        assert f.name == "front_wing"
        assert f.label == "前翼"
        assert f.min_val == 0.0
        assert f.max_val == 11.0

    def test_get_field_unknown_raises(self) -> None:
        """get_field 未知名抛 KeyError。"""
        with pytest.raises(KeyError):
            get_field("nonexistent")


# ===========================================================================
# 2. validate_value 校验
# ===========================================================================
class TestValidateValue:
    """validate_value 范围与步长校验。"""

    def test_valid_value_returns_snapped(self) -> None:
        """合法值返回对齐后的值。"""
        assert validate_value("front_wing", 5.0) == 5.0
        assert validate_value("front_wing", 0.0) == 0.0
        assert validate_value("front_wing", 11.0) == 11.0

    def test_out_of_range_raises(self) -> None:
        """超出范围抛 ValueError。"""
        with pytest.raises(ValueError):
            validate_value("front_wing", -1.0)
        with pytest.raises(ValueError):
            validate_value("front_wing", 12.0)

    def test_step_misalignment_raises(self) -> None:
        """不符合步长抛 ValueError。"""
        # front_wing step=1.0，传 5.5 不对齐
        with pytest.raises(ValueError):
            validate_value("front_wing", 5.5)

    def test_float_step_alignment(self) -> None:
        """浮点步长参数对齐。"""
        # active_aero_z step=0.1, min=0.0
        assert validate_value("active_aero_z", 0.5) == pytest.approx(0.5)
        assert validate_value("active_aero_z", 0.3) == pytest.approx(0.3)

    def test_unknown_param_raises(self) -> None:
        """未知参数名抛 KeyError。"""
        with pytest.raises(KeyError):
            validate_value("nonexistent", 1.0)


# ===========================================================================
# 3. CarSetup
# ===========================================================================
class TestCarSetup:
    """CarSetup 数据类。"""

    def test_default_has_23_fields(self) -> None:
        """default() 产出 23 字段。"""
        cs = CarSetup.default()
        d = cs.to_dict()
        assert len(d) == 23

    def test_default_values_match_spec(self) -> None:
        """default() 各字段值与 SetupField.default 一致。"""
        cs = CarSetup.default()
        for f in ALL_SETUP_FIELDS:
            assert getattr(cs, f.name) == f.default

    def test_validate_passes_for_default(self) -> None:
        """default() 校验通过。"""
        CarSetup.default().validate()

    def test_from_to_dict_roundtrip(self) -> None:
        """to_dict → from_dict 往返一致。"""
        cs = CarSetup.default()
        d = cs.to_dict()
        cs2 = CarSetup.from_dict(d)
        assert cs2.to_dict() == d

    def test_field_names_count_23(self) -> None:
        """field_names() 返回 23 个字段名。"""
        cs = CarSetup.default()
        assert len(cs.field_names()) == 23

    def test_diff_detects_changes(self) -> None:
        """diff 应检测出变更字段。"""
        cs1 = CarSetup.default()
        cs2 = CarSetup.default()
        cs2.front_wing = 7.0
        changes = cs1.diff(cs2)
        assert len(changes) == 1
        assert changes[0]["name"] == "front_wing"
        assert changes[0]["delta"] == 2.0
        assert changes[0]["direction"] == "increase"


# ===========================================================================
# 4. 症状枚举
# ===========================================================================
class TestSymptoms:
    """12 项症状 / 4 类枚举校验。"""

    def test_symptom_count_is_12(self) -> None:
        """Symptom 枚举必须恰好 12 项。"""
        assert len(list(Symptom)) == 12

    def test_category_count_is_4(self) -> None:
        """SymptomCategory 必须恰好 4 类。"""
        assert len(list(SymptomCategory)) == 4

    def test_categories_expected(self) -> None:
        """4 类名称。"""
        assert {c.value for c in SymptomCategory} == {"entry", "apex", "exit", "global"}

    def test_symptom_info_covers_all(self) -> None:
        """SYMPTOM_INFO 必须覆盖全部 12 症状。"""
        assert set(SYMPTOM_INFO.keys()) == set(Symptom)

    def test_tyre_wear_label_is_胎耗偏高(self) -> None:
        """tyre_wear 中文标签必须为「胎耗偏高」（非「胎温」）。"""
        assert SYMPTOM_INFO[Symptom.TYRE_WEAR]["label"] == "胎耗偏高"

    def test_understeer_label(self) -> None:
        """understeer 中文标签为「转向不足」。"""
        assert SYMPTOM_INFO[Symptom.UNDERSTEER]["label"] == "转向不足"

    def test_get_symptom_label(self) -> None:
        """get_symptom_label 返回中文标签。"""
        assert get_symptom_label(Symptom.UNDERSTEER) == "转向不足"
        assert get_symptom_label(Symptom.TYRE_WEAR) == "胎耗偏高"

    def test_get_symptom_category(self) -> None:
        """get_symptom_category 返回类别字符串。"""
        assert get_symptom_category(Symptom.UNDERSTEER) == "entry"
        assert get_symptom_category(Symptom.MIDCORNER_UNSTABLE) == "apex"
        assert get_symptom_category(Symptom.EXIT_WHEELSPIN) == "exit"
        assert get_symptom_category(Symptom.TYRE_WEAR) == "global"

    def test_get_symptoms_by_category_counts(self) -> None:
        """4 类症状数量分布：entry=5, apex=2, exit=1, global=4。"""
        assert len(get_symptoms_by_category(SymptomCategory.ENTRY)) == 5
        assert len(get_symptoms_by_category(SymptomCategory.APEX)) == 2
        assert len(get_symptoms_by_category(SymptomCategory.EXIT)) == 1
        assert len(get_symptoms_by_category(SymptomCategory.GLOBAL)) == 4

    def test_get_symptoms_by_category_string_arg(self) -> None:
        """get_symptoms_by_category 接受字符串参数。"""
        entry = get_symptoms_by_category("entry")
        assert Symptom.UNDERSTEER in entry
        assert len(entry) == 5

    def test_intensity_constants(self) -> None:
        """强度范围 0-5，默认 3。"""
        assert INTENSITY_MIN == 0
        assert INTENSITY_MAX == 5
        assert DEFAULT_INTENSITY == 3

    def test_validate_intensity_valid(self) -> None:
        """合法强度返回原值。"""
        assert validate_intensity(0) == 0
        assert validate_intensity(3) == 3
        assert validate_intensity(5) == 5

    def test_validate_intensity_invalid(self) -> None:
        """越界强度抛 ValueError。"""
        with pytest.raises(ValueError):
            validate_intensity(-1)
        with pytest.raises(ValueError):
            validate_intensity(6)

    def test_all_symptom_labels_nonempty(self) -> None:
        """每症状中文标签非空。"""
        for sym in Symptom:
            assert SYMPTOM_INFO[sym]["label"], f"症状 {sym!r} 标签为空"

    def test_all_symptom_descriptions_nonempty(self) -> None:
        """每症状描述非空。"""
        for sym in Symptom:
            assert SYMPTOM_INFO[sym]["description"], f"症状 {sym!r} 描述为空"