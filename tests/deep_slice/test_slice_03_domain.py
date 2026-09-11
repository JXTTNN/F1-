"""切片 3 深度测试：领域模型（setup + symptoms + track）。

覆盖：
    - setup_tuner.domain.setup（23 参数调教，7 大类，CarSetup + SetupField）
    - setup_tuner.domain.symptoms（12 症状，4 类，Symptom 枚举）
    - setup_tuner.domain.track（24 赛道，Corner/Track 数据模型）

5 种测试方式：
    1. unit     — get_field / validate_value / get_track_by_id 等正常输入正确性
    2. boundary — 越界值、未知名称、空输入、强度越界
    3. property — 往返一致性（to_dict→from_dict）、确定性、default 不变性
    4. static   — 23 参数 / 12 症状 / 24 赛道数量约束、值域约束
    5. smoke    — 真实构造完整 CarSetup → validate → diff 链路
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
from setup_tuner.domain.track import (
    ALL_TRACKS,
    Corner,
    CornerAnchor,
    Track,
    get_all_tracks,
    get_track_by_id,
    get_track_by_udp_id,
)


# ===========================================================================
# 1. 单元测试 (unit) — 每个公开函数的正常输入正确性
# ===========================================================================
class TestUnit:
    """单元测试：验证领域模型公开函数的正常输入正确性。"""

    def test_unit_get_field_returns_setup_field(self) -> None:
        """get_field 应返回对应 SetupField 实例。"""
        f = get_field("front_wing")
        assert isinstance(f, SetupField)
        assert f.name == "front_wing"
        assert f.group == "Aerodynamics"
        assert f.label == "前翼"

    def test_unit_get_fields_by_group(self) -> None:
        """get_fields_by_group 应返回该类下全部参数。"""
        aero = get_fields_by_group("Aerodynamics")
        assert len(aero) == 4
        assert all(f.group == "Aerodynamics" for f in aero)

    def test_unit_validate_value_in_range(self) -> None:
        """validate_value 对合法值应返回对齐后的值。"""
        result = validate_value("front_wing", 5.0)
        assert result == 5.0

    def test_unit_validate_value_returns_snapped_for_aligned(self) -> None:
        """validate_value 对已对齐步长的值应原样返回（对齐后的值）。"""
        # front_wing step=1.0, min=0.0 → 5.0 已在档位上
        result = validate_value("front_wing", 5.0)
        assert result == 5.0
        # active_aero_z step=0.1, min=0.0 → 0.5 已在档位上
        result = validate_value("active_aero_z", 0.5)
        assert result == pytest.approx(0.5)

    def test_unit_car_setup_default(self) -> None:
        """CarSetup.default() 应返回全部缺省值。"""
        s = CarSetup.default()
        assert s.front_wing == 5.0
        assert s.rear_wing == 5.0
        assert s.brake_pressure == 75.0

    def test_unit_car_setup_validate_passes(self) -> None:
        """CarSetup.default().validate() 应通过校验。"""
        s = CarSetup.default()
        assert s.validate() is s

    def test_unit_car_setup_to_dict(self) -> None:
        """CarSetup.to_dict 应返回 23 字段扁平字典。"""
        s = CarSetup.default()
        d = s.to_dict()
        assert len(d) == 23
        assert d["front_wing"] == 5.0

    def test_unit_car_setup_from_dict(self) -> None:
        """CarSetup.from_dict 应从字典构造 CarSetup。"""
        d = {"front_wing": 7.0, "rear_wing": 3.0, "unknown_field": 999}
        s = CarSetup.from_dict(d)
        assert s.front_wing == 7.0
        assert s.rear_wing == 3.0
        # 未提供字段取缺省值
        assert s.brake_pressure == 75.0

    def test_unit_get_symptoms_by_category(self) -> None:
        """get_symptoms_by_category 应返回该类别下全部症状。"""
        entry = get_symptoms_by_category(SymptomCategory.ENTRY)
        assert len(entry) == 5
        assert Symptom.UNDERSTEER in entry

    def test_unit_get_symptom_label(self) -> None:
        """get_symptom_label 应返回中文标签。"""
        assert get_symptom_label(Symptom.UNDERSTEER) == "转向不足"
        assert get_symptom_label(Symptom.TYRE_WEAR) == "胎耗偏高"

    def test_unit_get_symptom_category(self) -> None:
        """get_symptom_category 应返回类别字符串。"""
        assert get_symptom_category(Symptom.UNDERSTEER) == "entry"
        assert get_symptom_category(Symptom.BOTTOMING) == "global"

    def test_unit_validate_intensity_in_range(self) -> None:
        """validate_intensity 对合法值应原样返回。"""
        for v in (0, 1, 3, 5):
            assert validate_intensity(v) == v

    def test_unit_get_track_by_id(self) -> None:
        """get_track_by_id 应返回对应赛道。"""
        t = get_track_by_id("suzuka")
        assert t is not None
        assert t.track_id == "suzuka"
        assert t.country == "Japan"

    def test_unit_get_track_by_udp_id(self) -> None:
        """get_track_by_udp_id 应返回对应赛道。"""
        t = get_track_by_udp_id(2)
        assert t is not None
        assert t.track_id == "suzuka"

    def test_unit_get_all_tracks(self) -> None:
        """get_all_tracks 应返回全部赛道列表。"""
        tracks = get_all_tracks()
        assert len(tracks) == 24


# ===========================================================================
# 2. 边界/异常测试 (boundary) — 越界值、未知名称、空输入
# ===========================================================================
class TestBoundary:
    """边界/异常测试：验证越界与异常输入的处理。"""

    def test_boundary_get_field_unknown_raises(self) -> None:
        """get_field 对未知名称应抛 KeyError。"""
        with pytest.raises(KeyError):
            get_field("nonexistent_param")

    def test_boundary_validate_value_below_min_raises(self) -> None:
        """validate_value 对低于 min 的值应抛 ValueError。"""
        with pytest.raises(ValueError, match="超出允许范围"):
            validate_value("front_wing", -1.0)

    def test_boundary_validate_value_above_max_raises(self) -> None:
        """validate_value 对高于 max 的值应抛 ValueError。"""
        with pytest.raises(ValueError, match="超出允许范围"):
            validate_value("front_wing", 12.0)

    def test_boundary_validate_value_wrong_step_raises(self) -> None:
        """validate_value 对不符合步长的值应抛 ValueError。"""
        # active_aero_z step=0.1, min=0.0 → 0.25 不在档位上
        with pytest.raises(ValueError, match="不符合档位步长"):
            validate_value("active_aero_z", 0.25)

    def test_boundary_validate_value_at_min_max(self) -> None:
        """validate_value 对边界值（min/max）应通过。"""
        assert validate_value("front_wing", 0.0) == 0.0
        assert validate_value("front_wing", 11.0) == 11.0

    def test_boundary_validate_intensity_below_zero(self) -> None:
        """validate_intensity 对 < 0 应抛 ValueError。"""
        with pytest.raises(ValueError, match="越界"):
            validate_intensity(-1)

    def test_boundary_validate_intensity_above_five(self) -> None:
        """validate_intensity 对 > 5 应抛 ValueError。"""
        with pytest.raises(ValueError, match="越界"):
            validate_intensity(6)

    def test_boundary_get_track_by_id_unknown_returns_none(self) -> None:
        """get_track_by_id 对未知 ID 应返回 None。"""
        assert get_track_by_id("nonexistent") is None

    def test_boundary_get_track_by_udp_id_unknown_returns_none(self) -> None:
        """get_track_by_udp_id 对未知 udp_id 应返回 None。"""
        assert get_track_by_udp_id(999) is None
        assert get_track_by_udp_id(-1) is None

    def test_boundary_get_fields_by_group_unknown_returns_empty(self) -> None:
        """get_fields_by_group 对未知大类应返回空列表。"""
        assert get_fields_by_group("UnknownGroup") == []

    def test_boundary_get_symptoms_by_category_unknown_returns_empty(self) -> None:
        """get_symptoms_by_category 对未知类别应返回空列表。"""
        assert get_symptoms_by_category("unknown_category") == []

    def test_boundary_car_setup_validate_invalid_raises(self) -> None:
        """CarSetup 含非法字段值时 validate 应抛 ValueError。"""
        s = CarSetup(front_wing=99.0)  # 超出 max=11
        with pytest.raises(ValueError):
            s.validate()

    def test_boundary_car_setup_from_dict_ignores_extra_keys(self) -> None:
        """CarSetup.from_dict 应忽略多余键（防御性）。"""
        s = CarSetup.from_dict({"front_wing": 7.0, "junk": 999})
        assert s.front_wing == 7.0
        assert not hasattr(s, "junk")


# ===========================================================================
# 3. 属性不变量测试 (property) — 往返一致性、确定性
# ===========================================================================
class TestProperty:
    """属性不变量测试：验证往返一致性与确定性。"""

    def test_property_roundtrip_to_dict_from_dict(self) -> None:
        """往返一致性：to_dict → from_dict 应恢复相同 CarSetup。"""
        original = CarSetup(front_wing=7.0, rear_wing=3.0, brake_pressure=80.0)
        d = original.to_dict()
        restored = CarSetup.from_dict(d)
        assert restored.to_dict() == d

    def test_property_default_to_dict_has_23_fields(self) -> None:
        """不变量：default().to_dict() 恰好含 23 个字段。"""
        d = CarSetup.default().to_dict()
        assert len(d) == 23

    def test_property_default_is_idempotent(self) -> None:
        """幂等性：多次调用 default() 结果一致。"""
        d1 = CarSetup.default().to_dict()
        d2 = CarSetup.default().to_dict()
        assert d1 == d2

    def test_property_diff_symmetric_count(self) -> None:
        """不变量：A.diff(B) 与 B.diff(A) 的字段数相同（方向相反）。"""
        a = CarSetup(front_wing=5.0, rear_wing=5.0)
        b = CarSetup(front_wing=7.0, rear_wing=3.0)
        diff_ab = a.diff(b)
        diff_ba = b.diff(a)
        assert len(diff_ab) == len(diff_ba)
        # 方向相反
        for d1, d2 in zip(diff_ab, diff_ba, strict=True):
            assert d1["delta"] == -d2["delta"]

    def test_property_diff_no_change_returns_empty(self) -> None:
        """不变量：相同 CarSetup 的 diff 为空。"""
        s = CarSetup.default()
        assert s.diff(s) == []

    def test_property_symptom_info_covers_all(self) -> None:
        """不变量：SYMPTOM_INFO 覆盖全部 12 个 Symptom 枚举。"""
        for sym in Symptom:
            assert sym in SYMPTOM_INFO
            assert "label" in SYMPTOM_INFO[sym]
            assert "category" in SYMPTOM_INFO[sym]

    def test_property_track_udp_ids_unique(self) -> None:
        """不变量：24 条赛道的 udp_track_id 互不重复。"""
        udp_ids = [t.udp_track_id for t in ALL_TRACKS]
        assert len(udp_ids) == len(set(udp_ids)) == 24

    def test_property_track_ids_unique(self) -> None:
        """不变量：24 条赛道的 track_id 互不重复。"""
        track_ids = [t.track_id for t in ALL_TRACKS]
        assert len(track_ids) == len(set(track_ids)) == 24

    def test_property_corner_anchors_in_unit_square(self) -> None:
        """不变量：所有弯道锚点在 (0, 1) 开区间内。"""
        for track in ALL_TRACKS:
            for corner in track.corners:
                assert 0.0 < corner.anchor.anchor_x < 1.0
                assert 0.0 < corner.anchor.anchor_y < 1.0

    def test_property_corner_numbers_sequential(self) -> None:
        """不变量：每条赛道的弯道编号从 1 连续递增。"""
        for track in ALL_TRACKS:
            numbers = [c.number for c in track.corners]
            assert numbers == list(range(1, len(numbers) + 1))


# ===========================================================================
# 4. 静态分析 (static) — 数量约束、值域约束
# ===========================================================================
class TestStatic:
    """静态分析：验证数量与值域约束。"""

    def test_static_setup_field_count_is_23(self) -> None:
        """数量约束：ALL_SETUP_FIELDS 恰好 23 项。"""
        assert len(ALL_SETUP_FIELDS) == 23

    def test_static_group_count_is_7(self) -> None:
        """数量约束：ALL_GROUPS 恰好 7 大类。"""
        assert len(ALL_GROUPS) == 7
        assert ALL_GROUPS == [
            "Aerodynamics", "Differential", "Suspension Geometry",
            "Suspension", "Brakes", "Tyres", "New2026",
        ]

    def test_static_symptom_count_is_12(self) -> None:
        """数量约束：Symptom 枚举恰好 12 项。"""
        assert len(list(Symptom)) == 12

    def test_static_track_count_is_24(self) -> None:
        """数量约束：ALL_TRACKS 恰好 24 条赛道。"""
        assert len(ALL_TRACKS) == 24

    @pytest.mark.parametrize("field", ALL_SETUP_FIELDS)
    def test_static_each_field_min_le_default_le_max(self, field: SetupField) -> None:
        """参数化：每个参数的 default 在 [min, max] 区间内。"""
        assert field.min_val <= field.default <= field.max_val

    @pytest.mark.parametrize("field", ALL_SETUP_FIELDS)
    def test_static_each_field_max_delta_positive(self, field: SetupField) -> None:
        """参数化：每个参数的 max_delta > 0。"""
        assert field.max_delta > 0

    @pytest.mark.parametrize("field", ALL_SETUP_FIELDS)
    def test_static_each_field_step_positive(self, field: SetupField) -> None:
        """参数化：每个参数的 step > 0。"""
        assert field.step > 0

    def test_static_intensity_constants(self) -> None:
        """值域约束：强度常量 INTENSITY_MIN=0, MAX=5, DEFAULT=3。"""
        assert INTENSITY_MIN == 0
        assert INTENSITY_MAX == 5
        assert DEFAULT_INTENSITY == 3

    @pytest.mark.parametrize("sym", list(Symptom))
    def test_static_each_symptom_has_info(self, sym: Symptom) -> None:
        """参数化：每个症状在 SYMPTOM_INFO 中有完整元信息。"""
        info = SYMPTOM_INFO[sym]
        assert "label" in info
        assert "category" in info
        assert "description" in info
        assert info["category"] in ("entry", "apex", "exit", "global")

    @pytest.mark.parametrize("track", ALL_TRACKS)
    def test_static_each_track_has_corners(self, track: Track) -> None:
        """参数化：每条赛道至少有 1 个弯道。"""
        assert len(track.corners) >= 1

    @pytest.mark.parametrize("track", ALL_TRACKS)
    def test_static_each_track_length_positive(self, track: Track) -> None:
        """参数化：每条赛道长度 > 0。"""
        assert track.length_m > 0

    @pytest.mark.parametrize("track", ALL_TRACKS)
    def test_static_each_track_type_valid(self, track: Track) -> None:
        """参数化：每条赛道类型在合法枚举内。"""
        assert track.track_type in (
            "high_speed_low_downforce", "street", "high_downforce", "medium", "mixed",
        )

    def test_static_setup_field_is_frozen(self) -> None:
        """类型约束：SetupField 为 frozen dataclass。"""
        f = get_field("front_wing")
        with pytest.raises((AttributeError, TypeError)):
            f.name = "modified"  # type: ignore[misc]

    def test_static_corner_anchor_is_frozen(self) -> None:
        """类型约束：CornerAnchor 为 frozen dataclass。"""
        a = CornerAnchor(anchor_x=0.5, anchor_y=0.5)
        with pytest.raises((AttributeError, TypeError)):
            a.anchor_x = 0.9  # type: ignore[misc]


# ===========================================================================
# 5. 实际运行冒烟 (smoke) — 真实构造完整 CarSetup → validate → diff
# ===========================================================================
class TestSmoke:
    """实际运行冒烟：用真实数据走完整领域模型链路。"""

    def test_smoke_full_setup_validate_diff_chain(self) -> None:
        """冒烟：构造完整 CarSetup → validate → diff 链路。"""
        original = CarSetup.default()
        original.validate()

        # 构造修改后的调教
        modified = CarSetup(
            front_wing=7.0, rear_wing=3.0,
            brake_pressure=80.0, brake_bias=70.0,
            front_tyre_pressure=26.5, rear_tyre_pressure=24.5,
        )
        modified.validate()

        diff = original.diff(modified)
        # 应有 6 个字段变化
        changed_names = {d["name"] for d in diff}
        assert "front_wing" in changed_names
        assert "rear_wing" in changed_names
        assert "brake_pressure" in changed_names

    def test_smoke_all_24_tracks_accessible(self) -> None:
        """冒烟：全部 24 条赛道可通过 get_track_by_id 访问。"""
        for track in ALL_TRACKS:
            fetched = get_track_by_id(track.track_id)
            assert fetched is not None
            assert fetched.track_id == track.track_id
            assert fetched.udp_track_id == track.udp_track_id

    def test_smoke_all_12_symptoms_categorized(self) -> None:
        """冒烟：全部 12 症状可按类别检索。"""
        all_categorized: list[Symptom] = []
        for cat in SymptomCategory:
            all_categorized.extend(get_symptoms_by_category(cat))
        assert len(all_categorized) == 12
        assert set(all_categorized) == set(Symptom)

    def test_smoke_setup_roundtrip_with_modifications(self) -> None:
        """冒烟：修改多个参数 → to_dict → from_dict → diff 应一致。"""
        original = CarSetup.default()
        modified = CarSetup(
            front_wing=8.0, rear_wing=4.0,
            on_throttle_diff=60.0, off_throttle_diff=40.0,
            brake_pressure=78.0,
        )
        d = modified.to_dict()
        restored = CarSetup.from_dict(d)
        assert restored.to_dict() == d
        diff1 = original.diff(modified)
        diff2 = original.diff(restored)
        assert len(diff1) == len(diff2)

    def test_smoke_track_to_corners_chain(self) -> None:
        """冒烟：赛道 → 弯道列表 → 锚点链路完整可访问。"""
        suzuka = get_track_by_id("suzuka")
        assert suzuka is not None
        assert len(suzuka.corners) == 18
        # 第 1 个弯道
        c1 = suzuka.corners[0]
        assert isinstance(c1, Corner)
        assert c1.number == 1
        assert c1.corner_type in ("slow", "medium", "fast")
        assert isinstance(c1.anchor, CornerAnchor)

    def test_smoke_all_groups_have_fields(self) -> None:
        """冒烟：7 大类每类至少有 1 个参数。"""
        for group in ALL_GROUPS:
            fields = get_fields_by_group(group)
            assert len(fields) >= 1, f"大类 {group} 无参数"

    def test_smoke_validate_all_default_fields(self) -> None:
        """冒烟：全部 23 参数的缺省值都应通过 validate_value。"""
        for spec in ALL_SETUP_FIELDS:
            result = validate_value(spec.name, spec.default)
            assert result == pytest.approx(spec.default)