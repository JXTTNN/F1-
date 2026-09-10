"""CarSetup Schema 单元测试 (Task 2)。

覆盖：注册表规模/分组、默认调教合法性、越界与档位错配校验、
向量往返、游戏格式导出、差异比较。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from f1opt.data.setup_schema import (
    ALL_SETUP_FIELDS,
    DEFAULT_SETUP,
    SETUP_FIELDS,
    CarSetup,
)

EXPECTED_GROUP_ORDER = [
    "Aerodynamics",
    "Active Aero",
    "Transmission",
    "Suspension Geometry",
    "Suspension",
    "Brakes",
    "Tyres",
    "Fuel",
]


def test_setup_fields_registry_size_and_groups() -> None:
    """注册表恰好 23 项，分组集合等于 8 个 garage 分组。"""
    assert len(SETUP_FIELDS) == 23
    assert {f.group for f in SETUP_FIELDS.values()} == set(EXPECTED_GROUP_ORDER)


def test_all_setup_fields_order_grouped() -> None:
    """ALL_SETUP_FIELDS 按 garage 顺序返回，分组连续排列。"""
    fields = ALL_SETUP_FIELDS()
    assert len(fields) == 23
    # 顺序与注册表插入顺序一致
    assert [f.name for f in fields] == list(SETUP_FIELDS.keys())
    # 分组连续出现且顺序正确
    seen: list[str] = []
    for f in fields:
        if not seen or seen[-1] != f.group:
            seen.append(f.group)
    assert seen == EXPECTED_GROUP_ORDER


def test_default_setup_validates() -> None:
    """DEFAULT_SETUP 合法且关键字段符合预期基线。"""
    assert DEFAULT_SETUP.front_wing == 25
    assert DEFAULT_SETUP.rear_wing == 27
    assert DEFAULT_SETUP.on_throttle_diff == 80
    assert DEFAULT_SETUP.front_camber == -3.50
    assert DEFAULT_SETUP.fuel_load == 30.0
    # 再次显式构造不应抛异常
    assert CarSetup(**DEFAULT_SETUP.model_dump()) == DEFAULT_SETUP


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("front_wing", 60),
        ("rear_tyre_pressure", -1),
        ("front_camber", -1.0),
        ("brake_pressure", 79),
        ("front_brake_bias", 56),
        ("fuel_load", 4.0),
    ],
)
def test_out_of_range_raises(field: str, bad_value: object) -> None:
    """越界值必须触发 ValidationError。"""
    base = DEFAULT_SETUP.model_dump()
    base[field] = bad_value
    with pytest.raises(ValidationError):
        CarSetup(**base)


def test_step_misalignment_raises() -> None:
    """档位错配（front_toe=0.055）必须触发 ValidationError。"""
    base = DEFAULT_SETUP.model_dump()
    base["front_toe"] = 0.055
    with pytest.raises(ValidationError):
        CarSetup(**base)


def test_non_int_value_raises() -> None:
    """int 字段传入非整浮点（on_throttle_diff=80.5）必须触发 ValidationError。"""
    base = DEFAULT_SETUP.model_dump()
    base["on_throttle_diff"] = 80.5
    with pytest.raises(ValidationError):
        CarSetup(**base)


def test_boundary_values_accepted() -> None:
    """边界值（min/max）合法。"""
    base = DEFAULT_SETUP.model_dump()
    base["front_wing"] = 0
    base["rear_wing"] = 50
    base["front_camber"] = -2.50
    base["rear_tyre_pressure"] = 19.0
    setup = CarSetup(**base)
    assert setup.front_wing == 0
    assert setup.rear_wing == 50
    assert setup.front_camber == -2.50
    assert setup.rear_tyre_pressure == 19.0


def test_to_vector_length_and_range() -> None:
    """to_vector 长度为 21 且全部落在 [0,1]。"""
    vec = DEFAULT_SETUP.to_vector()
    assert len(vec) == 23
    assert all(0.0 <= v <= 1.0 for v in vec)


def test_vector_round_trip() -> None:
    """to_vector -> from_vector 应还原同一调教（档位对齐后逐字段相等）。"""
    rebuilt = CarSetup.from_vector(DEFAULT_SETUP.to_vector())
    for spec in ALL_SETUP_FIELDS():
        assert getattr(rebuilt, spec.name) == getattr(DEFAULT_SETUP, spec.name)


def test_vector_round_trip_random_setup() -> None:
    """对一份非默认调教同样能往返还原。"""
    custom = CarSetup(
        front_wing=10,
        rear_wing=40,
        on_throttle_diff=70,
        off_throttle_diff=100,
        engine_braking=80,
        front_camber=-3.00,
        rear_camber=-1.50,
        front_toe=0.08,
        rear_toe=0.15,
        front_suspension=5,
        rear_suspension=45,
        front_arb=30,
        rear_arb=2,
        front_ride_height=1,
        rear_ride_height=50,
        brake_pressure=90,
        front_brake_bias=50,
        front_tyre_pressure=26.5,
        rear_tyre_pressure=21.2,
        ballast=5,
        fuel_load=105.0,
        active_aero_mode=0,  # Iter-219
        x_mode_activations=0.0,  # Iter-219
    )
    rebuilt = CarSetup.from_vector(custom.to_vector())
    for spec in ALL_SETUP_FIELDS():
        assert getattr(rebuilt, spec.name) == getattr(custom, spec.name)


def test_from_vector_wrong_length_raises() -> None:
    """向量长度不符应抛出 ValueError。"""
    with pytest.raises(ValueError):
        CarSetup.from_vector([0.0] * 20)


def test_to_dict_all_floats() -> None:
    """to_dict 返回 23 项且值均为 float。"""
    d = DEFAULT_SETUP.to_dict()
    assert len(d) == 23
    assert all(isinstance(v, float) for v in d.values())
    assert d["front_wing"] == 25.0
    assert d["fuel_load"] == 30.0


def test_to_game_format_structure() -> None:
    """to_game_format 含 format 标签与全部 7 个分组。"""
    fmt = DEFAULT_SETUP.to_game_format()
    assert fmt["format"] == "f1-25-setup-v1"
    assert set(fmt["groups"].keys()) == set(EXPECTED_GROUP_ORDER)
    for items in fmt["groups"].values():
        assert len(items) > 0
        for item in items:
            assert {"name", "value", "unit"} <= set(item.keys())
    # 抽查一个具体值
    aero = {it["name"]: it for it in fmt["groups"]["Aerodynamics"]}
    assert aero["front_wing"]["value"] == 25
    assert aero["front_wing"]["unit"] == "clicks"


def test_diff_identical_is_empty() -> None:
    """相同调教的 diff 为空列表。"""
    assert DEFAULT_SETUP.diff(DEFAULT_SETUP) == []


def test_diff_changed_fields() -> None:
    """diff 正确报告变化字段及 before/after/delta。"""
    base = DEFAULT_SETUP.model_dump()
    base["front_wing"] = 30
    base["fuel_load"] = 25.0
    other = CarSetup(**base)
    changes = DEFAULT_SETUP.diff(other)
    names = {c["name"] for c in changes}
    assert names == {"front_wing", "fuel_load"}

    fw = next(c for c in changes if c["name"] == "front_wing")
    assert fw["before"] == 25
    assert fw["after"] == 30
    assert fw["delta"] == 5
    assert fw["group"] == "Aerodynamics"
    assert fw["unit"] == "clicks"

    fuel = next(c for c in changes if c["name"] == "fuel_load")
    assert fuel["before"] == 30.0
    assert fuel["after"] == 25.0
    assert fuel["delta"] == pytest.approx(-5.0)
    assert fuel["group"] == "Fuel"
    assert fuel["unit"] == "kg"
