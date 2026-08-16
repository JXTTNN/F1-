"""F1 25 / 2026 调教参数 Schema.

定义与游戏内 garage 完全一致的调教参数集合（23 项，分 8 组），
包含取值范围与档位步长。:class:`CarSetup` 提供：

- 合法性校验（范围 + 档位对齐）；
- 与 ML 向量互转（``to_vector`` / ``from_vector``，归一化到 [0,1]）；
- 导出为游戏可读分组结构（``to_game_format``）；
- 比较两份调教的差异（``diff``）。

实现说明：仅依赖 ``pydantic``（v2）。全部参数元数据集中在
:data:`SETUP_FIELDS` 注册表中，:class:`CarSetup` 的范围/档位校验通过
``model_validator(mode="after")`` 统一基于该注册表完成，避免为每个字段
重复编写校验器（任务允许的「generic validator keyed off SETUP_FIELDS」方案）。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field, model_validator

GroupName = Literal[
    "Aerodynamics",
    "Active Aero",
    "Transmission",
    "Suspension Geometry",
    "Suspension",
    "Brakes",
    "Tyres",
    "Fuel",
]
FieldKind = Literal["int", "float"]

GAME_FORMAT_TAG = "f1-25-setup-v1"


class SetupField(BaseModel):
    """单个调教参数的元数据描述（名称/分组/类型/范围/步长/单位/说明）。"""

    name: str
    group: GroupName
    kind: FieldKind
    min: float
    max: float
    step: float
    unit: str
    description: str


# (name, group, kind, min, max, step, unit, description) —— 按 garage 显示顺序排列
_FIELD_DEFS: list[tuple[str, GroupName, FieldKind, float, float, float, str, str]] = [
    # Aerodynamics
    ("front_wing", "Aerodynamics", "int", 0.0, 50.0, 1.0, "clicks", "前翼下压力等级"),
    ("rear_wing", "Aerodynamics", "int", 0.0, 50.0, 1.0, "clicks", "后翼下压力等级"),
    # Active Aero (Iter-194: F1 2026 active aero settings)
    ("active_aero_mode", "Active Aero", "int", 0.0, 2.0, 1.0, "mode", "主动空动模式 (0=Z-Mode, 1=Balanced, 2=X-Mode)"),
    ("x_mode_activations", "Active Aero", "int", 0.0, 3.0, 1.0, "count", "X-Mode 每圈激活次数"),
    # Transmission
    ("on_throttle_diff", "Transmission", "int", 50.0, 100.0, 1.0, "percent", "油门差速器锁止率"),
    ("off_throttle_diff", "Transmission", "int", 10.0, 100.0, 1.0, "percent", "收油滑行差速率"),
    # Iter-288: F1 26 发动机制动 (0-100%, 100% = 最大能量回收 + 入弯过度转向)
    ("engine_braking", "Transmission", "int", 0.0, 100.0, 1.0, "percent", "发动机制动"),
    # Suspension Geometry
    ("front_camber", "Suspension Geometry", "float", -3.50, -2.50, 0.01, "degrees", "前轮外倾角"),
    ("rear_camber", "Suspension Geometry", "float", -2.00, -1.00, 0.01, "degrees", "后轮外倾角"),
    ("front_toe", "Suspension Geometry", "float", 0.00, 0.10, 0.01, "degrees", "前轮前束角(外束)"),
    ("rear_toe", "Suspension Geometry", "float", 0.10, 0.30, 0.01, "degrees", "后轮前束角(内束)"),
    # Suspension
    ("front_suspension", "Suspension", "int", 1.0, 50.0, 1.0, "clicks", "前弹簧硬度"),
    ("rear_suspension", "Suspension", "int", 1.0, 50.0, 1.0, "clicks", "后弹簧硬度"),
    ("front_arb", "Suspension", "int", 1.0, 50.0, 1.0, "clicks", "前防倾杆硬度"),
    ("rear_arb", "Suspension", "int", 1.0, 50.0, 1.0, "clicks", "后防倾杆硬度"),
    ("front_ride_height", "Suspension", "int", 1.0, 50.0, 1.0, "clicks", "前离地间隙"),
    ("rear_ride_height", "Suspension", "int", 1.0, 50.0, 1.0, "clicks", "后离地间隙"),
    # Brakes
    ("brake_pressure", "Brakes", "int", 80.0, 100.0, 1.0, "percent", "制动压力上限"),
    ("front_brake_bias", "Brakes", "int", 45.0, 55.0, 1.0, "percent", "前制动分配比"),
    # Tyres
    ("front_tyre_pressure", "Tyres", "float", 21.0, 28.0, 0.1, "psi", "前轮胎压"),
    ("rear_tyre_pressure", "Tyres", "float", 19.0, 25.0, 0.1, "psi", "后轮胎压"),
    # Weight / Fuel
    # Iter-288: F1 26 ballast (My Team 配重, 线格式 uint8 m_ballast; 0-10 clicks 假设)
    ("ballast", "Fuel", "int", 0.0, 10.0, 1.0, "clicks", "配重 (重量分配)"),
    ("fuel_load", "Fuel", "float", 5.0, 110.0, 0.1, "kg", "燃油装载量"),
]

SETUP_FIELDS: dict[str, SetupField] = {
    d[0]: SetupField(
        name=d[0],
        group=d[1],
        kind=d[2],
        min=d[3],
        max=d[4],
        step=d[5],
        unit=d[6],
        description=d[7],
    )
    for d in _FIELD_DEFS
}


def ALL_SETUP_FIELDS() -> list[SetupField]:
    """按游戏 garage 显示顺序（分组连续）返回全部 23 项调教参数。"""
    return [SETUP_FIELDS[d[0]] for d in _FIELD_DEFS]


def _step_decimals(step: float) -> int:
    """根据步长推断小数位数，用于消除浮点噪声。"""
    if step >= 1.0:
        return 0
    return max(0, -int(math.floor(math.log10(step))))


def _snap_to_step(value: float, spec: SetupField) -> float:
    """将 ``value`` 对齐到最近的合法档位并消除浮点噪声。"""
    steps = round((value - spec.min) / spec.step)
    snapped = spec.min + steps * spec.step
    if spec.kind == "int":
        return float(int(round(snapped)))
    return round(snapped, _step_decimals(spec.step))


def _check_value(name: str, spec: SetupField, value: float) -> None:
    """校验范围与档位对齐；违规则抛出 :class:`ValueError`。"""
    if value < spec.min or value > spec.max:
        raise ValueError(
            f"{name}={value!r} 超出允许范围 "
            f"[{spec.min:g}, {spec.max:g}] (步长 {spec.step:g} {spec.unit})"
        )
    snapped = _snap_to_step(value, spec)
    if abs(snapped - value) > 1e-6:
        raise ValueError(
            f"{name}={value!r} 不符合档位步长 {spec.step:g} {spec.unit} "
            f"(最近合法值 {snapped:g})"
        )


class CarSetup(BaseModel):
    """一份完整的 F1 25 / 2026 调教（23 项参数，与游戏 garage 一一对应）。"""

    # --- Aerodynamics ---
    front_wing: int = Field(description="前翼下压力等级")
    rear_wing: int = Field(description="后翼下压力等级")
    # --- Active Aero (Iter-194) ---
    active_aero_mode: int = Field(description="主动空动模式 (0=Z-Mode, 1=Balanced, 2=X-Mode)")
    x_mode_activations: int = Field(description="X-Mode 每圈激活次数")
    # --- Transmission ---
    on_throttle_diff: int = Field(description="油门差速器锁止率")
    off_throttle_diff: int = Field(description="收油滑行差速率")
    engine_braking: int = Field(description="发动机制动 (0-100%)")
    # --- Suspension Geometry ---
    front_camber: float = Field(description="前轮外倾角 (degrees)")
    rear_camber: float = Field(description="后轮外倾角 (degrees)")
    front_toe: float = Field(description="前轮前束角 (degrees)")
    rear_toe: float = Field(description="后轮前束角 (degrees)")
    # --- Suspension ---
    front_suspension: int = Field(description="前弹簧硬度")
    rear_suspension: int = Field(description="后弹簧硬度")
    front_arb: int = Field(description="前防倾杆硬度")
    rear_arb: int = Field(description="后防倾杆硬度")
    front_ride_height: int = Field(description="前离地间隙")
    rear_ride_height: int = Field(description="后离地间隙")
    # --- Brakes ---
    brake_pressure: int = Field(description="制动压力上限")
    front_brake_bias: int = Field(description="前制动分配比")
    # --- Tyres ---
    front_tyre_pressure: float = Field(description="前轮胎压 (psi)")
    rear_tyre_pressure: float = Field(description="后轮胎压 (psi)")
    # --- Weight / Fuel ---
    ballast: int = Field(description="配重 (重量分配)")
    fuel_load: float = Field(description="燃油装载量 (kg)")

    @model_validator(mode="after")
    def _validate_all_fields(self) -> CarSetup:
        """基于 :data:`SETUP_FIELDS` 统一校验全部字段的范围与档位对齐。"""
        for name, spec in SETUP_FIELDS.items():
            _check_value(name, spec, getattr(self, name))
        return self

    def to_dict(self) -> dict[str, float]:
        """返回扁平字典，所有值归一为 float（供 ML 流水线使用）。"""
        return {spec.name: float(getattr(self, spec.name)) for spec in ALL_SETUP_FIELDS()}

    def to_vector(self) -> list[float]:
        """返回长度 = len(ALL_SETUP_FIELDS) 的向量, 归一化到 [0,1]。"""
        return [
            (float(getattr(self, spec.name)) - spec.min) / (spec.max - spec.min)
            for spec in ALL_SETUP_FIELDS()
        ]

    @classmethod
    def from_vector(cls, vec: Sequence[float]) -> CarSetup:
        """:func:`to_vector` 的逆运算：反归一化并对齐到最近档位。"""
        if len(vec) != len(SETUP_FIELDS):
            raise ValueError(
                f"向量长度 {len(vec)} 与调教参数数量 {len(SETUP_FIELDS)} 不一致"
            )
        kwargs: dict[str, int | float] = {}
        for spec, v in zip(ALL_SETUP_FIELDS(), vec, strict=True):
            denorm = spec.min + float(v) * (spec.max - spec.min)
            snapped = _snap_to_step(denorm, spec)
            kwargs[spec.name] = int(snapped) if spec.kind == "int" else snapped
        return cls(**kwargs)  # type: ignore[arg-type]

    def to_game_format(self) -> dict:
        """导出游戏可读的分组结构，含 ``format`` 标签与各组的参数/单位。"""
        groups: dict[str, list[dict]] = {}
        for spec in ALL_SETUP_FIELDS():
            groups.setdefault(spec.group, []).append(
                {
                    "name": spec.name,
                    "value": getattr(self, spec.name),
                    "unit": spec.unit,
                }
            )
        return {"format": GAME_FORMAT_TAG, "groups": groups}

    def diff(self, other: CarSetup) -> list[dict]:
        """返回与 ``other`` 不同的字段列表（每项含 before/after/delta + 语义增强).

        Iter-78 增强: 每项额外含 ``direction`` (increase/decrease) 与
        ``delta_steps`` (变化档数, 正=加档), 让车队工程师一眼看出方向与幅度.
        """
        changes: list[dict] = []
        for spec in ALL_SETUP_FIELDS():
            before = getattr(self, spec.name)
            after = getattr(other, spec.name)
            if before != after:
                delta = float(after) - float(before)
                changes.append(
                    {
                        "name": spec.name,
                        "group": spec.group,
                        "before": before,
                        "after": after,
                        "unit": spec.unit,
                        "delta": delta,
                        "direction": "increase" if delta > 0 else "decrease",
                        "delta_steps": round(delta / spec.step) if spec.step > 0 else 0,
                    }
                )
        return changes


DEFAULT_SETUP = CarSetup(
    front_wing=25,
    rear_wing=27,
    active_aero_mode=1,
    x_mode_activations=2,
    on_throttle_diff=80,
    off_throttle_diff=55,
    engine_braking=50,
    front_camber=-3.50,
    rear_camber=-2.00,
    front_toe=0.05,
    rear_toe=0.20,
    front_suspension=21,
    rear_suspension=11,
    front_arb=10,
    rear_arb=20,
    front_ride_height=20,
    rear_ride_height=40,
    brake_pressure=100,
    front_brake_bias=55,
    front_tyre_pressure=24.0,
    rear_tyre_pressure=20.5,
    ballast=0,
    fuel_load=30.0,
)

# Iter-195: 赛道类型预设调教 (high_downforce / low_downforce / street / mixed)
HIGH_DOWNFORCE_PRESET = CarSetup(
    front_wing=45,
    rear_wing=48,
    active_aero_mode=0,
    x_mode_activations=1,
    on_throttle_diff=85,
    off_throttle_diff=60,
    engine_braking=50,
    front_camber=-3.50,
    rear_camber=-2.00,
    front_toe=0.05,
    rear_toe=0.20,
    front_suspension=18,
    rear_suspension=10,
    front_arb=8,
    rear_arb=18,
    front_ride_height=18,
    rear_ride_height=38,
    brake_pressure=100,
    front_brake_bias=55,
    front_tyre_pressure=23.5,
    rear_tyre_pressure=20.0,
    ballast=0,
    fuel_load=30.0,
)

LOW_DOWNFORCE_PRESET = CarSetup(
    front_wing=5,
    rear_wing=3,
    active_aero_mode=2,
    x_mode_activations=3,
    on_throttle_diff=75,
    off_throttle_diff=50,
    engine_braking=50,
    front_camber=-3.20,
    rear_camber=-1.80,
    front_toe=0.06,
    rear_toe=0.22,
    front_suspension=25,
    rear_suspension=13,
    front_arb=12,
    rear_arb=22,
    front_ride_height=22,
    rear_ride_height=42,
    brake_pressure=100,
    front_brake_bias=55,
    front_tyre_pressure=24.5,
    rear_tyre_pressure=21.0,
    ballast=0,
    fuel_load=30.0,
)

STREET_PRESET = CarSetup(
    front_wing=42,
    rear_wing=45,
    active_aero_mode=0,
    x_mode_activations=0,
    on_throttle_diff=82,
    off_throttle_diff=58,
    engine_braking=50,
    front_camber=-3.40,
    rear_camber=-1.90,
    front_toe=0.04,
    rear_toe=0.18,
    front_suspension=16,
    rear_suspension=9,
    front_arb=7,
    rear_arb=16,
    front_ride_height=25,
    rear_ride_height=45,
    brake_pressure=98,
    front_brake_bias=54,
    front_tyre_pressure=23.0,
    rear_tyre_pressure=19.5,
    ballast=0,
    fuel_load=25.0,
)

MIXED_PRESET = CarSetup(
    front_wing=25,
    rear_wing=27,
    active_aero_mode=1,
    x_mode_activations=2,
    on_throttle_diff=80,
    off_throttle_diff=55,
    engine_braking=50,
    front_camber=-3.40,
    rear_camber=-1.90,
    front_toe=0.05,
    rear_toe=0.20,
    front_suspension=21,
    rear_suspension=11,
    front_arb=10,
    rear_arb=20,
    front_ride_height=20,
    rear_ride_height=40,
    brake_pressure=100,
    front_brake_bias=55,
    front_tyre_pressure=24.0,
    rear_tyre_pressure=20.5,
    ballast=0,
    fuel_load=30.0,
)

# Iter-195: 赛道类型 → 预设调教映射
TRACK_TYPE_PRESETS: dict[str, CarSetup] = {
    "high_downforce": HIGH_DOWNFORCE_PRESET,
    "high_speed_low_downforce": LOW_DOWNFORCE_PRESET,
    "street": STREET_PRESET,
    "mixed": MIXED_PRESET,
    "medium": MIXED_PRESET,
}


def get_track_type_preset(track_type: str) -> CarSetup:
    """Iter-195: 根据赛道类型返回预设调教."""
    return TRACK_TYPE_PRESETS.get(track_type, DEFAULT_SETUP)
