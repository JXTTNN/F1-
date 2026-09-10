"""F1 2026 调教参数全集（23 项，7 大类）。

本模块定义调教参数的取值范围、步长、缺省值、单次建议最大调整量与官方出处。
以 spec FR-PARAM-01 为准，与 legacy/f1opt/data/setup_schema.py 的差异如下：

- 新增「阻尼 damping」（legacy 无，缺省取值标注 EA F1 2026 官方调教指南）；
- 「主动空力」拆为 active_aero_z（Z 轴/弯道）与 active_aero_x（X 轴/直道）两个
  独立参数（legacy 为 active_aero_mode 三态 + x_mode_activations）；
- 不含「燃油 fuel_load」（spec 未列入）。

仅依赖标准库 dataclasses，零 ML 依赖，零 pydantic 依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


# 官方出处常量
SRC_UDP_PACKET5 = "EA F1 2026 UDP Specification, Packet 5"
SRC_OFFICIAL_GUIDE = "EA F1 2026 官方调教指南"


@dataclass(frozen=True, slots=True)
class SetupField:
    """单个调教参数定义。

    Attributes:
        name: 参数标识符（与 CarSetup 字段名一致）。
        group: 所属大类。
        label: 中文显示名。
        min_val: 最小值。
        max_val: 最大值。
        step: 步长（档位对齐用）。
        default: 缺省值。
        unit: 单位。
        max_delta: 单次建议最大调整量（用于 clamp）。
        source: 官方出处（EA F1 2026 官方调教指南 / UDP 规范）。
    """

    name: str
    group: str
    label: str
    min_val: float
    max_val: float
    step: float
    default: float
    unit: str
    max_delta: float
    source: str


# ---------------------------------------------------------------------------
# 23 项调教参数定义（按 7 大类顺序排列）
# ---------------------------------------------------------------------------
_FIELD_DEFS: list[SetupField] = [
    # 1. 空力 Aerodynamics (4)
    SetupField(
        name="front_wing",
        group="Aerodynamics",
        label="前翼",
        min_val=0.0,
        max_val=11.0,
        step=1.0,
        default=5.0,
        unit="clicks",
        max_delta=2.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="rear_wing",
        group="Aerodynamics",
        label="后翼",
        min_val=0.0,
        max_val=11.0,
        step=1.0,
        default=5.0,
        unit="clicks",
        max_delta=2.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="active_aero_z",
        group="Aerodynamics",
        label="主动空力Z轴/弯道",
        min_val=0.0,
        max_val=1.0,
        step=0.1,
        default=0.5,
        unit="ratio",
        max_delta=0.2,
        source=SRC_OFFICIAL_GUIDE,
    ),
    SetupField(
        name="active_aero_x",
        group="Aerodynamics",
        label="主动空力X轴/直道",
        min_val=0.0,
        max_val=1.0,
        step=0.1,
        default=0.5,
        unit="ratio",
        max_delta=0.2,
        source=SRC_OFFICIAL_GUIDE,
    ),
    # 2. 差速器 Differential (2)
    SetupField(
        name="on_throttle_diff",
        group="Differential",
        label="油门差速",
        min_val=0.0,
        max_val=100.0,
        step=1.0,
        default=50.0,
        unit="%",
        max_delta=10.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="off_throttle_diff",
        group="Differential",
        label="收油差速",
        min_val=0.0,
        max_val=100.0,
        step=1.0,
        default=50.0,
        unit="%",
        max_delta=10.0,
        source=SRC_UDP_PACKET5,
    ),
    # 3. 悬挂几何 Suspension Geometry (4)
    SetupField(
        name="front_camber",
        group="Suspension Geometry",
        label="前外倾角",
        min_val=-3.5,
        max_val=-1.0,
        step=0.1,
        default=-2.5,
        unit="°",
        max_delta=0.3,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="rear_camber",
        group="Suspension Geometry",
        label="后外倾角",
        min_val=-3.5,
        max_val=-1.0,
        step=0.1,
        default=-2.5,
        unit="°",
        max_delta=0.3,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="front_toe",
        group="Suspension Geometry",
        label="前束角",
        min_val=0.0,
        max_val=0.5,
        step=0.05,
        default=0.25,
        unit="°",
        max_delta=0.1,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="rear_toe",
        group="Suspension Geometry",
        label="后束角",
        min_val=0.0,
        max_val=0.5,
        step=0.05,
        default=0.25,
        unit="°",
        max_delta=0.1,
        source=SRC_UDP_PACKET5,
    ),
    # 4. 悬挂 Suspension (7)
    SetupField(
        name="front_spring",
        group="Suspension",
        label="前弹簧",
        min_val=1.0,
        max_val=500.0,
        step=1.0,
        default=250.0,
        unit="N/mm",
        max_delta=50.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="rear_spring",
        group="Suspension",
        label="后弹簧",
        min_val=1.0,
        max_val=500.0,
        step=1.0,
        default=250.0,
        unit="N/mm",
        max_delta=50.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="front_anti_roll_bar",
        group="Suspension",
        label="前防倾杆",
        min_val=1.0,
        max_val=500.0,
        step=1.0,
        default=250.0,
        unit="N/mm",
        max_delta=50.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="rear_anti_roll_bar",
        group="Suspension",
        label="后防倾杆",
        min_val=1.0,
        max_val=500.0,
        step=1.0,
        default=250.0,
        unit="N/mm",
        max_delta=50.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="front_ride_height",
        group="Suspension",
        label="前行驶高度",
        min_val=1.0,
        max_val=40.0,
        step=1.0,
        default=20.0,
        unit="mm",
        max_delta=5.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="rear_ride_height",
        group="Suspension",
        label="后行驶高度",
        min_val=1.0,
        max_val=40.0,
        step=1.0,
        default=20.0,
        unit="mm",
        max_delta=5.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="damping",
        group="Suspension",
        label="阻尼",
        min_val=1.0,
        max_val=500.0,
        step=1.0,
        default=250.0,
        unit="clicks",
        max_delta=50.0,
        source=SRC_OFFICIAL_GUIDE,  # legacy 无此参数，spec 新增
    ),
    # 5. 刹车 Brakes (2)
    SetupField(
        name="brake_pressure",
        group="Brakes",
        label="刹车压力",
        min_val=50.0,
        max_val=100.0,
        step=1.0,
        default=75.0,
        unit="%",
        max_delta=5.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="brake_bias",
        group="Brakes",
        label="刹车配比",
        min_val=40.0,
        max_val=90.0,
        step=1.0,
        default=65.0,
        unit="%",
        max_delta=5.0,
        source=SRC_UDP_PACKET5,
    ),
    # 6. 轮胎 Tyres (2)
    SetupField(
        name="front_tyre_pressure",
        group="Tyres",
        label="前胎压",
        min_val=16.0,
        max_val=35.0,
        step=0.1,
        default=25.5,
        unit="psi",
        max_delta=1.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="rear_tyre_pressure",
        group="Tyres",
        label="后胎压",
        min_val=16.0,
        max_val=35.0,
        step=0.1,
        default=25.5,
        unit="psi",
        max_delta=1.0,
        source=SRC_UDP_PACKET5,
    ),
    # 7. 2026 新增 New2026 (2)
    SetupField(
        name="engine_braking",
        group="New2026",
        label="发动机制动",
        min_val=0.0,
        max_val=100.0,
        step=1.0,
        default=50.0,
        unit="%",
        max_delta=10.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="ballast",
        group="New2026",
        label="配重",
        min_val=0.0,
        max_val=100.0,
        step=1.0,
        default=50.0,
        unit="kg",
        max_delta=10.0,
        source=SRC_UDP_PACKET5,
    ),
]


# 全部 23 项调教参数（按定义顺序，分组连续）
ALL_SETUP_FIELDS: list[SetupField] = list(_FIELD_DEFS)

# 名称 → SetupField 的快速索引
_FIELD_BY_NAME: dict[str, SetupField] = {f.name: f for f in ALL_SETUP_FIELDS}

# 全部 7 大类
ALL_GROUPS: list[str] = [
    "Aerodynamics",
    "Differential",
    "Suspension Geometry",
    "Suspension",
    "Brakes",
    "Tyres",
    "New2026",
]


def get_field(name: str) -> SetupField:
    """按名称查询单个调教参数定义。

    Args:
        name: 参数标识符。

    Returns:
        对应的 SetupField。

    Raises:
        KeyError: 名称不存在。
    """
    spec = _FIELD_BY_NAME.get(name)
    if spec is None:
        raise KeyError(f"未知的调教参数名: {name!r}")
    return spec


def get_fields_by_group(group: str) -> list[SetupField]:
    """按大类查询该类下全部调教参数（保持定义顺序）。

    Args:
        group: 大类名。

    Returns:
        该类下的 SetupField 列表（可能为空）。
    """
    return [f for f in ALL_SETUP_FIELDS if f.group == group]


def validate_value(name: str, value: float) -> float:
    """校验单个参数取值是否在合法范围且符合步长档位。

    Args:
        name: 参数标识符。
        value: 待校验值。

    Returns:
        对齐到步长档位后的值。

    Raises:
        KeyError: 参数名未知。
        ValueError: 超出范围或不符合步长。
    """
    spec = get_field(name)
    if value < spec.min_val or value > spec.max_val:
        raise ValueError(
            f"{name}={value!r} 超出允许范围 "
            f"[{spec.min_val:g}, {spec.max_val:g}] (步长 {spec.step:g} {spec.unit})"
        )
    # 步长档位对齐：(value - min) 应为 step 的整数倍
    steps = round((value - spec.min_val) / spec.step)
    snapped = spec.min_val + steps * spec.step
    if abs(snapped - value) > 1e-6:
        raise ValueError(
            f"{name}={value!r} 不符合档位步长 {spec.step:g} {spec.unit} "
            f"(最近合法值 {snapped:g})"
        )
    return snapped


# ---------------------------------------------------------------------------
# CarSetup：一份完整调教（23 字段 + 校验）
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class CarSetup:
    """一份完整的 F1 2026 调教（23 项参数，与游戏 garage 一一对应）。

    所有字段类型与 SetupField 定义一致；构造后可调用 :meth:`validate`
    进行范围与步长校验。
    """

    # 1. 空力 Aerodynamics
    front_wing: float = 5.0
    rear_wing: float = 5.0
    active_aero_z: float = 0.5
    active_aero_x: float = 0.5
    # 2. 差速器 Differential
    on_throttle_diff: float = 50.0
    off_throttle_diff: float = 50.0
    # 3. 悬挂几何 Suspension Geometry
    front_camber: float = -2.5
    rear_camber: float = -2.5
    front_toe: float = 0.25
    rear_toe: float = 0.25
    # 4. 悬挂 Suspension
    front_spring: float = 250.0
    rear_spring: float = 250.0
    front_anti_roll_bar: float = 250.0
    rear_anti_roll_bar: float = 250.0
    front_ride_height: float = 20.0
    rear_ride_height: float = 20.0
    damping: float = 250.0
    # 5. 刹车 Brakes
    brake_pressure: float = 75.0
    brake_bias: float = 65.0
    # 6. 轮胎 Tyres
    front_tyre_pressure: float = 25.5
    rear_tyre_pressure: float = 25.5
    # 7. 2026 新增 New2026
    engine_braking: float = 50.0
    ballast: float = 50.0

    def validate(self) -> CarSetup:
        """校验全部 23 字段的范围与步长档位对齐。

        Returns:
            self（校验通过后链式调用）。

        Raises:
            ValueError: 任一字段越界或不符合步长。
        """
        for spec in ALL_SETUP_FIELDS:
            validate_value(spec.name, getattr(self, spec.name))
        return self

    def to_dict(self) -> dict[str, float]:
        """返回扁平字典，所有值归一为 float。"""
        return {spec.name: float(getattr(self, spec.name)) for spec in ALL_SETUP_FIELDS}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CarSetup:
        """从扁平字典构造 CarSetup（仅取已知字段，忽略多余键）。"""
        kwargs: dict[str, float] = {}
        for spec in ALL_SETUP_FIELDS:
            if spec.name in data:
                kwargs[spec.name] = float(data[spec.name])
        return cls(**kwargs)

    @classmethod
    def default(cls) -> CarSetup:
        """返回全部参数取缺省值的 CarSetup。"""
        return cls(**{spec.name: spec.default for spec in ALL_SETUP_FIELDS})

    def diff(self, other: CarSetup) -> list[dict]:
        """返回与 other 不同的字段列表（每项含 name/before/after/delta/unit）。"""
        changes: list[dict] = []
        for spec in ALL_SETUP_FIELDS:
            before = getattr(self, spec.name)
            after = getattr(other, spec.name)
            if before != after:
                delta = float(after) - float(before)
                changes.append(
                    {
                        "name": spec.name,
                        "group": spec.group,
                        "label": spec.label,
                        "before": before,
                        "after": after,
                        "unit": spec.unit,
                        "delta": delta,
                        "direction": "increase" if delta > 0 else "decrease",
                        "delta_steps": round(delta / spec.step) if spec.step > 0 else 0,
                    }
                )
        return changes

    def field_names(self) -> list[str]:
        """返回全部 23 个字段名（按定义顺序）。"""
        return [f.name for f in fields(self)]