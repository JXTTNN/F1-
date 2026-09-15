"""F1 25 (2026 赛季) 调教参数全集（21 项，6 大类）。

本模块定义调教参数的取值范围、步长、缺省值、单次建议最大调整量。
参数严格对齐 EA F1 游戏（F1 25，UDP format=2026）CarSetups 包（Packet 5）
的真实字段，去掉早期版本中臆造的参数（active_aero_z/x、damping 等）。

6 大类 21 项（与游戏 Garage 调教界面一一对应）：
- 空气动力学 Aerodynamics: front_wing / rear_wing
- 变速箱 Transmission: on_throttle_diff / off_throttle_diff
- 悬挂几何 Suspension Geometry: front_camber / rear_camber / front_toe / rear_toe
- 悬挂 Suspension: front_suspension / rear_suspension / front_anti_roll_bar /
  rear_anti_roll_bar / front_ride_height / rear_ride_height
- 刹车 Brakes: brake_pressure / brake_bias / engine_braking
- 轮胎 Tyres: front_left_tyre_pressure / front_right_tyre_pressure /
  rear_left_tyre_pressure / rear_right_tyre_pressure

仅依赖标准库 dataclasses，零 ML 依赖，零 pydantic 依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# 官方出处常量（EA F1 UDP Telemetry Specification, Packet 5 CarSetups）
SRC_UDP_PACKET5 = "EA F1 UDP Spec, Packet 5 (CarSetups)"

# 浮点比较 epsilon（用于步长档位对齐校验）
FLOAT_COMPARE_EPSILON = 1e-6

# group（英文大类）→ category（中文类别名）映射
_GROUP_TO_CATEGORY: dict[str, str] = {
    "Aerodynamics": "空气动力学",
    "Transmission": "变速箱",
    "Suspension Geometry": "悬挂几何",
    "Suspension": "悬挂",
    "Brakes": "刹车",
    "Tyres": "轮胎",
}


@dataclass(frozen=True, slots=True)
class SetupField:
    """单个调教参数定义。

    Attributes:
        name: 参数标识符（与 CarSetup 字段名一致）。
        group: 所属大类（英文标识）。
        label: 中文显示名。
        min_val: 最小值。
        max_val: 最大值。
        step: 步长（档位对齐用）。
        default: 缺省值。
        unit: 单位。
        max_delta: 单次建议最大调整量（用于 clamp）。
        source: 官方出处。
        category: 中文类别名，缺省时从 group 自动派生。
        label_zh: 中文标签，缺省时取 label。
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
    category: str = ""
    label_zh: str = ""

    def __post_init__(self) -> None:
        if not self.category:
            object.__setattr__(
                self, "category",
                _GROUP_TO_CATEGORY.get(self.group, self.group),
            )
        if not self.label_zh:
            object.__setattr__(self, "label_zh", self.label)


# ---------------------------------------------------------------------------
# 21 项调教参数定义（按 6 大类顺序排列）
# ---------------------------------------------------------------------------
_FIELD_DEFS: list[SetupField] = [
    # 1. 空气动力学 Aerodynamics (2)
    SetupField(
        name="front_wing",
        group="Aerodynamics",
        label="前翼",
        min_val=0.0,
        max_val=50.0,
        step=1.0,
        default=25.0,
        unit="级",
        max_delta=5.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="rear_wing",
        group="Aerodynamics",
        label="后翼",
        min_val=0.0,
        max_val=50.0,
        step=1.0,
        default=25.0,
        unit="级",
        max_delta=5.0,
        source=SRC_UDP_PACKET5,
    ),
    # 2. 变速箱 Transmission (2)
    SetupField(
        name="on_throttle_diff",
        group="Transmission",
        label="差速器（开油门）",
        min_val=10.0,
        max_val=100.0,
        step=1.0,
        default=50.0,
        unit="%",
        max_delta=10.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="off_throttle_diff",
        group="Transmission",
        label="差速器（松油门）",
        min_val=10.0,
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
        max_val=-2.5,
        step=0.1,
        default=-3.5,
        unit="°",
        max_delta=0.5,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="rear_camber",
        group="Suspension Geometry",
        label="后外倾角",
        min_val=-2.0,
        max_val=-1.0,
        step=0.1,
        default=-1.5,
        unit="°",
        max_delta=0.5,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="front_toe",
        group="Suspension Geometry",
        label="前束角",
        min_val=0.0,
        max_val=0.2,
        step=0.01,
        default=0.0,
        unit="°",
        max_delta=0.05,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="rear_toe",
        group="Suspension Geometry",
        label="后束角",
        min_val=0.1,
        max_val=0.35,
        step=0.01,
        default=0.2,
        unit="°",
        max_delta=0.05,
        source=SRC_UDP_PACKET5,
    ),
    # 4. 悬挂 Suspension (6)
    SetupField(
        name="front_suspension",
        group="Suspension",
        label="前悬挂",
        min_val=1.0,
        max_val=41.0,
        step=1.0,
        default=6.0,
        unit="级",
        max_delta=2.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="rear_suspension",
        group="Suspension",
        label="后悬挂",
        min_val=1.0,
        max_val=41.0,
        step=1.0,
        default=6.0,
        unit="级",
        max_delta=2.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="front_anti_roll_bar",
        group="Suspension",
        label="前防倾杆",
        min_val=1.0,
        max_val=21.0,
        step=1.0,
        default=6.0,
        unit="级",
        max_delta=2.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="rear_anti_roll_bar",
        group="Suspension",
        label="后防倾杆",
        min_val=1.0,
        max_val=21.0,
        step=1.0,
        default=6.0,
        unit="级",
        max_delta=2.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="front_ride_height",
        group="Suspension",
        label="前行驶高度",
        min_val=15.0,
        max_val=35.0,
        step=1.0,
        default=25.0,
        unit="级",
        max_delta=2.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="rear_ride_height",
        group="Suspension",
        label="后行驶高度",
        min_val=40.0,
        max_val=60.0,
        step=1.0,
        default=50.0,
        unit="级",
        max_delta=2.0,
        source=SRC_UDP_PACKET5,
    ),
    # 5. 刹车 Brakes (3)
    SetupField(
        name="brake_pressure",
        group="Brakes",
        label="刹车压力",
        min_val=80.0,
        max_val=100.0,
        step=1.0,
        default=90.0,
        unit="%",
        max_delta=5.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="brake_bias",
        group="Brakes",
        label="刹车偏置",
        min_val=50.0,
        max_val=70.0,
        step=1.0,
        default=58.0,
        unit="%",
        max_delta=2.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="engine_braking",
        group="Brakes",
        label="引擎制动",
        min_val=0.0,
        max_val=100.0,
        step=1.0,
        default=50.0,
        unit="%",
        max_delta=10.0,
        source=SRC_UDP_PACKET5,
    ),
    # 6. 轮胎 Tyres (4 个独立胎压)
    SetupField(
        name="front_left_tyre_pressure",
        group="Tyres",
        label="前左胎压",
        min_val=22.5,
        max_val=29.5,
        step=0.1,
        default=23.5,
        unit="psi",
        max_delta=1.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="front_right_tyre_pressure",
        group="Tyres",
        label="前右胎压",
        min_val=22.5,
        max_val=29.5,
        step=0.1,
        default=23.5,
        unit="psi",
        max_delta=1.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="rear_left_tyre_pressure",
        group="Tyres",
        label="后左胎压",
        min_val=20.5,
        max_val=26.5,
        step=0.1,
        default=22.0,
        unit="psi",
        max_delta=1.0,
        source=SRC_UDP_PACKET5,
    ),
    SetupField(
        name="rear_right_tyre_pressure",
        group="Tyres",
        label="后右胎压",
        min_val=20.5,
        max_val=26.5,
        step=0.1,
        default=22.0,
        unit="psi",
        max_delta=1.0,
        source=SRC_UDP_PACKET5,
    ),
]


# 全部 21 项调教参数（按定义顺序，分组连续）
ALL_SETUP_FIELDS: list[SetupField] = list(_FIELD_DEFS)

# 名称 → SetupField 的快速索引
_FIELD_BY_NAME: dict[str, SetupField] = {f.name: f for f in ALL_SETUP_FIELDS}

# 全部 6 大类
ALL_GROUPS: list[str] = [
    "Aerodynamics",
    "Transmission",
    "Suspension Geometry",
    "Suspension",
    "Brakes",
    "Tyres",
]


def get_field(name: str) -> SetupField:
    """按名称查询单个调教参数定义。"""
    spec = _FIELD_BY_NAME.get(name)
    if spec is None:
        raise KeyError(f"未知的调教参数名: {name!r}")
    return spec


def get_fields_by_group(group: str) -> list[SetupField]:
    """按大类查询该类下全部调教参数（保持定义顺序）。"""
    return [f for f in ALL_SETUP_FIELDS if f.group == group]


def validate_value(name: str, value: float) -> float:
    """校验单个参数取值是否在合法范围且符合步长档位。"""
    spec = get_field(name)
    if value < spec.min_val or value > spec.max_val:
        raise ValueError(
            f"{name}={value!r} 超出允许范围 "
            f"[{spec.min_val:g}, {spec.max_val:g}] (步长 {spec.step:g} {spec.unit})",
        )
    steps = round((value - spec.min_val) / spec.step)
    snapped = spec.min_val + steps * spec.step
    if abs(snapped - value) > FLOAT_COMPARE_EPSILON:
        raise ValueError(
            f"{name}={value!r} 不符合档位步长 {spec.step:g} {spec.unit} "
            f"(最近合法值 {snapped:g})",
        )
    return snapped


# ---------------------------------------------------------------------------
# CarSetup：一份完整调教（21 字段 + 校验）
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class CarSetup:
    """一份完整的 F1 25 调教（21 项参数，与游戏 Garage 一一对应）。"""

    # 1. 空气动力学 Aerodynamics
    front_wing: float = 25.0
    rear_wing: float = 25.0
    # 2. 变速箱 Transmission
    on_throttle_diff: float = 50.0
    off_throttle_diff: float = 50.0
    # 3. 悬挂几何 Suspension Geometry
    front_camber: float = -3.5
    rear_camber: float = -1.5
    front_toe: float = 0.0
    rear_toe: float = 0.2
    # 4. 悬挂 Suspension
    front_suspension: float = 6.0
    rear_suspension: float = 6.0
    front_anti_roll_bar: float = 6.0
    rear_anti_roll_bar: float = 6.0
    front_ride_height: float = 25.0
    rear_ride_height: float = 50.0
    # 5. 刹车 Brakes
    brake_pressure: float = 90.0
    brake_bias: float = 58.0
    engine_braking: float = 50.0
    # 6. 轮胎 Tyres
    front_left_tyre_pressure: float = 23.5
    front_right_tyre_pressure: float = 23.5
    rear_left_tyre_pressure: float = 22.0
    rear_right_tyre_pressure: float = 22.0

    def validate(self) -> CarSetup:
        """校验全部 21 字段的范围与步长档位对齐。"""
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
                    },
                )
        return changes

    def field_names(self) -> list[str]:
        """返回全部 21 个字段名（按定义顺序）。"""
        return [spec.name for spec in ALL_SETUP_FIELDS]