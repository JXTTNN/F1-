"""圈级分析模型（lap model）—— 让调教建议真正"算"出来，而不是查表。

为什么需要这一层
----------------
此前的链路是 ``反馈/遥测 → Dx → Dx × C → clamp``：一次线性变换，没有任何
*评价*。它无法回答"这套调教对整圈是好是坏"，也就算不出权衡 —— 因为：

- ``C`` 只给出"每个诊断维度该动哪些参数"，不知道**哪些弯**在乎这个维度；
- 参数改动会**同时**帮一些弯、害另一些弯（加后翼：高速弯更稳，但直道更慢），
  线性一步法没有代价项，无法做这个取舍；
- 遥测是**全圈聚合**的，规则不知道问题出在慢弯还是快弯。

本模块提供三件东西，构成一个可优化的**目标函数**：

1. :data:`CORNER_CAP_DEMAND` —— 弯道类别 → 各能力维度的需求权重
   （慢弯要牵引/机械抓地/重刹，快弯要下压力/高速稳定性）。
2. :func:`corner_evaluations` —— 把赛道**逐个弯**展开：该弯的需求权重、
   重要度（按过弯耗时 ∝ 1/速度；车手报过反馈的弯额外加权）。
3. :func:`objective` —— 圈级目标：加权残差平方和 + 显式代价项
   （翼片阻力、刮底风险、胎温代价、改动幅度）。**代价项是产生"整体思维"的
   关键**：没有它，最优点必然是把所有能力拉满。

供给口径
--------
``C`` 的定义是"每 +1 诊断单位 → 参数应调整量"。反过来，给定参数改动 ``Δp``，
它对某能力维度 ``dim`` 的**供给**为
``supply[dim] = Σ_p C[dim][p] · Δp`` —— 与 ``Dx`` 同量纲（都是诊断单位），
因此残差 ``Dx[dim] − supply[dim]`` 是有意义的。为避免参数量纲不一（前翼
step=1 / 胎压 step=0.1）导致权重失衡，全部改动先按各自的 ``max_delta``
归一化到 ``u_p ∈ [−1, 1]`` 再入模。

所有常数都是**本项目标定的相对权重**，不是官方数值；只影响最优点落在
何处，不宣称绝对精度。方向性来自 F1 调教领域共识。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cache
from typing import Any

from setup_tuner.domain.setup import ALL_SETUP_FIELDS, SetupField
from setup_tuner.domain.track import get_track_by_id

from .coupling import get_coupling
from .diagnostic import DIAG_DIMS
from .holistic import FAST, MEDIUM, SLOW, corner_class, track_demand

# --------------------------------------------------------------------------- #
# 1. 弯道类别 → 能力需求权重
# --------------------------------------------------------------------------- #
# 依据（F1 调教领域共识）：
#   慢弯：气动效应弱，问题基本是机械的 —— 牵引出弯、机械前抓地、入弯响应、
#         重刹稳定；高速稳定性几乎无关。
#   快弯：气动载荷主导 —— 下压力与高速稳定性最关键；牵引需求低。
#   中速弯：折中。
CORNER_CAP_DEMAND: dict[str, dict[str, float]] = {
    SLOW: {
        "front_grip_req": 1.00, "rear_grip_req": 0.90, "turnin_req": 0.90,
        "exit_traction_req": 1.00, "brake_stab_req": 0.80,
        "brake_power_req": 0.70, "hi_speed_stab_req": 0.20,
        "ride_height_req": 0.40, "tyre_life_req": 0.50,
    },
    MEDIUM: {
        "front_grip_req": 1.00, "rear_grip_req": 1.00, "turnin_req": 0.70,
        "exit_traction_req": 0.70, "brake_stab_req": 0.70,
        "brake_power_req": 0.70, "hi_speed_stab_req": 0.70,
        "ride_height_req": 0.50, "tyre_life_req": 0.60,
    },
    FAST: {
        "front_grip_req": 0.80, "rear_grip_req": 0.90, "turnin_req": 0.30,
        "exit_traction_req": 0.20, "brake_stab_req": 0.50,
        "brake_power_req": 0.50, "hi_speed_stab_req": 1.00,
        "ride_height_req": 0.60, "tyre_life_req": 0.70,
    },
}

#: 湿地对**需求**的倾斜（比例）。
#: 依据：湿地下抓地总量下降、水滑风险上升，车更不稳、出弯更易打滑，
#: 因此高速稳定性与出弯牵引的需求上升；而空气动力学收益被水滑限制，
#: 单靠下压力解决前轴抓地的效率下降。
_WET_NEED_TILT: dict[str, float] = {
    "hi_speed_stab_req": 1.30,
    "exit_traction_req": 1.20,
    "brake_stab_req": 1.15,
    "front_grip_req": 0.90,
    "rear_grip_req": 0.95,
}
#: 湿地下可用抓地总量打折
_WET_GRIP_FACTOR = 0.88

#: 车手报过反馈的弯 → 重要度倍率（车手明确在乎的弯，权重更高）
_FEEDBACK_BOOST = 2.0
#: 过弯耗时基准速度（用于重要度 ∝ 1/速度 的归一化）
_TIME_REF_SPEED = 120.0


# --------------------------------------------------------------------------- #
# 2. 参数 → 归一化改动
# --------------------------------------------------------------------------- #
_FIELDS: dict[str, SetupField] = {f.name: f for f in ALL_SETUP_FIELDS}
#: 左右胎压在 F1 调教中必须一致 → 合并为一个自由度（右胎压跟随左胎压）。
_PRESSURE_PAIRS: tuple[tuple[str, str], ...] = (
    ("front_left_tyre_pressure", "front_right_tyre_pressure"),
    ("rear_left_tyre_pressure", "rear_right_tyre_pressure"),
)
_PRESSURE_FOLLOWER: dict[str, str] = {
    right: left for left, right in _PRESSURE_PAIRS
}

#: 可优化参数（右胎压由左胎压决定，不单独优化）
OPTIMIZABLE: tuple[str, ...] = tuple(
    f.name for f in ALL_SETUP_FIELDS if f.name not in _PRESSURE_FOLLOWER
)


@cache
def _c_eff() -> dict[str, dict[str, float]]:
    """``C_eff[dim][param] = C[dim][param].sign · magnitude``。

    含义：**每 1 个诊断单位的需求，该参数应改动的量**（与 C 同义，
    这里只是把符号与幅度显式拆开，供 :func:`supply_from_units` 反用）。
    """
    out: dict[str, dict[str, float]] = {}
    for dim in DIAG_DIMS:
        row: dict[str, float] = {}
        for p in OPTIMIZABLE:
            cell = get_coupling(dim, p)
            if cell is not None and cell.magnitude:
                row[p] = cell.sign * cell.magnitude
        out[dim] = row
    return out


@cache
def _gram_diag() -> dict[str, float]:
    """``(C·Cᵀ)`` 的对角元 ``Σ_p C[dim][p]²`` —— 供给量纲的自洽归一化。

    为什么需要它（两次建模错误的最终结论）：
        ``C`` 是"需求 → 参数调整"的线性设计矩阵，其每一行给出**解决该维度的
        多条可选路径**，行内幅度是按"所有路径同时施加后总效果达标"一起标定的。
        因此：
        - 用 ``Δp × C[dim][p]`` 当供给 → 量纲被放大 ``C²`` 倍，残差瞬间清零，
          最优点退化成"把所有能力拉满"；
        - 用 ``Δp / C[dim][p]`` 当供给 → 把并行的可选路径当成串联叠加，
          经多条参数求和后严重超供，优化器发现"删光改动"代价最低。

        正确做法：取 ``supply = Σ_p Δp·C[dim][p] / Σ_p C[dim][p]²``。
        此时**规则引擎自己那一步**（``Δp = Σ_d need_d·C[d][p]``）恰好使
        ``supply[dim] ≈ need[dim]``（对单维度严格成立），评分即为"缺口 0"。
        于是优化器的任务变成：**用更低的代价达成同样的效果** —— 这正是
        "整体思维"该做的事（少加翼片省阻力、用机械抓地替代气动等）。
    """
    return {
        dim: sum(v * v for v in row.values()) or 1.0
        for dim, row in _c_eff().items()
    }


def apply_weather_to_needs(
    needs: dict[str, dict[str, float]], wet: bool,
) -> dict[str, dict[str, float]]:
    """把天气折算进**需求**向量。

    为什么必须进模型（真实回归）：优化器从"规则引擎的湿地建议"和"干地建议"
    两个起点出发，若目标函数与天气无关，就会收敛到**同一个最优解** ——
    实测湿地/干地输出逐位相同，`_derive_telemetry_gain` 的湿地保守意图被
    完全抹平。天气只有进入目标函数，最优解才会真正随天气改变。
    """
    if not wet:
        return needs
    return {
        klass: {dim: v * _WET_NEED_TILT.get(dim, 1.0) for dim, v in vec.items()}
        for klass, vec in needs.items()
    }


def supply_from_units(units: dict[str, float]) -> dict[str, float]:
    """由归一化改动 ``u_p ∈ [−1,1]`` 算出各能力维度的**供给量**（与 Dx 同量纲）。"""
    c_eff = _c_eff()
    gram = _gram_diag()
    real = {
        p: u * _FIELDS[p].max_delta for p, u in units.items()
        if p in _FIELDS and u
    }
    out: dict[str, float] = {}
    for dim, row in c_eff.items():
        total = 0.0
        for p, delta in real.items():
            coeff = row.get(p)
            if coeff:
                total += delta * coeff
        out[dim] = total / gram[dim]
    return out


# --------------------------------------------------------------------------- #
# 3. 赛道上下文（遥测参与的地方）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class TrackContext:
    """由赛道特征 + 当前遥测状态确定的评价上下文。"""

    track_id: str
    #: 直道速度重要性（均速越高，加翼片的阻力代价越大）
    drag_weight: float
    #: 刮底风险权重（遥测报刮底则显著提高）
    bottoming_weight: float
    #: 胎温/胎耗代价权重（遥测胎温偏高或报过热则提高）
    tyre_heat_weight: float
    #: 抓地供给的遥测修正（胎温/胎压偏离工作窗口时 <1）
    grip_modifier: float
    notes: list[str] = field(default_factory=list)


def _num(value: Any, fallback: float = 0.0) -> float:
    """安全取数：非数值一律回退，避免遥测缺字段把模型带偏。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return float(value)


def track_context(
    track_id: str, telemetry: dict[str, Any] | None,
    dx: dict[str, float] | None = None, wet: bool = False,
) -> TrackContext:
    """由赛道画像 + 遥测摘要算出评价上下文。

    遥测在这里真正起作用：胎温/胎压偏离工作窗口 → 抓地供给打折；
    报刮底 → 刮底代价抬高；胎温高 → 胎耗代价抬高；
    赛道均速高 → 翼片阻力代价抬高。
    """
    demand = track_demand(track_id)
    telemetry = telemetry or {}
    dx = dx or {}
    notes: list[str] = []

    # 直道速度重要性：均速 250+ 的赛道（Monza/Jeddah 类）阻力代价最高
    drag_weight = max(0.5, min(1.6, demand.avg_speed / 160.0))

    bottoming_weight = 1.0
    if dx.get("ride_height_req", 0.0) > 0:
        bottoming_weight += 1.5
        notes.append("遥测报刮底 → 抬高降低离地间隙的代价")

    tyre_heat_weight = 1.0
    brake_temps = telemetry.get("m_brakesTemperature")
    if isinstance(brake_temps, list) and brake_temps:
        avg_brake = sum(_num(t) for t in brake_temps[:4]) / min(4, len(brake_temps))
        if avg_brake > 600:
            tyre_heat_weight += 0.8
            notes.append(f"刹车温度均值 {avg_brake:.0f}°C 偏高 → 抬高制动相关代价")
    if dx.get("tyre_life_req", 0.0) > 0:
        tyre_heat_weight += 0.5

    # 抓地供给修正：轮胎配方越软、胎温越接近窗口，供给越充足
    grip_modifier = 1.0
    if telemetry.get("is_soft_compound"):
        grip_modifier += 0.08
    if telemetry.get("tyres_age_laps") is not None:
        age = _num(telemetry.get("tyres_age_laps"))
        grip_modifier -= min(0.15, age * 0.006)   # 胎龄越高抓地越低
    if telemetry.get("is_hard_compound"):
        grip_modifier -= 0.04
    if wet:
        grip_modifier *= _WET_GRIP_FACTOR
        notes.append("湿地：可用抓地总量下降，稳定性/牵引需求上调")
    grip_modifier = max(0.6, min(1.15, grip_modifier))

    return TrackContext(
        track_id=track_id,
        drag_weight=round(drag_weight, 3),
        bottoming_weight=round(bottoming_weight, 3),
        tyre_heat_weight=round(tyre_heat_weight, 3),
        grip_modifier=round(grip_modifier, 3),
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# 4. 逐弯展开
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class CornerEval:
    """单个弯的评价项。"""

    number: int
    klass: str
    speed_kmh: float
    importance: float
    demand: dict[str, float]


def corner_evaluations(
    track_id: str, feedback_corners: set[int] | None = None,
) -> list[CornerEval]:
    """把赛道展开为逐弯评价项（需求权重 + 重要度）。

    重要度来源（优先级）：
        1. **遥测训练出的逐弯重要度**（``surrogate.corner_importance``）——
           来自 79k+ 个真实 2026 逐弯样本拟合的"该弯占整圈时间比例"；
        2. 回退到启发式 ``120 / 该弯参考速度``（无模型数据时）。

    为什么要把启发式换成模型输出：``120 / speed`` 只反映"弯越慢越耗时"，
    完全不知道这条赛道上**这个具体弯**实际占多少圈速 —— 而优化器正是按
    重要度分配"愿意为它付出多少代价"。换用实测占比后，权重才配得上
    "用遥测数据训练出来的模型"这句话。

    车手报过反馈的弯再乘 :data:`_FEEDBACK_BOOST` —— 车手明确指出的问题
    必须优先被优化，这是"结合车手反馈"的落点。
    """
    track = get_track_by_id(track_id)
    if track is None or not track.corners:
        return []
    boosted = feedback_corners or set()
    learned = _learned_importance(track_id)
    out: list[CornerEval] = []
    for c in track.corners:
        klass = corner_class(c.corner_type)
        speed = max(30.0, float(c.speed_kmh))
        if learned and c.number in learned:
            importance = float(learned[c.number])
        else:
            importance = _TIME_REF_SPEED / speed
        if c.number in boosted:
            importance *= _FEEDBACK_BOOST
        out.append(CornerEval(
            number=c.number, klass=klass, speed_kmh=speed,
            importance=importance, demand=dict(CORNER_CAP_DEMAND[klass]),
        ))
    total = sum(e.importance for e in out) or 1.0
    return [
        CornerEval(e.number, e.klass, e.speed_kmh, round(e.importance / total, 6), e.demand)
        for e in out
    ]


def _learned_importance(track_id: str) -> dict[int, float] | None:
    """取遥测训练出的逐弯重要度；模型不可用时返回 None（中性降级）。"""
    try:
        from .surrogate import get_surrogate

        return get_surrogate().corner_importance(track_id)
    except Exception:  # noqa: BLE001 — 模型层任何异常都不得打断优化主流程
        return None


# --------------------------------------------------------------------------- #
# 5. 目标函数
# --------------------------------------------------------------------------- #
#: 残差平方和之外的代价系数（本项目标定，仅决定最优点位置）
_K_DRAG = 0.06       # 净增翼片 → 阻力代价（乘 drag_weight）
_K_BOTTOMING = 0.10  # 降低离地间隙 → 刮底代价（乘 bottoming_weight）
_K_TYRE_HEAT = 0.05  # 加大负外倾 / 拉高胎压 → 胎温与胎耗代价
#: 改动幅度本身的小惩罚：同样收益下优先少改。
#: **不要再调小**（实测回归）：降到 0.002 时，Monza 与 Monaco 会收敛到
#: 逐位相同的解、湿/干地总幅度不变量也被打破 —— 这个系数正是"权衡"的
#: 交换率，它撑起了"赛道弯型决定取向"。悬挂/几何参数被优化器清零的问题
#: 改由 ``holistic.holistic_coherence`` 的「机械抓地参与度」规则定向修复，
#: 不动这个全局系数。
_K_EFFORT = 0.02
#: 正外倾超过默认值即视为"超出窗口"的阈值（cam·Δ>0 表示更负 → 更热）
_CAMBER_HEAT_GAIN = 1.0


@dataclass(frozen=True, slots=True)
class ObjectiveBreakdown:
    """目标函数分解（用于解释"为什么这么调"）。"""

    residual: float
    drag_cost: float
    bottoming_cost: float
    tyre_heat_cost: float
    effort_cost: float
    per_class: dict[str, float]

    @property
    def total(self) -> float:
        return (self.residual + self.drag_cost + self.bottoming_cost
                + self.tyre_heat_cost + self.effort_cost)


def _costs(units: dict[str, float], ctx: TrackContext) -> tuple[float, float, float, float]:
    """三个显式代价 + 改动幅度惩罚。"""
    fw = units.get("front_wing", 0.0)
    rw = units.get("rear_wing", 0.0)
    # 只有"净增下压力"付出阻力代价；减翼片不罚
    drag = _K_DRAG * ctx.drag_weight * max(0.0, fw + rw) ** 2

    frh = units.get("front_ride_height", 0.0)
    rrh = units.get("rear_ride_height", 0.0)
    bottoming = _K_BOTTOMING * ctx.bottoming_weight * max(0.0, -(frh + rrh)) ** 2

    camber = units.get("front_camber", 0.0) + units.get("rear_camber", 0.0)
    pressure = (
        units.get("front_left_tyre_pressure", 0.0)
        + units.get("rear_left_tyre_pressure", 0.0)
    )
    tyre_heat = _K_TYRE_HEAT * ctx.tyre_heat_weight * (
        (_CAMBER_HEAT_GAIN * max(0.0, camber)) ** 2 + max(0.0, pressure) ** 2
    )

    effort = _K_EFFORT * sum(u * u for u in units.values())
    return drag, bottoming, tyre_heat, effort


def needs_by_class_from_dx(
    dx: dict[str, float],
) -> dict[str, dict[str, float]]:
    """无逐弯信息时：三个类别共用同一份全圈需求（退化为单一需求）。"""
    return {k: dict(dx) for k in (SLOW, MEDIUM, FAST)}


def units_from_delta(
    delta: dict[str, float] | None,
    current_setup: dict[str, float] | None = None,
) -> dict[str, float]:
    """真实 delta → 归一化改动 ``u_p = Δp / max_delta``（与 :data:`OPTIMIZABLE` 同口径）。

    优化器内部用归一化单位评价目标函数；要把"规则引擎给出的初始建议"
    也放进同一套评价（例如算它的需求满足度），就需要这个反变换。
    不做档位对齐（那是 :func:`units_to_delta` 的职责）。
    """
    out: dict[str, float] = {}
    for p in OPTIMIZABLE:
        value = float((delta or {}).get(p, 0.0))
        spec = _FIELDS[p]
        out[p] = value / (spec.max_delta or 1.0)
    return out


def satisfaction_by_class(
    units: dict[str, float],
    needs: dict[str, dict[str, float]],
    track_id: str,
    ctx: TrackContext,
    corners: list[CornerEval] | None = None,
) -> dict[str, float]:
    """各弯道类别的**需求满足度**（0..1，按弯重要度与维度需求权重加权）。

    与 :func:`objective` 的分工：
        ``objective`` 是"越小越好"的圈级代价和，包含阻力/刮底/胎温/改动幅度等
        **代价项**；本函数只回答一个问题 —— **诊断出来的需求，这套调教覆盖了多少**。
    两者必须一起看：只优化目标函数可能靠"少改"把代价压下去却留下缺口；
    只看满足度又可能把车改得又硬又费油。同时给出才能判断"既有效又便宜"。

    满足度 = Σ_corners w(弯,维度) · min(实际供给, 需求) / Σ_corners w · 需求
    其中 ``w = 弯重要度 × 该类弯对该维度的需求权重``。
    """
    supply = supply_from_units(units)
    evals = corners if corners is not None else corner_evaluations(track_id)
    acc: dict[str, list[float]] = {k: [0.0, 0.0] for k in (SLOW, MEDIUM, FAST)}
    for e in evals:
        class_need = needs.get(e.klass) or {}
        for dim in DIAG_DIMS:
            need = class_need.get(dim, 0.0)
            if not need:
                continue
            weight = e.importance * e.demand.get(dim, 0.0)
            progress = (1.0 if need > 0 else -1.0) * supply.get(dim, 0.0) * ctx.grip_modifier
            acc[e.klass][0] += weight * min(max(progress, 0.0), abs(need))
            acc[e.klass][1] += weight * abs(need)
    return {
        k: (round(v[0] / v[1], 4) if v[1] > 1e-12 else 1.0)
        for k, v in acc.items()
    }


def objective(
    units: dict[str, float],
    needs: dict[str, dict[str, float]],
    track_id: str,
    ctx: TrackContext,
    corners: list[CornerEval] | None = None,
) -> ObjectiveBreakdown:
    """圈级目标函数（越小越好）。

    ``units`` 为归一化改动 ``u_p ∈ [−1,1]``；``needs`` 是**按弯道类别分列的
    需求向量**（``{slow|medium|fast: {dim: need}}``）。

    为什么必须按类别分列（这是"整体思维"能否成立的开关）：
        如果所有弯共用同一份全圈需求，那么"所有弯都想要同样的东西"，
        **根本不存在权衡** —— 目标函数退化成同一个二次型，最优点与赛道
        弯型构成无关（实测：Monza/Jeddah/Monaco/Zandvoort 曾算出逐位相同的
        调教）。只有让慢弯与快弯各自带着不同的需求（慢弯重牵引、快弯重
        稳定性），单一参数集才**不可能同时满足**，才会出现"帮了谁、害了谁"
        的取舍，赛道弯型占比才真正决定最优取向。

    评分 = 逐弯加权**缺口**平方和 + 显式代价项。

    残差口径（关键）：**只罚"供不应求"，不罚"供过于求"**。
        ``progress = sign(need) · supply``（与需求同向为正）
        ``deficit  = max(0, |need| − progress)``
    原因：早期版本对超供按同等权重平方惩罚，而规则引擎给出的初值本身就会
    供过于求（C 的幅度按"实际应改动量"标定，多参数叠加后总供给远超需求），
    结果目标被超供罚分完全主导 —— 优化器发现"什么都不改"分数最低，
    得出"最优调教 = 不动"的荒谬结论。
    超供的真实代价（阻力、胎温、改动幅度）由下面的显式代价项承担，
    这也正是"整体思维"里权衡的来源。
    """
    supply = supply_from_units(units)
    evals = corners if corners is not None else corner_evaluations(track_id)

    per_class: dict[str, float] = {SLOW: 0.0, MEDIUM: 0.0, FAST: 0.0}
    total = 0.0
    for e in evals:
        class_need = needs.get(e.klass) or {}
        term = 0.0
        for dim in DIAG_DIMS:
            need = class_need.get(dim, 0.0)
            if not need:
                continue
            progress = (1.0 if need > 0 else -1.0) * supply.get(dim, 0.0) * ctx.grip_modifier
            deficit = max(0.0, abs(need) - progress)
            term += e.demand.get(dim, 0.0) * deficit * deficit
        weighted = e.importance * term
        total += weighted
        per_class[e.klass] += weighted

    drag, bottoming, heat, effort = _costs(units, ctx)
    return ObjectiveBreakdown(
        residual=round(total, 6),
        drag_cost=round(drag, 6),
        bottoming_cost=round(bottoming, 6),
        tyre_heat_cost=round(heat, 6),
        effort_cost=round(effort, 6),
        per_class={k: round(v, 6) for k, v in per_class.items()},
    )


def units_to_delta(
    units: dict[str, float], current_setup: dict[str, float] | None = None,
) -> dict[str, float]:
    """归一化改动 → 真实参数改动。

    两处**正确性约束**（F1 调教参数是分档的，不能给小数）：
    1. **对齐到该参数的离散步长 ``spec.step``**（如悬挂 step=1、胎压 step=0.1）。
       早期版本直接输出 ``u × max_delta``，会得到 front_suspension=0.5 这类
       游戏里根本不存在的档位。
    2. **夹进参数上下限**（给了 ``current_setup`` 时），且不超过 ``max_delta`` ——
       对齐步长后可能略微越界，必须再夹一次。
    右胎压恒跟随左胎压（F1 调教要求左右一致）。
    """
    out: dict[str, float] = {f.name: 0.0 for f in ALL_SETUP_FIELDS}
    for p, u in units.items():
        if not u:
            continue
        spec = _FIELDS.get(p)
        if spec is None:
            continue
        step = spec.step or 0.01
        raw = u * spec.max_delta
        snapped = round(raw / step) * step
        # 对齐后不得超出 max_delta（向下取到档位内）
        limit = max(step, math.floor(spec.max_delta / step) * step)
        snapped = max(-limit, min(limit, snapped))
        if abs(snapped) < step / 2:
            continue
        if current_setup is not None:
            cur = float(current_setup.get(p, spec.default))
            snapped = max(spec.min_val - cur, min(spec.max_val - cur, snapped))
            snapped = round(snapped / step) * step
            if abs(snapped) < step / 2:
                continue
        out[p] = round(snapped, 6)
    for left, right in _PRESSURE_PAIRS:
        if left in out:
            out[right] = out[left]
    return out
