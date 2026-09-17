"""整体思维调教分析层（holistic）。

为什么需要这一层
----------------
原引擎的链路是 ``症状 → Dx → Dx × C → SetupDelta``，有两个结构性缺陷：

1. **弯道特性根本没参与**。``aggregate_feedback_symptoms`` 把反馈按
   ``(弯道, 症状)`` 聚合后**丢掉了弯道号**，``compute_dx`` 只拿到
   ``(症状, 强度, 阶段)``。于是"T4 慢发夹推头"和"T15 高速弯推头"在模型里
   完全等价 —— 但二者的机理和该动的参数完全不同：慢弯靠机械抓地/牵引
   （悬挂、差速器、离地间隙），快弯靠空气动力学与高速稳定性（翼片、离地间隙、
   防倾杆）。赛道差异此前只靠 ``track_type`` 分 5 组来体现，导致同组赛道
   （如 Monza 与 Jeddah）输出逐位相同。
2. **没有整体权衡**。``Dx × C`` 是线性映射，21 个参数各算各的，既不检查
   参数间的耦合（前后翼平衡、胎压左右对称），也不做"改多少算多"的预算控制，
   更不会告诉车手"这套调教牺牲了什么"。

本模块补齐这两点：

- :func:`track_demand`：由赛道**真实弯道数据**（弯型/速度/连续弯段）算出需求
  画像（牵引/空气动力学/制动三个指数），取代粗粒度的 ``track_type`` 分组。
- :data:`SYMPTOM_CLASS_WEIGHTS`：**分析模型核心** —— 同一症状在不同类型弯道
  上重加权 Dx 维度，让弯道特性真正改变调教方向与幅度。
- :func:`class_weighted_dx`：把逐弯反馈投影成带弯道类别权重的 Dx，并识别
  **跨弯道类别的需求冲突**（慢弯要的与快弯要的相反）。
- :func:`holistic_coherence`：整体性收口 —— 胎压左右对称、前后翼平衡窗口、
  改动预算、冲突提示，并产出可读的**权衡说明**供报告展示。

设计约束：纯函数、无 IO、无随机、无时间依赖（与 ``compute_dx`` 同约定），
保证建议可复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import Any

from setup_tuner.domain._track_arcs import TRACK_CORNER_ARCS
from setup_tuner.domain.corner_groups import build_corner_groups
from setup_tuner.domain.setup import ALL_SETUP_FIELDS
from setup_tuner.domain.track import get_track_by_id

from .coupling import get_coupling
from .diagnostic import DIAG_DIMS, SYMPTOM_STAGE_TO_DX

# --------------------------------------------------------------------------- #
# 弯道类别
# --------------------------------------------------------------------------- #
SLOW = "slow"
MEDIUM = "medium"
FAST = "fast"

#: ``corner_type`` → 弯道类别（当前一一对应，保留映射以便未来细分）。
_CLASS_OF_TYPE: dict[str, str] = {"slow": SLOW, "medium": MEDIUM, "fast": FAST}


def corner_class(corner_type: str) -> str:
    """把 ``Corner.corner_type`` 规整为弯道类别。"""
    return _CLASS_OF_TYPE.get(str(corner_type).strip().lower(), MEDIUM)


# --------------------------------------------------------------------------- #
# 赛道需求画像
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class TrackDemand:
    """由赛道真实弯道数据算出的需求画像。

    三个指数均归一化到 0..1，**不是**官方数值，而是"该赛道对某类能力的
    相对需求强度"，用于缩放敏感度与仲裁冲突。
    """

    track_id: str
    corner_count: int
    slow: int
    medium: int
    fast: int
    slow_share: float
    medium_share: float
    fast_share: float
    avg_speed: float
    min_speed: float
    max_speed: float
    longest_sequence: int
    #: 牵引/机械抓地需求（慢弯多 + 平均速度低 → 高）
    traction_index: float
    #: 下压力/高速稳定性需求（快弯多 + 最高速高 → 高）
    aero_index: float
    #: 制动需求（慢弯多且来速高 → 高）
    braking_index: float

    def describe(self) -> str:
        """可读画像，用于报告与调试。"""
        return (
            f"{self.track_id}: {self.corner_count} 弯"
            f"（慢 {self.slow}/中 {self.medium}/快 {self.fast}），"
            f"均速 {self.avg_speed:.0f} km/h，最长连续弯段 {self.longest_sequence}，"
            f"牵引 {self.traction_index:.2f} / 空力 {self.aero_index:.2f} / "
            f"制动 {self.braking_index:.2f}"
        )


def _longest_sequence(track_id: str) -> int:
    """最长连续弯段所含弯数（连续弯对平衡一致性的要求更高）。"""
    arcs = TRACK_CORNER_ARCS.get(track_id)
    if not arcs:
        return 0
    groups = build_corner_groups(arcs)
    best = 0
    for g in groups:
        members = g.get("members") or []
        best = max(best, len(members))
    return best


@cache
def track_demand(track_id: str) -> TrackDemand:
    """计算赛道需求画像（缓存；赛道数据是静态的）。

    未知赛道返回**中性画像**而不是抛错：引擎对未知/合成赛道 ID 必须保持宽容
    （既有 ``_derive_track_gain`` 对未知赛道也只是中性增益），否则会把
    "赛道表缺项"升级成"建议生成失败"——那属于跨层级的错误放大。
    """
    track = get_track_by_id(track_id)
    if track is None or not track.corners:
        return TrackDemand(
            track_id=track_id, corner_count=0, slow=0, medium=0, fast=0,
            slow_share=0.0, medium_share=0.0, fast_share=0.0,
            avg_speed=0.0, min_speed=0.0, max_speed=0.0, longest_sequence=0,
            traction_index=0.0, aero_index=0.0, braking_index=0.0,
        )

    classes = [corner_class(c.corner_type) for c in track.corners]
    n = len(classes) or 1
    slow = classes.count(SLOW)
    medium = classes.count(MEDIUM)
    fast = classes.count(FAST)
    speeds = [float(c.speed_kmh) for c in track.corners] or [0.0]
    avg_speed = sum(speeds) / len(speeds)
    min_speed, max_speed = min(speeds), max(speeds)

    slow_share = slow / n
    fast_share = fast / n

    # 三个指数的口径（0..1，纯几何/速度统计，无主观常数除了归一化锚点）：
    #   牵引：慢弯占比（慢弯必须靠机械抓地出弯）
    #   空力：快弯占比 × 最高速归一化（快弯越多、尾速越高，下压力越关键）
    #   制动：慢弯占比 × 平均速度归一化（来速高又要重刹 → 制动稳定需求高）
    traction_index = slow_share
    aero_index = fast_share * min(1.0, max_speed / 350.0)
    braking_index = slow_share * min(1.0, avg_speed / 250.0)

    return TrackDemand(
        track_id=track_id,
        corner_count=len(track.corners),
        slow=slow,
        medium=medium,
        fast=fast,
        slow_share=slow_share,
        medium_share=medium / n,
        fast_share=fast_share,
        avg_speed=avg_speed,
        min_speed=min_speed,
        max_speed=max_speed,
        longest_sequence=_longest_sequence(track_id),
        traction_index=round(traction_index, 4),
        aero_index=round(aero_index, 4),
        braking_index=round(braking_index, 4),
    )


# --------------------------------------------------------------------------- #
# 分析模型核心：症状 × 弯道类别 → Dx 维度重加权
# --------------------------------------------------------------------------- #
# 每项：(slow, medium, fast) 三个权重，作用在 ``SYMPTOM_STAGE_TO_DX`` 给出的
# 基础 Dx 贡献之上。权重 >1 放大该维度、<1 抑制。
#
# 机理依据（F1 调教领域共识）：
#   - 慢弯（发夹/低速弯）：速度低、气动效应弱 → 问题基本是**机械**的。
#     推头/转向不足主要靠前机械抓地、离地间隙、差速器解决，动翼片收益很小。
#   - 快弯（高速弯）：气动载荷主导 → 稳定性/抓地问题主要靠**翼片、离地间隙、
#     防倾杆**解决；此时改机械抓地收益有限且易引入不一致。
#   - 中速弯：介于两者之间，权重取 1.0 附近。
_S: dict[str, dict[str, tuple[float, float, float]]] = {
    # 转向不足（推头）
    "understeer": {
        "front_grip_req": (1.15, 1.0, 0.85),   # 慢弯：机械前抓地更关键
        "turnin_req": (1.20, 1.0, 0.80),       # 慢弯：入弯响应靠机械
        "hi_speed_stab_req": (0.50, 1.0, 1.35),  # 快弯：高速稳定性主导
    },
    "midcorner_understeer": {
        "front_grip_req": (1.20, 1.0, 0.85),
        "hi_speed_stab_req": (0.45, 1.0, 1.40),
    },
    # 转向过度
    "oversteer": {
        "rear_grip_req": (1.15, 1.0, 0.85),
        "hi_speed_stab_req": (0.50, 1.0, 1.45),  # 快弯甩尾最危险
        "exit_traction_req": (1.25, 1.0, 0.70),
    },
    "exit_oversteer": {
        "exit_traction_req": (1.30, 1.0, 0.70),  # 慢弯出弯靠牵引
        "hi_speed_stab_req": (0.55, 1.0, 1.30),
    },
    # 牵引 / 出弯打滑
    "exit_wheelspin": {
        "exit_traction_req": (1.35, 1.0, 0.65),
    },
    "midcorner_traction": {
        "exit_traction_req": (1.30, 1.0, 0.75),
        "rear_grip_req": (1.15, 1.0, 0.90),
    },
    # 高速稳定性
    "high_speed_instability": {
        "hi_speed_stab_req": (0.40, 1.0, 1.50),
        "ride_height_req": (0.60, 1.0, 1.25),
    },
    "midcorner_unstable": {
        "hi_speed_stab_req": (0.55, 1.0, 1.35),
    },
    # 入弯
    "turnin_unresponsive": {
        "turnin_req": (1.25, 1.0, 0.85),
    },
    # 制动
    "brake_long": {
        "brake_power_req": (1.20, 1.0, 0.95),
        "brake_stab_req": (0.80, 1.0, 1.20),   # 高速重刹更吃稳定性
    },
    "lockup": {
        "brake_stab_req": (1.15, 1.0, 1.10),
        "brake_power_req": (1.10, 1.0, 0.95),
    },
    # 刮底：慢弯出弯与快弯压缩都会触发，但快弯更依赖离地间隙
    "bottoming": {
        "ride_height_req": (1.10, 1.0, 1.30),
    },
    # 轮胎：慢弯以牵引磨耗为主，快弯以横向载荷为主
    "tyre_wear": {
        "tyre_life_req": (1.15, 1.0, 1.10),
    },
    "tyre_overheat": {
        "tyre_life_req": (1.10, 1.0, 1.15),
    },
}


def class_weights(symptom: str, klass: str) -> dict[str, float]:
    """取某症状在某弯道类别下各 Dx 维度的权重（未列出的维度权重 1.0）。"""
    idx = {SLOW: 0, MEDIUM: 1, FAST: 2}[klass]
    table = _S.get(symptom, {})
    return {dim: w[idx] for dim, w in table.items()}


#: 冲突判定门槛：两类的参数意图都至少达到该幅度才算"真的打架"。
_CONFLICT_MIN = 0.05

#: 参数 → 中文标签（取自 setup 字段定义，避免另造一套名字）。
_PARAM_LABELS: dict[str, str] = {f.name: f.label_zh for f in ALL_SETUP_FIELDS}


def _param_intent(class_dx: dict[str, float]) -> dict[str, float]:
    """把某一弯道类别的 Dx 诉求经耦合矩阵 C 投影为**逐参数方向意图**。

    正值表示该类弯希望该参数增大，负值表示希望减小。幅度为加权和，
    仅用于判断方向是否相冲，不是最终调整量。
    """
    intent: dict[str, float] = {}
    for param in _PARAM_LABELS:
        total = 0.0
        for dim, coef in class_dx.items():
            if not coef:
                continue
            cell = get_coupling(dim, param)
            if cell is None:
                continue
            total += coef * cell.sign * cell.magnitude
        intent[param] = round(total, 6)
    return intent


# --------------------------------------------------------------------------- #
# 逐弯反馈 → 带弯道类别权重的 Dx
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ParamConflict:
    """跨弯道类别的**参数意图冲突** —— 这是"整体思维"的核心矛盾。

    同一调教参数，慢弯区希望往一个方向动、快弯区希望往相反方向动。
    例：慢弯区希望前防倾杆变软（提高机械抓地），快弯区希望变硬
    （抑制侧倾、稳住高速弯）。不可能同时满足，必须按赛道弯型占比
    与车手反馈强度折中，并**明确告知车手牺牲了什么**。

    方向由既有耦合矩阵 C（``engine.coupling``）推出，不是另造一套映射。
    """

    param: str
    label: str
    slow_intent: float
    fast_intent: float

    def describe(self) -> str:
        return (
            f"{self.label}（{self.param}）：慢弯区倾向 {self.slow_intent:+.2f}、"
            f"快弯区倾向 {self.fast_intent:+.2f}，方向相反已折中"
        )


@dataclass(slots=True)
class ClassWeightedDx:
    """逐弯加权后的诊断结果。"""

    dx: dict[str, float]
    by_class: dict[str, dict[str, float]] = field(default_factory=dict)
    conflicts: list[ParamConflict] = field(default_factory=list)
    corner_notes: list[str] = field(default_factory=list)
    demand: TrackDemand | None = None


def _stage_of(symptom: str, stage: str | None) -> str:
    """取有效阶段；非法/缺失时回退到该症状的第一个阶段（默认阶段）。"""
    table = SYMPTOM_STAGE_TO_DX.get(symptom)
    if not table:
        return "global"
    if stage and stage in table:
        return stage
    return next(iter(table))


def class_weighted_dx(
    feedbacks: list[dict[str, Any]], track_id: str,
) -> ClassWeightedDx:
    """把逐弯反馈投影为**带弯道类别权重**的 Dx。

    与 ``aggregate_feedback_symptoms`` 的关键区别：**不丢弃弯道号** ——
    用弯道号查到该弯的类别（慢/中/快），再按类别重加权该症状的 Dx 维度。
    于是"T4 慢发夹推头"与"T15 高速弯推头"会得到不同的 Dx 形状。

    Args:
        feedbacks: 每条含 ``corner_number``（可为 None=全局）、``symptom``、
            ``strength``、``category``（作阶段用）。
        track_id: 赛道标识。

    Returns:
        :class:`ClassWeightedDx`，含最终 Dx、按类别分解、冲突列表、
        逐弯说明。同一 ``(corner, symptom)`` 只保留最强强度（与既有聚合口径一致）。
    """
    demand = track_demand(track_id)
    track = get_track_by_id(track_id)
    # 未知赛道：没有弯道可查 → 全部按中速类处理（保持宽容，不抛错）
    class_by_number = (
        {c.number: corner_class(c.corner_type) for c in track.corners}
        if track is not None else {}
    )

    # 同一 (弯号, 症状) 只取最强强度
    best: dict[tuple[Any, str], dict[str, Any]] = {}
    order: list[tuple[Any, str]] = []
    for fb in feedbacks:
        symptom = fb.get("symptom")
        if symptom not in SYMPTOM_STAGE_TO_DX:
            continue
        key = (fb.get("corner_number"), symptom)
        if key not in best:
            best[key] = fb
            order.append(key)
        elif float(fb.get("strength", 0)) > float(best[key].get("strength", 0)):
            best[key] = fb

    by_class: dict[str, dict[str, float]] = {c: dict.fromkeys(DIAG_DIMS, 0.0)
                                             for c in (SLOW, MEDIUM, FAST)}
    corner_notes: list[str] = []
    for key in order:
        fb = best[key]
        corner_number, symptom = key
        klass = class_by_number.get(corner_number, MEDIUM) if corner_number else MEDIUM
        stage = _stage_of(symptom, fb.get("category"))
        base = SYMPTOM_STAGE_TO_DX[symptom][stage]
        weights = class_weights(symptom, klass)
        strength = float(fb.get("strength", 0) or 0)
        for dim, coef in base.items():
            by_class[klass][dim] += coef * weights.get(dim, 1.0) * (strength / 3.0)
        where = f"T{corner_number}" if corner_number else "全局"
        corner_notes.append(
            f"{where}（{klass}）{symptom}@{stage} 强度{strength:g} "
            f"→ 按 {klass} 特性重加权"
        )

    # 合并：三类求和；再按赛道需求画像做**敏感度微调**（只改幅度不改方向）
    dx = dict.fromkeys(DIAG_DIMS, 0.0)
    for values in by_class.values():
        for dim, v in values.items():
            dx[dim] += v
    tilt = {
        "exit_traction_req": 0.85 + 0.30 * demand.traction_index,
        "front_grip_req": 0.90 + 0.20 * demand.traction_index,
        "rear_grip_req": 0.90 + 0.20 * demand.traction_index,
        "hi_speed_stab_req": 0.85 + 0.30 * demand.aero_index,
        "ride_height_req": 0.90 + 0.20 * demand.aero_index,
        "brake_stab_req": 0.85 + 0.30 * demand.braking_index,
        "brake_power_req": 0.90 + 0.20 * demand.braking_index,
    }
    for dim, factor in tilt.items():
        dx[dim] = round(dx[dim] * factor, 6)

    # 冲突：把两类的 Dx 诉求经耦合矩阵 C 投影到**参数方向**，方向相反即冲突。
    # 这比按 Dx 维度判符号有意义得多 —— 慢弯与快弯常常诉求不同的 Dx 维度，
    # 但真正打架的是对**同一个调教参数**的正反要求。
    slow_intent = _param_intent(by_class[SLOW])
    fast_intent = _param_intent(by_class[FAST])
    conflicts: list[ParamConflict] = []
    for param, label in _PARAM_LABELS.items():
        s_val, f_val = slow_intent.get(param, 0.0), fast_intent.get(param, 0.0)
        if s_val * f_val < 0 and min(abs(s_val), abs(f_val)) >= _CONFLICT_MIN:
            conflicts.append(
                ParamConflict(param, label, round(s_val, 4), round(f_val, 4))
            )
    conflicts.sort(key=lambda c: -min(abs(c.slow_intent), abs(c.fast_intent)))

    return ClassWeightedDx(
        dx=dx, by_class=by_class, conflicts=conflicts,
        corner_notes=corner_notes, demand=demand,
    )


# --------------------------------------------------------------------------- #
# 整体性收口
# --------------------------------------------------------------------------- #
#: 前后翼调整量之差的上限（超过则平衡被极端改变，需要收口）。
_WING_BALANCE_WINDOW = 3.0
#: 单次建议允许改动的参数个数上限（整体思维：一次不要动太多）。
_CHANGE_BUDGET = 12
#: 胎压左右必须一致的参数对。
_PRESSURE_PAIRS = (
    ("front_left_tyre_pressure", "front_right_tyre_pressure"),
    ("rear_left_tyre_pressure", "rear_right_tyre_pressure"),
)


def holistic_coherence(
    delta: dict[str, float], dx: dict[str, float], demand: TrackDemand,
) -> tuple[dict[str, float], list[str]]:
    """对参数调整量做整体性收口，返回 (收口后的 delta, 权衡说明)。

    收口内容：
    1. **胎压左右对称** —— F1 调教中左右胎压必须一致，取两侧均值；
    2. **前后翼平衡窗口** —— ``|Δ前翼 − Δ后翼|`` 不得超窗，否则等比回收，
       避免"为救一个弯把整车平衡推翻"；
    3. **改动预算** —— 一次最多改 ``_CHANGE_BUDGET`` 个参数，超出则保留
       幅度最大的（整体思维：改动越少越可归因）；
    4. **整车抓地不足** —— 前后抓地需求同时很高时提示优先胎温/胎压而非极端翼片。
    """
    out = dict(delta)
    notes: list[str] = []

    # 1. 胎压左右对称
    for left, right in _PRESSURE_PAIRS:
        a, b = out.get(left, 0.0), out.get(right, 0.0)
        if a != b:
            mean = round((a + b) / 2.0, 4)
            out[left] = out[right] = mean
            notes.append(
                f"胎压左右取齐（{left}/{right} → {mean:+.2f}）：左右胎压必须一致"
            )

    # 2. 前后翼平衡窗口
    fw, rw = out.get("front_wing", 0.0), out.get("rear_wing", 0.0)
    imbalance = fw - rw
    if abs(imbalance) > _WING_BALANCE_WINDOW:
        excess = abs(imbalance) - _WING_BALANCE_WINDOW
        shrink = excess / 2.0
        if imbalance > 0:
            out["front_wing"] = round(fw - shrink, 4)
            out["rear_wing"] = round(rw + shrink, 4)
        else:
            out["front_wing"] = round(fw + shrink, 4)
            out["rear_wing"] = round(rw - shrink, 4)
        notes.append(
            f"前后翼平衡收口（差值 {imbalance:+.2f} → "
            f"{out['front_wing'] - out['rear_wing']:+.2f}）：不超过 "
            f"{_WING_BALANCE_WINDOW:g} 级，避免整车平衡被单弯需求推翻"
        )

    # 3. 改动预算
    changed = [p for p, v in out.items() if v]
    if len(changed) > _CHANGE_BUDGET:
        ranked = sorted(changed, key=lambda p: abs(out[p]), reverse=True)
        dropped = ranked[_CHANGE_BUDGET:]
        for p in dropped:
            out[p] = 0.0
        notes.append(
            f"改动预算收口：一次只改 {_CHANGE_BUDGET} 项，"
            f"暂缓幅度最小的 {len(dropped)} 项（{', '.join(sorted(dropped))}）——"
            f"改动越少越容易归因，也避免同时引入多个变量"
        )

    # 4. 整车抓地不足
    if dx.get("front_grip_req", 0.0) > 0.5 and dx.get("rear_grip_req", 0.0) > 0.5:
        notes.append(
            "前/后抓地需求同时偏高：优先通过胎压与胎温进入工作窗口解决，"
            "单纯加大翼片只在高速弯有效，且会拖慢直道"
        )

    # 5. 赛道画像提示（让建议"看得出是为这条赛道做的"）
    # 阈值为弯型占比的经验分界：慢弯占比 ≥40% 即机械抓地主导（Monaco 0.42、
    # Monza 0.45 属此类），快弯占比 ≥35% 即空气动力学主导（Jeddah 0.37）。
    if demand.traction_index >= 0.40:
        notes.append(
            f"本条赛道慢弯占比 {demand.slow_share:.0%}，牵引/机械抓地优先："
            f"离地间隙与差速器比翼片更有效"
        )
    if demand.aero_index >= 0.35:
        notes.append(
            f"本条赛道快弯占比 {demand.fast_share:.0%}、尾速 {demand.max_speed:.0f} km/h，"
            f"下压力与高速稳定性优先"
        )
    if demand.longest_sequence >= 4:
        notes.append(
            f"存在 {demand.longest_sequence} 连弯段：整车平衡要一致，"
            f"不宜为单个弯做极端设定"
        )
    return out, notes
