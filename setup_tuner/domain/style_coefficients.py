"""车手风格系数 —— 让调教建议随「车手开法」变化（task-62 M3 / L1）。

本模块提供**两层**风格调制，职责严格分开：

1. :func:`gain_for_style` —— **幅度倍数**（0.85–1.25）。
   乘进 ``Dx × C`` 之后、clamp 之前。只改幅度、不改方向，因此单靠它
   在「参数已顶到 ``max_delta``」或「诊断本来就没动这个参数」时完全失效
   （实测 6 条风格规则里有 4 条在 24 站上零影响）。

2. :func:`style_trims` —— **风格补偿动作**（±1 档）。
   给风格相关参数一组「按开法偏好」的单档动作（如「转向修正频繁 →
   防倾杆各软 1 档」）。这一步会真的动参数，所以**必须由性能模型仲裁**：
   引擎只在预测圈速代价不超过预算时采纳，并在报告里逐条列出代价
   （见 ``engine.style_targeting``）。方向不改赛道最优解，只做「同速下的
   驾驶性偏好」。

设计约束：
- 倍数限制在 **0.85–1.25**：只改幅度、不改方向；
- 向量缺失 / 样本不足时返回全 1.0 / 空列表（退化为 L0，绝不臆测）；
- 补偿动作单次最多 1 档，且必须整组生效（胎压左右对称）；
- 每条映射都有出处（通用调校共识），倍数是本项目的"敏感度权重"。

映射依据（风格维度下标见 ``telemetry.style_extractor.STYLE_DIMS``）：

    ============================================  ==============================
    风格                                           调教含义
    ============================================  ==============================
    5 循迹刹车占比高                               制动中仍带转向 → 刹车压力
    （trail_braking）                              再大易锁死 → 刹车系增益 ↓
    2 转向平滑度低（修正频繁）                      平台需要更稳 → 悬挂/防倾杆
    （steer_smoothness）                           增益 ↓
    7 轮胎管理差（胎温离散大）                      轮胎已在临界 → 胎压/外倾
    （tyre_management）                            增益 ↓
    6 出弯给油早 + 8 滑移迹象多                     牵引是短板 → 差速/后翼
    （throttle_onset / slip_ratio）                增益 ↑
    1 攻弯强度高                                    入弯响应优先 → 前翼增益 ↑
    （steer_aggression）                           （幅度温和）
    ============================================  ==============================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from setup_tuner.domain.setup import ALL_SETUP_FIELDS

# 风格向量长度（StyleExtractor 输出）
STYLE_VECTOR_LEN = 12

# 中性基线：各维 0.5 视为"没有倾向"
NEUTRAL_BASELINE = 0.5

GAIN_MIN = 0.85
GAIN_MAX = 1.25

NEUTRAL_GAIN: dict[str, float] = {spec.name: 1.0 for spec in ALL_SETUP_FIELDS}

# (风格维度下标, 作用参数组, 每偏离基线 0.1 的增益系数 k)
#   gain = 1 + k * (v - 0.5)，v ∈ [0,1]，结果再夹到 [GAIN_MIN, GAIN_MAX]
_STYLE_RULES: list[tuple[int, tuple[str, ...], float]] = [
    # 5 循迹刹车：刹车系更谨慎
    (4, ("brake_pressure", "brake_bias"), -0.40),
    # 2 转向平滑度低 → 悬挂/防倾杆更保守（平滑度本身已是反向量：越低越粗糙）
    (1, ("front_anti_roll_bar", "rear_anti_roll_bar",
         "front_suspension", "rear_suspension"), -0.30),
    # 7 轮胎管理差 → 胎压/外倾更保守（tyre_management 低 = 离散大）
    (6, ("front_left_tyre_pressure", "front_right_tyre_pressure",
         "rear_left_tyre_pressure", "rear_right_tyre_pressure",
         "front_camber", "rear_camber"), -0.30),
    # 6 出弯给油早 → 差速与后翼 ↑
    (5, ("on_throttle_diff", "off_throttle_diff", "rear_wing"), 0.30),
    # 8 滑移迹象多 → 差速 ↑（锁止差速帮助牵引）
    (7, ("on_throttle_diff", "off_throttle_diff"), 0.20),
    # 1 攻弯强度高 → 前翼 ↑（温和）
    (0, ("front_wing",), 0.20),
]


def gain_for_style(vector: list[float] | None) -> dict[str, float]:
    """按 12 维风格向量返回 {param: 敏感度倍数}（覆盖全部 20 参数）。

    Args:
        vector: 风格向量（12 维，[0,1]）；None / 长度不足 / 含非数值时
            返回全 1.0（不臆测）。

    Returns:
        ``{param: gain}``，全部落在 [GAIN_MIN, GAIN_MAX]。
    """
    gain = dict(NEUTRAL_GAIN)
    if not vector or len(vector) < STYLE_VECTOR_LEN:
        return gain
    try:
        values = [float(v) for v in vector[:STYLE_VECTOR_LEN]]
    except (TypeError, ValueError):
        return gain
    if any(v != v for v in values):  # NaN 防御
        return gain

    for dim_index, params, k in _STYLE_RULES:
        v = values[dim_index]
        factor = 1.0 + k * (v - NEUTRAL_BASELINE)
        factor = max(GAIN_MIN, min(GAIN_MAX, factor))
        for name in params:
            gain[name] = factor
    return gain


def is_style_applicable(vector: list[float] | None, sample_count: int = 0,
                        min_samples: int = 3) -> bool:
    """判断风格向量是否可用于调制（样本数达标且长度合法）。"""
    return (
        vector is not None
        and len(vector) == STYLE_VECTOR_LEN
        and sample_count >= min_samples
    )


# ---------------------------------------------------------------------------
# 第二层：风格补偿动作（±1 档，由性能模型仲裁是否值得）
# ---------------------------------------------------------------------------
#: 偏离中性的门槛：|v-0.5| < 0.1（即 0.4~0.6）视为「没有倾向」，不动作。
#: 取 0.1 而不是更大值，是因为 StyleExtractor 的 12 维向量的中位偏离就在
#: 0.1~0.2 区间（见 usage_flow 评测的实测分布）；门槛过高会让绝大部分
#: 车手退化成"中性"，风格功能等于没开。
TRIM_MIN_DEVIATION = 0.1


@dataclass(frozen=True, slots=True)
class StyleTrim:
    """一条风格补偿动作。

    一条动作可含多个参数（胎压必须左右成对、防倾杆/悬挂成套），
    **整组一起采纳或一起拒绝** —— 半套会让车左右不平衡。
    """

    moves: tuple[tuple[str, float], ...]
    #: 风格维度下标（对应 ``telemetry.style_extractor.STYLE_DIMS``）
    dim: int
    dim_name: str
    #: 中文说明，直接进报告，用户要看得懂"为什么动这个"
    reason: str
    #: 偏离中性的强度 0~1（决定优先级：同样的代价先满足更强的倾向）
    weight: float

    @property
    def params(self) -> tuple[str, ...]:
        return tuple(p for p, _ in self.moves)


# (维度下标, 维度名, 触发侧, 动作, 说明)
# 「high」= 该维偏高才触发；「low」= 偏低才触发。
#
# 为什么有的动作写成"组"、有的写成单项：
# - 前后防倾杆必须成对（只软一端会把平衡打破，等于换了个问题）；
# - 前后悬挂同理；四轮胎压必须整体一致（左右不等直接违反对称性契约）；
# - 锁止差速/后翼/前翼各自独立，能单独存活 —— 捆在一起只会让"代价高的
#   那个成员"把整条动作拖死（实测：后翼 +1 顺带更快，却因为跟锁止差速
#   捆绑而一起被否）。
_TRIM_RULES: list[tuple[int, str, str, tuple[tuple[str, float], ...], str]] = [
    (0, "steer_aggression", "high", (("front_wing", 1.0),),
     "攻弯强度高 → 前翼 +1 档，入弯响应更直接"),
    (1, "steer_smoothness", "low",
     (("front_anti_roll_bar", -1.0), ("rear_anti_roll_bar", -1.0)),
     "转向修正频繁 → 前后防倾杆各软 1 档，扩大容错窗口"),
    (1, "steer_smoothness", "low",
     (("front_suspension", -1.0), ("rear_suspension", -1.0)),
     "转向修正频繁 → 前后悬挂各软 1 档，路肩/颠簸更吃得下"),
    (4, "trail_braking", "high", (("brake_pressure", -1.0),),
     "循迹刹车占比高 → 刹车压力 -1 档，降低带转向锁死风险"),
    (5, "throttle_onset", "high", (("on_throttle_diff", 1.0),),
     "出弯给油早 → 锁止差速 +1 档，牵引更可控"),
    (5, "throttle_onset", "high", (("rear_wing", 1.0),),
     "出弯给油早 → 后翼 +1 档，后轴更稳、敢于早给油"),
    (6, "tyre_management", "low",
     (("front_left_tyre_pressure", -1.0), ("front_right_tyre_pressure", -1.0),
      ("rear_left_tyre_pressure", -1.0), ("rear_right_tyre_pressure", -1.0)),
     "胎温离散大 → 四轮胎压 -1 档（0.1 bar），优先接地面积"),
    (7, "slip_ratio", "high", (("on_throttle_diff", 1.0),),
     "滑移迹象多 → 锁止差速 +1 档，抑制空转"),
    (7, "slip_ratio", "high", (("off_throttle_diff", 1.0),),
     "滑移迹象多 → 收油差速 +1 档，入弯更稳"),
]


def style_trims(vector: list[float] | None) -> list[StyleTrim]:
    """按风格向量返回**风格补偿动作**（可能为空）。

    与 :func:`gain_for_style` 的分工：这里返回的是"要不要额外动一档"，
    不是幅度倍数。返回顺序按 ``weight`` 降序（强倾向优先）。

    同一参数被多条规则命中时（例如给油早 + 滑移多都指向锁止差速），
    取 **单次最大档位（不叠加）**，避免"两条规则 → 改 2 档"的放大。

    Args:
        vector: 12 维风格向量；None / 长度不足 / 含非数值 → 空列表。

    Returns:
        ``list[StyleTrim]``，每条的 ``moves`` 已保证组内参数一整组生效。
    """
    if not vector or len(vector) < STYLE_VECTOR_LEN:
        return []
    try:
        values = [float(v) for v in vector[:STYLE_VECTOR_LEN]]
    except (TypeError, ValueError):
        return []
    if any(v != v for v in values):  # NaN 防御
        return []

    claimed: set[str] = set()
    trims: list[StyleTrim] = []
    for dim, name, side, moves, reason in _TRIM_RULES:
        deviation = (values[dim] - NEUTRAL_BASELINE) / NEUTRAL_BASELINE
        if side == "high" and deviation < TRIM_MIN_DEVIATION:
            continue
        if side == "low" and -deviation < TRIM_MIN_DEVIATION:
            continue
        # 同一参数被多条规则命中时（给油早 + 滑移多都指向锁止差速），
        # 由**先列的规则**认领，绝不叠加成 2 档。
        usable = tuple((p, s) for p, s in moves if p not in claimed)
        if not usable:
            continue
        claimed.update(p for p, _ in usable)
        trims.append(StyleTrim(
            moves=tuple(sorted(usable, key=lambda item: item[0])),
            dim=dim,
            dim_name=name,
            reason=reason,
            weight=round(min(1.0, abs(deviation)), 4),
        ))
    # 强倾向优先：同样的圈速代价，先满足更明显的开法特征
    trims.sort(key=lambda t: (-t.weight, t.dim))
    return trims


def describe_style(vector: list[float] | None,
                   top: int = 3) -> list[dict[str, Any]]:
    """把风格向量翻译成「用户能读的倾向清单」（报告用）。

    只输出偏离中性 ≥ ``TRIM_MIN_DEVIATION`` 的维度，按强度降序；
    偏高的用 ``high``、偏低的用 ``low``，并给出中文标签。
    """
    if not vector or len(vector) < STYLE_VECTOR_LEN:
        return []
    names = _DIM_LABELS
    out: list[dict[str, Any]] = []
    for index, raw in enumerate(vector[:STYLE_VECTOR_LEN]):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value != value:
            continue
        deviation = (value - NEUTRAL_BASELINE) / NEUTRAL_BASELINE
        if abs(deviation) < TRIM_MIN_DEVIATION:
            continue
        out.append({
            "dim": index,
            "name": names[index][0] if index < len(names) else f"dim{index}",
            "label": names[index][1] if index < len(names) else f"维度{index}",
            "side": "high" if deviation > 0 else "low",
            "value": round(value, 3),
            "strength": round(min(1.0, abs(deviation)), 3),
        })
    out.sort(key=lambda item: -item["strength"])
    return out[:top]


#: 与 ``telemetry.style_extractor.STYLE_DIMS`` 同序的中文标签（报告展示用）。
#: 这里**复制一份**而不是 import，是为了不让 domain 层依赖 telemetry 层
#: （domain 是纯规则，telemetry 需要解析报文）。两侧顺序由测试锁定。
_DIM_LABELS: list[tuple[str, str]] = [
    ("steer_aggression", "攻弯强度"),
    ("steer_smoothness", "转向平滑度"),
    ("throttle_aggression", "油门激进度"),
    ("brake_aggression", "刹车激进度"),
    ("trail_braking", "循迹刹车占比"),
    ("throttle_onset", "出弯给油时机"),
    ("tyre_management", "轮胎管理"),
    ("slip_ratio", "滑移迹象"),
    ("straight_speed_ratio", "直道末速比"),
    ("brake_thermal", "制动热负荷"),
    ("slow_corner_ratio", "慢弯占比"),
    ("lap_consistency", "圈速一致性"),
]
