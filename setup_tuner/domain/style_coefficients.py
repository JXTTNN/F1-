"""车手风格系数 —— 让调教建议随「车手开法」变化（task-62 M3 / L1）。

与 :mod:`domain.track_coefficients` 同构：把 12 维风格向量映射为
``{param: 敏感度倍数}``，乘进 ``Dx × C`` 之后、clamp 之前。

设计约束（与 track_coefficients 一致）：
- 倍数限制在 **0.85–1.25**：只改幅度、不改方向；
- 向量缺失 / 样本不足时返回全 1.0（退化为 L0，绝不臆测）；
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
    """按 12 维风格向量返回 {param: 敏感度倍数}（覆盖全部 21 参数）。

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
