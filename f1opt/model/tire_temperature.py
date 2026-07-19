"""EA F1 2026 轮胎温度窗口模型 (Iter-64).

EA F1 2026 物理引擎中, 轮胎温度是核心抓地力因素:
- 每个化合物有最优工作温度窗口 (°C)
- 低于窗口: 冷胎 = 抓地不足, 圈速损失 (冷启动/暖胎阶段)
- 高于窗口: 过热 = 退化加速 + 抓地下降 (高龄胎/热赛道/软胎)

**Pirelli 2026 工作温度窗口 (基于 Pirelli 公开技术资料 + EA F1 2026 量级)**:
- hard:   95-115°C (窗口宽 20°C, 热稳定)
- medium: 90-110°C (窗口宽 20°C)
- soft:   85-100°C (窗口宽 15°C, 灵敏度高)
- intermediate: 70-90°C
- wet:    60-80°C

**轮胎温度估算物理**:
- 基线 = track_temp + 60°C (摩擦/胎压热高于赛道)
- 冷启动偏移: 第 0 圈 -25°C, 3 圈暖胎至工作温度 (warmup_progress = lap/3)
- 高龄胎热积累: +0.3°C/lap (摩擦磨损累积热)
- 湿地降温: -15°C (水膜蒸发+导热)

公开 API:
    - :func:`tire_temp_at_lap` — 单圈轮胎温度估算 (°C).
    - :func:`tire_temp_penalty_s` — 偏离窗口的圈速惩罚 (s).
    - :func:`tire_temp_window` — 查询化合物工作窗口.
    - :func:`is_in_optimal_window` — 判断当前胎温是否在窗口内.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# EA F1 2026 轮胎温度窗口参数
# --------------------------------------------------------------------------- #
# (optimal_min_c, optimal_max_c, cold_penalty_per_c_s, hot_penalty_per_c_s)
# - cold_penalty: 每低于下界 1°C 的圈速损失 (s/°C)
# - hot_penalty:  每高于上界 1°C 的圈速损失 (s/°C)
_TIRE_TEMP_PARAMS: dict[str, tuple[float, float, float, float]] = {
    # Slick compounds (Pirelli 2026 命名)
    "c0": (95.0, 115.0, 0.020, 0.030),  # 最硬, 窗口宽, 灵敏度低
    "c1": (93.0, 113.0, 0.022, 0.032),
    "c2": (90.0, 110.0, 0.025, 0.040),
    "c3": (88.0, 108.0, 0.028, 0.050),
    "c4": (85.0, 100.0, 0.030, 0.060),  # ~ soft
    "c5": (82.0,  97.0, 0.035, 0.070),  # 最软, 窗口窄, 灵敏度高
    # EA F1 别名
    "hard":         (95.0, 115.0, 0.020, 0.030),
    "medium":       (90.0, 110.0, 0.025, 0.040),
    "soft":         (85.0, 100.0, 0.030, 0.060),
    # 雨胎 (低温窗口)
    "intermediate": (70.0,  90.0, 0.020, 0.030),
    "wet":          (60.0,  80.0, 0.015, 0.025),
}

# 物理常量
_WARMUP_LAPS = 3              # 暖胎圈数 (前 3 圈冷启动偏移)
_BASE_FRICTION_HEAT_C = 60.0  # 摩擦/胎压热高于赛道温度 (°C)
_COLD_START_OFFSET_C = -25.0  # 第 0 圈冷启动偏移 (°C)
_AGE_HEAT_PER_LAP_C = 0.3     # 每圈热积累 (°C/lap, 磨损累积)
_WET_COOLING_C = -15.0        # 湿地降温 (°C)


# --------------------------------------------------------------------------- #
# 核心温度估算
# --------------------------------------------------------------------------- #
def tire_temp_at_lap(
    compound: str,
    track_temp_c: float,
    ambient_temp_c: float,
    lap_in_stint: int,
    tire_age_laps: float,
    wet: bool,
) -> float:
    """估算单圈内轮胎平均工作温度 (°C).

    Args:
        compound: 化合物名 (hard/medium/soft/c0-c5/intermediate/wet).
        track_temp_c: 赛道表面温度 (°C).
        ambient_temp_c: 环境温度 (°C). (保留参数, 当前模型以 track_temp 为主)
        lap_in_stint: 当前 stint 内的圈数 (0-based, 用于冷启动判断).
        tire_age_laps: 轮胎总年龄 (圈, 可分数 — SC 后磨损降速).
        wet: 是否湿地.

    Returns:
        轮胎平均温度 (°C).

    Physics:
        temp = track_temp + 60                          # 摩擦热基线
              + (-25) * (1 - lap/3)  if lap < 3         # 冷启动偏移
              + 0.3 * tire_age                          # 高龄胎热积累
              + (-15)               if wet              # 湿地降温
    """
    # 基线: 摩擦热 + 胎压热 (~60°C) 高于赛道温度
    base = track_temp_c + _BASE_FRICTION_HEAT_C

    # 冷启动偏移 (前 3 圈渐进升温)
    lap = max(0, int(lap_in_stint))
    if lap < _WARMUP_LAPS:
        warmup_progress = lap / _WARMUP_LAPS
        cold_offset = _COLD_START_OFFSET_C * (1.0 - warmup_progress)
    else:
        cold_offset = 0.0

    # 高龄胎热积累 (磨损+摩擦累积热)
    age_heat = max(0.0, float(tire_age_laps)) * _AGE_HEAT_PER_LAP_C

    # 湿地降温 (水膜蒸发+导热)
    wet_cooling = _WET_COOLING_C if wet else 0.0

    return base + cold_offset + age_heat + wet_cooling


def tire_temp_penalty_s(
    compound: str,
    track_temp_c: float,
    ambient_temp_c: float,
    lap_in_stint: int,
    tire_age_laps: float,
    wet: bool,
) -> float:
    """轮胎温度偏离工作窗口的圈速惩罚 (s).

    正值 = 比最优窗口慢 (冷或热).
    0 = 在窗口内 (无惩罚).

    Args:
        同 :func:`tire_temp_at_lap`.

    Returns:
        圈速惩罚 (s).
    """
    params = _TIRE_TEMP_PARAMS.get(compound)
    if params is None:
        return 0.0  # 未知化合物不惩罚
    opt_min, opt_max, cold_pen, hot_pen = params
    temp = tire_temp_at_lap(
        compound, track_temp_c, ambient_temp_c, lap_in_stint, tire_age_laps, wet
    )
    if temp < opt_min:
        return (opt_min - temp) * cold_pen
    if temp > opt_max:
        return (temp - opt_max) * hot_pen
    return 0.0


# --------------------------------------------------------------------------- #
# 工作窗口查询
# --------------------------------------------------------------------------- #
def tire_temp_window(compound: str) -> tuple[float, float]:
    """查询化合物最优工作温度窗口 (°C).

    Returns:
        (optimal_min_c, optimal_max_c). 未知化合物返回 (0.0, 200.0).
    """
    params = _TIRE_TEMP_PARAMS.get(compound)
    if params is None:
        return (0.0, 200.0)
    return (params[0], params[1])


def is_in_optimal_window(
    compound: str,
    track_temp_c: float,
    ambient_temp_c: float,
    lap_in_stint: int,
    tire_age_laps: float,
    wet: bool,
) -> bool:
    """判断当前胎温是否在最优窗口内."""
    params = _TIRE_TEMP_PARAMS.get(compound)
    if params is None:
        return True
    opt_min, opt_max = params[0], params[1]
    temp = tire_temp_at_lap(
        compound, track_temp_c, ambient_temp_c, lap_in_stint, tire_age_laps, wet
    )
    return opt_min <= temp <= opt_max


def temp_state(
    compound: str,
    track_temp_c: float,
    ambient_temp_c: float,
    lap_in_stint: int,
    tire_age_laps: float,
    wet: bool,
) -> str:
    """返回胎温状态: 'cold' / 'optimal' / 'hot'."""
    params = _TIRE_TEMP_PARAMS.get(compound)
    if params is None:
        return "optimal"
    opt_min, opt_max = params[0], params[1]
    temp = tire_temp_at_lap(
        compound, track_temp_c, ambient_temp_c, lap_in_stint, tire_age_laps, wet
    )
    if temp < opt_min:
        return "cold"
    if temp > opt_max:
        return "hot"
    return "optimal"


def all_compounds() -> list[str]:
    """所有支持的化合物名."""
    return list(_TIRE_TEMP_PARAMS.keys())
