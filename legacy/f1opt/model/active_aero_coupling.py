"""F1 2026 主动空动与圈速精确耦合 (Iter-55).

将 :mod:`f1opt.model.active_aero` 的 X-mode 激活收益精确耦合到圈速模型,
严格契合 EA F1 2026 物理引擎.

**EA F1 2026 主动空动物理 (对标游戏引擎)**:
- **Z-mode (弯角)**: 默认高下压力, 弯角抓地力基准.
- **X-mode (直道)**: 翼面"张开"减阻, 直道圈速收益.
- **耦合效应**: X-mode 不仅减阻, 还影响:
  - 直道极速 (+8-12 km/h)
  - 燃油消耗 (减阻 → 省油 ~2%)
  - 轮胎负载 (低阻力 → 直道磨损略降)
  - 电池部署效率 (低阻 → ERS 收益增 ~5%)

**圈速量化 (EA F1 2026 物理量级)**:
- 单次 X-mode 激活 (800m 直道): ~0.24 s 收益
- 一圈 3 次激活 (Monza): ~0.7 s 总收益
- 湿地禁用 X-mode: 圈速损失 0.5-1.0 s (高下压力需求)

**本模块提供**:
- X-mode 激活对圈速的精确收益 (基于直道几何).
- 主动空动与 PU 部署的协同效应.
- Z-mode/X-mode 切换的过渡圈速影响.

公开 API:
    - :func:`active_aero_lap_gain_s` — 单圈主动空动总收益.
    - :func:`x_mode_straight_gain_s` — 单直道 X-mode 收益.
    - :func:`z_mode_corner_baseline_s` — Z-mode 弯角基准.
    - :func:`aero_pu_synergy_gain_s` — 主动空动 + PU 协同收益.
    - :func:`fuel_save_from_aero_pct` — 减阻省油百分比.
"""

from __future__ import annotations

from f1opt.model.active_aero import (
    _MIN_STRAIGHT_M_FOR_ACTIVATION,
    _X_MODE_GAIN_PER_M_S,
    optimal_plan_for_track,
)

# --------------------------------------------------------------------------- #
# EA F1 2026 主动空动物理常量
# --------------------------------------------------------------------------- #
# X-mode 减阻对极速提升 (km/h per activation, EA F1 2026 量级)
_X_MODE_TOP_SPEED_BOOST_KMH = 10.0

# X-mode 减阻省油 (百分比, EA F1 2026)
_X_MODE_FUEL_SAVE_PCT = 2.0

# X-mode 与 PU 部署协同: 低阻直道上 PU 部署收益增 (系数)
_AERO_PU_SYNERGY_FACTOR = 1.05

# Z-mode 弯角抓地力基准 (相对无主动空动, 正=快)
_Z_MODE_CORNER_GAIN_S = 0.15  # 每圈弯角段基准收益

# X→Z 切换过渡损失 (制动时翼面复位, 单次)
_X_Z_TRANSITION_PENALTY_S = 0.02

# 湿地禁用 X-mode 的圈速损失 (需回到高下压力, 直道慢)
_WET_X_MODE_DISABLED_PENALTY_S = 0.80


# --------------------------------------------------------------------------- #
# 单直道 X-mode 收益
# --------------------------------------------------------------------------- #
def x_mode_straight_gain_s(straight_length_m: float) -> float:
    """单直道 X-mode 激活的圈速收益 (s).

    基于 EA F1 2026 物理模型: 收益 = 直道长度 × 系数.
    短直道 (< 400m) 无收益 (激活无意义).

    Args:
        straight_length_m: 直道长度 (m).

    Returns:
        圈速收益 (s, 正=快). 0.0 表示无收益.
    """
    if straight_length_m < _MIN_STRAIGHT_M_FOR_ACTIVATION:
        return 0.0
    return straight_length_m * _X_MODE_GAIN_PER_M_S


def x_mode_top_speed_boost_kmh() -> float:
    """X-mode 单次激活极速提升 (km/h)."""
    return _X_MODE_TOP_SPEED_BOOST_KMH


def fuel_save_from_aero_pct(n_activations: int) -> float:
    """X-mode 减阻省油百分比.

    每次激活省 ~2%, 一圈最多 3 次 → ~6% (上限).

    Args:
        n_activations: 一圈激活次数.

    Returns:
        省油百分比 (0..6).
    """
    return min(6.0, n_activations * _X_MODE_FUEL_SAVE_PCT)


# --------------------------------------------------------------------------- #
# 单圈主动空动总收益
# --------------------------------------------------------------------------- #
def active_aero_lap_gain_s(
    track_id: str,
    gap_to_ahead_s: float = 0.8,
    wet_conditions: bool = False,
) -> float:
    """单圈主动空动总收益 (s, 正=快).

    综合 X-mode 直道收益 + Z-mode 弯角基准 - 切换过渡损失.

    Args:
        track_id: 赛道 ID.
        gap_to_ahead_s: 与前车差距 (s), > 1.0 无法激活 X-mode.
        wet_conditions: 湿地 (禁用 X-mode).

    Returns:
        圈速收益 (s, 正=快). 负值表示损失 (湿地).
    """
    if wet_conditions:
        # 湿地禁用 X-mode, 圈速损失 (需高下压力, 直道慢)
        return -_WET_X_MODE_DISABLED_PENALTY_S

    plan = optimal_plan_for_track(track_id, gap_to_ahead_s, wet_conditions)
    if plan.n_activations == 0:
        # 无激活: 仅 Z-mode 弯角基准
        return _Z_MODE_CORNER_GAIN_S

    # X-mode 直道收益
    x_gain = plan.total_gain_s
    # Z-mode 弯角基准
    z_gain = _Z_MODE_CORNER_GAIN_S
    # 切换过渡损失 (每次激活一次 X→Z 切换)
    transition_loss = plan.n_activations * _X_Z_TRANSITION_PENALTY_S

    return x_gain + z_gain - transition_loss


def z_mode_corner_baseline_s() -> float:
    """Z-mode 弯角基准收益 (s/圈)."""
    return _Z_MODE_CORNER_GAIN_S


# --------------------------------------------------------------------------- #
# 主动空动 + PU 协同
# --------------------------------------------------------------------------- #
def aero_pu_synergy_gain_s(
    track_id: str,
    pu_deploy_mj: float,
    gap_to_ahead_s: float = 0.8,
    wet_conditions: bool = False,
) -> float:
    """主动空动 + PU 部署协同收益 (s).

    EA F1 2026 物理: X-mode 减阻直道上, PU 部署收益增 ~5% (低阻 →
    更高尾速 → 部署能量转化效率更高).

    Args:
        track_id: 赛道 ID.
        pu_deploy_mj: 本圈 PU 部署能量 (MJ).
        gap_to_ahead_s: 与前车差距.
        wet_conditions: 湿地.

    Returns:
        协同额外收益 (s, 正=快). 0.0 表示无协同 (湿地/无激活).
    """
    if wet_conditions:
        return 0.0

    plan = optimal_plan_for_track(track_id, gap_to_ahead_s, wet_conditions)
    if plan.n_activations == 0:
        return 0.0

    # X-mode 直道上的 PU 部署收益增
    # 假设 60% 部署在直道 (X-mode 区), 增益 5%
    from f1opt.model.pu_2026 import DEPLOY_GAIN_S_PER_MJ
    straight_deploy = pu_deploy_mj * 0.6
    synergy = straight_deploy * DEPLOY_GAIN_S_PER_MJ * (_AERO_PU_SYNERGY_FACTOR - 1.0)
    return synergy


# --------------------------------------------------------------------------- #
# 完整主动空动圈速影响
# --------------------------------------------------------------------------- #
def active_aero_total_lap_effect_s(
    track_id: str,
    gap_to_ahead_s: float = 0.8,
    wet_conditions: bool = False,
    pu_deploy_mj: float = 6.0,
) -> dict[str, float]:
    """主动空动对圈速的完整影响 (对标 EA F1 2026).

    Returns:
        包含各分量收益的字典:
        - ``total_gain_s``: 总圈速收益 (正=快).
        - ``x_mode_gain_s``: X-mode 直道收益.
        - ``z_mode_gain_s``: Z-mode 弯角基准.
        - ``transition_penalty_s``: 切换损失.
        - ``synergy_gain_s``: PU 协同收益.
        - ``fuel_save_pct``: 省油百分比.
        - ``top_speed_boost_kmh``: 极速提升.
    """
    if wet_conditions:
        return {
            "total_gain_s": -_WET_X_MODE_DISABLED_PENALTY_S,
            "x_mode_gain_s": 0.0,
            "z_mode_gain_s": 0.0,
            "transition_penalty_s": 0.0,
            "synergy_gain_s": 0.0,
            "fuel_save_pct": 0.0,
            "top_speed_boost_kmh": 0.0,
        }

    plan = optimal_plan_for_track(track_id, gap_to_ahead_s, wet_conditions)
    n_act = plan.n_activations

    x_gain = plan.total_gain_s
    z_gain = _Z_MODE_CORNER_GAIN_S
    transition = n_act * _X_Z_TRANSITION_PENALTY_S
    synergy = aero_pu_synergy_gain_s(track_id, pu_deploy_mj, gap_to_ahead_s, wet_conditions)
    fuel_pct = fuel_save_from_aero_pct(n_act)
    speed_boost = _X_MODE_TOP_SPEED_BOOST_KMH if n_act > 0 else 0.0
    total = x_gain + z_gain - transition + synergy

    return {
        "total_gain_s": total,
        "x_mode_gain_s": x_gain,
        "z_mode_gain_s": z_gain,
        "transition_penalty_s": -transition,
        "synergy_gain_s": synergy,
        "fuel_save_pct": fuel_pct,
        "top_speed_boost_kmh": speed_boost,
    }


# --------------------------------------------------------------------------- #
# 便捷函数
# --------------------------------------------------------------------------- #
def wet_aero_penalty_s() -> float:
    """湿地禁用 X-mode 的圈速损失."""
    return _WET_X_MODE_DISABLED_PENALTY_S


def transition_penalty_per_activation_s() -> float:
    """单次 X→Z 切换损失."""
    return _X_Z_TRANSITION_PENALTY_S


def max_fuel_save_pct() -> float:
    """最大省油百分比 (3 次激活)."""
    return 3 * _X_MODE_FUEL_SAVE_PCT
