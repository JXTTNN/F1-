"""F1 2026 主动空气动力学激活策略 (Iter-50).

F1 2026 引入主动空气动力学 (Active Aero), 替代/增强传统 DRS:

- **Z-mode (弯角模式)**: 高下压力, 拆分翼面调整, 增加弯角抓地力.
- **X-mode (直道模式)**: 低阻力, 翼面"张开"减阻, 类似 DRS 但更强.
- **激活限制**: 每圈最多 3 次激活 (X-mode), 需在指定检测点后激活.
- **资格条件**: 与前车距离 ≤ 1 秒 (类似 DRS 检测).
- **取消条件**: 制动时自动取消, 回到 Z-mode.

EA F1 2026 主动空动策略:
- AI 车手智能选择激活点 (长直道最优).
- "Manual override" 模式允许玩家手动激活.
- 激活次数管理: 3 次/圈限制下的最优分配.

策略要素:
- 长直道激活 X-mode 收益最大 (~0.3-0.5s).
- 短直道激活收益小, 节省激活次数.
- 防守时也可用 (前车激活, 后车跟随激活).
- 湿地条件下 X-mode 禁用.

公开 API:
    - :class:`ActiveAeroMode` — 模式枚举.
    - :class:`ActivationOpportunity` — 单个激活机会.
    - :class:`ActiveAeroStrategy` — 激活策略.
    - :func:`optimal_activation_plan` — 最优激活计划.
    - :func:`active_aero_gain_s` — 激活收益估算.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
class ActiveAeroMode(Enum):
    """主动空动模式."""

    Z_MODE = "z_mode"    # 高下压力 (弯角)
    X_MODE = "x_mode"    # 低阻力 (直道, 类 DRS)


# 每圈最大激活次数 (FIA 2026 规则)
_MAX_ACTIVATIONS_PER_LAP = 3

# X-mode 收益系数 (s/m 直道长度)
_X_MODE_GAIN_PER_M_S = 0.0003  # 每 m 直道约 0.0003s 收益

# Iter-188: 空气阻力减少估算 (X-mode 的低阻力特性).
# X-mode 下翼面张开, 阻力系数 Cd 降低约 15%, 等效于 ~25% 下压力降低.
# 在 300 km/h (83.3 m/s) 时, 阻力降低约 60 N, 对应 ~0.15s 增益.
_X_MODE_DRAG_REDUCTION_PCT = 0.15  # X-mode 阻力降低百分比
_X_MODE_REF_SPEED_MS = 83.3  # 参考速度 300 km/h
_X_MODE_REF_DRAG_N = 400.0  # 参考速度下参考阻力 (N)

# 最小直道长度 (m), 短于此激活无意义
_MIN_STRAIGHT_M_FOR_ACTIVATION = 400

# DRS 检测点前车距离阈值 (s)
_DETECTION_GAP_S = 1.0

# 湿地禁用
_WET_TRACK_THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
# 数据类
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ActivationOpportunity:
    """单个 X-mode 激活机会.

    - ``straight_m``: 直道长度 (m).
    - ``sector_idx``: 所在扇区 (1-3).
    - ``gain_s``: 激活收益 (s).
    - ``priority``: 优先级 (1=最高).
    """

    straight_m: float
    sector_idx: int
    gain_s: float
    priority: int = 1


@dataclass
class ActiveAeroStrategy:
    """单圈激活策略."""

    lap: int
    activations: list[ActivationOpportunity] = field(default_factory=list)
    mode: ActiveAeroMode = ActiveAeroMode.Z_MODE
    wet_conditions: bool = False
    gap_to_ahead_s: float = 999.0

    @property
    def n_activations(self) -> int:
        return len(self.activations)

    @property
    def can_activate(self) -> bool:
        """是否还能激活 (未超限 + 非湿地 + 有前车)."""
        if self.wet_conditions:
            return False
        if self.n_activations >= _MAX_ACTIVATIONS_PER_LAP:
            return False
        if self.gap_to_ahead_s > _DETECTION_GAP_S:
            return False
        return True

    @property
    def total_gain_s(self) -> float:
        return sum(a.gain_s for a in self.activations)


# --------------------------------------------------------------------------- #
# 最优激活计划
# --------------------------------------------------------------------------- #
def optimal_activation_plan(
    straight_lengths_m: list[float],
    sector_for_straight: list[int],
    gap_to_ahead_s: float = 0.8,
    wet_conditions: bool = False,
) -> ActiveAeroStrategy:
    """计算一圈的最优 X-mode 激活计划.

    Args:
        straight_lengths_m: 各直道长度 (m).
        sector_for_straight: 各直道所在扇区 (1-3).
        gap_to_ahead_s: 与前车差距 (s), > 1.0 无法激活.
        wet_conditions: 湿地条件 (禁用 X-mode).

    Returns:
        :class:`ActiveAeroStrategy` 包含最多 3 个激活机会.
    """
    strategy = ActiveAeroStrategy(
        lap=0,
        wet_conditions=wet_conditions,
        gap_to_ahead_s=gap_to_ahead_s,
    )

    if wet_conditions or gap_to_ahead_s > _DETECTION_GAP_S:
        return strategy  # 无法激活

    # 评估每个直道的激活收益
    opportunities: list[ActivationOpportunity] = []
    for i, length in enumerate(straight_lengths_m):
        if length < _MIN_STRAIGHT_M_FOR_ACTIVATION:
            continue
        gain = length * _X_MODE_GAIN_PER_M_S
        sector = sector_for_straight[i] if i < len(sector_for_straight) else 1
        opportunities.append(ActivationOpportunity(
            straight_m=length,
            sector_idx=sector,
            gain_s=gain,
        ))

    # 按收益降序排序, 选 top 3
    opportunities.sort(key=lambda x: x.gain_s, reverse=True)
    selected = opportunities[:_MAX_ACTIVATIONS_PER_LAP]

    # 按扇区顺序排序 (激活顺序)
    selected.sort(key=lambda x: x.sector_idx)

    # 设置优先级
    for i, opp in enumerate(selected):
        opp_with_priority = ActivationOpportunity(
            straight_m=opp.straight_m,
            sector_idx=opp.sector_idx,
            gain_s=opp.gain_s,
            priority=i + 1,
        )
        strategy.activations.append(opp_with_priority)

    return strategy


# --------------------------------------------------------------------------- #
# 便捷函数
# --------------------------------------------------------------------------- #
def active_aero_gain_s(straight_length_m: float, speed_ms: float = 83.3) -> float:
    """便捷: 估算直道激活收益 (s), 含速度依赖的阻力减少.

    Iter-188: 加入速度依赖的阻力减少估算. 低速时阻力减少更小.
    """
    if straight_length_m < _MIN_STRAIGHT_M_FOR_ACTIVATION:
        return 0.0
    base_gain = straight_length_m * _X_MODE_GAIN_PER_M_S
    # 速度依赖的阻力减少: 阻力 ∝ v², 低速时阻力减少更小
    speed_factor = min(1.0, (speed_ms / _X_MODE_REF_SPEED_MS) ** 2)
    drag_reduction = _X_MODE_DRAG_REDUCTION_PCT * _X_MODE_REF_DRAG_N * speed_factor
    # 阻力减少 → 驱动力更多用于加速, 近似线性增益
    drag_benefit_s = drag_reduction / 4000.0 * straight_length_m / 1000.0  # ~0.02s per 100m
    return base_gain + drag_benefit_s


def can_activate_x_mode(
    gap_to_ahead_s: float,
    wet_conditions: bool = False,
    activations_this_lap: int = 0,
) -> bool:
    """便捷: 判断是否可激活 X-mode."""
    if wet_conditions:
        return False
    if gap_to_ahead_s > _DETECTION_GAP_S:
        return False
    if activations_this_lap >= _MAX_ACTIVATIONS_PER_LAP:
        return False
    return True


def max_activations_per_lap() -> int:
    """便捷: 每圈最大激活次数."""
    return _MAX_ACTIVATIONS_PER_LAP


# --------------------------------------------------------------------------- #
# 赛道特定激活机会 (基于 sector_times 数据)
# --------------------------------------------------------------------------- #
# 每条赛道的直道长度 (m), 按 DRS 区排序
_TRACK_STRAIGHTS_M: dict[str, list[float]] = {
    "monza": [1200, 800, 600],          # 主直道 + 回弯直道
    "spa": [1100, 900],                  # Kemmel + Stavelot
    "baku": [2200, 600],                 # 2.2km 超长直道
    "silverstone": [800, 700],           # Hangar + Wellington
    "monaco": [250],                     # 仅起点
    "suzuka": [900],                     # 主直道
    "bahrain": [900, 700, 600],
    "jeddah": [800, 700, 600],
    "miami": [700, 600, 500],
    "montreal": [800, 600],
    "barcelona": [900, 700],
    "spielberg": [700, 600, 500],
    "hungaroring": [600],
    "zandvoort": [500, 400],
    "shanghai": [1000, 800, 600],
    "melbourne": [800, 700],
    "madrid": [800, 700, 600],
    "singapore": [700, 600, 500],
    "austin": [800, 700],
    "mexico_city": [800, 700, 600],
    "interlagos": [700, 600],
    "las_vegas": [1500, 800, 700],
    "losail": [900, 800],
    "yas_marina": [900, 800, 700],
}


def straights_for_track(track_id: str) -> list[float]:
    """查询赛道直道长度列表 (m)."""
    return _TRACK_STRAIGHTS_M.get(track_id, [600.0])


def optimal_plan_for_track(
    track_id: str,
    gap_to_ahead_s: float = 0.8,
    wet_conditions: bool = False,
) -> ActiveAeroStrategy:
    """便捷: 计算赛道最优激活计划."""
    straights = straights_for_track(track_id)
    sectors = [(i % 3) + 1 for i in range(len(straights))]
    return optimal_activation_plan(straights, sectors, gap_to_ahead_s, wet_conditions)
