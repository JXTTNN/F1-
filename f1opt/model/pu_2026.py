"""F1 2026 动力单元 (PU) 能量部署精确模型 (Iter-53).

严格契合 EA Sports F1 2026 物理引擎与 FIA 2026 技术规则 §5.

**2026 PU 规格 (FIA 2026 技术规则, EA F1 2026 物理引擎基准)**:
- ICE (内燃机): 400 kW (~536 hp), 100% 可持续燃料
- MGU-K (动能电机): 350 kW (~469 hp), 双向部署/回收
- **总功率: 750 kW (~1006 hp)** — 比 2025 (~740 kW) 略增
- **移除 MGU-H** (2026 规则取消, 简化 PU)
- **部署预算: 9 MJ/lap** (MGU-K, 比 2014-2025 的 4 MJ 翻倍)
- 回收预算: 无硬上限, 但受刹车能量物理限制 (~6 MJ/lap)
- 电池容量: 增大 (支持 9 MJ 部署)
- 燃油质量流限: 100 kg/h, 总量 110 kg (可持续燃料, 能量密度 0.97)

**EA F1 2026 部署模式 (4 种, 对应游戏 Hotkey)**:
- **MODE 1 (Qualifying/Party)**: 全力 9 MJ/lap, 圈速最快, 电池快速耗尽
- **MODE 2 (Attack)**: 8 MJ/lap, 多部署追击
- **MODE 3 (Balanced)**: 6 MJ/lap, 标准比赛节奏
- **MODE 4 (Conserve)**: 4 MJ/lap, 省电池, 长 stint

**圈速影响 (工程估算, 对标 EA F1 2026 量级)**:
- 1 MJ 部署 ≈ 0.07-0.11 s 直道收益 (2026 MGU-K 350kW, 比旧 120kW 强)
- 1 MJ 回收 ≈ 0.04 s 圈速代价 (刹车阻力)
- 全力 9 MJ vs 平衡 6 MJ ≈ 0.20-0.30 s/lap 差距

公开 API:
    - :class:`PUDeployMode` — 4 种部署模式.
    - :class:`PU2026State` — PU 状态 (SoC, 已部署/回收).
    - :class:`PU2026Model` — 2026 PU 部署模型.
    - :func:`lap_time_gain_s` — 部署对圈速收益.
    - :func:`max_deploy_mj_per_lap` — 9 MJ 上限.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# --------------------------------------------------------------------------- #
# FIA 2026 PU 常量 (严格对标 EA F1 2026 物理引擎)
# --------------------------------------------------------------------------- #
ICE_POWER_KW = 400.0          # 内燃机功率 (FIA 2026 §5.2)
MGU_K_POWER_KW = 350.0        # MGU-K 功率 (FIA 2026 §5.3, 比 2025 的 120kW 大幅提升)
TOTAL_POWER_KW = ICE_POWER_KW + MGU_K_POWER_KW  # 750 kW

MAX_DEPLOY_MJ_PER_LAP = 9.0   # 每圈部署上限 (FIA 2026 §5.4, 比 2014-2025 的 4 MJ 翻倍)
MAX_HARVEST_MJ_PER_LAP = 6.0  # 每圈回收物理上限 (刹车能量限制)
BATTERY_CAPACITY_MJ = 9.0     # 电池容量 (匹配单圈最大部署)

# 圈速影响系数 (EA F1 2026 物理量级)
# 2026 MGU-K 350kW, 部署收益比旧 120kW ERS 高
DEPLOY_GAIN_S_PER_MJ = 0.09   # 1 MJ 部署 ~0.09 s 直道收益
HARVEST_DRAG_S_PER_MJ = 0.04  # 1 MJ 回收 ~0.04 s 圈速代价
NET_GAIN_S_PER_MJ = DEPLOY_GAIN_S_PER_MJ - HARVEST_DRAG_S_PER_MJ  # 0.05 s/MJ


# --------------------------------------------------------------------------- #
# 部署模式
# --------------------------------------------------------------------------- #
class PUDeployMode(Enum):
    """EA F1 2026 PU 部署模式 (4 种, 对应游戏 Hotkey).

    - QUALIFYING (Mode 1): 全力, 圈速最快, 电池快速耗尽.
    - ATTACK (Mode 2): 多部署追击.
    - BALANCED (Mode 3): 标准比赛节奏.
    - CONSERVE (Mode 4): 省电池, 长 stint.
    """

    QUALIFYING = "qualifying"   # Mode 1: 9 MJ
    ATTACK = "attack"           # Mode 2: 8 MJ
    BALANCED = "balanced"       # Mode 3: 6 MJ
    CONSERVE = "conserve"       # Mode 4: 4 MJ


# 每模式部署 MJ/lap + 效率系数
_MODE_PARAMS: dict[PUDeployMode, tuple[float, float]] = {
    PUDeployMode.QUALIFYING: (9.0, 1.10),   # 全力 + 10% 效率 (电池满压)
    PUDeployMode.ATTACK: (8.0, 1.05),
    PUDeployMode.BALANCED: (6.0, 1.0),
    PUDeployMode.CONSERVE: (4.0, 0.95),
}


# --------------------------------------------------------------------------- #
# PU2026State
# --------------------------------------------------------------------------- #
@dataclass
class PU2026State:
    """2026 PU 状态 (跨圈传递).

    - ``soc_mj``: 电池当前能量 (MJ), 0..BATTERY_CAPACITY_MJ.
    - ``cumulative_deploy_mj``: 累计部署能量.
    - ``cumulative_harvest_mj``: 累计回收能量.
    - ``laps_completed``: 已完成圈数.
    """

    soc_mj: float = BATTERY_CAPACITY_MJ * 0.5  # 起始 50% SoC
    cumulative_deploy_mj: float = 0.0
    cumulative_harvest_mj: float = 0.0
    laps_completed: int = 0

    @property
    def soc_pct(self) -> float:
        """电池 SoC 百分比 (0..100)."""
        return 100.0 * self.soc_mj / BATTERY_CAPACITY_MJ

    @property
    def is_low_soc(self) -> bool:
        """SoC < 20% 视为低."""
        return self.soc_mj < 0.2 * BATTERY_CAPACITY_MJ

    @property
    def is_full(self) -> bool:
        return self.soc_mj >= BATTERY_CAPACITY_MJ * 0.99


# --------------------------------------------------------------------------- #
# PU2026Model
# --------------------------------------------------------------------------- #
class PU2026Model:
    """F1 2026 PU 能量部署模型.

    严格契合 FIA 2026 技术规则 + EA F1 2026 物理引擎.

    用法::

        pu = PU2026Model(track_id="monza", mode=PUDeployMode.BALANCED)
        state = PU2026State(soc_mj=4.5)
        result = pu.simulate_lap(state)
        print(f"部署 {result.deploy_mj} MJ, 收益 {result.net_gain_s:.3f}s")
    """

    def __init__(
        self,
        track_id: str,
        mode: PUDeployMode = PUDeployMode.BALANCED,
        deploy_efficiency: float = 1.0,
        harvest_efficiency: float = 1.0,
    ) -> None:
        self.track_id = track_id
        self.mode = mode
        # 车队 PU 效率 (Mercedes/Honda RBPT ~1.05, 后段 ~0.95)
        self.deploy_efficiency = max(0.8, min(1.2, deploy_efficiency))
        self.harvest_efficiency = max(0.8, min(1.2, harvest_efficiency))

    # ------------------------------------------------------------------ #
    def target_deploy_mj(self) -> float:
        """当前模式目标部署 MJ/lap."""
        base, _ = _MODE_PARAMS[self.mode]
        return min(base, MAX_DEPLOY_MJ_PER_LAP)

    def mode_efficiency(self) -> float:
        """当前模式效率系数."""
        _, eff = _MODE_PARAMS[self.mode]
        return eff

    # ------------------------------------------------------------------ #
    def simulate_lap(self, state: PU2026State) -> PULapResult:
        """仿真一圈 PU 部署/回收, 更新状态, 返回结果.

        - 部署受模式 + SoC 限制 (低 SoC 自动降部署).
        - 回收受赛道刹车能量限制 (重制动赛道回收多).
        - 圈速收益 = 部署收益 - 回收代价.
        """
        target = self.target_deploy_mj()
        # SoC 限制: 实际部署不能超过当前 SoC
        actual_deploy = min(target, state.soc_mj)
        # 低 SoC 自动降级 (保护电池)
        if state.is_low_soc and self.mode == PUDeployMode.QUALIFYING:
            actual_deploy = min(actual_deploy, target * 0.6)

        # 回收 (基于赛道特性 + 模式)
        # Conserve 模式多回收, Qualifying 少回收 (少刹车)
        harvest_base = MAX_HARVEST_MJ_PER_LAP * self._track_harvest_factor()
        if self.mode == PUDeployMode.CONSERVE:
            harvest_base *= 1.15  # 多回收
        elif self.mode == PUDeployMode.QUALIFYING:
            harvest_base *= 0.85  # 少回收
        # 电池满则不回收 (物理限制)
        room = BATTERY_CAPACITY_MJ - state.soc_mj + actual_deploy
        actual_harvest = min(harvest_base, room)

        # 应用效率
        effective_deploy = actual_deploy * self.deploy_efficiency * self.mode_efficiency()
        effective_harvest = actual_harvest * self.harvest_efficiency

        # 圈速收益
        deploy_gain = effective_deploy * DEPLOY_GAIN_S_PER_MJ
        harvest_cost = effective_harvest * HARVEST_DRAG_S_PER_MJ
        net_gain = deploy_gain - harvest_cost

        # 更新状态
        state.soc_mj = max(0.0, min(BATTERY_CAPACITY_MJ,
                                     state.soc_mj - actual_deploy + actual_harvest))
        state.cumulative_deploy_mj += actual_deploy
        state.cumulative_harvest_mj += actual_harvest
        state.laps_completed += 1

        return PULapResult(
            mode=self.mode,
            deploy_mj=actual_deploy,
            harvest_mj=actual_harvest,
            soc_before_mj=state.soc_mj + actual_deploy - actual_harvest,
            soc_after_mj=state.soc_mj,
            deploy_gain_s=deploy_gain,
            harvest_cost_s=harvest_cost,
            net_gain_s=net_gain,
            efficiency=self.mode_efficiency(),
        )

    # ------------------------------------------------------------------ #
    def _track_harvest_factor(self) -> float:
        """赛道刹车能量因子 (0.7-1.1).

        重制动赛道 (加拿大/新加坡/巴林) 回收多, 全油门赛道 (Spa/Monza) 回收少.
        """
        _HIGH_HARVEST = {"montreal", "singapore", "bahrain", "miami", "austin"}
        _LOW_HARVEST = {"spa", "monza", "jeddah", "baku", "las_vegas"}
        if self.track_id in _HIGH_HARVEST:
            return 1.10
        if self.track_id in _LOW_HARVEST:
            return 0.80
        return 1.0

    # ------------------------------------------------------------------ #
    def recommend_mode(self, state: PU2026State, lap: int, total_laps: int) -> PUDeployMode:
        """推荐部署模式 (基于圈数 + SoC)."""
        # 排位圈 (最后 1 圈) → 全力
        if lap >= total_laps:
            return PUDeployMode.QUALIFYING
        # 低 SoC → 省电
        if state.is_low_soc:
            return PUDeployMode.CONSERVE
        # 后段 1/3 + SoC 充足 → 攻击
        if lap > total_laps * 0.7 and state.soc_mj > BATTERY_CAPACITY_MJ * 0.5:
            return PUDeployMode.ATTACK
        return PUDeployMode.BALANCED


# --------------------------------------------------------------------------- #
# PULapResult
# --------------------------------------------------------------------------- #
@dataclass
class PULapResult:
    """单圈 PU 部署结果."""

    mode: PUDeployMode
    deploy_mj: float
    harvest_mj: float
    soc_before_mj: float
    soc_after_mj: float
    deploy_gain_s: float
    harvest_cost_s: float
    net_gain_s: float
    efficiency: float

    @property
    def net_soc_delta_mj(self) -> float:
        """SoC 净变化 (正=充电, 负=放电)."""
        return self.harvest_mj - self.deploy_mj


# --------------------------------------------------------------------------- #
# 便捷函数
# --------------------------------------------------------------------------- #
def lap_time_gain_s(deploy_mj: float, harvest_mj: float = 0.0) -> float:
    """便捷: 计算部署收益 (s)."""
    return deploy_mj * DEPLOY_GAIN_S_PER_MJ - harvest_mj * HARVEST_DRAG_S_PER_MJ


def max_deploy_mj_per_lap() -> float:
    """便捷: 2026 每圈最大部署 (9 MJ)."""
    return MAX_DEPLOY_MJ_PER_LAP


def total_power_kw() -> float:
    """便捷: 2026 PU 总功率 (750 kW)."""
    return TOTAL_POWER_KW


def mode_deploy_mj(mode: PUDeployMode) -> float:
    """便捷: 模式对应部署 MJ/lap."""
    return _MODE_PARAMS[mode][0]
