"""F1 2026 — 50% 电动力分配规则 (Iter-26).

FIA 2026 技术规则核心创新: **Power Unit** 完全重新设计, 电气化大幅提升:

1. **MGU-K 升级**: 最大功率 350 kW (vs 2025 120 kW), 部署上限大增.
2. **MGU-H 取消**: 2026 起 MGU-H 被禁, 所有电能量来自 MGU-K 制动回收
   + 电池储能.
3. **50/50 电液力分配**: 直道上车手可控制电动力 vs 内燃机动力的比例:
   - 内燃机 (ICE): ~400 kW (540 hp)
   - 电动力 (MGU-K): ~350 kW (470 hp)
   - 总功率 ~750 kW (1000+ hp)
4. **部署模式 (PUI - Power Unit Interface)**:
   - **Quali Mode**: 100% 电力全功率, 短期, 高电池消耗
   - **Race Mode**: 60% 电力, 长期平衡
   - **Save Mode**: 30% 电力, 节能, 高回收
   - **Attack Mode**: 90% 电力, 4s burst (类似 Formula E 攻击模式)
5. **电池能量上限**: 每圈 9 MJ 回收上限, 9 MJ 部署上限.
6. **燃油流量上限**: 2026 起 30 kg/h (vs 2025 100 kg/h), 强制电气化.

公开 API:
    - :class:`PowerUnit2026` — 单圈 PU 状态仿真.
    - :func:`simulate_pu_lap` — 便捷函数.
"""

from __future__ import annotations

from dataclasses import dataclass

# FIA 2026 PU 规格
_ICE_POWER_KW = 400.0
_MGUK_POWER_KW = 350.0
_TOTAL_POWER_KW = _ICE_POWER_KW + _MGUK_POWER_KW  # ~750 kW

# 部署模式系数 (相对最大电功率)
_DEPLOY_MODE_FACTOR: dict[str, float] = {
    "quali": 1.00,  # 100% MGU-K
    "race": 0.60,
    "save": 0.30,
    "attack": 0.90,  # burst 模式
}
_ATTACK_MODE_DURATION_S = 4.0  # 4s burst
_ATTACK_MODE_COOLDOWN_S = 30.0  # 30s 冷却

# 电池容量与能量流
_BATTERY_CAPACITY_MJ = 9.0  # 每圈上限
_RECOVERY_EFFICIENCY = 0.85  # MGU-K 回收效率
# 每圈可用部署能量 = min(电池容量, recovery + 电池存量)
# 简化: 每圈 9 MJ 回收 + 9 MJ 部署, 取决于模式

# 燃油流量上限 (2026: 30 kg/h, 一圈 ~90s = 0.75 kg)
_FUEL_FLOW_KG_PER_H = 30.0
_FUEL_PER_LAP_KG = _FUEL_FLOW_KG_PER_H / 3600.0 * 90.0  # ~0.75 kg

# 电液分配 (50/50 默认)
_DEFAULT_ELECTRIC_FRACTION = 0.50


# --------------------------------------------------------------------------- #
# PULapResult
# --------------------------------------------------------------------------- #
@dataclass
class PULapResult:
    """单圈 PU 仿真结果."""

    lap_idx: int
    deploy_mode: str
    """使用模式: quali/race/save/attack."""
    ice_power_kw: float
    """内燃机功率 kW."""
    mguk_power_kw: float
    """MGU-K 电功率 kW."""
    total_power_kw: float
    """总功率 kW."""
    electric_fraction: float
    """电动力占比 0..1."""
    energy_deployed_mj: float
    """本圈部署能量 MJ."""
    energy_recovered_mj: float
    """本圈回收能量 MJ."""
    fuel_used_kg: float
    """本圈燃油消耗 kg."""
    battery_soc: float
    """圈末电池 SoC 0..1."""
    attack_mode_activated: bool
    """本圈是否激活 Attack Mode."""


# --------------------------------------------------------------------------- #
# PowerUnit2026
# --------------------------------------------------------------------------- #
@dataclass
class PowerUnit2026:
    """F1 2026 Power Unit 仿真 (Iter-26).

    用法::

        pu = PowerUnit2026(track_id="monza", initial_soc=0.6)
        r = pu.simulate_lap(lap_idx=0, deploy_mode="race")
        print(r.total_power_kw, r.electric_fraction)
    """

    track_id: str
    initial_soc: float = 0.6
    """初始电池 SoC 0..1."""

    # ------------------------------------------------------------------ #
    def simulate_lap(
        self,
        lap_idx: int,
        deploy_mode: str = "race",
        attack_mode: bool = False,
        recovery_intensity: float = 0.7,
        track_wetness: float = 0.0,
    ) -> PULapResult:
        """仿真单圈 PU 状态.

        Args:
            deploy_mode: quali/race/save/attack.
            attack_mode: 本圈是否激活 Attack Mode (4s burst).
            recovery_intensity: 制动回收强度 0..1 (赛道制动能量).
            track_wetness: 湿润度 0..1 (湿地下部署降级).
        """
        # 湿地部署降级
        wet_factor = 1.0 - track_wetness * 0.3

        # 模式系数
        mode_factor = _DEPLOY_MODE_FACTOR.get(deploy_mode, 0.60)
        # Attack Mode: 4s 100% 部署 + 其余时间 race 模式
        attack_active = attack_mode and deploy_mode != "quali"
        if attack_active:
            # 平均: 4s/90s × 100% + 86s/90s × mode_factor
            attack_fraction = _ATTACK_MODE_DURATION_S / 90.0
            effective_factor = (attack_fraction * 1.0
                                + (1 - attack_fraction) * mode_factor)
        else:
            effective_factor = mode_factor * wet_factor

        mguk_power = _MGUK_POWER_KW * effective_factor
        ice_power = _ICE_POWER_KW
        total_power = ice_power + mguk_power
        electric_fraction = mguk_power / total_power if total_power > 0 else 0.0

        # 能量流 (简化: 总功率 × 90s 圈时间, 但需 MJ)
        lap_duration_s = 90.0
        # 部署能量 (MGU-K 输出, MJ)
        # 模式决定实际部署能量 (quali 用满 9 MJ, save 仅用 30%)
        max_deployable_mj = _BATTERY_CAPACITY_MJ * effective_factor
        energy_deployed = min(
            mguk_power * lap_duration_s / 1000.0,
            max_deployable_mj,
        )
        # 回收能量 (MGU-K 制动回收, 简化: 制动强度 × 上限)
        # 制动能量粗估: 总功率 × 制动时间占比 × 效率
        braking_fraction = recovery_intensity * 0.20  # 20% 制动时间
        energy_recovered = (
            _MGUK_POWER_KW * braking_fraction * lap_duration_s / 1000.0
            * _RECOVERY_EFFICIENCY
        )
        # 限制回收 ≤ 部署上限 9 MJ
        energy_recovered = min(energy_recovered, _BATTERY_CAPACITY_MJ)

        # 燃油消耗 (ICE 工作, 简化 0.75 kg/lap × ICE 满负荷)
        fuel_used = _FUEL_PER_LAP_KG

        # 电池 SoC 更新
        # net_energy = recovered - deployed (MJ)
        net_energy = energy_recovered - energy_deployed
        # 容量 9 MJ, SoC 变化 = net_energy / capacity
        soc_change = net_energy / _BATTERY_CAPACITY_MJ
        new_soc = max(0.0, min(1.0, self.initial_soc + soc_change))

        return PULapResult(
            lap_idx=lap_idx,
            deploy_mode=deploy_mode,
            ice_power_kw=ice_power,
            mguk_power_kw=mguk_power,
            total_power_kw=total_power,
            electric_fraction=electric_fraction,
            energy_deployed_mj=energy_deployed,
            energy_recovered_mj=energy_recovered,
            fuel_used_kg=fuel_used,
            battery_soc=new_soc,
            attack_mode_activated=attack_active,
        )


# --------------------------------------------------------------------------- #
# 便捷函数
# --------------------------------------------------------------------------- #
def simulate_pu_lap(
    track_id: str,
    lap_idx: int,
    deploy_mode: str = "race",
    initial_soc: float = 0.6,
    **kwargs,
) -> PULapResult:
    """便捷函数."""
    pu = PowerUnit2026(track_id=track_id, initial_soc=initial_soc)
    return pu.simulate_lap(lap_idx=lap_idx, deploy_mode=deploy_mode, **kwargs)


def total_power_for_mode(deploy_mode: str, attack: bool = False) -> float:
    """返回模式的总功率 kW."""
    mode_factor = _DEPLOY_MODE_FACTOR.get(deploy_mode, 0.60)
    if attack and deploy_mode != "quali":
        attack_fraction = _ATTACK_MODE_DURATION_S / 90.0
        effective = attack_fraction * 1.0 + (1 - attack_fraction) * mode_factor
    else:
        effective = mode_factor
    return _ICE_POWER_KW + _MGUK_POWER_KW * effective
