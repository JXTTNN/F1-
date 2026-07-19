"""F1 2026 — 可持续燃料 E10+ 模型 (Iter-27).

FIA 2026 强制使用 **100% 可持续燃料** (advanced sustainable fuel).
F1 2026 (EA Sports 官方) 还原此规则:

1. **燃料组成**: 100% 可持续来源 (生物乙醇 + e-fuel 合成).
2. **能量密度**: 比 2025 E10 (10% 乙醇) 略低 (~3% lower energy density).
3. **冷却需求**: 可持续燃料燃烧温度高, 散热器需更大.
4. **每圈消耗**: 30 kg/h 上限 (vs 2025 100 kg/h) — F1 2026 强制电气化.
5. **赛道温度耦合**: 高温环境冷却不足时, 引擎降功率.
6. **海拔耦合**: 高海拔 (Mexico City) 空气稀薄 → 燃烧效率下降.

公开 API:
    - :class:`SustainableFuelModel` — 单圈燃料消耗仿真.
    - :func:`fuel_consumption_lap` — 便捷函数.
"""

from __future__ import annotations

from dataclasses import dataclass

# FIA 2026 燃料规格
_FUEL_FLOW_KG_PER_H = 30.0  # max flow rate
_LAP_DURATION_S = 90.0
_FUEL_PER_LAP_KG_BASE = _FUEL_FLOW_KG_PER_H / 3600.0 * _LAP_DURATION_S  # ~0.75

# 可持续燃料能量密度 (vs 2025 E10 = 1.0)
_ENERGY_DENSITY_FACTOR = 0.97  # 3% lower
# 引擎热效率 (F1 2026 PU ~52%)
_ENGINE_THERMAL_EFFICIENCY = 0.52
# 高温降功率阈值
_TEMP_DERATING_THRESHOLD_C = 50.0
_TEMP_DERATING_PER_DEG_C = 0.015  # 每度降 1.5%

# 海拔空气密度对燃烧影响
def _air_density_factor(altitude_m: float) -> float:
    return (1.0 - 2.25577e-5 * altitude_m) ** 4.2559


# 赛道默认海拔
_DEFAULT_ALTITUDE_M = 100.0


@dataclass
class FuelLapResult:
    """单圈燃料消耗仿真结果."""

    fuel_used_kg: float
    fuel_flow_kg_per_h: float
    energy_density_factor: float
    effective_power_factor: float
    """有效功率因子 (1.0 = 满功率, <1 = 降功率)."""
    derating_reason: str
    """降功率原因 ("none" / "high_temp" / "altitude" / "both")."""


@dataclass
class SustainableFuelModel:
    """F1 2026 可持续燃料模型 (Iter-27).

    用法::

        fuel = SustainableFuelModel(
            track_id="mexico_city", altitude_m=2286,
            track_temp_c=35.0,
        )
        r = fuel.simulate_lap(deploy_mode="race")
        print(r.fuel_used_kg, r.effective_power_factor)
    """

    track_id: str
    altitude_m: float = _DEFAULT_ALTITUDE_M
    track_temp_c: float = 35.0
    cooling_duct_size: float = 1.0
    """散热器尺寸 0.7-1.3 (1.0 = 标准)."""

    # ------------------------------------------------------------------ #
    def simulate_lap(self, deploy_mode: str = "race") -> FuelLapResult:
        """仿真单圈燃料消耗."""
        # 基础流量
        base_flow = _FUEL_FLOW_KG_PER_H
        # 实际流量受模式影响 (quali 全油门, save 节流)
        mode_factor = {
            "quali": 1.0, "race": 0.85, "save": 0.65, "attack": 0.95,
        }.get(deploy_mode, 0.85)
        fuel_flow = base_flow * mode_factor
        fuel_used = fuel_flow / 3600.0 * _LAP_DURATION_S

        # 高温降功率
        temp_factor = 1.0
        derating_reason = "none"
        if self.track_temp_c > _TEMP_DERATING_THRESHOLD_C:
            excess = self.track_temp_c - _TEMP_DERATING_THRESHOLD_C
            temp_factor = max(0.85, 1.0 - excess * _TEMP_DERATING_PER_DEG_C)
            # 散热器大可部分缓解
            temp_factor = min(1.0, temp_factor + (self.cooling_duct_size - 1.0) * 0.05)
            derating_reason = "high_temp"

        # 海拔降功率 (空气稀薄, 燃烧不充分)
        alt_factor = _air_density_factor(self.altitude_m)
        if alt_factor < 0.95 and derating_reason == "high_temp":
            derating_reason = "both"
        elif alt_factor < 0.95:
            derating_reason = "altitude"

        effective_power = _ENERGY_DENSITY_FACTOR * temp_factor * alt_factor

        return FuelLapResult(
            fuel_used_kg=fuel_used,
            fuel_flow_kg_per_h=fuel_flow,
            energy_density_factor=_ENERGY_DENSITY_FACTOR,
            effective_power_factor=effective_power,
            derating_reason=derating_reason,
        )


def fuel_consumption_lap(
    track_id: str,
    deploy_mode: str = "race",
    altitude_m: float = _DEFAULT_ALTITUDE_M,
    track_temp_c: float = 35.0,
    **kwargs,
) -> FuelLapResult:
    """便捷函数."""
    model = SustainableFuelModel(
        track_id=track_id, altitude_m=altitude_m,
        track_temp_c=track_temp_c, **kwargs,
    )
    return model.simulate_lap(deploy_mode=deploy_mode)
