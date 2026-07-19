"""F1 2026 燃油消耗模型 (Iter-49).

F1 2026 燃油规则:
- **质量流限制**: 100 kg/h 最大 (FIA 规则).
- **总燃油限制**: 110 kg 正赛上限 (无额外消耗).
- **可持续燃料**: 100% 可持续燃料 (Iter-27), 能量密度 0.97.
- **节油模式**: lean / normal / rich, 影响圈速与消耗.

燃油物理:
- 每圈消耗取决于赛道几何 (长直道多 + 重制动 = 高消耗).
- 燃油质量影响圈速: 每 10 kg 燃油约 0.3-0.4 s/lap.
- 节油模式降低消耗但牺牲圈速 (~0.3-0.5 s/lap).
- 富油模式提升圈速 (~0.2 s/lap) 但消耗高.

EA F1 2026 燃油管理特性:
- 实时燃油余量显示.
- "Fuel save" 按钮激活节油模式.
- 燃油预算规划: 起步油量 vs 进站策略.
- 混合模式: 燃油 + ERS 协同管理.

数据来源: FIA 2026 规则 + 车队 simulator 量级估计.

公开 API:
    - :class:`FuelConsumptionModel` — 燃油消耗模型.
    - :func:`fuel_per_lap` — 每圈燃油消耗.
    - :func:`fuel_effect_on_lap_time` — 燃油对圈速影响.
    - :func:`recommended_start_fuel_kg` — 推荐起步油量.
    - :class:`FuelMode` — 燃油模式枚举.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
class FuelMode(Enum):
    """燃油模式."""

    LEAN = "lean"        # 节油: 慢 0.4s, 省 12%
    NORMAL = "normal"    # 标准
    RICH = "rich"        # 富油: 快 0.2s, 多耗 15%
    PARTY = "party"      # 极致: 快 0.4s, 多耗 25% (排位)


# 燃油模式系数: (圈速 delta s, 消耗系数)
_FUEL_MODE_PARAMS: dict[FuelMode, tuple[float, float]] = {
    FuelMode.LEAN: (0.4, 0.88),     # 慢 0.4s, 省 12%
    FuelMode.NORMAL: (0.0, 1.0),
    FuelMode.RICH: (-0.2, 1.15),    # 快 0.2s, 多 15%
    FuelMode.PARTY: (-0.4, 1.25),   # 快 0.4s, 多 25%
}

# 燃油对圈速影响: 每 10 kg 燃油慢多少秒 (F1 2026 量级)
_FUEL_PENALTY_PER_10KG_S = 0.35

# 最大燃油上限 (FIA 2026)
_MAX_FUEL_KG = 110.0

# 赛道燃油消耗基准 (kg/lap, 来自 track_engineering 数据)
_TRACK_FUEL_CONSUMPTION: dict[str, float] = {
    "melbourne": 1.55, "shanghai": 1.65, "suzuka": 1.75, "bahrain": 1.65,
    "jeddah": 1.85, "miami": 1.60, "montreal": 1.55, "monaco": 1.20,
    "barcelona": 1.60, "spielberg": 1.50, "silverstone": 1.85, "spa": 2.10,
    "hungaroring": 1.50, "zandvoort": 1.55, "monza": 1.90, "madrid": 1.65,
    "baku": 1.85, "singapore": 1.55, "austin": 1.75, "mexico_city": 1.65,
    "interlagos": 1.60, "las_vegas": 1.95, "losail": 1.80, "yas_marina": 1.70,
}

_DEFAULT_FUEL_KG_PER_LAP = 1.65


# --------------------------------------------------------------------------- #
# 数据类
# --------------------------------------------------------------------------- #
@dataclass
class FuelState:
    """车手燃油状态."""

    current_fuel_kg: float
    total_laps: int
    current_lap: int = 0
    mode: FuelMode = FuelMode.NORMAL


@dataclass
class FuelConsumptionResult:
    """燃油消耗计算结果."""

    fuel_used_kg: float
    fuel_remaining_kg: float
    laps_remaining: int
    fuel_per_lap_avg: float
    fuel_deficit_kg: float        # 负 = 富余, 正 = 不足
    needs_fuel_save: bool         # 是否需要节油
    projected_finish_fuel_kg: float  # 预测完赛时余油
    lap_time_delta_s: float       # 燃油模式对圈速影响


# --------------------------------------------------------------------------- #
# FuelConsumptionModel
# --------------------------------------------------------------------------- #
class FuelConsumptionModel:
    """燃油消耗模型.

    用法::

        model = FuelConsumptionModel(track_id="monza")
        state = FuelState(current_fuel_kg=110.0, total_laps=53, current_lap=20)
        result = model.calculate(state)
        if result.needs_fuel_save:
            print(f"需节油! 不足 {result.fuel_deficit_kg:.1f}kg")
    """

    def __init__(self, track_id: str) -> None:
        if track_id not in _TRACK_FUEL_CONSUMPTION:
            # 未知赛道用默认
            self.track_id = track_id
            self.base_consumption_kg_per_lap = _DEFAULT_FUEL_KG_PER_LAP
        else:
            self.track_id = track_id
            self.base_consumption_kg_per_lap = _TRACK_FUEL_CONSUMPTION[track_id]

    # ------------------------------------------------------------------ #
    def consumption_for_mode(self, mode: FuelMode) -> float:
        """给定模式下的每圈消耗 (kg)."""
        _, factor = _FUEL_MODE_PARAMS[mode]
        return self.base_consumption_kg_per_lap * factor

    def lap_time_delta_for_mode(self, mode: FuelMode) -> float:
        """给定模式下的圈速 delta (s)."""
        delta, _ = _FUEL_MODE_PARAMS[mode]
        return delta

    def fuel_effect_on_lap_time(self, current_fuel_kg: float) -> float:
        """当前燃油质量对圈速的影响 (s, 正=慢).

        燃油越多越慢. 基准: 0 kg 燃油时无影响.
        """
        return (current_fuel_kg / 10.0) * _FUEL_PENALTY_PER_10KG_S

    # ------------------------------------------------------------------ #
    def calculate(self, state: FuelState) -> FuelConsumptionResult:
        """计算燃油状态."""
        consumption = self.consumption_for_mode(state.mode)
        mode_delta = self.lap_time_delta_for_mode(state.mode)

        laps_remaining = state.total_laps - state.current_lap
        if laps_remaining <= 0:
            return FuelConsumptionResult(
                fuel_used_kg=0.0,
                fuel_remaining_kg=state.current_fuel_kg,
                laps_remaining=0,
                fuel_per_lap_avg=consumption,
                fuel_deficit_kg=0.0,
                needs_fuel_save=False,
                projected_finish_fuel_kg=state.current_fuel_kg,
                lap_time_delta_s=mode_delta,
            )

        projected_total_use = consumption * laps_remaining
        projected_finish = state.current_fuel_kg - projected_total_use
        deficit = max(0.0, -projected_finish)  # 正 = 不足

        needs_save = deficit > 0.5  # 超过 0.5kg 不足需节油

        return FuelConsumptionResult(
            fuel_used_kg=consumption,
            fuel_remaining_kg=state.current_fuel_kg - consumption,
            laps_remaining=laps_remaining,
            fuel_per_lap_avg=consumption,
            fuel_deficit_kg=deficit,
            needs_fuel_save=needs_save,
            projected_finish_fuel_kg=projected_finish,
            lap_time_delta_s=mode_delta,
        )

    # ------------------------------------------------------------------ #
    def recommend_mode(self, state: FuelState) -> FuelMode:
        """推荐燃油模式 (基于余量)."""
        result = self.calculate(state)
        if result.needs_fuel_save:
            return FuelMode.LEAN
        # 余油充足 + 后段可推
        if result.projected_finish_fuel_kg > 8.0 and state.current_lap > state.total_laps * 0.7:
            return FuelMode.RICH
        return FuelMode.NORMAL

    def recommended_start_fuel_kg(self, total_laps: int, mode: FuelMode = FuelMode.NORMAL) -> float:
        """推荐起步油量 (kg).

        留 2 kg 安全余量.
        """
        consumption = self.consumption_for_mode(mode)
        needed = consumption * total_laps + 2.0
        return min(needed, _MAX_FUEL_KG)


# --------------------------------------------------------------------------- #
# 便捷函数
# --------------------------------------------------------------------------- #
def fuel_per_lap(track_id: str, mode: FuelMode = FuelMode.NORMAL) -> float:
    """便捷: 查询赛道每圈燃油消耗."""
    return FuelConsumptionModel(track_id).consumption_for_mode(mode)


def fuel_effect_on_lap_time(current_fuel_kg: float) -> float:
    """便捷: 燃油质量对圈速影响."""
    return (current_fuel_kg / 10.0) * _FUEL_PENALTY_PER_10KG_S


def recommended_start_fuel_kg(track_id: str, total_laps: int) -> float:
    """便捷: 推荐起步油量."""
    return FuelConsumptionModel(track_id).recommended_start_fuel_kg(total_laps)


def all_track_consumptions() -> dict[str, float]:
    """所有赛道基准燃油消耗."""
    return dict(_TRACK_FUEL_CONSUMPTION)
