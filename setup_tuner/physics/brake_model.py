"""制动模型 — 刹车压力/偏置 → 制动力 + 锁死阈值。

物理原理：
    F1 2026规则下的制动系统简化模型。刹车压力决定总制动力大小，
    刹车偏置决定前后轴制动力分配比例。

    - 总制动力 = MAX_BRAKE_FORCE_PER_WHEEL * 4 * (pressure/100)
    - 前轴制动力 = 总制动力 * (bias/100)
    - 后轴制动力 = 总制动力 * (1 - bias/100)
    - 锁死阈值（减速度）= μ_tyre * (静态载荷 + 下压力) / CAR_MASS
      轮胎能承受的最大摩擦力 = μ * 法向力，当制动力超过此值时锁死。
      前轴静态载荷 ≈ CAR_MASS * G * 0.5（50:50重量分配）
    - 制动稳定性：偏置越靠前越稳定（50%→0.5, 70%→1.0）
    - 最大减速度 = 总制动力 / CAR_MASS
"""
from __future__ import annotations

from dataclasses import dataclass

# ── 物理常数 ──────────────────────────────────────────────
G: float = 9.81                       # m/s², 重力加速度
CAR_MASS: float = 798.0               # kg, F1 2026规则最低重量
MAX_BRAKE_FORCE_PER_WHEEL: float = 2000.0  # N, 100%压力时单轮最大制动力
WHEEL_COUNT: int = 4                  # 制动轮数量
STATIC_WEIGHT_DIST_FRONT: float = 0.5  # 前轴静态重量分配比例

# ── 刹车偏置范围 ──────────────────────────────────────────
BRAKE_BIAS_MIN: float = 50.0  # %, 最小前轴偏置
BRAKE_BIAS_MAX: float = 70.0  # %, 最大前轴偏置

# ── 制动稳定性映射 ────────────────────────────────────────
STABILITY_AT_MIN_BIAS: float = 0.5   # 50%偏置时的稳定性
STABILITY_AT_MAX_BIAS: float = 1.0   # 70%偏置时的稳定性


@dataclass
class BrakeOutput:
    """制动模型计算输出。

    力的单位为牛顿(N)，减速度单位为m/s²。
    """
    brake_force_front: float     # N, 前轴制动力
    brake_force_rear: float      # N, 后轴制动力
    brake_force_total: float     # N, 总制动力
    lock_threshold_front: float  # m/s², 前轮锁死阈值（减速度）
    lock_threshold_rear: float   # m/s², 后轮锁死阈值
    brake_stability: float       # 0-1, 制动稳定性评分
    deceleration_max: float      # m/s², 最大减速度


class BrakeModel:
    """制动模型。

    通过刹车压力和刹车偏置，计算制动力分配、锁死阈值和制动稳定性。
    """

    def __init__(self, brake_pressure: float, brake_bias: float) -> None:
        """初始化制动模型。

        Args:
            brake_pressure: 刹车压力百分比 (80-100%)
            brake_bias: 刹车偏置百分比 (50-70%, 前轴占比)
        """
        self.brake_pressure: float = brake_pressure
        self.brake_bias: float = brake_bias

    def compute(
        self,
        downforce_front: float = 0.0,
        downforce_rear: float = 0.0,
        tyre_mu_front: float = 1.5,
        tyre_mu_rear: float = 1.5,
    ) -> BrakeOutput:
        """计算给定条件下的制动输出。

        Args:
            downforce_front: 前轴下压力 (N)
            downforce_rear: 后轴下压力 (N)
            tyre_mu_front: 前轮轮胎摩擦系数
            tyre_mu_rear: 后轮轮胎摩擦系数

        Returns:
            BrakeOutput 包含制动力、锁死阈值、稳定性等物理量
        """
        brake_force_total = self._calc_total_brake_force()
        brake_force_front = self._calc_front_brake_force(brake_force_total)
        brake_force_rear = self._calc_rear_brake_force(brake_force_total)

        lock_threshold_front = self._calc_lock_threshold(
            tyre_mu_front, downforce_front,
        )
        lock_threshold_rear = self._calc_lock_threshold(
            tyre_mu_rear, downforce_rear,
        )

        brake_stability = self._calc_brake_stability()
        deceleration_max = self._calc_deceleration_max(brake_force_total)

        return BrakeOutput(
            brake_force_front=brake_force_front,
            brake_force_rear=brake_force_rear,
            brake_force_total=brake_force_total,
            lock_threshold_front=lock_threshold_front,
            lock_threshold_rear=lock_threshold_rear,
            brake_stability=brake_stability,
            deceleration_max=deceleration_max,
        )

    def _calc_total_brake_force(self) -> float:
        """计算总制动力 = MAX_BRAKE_FORCE_PER_WHEEL * 4 * (pressure/100)。"""
        pressure_ratio = self.brake_pressure / 100.0
        return MAX_BRAKE_FORCE_PER_WHEEL * WHEEL_COUNT * pressure_ratio

    def _calc_front_brake_force(self, total: float) -> float:
        """计算前轴制动力 = 总制动力 * (bias/100)。"""
        return total * (self.brake_bias / 100.0)

    def _calc_rear_brake_force(self, total: float) -> float:
        """计算后轴制动力 = 总制动力 * (1 - bias/100)。"""
        return total * (1.0 - self.brake_bias / 100.0)

    @staticmethod
    def _calc_lock_threshold(tyre_mu: float, downforce: float) -> float:
        """计算锁死阈值减速度 = μ * (静态载荷 + 下压力) / CAR_MASS。

        轮胎能承受的最大摩擦力 = μ * 法向力。
        锁死阈值减速度 = 最大摩擦力 / 车辆质量。
        """
        static_load = CAR_MASS * G * STATIC_WEIGHT_DIST_FRONT
        normal_force = static_load + downforce
        max_friction = tyre_mu * normal_force
        return max_friction / CAR_MASS

    def _calc_brake_stability(self) -> float:
        """计算制动稳定性评分（50%→0.5, 70%→1.0，线性映射）。"""
        if self.brake_bias <= BRAKE_BIAS_MIN:
            return STABILITY_AT_MIN_BIAS
        if self.brake_bias >= BRAKE_BIAS_MAX:
            return STABILITY_AT_MAX_BIAS
        bias_range = BRAKE_BIAS_MAX - BRAKE_BIAS_MIN
        stability_range = STABILITY_AT_MAX_BIAS - STABILITY_AT_MIN_BIAS
        t = (self.brake_bias - BRAKE_BIAS_MIN) / bias_range
        return STABILITY_AT_MIN_BIAS + t * stability_range

    @staticmethod
    def _calc_deceleration_max(brake_force_total: float) -> float:
        """计算最大减速度 = 总制动力 / CAR_MASS。"""
        return brake_force_total / CAR_MASS