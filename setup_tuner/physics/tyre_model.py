"""Pirelli轮胎模型 — 胎温/胎压/外倾 → 抓地力 + 磨耗.

物理原理:
    - 温度惩罚: 抛物线模型, 偏离最优温度时抓地力下降.
      temp_penalty = max(0, 1 - ((temp - temp_opt) / temp_window)^2)
    - 胎压惩罚: 偏离最优胎压时接触面积变化, 抓地力下降.
      pressure_penalty = max(0.5, 1 - ((pressure - pressure_opt) / 2.0)^2)
    - 外倾角惩罚: 外倾不足(<1°)或过大(>3.5°)均降低抓地力.
      |camber| < 1.0 → 0.7 + 0.3 * |camber|
      1.0 ≤ |camber| ≤ 3.5 → 1.0 (最优区间)
      |camber| > 3.5 → max(0.5, 1 - (|camber| - 3.5) * 0.2)
    - 综合抓地力: μ = mu_peak * temp_penalty * pressure_penalty * camber_penalty
    - 磨耗率: wear_rate = wear_rate_base * (1 + slip_angle * 0.1) * (vertical_load / 5000)
      过热时(temp > temp_opt + temp_window)磨耗率 ×1.5
    - 侧向力: cornering_force = μ * vertical_load * min(1.0, slip_angle / 5.0)
"""

from __future__ import annotations

from dataclasses import dataclass

G = 9.81  # m/s², 重力加速度

# Pirelli F1轮胎参数（基于公开数据估算）
TYRE_PARAMS: dict[str, dict[str, float]] = {
    "soft": {"mu_peak": 1.8, "temp_opt": 95.0, "temp_window": 25.0,
             "wear_rate_base": 1.5, "pressure_opt": 23.5},
    "medium": {"mu_peak": 1.6, "temp_opt": 100.0, "temp_window": 30.0,
               "wear_rate_base": 1.0, "pressure_opt": 23.0},
    "hard": {"mu_peak": 1.4, "temp_opt": 105.0, "temp_window": 35.0,
             "wear_rate_base": 0.7, "pressure_opt": 22.5},
    "intermediate": {"mu_peak": 1.2, "temp_opt": 70.0, "temp_window": 20.0,
                     "wear_rate_base": 2.0, "pressure_opt": 22.0},
    "wet": {"mu_peak": 0.9, "temp_opt": 50.0, "temp_window": 15.0,
            "wear_rate_base": 3.0, "pressure_opt": 21.5},
}

# 外倾角惩罚区间边界
CAMBER_LOW_THRESHOLD = 1.0    # 度, 外倾不足下界
CAMBER_HIGH_THRESHOLD = 3.5   # 度, 外倾过大上界
CAMBER_LOW_BASE = 0.7         # 外倾不足时的基础惩罚
CAMBER_LOW_SCALE = 0.3        # 外倾不足时的线性系数
CAMBER_HIGH_SCALE = 0.2       # 外倾过大时的线性系数
CAMBER_MIN_PENALTY = 0.5      # 外倾惩罚下限

# 胎压惩罚参数
PRESSURE_TOLERANCE = 2.0      # psi, 胎压容忍窗口
PRESSURE_MIN_PENALTY = 0.5    # 胎压惩罚下限

# 磨耗率参数
WEAR_LOAD_REFERENCE = 5000.0  # N, 磨耗率参考垂直载荷
WEAR_SLIP_SCALE = 0.1         # 滑移角对磨耗的影响系数
WEAR_OVERHEAT_MULTIPLIER = 1.5  # 过热时磨耗倍率

# 侧向力参数
SLIP_ANGLE_REFERENCE = 5.0    # 度, 侧向力饱和滑移角


@dataclass
class TyreOutput:
    """轮胎模型计算输出."""
    grip_coefficient: float    # 抓地力系数 μ (无量纲)
    wear_rate: float           # 磨耗率 (%/lap)
    temp_penalty: float        # 温度惩罚因子 (0-1, 1=最优)
    pressure_penalty: float    # 胎压惩罚因子 (0-1, 1=最优)
    camber_penalty: float      # 外倾角惩罚因子 (0-1, 1=最优)
    cornering_force: float     # 侧向力 (N)
    vertical_load: float       # 垂直载荷 (N)


class TyreModel:
    """Pirelli轮胎模型.

    根据轮胎类型、胎压、外倾角初始化, 然后在给定温度、垂直载荷和
    滑移角条件下计算抓地力系数、磨耗率和侧向力.
    """

    def __init__(self, tyre_type: str, pressure: float, camber: float) -> None:
        self.tyre_type = tyre_type
        self.pressure = pressure       # psi
        self.camber = camber           # 度 (负值)
        self.params = TYRE_PARAMS.get(tyre_type, TYRE_PARAMS["medium"])

    def compute(self, temp: float, vertical_load: float,
                slip_angle: float = 0.0) -> TyreOutput:
        """计算给定条件下的轮胎输出.

        Args:
            temp: 轮胎温度 (°C)
            vertical_load: 垂直载荷 (N)
            slip_angle: 滑移角 (度, 默认0)

        Returns:
            TyreOutput 包含抓地力系数、磨耗率、各惩罚因子、侧向力等
        """
        temp_penalty = self._calc_temp_penalty(temp)
        pressure_penalty = self._calc_pressure_penalty()
        camber_penalty = self._calc_camber_penalty()

        grip = (self.params["mu_peak"] * temp_penalty
                * pressure_penalty * camber_penalty)
        wear = self._calc_wear_rate(temp, vertical_load, slip_angle)
        cornering_force = grip * vertical_load * min(
            1.0, abs(slip_angle) / SLIP_ANGLE_REFERENCE)

        return TyreOutput(
            grip_coefficient=grip,
            wear_rate=wear,
            temp_penalty=temp_penalty,
            pressure_penalty=pressure_penalty,
            camber_penalty=camber_penalty,
            cornering_force=cornering_force,
            vertical_load=vertical_load,
        )

    def _calc_temp_penalty(self, temp: float) -> float:
        """计算温度惩罚因子（抛物线模型）."""
        temp_opt = self.params["temp_opt"]
        temp_window = self.params["temp_window"]
        deviation = (temp - temp_opt) / temp_window
        return max(0.0, 1.0 - deviation * deviation)

    def _calc_pressure_penalty(self) -> float:
        """计算胎压惩罚因子."""
        pressure_opt = self.params["pressure_opt"]
        deviation = (self.pressure - pressure_opt) / PRESSURE_TOLERANCE
        return max(PRESSURE_MIN_PENALTY, 1.0 - deviation * deviation)

    def _calc_camber_penalty(self) -> float:
        """计算外倾角惩罚因子."""
        camber_abs = abs(self.camber)
        if camber_abs < CAMBER_LOW_THRESHOLD:
            return CAMBER_LOW_BASE + CAMBER_LOW_SCALE * camber_abs
        if camber_abs <= CAMBER_HIGH_THRESHOLD:
            return 1.0
        excess = camber_abs - CAMBER_HIGH_THRESHOLD
        return max(CAMBER_MIN_PENALTY, 1.0 - excess * CAMBER_HIGH_SCALE)

    def _calc_wear_rate(self, temp: float, vertical_load: float,
                        slip_angle: float) -> float:
        """计算磨耗率 (%/lap)."""
        base = self.params["wear_rate_base"]
        load_factor = vertical_load / WEAR_LOAD_REFERENCE
        slip_factor = 1.0 + abs(slip_angle) * WEAR_SLIP_SCALE
        wear = base * slip_factor * load_factor

        temp_opt = self.params["temp_opt"]
        temp_window = self.params["temp_window"]
        if temp > temp_opt + temp_window:
            wear *= WEAR_OVERHEAT_MULTIPLIER
        return wear