"""F1 刹车系统模型 (Iter-7).

F1 用碳纤维刹车盘 (carbon-carbon) 工作温度窗口 400-700 °C, 远高于民用
钢盘. 刹车偏置 (front/rear balance) 影响入弯稳定性 — 真实车队每圈都会
动态调整 (corner-by-corner brake bias migration).

本模块实现:

- :class:`BrakeBias`: 前后刹车偏置 (0.50-0.65 前), 含 ERS 回收影响.
- :class:`BrakeThermalModel`: 刹车盘温度仿真, 工作窗口 400-700 °C,
  过热/过冷惩罚.
- :class:`BrakeWearModel`: 碳盘磨损 (mm/lap), 受温度 + 偏置 + 赛道影响.
- :class:`BrakeModel`: 综合模型, 给定偏置+冷却设置+圈数, 输出单圈影响.

公开 API:
    - :class:`BrakeBias`, :class:`BrakeThermalModel`, :class:`BrakeWearModel`,
      :class:`BrakeModel`
    - :data:`BRAKE_TRACK_LOAD` — 24 条赛道的刹车负载等级 (1-5).

参考 (FIA 公开技术规则 + Brembo 公开刹车数据):
    FIA F1 Technical Regulations 2026 §11 (Brakes).
    Brembo Motorsport "F1 brake track profile" 公开数据.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 物理常数 / 默认参数
_BRAKE_DISC_THICKNESS_MM_NEW = 32.0       # 新盘厚度
_BRAKE_DISC_THICKNESS_MM_MIN = 24.0       # 磨损极限 (8 mm 可磨)
_BRAKE_DISC_MASS_KG = 1.2                 # 单盘质量
_BRAKE_SPECIFIC_HEAT = 800.0              # J/(kg·K) 碳纤维
_BRAKE_COOLING_AMBIENT_C = 60.0           # 刹车气道环境温度 (赛道辐射热)

# 工作窗口 (°C)
_OPTIMAL_TEMP_LOW = 400.0
_OPTIMAL_TEMP_HIGH = 700.0
_PEAK_GRIP_TEMP_C = 550.0                 # 最大摩擦系数温度
_OVERHEAT_PENALTY_PER_C = 0.05            # s/lap per °C above 700
_UNDERTEMP_PENALTY_PER_C = 0.08           # s/lap per °C below 400

# 偏置参数
_BRAKE_BIAS_FRONT_MIN = 0.50              # 50% 前 (过度偏后)
_BRAKE_BIAS_FRONT_MAX = 0.65              # 65% 前 (过度偏前)
_BRAKE_BIAS_OPTIMAL = 0.56                # 56% 前 (典型 F1)
_BRAKE_BIAS_PENALTY_PER_PCT = 0.015       # 每 1% 偏离最优的圈速损失

# 磨损参数
_BRAKE_WEAR_MM_PER_LAP_BASE = 0.05        # 基础磨损 mm/lap
_BRAKE_WEAR_TEMP_FACTOR = 0.0015          # 温度系数 (per °C above 500)
_BRAKE_WEAR_BIAS_FACTOR = 0.02            # 偏置系数 (前偏置高 → 前盘磨损快)

# 热模型校准 (F1 公开工程估算)
# 单盘每圈能量 ~150-400 kJ (轻赛道-重赛道), 4 盘共 ~0.6-1.6 MJ/lap.
# 1.2 kg × 800 J/kgK = 960 J/K → 300 kJ 输入 → 312°C 温升.
# 冷却率必须使温度在 400-700°C 平衡 (热输入 = 冷却).
_BRAKE_ENERGY_PER_LAP_KJ_DEFAULT = 300.0  # 单盘每圈能量 (kJ)
_BRAKE_COOLING_RATE_PER_K = 0.0008        # per mm² duct area, per K delta


# --------------------------------------------------------------------------- #
# 赛道刹车负载 (1=轻 5=重)
# --------------------------------------------------------------------------- #
BRAKE_TRACK_LOAD: dict[str, int] = {
    "monaco": 5, "singapore": 5, "budapest": 4, "montreal": 4, "austin": 4,
    "suzuka": 3, "silverstone": 2, "spa": 2, "monza": 2, "bahrain": 4,
    "jeddah": 2, "melbourne": 3, "shanghai": 4, "miami": 3, "barcelona": 3,
    "amsterdam": 3, "baku": 3, "losail": 3, "madrid": 4, "interlagos": 4,
    "las_vegas": 2, "yas_marina": 3,
}
_DEFAULT_TRACK_LOAD = 3


def get_brake_track_load(track_id: str) -> int:
    return BRAKE_TRACK_LOAD.get(track_id, _DEFAULT_TRACK_LOAD)


# --------------------------------------------------------------------------- #
# BrakeBias
# --------------------------------------------------------------------------- #
@dataclass
class BrakeBias:
    """刹车偏置 (front fraction 0.5-0.65), 含 ERS 回收影响.

    - ``front_fraction``: 前刹车比例 (0.50-0.65).
    - ``ers_migration``: ERS 回收时偏置后移 (向后 0-5%), 因为 MGU-K 在
      后轴回收相当于增加了后刹.
    """

    front_fraction: float = _BRAKE_BIAS_OPTIMAL
    ers_migration: float = 0.0  # 0-0.05 (0-5% 后移)

    def __post_init__(self) -> None:
        self.front_fraction = max(_BRAKE_BIAS_FRONT_MIN,
                                  min(_BRAKE_BIAS_FRONT_MAX, self.front_fraction))
        self.ers_migration = max(0.0, min(0.05, self.ers_migration))

    @property
    def effective_front_fraction(self) -> float:
        """实际前刹比例 (考虑 ERS 后移)."""
        return max(_BRAKE_BIAS_FRONT_MIN,
                   self.front_fraction - self.ers_migration)

    @property
    def rear_fraction(self) -> float:
        return 1.0 - self.effective_front_fraction

    def deviation_penalty_s(self) -> float:
        """偏离最优偏置 (56% 前) 的圈速惩罚 (s/lap)."""
        dev = abs(self.effective_front_fraction - _BRAKE_BIAS_OPTIMAL)
        return dev * 100.0 * _BRAKE_BIAS_PENALTY_PER_PCT

    def lockup_risk(self) -> float:
        """前轮锁死风险 (0-1, 1=最高).

        前偏置高 → 前轮易锁死 (高刹车压力时).
        """
        excess_front = max(0.0, self.effective_front_fraction - _BRAKE_BIAS_OPTIMAL)
        return min(1.0, excess_front * 12.0)

    def rear_instability_risk(self) -> float:
        """后轮不稳定风险 (0-1, 1=最高).

        前偏置低 → 后轮承担多 → 入弯后部不稳定.
        """
        excess_rear = max(0.0, _BRAKE_BIAS_OPTIMAL - self.effective_front_fraction)
        return min(1.0, excess_rear * 15.0)


# --------------------------------------------------------------------------- #
# BrakeThermalModel
# --------------------------------------------------------------------------- #
@dataclass
class BrakeThermalModel:
    """单盘温度仿真, 工作窗口 400-700 °C.

    模型: 每圈温度变化 = (刹车能量输入 - 冷却) / (质量 × 比热).
    校准: 默认 300 kJ/lap 单盘, 100 mm² 气道 → 在 ~550°C 平衡.
    """

    initial_temp_c: float = 450.0
    cooling_duct_area_mm2: float = 100.0  # 冷却气道面积 (mm²)
    brake_energy_kj_per_lap: float = _BRAKE_ENERGY_PER_LAP_KJ_DEFAULT

    def temp_after_lap(
        self, current_temp_c: float, track_load: int = 3
    ) -> float:
        """仿真一圈后刹车盘温度 (°C).

        track_load 1-5: 高负载赛道能量输入更大.
        """
        # 输入能量 (受赛道负载影响): load=1 → 0.85x, load=5 → 1.45x
        energy_in_kj = self.brake_energy_kj_per_lap * (0.7 + 0.15 * track_load)
        # 温升 (J / (kg·J/kgK) = K)
        delta_t_in = energy_in_kj * 1000.0 / (
            _BRAKE_DISC_MASS_KG * _BRAKE_SPECIFIC_HEAT
        )
        # 冷却: 气道面积越大冷却越快, 与温差成正比.
        # 校准: 100 mm² 气道, 300 kJ 输入 → 550°C 平衡.
        #   delta_in(312) = (550-60) * rate * 100 → rate = 0.0064
        cooling_rate = _BRAKE_COOLING_RATE_PER_K * 8.0 * self.cooling_duct_area_mm2
        delta_t_out = (current_temp_c - _BRAKE_COOLING_AMBIENT_C) * cooling_rate
        new_temp = current_temp_c + delta_t_in - delta_t_out
        return max(_BRAKE_COOLING_AMBIENT_C, min(1500.0, new_temp))

    def thermal_penalty_s(self, temp_c: float) -> float:
        """温度偏离工作窗口的圈速惩罚 (s/lap)."""
        if temp_c < _OPTIMAL_TEMP_LOW:
            return ( _OPTIMAL_TEMP_LOW - temp_c) * _UNDERTEMP_PENALTY_PER_C
        if temp_c > _OPTIMAL_TEMP_HIGH:
            return (temp_c - _OPTIMAL_TEMP_HIGH) * _OVERHEAT_PENALTY_PER_C
        return 0.0

    def in_window(self, temp_c: float) -> bool:
        return _OPTIMAL_TEMP_LOW <= temp_c <= _OPTIMAL_TEMP_HIGH


# --------------------------------------------------------------------------- #
# BrakeWearModel
# --------------------------------------------------------------------------- #
@dataclass
class BrakeWearModel:
    """碳盘磨损 (mm/lap), 受温度 + 偏置 + 赛道负载影响.

    新盘厚度 32 mm, 极限 24 mm (8 mm 可用).
    """

    current_thickness_mm: float = _BRAKE_DISC_THICKNESS_MM_NEW

    def wear_per_lap(
        self,
        temp_c: float,
        bias: BrakeBias,
        track_load: int = 3,
        is_front: bool = True,
    ) -> float:
        """单圈磨损 (mm)."""
        # 基础磨损受赛道负载
        base = _BRAKE_WEAR_MM_PER_LAP_BASE * (0.6 + 0.2 * track_load)
        # 温度系数: 500°C 以上加速磨损
        temp_factor = 1.0 + max(0.0, temp_c - 500.0) * _BRAKE_WEAR_TEMP_FACTOR
        # 偏置系数: 前偏置高 → 前盘磨损更快, 后盘磨损更慢
        if is_front:
            bias_factor = 1.0 + (bias.effective_front_fraction - 0.5) * 2.0
        else:
            bias_factor = 1.0 + (0.5 - bias.effective_front_fraction) * 2.0
        return base * temp_factor * max(0.5, bias_factor)

    def laps_remaining(
        self,
        temp_c: float,
        bias: BrakeBias,
        track_load: int = 3,
        is_front: bool = True,
    ) -> int:
        """剩余可用圈数 (达到磨损极限)."""
        wear = self.wear_per_lap(temp_c, bias, track_load, is_front)
        if wear <= 0:
            return 9999
        remaining_mm = self.current_thickness_mm - _BRAKE_DISC_THICKNESS_MM_MIN
        return max(0, int(remaining_mm / wear))


# --------------------------------------------------------------------------- #
# BrakeModel (综合)
# --------------------------------------------------------------------------- #
@dataclass
class BrakeModel:
    """综合刹车模型: 给定偏置 + 冷却 + 圈数, 输出单圈影响.

    用法::

        bm = BrakeModel(track_id="monaco", front_fraction=0.58,
                        cooling_duct_area_mm2=120, brake_energy_kj_per_lap=2200)
        lap = bm.simulate_lap(current_temp_c=480.0)
        # lap = {bias_penalty_s, thermal_penalty_s, new_temp_c, ...}
        stint = bm.simulate_stint(laps=20)
    """

    track_id: str
    front_fraction: float = _BRAKE_BIAS_OPTIMAL
    cooling_duct_area_mm2: float = 100.0
    brake_energy_kj_per_lap: float = _BRAKE_ENERGY_PER_LAP_KJ_DEFAULT
    initial_disc_thickness_mm: float = _BRAKE_DISC_THICKNESS_MM_NEW

    _bias: BrakeBias = field(init=False, repr=False)
    _thermal: BrakeThermalModel = field(init=False, repr=False)
    _wear: BrakeWearModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._bias = BrakeBias(front_fraction=self.front_fraction)
        self._thermal = BrakeThermalModel(
            cooling_duct_area_mm2=self.cooling_duct_area_mm2,
            brake_energy_kj_per_lap=self.brake_energy_kj_per_lap,
        )
        self._wear = BrakeWearModel(current_thickness_mm=self.initial_disc_thickness_mm)

    @property
    def bias(self) -> BrakeBias:
        return self._bias

    @property
    def thermal(self) -> BrakeThermalModel:
        return self._thermal

    @property
    def wear(self) -> BrakeWearModel:
        return self._wear

    # ------------------------------------------------------------------ #
    def simulate_lap(self, current_temp_c: float) -> dict[str, Any]:
        """仿真单圈刹车系统影响."""
        track_load = get_brake_track_load(self.track_id)
        new_temp = self._thermal.temp_after_lap(current_temp_c, track_load)
        thermal_pen = self._thermal.thermal_penalty_s(new_temp)
        bias_pen = self._bias.deviation_penalty_s()
        front_wear = self._wear.wear_per_lap(
            new_temp, self._bias, track_load, is_front=True
        )
        rear_wear = self._wear.wear_per_lap(
            new_temp, self._bias, track_load, is_front=False
        )
        # 锁死/不稳定风险 → 圈速小损失
        lockup_pen = self._bias.lockup_risk() * 0.04
        instability_pen = self._bias.rear_instability_risk() * 0.06
        return {
            "track_id": self.track_id,
            "track_load": int(track_load),
            "front_fraction": self._bias.effective_front_fraction,
            "rear_fraction": self._bias.rear_fraction,
            "temp_before_c": float(current_temp_c),
            "temp_after_c": float(new_temp),
            "in_window": self._thermal.in_window(new_temp),
            "bias_penalty_s": float(bias_pen),
            "thermal_penalty_s": float(thermal_pen),
            "lockup_penalty_s": float(lockup_pen),
            "instability_penalty_s": float(instability_pen),
            "front_wear_mm": float(front_wear),
            "rear_wear_mm": float(rear_wear),
            "total_lap_penalty_s": float(
                bias_pen + thermal_pen + lockup_pen + instability_pen
            ),
        }

    # ------------------------------------------------------------------ #
    def simulate_stint(
        self, laps: int, initial_temp_c: float = 450.0
    ) -> list[dict[str, Any]]:
        """仿真多圈, 跨圈温度与磨损传递."""
        temp = float(initial_temp_c)
        out: list[dict[str, Any]] = []
        cumulative_wear_front = 0.0
        cumulative_wear_rear = 0.0
        for k in range(int(laps)):
            r = self.simulate_lap(temp)
            r["lap"] = k + 1
            cumulative_wear_front += r["front_wear_mm"]
            cumulative_wear_rear += r["rear_wear_mm"]
            r["cumulative_front_wear_mm"] = float(cumulative_wear_front)
            r["cumulative_rear_wear_mm"] = float(cumulative_wear_rear)
            r["disc_thickness_front_mm"] = float(
                self.initial_disc_thickness_mm - cumulative_wear_front
            )
            r["disc_thickness_rear_mm"] = float(
                self.initial_disc_thickness_mm - cumulative_wear_rear
            )
            temp = r["temp_after_c"]
            out.append(r)
        return out

    # ------------------------------------------------------------------ #
    def optimize_bias(
        self, temp_c: float = 500.0, step: float = 0.01
    ) -> tuple[float, float]:
        """寻找最优前偏置 (minimize total_lap_penalty_s).

        返回 (best_front_fraction, best_penalty_s).
        """
        best_ff = _BRAKE_BIAS_OPTIMAL
        best_pen = float("inf")
        ff = _BRAKE_BIAS_FRONT_MIN
        while ff <= _BRAKE_BIAS_FRONT_MAX:
            self._bias = BrakeBias(front_fraction=ff)
            r = self.simulate_lap(temp_c)
            if r["total_lap_penalty_s"] < best_pen:
                best_pen = r["total_lap_penalty_s"]
                best_ff = ff
            ff += step
        # 恢复
        self._bias = BrakeBias(front_fraction=self.front_fraction)
        return (best_ff, best_pen)

    # ------------------------------------------------------------------ #
    def optimize_cooling(
        self, temp_c: float = 500.0, target_temp_c: float = 550.0,
        search_range: tuple[float, float] = (50.0, 200.0),
        step: float = 5.0
    ) -> tuple[float, float]:
        """寻找使刹车盘温度最接近 target_temp_c 的冷却气道面积 (mm²).

        返回 (best_area, achieved_temp).
        """
        best_area = search_range[0]
        best_diff = float("inf")
        best_temp = target_temp_c
        orig_area = self._thermal.cooling_duct_area_mm2
        area = search_range[0]
        try:
            while area <= search_range[1]:
                self._thermal.cooling_duct_area_mm2 = area
                t = self._thermal.temp_after_lap(
                    temp_c, get_brake_track_load(self.track_id)
                )
                diff = abs(t - target_temp_c)
                if diff < best_diff:
                    best_diff = diff
                    best_area = area
                    best_temp = t
                area += step
        finally:
            self._thermal.cooling_duct_area_mm2 = orig_area
        return (best_area, best_temp)

    def brake_migration_analysis(self, laps: int = 10) -> dict:
        """Iter-191: 刹车偏置迁移分析 — 多圈偏置变化对圈速的影响.

        模拟逐年 (每圈) 偏置微调的效果, 返回最优偏置路径.
        """
        track_load = get_brake_track_load(self.track_id)
        best_penalty = float("inf")
        best_path: list[float] = []
        # 搜索从 0.52 到 0.60 的偏置路径
        for start_bias in (0.52, 0.54, 0.56, 0.58, 0.60):
            total_pen = 0.0
            path: list[float] = []
            current_bias = start_bias
            temp = 450.0
            for _ in range(laps):
                self._bias = BrakeBias(front_fraction=current_bias)
                r = self.simulate_lap(temp)
                total_pen += r["total_lap_penalty_s"]
                path.append(current_bias)
                temp = r["temp_after_c"]
                # 偏置微调: 向最优偏置渐进
                current_bias += 0.005 * (0.56 - current_bias)
            if total_pen < best_penalty:
                best_penalty = total_pen
                best_path = path
        self._bias = BrakeBias(front_fraction=self.front_fraction)
        return {
            "optimal_path": best_path,
            "total_penalty_s": float(best_penalty),
            "laps": laps,
        }
