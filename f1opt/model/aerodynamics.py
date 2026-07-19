"""F1 空气动力学模型 (Iter-6).

F1 圈速最大单一影响因素是空气动力学 — 下压力 (Cl·A) 决定弯中最大
横向 G, 阻力 (Cd·A) 决定直道尾速. 真实车队 (Red Bull/Mercedes/Ferrari
等) 通过 CFD + 风洞绘制 "aero map" 表征每个设置参数对 Cl/Cd 的影响.

本模块实现工程化的 aero map:

- 下压力系数 ``Cl``: 由前翼 / 后翼 / 离地间隙 (front & rear) / rake 决定.
- 阻力系数 ``Cd``: 由前翼 / 后翼决定, 受 DRS 影响可大幅降低.
- **地面效应** (ground effect): 2022+ F1 地板规则下, 离地间隙过低会触发
  "porpoising" (海豚跳), 过高则失去地面效应增益 — 存在最优离地间隙.
- **Mach 数修正**: 350 km/h ≈ Mach 0.28, 阻力在高速区非线性上升.
- **DRS 效果**: 按赛道 DRS 区长度估算圈速收益.

公开 API:
    - :class:`AeroMap` — 单车的 aero map (Cl/Cd vs 设置).
    - :class:`AerodynamicsModel` — 单圈 aero 仿真 (downforce N, drag N, 圈速增益).
    - :data:`DRS_TRACK_DATA` — 24 条赛道的 DRS 区数量与平均长度.

参考 (FIA 公开技术规则 + 公开工程资料):
    FIA F1 Technical Regulations 2026 §1 (Aerodynamic constraints).
    "How F1 ground effect works" — F1.com 公开技术文章.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# 物理常数
_AIR_DENSITY_SEA_LEVEL = 1.225  # kg/m³
_GRAVITY = 9.81                  # m/s²
_KMH_TO_MS = 1.0 / 3.6
_MACH_1_AT_SEA_LEVEL_MS = 340.0  # m/s

# 默认车辆参数 (2026 F1 公开估算)
_DEFAULT_REFERENCE_AREA_M2 = 1.5    # F1 前视投影面积 ~1.5 m²
_DEFAULT_MASS_KG = 798.0            # FIA 2026 最低车重 (车+车手)
_DEFAULT_WHEELBASE_M = 3.6
_DEFAULT_COP_HEIGHT_M = 0.5         # 压力中心高度 (估算)

# 翼面系数 (单格设置对应的 Cl/Cd 贡献, 基于 F1 公开工程估算)
_FRONT_WING_CL_PER_STEP = 0.04    # 前翼每格 +0.04 Cl
_REAR_WING_CL_PER_STEP = 0.06     # 后翼每格 +0.06 Cl
_FRONT_WING_CD_PER_STEP = 0.012   # 前翼每格 +0.012 Cd
_REAR_WING_CD_PER_STEP = 0.020    # 后翼每格 +0.020 Cd

# 地面效应: 最优离地间隙 ~25 mm, 增益钟形曲线 (sigma=10 mm)
_GROUND_EFFECT_OPTIMAL_MM = 25.0
_GROUND_EFFECT_SIGMA_MM = 10.0
_GROUND_EFFECT_MAX_GAIN = 0.45    # 最优时下压力 +45%
_GROUND_EFFECT_PORPOISING_THRESHOLD_MM = 10.0   # 低于此触发海豚跳惩罚

# DRS: 启用时后翼 Cd 降低 60-80%
_DRS_DRAG_REDUCTION_FACTOR = 0.30  # 启用 DRS 后 Cd 保留 30%
_DRS_DOWNFORCE_REDUCTION_FACTOR = 0.85  # 启用 DRS 后下压力保留 85%

# Mach 修正: 阻力在高速区非线性上升 (Prandtl-Glauert 简化)
_MACH_DRAG_EXPONENT = 2.5  # 简化指数


# --------------------------------------------------------------------------- #
# AeroMap
# --------------------------------------------------------------------------- #
@dataclass
class AeroMap:
    """单车的 aero map: 给定设置计算 Cl/Cd.

    所有设置值在 ``[0, 1]`` 归一化 (与 ``CarSetup.to_vector()`` 一致).
    """

    front_wing: float = 0.5          # 前翼角度 (0-1)
    rear_wing: float = 0.5           # 后翼角度 (0-1)
    ride_height_front_mm: float = 25.0   # 前离地间隙 (mm)
    ride_height_rear_mm: float = 35.0    # 后离地间隙 (mm)
    drs_active: bool = False         # DRS 是否启用

    def _ground_effect_factor(self, ride_height_mm: float) -> tuple[float, float]:
        """返回 (downforce_factor, porpoising_penalty).

        - ride_height ≈ 25 mm: downforce_factor=1+0.45, porpoising=0
        - ride_height 过低 (<10 mm): porpoising 损失大
        - ride_height 过高 (>50 mm): 地面效应损失
        """
        diff = ride_height_mm - _GROUND_EFFECT_OPTIMAL_MM
        # 钟形增益
        gain = _GROUND_EFFECT_MAX_GAIN * math.exp(
            -0.5 * (diff / _GROUND_EFFECT_SIGMA_MM) ** 2
        )
        downforce_factor = 1.0 + gain

        # 海豚跳惩罚: 离地间隙过低
        porpoising = 0.0
        if ride_height_mm < _GROUND_EFFECT_PORPOISING_THRESHOLD_MM:
            deficit = _GROUND_EFFECT_PORPOISING_THRESHOLD_MM - ride_height_mm
            porpoising = deficit * 0.05  # 每 mm 损失 5% 下压力

        return downforce_factor, porpoising

    @property
    def rake_mm(self) -> float:
        """rake = 后离地 - 前离地 (mm)."""
        return self.ride_height_rear_mm - self.ride_height_front_mm

    def cl(self) -> float:
        """下压力系数 Cl (无量纲).

        基础 Cl 来自翼面, 乘以地面效应 (前后分别), 减去海豚跳惩罚.
        """
        # 地面效应 (前后分别)
        f_factor, f_porpoise = self._ground_effect_factor(self.ride_height_front_mm)
        r_factor, r_porpoise = self._ground_effect_factor(self.ride_height_rear_mm)

        # 前翼 + 前地板用前 factor; 后翼 + 后地板用后 factor
        cl_front = (self.front_wing * _FRONT_WING_CL_PER_STEP * 10.0 + 0.5) * f_factor
        cl_rear = (self.rear_wing * _REAR_WING_CL_PER_STEP * 10.0 + 0.7) * r_factor
        cl_total = cl_front + cl_rear

        # 海豚跳惩罚
        cl_total *= (1.0 - f_porpoise * 0.5 - r_porpoise * 0.5)

        # DRS: 启用时后翼下压力降低
        if self.drs_active:
            cl_total *= _DRS_DOWNFORCE_REDUCTION_FACTOR

        return max(0.0, cl_total)

    def cd(self) -> float:
        """阻力系数 Cd (无量纲)."""
        cd_base = (
            self.front_wing * _FRONT_WING_CD_PER_STEP * 10.0
            + self.rear_wing * _REAR_WING_CD_PER_STEP * 10.0
            + 0.8   # 车身基础 Cd
        )
        if self.drs_active:
            cd_base *= _DRS_DRAG_REDUCTION_FACTOR
        return max(0.1, cd_base)

    def cl_cd_ratio(self) -> float:
        """空气动力学效率 Cl/Cd."""
        cd = self.cd()
        return self.cl() / cd if cd > 0 else 0.0


# --------------------------------------------------------------------------- #
# DRS 赛道数据 (24 条赛道 DRS 区数量 + 平均长度)
# --------------------------------------------------------------------------- #
DRS_TRACK_DATA: dict[str, dict[str, Any]] = {
    "melbourne": {"n_drs_zones": 4, "avg_zone_length_m": 700.0},
    "shanghai": {"n_drs_zones": 2, "avg_zone_length_m": 1100.0},
    "suzuka": {"n_drs_zones": 1, "avg_zone_length_m": 800.0},
    "bahrain": {"n_drs_zones": 3, "avg_zone_length_m": 750.0},
    "jeddah": {"n_drs_zones": 3, "avg_zone_length_m": 850.0},
    "miami": {"n_drs_zones": 3, "avg_zone_length_m": 600.0},
    "monaco": {"n_drs_zones": 1, "avg_zone_length_m": 350.0},
    "montreal": {"n_drs_zones": 2, "avg_zone_length_m": 700.0},
    "barcelona": {"n_drs_zones": 2, "avg_zone_length_m": 800.0},
    "silverstone": {"n_drs_zones": 2, "avg_zone_length_m": 800.0},
    "spa": {"n_drs_zones": 2, "avg_zone_length_m": 700.0},
    "budapest": {"n_drs_zones": 2, "avg_zone_length_m": 600.0},
    "amsterdam": {"n_drs_zones": 2, "avg_zone_length_m": 700.0},
    "monza": {"n_drs_zones": 2, "avg_zone_length_m": 1000.0},
    "baku": {"n_drs_zones": 2, "avg_zone_length_m": 1200.0},
    "singapore": {"n_drs_zones": 4, "avg_zone_length_m": 500.0},
    "austin": {"n_drs_zones": 2, "avg_zone_length_m": 900.0},
    "losail": {"n_drs_zones": 1, "avg_zone_length_m": 800.0},
    "madrid": {"n_drs_zones": 3, "avg_zone_length_m": 600.0},
    "interlagos": {"n_drs_zones": 2, "avg_zone_length_m": 700.0},
    "las_vegas": {"n_drs_zones": 2, "avg_zone_length_m": 1000.0},
    "yas_marina": {"n_drs_zones": 2, "avg_zone_length_m": 800.0},
}
_DEFAULT_DRS_DATA = {"n_drs_zones": 2, "avg_zone_length_m": 700.0}


def get_drs_data(track_id: str) -> dict[str, Any]:
    """获取赛道 DRS 数据, 未知返回默认."""
    return DRS_TRACK_DATA.get(track_id, _DEFAULT_DRS_DATA)


# --------------------------------------------------------------------------- #
# AerodynamicsModel
# --------------------------------------------------------------------------- #
@dataclass
class AerodynamicsModel:
    """单圈 aero 仿真: 给定设置 + 赛道, 计算下压力/阻力/圈速增益.

    用法::

        am = AerodynamicsModel(track_id="monza", front_wing=0.3, rear_wing=0.4,
                               ride_height_front_mm=20, ride_height_rear_mm=35)
        out = am.compute_lap_aero(avg_speed_ms=80.0, max_speed_ms=95.0)
        # out = {downforce_N, drag_N, cl, cd, drs_gain_s, ...}
    """

    track_id: str
    front_wing: float = 0.5
    rear_wing: float = 0.5
    ride_height_front_mm: float = 25.0
    ride_height_rear_mm: float = 35.0
    reference_area_m2: float = _DEFAULT_REFERENCE_AREA_M2
    air_density: float = _AIR_DENSITY_SEA_LEVEL
    mass_kg: float = _DEFAULT_MASS_KG
    _aero: AeroMap = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._aero = AeroMap(
            front_wing=self.front_wing,
            rear_wing=self.rear_wing,
            ride_height_front_mm=self.ride_height_front_mm,
            ride_height_rear_mm=self.ride_height_rear_mm,
            drs_active=False,
        )

    @property
    def aero_map(self) -> AeroMap:
        return self._aero

    # ------------------------------------------------------------------ #
    # 力计算
    # ------------------------------------------------------------------ #
    def downforce_n(self, speed_ms: float, drs_active: bool = False) -> float:
        """给定速度的下压力 (N). 公式: 0.5 * rho * v^2 * Cl * A."""
        self._aero.drs_active = drs_active
        try:
            cl = self._aero.cl()
            return 0.5 * self.air_density * speed_ms * speed_ms * cl * self.reference_area_m2
        finally:
            self._aero.drs_active = False  # restore

    def drag_n(self, speed_ms: float, drs_active: bool = False) -> float:
        """给定速度的阻力 (N). 公式: 0.5 * rho * v^2 * Cd * A * mach_correction."""
        self._aero.drs_active = drs_active
        try:
            cd = self._aero.cd()
            mach = speed_ms / _MACH_1_AT_SEA_LEVEL_MS
            # 简化 Prandtl-Glauert: 高速时阻力略增 (mach^2.5 因子, 但截断)
            mach_factor = max(1.0, 1.0 + 0.1 * mach ** _MACH_DRAG_EXPONENT)
            return (0.5 * self.air_density * speed_ms * speed_ms * cd
                    * self.reference_area_m2 * mach_factor)
        finally:
            self._aero.drs_active = False

    # ------------------------------------------------------------------ #
    # 单圈 aero 摘要
    # ------------------------------------------------------------------ #
    def compute_lap_aero(
        self, avg_speed_ms: float, max_speed_ms: float | None = None
    ) -> dict[str, Any]:
        """返回单圈 aero 摘要: 下压力/阻力 (avg & max), Cl/Cd, DRS 收益."""
        if max_speed_ms is None:
            max_speed_ms = avg_speed_ms * 1.4

        downforce_avg = self.downforce_n(avg_speed_ms)
        downforce_max = self.downforce_n(max_speed_ms)
        drag_avg = self.drag_n(avg_speed_ms)
        drag_max = self.drag_n(max_speed_ms)

        # DRS 收益估算: 每个 DRS 区节省 ~0.2 s/100 m 直道
        drs_data = get_drs_data(self.track_id)
        n_drs = drs_data["n_drs_zones"]
        avg_zone = drs_data["avg_zone_length_m"]
        # 估算 DRS 圈速收益
        drs_gain_s = self._estimate_drs_gain_s(max_speed_ms, n_drs, avg_zone)

        # 下压力 → 弯速收益估算: 每 1000 N 下压力 ≈ 0.05 s/lap
        corner_gain_s = downforce_avg / 1000.0 * 0.05

        # 阻力 → 直道损失: 每 1000 N 阻力 @ 80 m/s ≈ 0.04 s/lap
        drag_loss_s = drag_avg / 1000.0 * 0.04

        return {
            "track_id": self.track_id,
            "front_wing": self.front_wing,
            "rear_wing": self.rear_wing,
            "ride_height_front_mm": self.ride_height_front_mm,
            "ride_height_rear_mm": self.ride_height_rear_mm,
            "rake_mm": self._aero.rake_mm,
            "cl": self._aero.cl(),
            "cd": self._aero.cd(),
            "cl_cd_ratio": self._aero.cl_cd_ratio(),
            "downforce_avg_N": float(downforce_avg),
            "downforce_max_N": float(downforce_max),
            "drag_avg_N": float(drag_avg),
            "drag_max_N": float(drag_max),
            "drs_zones": int(n_drs),
            "drs_avg_zone_length_m": float(avg_zone),
            "drs_gain_s": float(drs_gain_s),
            "corner_gain_from_downforce_s": float(corner_gain_s),
            "drag_loss_s": float(drag_loss_s),
            "net_lap_gain_s": float(corner_gain_s + drs_gain_s - drag_loss_s),
        }

    def _estimate_drs_gain_s(
        self, max_speed_ms: float, n_drs: int, avg_zone_m: float
    ) -> float:
        """估算 DRS 圈速收益 (s).

        模型: 每个 DRS 区 = 直道长度 × 速度差.
        速度差 ≈ 12 km/h (3.3 m/s) DRS 启用 vs 关闭.
        时间收益 = zone_length / (v - dv/2) - zone_length / (v + dv/2)
        """
        if n_drs <= 0 or max_speed_ms <= 0:
            return 0.0
        dv = 3.3  # m/s 速度增量
        v_no_drs = max_speed_ms
        v_drs = max_speed_ms + dv
        # 单区时间差
        if v_no_drs <= 0 or v_drs <= 0:
            return 0.0
        time_diff = avg_zone_m * (1.0 / v_no_drs - 1.0 / v_drs)
        return max(0.0, time_diff * n_drs)

    # ------------------------------------------------------------------ #
    # 设置敏感性分析
    # ------------------------------------------------------------------ #
    def sensitivity_analysis(self, delta: float = 0.05) -> dict[str, float]:
        """计算每个设置参数对圈速净收益的敏感性 (s per unit delta).

        返回字典: ``{front_wing, rear_wing, ride_height_front, ride_height_rear}``.
        """
        base = self.compute_lap_aero(avg_speed_ms=80.0, max_speed_ms=110.0)
        base_gain = base["net_lap_gain_s"]
        out: dict[str, float] = {}

        for param in ("front_wing", "rear_wing"):
            old = getattr(self, param)
            setattr(self, param, min(1.0, old + delta))
            self._rebuild_aero()
            new_gain = self.compute_lap_aero(80.0, 110.0)["net_lap_gain_s"]
            out[param] = (new_gain - base_gain) / delta
            setattr(self, param, old)
            self._rebuild_aero()

        for param in ("ride_height_front_mm", "ride_height_rear_mm"):
            old = getattr(self, param)
            setattr(self, param, max(5.0, old + delta * 100.0))  # delta in m → mm
            self._rebuild_aero()
            new_gain = self.compute_lap_aero(80.0, 110.0)["net_lap_gain_s"]
            # 敏感性 per mm
            out[param] = (new_gain - base_gain) / (delta * 100.0)
            setattr(self, param, old)
            self._rebuild_aero()

        return out

    def _rebuild_aero(self) -> None:
        self._aero = AeroMap(
            front_wing=self.front_wing,
            rear_wing=self.rear_wing,
            ride_height_front_mm=self.ride_height_front_mm,
            ride_height_rear_mm=self.ride_height_rear_mm,
            drs_active=False,
        )

    # ------------------------------------------------------------------ #
    # 寻找最优设置 (简单网格)
    # ------------------------------------------------------------------ #
    def optimize_ride_height(
        self, speed_ms: float = 80.0, search_range_mm: tuple[float, float] = (10.0, 50.0),
        step_mm: float = 2.0
    ) -> tuple[float, float]:
        """寻找最优前离地间隙 (mm), 返回 (best_rh, best_cl).

        权衡: 低 = 高下压力但有海豚跳; 高 = 失去地面效应.
        不修改模型自身状态 (保存/恢复).
        """
        # 保存原状
        orig_rhf = self.ride_height_front_mm
        orig_rhr = self.ride_height_rear_mm
        best_rh = search_range_mm[0]
        best_cl = -1.0
        rh = search_range_mm[0]
        try:
            while rh <= search_range_mm[1]:
                self.ride_height_front_mm = rh
                self.ride_height_rear_mm = rh + 10.0  # 保持 10 mm rake
                self._rebuild_aero()
                cl = self._aero.cl()
                if cl > best_cl:
                    best_cl = cl
                    best_rh = rh
                rh += step_mm
        finally:
            # 恢复原状
            self.ride_height_front_mm = orig_rhf
            self.ride_height_rear_mm = orig_rhr
            self._rebuild_aero()
        return (best_rh, best_cl)
