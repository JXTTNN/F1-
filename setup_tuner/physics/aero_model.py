"""空气动力学模型 — 前翼/后翼 → 下压力 + 阻力。

物理原理：
    F1 2026规则下的空气动力学简化模型。前翼和后翼的调教级别决定了
    升力系数(CL)，进而通过伯努利方程计算下压力。

    - 前翼级别 0-10 → CL_front 从 0.3(最低)到 2.0(最高)，线性映射
    - 后翼级别 0-11 → CL_rear 从 0.2(最低)到 1.6(最高)，线性映射
    - 行驶高度影响下压力（地面效应）：
        level 2 → 乘数 1.15（低行驶高度，地面效应增强）
        level 4 → 乘数 1.0（基准）
        level 7 → 乘数 0.85（高行驶高度，地面效应减弱）
    - 下压力 = CL * 0.5 * ρ * v² * A
    - 阻力系数 CD ≈ (CL_front + CL_rear) * 0.15 + 0.3
    - 阻力 = CD * 0.5 * ρ * v² * (前翼面积 + 后翼面积)
    - 空力平衡 = 前轴下压力 / 总下压力 * 100%
"""
from __future__ import annotations

from dataclasses import dataclass

# ── 物理常数（F1 2026规则）──────────────────────────────
RHO_AIR: float = 1.225        # kg/m³, 空气密度（海平面标准）
G: float = 9.81               # m/s², 重力加速度
CAR_MASS: float = 798.0       # kg, F1 2026规则最低重量
FRONT_WING_AREA: float = 1.0  # m², 前翼面积估算
REAR_WING_AREA: float = 0.8   # m², 后翼面积估算

# ── 翼片级别范围 ──────────────────────────────────────────
FRONT_WING_MIN_LEVEL: int = 0
FRONT_WING_MAX_LEVEL: int = 10
REAR_WING_MIN_LEVEL: int = 0
REAR_WING_MAX_LEVEL: int = 11

# ── 升力系数范围 ──────────────────────────────────────────
CL_FRONT_MIN: float = 0.3
CL_FRONT_MAX: float = 2.0
CL_REAR_MIN: float = 0.2
CL_REAR_MAX: float = 1.6

# ── 阻力系数参数 ──────────────────────────────────────────
CD_BASE: float = 0.3          # 基础阻力系数
CD_LIFT_RATIO: float = 0.15   # 升力系数对阻力系数的贡献比

# ── 行驶高度乘数（地面效应）──────────────────────────────
RIDE_HEIGHT_LOW: float = 2.0     # 低行驶高度级别
RIDE_HEIGHT_MID: float = 4.0     # 中行驶高度级别（基准）
RIDE_HEIGHT_HIGH: float = 7.0    # 高行驶高度级别
RIDE_HEIGHT_MULT_LOW: float = 1.15   # 低行驶高度乘数
RIDE_HEIGHT_MULT_MID: float = 1.0    # 中行驶高度乘数
RIDE_HEIGHT_MULT_HIGH: float = 0.85  # 高行驶高度乘数


@dataclass
class AeroOutput:
    """空气动力学计算输出。

    所有力的单位为牛顿(N)，速度单位为m/s。
    """
    downforce_front: float   # N, 前轴下压力
    downforce_rear: float    # N, 后轴下压力
    downforce_total: float   # N, 总下压力
    drag_force: float        # N, 阻力
    aero_balance: float      # %, 前轴下压力占比 (0-100)
    cl_front: float          # 前翼下压力系数
    cl_rear: float           # 后翼下压力系数
    cd_total: float          # 总阻力系数


class AeroModel:
    """空气动力学模型。

    通过前翼和后翼的调教级别，计算给定速度下的下压力、阻力和空力平衡。
    """

    def __init__(self, front_wing: float, rear_wing: float) -> None:
        """初始化空气动力学模型。

        Args:
            front_wing: 前翼级别 (0-10)
            rear_wing: 后翼级别 (0-11)
        """
        self.front_wing: float = front_wing
        self.rear_wing: float = rear_wing

    def compute(
        self,
        speed_ms: float,
        ride_height_front: float = 4.0,
        ride_height_rear: float = 4.0,
    ) -> AeroOutput:
        """计算给定速度下的空气动力学输出。

        Args:
            speed_ms: 车速 (m/s)
            ride_height_front: 前轴行驶高度级别
            ride_height_rear: 后轴行驶高度级别

        Returns:
            AeroOutput 包含下压力、阻力、空力平衡等物理量
        """
        cl_front = self._calc_cl_front()
        cl_rear = self._calc_cl_rear()

        front_mult = self._ride_height_multiplier(ride_height_front)
        rear_mult = self._ride_height_multiplier(ride_height_rear)

        downforce_front = self._calc_downforce(cl_front * front_mult, speed_ms, FRONT_WING_AREA)
        downforce_rear = self._calc_downforce(cl_rear * rear_mult, speed_ms, REAR_WING_AREA)
        downforce_total = downforce_front + downforce_rear

        cd_total = self._calc_cd_total(cl_front, cl_rear)
        drag_force = self._calc_drag(cd_total, speed_ms)

        aero_balance = self._calc_aero_balance(downforce_front, downforce_total)

        return AeroOutput(
            downforce_front=downforce_front,
            downforce_rear=downforce_rear,
            downforce_total=downforce_total,
            drag_force=drag_force,
            aero_balance=aero_balance,
            cl_front=cl_front,
            cl_rear=cl_rear,
            cd_total=cd_total,
        )

    def _calc_cl_front(self) -> float:
        """计算前翼升力系数（线性映射 0-10 → 0.3-2.0）。"""
        level_range = FRONT_WING_MAX_LEVEL - FRONT_WING_MIN_LEVEL
        cl_range = CL_FRONT_MAX - CL_FRONT_MIN
        return CL_FRONT_MIN + (self.front_wing / level_range) * cl_range

    def _calc_cl_rear(self) -> float:
        """计算后翼升力系数（线性映射 0-11 → 0.2-1.6）。"""
        level_range = REAR_WING_MAX_LEVEL - REAR_WING_MIN_LEVEL
        cl_range = CL_REAR_MAX - CL_REAR_MIN
        return CL_REAR_MIN + (self.rear_wing / level_range) * cl_range

    def _ride_height_multiplier(self, ride_height: float) -> float:
        """计算行驶高度对下压力的乘数（地面效应插值）。

        在 RIDE_HEIGHT_LOW/MID/HIGH 三个锚点之间线性插值。
        低于 LOW 取 LOW 值，高于 HIGH 取 HIGH 值。
        """
        if ride_height <= RIDE_HEIGHT_LOW:
            return RIDE_HEIGHT_MULT_LOW
        if ride_height >= RIDE_HEIGHT_HIGH:
            return RIDE_HEIGHT_MULT_HIGH
        if ride_height <= RIDE_HEIGHT_MID:
            return self._lerp(
                RIDE_HEIGHT_LOW, RIDE_HEIGHT_MID,
                RIDE_HEIGHT_MULT_LOW, RIDE_HEIGHT_MULT_MID,
                ride_height,
            )
        return self._lerp(
            RIDE_HEIGHT_MID, RIDE_HEIGHT_HIGH,
            RIDE_HEIGHT_MULT_MID, RIDE_HEIGHT_MULT_HIGH,
            ride_height,
        )

    @staticmethod
    def _lerp(
        x0: float, x1: float, y0: float, y1: float, x: float,
    ) -> float:
        """线性插值：在 (x0,y0) 和 (x1,y1) 之间对 x 插值。"""
        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    @staticmethod
    def _calc_downforce(cl: float, speed_ms: float, area: float) -> float:
        """计算下压力 = CL * 0.5 * ρ * v² * A。"""
        return cl * 0.5 * RHO_AIR * speed_ms * speed_ms * area

    @staticmethod
    def _calc_cd_total(cl_front: float, cl_rear: float) -> float:
        """计算总阻力系数 CD ≈ (CL_front + CL_rear) * 0.15 + 0.3。"""
        return (cl_front + cl_rear) * CD_LIFT_RATIO + CD_BASE

    @staticmethod
    def _calc_drag(cd: float, speed_ms: float) -> float:
        """计算阻力 = CD * 0.5 * ρ * v² * (前翼面积 + 后翼面积)。"""
        total_area = FRONT_WING_AREA + REAR_WING_AREA
        return cd * 0.5 * RHO_AIR * speed_ms * speed_ms * total_area

    @staticmethod
    def _calc_aero_balance(downforce_front: float, downforce_total: float) -> float:
        """计算空力平衡 = 前轴下压力 / 总下压力 * 100%。"""
        if downforce_total == 0.0:
            return 0.0
        return downforce_front / downforce_total * 100.0