"""悬挂动力学模型 — 悬挂/防倾杆/行驶高度 → 重量转移.

物理原理:
    - 弹簧刚度: 从SUSPENSION_STIFFNESS字典按级别取值 (N/mm)
    - 防倾杆刚度: ARB_STIFFNESS_BASE + (level-1) * ARB_STIFFNESS_STEP (N·m/deg)
    - 行驶高度: 从RIDE_HEIGHT_MAP字典按级别取值 (mm)
    - 总侧倾刚度: 弹簧贡献(刚度*SPRING_ROLL_FACTOR) + 防倾杆刚度
    - 侧倾平衡: 前轴侧倾刚度 / 总侧倾刚度 * 100%
    - 重量转移: 总转移 = CAR_MASS * lateral_accel * G * CG_HEIGHT / TRACK_WIDTH
      按侧倾刚度比例分配到前后轴
    - 刮底风险: min(ride_height) < BOTTOMING_THRESHOLD时有风险
      risk = (BOTTOMING_THRESHOLD - min_height) / BOTTOMING_RANGE
    - 俯仰响应: (spring_rate_front + spring_rate_rear) / PITCH_NORMALIZATION
"""

from __future__ import annotations

from dataclasses import dataclass

G = 9.81                # m/s², 重力加速度
CAR_MASS = 798.0        # kg, F1赛车最小重量
TRACK_WIDTH = 2.0       # m, 轴距宽度
WHEELBASE = 3.6         # m, 轴距
CG_HEIGHT = 0.25        # m, 重心高度

# 悬挂级别 → 弹簧刚度 (N/mm), level 1=最软, 6=最硬
SUSPENSION_STIFFNESS: dict[int, float] = {
    1: 80.0, 2: 100.0, 3: 120.0, 4: 140.0, 5: 160.0, 6: 180.0,
}

# 防倾杆级别 → 侧倾刚度 (N·m/deg), level 1=最软, 11=最硬
ARB_STIFFNESS_BASE = 500.0    # level 1基础刚度
ARB_STIFFNESS_STEP = 200.0    # 每级增加量

# 行驶高度级别 → 离地间隙 (mm), level 2=最低, 7=最高
RIDE_HEIGHT_MAP: dict[int, float] = {
    2: 15.0, 3: 20.0, 4: 25.0, 5: 30.0, 6: 35.0, 7: 40.0,
}

# 侧倾刚度计算参数
SPRING_ROLL_FACTOR = 10.0     # 弹簧刚度→侧倾刚度转换系数

# 刮底风险参数
BOTTOMING_THRESHOLD = 25.0    # mm, 开始有刮底风险的离地间隙
BOTTOMING_RANGE = 15.0        # mm, 刮底风险归一化范围

# 俯仰响应参数
PITCH_NORMALIZATION = 360.0   # 弹簧刚度归一化分母


@dataclass
class SuspensionOutput:
    """悬挂模型计算输出."""
    spring_rate_front: float      # N/mm, 前弹簧刚度
    spring_rate_rear: float       # N/mm, 后弹簧刚度
    arb_stiffness_front: float    # N·m/deg, 前防倾杆刚度
    arb_stiffness_rear: float     # N·m/deg, 后防倾杆刚度
    ride_height_front: float      # mm, 前离地间隙
    ride_height_rear: float       # mm, 后离地间隙
    roll_stiffness_front: float   # N·m/deg, 前轴总侧倾刚度
    roll_stiffness_rear: float    # N·m/deg, 后轴总侧倾刚度
    roll_stiffness_total: float   # N·m/deg, 总侧倾刚度
    roll_balance: float           # %, 前轴侧倾刚度占比
    weight_transfer_front: float  # N, 前轴重量转移量
    weight_transfer_rear: float   # N, 后轴重量转移量
    bottoming_risk: float         # 0-1, 刮底风险
    pitch_response: float         # 0-1, 俯仰响应


class SuspensionModel:
    """悬挂动力学模型.

    根据前后悬挂级别、防倾杆级别、行驶高度级别初始化, 在给定加速度
    条件下计算侧倾刚度、重量转移、刮底风险和俯仰响应.
    """

    def __init__(self, front_suspension: float, rear_suspension: float,
                 front_anti_roll_bar: float, rear_anti_roll_bar: float,
                 front_ride_height: float, rear_ride_height: float) -> None:
        self.front_suspension = int(front_suspension)
        self.rear_suspension = int(rear_suspension)
        self.front_anti_roll_bar = int(front_anti_roll_bar)
        self.rear_anti_roll_bar = int(rear_anti_roll_bar)
        self.front_ride_height = int(front_ride_height)
        self.rear_ride_height = int(rear_ride_height)

    def compute(self, lateral_accel: float = 0.0,
                longitudinal_accel: float = 0.0) -> SuspensionOutput:
        """计算给定加速度下的悬挂输出.

        Args:
            lateral_accel: 侧向加速度 (g)
            longitudinal_accel: 纵向加速度 (g)

        Returns:
            SuspensionOutput 包含弹簧刚度、侧倾刚度、重量转移等
        """
        spring_f = self._get_spring_rate(self.front_suspension)
        spring_r = self._get_spring_rate(self.rear_suspension)
        arb_f = self._get_arb_stiffness(self.front_anti_roll_bar)
        arb_r = self._get_arb_stiffness(self.rear_anti_roll_bar)
        height_f = self._get_ride_height(self.front_ride_height)
        height_r = self._get_ride_height(self.rear_ride_height)

        roll_f = spring_f * SPRING_ROLL_FACTOR + arb_f
        roll_r = spring_r * SPRING_ROLL_FACTOR + arb_r
        roll_total = roll_f + roll_r
        roll_balance = roll_f / roll_total * 100.0 if roll_total > 0 else 0.0

        wt_front, wt_rear = self._calc_weight_transfer(
            lateral_accel, roll_f, roll_r, roll_total)
        bottoming_risk = self._calc_bottoming_risk(height_f, height_r)
        pitch = self._calc_pitch_response(spring_f, spring_r)

        return SuspensionOutput(
            spring_rate_front=spring_f, spring_rate_rear=spring_r,
            arb_stiffness_front=arb_f, arb_stiffness_rear=arb_r,
            ride_height_front=height_f, ride_height_rear=height_r,
            roll_stiffness_front=roll_f, roll_stiffness_rear=roll_r,
            roll_stiffness_total=roll_total, roll_balance=roll_balance,
            weight_transfer_front=wt_front, weight_transfer_rear=wt_rear,
            bottoming_risk=bottoming_risk, pitch_response=pitch,
        )

    def _get_spring_rate(self, level: int) -> float:
        """获取指定级别的弹簧刚度 (N/mm)."""
        return SUSPENSION_STIFFNESS.get(level, SUSPENSION_STIFFNESS[3])

    def _get_arb_stiffness(self, level: int) -> float:
        """获取指定级别的防倾杆刚度 (N·m/deg)."""
        return ARB_STIFFNESS_BASE + (level - 1) * ARB_STIFFNESS_STEP

    def _get_ride_height(self, level: int) -> float:
        """获取指定级别的离地间隙 (mm)."""
        return RIDE_HEIGHT_MAP.get(level, RIDE_HEIGHT_MAP[4])

    def _calc_weight_transfer(self, lateral_accel: float,
                              roll_f: float, roll_r: float,
                              roll_total: float) -> tuple[float, float]:
        """计算前后轴重量转移量 (N)."""
        total_transfer = (CAR_MASS * lateral_accel * G
                          * CG_HEIGHT / TRACK_WIDTH)
        if roll_total <= 0:
            return total_transfer / 2.0, total_transfer / 2.0
        wt_front = total_transfer * roll_f / roll_total
        wt_rear = total_transfer * roll_r / roll_total
        return wt_front, wt_rear

    def _calc_bottoming_risk(self, height_f: float,
                             height_r: float) -> float:
        """计算刮底风险 (0-1)."""
        min_height = min(height_f, height_r)
        if min_height >= BOTTOMING_THRESHOLD:
            return 0.0
        return (BOTTOMING_THRESHOLD - min_height) / BOTTOMING_RANGE

    def _calc_pitch_response(self, spring_f: float,
                             spring_r: float) -> float:
        """计算俯仰响应 (0-1)."""
        return (spring_f + spring_r) / PITCH_NORMALIZATION