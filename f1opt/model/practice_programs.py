"""F1 2026 自由练习赛项目模型 (Iter-45).

EA Sports F1 2026 自由练习赛 (FP1/FP2/FP3) 中, 车队执行多个"练习项目"
(Practice Programs) 收集数据并验证调校. 每个项目有:

- **目标**: 圈速 delta、圈数、燃油预算等.
- **执行**: 车手按项目跑指定圈数, 系统记录圈速/磨损/燃油.
- **评分**: 成功 / 部分成功 / 失败.
- **反馈**: 推荐调校方向 (aero 平衡、刹车偏置、轮胎压力等).

EA F1 2026 标准练习项目:
- **AP** (Acclimatization Program): 适应赛道, 3 圈在 +2.0s delta 内.
- **AER** (Aero Testing): 空气动力学测试, 评估下压力/阻力平衡.
- **RT** (Race Trim): 长跑, 8-12 圈模拟正赛节奏.
- **QS** (Qualifying Simulation): 推飞圈, 单圈在 -0.5s delta 内.
- **TS** (Tire Strategy): 测试不同化合物, 评估磨损.
- **FM** (Fuel Management): 燃油管理, 节油模式跑圈.
- **SV** (Setup Verification): 调校验证, 综合反馈.

项目成功度影响:
- R&D 点数奖励 (EA F1 2026: 每成功项目 +5-15 R&D 点).
- 车手资源分数 (resource points).
- 正赛 setup 精度 (失败项目 → setup 不确定度增加).

公开 API:
    - :class:`PracticeProgram` — 单项目定义.
    - :class:`ProgramResult` — 项目执行结果.
    - :class:`PracticeSessionSimulator` — 练习赛仿真器.
    - :func:`simulate_practice_program` — 便捷函数.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum


# --------------------------------------------------------------------------- #
# 枚举
# --------------------------------------------------------------------------- #
class ProgramType(Enum):
    """EA F1 2026 练习项目类型."""

    ACCLIMATIZATION = "AP"       # 适应赛道
    AERO_TESTING = "AER"         # 空气动力学测试
    RACE_TRIM = "RT"             # 长跑 (正赛节奏)
    QUALIFYING_SIM = "QS"        # 排位模拟
    TIRE_STRATEGY = "TS"         # 轮胎策略
    FUEL_MANAGEMENT = "FM"       # 燃油管理
    SETUP_VERIFICATION = "SV"    # 调校验证


class ProgramOutcome(Enum):
    """项目结果."""

    SUCCESS = "success"          # 达成目标
    PARTIAL = "partial"          # 部分达成
    FAILED = "failed"            # 未达成
    ABORTED = "aborted"          # 中止 (事故/机械问题)


# --------------------------------------------------------------------------- #
# 项目目标参数 (EA F1 2026 标准)
# --------------------------------------------------------------------------- #
# 每个项目: (目标圈数, delta_to_reference_s, 容差_s)
# delta_to_reference: 负 = 需比参考快, 正 = 允许慢
_PROGRAM_TARGETS: dict[ProgramType, tuple[int, float, float]] = {
    ProgramType.ACCLIMATIZATION: (3, 2.0, 0.5),    # 3 圈, +2.0s 内, ±0.5 容差
    ProgramType.AERO_TESTING: (5, 1.0, 0.3),        # 5 圈, +1.0s 内
    ProgramType.RACE_TRIM: (10, 1.5, 0.4),          # 10 圈长跑, +1.5s 内
    ProgramType.QUALIFYING_SIM: (3, -0.3, 0.2),     # 3 推飞圈, 需比参考快 0.3s
    ProgramType.TIRE_STRATEGY: (8, 2.0, 0.5),       # 8 圈, 评估磨损
    ProgramType.FUEL_MANAGEMENT: (6, 1.8, 0.4),     # 6 圈节油, +1.8s 内
    ProgramType.SETUP_VERIFICATION: (5, 0.8, 0.3),  # 5 圈综合, +0.8s 内
}

# R&D 点数奖励 (EA F1 2026: 每项目成功奖励)
_RD_POINTS_REWARD: dict[ProgramType, int] = {
    ProgramType.ACCLIMATIZATION: 5,
    ProgramType.AERO_TESTING: 10,
    ProgramType.RACE_TRIM: 15,
    ProgramType.QUALIFYING_SIM: 12,
    ProgramType.TIRE_STRATEGY: 12,
    ProgramType.FUEL_MANAGEMENT: 8,
    ProgramType.SETUP_VERIFICATION: 10,
}


# --------------------------------------------------------------------------- #
# 数据类
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PracticeProgram:
    """单个练习项目定义.

    - ``target_laps``: 目标圈数.
    - ``delta_target_s``: 相对参考圈速的目标 delta (s). 负 = 需更快.
    - ``tolerance_s``: 容差 — delta 在容差内算"部分成功".
    - ``rd_points``: 成功时奖励的 R&D 点数.
    """

    program_type: ProgramType
    target_laps: int
    delta_target_s: float
    tolerance_s: float
    rd_points: int
    compound: str = "medium"
    fuel_mode: str = "normal"  # normal / lean / rich

    @classmethod
    def standard(cls, ptype: ProgramType, compound: str = "medium") -> PracticeProgram:
        """创建标准 EA F1 2026 项目."""
        laps, delta, tol = _PROGRAM_TARGETS[ptype]
        return cls(
            program_type=ptype,
            target_laps=laps,
            delta_target_s=delta,
            tolerance_s=tol,
            rd_points=_RD_POINTS_REWARD[ptype],
            compound=compound,
            fuel_mode="lean" if ptype == ProgramType.FUEL_MANAGEMENT else "normal",
        )


@dataclass
class ProgramResult:
    """项目执行结果."""

    program_type: ProgramType
    outcome: ProgramOutcome
    completed_laps: int
    avg_delta_s: float              # 平均圈速 delta vs 参考
    best_delta_s: float             # 最佳圈 delta
    tire_wear_pct: float            # 累计轮胎磨损 %
    fuel_used_kg: float             # 燃油消耗
    rd_points_earned: int           # 实际获得 R&D 点
    feedback: list[str] = field(default_factory=list)
    """车队反馈 (调校建议等)."""

    @property
    def is_success(self) -> bool:
        return self.outcome == ProgramOutcome.SUCCESS


# --------------------------------------------------------------------------- #
# PracticeSessionSimulator
# --------------------------------------------------------------------------- #
class PracticeSessionSimulator:
    """练习赛项目仿真器.

    模拟车手执行一个练习项目, 综合考虑车手能力、调校、疲劳、轮胎.

    用法::

        sim = PracticeSessionSimulator(
            driver_pace=0.85, driver_consistency=0.80,
            setup_quality=0.70, track_id="monza", seed=42,
        )
        prog = PracticeProgram.standard(ProgramType.QUALIFYING_SIM)
        result = sim.run_program(prog)
        if result.is_success:
            print(f"+{result.rd_points_earned} R&D 点")
    """

    def __init__(
        self,
        driver_pace: float,           # 0..1, 越高越快
        driver_consistency: float,    # 0..1
        setup_quality: float,         # 0..1, 当前调校与最优的匹配度
        track_id: str,
        seed: int | None = None,
        reference_lap_time_s: float = 90.0,  # 参考圈速 (基准)
        track_difficulty: float = 0.5,       # 0..1, Monaco 高, Monza 低
    ) -> None:
        self.driver_pace = max(0.0, min(1.0, driver_pace))
        self.driver_consistency = max(0.0, min(1.0, driver_consistency))
        self.setup_quality = max(0.0, min(1.0, setup_quality))
        self.track_id = track_id
        self.rng = random.Random(seed)
        self.reference_lap_time_s = float(reference_lap_time_s)
        self.track_difficulty = max(0.0, min(1.0, track_difficulty))

    # ------------------------------------------------------------------ #
    def run_program(self, program: PracticeProgram) -> ProgramResult:
        """执行一个练习项目, 返回结果."""
        # 单圈 delta 估计: 车手能力 + 调校 + 项目类型 + 噪声
        laps: list[float] = []  # 每圈 delta
        tire_wear_total = 0.0
        fuel_total = 0.0
        aborted = False

        # 项目类型对圈速 delta 的基础偏移
        type_offset = self._type_delta_offset(program.program_type)

        for lap_idx in range(program.target_laps):
            # 疲劳: 每圈累积微小惩罚
            fatigue_pen = lap_idx * 0.02 * self.track_difficulty

            # 车手能力 delta: pace 越高 delta 越负 (越快)
            pace_delta = (1.0 - self.driver_pace) * 1.5  # 0..1.5s 慢

            # 调校质量 delta: 0..1, 1=最优, 差 0.8s
            setup_delta = (1.0 - self.setup_quality) * 0.8

            # 一致性噪声: 一致性低 → 噪声大
            noise = self.rng.gauss(0.0, (1.0 - self.driver_consistency) * 0.3)

            # 燃油模式影响
            fuel_mode_delta = 0.0
            if program.fuel_mode == "lean":
                fuel_mode_delta = 0.3  # 节油慢一点
            elif program.fuel_mode == "rich":
                fuel_mode_delta = -0.2  # 富油快一点

            # 轮胎磨损 (随圈数增加)
            wear_per_lap = 1.5 + lap_idx * 0.1
            tire_wear_total += wear_per_lap
            wear_delta = lap_idx * 0.03  # 磨损导致变慢

            # 燃油消耗 (kg/lap)
            fuel_per_lap = 1.6
            if program.fuel_mode == "lean":
                fuel_per_lap = 1.4
            elif program.fuel_mode == "rich":
                fuel_per_lap = 1.8
            fuel_total += fuel_per_lap

            # 事故概率 (低, 但调校差 + 高难度赛道更高)
            accident_prob = 0.002 + (1.0 - self.setup_quality) * 0.003 * self.track_difficulty
            if self.rng.random() < accident_prob:
                aborted = True
                break

            delta = (type_offset + pace_delta + setup_delta + fatigue_pen
                     + fuel_mode_delta + wear_delta + noise)
            laps.append(delta)

        # 评估结果
        if aborted or len(laps) == 0:
            return ProgramResult(
                program_type=program.program_type,
                outcome=ProgramOutcome.ABORTED,
                completed_laps=len(laps),
                avg_delta_s=999.0,
                best_delta_s=999.0,
                tire_wear_pct=tire_wear_total,
                fuel_used_kg=fuel_total,
                rd_points_earned=0,
                feedback=["项目中止 — 检查车手状态与调校稳定性"],
            )

        avg_delta = sum(laps) / len(laps)
        best_delta = min(laps)

        # 评分: delta <= target → success; |delta - target| <= tol → partial
        target = program.delta_target_s
        tol = program.tolerance_s
        if avg_delta <= target:
            outcome = ProgramOutcome.SUCCESS
            rd = program.rd_points
        elif avg_delta <= target + tol:
            outcome = ProgramOutcome.PARTIAL
            rd = program.rd_points // 2
        else:
            outcome = ProgramOutcome.FAILED
            rd = 0

        # 完成圈数不足也算失败/部分
        if len(laps) < program.target_laps * 0.7:
            outcome = ProgramOutcome.FAILED
            rd = 0

        # 反馈
        feedback = self._generate_feedback(program, avg_delta, outcome, tire_wear_total)

        return ProgramResult(
            program_type=program.program_type,
            outcome=outcome,
            completed_laps=len(laps),
            avg_delta_s=avg_delta,
            best_delta_s=best_delta,
            tire_wear_pct=tire_wear_total,
            fuel_used_kg=fuel_total,
            rd_points_earned=rd,
            feedback=feedback,
        )

    # ------------------------------------------------------------------ #
    def _type_delta_offset(self, ptype: ProgramType) -> float:
        """项目类型对圈速的基础偏移 (s)."""
        # QS 是推飞圈, 应该接近极限; RT 是正赛节奏, 较保守
        if ptype == ProgramType.QUALIFYING_SIM:
            return -0.3  # 推飞, 比 reference 快
        if ptype == ProgramType.RACE_TRIM:
            return 0.8   # 保守节奏
        if ptype == ProgramType.FUEL_MANAGEMENT:
            return 0.6   # 节油慢
        if ptype == ProgramType.ACCLIMATIZATION:
            return 1.2   # 适应阶段慢
        if ptype == ProgramType.AERO_TESTING:
            return 0.5
        if ptype == ProgramType.TIRE_STRATEGY:
            return 0.7
        return 0.5  # SV

    # ------------------------------------------------------------------ #
    def _generate_feedback(
        self,
        program: PracticeProgram,
        avg_delta: float,
        outcome: ProgramOutcome,
        tire_wear: float,
    ) -> list[str]:
        """生成车队反馈."""
        fb: list[str] = []
        if outcome == ProgramOutcome.SUCCESS:
            fb.append("项目达成目标, 数据收集完成.")
        elif outcome == ProgramOutcome.PARTIAL:
            fb.append("部分达成, 建议下一节调整调校后再试.")
        else:
            fb.append("未达目标, 需大幅调整调校或车手节奏.")

        # 圈速反馈
        if avg_delta > 1.0:
            fb.append("圈速偏慢 — 检查下压力等级与刹车偏置.")
        elif avg_delta < -0.2:
            fb.append("圈速优秀 — 当前调校接近最优.")

        # 轮胎反馈
        if tire_wear > 15.0:
            fb.append("轮胎磨损偏高 — 建议提高胎压或减少滑动.")
        elif tire_wear < 5.0:
            fb.append("轮胎工作温度不足 — 建议降低胎压或更激进驾驶.")

        # 项目特定反馈
        if program.program_type == ProgramType.QUALIFYING_SIM and outcome != ProgramOutcome.SUCCESS:
            fb.append("排位模拟未达标 — 检查 DRS 激活点与燃油配比.")
        if program.program_type == ProgramType.FUEL_MANAGEMENT:
            fb.append("燃油消耗记录完成, 用于正赛节油策略规划.")
        if program.program_type == ProgramType.TIRE_STRATEGY:
            fb.append("轮胎数据已收集, 用于进站策略优化.")

        return fb


# --------------------------------------------------------------------------- #
# Convenience function
# --------------------------------------------------------------------------- #
def simulate_practice_program(
    program_type: ProgramType,
    driver_pace: float,
    driver_consistency: float,
    setup_quality: float,
    track_id: str,
    seed: int | None = None,
    compound: str = "medium",
) -> ProgramResult:
    """便捷函数: 仿真单个练习项目."""
    prog = PracticeProgram.standard(program_type, compound=compound)
    sim = PracticeSessionSimulator(
        driver_pace=driver_pace,
        driver_consistency=driver_consistency,
        setup_quality=setup_quality,
        track_id=track_id,
        seed=seed,
    )
    return sim.run_program(prog)
