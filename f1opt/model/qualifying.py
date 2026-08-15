"""F1 三阶段排位赛仿真 (Iter-20).

FIA 真实排位赛格式 (Sporting Regulations 33.1-33.5):

- **Q1**: 18 分钟, 全部 22 车手 → 淘汰最慢 6 名 (P17-P22).
- **Q2**: 15 分钟, 剩余 16 车手 → 淘汰最慢 6 名 (P11-P16).
- **Q3**: 12 分钟, 前 10 车手 → 决定 P1-P10.

每阶段车手需跑 2-3 个计时圈 (out-lap + flying + in-lap).
新胎在 Q2/Q3 提供 grip 优势 (Pirelli 软胎 Q2 强制规则已废除, 但
车队通常在 Q2 用硬胎以正赛首段轮胎起步, 即 "alternate strategy").

每个阶段的圈速基于:
- 车手单圈潜力 (skill + aggression/smoothness/consistency)
- 赛车潜力 (setup-based via surrogate model)
- 轮胎 (新软胎 vs 旧胎)
- 燃油量 (轻 = 快)
- 随机扰动 (车手失误, 交通, 黄旗)
- 排位赛专用调教 (Q3 车队走更激进 setup)

公开 API:
    - :class:`QualifyingSession` — 三阶段排位赛仿真.
    - :func:`simulate_qualifying` — 便捷函数.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from f1opt.data.setup_schema import CarSetup
from f1opt.model.surrogate import track_prior

# --------------------------------------------------------------------------- #
# FIA 规则常量
# --------------------------------------------------------------------------- #
_Q1_DURATION_MIN = 18.0
_Q2_DURATION_MIN = 15.0
_Q3_DURATION_MIN = 12.0

_Q1_ELIMINATE = 6   # P17-P22 eliminated in Q1
_Q2_ELIMINATE = 6   # P11-P16 eliminated in Q2

# 新软胎相对基础圈速的提升 (Pirelli 软胎 vs 中性胎)
_TIRE_GAIN_SOFT_S = 0.45
_TIRE_GAIN_MEDIUM_S = 0.0
_TIRE_GAIN_HARD_S = -0.35  # hard 略慢
# 用过的胎 (磨损 > 30%) 损失
_USED_TIRE_PENALTY_S = 0.15

# 燃油量惩罚 (1 kg = 0.030 s, 起步 110 kg, Q1 末约 95 kg, Q3 末约 80 kg)
_FUEL_PENALTY_S_PER_KG = 0.030
_Q1_FUEL_KG = 100.0
_Q2_FUEL_KG = 90.0
_Q3_FUEL_KG = 80.0

# 单阶段最多跑的飞驰圈数 (out-lap/flying/in-lap 占时间)
_MAX_FLYING_LAPS_Q1 = 7
_MAX_FLYING_LAPS_Q2 = 6
_MAX_FLYING_LAPS_Q3 = 5

# 车手失误概率 (每圈)
_MISTAKE_PROB_Q1 = 0.06
_MISTAKE_PROB_Q2 = 0.05
_MISTAKE_PROB_Q3 = 0.04
_MISTAKE_PENALTY_S_RANGE = (0.2, 1.5)  # 小失误到大失误

# 交通阻塞概率 (后车在飞驰圈遇到前车慢车)
_TRAFFIC_PROB_Q1 = 0.12
_TRAFFIC_PROB_Q2 = 0.08
_TRAFFIC_PROB_Q3 = 0.03
_TRAFFIC_PENALTY_S_RANGE = (0.1, 0.6)


# --------------------------------------------------------------------------- #
# DriverQualifyingInput
# --------------------------------------------------------------------------- #
@dataclass
class DriverQualifyingInput:
    """车手排位赛输入."""

    driver_id: str
    driver_name: str
    team_id: str
    setup: CarSetup
    skill: float
    """0..1, 综合实力 (consistency/smoothness/aggression 加权)."""
    aggression: float = 0.7
    smoothness: float = 0.7
    consistency: float = 0.7
    q3_setup_boost: float = 0.0
    """车队在 Q3 走更激进 setup 的圈速提升 s (0 = 不变)."""


# --------------------------------------------------------------------------- #
# QualifyingResult
# --------------------------------------------------------------------------- #
@dataclass
class QualifyingLap:
    """单次排位赛计时圈."""

    phase: str  # "Q1" / "Q2" / "Q3"
    lap_time_s: float
    fuel_kg: float
    tire_compound: str
    tire_age_laps: int
    mistake: bool
    traffic: bool


@dataclass
class DriverQualifyingResult:
    """单车手排位赛结果."""

    driver_id: str
    driver_name: str
    team_id: str
    grid_position: int
    """最终发车位 1..22."""
    best_lap_time_s: float | None
    """该车手最佳圈速 (None = 未出圈, 但应极少发生)."""
    q1_laps: list[QualifyingLap] = field(default_factory=list)
    q2_laps: list[QualifyingLap] = field(default_factory=list)
    q3_laps: list[QualifyingLap] = field(default_factory=list)
    eliminated_in: str = "Q3"
    """被淘汰阶段 ("Q1"/"Q2"/"Q3-完成")."""
    q2_tire_for_race: str = "medium"
    """正赛首段使用的轮胎 (Q2 通过者必须用其 Q2 最快圈所用胎起跑)."""


# --------------------------------------------------------------------------- #
# QualifyingSession
# --------------------------------------------------------------------------- #
@dataclass
class QualifyingSession:
    """F1 三阶段排位赛仿真.

    用法::

        sess = QualifyingSession(track_id="monza", drivers=inputs, seed=42)
        grid = sess.run()  # list[DriverQualifyingResult] 按 grid_position 排序
    """

    track_id: str
    drivers: list[DriverQualifyingInput]
    seed: int | None = None

    # ------------------------------------------------------------------ #
    def run(self) -> list[DriverQualifyingResult]:
        """运行三阶段排位赛, 返回按发车位排序的结果列表."""
        rng = random.Random(self.seed)
        # 赛道基础圈速 (来自 surrogate prior)
        base_lap = track_prior(self.track_id, self.drivers[0].setup)

        # 所有车手初始结果
        results: dict[str, DriverQualifyingResult] = {
            d.driver_id: DriverQualifyingResult(
                driver_id=d.driver_id,
                driver_name=d.driver_name,
                team_id=d.team_id,
                grid_position=0,
                best_lap_time_s=None,
            )
            for d in self.drivers
        }

        # === Q1: 全部 22 车手, 淘汰 6 ===
        q1_standings = self._run_phase(
            phase="Q1",
            drivers=self.drivers,
            base_lap=base_lap,
            fuel_kg=_Q1_FUEL_KG,
            max_laps=_MAX_FLYING_LAPS_Q1,
            mistake_prob=_MISTAKE_PROB_Q1,
            traffic_prob=_TRAFFIC_PROB_Q1,
            rng=rng,
            results=results,
        )
        # Q1 后 6 名 → P17-P22
        q1_bottom_6 = [d_id for d_id, _ in q1_standings[-_Q1_ELIMINATE:]]
        for rank_offset, d_id in enumerate(q1_bottom_6):
            results[d_id].grid_position = 22 - rank_offset
            results[d_id].eliminated_in = "Q1"

        # === Q2: 前 16 车手, 淘汰 6 ===
        q2_drivers_ids = [d_id for d_id, _ in q1_standings[:16]]
        q2_driver_inputs = [d for d in self.drivers if d.driver_id in q2_drivers_ids]
        # Q2 用不同胎 (Pirelli soft/medium), Q2 最快圈所用胎 = 正赛首段胎
        q2_standings = self._run_phase(
            phase="Q2",
            drivers=q2_driver_inputs,
            base_lap=base_lap,
            fuel_kg=_Q2_FUEL_KG,
            max_laps=_MAX_FLYING_LAPS_Q2,
            mistake_prob=_MISTAKE_PROB_Q2,
            traffic_prob=_TRAFFIC_PROB_Q2,
            rng=rng,
            results=results,
            record_q2_tire=True,
        )
        q2_bottom_6 = [d_id for d_id, _ in q2_standings[-_Q2_ELIMINATE:]]
        for rank_offset, d_id in enumerate(q2_bottom_6):
            results[d_id].grid_position = 16 - rank_offset
            results[d_id].eliminated_in = "Q2"

        # === Q3: 前 10 车手 ===
        q3_drivers_ids = [d_id for d_id, _ in q2_standings[:10]]
        q3_driver_inputs = [d for d in self.drivers if d.driver_id in q3_drivers_ids]
        q3_standings = self._run_phase(
            phase="Q3",
            drivers=q3_driver_inputs,
            base_lap=base_lap,
            fuel_kg=_Q3_FUEL_KG,
            max_laps=_MAX_FLYING_LAPS_Q3,
            mistake_prob=_MISTAKE_PROB_Q3,
            traffic_prob=_TRAFFIC_PROB_Q3,
            rng=rng,
            results=results,
            is_q3=True,
        )
        # Q3 排名直接对应 P1-P10
        for rank, (d_id, _) in enumerate(q3_standings):
            results[d_id].grid_position = rank + 1
            results[d_id].eliminated_in = "Q3-complete"

        # 返回按 grid_position 排序
        all_results = list(results.values())
        all_results.sort(key=lambda r: r.grid_position)
        return all_results

    # ------------------------------------------------------------------ #
    def _run_phase(
        self,
        phase: str,
        drivers: list[DriverQualifyingInput],
        base_lap: float,
        fuel_kg: float,
        max_laps: int,
        mistake_prob: float,
        traffic_prob: float,
        rng: random.Random,
        results: dict[str, DriverQualifyingResult],
        record_q2_tire: bool = False,
        is_q3: bool = False,
    ) -> list[tuple[str, float]]:
        """仿真单个阶段, 返回 [(driver_id, best_lap), ...] 按快到慢排序."""
        phase_laps: list[tuple[str, float, str]] = []  # (driver_id, best_lap, tire)

        for d in drivers:
            # 车手决定跑几圈 (1 to max_laps, 偏向 2-3)
            n_laps = rng.randint(2, max(2, max_laps - 2))

            # Q3 走更激进 setup, 给圈速提升
            setup_boost = d.q3_setup_boost if is_q3 else 0.0
            # Q2/Q3 通常用新软胎; Q1 可能用中性胎省胎
            if phase == "Q1":
                tire = "medium"
            else:
                tire = "soft"
            tire_age = 0 if phase in ("Q2", "Q3") else rng.randint(0, 1)

            best_lap: float | None = None
            best_tire_used = tire
            for lap_idx in range(n_laps):
                # 单圈圈速
                lap_time = self._compute_lap_time(
                    base_lap=base_lap,
                    skill=d.skill,
                    smoothness=d.smoothness,
                    aggression=d.aggression,
                    consistency=d.consistency,
                    fuel_kg=fuel_kg - lap_idx * 1.2,
                    tire_compound=tire,
                    tire_age_laps=tire_age + lap_idx,
                    setup_boost_s=setup_boost,
                    mistake_prob=mistake_prob,
                    traffic_prob=traffic_prob,
                    rng=rng,
                )
                # 记录圈
                mistake = rng.random() < mistake_prob
                traffic = rng.random() < traffic_prob
                # 若出失误, 上面 lap_time 已含惩罚
                ql = QualifyingLap(
                    phase=phase,
                    lap_time_s=lap_time,
                    fuel_kg=max(60.0, fuel_kg - lap_idx * 1.2),
                    tire_compound=tire,
                    tire_age_laps=tire_age + lap_idx,
                    mistake=mistake,
                    traffic=traffic,
                )
                if phase == "Q1":
                    results[d.driver_id].q1_laps.append(ql)
                elif phase == "Q2":
                    results[d.driver_id].q2_laps.append(ql)
                else:
                    results[d.driver_id].q3_laps.append(ql)

                if best_lap is None or lap_time < best_lap:
                    best_lap = lap_time
                    best_tire_used = tire

            if best_lap is None:
                # 没出圈 — 给一个很慢的圈速保证被淘汰
                best_lap = base_lap + 5.0
                best_tire_used = tire

            phase_laps.append((d.driver_id, best_lap, best_tire_used))

            # 更新该车手历史最佳
            cur = results[d.driver_id]
            if cur.best_lap_time_s is None or best_lap < cur.best_lap_time_s:
                cur.best_lap_time_s = best_lap

            if record_q2_tire:
                cur.q2_tire_for_race = best_tire_used

        # 按快到慢排序
        phase_laps.sort(key=lambda x: x[1])
        return [(d_id, lap) for d_id, lap, _ in phase_laps]

    # ------------------------------------------------------------------ #
    def _compute_lap_time(
        self,
        base_lap: float,
        skill: float,
        smoothness: float,
        aggression: float,
        consistency: float,
        fuel_kg: float,
        tire_compound: str,
        tire_age_laps: int,
        setup_boost_s: float,
        mistake_prob: float,
        traffic_prob: float,
        rng: random.Random,
    ) -> float:
        """计算单圈圈速 (排位赛专用)."""
        # 基础圈速 + 车手 skill (skill 1.0 = -1.5s, skill 0.0 = +0.5s)
        skill_offset = (0.5 - skill) * 2.0  # ±1.0 s
        # smoothness 偏移
        smooth_offset = max(0.0, 0.5 - smoothness) * 0.5
        # consistency 偏移 (低 consistency = 圈速不稳, 慢)
        cons_offset = max(0.0, 0.5 - consistency) * 0.3
        # aggression: 排位赛激进略快 (但风险高)
        aggr_offset = (0.5 - aggression) * 0.4  # ±0.2 s

        # 轮胎
        if tire_compound == "soft":
            tire_gain = _TIRE_GAIN_SOFT_S
        elif tire_compound == "hard":
            tire_gain = _TIRE_GAIN_HARD_S
        else:
            tire_gain = _TIRE_GAIN_MEDIUM_S
        if tire_age_laps > 0:
            tire_gain -= _USED_TIRE_PENALTY_S * tire_age_laps

        # 燃油 (相对 80 kg 基准, 100 kg 慢 0.6s)
        fuel_offset = (fuel_kg - 80.0) * _FUEL_PENALTY_S_PER_KG

        # 失误
        mistake_offset = 0.0
        if rng.random() < mistake_prob:
            mistake_offset = rng.uniform(*_MISTAKE_PENALTY_S_RANGE)
        # 交通
        traffic_offset = 0.0
        if rng.random() < traffic_prob:
            traffic_offset = rng.uniform(*_TRAFFIC_PENALTY_S_RANGE)

        # 随机噪声 (车手每次飞驰圈的圈速抖动)
        noise = rng.gauss(0.0, 0.12)

        lap_time = (
            base_lap
            + skill_offset
            + smooth_offset
            + cons_offset
            + aggr_offset
            - tire_gain        # 正 gain = 圈速减少
            + fuel_offset
            + mistake_offset
            + traffic_offset
            + noise
            - setup_boost_s    # Q3 激进 setup 提升 (s)
        )
        # 物理边界
        return max(60.0, min(180.0, float(lap_time)))


# --------------------------------------------------------------------------- #
# Convenience function
# --------------------------------------------------------------------------- #
def simulate_qualifying(
    track_id: str,
    drivers: list[DriverQualifyingInput],
    seed: int | None = None,
) -> list[DriverQualifyingResult]:
    """便捷函数: 运行三阶段排位赛."""
    sess = QualifyingSession(track_id=track_id, drivers=drivers, seed=seed)
    return sess.run()
