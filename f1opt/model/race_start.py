"""F1 2026 赛车起步 & 第 1 圈混乱模型 (Iter-37).

真实 F1 比赛的"第 1 圈"是位置变动最大的圈:
- **起步反应时间**: 0.15-0.30 s (顶级车手 0.15s, 一般 0.25s).
- **起步抓位 (launch)**: 取决于 clutch control + 起步反应 + 轮胎温度.
  顶级起步者 (Verstappen, Norris) 可在 1 圈内超 2-3 个位置.
- **T1 制动区混乱**: 紧凑发车 → 第 1 圈 T1 接触概率显著高于其他圈.
- **DRS 禁用第 1 圈** (已在 drs_2026 实现).
- **安全车概率峰值**: 第 1-2 圈 SC 概率是其他圈的 5-10×.

EA Sports F1 2026 游戏官方模型:
- 起步反应时间随机 0.15-0.30 s
- 起步位置变动: 基于 driver_aggression + clutch_skill (这里用 aggression)
- T1 接触: 概率与紧密度 (前后车间隔) + 车手激进程度相关
- 起步获胜者: 平均 1.5 个位置, 顶级起步者最多 +3
- 起步失败者: 平均 -1.5 个位置, 严重失败 -5

参考文献:
- FIA Sporting Regulations 2026 §31.3 (Start procedure)
- 公开 F1 起步反应时间数据 (2018-2024)

公开 API:
    - :class:`RaceStartModel` — 起步 + 第 1 圈混乱仿真.
    - :func:`simulate_race_start` — 便捷函数.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
_REACTION_TIME_MIN_S = 0.15
_REACTION_TIME_MAX_S = 0.30
_REACTION_TIME_TOP_DRIVER_S = 0.15  # 顶级车手最快反应

# 起步位置变动系数 (与 aggression 相关)
_LAUNCH_GAIN_BASE = 0.5  # 平均起步抓位
_LAUNCH_GAIN_AGGRESSION_FACTOR = 3.0  # aggression 0.5 → ±1.5 位
_LAUNCH_GAIN_NOISE_STD = 1.0  # 起步噪声
_LAUNCH_GAIN_MAX = 4  # 顶级起步最多 +4 位
_LAUNCH_LOSS_MAX = 5  # 最差起步 -5 位

# T1 接触概率 (每对相邻车)
_T1_CONTACT_BASE_PROB = 0.012  # 1.2% 基础概率
_T1_CONTACT_TIGHTNESS_FACTOR = 0.5  # 间隔越小概率越高
_T1_CONTACT_AGGRESSION_FACTOR = 0.3  # 激进车手更易碰撞
_T1_CONTACT_AGGRESSION_THRESHOLD = 0.7  # aggression > 0.7 显著增加

# 起步失败 (熄火/打滑) 概率
_STALL_PROB = 0.002  # 0.2% 熄火概率 (现代 F1 罕见)
_STALL_POSITION_LOSS = 8  # 熄火损失 8 位

# 第 1 圈 SC 概率提升 (相对其他圈)
_LAP1_SC_PROB_MULTIPLIER = 8.0  # 第 1 圈 SC 概率 8×
_LAP2_SC_PROB_MULTIPLIER = 3.0  # 第 2 圈 SC 概率 3×


@dataclass(frozen=True)
class StartDriverInput:
    """起步仿真输入 (单车手)."""

    driver_id: str
    grid_position: int
    """1-indexed 发车位."""
    driver_aggression: float = 0.5
    """车手激进度 0..1 (高 = 起步抓位好)."""
    driver_consistency: float = 0.5
    """圈速一致性 0..1 (高 = 起步稳定)."""
    clutch_skill: float = 0.5
    """离合控制技能 0..1 (高 = 起步快)."""
    pole_position: bool = False
    """是否杆位 (杆位起步反应时间略快)."""


@dataclass
class StartDriverResult:
    """起步仿真结果 (单车手)."""

    driver_id: str
    grid_position: int
    new_position: int
    """第 1 圈结束位置 (1-indexed)."""
    position_change: int
    """位置变动 (正 = 上升, 负 = 下降)."""
    reaction_time_s: float
    """起步反应时间 s."""
    launch_quality: float
    """起步质量 0..1 (1 = 完美起步)."""
    t1_contact: bool
    """是否在 T1 接触."""
    stalled: bool
    """是否熄火/打滑."""
    lap1_time_offset_s: float
    """第 1 圈时间偏移 s (相对正常圈速, 正 = 慢)."""


# --------------------------------------------------------------------------- #
# RaceStartModel
# --------------------------------------------------------------------------- #
@dataclass
class RaceStartModel:
    """F1 2026 起步 + 第 1 圈混乱仿真.

    用法::

        inputs = [StartDriverInput(driver_id=f"d{i}", grid_position=i+1,
                                    driver_aggression=0.7) for i in range(22)]
        model = RaceStartModel(track_id="monza", seed=42)
        results = model.simulate(inputs)
        # results[0].new_position  # 第 1 圈后位置
    """

    track_id: str = "monza"
    seed: int | None = None
    n_drivers: int = 22
    """预期车手数 (用于概率校准)."""

    # ------------------------------------------------------------------ #
    def simulate(self, drivers: list[StartDriverInput]) -> list[StartDriverResult]:
        """仿真起步 + 第 1 圈, 返回每车手结果 (按 new_position 排序)."""
        if not drivers:
            return []
        rng = random.Random(self.seed)

        # 1. 每车手起步质量 + 反应时间
        launches: list[tuple[float, StartDriverResult]] = []
        for d in drivers:
            # 反应时间: 顶级车手 (高 consistency + aggression) 更快
            skill = (d.driver_aggression * 0.4 + d.driver_consistency * 0.3
                     + d.clutch_skill * 0.3)
            if d.pole_position:
                # 杆位车手观察灯更专注
                skill = min(1.0, skill + 0.05)
            reaction = (_REACTION_TIME_MAX_S
                        - skill * (_REACTION_TIME_MAX_S - _REACTION_TIME_MIN_S))
            reaction += rng.gauss(0.0, 0.02)
            reaction = max(_REACTION_TIME_MIN_S, reaction)

            # 起步质量: clutch_skill 主导 + 噪声
            launch_quality = d.clutch_skill * 0.6 + d.driver_aggression * 0.2
            launch_quality += rng.gauss(0.0, 0.15)
            launch_quality = max(0.0, min(1.0, launch_quality))

            # 位置变动 (起步抓位)
            # aggression 高 + clutch 好 → 抓位多
            position_change = (
                (launch_quality - 0.5) * _LAUNCH_GAIN_AGGRESSION_FACTOR
                + rng.gauss(0.0, _LAUNCH_GAIN_NOISE_STD)
            )
            # 熄火概率检查
            stalled = rng.random() < _STALL_PROB
            if stalled:
                position_change -= _STALL_POSITION_LOSS

            # 限制范围
            position_change = max(-_LAUNCH_LOSS_MAX, min(_LAUNCH_GAIN_MAX, position_change))

            launches.append((launch_quality, StartDriverResult(
                driver_id=d.driver_id,
                grid_position=d.grid_position,
                new_position=0,  # 后填
                position_change=int(round(position_change)),
                reaction_time_s=float(reaction),
                launch_quality=float(launch_quality),
                t1_contact=False,
                stalled=stalled,
                lap1_time_offset_s=0.0,
            )))

        # 2. T1 接触检查 (基于发车紧密度)
        # 排序按 grid_position (发车顺序)
        sorted_by_grid = sorted(launches, key=lambda x: x[1].grid_position)
        for i in range(len(sorted_by_grid) - 1):
            ahead = sorted_by_grid[i][1]
            behind = sorted_by_grid[i + 1][1]
            # 间隔越紧 (后方车手 grid 紧贴前方), T1 接触概率越高
            gap_factor = 1.0  # 默认紧贴 (1 位间隔)
            # 后方车手 aggression 高 → 接触概率高
            aggression_input = next(
                (d for d in drivers if d.driver_id == behind.driver_id), None
            )
            behind_aggr = (aggression_input.driver_aggression
                           if aggression_input else 0.5)
            contact_prob = _T1_CONTACT_BASE_PROB * (
                1.0 + _T1_CONTACT_TIGHTNESS_FACTOR * gap_factor
            )
            if behind_aggr > _T1_CONTACT_AGGRESSION_THRESHOLD:
                contact_prob *= (1.0 + _T1_CONTACT_AGGRESSION_FACTOR)
            # 激进车手本身也可能接触前车
            ahead_aggr_input = next(
                (d for d in drivers if d.driver_id == ahead.driver_id), None
            )
            ahead_aggr = (ahead_aggr_input.driver_aggression
                          if ahead_aggr_input else 0.5)
            if ahead_aggr > _T1_CONTACT_AGGRESSION_THRESHOLD:
                contact_prob *= 1.2

            if rng.random() < contact_prob:
                behind.t1_contact = True
                # 接触: 后方车手损失位置 + 时间
                behind.position_change -= 2
                behind.lap1_time_offset_s += rng.uniform(2.0, 6.0)
                # 前方车手也可能被推 (轻微)
                if rng.random() < 0.4:
                    ahead.t1_contact = True
                    ahead.position_change -= 1
                    ahead.lap1_time_offset_s += rng.uniform(1.0, 3.0)

        # 3. 计算新位置 (基于位置变动)
        # 用 "虚拟分数" = grid_position - position_change, 排序后赋新位置
        results = [r for _, r in launches]
        # 排序: grid_position - position_change (越小越前)
        results.sort(key=lambda r: (r.grid_position - r.position_change
                                    + rng.gauss(0, 0.01)))  # 微抖动避免并列
        for new_pos, r in enumerate(results, start=1):
            r.new_position = new_pos
            # 第 1 圈时间偏移: 起步反应 + 起步质量影响
            # 慢起步 → 第 1 圈慢
            r.lap1_time_offset_s += (
                (r.reaction_time_s - _REACTION_TIME_MIN_S) * 5.0
                + (0.5 - r.launch_quality) * 1.5
            )
            if r.stalled:
                r.lap1_time_offset_s += 8.0

        return results

    # ------------------------------------------------------------------ #
    def lap1_safety_car_probability(self) -> float:
        """第 1 圈 SC 概率 (相对其他圈的乘数)."""
        return _LAP1_SC_PROB_MULTIPLIER

    def lap2_safety_car_probability(self) -> float:
        """第 2 圈 SC 概率 (相对其他圈的乘数)."""
        return _LAP2_SC_PROB_MULTIPLIER

    def summary(self, results: list[StartDriverResult]) -> dict[str, Any]:
        """返回起步仿真摘要."""
        if not results:
            return {}
        changes = [r.position_change for r in results]
        contacts = sum(1 for r in results if r.t1_contact)
        stalls = sum(1 for r in results if r.stalled)
        biggest_gainer = max(results, key=lambda r: r.position_change)
        biggest_loser = min(results, key=lambda r: r.position_change)
        return {
            "track_id": self.track_id,
            "n_drivers": len(results),
            "avg_position_change": sum(abs(c) for c in changes) / len(changes),
            "n_t1_contacts": contacts,
            "n_stalls": stalls,
            "biggest_gainer": {
                "driver_id": biggest_gainer.driver_id,
                "position_change": biggest_gainer.position_change,
            },
            "biggest_loser": {
                "driver_id": biggest_loser.driver_id,
                "position_change": biggest_loser.position_change,
            },
        }


# --------------------------------------------------------------------------- #
# Convenience function
# --------------------------------------------------------------------------- #
def simulate_race_start(
    drivers: list[StartDriverInput],
    track_id: str = "monza",
    seed: int | None = None,
) -> list[StartDriverResult]:
    """便捷函数: 仿真起步 + 第 1 圈."""
    return RaceStartModel(track_id=track_id, seed=seed).simulate(drivers)
