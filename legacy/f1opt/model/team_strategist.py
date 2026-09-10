"""F1 车队级双层策略决策 (Iter-24).

真实 F1 车队 (Red Bull, Mercedes, Ferrari, McLaren) 每场比赛为两辆赛车
制定互补策略 — "cover the field" — 以应对不同比赛情景:

1. **两车均前列 (前 5)**: 分策略 — 一车激进 (soft-soft-hard), 一车保守
   (medium-hard). 保护 SC 早期/晚期两种情景.
2. **两车均中游 (6-12)**: 同策略 — 维持赛道位置, 防御对手 undercut.
3. **一车前一车后 (split)**: 后车走激进风险策略 (soft 起步, 早 undercut),
   前车走保守策略稳住位置.
4. **两车均后段 (13+)**: 分策略最大化一车机会 — 一车 soft-low-downforce
   抢早期位置, 一车 medium-hard 等待 SC.

公开 API:
    - :class:`TeamStrategist` — 双车策略决策.
    - :func:`decide_team_strategies` — 便捷函数.
"""

from __future__ import annotations

from dataclasses import dataclass

from f1opt.data.setup_schema import CarSetup
from f1opt.model.race_simulator import RaceStrategy
from f1opt.model.strategy_optimizer import (
    optimize_strategy,
)
from f1opt.model.weather import WeatherModel

# --------------------------------------------------------------------------- #
# 预定义策略模板
# --------------------------------------------------------------------------- #
# 激进: soft-soft-hard, 早进站 (短 stint 利用 soft 速度)
_AGGRESSIVE_2STOP = RaceStrategy(
    pit_laps=(12, 32), compounds=("soft", "soft", "hard")
)
# 保守: medium-hard, 中段单进站 (长 stint 利用 hard 耐久)
_CONSERVATIVE_1STOP = RaceStrategy(
    pit_laps=(30,), compounds=("medium", "hard")
)
# 均衡: medium-hard-medium, 双进站
_BALANCED_2STOP = RaceStrategy(
    pit_laps=(22, 42), compounds=("medium", "hard", "medium")
)
# 极端激进: soft-medium-soft-hard, 三次进站
_EXTREME_3STOP = RaceStrategy(
    pit_laps=(10, 28, 45), compounds=("soft", "medium", "soft", "hard")
)
# 后车翻盘: soft 起步 + 早 undercut
_UNDERCUT_2STOP = RaceStrategy(
    pit_laps=(10, 35), compounds=("soft", "hard", "medium")
)


@dataclass
class TeamStrategyDecision:
    """车队双层策略决策."""

    car1_strategy: RaceStrategy
    car2_strategy: RaceStrategy
    car1_role: str
    """车 1 角色: "aggressive" / "conservative" / "balanced" / "undercut"."""
    car2_role: str
    rationale: str
    """决策原因 (人类可读)."""


# --------------------------------------------------------------------------- #
# TeamStrategist
# --------------------------------------------------------------------------- #
@dataclass
class TeamStrategist:
    """车队级双层策略决策器 (Iter-24).

    用法::

        strategist = TeamStrategist(track_id="monza", total_laps=58,
                                     pit_loss_s=23.0)
        decision = strategist.decide(
            car1_grid=2, car2_grid=5,
            car1_skill=0.85, car2_skill=0.78,
            track_type="high_speed_low_downforce",
            forecast_wet=False,
        )
        print(decision.car1_strategy, decision.car2_strategy)
    """

    track_id: str
    total_laps: int
    pit_loss_s: float = 23.0

    # ------------------------------------------------------------------ #
    def decide(
        self,
        car1_grid: int,
        car2_grid: int,
        car1_skill: float = 0.75,
        car2_skill: float = 0.75,
        track_type: str = "medium",
        forecast_wet: bool = False,
        is_sprint: bool = False,
    ) -> TeamStrategyDecision:
        """为车队两车制定互补策略.

        Args:
            car1_grid, car2_grid: 两车发车位 (1-20).
            car1_skill, car2_skill: 两车综合实力 (0-1).
            track_type: 赛道类型 (high_speed_low_downforce/street/etc).
            forecast_wet: 是否预报雨天.
            is_sprint: 是否 Sprint 周末 (短赛 → 单进站).
        """
        # 湿地: 都走 wet 策略 (没分策略意义)
        if forecast_wet:
            return TeamStrategyDecision(
                car1_strategy=RaceStrategy(
                    pit_laps=(20,), compounds=("intermediate", "wet")
                ),
                car2_strategy=RaceStrategy(
                    pit_laps=(20,), compounds=("intermediate", "wet")
                ),
                car1_role="balanced",
                car2_role="balanced",
                rationale="Wet forecast: both cars on intermediate-wet",
            )

        # Sprint: 短赛 (~100km), 单进站或无进站
        if is_sprint:
            return TeamStrategyDecision(
                car1_strategy=_CONSERVATIVE_1STOP,
                car2_strategy=_CONSERVATIVE_1STOP,
                car1_role="balanced",
                car2_role="balanced",
                rationale="Sprint: both cars on conservative 1-stop",
            )

        # 谁是前车 (更低 grid_position)
        if car1_grid <= car2_grid:
            front_grid, back_grid = car1_grid, car2_grid
            front_idx, back_idx = 1, 2
        else:
            front_grid, back_grid = car2_grid, car1_grid
            front_idx, back_idx = 2, 1

        # === 情景 1: 两车均前 5 ===
        if front_grid <= 5 and back_grid <= 5:
            # 分策略: 前车保守 (守住位置), 后车激进 (尝试超越)
            front_strat = _CONSERVATIVE_1STOP
            back_strat = _AGGRESSIVE_2STOP
            rationale = (
                f"Both top-5: split strategy — P{front_grid} conservative "
                f"(hold position), P{back_grid} aggressive (attack)"
            )
        # === 情景 2: 两车均 6-12 (中游) ===
        elif 6 <= front_grid <= 12 and 6 <= back_grid <= 12:
            # 同策略: 维持赛道位置
            front_strat = _BALANCED_2STOP
            back_strat = _BALANCED_2STOP
            rationale = (
                f"Both mid-field (P{front_grid}, P{back_grid}): same balanced "
                f"strategy to maintain track position"
            )
        # === 情景 3: split (前车前 8, 后车后 8) ===
        elif front_grid <= 8 and back_grid >= 13:
            # 后车走激进 undercut
            front_strat = _CONSERVATIVE_1STOP
            back_strat = _UNDERCUT_2STOP
            rationale = (
                f"Split grid: P{front_grid} conservative, P{back_grid} "
                f"undercut (early soft pit for track position)"
            )
        # === 情景 4: 两车均后段 (13+) ===
        elif front_grid >= 13:
            # 极端分策略最大化一车机会
            front_strat = _BALANCED_2STOP
            back_strat = _EXTREME_3STOP
            rationale = (
                f"Both back-markers (P{front_grid}, P{back_grid}): split "
                f"aggressively to maximize one car's chance"
            )
        # === 默认: 均衡 2-stop ===
        else:
            front_strat = _BALANCED_2STOP
            back_strat = _BALANCED_2STOP
            rationale = (
                f"Default balanced 2-stop for P{front_grid}, P{back_grid}"
            )

        # 根据车手 skill 微调: 高 skill 车手可走激进
        # 如果后车 skill 高, 让后车走更激进
        if back_idx == 1 and car1_skill > car2_skill + 0.1:
            # 实际后车是 car1, 但 skill 高 → 可以更激进
            back_strat = _AGGRESSIVE_2STOP if back_strat is _BALANCED_2STOP else back_strat
            rationale += "; back car higher-skill → more aggressive"

        # 分配策略到 car1/car2
        if front_idx == 1:
            return TeamStrategyDecision(
                car1_strategy=front_strat,
                car2_strategy=back_strat,
                car1_role=_role_for(front_strat),
                car2_role=_role_for(back_strat),
                rationale=rationale,
            )
        else:
            return TeamStrategyDecision(
                car1_strategy=back_strat,
                car2_strategy=front_strat,
                car1_role=_role_for(back_strat),
                car2_role=_role_for(front_strat),
                rationale=rationale,
            )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _role_for(strat: RaceStrategy) -> str:
    """从策略推断角色."""
    if strat is _AGGRESSIVE_2STOP:
        return "aggressive"
    if strat is _CONSERVATIVE_1STOP:
        return "conservative"
    if strat is _EXTREME_3STOP:
        return "extreme"
    if strat is _UNDERCUT_2STOP:
        return "undercut"
    return "balanced"


def decide_team_strategies(
    track_id: str,
    total_laps: int,
    car1_grid: int,
    car2_grid: int,
    **kwargs,
) -> TeamStrategyDecision:
    """便捷函数."""
    s = TeamStrategist(track_id=track_id, total_laps=total_laps)
    return s.decide(car1_grid=car1_grid, car2_grid=car2_grid, **kwargs)


def optimize_team_strategies(
    setup1: CarSetup,
    setup2: CarSetup,
    track_id: str,
    total_laps: int,
    car1_grid: int,
    car2_grid: int,
    weather: WeatherModel | None = None,
) -> tuple[RaceStrategy, RaceStrategy]:
    """优化两车策略 — 用 StrategyOptimizer 找各自最优, 然后强制差异化.

    车队 wall 实际做法: 先算每车最优策略, 若两车策略相同且都属于前列,
    强制一车改为次优策略以分险.
    """
    # 各自最优
    opt1 = optimize_strategy(setup1, track_id, total_laps, weather=weather)
    opt2 = optimize_strategy(setup2, track_id, total_laps, weather=weather)

    s1 = RaceStrategy(pit_laps=opt1.pit_laps, compounds=opt1.compounds)
    s2 = RaceStrategy(pit_laps=opt2.pit_laps, compounds=opt2.compounds)

    # 如果两车策略相同 + 前列, 强制分化
    if (s1.pit_laps == s2.pit_laps
            and s1.compounds == s2.compounds
            and car1_grid <= 5 and car2_grid <= 5):
        # 后车改为激进策略
        if car1_grid < car2_grid:
            s2 = _AGGRESSIVE_2STOP
        else:
            s1 = _AGGRESSIVE_2STOP

    return s1, s2
