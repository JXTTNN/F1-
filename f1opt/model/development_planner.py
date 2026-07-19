"""F1 2026 — 车队赛季研发升级路径 (Iter-34).

车队在赛季中根据 Cost Cap + 赛道特点制定研发升级路径:

1. **赛季前**: 冬测完成基础研发 (aero_floor + powertrain).
2. **赛季中升级窗口**: 每场可带 1 个升级包 (受 Cost Cap 限制).
3. **赛道匹配**: 高速赛道前升级 aero_low_downforce, 高磨蚀前升级 mech_brakes.
4. **赛季末冲刺**: 后段车队若积分落后, 全力冲刺 powertrain.
5. **升级影响**: 升级提升赛车性能 (圈速增益 s/lap, 体现在 setup).

公开 API:
    - :class:`DevelopmentPlanner` — 赛季研发路径规划.
    - :func:`plan_season_development` — 便捷函数.
"""

from __future__ import annotations

from dataclasses import dataclass

from f1opt.data.tracks import ALL_TRACKS
from f1opt.model.cost_cap import (
    _UPGRADE_COSTS,
    _UPGRADE_GAINS,
    CostCapBudget,
    allocate_budget,
)

# 赛道特定升级优先级 (赛道类型 → 推荐升级)
_TRACK_UPGRADE_PRIORITY: dict[str, list[str]] = {
    "high_speed_low_downforce": ["aero_rear_wing", "aero_front_wing", "powertrain_ice"],
    "high_downforce": ["aero_floor", "aero_sidepod", "mech_suspension"],
    "street": ["mech_brakes", "mech_suspension", "aero_front_wing"],
    "medium": ["aero_floor", "powertrain_mguk", "mech_suspension"],
    "mixed": ["aero_floor", "aero_sidepod", "mech_suspension"],
}


@dataclass
class UpgradeEvent:
    """单次升级事件."""

    race_idx: int
    """0-indexed 赛季场次."""
    track_id: str
    upgrade_id: str
    cost_usd_m: float
    expected_gain_s: float
    cumulative_gain_s: float
    """累计性能增益 s/lap."""


@dataclass
class DevelopmentPlanner:
    """赛季研发路径规划器 (Iter-34).

    用法::

        planner = DevelopmentPlanner(
            team_id="red_bull",
            team_strength="balanced",
            initial_budget_m=60.0,
        )
        path = planner.plan_season()
        # path = [UpgradeEvent, ...]
    """

    team_id: str
    team_strength: str = "balanced"
    """车队强度: balanced/aero_focused/powertrain_focused/backmarker."""
    initial_budget_m: float = 60.0
    """初始研发预算 $M."""

    # ------------------------------------------------------------------ #
    def plan_season(self) -> list[UpgradeEvent]:
        """规划赛季升级路径."""
        budget = CostCapBudget(
            team_id=self.team_id,
            initial_budget_m=self.initial_budget_m,
        )
        # 基础升级顺序 (按车队强度)
        base_order = allocate_budget(self.team_strength)

        events: list[UpgradeEvent] = []
        cumulative_gain = 0.0

        for race_idx, track in enumerate(ALL_TRACKS):
            # 选升级: 赛道优先 + base_order 顺序
            upgrade_id = self._pick_upgrade(
                track.track_type, base_order, budget
            )
            if upgrade_id is None:
                continue

            cost = _UPGRADE_COSTS[upgrade_id]
            gain = _UPGRADE_GAINS[upgrade_id]
            if not budget.can_afford(upgrade_id):
                continue

            budget.spend(race_idx=race_idx, upgrade_id=upgrade_id)
            cumulative_gain += gain
            events.append(UpgradeEvent(
                race_idx=race_idx,
                track_id=track.track_id,
                upgrade_id=upgrade_id,
                cost_usd_m=cost,
                expected_gain_s=gain,
                cumulative_gain_s=cumulative_gain,
            ))

            if budget.remaining_budget_m < 1.0:
                break  # 预算耗尽

        return events

    # ------------------------------------------------------------------ #
    def _pick_upgrade(
        self,
        track_type: str,
        base_order: list[str],
        budget: CostCapBudget,
    ) -> str | None:
        """选择该场升级."""
        # 已升级项不再选
        upgraded = {uid for _, uid, _ in budget.spent_history}

        # 1. 赛道类型优先升级
        track_priority = _TRACK_UPGRADE_PRIORITY.get(track_type, [])
        for uid in track_priority:
            if uid in upgraded:
                continue
            if budget.can_afford(uid):
                return uid

        # 2. base_order 顺序
        for uid in base_order:
            if uid in upgraded:
                continue
            if budget.can_afford(uid):
                return uid

        return None

    # ------------------------------------------------------------------ #
    def summary(self) -> dict:
        events = self.plan_season()
        total_spent = sum(e.cost_usd_m for e in events)
        return {
            "team_id": self.team_id,
            "team_strength": self.team_strength,
            "n_upgrades": len(events),
            "total_spent_m": round(total_spent, 2),
            "remaining_budget_m": round(self.initial_budget_m - total_spent, 2),
            "total_gain_s": round(events[-1].cumulative_gain_s if events else 0.0, 3),
            "upgrade_path": [
                {
                    "race_idx": e.race_idx,
                    "track_id": e.track_id,
                    "upgrade_id": e.upgrade_id,
                    "cost_usd_m": e.cost_usd_m,
                    "gain_s": e.expected_gain_s,
                    "cumulative_gain_s": e.cumulative_gain_s,
                }
                for e in events
            ],
        }


def plan_season_development(
    team_id: str,
    team_strength: str = "balanced",
    initial_budget_m: float = 60.0,
) -> list[UpgradeEvent]:
    """便捷函数."""
    planner = DevelopmentPlanner(
        team_id=team_id,
        team_strength=team_strength,
        initial_budget_m=initial_budget_m,
    )
    return planner.plan_season()
