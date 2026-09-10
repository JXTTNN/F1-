"""F1 2026 — 车队研发预算上限 (Cost Cap) (Iter-33).

FIA 2026 财务规则: **Cost Cap** $135M/年 (含车手工资豁免), 用于限制
车队研发投入. EA F1 2026 还原此机制作为 "My Team" 模式核心.

主要影响:
1. **研发分配**: 车队需在 aero/mech/powertrain/driver 间分配预算.
2. **升级路径**: 每场可带 1 个升级包, 受预算限制.
3. **赛道特定**: 低成本赛道 (Bahrain 测试) 不需新升级.
4. **赛季中期**: 升级越多赛季后段越强, 但预算消耗大.

公开 API:
    - :class:`CostCapBudget` — 单车队预算管理.
    - :func:`allocate_budget` — 默认分配方案.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# FIA 2026 Cost Cap (USD million)
_TOTAL_BUDGET_USD_M = 135.0
_DRIVER_SALARY_EXEMPT_USD_M = 35.0  # 车手工资不计入
_DEVELOPMENT_BUDGET_USD_M = _TOTAL_BUDGET_USD_M - 75.0  # 运营 ~$75M
"""实际研发预算 ~$60M (135 - 75 运营)."""

# 单项升级成本 (USD M)
_UPGRADE_COSTS: dict[str, float] = {
    "aero_floor": 2.5,
    "aero_front_wing": 1.8,
    "aero_rear_wing": 1.5,
    "aero_sidepod": 2.2,
    "mech_suspension": 3.0,
    "mech_gearbox": 4.5,
    "mech_brakes": 2.0,
    "powertrain_ice": 5.5,
    "powertrain_mguk": 4.0,
    "powertrain_battery": 3.5,
    "driver_skill_development": 1.0,
}

# 性能增益 (s/lap, 满升级)
_UPGRADE_GAINS: dict[str, float] = {
    "aero_floor": 0.45,
    "aero_front_wing": 0.20,
    "aero_rear_wing": 0.15,
    "aero_sidepod": 0.30,
    "mech_suspension": 0.25,
    "mech_gearbox": 0.20,
    "mech_brakes": 0.10,
    "powertrain_ice": 0.35,
    "powertrain_mguk": 0.30,
    "powertrain_battery": 0.25,
    "driver_skill_development": 0.15,
}


@dataclass
class UpgradeDecision:
    """单次升级决策."""

    upgrade_id: str
    cost_usd_m: float
    expected_gain_s: float
    """预计圈速增益 s/lap."""
    cost_per_tenth_s: float
    """每 0.1s 圈速增益的成本 (USD M) — 效率指标."""


@dataclass
class CostCapBudget:
    """车队 Cost Cap 预算管理 (Iter-33).

    用法::

        budget = CostCapBudget(team_id="red_bull", initial_budget_m=60.0)
        d = budget.evaluate_upgrade("aero_floor")
        budget.spend("aero_floor")  # 实际支出
        print(budget.remaining_budget_m)
    """

    team_id: str
    initial_budget_m: float = _DEVELOPMENT_BUDGET_USD_M
    spent_history: list[tuple[int, str, float]] = field(default_factory=list)
    """(race_idx, upgrade_id, cost) 历史."""

    # ------------------------------------------------------------------ #
    @property
    def remaining_budget_m(self) -> float:
        return self.initial_budget_m - sum(c for _, _, c in self.spent_history)

    @property
    def total_spent_m(self) -> float:
        return sum(c for _, _, c in self.spent_history)

    # ------------------------------------------------------------------ #
    def evaluate_upgrade(self, upgrade_id: str) -> UpgradeDecision:
        """评估升级的成本效益."""
        if upgrade_id not in _UPGRADE_COSTS:
            raise ValueError(f"Unknown upgrade: {upgrade_id!r}")
        cost = _UPGRADE_COSTS[upgrade_id]
        gain = _UPGRADE_GAINS[upgrade_id]
        cost_per_tenth = cost / (gain * 10) if gain > 0 else float("inf")
        return UpgradeDecision(
            upgrade_id=upgrade_id,
            cost_usd_m=cost,
            expected_gain_s=gain,
            cost_per_tenth_s=cost_per_tenth,
        )

    # ------------------------------------------------------------------ #
    def can_afford(self, upgrade_id: str) -> bool:
        """是否负担得起."""
        cost = _UPGRADE_COSTS.get(upgrade_id, 0.0)
        return cost <= self.remaining_budget_m

    # ------------------------------------------------------------------ #
    def spend(self, race_idx: int, upgrade_id: str) -> bool:
        """支出升级费用. 返回是否成功."""
        if not self.can_afford(upgrade_id):
            return False
        cost = _UPGRADE_COSTS[upgrade_id]
        self.spent_history.append((race_idx, upgrade_id, cost))
        return True

    # ------------------------------------------------------------------ #
    def total_performance_gain_s(self) -> float:
        """累计性能增益 s/lap."""
        return sum(
            _UPGRADE_GAINS[uid] for _, uid, _ in self.spent_history
            if uid in _UPGRADE_GAINS
        )

    # ------------------------------------------------------------------ #
    def recommend_next_upgrade(self) -> UpgradeDecision | None:
        """推荐下次升级 (按 cost_per_tenth 排序, 取最优且负担得起)."""
        candidates: list[UpgradeDecision] = []
        # 已升级项不再考虑 (重复升级边际收益递减)
        upgraded = {uid for _, uid, _ in self.spent_history}
        for uid in _UPGRADE_COSTS:
            if uid in upgraded:
                continue
            if not self.can_afford(uid):
                continue
            candidates.append(self.evaluate_upgrade(uid))
        if not candidates:
            return None
        # 最优 cost_per_tenth
        return min(candidates, key=lambda x: x.cost_per_tenth_s)

    # ------------------------------------------------------------------ #
    def summary(self) -> dict:
        return {
            "team_id": self.team_id,
            "initial_budget_m": self.initial_budget_m,
            "total_spent_m": round(self.total_spent_m, 2),
            "remaining_budget_m": round(self.remaining_budget_m, 2),
            "n_upgrades": len(self.spent_history),
            "total_gain_s": round(self.total_performance_gain_s(), 3),
            "upgrades_done": [uid for _, uid, _ in self.spent_history],
        }


# --------------------------------------------------------------------------- #
# 默认预算分配方案
# --------------------------------------------------------------------------- #
def allocate_budget(team_strength: str = "balanced") -> list[str]:
    """按车队强度给出推荐升级顺序.

    Args:
        team_strength: "balanced" / "aero_focused" / "powertrain_focused" /
                       "backmarker" (后段车队, 抢速度).
    """
    if team_strength == "aero_focused":
        return ["aero_floor", "aero_sidepod", "aero_front_wing", "aero_rear_wing",
                "mech_suspension", "powertrain_mguk", "powertrain_ice",
                "mech_brakes", "powertrain_battery", "mech_gearbox",
                "driver_skill_development"]
    if team_strength == "powertrain_focused":
        return ["powertrain_ice", "powertrain_mguk", "powertrain_battery",
                "aero_floor", "mech_gearbox", "aero_sidepod", "aero_front_wing",
                "mech_suspension", "aero_rear_wing", "mech_brakes",
                "driver_skill_development"]
    if team_strength == "backmarker":
        # 后段车队抢早期速度: 便宜高效升级优先
        return ["aero_front_wing", "aero_rear_wing", "mech_brakes",
                "driver_skill_development", "aero_floor", "mech_suspension",
                "powertrain_mguk", "aero_sidepod", "powertrain_battery",
                "mech_gearbox", "powertrain_ice"]
    # balanced
    return ["aero_floor", "powertrain_mguk", "mech_suspension", "aero_sidepod",
            "powertrain_ice", "aero_front_wing", "powertrain_battery",
            "mech_gearbox", "aero_rear_wing", "mech_brakes",
            "driver_skill_development"]
