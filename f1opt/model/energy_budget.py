"""F1 2026 — 车手能量预算管理 (Iter-29).

FIA 2026 引入 **Power Unit Interface (PUI)** 让车手手动管理电能量:

1. **每圈 9 MJ 部署上限** + **9 MJ 回收上限** (FIA 强制).
2. **车手 UI 显示**: SoC %, 部署剩余 MJ, 回收潜力.
3. **节省模式**: 车手在前段圈保存能量, 末段用于冲刺.
4. **冲刺模式**: 末段 5 圈全力部署, 但需提前储备.
5. **风险/收益**: 部署多 = 圈速快但电池亏, 后段圈速掉.

公开 API:
    - :class:`EnergyBudgetPlanner` — 整场比赛能量预算规划.
    - :func:`plan_energy_budget` — 便捷函数.
"""

from __future__ import annotations

from dataclasses import dataclass

_MAX_DEPLOY_MJ_PER_LAP = 9.0
_MAX_RECOVER_MJ_PER_LAP = 9.0
_BATTERY_CAPACITY_MJ = 9.0
_DEFAULT_RECOVERY_PER_LAP = 5.5  # 平均每圈回收 5.5 MJ (制动 + MGU-K)


@dataclass
class LapEnergyPlan:
    """单圈能量计划."""

    lap: int
    deploy_mj: float
    recover_mj: float
    soc_before: float
    soc_after: float
    deploy_mode: str
    """该圈部署模式: save/balanced/attack/quali."""
    rationale: str


@dataclass
class EnergyBudgetPlanner:
    """F1 2026 整场比赛能量预算规划 (Iter-29).

    用法::

        planner = EnergyBudgetPlanner(total_laps=58, initial_soc=0.7)
        plan = planner.plan(recovery_per_lap=5.5, final_attack_laps=5)
        # plan = [LapEnergyPlan, ...]
    """

    total_laps: int
    initial_soc: float = 0.7
    """初始电池 SoC 0..1."""

    # ------------------------------------------------------------------ #
    def plan(
        self,
        recovery_per_lap: float = _DEFAULT_RECOVERY_PER_LAP,
        final_attack_laps: int = 5,
        quali_mode_laps: tuple[int, ...] = (),
    ) -> list[LapEnergyPlan]:
        """生成整场比赛能量部署计划.

        Args:
            recovery_per_lap: 平均每圈回收 MJ.
            final_attack_laps: 末段全力冲刺圈数.
            quali_mode_laps: 圈号 (1-indexed) 列表, 这些圈用 quali 模式
                (e.g. 排位赛, 或正赛关键超车圈).
        """
        plan: list[LapEnergyPlan] = []
        soc = self.initial_soc

        for lap_idx in range(1, self.total_laps + 1):
            soc_before = soc
            is_final_attack = lap_idx > self.total_laps - final_attack_laps
            is_quali = lap_idx in quali_mode_laps

            if is_quali:
                deploy_mode = "quali"
                deploy = _MAX_DEPLOY_MJ_PER_LAP  # 100% 部署
                rationale = "Qualifying-style full deployment"
            elif is_final_attack:
                deploy_mode = "attack"
                # 末段冲刺: 平均剩余能量 / 剩余圈数, 但每圈不超过 9
                remaining_laps = self.total_laps - lap_idx + 1
                available = soc * _BATTERY_CAPACITY_MJ + recovery_per_lap * remaining_laps
                deploy = min(_MAX_DEPLOY_MJ_PER_LAP,
                             available / max(1, remaining_laps) * 1.5)
                rationale = f"Final attack ({remaining_laps} laps left)"
            else:
                # 平衡: 每圈回收 - 部署, SoC 缓慢上升为末段冲刺储备
                # 目标: 前 (total - final_attack) 圈储备 SoC ~ 0.85
                target_soc = 0.85
                remaining_to_attack = self.total_laps - final_attack_laps - lap_idx + 1
                if remaining_to_attack > 0:
                    needed = (target_soc - soc) * _BATTERY_CAPACITY_MJ
                    # 部署 = 回收 - 储备
                    deploy = max(0.0, recovery_per_lap - needed / remaining_to_attack)
                else:
                    deploy = recovery_per_lap  # 平衡
                deploy = min(deploy, _MAX_DEPLOY_MJ_PER_LAP)
                deploy_mode = "balanced"
                rationale = "Balanced deployment, storing energy for final attack"

            # 限制: 部署不能超过 SoC + 回收
            available_this_lap = soc * _BATTERY_CAPACITY_MJ + recovery_per_lap
            deploy = min(deploy, available_this_lap, _MAX_DEPLOY_MJ_PER_LAP)
            deploy = max(0.0, deploy)

            # 更新 SoC
            net = recovery_per_lap - deploy
            soc_change = net / _BATTERY_CAPACITY_MJ
            soc = max(0.0, min(1.0, soc + soc_change))

            plan.append(LapEnergyPlan(
                lap=lap_idx,
                deploy_mj=deploy,
                recover_mj=recovery_per_lap,
                soc_before=soc_before,
                soc_after=soc,
                deploy_mode=deploy_mode,
                rationale=rationale,
            ))

        return plan

    # ------------------------------------------------------------------ #
    def summary(self, plan: list[LapEnergyPlan]) -> dict:
        """生成预算摘要."""
        if not plan:
            return {}
        total_deploy = sum(p.deploy_mj for p in plan)
        total_recover = sum(p.recover_mj for p in plan)
        return {
            "total_laps": len(plan),
            "total_deploy_mj": total_deploy,
            "total_recover_mj": total_recover,
            "net_energy_mj": total_recover - total_deploy,
            "final_soc": plan[-1].soc_after,
            "initial_soc": plan[0].soc_before,
            "attack_laps": sum(1 for p in plan if p.deploy_mode == "attack"),
            "quali_laps": sum(1 for p in plan if p.deploy_mode == "quali"),
            "balanced_laps": sum(1 for p in plan if p.deploy_mode == "balanced"),
        }


def plan_energy_budget(
    total_laps: int,
    initial_soc: float = 0.7,
    recovery_per_lap: float = _DEFAULT_RECOVERY_PER_LAP,
    final_attack_laps: int = 5,
) -> list[LapEnergyPlan]:
    """便捷函数."""
    planner = EnergyBudgetPlanner(total_laps=total_laps, initial_soc=initial_soc)
    return planner.plan(recovery_per_lap=recovery_per_lap,
                        final_attack_laps=final_attack_laps)
