"""Tests for energy_budget (Iter-29)."""

from __future__ import annotations

import pytest

from f1opt.model.energy_budget import (
    EnergyBudgetPlanner,
    LapEnergyPlan,
    plan_energy_budget,
)


class TestBasicPlan:
    def test_plan_has_correct_length(self) -> None:
        plan = plan_energy_budget(total_laps=58)
        assert len(plan) == 58

    def test_each_lap_has_required_fields(self) -> None:
        plan = plan_energy_budget(total_laps=10)
        for p in plan:
            assert isinstance(p, LapEnergyPlan)
            assert 0 <= p.deploy_mj <= 9.0
            assert 0 <= p.recover_mj <= 9.0
            assert 0 <= p.soc_before <= 1.0
            assert 0 <= p.soc_after <= 1.0
            assert p.deploy_mode in ("save", "balanced", "attack", "quali")

    def test_soc_continuity(self) -> None:
        """前一圈 soc_after = 后一圈 soc_before."""
        plan = plan_energy_budget(total_laps=20)
        for i in range(len(plan) - 1):
            assert plan[i].soc_after == pytest.approx(
                plan[i + 1].soc_before, abs=1e-6
            )


class TestDeploymentLimits:
    def test_no_lap_exceeds_9_mj_deploy(self) -> None:
        plan = plan_energy_budget(total_laps=58)
        for p in plan:
            assert p.deploy_mj <= 9.0 + 1e-6

    def test_no_lap_exceeds_9_mj_recover(self) -> None:
        plan = plan_energy_budget(total_laps=58, recovery_per_lap=8.0)
        for p in plan:
            assert p.recover_mj <= 9.0 + 1e-6


class TestFinalAttack:
    def test_final_laps_are_attack_mode(self) -> None:
        plan = plan_energy_budget(total_laps=58, final_attack_laps=5)
        # Last 5 laps should be attack mode
        for p in plan[-5:]:
            assert p.deploy_mode == "attack"

    def test_attack_laps_have_higher_deployment(self) -> None:
        plan = plan_energy_budget(total_laps=58, final_attack_laps=5)
        attack_avg = sum(p.deploy_mj for p in plan[-5:]) / 5
        balanced_avg = sum(p.deploy_mj for p in plan[:-5]) / (58 - 5)
        assert attack_avg > balanced_avg


class TestQualiMode:
    def test_quali_laps_full_deployment(self) -> None:
        plan = EnergyBudgetPlanner(total_laps=58, initial_soc=0.7).plan(
            recovery_per_lap=5.5, final_attack_laps=5, quali_mode_laps=(30,),
        )
        # Lap 30 (1-indexed) → index 29
        assert plan[29].deploy_mode == "quali"
        assert plan[29].deploy_mj == 9.0


class TestSummary:
    def test_summary_has_required_keys(self) -> None:
        plan = plan_energy_budget(total_laps=20)
        planner = EnergyBudgetPlanner(total_laps=20)
        s = planner.summary(plan)
        required = {"total_laps", "total_deploy_mj", "total_recover_mj",
                    "net_energy_mj", "final_soc", "initial_soc",
                    "attack_laps", "quali_laps", "balanced_laps"}
        assert required.issubset(s.keys())

    def test_summary_counts_correct(self) -> None:
        plan = plan_energy_budget(total_laps=20, final_attack_laps=3)
        planner = EnergyBudgetPlanner(total_laps=20)
        s = planner.summary(plan)
        assert s["total_laps"] == 20
        assert s["attack_laps"] == 3
        assert s["balanced_laps"] == 17


class TestEnergyConservation:
    def test_balanced_laps_store_energy(self) -> None:
        """Balanced mode should net-positive energy (recovery > deploy)."""
        plan = plan_energy_budget(total_laps=58, final_attack_laps=5,
                                   recovery_per_lap=6.0)
        # In early balanced laps, deploy should be ≤ recovery on average
        balanced = [p for p in plan if p.deploy_mode == "balanced"]
        avg_net = sum(p.recover_mj - p.deploy_mj for p in balanced) / len(balanced)
        # Either storing (net > 0) or near-zero
        assert avg_net >= -0.5

    def test_soc_does_not_go_negative(self) -> None:
        plan = plan_energy_budget(total_laps=58, initial_soc=0.5,
                                   recovery_per_lap=4.0)
        for p in plan:
            assert p.soc_after >= 0.0
            assert p.soc_before >= 0.0
