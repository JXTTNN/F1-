"""F1 2026 R&D 升级树模型测试 (Iter-51)."""

from __future__ import annotations

import pytest

from f1opt.model.rnd_tree import (
    RnDBranch,
    RnDTree,
    TeamRnDState,
    apply_upgrades_to_team,
    nodes_per_branch,
    total_branches,
    total_nodes,
)


# --------------------------------------------------------------------------- #
# RnDBranch 枚举
# --------------------------------------------------------------------------- #
def test_branch_values():
    assert RnDBranch.AERODYNAMICS.value == "aero"
    assert RnDBranch.POWER_UNIT.value == "pu"
    assert RnDBranch.CHASSIS.value == "chassis"
    assert RnDBranch.DURABILITY.value == "durability"


def test_four_branches():
    assert total_branches() == 4


# --------------------------------------------------------------------------- #
# RnDTree 基础
# --------------------------------------------------------------------------- #
def test_tree_has_nodes():
    tree = RnDTree()
    assert len(tree.all_nodes()) == 20  # 4 分支 × 5 节点


def test_total_nodes():
    assert total_nodes() == 20


def test_nodes_per_branch():
    npb = nodes_per_branch()
    assert npb["aero"] == 5
    assert npb["pu"] == 5
    assert npb["chassis"] == 5
    assert npb["durability"] == 5


def test_get_node():
    tree = RnDTree()
    node = tree.get_node("aero_1")
    assert node.node_id == "aero_1"
    assert node.branch == RnDBranch.AERODYNAMICS


def test_get_node_unknown_raises():
    tree = RnDTree()
    with pytest.raises(ValueError):
        tree.get_node("nonexistent")


def test_nodes_in_branch():
    tree = RnDTree()
    aero_nodes = tree.nodes_in_branch(RnDBranch.AERODYNAMICS)
    assert len(aero_nodes) == 5
    for n in aero_nodes:
        assert n.branch == RnDBranch.AERODYNAMICS


def test_first_node_in_branch():
    tree = RnDTree()
    first = tree.first_node_in_branch(RnDBranch.POWER_UNIT)
    assert first.node_id == "pu_1"


# --------------------------------------------------------------------------- #
# 前置依赖
# --------------------------------------------------------------------------- #
def test_first_node_no_prereq():
    tree = RnDTree()
    for branch in RnDBranch:
        first = tree.first_node_in_branch(branch)
        assert first.prereq_id is None


def test_can_unlock_first_node():
    tree = RnDTree()
    assert tree.can_unlock("aero_1", set()) is True


def test_cannot_unlock_second_without_first():
    tree = RnDTree()
    assert tree.can_unlock("aero_2", set()) is False


def test_can_unlock_second_with_first():
    tree = RnDTree()
    assert tree.can_unlock("aero_2", {"aero_1"}) is True


def test_next_unlockable_returns_first():
    tree = RnDTree()
    n = tree.next_unlockable(RnDBranch.AERODYNAMICS, set())
    assert n is not None
    assert n.node_id == "aero_1"


def test_next_unlockable_after_first():
    tree = RnDTree()
    n = tree.next_unlockable(RnDBranch.AERODYNAMICS, {"aero_1"})
    assert n is not None
    assert n.node_id == "aero_2"


def test_next_unlockable_none_when_all_done():
    tree = RnDTree()
    all_aero = {f"aero_{i}" for i in range(1, 6)}
    n = tree.next_unlockable(RnDBranch.AERODYNAMICS, all_aero)
    assert n is None


# --------------------------------------------------------------------------- #
# 升级节点属性
# --------------------------------------------------------------------------- #
def test_node_has_cost():
    tree = RnDTree()
    node = tree.get_node("aero_1")
    assert node.cost_rp > 0


def test_node_has_effects():
    tree = RnDTree()
    node = tree.get_node("aero_1")
    assert "aero_efficiency" in node.effects
    assert node.effects["aero_efficiency"] > 0


def test_node_costs_increase_with_level():
    """越高级节点越贵."""
    tree = RnDTree()
    costs = [tree.get_node(f"aero_{i}").cost_rp for i in range(1, 6)]
    for i in range(4):
        assert costs[i] < costs[i + 1]


def test_node_weeks_increase_with_level():
    """越高级节点耗时越长."""
    tree = RnDTree()
    weeks = [tree.get_node(f"pu_{i}").weeks_to_complete for i in range(1, 6)]
    for i in range(4):
        assert weeks[i] <= weeks[i + 1]


# --------------------------------------------------------------------------- #
# TeamRnDState 基础
# --------------------------------------------------------------------------- #
def test_state_initial():
    s = TeamRnDState(team_id="rbr")
    assert s.resource_points == 1000
    assert len(s.completed_nodes) == 0
    assert len(s.in_progress) == 0
    assert s.total_invested_rp == 0


def test_add_resource_points():
    s = TeamRnDState(team_id="rbr")
    s.add_resource_points(100)
    assert s.resource_points == 1100


def test_add_negative_points_raises():
    s = TeamRnDState(team_id="rbr")
    with pytest.raises(ValueError):
        s.add_resource_points(-10)


# --------------------------------------------------------------------------- #
# start_upgrade
# --------------------------------------------------------------------------- #
def test_start_upgrade_success():
    tree = RnDTree()
    s = TeamRnDState(team_id="rbr", resource_points=500)
    assert s.start_upgrade(tree, "aero_1") is True
    assert s.resource_points == 300  # 500 - 200
    assert "aero_1" in s.in_progress
    assert s.total_invested_rp == 200


def test_start_upgrade_insufficient_points():
    tree = RnDTree()
    s = TeamRnDState(team_id="rbr", resource_points=100)
    assert s.start_upgrade(tree, "aero_1") is False
    assert s.resource_points == 100  # 未扣


def test_start_upgrade_prereq_not_met():
    tree = RnDTree()
    s = TeamRnDState(team_id="rbr", resource_points=10000)
    assert s.start_upgrade(tree, "aero_2") is False


def test_start_upgrade_already_completed():
    tree = RnDTree()
    s = TeamRnDState(team_id="rbr", resource_points=10000,
                     completed_nodes={"aero_1"})
    assert s.start_upgrade(tree, "aero_1") is False


def test_start_upgrade_already_in_progress():
    tree = RnDTree()
    s = TeamRnDState(team_id="rbr", resource_points=10000)
    s.start_upgrade(tree, "aero_1")
    assert s.start_upgrade(tree, "aero_1") is False


# --------------------------------------------------------------------------- #
# advance_week
# --------------------------------------------------------------------------- #
def test_advance_week_decrements():
    tree = RnDTree()
    s = TeamRnDState(team_id="rbr", resource_points=10000)
    s.start_upgrade(tree, "aero_1")  # 1 周
    completed = s.advance_week(tree)
    assert "aero_1" in completed
    assert "aero_1" in s.completed_nodes
    assert "aero_1" not in s.in_progress


def test_advance_week_multiweek():
    tree = RnDTree()
    s = TeamRnDState(team_id="rbr", resource_points=10000,
                     completed_nodes={"aero_1"})  # 先完成前置
    s.start_upgrade(tree, "aero_2")  # 2 周
    completed = s.advance_week(tree)
    assert completed == []  # 第 1 周未完成
    assert "aero_2" in s.in_progress
    assert s.in_progress["aero_2"] == 1
    completed = s.advance_week(tree)
    assert "aero_2" in completed


def test_advance_week_multiple_nodes():
    tree = RnDTree()
    s = TeamRnDState(team_id="rbr", resource_points=10000)
    s.start_upgrade(tree, "aero_1")
    s.start_upgrade(tree, "pu_1")
    s.start_upgrade(tree, "cha_1")
    completed = s.advance_week(tree)
    # 三个都是 1 周, 应全部完成
    assert len(completed) == 3


def test_advance_week_nothing_in_progress():
    tree = RnDTree()
    s = TeamRnDState(team_id="rbr")
    completed = s.advance_week(tree)
    assert completed == []


# --------------------------------------------------------------------------- #
# total_effects
# --------------------------------------------------------------------------- #
def test_total_effects_empty():
    tree = RnDTree()
    s = TeamRnDState(team_id="rbr")
    assert s.total_effects(tree) == {}


def test_total_effects_single_node():
    tree = RnDTree()
    s = TeamRnDState(team_id="rbr", completed_nodes={"aero_1"})
    effects = s.total_effects(tree)
    assert "aero_efficiency" in effects
    assert effects["aero_efficiency"] > 0


def test_total_effects_accumulate():
    tree = RnDTree()
    s = TeamRnDState(team_id="rbr", completed_nodes={"aero_1", "aero_2"})
    effects = s.total_effects(tree)
    e1 = tree.get_node("aero_1").effects["aero_efficiency"]
    e2 = tree.get_node("aero_2").effects["aero_efficiency"]
    assert abs(effects["aero_efficiency"] - (e1 + e2)) < 1e-9


def test_total_effects_negative_for_tire_deg():
    """底盘升级应降低 tire_degradation_factor (负值)."""
    tree = RnDTree()
    s = TeamRnDState(team_id="rbr", completed_nodes={"cha_1"})
    effects = s.total_effects(tree)
    assert effects["tire_degradation_factor"] < 0


# --------------------------------------------------------------------------- #
# branch_progress
# --------------------------------------------------------------------------- #
def test_branch_progress_empty():
    tree = RnDTree()
    s = TeamRnDState(team_id="rbr")
    done, total = s.branch_progress(tree, RnDBranch.AERODYNAMICS)
    assert done == 0
    assert total == 5


def test_branch_progress_partial():
    tree = RnDTree()
    s = TeamRnDState(team_id="rbr", completed_nodes={"aero_1", "aero_2"})
    done, total = s.branch_progress(tree, RnDBranch.AERODYNAMICS)
    assert done == 2
    assert total == 5


def test_branch_progress_full():
    tree = RnDTree()
    all_aero = {f"aero_{i}" for i in range(1, 6)}
    s = TeamRnDState(team_id="rbr", completed_nodes=all_aero)
    done, total = s.branch_progress(tree, RnDBranch.AERODYNAMICS)
    assert done == 5
    assert total == 5


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def test_summary_structure():
    tree = RnDTree()
    s = TeamRnDState(team_id="rbr", resource_points=500,
                     completed_nodes={"aero_1"})
    summary = s.summary(tree)
    assert summary["team_id"] == "rbr"
    assert summary["resource_points"] == 500
    assert summary["completed"] == 1
    assert "branches" in summary
    assert "effects" in summary


# --------------------------------------------------------------------------- #
# apply_upgrades_to_team
# --------------------------------------------------------------------------- #
def test_apply_upgrades_returns_effects():
    tree = RnDTree()
    s = TeamRnDState(team_id="rbr", completed_nodes={"aero_1", "pu_1"})
    effects = apply_upgrades_to_team(s, tree)
    assert "aero_efficiency" in effects
    assert "power_unit_kW" in effects


def test_apply_upgrades_aero_positive():
    tree = RnDTree()
    s = TeamRnDState(team_id="rbr", completed_nodes={"aero_1"})
    effects = apply_upgrades_to_team(s, tree)
    assert effects["aero_efficiency"] > 0


# --------------------------------------------------------------------------- #
# 实战场景
# --------------------------------------------------------------------------- #
def test_season_rnd_progression():
    """完整赛季 R&D 进展: 每周推进, 累计升级."""
    tree = RnDTree()
    s = TeamRnDState(team_id="rbr", resource_points=2000)
    # 赛季 24 周, 每周推进
    for _week in range(24):
        # 每周获得练习/完赛点
        s.add_resource_points(30)
        # 尝试开始可解锁的升级
        for branch in RnDBranch:
            next_node = tree.next_unlockable(branch, s.completed_nodes)
            if next_node and s.resource_points >= next_node.cost_rp:
                s.start_upgrade(tree, next_node.node_id)
        s.advance_week(tree)
    # 赛季结束应有若干升级完成
    assert len(s.completed_nodes) >= 5


def test_top_team_maxed_branch():
    """顶级车队某分支全升级."""
    tree = RnDTree()
    all_aero = {f"aero_{i}" for i in range(1, 6)}
    s = TeamRnDState(team_id="rbr", resource_points=100000,
                     completed_nodes=all_aero)
    effects = s.total_effects(tree)
    # 累计空动效率提升应显著
    assert effects["aero_efficiency"] >= 0.10


def test_backmarker_limited_progress():
    """后段车队 R&D 点有限, 进展少."""
    tree = RnDTree()
    s = TeamRnDState(team_id="has", resource_points=300)
    # 只能做 1 个低成本升级
    assert s.start_upgrade(tree, "dur_1") is True  # 150 RP
    assert s.resource_points == 150
    # 剩余不够第二个
    assert s.start_upgrade(tree, "dur_2") is False  # 需 250


def test_balanced_development():
    """均衡发展: 4 分支各升级 1 节点."""
    tree = RnDTree()
    s = TeamRnDState(team_id="mer", resource_points=2000)
    for branch in RnDBranch:
        first = tree.first_node_in_branch(branch)
        s.start_upgrade(tree, first.node_id)
    s.advance_week(tree)  # 全部 1 周完成
    assert len(s.completed_nodes) == 4
    effects = s.total_effects(tree)
    # 应有 4 个不同效果
    assert len(effects) >= 3
