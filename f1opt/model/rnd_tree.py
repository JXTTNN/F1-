"""F1 2026 R&D 升级树模型 (Iter-51).

EA Sports F1 2026 生涯模式核心: 车队通过 R&D 点 (Resource Points) 升级
赛车性能. R&D 树分为 4 大分支:

- **Aerodynamics (空动)**: 下压力、阻力效率、DRS 效果.
- **Power Unit (动力单元)**: 引擎功率、燃油效率、ERS 部署.
- **Chassis (底盘)**: 重量、重心、悬挂几何.
- **Durability (耐久)**: 引擎/变速箱/刹车可靠性.

每条升级线有多个节点, 需按顺序解锁. 升级消耗 R&D 点, 完成需时间
(赛道周数). 升级影响赛车性能参数:

- aero_efficiency (空动效率): +0.01..+0.03 per node
- power_unit_kW (动力): +1..+3 kW per node
- reliability (可靠性): +0.01..+0.02 per node
- tire_degradation_factor (轮胎磨损): -0.01..-0.02 per node
- fuel_efficiency (燃油效率): +0.005..+0.015 per node

EA F1 2026 R&D 规则:
- 起始 R&D 点: 取决于车队预算帽 ($60M 开发预算).
- 每场练习项目成功: +5..15 R&D 点 (Iter-45).
- 每场完赛: +10 R&D 点.
- 升级节点需 1-3 个赛道周完成.
- 2026 规则: 简化 R&D 树 (减少跨季结转).

公开 API:
    - :class:`RnDNode` — 单升级节点.
    - :class:`RnDTree` — 升级树.
    - :class:`TeamRnDState` — 车队 R&D 状态.
    - :func:`apply_upgrades_to_team` — 应用升级到车队性能参数.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# --------------------------------------------------------------------------- #
# 枚举
# --------------------------------------------------------------------------- #
class RnDBranch(Enum):
    """R&D 升级分支."""

    AERODYNAMICS = "aero"
    POWER_UNIT = "pu"
    CHASSIS = "chassis"
    DURABILITY = "durability"


# --------------------------------------------------------------------------- #
# RnDNode: 单升级节点
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RnDNode:
    """单升级节点.

    - ``node_id``: 唯一 ID.
    - ``branch``: 所属分支.
    - ``name``: 升级名称.
    - ``cost_rp``: R&D 点消耗.
    - ``weeks_to_complete``: 完成所需赛道周.
    - ``prereq_id``: 前置节点 ID (None = 无).
    - ``effects``: 性能影响字典.
    """

    node_id: str
    branch: RnDBranch
    name: str
    cost_rp: int
    weeks_to_complete: int
    prereq_id: str | None = None
    effects: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# RnDTree: 升级树定义
# --------------------------------------------------------------------------- #
# EA F1 2026 简化 R&D 树: 每分支 5 节点, 累计升级
_RND_TREE: list[RnDNode] = [
    # === Aerodynamics (空动) ===
    RnDNode("aero_1", RnDBranch.AERODYNAMICS, "前翼优化", 200, 1,
            effects={"aero_efficiency": 0.02}),
    RnDNode("aero_2", RnDBranch.AERODYNAMICS, "底板升级", 350, 2, "aero_1",
            effects={"aero_efficiency": 0.02}),
    RnDNode("aero_3", RnDBranch.AERODYNAMICS, "侧箱重塑", 500, 2, "aero_2",
            effects={"aero_efficiency": 0.03, "tire_degradation_factor": -0.01}),
    RnDNode("aero_4", RnDBranch.AERODYNAMICS, "扩散器 v2", 700, 3, "aero_3",
            effects={"aero_efficiency": 0.03}),
    RnDNode("aero_5", RnDBranch.AERODYNAMICS, "主动空动优化", 900, 3, "aero_4",
            effects={"aero_efficiency": 0.04, "drs_effectiveness": 0.05}),

    # === Power Unit (动力) ===
    RnDNode("pu_1", RnDBranch.POWER_UNIT, "ICE 燃烧优化", 250, 1,
            effects={"power_unit_kW": 2.0, "fuel_efficiency": 0.01}),
    RnDNode("pu_2", RnDBranch.POWER_UNIT, "MGU-K 升级", 400, 2, "pu_1",
            effects={"power_unit_kW": 2.0}),
    RnDNode("pu_3", RnDBranch.POWER_UNIT, "电池能量管理", 550, 2, "pu_2",
            effects={"power_unit_kW": 1.0, "fuel_efficiency": 0.015}),
    RnDNode("pu_4", RnDBranch.POWER_UNIT, "可持续燃料适配", 700, 3, "pu_3",
            effects={"power_unit_kW": 1.0, "fuel_efficiency": 0.02}),
    RnDNode("pu_5", RnDBranch.POWER_UNIT, "PU 整体集成", 1000, 3, "pu_4",
            effects={"power_unit_kW": 3.0, "fuel_efficiency": 0.01}),

    # === Chassis (底盘) ===
    RnDNode("cha_1", RnDBranch.CHASSIS, "悬挂几何", 200, 1,
            effects={"tire_degradation_factor": -0.01}),
    RnDNode("cha_2", RnDBranch.CHASSIS, "重量分布优化", 350, 2, "cha_1",
            effects={"tire_degradation_factor": -0.015}),
    RnDNode("cha_3", RnDBranch.CHASSIS, "刹车冷却升级", 450, 2, "cha_2",
            effects={"tire_degradation_factor": -0.01, "reliability": 0.01}),
    RnDNode("cha_4", RnDBranch.CHASSIS, "车身刚性提升", 650, 3, "cha_3",
            effects={"tire_degradation_factor": -0.02}),
    RnDNode("cha_5", RnDBranch.CHASSIS, "主动悬挂原型", 900, 3, "cha_4",
            effects={"tire_degradation_factor": -0.02, "aero_efficiency": 0.01}),

    # === Durability (耐久) ===
    RnDNode("dur_1", RnDBranch.DURABILITY, "ICE 可靠性", 150, 1,
            effects={"reliability": 0.02}),
    RnDNode("dur_2", RnDBranch.DURABILITY, "变速箱强化", 250, 1, "dur_1",
            effects={"reliability": 0.02}),
    RnDNode("dur_3", RnDBranch.DURABILITY, "刹车耐久", 350, 2, "dur_2",
            effects={"reliability": 0.015}),
    RnDNode("dur_4", RnDBranch.DURABILITY, "电子系统冗余", 500, 2, "dur_3",
            effects={"reliability": 0.02}),
    RnDNode("dur_5", RnDBranch.DURABILITY, "全车耐久包", 700, 3, "dur_4",
            effects={"reliability": 0.03}),
]


class RnDTree:
    """R&D 升级树."""

    def __init__(self) -> None:
        self._nodes: dict[str, RnDNode] = {n.node_id: n for n in _RND_TREE}

    def get_node(self, node_id: str) -> RnDNode:
        if node_id not in self._nodes:
            raise ValueError(f"Unknown R&D node: {node_id!r}")
        return self._nodes[node_id]

    def all_nodes(self) -> list[RnDNode]:
        return list(self._nodes.values())

    def nodes_in_branch(self, branch: RnDBranch) -> list[RnDNode]:
        return [n for n in _RND_TREE if n.branch == branch]

    def first_node_in_branch(self, branch: RnDBranch) -> RnDNode:
        nodes = self.nodes_in_branch(branch)
        if not nodes:
            raise ValueError(f"No nodes in branch: {branch}")
        return nodes[0]

    def can_unlock(self, node_id: str, completed: set[str]) -> bool:
        """节点是否可解锁 (前置已完成)."""
        node = self.get_node(node_id)
        if node.prereq_id is None:
            return True
        return node.prereq_id in completed

    def next_unlockable(self, branch: RnDBranch, completed: set[str]) -> RnDNode | None:
        """分支中下一个可解锁节点."""
        for node in self.nodes_in_branch(branch):
            if node.node_id not in completed and self.can_unlock(node.node_id, completed):
                return node
        return None


# --------------------------------------------------------------------------- #
# TeamRnDState: 车队 R&D 状态
# --------------------------------------------------------------------------- #
@dataclass
class TeamRnDState:
    """车队 R&D 状态 — 跟踪 R&D 点、已完成升级、进行中升级.

    - ``resource_points``: 当前 R&D 点.
    - ``completed_nodes``: 已完成升级节点 ID 集合.
    - ``in_progress``: 进行中升级 {node_id: 剩余周数}.
    - ``total_invested_rp``: 累计投入 R&D 点.
    """

    team_id: str
    resource_points: int = 1000
    completed_nodes: set[str] = field(default_factory=set)
    in_progress: dict[str, int] = field(default_factory=dict)
    total_invested_rp: int = 0

    # ------------------------------------------------------------------ #
    def start_upgrade(self, tree: RnDTree, node_id: str) -> bool:
        """开始升级节点, 返回是否成功.

        - 检查 R&D 点足够 + 前置完成 + 未在升级中.
        """
        node = tree.get_node(node_id)
        if node_id in self.completed_nodes:
            return False
        if node_id in self.in_progress:
            return False
        if not tree.can_unlock(node_id, self.completed_nodes):
            return False
        if self.resource_points < node.cost_rp:
            return False
        self.resource_points -= node.cost_rp
        self.total_invested_rp += node.cost_rp
        self.in_progress[node_id] = node.weeks_to_complete
        return True

    # ------------------------------------------------------------------ #
    def advance_week(self, tree: RnDTree) -> list[str]:
        """推进一周, 返回本周完成的节点列表."""
        completed_this_week: list[str] = []
        for node_id in list(self.in_progress.keys()):
            self.in_progress[node_id] -= 1
            if self.in_progress[node_id] <= 0:
                del self.in_progress[node_id]
                self.completed_nodes.add(node_id)
                completed_this_week.append(node_id)
        return completed_this_week

    # ------------------------------------------------------------------ #
    def add_resource_points(self, points: int) -> None:
        """增加 R&D 点 (练习项目/完赛奖励)."""
        if points < 0:
            raise ValueError("Points must be non-negative")
        self.resource_points += points

    # ------------------------------------------------------------------ #
    def total_effects(self, tree: RnDTree) -> dict[str, float]:
        """所有已完成升级的累计效果."""
        effects: dict[str, float] = {}
        for node_id in self.completed_nodes:
            node = tree.get_node(node_id)
            for k, v in node.effects.items():
                effects[k] = effects.get(k, 0.0) + v
        return effects

    # ------------------------------------------------------------------ #
    def branch_progress(self, tree: RnDTree, branch: RnDBranch) -> tuple[int, int]:
        """分支进度 (已完成数, 总数)."""
        total = len(tree.nodes_in_branch(branch))
        done = sum(1 for n in tree.nodes_in_branch(branch)
                   if n.node_id in self.completed_nodes)
        return done, total

    def summary(self, tree: RnDTree) -> dict[str, object]:
        return {
            "team_id": self.team_id,
            "resource_points": self.resource_points,
            "completed": len(self.completed_nodes),
            "in_progress": len(self.in_progress),
            "total_invested_rp": self.total_invested_rp,
            "branches": {
                b.value: self.branch_progress(tree, b) for b in RnDBranch
            },
            "effects": self.total_effects(tree),
        }


# --------------------------------------------------------------------------- #
# 便捷函数
# --------------------------------------------------------------------------- #
def apply_upgrades_to_team(team_state: TeamRnDState, tree: RnDTree) -> dict[str, float]:
    """将 R&D 升级效果应用到车队性能参数.

    Returns:
        累计效果字典, 可加到 :class:`TeamCarProfile2026` 字段上.
    """
    return team_state.total_effects(tree)


def total_branches() -> int:
    return len(list(RnDBranch))


def total_nodes() -> int:
    return len(_RND_TREE)


def nodes_per_branch() -> dict[str, int]:
    return {b.value: sum(1 for n in _RND_TREE if n.branch == b) for b in RnDBranch}
