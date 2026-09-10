"""F1 2026 动力单元组件配额与罚位模型 (Iter-44).

FIA 2026 体育规则 §23 限制每位车手每赛季可用 PU 组件数量:

2026 PU 组件 (移除 MGU-H, 5 大件 + 排气):
- **ICE** (内燃机): 4 件/季
- **MGU-K** (动能电机): 4 件/季 (2026 升级到 350kW)
- **TC** (涡轮增压器): 4 件/季
- **ES** (能量储存/电池): 4 件/季
- **CE** (控制电子): 4 件/季
- **EX** (排气系统): 8 件/季

超额罚位规则 (FIA 2026 §23.3):
- 第 1 次超额某组件: 退后 5 位发车.
- 第 2 次超额同组件: 退后 5 位 (累计 10).
- 第 3+ 次超额同组件: 退后 5 位 + 维修区发车 (若 >15 位).

罚位应用:
- 排位赛后, 若总罚位 > 15, 剩余转为维修区发车.
- 多车罚位时, 按申请顺序应用 (先到先得).
- 罚位不能让车手退到最后发车位之后 (最多 22 位).

EA Sports F1 2026 游戏官方 PU 管理功能:
- R&D 树升级 PU 可靠性 (减少组件损耗).
- 跨赛季组件结转 (有限制).
- 罚位影响排位赛策略 (Q2 退出避免浪费里程).

公开 API:
    - :class:`PUComponentInventory` — 单车手 PU 组件库存.
    - :class:`GridPenaltyCalculator` — 罚位计算.
    - :func:`apply_grid_penalties` — 应用罚位到发车顺序.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# 2026 PU 组件配额 (FIA §23)
# --------------------------------------------------------------------------- #
PU_COMPONENTS_2026: dict[str, int] = {
    "ice": 4,        # 内燃机
    "mgu_k": 4,      # MGU-K (2026: 350kW, 移除 MGU-H)
    "tc": 4,         # 涡轮增压器
    "es": 4,         # 能量储存
    "ce": 4,         # 控制电子
    "exhaust": 8,    # 排气系统
}

# 罚位规则 (FIA 2026 §23.3)
_PENALTY_PER_EXCEED = 5  # 每次超额罚 5 位
_PENALTY_THRESHOLD_PIT_LANE_START = 15  # 总罚位 > 15 → 维修区发车


@dataclass
class PUComponentInventory:
    """单车手 PU 组件库存与使用追踪.

    用法::

        inv = PUComponentInventory(driver_id="ver")
        # 赛季中安装新组件
        inv.install_component("ice", race_idx=5)
        # 检查是否超额
        if inv.is_over_limit("ice"):
            penalty = inv.penalty_for_component("ice")
        # 整场罚位
        total = inv.total_grid_penalty()
    """

    driver_id: str
    # 各组件已使用数量
    used: dict[str, int] = field(default_factory=lambda: {
        "ice": 0, "mgu_k": 0, "tc": 0, "es": 0, "ce": 0, "exhaust": 0
    })
    # 各组件超额次数 (用于累计罚位)
    exceeded: dict[str, int] = field(default_factory=lambda: {
        "ice": 0, "mgu_k": 0, "tc": 0, "es": 0, "ce": 0, "exhaust": 0
    })
    # 安装历史: [(race_idx, component)]
    history: list[tuple[int, str]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def install_component(self, component: str, race_idx: int = 0) -> bool:
        """安装一个新组件, 返回是否触发超额.

        Args:
            component: 组件类型 (ice/mgu_k/tc/es/ce/exhaust).
            race_idx: 当前比赛轮次 (用于历史记录).

        Returns:
            True 如果此次安装触发超额 (会罚位).
        """
        if component not in PU_COMPONENTS_2026:
            raise ValueError(f"Unknown PU component: {component!r}")
        self.used[component] += 1
        self.history.append((race_idx, component))
        # 检查是否超额
        if self.used[component] > PU_COMPONENTS_2026[component]:
            self.exceeded[component] += 1
            return True
        return False

    # ------------------------------------------------------------------ #
    def is_over_limit(self, component: str) -> bool:
        """该组件是否已超额."""
        return self.used.get(component, 0) > PU_COMPONENTS_2026.get(component, 0)

    def remaining(self, component: str) -> int:
        """该组件剩余配额."""
        return max(0, PU_COMPONENTS_2026.get(component, 0) - self.used.get(component, 0))

    def penalty_for_component(self, component: str) -> int:
        """该组件累计罚位 (位数)."""
        return self.exceeded.get(component, 0) * _PENALTY_PER_EXCEED

    def total_grid_penalty(self) -> int:
        """所有组件累计总罚位 (位数)."""
        return sum(self.exceeded.values()) * _PENALTY_PER_EXCEED

    def needs_pit_lane_start(self) -> bool:
        """是否需要维修区发车 (总罚位 > 15)."""
        return self.total_grid_penalty() > _PENALTY_THRESHOLD_PIT_LANE_START

    # ------------------------------------------------------------------ #
    def summary(self) -> dict[str, object]:
        """返回库存摘要."""
        return {
            "driver_id": self.driver_id,
            "used": dict(self.used),
            "quota": dict(PU_COMPONENTS_2026),
            "exceeded": dict(self.exceeded),
            "total_grid_penalty": self.total_grid_penalty(),
            "needs_pit_lane_start": self.needs_pit_lane_start(),
            "n_installs": len(self.history),
        }


# --------------------------------------------------------------------------- #
# GridPenaltyCalculator
# --------------------------------------------------------------------------- #
@dataclass
class GridPenaltyCalculator:
    """罚位计算器 — 处理多车罚位排序.

    FIA 规则: 多车罚位时, 按申请顺序 (先到先得) 应用.
    若罚位让车手退到最后, 剩余罚位"浪费" (不传递给其他车手).
    """

    # ------------------------------------------------------------------ #
    def apply_penalties(
        self,
        qualifying_order: list[str],
        penalties: dict[str, int],
        pit_lane_starts: set[str] | None = None,
    ) -> list[str]:
        """应用罚位到排位赛顺序, 返回最终发车顺序 (FIA 标准 gap-filling).

        FIA 规则:
        - 罚位车手目标位置 = 原排位 + 罚位 (上限 n-1).
        - 非罚位车手按原排位顺序填充剩余空位.
        - 多车目标重叠时, 原排位靠前者优先占位, 后者继续后移.

        Args:
            qualifying_order: 排位赛名次 (driver_id 列表, 第 0 = 杆位).
            penalties: {driver_id: 罚位数}.
            pit_lane_starts: 需维修区发车的车手集合.

        Returns:
            最终发车顺序 (driver_id 列表, 第 0 = 第 1 位发车).
        """
        if pit_lane_starts is None:
            pit_lane_starts = set()

        # 维修区发车的车手放最后 (按原排位顺序)
        pit_lane_drivers = [d for d in qualifying_order if d in pit_lane_starts]
        # 正常发车的车手 (按排位顺序)
        normal = [d for d in qualifying_order if d not in pit_lane_starts]
        n = len(normal)
        if n == 0:
            return pit_lane_drivers

        # 计算罚位车手目标位置 (orig_idx + penalty, 上限 n-1)
        penalized: list[tuple[int, int, str]] = []  # (target, orig_idx, driver)
        for i, d in enumerate(normal):
            pen = penalties.get(d, 0)
            if pen > 0:
                target = min(i + pen, n - 1)
                penalized.append((target, i, d))

        # 按 (target, orig_idx) 排序 — 原排位靠前者优先占位
        penalized.sort(key=lambda x: (x[0], x[1]))

        # 占位: 冲突时向后找下一个空位
        occupied: set[int] = set()
        placement: dict[str, int] = {}
        for target, _orig, d in penalized:
            pos = target
            while pos in occupied and pos < n - 1:
                pos += 1
            # 若到末尾仍冲突, 向前找
            while pos in occupied and pos > 0:
                pos -= 1
            placement[d] = pos
            occupied.add(pos)

        # 构建发车网格: 罚位车手在占位, 其余由非罚位车手按原序填充
        result: list[str | None] = [None] * n
        for d, pos in placement.items():
            result[pos] = d
        non_penalized = [d for d in normal if d not in placement]
        np_idx = 0
        for i in range(n):
            if result[i] is None:
                result[i] = non_penalized[np_idx]
                np_idx += 1

        return [d for d in result if d is not None] + pit_lane_drivers

    # ------------------------------------------------------------------ #
    def summary(
        self,
        qualifying_order: list[str],
        penalties: dict[str, int],
        pit_lane_starts: set[str] | None = None,
    ) -> dict[str, object]:
        """返回罚位应用摘要."""
        final = self.apply_penalties(qualifying_order, penalties, pit_lane_starts)
        changes = {}
        for d in qualifying_order:
            old_pos = qualifying_order.index(d) + 1
            new_pos = final.index(d) + 1
            changes[d] = {"from": old_pos, "to": new_pos,
                          "change": new_pos - old_pos}
        return {
            "qualifying_order": qualifying_order,
            "final_grid": final,
            "penalties": penalties,
            "pit_lane_starts": list(pit_lane_starts or []),
            "position_changes": changes,
        }


# --------------------------------------------------------------------------- #
# Convenience function
# --------------------------------------------------------------------------- #
def apply_grid_penalties(
    qualifying_order: list[str],
    penalties: dict[str, int],
    pit_lane_starts: set[str] | None = None,
) -> list[str]:
    """便捷函数: 应用罚位到排位赛顺序."""
    return GridPenaltyCalculator().apply_penalties(
        qualifying_order, penalties, pit_lane_starts
    )
