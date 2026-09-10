"""迭代闭环管理 —— 历史反馈/建议留存与前后对比。

对齐 design.md 2.1.3「迭代闭环」模块职责与 spec FR-ITER-01/02：
支撑「跑圈 → 反馈 → 建议 → 再跑」闭环的可追溯记录与对比视图。
"""

from __future__ import annotations

from typing import Any

from setup_tuner.db.store import Store


class IterationService:
    """迭代闭环管理服务。

    Args:
        store: 注入的 Store 实例。
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # 创建迭代记录
    # ------------------------------------------------------------------
    def create_iteration(
        self,
        track_id: str,
        before_setup_id: int | None,
        after_setup_id: int | None,
        suggestion_id: int | None,
    ) -> int:
        """创建一条迭代闭环记录，轮次号自动递增。

        Args:
            track_id: 赛道标识。
            before_setup_id: 调整前调教快照 id。
            after_setup_id: 调整后调教快照 id。
            suggestion_id: 对应建议 id。

        Returns:
            新插入的 iteration.id。
        """
        round_no = self.get_latest_round(track_id) + 1
        return self._store.save_iteration(
            track_id=track_id,
            round_no=round_no,
            before_setup_id=before_setup_id,
            after_setup_id=after_setup_id,
            suggestion_id=suggestion_id,
        )

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get_history(self, track_id: str) -> list[dict[str, Any]]:
        """查询某赛道的迭代历史（按 round_no 升序）。"""
        return self._store.get_iterations(track_id)

    def get_latest_round(self, track_id: str) -> int:
        """获取某赛道最新轮次号。

        Returns:
            最新轮次号；无历史记录时返回 0（下一次 create_iteration 将使用 1）。
        """
        iterations = self._store.get_iterations(track_id)
        if not iterations:
            return 0
        return max(it["round_no"] for it in iterations)

    # ------------------------------------------------------------------
    # 前后调教对比
    # ------------------------------------------------------------------
    @staticmethod
    def compare_setups(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        """对比前后两份调教快照的差异。

        输入为扁平参数字典（参数名 → 数值）。仅比较两份字典共有的键，
        返回变化项列表与统计摘要。

        Args:
            before: 调整前参数字典。
            after: 调整后参数字典。

        Returns:
            ``{
                "changes": [{"name", "before", "after", "delta"}, ...],
                "changed_count": int,
                "total_params": int,
                "unchanged_count": int,
            }``
        """
        common_keys = set(before.keys()) & set(after.keys())
        changes: list[dict[str, Any]] = []
        for name in sorted(common_keys):
            b_val = before[name]
            a_val = after[name]
            if b_val != a_val:
                changes.append(
                    {
                        "name": name,
                        "before": b_val,
                        "after": a_val,
                        "delta": float(a_val) - float(b_val),
                    }
                )
        total = len(common_keys)
        return {
            "changes": changes,
            "changed_count": len(changes),
            "total_params": total,
            "unchanged_count": total - len(changes),
        }