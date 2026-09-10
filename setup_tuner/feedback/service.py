"""反馈管理服务 —— 封装反馈业务逻辑。

核心语义（对齐 design.md 2.9.2 / spec FR-FBK-01/03/04）：

- **未点击弯道默认"正常"**：玩家只在有问题的弯道上点击并录入反馈；
  未点击的弯道不产生任何记录，视为正常。
- **请求建议前校验**：至少有 1 条反馈（含全局症状），否则返回引导消息。
- 12 症状枚举与强度 0–5 校验复用 ``domain.symptoms``。
"""

from __future__ import annotations

from typing import Any

from setup_tuner.db.store import Store
from setup_tuner.domain.symptoms import (
    DEFAULT_INTENSITY,
    INTENSITY_MAX,
    INTENSITY_MIN,
    Symptom,
    get_symptom_category,
)

# 预构建 symptom 字符串 → Symptom 枚举的快速索引（12 项）
_VALID_SYMPTOMS: dict[str, Symptom] = {sym.value: sym for sym in Symptom}

# 无反馈时请求建议的引导消息（逐字对齐 spec FR-FBK-04 / design 2.9.2）
_NO_FEEDBACK_HINT = "未发现反馈，请先点击赛道图上的问题弯道并录入反馈"


class FeedbackService:
    """反馈录入/查询/未点击默认正常 业务服务。

    Args:
        store: 注入的 Store 实例。
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # 录入
    # ------------------------------------------------------------------
    def submit_feedback(
        self,
        track_id: str,
        corner_number: int | None,
        symptom: str,
        strength: int = DEFAULT_INTENSITY,
        setup_id: int | None = None,
    ) -> dict[str, Any]:
        """提交一条玩家反馈。

        Args:
            track_id: 赛道标识。
            corner_number: 弯道编号（1-based）；``None`` 表示全局症状
                （如直道刮底、胎耗偏高、直道速度低、圈速不高）。
            symptom: 12 症状标识之一（字符串值）。
            strength: 强度 0–5，默认 3。
            setup_id: 关联的调教快照 id（可选）。

        Returns:
            反馈记录字典（含 id / track_id / corner_number / symptom /
            category / strength / created_at）。

        Raises:
            ValueError: symptom 不在 12 枚举内，或 strength 越界。
        """
        # 校验 symptom 在 12 枚举内
        sym_enum = _VALID_SYMPTOMS.get(symptom)
        if sym_enum is None:
            valid = ", ".join(sorted(_VALID_SYMPTOMS))
            raise ValueError(
                f"未知症状标识 {symptom!r}，合法值：{valid}",
            )
        # 校验 strength 0–5
        if strength < INTENSITY_MIN or strength > INTENSITY_MAX:
            raise ValueError(
                f"症状强度 {strength} 越界，合法范围 [{INTENSITY_MIN}, {INTENSITY_MAX}]",
            )

        category = get_symptom_category(sym_enum)
        feedback_id = self._store.add_feedback(
            track_id=track_id,
            corner_number=corner_number,
            symptom=symptom,
            category=category,
            strength=strength,
            setup_id=setup_id,
        )
        return {
            "id": feedback_id,
            "track_id": track_id,
            "corner_number": corner_number,
            "symptom": symptom,
            "category": category,
            "strength": strength,
            "setup_id": setup_id,
        }

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get_feedbacks(self, track_id: str) -> list[dict[str, Any]]:
        """查询某赛道的全部反馈列表（按 created_at 升序）。"""
        return self._store.get_feedbacks(track_id)

    def get_corner_feedbacks(
        self, track_id: str,
    ) -> dict[int | None, list[dict[str, Any]]]:
        """按弯道编号分组返回反馈。

        Returns:
            字典 ``{corner_number: [反馈记录, ...]}``。
            ``corner_number=None`` 的键对应全局症状反馈。
        """
        grouped: dict[int | None, list[dict[str, Any]]] = {}
        for fb in self._store.get_feedbacks(track_id):
            key = fb["corner_number"]
            grouped.setdefault(key, []).append(fb)
        return grouped

    def get_normal_corners(
        self, track_id: str, total_corners: int,
    ) -> list[int]:
        """返回「正常」弯道编号列表 —— 即没有反馈的弯道。

        核心语义（对齐 FR-FBK-01）：未点击的弯道不提交反馈 = 默认正常。
        本方法返回 ``[1..total_corners]`` 中**没有任何反馈记录**的弯道编号。

        Args:
            track_id: 赛道标识。
            total_corners: 该赛道弯道总数。

        Returns:
            无反馈的弯道编号列表（升序）。全局症状（corner_number=None）
            不影响某弯道是否"正常"。
        """
        all_corners = set(range(1, total_corners + 1))
        # 收集所有有反馈的弯道编号（排除全局症状的 None）
        clicked: set[int] = set()
        for fb in self._store.get_feedbacks(track_id):
            cn = fb["corner_number"]
            if cn is not None:
                clicked.add(cn)
        return sorted(all_corners - clicked)

    # ------------------------------------------------------------------
    # 请求建议前校验
    # ------------------------------------------------------------------
    def validate_before_suggest(self, track_id: str) -> tuple[bool, str]:
        """请求建议前的校验：至少 1 条反馈（含全局症状）。

        对齐 spec FR-FBK-04 / design 2.9.2：无反馈时返回 ``(False, 引导消息)``。

        Args:
            track_id: 赛道标识。

        Returns:
            ``(True, "")`` 表示校验通过，可以生成建议；
            ``(False, 引导消息)`` 表示无反馈，应引导玩家先录入。
        """
        if self._store.has_feedback(track_id):
            return True, ""
        return False, _NO_FEEDBACK_HINT