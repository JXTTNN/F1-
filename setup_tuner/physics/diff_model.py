"""差速器模型 — 开/松油门差速 → 入弯/出弯稳定性。

物理原理：
    F1差速器通过控制左右车轮的锁定程度来影响弯道行为。

    - 锁定系数 = (diff_value - 50) / 50
        0 = 最开（50%，允许两轮自由转速差）
        1 = 全锁（100%，两轮强制同速）
    - 开油门差速（on-throttle）：
        值低（开）→ 出弯牵引好但稳定性差
        值高（锁）→ 稳定但牵引差
        corner_exit_traction = 1.0 - on_throttle_lock * 0.7
        on_throttle_oversteer = on_throttle_lock * 0.5
    - 松油门差速（off-throttle）：
        值低（开）→ 入弯响应好但稳定性差
        值高（锁）→ 稳定但响应差
        turn_in_response = 1.0 - off_throttle_lock * 0.7
        corner_entry_stability = 0.3 + off_throttle_lock * 0.7
        off_throttle_oversteer = (1.0 - off_throttle_lock) * 0.3
"""
from __future__ import annotations

from dataclasses import dataclass

# ── 差速器参数范围 ────────────────────────────────────────
DIFF_MIN_VALUE: float = 50.0  # %, 最开差速值
DIFF_MAX_VALUE: float = 100.0  # %, 全锁差速值

# ── 开油门差速系数 ────────────────────────────────────────
ON_THROTTLE_TRACTION_FACTOR: float = 0.7    # 锁定对牵引力的影响系数
ON_THROTTLE_OVERSTEER_FACTOR: float = 0.5   # 锁定对转向过度的影响系数

# ── 松油门差速系数 ────────────────────────────────────────
OFF_THROTTLE_RESPONSE_FACTOR: float = 0.7    # 锁定对入弯响应的影响系数
OFF_THROTTLE_STABILITY_BASE: float = 0.3     # 入弯稳定性基础值
OFF_THROTTLE_STABILITY_FACTOR: float = 0.7   # 锁定对入弯稳定性的影响系数
OFF_THROTTLE_OVERSTEER_FACTOR: float = 0.3   # 开放对转向过度的影响系数


@dataclass
class DiffOutput:
    """差速器模型计算输出。

    所有评分值为 0-1 的无量纲系数。
    """
    on_throttle_lock: float        # 0-1, 开油门锁定系数 (1=全锁, 0=全开)
    off_throttle_lock: float       # 0-1, 松油门锁定系数
    corner_entry_stability: float  # 0-1, 入弯稳定性
    corner_exit_traction: float    # 0-1, 出弯牵引力
    turn_in_response: float        # 0-1, 入弯响应
    on_throttle_oversteer: float   # 0-1, 开油门转向过度倾向
    off_throttle_oversteer: float  # 0-1, 松油门转向过度倾向


class DiffModel:
    """差速器模型。

    通过开油门和松油门差速值，计算入弯/出弯的稳定性、牵引力和响应特性。
    """

    def __init__(self, on_throttle_diff: float, off_throttle_diff: float) -> None:
        """初始化差速器模型。

        Args:
            on_throttle_diff: 开油门差速值 (50-100%)
            off_throttle_diff: 松油门差速值 (50-100%)
        """
        self.on_throttle_diff: float = on_throttle_diff
        self.off_throttle_diff: float = off_throttle_diff

    def compute(self) -> DiffOutput:
        """计算差速器输出。

        Returns:
            DiffOutput 包含锁定系数、稳定性、牵引力等评分
        """
        on_throttle_lock = self._calc_lock_coefficient(self.on_throttle_diff)
        off_throttle_lock = self._calc_lock_coefficient(self.off_throttle_diff)

        corner_exit_traction = self._calc_corner_exit_traction(on_throttle_lock)
        on_throttle_oversteer = self._calc_on_throttle_oversteer(on_throttle_lock)

        turn_in_response = self._calc_turn_in_response(off_throttle_lock)
        corner_entry_stability = self._calc_corner_entry_stability(off_throttle_lock)
        off_throttle_oversteer = self._calc_off_throttle_oversteer(off_throttle_lock)

        return DiffOutput(
            on_throttle_lock=on_throttle_lock,
            off_throttle_lock=off_throttle_lock,
            corner_entry_stability=corner_entry_stability,
            corner_exit_traction=corner_exit_traction,
            turn_in_response=turn_in_response,
            on_throttle_oversteer=on_throttle_oversteer,
            off_throttle_oversteer=off_throttle_oversteer,
        )

    @staticmethod
    def _calc_lock_coefficient(diff_value: float) -> float:
        """计算锁定系数 = (diff_value - 50) / 50。

        50% → 0（最开），100% → 1（全锁）。
        """
        return (diff_value - DIFF_MIN_VALUE) / (DIFF_MAX_VALUE - DIFF_MIN_VALUE)

    @staticmethod
    def _calc_corner_exit_traction(on_throttle_lock: float) -> float:
        """计算出弯牵引力 = 1.0 - lock * 0.7。"""
        return 1.0 - on_throttle_lock * ON_THROTTLE_TRACTION_FACTOR

    @staticmethod
    def _calc_on_throttle_oversteer(on_throttle_lock: float) -> float:
        """计算开油门转向过度 = lock * 0.5。"""
        return on_throttle_lock * ON_THROTTLE_OVERSTEER_FACTOR

    @staticmethod
    def _calc_turn_in_response(off_throttle_lock: float) -> float:
        """计算入弯响应 = 1.0 - lock * 0.7。"""
        return 1.0 - off_throttle_lock * OFF_THROTTLE_RESPONSE_FACTOR

    @staticmethod
    def _calc_corner_entry_stability(off_throttle_lock: float) -> float:
        """计算入弯稳定性 = 0.3 + lock * 0.7。"""
        return OFF_THROTTLE_STABILITY_BASE + off_throttle_lock * OFF_THROTTLE_STABILITY_FACTOR

    @staticmethod
    def _calc_off_throttle_oversteer(off_throttle_lock: float) -> float:
        """计算松油门转向过度 = (1.0 - lock) * 0.3。"""
        return (1.0 - off_throttle_lock) * OFF_THROTTLE_OVERSTEER_FACTOR