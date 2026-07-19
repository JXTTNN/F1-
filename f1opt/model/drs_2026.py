"""F1 2026 — DRS 替代规则 (Iter-31).

FIA 2026 重大变化: **DRS 在部分场景被 Manual Override + Active Aero 替代**.

但 DRS 仍保留用于:
1. **正赛 DRS 区**: 仍然每圈可用 (检测 + 激活).
2. **取消 "DRS Disabled" 后段**: 改为 Manual Override.
3. **Train of cars**: 第 2 圈起 DRS 启用 (避免第 1 圈事故).
4. **Train lap / SC restart 后**: DRS 启用延迟 2 圈.

公开 API:
    - :class:`DRS2026Model` — 单圈 DRS 状态.
    - :func:`drs_for_lap` — 便捷函数.
"""

from __future__ import annotations

from dataclasses import dataclass

# FIA 2026 DRS 规则
_DRS_GAP_THRESHOLD_S = 1.0
"""前车 1s 内可激活 DRS (传统 F1)."""
_DRS_GAIN_S_PER_ZONE = 0.45
"""每 DRS 区圈速增益 ~0.45s."""
_DRS_DISABLED_AFTER_SC_LAPS = 2
"""SC restart 后 2 圈 DRS 禁用."""
_DRS_DISABLED_LAP_1 = True
"""第 1 圈 DRS 禁用."""


@dataclass
class DRS2026State:
    """单圈 DRS 状态."""

    lap: int
    drs_available: bool
    """本圈 DRS 是否可用."""
    n_drs_zones: int
    """该赛道 DRS 区数量."""
    drs_active_zones: int
    """本圈实际激活的 DRS 区数 (车手在该圈通过 DRS 检测点)."""
    lap_time_gain_s: float
    """本圈 DRS 增益圈速 (s)."""
    reason: str
    """状态原因."""


@dataclass
class DRS2026Model:
    """F1 2026 DRS 状态模型 (Iter-31).

    用法::

        drs = DRS2026Model(track_id="monza", n_drs_zones=2)
        state = drs.simulate_lap(
            lap=10, gap_ahead_s=0.8,
            sc_just_ended_lap=0,  # 0 = 没有 SC
        )
    """

    track_id: str
    n_drs_zones: int = 2
    """该赛道 DRS 区数量."""

    # ------------------------------------------------------------------ #
    def simulate_lap(
        self,
        lap: int,
        gap_ahead_s: float | None,
        sc_just_ended_lap: int = 0,
        track_wetness: float = 0.0,
    ) -> DRS2026State:
        """仿真单圈 DRS 状态."""
        # 第 1 圈禁用
        if lap == 1 and _DRS_DISABLED_LAP_1:
            return DRS2026State(
                lap=lap, drs_available=False, n_drs_zones=self.n_drs_zones,
                drs_active_zones=0, lap_time_gain_s=0.0,
                reason="DRS disabled lap 1 (F1 2026 rule)",
            )

        # SC 后 2 圈禁用 (含 SC 结束圈本身: lap=11,12 禁用, lap=13 启用)
        if sc_just_ended_lap > 0 and lap - sc_just_ended_lap <= _DRS_DISABLED_AFTER_SC_LAPS:
            return DRS2026State(
                lap=lap, drs_available=False, n_drs_zones=self.n_drs_zones,
                drs_active_zones=0, lap_time_gain_s=0.0,
                reason=f"DRS disabled (SC ended lap {sc_just_ended_lap}, "
                       f"need 2 laps)",
            )

        # 湿地禁用 DRS
        if track_wetness > 0.30:
            return DRS2026State(
                lap=lap, drs_available=False, n_drs_zones=self.n_drs_zones,
                drs_active_zones=0, lap_time_gain_s=0.0,
                reason="DRS disabled (wet conditions)",
            )

        # 没有前车 → DRS 不可用
        if gap_ahead_s is None:
            return DRS2026State(
                lap=lap, drs_available=False, n_drs_zones=self.n_drs_zones,
                drs_active_zones=0, lap_time_gain_s=0.0,
                reason="No car ahead",
            )

        # DRS 可用条件: 前车在 1s 内
        if gap_ahead_s <= _DRS_GAP_THRESHOLD_S:
            # 所有 DRS 区都可用 (假设车手在每个检测点都满足条件)
            n_active = self.n_drs_zones
            gain = n_active * _DRS_GAIN_S_PER_ZONE
            return DRS2026State(
                lap=lap, drs_available=True, n_drs_zones=self.n_drs_zones,
                drs_active_zones=n_active, lap_time_gain_s=gain,
                reason=f"DRS active (gap {gap_ahead_s:.2f}s, {n_active} zones)",
            )

        # 不满足 DRS 条件
        return DRS2026State(
            lap=lap, drs_available=False, n_drs_zones=self.n_drs_zones,
            drs_active_zones=0, lap_time_gain_s=0.0,
            reason=f"Gap too large ({gap_ahead_s:.2f}s > {_DRS_GAP_THRESHOLD_S}s)",
        )


def drs_for_lap(
    track_id: str,
    lap: int,
    n_drs_zones: int = 2,
    gap_ahead_s: float | None = None,
    sc_just_ended_lap: int = 0,
    track_wetness: float = 0.0,
) -> DRS2026State:
    """便捷函数."""
    m = DRS2026Model(track_id=track_id, n_drs_zones=n_drs_zones)
    return m.simulate_lap(
        lap=lap, gap_ahead_s=gap_ahead_s,
        sc_just_ended_lap=sc_just_ended_lap,
        track_wetness=track_wetness,
    )
