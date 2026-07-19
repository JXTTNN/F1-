"""F1 2026 进站窗口优化器 (Iter-46).

进站窗口 (Pit Window) 是车队决定何时进站的核心策略问题. 本模块
基于赛道几何 (pit loss)、轮胎性能曲线、交通状况、安全车概率,
计算最优进站窗口.

F1 2026 进站窗口决策因素:
- **轮胎性能曲线**: 化合物随圈数的圈速损失 (degradation curve).
- **Undercut**: 早进站用新胎追前车, 但有 pit loss.
- **Overcut**: 晚进站利用旧胎维持圈速, 等前车进站后超车.
- **安全车概率**: SC 期间进站 "free pit" (pit loss 折扣).
- **交通**: 前后车距离影响进站后是否陷入慢车阵.
- **轮胎温度**: 新胎需 warmup 圈数, 进站后初期圈速慢.
- **2-stop vs 1-stop**: 总圈数决定策略骨架.

EA F1 2026 策略助手特性:
- 实时推荐进站窗口 (lap range).
- "Pit now" vs "Stay out" 决策.
- 安全车响应策略.
- Undercut/Overcut 机会识别.

公开 API:
    - :class:`PitWindowState` — 当前车手状态.
    - :class:`PitWindowRecommendation` — 推荐结果.
    - :class:`PitWindowOptimizer` — 优化器.
    - :func:`recommend_pit_window` — 便捷函数.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
# 轮胎性能曲线参数: 每圈圈速损失 (s/lap) — 简化线性 + 临界点
# 实际曲线为非线性, 临界点后磨损加速
_TIRE_DEG_PER_LAP: dict[str, float] = {
    "soft": 0.08,     # 软胎磨耗快
    "medium": 0.045,
    "hard": 0.025,
    "intermediate": 0.06,
    "wet": 0.04,
}

# 轮胎临界圈数 (超过后磨耗加速)
_TIRE_CLIFF_LAPS: dict[str, int] = {
    "soft": 18,
    "medium": 28,
    "hard": 40,
    "intermediate": 22,
    "wet": 30,
}

# 临界后磨耗倍数
_CLIFF_MULTIPLIER = 2.5

# 新胎 warmup 圈数 + 圈速损失
_WARMUP_LAPS = 2
_WARMUP_PENALTY_S = 0.5  # 每圈 warmup 损失

# 默认 pit loss (秒)
_DEFAULT_PIT_LOSS_S = 23.0


# --------------------------------------------------------------------------- #
# 数据类
# --------------------------------------------------------------------------- #
@dataclass
class PitWindowState:
    """车手当前状态 — 用于进站窗口计算."""

    current_lap: int                  # 当前圈数
    total_laps: int                   # 总圈数
    current_compound: str             # 当前化合物
    tire_age_laps: int                # 当前轮胎已跑圈数
    current_position: int             # 当前位置
    gap_ahead_s: float                # 与前车差距 (s)
    gap_behind_s: float               # 与后车差距 (s)
    pit_loss_s: float = _DEFAULT_PIT_LOSS_S
    next_compound: str = "medium"     # 进站后换的化合物
    sc_probability: float = 0.0       # 未来 5 圈安全车概率
    sc_pit_discount: float = 1.0      # SC 期间 pit loss 折扣 (1.0=无折扣)
    target_compound_for_finish: str = "medium"  # 完赛用化合物


@dataclass
class PitWindowRecommendation:
    """进站窗口推荐结果."""

    recommend_pit_now: bool           # 是否推荐立即进站
    optimal_pit_lap: int              # 最优进站圈
    pit_window_start: int             # 窗口开始圈
    pit_window_end: int               # 窗口结束圈
    reason: str                       # 推荐理由
    undercut_opportunity: bool        # 是否有 undercut 机会
    overcut_opportunity: bool         # 是否有 overcut 机会
    sc_pit_opportunity: bool          # 是否有 SC 进站机会
    projected_position_after_pit: int # 预测进站后位置
    projected_final_position: int     # 预测完赛位置
    risk_level: str = "medium"        # low / medium / high
    alternatives: list[str] = field(default_factory=list)
    """备选策略说明."""


# --------------------------------------------------------------------------- #
# PitWindowOptimizer
# --------------------------------------------------------------------------- #
class PitWindowOptimizer:
    """进站窗口优化器.

    用法::

        opt = PitWindowOptimizer()
        state = PitWindowState(
            current_lap=15, total_laps=53, current_compound="medium",
            tire_age_laps=15, current_position=3,
            gap_ahead_s=2.5, gap_behind_s=4.0,
            next_compound="hard", sc_probability=0.15,
        )
        rec = opt.optimize(state)
        if rec.recommend_pit_now:
            print(f"进站! 原因: {rec.reason}")
    """

    def __init__(self, pit_loss_s: float = _DEFAULT_PIT_LOSS_S) -> None:
        self.default_pit_loss_s = float(pit_loss_s)

    # ------------------------------------------------------------------ #
    def optimize(self, state: PitWindowState) -> PitWindowRecommendation:
        """计算最优进站窗口."""
        laps_remaining = state.total_laps - state.current_lap
        pit_loss = state.pit_loss_s

        # 轮胎临界检查
        cliff = _TIRE_CLIFF_LAPS.get(state.current_compound, 25)
        past_cliff = state.tire_age_laps >= cliff
        approaching_cliff = state.tire_age_laps >= cliff - 3

        # 计算最优进站圈: 最大化剩余 stint 长度, 同时利用轮胎性能
        # 简化: 最优进站圈 = 当前圈 + max(1, (laps_remaining - stint_target) // 2)
        next_cliff = _TIRE_CLIFF_LAPS.get(state.next_compound, 25)
        # 下一段 stint 应在临界前完赛
        max_next_stint = next_cliff - 2  # 留 2 圈余量
        # 最优进站圈使得下一段刚好在临界前完赛
        optimal_pit_lap = state.total_laps - max_next_stint
        optimal_pit_lap = max(optimal_pit_lap, state.current_lap + 1)
        optimal_pit_lap = min(optimal_pit_lap, state.total_laps - 3)

        # 窗口范围: 最优圈 ± 3
        window_start = max(state.current_lap + 1, optimal_pit_lap - 3)
        window_end = min(state.total_laps - 2, optimal_pit_lap + 3)

        # Undercut 机会: 前车 1-3s 内, 早进站可能超越
        undercut = (1.0 <= state.gap_ahead_s <= 3.0 and
                    state.tire_age_laps >= cliff - 5)

        # Overcut 机会: 前车刚进站, 自己用旧胎多跑几圈
        overcut = (state.gap_ahead_s < 1.0 and
                   state.tire_age_laps < cliff and
                   not past_cliff)

        # SC 进站机会: 当前 SC 概率高, 等 SC 进站省 pit loss
        sc_pit = (state.sc_probability > 0.20 and
                  state.tire_age_laps < cliff + 3)

        # 立即进站判断
        reasons: list[str] = []
        recommend_now = False

        if past_cliff:
            reasons.append("轮胎已过临界, 必须进站")
            recommend_now = True
        elif approaching_cliff and state.current_lap >= optimal_pit_lap - 2:
            reasons.append("接近轮胎临界 + 进入窗口")
            recommend_now = True
        elif undercut and state.current_lap >= window_start:
            reasons.append(f"Undercut 机会: 前车 {state.gap_ahead_s:.1f}s 内")
            recommend_now = True
        elif sc_pit and state.sc_probability > 0.35:
            reasons.append(f"高 SC 概率 ({state.sc_probability:.0%}), 立即进站利用 free pit")
            recommend_now = True
        elif state.current_lap >= optimal_pit_lap and laps_remaining <= max_next_stint:
            reasons.append("到达最优进站圈, 余下圈数匹配新胎 stint")
            recommend_now = True

        # 风险评估
        if past_cliff:
            risk = "high"
        elif state.sc_probability > 0.30:
            risk = "medium"
        elif undercut or overcut:
            risk = "medium"
        else:
            risk = "low"

        # 预测位置 (简化)
        # 进站后掉 ~pit_loss / avg_lap_time 个位置
        avg_lap = 90.0
        positions_lost = max(1, int(pit_loss * state.sc_pit_discount / avg_lap * 20))
        if sc_pit and state.sc_pit_discount < 0.6:
            positions_lost = max(0, positions_lost - 2)
        projected_after_pit = min(20, state.current_position + positions_lost)
        projected_final = max(1, projected_after_pit - 1)  # 新胎追回

        # 备选策略
        alternatives: list[str] = []
        if laps_remaining > next_cliff + 5:
            alternatives.append("考虑 2-stop: 多一次进站但轮胎更新")
        if not past_cliff and laps_remaining < cliff - state.tire_age_laps + 3:
            alternatives.append("延长 stint: 当前轮胎仍可坚持到完赛")
        if state.sc_probability > 0.20:
            alternatives.append("等 SC: 维修区损失折扣可观")

        reason = reasons[0] if reasons else "未到最优窗口, 继续观察"
        if not recommend_now and not reasons:
            reason = (f"当前圈 {state.current_lap}, 最优窗口 "
                      f"{window_start}-{window_end}, 建议等待")

        return PitWindowRecommendation(
            recommend_pit_now=recommend_now,
            optimal_pit_lap=optimal_pit_lap,
            pit_window_start=window_start,
            pit_window_end=window_end,
            reason=reason,
            undercut_opportunity=undercut,
            overcut_opportunity=overcut,
            sc_pit_opportunity=sc_pit,
            projected_position_after_pit=projected_after_pit,
            projected_final_position=projected_final,
            risk_level=risk,
            alternatives=alternatives,
        )


# --------------------------------------------------------------------------- #
# Convenience function
# --------------------------------------------------------------------------- #
def recommend_pit_window(state: PitWindowState) -> PitWindowRecommendation:
    """便捷函数: 计算进站窗口推荐."""
    return PitWindowOptimizer().optimize(state)
