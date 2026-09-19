"""调教方案的整体优化器（deterministic coordinate ascent）。

定位
----
``compute_setup_delta`` 是"一次线性步"：``Δp = Σ_dim Dx[dim]·C[dim][p]``，取完
即止。它的问题不是方向错，而是**没有评价、没有取舍、无法回头**：

- 无法回答"这一步对整圈是净赚还是净亏"；
- 无法处理"加后翼帮高速弯、但害直道"这类互相拉扯的需求；
- 一次取满，不会因为代价太高而少取一点。

本模块在 :mod:`lap_model` 的目标函数上做**确定性**坐标上升：逐个参数试
±一个网格步长，只接受让整圈目标下降的改动，反复若干轮直到不再改善。
从两个起点出发（规则引擎的解、以及零改动）取更优者，避免被单一起点困住。

确定性：无随机、无时间依赖；参数按固定顺序遍历，候选按固定顺序尝试；
同输入必得同输出（与 ``compute_dx`` 的约定一致）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from setup_tuner.domain.setup import ALL_SETUP_FIELDS

from .lap_model import (
    _FIELDS,
    OPTIMIZABLE,
    CornerEval,
    ObjectiveBreakdown,
    TrackContext,
    apply_weather_to_needs,
    corner_evaluations,
    needs_by_class_from_dx,
    objective,
    satisfaction_by_class,
    track_context,
    units_to_delta,
)

#: 坐标上升的网格步长（归一化单位，1.0 = 走满该参数的 max_delta）
_GRID = 0.125
#: 二级精修网格（粗搜收敛后再用半格长重启，只接受更优解）
_GRID_FINE = 0.0625
#: 最大轮数（每轮遍历全部参数）
_MAX_PASSES = 8
#: 精修阶段的最大轮数（步长更小，需要更多轮才能走完同样的距离）
_MAX_PASSES_FINE = 12
#: 改善小于该值即视为收敛（避免在噪声级别反复微调）
_MIN_GAIN = 1e-6


@dataclass(slots=True)
class OptimizeResult:
    """优化结果。"""

    delta: dict[str, float]
    units: dict[str, float]
    before: ObjectiveBreakdown
    after: ObjectiveBreakdown
    trace: list[str] = field(default_factory=list)
    #: 各弯道类别的残差改善（正 = 该类弯变好）
    per_class_gain: dict[str, float] = field(default_factory=dict)
    #: 优化前（规则引擎初始建议）的需求满足度（0..1，按弯道类别）
    satisfaction_before: dict[str, float] = field(default_factory=dict)
    #: 优化后的需求满足度（0..1，按弯道类别）
    satisfaction_after: dict[str, float] = field(default_factory=dict)

    @property
    def satisfaction_gain(self) -> dict[str, float]:
        """各组满足度提升（优化后 − 优化前）。"""
        return {
            k: round(self.satisfaction_after.get(k, 0.0)
                     - self.satisfaction_before.get(k, 0.0), 4)
            for k in self.satisfaction_after
        }

    @property
    def improvement(self) -> float:
        return self.before.total - self.after.total

    @property
    def improved(self) -> bool:
        return self.improvement > _MIN_GAIN


def _feasible_range(param: str, current_value: float) -> tuple[float, float]:
    """该参数在归一化单位下的可行区间（受参数上下限约束）。"""
    spec = _FIELDS[param]
    max_delta = spec.max_delta or 1.0
    lo = max(-1.0, (spec.min_val - current_value) / max_delta)
    hi = min(1.0, (spec.max_val - current_value) / max_delta)
    if hi < lo:
        return 0.0, 0.0
    return lo, hi


def _initial_units(
    initial_delta: dict[str, float] | None,
    current_setup: dict[str, float],
) -> dict[str, float]:
    """把初始改动（通常来自规则引擎）折算为归一化单位并夹进可行域。"""
    units: dict[str, float] = {}
    for p in OPTIMIZABLE:
        raw = (initial_delta or {}).get(p, 0.0)
        spec = _FIELDS[p]
        u = raw / (spec.max_delta or 1.0)
        lo, hi = _feasible_range(p, float(current_setup.get(p, spec.default)))
        units[p] = max(lo, min(hi, round(u / _GRID) * _GRID))
    return units


def _ascend(
    units: dict[str, float],
    needs: dict[str, dict[str, float]],
    ctx: TrackContext,
    corners: list[CornerEval],
    track_id: str,
    current_setup: dict[str, float],
    grid: float = _GRID,
    max_passes: int = _MAX_PASSES,
) -> tuple[dict[str, float], list[str], ObjectiveBreakdown]:
    """坐标上升主循环，返回 (最优 units, 轨迹, 最终分解)。

    Args:
        grid: 本阶段的步长（归一化单位）。精修阶段传更小的值。
        max_passes: 本阶段最大轮数。
    """
    cur = dict(units)
    cur_val = objective(cur, needs, track_id, ctx, corners)
    trace: list[str] = []

    for _ in range(max_passes):
        moved = False
        for p in OPTIMIZABLE:                      # 固定顺序 → 确定性
            spec = _FIELDS[p]
            lo, hi = _feasible_range(p, float(current_setup.get(p, spec.default)))
            base = cur.get(p, 0.0)
            best_u, best_val = base, cur_val
            for cand in (base + grid, base - grid):    # 固定顺序 → 确定性
                if cand < lo - 1e-9 or cand > hi + 1e-9:
                    continue
                trial = dict(cur)
                trial[p] = round(cand, 6)
                val = objective(trial, needs, track_id, ctx, corners)
                if val.total < best_val.total - _MIN_GAIN:
                    best_u, best_val = trial[p], val
            if best_u != base:
                gain = cur_val.total - best_val.total
                cur_val = best_val
                cur[p] = best_u
                moved = True
                trace.append(
                    f"{spec.label_zh}：{base * spec.max_delta:+.2f} → "
                    f"{best_u * spec.max_delta:+.2f}（圈级目标 −{gain:.5f}）"
                )
        if not moved:
            break
    return cur, trace, cur_val


def optimize_setup(
    dx: dict[str, float],
    current_setup: dict[str, float],
    track_id: str,
    telemetry: dict[str, Any] | None = None,
    feedbacks: list[dict[str, Any]] | None = None,
    initial_delta: dict[str, float] | None = None,
    needs_by_class: dict[str, dict[str, float]] | None = None,
    wet: bool = False,
) -> OptimizeResult:
    """在圈级目标上搜索更优调教。

    Args:
        dx: 诊断向量（9 维）—— 与遥测/反馈合成后的需求。
        current_setup: 当前调教（21 参数，真实单位）。
        track_id: 赛道标识。
        telemetry: 遥测摘要（参与代价项与抓地修正）。
        feedbacks: 逐弯反馈，用于抬高车手关心弯的重要度。
        initial_delta: 起点（通常为规则引擎给出的 ``setup_delta``）。
        needs_by_class: **按弯道类别分列的需求**（来自
            ``holistic.class_weighted_dx().by_class``）。这是权衡能否成立的
            关键：慢弯与快弯带着各自的需求，单一参数集不可能同时满足，
            赛道弯型占比才真正决定取向。缺省时退化为全圈共用一份需求。
        wet: 是否湿地。天气必须进入目标函数，否则不同起点的优化会收敛到
            同一个解，湿地保守意图被抹平（真实回归）。

    Returns:
        :class:`OptimizeResult`，含优化后的 ``delta``、前后目标分解、
        改动轨迹、以及各弯道类别的残差改善。
    """
    corners = corner_evaluations(
        track_id,
        {fb.get("corner_number") for fb in (feedbacks or [])
         if isinstance(fb.get("corner_number"), int)},
    )
    ctx = track_context(track_id, telemetry, dx, wet=wet)
    needs = apply_weather_to_needs(
        needs_by_class or needs_by_class_from_dx(dx), wet,
    )

    # 没有弯道基础（未知/合成 track_id、赛道上无弯道数据）→ **不做圈级优化**，
    # 原样返回起点。原因：此时逐弯残差恒为 0，目标函数只剩代价项，
    # 而代价项的最小值就是"什么都不改"—— 优化器会把规则引擎的整份建议清空
    # （实测打挂 5 个既有测试，表现为"多症状叠加未产生差异"）。
    # 圈级优化必须有赛道弯道数据才有意义（与 track_demand 的中性降级同理）。
    if not corners:
        identity = objective(
            _initial_units(initial_delta, current_setup), needs, track_id, ctx, [],
        )
        base_units = _initial_units(initial_delta, current_setup)
        flat = satisfaction_by_class(base_units, needs, track_id, ctx, [])
        return OptimizeResult(
            delta=dict(initial_delta or {f.name: 0.0 for f in ALL_SETUP_FIELDS}),
            units=base_units,
            before=identity,
            after=identity,
            trace=[],
            per_class_gain={},
            satisfaction_before=flat,
            satisfaction_after=flat,
        )

    # 多起点：规则引擎的解 + 零改动，取更优者
    starts = [_initial_units(initial_delta, current_setup)]
    if initial_delta:
        starts.append(_initial_units(None, current_setup))

    best_units: dict[str, float] | None = None
    best_trace: list[str] = []
    best_after: ObjectiveBreakdown | None = None
    for start in starts:
        units, trace, after = _ascend(
            start, needs, ctx, corners, track_id, current_setup,
        )
        if best_after is None or after.total < best_after.total - _MIN_GAIN:
            best_units, best_trace, best_after = units, trace, after

    assert best_units is not None and best_after is not None

    # 二级精修：粗搜用的是 0.125 的网格，最优点可能落在格点之间。
    # 从粗搜最优点出发、用半格长（0.0625）再跑一轮，只接受更优解 ——
    # 这一步**只会变好、不会变差**（严格比较 total 才替换）。
    fine_units, fine_trace, fine_after = _ascend(
        best_units, needs, ctx, corners, track_id, current_setup,
        grid=_GRID_FINE, max_passes=_MAX_PASSES_FINE,
    )
    if fine_after.total < best_after.total - _MIN_GAIN:
        best_units, best_after = fine_units, fine_after
        best_trace = [*best_trace, *fine_trace]
    before = objective(
        _initial_units(initial_delta, current_setup), needs, track_id, ctx, corners,
    )
    per_class_gain = {
        k: round(before.per_class.get(k, 0.0) - best_after.per_class.get(k, 0.0), 6)
        for k in ("slow", "medium", "fast")
    }
    settle_before = _initial_units(initial_delta, current_setup)
    return OptimizeResult(
        delta=units_to_delta(best_units, current_setup),
        units=best_units,
        before=before,
        after=best_after,
        trace=best_trace,
        per_class_gain=per_class_gain,
        satisfaction_before=satisfaction_by_class(
            settle_before, needs, track_id, ctx, corners,
        ),
        satisfaction_after=satisfaction_by_class(
            best_units, needs, track_id, ctx, corners,
        ),
    )


def describe_tradeoff(result: OptimizeResult) -> list[str]:
    """把优化结果翻译成可读的取舍说明。"""
    notes: list[str] = []
    gains = result.per_class_gain
    winners = sorted(
        (k for k, v in gains.items() if v > 1e-4), key=lambda k: -gains[k],
    )
    labels = {"slow": "慢弯区", "medium": "中速弯区", "fast": "快弯区"}
    if winners:
        notes.append(
            "整圈最优解相对规则建议的改善集中于："
            + "、".join(f"{labels[k]}(−{gains[k]:.4f} 残差)" for k in winners)
        )
    losers = sorted(
        (k for k, v in gains.items() if v < -1e-4), key=lambda k: gains[k],
    )
    if losers:
        notes.append(
            "为避免整圈净亏，以下区域做了让步（残差回升）："
            + "、".join(f"{labels[k]}(+{-gains[k]:.4f})" for k in losers)
        )
    b, a = result.before, result.after
    notes.append(
        f"代价结构：阻力 {b.drag_cost:.4f}→{a.drag_cost:.4f}、"
        f"刮底 {b.bottoming_cost:.4f}→{a.bottoming_cost:.4f}、"
        f"胎温 {b.tyre_heat_cost:.4f}→{a.tyre_heat_cost:.4f}"
    )
    return notes
