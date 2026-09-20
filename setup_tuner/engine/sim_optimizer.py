"""调教模拟优化器 —— 「参数矩阵给方向，神经网络不断模拟优化」的落地。

用户明确的优化方式
------------------
    1. **方向**：``engine.compute_setup_delta``（Dx × C 矩阵 6 步流水线）
       经 ``optimizer.optimize_setup`` + ``holistic.holistic_coherence`` 收口，
       给出初始调教改动 ``delta0``；
    2. **不断模拟优化**：从 ``delta0`` 出发做确定性坐标上升 —— 每次改动一个
       参数（固定顺序、固定步长），用**训练好的调教性能神经网络**
       （``engine.setup_sim``，遥测锚定仿真训练，纯标准库推理）预测该候选
       调教的圈速，保留预测更快的候选；反复若干轮直到不再改善。

接受判据（双条件，防退化）
--------------------------
对每个候选改动，**同时**满足才接受：

    a) 模拟圈速提升 ≥ ``min_gain_s``（默认 2ms）——
       神经网络主导的「不断模拟」；
    b) 手写圈级目标不显著恶化（容差 ``obj_tol``）——
       保证车手反馈的需求满足度（Dx 残差）与显式代价（阻力/刮底/胎温/
       改动幅度）不被模拟优化悄悄牺牲 —— 「整体性」的守门员。

安全性
------
- 模型不可用 / 赛道不在覆盖范围 → **原样返回**（``available=False``），
  与纯规则引擎行为逐位一致（未知赛道/合成 id 的既有测试不受影响）；
- 完全确定性：无随机、无时间依赖；参数固定顺序、候选固定顺序；
- 改动受 ``max_delta`` 与参数上下限约束，并按 ``step`` 档位对齐；
- 左右胎压保持一致（跟随左胎压）。

报告
----
返回 :class:`SimRefineResult`：最终 delta、逐次接受记录（trace）、
模拟前后圈速增量、迭代次数 —— 引擎写进报告的 ``holistic.simulation``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from setup_tuner.domain.setup import ALL_SETUP_FIELDS, get_field

from .lap_model import (
    OPTIMIZABLE,
    corner_evaluations,
    needs_by_class_from_dx,
    objective,
    track_context,
    units_from_delta,
)
from .setup_sim import get_setup_sim, normalize_setup

#: 坐标上升步长（占该参数 max_delta 的比例；与 optimizer._GRID 同口径量级）
_GRID = 0.25
#: 最大轮数（每轮遍历全部可优化参数）
_MAX_PASSES = 3
#: 接受阈值：模拟圈速提升小于该值视为噪声（秒）。
#: 标定依据（2026-09-19 实测）：模型对「相邻候选的差异」预测误差中位数
#: ~15ms；只接受 ≥20ms 的预测提升，落在模型可靠区间内（>30ms 的移动
#: 方向正确率 95%），避免坐标上升追逐噪声。
_MIN_GAIN_S = 0.020
#: 目标代价兑换率（秒 / 目标单位）：模拟优化必须「付得起」它造成的目标代价 ——
#: 接受条件为 ``模拟提升 ≥ max(_MIN_GAIN_S, _OBJ_EXCHANGE_S × 目标代价增量)``。
#: 目标单位 = 手写圈级目标（需求残差 + 阻力/刮底/胎温/改动幅度代价）的总和。
#: 标定：单参数整幅改动的 effort 代价 = 0.02 → 需 ≥100ms 模拟收益才动整幅；
#: 需求残差增加 0.01（缺口显著扩大）→ 需 ≥50ms 收益。这防止模拟优化
#: 把车手反馈的需求满足度与安全性代价当空气。
_OBJ_EXCHANGE_S = 5.0
#: 整轮允许支付的目标代价总预算（所有被采纳改动累计）。
_OBJ_BUDGET = 0.08
#: 左右胎压跟随（右跟随左，保持 F1 调教的左右一致约定）
_PRESSURE_FOLLOW: tuple[tuple[str, str], ...] = (
    ("front_left_tyre_pressure", "front_right_tyre_pressure"),
    ("rear_left_tyre_pressure", "rear_right_tyre_pressure"),
)


@dataclass(slots=True)
class SimRefineResult:
    """模拟优化结果。"""

    delta: dict[str, float]
    available: bool
    reason: str = ""
    iterations: int = 0
    accepted: int = 0
    before_s: float | None = None
    after_s: float | None = None
    trace: list[str] = field(default_factory=list)

    @property
    def gain_s(self) -> float | None:
        """模拟圈速提升（秒；正 = 更快）。"""
        if self.before_s is None or self.after_s is None:
            return None
        return self.before_s - self.after_s


def _apply_step_alignment(name: str, value: float) -> float:
    """把目标值对齐到参数 step 档位（相对 min_val 的整数倍）。"""
    spec = get_field(name)
    step = spec.step or 0.01
    steps = round((value - spec.min_val) / step)
    aligned = spec.min_val + steps * step
    if spec.step >= 1.0 and float(spec.step).is_integer():
        aligned = float(round(aligned))
    return aligned


def _candidate_value(
    name: str, current_value: float, direction: int, grid: float,
) -> float | None:
    """生成一个候选真实值（含 max_delta 上限与上下限约束）；越界返回 None。"""
    spec = get_field(name)
    target = current_value + direction * grid * spec.max_delta
    aligned = _apply_step_alignment(name, target)
    delta = aligned - current_value
    if abs(delta) < 1e-9:
        return None
    if abs(delta) > spec.max_delta + 1e-9:
        return None
    if aligned < spec.min_val - 1e-9 or aligned > spec.max_val + 1e-9:
        return None
    return aligned


def sim_refine(
    delta0: dict[str, float],
    current_setup: dict[str, float],
    track_id: str,
    dx: dict[str, float] | None = None,
    telemetry: dict[str, Any] | None = None,
    wet: bool = False,
    grid: float = _GRID,
    max_passes: int = _MAX_PASSES,
) -> SimRefineResult:
    """从矩阵方向出发，用神经网络模拟循环精修调教（确定性）。

    Args:
        delta0: 初始改动（规则引擎 + 圈级优化 + 收口后的结果，真实单位）。
        current_setup: 当前调教（真实单位，20 参数）。
        track_id: 赛道标识；不在模型覆盖范围时原样返回。
        dx: 诊断向量（用于手写目标第二判据）；缺省时跳过第二判据。
        telemetry: 遥测摘要（参与手写目标上下文）。
        wet: 是否湿地（参与手写目标上下文）。
        grid: 坐标上升步长（占 max_delta 的比例）。
        max_passes: 最大轮数。

    Returns:
        :class:`SimRefineResult`；模型不可用/赛道未覆盖时 ``available=False``
        且 ``delta`` 与输入逐位一致。
    """
    model = get_setup_sim()
    if not model.available:
        return SimRefineResult(delta=dict(delta0), available=False,
                               reason=f"模型不可用：{model.reason}")
    if not model.covers(track_id):
        return SimRefineResult(
            delta=dict(delta0), available=False,
            reason=f"赛道 {track_id} 不在模型覆盖范围（{len(model.track_ids)} 赛道）",
        )

    # 初始状态
    cur: dict[str, float] = {f.name: float(delta0.get(f.name, 0.0)) for f in ALL_SETUP_FIELDS}

    def _sim_time_s(delta: dict[str, float]) -> float | None:
        trial_setup = {
            f.name: float(current_setup.get(f.name, f.default)) + delta.get(f.name, 0.0)
            for f in ALL_SETUP_FIELDS
        }
        return model.predict_delta_s(track_id, normalize_setup(trial_setup))

    def _obj_total(delta: dict[str, float]) -> float | None:
        if dx is None:
            return None
        ctx = track_context(track_id, telemetry, dx, wet=wet)
        corners = corner_evaluations(track_id)
        if not corners:
            return None
        units = units_from_delta(delta, current_setup)
        return objective(
            units, needs_by_class_from_dx(dx), track_id, ctx, corners,
        ).total

    before_s = _sim_time_s(cur)
    if before_s is None:
        return SimRefineResult(delta=dict(delta0), available=False,
                               reason="模型推理失败")
    cur_obj = _obj_total(cur)
    trace: list[str] = []
    accepted = 0
    iterations = 0
    used_budget = 0.0

    for _ in range(max_passes):
        moved = False
        for p in OPTIMIZABLE:
            spec = get_field(p)
            base = float(cur.get(p, 0.0))
            base_val = float(current_setup.get(p, spec.default))
            base_s = _sim_time_s(cur)
            if base_s is None:
                break
            for direction in (+1, -1):            # 固定顺序 → 确定性
                cand = _candidate_value(
                    p, base_val + base, direction, grid,
                )
                if cand is None:
                    continue
                new_delta = cand - base_val
                # 单次上限：总改动不得超过该参数的 max_delta（与规则流水线同约定）
                if abs(new_delta) > spec.max_delta + 1e-9:
                    continue
                trial = dict(cur)
                trial[p] = new_delta
                # 左右胎压一致：右胎压跟随左胎压
                for left, right in _PRESSURE_FOLLOW:
                    if p == left:
                        trial[right] = new_delta
                trial_s = _sim_time_s(trial)
                if trial_s is None:
                    continue
                sim_gain = base_s - trial_s
                if sim_gain < _MIN_GAIN_S:
                    continue
                # 目标代价必须「付得起」：模拟提升 ≥ 兑换率 × 代价增量，
                # 且整轮预算未超支（整体性守门，见模块文档）
                if cur_obj is not None:
                    trial_obj = _obj_total(trial)
                    if trial_obj is not None:
                        penalty = trial_obj - cur_obj
                        if penalty > 0.0:
                            if sim_gain < _OBJ_EXCHANGE_S * penalty:
                                continue
                            if used_budget + penalty > _OBJ_BUDGET:
                                continue
                        used_budget += max(0.0, penalty)
                        cur_obj = trial_obj
                # 接受
                gain_ms = sim_gain * 1000.0
                cur = trial
                accepted += 1
                moved = True
                trace.append(
                    f"{spec.label_zh if hasattr(spec, 'label_zh') else p}："
                    f"{base:+.3f} → {new_delta:+.3f}（模拟 −{gain_ms:.1f} ms）"
                )
                break
        iterations += 1
        if not moved:
            break

    after_s = _sim_time_s(cur)
    if after_s is None:
        after_s = before_s
    return SimRefineResult(
        delta=cur, available=True, reason="ok",
        iterations=iterations, accepted=accepted,
        before_s=before_s, after_s=after_s, trace=trace,
    )
