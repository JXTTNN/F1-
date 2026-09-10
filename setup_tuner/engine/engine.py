"""SetupDelta 计算引擎（核心，6 步确定性流水线）。

实现 design 2.7.4 的 ``SetupDelta = clamp(Dx × C)`` 计算流水线：

    1. 矩阵乘法：raw[p] = Σ_d Dx[d] × C[d][p]
    2. 遥测校准：raw[p] *= telemetry_gain[p]（默认 1.0）
    3. 单次上限约束：raw[p] = clip(raw[p], -max_delta[p], +max_delta[p])
    4. 合法区间约束：next[p] = clip(current[p] + raw[p], min[p], max[p])，
       Δ[p] = next[p] - current[p]
    5. 档位对齐：整数参数 round；浮点参数 round 到 step
    6. 输出 SetupDelta = {param: Δ_value}

纯函数、零 IO、零随机、零时间依赖，满足 FR-ENG-05 / FR-NFR-R1（可复现）。
任意单症状产出 SetupDelta 覆盖全部 23 参数；相同输入输出完全一致。
"""

from __future__ import annotations

from typing import Any

from setup_tuner.domain.setup import ALL_SETUP_FIELDS

from .confidence import assess_confidence
from .coupling import COUPLING_MATRIX, CouplingCell
from .diagnostic import (
    DIAG_DIMS,
    DIAG_DIMS_POSITIVE_SEMANTICS,
    DIAG_DIMS_ZH,
    compute_dx,
    is_zero_dx,
)


# ---------------------------------------------------------------------------
# 数值工具
# ---------------------------------------------------------------------------
def _clip(value: float, lo: float, hi: float) -> float:
    """将 value 裁剪到 [lo, hi] 区间。"""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _align_to_step(value: float, step: float, lo: float) -> float:
    """将 value 对齐到 step 档位（相对 lo 的整数倍）。

    Args:
        value: 待对齐值。
        step: 档位步长。
        lo: 区间下界（档位基准）。

    Returns:
        对齐到 step 档位的值。
    """
    if step <= 0:
        return value
    steps = round((value - lo) / step)
    return lo + steps * step


# ---------------------------------------------------------------------------
# 遥测校准增益提取（design 2.7.5）
# ---------------------------------------------------------------------------
def _derive_telemetry_gain(telemetry: dict[str, Any] | None) -> dict[str, float]:
    """从遥测字典提取每参数的幅度增益（参数级增益）。

    规则（design 2.7.5，仅实现参数级增益，Dx 级增益在 confidence 体现方向印证）：
        - 天气湿（``telemetry["weather"]`` 为 ``"wet"``/``"rainy"`` 或
          ``m_weather`` >= 1）→ 全参数 ×0.7（湿地下保守调整）；
        - 其余默认 1.0（不臆测）。

    Args:
        telemetry: 遥测字典；None 时返回全 1.0 增益。

    Returns:
        {param: gain} 字典，覆盖全部 23 参数。
    """
    gain = {f.name: 1.0 for f in ALL_SETUP_FIELDS}
    if not telemetry:
        return gain

    # 天气降幅：湿地下全参数保守调整 ×0.7
    weather = telemetry.get("weather") or telemetry.get("m_weather")
    is_wet = False
    if isinstance(weather, str):
        is_wet = weather.lower() in {"wet", "rainy", "rain", "drizzle"}
    elif isinstance(weather, (int, float)):
        # F1 UDP m_weather: 0=clear, 1=light rain, 2=heavy rain, 3=storm
        is_wet = weather >= 1
    if is_wet:
        for name in gain:
            gain[name] = 0.7

    return gain


# ---------------------------------------------------------------------------
# 核心：compute_setup_delta（6 步确定性流水线）
# ---------------------------------------------------------------------------
def compute_setup_delta(
    dx: dict[str, float],
    current_setup: dict[str, float],
    telemetry_gain: dict[str, float] | None = None,
) -> dict[str, float]:
    """执行完整的 6 步 SetupDelta 计算流水线。

    确定性纯函数：相同输入必得相同输出，无 IO、无随机、无时间依赖。

    Args:
        dx: 诊断向量字典 {dim_key: value}（9 维，来自 compute_dx）。
        current_setup: 当前调教快照 {param: value}（23 参数）。
        telemetry_gain: 每参数的遥测幅度增益 {param: gain}；
            None 时全部默认 1.0。

    Returns:
        SetupDelta 字典 {param: delta_value}，覆盖全部 23 参数。
        每参数 delta 满足：
            - |delta| <= max_delta[p]（单次上限）；
            - current[p] + delta ∈ [min[p], max[p]]（合法区间）；
            - delta 对齐到 step 档位。

    Raises:
        KeyError: current_setup 缺失某参数。
    """
    if telemetry_gain is None:
        telemetry_gain = {f.name: 1.0 for f in ALL_SETUP_FIELDS}

    setup_delta: dict[str, float] = {}

    for spec in ALL_SETUP_FIELDS:
        p = spec.name

        # 步骤 1：矩阵乘法 raw[p] = Σ_d Dx[d] × C[d][p]
        raw = 0.0
        for dim in DIAG_DIMS:
            dx_val = dx.get(dim, 0.0)
            if dx_val == 0.0:
                continue
            cell: CouplingCell | None = COUPLING_MATRIX[dim][p]
            if cell is None:
                continue
            raw += dx_val * cell.value

        # 步骤 2：遥测校准 raw[p] *= telemetry_gain[p]
        gain = telemetry_gain.get(p, 1.0)
        raw *= gain

        # 步骤 3：单次上限约束 raw[p] = clip(raw, -max_delta, +max_delta)
        raw = _clip(raw, -spec.max_delta, spec.max_delta)

        # 步骤 4：合法区间约束
        current = float(current_setup[p])
        next_val = _clip(current + raw, spec.min_val, spec.max_val)


        # 步骤 5：档位对齐（对 delta 对齐到 step，再回推 next）
        # 先对 next 对齐档位，再算 delta，保证 next 与 delta 都在档位上
        next_aligned = _align_to_step(next_val, spec.step, spec.min_val)
        next_aligned = _clip(next_aligned, spec.min_val, spec.max_val)
        delta_aligned = next_aligned - current

        # 整数参数（step >= 1 且为整数）round 到整数
        if spec.step >= 1.0 and float(spec.step).is_integer():
            delta_aligned = float(round(delta_aligned))

        # 步骤 6：输出
        setup_delta[p] = delta_aligned

    return setup_delta


# ---------------------------------------------------------------------------
# 完整建议生成
# ---------------------------------------------------------------------------
# 简单的副作用提示表（参数 → 方向 → tradeoff 文案）
_TRADEOFF_NOTES: dict[str, dict[str, str]] = {
    "front_wing": {
        "increase": "可能牺牲直道极速",
        "decrease": "可能提升直道极速但降低前轴下压力",
    },
    "rear_wing": {
        "increase": "可能牺牲直道极速",
        "decrease": "可能提升直道极速但降低后轴下压力",
    },
    "brake_pressure": {
        "increase": "可能增加轮胎锁死风险",
        "decrease": "可能延长制动距离",
    },
    "front_ride_height": {
        "increase": "可能降低前轴下压力中心",
        "decrease": "可能增加刮底风险",
    },
    "rear_ride_height": {
        "increase": "可能降低后轴下压力中心",
        "decrease": "可能增加刮底风险",
    },
    "on_throttle_diff": {
        "decrease": "可能增加出弯转向过度倾向",
    },
    "front_spring": {
        "decrease": "可能增加车身侧倾",
        "increase": "可能降低机械抓地",
    },
    "rear_spring": {
        "decrease": "可能增加车身侧倾",
        "increase": "可能降低机械抓地",
    },
}


def _build_param_detail(
    spec_name: str,
    current: float,
    delta: float,
    dx: dict[str, float],
) -> dict[str, Any]:
    """组装单参数的报告详情（联动说明 / 出处 / tradeoff）。"""
    # 收集贡献该参数的诊断维度
    linkages: list[str] = []
    sources: list[str] = []
    for dim in DIAG_DIMS:
        dx_val = dx.get(dim, 0.0)
        if dx_val == 0.0:
            continue
        cell: CouplingCell | None = COUPLING_MATRIX[dim][spec_name]
        if cell is None:
            continue
        linkages.append(
            f"{dim}({DIAG_DIMS_ZH[dim]}) Dx={dx_val:+.2f} × C={cell.value:+.2f}"
        )
        if cell.source not in sources:
            sources.append(cell.source)

    # 联动说明（中文）
    if linkages:
        linked_notes = "、".join(
            f"{DIAG_DIMS_POSITIVE_SEMANTICS.get(dim, dim)}"
            for dim in DIAG_DIMS
            if dx.get(dim, 0.0) != 0.0
            and COUPLING_MATRIX[dim][spec_name] is not None
        )
        if not linked_notes:
            linked_notes = "由多个诊断维度联动调整"
    else:
        linked_notes = "本次无需调整"

    # 主出处（多出处用逗号 join）
    source = ",".join(sources) if sources else ""

    # tradeoff
    tradeoff: str | None = None
    if abs(delta) > 1e-12:
        direction = "increase" if delta > 0 else "decrease"
        tradeoff = _TRADEOFF_NOTES.get(spec_name, {}).get(direction)

    return {
        "param": spec_name,
        "current": current,
        "next": current + delta,
        "setup_delta": delta,
        "linkages": linkages,
        "linked_notes": linked_notes,
        "source": source,
        "tradeoff": tradeoff,
    }


def generate_suggestion(
    symptoms: list[tuple[str, int]],
    current_setup: dict[str, float],
    track_id: str,
    telemetry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """完整建议生成（Dx → SetupDelta → 报告组装）。

    确定性纯函数：相同输入必得相同输出，无 IO、无随机、无时间依赖。
    时间戳由报告落库层（T7）在持久化时补充，本函数不引入时间依赖。

    Args:
        symptoms: 症状列表 [(symptom_key, strength), ...]。
        current_setup: 当前调教快照 {param: value}（23 参数）。
        track_id: 赛道标识。
        telemetry: 遥测客观数据（用于校准增益与置信度）；None 表示无遥测。

    Returns:
        建议报告字典，结构对齐 design 2.7.7：
        ::
            {
              "track_id": str,
              "dx": {dim: value},
              "setup_delta": {param: delta},
              "parameters": [param_detail, ...],   # 23 项
              "confidence": "high|medium|low",
              "summary": str
            }
    """
    # ① 症状 → Dx
    dx = compute_dx(symptoms)

    # ② 遥测校准增益
    telemetry_gain = _derive_telemetry_gain(telemetry)

    # ③ SetupDelta = clamp(Dx × C)
    setup_delta = compute_setup_delta(dx, current_setup, telemetry_gain)

    # ④ 逐参数报告详情
    parameters: list[dict[str, Any]] = []
    for spec in ALL_SETUP_FIELDS:
        current = float(current_setup[spec.name])
        delta = setup_delta[spec.name]
        parameters.append(_build_param_detail(spec.name, current, delta, dx))

    # ⑤ 置信度
    confidence = assess_confidence(symptoms, telemetry)

    # ⑥ 摘要
    nonzero_count = sum(1 for d in setup_delta.values() if abs(d) > 1e-12)
    if is_zero_dx(dx):
        summary = "未检测到有效症状，本次无调整建议"
    else:
        summary = (
            f"本次建议共关联 {len(parameters)} 项参数，"
            f"其中 {nonzero_count} 项非零调整，整体性调教"
        )

    return {
        "track_id": track_id,
        "dx": dx,
        "setup_delta": setup_delta,
        "parameters": parameters,
        "confidence": confidence,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# 确定性自校验（任意单症状覆盖全部 23 参数）
# ---------------------------------------------------------------------------
def validate_engine() -> None:
    """构建期校验引擎确定性与覆盖性。

    校验项：
        1. 任意单症状（强度 3）产出 SetupDelta 覆盖全部 23 参数；
        2. 确定性：相同输入跑两次结果完全一致；
        3. 不越界：current + delta ∈ [min, max] 且 |delta| <= max_delta；
        4. 每参数非零 delta 的 source 非空。

    Raises:
        AssertionError: 任一校验不通过。
    """
    from setup_tuner.domain.setup import CarSetup

    default_setup = CarSetup.default().to_dict()

    for symptom in (s.value for s in __import__("setup_tuner.domain.symptoms", fromlist=["Symptom"]).Symptom):
        # 单症状强度 3
        result1 = generate_suggestion(
            [(symptom, 3)], default_setup, "validate_track", None
        )
        result2 = generate_suggestion(
            [(symptom, 3)], default_setup, "validate_track", None
        )

        # 1. 覆盖全部 23 参数
        assert set(result1["setup_delta"].keys()) == set(default_setup.keys()), (
            f"症状 {symptom!r} SetupDelta 未覆盖全部 23 参数"
        )

        # 2. 确定性
        assert result1["setup_delta"] == result2["setup_delta"], (
            f"症状 {symptom!r} 两次运行结果不一致（非确定性）"
        )

        # 3. 不越界 + 4. 非零 delta 有出处
        for spec in ALL_SETUP_FIELDS:
            p = spec.name
            current = default_setup[p]
            delta = result1["setup_delta"][p]
            next_val = current + delta
            assert next_val >= spec.min_val - 1e-6, (
                f"症状 {symptom!r} 参数 {p!r} next={next_val} < min={spec.min_val}"
            )
            assert next_val <= spec.max_val + 1e-6, (
                f"症状 {symptom!r} 参数 {p!r} next={next_val} > max={spec.max_val}"
            )
            assert abs(delta) <= spec.max_delta + 1e-6, (
                f"症状 {symptom!r} 参数 {p!r} |delta|={abs(delta)} > max_delta={spec.max_delta}"
            )
            # 非零 delta 对应参数详情 source 非空
            if abs(delta) > 1e-6:
                detail = next(pd for pd in result1["parameters"] if pd["param"] == p)
                assert detail["source"], (
                    f"症状 {symptom!r} 参数 {p!r} 非零 delta 但 source 为空"
                )