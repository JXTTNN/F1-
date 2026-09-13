"""SetupDelta 计算引擎（核心，6 步确定性流水线 + 混合模型）。

实现 design 2.7.4 的 ``SetupDelta = clamp(Dx × C)`` 计算流水线：

    1. 矩阵乘法：raw[p] = Σ_d Dx[d] × C[d][p]
    2. 遥测校准：raw[p] *= telemetry_gain[p]（默认 1.0）
    3. 单次上限约束：raw[p] = clip(raw[p], -max_delta[p], +max_delta[p])
    4. 合法区间约束：next[p] = clip(current[p] + raw[p], min[p], max[p])，
       Δ[p] = next[p] - current[p]
    5. 档位对齐：整数参数 round；浮点参数 round 到 step
    6. 输出 SetupDelta = {param: Δ_value}

纯函数、零 IO、零随机、零时间依赖，满足 FR-ENG-05 / FR-NFR-R1（可复现）。
任意单症状产出 SetupDelta 覆盖全部 20 参数；相同输入输出完全一致。

混合模型扩展（task-43）：
    ``generate_suggestion`` 支持 ``model_type`` 参数：
        - ``"rule"``：纯规则引擎（默认确定性流水线）
        - ``"nn"``：纯神经网络（PyTorch 不可用时自动降级为规则引擎）
        - ``"hybrid"``：混合模型（规则 60% + 神经网络 40%，神经网络不可用时降级）

    神经网络分支由 :mod:`setup_tuner.engine.nn_model` 提供，
    PyTorch 为可选依赖，不可用时自动降级为纯规则引擎。
"""

from __future__ import annotations

from typing import Any

from setup_tuner.domain.setup import ALL_SETUP_FIELDS

from .confidence import assess_confidence
from .coupling import nonzero_cells_for_param_cached
from .diagnostic import (
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
def _is_wet_weather(telemetry: dict[str, Any]) -> bool:
    """判断遥测指示湿滑天气。"""
    weather = telemetry.get("weather") or telemetry.get("m_weather")
    if isinstance(weather, str):
        return weather.lower() in {"wet", "rainy", "rain", "drizzle"}
    if isinstance(weather, (int, float)):
        # F1 UDP m_weather: 0=clear, 1=4light rain, 2=heavy rain, 3=storm
        return weather >= 1
    return False


def _apply_tyre_temp_gain(telemetry: dict[str, Any], gain: dict[str, float]) -> None:
    """根据胎温校准胎压参数增益（原地修改 gain）。"""
    tyre_temps = telemetry.get("m_tyresSurfaceTemperature")
    if not (isinstance(tyre_temps, list) and len(tyre_temps) >= 4):
        return
    avg_tyre_temp = sum(tyre_temps[:4]) / 4.0
    tyre_pressure_params = (
        "front_left_tyre_pressure", "front_right_tyre_pressure",
        "rear_left_tyre_pressure", "rear_right_tyre_pressure",
    )
    if avg_tyre_temp > 100.0:
        # 胎温过高 → 胎压调整更敏感
        for p in tyre_pressure_params:
            gain[p] *= 1.3
    elif avg_tyre_temp < 80.0:
        # 胎温过低 → 胎压调整保守
        for p in tyre_pressure_params:
            gain[p] *= 0.8


def _apply_throttle_brake_gain(telemetry: dict[str, Any], gain: dict[str, float]) -> None:
    """根据油门/刹车均值校准差速器/刹车参数增益（原地修改 gain）。"""
    throttle = telemetry.get("m_throttle")
    if isinstance(throttle, (int, float)) and throttle > 0.7:
        for p in ("on_throttle_diff", "off_throttle_diff"):
            gain[p] *= 1.2
    brake = telemetry.get("m_brake")
    if isinstance(brake, (int, float)) and brake > 0.5:
        for p in ("brake_pressure", "brake_bias"):
            gain[p] *= 1.2


def _derive_telemetry_gain(telemetry: dict[str, Any] | None) -> dict[str, float]:
    """从遥测字典提取每参数的幅度增益（参数级增益）。

    规则（扩展版，利用更多遥测数据校准）：
        - 天气湿（``telemetry["weather"]`` 为 ``"wet"``/``"rainy"`` 或
          ``m_weather`` >= 1）→ 全参数 ×0.7（湿地下保守调整）；
        - 胎温过高（``m_tyresSurfaceTemperature`` 均值 > 100°C）→ 胎压参数 ×1.3
          （胎温过高时胎压调整更敏感）；
        - 胎温过低（``m_tyresSurfaceTemperature`` 均值 < 80°C）→ 胎压参数 ×0.8
          （胎温过低时胎压调整保守）；
        - 油门均值高（``m_throttle`` 均值 > 0.7）→ 差速器参数 ×1.2
          （高油门占比时差速器调整更关键）；
        - 刹车均值高（``m_brake`` 均值 > 0.5）→ 刹车参数 ×1.2
          （高刹车占比时刹车调整更关键）；
        - 其余默认 1.0（不臆测）。

    Args:
        telemetry: 遥测字典；None 时返回全 1.0 增益。

    Returns:
        {param: gain} 字典，覆盖全部 20 参数。
    """
    gain = {f.name: 1.0 for f in ALL_SETUP_FIELDS}
    if not telemetry:
        return gain

    if _is_wet_weather(telemetry):
        for name in gain:
            gain[name] = 0.7
    _apply_tyre_temp_gain(telemetry, gain)
    _apply_throttle_brake_gain(telemetry, gain)
    return gain


# ---------------------------------------------------------------------------
# 核心：compute_setup_delta（6 步确定性流水线）
# ---------------------------------------------------------------------------
def _compute_param_raw_delta(
    p: str, dx: dict[str, float], telemetry_gain: dict[str, float],
) -> float:
    """步骤 1-2：矩阵乘法 + 遥测校准，返回校准后 raw delta。"""
    raw = 0.0
    for cell in nonzero_cells_for_param_cached(p):
        dx_val = dx.get(cell.diag, 0.0)
        if dx_val == 0.0:
            continue
        raw += dx_val * cell.value
    gain = telemetry_gain.get(p, 1.0)
    return raw * gain


def _align_param_delta(spec: Any, raw: float, current: float) -> float:
    """步骤 3-6：单次上限 + 合法区间 + 档位对齐，返回最终 delta。"""
    raw = _clip(raw, -spec.max_delta, spec.max_delta)
    next_val = _clip(current + raw, spec.min_val, spec.max_val)
    next_aligned = _align_to_step(next_val, spec.step, spec.min_val)
    next_aligned = _clip(next_aligned, spec.min_val, spec.max_val)
    delta_aligned = next_aligned - current
    # 整数参数（step >= 1 且为整数）round 到整数
    if spec.step >= 1.0 and float(spec.step).is_integer():
        delta_aligned = float(round(delta_aligned))
    return delta_aligned


def compute_setup_delta(
    dx: dict[str, float],
    current_setup: dict[str, float],
    telemetry_gain: dict[str, float] | None = None,
) -> dict[str, float]:
    """执行完整的 6 步 SetupDelta 计算流水线。

    确定性纯函数：相同输入必得相同输出，无 IO、无随机、无时间依赖。

    Args:
        dx: 诊断向量字典 {dim_key: value}（9 维，来自 compute_dx）。
        current_setup: 当前调教快照 {param: value}（20 参数）。
        telemetry_gain: 每参数的遥测幅度增益 {param: gain}；
            None 时全部默认 1.0。

    Returns:
        SetupDelta 字典 {param: delta_value}，覆盖全部 20 参数。
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
        raw = _compute_param_raw_delta(p, dx, telemetry_gain)
        current = float(current_setup[p])
        setup_delta[p] = _align_param_delta(spec, raw, current)
    return setup_delta


# ---------------------------------------------------------------------------
# 完整建议生成
# ---------------------------------------------------------------------------
# 完整的参数 tradeoff 提示表（参数 → 方向 → tradeoff 文案）
_TRADEOFF_NOTES: dict[str, dict[str, str]] = {
    "front_wing": {
        "increase": "可能牺牲直道极速",
        "decrease": "可能提升直道极速但降低前轴下压力",
    },
    "rear_wing": {
        "increase": "可能牺牲直道极速",
        "decrease": "可能提升直道极速但降低后轴下压力",
    },
    "on_throttle_diff": {
        "increase": "可能增加出弯转向不足倾向",
        "decrease": "可能增加出弯转向过度倾向",
    },
    "off_throttle_diff": {
        "increase": "可能增加入弯转向过度倾向",
        "decrease": "可能增加入弯转向不足倾向",
    },
    "front_camber": {
        "increase": "可能增加直道轮胎内缘磨耗",
        "decrease": "可能降低弯中前轮抓地",
    },
    "rear_camber": {
        "increase": "可能增加直道轮胎内缘磨耗",
        "decrease": "可能降低弯中后轮抓地",
    },
    "front_toe": {
        "increase": "可能增加直道轮胎磨耗和阻力",
        "decrease": "可能降低前轴指向精度",
    },
    "rear_toe": {
        "increase": "可能增加直道轮胎磨耗和阻力",
        "decrease": "可能降低后轴稳定性",
    },
    "front_suspension": {
        "increase": "可能降低机械抓地但增高速稳定性",
        "decrease": "可能增加车身侧倾但增机械抓地",
    },
    "rear_suspension": {
        "increase": "可能降低机械抓地但增高速稳定性",
        "decrease": "可能增加车身侧倾但增机械抓地",
    },
    "front_anti_roll_bar": {
        "increase": "可能降低前轴独立抓地但增侧倾刚度",
        "decrease": "可能增加车身侧倾但增前轴独立抓地",
    },
    "rear_anti_roll_bar": {
        "increase": "可能降低后轴独立抓地但增侧倾刚度",
        "decrease": "可能增加车身侧倾但增后轴独立抓地",
    },
    "front_ride_height": {
        "increase": "可能降低前轴下压力中心",
        "decrease": "可能增加刮底风险",
    },
    "rear_ride_height": {
        "increase": "可能降低后轴下压力中心",
        "decrease": "可能增加刮底风险",
    },
    "brake_pressure": {
        "increase": "可能增加轮胎锁死风险",
        "decrease": "可能延长制动距离",
    },
    "brake_bias": {
        "increase": "可能增加前轮锁死倾向",
        "decrease": "可能增加后轮锁死倾向",
    },
    "front_left_tyre_pressure": {
        "increase": "可能减小轮胎接触面积但增响应",
        "decrease": "可能增大轮胎接触面积但降响应",
    },
    "front_right_tyre_pressure": {
        "increase": "可能减小轮胎接触面积但增响应",
        "decrease": "可能增大轮胎接触面积但降响应",
    },
    "rear_left_tyre_pressure": {
        "increase": "可能减小轮胎接触面积但增响应",
        "decrease": "可能增大轮胎接触面积但降响应",
    },
    "rear_right_tyre_pressure": {
        "increase": "可能减小轮胎接触面积但增响应",
        "decrease": "可能增大轮胎接触面积但降响应",
    },
}


def _collect_param_linkages(
    spec_name: str, dx: dict[str, float],
) -> tuple[list[str], list[str]]:
    """收集参数的诊断维度联动描述与出处列表。"""
    linkages: list[str] = []
    sources: list[str] = []
    for cell in nonzero_cells_for_param_cached(spec_name):
        dx_val = dx.get(cell.diag, 0.0)
        if dx_val == 0.0:
            continue
        linkages.append(
            f"{cell.diag}({DIAG_DIMS_ZH[cell.diag]}) Dx={dx_val:+.2f} × C={cell.value:+.2f}",
        )
        if cell.source not in sources:
            sources.append(cell.source)
    return linkages, sources


def _build_linked_notes(spec_name: str, dx: dict[str, float], linkages: list[str]) -> str:
    """构造参数的中文联动说明。"""
    if not linkages:
        return "本次无需调整"
    linked_notes = "、".join(
        f"{DIAG_DIMS_POSITIVE_SEMANTICS.get(cell.diag, cell.diag)}"
        for cell in nonzero_cells_for_param_cached(spec_name)
        if dx.get(cell.diag, 0.0) != 0.0
    )
    return linked_notes or "由多个诊断维度联动调整"


def _build_param_detail(
    spec_name: str,
    current: float,
    delta: float,
    dx: dict[str, float],
) -> dict[str, Any]:
    """组装单参数的报告详情（联动说明 / 出处 / tradeoff）。"""
    linkages, sources = _collect_param_linkages(spec_name, dx)
    linked_notes = _build_linked_notes(spec_name, dx, linkages)
    source = ",".join(sources) if sources else ""

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


def _compute_nn_delta(
    model_type: str, symptoms: list, dx: dict, current_setup: dict[str, float],
    track_id: str,
) -> tuple[dict[str, float] | None, bool]:
    """计算神经网络 delta（若需要且可用），返回 (nn_delta, nn_available)。"""
    if model_type not in ("nn", "hybrid"):
        return None, False
    nn_manager = _get_nn_manager()
    if nn_manager is None or not nn_manager.available:
        return None, False
    nn_delta = nn_manager.predict(symptoms, dx, current_setup, track_id)
    return nn_delta, True


def _build_param_details(
    final_delta: dict[str, float], current_setup: dict[str, float], dx: dict,
) -> list[dict[str, Any]]:
    """逐参数构造报告详情列表。"""
    parameters: list[dict[str, Any]] = []
    for spec in ALL_SETUP_FIELDS:
        current = float(current_setup[spec.name])
        delta = final_delta[spec.name]
        parameters.append(_build_param_detail(spec.name, current, delta, dx))
    return parameters


def _build_suggestion_summary(
    final_delta: dict[str, float], dx: dict, parameters: list[dict[str, Any]],
) -> str:
    """构造建议摘要文本。"""
    nonzero_count = sum(1 for d in final_delta.values() if abs(d) > 1e-12)
    if is_zero_dx(dx):
        return "未检测到有效症状，本次无调整建议"
    return (
        f"本次建议共关联 {len(parameters)} 项参数，"
        f"其中 {nonzero_count} 项非零调整，整体性调教"
    )


_GENERATE_SUGGESTION_DOC = """完整建议生成（Dx → SetupDelta → 报告组装）。

确定性纯函数：相同输入必得相同输出，无 IO、无随机、无时间依赖。
时间戳由报告落库层（T7）在持久化时补充，本函数不引入时间依赖。

混合模型（task-43）：
    ``model_type`` 控制使用哪种模型分支：
        - ``"rule"``：纯规则引擎（6 步确定性流水线）
        - ``"nn"``：纯神经网络（不可用时降级为规则引擎）
        - ``"hybrid"``：混合（规则 60% + 神经网络 40%，不可用时降级）

    神经网络分支为可选依赖（PyTorch），不可用时自动降级为纯规则引擎，
    保证向后兼容。

阶段敏感扩展（task-60）：
    ``symptoms`` 支持二元组和三元组混合：
        - 二元组 ``(symptom, strength)``：使用症状的默认阶段（向后兼容）；
        - 三元组 ``(symptom, strength, stage)``：使用指定阶段的 Dx 映射。

Args:
    symptoms: 症状列表，每项为二元组或三元组：
        - ``(symptom_key, strength)``：使用默认阶段；
        - ``(symptom_key, strength, stage)``：使用指定阶段。
    current_setup: 当前调教快照 {param: value}（20 参数）。
    track_id: 赛道标识。
    telemetry: 遥测客观数据（用于校准增益与置信度）；None 表示无遥测。
    model_type: 模型类型，``"rule"`` | ``"nn"`` | ``"hybrid"``。
        默认 ``"hybrid"``。未知值按 ``"rule"`` 处理。

Returns:
    建议报告字典，结构对齐 design 2.7.7：
    ::
        {{
          "track_id": str,
          "dx": {{dim: value}},
          "setup_delta": {{param: delta}},
          "parameters": [param_detail, ...],   # 20 项
          "confidence": "high|medium|low",
          "summary": str,
          "model_type": str,            # 实际使用的模型类型
          "nn_available": bool,         # 神经网络是否可用
        }}
"""


def generate_suggestion(
    symptoms: list[tuple[str, int]] | list[tuple[str, int, str]],
    current_setup: dict[str, float],
    track_id: str,
    telemetry: dict[str, Any] | None = None,
    model_type: str = "hybrid",
) -> dict[str, Any]:
    """完整建议生成（Dx → SetupDelta → 报告组装）。详见模块级文档。"""
    dx = compute_dx(symptoms)
    telemetry_gain = _derive_telemetry_gain(telemetry)
    rule_delta = compute_setup_delta(dx, current_setup, telemetry_gain)

    nn_delta, nn_available = _compute_nn_delta(
        model_type, symptoms, dx, current_setup, track_id,
    )
    final_delta, actual_model_type = _blend_delta(
        rule_delta, nn_delta, model_type, nn_available,
    )
    parameters = _build_param_details(final_delta, current_setup, dx)
    confidence = assess_confidence(symptoms, telemetry)
    summary = _build_suggestion_summary(final_delta, dx, parameters)

    return {
        "track_id": track_id,
        "dx": dx,
        "setup_delta": final_delta,
        "parameters": parameters,
        "confidence": confidence,
        "summary": summary,
        "model_type": actual_model_type,
        "nn_available": nn_available,
    }


generate_suggestion.__doc__ = _GENERATE_SUGGESTION_DOC


def _blend_delta(
    rule_delta: dict[str, float],
    nn_delta: dict[str, float] | None,
    model_type: str,
    nn_available: bool,
) -> tuple[dict[str, float], str]:
    """根据 model_type 混合规则引擎与神经网络结果。

    Args:
        rule_delta: 规则引擎的 SetupDelta。
        nn_delta: 神经网络的 SetupDelta（None 表示不可用）。
        model_type: 请求的模型类型。
        nn_available: 神经网络是否可用。

    Returns:
        (final_delta, actual_model_type) 二元组。
        actual_model_type 为实际使用的模型类型（可能因降级而与请求不同）。
    """
    # 权重：hybrid 模式下规则 60% + 神经网络 40%
    RULE_WEIGHT = 0.6
    NN_WEIGHT = 0.4

    if model_type == "nn" and nn_delta is not None:
        # 纯神经网络模式
        return nn_delta, "nn"

    if model_type == "hybrid" and nn_delta is not None:
        # 混合模式：加权平均
        blended = {
            p: rule_delta[p] * RULE_WEIGHT + nn_delta[p] * NN_WEIGHT
            for p in rule_delta
        }
        return blended, "hybrid"

    # 降级为纯规则引擎
    if model_type in ("nn", "hybrid") and not nn_available:
        # 请求了 nn/hybrid 但神经网络不可用，降级
        return rule_delta, "rule"
    # model_type == "rule" 或未知值
    return rule_delta, "rule"


# ---------------------------------------------------------------------------
# 神经网络管理器单例（延迟加载，PyTorch 不可用时返回 None）
# ---------------------------------------------------------------------------
_NN_MANAGER: Any = None


def _get_nn_manager() -> Any:
    """获取神经网络模型管理器单例（延迟加载）。

    PyTorch 不可用或加载失败时返回 None，引擎自动降级为纯规则引擎。

    Returns:
        NNModelManager 实例或 None。
    """
    global _NN_MANAGER
    if _NN_MANAGER is None:
        try:
            from .nn_model import NNModelManager
            _NN_MANAGER = NNModelManager()
        except Exception:
            _NN_MANAGER = None
    return _NN_MANAGER


def reset_nn_manager() -> None:
    """重置神经网络管理器单例（供测试使用）。"""
    global _NN_MANAGER
    _NN_MANAGER = None


# ---------------------------------------------------------------------------
# 确定性自校验（任意单症状覆盖全部 20 参数）
# ---------------------------------------------------------------------------
def _validate_symptom_invariants(
    symptom: str, result1: dict, result2: dict, default_setup: dict,
) -> None:
    """校验单症状的覆盖性、确定性、不越界、出处完整性。"""
    # 1. 覆盖全部 20 参数
    assert set(result1["setup_delta"].keys()) == set(default_setup.keys()), (
        f"症状 {symptom!r} SetupDelta 未覆盖全部 20 参数"
    )
    # 2. 确定性
    assert result1["setup_delta"] == result2["setup_delta"], (
        f"症状 {symptom!r} 两次运行结果不一致（非确定性）"
    )
    # 3. 不越界 + 4. 非零 delta 有出处
    for spec in ALL_SETUP_FIELDS:
        _validate_param_in_bounds(symptom, spec, result1, default_setup)


def _validate_param_in_bounds(
    symptom: str, spec: Any, result1: dict, default_setup: dict,
) -> None:
    """校验单参数 next 在 [min, max] 且 |delta| <= max_delta，非零 delta 有出处。"""
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
    if abs(delta) > 1e-6:
        detail = next(pd for pd in result1["parameters"] if pd["param"] == p)
        assert detail["source"], (
            f"症状 {symptom!r} 参数 {p!r} 非零 delta 但 source 为空"
        )


def validate_engine() -> None:
    """构建期校验引擎确定性与覆盖性。

    校验项：
        1. 任意单症状（强度 3）产出 SetupDelta 覆盖全部 20 参数；
        2. 确定性：相同输入跑两次结果完全一致；
        3. 不越界：current + delta ∈ [min, max] 且 |delta| <= max_delta；
        4. 每参数非零 delta 的 source 非空。

    Raises:
        AssertionError: 任一校验不通过。
    """
    from setup_tuner.domain.setup import CarSetup
    from setup_tuner.domain.symptoms import Symptom

    default_setup = CarSetup.default().to_dict()

    for symptom in (s.value for s in Symptom):
        # 单症状强度 3，使用 rule 模式保持确定性自校验
        result1 = generate_suggestion(
            [(symptom, 3)], default_setup, "validate_track", None,
            model_type="rule",
        )
        result2 = generate_suggestion(
            [(symptom, 3)], default_setup, "validate_track", None,
            model_type="rule",
        )
        _validate_symptom_invariants(symptom, result1, result2, default_setup)