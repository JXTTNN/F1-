"""诊断向量 Dx（9 维 = 8 维能力需求 + 底盘离地）。

Dx 为 9 维**带符号实数行向量**，每维语义为「某种能力的需求量」，
**正值 = 该能力不足、需增强**（方向统一，便于与耦合矩阵符号相乘）。

9 个维度（逐字对齐 design.md 2.7.2）：
    1. front_grip_req      前轴抓地
    2. rear_grip_req       后轴抓地
    3. turnin_req          入弯响应
    4. hi_speed_stab_req   高速稳定性
    5. brake_stab_req      制动稳定性
    6. brake_power_req     制动力
    7. exit_traction_req   出弯牵引
    8. tyre_life_req       轮胎寿命
    9. ride_height_req     底盘离地

症状 → Dx 映射为确定性规则，多症状叠加时 Dx 分量**代数求和**；
「未点击弯道=正常」不产生贡献。

task-60 扩展：同一症状在不同弯道阶段（entry/apex/exit/global）有不同的
Dx 系数映射。``SYMPTOM_STAGE_TO_DX`` 为 ``(症状, 阶段) → Dx`` 映射，
``SYMPTOM_DEFAULT_STAGE`` 指定每个症状的默认阶段（向后兼容二元组调用）。
``SYMPTOM_TO_DX`` 从阶段映射自动生成，保持向后兼容。

出处：F1 25 官方调教指南（症状-机理对应章节）。
本模块为纯函数、零 IO、零随机，满足 FR-ENG-05 / FR-NFR-R1（可复现）。
"""

from __future__ import annotations

from collections.abc import Iterable

# ---------------------------------------------------------------------------
# 官方出处常量
# ---------------------------------------------------------------------------
SOURCE_DX = "F1 25 官方调教指南"

# 浮点零值判定 epsilon（用于 Dx 向量全零判断）
ZERO_DX_EPSILON = 1e-12


# ---------------------------------------------------------------------------
# 9 个诊断维度定义
# ---------------------------------------------------------------------------
# 维度 key 列表（固定顺序，作为向量下标基准）
DIAG_DIMS: list[str] = [
    "front_grip_req",       # 前轴抓地
    "rear_grip_req",        # 后轴抓地
    "turnin_req",           # 入弯响应
    "hi_speed_stab_req",    # 高速稳定性
    "brake_stab_req",       # 制动稳定性
    "brake_power_req",      # 制动力
    "exit_traction_req",    # 出弯牵引
    "tyre_life_req",        # 轮胎寿命
    "ride_height_req",      # 底盘离地
]

# 维度 key → 中文名称
DIAG_DIMS_ZH: dict[str, str] = {
    "front_grip_req": "前轴抓地",
    "rear_grip_req": "后轴抓地",
    "turnin_req": "入弯响应",
    "hi_speed_stab_req": "高速稳定性",
    "brake_stab_req": "制动稳定性",
    "brake_power_req": "制动力",
    "exit_traction_req": "出弯牵引",
    "tyre_life_req": "轮胎寿命",
    "ride_height_req": "底盘离地",
}

# 维度 key → 正值语义说明（用于报告联动说明）
DIAG_DIMS_POSITIVE_SEMANTICS: dict[str, str] = {
    "front_grip_req": "前轴抓地不足（转向不足）",
    "rear_grip_req": "后轴抓地不足（转向过度/打滑）",
    "turnin_req": "入弯响应不足（迟钝）",
    "hi_speed_stab_req": "高速/弯中不稳定",
    "brake_stab_req": "制动不稳定/易锁死",
    "brake_power_req": "制动力不足（刹距长）",
    "exit_traction_req": "出弯牵引不足（打滑）",
    "tyre_life_req": "胎耗过高",
    "ride_height_req": "离地过低（刮底）",
}


# ---------------------------------------------------------------------------
# 症状 × 阶段 → Dx 映射规则（确定性，task-60 阶段敏感扩展）
# ---------------------------------------------------------------------------
# 同一症状在不同弯道阶段有不同的 Dx 系数映射。
# 每个症状有一个"默认阶段"，当不指定阶段时使用默认阶段的映射。
SYMPTOM_STAGE_TO_DX: dict[str, dict[str, dict[str, float]]] = {
    # 入弯 entry
    "understeer": {
        "entry": {"front_grip_req": 0.80, "turnin_req": 0.30, "brake_stab_req": 0.10},
        "apex": {"front_grip_req": 0.60, "hi_speed_stab_req": 0.30, "turnin_req": 0.10},
    },
    "oversteer": {
        "entry": {"rear_grip_req": 0.80, "hi_speed_stab_req": 0.30, "exit_traction_req": 0.10},
        "exit": {"rear_grip_req": 0.50, "exit_traction_req": 0.40, "hi_speed_stab_req": 0.10},
    },
    "turnin_unresponsive": {
        "entry": {"turnin_req": 0.80, "front_grip_req": 0.30, "brake_stab_req": 0.10},
    },
    "brake_long": {
        "entry": {"brake_power_req": 0.80, "brake_stab_req": 0.20},
    },
    "lockup": {
        "entry": {"brake_stab_req": 0.70, "brake_power_req": -0.30, "front_grip_req": 0.10},
    },
    # 弯中 apex
    "midcorner_understeer": {
        "apex": {"front_grip_req": 0.60, "hi_speed_stab_req": 0.30, "turnin_req": 0.10},
    },
    "midcorner_unstable": {
        "apex": {"hi_speed_stab_req": 0.70, "rear_grip_req": 0.30, "front_grip_req": 0.10},
    },
    "midcorner_traction": {
        "apex": {"exit_traction_req": 0.60, "rear_grip_req": 0.40, "hi_speed_stab_req": 0.10},
    },
    # 出弯 exit
    "exit_wheelspin": {
        "exit": {"exit_traction_req": 0.70, "rear_grip_req": 0.30, "hi_speed_stab_req": 0.10},
    },
    "exit_oversteer": {
        "exit": {"rear_grip_req": 0.60, "exit_traction_req": 0.30, "hi_speed_stab_req": 0.10},
    },
    "exit_understeer": {
        # task-61：出弯给油推头 = 牵引与前轴平衡（出弯原仅 2 项，补齐）
        "exit": {"front_grip_req": 0.50, "exit_traction_req": 0.30, "hi_speed_stab_req": 0.20},
    },
    "exit_unstable": {
        # task-61：出弯姿态晃动
        "exit": {"hi_speed_stab_req": 0.50, "rear_grip_req": 0.30, "exit_traction_req": 0.20},
    },
    # 全局 global
    "bottoming": {
        "global": {"ride_height_req": 0.90, "hi_speed_stab_req": 0.10},
    },
    "tyre_wear": {
        "global": {"tyre_life_req": 0.80, "brake_stab_req": 0.10, "exit_traction_req": 0.10},
    },
    "tyre_overheat": {
        # task-61：胎温过高 —— 补上"有规则（1/2/12）没反馈入口"的缺口
        "global": {"tyre_life_req": 0.60, "front_grip_req": 0.20, "rear_grip_req": 0.20},
        "apex": {"tyre_life_req": 0.40, "front_grip_req": 0.40, "hi_speed_stab_req": 0.20},
    },
    "straight_slow": {
        "global": {"front_grip_req": -0.50, "rear_grip_req": -0.50, "hi_speed_stab_req": -0.20},
    },
    "lap_slow": {
        "global": {
            "brake_power_req": 0.30, "exit_traction_req": 0.30, "turnin_req": 0.30,
            "hi_speed_stab_req": 0.20, "front_grip_req": 0.10,
        },
    },
    "high_speed_instability": {
        "global": {"hi_speed_stab_req": 0.80, "front_grip_req": 0.10, "rear_grip_req": 0.10},
    },
}


# 症状默认阶段映射（用于不指定阶段时的 fallback）
SYMPTOM_DEFAULT_STAGE: dict[str, str] = {
    "understeer": "entry",
    "oversteer": "entry",
    "turnin_unresponsive": "entry",
    "brake_long": "entry",
    "lockup": "entry",
    "midcorner_understeer": "apex",
    "midcorner_unstable": "apex",
    "midcorner_traction": "apex",
    "exit_wheelspin": "exit",
    "exit_oversteer": "exit",
    "exit_understeer": "exit",
    "exit_unstable": "exit",
    "tyre_overheat": "global",
    "bottoming": "global",
    "tyre_wear": "global",
    "straight_slow": "global",
    "lap_slow": "global",
    "high_speed_instability": "global",
}


# ---------------------------------------------------------------------------
# 向后兼容：SYMPTOM_TO_DX 从阶段映射自动生成
# ---------------------------------------------------------------------------
# 使用每个症状的默认阶段的映射，保持与旧代码完全兼容。
SYMPTOM_TO_DX: dict[str, dict[str, float]] = {
    symptom: SYMPTOM_STAGE_TO_DX[symptom][SYMPTOM_DEFAULT_STAGE[symptom]]
    for symptom in SYMPTOM_STAGE_TO_DX
}


# ---------------------------------------------------------------------------
# 核心求值函数
# ---------------------------------------------------------------------------
# 症状输入类型：二元组 (symptom, strength) 或三元组 (symptom, strength, stage)
SymptomInput = tuple[str, int | float] | tuple[str, int | float, str]


def _resolve_stage(symptom: str, stage: str | None) -> str:
    """解析症状的有效阶段。

    Args:
        symptom: 症状标识字符串。
        stage: 显式指定的阶段；None 时使用默认阶段。

    Returns:
        有效阶段字符串。

    Raises:
        KeyError: 症状未知，或指定阶段不在该症状的阶段映射中且无默认阶段。
    """
    if symptom not in SYMPTOM_STAGE_TO_DX:
        raise KeyError(f"未知症状标识: {symptom!r}")

    stage_map = SYMPTOM_STAGE_TO_DX[symptom]

    if stage is not None and stage in stage_map:
        return stage

    # 指定阶段不存在时 fallback 到默认阶段
    default_stage = SYMPTOM_DEFAULT_STAGE.get(symptom)
    if default_stage is None or default_stage not in stage_map:
        raise KeyError(f"症状 {symptom!r} 无可用阶段映射")
    return default_stage


def _unpack_symptom_item(
    item: SymptomInput,
) -> tuple[str, int | float, str | None]:
    """解包症状输入项为 (symptom, strength, stage) 三元组。

    Args:
        item: 二元组 (symptom, strength) 或三元组 (symptom, strength, stage)。

    Returns:
        (symptom, strength, stage) 三元组，stage 可能为 None。

    Raises:
        ValueError: 输入格式非法（非二元组或三元组）。
    """
    if len(item) == 3:
        symptom, strength, stage = item  # type: ignore[misc]
        return symptom, strength, stage
    if len(item) == 2:
        symptom, strength = item  # type: ignore[misc]
        return symptom, strength, None
    raise ValueError(f"症状输入格式非法: {item!r}，应为二元组或三元组")


def _validate_symptom_strength(symptom: str, strength: int | float) -> float:
    """校验症状强度类型与范围，返回 float 化的强度值。

    Args:
        symptom: 症状标识字符串。
        strength: 原始强度值。

    Returns:
        float 化的强度值。

    Raises:
        KeyError: 症状未知。
        ValueError: 强度类型非法或越界 [0, 5]。
    """
    if symptom not in SYMPTOM_STAGE_TO_DX:
        raise KeyError(f"未知症状标识: {symptom!r}")
    if not isinstance(strength, (int, float)):
        raise ValueError(
            f"症状 {symptom!r} 强度类型非法: {type(strength).__name__}，应为 int/float",
        )
    s = float(strength)
    if s < 0.0 or s > 5.0:
        raise ValueError(f"症状 {symptom!r} 强度 {s} 越界，合法范围 [0, 5]")
    return s


def compute_dx(symptoms: Iterable[SymptomInput]) -> dict[str, float]:
    """从症状列表计算诊断向量 Dx（多症状代数求和）。

    确定性纯函数：相同输入必得相同输出，无 IO、无随机、无时间依赖。

    Args:
        symptoms: 可迭代的症状序列，每项为：
            - 二元组 ``(symptom_key, strength)``：使用症状的默认阶段（向后兼容）；
            - 三元组 ``(symptom_key, strength, stage)``：使用指定阶段。
            - symptom_key: 症状标识字符串（见 SYMPTOM_STAGE_TO_DX 的 key）。
            - strength: 症状强度 s∈[0,5]，作为 Dx 系数的乘数。
              强度 0 表示该症状不产生贡献（与未触发等价）。
            - stage: 弯道阶段（``"entry"``/``"apex"``/``"exit"``/``"global"``）。
              若指定阶段不在该症状的阶段映射中，自动 fallback 到默认阶段。

    Returns:
        9 维 Dx 字典 {dim_key: value}，所有 9 个维度均出现（无触发维度为 0.0）。
        正值表示该能力不足需增强，负值表示该能力过强需减弱。

    Raises:
        KeyError: 出现未知症状标识。
        ValueError: 强度超出 [0, 5] 区间。

    出处:
        F1 25 官方调教指南（症状-机理对应章节）。
    """
    dx: dict[str, float] = dict.fromkeys(DIAG_DIMS, 0.0)

    for item in symptoms:
        symptom, strength, stage = _unpack_symptom_item(item)
        s = _validate_symptom_strength(symptom, strength)
        # 强度 0 直接跳过（不产生贡献）
        if s == 0.0:
            continue
        _accumulate_dx_contribution(dx, symptom, s, stage)

    return dx


def _accumulate_dx_contribution(
    dx: dict[str, float], symptom: str, strength: float, stage: str | None,
) -> None:
    """将单个症状的 Dx 贡献累加到 dx（原地修改）。"""
    effective_stage = _resolve_stage(symptom, stage)
    contribution = SYMPTOM_STAGE_TO_DX[symptom][effective_stage]
    for dim, coef in contribution.items():
        dx[dim] += coef * strength


def empty_dx() -> dict[str, float]:
    """返回全零 Dx 向量（所有 9 维均为 0.0）。

    用于「无反馈」场景的占位，避免空字典导致的维度缺失。
    """
    return dict.fromkeys(DIAG_DIMS, 0.0)


def is_zero_dx(dx: dict[str, float]) -> bool:
    """判断 Dx 向量是否全零（无任何有效需求）。

    Args:
        dx: 诊断向量字典。

    Returns:
        全部分量绝对值 < 1e-12 时返回 True。
    """
    return all(abs(v) < ZERO_DX_EPSILON for v in dx.values())


def dx_to_vector(dx: dict[str, float]) -> list[float]:
    """将 Dx 字典转为按 DIAG_DIMS 顺序的浮点列表（便于矩阵乘法）。

    Args:
        dx: 诊断向量字典。

    Returns:
        长度 9 的浮点列表，顺序与 DIAG_DIMS 一致。
    """
    return [float(dx.get(dim, 0.0)) for dim in DIAG_DIMS]