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

出处：EA F1 2026 官方调教指南（症状-机理对应章节）。
本模块为纯函数、零 IO、零随机，满足 FR-ENG-05 / FR-NFR-R1（可复现）。
"""

from __future__ import annotations

from typing import Iterable


# ---------------------------------------------------------------------------
# 官方出处常量
# ---------------------------------------------------------------------------
SOURCE_DX = "EA F1 2026 官方调教指南"


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
# 症状 → Dx 映射规则（确定性，逐字对齐 design.md 2.7.2 表）
# ---------------------------------------------------------------------------
# 每个症状映射到一个 {维度key: 系数} 字典；
# Dx 分量 = 系数 × 强度 s（s∈[0,5]，默认 3，归一化系数 k=1.0）。
# 多症状叠加时各维度代数求和。
SYMPTOM_TO_DX: dict[str, dict[str, float]] = {
    # 入弯 entry
    "understeer": {
        "front_grip_req": 0.80,
        "turnin_req": 0.30,
    },
    "oversteer": {
        "rear_grip_req": 0.80,
        "hi_speed_stab_req": 0.30,
    },
    "turnin_unresponsive": {
        "turnin_req": 0.80,
        "front_grip_req": 0.30,
    },
    "brake_long": {
        "brake_power_req": 0.80,
    },
    "lockup": {
        "brake_stab_req": 0.70,
        "brake_power_req": -0.30,  # 锁死需减压（负号）
    },
    # 弯中 apex
    "midcorner_unstable": {
        "hi_speed_stab_req": 0.70,
        "rear_grip_req": 0.30,
    },
    "midcorner_traction": {
        "exit_traction_req": 0.60,
        "rear_grip_req": 0.40,
    },
    # 出弯 exit
    "exit_wheelspin": {
        "exit_traction_req": 0.70,
        "rear_grip_req": 0.30,
    },
    # 全局 global
    "bottoming": {
        "ride_height_req": 0.90,
    },
    "tyre_wear": {
        "tyre_life_req": 0.80,
    },
    "straight_slow": {
        "front_grip_req": -0.50,  # 下压力过大致阻力（负号）
        "rear_grip_req": -0.50,
    },
    "lap_slow": {
        "brake_power_req": 0.30,
        "exit_traction_req": 0.30,
        "turnin_req": 0.30,
        "hi_speed_stab_req": 0.20,
    },
}


# ---------------------------------------------------------------------------
# 核心求值函数
# ---------------------------------------------------------------------------
def compute_dx(symptoms: Iterable[tuple[str, int]]) -> dict[str, float]:
    """从症状列表计算诊断向量 Dx（多症状代数求和）。

    确定性纯函数：相同输入必得相同输出，无 IO、无随机、无时间依赖。

    Args:
        symptoms: 可迭代的 (symptom_key, strength) 二元组序列。
            - symptom_key: 症状标识字符串（见 SYMPTOM_TO_DX 的 key）。
            - strength: 症状强度 s∈[0,5]，作为 Dx 系数的乘数。
              强度 0 表示该症状不产生贡献（与未触发等价）。

    Returns:
        9 维 Dx 字典 {dim_key: value}，所有 9 个维度均出现（无触发维度为 0.0）。
        正值表示该能力不足需增强，负值表示该能力过强需减弱。

    Raises:
        KeyError: 出现未知症状标识。
        ValueError: 强度超出 [0, 5] 区间。

    出处:
        EA F1 2026 官方调教指南（症状-机理对应章节）。
    """
    dx: dict[str, float] = {dim: 0.0 for dim in DIAG_DIMS}

    for symptom, strength in symptoms:
        if symptom not in SYMPTOM_TO_DX:
            raise KeyError(f"未知症状标识: {symptom!r}")
        if not isinstance(strength, (int, float)):
            raise ValueError(
                f"症状 {symptom!r} 强度类型非法: {type(strength).__name__}，应为 int/float"
            )
        s = float(strength)
        if s < 0.0 or s > 5.0:
            raise ValueError(f"症状 {symptom!r} 强度 {s} 越界，合法范围 [0, 5]")

        # 强度 0 直接跳过（不产生贡献）
        if s == 0.0:
            continue

        contribution = SYMPTOM_TO_DX[symptom]
        for dim, coef in contribution.items():
            dx[dim] += coef * s

    return dx


def empty_dx() -> dict[str, float]:
    """返回全零 Dx 向量（所有 9 维均为 0.0）。

    用于「无反馈」场景的占位，避免空字典导致的维度缺失。
    """
    return {dim: 0.0 for dim in DIAG_DIMS}


def is_zero_dx(dx: dict[str, float]) -> bool:
    """判断 Dx 向量是否全零（无任何有效需求）。

    Args:
        dx: 诊断向量字典。

    Returns:
        全部分量绝对值 < 1e-12 时返回 True。
    """
    return all(abs(v) < 1e-12 for v in dx.values())


def dx_to_vector(dx: dict[str, float]) -> list[float]:
    """将 Dx 字典转为按 DIAG_DIMS 顺序的浮点列表（便于矩阵乘法）。

    Args:
        dx: 诊断向量字典。

    Returns:
        长度 9 的浮点列表，顺序与 DIAG_DIMS 一致。
    """
    return [float(dx.get(dim, 0.0)) for dim in DIAG_DIMS]