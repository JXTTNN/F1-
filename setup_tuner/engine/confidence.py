"""置信度评估（证据充分度 + 反馈明确度）。

返回 ``"high" | "medium" | "low"``，对齐 design 2.7.7：
    - **high**：有遥测佐证 + 反馈明确（强度 ≥ 2 且无矛盾）；
    - **medium**：有反馈但无遥测佐证；
    - **low**：反馈模糊（强度 0/1）或矛盾（同时触发正负矛盾的症状）。

纯函数、零 IO、零随机，满足 FR-ENG-05 / FR-NFR-R1。
"""

from __future__ import annotations

from typing import Any

from .diagnostic import SYMPTOM_TO_DX

# ---------------------------------------------------------------------------
# 矛盾症状对（同时触发会产生方向矛盾的 Dx 贡献）
# ---------------------------------------------------------------------------
# 例：understeer（前轴抓地不足）与 oversteer（后轴抓地不足）不矛盾；
# 但 understeer 与 straight_slow（下压力过大）在 front_grip_req 上方向相反，
# 属于"反馈矛盾"——玩家既说转向不足又说直道慢，可能下压力已过高。
_CONFLICT_PAIRS: frozenset[tuple[str, str]] = frozenset({
    ("understeer", "straight_slow"),   # 前轴抓地 vs 下压力过大
    ("oversteer", "straight_slow"),    # 后轴抓地 vs 下压力过大
    ("brake_long", "lockup"),          # 制动力不足 vs 易锁死（制动方向矛盾）
})


def _has_telemetry_evidence(telemetry: dict[str, Any] | None) -> bool:
    """判断遥测是否提供有效佐证。

    有效佐证：遥测非空且含至少一个可印证字段（胎耗/胎温/刹车/天气/圈速等）。
    """
    if not telemetry:
        return False
    evidence_keys = {
        "m_tyresAgeLaps",
        "tyres_age_laps",
        "m_brake",
        "brake",
        "m_weather",
        "weather",
        "m_lastLapTimeInMS",
        "last_lap_time_ms",
        "m_sector1TimeInMS",
        "sector1_time_ms",
        "m_sector2TimeInMS",
        "sector2_time_ms",
        "tyres_surface_temperature",
        "front_tyre_temp",
        "rear_tyre_temp",
    }
    return any(k in telemetry and telemetry[k] is not None for k in evidence_keys)


def _has_conflict(symptoms: list[tuple[str, int]]) -> bool:
    """判断症状列表是否包含矛盾对（且两者强度均 ≥ 2）。"""
    active = {s for s, strength in symptoms if strength >= 2}
    return any(a in active and b in active for a, b in _CONFLICT_PAIRS)


def _is_feedback_clear(symptoms: list[tuple[str, int]]) -> bool:
    """判断反馈是否明确（至少一条强度 ≥ 2 的有效症状）。"""
    return any(symptom in SYMPTOM_TO_DX and strength >= 2 for symptom, strength in symptoms)


def _is_feedback_vague(symptoms: list[tuple[str, int]]) -> bool:
    """判断反馈是否模糊（全部强度 ≤ 1 或无有效症状）。"""
    if not symptoms:
        return True
    return all(not (symptom in SYMPTOM_TO_DX and strength >= 2) for symptom, strength in symptoms)


def assess_confidence(
    symptoms: list[tuple[str, int]],
    telemetry: dict[str, Any] | None,
) -> str:
    """评估建议置信度。

    确定性纯函数：相同输入必得相同输出。

    Args:
        symptoms: 症状列表 [(symptom_key, strength), ...]。
        telemetry: 遥测客观数据；None 表示无遥测。

    Returns:
        ``"high"`` | ``"medium"`` | ``"low"``：
            - **high**：有遥测佐证 + 反馈明确 + 无矛盾；
            - **medium**：反馈明确 + 无矛盾，但无遥测佐证；
            - **low**：反馈模糊 或 存在矛盾症状对。
    """
    # 反馈模糊或矛盾 → low
    if _is_feedback_vague(symptoms):
        return "low"
    if _has_conflict(symptoms):
        return "low"

    # 反馈明确且无矛盾
    if not _is_feedback_clear(symptoms):
        return "low"

    # 有遥测佐证 → high；否则 → medium
    if _has_telemetry_evidence(telemetry):
        return "high"
    return "medium"