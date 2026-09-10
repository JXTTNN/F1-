"""建议报告组装模块 —— 将规则引擎输出组装为 design 2.7.7 格式的报告 JSON。

对齐 design.md 2.7.7「建议报告输出结构」与 spec FR-RPT-01 ~ FR-RPT-04：

    - 每参数含 param / current / setup_delta / linkages / linked_notes /
      source / confidence / tradeoff 八字段；
    - 报告顶层含 track_id / generated_at(ISO8601) / parameters[] / summary；
    - source 取值受限枚举：EA_UDP_2026 / EA_SETUP_GUIDE / PIRELLI
      （对齐 FR-RPT-02）；
    - confidence 取值 high|medium|low；
    - tradeoff 仅在存在副作用时出现（对齐 FR-RPT-03/04）。

本模块为纯函数组装，零 IO、零随机；时间戳由本模块在组装时补充 ISO8601 UTC。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from setup_tuner.domain.setup import ALL_SETUP_FIELDS, get_field


# ---------------------------------------------------------------------------
# ISO8601 时间戳
# ---------------------------------------------------------------------------
def _now_iso8601() -> str:
    """返回当前 UTC 时间的 ISO8601 字符串（带 Z 后缀，对齐 design 2.7.7 示例）。

    示例：``2026-09-10T12:00:00Z``
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# 联动说明格式化
# ---------------------------------------------------------------------------
def format_linkages(param_detail: dict[str, Any]) -> list[str]:
    """格式化单参数的联动说明列表。

    输入为 engine.generate_suggestion 输出 ``parameters[]`` 中的单参数详情，
    其 ``linkages`` 字段为形如 ``"front_grip_req(+0.8)"`` 的字符串列表
    （或 engine 内部使用的 ``"dim(中文) Dx=+x.xx × C=+x.xx"`` 格式）。

    本函数做归一化处理：保留原字符串，确保返回 ``list[str]``；空输入返回空列表。

    Args:
        param_detail: 单参数详情字典（含 ``linkages`` 字段）。

    Returns:
        联动说明字符串列表。
    """
    raw = param_detail.get("linkages")
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in raw]


# ---------------------------------------------------------------------------
# 摘要生成
# ---------------------------------------------------------------------------
def build_summary(parameters: list[dict[str, Any]]) -> str:
    """根据参数列表生成摘要文本。

    统计非零调整参数数与总参数数，输出对齐 design 2.7.7 的中文摘要。

    Args:
        parameters: 参数详情列表（每项含 ``setup_delta`` 字段）。

    Returns:
        摘要文本。
    """
    total = len(parameters)
    nonzero = sum(
        1 for p in parameters if abs(float(p.get("setup_delta", 0.0))) > 1e-12
    )
    if total == 0:
        return "本次建议无参数"
    if nonzero == 0:
        return f"本次建议共关联 {total} 项参数，均无需调整"
    return (
        f"本次建议共关联 {total} 项参数，"
        f"其中 {nonzero} 项非零调整，整体性调教"
    )


# ---------------------------------------------------------------------------
# 单参数报告项组装
# ---------------------------------------------------------------------------
def _build_param_entry(
    param_detail: dict[str, Any],
    confidence: str,
) -> dict[str, Any]:
    """将 engine 输出的单参数详情组装为 design 2.7.7 报告项。

    输出字段（对齐 design 2.7.7）：
        - param: 参数标识符
        - current: 当前值
        - setup_delta: 调整量
        - linkages: 联动说明列表
        - linked_notes: 联动中文说明
        - source: 官方出处
        - confidence: 置信度 high|medium|low
        - tradeoff: 副作用提示（无则省略）
    """
    entry: dict[str, Any] = {
        "param": param_detail.get("param", ""),
        "current": param_detail.get("current", 0.0),
        "setup_delta": param_detail.get("setup_delta", 0.0),
        "linkages": format_linkages(param_detail),
        "linked_notes": param_detail.get("linked_notes", ""),
        "source": param_detail.get("source", ""),
        "confidence": confidence,
    }
    # tradeoff 仅在存在副作用时出现（对齐 FR-RPT-03/04）
    tradeoff = param_detail.get("tradeoff")
    if tradeoff:
        entry["tradeoff"] = tradeoff
    return entry


# ---------------------------------------------------------------------------
# 报告组装主入口
# ---------------------------------------------------------------------------
def build_report(
    suggestion_result: dict[str, Any],
    track_id: str,
    setup_id: int | None = None,
) -> dict[str, Any]:
    """将 engine.generate_suggestion 的输出组装为 design 2.7.7 格式的报告 JSON。

    Args:
        suggestion_result: ``engine.generate_suggestion`` 的返回字典，结构：
            ``{track_id, dx, setup_delta, parameters[], confidence, summary}``。
        track_id: 赛道标识（用于报告顶层；与 suggestion_result["track_id"] 一致）。
        setup_id: 关联的调教快照 id（可选，用于追溯）。

    Returns:
        报告 JSON 字典，结构对齐 design 2.7.7：
        ::
            {
              "track_id": str,
              "setup_id": int | None,
              "generated_at": "2026-09-10T12:00:00Z",
              "parameters": [{param, current, setup_delta, linkages,
                              linked_notes, source, confidence, tradeoff?}, ...],
              "summary": str,
              "confidence": "high|medium|low",
              "dx": {dim: value},          # 诊断向量（调试/追溯用）
              "setup_delta": {param: delta}  # 扁平增量（便于前端批量应用）
            }
    """
    # 置信度（取引擎结果；缺省 medium）
    confidence = str(suggestion_result.get("confidence", "medium"))

    # 参数列表（取引擎结果；缺省空列表）
    raw_params: list[dict[str, Any]] = suggestion_result.get("parameters", [])
    parameters = [_build_param_entry(pd, confidence) for pd in raw_params]

    # 摘要（优先使用引擎结果；缺省则按参数列表重新生成）
    summary = suggestion_result.get("summary") or build_summary(parameters)

    # 顶层组装
    report: dict[str, Any] = {
        "track_id": track_id,
        "setup_id": setup_id,
        "generated_at": _now_iso8601(),
        "parameters": parameters,
        "summary": summary,
        "confidence": confidence,
        # 扁平增量与诊断向量（便于前端/迭代对比）
        "setup_delta": suggestion_result.get("setup_delta", {}),
        "dx": suggestion_result.get("dx", {}),
    }
    return report


# ---------------------------------------------------------------------------
# 辅助：从 CarSetups 包（遥测 Packet 5）提取 23 参数快照
# ---------------------------------------------------------------------------
def extract_setup_from_packet5(packet5: dict[str, Any]) -> dict[str, float]:
    """从遥测 CarSetups 包（packet_id=5）提取 23 项调教参数快照。

    对齐 design 1.2.1 Packet 5 字段映射与 domain.setup 23 参数全集。
    缺失字段取 SetupField.default。

    Args:
        packet5: ``parse_car_setups`` 返回的字典（含 m_frontWing 等字段）。

    Returns:
        23 参数扁平字典 ``{param_name: value}``。
    """
    # UDP 字段名 → domain.setup 参数名 的映射
    # （对照 design 1.2.1 Packet 5 与 domain/setup.py 23 参数定义）
    field_map: dict[str, str] = {
        "m_frontWing": "front_wing",
        "m_rearWing": "rear_wing",
        # 主动空力 Z/X：UDP 仅有 m_activeAeroMode（0=Z/弯, 1=X/直），
        # 此处将 mode 映射为 ratio 占位（0.5），实际值需玩家在 garage 确认
        "m_onThrottleDiff": "on_throttle_diff",
        "m_offThrottleDiff": "off_throttle_diff",
        "m_frontCamber": "front_camber",
        "m_rearCamber": "rear_camber",
        "m_frontToe": "front_toe",
        "m_rearToe": "rear_toe",
        "m_frontSuspension": "front_spring",
        "m_rearSuspension": "rear_spring",
        "m_frontAntiRollBar": "front_anti_roll_bar",
        "m_rearAntiRollBar": "rear_anti_roll_bar",
        "m_frontSuspensionHeight": "front_ride_height",
        "m_rearSuspensionHeight": "rear_ride_height",
        # damping：UDP Packet 5 无此字段，取缺省
        "m_brakePressure": "brake_pressure",
        "m_brakeBias": "brake_bias",
        # 胎压：UDP 为 tyresPressure[4]，取前两轴均值作为前/后胎压占位
        "m_engineBraking": "engine_braking",
        "m_ballast": "ballast",
    }

    result: dict[str, float] = {}
    for spec in ALL_SETUP_FIELDS:
        udp_name = next(
            (u for u, d in field_map.items() if d == spec.name), None
        )
        if udp_name is not None and udp_name in packet5:
            result[spec.name] = float(packet5[udp_name])
        else:
            result[spec.name] = spec.default

    # 主动空力 Z/X：从 m_activeAeroMode 推断占位
    # （0=Z/弯道模式 → active_aero_z 取 0.6, active_aero_x 取 0.4；
    #  1=X/直道模式 → active_aero_z 取 0.4, active_aero_x 取 0.6；
    #  其他取缺省 0.5）
    aero_mode = packet5.get("m_activeAeroMode")
    if aero_mode == 0:
        result["active_aero_z"] = 0.6
        result["active_aero_x"] = 0.4
    elif aero_mode == 1:
        result["active_aero_z"] = 0.4
        result["active_aero_x"] = 0.6

    # 胎压：从 tyresPressure[4] 取前轴/后轴均值
    pressures = packet5.get("tyresPressure") or packet5.get("m_tyresPressure")
    if isinstance(pressures, (list, tuple)) and len(pressures) >= 4:
        try:
            front_p = (float(pressures[0]) + float(pressures[1])) / 2.0
            rear_p = (float(pressures[2]) + float(pressures[3])) / 2.0
            result["front_tyre_pressure"] = front_p
            result["rear_tyre_pressure"] = rear_p
        except (TypeError, ValueError):
            pass

    # clamp 到合法区间（防御性，UDP 值可能越界）
    for spec in ALL_SETUP_FIELDS:
        val = result[spec.name]
        if val < spec.min_val:
            result[spec.name] = spec.min_val
        elif val > spec.max_val:
            result[spec.name] = spec.max_val

    return result


# ---------------------------------------------------------------------------
# 辅助：从遥测帧缓存提取用于建议生成的遥测摘要
# ---------------------------------------------------------------------------
def extract_telemetry_summary(
    all_latest: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """从 TelemetryStream.get_all_latest() 输出提取用于建议生成的遥测摘要。

    提取 weather / track_temp / air_temp / speed / tyre_compound 等关键字段，
    供 engine._derive_telemetry_gain 与 confidence.assess_confidence 使用。

    Args:
        all_latest: ``TelemetryStream.get_all_latest()`` 返回的
            ``{packet_id: parsed_dict}`` 字典。

    Returns:
        遥测摘要字典。
    """
    summary: dict[str, Any] = {}

    # Packet 1 Session：天气/温度
    session = all_latest.get(1)
    if session:
        summary["weather"] = session.get("m_weather")
        summary["track_temp"] = session.get("m_trackTemperature")
        summary["air_temp"] = session.get("m_airTemperature")
        summary["track_id_udp"] = session.get("m_trackId")

    # Packet 6 CarTelemetry：速度/油门/刹车/挡位/转速
    telemetry = all_latest.get(6)
    if telemetry:
        summary["speed"] = telemetry.get("m_speed")
        summary["throttle"] = telemetry.get("m_throttle")
        summary["brake"] = telemetry.get("m_brake")
        summary["gear"] = telemetry.get("m_gear")
        summary["engine_rpm"] = telemetry.get("m_engineRPM")

    # Packet 7 CarStatus：轮胎配方/胎龄/燃油
    status = all_latest.get(7)
    if status:
        summary["tyre_compound"] = status.get("m_visualTyreCompound")
        summary["tyres_age_laps"] = status.get("m_tyresAgeLaps")
        summary["fuel_in_tank"] = status.get("m_fuelInTank")

    # Packet 2 LapData：圈速/扇区/圈距离
    lap = all_latest.get(2)
    if lap:
        summary["lap_distance"] = lap.get("m_lapDistance")
        summary["sector"] = lap.get("m_sector")
        summary["current_lap_num"] = lap.get("m_currentLapNum")
        summary["last_lap_time_ms"] = lap.get("m_lastLapTimeInMS")

    return summary


# ---------------------------------------------------------------------------
# 辅助：从反馈记录列表提取症状列表（供 engine.generate_suggestion 使用）
# ---------------------------------------------------------------------------
def feedbacks_to_symptoms(
    feedbacks: list[dict[str, Any]],
) -> list[tuple[str, int]]:
    """将反馈记录列表转换为 engine.generate_suggestion 所需的 symptoms 列表。

    Args:
        feedbacks: ``Store.get_feedbacks`` 返回的反馈记录列表。

    Returns:
        ``[(symptom_key, strength), ...]`` 列表。
    """
    return [
        (fb["symptom"], int(fb["strength"]))
        for fb in feedbacks
        if fb.get("symptom") and fb.get("strength") is not None
    ]