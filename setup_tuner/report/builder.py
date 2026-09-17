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

from datetime import UTC, datetime
from typing import Any

from setup_tuner.domain.setup import ALL_SETUP_FIELDS
from setup_tuner.telemetry.packets import to_sector_1based

# 浮点比较 epsilon（用于 delta 零值判定）
DELTA_ZERO_EPSILON = 1e-12

# ---------------------------------------------------------------------------
# Packet 5 字段映射（UDP 字段名 → domain.setup 参数名）
# 对照 design 1.2.1 Packet 5 与 domain/setup.py 21 参数定义
# ---------------------------------------------------------------------------
_PACKET5_FIELD_MAP: dict[str, str] = {
    "m_frontWing": "front_wing",
    "m_rearWing": "rear_wing",
    "m_onThrottleDiff": "on_throttle_diff",
    "m_offThrottleDiff": "off_throttle_diff",
    "m_frontCamber": "front_camber",
    "m_rearCamber": "rear_camber",
    "m_frontToe": "front_toe",
    "m_rearToe": "rear_toe",
    "m_frontSuspension": "front_suspension",
    "m_rearSuspension": "rear_suspension",
    "m_frontAntiRollBar": "front_anti_roll_bar",
    "m_rearAntiRollBar": "rear_anti_roll_bar",
    "m_frontSuspensionHeight": "front_ride_height",
    "m_rearSuspensionHeight": "rear_ride_height",
    "m_brakePressure": "brake_pressure",
    "m_brakeBias": "brake_bias",
    "m_engineBraking": "engine_braking",
    "m_rearLeftTyrePressure": "rear_left_tyre_pressure",
    "m_rearRightTyrePressure": "rear_right_tyre_pressure",
    "m_frontLeftTyrePressure": "front_left_tyre_pressure",
    "m_frontRightTyrePressure": "front_right_tyre_pressure",
}

# ---------------------------------------------------------------------------
# UDP 值域换算策略（2026-09 修正 —— 原实现会静默算错基线调教）
# ---------------------------------------------------------------------------
# 这里曾有一张 ``_UDP_VALUE_RANGE_MAP``，假设 EA 的 uint8 字段是 0–250 的编码值，
# 再线性压缩到车库值域（例如 ``m_frontWing=34`` → 34/250*50 = **6.8**）。
#
# 该假设是错的。用仓库内 403 帧真实 F1 2026 抓包（``packetFormat=2026``）核对后：
# EA 下发的 ``m_frontWing`` / ``m_frontSuspension`` / ``m_brakePressure`` /
# ``m_engineBraking`` / ``m_frontToe`` … **已经是车库内显示的值** ——
# 21/21 个映射字段的原始值全部落在 ``domain.setup`` 定义的合法区间内，例如：
#
#     m_frontWing=34            ∈ [0, 50]
#     m_frontSuspension=37      ∈ [1, 41]
#     m_frontAntiRollBar=15     ∈ [1, 21]
#     m_brakePressure=97        ∈ [80, 100]
#     m_brakeBias=57            ∈ [50, 70]
#     m_engineBraking=50        ∈ [0, 100]
#     m_frontToe=0.04           ∈ [0, 0.2]
#
# 错误换算不会报错（结果仍落在合法区间内，clamp 与全部测试都放行），
# 因此属于"静默污染"：导入的当前调教基线全错 → 之后的每一个 setup_delta 都基于错基线。
#
# 现在改为**恒等映射**，只保留 clamp 到合法区间（防御越界值）。
# 若将来确认某个字段 EA 确实做了编码，请**单独**为该字段加映射并附实测依据。



# ---------------------------------------------------------------------------
# ISO8601 时间戳
# ---------------------------------------------------------------------------
def _now_iso8601() -> str:
    """返回当前 UTC 时间的 ISO8601 字符串（带 Z 后缀，对齐 design 2.7.7 示例）。

    示例：``2026-09-10T12:00:00Z``
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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
        1 for p in parameters if abs(float(p.get("setup_delta", 0.0))) > DELTA_ZERO_EPSILON
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
    confidence = str(suggestion_result.get("confidence", "medium"))
    raw_params: list[dict[str, Any]] = suggestion_result.get("parameters", [])
    parameters = [_build_param_entry(pd, confidence) for pd in raw_params]
    summary = suggestion_result.get("summary") or build_summary(parameters)
    report = _assemble_report_dict(
        track_id, setup_id, parameters, summary, confidence, suggestion_result,
    )
    # task-62：报告带实际生效的模型类型（nn/hybrid 未装 torch 时会降级为 rule，
    # 前端据此如实提示，不让用户误以为神经网络在跑）
    report["model_type"] = str(suggestion_result.get("model_type", "rule"))
    # 整体思维层：赛道需求画像 / 逐弯加权说明 / 跨弯道类别冲突 / 收口说明。
    # 这部分是"为什么这么调"的依据链，前端据此向车手解释取舍，而不是只给数字。
    holistic = suggestion_result.get("holistic")
    if holistic:
        report["holistic"] = holistic
    return report


def _assemble_report_dict(
    track_id: str,
    setup_id: int | None,
    parameters: list[dict[str, Any]],
    summary: str,
    confidence: str,
    suggestion_result: dict[str, Any],
) -> dict[str, Any]:
    """组装报告顶层字典（对齐 design 2.7.7）。"""
    return {
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


# ---------------------------------------------------------------------------
# UDP 值域换算：Packet 5 原始值 → 游戏车库值
# ---------------------------------------------------------------------------
def convert_udp_to_game_value(param_name: str, udp_value: float) -> float:
    """把 Packet 5 原始值转换为游戏内车库值。

    2026-09 修正：真实抓包证明 EA 下发的就是车库值，因此本函数为**恒等映射**
    （详见本模块 ``UDP 值域换算策略`` 注释）。保留函数是为了不动调用方接口，
    并作为"是否需要按字段换算"的唯一收敛点。

    Args:
        param_name: 参数标识（保留参数：将来若某字段确需换算，在此按名分派）。
        udp_value: UDP Packet 5 中的原始值。

    Returns:
        游戏内车库值（当前等于 ``udp_value``）。
    """
    return udp_value


# ---------------------------------------------------------------------------
# 辅助：从 CarSetups 包（遥测 Packet 5）提取 21 参数快照
# ---------------------------------------------------------------------------
def extract_setup_from_packet5(packet5: dict[str, Any]) -> dict[str, float]:
    """从遥测 CarSetups 包（packet_id=5）提取 21 项调教参数快照。

    对齐 design 1.2.1 Packet 5 字段映射与 domain.setup 21 参数全集。
    缺失字段取 SetupField.default；值经 :func:`convert_udp_to_game_value`
    （当前为恒等映射）后 clamp 到合法区间。

    Args:
        packet5: ``parse_car_setups`` 返回的字典（含 m_frontWing 等字段）。

    Returns:
        21 参数扁平字典 ``{param_name: value}``。
    """
    result: dict[str, float] = {}
    for spec in ALL_SETUP_FIELDS:
        udp_name = next(
            (u for u, d in _PACKET5_FIELD_MAP.items() if d == spec.name), None,
        )
        if udp_name is not None and udp_name in packet5:
            udp_value = float(packet5[udp_name])
            result[spec.name] = convert_udp_to_game_value(spec.name, udp_value)
        else:
            result[spec.name] = spec.default

    # clamp 到合法区间（防御性，UDP 值可能越界）
    _clamp_setup_to_bounds(result)
    return result


def _clamp_setup_to_bounds(result: dict[str, float]) -> None:
    """将 21 参数值 clamp 到各自合法区间（原地修改）。"""
    for spec in ALL_SETUP_FIELDS:
        val = result[spec.name]
        if val < spec.min_val:
            result[spec.name] = spec.min_val
        elif val > spec.max_val:
            result[spec.name] = spec.max_val


# ---------------------------------------------------------------------------
# 辅助：从遥测帧缓存提取用于建议生成的遥测摘要
# ---------------------------------------------------------------------------
def extract_telemetry_summary(
    all_latest: dict[int, dict[str, Any]],
    lap_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """从遥测帧缓存 + 整圈统计提取用于建议生成的遥测摘要。

    提取 weather / track_temp / air_temp / speed / tyre_compound 等关键字段，
    供 engine._derive_telemetry_gain / _derive_telemetry_dx / confidence 使用。

    Args:
        all_latest: ``TelemetryStream.get_all_latest()`` 返回的
            ``{packet_id: parsed_dict}`` 字典。
        lap_stats: 可选，``LapAggregator.snapshot()`` 产出的整圈统计。
            提供后会合并进摘要，并用整圈均值覆盖 ``m_throttle`` / ``m_brake``
            （引擎明确按"均值"语义使用这两个键）。
            这是让遥测规则 6/7/10/15 生效的关键：单帧路径永远拿不到
            ``max_speed`` / ``avg_steer`` / ``max_steer`` / ``on_straight``。

    Returns:
        遥测摘要字典。
    """
    summary: dict[str, Any] = {}
    _merge_session_summary(summary, all_latest.get(1))
    _merge_telemetry_summary(summary, all_latest.get(6))
    _merge_status_summary(summary, all_latest.get(7))
    _merge_lap_summary(summary, all_latest.get(2))
    if lap_stats:
        summary.update(lap_stats)
        if lap_stats.get("avg_throttle") is not None:
            summary["m_throttle"] = lap_stats["avg_throttle"]
        if lap_stats.get("avg_brake") is not None:
            summary["m_brake"] = lap_stats["avg_brake"]
    return summary


def _merge_session_summary(
    summary: dict[str, Any], session: dict[str, Any] | None,
) -> None:
    """从 Packet 1 Session 提取天气/温度。"""
    if not session:
        return
    summary["weather"] = session.get("m_weather")
    summary["track_temp"] = session.get("m_trackTemperature")
    summary["air_temp"] = session.get("m_airTemperature")
    summary["track_id_udp"] = session.get("m_trackId")


def _merge_telemetry_summary(
    summary: dict[str, Any], telemetry: dict[str, Any] | None,
) -> None:
    """从 Packet 6 CarTelemetry 提取速度/油门/刹车/挡位/转速/胎温/制动温度/胎压。"""
    if not telemetry:
        return
    summary["speed"] = telemetry.get("m_speed")
    summary["throttle"] = telemetry.get("m_throttle")
    summary["m_throttle"] = telemetry.get("m_throttle")
    summary["brake"] = telemetry.get("m_brake")
    summary["m_brake"] = telemetry.get("m_brake")
    summary["gear"] = telemetry.get("m_gear")
    summary["engine_rpm"] = telemetry.get("m_engineRPM")
    summary["m_tyresSurfaceTemperature"] = telemetry.get("m_tyresSurfaceTemperature")
    summary["m_tyresInnerTemperature"] = telemetry.get("m_tyresInnerTemperature")
    summary["m_brakesTemperature"] = telemetry.get("m_brakesTemperature")
    summary["m_tyresPressure"] = telemetry.get("m_tyresPressure")


def _merge_status_summary(
    summary: dict[str, Any], status: dict[str, Any] | None,
) -> None:
    """从 Packet 7 CarStatus 提取轮胎配方/胎龄/燃油/刹车平衡。

    ``front_brake_bias``（游戏内实际读数）供引擎规则16 与写入调教比对：
    两者偏差过大说明设置未生效，此时继续调参无意义。
    """
    if not status:
        return
    summary["tyre_compound"] = status.get("m_visualTyreCompound")
    summary["tyres_age_laps"] = status.get("m_tyresAgeLaps")
    summary["fuel_in_tank"] = status.get("m_fuelInTank")
    summary["fuel_remaining_laps"] = status.get("m_fuelRemainingLaps")
    # 实际刹车平衡（Packet 7 字段名 m_frontBrakeBias；模拟器直接沿用该键）
    summary["front_brake_bias"] = status.get("m_frontBrakeBias")


def _merge_lap_summary(
    summary: dict[str, Any], lap: dict[str, Any] | None,
) -> None:
    """从 Packet 2 LapData 提取圈速/扇区/圈距离。

    ``sector`` 统一转为 **1 基**（0/1/2 → 1/2/3）：前端按 ``S1/S2/S3`` 展示，
    规则 5 判断 ``sector == 3``。早期直传 0 基原始值导致前端显示 S0/S1/S2、
    且规则 5 永不触发。
    """
    if not lap:
        return
    summary["lap_distance"] = lap.get("m_lapDistance")
    summary["sector"] = to_sector_1based(lap.get("m_sector"))
    summary["current_lap_num"] = lap.get("m_currentLapNum")
    summary["last_lap_time_ms"] = lap.get("m_lastLapTimeInMS")


# ---------------------------------------------------------------------------
# 辅助：反馈列表 → 症状列表（含聚合，用于 /suggest）
# ---------------------------------------------------------------------------
def feedbacks_to_symptoms(
    feedbacks: list[dict[str, Any]],
) -> list[tuple[str, int]]:
    """将反馈记录逐条投影为 ``(symptom, strength)``（不做聚合，保留原语义）。

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


def aggregate_feedback_symptoms(
    feedbacks: list[dict[str, Any]],
) -> list[tuple[str, int, str]]:
    """把反馈按 ``(弯道, 症状)`` 聚合后投影为**三元组**（症状, 强度, 阶段）。

    为什么需要（2026-09 修正）：
        早期 ``/suggest`` 直接把每条反馈都作为一个症状丢给 ``compute_dx``。
        而 ``compute_dx`` 对同症状是**代数求和**，于是同一条反馈重复 N 次就会让
        Dx 线性放大 —— 云端实测：同一条反馈 ×20 时 ``|Dx|max`` 从 2.4 涨到 48，
        21 个参数里 16 个被 ``max_delta`` 顶满 → 建议不再随输入变化，
        且反馈越多越极端。这既不是"更多信息"，也不是个性化。

        聚合规则：同一 ``(corner_number, symptom)`` 只保留**最强强度**；
        不同弯道的同一症状仍然各算一份（保留"多个弯都推头"的信息量，
        但不会因重复点击而虚假放大）。

    task-62：输出升级为**三元组** ``(symptom, strength, stage)`` ——
        ``compute_dx`` 原生支持阶段化映射（``SYMPTOM_STAGE_TO_DX``），此前管线
        把阶段丢掉，等于放弃了引擎的阶段感知能力。stage 取反馈的
        ``category``（entry/apex/exit/global，与"反馈阶段"同义）。

    Args:
        feedbacks: ``Store.get_feedbacks`` 返回的反馈记录列表。

    Returns:
        聚合后的 ``[(symptom, strength, stage), ...]``，按首次出现顺序。
    """
    best: dict[tuple[Any, str], tuple[int, str]] = {}
    order: list[tuple[Any, str]] = []
    for fb in feedbacks:
        symptom = fb.get("symptom")
        strength = fb.get("strength")
        if not symptom or strength is None:
            continue
        stage = fb.get("category") or "global"
        key = (fb.get("corner_number"), symptom)
        value = int(strength)
        if key not in best:
            best[key] = (value, stage)
            order.append(key)
        elif value > best[key][0]:
            best[key] = (value, stage)
    return [
        (symptom, best[(corner, symptom)][0], best[(corner, symptom)][1])
        for corner, symptom in order
    ]