"""遥测自动诊断 —— 让**车手没有反馈到的问题**也能被发现并进入优化。

为什么必须有这一层
------------------
原链路的症状只有一个来源：车手在界面上点选的反馈。车手没点的地方，
引擎完全不知道有问题 —— 但遥测里全都写着：压路肩、刮底、胎温前后失衡、
刹车过热、出弯牵引损失、高速弯时间损失……

本模块把这些遥测信号翻译成**与车手反馈完全同构的症状条目**
（``corner_number`` / ``symptom`` / ``strength`` / ``category``），
于是它们可以直接喂进既有的 ``holistic.class_weighted_dx``，
走同一条 Dx → 优化器 → 收口 的链路 —— 不需要为"自动发现问题"另开一条路径。

两类证据
--------
1. **规则证据**（整圈聚合量）：路肩、刮底、胎温/胎压、刹车温度、
   转向输入、直道比例、防抱死/牵引控制介入等。
2. **模型证据**（代理模型残差）：当提供了逐弯实际用时 ``corner_times`` 时，
   用遥测训练出的弯速代理模型（:mod:`setup_tuner.engine.surrogate`）算出该弯
   在**当前工况**下的期望用时，实测明显更慢 → 按弯型判定症状并给出强度。
   这是"用遥测训练出来的模型"真正参与"发现问题"的地方。

冲突处理
--------
车手反馈优先：同一 ``(弯号, 症状)`` 若车手已反馈，隐式条目被丢弃
（车手的感知比模型推断更可信）。同一弯的不同症状则保留，共同参与优化。
"""

from __future__ import annotations

from typing import Any

#: 症状强度上限（与引擎侧 1-3 档一致）
_MAX_STRENGTH = 3

#: 各症状对应的通用阶段（用于报告文案，Dx 映射由 diagnostic 决定）
_STAGE_HINT: dict[str, str] = {
    "understeer": "entry",
    "exit_wheelspin": "exit",
    "high_speed_instability": "global",
    "bottoming": "global",
    "tyre_graining": "global",
    "brake_fade": "global",
}


def _num(value: Any) -> float | None:
    """安全取数：非数值（含 bool / 'None'）一律返回 None。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _nums(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for v in values:
        n = _num(v)
        if n is not None:
            out.append(n)
    return out


#: 由「隐式诊断 → 反馈路径」接管的证据来源 —— 即**引擎现有聚合规则没有覆盖**的盲区。
#:
#: 为什么必须逐个甄别（而不是"全部接管"）：``_derive_telemetry_dx`` 里已有 17 条
#: 聚合规则覆盖了胎温过高/过低/不均、胎压异常、刹车过热、入弯/弯中转向、
#: 直道速度、刮底、路肩。这些信号若再经反馈路径注入会**重复计入 Dx**。
#: 下面是实测「引擎里完全没有对应规则」的信号（grep 确认），因此必须接管，
#: 否则"车手没反馈的问题"永远进不了优化：
#:
#:   ``anti_lock_brakes``   → lockup（锁死倾向）
#:   ``tyres_wear``         → tyre_wear / 前后胎耗失衡
#:   ``traction_control``   → exit_wheelspin（出弯打滑）
#:   ``floor_damage``       → 底板损伤 → 稳定性/离地
#:   ``wing_damage_max``    → 翼片损伤 → 前后轴气动失衡
#:   ``kerb_corners``       → 逐弯路肩冲击（聚合规则只有全圈判定，无弯道归因）
#:   ``surrogate``          → 弯速代理模型残差（实测用时 vs 模型期望）
ENGINE_OWNED_SOURCES: frozenset[str] = frozenset({
    "telemetry:kerb",
    "telemetry:tyre_wear",
    "telemetry:abs",
    "telemetry:tc",
    "telemetry:damage",
    "surrogate:corner_residual",
})


def implicit_for_engine(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """筛出应由反馈路径接管（而非聚合规则）的隐式症状。"""
    return [i for i in items if i.get("source") in ENGINE_OWNED_SOURCES]


def _fb(
    symptom: str,
    strength: int,
    corner: int | None,
    evidence: str,
    source: str,
) -> dict[str, Any]:
    return {
        "corner_number": corner,
        "symptom": symptom,
        "strength": max(1, min(_MAX_STRENGTH, int(strength))),
        "category": _STAGE_HINT.get(symptom, "global"),
        "source": source,
        "evidence": evidence,
        "implicit": True,
    }


# --------------------------------------------------------------------------- #
# 规则证据
# --------------------------------------------------------------------------- #
def _from_kerbs(telemetry: dict[str, Any]) -> list[dict[str, Any]]:
    """按弯路肩检测 → 路肩冲击症状（路肩不得不压，调教要迁就它）。"""
    out: list[dict[str, Any]] = []
    kerbs = telemetry.get("kerb_corners")
    if not isinstance(kerbs, list):
        return out
    for k in kerbs:
        if not isinstance(k, dict):
            continue
        ratio = _num(k.get("ratio"))
        corner = k.get("corner")
        if ratio is None or not isinstance(corner, int):
            continue
        if ratio >= 2.0:
            out.append(_fb("kerb_instability", 3, corner,
                           f"T{corner} 悬挂冲击比 {ratio:.2f}（重度压路肩）", "telemetry:kerb"))
        elif ratio >= 1.6:
            out.append(_fb("kerb_instability", 2, corner,
                           f"T{corner} 悬挂冲击比 {ratio:.2f}（轻度压路肩）", "telemetry:kerb"))
    return out


def _from_ride_height(telemetry: dict[str, Any]) -> list[dict[str, Any]]:
    """刮底 / 底板离地过低 → 需要抬高底盘或加硬弹簧。"""
    out: list[dict[str, Any]] = []
    bottoming = telemetry.get("plank_bottoming")
    ratio = _num(telemetry.get("plank_bottoming_ratio")) or 0.0
    minima = [
        _num(telemetry.get("plank_front_height_min")),
        _num(telemetry.get("plank_rear_height_min")),
        _num(telemetry.get("suspension_height_min")),
    ]
    numeric = [m for m in minima if m is not None]
    if bottoming:
        out.append(_fb("bottoming", 3, None,
                       f"底板触地（触地帧占比 {ratio:.3f}）", "telemetry:plank"))
    elif ratio >= 0.05:
        out.append(_fb("bottoming", 2, None,
                       f"底板接近触地（占比 {ratio:.3f}）", "telemetry:plank"))
    elif numeric and min(numeric) <= 0.012:
        out.append(_fb("bottoming", 2, None,
                       f"最小离地 {min(numeric) * 1000:.1f} mm 落入触地带", "telemetry:plank"))
    return out


def _from_tyres(telemetry: dict[str, Any]) -> list[dict[str, Any]]:
    """胎温/胎压：前后失衡、左右失衡、整体过热、以及胎压偏离工作窗口。"""
    out: list[dict[str, Any]] = []
    # 官方车轮顺序 RL, RR, FL, FR（[0][1]=后，[2][3]=前）
    surf = _nums(telemetry.get("m_tyresSurfaceTemperature"))
    inner = _nums(telemetry.get("m_tyresInnerTemperature"))
    temps = surf if len(surf) >= 4 else inner
    if len(temps) >= 4:
        rear = (temps[0] + temps[1]) / 2.0
        front = (temps[2] + temps[3]) / 2.0
        avg = sum(temps[:4]) / 4.0
        left = (temps[0] + temps[2]) / 2.0
        right = (temps[1] + temps[3]) / 2.0
        if avg > 105.0:
            out.append(_fb("tyre_overheat", 3, None,
                           f"胎面均温 {avg:.1f}°C（>105）", "telemetry:tyre_temp"))
        elif avg > 95.0:
            out.append(_fb("tyre_overheat", 2, None,
                           f"胎面均温 {avg:.1f}°C 偏高", "telemetry:tyre_temp"))
        diff_fr = front - rear
        if abs(diff_fr) > 15.0:
            if diff_fr > 0:
                out.append(_fb("midcorner_understeer", 3, None,
                               f"前胎比后胎热 {diff_fr:+.1f}°C（前轴在滑）",
                               "telemetry:tyre_temp"))
            else:
                out.append(_fb("exit_oversteer", 3, None,
                               f"后胎比前胎热 {diff_fr:+.1f}°C（后轴在滑）",
                               "telemetry:tyre_temp"))
        if abs(left - right) > 15.0:
            out.append(_fb("tyre_graining", 2, None,
                           f"左右胎温差 {left - right:+.1f}°C（单侧滑移起粒）",
                           "telemetry:tyre_temp"))
    pressures = _nums(telemetry.get("m_tyresPressure"))
    if len(pressures) >= 4:
        rear_p = (pressures[0] + pressures[1]) / 2.0
        front_p = (pressures[2] + pressures[3]) / 2.0
        if front_p > 26.0 or front_p < 22.0:
            out.append(_fb("midcorner_understeer", 2, None,
                           f"前胎压 {front_p:.1f} psi 偏离工作窗口 22–26",
                           "telemetry:tyre_pressure"))
        if rear_p > 26.0 or rear_p < 22.0:
            out.append(_fb("exit_wheelspin", 2, None,
                           f"后胎压 {rear_p:.1f} psi 偏离工作窗口 22–26",
                           "telemetry:tyre_pressure"))
    wear = _nums(telemetry.get("tyres_wear"))
    if len(wear) >= 4:
        avg_wear = sum(wear[:4]) / 4.0
        front_wear = (wear[2] + wear[3]) / 2.0
        rear_wear = (wear[0] + wear[1]) / 2.0
        if avg_wear > 25.0:
            out.append(_fb("tyre_wear", 3, None,
                           f"平均胎耗 {avg_wear:.1f}%（>25）", "telemetry:tyre_wear"))
        elif avg_wear > 15.0:
            out.append(_fb("tyre_wear", 2, None,
                           f"平均胎耗 {avg_wear:.1f}% 偏高", "telemetry:tyre_wear"))
        if front_wear - rear_wear > 8.0:
            out.append(_fb("midcorner_understeer", 2, None,
                           f"前胎比后胎多磨 {front_wear - rear_wear:.1f}%",
                           "telemetry:tyre_wear"))
        elif rear_wear - front_wear > 8.0:
            out.append(_fb("exit_wheelspin", 2, None,
                           f"后胎比前胎多磨 {rear_wear - front_wear:.1f}%",
                           "telemetry:tyre_wear"))
    return out


def _from_brakes(telemetry: dict[str, Any]) -> list[dict[str, Any]]:
    """刹车温度过热/前后失衡 → 制动衰减与制动稳定性；ABS 介入 → 锁死倾向。"""
    out: list[dict[str, Any]] = []
    temps = _nums(telemetry.get("m_brakesTemperature"))
    # 注意：刹车温度**可能缺失**（老包/模拟数据），但下面的 ABS 判定与温度无关，
    # 因此不能在这里提前 return（早期版本如此，导致 `m_antiLockBrakes` 这一
    # 全新信号永远检测不到 —— 实测测试抓到）。
    if len(temps) >= 4:
        warn, severe = 500.0, 600.0
        if telemetry.get("is_soft_compound"):
            warn, severe = 450.0, 550.0
        elif telemetry.get("is_hard_compound"):
            warn, severe = 550.0, 660.0
        avg = sum(temps[:4]) / 4.0
        if avg > severe:
            out.append(_fb("brake_fade", 3, None,
                           f"刹车均温 {avg:.0f}°C（>{severe:.0f}）热衰减",
                           "telemetry:brake_temp"))
        elif avg > warn:
            out.append(_fb("brake_fade", 2, None,
                           f"刹车均温 {avg:.0f}°C（>{warn:.0f}）偏高",
                           "telemetry:brake_temp"))
        rear = (temps[0] + temps[1]) / 2.0
        front = (temps[2] + temps[3]) / 2.0
        if abs(front - rear) > 120.0:
            out.append(_fb("brake_instability", 2, None,
                           f"前后轴刹车温差 {front - rear:+.0f}°C（制动力分配失衡）",
                           "telemetry:brake_temp"))
    if telemetry.get("anti_lock_brakes"):
        out.append(_fb("lockup", 2, None, "检测到防抱死介入（锁死倾向）",
                       "telemetry:abs"))
    return out


def _from_damage(telemetry: dict[str, Any]) -> list[dict[str, Any]]:
    """车损 → 气动/底盘失衡（引擎里**没有**任何规则覆盖这一项）。

    机理（F1 领域共识）：
    - 前翼损伤 → 前轴下压力损失 → 转向不足，且随速度升高而加剧；
    - 底板/地板损伤 → 扩散器失效 → 高速稳定性下降 + 车高利用变差；
    - 后翼损伤 → 后轴下压力损失 → 高速弯转向过度。
    车手通常只会说"车不对了"，不会告诉你伤了哪一块 —— 遥测知道。
    """
    out: list[dict[str, Any]] = []
    floor = _num(telemetry.get("floor_damage"))
    wing = _num(telemetry.get("wing_damage_max"))
    severe = bool(telemetry.get("damage_severe"))
    if wing is not None and wing > 0.0:
        strength = 3 if wing >= 40.0 else 2
        out.append(_fb("understeer", strength, None,
                       f"前翼损伤 {wing:.0f}%（前轴下压力损失 → 推头，速度越高越明显）",
                       "telemetry:damage"))
    if floor is not None and floor > 0.0:
        strength = 3 if floor >= 40.0 else 2
        out.append(_fb("high_speed_instability", strength, None,
                       f"底板损伤 {floor:.0f}%（扩散器效率下降 → 高速失稳）",
                       "telemetry:damage"))
        out.append(_fb("bottoming", 2, None,
                       f"底板损伤 {floor:.0f}%（车高利用变差，容易触地）",
                       "telemetry:damage"))
    if severe:
        out.append(_fb("high_speed_instability", 3, None,
                       "整圈存在严重车损（damage_severe） → 稳定性需求上调",
                       "telemetry:damage"))
    return out


def _from_driving_inputs(telemetry: dict[str, Any]) -> list[dict[str, Any]]:
    """转向/油门/直道比例 → 入弯迟钝、出弯牵引、直道阻力。"""
    out: list[dict[str, Any]] = []
    max_steer = _num(telemetry.get("max_steer"))
    avg_steer = _num(telemetry.get("avg_steer"))
    avg_speed = _num(telemetry.get("avg_speed"))
    if max_steer is not None and max_steer > 0.35:
        out.append(_fb("turnin_unresponsive", 2, None,
                       f"整圈最大转向 {max_steer:.2f}（入弯需大幅打方向）",
                       "telemetry:steer"))
    if avg_steer is not None and avg_steer > 0.20:
        out.append(_fb("midcorner_unstable", 2, None,
                       f"整圈平均转向 {avg_steer:.2f}（反复修正）",
                       "telemetry:steer"))
    straight = _num(telemetry.get("aero_straight_ratio")) \
        or _num(telemetry.get("straight_ratio"))
    if straight is not None and straight > 0.45 and avg_speed is not None \
            and avg_speed > 150.0:
        out.append(_fb("straight_slow", 2, None,
                       f"直道占比 {straight:.2f} 且均速 {avg_speed:.0f} km/h（阻力偏大）",
                       "telemetry:aero"))
    if telemetry.get("traction_control"):
        out.append(_fb("exit_wheelspin", 2, None, "检测到牵引控制介入（出弯打滑）",
                       "telemetry:tc"))
    return out


# --------------------------------------------------------------------------- #
# 模型证据：代理模型残差
# --------------------------------------------------------------------------- #
def link_findings_to_changes(
    findings: list[dict[str, Any]],
    setup_delta: dict[str, float] | None,
) -> list[dict[str, Any]]:
    """把每条遥测发现映射到它**经由哪些参数**得到处理（"问题 → 优化方案"）。

    归因口径（与引擎同源，不是另编一套）：
        ``症状 → SYMPTOM_STAGE_TO_DX(阶段)`` 得到诊断维度；
        再用耦合矩阵 ``C`` 查出该维度上非零的参数 —— 这些参数若在最终建议里
        有非零改动，即为该发现对应的优化动作。

    这不是严格的因果归因（一个参数可能同时服务多个需求），而是**可追溯的
    联动说明**：每个改动都能说清"它由哪个遥测发现引入的需求驱动"。
    若某条发现在最终建议里没有任何对应改动，会显式给出 ``unresolved`` 提示 ——
    不假装问题已被解决。

    Args:
        findings: :func:`diagnose_from_telemetry` 的输出。
        setup_delta: 最终调教调整量 ``{param: delta}``。

    Returns:
        每项含 ``finding``（原始发现）、``dx_dims``（联动维度）、
        ``changes``（参数改动列表）、``resolved``（是否有对应改动）。
    """
    from setup_tuner.domain.setup import ALL_SETUP_FIELDS

    from .coupling import get_coupling
    from .diagnostic import SYMPTOM_STAGE_TO_DX

    labels = {f.name: f.label_zh for f in ALL_SETUP_FIELDS}
    delta = setup_delta or {}
    plan: list[dict[str, Any]] = []
    for item in findings or []:
        symptom = str(item.get("symptom") or "")
        table = SYMPTOM_STAGE_TO_DX.get(symptom)
        if not table:
            continue
        stage = item.get("category")
        dims = table.get(stage) or next(iter(table.values()))
        active_dims = [d for d, v in dims.items() if v]
        changes: list[dict[str, Any]] = []
        for param, value in delta.items():
            if not value:
                continue
            for dim in active_dims:
                cell = get_coupling(dim, param)
                if cell is not None and cell.value:
                    changes.append({
                        "param": param,
                        "label": labels.get(param, param),
                        "delta": round(float(value), 4),
                        "via_dim": dim,
                    })
                    break
        plan.append({
            "symptom": symptom,
            "corner": item.get("corner_number"),
            "strength": item.get("strength"),
            "source": item.get("source"),
            "evidence": item.get("evidence"),
            "dx_dims": active_dims,
            "changes": changes,
            "resolved": bool(changes),
        })
    return plan


def _from_surrogate_residual(
    track_id: str,
    corner_times: dict[int, float],
    conditions: dict[str, Any],
    tolerance: float = 0.15,
) -> list[dict[str, Any]]:
    """用遥测训练的弯速模型算出期望用时，实测偏慢 → 推断症状。

    关键点：输出的症状将在 merge_with_driver_feedbacks 中与车手反馈合并，
    并通过 class_weighted_dx 进入 engine 的 dx 计算链路，真正影响
    setup_delta 与调教建议。

    Args:
        track_id: 赛道标识。
        corner_times: 逐弯**实际**通过时间（秒），来自逐点遥测切弯。
        conditions: 工况（轮胎/天气/温度），用于模型推理。
        tolerance: 允许的相对偏差（超过即视为损失时间）。

    Returns:
        隐式症状列表；代理模型不可用或该赛道无期望值时返回空列表。
    """
    from .surrogate import get_surrogate

    model = get_surrogate()
    if not model.available:
        return []
    expected = model.expected_corner_s(track_id)
    if not expected:
        return []
    from setup_tuner.domain.track import get_track_by_id

    track = get_track_by_id(track_id)
    class_by_number = (
        {c.number: str(c.corner_type).lower() for c in track.corners}
        if track is not None else {}
    )
    out: list[dict[str, Any]] = []
    for corner, actual in corner_times.items():
        exp = expected.get(int(corner))
        if not exp or exp <= 0.0:
            continue
        deficit = (actual - exp) / exp
        if deficit <= tolerance:
            continue
        cls = class_by_number.get(int(corner), "medium")
        strength = 3 if deficit > tolerance * 2.5 else 2
        if cls == "slow":
            symptom, why = "exit_wheelspin", "慢弯（机械抓地/牵引）"
        elif cls == "fast":
            symptom, why = "high_speed_instability", "快弯（下压力/稳定性）"
        else:
            symptom, why = "midcorner_understeer", "中速弯（前轴平衡）"
        out.append(_fb(
            symptom, strength, int(corner),
            f"T{corner} 实测 {actual:.3f}s vs 模型期望 {exp:.3f}s"
            f"（慢 {deficit * 100:.1f}%）→ 归因 {why}",
            "surrogate:corner_residual",
        ))
    return out


# --------------------------------------------------------------------------- #
# 对外入口
# --------------------------------------------------------------------------- #
def diagnose_from_telemetry(
    telemetry: dict[str, Any] | None,
    track_id: str,
    corner_times: dict[int, float] | None = None,
    conditions: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """从遥测自动发现车手未反馈的问题，返回与车手反馈同构的症状条目。

    Args:
        telemetry: 整圈遥测摘要（LapAggregator / 训练导出产出的 lap_agg 结构）。
        track_id: 赛道标识。
        corner_times: 可选的逐弯实际用时（有逐点遥测时提供，启用模型证据）。
        conditions: 可选的工况（轮胎/天气/温度），供模型证据使用。

    Returns:
        隐式症状列表，每项含 ``corner_number`` / ``symptom`` / ``strength`` /
        ``category`` / ``source`` / ``evidence`` / ``implicit=True``。
        无遥测时返回空列表（中性降级，不抛错）。
    """
    if not telemetry:
        return []
    out: list[dict[str, Any]] = []
    out.extend(_from_kerbs(telemetry))
    out.extend(_from_ride_height(telemetry))
    out.extend(_from_tyres(telemetry))
    out.extend(_from_brakes(telemetry))
    out.extend(_from_damage(telemetry))
    out.extend(_from_driving_inputs(telemetry))
    if corner_times:
        out.extend(_from_surrogate_residual(
            track_id, corner_times, conditions or {},
        ))
    # 同一 (弯, 症状) 只保留最强的一条
    best: dict[tuple[Any, str], dict[str, Any]] = {}
    for item in out:
        key = (item.get("corner_number"), item.get("symptom"))
        cur = best.get(key)
        if cur is None or float(item["strength"]) > float(cur["strength"]):
            best[key] = item
    return sorted(
        best.values(),
        key=lambda d: (-float(d["strength"]), str(d["symptom"])),
    )


def merge_with_driver_feedbacks(
    driver_feedbacks: list[dict[str, Any]] | None,
    implicit: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """合并车手反馈与隐式诊断；车手反馈优先（同弯同症状时不重复）。"""
    merged: list[dict[str, Any]] = list(driver_feedbacks or [])
    taken = {
        (fb.get("corner_number"), fb.get("symptom"))
        for fb in merged
    }
    for item in implicit or []:
        key = (item.get("corner_number"), item.get("symptom"))
        if key in taken:
            continue
        merged.append(item)
        taken.add(key)
    return merged
