"""Issue ID 注册表 — 基于 F1 physics.py downforce_balance 的物理闭环规则

=== 设计哲学 ===
一个车手反馈触发的不是单参数调教，而是“一套整体调校方案”。
参数在 F1 赛车中天然耦合：
    - AERO_BALANCE: 前翼 + 后翼 (气动前后比例)
    - MECHANICAL_ROLL: 前/后防倾 + 前/后外倾 (侧倾刚度分配)
    - TIRE_THERMAL: 胎压 + 外倾 (热管理/磨损)
    - DRIVETRAIN: 油门/滑行差速 (牵引/收尾)

每个 issue 可提供多套 COUPLED_SOLUTIONS：
    - 方案 A: 气动主导 (翼面调为主)
    - 方案 B: 机械主导 (防倾/外倾为主)
    - 方案 C: 综合均衡 (翼面+机械均调)

DIRECTION 仍保留作为“默认综合方案”的收敛来源，供向后兼容。
"""

from __future__ import annotations

from typing import Any

# =========================================================================== #
# 1. 耦合组定义：哪些参数需要协同调节
# =========================================================================== #
COUPLING_GROUPS: dict[str, list[str]] = {
    "AERO_BALANCE": ["front_wing", "rear_wing"],
    "MECHANICAL_ROLL": ["front_arb", "rear_arb", "front_camber", "rear_camber"],
    "TIRE_THERMAL": ["front_tyre_pressure", "rear_tyre_pressure", "front_camber", "rear_camber"],
    "DRIVETRAIN": ["on_throttle_diff", "off_throttle_diff"],
}

# =========================================================================== #
# 2. 单问题调校方向 (为 COUPLED_SOLUTIONS 提供基准)
# =========================================================================== #

DIRECTION: dict[str, dict[str, tuple[float, str]]] = {}

DIRECTION["understeer_in"] = {
    "front_wing": (+2.0, "增前翼 2 档：前下压↑，提高前轮载荷"),
    "rear_wing": (-1.0, "减后翼 1 档：后下压↓，平衡前后"),
    "front_arb": (-2.0, "软化前防倾 2 档：提升机械抓地"),
    "rear_arb": (+0.5, "硬化后防倾 0.5 档：限制尾部侧滑"),
}

DIRECTION["oversteer_in"] = {
    "rear_wing": (+2.0, "增后翼 2 档：后下压↑，稳住车尾"),
    "front_wing": (-1.0, "减前翼 1 档：前下压↓，减轻车头拉力"),
    "rear_arb": (-1.5, "软化后防倾 1.5 档：后轮贴地↑"),
    "on_throttle_diff": (-5.0, "软滑行差速 5%：收尾更稳"),
}

DIRECTION["brake_lock"] = {
    "front_brake_bias": (-2.0, "刹车分配前移 2%：减轻前轮锁死倾向"),
    "brake_pressure": (-5.0, "降刹车压力 5%：降低整体锁死风险"),
}

DIRECTION["steer_delay"] = {
    "front_wing": (+1.5, "增前翼 1.5 档：提升转向响应性"),
}

DIRECTION["steer_sharp"] = {
    "front_wing": (-1.0, "减前翼 1 档：降低前轮活跃度"),
    "rear_wing": (-0.5, "减后翼 0.5 档：降低尾部抖摆"),
}

DIRECTION["understeer_apex"] = {
    "front_wing": (+1.0, "增前翼 1 档：提升弯中前轮载荷"),
    "front_camber": (-0.2, "减负倾角 0.2°：提升弯中抓地"),
}

DIRECTION["oversteer_apex"] = {
    "rear_wing": (+1.5, "增后翼 1.5 档：支住车尾"),
    "front_wing": (-1.0, "减前翼 1 档：减少前轮被拉"),
    "rear_camber": (+0.3, "增负倾角 0.3°：提升尾部抓地极限"),
}

DIRECTION["rear_slip"] = {
    "rear_wing": (+1.0, "增后翼 1 档：支住车尾"),
    "on_throttle_diff": (-5.0, "差速激进 5%：收尾更稳"),
    "front_brake_bias": (+2.0, "刹车分配后移 2%：减轻前轮抱死"),
}

DIRECTION["steer_delay_apex"] = {
    "front_wing": (+1.0, "增前翼 1 档：提升响应"),
}

DIRECTION["steer_sharp_apex"] = {
    "front_wing": (-0.5, "减前翼 0.5 档：降低活跃度"),
    "rear_camber": (+0.2, "稍增负倾 0.2°：提升尾部抓地"),
}

DIRECTION["understeer_out"] = {
    "rear_wing": (+1.0, "增后翼 1 档：平衡出弎时前后荷重"),
    "off_throttle_diff": (-3.0, "软滑行差速 3%：收尾更顺滑"),
}

DIRECTION["oversteer_out"] = {
    "rear_wing": (+1.0, "增后翼 1 档：稳住尾部"),
    "off_throttle_diff": (-4.0, "软滑行差速 4%：收油时尾部收心"),
}

DIRECTION["acceleration"] = {
    "on_throttle_diff": (-5.0, "差速激进 5%：出弎牵引↑"),
}

DIRECTION["rear_slip_out"] = {
    "on_throttle_diff": (-5.0, "差速激进 5%：稳收尾"),
    "rear_camber": (+0.2, "稍增负倾 0.2°：提升尾部抓地"),
}

DIRECTION["steer_delay_out"] = {
    "rear_wing": (+1.0, "增后翼 1 档：提升尾部稳定"),
}

DIRECTION["steer_sharp_out"] = {
    "rear_wing": (-1.0, "减后翼 1 档：降低尾部活跃"),
}

DIRECTION["straight_slow"] = {
    "front_wing": (-3.0, "减前翼 3 档：显著降低阻力"),
    "rear_wing": (-3.0, "减后翼 3 档：降低阻力"),
}

DIRECTION["lap_slow"] = {
    "front_wing": (-1.5, "减前翼 1.5 档：提升直道速度"),
    "rear_wing": (-1.5, "减后翼 1.5 档：提升直道速度"),
    "fuel_mix": (-1.0, "降油耗混合比 1 级"),
}

DIRECTION["tire_wear"] = {
    "front_tyre_pressure": (+1.0, "增前胎压 ~0.5 PSI：减小接触面积降低磨损"),
    "rear_tyre_pressure": (+1.0, "增后胎压 ~0.5 PSI：缓解后轮磨损"),
    "front_wing": (-2.0, "减前翼 2 档：降低前轮胎温负荷"),
    "rear_wing": (-2.0, "减后翼 2 档：降低尾部胎温负荷"),
    "front_camber": (+0.2, "略减负倾 0.2°：减小内缘热点"),
    "rear_camber": (+0.2, "略减负倾 0.2°：减小尾部热点"),
}

DIRECTION["fuel_wear"] = {
    "fuel_mix": (-1.0, "降油耗混合比 1 级"),
}

DIRECTION["balance"] = {
    "front_arb": (-1.0, "软化前防倾 1 档"),
    "rear_arb": (+1.0, "硬化后防倾 1 档"),
    "front_wing": (-0.5, "减前翼 0.5 档"),
    "rear_wing": (+0.5, "增后翼 0.5 档"),
}

DIRECTION["ers"] = {
    "ers_deployment_mode": (-1.0, "调节 ERS 部署模式：降低能耗"),
}

DIRECTION["traction"] = {
    "front_wing": (-1.0, "减前翼 1 档：减轻车头拉扭力"),
    "on_throttle_diff": (-5.0, "软滑行差速 5%：让轮子顺势转"),
}

# =========================================================================== #
# 3. 耦合解决方案：每个问题的多套整体调校方案
# =========================================================================== #

COUPLED_SOLUTIONS: dict[str, list[dict[str, Any]]] = {
    # ------------------- 推头 (understeer) -------------------
    "understeer_in": [
        {
            "name": "气动平衡法",
            "id": "aero_balance",
            "coupling_groups": ["AERO_BALANCE"],
            "params": {"front_wing": +2.0, "rear_wing": -1.5},
            "summary": "增前翼提升前下压，减后翼降低后负荷，显著纠正推头。直道需权衡后翼减小的阻力损失。",
            "tradeoff": "直道速度略降低 1-2 km/h，优势在弯道转向能力提升。",
        },
        {
            "name": "机械抓地法",
            "id": "mechanical_grip",
            "coupling_groups": ["MECHANICAL_ROLL", "TIRE_THERMAL"],
            "params": {"front_arb": -2.0, "front_camber": -0.3, "front_tyre_pressure": +0.5},
            "summary": "软化前防倾、增前负倾、升前胎压，三者协同提升前轮机械抓地，纠正推头。",
            "tradeoff": "轮胎磨损从内缘转向外缘，需后期调整胎温管理。",
        },
        {
            "name": "综合均衡法",
            "id": "balanced",
            "coupling_groups": ["AERO_BALANCE", "MECHANICAL_ROLL"],
            "params": {"front_wing": +1.5, "rear_wing": -1.0, "front_arb": -1.0, "rear_arb": +0.5},
            "summary": "翼面 + 前后防倾协同，兼顾气动与机械，适合不确定推头程度时。",
            "tradeoff": "治病效果温和，需循环效果。",
        },
    ],
    # ------------------- 甩尾 (oversteer) -------------------
    "oversteer_in": [
        {
            "name": "后翼稳尾法",
            "id": "tail_stabilize",
            "coupling_groups": ["AERO_BALANCE"],
            "params": {"rear_wing": +2.0, "front_wing": -1.0},
            "summary": "增后翼压实车尾，减前翼防止车头被拉，快速止滑。",
            "tradeoff": "增后翼略增总阻力，出弎转向反应减缓。",
        },
        {
            "name": "差速收心法",
            "id": "diff_control",
            "coupling_groups": ["MECHANICAL_ROLL", "DRIVETRAIN"],
            "params": {"rear_arb": -1.5, "on_throttle_diff": -4.0, "off_throttle_diff": -3.0},
            "summary": "软后防倾+滑行差速收油，收尾更稳，适合尾轻易打滑。",
            "tradeoff": "差速收紧后，出弎抓地变化剧烈，需适应性上滑。",
        },
    ],
    # ------------------- 胎耗 -------------------
    "tire_wear": [
        {
            "name": "阻力减负法",
            "id": "low_drag_thermal",
            "coupling_groups": ["AERO_BALANCE", "TIRE_THERMAL"],
            "params": {"front_wing": -2.0, "rear_wing": -2.0, "front_tyre_pressure": +0.8, "rear_tyre_pressure": +0.8},
            "summary": "减翼面降低下压热负荷，增胎压缩小接触面积，双管齐下降低磨损。",
            "tradeoff": "后轮抓地下降，高速弯需靠机械调校弥补。",
        },
        {
            "name": "负倾均衡法",
            "id": "camber_balance",
            "coupling_groups": ["MECHANICAL_ROLL", "TIRE_THERMAL"],
            "params": {"front_camber": +0.3, "rear_camber": +0.3, "front_tyre_pressure": +0.5, "rear_tyre_pressure": +0.5},
            "summary": "减小负倾角 + 增胎压，直接降低内缘过热磨损。",
            "tradeoff": "外缘抓地稍减，需轮胎寿命跟踪。",
        },
    ],
    # ------------------- 直道慢 (straight_slow) -------------------
    "straight_slow": [
        {
            "name": "减阻主力法",
            "id": "drag_reduction",
            "coupling_groups": ["AERO_BALANCE"],
            "params": {"front_wing": -4.0, "rear_wing": -4.0},
            "summary": "大幅减前后翼，显著降低 D 项阻力，直道 10-15 km/h 速度提升。",
            "tradeoff": "弯道下压力下降，推头/甩尾问题可能浮现，需要弯道局部加固。",
        },
        {
            "name": "燃油提升法",
            "id": "fuel_boost",
            "coupling_groups": ["AERO_BALANCE", "DRIVETRAIN"],
            "params": {"front_wing": -2.0, "rear_wing": -2.0, "fuel_mix": +1.0, "ers_deployment_mode": +1.0},
            "summary": "减翼降阻 + 提混合比/ERS，兼顾燃油消耗与直道动力输出。",
            "tradeoff": "油耗可能反弹，需观察燃油箱消耗曲线。",
        },
    ],
}

# =========================================================================== #
# 4. 自动生成函数
# =========================================================================== #

def _build_adjustments(issue_id: str) -> list[tuple[str, float, str]]:
    """从 DIRECTION 生成默认 (field, delta, reason) 列表"""
    result = []
    for field, (delta, reason) in DIRECTION.get(issue_id, {}).items():
        result.append((field, float(delta), reason))
    return result


def _build_contradictions(issue_id: str) -> list[dict[str, Any]]:
    """从 DIRECTION 生成矛盾规则：相反方向调节即为矛盾"""
    result = []
    for field, (delta, _) in DIRECTION.get(issue_id, {}).items():
        opp_dir = "increase" if delta < 0 else "decrease"
        result.append({"param": field, "direction": opp_dir,
                       "reason": f"{field} {opp_dir} 方向会抵消本次调教效果"})
    return result


def _infer_fields(issue_id: str) -> list[str]:
    return list(DIRECTION.get(issue_id, {}).keys())


# =========================================================================== #
# 5. Issue 注册表 (ISSUE_REGISTRY)
# =========================================================================== #

ISSUE_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {}

_phase_map = {
    "turn-in": ["understeer_in", "oversteer_in", "brake_lock", "steer_delay", "steer_sharp"],
    "apex": ["understeer_apex", "oversteer_apex", "rear_slip", "steer_delay_apex", "steer_sharp_apex"],
    "turn-out": ["understeer_out", "oversteer_out", "acceleration", "rear_slip_out", "steer_delay_out", "steer_sharp_out"],
    "straight": ["straight_slow", "lap_slow"],
    "global": ["tire_wear", "fuel_wear", "balance", "ers", "traction"],
}

_phase_labels = {
    "understeer_in": "转向不足 / 推头", "oversteer_in": "转向过度 / 甩尾",
    "brake_lock": "刹车锁死", "steer_delay": "转向迟钝", "steer_sharp": "转向过于灵敏",
    "understeer_apex": "弯中推头", "oversteer_apex": "弯中甩尾", "rear_slip": "车尾易打滑",
    "steer_delay_apex": "弯中转向迟钝", "steer_sharp_apex": "弯中转向过度灵敏",
    "understeer_out": "出弎推头", "oversteer_out": "出弎甩尾", "acceleration": "加速不足",
    "rear_slip_out": "出弎车尾易打滑", "steer_delay_out": "出弎转向迟钝", "steer_sharp_out": "出弎过于灵敏",
    "straight_slow": "直道慢", "lap_slow": "单圈慢",
    "tire_wear": "轮胎磨损过快", "fuel_wear": "油耗", "balance": "前后平衡",
    "ers": "ERS 能量管理", "traction": "牵引力不足",
}

for phase, issues in _phase_map.items():
    ISSUE_REGISTRY[phase] = {}
    for iid in issues:
        ISSUE_REGISTRY[phase][iid] = {
            "label": _phase_labels.get(iid, iid),
            "phase": phase,
            "physics": "f1opt.model.physics.AeroModel + TireThermalModel",
            "fields": _infer_fields(iid),
            "adjustments": _build_adjustments(iid),
            "contradictions": _build_contradictions(iid),
        }

# =========================================================================== #
# 6. 查询函数
# =========================================================================== #

def get_sub_intent_for_issue(issue_id: str) -> str | None:
    core_map = {
        "understeer_in": "understeer", "understeer_apex": "understeer", "understeer_out": "understeer",
        "oversteer_in": "oversteer", "oversteer_apex": "oversteer", "oversteer_out": "oversteer",
        "brake_lock": "brake", "brake_lift": "brake", "steer_delay": "steering", "steer_sharp": "steering",
        "steer_delay_apex": "steering", "steer_sharp_apex": "steering", "steer_delay_out": "steering", "steer_sharp_out": "steering",
        "traction": "traction", "acceleration": "traction",
        "tyre_wear": "tyre_wear", "fuel_wear": "ers", "ers": "ers",
        "balance": "balance", "straight_slow": "aero", "lap_slow": "aero",
    }
    return core_map.get(issue_id)


def get_fields_for_issue(issue_id: str) -> list[str]:
    for phase, issues in ISSUE_REGISTRY.items():
        if issue_id in issues:
            return issues[issue_id]["fields"]
    return []


def get_adjustments_for_issue(issue_id: str) -> list[tuple[str, float, str]]:
    for phase, issues in ISSUE_REGISTRY.items():
        if issue_id in issues:
            return issues[issue_id]["adjustments"]
    return []


def get_contradictions_for_issue(issue_id: str) -> list[dict[str, Any]]:
    for phase, issues in ISSUE_REGISTRY.items():
        if issue_id in issues:
            return issues[issue_id].get("contradictions", [])
    return []


def get_coupled_solutions_for_issue(issue_id: str) -> list[dict[str, Any]]:
    """返回该 issue 的 多套整体调校方案。若无耦合方案，返回默认综合方案。"""
    return COUPLED_SOLUTIONS.get(issue_id, [{
        "name": "默认综合方案",
        "id": "default",
        "coupling_groups": list(set(sum([COUPLING_GROUPS[g] for g in ["AERO_BALANCE", "MECHANICAL_ROLL"] if g in COUPLING_GROUPS], []))),
        "params": dict(DIRECTION.get(issue_id, {})),
        "summary": f"基于 {issue_id} 的默认调校方向。",
        "tradeoff": "综合考虑气动与机械，不极端。",
    }])


def get_issues_by_phase(phase: str) -> list[str]:
    return list(ISSUE_REGISTRY.get(phase, {}).keys())


def get_all_issue_ids() -> list[str]:
    result = []
    for phase, issues in ISSUE_REGISTRY.items():
        result.extend(issues.keys())
    return result


__all__ = [
    "ISSUE_REGISTRY", "COUPLED_SOLUTIONS", "COUPLING_GROUPS", "DIRECTION",
    "get_all_issue_ids", "get_issues_by_phase", "get_sub_intent_for_issue",
    "get_fields_for_issue", "get_adjustments_for_issue", "get_contradictions_for_issue",
    "get_coupled_solutions_for_issue",
]