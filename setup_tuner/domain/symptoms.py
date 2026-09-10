"""F1 2026 调教症状枚举（12 项，4 类）。

逐字对齐 spec FR-FBK-02：12 种症状分入弯/弯中/出弯/全局四类，
每项含中文标签、类别与描述。强度 0–5，默认 3。

关键：tyre_wear 中文标签为「胎耗偏高」（非「胎温」），对应轮胎寿命维度，
与胎温（tyre temperature）是不同概念。
"""

from __future__ import annotations

from enum import Enum


class SymptomCategory(str, Enum):
    """症状所属的弯道阶段类别。"""

    ENTRY = "entry"    # 入弯
    APEX = "apex"      # 弯中
    EXIT = "exit"      # 出弯
    GLOBAL = "global"  # 全局


class Symptom(str, Enum):
    """12 项调教症状枚举（逐字对齐 spec FR-FBK-02）。"""

    # 入弯 entry (5)
    UNDERSTEER = "understeer"              # 转向不足
    OVERSTEER = "oversteer"                # 转向过度
    TURNIN_UNRESPONSIVE = "turnin_unresponsive"  # 转向不灵敏
    BRAKE_LONG = "brake_long"              # 刹车距离长
    LOCKUP = "lockup"                      # 轮胎锁死
    # 弯中 apex (2)
    MIDCORNER_UNSTABLE = "midcorner_unstable"    # 车身不稳定
    MIDCORNER_TRACTION = "midcorner_traction"    # 弯中不能稳定加速
    # 出弯 exit (1)
    EXIT_WHEELSPIN = "exit_wheelspin"      # 出弯打滑
    # 全局 global (4)
    BOTTOMING = "bottoming"                # 直道刮底
    TYRE_WEAR = "tyre_wear"                # 胎耗偏高（非胎温！）
    STRAIGHT_SLOW = "straight_slow"        # 直道速度低
    LAP_SLOW = "lap_slow"                  # 圈速不高


# 每症状的元信息：中文标签 / 类别 / 描述
SYMPTOM_INFO: dict[Symptom, dict[str, str]] = {
    # 入弯 entry
    Symptom.UNDERSTEER: {
        "label": "转向不足",
        "category": SymptomCategory.ENTRY.value,
        "description": "入弯时车头不愿指向弯心，前轮抓地不足。",
    },
    Symptom.OVERSTEER: {
        "label": "转向过度",
        "category": SymptomCategory.ENTRY.value,
        "description": "入弯时车尾外甩，后轮抓地不足。",
    },
    Symptom.TURNIN_UNRESPONSIVE: {
        "label": "转向不灵敏",
        "category": SymptomCategory.ENTRY.value,
        "description": "转动方向盘后车辆响应迟缓。",
    },
    Symptom.BRAKE_LONG: {
        "label": "刹车距离长",
        "category": SymptomCategory.ENTRY.value,
        "description": "制动距离过长，入弯点偏晚。",
    },
    Symptom.LOCKUP: {
        "label": "轮胎锁死",
        "category": SymptomCategory.ENTRY.value,
        "description": "制动时车轮抱死打滑。",
    },
    # 弯中 apex
    Symptom.MIDCORNER_UNSTABLE: {
        "label": "车身不稳定",
        "category": SymptomCategory.APEX.value,
        "description": "弯中车身姿态抖动或难以保持线路。",
    },
    Symptom.MIDCORNER_TRACTION: {
        "label": "弯中不能稳定加速",
        "category": SymptomCategory.APEX.value,
        "description": "弯中给油后驱动轮打滑，无法稳定加速。",
    },
    # 出弯 exit
    Symptom.EXIT_WHEELSPIN: {
        "label": "出弯打滑",
        "category": SymptomCategory.EXIT.value,
        "description": "出弯加速时车轮空转打滑。",
    },
    # 全局 global
    Symptom.BOTTOMING: {
        "label": "直道刮底",
        "category": SymptomCategory.GLOBAL.value,
        "description": "底盘在直道或颠簸处触地刮底。",
    },
    Symptom.TYRE_WEAR: {
        "label": "胎耗偏高",  # 逐字对齐 spec FR-FBK-02，非「胎温」
        "category": SymptomCategory.GLOBAL.value,
        "description": "轮胎磨损速率偏高（轮胎寿命维度，非胎温）。",
    },
    Symptom.STRAIGHT_SLOW: {
        "label": "直道速度低",
        "category": SymptomCategory.GLOBAL.value,
        "description": "直道尾速不足。",
    },
    Symptom.LAP_SLOW: {
        "label": "圈速不高",
        "category": SymptomCategory.GLOBAL.value,
        "description": "整体圈速未达预期。",
    },
}


# 强度范围 0–5，默认 3
INTENSITY_MIN = 0
INTENSITY_MAX = 5
DEFAULT_INTENSITY = 3


def get_symptoms_by_category(category: SymptomCategory | str) -> list[Symptom]:
    """按弯道阶段类别返回该类下全部症状（保持枚举定义顺序）。

    Args:
        category: SymptomCategory 枚举或其字符串值。

    Returns:
        该类别下的 Symptom 列表。
    """
    cat_value = category.value if isinstance(category, SymptomCategory) else str(category)
    return [sym for sym in Symptom if SYMPTOM_INFO[sym]["category"] == cat_value]


def get_symptom_label(sym: Symptom) -> str:
    """返回症状的中文标签。

    Args:
        sym: Symptom 枚举成员。

    Returns:
        中文标签字符串。
    """
    return SYMPTOM_INFO[sym]["label"]


def get_symptom_category(sym: Symptom) -> str:
    """返回症状所属的弯道阶段类别字符串值。"""
    return SYMPTOM_INFO[sym]["category"]


def validate_intensity(intensity: int) -> int:
    """校验强度值在 [0, 5] 区间。

    Raises:
        ValueError: 强度越界。
    """
    if intensity < INTENSITY_MIN or intensity > INTENSITY_MAX:
        raise ValueError(
            f"症状强度 {intensity} 越界，合法范围 [{INTENSITY_MIN}, {INTENSITY_MAX}]"
        )
    return intensity