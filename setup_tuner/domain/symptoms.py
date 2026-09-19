"""F1 2026 调教症状枚举（22 项，4 类）。

逐字对齐 spec FR-FBK-02：15 种症状分入弯/弯中/出弯/全局四类，
每项含中文标签、类别与描述。强度 1–3（轻微/明显/严重），默认 2。

task-82 扩展（2026-09-17，用户要求「反馈选项偏少，增加 3~4 个并优化现有」）：
- 新增 4 项：brake_instability（刹车中车身不稳）/ kerb_instability（压路肩弹跳失控）/
  tyre_graining（轮胎粒化）/ brake_fade（刹车过热衰减）；
  四项全部可被遥测佐证（悬挂加速度/刹车温度/滑移），与「无反馈走遥测发现」闭环。
- bottoming 标签由「直道刮底」放宽为「刮底」：触地不限于直道（快弯压缩同样触发，
  遥测 plank_bottoming 也不分直道/弯中）。

task-61 扩展（2026-09-16）：
- 新增 3 项：exit_understeer（出弯转向不足）/ exit_unstable（出弯车身不稳）/
  tyre_overheat（胎温过高）—— 补齐出弯阶段（原仅 2 项）与"有规则没入口"的胎温缺口；
- ``lap_slow``（圈速不高）从 UI 可勾列表降级为报告综合结论：其 Dx 弥散在 5 个维度，
  属于"其他症状的结果"；枚举与 Dx 映射保留以兼容旧数据，UI 不再展示。

关键：tyre_wear 中文标签为「胎耗偏高」（非「胎温」），对应轮胎寿命维度，
与胎温（tyre temperature）是不同概念。

task-60 扩展：新增 3 项症状（midcorner_understeer / exit_oversteer /
high_speed_instability），支持同一症状在不同弯道阶段有不同的 Dx 映射。
"""

from __future__ import annotations

from enum import StrEnum


class SymptomCategory(StrEnum):
    """症状所属的弯道阶段类别。"""

    ENTRY = "entry"    # 入弯
    APEX = "apex"      # 弯中
    EXIT = "exit"      # 出弯
    GLOBAL = "global"  # 全局


class Symptom(StrEnum):
    """22 项调教症状枚举（spec FR-FBK-02 15 项 + task-60/61/82 扩展）。"""

    # 入弯 entry (5)
    UNDERSTEER = "understeer"              # 转向不足
    OVERSTEER = "oversteer"                # 转向过度
    TURNIN_UNRESPONSIVE = "turnin_unresponsive"  # 转向不灵敏
    BRAKE_LONG = "brake_long"              # 刹车距离长
    LOCKUP = "lockup"                      # 轮胎锁死
    BRAKE_INSTABILITY = "brake_instability"  # 刹车中车身不稳（task-82 新增）
    # 弯中 apex (3)
    MIDCORNER_UNDERSTEER = "midcorner_understeer"  # 弯中推头（task-60 新增）
    MIDCORNER_UNSTABLE = "midcorner_unstable"    # 车身不稳定
    MIDCORNER_TRACTION = "midcorner_traction"    # 弯中不能稳定加速
    # 出弯 exit (4)
    EXIT_WHEELSPIN = "exit_wheelspin"      # 出弯打滑
    EXIT_OVERSTEER = "exit_oversteer"      # 出弯甩尾（task-60 新增）
    EXIT_UNDERSTEER = "exit_understeer"    # 出弯转向不足（task-61 新增）
    EXIT_UNSTABLE = "exit_unstable"        # 出弯车身不稳（task-61 新增）
    # 全局 global (5)
    BOTTOMING = "bottoming"                # 刮底（底板触地）
    TYRE_WEAR = "tyre_wear"                # 胎耗偏高（非胎温！）
    TYRE_OVERHEAT = "tyre_overheat"        # 胎温过高（task-61 新增）
    STRAIGHT_SLOW = "straight_slow"        # 直道速度低
    LAP_SLOW = "lap_slow"                  # 圈速不高
    HIGH_SPEED_INSTABILITY = "high_speed_instability"  # 高速不稳（task-60 新增）
    KERB_INSTABILITY = "kerb_instability"  # 压路肩弹跳失控（task-82 新增）
    TYRE_GRAINING = "tyre_graining"        # 轮胎粒化（task-82 新增）
    BRAKE_FADE = "brake_fade"              # 刹车过热衰减（task-82 新增）


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
    Symptom.BRAKE_INSTABILITY: {
        "label": "刹车中车身不稳",
        "category": SymptomCategory.ENTRY.value,
        "description": "重刹时车尾摆动或车身晃动，难以稳定把车速降下来。",
    },
    # 弯中 apex
    Symptom.MIDCORNER_UNDERSTEER: {
        "label": "弯中推头",
        "category": SymptomCategory.APEX.value,
        "description": "弯中持续转向不足，前轮抓地不足。",
    },
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
    Symptom.EXIT_OVERSTEER: {
        "label": "出弯甩尾",
        "category": SymptomCategory.EXIT.value,
        "description": "出弯加速时车尾外甩。",
    },
    Symptom.EXIT_UNDERSTEER: {
        "label": "出弯转向不足",
        "category": SymptomCategory.EXIT.value,
        "description": "出弯给油后车头推向外侧，前轴抓地不足。",
    },
    Symptom.EXIT_UNSTABLE: {
        "label": "出弯车身不稳",
        "category": SymptomCategory.EXIT.value,
        "description": "出弯阶段车身姿态晃动，难以保持线路。",
    },
    # 全局 global
    Symptom.BOTTOMING: {
        "label": "刮底",  # 不限直道：快弯/颠簸/压缩都会触地（task-82 由「直道刮底」放宽）
        "category": SymptomCategory.GLOBAL.value,
        "description": "底盘或底板触地刮擦（直道、快弯压缩或颠簸处都可能）。",
    },
    Symptom.TYRE_WEAR: {
        "label": "胎耗偏高",  # 逐字对齐 spec FR-FBK-02，非「胎温」
        "category": SymptomCategory.GLOBAL.value,
        "description": "轮胎磨损速率偏高（轮胎寿命维度，非胎温）。",
    },
    Symptom.TYRE_OVERHEAT: {
        "label": "胎温过高",
        "category": SymptomCategory.GLOBAL.value,
        "description": "轮胎工作温度超出最佳窗口，抓地力衰减。",
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
    Symptom.HIGH_SPEED_INSTABILITY: {
        "label": "高速不稳",
        "category": SymptomCategory.GLOBAL.value,
        "description": "高速段方向不稳定。",
    },
    Symptom.KERB_INSTABILITY: {
        "label": "压路肩弹跳失控",
        "category": SymptomCategory.GLOBAL.value,
        "description": "压路肩后车身弹跳剧烈甚至失稳——路肩不得不压（task-82 新增）。",
    },
    Symptom.TYRE_GRAINING: {
        "label": "轮胎粒化",
        "category": SymptomCategory.GLOBAL.value,
        "description": "轮胎表面因滑动起粒、抓地骤降（与磨损/过热不同的失效模式）。",
    },
    Symptom.BRAKE_FADE: {
        "label": "刹车过热衰减",
        "category": SymptomCategory.GLOBAL.value,
        "description": "长刹车段之后踏板变软、制动力明显下降。",
    },
}


# 强度范围 0–5，默认 3
INTENSITY_MIN = 1
INTENSITY_MAX = 3
DEFAULT_INTENSITY = 2

# 强度档位锚定文案（task-61：0–5 过细，收敛为 3 档）
INTENSITY_LABELS: dict[int, str] = {1: "轻微", 2: "明显", 3: "严重"}


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
            f"症状强度 {intensity} 越界，合法范围 [{INTENSITY_MIN}, {INTENSITY_MAX}]",
        )
    return intensity