"""赛道特征系数 —— 让调教建议真正「因赛道而异」（S1）。

问题背景（2026-09 审计）：
    ``engine.generate_suggestion`` 虽然接收 ``track_id``，但此前**只把它传给神经网络分支**
    （而该分支因维度常量失配实际不可用）。规则引擎的 ``Dx × C`` 与赛道无关，
    因此云端实测 **suzuka 与 monza 在相同症状下输出逐位相同** —— 24 条赛道共用一套建议。

设计：
    在 ``Dx × C`` 之后、clamp 之前引入一层「赛道系数」（与 ``telemetry_gain`` 同构，
    按参数相乘）。系数按 ``track_type`` 分组，取值来自通用调校共识而非臆测的精确数值，
    因此刻意控制在 0.85–1.25 的温和区间：**只改敏感度，不改方向**。

系数含义与依据（每题均为"该参数在该类赛道上的调整敏感度"）：

    =========================== ==================================================
    赛道类型                     依据
    =========================== ==================================================
    high_speed_low_downforce     低下压力赛道（Monza/Spielberg/Baku/Las Vegas/
    （5 条：jeddah/montreal/     Jeddah/Montreal）：翼片直接影响极速，故翼片增益↑；
    monza/baku/las_vegas）       离地间隙已压到极限、再压收益小而刮底风险高，故↓；
                                 长直线依赖差速锁止，故该项↑。
    high_downforce               高下压力赛道（Hungaroring/Zandvoort）：弯中抓地与
    （2 条）                     轮胎几何（外倾/束角）主导，故翼片与几何增益↑；
                                 侧倾控制更关键，防倾杆↑。
    street                       街道赛（Monaco/Singapore/Miami/Madrid）：路面颠簸
    （4 条）                     且抓地力低 → 离地间隙、机械抓地、防倾杆更敏感；
                                 直道极速影响有限，故翼片增益↓。
    medium / mixed               无一致偏向，保持 1.0（**不臆测**）。
    =========================== ==================================================

出处：F1 官方调教指南中按赛道类型给出的调校优先级；具体倍数为本项目标定的
"敏感度权重"，不是官方数值，仅用于缩放调整幅度。
"""

from __future__ import annotations

from setup_tuner.domain.setup import ALL_SETUP_FIELDS
from setup_tuner.domain.track import get_track_by_id

# 参数分组（便于按类给系数）
PARAM_GROUPS: dict[str, tuple[str, ...]] = {
    "aero": ("front_wing", "rear_wing"),
    "ride_height": ("front_ride_height", "rear_ride_height"),
    "arb": ("front_anti_roll_bar", "rear_anti_roll_bar"),
    "mech_grip": ("front_suspension", "rear_suspension"),
    "geometry": ("front_camber", "rear_camber", "front_toe", "rear_toe"),
    "brake": ("brake_pressure", "brake_bias"),
    "diff": ("on_throttle_diff", "off_throttle_diff"),
    "tyre_pressure": (
        "front_left_tyre_pressure", "front_right_tyre_pressure",
        "rear_left_tyre_pressure", "rear_right_tyre_pressure",
    ),
}

# 赛道类型 → {参数组: 敏感度倍数}；未列出的组与赛道类型一律 1.0
TRACK_TYPE_MULTIPLIERS: dict[str, dict[str, float]] = {
    "high_speed_low_downforce": {
        "aero": 1.25,
        "ride_height": 0.90,
        "diff": 1.10,
        "arb": 0.95,
    },
    "high_downforce": {
        "aero": 1.20,
        "geometry": 1.10,
        "arb": 1.10,
    },
    "street": {
        "ride_height": 1.25,
        "arb": 1.15,
        "mech_grip": 1.15,
        "aero": 0.85,
        "geometry": 0.90,
    },
    "medium": {},
    "mixed": {},
}

# 单一事实来源：全部 21 参数的默认系数
NEUTRAL_GAIN: dict[str, float] = {spec.name: 1.0 for spec in ALL_SETUP_FIELDS}


def gain_for_track_type(track_type: str) -> dict[str, float]:
    """按赛道类型返回 {param: 敏感度倍数}（覆盖全部 21 参数）。"""
    gain = dict(NEUTRAL_GAIN)
    multipliers = TRACK_TYPE_MULTIPLIERS.get(track_type, {})
    for group, factor in multipliers.items():
        for param in PARAM_GROUPS.get(group, ()):
            gain[param] = factor
    return gain


def track_gain_for(track_id: str) -> dict[str, float]:
    """按赛道标识返回 {param: 敏感度倍数}。

    未知赛道返回全 1.0（不臆测）；赛道类型为 ``medium``/``mixed`` 时同样为全 1.0。
    """
    track = get_track_by_id(track_id)
    if track is None:
        return dict(NEUTRAL_GAIN)
    return gain_for_track_type(track.track_type)
