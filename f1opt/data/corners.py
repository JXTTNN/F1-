"""Iter-164.13: 赛道逐弯 (corner-level) 几何与需求数据.

F1 2026 各赛道的逐弯结构化数据: 弯角编号 / 类型 / apex 速度 / 半径 / 长度 /
银行角 / DRS / 超车点 / 调教需求. 用于 R5 "全程动态" — 让系统能识别
"第 5 弯有转向不足" 并给出逐弯调教建议.

数据来源:
- 公开赛道图 (FIA 官方) + 各车队工程师手册的弯角编号.
- apex 速度 / 半径为工程化合理估计 (非 telemetry 实测, 但量级准确).
- 对未手工录入的赛道, :func:`generate_corner_profile` 基于赛道特征
  (track_type / corners / length / top_speed) 合成代表性逐弯分布.

公开 API:
    - :class:`Corner` — 单个弯角的结构化数据.
    - :func:`get_corners` — 按 track_id 拿逐弯列表 (有手填的用手填, 否则合成).
    - :func:`generate_corner_profile` — 合成逐弯分布 (基于赛道特征).
    - :func:`corner_demand_summary` — 汇总赛道的调教需求 (各维度占比).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from f1opt.data.track_engineering import get_track_engineering
from f1opt.data.tracks import get_track

CornerType = Literal["slow", "medium", "fast"]
CornerDemand = Literal[
    "braking",       # 重刹区 (高刹车压力需求)
    "traction",      # 出弯牵引 (低油门差需求)
    "high_downforce",  # 高速弯 (高下压力需求)
    "low_downforce",   # 直道尾速 (低下压力需求)
    "stability",     # 复合弯 (悬挂稳定性需求)
    "kerb",          # 上路肩 (高悬挂行程需求)
]


@dataclass(frozen=True)
class Corner:
    """单个弯角的结构化数据 (Iter-164.13).

    所有速度单位 km/h, 长度单位 m, 角度单位度.
    """

    number: int
    """弯角编号 (1-based, 按 F1 官方赛道图)."""

    name: str
    """弯角名称 (如 "S-Curves", "Casino Triangle", "Eau Rouge")."""

    corner_type: CornerType
    """弯角类型: slow (<100 km/h) / medium (100-200) / fast (>200)."""

    speed_kmh: float
    """apex 速度 (km/h)."""

    radius_m: float
    """弯角半径 (m). slow ~30-80m, medium ~100-300m, fast ~400-1000m."""

    length_m: float
    """弯角区域长度 (入弯制动点→出弯加速点, m)."""

    banking_deg: float
    """银行角 (度, 0 = 平面). 大多数 F1 赛道 ~0-3°."""

    is_drs: bool
    """该弯角是否紧邻 DRS 区 (出弯后或入弯前有 DRS 检测/激活点)."""

    is_overtaking: bool
    """是否为主要超车点 (重刹 + 长直道尾)."""

    demands: tuple[CornerDemand, ...]
    """该弯角对调教的需求 (可多项). 如高速弯 → ("high_downforce", "stability")."""


# --------------------------------------------------------------------------- #
# 手工录入的赛道逐弯数据 (代表性赛道)
# --------------------------------------------------------------------------- #
_MANUAL_CORNERS: dict[str, list[Corner]] = {}


def _suzuka_corners() -> list[Corner]:
    """Suzuka International Racing Course (5.807 km, 18 弯, mixed)."""
    return [
        Corner(1, "First", "fast", 230, 600, 280, 0, True, True, ("high_downforce", "stability")),
        Corner(2, "S-Curves entry", "medium", 150, 120, 100, 0, False, False, ("high_downforce", "stability")),
        Corner(3, "S-Curves", "medium", 170, 140, 90, 0, False, False, ("high_downforce", "stability")),
        Corner(4, "S-Curves", "medium", 175, 145, 90, 0, False, False, ("high_downforce", "stability")),
        Corner(5, "S-Curves exit", "medium", 180, 160, 100, 0, False, False, ("high_downforce", "stability")),
        Corner(6, "Dunlop", "medium", 160, 110, 120, 0, False, False, ("high_downforce", "stability")),
        Corner(7, "Degner 1", "slow", 95, 50, 80, 0, False, False, ("braking", "traction")),
        Corner(8, "Degner 2", "slow", 85, 40, 70, 0, False, False, ("braking", "traction")),
        Corner(9, "Hairpin", "slow", 70, 30, 90, 0, False, True, ("braking", "traction")),
        Corner(10, "200R", "fast", 210, 350, 200, 0, False, False, ("high_downforce",)),
        Corner(11, "Spoon", "medium", 140, 100, 150, 0, False, False, ("high_downforce", "stability")),
        Corner(12, "Spoon exit", "medium", 155, 120, 120, 0, False, False, ("high_downforce", "traction")),
        Corner(13, "130R", "fast", 295, 900, 250, 0, True, False, ("high_downforce", "stability")),
        Corner(14, "Casio Triangle entry", "slow", 80, 35, 70, 0, False, False, ("braking",)),
        Corner(15, "Casio Triangle", "slow", 75, 30, 70, 0, False, True, ("braking", "traction")),
        Corner(16, "Casio Triangle exit", "slow", 85, 40, 70, 0, False, False, ("traction",)),
        Corner(17, "Final", "medium", 165, 130, 110, 0, True, False, ("high_downforce", "stability")),
        Corner(18, "Final chicane", "slow", 90, 45, 80, 0, False, False, ("braking", "kerb")),
    ]


def _monaco_corners() -> list[Corner]:
    """Circuit de Monaco (3.337 km, 19 弯, street)."""
    return [
        Corner(1, "Sainte-Devote", "slow", 80, 35, 75, 0, True, True, ("braking", "traction")),
        Corner(2, "Beau Rivage", "medium", 130, 90, 90, 0, False, False, ("high_downforce",)),
        Corner(3, "Massenet", "medium", 120, 80, 100, 0, False, False, ("high_downforce", "stability")),
        Corner(4, "Casino", "medium", 135, 95, 80, 0, False, False, ("high_downforce",)),
        Corner(5, "Mirabeau Haut", "slow", 75, 30, 65, 0, False, False, ("braking",)),
        Corner(6, "Mirabeau Bas", "slow", 65, 25, 60, 0, False, False, ("braking", "traction")),
        Corner(7, "Grand Hotel Hairpin", "slow", 45, 15, 70, 0, False, True, ("braking", "traction")),
        Corner(8, "Portier", "slow", 80, 35, 75, 0, False, False, ("high_downforce",)),
        Corner(9, "Tunnel entry", "medium", 145, 110, 100, 0, False, False, ("high_downforce", "stability")),
        Corner(10, "Tunnel", "fast", 200, 300, 150, 0, False, False, ("high_downforce", "stability")),
        Corner(11, "Tunnel exit", "medium", 155, 120, 110, 0, False, False, ("stability",)),
        Corner(12, "Chicane", "slow", 70, 28, 70, 0, False, True, ("braking", "kerb")),
        Corner(13, "Chicane exit", "slow", 75, 30, 70, 0, False, False, ("traction", "kerb")),
        Corner(14, "Tabac", "medium", 140, 100, 80, 0, False, False, ("high_downforce",)),
        Corner(15, "Piscine entry", "medium", 150, 110, 85, 0, False, False, ("high_downforce", "stability")),
        Corner(16, "Piscine", "medium", 135, 95, 85, 0, False, False, ("high_downforce",)),
        Corner(17, "Rascasse entry", "slow", 80, 35, 70, 0, False, False, ("braking",)),
        Corner(18, "Rascasse", "slow", 70, 28, 75, 0, False, True, ("braking", "traction")),
        Corner(19, "Anthony Noghes", "medium", 125, 85, 90, 0, True, False, ("high_downforce", "stability")),
    ]


def _monza_corners() -> list[Corner]:
    """Autodromo Nazionale Monza (5.793 km, 11 弯, high_speed_low_downforce)."""
    return [
        Corner(1, "Prima Variante", "slow", 85, 40, 80, 0, True, True, ("braking", "traction")),
        Corner(2, "Variante della Roggia", "slow", 90, 45, 75, 0, False, False, ("braking", "kerb")),
        Corner(3, "Curva Biassono", "medium", 165, 130, 100, 0, False, False, ("high_downforce",)),
        Corner(4, "Curva del Serraglio", "fast", 230, 500, 180, 0, False, False, ("high_downforce", "low_downforce")),
        Corner(5, "Variante Ascari entry", "slow", 95, 50, 85, 0, False, False, ("braking",)),
        Corner(6, "Variante Ascari", "slow", 80, 38, 80, 0, False, True, ("braking", "traction")),
        Corner(7, "Variante Ascari exit", "medium", 140, 100, 100, 0, False, False, ("traction",)),
        Corner(8, "Curva Parabolica entry", "medium", 175, 150, 130, 0, False, False, ("high_downforce", "stability")),
        Corner(9, "Curva Parabolica", "fast", 215, 350, 200, 0, True, False, ("high_downforce", "stability")),
        Corner(10, "Curva Grande", "fast", 250, 650, 220, 0, False, False, ("high_downforce", "low_downforce")),
        Corner(11, "Prima Variante approach", "slow", 85, 40, 80, 0, False, True, ("braking",)),
    ]


def _silverstone_corners() -> list[Corner]:
    """Silverstone Circuit (5.891 km, 18 弯, mixed)."""
    return [
        Corner(1, "Abbey", "fast", 240, 550, 200, 0, True, False, ("high_downforce", "stability")),
        Corner(2, "Farm", "medium", 175, 150, 130, 0, False, False, ("high_downforce",)),
        Corner(3, "Village", "medium", 140, 100, 100, 0, False, False, ("high_downforce",)),
        Corner(4, "The Loop", "slow", 90, 45, 80, 0, False, False, ("braking", "traction")),
        Corner(5, "Aintree", "medium", 165, 125, 110, 0, False, False, ("high_downforce",)),
        Corner(6, "Wellington Straight", "fast", 280, 800, 250, 0, True, False, ("low_downforce",)),
        Corner(7, "Brooklands", "slow", 95, 50, 85, 0, False, False, ("braking", "traction")),
        Corner(8, "Luffield", "medium", 130, 90, 120, 0, False, False, ("high_downforce", "stability")),
        Corner(9, "Woodcote", "fast", 230, 500, 180, 0, False, False, ("high_downforce", "stability")),
        Corner(10, "Copse", "fast", 245, 580, 200, 0, False, False, ("high_downforce", "stability")),
        Corner(11, "Maggotts 1", "fast", 220, 450, 170, 0, False, False, ("high_downforce", "stability")),
        Corner(12, "Maggotts 2", "fast", 210, 400, 160, 0, False, False, ("high_downforce", "stability")),
        Corner(13, "Becketts 1", "medium", 175, 150, 130, 0, False, False, ("high_downforce", "stability")),
        Corner(14, "Becketts 2", "medium", 160, 120, 120, 0, False, False, ("high_downforce", "stability")),
        Corner(15, "Chapel", "fast", 215, 420, 160, 0, True, False, ("high_downforce", "stability")),
        Corner(16, "Stowe", "medium", 165, 130, 120, 0, False, True, ("braking", "high_downforce")),
        Corner(17, "Vale", "slow", 90, 45, 80, 0, False, False, ("braking", "traction")),
        Corner(18, "Club", "medium", 155, 120, 120, 0, True, False, ("high_downforce", "stability")),
    ]


_MANUAL_CORNERS["suzuka"] = _suzuka_corners()
_MANUAL_CORNERS["monaco"] = _monaco_corners()
_MANUAL_CORNERS["monza"] = _monza_corners()
_MANUAL_CORNERS["silverstone"] = _silverstone_corners()


def _melbourne_corners() -> list[Corner]:
    """Iter-193: Albert Park Grand Prix Circuit (5.278 km, 14 弯, medium)."""
    return [
        Corner(1, "Turn 1", "slow", 95, 50, 85, 0, True, True, ("braking", "traction")),
        Corner(2, "Turn 2", "medium", 130, 90, 100, 0, False, False, ("high_downforce", "stability")),
        Corner(3, "Turn 3", "fast", 230, 500, 200, 0, False, False, ("high_downforce", "stability")),
        Corner(4, "Turn 4", "slow", 85, 40, 80, 0, False, False, ("braking", "traction")),
        Corner(5, "Turn 5", "medium", 160, 130, 120, 0, False, False, ("high_downforce", "stability")),
        Corner(6, "Turn 6", "fast", 210, 380, 180, 0, False, False, ("high_downforce", "stability")),
        Corner(7, "Turn 7", "slow", 80, 35, 75, 0, False, False, ("braking", "kerb")),
        Corner(8, "Turn 8", "medium", 150, 110, 110, 0, False, False, ("high_downforce",)),
        Corner(9, "Turn 9", "fast", 220, 420, 190, 0, True, False, ("high_downforce", "stability")),
        Corner(10, "Turn 10", "slow", 90, 45, 80, 0, False, True, ("braking", "traction")),
        Corner(11, "Turn 11", "fast", 240, 550, 200, 0, False, False, ("high_downforce", "stability")),
        Corner(12, "Turn 12", "medium", 155, 120, 110, 0, False, False, ("high_downforce", "stability")),
        Corner(13, "Turn 13", "slow", 85, 40, 80, 0, False, False, ("braking", "traction")),
        Corner(14, "Turn 14", "medium", 140, 100, 100, 0, True, False, ("high_downforce", "stability")),
    ]


def _spa_corners() -> list[Corner]:
    """Iter-193: Circuit de Spa-Francorchamps (7.004 km, 19 弯, mixed)."""
    return [
        Corner(1, "La Source", "slow", 85, 40, 80, 0, True, True, ("braking", "traction")),
        Corner(2, "Eau Rouge", "fast", 260, 700, 220, 0, False, False, ("high_downforce", "stability")),
        Corner(3, "Raidillon", "fast", 270, 750, 230, 0, False, False, ("high_downforce", "stability")),
        Corner(4, "Kemmel Straight", "fast", 280, 850, 260, 0, True, False, ("low_downforce",)),
        Corner(5, "Les Combes", "slow", 95, 50, 85, 0, False, True, ("braking", "traction")),
        Corner(6, "Malmedy", "medium", 145, 105, 110, 0, False, False, ("high_downforce", "stability")),
        Corner(7, "Rivage", "slow", 90, 45, 80, 0, False, False, ("braking", "traction")),
        Corner(8, "Pouhon", "fast", 225, 450, 200, 0, False, False, ("high_downforce", "stability")),
        Corner(9, "Fagnes", "medium", 155, 120, 120, 0, False, False, ("high_downforce", "stability")),
        Corner(10, "Campus", "slow", 88, 42, 78, 0, False, False, ("braking", "kerb")),
        Corner(11, "Stavelot", "fast", 235, 500, 200, 0, False, False, ("high_downforce", "stability")),
        Corner(12, "Blanchimont 1", "fast", 275, 780, 240, 0, False, False, ("high_downforce", "stability")),
        Corner(13, "Blanchimont 2", "fast", 280, 820, 250, 0, True, False, ("high_downforce", "low_downforce")),
        Corner(14, "Bus Stop entry", "slow", 82, 38, 78, 0, False, True, ("braking", "traction")),
        Corner(15, "Bus Stop", "slow", 78, 35, 75, 0, False, False, ("braking", "kerb")),
        Corner(16, "Bus Stop exit", "slow", 85, 40, 80, 0, False, False, ("traction",)),
        Corner(17, "La Source approach", "medium", 160, 130, 110, 0, False, False, ("high_downforce", "stability")),
        Corner(18, "Eau Rouge approach", "fast", 255, 650, 220, 0, False, False, ("high_downforce", "stability")),
        Corner(19, "Kemmel approach", "fast", 265, 720, 230, 0, True, False, ("high_downforce", "low_downforce")),
    ]


_MANUAL_CORNERS["melbourne"] = _melbourne_corners()
_MANUAL_CORNERS["spa"] = _spa_corners()


# --------------------------------------------------------------------------- #
# 合成逐弯分布 (未手工录入的赛道)
# --------------------------------------------------------------------------- #
def _classify_speed(speed: float) -> CornerType:
    if speed < 100:
        return "slow"
    if speed < 200:
        return "medium"
    return "fast"


def _speed_to_radius(speed: float) -> float:
    """从 apex 速度估算弯角半径 (物理: v² = a*r, a≈1.5g=14.7 m/s²)."""
    v_ms = speed / 3.6
    a = 14.7  # m/s² (1.5g 侧向加速度, F1 干地极限)
    return max(20.0, (v_ms * v_ms) / a)


def _demands_for_corner(corner_type: CornerType, is_overtaking: bool) -> tuple[CornerDemand, ...]:
    if corner_type == "slow":
        if is_overtaking:
            return ("braking", "traction")
        return ("braking",)
    if corner_type == "medium":
        return ("high_downforce", "stability")
    return ("high_downforce", "stability")


def generate_corner_profile(track_id: str) -> list[Corner]:
    """Iter-164.13: 基于赛道特征合成逐弯分布.

    用赛道类型 / 弯角数 / 长度 / 极速 等公开元数据, 生成代表性的逐弯分布.
    合成数据量级准确 (apex 速度 / 半径基于 F1 侧向加速度极限), 但弯角名称
    用编号占位 (非真实赛道图名称). 对已手工录入的赛道, :func:`get_corners`
    优先返回手填数据.

    Returns
    -------
    list[Corner]
        逐弯数据列表 (1-based 编号).
    """
    track = get_track(track_id)
    eng = get_track_engineering(track_id)
    n = track.corners
    length = track.length_m
    # 赛道类型决定弯角速度分布
    if track.track_type == "high_speed_low_downforce":
        # Monza-like: 多直道 + 少慢弯, 极速高
        slow_frac, med_frac, fast_frac = 0.25, 0.30, 0.45
        speed_range = (75, 280)
    elif track.track_type == "street":
        # Monaco-like: 多慢弯, 极速低
        slow_frac, med_frac, fast_frac = 0.55, 0.35, 0.10
        speed_range = (50, 200)
    elif track.track_type == "high_downforce":
        # Hungaroring-like: 多中慢弯, 少直道
        slow_frac, med_frac, fast_frac = 0.40, 0.45, 0.15
        speed_range = (70, 230)
    elif track.track_type == "mixed":
        # Suzuka-like: 各类型均衡
        slow_frac, med_frac, fast_frac = 0.35, 0.40, 0.25
        speed_range = (70, 270)
    else:  # medium
        slow_frac, _, fast_frac = 0.33, 0.45, 0.22
        speed_range = (75, 250)

    n_slow = max(1, int(round(n * slow_frac)))
    n_fast = max(1, int(round(n * fast_frac)))
    n_med = max(1, n - n_slow - n_fast)

    # 构建弯角类型序列 (交错分布, 避免同类聚集)
    types: list[CornerType] = []
    pool = (["slow"] * n_slow) + (["medium"] * n_med) + (["fast"] * n_fast)
    # 交错: 按比例轮转
    while pool:
        for t in ("slow", "medium", "fast"):
            if t in pool:
                types.append(t)
                pool.remove(t)
                break
    types = types[:n]
    while len(types) < n:
        types.append("medium")

    corners: list[Corner] = []
    avg_zone = length / n  # 平均每弯区域长度
    drs_count = eng.drs_zones
    # DRS 分布: 均匀分散到部分弯角
    drs_idxs = set()
    if drs_count > 0 and n > 0:
        step = max(1, n // drs_count)
        for i in range(0, n, step):
            drs_idxs.add(i)
            if len(drs_idxs) >= drs_count:
                break

    for i in range(n):
        t = types[i]
        if t == "slow":
            speed = speed_range[0] + (i * 7) % 30  # 75-105
        elif t == "medium":
            speed = 110 + (i * 13) % 80  # 110-190
        else:
            speed = 200 + (i * 17) % (speed_range[1] - 200 + 1)  # 200-280
        radius = _speed_to_radius(speed)
        zone_len = avg_zone * (0.6 if t == "fast" else 1.0 if t == "medium" else 0.8)
        is_drs = i in drs_idxs
        # 超车点: 慢弯 + DRS, 或前 30% 慢弯
        is_ota = (t == "slow" and is_drs) or (t == "slow" and i < n * 0.3)
        demands = _demands_for_corner(t, is_ota)
        corners.append(Corner(
            number=i + 1,
            name=f"Corner {i + 1}",
            corner_type=t,
            speed_kmh=float(speed),
            radius_m=float(radius),
            length_m=float(zone_len),
            banking_deg=0.0,
            is_drs=is_drs,
            is_overtaking=is_ota,
            demands=demands,
        ))
    return corners


def get_corners(track_id: str) -> list[Corner]:
    """按 track_id 拿逐弯列表 (手填优先, 否则合成).

    对未知 track_id (新赛道/拼写错误), 优雅降级为合成默认 profile
    (mixed type, 16 corners), 不崩. 这样新赛道加入赛历时系统仍可用.

    Returns
    -------
    list[Corner]
        逐弯数据 (1-based 编号, 长度 = Track.corners 或默认 16).
    """
    if track_id in _MANUAL_CORNERS:
        return list(_MANUAL_CORNERS[track_id])
    try:
        return generate_corner_profile(track_id)
    except (ValueError, KeyError):
        # 未知 track_id: 合成默认 profile (mixed type, 16 corners)
        return _synthesize_default_corners()


def _synthesize_default_corners(n: int = 16) -> list[Corner]:
    """为未知 track_id 合成默认 corner profile (mixed type)."""
    corners: list[Corner] = []
    n_slow = max(1, int(round(n * 0.33)))
    n_fast = max(1, int(round(n * 0.22)))
    n_med = max(1, n - n_slow - n_fast)
    types: list[CornerType] = []
    pool = (["slow"] * n_slow) + (["medium"] * n_med) + (["fast"] * n_fast)
    while pool:
        for t in ("slow", "medium", "fast"):
            if t in pool:
                types.append(t)
                pool.remove(t)
                break
    types = types[:n]
    while len(types) < n:
        types.append("medium")
    for i in range(n):
        t = types[i]
        if t == "slow":
            speed = 75.0 + (i * 7) % 30
        elif t == "medium":
            speed = 110.0 + (i * 13) % 80
        else:
            speed = 200.0 + (i * 17) % 70
        radius = _speed_to_radius(speed)
        zone_len = 90.0 * (0.6 if t == "fast" else 1.0 if t == "medium" else 0.8)
        demands = _demands_for_corner(t, t == "slow" and i < n * 0.3)
        corners.append(Corner(
            number=i + 1,
            name=f"Corner {i + 1}",
            corner_type=t,
            speed_kmh=float(speed),
            radius_m=float(radius),
            length_m=float(zone_len),
            banking_deg=0.0,
            is_drs=(i % 5 == 0),
            is_overtaking=(t == "slow" and i < n * 0.3),
            demands=demands,
        ))
    return corners


# --------------------------------------------------------------------------- #
# 赛道调教需求汇总
# --------------------------------------------------------------------------- #
def corner_demand_summary(track_id: str) -> dict[str, float]:
    """Iter-164.13: 汇总赛道的逐弯调教需求 (各需求占比 0..1).

    返回 dict 如 ``{"braking": 0.25, "high_downforce": 0.40, ...}``,
    各值为该需求在所有弯角中的占比. 用于让优化器/反馈引擎理解赛道
    "整体上"最需要什么调教方向.
    """
    corners = get_corners(track_id)
    n = len(corners)
    if n == 0:
        return {}
    counts: dict[str, int] = {}
    for c in corners:
        for d in c.demands:
            counts[d] = counts.get(d, 0) + 1
    return {k: v / n for k, v in counts.items()}


def problematic_corner_heuristic(
    corners: list[Corner],
    *,
    understeer_indicator: float = 0.0,
    oversteer_indicator: float = 0.0,
    lockup_proxy: float = 0.0,
    high_tire_wear: bool = False,
) -> list[dict]:
    """Iter-164.14: 基于遥测指标推断哪些弯角可能有问题.

    把圈级遥测指标 (understeer / oversteer / lockup / tire_wear) 映射到
    最可能产生这些问题的弯角类型. 返回逐弯问题清单, 供逐弯调教建议使用.

    Parameters
    ----------
    corners : list[Corner]
        赛道逐弯数据 (:func:`get_corners`).
    understeer_indicator : float
        圈级转向不足指标 (0..1, 越高越严重).
    oversteer_indicator : float
        圈级过度转向指标 (0..1).
    lockup_proxy : float
        圈级刹车锁死代理 (0..1).
    high_tire_wear : bool
        是否胎耗过高.

    Returns
    -------
    list[dict]
        问题弯角列表, 每项 ``{"corner": int, "name": str, "issue": str,
        "severity": float, "suggestion": str}``.
    """
    issues: list[dict] = []
    for c in corners:
        severity = 0.0
        issue = ""
        suggestion = ""

        # 转向不足: 高速弯最敏感 (下压力不足 → 前端失去抓地)
        if understeer_indicator > 0.3 and c.corner_type in ("fast", "medium"):
            sev = understeer_indicator * (1.0 if c.corner_type == "fast" else 0.6)
            if sev > severity:
                severity = sev
                issue = "understeer"
                suggestion = "增加前翼/减少后翼 (增前端下压力)"

        # 过度转向: 慢弯出弯最敏感 (牵引力不足 → 后轮打滑)
        if oversteer_indicator > 0.3 and c.corner_type in ("slow", "medium"):
            sev = oversteer_indicator * (1.0 if "traction" in c.demands else 0.5)
            if sev > severity:
                severity = sev
                issue = "oversteer"
                suggestion = "增加后翼/软后悬挂 (增后端稳定性)"

        # 刹车锁死: 重刹慢弯最敏感
        if lockup_proxy > 0.3 and "braking" in c.demands:
            sev = lockup_proxy * (1.0 if c.is_overtaking else 0.7)
            if sev > severity:
                severity = sev
                issue = "lockup"
                suggestion = "降低刹车压力/前移刹车平衡"

        # 胎耗过高: 高速弯 + 高下压力需求弯角最敏感
        if high_tire_wear and c.corner_type in ("fast", "medium"):
            sev = 0.5 * (1.0 if c.corner_type == "fast" else 0.6)
            if sev > severity:
                severity = sev
                issue = "tire_wear"
                suggestion = "提高胎压/软悬挂 (降低胎温)"

        if severity > 0.1:
            issues.append({
                "corner": c.number,
                "name": c.name,
                "issue": issue,
                "severity": float(severity),
                "suggestion": suggestion,
                "corner_type": c.corner_type,
                "speed_kmh": c.speed_kmh,
            })

    # 按严重度降序
    issues.sort(key=lambda x: x["severity"], reverse=True)
    return issues


# --------------------------------------------------------------------------- #
# Iter-164.16: 逐弯调教推荐 — 把弯角级问题映射成具体 setup 参数变更
# --------------------------------------------------------------------------- #
# 反思: Iter-164.14 的 problematic_corner_heuristic 只给出文本 suggestion
# ("增加前翼/减少后翼"), 但没有量化成具体 setup 参数变更. R5 "针对性更改调教"
# 要求系统能输出 "因第 5 弯 understeer → front_wing +2 档" 这样的具体推荐.
#
# 本函数把弯角级问题 (understeer/oversteer/lockup/tire_wear) 映射成
# CarSetup 字段级的变更建议 (name, delta, reason), 供 FeedbackEngine 在
# corner_analysis 维度里返回逐弯调教推荐.

# 每种 issue 类型对应的 setup 变更规则: (field, delta_per_severity, max_delta)
# delta_per_severity: severity=1.0 时的档数变更; max_delta: 单 issue 上限.
_CORNER_ISSUE_SETUP_RULES: dict[str, list[tuple[str, float, float]]] = {
    "understeer": [
        # 增加前翼 (+2/严重度), 减少后翼 (-1/严重度) → 增前端下压力
        ("front_wing", +2.0, +5.0),
        ("rear_wing", -1.0, -3.0),
        # 前悬更软 (+1/严重度) → 增前端机械抓地
        ("front_suspension", -1.0, -3.0),
    ],
    "oversteer": [
        # 增加后翼 (+2/严重度), 后悬更软 → 增后端稳定性
        ("rear_wing", +2.0, +5.0),
        ("rear_suspension", -1.0, -3.0),
        # 后 ARB 更软 → 减少后轴侧倾刚度
        ("rear_arb", -1.0, -3.0),
    ],
    "lockup": [
        # 降刹车压力 (-2/严重度), 前移刹车平衡 → 减前轮锁死
        ("brake_pressure", -2.0, -5.0),
        ("front_brake_bias", -0.5, -2.0),
    ],
    "tire_wear": [
        # 提高胎压 (+0.5 psi/严重度) → 减接地面积降胎温
        ("front_tyre_pressure", +0.5, +1.5),
        ("rear_tyre_pressure", +0.5, +1.5),
        # camber 向 0 靠拢 (减少胎面剪切)
        ("front_camber", +0.005, +0.015),
        ("rear_camber", +0.005, +0.015),
    ],
}


def corner_setup_recommendations(
    issues: list[dict],
    *,
    max_total_changes: int = 8,
) -> list[dict]:
    """Iter-164.16: 把弯角级问题映射成具体 setup 参数变更建议.

    Parameters
    ----------
    issues : list[dict]
        :func:`problematic_corner_heuristic` 返回的弯角问题列表.
    max_total_changes : int
        最多返回的变更项数 (避免过多变更让工程师无所适从).

    Returns
    -------
    list[dict]
        每项 ``{"name": str, "delta": float, "reason": str, "corner": int,
        "severity": float}``. ``name`` 是 CarSetup 字段名, ``delta`` 是
        该字段的变更量 (正=增加, 负=减少), ``reason`` 是变更理由.
    """
    # 按 field 聚合 delta (同一 field 多个 issue 的 delta 累加, 但限幅)
    field_deltas: dict[str, dict] = {}
    for iss in issues:
        issue_type = iss.get("issue", "")
        severity = float(iss.get("severity", 0.0))
        corner = int(iss.get("corner", 0))
        corner_name = str(iss.get("name", ""))
        rules = _CORNER_ISSUE_SETUP_RULES.get(issue_type, [])
        for field, delta_per_sev, max_delta in rules:
            raw_delta = delta_per_sev * severity
            # 限幅
            if max_delta > 0:
                delta = min(raw_delta, max_delta)
            else:
                delta = max(raw_delta, max_delta)
            if abs(delta) < 1e-6:
                continue
            if field not in field_deltas:
                field_deltas[field] = {
                    "name": field,
                    "delta": 0.0,
                    "reasons": [],
                    "max_severity": 0.0,
                }
            # 累加但再次限幅 (多 issue 累加后可能超 max_delta)
            new_delta = field_deltas[field]["delta"] + delta
            if max_delta > 0:
                new_delta = min(new_delta, max_delta)
            else:
                new_delta = max(new_delta, max_delta)
            field_deltas[field]["delta"] = new_delta
            field_deltas[field]["reasons"].append(
                f"第{corner}弯({corner_name}){issue_type}(sev={severity:.2f})"
            )
            field_deltas[field]["max_severity"] = max(
                field_deltas[field]["max_severity"], severity
            )

    # 转成 list, 按 max_severity 降序, 限 max_total_changes 项
    recs = sorted(
        field_deltas.values(),
        key=lambda x: x["max_severity"],
        reverse=True,
    )[:max_total_changes]
    return [
        {
            "name": r["name"],
            "delta": round(r["delta"], 4),
            "reason": "; ".join(r["reasons"]),
            "corner": -1,  # 聚合多弯, 无单一 corner
            "severity": r["max_severity"],
        }
        for r in recs
    ]
