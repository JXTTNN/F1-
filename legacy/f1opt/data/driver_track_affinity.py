"""F1 2026 车手-赛道亲和度模型 (Iter-38).

真实 F1 车手在特定赛道表现出明显差异:
- **Verstappen @ Suzuka**: 3 连胜 (2022-2024), 高下压力赛道适合其风格.
- **Hamilton @ Silverstone**: 8+ 胜, 主场 + 高速弯.
- **Leclerc @ Monza**: Ferrari 主场, 低下压力直道.
- **Alonso @ Monaco**: 街道赛大师, 2 胜 + 多次 pole.
- **Norris @ Zandvoort/Miami**: 高速弯 + 制动稳定性.

EA Sports F1 2026 游戏官方 "Track Specialization" 评级:
- 每位车手有 0-3 个 "擅长赛道".
- 擅长赛道: 圈速 +0.15-0.30 s (相对基准).
- 不擅长赛道: 圈速 -0.10-0.20 s.

数据来源 (Iter-38):
- 2018-2025 公开 F1 比赛结果 (driver-track 历史表现).
- EA Sports F1 2026 游戏官方车手-赛道评级.
- The Race / Auto Motor und Sport 车手风格分析.

公开 API:
    - :func:`driver_track_affinity` — 返回亲和度系数 (-0.30 .. +0.30 s).
    - :func:`track_specialists` — 返回该赛道的专家车手列表.
    - :func:`driver_specialist_tracks` — 返回车手擅长赛道列表.
"""

from __future__ import annotations

from f1opt.data.drivers_2026 import all_drivers_2026, get_driver_2026

# --------------------------------------------------------------------------- #
# 亲和度系数 (s/lap, 正 = 圈速更快, 负 = 圈速更慢)
# --------------------------------------------------------------------------- #
# 格式: { driver_id: { track_id: affinity_s } }
# 仅记录显著亲和 (>0.10 或 < -0.10), 其他默认 0
_DRIVER_TRACK_AFFINITY: dict[str, dict[str, float]] = {
    # Max Verstappen — Suzuka (3 连胜), Spa (4 连胜), Austin, Interlagos
    "ver": {
        "suzuka": 0.25,      # 高下压力 + 流畅节奏
        "spa": 0.22,         # 长直道 + 高速弯组合
        "austin": 0.18,      # COTA 多变布局
        "interlagos": 0.20,  # 起伏 + 反向下滑
        "silverstone": 0.15,  # 高速弯
    },
    # Lewis Hamilton — Silverstone (8 胜), Hungaroring, Canada, Shanghai
    "ham": {
        "silverstone": 0.28,  # 主场 + 高速弯大师
        "hungaroring": 0.20,  # 类卡丁车赛道
        "montreal": 0.18,     # 制动 + 牵引
        "shanghai": 0.15,     # 长直道 + 复杂弯
        "spa": 0.12,
    },
    # Charles Leclerc — Monza (Ferrari 主场), Spa, Baku (街道路型)
    "lec": {
        "monza": 0.25,        # Ferrari 主场 + 低阻
        "spa": 0.18,
        "baku": 0.20,         # 街道 + 长直道
        "monaco": 0.15,       # 排位赛专家
        "zandvoort": 0.12,
    },
    # Fernando Alonso — Monaco (2 胜), Singapore (街赛大师), Bahrain
    "alo": {
        "monaco": 0.25,       # 街道赛大师
        "singapore": 0.20,    # 高下压力街道
        "bahrain": 0.15,
        "suzuka": 0.12,       # 高速弯专家
        "interlagos": 0.10,
    },
    # Lando Norris — Zandvoort, Miami, Silverstone (高速弯)
    "nor": {
        "zandvoort": 0.22,    # 高速倾斜弯
        "miami": 0.18,        # 首胜所在地
        "silverstone": 0.15,
        "singapore": -0.12,   # 街道赛相对弱
    },
    # Oscar Piastri — Bahrain, Hungary, Baku
    "pia": {
        "bahrain": 0.18,
        "hungaroring": 0.20,  # 首胜
        "baku": 0.15,
    },
    # George Russell — Brazil, Canada, Silverstone
    "rus": {
        "interlagos": 0.18,   # 首胜
        "montreal": 0.12,
        "silverstone": 0.10,
    },
    # Carlos Sainz — Singapore, Mexico, Monza
    "sai": {
        "singapore": 0.22,    # 2023 胜
        "monza": 0.15,
        "las_vegas": 0.12,    # 街道 + 低下压力
        "bahrain": 0.10,
    },
    # Alex Albon — Singapore, Monaco (威廉姆斯时期)
    "alb": {
        "singapore": 0.12,
        "monaco": 0.10,
        "silverstone": 0.10,
    },
    # Pierre Gasly — Baku (2020 首胜), Zandvoort
    "gas": {
        "baku": 0.20,         # 首胜
        "zandvoort": 0.12,
    },
    # Nico Hulkenberg — 巴西雨战大师
    "hul": {
        "interlagos": 0.18,   # 2019 pole + 多次雨战好成绩
        "spa": 0.10,
    },
    # Yuki Tsunoda — Suzuka (主场)
    "tsu": {
        "suzuka": 0.18,       # 主场
    },
    # Esteban Ocon — Monaco (排位好)
    "oco": {
        "monaco": 0.12,
        "hungaroring": 0.10,  # 2021 首胜
    },
    # Lance Stroll — Singapore, Canada (主场)
    "str": {
        "singapore": 0.12,
        "montreal": 0.15,     # 主场
    },
    # Kimi Antonelli — Monza (主场首秀)
    "ant": {
        "monza": 0.15,        # 主场
    },
    # Oliver Bearman — Saudi (首秀), Mexico
    "bears": {
        "jeddah": 0.12,       # 首秀
        "monaco": 0.10,
    },
    # Liam Lawson — Singapore
    "had": {
        "singapore": 0.12,
    },
    # Isack Hadjar — Monaco (F2 强项)
    "bea": {
        "monaco": 0.10,
    },
    # Jack Doohan — Melbourne (主场)
    "doo": {
        "melbourne": 0.15,    # 主场
    },
    # Gabriel Bortoleto — Interlagos (主场)
    "bor": {
        "interlagos": 0.18,   # 主场
    },
}

# 反向: 车手明显不擅长的赛道 (圈内慢)
_DRIVER_TRACK_WEAKNESS: dict[str, dict[str, float]] = {
    "had": {"monaco": -0.15},          # Lawson 不擅街道
    "bea": {"suzuka": -0.12},          # Hadjar Suzuka 经验少
    "doo": {"monaco": -0.15},          # Doohan 街道赛经验少
    "bor": {"monaco": -0.18},          # Bortoleto 新秀 + Monaco 难
    "bears": {"suzuka": -0.12},        # Bearman Suzuka 经验少
}


# --------------------------------------------------------------------------- #
# 公开 API
# --------------------------------------------------------------------------- #
def driver_track_affinity(driver_id: str, track_id: str) -> float:
    """返回车手在该赛道的亲和度 (s/lap, 正=快, 负=慢).

    范围: -0.30 .. +0.30. 默认 0.0 (无显著亲和).

    用法::

        off = driver_track_affinity("ver", "suzuka")  # 0.25
        # 在 lap_simulator 中: lap_time -= off
    """
    # 检查车手存在
    try:
        get_driver_2026(driver_id)
    except ValueError as e:
        raise ValueError(f"Unknown driver_id: {driver_id!r}") from e

    affinity = _DRIVER_TRACK_AFFINITY.get(driver_id, {}).get(track_id, 0.0)
    weakness = _DRIVER_TRACK_WEAKNESS.get(driver_id, {}).get(track_id, 0.0)
    return float(affinity + weakness)


def track_specialists(track_id: str) -> list[tuple[str, float]]:
    """返回该赛道的专家车手 (driver_id, affinity_s) 列表, 按亲和度降序.

    仅返回亲和度 > 0.10 的车手.
    """
    specialists: list[tuple[str, float]] = []
    for d in all_drivers_2026():
        aff = driver_track_affinity(d.driver_id, track_id)
        if aff > 0.10:
            specialists.append((d.driver_id, aff))
    specialists.sort(key=lambda x: -x[1])
    return specialists


def driver_specialist_tracks(driver_id: str) -> list[tuple[str, float]]:
    """返回车手擅长的赛道 (track_id, affinity_s) 列表, 按亲和度降序.

    仅返回亲和度 > 0.10 的赛道.
    """
    try:
        get_driver_2026(driver_id)
    except ValueError as e:
        raise ValueError(f"Unknown driver_id: {driver_id!r}") from e

    affinities = _DRIVER_TRACK_AFFINITY.get(driver_id, {})
    out = [(t, a) for t, a in affinities.items() if a > 0.10]
    out.sort(key=lambda x: -x[1])
    return out


def top_specialist_for_track(track_id: str) -> str | None:
    """返回该赛道的头号专家 driver_id (无则 None)."""
    specs = track_specialists(track_id)
    return specs[0][0] if specs else None


def has_affinity(driver_id: str, track_id: str) -> bool:
    """车手对该赛道是否有显著亲和 (|affinity| > 0.10)."""
    return abs(driver_track_affinity(driver_id, track_id)) > 0.10
