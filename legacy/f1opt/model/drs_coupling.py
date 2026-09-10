"""F1 2026 DRS 圈速精确耦合 (Iter-57).

将 DRS (Drag Reduction System) 精确耦合到圈速模型, 严格契合 EA F1 2026
物理引擎与 FIA 2026 体育规则.

**EA F1 2026 DRS 物理 (对标游戏引擎)**:
- **排位赛**: DRS 全程可用 (每圈每个 DRS 区, 无需前车).
- **正赛**: DRS 仅在检测点前 1s 内有前车时可用 (FIA 规则).
- **第 1 圈**: DRS 禁用 (避免起步事故).
- **SC 重启后 2 圈**: DRS 禁用 (FIA 2026 §23.5).
- **湿地 (wetness > 0.30)**: DRS 禁用 (安全考虑).

**圈速量化 (EA F1 2026 物理量级)**:
- 单 DRS 区 (800m 直道): ~0.25-0.35s 收益
- 收益 ∝ 直道长度 × 速度差 (DRS 启用 vs 关闭 ~12-15 km/h)
- 24 赛道 DRS 区数量: 1 (Suzuka/Losail) ~ 4 (Melbourne/Singapore)

**24 赛道 DRS 区数据 (EA F1 2026 日历)**:
完整覆盖 2026 赛历 24 站, 每站 DRS 区数量 + 总 DRS 直道长度.

公开 API:
    - :data:`DRS_ZONES_2026` — 24 赛道 DRS 区数据.
    - :func:`drs_lap_gain_s` — 单圈 DRS 圈速收益.
    - :func:`drs_zone_gain_s` — 单 DRS 区收益.
    - :func:`drs_available` — DRS 是否可用 (规则判断).
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# EA F1 2026 DRS 物理常量
# --------------------------------------------------------------------------- #
# DRS 启用 vs 关闭速度差 (km/h, EA F1 2026 量级)
_DRS_SPEED_DELTA_KMH = 13.0

# DRS 圈速收益系数: s per meter of DRS zone
# 校准: 800m 直道 → ~0.30s → 0.000375 s/m
_DRS_GAIN_S_PER_M = 0.000375

# DRS 检测间隙阈值 (s, FIA 规则)
_DRS_GAP_THRESHOLD_S = 1.0

# SC 重启后 DRS 禁用圈数 (FIA 2026)
_DRS_DISABLED_AFTER_SC_LAPS = 2

# 湿地阈值 (wetness > 此值禁用 DRS)
_DRS_WET_THRESHOLD = 0.30

# 基准速度 (用于长度→时间转换, m/s, ~330 km/h 直道尾速)
_REF_SPEED_MS = 92.0  # ~331 km/h


# --------------------------------------------------------------------------- #
# 24 赛道 DRS 区数据 (EA F1 2026 日历)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DRSZoneData:
    """单赛道 DRS 区数据.

    - ``n_zones``: DRS 区数量.
    - ``total_zone_length_m``: 所有 DRS 区直道总长 (m).
    - ``avg_zone_length_m``: 平均 DRS 区长度 (m).
    """

    n_zones: int
    total_zone_length_m: float
    avg_zone_length_m: float

    @property
    def max_lap_gain_s(self) -> float:
        """全 DRS 区激活的最大圈速收益 (s)."""
        return self.total_zone_length_m * _DRS_GAIN_S_PER_M


# 24 赛道 DRS 数据 (基于 2024-2025 F1 DRS 区配置 + 2026 赛历调整)
# (n_zones, total_zone_length_m)
_DRS_ZONE_RAW: dict[str, tuple[int, float]] = {
    "melbourne": (4, 2400.0),    # Albert Park: 4 DRS zones (2024)
    "shanghai": (2, 1900.0),     # Shanghai: 2 zones
    "suzuka": (1, 800.0),        # Suzuka: 1 zone (main straight)
    "bahrain": (3, 2100.0),      # Sakhir: 3 zones
    "jeddah": (3, 2400.0),       # Jeddah: 3 zones (fast street)
    "miami": (3, 1800.0),        # Miami: 3 zones
    "montreal": (2, 1400.0),     # Gilles Villeneuve: 2 zones
    "monaco": (1, 350.0),        # Monaco: 1 zone (short main straight)
    "barcelona": (2, 1600.0),    # Catalunya: 2 zones
    "spielberg": (3, 1500.0),    # Red Bull Ring: 3 zones (2024)
    "silverstone": (2, 1700.0),  # Silverstone: 2 zones
    "spa": (2, 1400.0),          # Spa: 2 zones (Kemmel + main)
    "hungaroring": (2, 1200.0),  # Hungaroring: 2 zones
    "zandvoort": (2, 1100.0),    # Zandvoort: 2 zones
    "monza": (2, 2000.0),        # Monza: 2 zones (long main + Curva Grande)
    "madrid": (3, 1700.0),       # Madrid 2026 debut: 3 zones
    "baku": (2, 2400.0),         # Baku: 2 zones (very long main straight)
    "singapore": (4, 1800.0),    # Marina Bay: 4 zones (2024)
    "austin": (2, 1800.0),       # COTA: 2 zones
    "mexico_city": (3, 1700.0),  # Hermanos Rodriguez: 3 zones
    "interlagos": (2, 1400.0),   # Interlagos: 2 zones
    "las_vegas": (2, 2000.0),    # Las Vegas: 2 zones (long strip)
    "losail": (1, 1000.0),       # Lusail: 1 zone (main straight)
    "yas_marina": (2, 1600.0),   # Yas Marina: 2 zones
}

# 构建 DRS_ZONES_2026 (frozen dataclass 实例)
DRS_ZONES_2026: dict[str, DRSZoneData] = {
    tid: DRSZoneData(
        n_zones=n,
        total_zone_length_m=total,
        avg_zone_length_m=total / n,
    )
    for tid, (n, total) in _DRS_ZONE_RAW.items()
}

_DEFAULT_DRS = DRSZoneData(n_zones=2, total_zone_length_m=1400.0, avg_zone_length_m=700.0)


def get_drs_zone_data(track_id: str) -> DRSZoneData:
    """查询赛道 DRS 区数据, 未知返回默认."""
    return DRS_ZONES_2026.get(track_id, _DEFAULT_DRS)


# --------------------------------------------------------------------------- #
# DRS 可用性判断 (FIA 2026 规则)
# --------------------------------------------------------------------------- #
def drs_available(
    lap: int,
    gap_to_ahead_s: float | None,
    session_type: str = "race",
    wetness: float = 0.0,
    sc_just_ended_lap: int = 0,
) -> tuple[bool, str]:
    """判断 DRS 是否可用 (FIA 2026 规则).

    Args:
        lap: 当前圈数 (1-based).
        gap_to_head_s: 与前车差距 (s), None = 无前车 (净空).
        session_type: "qualifying" 或 "race".
        wetness: 赛道湿润度 (0..1).
        sc_just_ended_lap: 安全车刚结束的圈数 (0 = 无 SC).

    Returns:
        (available, reason) 元组.
    """
    # 湿地禁用
    if wetness > _DRS_WET_THRESHOLD:
        return False, "DRS disabled (wet conditions)"

    # 排位赛: DRS 全程可用 (无需前车)
    if session_type == "qualifying":
        return True, "DRS available (qualifying, free use)"

    # 正赛规则:
    # 第 1 圈禁用
    if lap == 1:
        return False, "DRS disabled lap 1 (F1 2026 rule)"

    # SC 后 2 圈禁用
    if sc_just_ended_lap > 0 and lap - sc_just_ended_lap <= _DRS_DISABLED_AFTER_SC_LAPS:
        return False, (
            f"DRS disabled (SC ended lap {sc_just_ended_lap}, "
            f"need {_DRS_DISABLED_AFTER_SC_LAPS} laps)"
        )

    # 无前车 → DRS 不可用
    if gap_to_ahead_s is None:
        return False, "No car ahead (DRS needs target)"

    # 前车在 1s 内 → DRS 可用
    if gap_to_ahead_s <= _DRS_GAP_THRESHOLD_S:
        return True, f"DRS active (gap {gap_to_ahead_s:.2f}s)"

    return False, f"Gap too large ({gap_to_ahead_s:.2f}s > {_DRS_GAP_THRESHOLD_S}s)"


# --------------------------------------------------------------------------- #
# DRS 圈速收益
# --------------------------------------------------------------------------- #
def drs_zone_gain_s(zone_length_m: float) -> float:
    """单 DRS 区圈速收益 (s).

    基于 EA F1 2026 物理: 收益 = 直道长度 × 系数.
    短直道 (< 200m) 收益微弱.

    Args:
        zone_length_m: DRS 区直道长度 (m).

    Returns:
        圈速收益 (s, 正=快). 0.0 表示无收益.
    """
    if zone_length_m < 200.0:
        return 0.0
    return zone_length_m * _DRS_GAIN_S_PER_M


def drs_lap_gain_s(
    track_id: str,
    lap: int = 2,
    gap_to_ahead_s: float | None = None,
    session_type: str = "race",
    wetness: float = 0.0,
    sc_just_ended_lap: int = 0,
    n_active_zones: int | None = None,
) -> float:
    """单圈 DRS 圈速收益 (s, 正=快).

    综合 FIA 2026 规则判断 + EA F1 2026 物理量化.

    Args:
        track_id: 赛道 ID.
        lap: 当前圈 (1-based).
        gap_to_ahead_s: 与前车差距 (s), None = 净空.
        session_type: "qualifying" 或 "race".
        wetness: 湿润度.
        sc_just_ended_lap: SC 刚结束圈 (0 = 无).
        n_active_zones: 强制指定激活区数 (None = 全部可用区).

    Returns:
        圈速收益 (s, 正=快). 0.0 = DRS 不可用.
    """
    available, _reason = drs_available(
        lap, gap_to_ahead_s, session_type, wetness, sc_just_ended_lap
    )
    if not available:
        return 0.0

    zone_data = get_drs_zone_data(track_id)
    if n_active_zones is None:
        n_active = zone_data.n_zones
    else:
        n_active = min(n_active_zones, zone_data.n_zones)

    if n_active <= 0:
        return 0.0

    # 收益 = 平均区长度 × 激活区数 × 系数
    return n_active * zone_data.avg_zone_length_m * _DRS_GAIN_S_PER_M


# --------------------------------------------------------------------------- #
# 便捷函数
# --------------------------------------------------------------------------- #
def max_drs_gain_s(track_id: str) -> float:
    """便捷: 赛道最大 DRS 收益 (全激活)."""
    return get_drs_zone_data(track_id).max_lap_gain_s


def n_drs_zones(track_id: str) -> int:
    """便捷: 赛道 DRS 区数量."""
    return get_drs_zone_data(track_id).n_zones


def drs_speed_delta_kmh() -> float:
    """便捷: DRS 启用 vs 关闭速度差 (km/h)."""
    return _DRS_SPEED_DELTA_KMH


def all_drs_tracks() -> list[str]:
    """便捷: 所有有 DRS 数据的赛道."""
    return list(DRS_ZONES_2026.keys())
