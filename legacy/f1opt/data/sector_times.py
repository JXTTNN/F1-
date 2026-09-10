"""F1 2026 赛道扇区时间数据库 (Iter-48).

每条 F1 2026 赛道的 3 个扇区基准时间. 扇区数据用于:

- **圈速分解**: 总圈速 = S1 + S2 + S3, 分析车手强弱段.
- **DRS 评估**: 哪些扇区含长直道 (DRS 价值高).
- **轮胎负载**: 高速弯扇区磨损大, 直道扇区磨损小.
- **策略分段**: 进站损失按扇区分布 (含直道的扇区 pit loss 高).

数据来源: F1 官方计时 + 车队 simulator 量级估计.
所有时间为 2026 赛车 (750kW PU + 主动空动) 在各赛道的预估基准圈速,
基于 2024-2025 实测数据外推 + 2026 规则变化调整.

注意: 时间为合理工程估计, 维护一致性比逐位精度更重要.

公开 API:
    - :class:`TrackSectorData` — 单赛道扇区数据.
    - :func:`sector_times_for` — 查询赛道扇区时间.
    - :func:`total_lap_time_s` — 总圈速.
    - :func:`sector_with_longest_straight` — 最长直道扇区.
    - :func:`high_wear_sector` — 最高磨损扇区.
"""

from __future__ import annotations

from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# 数据类
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrackSectorData:
    """单赛道扇区数据.

    - ``s1_s/s2_s/s3_s``: 各扇区基准时间 (s).
    - ``s1_drs/s2_drs/s3_drs``: 该扇区是否含 DRS 区.
    - ``s1_wear/s2_wear/s3_wear``: 扇区轮胎磨损系数 (1.0 = 平均).
    - ``s1_overtake/s2_overtake/s3_overtake``: 扇区超车可能性 (0..1).
    """

    track_id: str
    s1_s: float
    s2_s: float
    s3_s: float
    s1_drs: bool = False
    s2_drs: bool = False
    s3_drs: bool = False
    s1_wear: float = 1.0
    s2_wear: float = 1.0
    s3_wear: float = 1.0
    s1_overtake: float = 0.3
    s2_overtake: float = 0.3
    s3_overtake: float = 0.3

    @property
    def total_lap_time_s(self) -> float:
        return self.s1_s + self.s2_s + self.s3_s

    def sector_time(self, idx: int) -> float:
        """返回第 idx 扇区时间 (1-indexed)."""
        if idx == 1:
            return self.s1_s
        if idx == 2:
            return self.s2_s
        if idx == 3:
            return self.s3_s
        raise ValueError(f"Sector index must be 1, 2, or 3, got {idx}")

    def sector_drs(self, idx: int) -> bool:
        if idx == 1:
            return self.s1_drs
        if idx == 2:
            return self.s2_drs
        if idx == 3:
            return self.s3_drs
        raise ValueError(f"Sector index must be 1, 2, or 3, got {idx}")

    def sector_wear(self, idx: int) -> float:
        if idx == 1:
            return self.s1_wear
        if idx == 2:
            return self.s2_wear
        if idx == 3:
            return self.s3_wear
        raise ValueError(f"Sector index must be 1, 2, or 3, got {idx}")

    def sector_overtake(self, idx: int) -> float:
        if idx == 1:
            return self.s1_overtake
        if idx == 2:
            return self.s2_overtake
        if idx == 3:
            return self.s3_overtake
        raise ValueError(f"Sector index must be 1, 2, or 3, got {idx}")

    def sector_with_longest_straight(self) -> int:
        """含最长直道的扇区 (DRS 价值最高)."""
        # 简化: DRS 扇区中超车可能性最高的
        drs_sectors = [i for i in (1, 2, 3) if self.sector_drs(i)]
        if not drs_sectors:
            return 1
        return max(drs_sectors, key=lambda i: self.sector_overtake(i))

    def high_wear_sector(self) -> int:
        """最高轮胎磨损扇区."""
        return max((1, 2, 3), key=lambda i: self.sector_wear(i))

    def best_overtake_sector(self) -> int:
        """最佳超车扇区."""
        return max((1, 2, 3), key=lambda i: self.sector_overtake(i))


# --------------------------------------------------------------------------- #
# 24 条赛道扇区数据 (2026 赛历)
# --------------------------------------------------------------------------- #
# 时间为 2026 赛车预估基准 (基于 2024 数据 + 2026 规则调整)
_SECTOR_DATA: dict[str, TrackSectorData] = {
    "melbourne": TrackSectorData(
        "melbourne", 30.5, 35.2, 28.8,
        s2_drs=True, s3_drs=True,
        s2_wear=1.15, s3_wear=0.85,
        s2_overtake=0.55, s3_overtake=0.60,
    ),
    "shanghai": TrackSectorData(
        "shanghai", 32.0, 38.5, 30.0,
        s1_drs=True, s2_drs=True,
        s1_wear=1.10, s2_wear=1.20,
        s1_overtake=0.55, s2_overtake=0.50,
    ),
    "suzuka": TrackSectorData(
        "suzuka", 31.5, 37.0, 30.5,
        s1_drs=True,
        s1_wear=1.30, s2_wear=1.25, s3_wear=0.95,
        s1_overtake=0.40, s3_overtake=0.45,
    ),
    "bahrain": TrackSectorData(
        "bahrain", 30.0, 38.0, 30.5,
        s1_drs=True, s2_drs=True, s3_drs=True,
        s2_wear=1.15, s3_wear=0.90,
        s1_overtake=0.55, s2_overtake=0.55, s3_overtake=0.60,
    ),
    "jeddah": TrackSectorData(
        "jeddah", 32.5, 40.0, 32.0,
        s1_drs=True, s2_drs=True, s3_drs=True,
        s2_wear=0.95, s3_wear=0.90,
        s1_overtake=0.50, s2_overtake=0.55, s3_overtake=0.50,
    ),
    "miami": TrackSectorData(
        "miami", 31.0, 36.5, 30.0,
        s1_drs=True, s2_drs=True, s3_drs=True,
        s2_wear=1.10,
        s1_overtake=0.55, s3_overtake=0.60,
    ),
    "montreal": TrackSectorData(
        "montreal", 29.5, 34.0, 28.5,
        s1_drs=True, s2_drs=True,
        s2_wear=1.20,
        s1_overtake=0.60, s2_overtake=0.55,
    ),
    "monaco": TrackSectorData(
        "monaco", 25.0, 26.5, 24.0,
        s2_drs=True,  # 仅隧道后
        s1_wear=1.10, s2_wear=1.20,
        s1_overtake=0.05, s2_overtake=0.10, s3_overtake=0.15,
    ),
    "barcelona": TrackSectorData(
        "barcelona", 30.5, 36.0, 29.0,
        s1_drs=True, s2_drs=True,
        s1_wear=1.25, s2_wear=1.20,
        s1_overtake=0.35, s2_overtake=0.40,
    ),
    "spielberg": TrackSectorData(
        "spielberg", 28.0, 32.5, 27.0,
        s1_drs=True, s2_drs=True, s3_drs=True,
        s2_wear=1.05,
        s1_overtake=0.60, s2_overtake=0.55, s3_overtake=0.55,
    ),
    "silverstone": TrackSectorData(
        "silverstone", 31.0, 36.5, 30.0,
        s1_drs=True, s2_drs=True,
        s1_wear=1.35, s2_wear=1.30, s3_wear=1.10,
        s1_overtake=0.45, s2_overtake=0.50,
    ),
    "spa": TrackSectorData(
        "spa", 35.0, 42.0, 33.5,
        s1_drs=True, s2_drs=True,
        s1_wear=1.25, s2_wear=1.30, s3_wear=1.10,
        s1_overtake=0.60, s2_overtake=0.55,
    ),
    "hungaroring": TrackSectorData(
        "hungaroring", 28.5, 33.0, 27.5,
        s1_drs=True,
        s1_wear=1.15, s2_wear=1.20,
        s1_overtake=0.25, s2_overtake=0.20, s3_overtake=0.25,
    ),
    "zandvoort": TrackSectorData(
        "zandvoort", 28.0, 32.5, 27.0,
        s1_drs=True, s2_drs=True,
        s1_wear=1.20, s2_wear=1.15,
        s1_overtake=0.30, s2_overtake=0.25,
    ),
    "monza": TrackSectorData(
        "monza", 27.5, 33.0, 26.5,
        s1_drs=True, s2_drs=True,
        s1_wear=1.15, s2_wear=0.85, s3_wear=0.90,
        s1_overtake=0.70, s2_overtake=0.65, s3_overtake=0.65,
    ),
    "madrid": TrackSectorData(
        "madrid", 30.0, 35.5, 29.0,
        s1_drs=True, s2_drs=True, s3_drs=True,
        s2_wear=1.10,
        s1_overtake=0.55, s2_overtake=0.50, s3_overtake=0.55,
    ),
    "baku": TrackSectorData(
        "baku", 32.0, 42.0, 30.0,
        s1_drs=True, s2_drs=True,
        s2_wear=0.80,  # 长直道低磨损
        s1_overtake=0.65, s2_overtake=0.70, s3_overtake=0.55,
    ),
    "singapore": TrackSectorData(
        "singapore", 30.5, 35.5, 29.0,
        s1_drs=True, s2_drs=True, s3_drs=True,
        s1_wear=1.10, s2_wear=1.15,
        s1_overtake=0.25, s2_overtake=0.20, s3_overtake=0.30,
    ),
    "austin": TrackSectorData(
        "austin", 30.0, 35.5, 29.0,
        s1_drs=True, s2_drs=True,
        s1_wear=1.20, s2_wear=1.15,
        s1_overtake=0.55, s2_overtake=0.50,
    ),
    "mexico_city": TrackSectorData(
        "mexico_city", 28.5, 33.0, 27.5,
        s1_drs=True, s2_drs=True, s3_drs=True,
        s2_wear=0.95,
        s1_overtake=0.60, s2_overtake=0.55, s3_overtake=0.55,
    ),
    "interlagos": TrackSectorData(
        "interlagos", 27.5, 32.0, 26.5,
        s1_drs=True, s2_drs=True,
        s1_wear=1.15, s3_wear=1.10,
        s1_overtake=0.55, s2_overtake=0.60, s3_overtake=0.55,
    ),
    "las_vegas": TrackSectorData(
        "las_vegas", 30.0, 37.0, 31.0,
        s1_drs=True, s2_drs=True, s3_drs=True,
        s2_wear=0.85,
        s1_overtake=0.60, s2_overtake=0.65, s3_overtake=0.55,
    ),
    "losail": TrackSectorData(
        "losail", 31.0, 36.0, 30.0,
        s1_drs=True, s2_drs=True,
        s1_wear=1.25, s2_wear=1.20,
        s1_overtake=0.45, s2_overtake=0.50,
    ),
    "yas_marina": TrackSectorData(
        "yas_marina", 30.5, 36.5, 29.5,
        s1_drs=True, s2_drs=True, s3_drs=True,
        s2_wear=1.05,
        s1_overtake=0.55, s2_overtake=0.55, s3_overtake=0.60,
    ),
}


# --------------------------------------------------------------------------- #
# 公开 API
# --------------------------------------------------------------------------- #
def sector_times_for(track_id: str) -> TrackSectorData:
    """查询赛道扇区数据.

    自动解析别名 (sakhir↔bahrain, sao_paulo↔interlagos, lusail↔losail),
    与 ``tracks.TRACKS_BY_ID`` 的城市命名兼容 (Iter-67).

    Raises:
        ValueError: 未知赛道.
    """
    from f1opt.data.ea_f1_2026_benchmark import resolve_track_id

    cid = resolve_track_id(track_id)
    if cid not in _SECTOR_DATA:
        raise ValueError(f"Unknown track_id: {track_id!r}")
    return _SECTOR_DATA[cid]


def total_lap_time_s(track_id: str) -> float:
    """便捷: 查询赛道总圈速 (s)."""
    return sector_times_for(track_id).total_lap_time_s


def sector_with_longest_straight(track_id: str) -> int:
    """便捷: 含最长直道的扇区 (DRS 价值最高)."""
    return sector_times_for(track_id).sector_with_longest_straight()


def high_wear_sector(track_id: str) -> int:
    """便捷: 最高磨损扇区."""
    return sector_times_for(track_id).high_wear_sector()


def all_track_ids() -> list[str]:
    """所有支持的赛道 ID."""
    return list(_SECTOR_DATA.keys())


def tracks_sorted_by_lap_time() -> list[tuple[str, float]]:
    """按圈速排序的赛道 (快→慢)."""
    return sorted(
        ((tid, data.total_lap_time_s) for tid, data in _SECTOR_DATA.items()),
        key=lambda x: x[1],
    )
