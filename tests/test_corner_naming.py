"""弯道编号/命名 —— 以 **F1 官方文档**为准的回归测试。

本文件的历史与教训
------------------
1. task-64 曾把"用 Turn N 占位"当成错误，规定若干赛道**不许**出现 ``Turn N``。
   但查证 F1 官方文档后确认：官方只对部分赛道命名弯道，其余赛道的弯道
   **在官方口径里就没有专名**，官方直接写 ``Turn N``。所以 ``Turn N``
   不是占位符，而是这些弯的正式写法；旧的规则会逼着人编名字。
2. 为了满足"必须有专名"，代码里出现了 ``Bocht 8`` / ``Bocht 9`` /
   ``Curve 10`` / ``Turn 3 Right`` / ``Circuit Zandvoort Bocht`` 这类
   **自造描述**，以及把直道名 ``Reta Oposta``、路段名 ``Subida dos Boxes``
   / ``Arquibancada`` 当作弯名 —— 这正是用户反馈「弯道号就是乱的」。

现在的口径（三层）
------------------
- ``OFFICIAL_NAMES``（``_track_official``）：F1 官方文档**明确给出**的
  编号↔名称 → 逐条锁定，改动即失败。
- ``TRADITIONAL_TURN_LOCKS``：赛道/业界长期通用但**无 F1 官方出处**的名称 →
  仅作防漂移锁定，**不得**当官方事实引用。
- 其余弯使用 ``Turn N``（官方写法）。

另设"禁止虚构"规则：人造占位模式（``Bocht N`` / ``Curve N`` /
``Turn N Left|Right`` / straight / approach）与已知的直道/路段名一律禁止。
"""

from __future__ import annotations

import re

import pytest

from setup_tuner.domain._track_official import (
    FORBIDDEN_NAME_PATTERNS,
    OFFICIAL_NAMES,
    OFFICIAL_SOURCES,
    OFFICIAL_TURN_COUNTS,
)
from setup_tuner.domain.track import ALL_TRACKS, get_track_by_id

#: 明确属于"直道 / 路段 / 区段"而非弯道的名词 —— 绝不可作弯名。
FORBIDDEN_EXACT_NAMES = {
    "Reta Oposta",        # 直道名（T3–T4 之间）
    "Subida dos Boxes",   # 维修区上坡路段
    "Arquibancada",       # 看台路段
    "Wellington Straight",
    "Kemmel Straight",
}

#: 各赛道**传统**命名（无 F1 官方出处）。仅锁定防漂移，不代表官方。
TRADITIONAL_TURN_LOCKS: dict[str, dict[int, str]] = {
    "monza": {6: "Lesmo 1", 7: "Lesmo 2"},
    "spa": {5: "Les Combes", 7: "Malmedy", 14: "Stavelot", 18: "Bus Stop"},
    "suzuka": {7: "Dunlop", 11: "Hairpin", 12: "200R", 15: "130R"},
    "zandvoort": {6: "Scheivlak"},
    "melbourne": {1: "Jones", 2: "Brabham", 13: "Ascari", 14: "Stewart"},
    "monaco": {4: "Casino", 6: "Grand Hotel Hairpin", 8: "Portier", 12: "Tabac"},
    "sao_paulo": {1: "Senna S"},
}


def _corners(track_id: str) -> dict[int, str]:
    track = get_track_by_id(track_id)
    assert track is not None
    return {c.number: c.name for c in track.corners}


# ===========================================================================
# 1. 官方来源与弯数
# ===========================================================================
class TestOfficialProvenance:
    """弯道数据必须有 F1 官方的可追溯来源。"""

    def test_every_track_has_official_source(self) -> None:
        """每条赛道都要有官方来源 URL（防止再次出现无出处的臆造数据）。"""
        missing = sorted(t.track_id for t in ALL_TRACKS if t.track_id not in OFFICIAL_SOURCES)
        assert not missing, f"以下赛道缺少官方来源登记：{missing}"

    def test_sources_are_formula1_urls(self) -> None:
        """来源必须是 formula1.com 官方域名。"""
        bad = [
            (tid, url)
            for tid, urls in OFFICIAL_SOURCES.items()
            for url in urls
            if not url.startswith("https://formula1.com/")
        ]
        assert not bad, f"来源 URL 非 F1 官方域名：{bad}"

    def test_counts_cover_all_tracks(self) -> None:
        assert {t.track_id for t in ALL_TRACKS} == set(OFFICIAL_TURN_COUNTS)

    @pytest.mark.parametrize("track_id", sorted(OFFICIAL_TURN_COUNTS))
    def test_corner_count_matches_official(self, track_id: str) -> None:
        """弯数不可被"补数式"增删。"""
        expected, _evidence = OFFICIAL_TURN_COUNTS[track_id]
        assert len(_corners(track_id)) == expected, (
            f"{track_id}: 弯数 {len(_corners(track_id))} != 官方 {expected}"
        )

    @pytest.mark.parametrize("track_id", sorted(OFFICIAL_TURN_COUNTS))
    def test_corner_numbers_are_contiguous_1based(self, track_id: str) -> None:
        numbers = sorted(_corners(track_id))
        assert numbers == list(range(1, len(numbers) + 1))

    @pytest.mark.parametrize("track_id", sorted(OFFICIAL_TURN_COUNTS))
    def test_official_count_endorses_data_count(self, track_id: str) -> None:
        """弯数与代码里实际建出的弯数必须一致（防两处数据分叉）。"""
        track = get_track_by_id(track_id)
        assert track is not None
        assert len(track.corners) == OFFICIAL_TURN_COUNTS[track_id][0]


# ===========================================================================
# 2. 官方编号↔名称逐条锁定
# ===========================================================================
class TestOfficialNamesMatch:
    """F1 官方文档给出的编号↔名称必须逐条对上。"""

    @pytest.mark.parametrize("track_id", sorted(OFFICIAL_NAMES))
    def test_official_names_at_official_numbers(self, track_id: str) -> None:
        by_number = _corners(track_id)
        for number, name in OFFICIAL_NAMES[track_id].items():
            assert number in by_number, f"{track_id} 缺少 T{number}（官方名 {name}）"
            assert name.lower() in by_number[number].lower(), (
                f"{track_id} T{number} 实际是 {by_number[number]!r}，"
                f"官方文档为 {name!r}"
            )


# ===========================================================================
# 3. 禁止虚构弯位
# ===========================================================================
class TestNoFabricatedCorners:
    """不得用自造描述、直道名或路段名充当弯名。"""

    @pytest.mark.parametrize("track_id", sorted(OFFICIAL_TURN_COUNTS))
    def test_no_fabricated_placeholder_patterns(self, track_id: str) -> None:
        """禁止 Bocht N / Curve N / "Turn N Left|Right" / straight / approach。"""
        pats = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN_NAME_PATTERNS]
        offenders = [
            f"T{n} {name}"
            for n, name in _corners(track_id).items()
            if any(p.search(name) for p in pats)
        ]
        assert not offenders, f"{track_id} 存在自造占位弯名: {offenders}"

    @pytest.mark.parametrize("track_id", sorted(OFFICIAL_TURN_COUNTS))
    def test_no_straight_or_section_names(self, track_id: str) -> None:
        """直道名 / 路段名不得作弯名。"""
        offenders = [
            f"T{n} {name}"
            for n, name in _corners(track_id).items()
            if name in FORBIDDEN_EXACT_NAMES
        ]
        assert not offenders, f"{track_id} 把直道/路段名当弯名: {offenders}"

    @pytest.mark.parametrize("track_id", sorted(OFFICIAL_TURN_COUNTS))
    def test_no_duplicate_corner_names(self, track_id: str) -> None:
        """同一赛道内弯名不得重复（重复名无法区分反馈归属）。"""
        names = list(_corners(track_id).values())
        dupes = {n for n in names if names.count(n) > 1}
        assert not dupes, f"{track_id} 存在重名弯: {sorted(dupes)}"

    @pytest.mark.parametrize("track_id", sorted(OFFICIAL_NAMES))
    def test_officially_named_corners_not_overwritten_by_turn_n(
        self, track_id: str
    ) -> None:
        """F1 官方有专名的弯不得退化成 "Turn N"。"""
        by_number = _corners(track_id)
        offenders = [
            f"T{n}"
            for n in OFFICIAL_NAMES[track_id]
            if re.fullmatch(r"Turn\s+\d+", by_number.get(n, ""))
        ]
        assert not offenders, (
            f"{track_id} 的这些弯在 F1 官方文档里有专名，却被写成 Turn N: {offenders}"
        )


# ===========================================================================
# 4. 传统命名防漂移（非官方，仅锁定）
# ===========================================================================
class TestTraditionalTurnLocks:
    """传统名称位置锁定 —— 起因是连续弯区域曾整体错位。"""

    @pytest.mark.parametrize("track_id", sorted(TRADITIONAL_TURN_LOCKS))
    def test_traditional_locks_hold(self, track_id: str) -> None:
        by_number = _corners(track_id)
        for number, fragment in TRADITIONAL_TURN_LOCKS[track_id].items():
            assert number in by_number, f"{track_id} 缺少 T{number}（期望 {fragment}）"
            assert fragment.lower() in by_number[number].lower(), (
                f"{track_id} T{number} 是 {by_number[number]!r}，期望含 {fragment!r}"
            )


# ===========================================================================
# 5. 弯型与速度自洽
# ===========================================================================
class TestCornerTypeSpeedConsistency:
    """``corner_type`` 必须与 ``speed_kmh`` 落在同一速度带。"""

    BANDS = {"slow": (0.0, 115.0), "medium": (105.0, 205.0), "fast": (195.0, 400.0)}

    @pytest.mark.parametrize("track_id", sorted(OFFICIAL_TURN_COUNTS))
    def test_type_matches_speed_band(self, track_id: str) -> None:
        track = get_track_by_id(track_id)
        assert track is not None
        bad = []
        for c in track.corners:
            lo, hi = self.BANDS[c.corner_type]
            if not (lo <= c.speed_kmh <= hi):
                bad.append((c.number, c.name, c.corner_type, c.speed_kmh))
        assert not bad, f"{track_id} 弯型/速度不自洽: {bad}"

    @pytest.mark.parametrize("track_id", sorted(OFFICIAL_TURN_COUNTS))
    def test_speed_is_positive(self, track_id: str) -> None:
        track = get_track_by_id(track_id)
        assert track is not None
        assert all(c.speed_kmh > 0 for c in track.corners)
