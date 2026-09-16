"""弯道命名与编号真实性回归测试（task-64）。

背景
----
``_turn_data.TURNS`` 与 ``track.py`` 的手工 builder 共同定义了 24 条赛道的
逐弯元数据。此前的数据存在三类**虚构弯位**问题，会直接导致游内弯号错位
（用户反馈的「连续弯标号直接是错误的」）：

1. **占位命名**：用 ``"Turn N"`` 填充本有专名的弯（melbourne / zandvoort）；
2. **非弯道占位**：把直道或赛道区段当作弯，例如 silverstone 的
   ``"Wellington Straight"``、spa 的 ``"Kemmel Straight"``；
3. **人造拆分补数**：为凑够弯数而拆出 ``"Parabolica entry"``、
   ``"La Source approach"`` 之类的伪弯。

本文件锁定修正后的关键事实，防止上述问题回归。
"""

from __future__ import annotations

import re

import pytest

from setup_tuner.domain.track import ALL_TRACKS, get_track_by_id

# FIA 官方弯数（2026 赛历）。弯数是赛道布局的硬事实，不可随意增删。
OFFICIAL_CORNER_COUNTS: dict[str, int] = {
    "melbourne": 14,
    "shanghai": 16,
    "suzuka": 18,
    "sakhir": 15,
    "jeddah": 27,
    "miami": 19,
    "montreal": 14,
    "monaco": 19,
    "barcelona": 14,
    "spielberg": 10,
    "silverstone": 18,
    "spa": 19,
    "hungaroring": 14,
    "zandvoort": 14,
    "monza": 11,
    "madrid": 22,
    "baku": 20,
    "singapore": 19,
    "austin": 20,
    "mexico_city": 17,
    "sao_paulo": 15,
    "las_vegas": 17,
    "lusail": 16,
    "yas_marina": 16,
}

# 命名中**绝不能出现**的伪弯标记（直道 / 区段 / 人造拆分后缀）。
# 注意：纯 "Turn N" 不在此列 —— 部分赛道（sakhir / spielberg / baku /
# miami / las_vegas 等）的弯道在 FIA 官方资料中确实没有专名，用
# "Turn N" 是正确写法。对于**确有专名**却被占位的赛道，由
# PLACEHOLDER_FORBIDDEN_TRACKS 单独约束。
FORBIDDEN_NAME_PATTERNS = [
    r"\bstraight\b",          # "Wellington Straight" / "Kemmel Straight"
    r"\bapproach\b",          # "La Source approach"
]

# 官方弯名确凿、但曾被 "Turn N" 占位覆盖的赛道（task-64 已修正，禁止回归）
PLACEHOLDER_FORBIDDEN_TRACKS = {
    "melbourne",     # Jones / Brabham / Whiteford / Marina / Lauda / Ascari ...
    "zandvoort",     # Tarzanbocht / Hugenholtzbocht / Scheivlak / Arie Luyendyk ...
    "silverstone",   # Abbey / Brooklands / Copse / Stowe / Vale / Club
    "spa",           # La Source / Eau Rouge / Pouhon / Blanchimont ...
    "monza",         # Rettifilo / Curva Grande / Lesmo / Ascari / Parabolica
    "suzuka",        # First Curve / Dunlop / Degner / Hairpin / Spoon / 130R
    "monaco",        # Sainte-Devote / Massenet / Tabac / Rascasse ...
    "sao_paulo",     # Senna S / Curva do Sol / Ferradura / Mergulho ...
}

# 各赛道的官方关键弯位（1-based 弯号 -> 名称片段），用于锁定编号正确性
AUTHORITATIVE_TURNS: dict[str, dict[int, str]] = {
    # T1/2 Rettifilo、T3 Curva Grande、T4/5 Roggia、T6/7 Lesmo、
    # T8/9/10 Ascari、T11 Parabolica(Alboreto)
    "monza": {3: "Curva Grande", 6: "Lesmo 1", 7: "Lesmo 2", 11: "Alboreto"},
    # T1 La Source、T2-4 Eau Rouge/Raidillon、T5/6 Combes、T7 Malmedy、
    # T8 Bruxelles、T9 Speakers、T10/11 Pouhon、T12/13 Fagnes、
    # T14 Stavelot、T15 Paul Frere、T16/17 Blanchimont、T18/19 Bus Stop
    "spa": {
        1: "La Source", 2: "Eau Rouge", 5: "Les Combes", 7: "Malmedy",
        10: "Pouhon", 14: "Stavelot", 18: "Bus Stop",
    },
    # T1 Abbey、T6 Brooklands、T9 Copse、T15 Stowe、T17/18 Club
    "silverstone": {1: "Abbey", 6: "Brooklands", 9: "Copse", 15: "Stowe"},
    # T1/2 First/Second、T7 Dunlop、T8/9 Degner、T11 Hairpin、
    # T12 200R、T13/14 Spoon、T15 130R、T16/17 Casio、T18 Final
    "suzuka": {1: "First Curve", 7: "Dunlop", 9: "Degner 2", 11: "Hairpin",
               12: "200R", 15: "130R", 18: "Final"},
    # T1 Tarzan、T3 Hugenholtz（倾斜弯）、T6 Scheivlak、T14 Arie Luyendyk
    "zandvoort": {1: "Tarzan", 3: "Hugenholtz", 6: "Scheivlak",
                  14: "Luyendyk"},
    # T1 Jones、T2 Brabham、T13 Ascari、T14 Stewart
    "melbourne": {1: "Jones", 2: "Brabham", 13: "Ascari", 14: "Stewart"},
}


# ===========================================================================
# 1. 弯数必须与 FIA 官方一致
# ===========================================================================
class TestOfficialCornerCounts:
    """每条赛道的弯数必须与 FIA 官方布局一致。"""

    def test_all_tracks_covered(self) -> None:
        """官方弯数表必须覆盖全部 24 条赛道。"""
        assert len(OFFICIAL_CORNER_COUNTS) == 24
        assert len(ALL_TRACKS) == 24

    @pytest.mark.parametrize("track_id", sorted(OFFICIAL_CORNER_COUNTS))
    def test_corner_count_matches_official(self, track_id: str) -> None:
        """弯数不可被"补数式"增删。"""
        track = get_track_by_id(track_id)
        assert track is not None
        expected = OFFICIAL_CORNER_COUNTS[track_id]
        assert len(track.corners) == expected, (
            f"{track_id}: 弯数 {len(track.corners)} != 官方 {expected}"
        )

    @pytest.mark.parametrize("track_id", sorted(OFFICIAL_CORNER_COUNTS))
    def test_corner_numbers_are_contiguous_1based(self, track_id: str) -> None:
        """弯号必须是 1..N 连续整数（无跳号/重复）。"""
        track = get_track_by_id(track_id)
        assert track is not None
        numbers = [c.number for c in track.corners]
        assert numbers == list(range(1, len(numbers) + 1))


# ===========================================================================
# 2. 禁止虚构弯位
# ===========================================================================
class TestNoFabricatedCorners:
    """不得用直道/区段/占位名充当弯道。"""

    @pytest.mark.parametrize("track_id", sorted(OFFICIAL_CORNER_COUNTS))
    def test_no_forbidden_name_patterns(self, track_id: str) -> None:
        """弯名不得含直道或 approach 后缀（不得把区段当弯）。"""
        track = get_track_by_id(track_id)
        assert track is not None
        offenders = [
            c.name
            for c in track.corners
            if any(re.search(p, c.name, re.IGNORECASE) for p in FORBIDDEN_NAME_PATTERNS)
        ]
        assert not offenders, f"{track_id} 存在伪弯命名: {offenders}"

    @pytest.mark.parametrize("track_id", sorted(PLACEHOLDER_FORBIDDEN_TRACKS))
    def test_placeholder_names_replaced_where_official_names_exist(
        self, track_id: str
    ) -> None:
        """官方弯名确凿的赛道，不得再用 "Turn N" 占位。"""
        track = get_track_by_id(track_id)
        assert track is not None
        offenders = [
            c.name for c in track.corners if re.fullmatch(r"Turn\s+\d+", c.name)
        ]
        assert not offenders, (
            f"{track_id} 仍有 Turn N 占位命名（官方弯名已确凿）: {offenders}"
        )

    @pytest.mark.parametrize("track_id", sorted(OFFICIAL_CORNER_COUNTS))
    def test_no_duplicate_corner_names(self, track_id: str) -> None:
        """同一赛道内弯名不得重复（重复名无法区分反馈归属）。"""
        track = get_track_by_id(track_id)
        assert track is not None
        names = [c.name for c in track.corners]
        dupes = {n for n in names if names.count(n) > 1}
        assert not dupes, f"{track_id} 存在重名弯: {sorted(dupes)}"


# ===========================================================================
# 3. 关键弯位编号正确性
# ===========================================================================
class TestAuthoritativeTurnNumbers:
    """锁定各赛道官方关键弯位的弯号。"""

    @pytest.mark.parametrize("track_id", sorted(AUTHORITATIVE_TURNS))
    def test_key_turns_at_expected_numbers(self, track_id: str) -> None:
        """官方关键弯必须落在正确弯号上。"""
        track = get_track_by_id(track_id)
        assert track is not None
        by_number = {c.number: c.name for c in track.corners}
        for number, fragment in AUTHORITATIVE_TURNS[track_id].items():
            assert number in by_number, (
                f"{track_id} 缺少 T{number}（期望 {fragment}）"
            )
            assert fragment.lower() in by_number[number].lower(), (
                f"{track_id} T{number} 是 {by_number[number]!r}，"
                f"期望包含 {fragment!r}"
            )


# ===========================================================================
# 4. 弯型与速度自洽
# ===========================================================================
class TestCornerTypeSpeedConsistency:
    """``corner_type`` 必须与 ``speed_kmh`` 落在同一速度带。"""

    # slow 弯不应快于 medium 上限；fast 弯不应慢于 medium 下限
    BANDS = {"slow": (0.0, 115.0), "medium": (105.0, 205.0), "fast": (195.0, 400.0)}

    @pytest.mark.parametrize("track_id", sorted(OFFICIAL_CORNER_COUNTS))
    def test_type_matches_speed_band(self, track_id: str) -> None:
        """弯型标注与过弯速度必须自洽。"""
        track = get_track_by_id(track_id)
        assert track is not None
        bad = []
        for c in track.corners:
            lo, hi = self.BANDS[c.corner_type]
            if not (lo <= c.speed_kmh <= hi):
                bad.append((c.number, c.name, c.corner_type, c.speed_kmh))
        assert not bad, f"{track_id} 弯型/速度不自洽: {bad}"

    @pytest.mark.parametrize("track_id", sorted(OFFICIAL_CORNER_COUNTS))
    def test_speed_is_positive(self, track_id: str) -> None:
        """过弯速度必须为正。"""
        track = get_track_by_id(track_id)
        assert track is not None
        assert all(c.speed_kmh > 0 for c in track.corners)
