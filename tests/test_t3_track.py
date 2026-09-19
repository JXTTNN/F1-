"""T3 赛道数据单元测试。

覆盖验收标准：
1. ALL_TRACKS 长度 == 24
2. 每赛道含弯道列表，anchor 在 0~1 区间
3. udp_track_id 唯一且与 EA 官方 m_trackId 枚举一致
4. get_track_by_id / get_track_by_udp_id 查找正确
"""

from __future__ import annotations

from setup_tuner.domain.track import (
    ALL_TRACKS,
    UDP_TRACK_ID_UNVERIFIED,
    Corner,
    CornerAnchor,
    Track,
    get_all_tracks,
    get_track_by_id,
    get_track_by_udp_id,
)


# ===========================================================================
# 1. ALL_TRACKS 基数与结构
# ===========================================================================
class TestTrackRoster:
    """24 条 F1 2026 赛历赛道静态数据校验。"""

    def test_all_tracks_count_is_24(self) -> None:
        """ALL_TRACKS 必须恰好 24 条赛道。"""
        assert len(ALL_TRACKS) == 24

    def test_get_all_tracks_returns_copy(self) -> None:
        """get_all_tracks 返回列表副本，长度 24。"""
        tracks = get_all_tracks()
        assert len(tracks) == 24
        assert tracks is not ALL_TRACKS  # 副本，非同一对象

    def test_all_entries_are_track_instances(self) -> None:
        """每条赛道均为 Track 实例。"""
        for t in ALL_TRACKS:
            assert isinstance(t, Track)

    def test_track_id_unique(self) -> None:
        """track_id 必须唯一。"""
        ids = [t.track_id for t in ALL_TRACKS]
        assert len(ids) == len(set(ids)), f"track_id 重复: {ids}"

    def test_udp_track_id_unique(self) -> None:
        """udp_track_id 必须唯一。"""
        udp_ids = [t.udp_track_id for t in ALL_TRACKS]
        assert len(udp_ids) == len(set(udp_ids)), f"udp_track_id 重复: {udp_ids}"

    def test_udp_track_id_matches_official_enum(self) -> None:
        """udp_track_id 必须与 EA UDP 规范的 m_trackId 官方枚举一致。

        历史问题：早期按「赛历轮次 - 1」占位分配，24 条赛道里 20 条会认错赛道
        （例如 suzuka 被赋 2，官方是 13；jeddah 被赋 4，官方是 29）。
        """
        official = {
            "melbourne": 0, "shanghai": 2, "sakhir": 3, "barcelona": 4,
            "monaco": 5, "montreal": 6, "silverstone": 7, "hungaroring": 9,
            "spa": 10, "monza": 11, "singapore": 12, "suzuka": 13,
            "yas_marina": 14, "austin": 15, "sao_paulo": 16, "spielberg": 17,
            "mexico_city": 19, "baku": 20, "zandvoort": 26, "jeddah": 29,
            "miami": 30, "las_vegas": 31, "lusail": 32,
        }
        for t in ALL_TRACKS:
            if t.track_id in UDP_TRACK_ID_UNVERIFIED:
                continue
            assert t.track_id in official, f"{t.track_id} 缺少官方枚举"
            assert t.udp_track_id == official[t.track_id], (
                f"{t.track_id}: udp_track_id={t.udp_track_id}，官方={official[t.track_id]}"
            )

    def test_udp_track_id_unverified_is_explicit(self) -> None:
        """未确认官方枚举的赛道必须有显式标记，不许"看起来已校准"。"""
        flagged = {t.track_id for t in ALL_TRACKS} & set(UDP_TRACK_ID_UNVERIFIED)
        assert flagged == set(UDP_TRACK_ID_UNVERIFIED)
        for tid in UDP_TRACK_ID_UNVERIFIED:
            track = get_track_by_id(tid)
            assert track is not None and track.udp_track_id > 32, (
                f"{tid} 的占位 id 应大于已知枚举上限 32"
            )

    def test_track_type_valid(self) -> None:
        """每赛道 track_type 必须在合法枚举内。"""
        valid_types = {
            "high_speed_low_downforce",
            "street",
            "high_downforce",
            "medium",
            "mixed",
        }
        for t in ALL_TRACKS:
            assert t.track_type in valid_types, (
                f"赛道 {t.track_id!r} track_type {t.track_type!r} 非法"
            )

    def test_length_positive(self) -> None:
        """每赛道长度必须为正。"""
        for t in ALL_TRACKS:
            assert t.length_m > 0, f"赛道 {t.track_id!r} 长度非正: {t.length_m}"


# ===========================================================================
# 2. 弯道列表与锚点
# ===========================================================================
class TestCorners:
    """每赛道弯道列表与锚点校验。"""

    def test_every_track_has_corners(self) -> None:
        """每赛道至少含 1 个弯道。"""
        for t in ALL_TRACKS:
            assert len(t.corners) >= 1, f"赛道 {t.track_id!r} 无弯道"

    def test_corner_numbering_1_based(self) -> None:
        """弯道编号从 1 开始连续递增。"""
        for t in ALL_TRACKS:
            numbers = [c.number for c in t.corners]
            assert numbers == list(range(1, len(t.corners) + 1)), (
                f"赛道 {t.track_id!r} 弯道编号不连续: {numbers}"
            )

    def test_corner_anchor_in_unit_interval(self) -> None:
        """每弯道锚点 anchor_x / anchor_y 必须在 (0, 1) 开区间内。"""
        for t in ALL_TRACKS:
            for c in t.corners:
                assert isinstance(c.anchor, CornerAnchor)
                assert 0.0 < c.anchor.anchor_x < 1.0, (
                    f"赛道 {t.track_id!r} 弯道 {c.number} anchor_x={c.anchor.anchor_x} 越界"
                )
                assert 0.0 < c.anchor.anchor_y < 1.0, (
                    f"赛道 {t.track_id!r} 弯道 {c.number} anchor_y={c.anchor.anchor_y} 越界"
                )

    def test_corner_type_valid(self) -> None:
        """弯道类型必须为 slow/medium/fast。"""
        valid = {"slow", "medium", "fast"}
        for t in ALL_TRACKS:
            for c in t.corners:
                assert c.corner_type in valid, (
                    f"赛道 {t.track_id!r} 弯道 {c.number} type {c.corner_type!r} 非法"
                )

    def test_corner_speed_positive(self) -> None:
        """弯道预计速度必须为正。"""
        for t in ALL_TRACKS:
            for c in t.corners:
                assert c.speed_kmh > 0, (
                    f"赛道 {t.track_id!r} 弯道 {c.number} 速度非正: {c.speed_kmh}"
                )

    def test_corner_instances(self) -> None:
        """弯道对象均为 Corner 实例。"""
        for t in ALL_TRACKS:
            for c in t.corners:
                assert isinstance(c, Corner)


# ===========================================================================
# 3. 查询函数
# ===========================================================================
class TestTrackLookup:
    """get_track_by_id / get_track_by_udp_id 查询正确性。"""

    def test_get_track_by_id_known(self) -> None:
        """已知 track_id 应返回对应赛道。"""
        t = get_track_by_id("suzuka")
        assert t is not None
        assert t.track_id == "suzuka"
        assert t.official_name == "Japanese Grand Prix"
        assert t.country == "Japan"

    def test_get_track_by_id_unknown_returns_none(self) -> None:
        """未知 track_id 应返回 None。"""
        assert get_track_by_id("nonexistent") is None
        assert get_track_by_id("") is None

    def test_get_track_by_udp_id_known(self) -> None:
        """已知 udp_track_id 应返回对应赛道。"""
        t = get_track_by_udp_id(13)  # suzuka（官方 m_trackId=13）
        assert t is not None
        assert t.track_id == "suzuka"

    def test_get_track_by_udp_id_unknown_returns_none(self) -> None:
        """未知 udp_track_id 应返回 None。"""
        assert get_track_by_udp_id(99) is None
        assert get_track_by_udp_id(-1) is None

    def test_id_and_udp_lookup_consistent(self) -> None:
        """对每条赛道，两种查询方式应返回同一对象。"""
        for t in ALL_TRACKS:
            by_id = get_track_by_id(t.track_id)
            by_udp = get_track_by_udp_id(t.udp_track_id)
            assert by_id is t
            assert by_udp is t

    def test_all_24_tracks_findable_by_id(self) -> None:
        """全部 24 条赛道都能通过 track_id 查到。"""
        for t in ALL_TRACKS:
            assert get_track_by_id(t.track_id) is not None

    def test_all_24_tracks_findable_by_udp(self) -> None:
        """全部 24 条赛道都能通过 udp_track_id 查到。"""
        for t in ALL_TRACKS:
            assert get_track_by_udp_id(t.udp_track_id) is not None


# ===========================================================================
# 4. 已知赛道数据抽样断言
# ===========================================================================
class TestKnownTrackData:
    """对几条手工录入赛道的已知数据做抽样断言。"""

    def test_melbourne_metadata(self) -> None:
        """Melbourne 元数据核对。"""
        t = get_track_by_id("melbourne")
        assert t is not None
        assert t.circuit_name == "Albert Park Grand Prix Circuit"
        assert t.track_type == "medium"
        assert t.length_m == 5278.0
        assert t.udp_track_id == 0
        assert len(t.corners) == 14

    def test_suzuka_corners(self) -> None:
        """Suzuka 18 弯，按 FIA 官方弯号（task-64 二次修正）。"""
        t = get_track_by_id("suzuka")
        assert t is not None
        assert len(t.corners) == 18
        assert t.corners[0].name == "First Curve"
        # 按 F1 官方口径校正：Esses=T3-T6、Dunlop=T7、Degner=T8/T9、
        # T10 官方无专名（故写作 "Turn 10"，这是官方对无专名弯的写法）、
        # Hairpin=T11、200R=T12、Spoon=T13/T14、130R=T15、
        # Casio=T16/T17、Final Corner=T18
        names = {c.number: c.name for c in t.corners}
        assert names[7] == "Dunlop Curve"
        assert names[8] == "Degner 1"
        assert names[10] == "Turn 10"
        assert names[11] == "Hairpin"
        assert names[12] == "200R"
        assert names[15] == "130R"
        assert names[18] == "Final Corner"
        assert t.corners[0].number == 1

    def test_monaco_street_circuit(self) -> None:
        """Monaco 为街道赛。"""
        t = get_track_by_id("monaco")
        assert t is not None
        assert t.track_type == "street"
        assert t.length_m == 3337.0

    def test_monza_high_speed(self) -> None:
        """Monza 为高速低下压力赛道。"""
        t = get_track_by_id("monza")
        assert t is not None
        assert t.track_type == "high_speed_low_downforce"

    def test_svg_path_matches_track_id(self) -> None:
        """每赛道 svg_path 应为 tracks/<track_id>.svg。"""
        for t in ALL_TRACKS:
            assert t.svg_path == f"tracks/{t.track_id}.svg", (
                f"赛道 {t.track_id!r} svg_path {t.svg_path!r} 不符"
            )