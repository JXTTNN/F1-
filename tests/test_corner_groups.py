"""弯道段（corner group）单元测试 —— 连续弯整改（task-63）的数据层验收。"""
from __future__ import annotations

from setup_tuner.domain._track_arcs import TRACK_CORNER_ARCS
from setup_tuner.domain.corner_groups import (
    GROUP_MAX_SPAN,
    build_corner_groups,
    corner_group_index,
    group_for_progress,
    nearest_member,
)


class TestBuildGroups:
    """段推导。"""

    def test_dense_corners_chain_into_segment(self) -> None:
        """间距 ≤ 阈值的弯链成一段。"""
        arcs = {1: 0.00, 2: 0.01, 3: 0.02, 4: 0.50, 5: 0.56}
        groups = build_corner_groups(arcs)
        multi = [g for g in groups if g["size"] > 1]
        assert [g["members"] for g in multi] == [[1, 2, 3]]

    def test_gap_over_threshold_breaks_chain(self) -> None:
        """间距超过阈值 → 断链分段（1→2 间距 1% 密集成段，2→3 间距 29% 断开）。"""
        arcs = {1: 0.00, 2: 0.01, 3: 0.30}
        groups = build_corner_groups(arcs)
        assert [(g["name"], g["members"]) for g in groups] == [
            ("T1-T2", [1, 2]), ("T3", [3]),
        ]

    def test_span_cap_prevents_runaway_merge(self) -> None:
        """等间距但总跨度超上限 → 不无限合并。"""
        arcs = {i + 1: i * 0.06 for i in range(5)}  # 0.00/0.06/0.12/0.18/0.24
        groups = build_corner_groups(arcs)
        for g in groups:
            assert g["arc_end"] - g["arc_start"] <= GROUP_MAX_SPAN + 1e-9

    def test_single_corner_is_segment(self) -> None:
        """单弯也是段（members=[n]），调用方无需判空。"""
        groups = build_corner_groups({7: 0.42})
        assert groups == [{
            "name": "T7", "members": [7],
            "arc_start": 0.42, "arc_end": 0.42, "size": 1,
        }]

    def test_cyclic_chain_across_wrap(self) -> None:
        """回绕安全：跨 0/1 边界的连续弯链成一段，且段不跨边界。"""
        # corner 1 在 0.99，corner 2 在 0.005，corner 3 在 0.02 —— 实际连续
        arcs = {1: 0.99, 2: 0.005, 3: 0.02}
        groups = build_corner_groups(arcs)
        multi = [g for g in groups if g["size"] > 1]
        assert multi and set(multi[0]["members"]) == {1, 2, 3}
        for g in groups:
            assert g["arc_start"] <= g["arc_end"], "段不得跨 0/1 边界"

    def test_empty(self) -> None:
        assert build_corner_groups({}) == []


class TestRealTracks:
    """24 条真实赛道的段推导不变量。"""

    def test_all_tracks_covered_exactly_once(self) -> None:
        """每个弯恰好属于一个段（无遗漏、无重复）。"""
        for track_id, arcs in TRACK_CORNER_ARCS.items():
            groups = build_corner_groups(arcs)
            covered = [m for g in groups for m in g["members"]]
            assert sorted(covered) == sorted(arcs), f"{track_id} 覆盖不完整"
            assert len(covered) == len(set(covered)), f"{track_id} 有重复成员"

    def test_no_group_crosses_wrap(self) -> None:
        """任何段都不跨 0/1 边界（arc_start <= arc_end）。"""
        for track_id, arcs in TRACK_CORNER_ARCS.items():
            for g in build_corner_groups(arcs):
                assert g["arc_start"] <= g["arc_end"], f"{track_id}/{g['name']} 跨边界"

    def test_dense_region_produces_multi_member_group(self) -> None:
        """Suzuka S 弯（T2-T7 密集区）应产生多成员段。"""
        groups = build_corner_groups(TRACK_CORNER_ARCS["suzuka"])
        multi = [g for g in groups if g["size"] > 1]
        assert multi, "Suzuka 连续弯未产生任何段"
        big = max(multi, key=lambda g: g["size"])
        assert big["size"] >= 2

    def test_monza_wrap_fixed(self) -> None:
        """monza（回绕赛道）不应再出现负间距式的段。"""
        for g in build_corner_groups(TRACK_CORNER_ARCS["monza"]):
            assert g["arc_start"] <= g["arc_end"]


class TestMapping:
    """进度 → 段 → 成员。"""

    ARCS = {1: 0.00, 2: 0.01, 3: 0.02, 4: 0.50, 5: 0.56}

    def test_containment(self) -> None:
        groups = build_corner_groups(self.ARCS)
        g = group_for_progress(0.015, groups)
        assert g is not None and g["members"] == [1, 2, 3]

    def test_gap_maps_to_nearest_group(self) -> None:
        """组间空隙 → 最近段（不再跳变）。"""
        groups = build_corner_groups(self.ARCS)
        g = group_for_progress(0.25, groups)  # 0.02 与 0.50 之间的空隙
        assert g is not None

    def test_nearest_member(self) -> None:
        groups = build_corner_groups(self.ARCS)
        g = group_for_progress(0.015, groups)
        assert nearest_member(0.001, g, self.ARCS) == 1
        assert nearest_member(0.019, g, self.ARCS) == 3

    def test_index_covers_all(self) -> None:
        groups = build_corner_groups(self.ARCS)
        index = corner_group_index(groups)
        assert set(index) == set(self.ARCS)

    def test_progress_wraps(self) -> None:
        """进度 1.0 与 0.0 等价（环形）。"""
        groups = build_corner_groups(self.ARCS)
        assert group_for_progress(0.0, groups) is not None
        assert group_for_progress(1.0, groups) is not None
