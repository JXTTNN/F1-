"""``f1opt.data.tracks`` 2026 赛道数据库测试."""

from __future__ import annotations

import pytest

from f1opt.data import tracks
from f1opt.data.tracks import (
    ALL_TRACKS,
    TRACKS_BY_ID,
    Track,
    all_tracks,
    get_track,
    get_track_by_round,
    sprint_tracks,
)

EXPECTED_SPRINT_IDS = ["shanghai", "miami", "montreal", "silverstone", "zandvoort", "singapore"]


def test_all_tracks_returns_24_entries_in_calendar_order() -> None:
    """共 24 条赛道, round_number 1..24 唯一且连续, 与赛历顺序一致."""
    result = all_tracks()
    assert len(result) == 24
    assert result == ALL_TRACKS
    rounds = [t.round_number for t in result]
    assert rounds == list(range(1, 25))
    assert len(set(rounds)) == 24


def test_melbourne_is_round_1_and_yas_marina_is_round_24() -> None:
    """揭幕战墨尔本为第 1 轮, 收官战阿布扎比为第 24 轮."""
    assert get_track("melbourne").round_number == 1
    assert get_track("yas_marina").round_number == 24


def test_madrid_properties() -> None:
    """马德里 2026 首秀: 第 16 轮, 非 Sprint, 街道赛."""
    madrid = get_track("madrid")
    assert madrid.round_number == 16
    assert madrid.is_sprint is False
    assert madrid.track_type == "street"


def test_sprint_tracks_returns_exactly_six_expected() -> None:
    """Sprint 周末恰好 6 场, 与官方赛历一致."""
    sprints = sprint_tracks()
    assert len(sprints) == 6
    assert [t.track_id for t in sprints] == EXPECTED_SPRINT_IDS
    assert all(t.is_sprint for t in sprints)
    # 非 Sprint 的赛道数量 = 24 - 6 = 18
    non_sprint_ids = {t.track_id for t in ALL_TRACKS if not t.is_sprint}
    assert len(non_sprint_ids) == 18
    assert set(EXPECTED_SPRINT_IDS).isdisjoint(non_sprint_ids)


def test_every_track_has_positive_length_and_corners() -> None:
    """所有赛道长度与弯角数必须为正."""
    for track in ALL_TRACKS:
        assert isinstance(track, Track)
        assert track.length_m > 0, track.track_id
        assert track.corners > 0, track.track_id


def test_get_track_unknown_id_raises_value_error() -> None:
    """未知 track_id 抛出 ValueError."""
    with pytest.raises(ValueError):
        get_track("nonexistent")


def test_get_track_by_round_returns_matching_track() -> None:
    """按轮次查询应返回与 track_id 查询一致的结果."""
    for track in ALL_TRACKS:
        assert get_track_by_round(track.round_number) is track
    with pytest.raises(ValueError):
        get_track_by_round(0)
    with pytest.raises(ValueError):
        get_track_by_round(25)


def test_tracks_by_id_matches_all_tracks() -> None:
    """TRACKS_BY_ID 的值集合应与 ALL_TRACKS 完全一致."""
    assert set(TRACKS_BY_ID.values()) == set(ALL_TRACKS)
    assert len(TRACKS_BY_ID) == len(ALL_TRACKS) == 24
    for track in ALL_TRACKS:
        assert TRACKS_BY_ID[track.track_id] is track


def test_track_ids_are_unique_and_match_expected_set() -> None:
    """24 个 track_id 唯一, 且与规范约定的 id 列表一致."""
    expected_ids = [
        "melbourne", "shanghai", "suzuka", "sakhir", "jeddah", "miami", "montreal",
        "monaco", "barcelona", "spielberg", "silverstone", "spa", "hungaroring",
        "zandvoort", "monza", "madrid", "baku", "singapore", "austin",
        "mexico_city", "sao_paulo", "las_vegas", "lusail", "yas_marina",
    ]
    actual_ids = [t.track_id for t in ALL_TRACKS]
    assert len(set(actual_ids)) == 24
    assert actual_ids == expected_ids


def test_module_constants_consistent() -> None:
    """重复导入应保持一致性 (模块级常量未被破坏)."""
    import importlib

    reloaded = importlib.reload(tracks)
    try:
        assert len(reloaded.ALL_TRACKS) == 24
        assert set(reloaded.TRACKS_BY_ID.values()) == set(reloaded.ALL_TRACKS)
        assert reloaded.get_track("melbourne").round_number == 1
    finally:
        importlib.reload(tracks)
