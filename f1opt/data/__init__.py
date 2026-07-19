"""``f1opt.data`` 子包: 赛道数据库与运行期数据存储.

公开 :mod:`f1opt.data.tracks` 的核心 API 供上层模块直接导入.
"""

from __future__ import annotations

from f1opt.data.tracks import (
    ALL_TRACKS,
    TRACKS_BY_ID,
    Track,
    TrackType,
    all_tracks,
    get_track,
    get_track_by_round,
    sprint_tracks,
)

__all__ = [
    "ALL_TRACKS",
    "TRACKS_BY_ID",
    "Track",
    "TrackType",
    "all_tracks",
    "get_track",
    "get_track_by_round",
    "sprint_tracks",
]
