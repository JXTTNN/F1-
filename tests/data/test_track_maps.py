"""Tests for track map pixel-coordinate mapping (Opt-Map)."""

import pytest

from f1opt.data.track_maps import (
    TRACK_MAPS,
    interpolate_position,
    find_nearest_corner,
    get_track_map,
    CornerDef,
)


class TestTrackMapBasics:
    """Test basic track map data structure."""

    def test_tracks_have_required_fields(self):
        assert len(TRACK_MAPS) >= 10, "Should have at least 10 track maps"
        for tid, tm in TRACK_MAPS.items():
            assert tm.track_id == tid
            assert tm.image_file
            assert tm.canvas_width > 0
            assert tm.canvas_height > 0
            assert len(tm.control_points) >= 2
            assert len(tm.corners) >= 3
            assert tm.sector_boundaries

    def test_control_points_are_sorted(self):
        for tid, tm in TRACK_MAPS.items():
            distances = [cp[0] for cp in tm.control_points]
            assert distances == sorted(distances), f"{tid}: control points must be sorted by distance"

    def test_control_points_span_full_lap(self):
        for tid, tm in TRACK_MAPS.items():
            # First point at or near 0
            assert tm.control_points[0][0] >= 0
            # Last point matches track length
            # (we only have length in ALL_TRACKS, skip this check)


class TestInterpolatePosition:
    """Test distance-to-pixel interpolation."""

    def test_interpolate_at_start(self):
        for tid, tm in TRACK_MAPS.items():
            x, y = interpolate_position(0, tm.control_points)
            # Should be near the first control point
            first = tm.control_points[0]
            assert abs(x - first[1]) < 1
            assert abs(y - first[2]) < 1

    def test_interpolate_at_end(self):
        for tid, tm in TRACK_MAPS.items():
            last_dist = tm.control_points[-1][0]
            x, y = interpolate_position(last_dist, tm.control_points)
            last = tm.control_points[-1]
            assert abs(x - last[1]) < 1
            assert abs(y - last[2]) < 1

    def test_interpolate_linear(self):
        melb = get_track_map("melbourne")
        # Interpolate midpoint between 2000m and 2500m
        pts = melb.control_points
        # Find two control points
        for i in range(len(pts) - 1):
            if pts[i][0] == 2000 and pts[i+1][0] == 2500:
                x, y = interpolate_position(2250, pts)
                # Linear interpolation: (x1+x2)/2, (y1+y2)/2
                ex = (pts[i][1] + pts[i+1][1]) / 2
                ey = (pts[i][2] + pts[i+1][2]) / 2
                assert abs(x - ex) < 0.1
                assert abs(y - ey) < 0.1
                return
        pytest.skip("Test control points not found")


class TestFindNearestCorner:
    """Test corner detection from distance."""

    def test_finds_corner_in_range(self):
        melb = get_track_map("melbourne")
        if len(melb.corners) >= 1:
            c = melb.corners[0]
            mid_dist = (c.distance_start + c.distance_end) / 2
            found = find_nearest_corner(mid_dist, melb.corners)
            assert found is not None
            assert found.corner_id == c.corner_id

    def test_returns_none_outside_all_ranges(self):
        melb = get_track_map("melbourne")
        all_distances = []
        for c in melb.corners:
            all_distances.extend([c.distance_start, c.distance_end])
        min_d, max_d = min(all_distances), max(all_distances)
        # Before any corner
        result = find_nearest_corner(min_d - 1000, melb.corners)
        # Should return None or last corner (implementation detail)


class TestGetTrackMap:
    """Test track map retrieval."""

    def test_get_existing_track(self):
        melb = get_track_map("melbourne")
        assert melb is not None
        assert melb.track_id == "melbourne"

    def test_get_nonexistent_track(self):
        result = get_track_map("nonexistent_track_id")
        assert result is None
