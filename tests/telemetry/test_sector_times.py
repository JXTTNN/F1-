"""Tests for :mod:`f1opt.telemetry.sector_times` (Iter-151)."""
from __future__ import annotations

import pytest

from f1opt.telemetry.sector_times import (
    SectorTimeExtractor,
    extract_sector_times,
)


def _make_frames(n: int, lap_distance_fn) -> list[dict]:
    """Generate telemetry frames with lap_distance and frame_t."""
    frames = []
    for i in range(n):
        frames.append({
            "frame_t": float(i) * 0.5,
            "lap_distance": lap_distance_fn(i, n),
            "speed": 250.0 + i * 0.1,
        })
    return frames


class TestSectorTimeExtractor:
    def test_basic_three_sectors(self) -> None:
        ex = SectorTimeExtractor()
        frames = _make_frames(60, lambda i, n: i / (n - 1))
        sectors = ex.feed_batch(frames)
        sectors.extend(ex.flush())
        assert len(sectors) == 3
        assert sectors[0].sector == 1
        assert sectors[1].sector == 2
        assert sectors[2].sector == 3

    def test_sector_time_positive(self) -> None:
        SectorTimeExtractor()
        frames = _make_frames(60, lambda i, n: i / (n - 1))
        sectors = extract_sector_times(frames)
        for s in sectors:
            assert s.time_s >= 0
            assert s.n_frames > 0

    def test_custom_splits(self) -> None:
        SectorTimeExtractor(sector_splits=[0.0, 0.5, 1.0])
        frames = _make_frames(60, lambda i, n: i / (n - 1))
        sectors = extract_sector_times(frames, sector_splits=[0.0, 0.5, 1.0])
        assert len(sectors) == 2

    def test_lap_reset(self) -> None:
        SectorTimeExtractor()
        # First lap
        frames1 = _make_frames(30, lambda i, n: i / (n - 1))
        # Second lap (reset distance)
        frames2 = _make_frames(30, lambda i, n: i / (n - 1))
        sectors = extract_sector_times(frames1 + frames2)
        assert len(sectors) >= 3  # Should have sectors from both laps

    def test_single_frame(self) -> None:
        ex = SectorTimeExtractor()
        result = ex.feed({"frame_t": 0.0, "lap_distance": 0.0, "speed": 300.0})
        assert result is None  # First frame doesn't complete a sector

    def test_avg_speed(self) -> None:
        SectorTimeExtractor()
        frames = _make_frames(30, lambda i, n: i / (n - 1))
        sectors = extract_sector_times(frames)
        assert sectors[0].avg_speed_kph > 0

    def test_invalid_splits(self) -> None:
        with pytest.raises(ValueError):
            SectorTimeExtractor(sector_splits=[0.5, 1.0])  # doesn't start at 0
        with pytest.raises(ValueError):
            SectorTimeExtractor(sector_splits=[0.0, 0.5])  # doesn't end at 1
        with pytest.raises(ValueError):
            SectorTimeExtractor(sector_splits=[0.0, 0.5, 0.3, 1.0])  # not increasing

    def test_flush_empty(self) -> None:
        ex = SectorTimeExtractor()
        assert ex.flush() == []

    def test_n_sectors_property(self) -> None:
        ex = SectorTimeExtractor(sector_splits=[0.0, 0.25, 0.5, 0.75, 1.0])
        assert ex.n_sectors == 4