"""Sector time extraction from telemetry frames (Iter-151).

EA F1 2026 professional standard: sector times are fundamental to lap
analysis. The :class:`SectorTimeExtractor` computes sector times from
telemetry frame data by detecting sector boundary crossings based on
lap distance progression.

F1 25 telemetry provides ``lap_distance`` as a fraction of the total lap
distance (0.0–1.0). Sector boundaries are typically at 0.0, 0.33, 0.66,
and 1.0 (or custom splits from track-specific metadata).

Usage::

    from f1opt.telemetry.sector_times import SectorTimeExtractor, extract_sector_times

    sectors = extract_sector_times(frames, sector_splits=[0.0, 0.33, 0.66, 1.0])
    print(sectors)  # [{"sector": 1, "time_s": 28.5, "n_frames": 143}, ...]
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass


@_dataclass
class SectorTime:
    """A single sector time measurement."""

    sector: int
    """Sector number (1, 2, or 3)."""

    time_s: float
    """Sector time in seconds."""

    n_frames: int
    """Number of telemetry frames in this sector."""

    start_distance: float
    """Lap distance at sector start (0.0–1.0)."""

    end_distance: float
    """Lap distance at sector end (0.0–1.0)."""

    avg_speed_kph: float
    """Average speed across the sector (km/h)."""


class SectorTimeExtractor:
    """Extract sector times from telemetry frame data (Iter-151).

    Detects sector boundary crossings by monitoring ``lap_distance``
    progression. When ``lap_distance`` crosses a sector split boundary,
    the current sector is closed and a new sector begins.

    Handles lap resets (``lap_distance`` drops from near 1.0 to near 0.0)
    gracefully by starting a new lap.
    """

    def __init__(self, sector_splits: list[float] | None = None) -> None:
        """Initialise with optional sector boundary definitions.

        Args:
            sector_splits: Lap-distance fractions at sector boundaries.
                Default ``[0.0, 0.33, 0.66, 1.0]`` for three equal sectors.
        """
        self._splits = sector_splits or [0.0, 0.33, 0.66, 1.0]
        self._validate_splits()
        self._current_sector = 0
        self._sector_start_time: float | None = None
        self._sector_start_distance: float = 0.0
        self._sector_frames: list[dict] = []
        self._sectors: list[SectorTime] = []

    def _validate_splits(self) -> None:
        if len(self._splits) < 2:
            raise ValueError("Need at least 2 sector splits (start and end)")
        if self._splits[0] != 0.0:
            raise ValueError("First sector split must be 0.0")
        if self._splits[-1] != 1.0:
            raise ValueError("Last sector split must be 1.0")
        for i in range(1, len(self._splits)):
            if self._splits[i] <= self._splits[i - 1]:
                raise ValueError(
                    f"Sector splits must be strictly increasing: "
                    f"{self._splits[i]} <= {self._splits[i - 1]}"
                )

    @property
    def n_sectors(self) -> int:
        """Number of sectors (one less than the number of splits)."""
        return len(self._splits) - 1

    def feed(self, frame: dict) -> SectorTime | None:
        """Feed a single telemetry frame and return a completed sector if any.

        Args:
            frame: A telemetry frame dict with at least ``lap_distance``,
                ``frame_t``, and optionally ``speed``.

        Returns:
            A :class:`SectorTime` when a sector boundary is crossed,
            or ``None`` if still within the current sector.
        """
        lap_distance = frame.get("lap_distance", 0.0)
        frame_t = frame.get("frame_t", 0.0)

        # Detect lap reset (distance drops significantly)
        if (
            self._sector_start_distance is not None
            and lap_distance < self._sector_start_distance - 0.1
        ):
            # Lap reset: close current sector and restart
            result = self._finalize_current_sector(
                frame_t, lap_distance=1.0, force=True
            )
            self._reset()
            self._sector_start_distance = 0.0
            self._sector_start_time = frame_t
            self._sector_frames.append(frame)
            return result

        # First frame: start tracking
        if self._sector_start_time is None:
            self._sector_start_time = frame_t
            self._sector_start_distance = lap_distance
            self._sector_frames.append(frame)
            return None

        # Check if we crossed a sector boundary
        if self._current_sector >= self.n_sectors:
            # All sectors already completed; just accumulate frames
            self._sector_frames.append(frame)
            return None

        next_boundary = self._splits[self._current_sector + 1]
        if lap_distance >= next_boundary:
            result = self._finalize_current_sector(frame_t, lap_distance)
            # Advance to next sector
            self._current_sector += 1
            self._sector_start_time = frame_t
            self._sector_start_distance = lap_distance
            self._sector_frames.append(frame)
            return result

        self._sector_frames.append(frame)
        return None

    def feed_batch(self, frames: list[dict]) -> list[SectorTime]:
        """Feed a batch of frames and return all completed sectors."""
        results: list[SectorTime] = []
        for f in frames:
            s = self.feed(f)
            if s is not None:
                results.append(s)
        return results

    def flush(self) -> list[SectorTime]:
        """Flush any incomplete sector and return it (if any)."""
        if (
            self._sector_frames
            and self._sector_start_time is not None
            and self._current_sector < self.n_sectors
        ):
            last = self._sector_frames[-1]
            sector = self._finalize_current_sector(
                last.get("frame_t", 0.0),
                last.get("lap_distance", 1.0),
                force=True,
            )
            return [sector] if sector else []
        return []

    def _finalize_current_sector(
        self,
        end_time: float,
        lap_distance: float,
        force: bool = False,
    ) -> SectorTime | None:
        if not self._sector_frames or self._sector_start_time is None:
            return None

        sector_time = end_time - self._sector_start_time
        if sector_time <= 0 and not force:
            return None

        # Compute average speed
        speeds = [
            f.get("speed", 0.0)
            for f in self._sector_frames
            if "speed" in f
        ]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

        sector = SectorTime(
            sector=self._current_sector + 1,
            time_s=round(sector_time, 4),
            n_frames=len(self._sector_frames),
            start_distance=self._sector_start_distance,
            end_distance=lap_distance,
            avg_speed_kph=round(avg_speed, 2),
        )
        self._sectors.append(sector)
        return sector

    def _reset(self) -> None:
        self._current_sector = 0
        self._sector_start_time = None
        self._sector_start_distance = 0.0
        self._sector_frames = []


def extract_sector_times(
    frames: list[dict],
    sector_splits: list[float] | None = None,
) -> list[SectorTime]:
    """Convenience function: extract sector times from a list of frames."""
    ex = SectorTimeExtractor(sector_splits)
    results = ex.feed_batch(frames)
    results.extend(ex.flush())
    return results


__all__ = [
    "SectorTime",
    "SectorTimeExtractor",
    "extract_sector_times",
]