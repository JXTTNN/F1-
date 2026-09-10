"""Telemetry gap filling for frame sequences (Iter-155).

When packet loss occurs (detected by :class:`~f1opt.telemetry.validation.FrameTracker`
or :class:`~f1opt.telemetry.packet_loss.PacketLossDetector`), the resulting
frame sequence has gaps. This module provides utilities to fill those gaps
with interpolated frames, so downstream analytics (lap-time computation,
sector analysis, performance benchmarking) can operate on a continuous
time series without artifacts.

Filling strategies:

- **Linear interpolation** (default): for numeric fields, linearly interpolate
  between the last frame before the gap and the first frame after. Suitable
  for smoothly-varying channels (speed, throttle, brake, steering, ERS).
- **Hold-last**: for discrete/categorical fields (gear, DRS status, pit
  status), hold the last known value until the gap ends.
- **No-fill**: explicitly skip certain fields (e.g. timestamp-derived fields
  that should be recomputed, not interpolated).

Usage::

    from f1opt.telemetry.gap_filler import GapFiller

    filler = GapFiller(
        float_fields=["speed", "throttle", "brake", "steer"],
        int_fields=["gear", "drs"],
    )
    filled = filler.fill_gaps(frames, max_gap=5)
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "GapFiller",
    "fill_frame_gaps",
]

#: Default fields to linearly interpolate (smoothly-varying channels).
_DEFAULT_FLOAT_FIELDS: tuple[str, ...] = (
    "speed",
    "throttle",
    "brake",
    "steer",
    "g_lat",
    "g_long",
    "rpm",
    "ers_store",
    "ers_harvested_this_lap",
    "ers_deployed_this_lap",
    "tyre_temp_fl",
    "tyre_temp_fr",
    "tyre_temp_rl",
    "tyre_temp_rr",
    "fuel_in_tank",
    "fuel_remaining_laps",
    "lap_distance",
    "lap_time",
)

#: Default fields to hold-last (discrete/categorical channels).
# Iter-264: 对齐 aligner 字段名 (ers_deploy_mode 是离散模式 0-3, 应 hold-last;
# drs -> drs_allowed/drs_active)。
_DEFAULT_INT_FIELDS: tuple[str, ...] = (
    "gear",
    "ers_deploy_mode",
    "drs_allowed",
    "drs_active",
    "pit_status",
    "session_time",
)


class GapFiller:
    """Fill gaps in a frame sequence with interpolated frames (Iter-155).

    Args:
        float_fields: Field names to linearly interpolate across gaps.
        int_fields: Field names to hold-last across gaps (discrete values).
        max_gap: Maximum gap size to fill. Gaps larger than this are left
            unfilled (likely a session interruption or flashback, not a
            simple packet drop). Default 10.
    """

    def __init__(
        self,
        float_fields: tuple[str, ...] | list[str] | None = None,
        int_fields: tuple[str, ...] | list[str] | None = None,
        max_gap: int = 10,
    ) -> None:
        self._float_fields: set[str] = (
            set(float_fields) if float_fields is not None
            else set(_DEFAULT_FLOAT_FIELDS)
        )
        self._int_fields: set[str] = (
            set(int_fields) if int_fields is not None
            else set(_DEFAULT_INT_FIELDS)
        )
        self._max_gap = max_gap

    @property
    def max_gap(self) -> int:
        return self._max_gap

    def fill_gaps(self, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fill gaps in a frame sequence.

        Args:
            frames: List of frame dicts, each expected to have at least a
                ``"frame_id"`` key (int) that is monotonically increasing.
                Frames must be sorted by ``frame_id``.

        Returns:
            A new list of frames with gaps filled. If the input has no gaps
            (or gaps > ``max_gap``), the original list is returned unchanged.
            Original frames are not modified; filled frames are new dicts.
        """
        if len(frames) < 2:
            return list(frames)

        result: list[dict[str, Any]] = [frames[0]]

        for i in range(1, len(frames)):
            prev_frame = frames[i - 1]
            curr_frame = frames[i]
            prev_id = prev_frame.get("frame_id", 0)
            curr_id = curr_frame.get("frame_id", 0)
            gap = curr_id - prev_id - 1

            if gap <= 0:
                # No gap (or duplicate / out-of-order) — just append
                result.append(curr_frame)
                continue

            if gap > self._max_gap:
                # Gap too large — don't fill (likely session interruption)
                result.append(curr_frame)
                continue

            # Fill the gap with interpolated frames
            for g in range(1, gap + 1):
                alpha = g / (gap + 1)
                filled = self._interpolate_frame(prev_frame, curr_frame, alpha, prev_id + g)
                result.append(filled)

            result.append(curr_frame)

        return result

    def _interpolate_frame(
        self,
        prev: dict[str, Any],
        curr: dict[str, Any],
        alpha: float,
        frame_id: int,
    ) -> dict[str, Any]:
        """Create a single interpolated frame at position ``alpha`` between prev and curr.

        Args:
            prev: Frame before the gap.
            curr: Frame after the gap.
            alpha: Interpolation factor in [0, 1] (0 = prev, 1 = curr).
            frame_id: The frame_id for the interpolated frame.

        Returns:
            A new frame dict with interpolated values.
        """
        filled: dict[str, Any] = {}

        # Collect all keys from both frames
        all_keys = set(prev.keys()) | set(curr.keys())

        for key in all_keys:
            if key == "frame_id":
                filled[key] = frame_id
                continue

            prev_val = prev.get(key)
            curr_val = curr.get(key)

            # If either is None, hold the non-None value
            if prev_val is None and curr_val is not None:
                filled[key] = curr_val
                continue
            if curr_val is None and prev_val is not None:
                filled[key] = prev_val
                continue
            if prev_val is None and curr_val is None:
                filled[key] = None
                continue

            # 走到这里 prev_val / curr_val 必非 None (4 种组合已全部 continue).
            assert prev_val is not None and curr_val is not None
            # Linear interpolation for float fields
            if key in self._float_fields:
                try:
                    filled[key] = float(prev_val) + alpha * (float(curr_val) - float(prev_val))
                except (TypeError, ValueError):
                    filled[key] = prev_val  # hold-last fallback
            # Hold-last for int/categorical fields
            elif key in self._int_fields:
                filled[key] = prev_val
            else:
                # Unknown field: hold-last by default (safe)
                filled[key] = prev_val

        return filled


def fill_frame_gaps(
    frames: list[dict[str, Any]],
    *,
    float_fields: tuple[str, ...] | list[str] | None = None,
    int_fields: tuple[str, ...] | list[str] | None = None,
    max_gap: int = 10,
) -> list[dict[str, Any]]:
    """Convenience function: fill gaps in a frame sequence (Iter-155).

    See :class:`GapFiller` for details.
    """
    filler = GapFiller(float_fields=float_fields, int_fields=int_fields, max_gap=max_gap)
    return filler.fill_gaps(frames)
