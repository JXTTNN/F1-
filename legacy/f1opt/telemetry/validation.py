"""Field-level validation + flashback / gap detection for F1 25 telemetry.

Two responsibilities:

1. :func:`validate_sample` — cheap range checks on the parsed payload of a
   single packet (throttle/brake/steer in range, tyre pressures/temps
   non-negative, fuel/ERS stores non-negative, wear percentages in [0, 100]).

2. :class:`FrameTracker` — tracks the last seen ``m_overallFrameIdentifier``
   per ``m_sessionUID``. The overall frame identifier does NOT reset on
   flashback, so a regression (new frame < last) indicates packet reordering,
   a replayed datagram, or a corrupted stream. A gap (new frame > last + 1)
   indicates dropped datagrams. Both are reported so the aggregator can
   discard laps that span a flashback.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from typing import Any


class FrameTracker:
    """Track ``m_overallFrameIdentifier`` per ``m_sessionUID``.

    The overall frame identifier is monotonic across flashbacks (unlike
    ``m_frameIdentifier`` which resets). Regression therefore signals
    reordering/replay/corruption; a gap > 1 signals dropped datagrams.
    """

    def __init__(self) -> None:
        self._last: dict[int, int] = {}

    def observe(
        self, session_uid: int, overall_frame: int
    ) -> tuple[bool, bool, int]:
        """Record a frame observation.

        Returns ``(regressed, gap, delta)``:

        - ``regressed`` — True if ``overall_frame < last`` (flashback / reorder).
        - ``gap``       — True if ``overall_frame > last + 1`` (dropped packets).
        - ``delta``     — ``overall_frame - last`` (0 for the first observation
          of a session).
        """
        last = self._last.get(session_uid)
        self._last[session_uid] = overall_frame
        if last is None:
            return False, False, 0
        delta = overall_frame - last
        return delta < 0, delta > 1, delta

    def last(self, session_uid: int) -> int | None:
        """Return the last observed overall frame for a session, or None."""
        return self._last.get(session_uid)

    def reset(self, session_uid: int | None = None) -> None:
        """Forget frame history for one session (or all sessions when None)."""
        if session_uid is None:
            self._last.clear()
        else:
            self._last.pop(session_uid, None)


# --------------------------------------------------------------------------- #
# Sample quality flags (field-level validation closure)
# --------------------------------------------------------------------------- #
class SampleFlag(enum.StrEnum):
    """Severity-ordered sample quality flags (higher = worse).

    Closing the field-level validation loop: rather than silently writing
    anomalous samples, each frame is classified into one of these flags so
    the aggregator can expose a per-lap ``quality_flag`` and decide
    ``clean`` against a configurable threshold.

    - ``OK``                   — sample passed all checks.
    - ``SUSPECT_RANGE``        — a field value was out of its expected range.
    - ``SUSPECT_STALE``        — the sample appears stale (no time advancement).
    - ``SUSPECT_FRAME_REGRESS``— ``m_overallFrameIdentifier`` regressed
      (packet reordering / flashback replay / stream corruption).
    - ``INVALID``              — the sample is unusable; reserved for severe
      structural failures. Not produced by :func:`flag_sample` on range
      violations alone, but honored by the aggregator when set upstream.
    """

    OK = "OK"
    SUSPECT_RANGE = "SUSPECT_RANGE"
    SUSPECT_STALE = "SUSPECT_STALE"
    SUSPECT_FRAME_REGRESS = "SUSPECT_FRAME_REGRESS"
    INVALID = "INVALID"


# Detailed rank for deterministic tie-breaking *within* the same coarse
# severity (used by :func:`merge_flags`). SUSPECT_FRAME_REGRESS ranks above
# SUSPECT_RANGE so a frame that both fails a range check and regresses is
# reported as a regression.
_DETAIL_RANK: dict[SampleFlag, int] = {
    SampleFlag.OK: 0,
    SampleFlag.SUSPECT_STALE: 1,
    SampleFlag.SUSPECT_RANGE: 2,
    SampleFlag.SUSPECT_FRAME_REGRESS: 3,
    SampleFlag.INVALID: 4,
}


def _coerce_flag(flag: str) -> SampleFlag:
    """Coerce a plain string to a :class:`SampleFlag` (unknown → ``OK``)."""
    if isinstance(flag, SampleFlag):
        return flag
    try:
        return SampleFlag(flag)
    except ValueError:
        return SampleFlag.OK


def flag_severity(flag: str) -> int:
    """Return the coarse severity of a flag: ``OK=0``, ``SUSPECT_*=1``, ``INVALID=2``.

    Higher means worse. The generic level name ``"SUSPECT"`` (used as a
    ``LapAggregator.clean_threshold`` value) also maps to 1, so a threshold of
    ``"SUSPECT"`` admits any ``SUSPECT_*`` lap while rejecting ``INVALID``.
    Used to rank laps against ``LapAggregator.clean_threshold``.
    """
    # Generic threshold level "SUSPECT" → severity 1 (any SUSPECT_*).
    if isinstance(flag, str) and flag == "SUSPECT":
        return 1
    f = _coerce_flag(flag)
    if f is SampleFlag.OK:
        return 0
    if f is SampleFlag.INVALID:
        return 2
    return 1  # any SUSPECT_*


def merge_flags(flags: list[str]) -> str:
    """Return the worst (highest-severity) flag from ``flags``.

    Ties within the same coarse severity are broken by :data:`_DETAIL_RANK`
    so that, e.g., a frame that both fails a range check and regresses is
    reported as ``SUSPECT_FRAME_REGRESS``. An empty list returns ``OK``.
    """
    if not flags:
        return SampleFlag.OK
    return max(
        flags,
        key=lambda f: (flag_severity(f), _DETAIL_RANK.get(_coerce_flag(f), 0)),
    )


def flag_sample(
    frame: dict, validation_result: dict, frame_tracker_result: tuple
) -> str:
    """Classify a single frame into the worst applicable :class:`SampleFlag`.

    Logic:

    - If ``validation_result.ok`` is False, the failure reason is classified:
      range violations → ``SUSPECT_RANGE``, stale markers → ``SUSPECT_STALE``,
      regression/reorder markers → ``SUSPECT_FRAME_REGRESS``. All
      :func:`validate_sample` failures are range-type (``out of [lo, hi]`` /
      ``< lo``), so they map to ``SUSPECT_RANGE`` by default.
    - If ``frame_tracker_result`` reports a regression (first element True),
      ``SUSPECT_FRAME_REGRESS`` is added.
    - Otherwise the frame is ``OK``.

    ``frame`` is the parsed packet dict (reserved for future staleness
    detection); ``validation_result`` is the ``{"ok", "reason"}`` dict stamped
    by :func:`validate_sample`; ``frame_tracker_result`` is the
    ``(regressed, gap, delta)`` tuple from :meth:`FrameTracker.observe`.
    """
    candidates: list[str] = [SampleFlag.OK]

    if frame_tracker_result is not None and len(frame_tracker_result) >= 1:
        if frame_tracker_result[0]:  # regressed
            candidates.append(SampleFlag.SUSPECT_FRAME_REGRESS)

    if validation_result is not None and validation_result.get("ok") is False:
        reason = str(validation_result.get("reason") or "").lower()
        if "stale" in reason:
            candidates.append(SampleFlag.SUSPECT_STALE)
        elif "regress" in reason or "reorder" in reason:
            candidates.append(SampleFlag.SUSPECT_FRAME_REGRESS)
        else:
            # validate_sample only emits range-type failures.
            candidates.append(SampleFlag.SUSPECT_RANGE)

    return merge_flags(candidates)


def _check_range(value: float, lo: float, hi: float, name: str) -> str | None:
    if not (lo <= value <= hi):
        return f"{name}={value} out of range [{lo}, {hi}]"
    return None


def _first_non_negative(
    values: Any, name: str, lo: float = 0.0
) -> str | None:
    """Return a reason string for the first value below ``lo``, else None.

    Handles both scalar numeric values and iterables thereof.
    """
    if values is None:
        return None
    if isinstance(values, (int, float)):
        if values < lo:
            return f"{name}={values} < {lo}"
        return None
    for j, v in enumerate(values):
        if v < lo:
            return f"{name}[{j}]={v} < {lo}"
    return None


def validate_sample(
    packet_id: int, parsed: Mapping[str, Any]
) -> tuple[bool, str | None]:
    """Validate key field ranges for a parsed packet.

    Returns ``(ok, reason)``. ``ok`` is True iff no validation failure was
    found; ``reason`` is a short human-readable string explaining the first
    failure encountered, or None when ``ok``.

    Only well-understood per-car fields are checked; under-documented packets
    (TimeTrial, MotionEx) are accepted as-is.
    """
    if packet_id == 6:  # CarTelemetry
        cars = parsed.get("m_carTelemetryData") or []
        for i, c in enumerate(cars):
            for fld in ("m_throttle", "m_brake"):
                v = c.get(fld)
                if v is not None:
                    reason = _check_range(float(v), 0.0, 1.0, f"car[{i}].{fld}")
                    if reason:
                        return False, reason
            v = c.get("m_steer")
            if v is not None:
                reason = _check_range(float(v), -1.0, 1.0, f"car[{i}].m_steer")
                if reason:
                    return False, reason
            reason = _first_non_negative(c.get("m_tyresPressure"), f"car[{i}].m_tyresPressure")
            if reason:
                return False, reason
            for fld in (
                "m_brakesTemperature",
                "m_tyresSurfaceTemperature",
                "m_tyresInnerTemperature",
                "m_engineTemperature",
            ):
                reason = _first_non_negative(c.get(fld), f"car[{i}].{fld}")
                if reason:
                    return False, reason
    elif packet_id == 7:  # CarStatus
        cars = parsed.get("m_carStatusData") or []
        for i, c in enumerate(cars):
            for fld in (
                "m_fuelInTank",
                "m_fuelCapacity",
                "m_fuelRemainingLaps",
                "m_ersStoreEnergy",
                "m_ersHarvestedThisLapMGUK",
                "m_ersHarvestedThisLapMGUH",
                "m_ersDeployedThisLap",
            ):
                v = c.get(fld)
                if v is not None and float(v) < 0:
                    return False, f"car[{i}].{fld}={v} < 0"
    elif packet_id == 10:  # CarDamage
        cars = parsed.get("m_carDamageData") or []
        for i, c in enumerate(cars):
            for fld in ("m_tyresWear", "m_tyresDamage"):
                arr = c.get(fld) or []
                for j, w in enumerate(arr):
                    if w < 0 or w > 100:
                        return False, f"car[{i}].{fld}[{j}]={w} out of [0, 100]"
    elif packet_id == 2:  # LapData
        cars = parsed.get("m_lapData") or []
        for i, c in enumerate(cars):
            for fld in ("m_lastLapTimeInMS", "m_currentLapTimeInMS"):
                v = c.get(fld)
                if v is not None and v < 0:
                    return False, f"car[{i}].{fld}={v} < 0"
    return True, None


__all__ = [
    "FrameTracker",
    "SampleFlag",
    "validate_sample",
    "flag_sample",
    "flag_severity",
    "merge_flags",
]
