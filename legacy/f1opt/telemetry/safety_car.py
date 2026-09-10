"""Safety Car / Virtual Safety Car scenario detection (Iter-135).

Consumes parsed F1 25 Session packets and tracks Safety Car (SC) / Virtual
Safety Car (VSC) state transitions over time. Produces a structured event
log with start/end timestamps, duration, and type, plus convenience queries
for the current state and active-window filtering.

The Session packet's ``m_safetyCarStatus`` byte encodes (per F1 24/25 spec):

* 0 — no safety car (green / racing)
* 1 — full safety car deployed
* 2 — virtual safety car deployed

This module is defensive: out-of-order arrival, missing fields, and unknown
status codes are handled gracefully (unknown codes are treated as "no SC"
but logged as ``raw_status`` in the event for post-hoc inspection).

Public API:

* :class:`SafetyCarTracker` — stateful tracker; call
  :meth:`on_session_packet` for each parsed Session packet, then query
  :meth:`current_state` / :meth:`events` / :meth:`is_active` /
  :meth:`active_type` / :meth:`frames_during_safety_car`.
* :data:`SC_STATUS_NONE` / :data:`SC_STATUS_FULL` / :data:`SC_STATUS_VIRTUAL`
  — status byte constants.
* :data:`SC_TYPE_FULL` / :data:`SC_TYPE_VIRTUAL` — event type labels.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any

# Status byte values (per F1 24/25 m_safetyCarStatus spec).
SC_STATUS_NONE = 0
SC_STATUS_FULL = 1
SC_STATUS_VIRTUAL = 2

#: Human-readable event type labels (used in ``event["type"]``).
SC_TYPE_FULL = "safety_car"
SC_TYPE_VIRTUAL = "virtual_safety_car"

#: Map status byte -> event type label (unknown codes excluded).
_STATUS_TO_TYPE: dict[int, str] = {
    SC_STATUS_FULL: SC_TYPE_FULL,
    SC_STATUS_VIRTUAL: SC_TYPE_VIRTUAL,
}


def _status_label(status: int) -> str:
    """Return a human-readable label for a status byte (defensive)."""
    if status == SC_STATUS_NONE:
        return "none"
    if status == SC_STATUS_FULL:
        return SC_TYPE_FULL
    if status == SC_STATUS_VIRTUAL:
        return SC_TYPE_VIRTUAL
    return f"unknown({status})"


class SafetyCarTracker:
    """Stateful Safety Car / Virtual Safety Car scenario tracker.

    Maintains the current SC/VSC status, a bounded event log of transitions
    (start -> end with duration), and supports filtering unified telemetry
    frames to those that fall within an SC/VSC activation window.

    Out-of-order arrival: packets are accepted in any order; the tracker
    only advances state when ``session_time`` moves forward. A transition
    is recorded when the status byte changes between two *time-ordered*
    observations. Late-arriving packets with ``session_time`` earlier than
    the last seen are dropped (defensive — preserves monotonic state).
    """

    def __init__(self, max_events: int = 200) -> None:
        if max_events < 1:
            raise ValueError("max_events must be >= 1")
        self._max_events = max_events
        #: Current status byte (0 = none).
        self._status: int = SC_STATUS_NONE
        #: Last seen session_time (seconds). -1 = no packet seen yet.
        self._last_t: float = -1.0
        #: Currently-open event (None when status == none).
        self._open_event: dict[str, Any] | None = None
        #: Bounded event log (most recent first is *not* guaranteed; we
        #: append in time order and trim the oldest when over capacity).
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)

    # ------------------------------------------------------------------ #
    # Ingest
    # ------------------------------------------------------------------ #
    def on_session_packet(self, header: Any, parsed: dict[str, Any]) -> None:
        """Ingest a parsed Session packet.

        Extracts ``m_safetyCarStatus`` and ``header.session_time`` and
        advances the tracker state. Late / out-of-order packets (session_time
        < last seen) are silently dropped to preserve monotonic state.

        Args:
            header: Packet header (any object with ``session_time`` attribute
                or a dict with ``"session_time"`` key).
            parsed: Parsed Session body dict (must contain
                ``"m_safetyCarStatus"``).
        """
        t = self._extract_session_time(header)
        if t is None:
            return
        # Defensive: drop late/out-of-order packets.
        if self._last_t >= 0.0 and t < self._last_t:
            return
        status = int(parsed.get("m_safetyCarStatus", SC_STATUS_NONE))
        # Clamp to known range; unknown codes are normalised to NONE but the
        # raw value is recorded in any event that opens.
        if status not in _STATUS_TO_TYPE:
            status = SC_STATUS_NONE
        self._advance(t, status)

    def _advance(self, t: float, status: int) -> None:
        """Advance state to time ``t`` with status ``status`` (monotonic)."""
        prev_status = self._status
        self._last_t = t
        if status == prev_status:
            return  # no transition
        # Transition detected.
        if status == SC_STATUS_NONE:
            # SC/VSC ending -> close open event.
            if self._open_event is not None:
                self._open_event["end_time"] = t
                self._open_event["end_status"] = SC_STATUS_NONE
                self._open_event["duration_s"] = t - float(
                    self._open_event["start_time"]
                )
                self._events.append(dict(self._open_event))
                self._open_event = None
        else:
            # SC/VSC starting. If a different SC type was already open
            # (e.g. VSC -> full SC), close the prior event first.
            if self._open_event is not None:
                self._open_event["end_time"] = t
                self._open_event["end_status"] = status
                self._open_event["duration_s"] = t - float(
                    self._open_event["start_time"]
                )
                self._events.append(dict(self._open_event))
                self._open_event = None
            self._open_event = {
                "type": _STATUS_TO_TYPE[status],
                "start_time": t,
                "start_status": status,
                "end_time": None,
                "end_status": None,
                "duration_s": None,
            }
        self._status = status

    @staticmethod
    def _extract_session_time(header: Any) -> float | None:
        """Extract ``session_time`` (float seconds) from a header object."""
        if hasattr(header, "session_time"):
            try:
                return float(header.session_time)
            except (TypeError, ValueError):
                return None
        if isinstance(header, dict):
            v = header.get("session_time")
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        return None

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def current_state(self) -> dict[str, Any]:
        """Return the current SC/VSC state as a dict.

        Keys: ``status`` (int byte), ``label`` (human-readable str),
        ``is_active`` (bool), ``active_type`` (str | None), ``last_time``
        (float, -1 if no packets seen).
        """
        return {
            "status": self._status,
            "label": _status_label(self._status),
            "is_active": self._status != SC_STATUS_NONE,
            "active_type": _STATUS_TO_TYPE.get(self._status),
            "last_time": self._last_t,
        }

    def is_active(self) -> bool:
        """Return True if an SC or VSC is currently active."""
        return self._status != SC_STATUS_NONE

    def active_type(self) -> str | None:
        """Return the active event type (``safety_car`` / ``virtual_safety_car``).

        Returns ``None`` when no SC/VSC is active.
        """
        return _STATUS_TO_TYPE.get(self._status)

    def events(self) -> list[dict[str, Any]]:
        """Return a list of completed SC/VSC events (time-ordered, oldest first).

        Each event dict has keys: ``type``, ``start_time``, ``start_status``,
        ``end_time``, ``end_status``, ``duration_s``. The currently-open
        event (if any) is NOT included; see :meth:`open_event`.
        """
        return [dict(e) for e in self._events]

    def open_event(self) -> dict[str, Any] | None:
        """Return the currently-open event (a copy), or None."""
        if self._open_event is None:
            return None
        return dict(self._open_event)

    def event_count(self) -> int:
        """Return the number of *completed* events in the log."""
        return len(self._events)

    def total_safety_car_time(self) -> float:
        """Return total elapsed time (seconds) across all completed SC/VSC events."""
        return sum(
            float(e.get("duration_s") or 0.0) for e in self._events
        )

    def frames_during_safety_car(
        self,
        frames: Iterable[dict[str, Any]],
        *,
        include_open: bool = True,
    ) -> list[dict[str, Any]]:
        """Filter ``frames`` to those falling within an SC/VSC activation window.

        Each frame must have a ``session_time`` key (float seconds). A frame
        is kept if its ``session_time`` falls within any completed event's
        ``[start_time, end_time]`` window, or — when ``include_open`` is True
        — within the currently-open event's ``[start_time, +inf)`` window.

        Args:
            frames: Iterable of unified frame dicts (e.g. from
                :meth:`TelemetryAligner.sample_60hz`).
            include_open: When True (default), also keep frames inside the
                currently-open (not-yet-closed) event window.

        Returns:
            List of frames (in input order) that fall within an SC/VSC window.
        """
        windows: list[tuple[float, float | None]] = [
            (float(e["start_time"]), float(e["end_time"]))
            for e in self._events
            if e.get("end_time") is not None
        ]
        if include_open and self._open_event is not None:
            windows.append((float(self._open_event["start_time"]), None))
        out: list[dict[str, Any]] = []
        for f in frames:
            t = f.get("session_time")
            if t is None:
                continue
            try:
                tf = float(t)
            except (TypeError, ValueError):
                continue
            for start, end in windows:
                if end is None:
                    if tf >= start:
                        out.append(f)
                        break
                elif start <= tf <= end:
                    out.append(f)
                    break
        return out

    def reset(self) -> None:
        """Clear all state (for testing / session restart)."""
        self._status = SC_STATUS_NONE
        self._last_t = -1.0
        self._open_event = None
        self._events.clear()
