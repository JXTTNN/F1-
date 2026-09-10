"""Unit tests for :mod:`f1opt.telemetry.safety_car` (Iter-135).

Feeds crafted ``(header, parsed)`` pairs directly to
:class:`SafetyCarTracker` (no UDP socket needed). Verifies:
- Initial state is empty / inactive.
- Single SC and VSC events produce one completed event each.
- Type switch (full -> virtual) closes the prior event and opens a new one.
- Out-of-order packets are dropped (state unchanged).
- Unknown status codes are normalised to NONE.
- ``frames_during_safety_car`` filters frames within SC/VSC windows and
  honours ``include_open``.
- ``total_safety_car_time`` sums completed durations only.
- ``reset()`` clears all state.
- ``max_events`` deque trimming drops the oldest event.
- Both object headers and dict headers are accepted.
- Repeated same-status packets do not create events.
- A missing ``m_safetyCarStatus`` key defaults to NONE.
"""

from __future__ import annotations

import types

from f1opt.telemetry.safety_car import (
    SC_STATUS_FULL,
    SC_STATUS_NONE,
    SC_STATUS_VIRTUAL,
    SC_TYPE_FULL,
    SC_TYPE_VIRTUAL,
    SafetyCarTracker,
)


def _hdr(t: float) -> types.SimpleNamespace:
    """Build a fake header with a ``session_time`` attribute."""
    return types.SimpleNamespace(session_time=t)


def _parsed(status: int) -> dict[str, int]:
    """Build a fake parsed Session payload."""
    return {"m_safetyCarStatus": status}


class TestSafetyCarTracker:
    def test_initial_state(self) -> None:
        tr = SafetyCarTracker()
        s = tr.current_state()
        assert s["status"] == SC_STATUS_NONE
        assert s["label"] == "none"
        assert s["is_active"] is False
        assert s["active_type"] is None
        assert s["last_time"] == -1.0
        assert tr.is_active() is False
        assert tr.active_type() is None
        assert tr.events() == []
        assert tr.open_event() is None
        assert tr.event_count() == 0
        assert tr.total_safety_car_time() == 0.0

    def test_single_sc_event(self) -> None:
        tr = SafetyCarTracker()
        tr.on_session_packet(_hdr(10.0), _parsed(SC_STATUS_FULL))
        # Open event mid-flight.
        assert tr.is_active() is True
        assert tr.active_type() == SC_TYPE_FULL
        oe = tr.open_event()
        assert oe is not None
        assert oe["type"] == SC_TYPE_FULL
        assert oe["start_time"] == 10.0
        assert oe["start_status"] == SC_STATUS_FULL
        assert oe["end_time"] is None
        assert oe["end_status"] is None
        assert oe["duration_s"] is None
        assert tr.event_count() == 0
        # Close.
        tr.on_session_packet(_hdr(20.0), _parsed(SC_STATUS_NONE))
        assert tr.is_active() is False
        assert tr.open_event() is None
        assert tr.event_count() == 1
        e = tr.events()[0]
        assert e["type"] == SC_TYPE_FULL
        assert e["start_time"] == 10.0
        assert e["start_status"] == SC_STATUS_FULL
        assert e["end_time"] == 20.0
        assert e["end_status"] == SC_STATUS_NONE
        assert e["duration_s"] == 10.0

    def test_single_vsc_event(self) -> None:
        tr = SafetyCarTracker()
        tr.on_session_packet(_hdr(5.0), _parsed(SC_STATUS_VIRTUAL))
        assert tr.active_type() == SC_TYPE_VIRTUAL
        tr.on_session_packet(_hdr(12.5), _parsed(SC_STATUS_NONE))
        assert tr.event_count() == 1
        e = tr.events()[0]
        assert e["type"] == SC_TYPE_VIRTUAL
        assert e["start_time"] == 5.0
        assert e["end_time"] == 12.5
        assert e["start_status"] == SC_STATUS_VIRTUAL
        assert e["end_status"] == SC_STATUS_NONE
        assert e["duration_s"] == 7.5

    def test_type_switch_full_to_virtual(self) -> None:
        tr = SafetyCarTracker()
        tr.on_session_packet(_hdr(10.0), _parsed(SC_STATUS_FULL))
        tr.on_session_packet(_hdr(15.0), _parsed(SC_STATUS_VIRTUAL))
        # Prior full event closed.
        assert tr.event_count() == 1
        e0 = tr.events()[0]
        assert e0["type"] == SC_TYPE_FULL
        assert e0["start_time"] == 10.0
        assert e0["end_time"] == 15.0
        assert e0["end_status"] == SC_STATUS_VIRTUAL
        assert e0["duration_s"] == 5.0
        # New virtual event open.
        assert tr.active_type() == SC_TYPE_VIRTUAL
        oe = tr.open_event()
        assert oe is not None
        assert oe["type"] == SC_TYPE_VIRTUAL
        assert oe["start_time"] == 15.0
        # Close virtual.
        tr.on_session_packet(_hdr(25.0), _parsed(SC_STATUS_NONE))
        assert tr.event_count() == 2
        e1 = tr.events()[1]
        assert e1["type"] == SC_TYPE_VIRTUAL
        assert e1["start_time"] == 15.0
        assert e1["end_time"] == 25.0
        assert e1["duration_s"] == 10.0

    def test_out_of_order_packet_dropped(self) -> None:
        tr = SafetyCarTracker()
        tr.on_session_packet(_hdr(10.0), _parsed(SC_STATUS_FULL))
        before = tr.current_state()
        # Late packet would close the event if accepted.
        tr.on_session_packet(_hdr(5.0), _parsed(SC_STATUS_NONE))
        after = tr.current_state()
        assert after["status"] == before["status"]
        assert after["last_time"] == before["last_time"]
        assert tr.is_active() is True
        assert tr.event_count() == 0
        assert tr.open_event() is not None

    def test_unknown_status_normalised_to_none(self) -> None:
        tr = SafetyCarTracker()
        tr.on_session_packet(_hdr(10.0), _parsed(SC_STATUS_FULL))
        tr.on_session_packet(_hdr(18.0), _parsed(3))  # unknown -> NONE
        assert tr.is_active() is False
        assert tr.event_count() == 1
        e = tr.events()[0]
        assert e["end_status"] == SC_STATUS_NONE
        assert e["end_time"] == 18.0
        assert e["duration_s"] == 8.0

    def test_frames_during_safety_car_filters(self) -> None:
        tr = SafetyCarTracker()
        tr.on_session_packet(_hdr(10.0), _parsed(SC_STATUS_FULL))
        tr.on_session_packet(_hdr(20.0), _parsed(SC_STATUS_NONE))
        frames = [
            {"session_time": 5.0, "id": "a"},
            {"session_time": 10.0, "id": "b"},
            {"session_time": 15.0, "id": "c"},
            {"session_time": 20.0, "id": "d"},
            {"session_time": 25.0, "id": "e"},
        ]
        out = tr.frames_during_safety_car(frames)
        assert [f["id"] for f in out] == ["b", "c", "d"]

    def test_frames_include_open_false_excludes_open(self) -> None:
        tr = SafetyCarTracker()
        tr.on_session_packet(_hdr(10.0), _parsed(SC_STATUS_FULL))
        frames = [
            {"session_time": 5.0, "id": "a"},
            {"session_time": 12.0, "id": "b"},
            {"session_time": 15.0, "id": "c"},
        ]
        assert [f["id"] for f in tr.frames_during_safety_car(frames)] == ["b", "c"]
        assert tr.frames_during_safety_car(frames, include_open=False) == []

    def test_total_safety_car_time_sums_completed(self) -> None:
        tr = SafetyCarTracker()
        tr.on_session_packet(_hdr(10.0), _parsed(SC_STATUS_FULL))
        tr.on_session_packet(_hdr(20.0), _parsed(SC_STATUS_NONE))
        tr.on_session_packet(_hdr(30.0), _parsed(SC_STATUS_VIRTUAL))
        tr.on_session_packet(_hdr(37.5), _parsed(SC_STATUS_NONE))
        assert tr.event_count() == 2
        assert tr.total_safety_car_time() == 17.5
        # Open event does not contribute.
        tr.on_session_packet(_hdr(50.0), _parsed(SC_STATUS_FULL))
        assert tr.total_safety_car_time() == 17.5

    def test_reset_clears_state(self) -> None:
        tr = SafetyCarTracker()
        tr.on_session_packet(_hdr(10.0), _parsed(SC_STATUS_FULL))
        tr.on_session_packet(_hdr(20.0), _parsed(SC_STATUS_NONE))
        tr.on_session_packet(_hdr(30.0), _parsed(SC_STATUS_VIRTUAL))
        assert tr.event_count() == 1
        assert tr.is_active() is True
        tr.reset()
        s = tr.current_state()
        assert s["status"] == SC_STATUS_NONE
        assert s["last_time"] == -1.0
        assert tr.is_active() is False
        assert tr.events() == []
        assert tr.open_event() is None
        assert tr.event_count() == 0
        assert tr.total_safety_car_time() == 0.0

    def test_max_events_deque_trimming(self) -> None:
        tr = SafetyCarTracker(max_events=2)
        for start, end in [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]:
            tr.on_session_packet(_hdr(start), _parsed(SC_STATUS_FULL))
            tr.on_session_packet(_hdr(end), _parsed(SC_STATUS_NONE))
        assert tr.event_count() == 2
        evs = tr.events()
        assert evs[0]["start_time"] == 30.0  # oldest (10..20) trimmed
        assert evs[1]["start_time"] == 50.0
        assert tr.total_safety_car_time() == 20.0

    def test_object_header_works(self) -> None:
        tr = SafetyCarTracker()
        tr.on_session_packet(
            types.SimpleNamespace(session_time=10.0), _parsed(SC_STATUS_FULL)
        )
        tr.on_session_packet(
            types.SimpleNamespace(session_time=20.0), _parsed(SC_STATUS_NONE)
        )
        assert tr.event_count() == 1
        assert tr.events()[0]["duration_s"] == 10.0

    def test_dict_header_works(self) -> None:
        tr = SafetyCarTracker()
        tr.on_session_packet({"session_time": 10.0}, _parsed(SC_STATUS_VIRTUAL))
        tr.on_session_packet({"session_time": 25.0}, _parsed(SC_STATUS_NONE))
        assert tr.event_count() == 1
        e = tr.events()[0]
        assert e["type"] == SC_TYPE_VIRTUAL
        assert e["duration_s"] == 15.0

    def test_no_transition_no_event(self) -> None:
        # Repeated same-status packets do not create or close events.
        tr = SafetyCarTracker()
        tr.on_session_packet(_hdr(10.0), _parsed(SC_STATUS_FULL))
        tr.on_session_packet(_hdr(15.0), _parsed(SC_STATUS_FULL))  # same status
        assert tr.event_count() == 0
        assert tr.is_active() is True
        oe = tr.open_event()
        assert oe is not None
        assert oe["start_time"] == 10.0  # unchanged

    def test_missing_safety_car_status_defaults_none(self) -> None:
        tr = SafetyCarTracker()
        tr.on_session_packet(_hdr(10.0), _parsed(SC_STATUS_FULL))
        # Missing key -> defaults to 0 (NONE) -> closes the open SC event.
        tr.on_session_packet(_hdr(20.0), {})
        assert tr.event_count() == 1
        assert tr.is_active() is False
        assert tr.events()[0]["end_status"] == SC_STATUS_NONE

    def test_events_returns_copies(self) -> None:
        # Mutating a returned event must not corrupt internal state.
        tr = SafetyCarTracker()
        tr.on_session_packet(_hdr(10.0), _parsed(SC_STATUS_FULL))
        tr.on_session_packet(_hdr(20.0), _parsed(SC_STATUS_NONE))
        e = tr.events()[0]
        e["duration_s"] = 999.0
        assert tr.events()[0]["duration_s"] == 10.0
