"""Tests for :class:`FeedbackMemory` (Iter-142)."""
from __future__ import annotations

from f1opt.feedback.engine import FeedbackMemory, FeedbackMemoryEntry


class TestFeedbackMemory:
    def test_empty_memory(self) -> None:
        fm = FeedbackMemory(max_entries=10)
        assert len(fm) == 0
        assert fm.recent(5) == []
        assert fm.context_for_prompt() == ""

    def test_add_and_recent(self) -> None:
        fm = FeedbackMemory(max_entries=10)
        fm.add("Summary 1", 85.5, "melbourne",
               ["balance", "grip"], {"balance": "understeer"}, ["front_wing: 25 -> 27"])
        fm.add("Summary 2", 85.0, "melbourne",
               ["balance", "grip"], {"balance": "neutral"}, [])
        assert len(fm) == 2
        recent = fm.recent(2)
        assert len(recent) == 2
        assert recent[-1].summary == "Summary 2"

    def test_ring_buffer_overflow(self) -> None:
        fm = FeedbackMemory(max_entries=3)
        for i in range(5):
            fm.add(f"Summary {i}", 85.0, "track",
                   ["balance"], {}, [])
        assert len(fm) == 3
        recent = fm.recent(3)
        assert recent[0].summary == "Summary 2"
        assert recent[-1].summary == "Summary 4"

    def test_context_for_prompt_empty(self) -> None:
        fm = FeedbackMemory()
        assert fm.context_for_prompt() == ""

    def test_context_for_prompt_with_data(self) -> None:
        fm = FeedbackMemory()
        fm.add("Summary 1", 85.5, "melbourne",
               ["balance", "grip", "tyres", "braking", "ers_drs",
                "throttle_brake_smoothness", "confidence",
                "lap_time_potential", "sector_compare", "setup_advice"],
               {"balance": "understeer", "braking": "lockup_risk"},
               ["front_wing: 25 -> 27", "brake_pressure: 95 -> 93"])
        fm.add("Summary 2", 85.1, "melbourne",
               ["balance", "grip", "tyres", "braking", "ers_drs",
                "throttle_brake_smoothness", "confidence",
                "lap_time_potential", "sector_compare", "setup_advice"],
               {"balance": "neutral", "grip": "low"},
               ["rear_wing: 27 -> 28"])
        ctx = fm.context_for_prompt()
        assert "Recent Feedback History" in ctx
        assert "85.5" in ctx
        assert "improving" in ctx
        assert "front_wing" in ctx
        assert "understeer" in ctx
        assert "low" in ctx

    def test_context_for_prompt_slowing_trend(self) -> None:
        fm = FeedbackMemory()
        fm.add("S1", 85.0, "track", ["balance"], {}, [])
        fm.add("S2", 85.5, "track", ["balance"], {}, [])
        ctx = fm.context_for_prompt()
        assert "slowing" in ctx

    def test_reset_clears(self) -> None:
        fm = FeedbackMemory()
        fm.add("S", 85.0, "track", ["balance"], {}, [])
        assert len(fm) == 1
        fm.reset()
        assert len(fm) == 0

    def test_negative_max_entries_rejected(self) -> None:
        import pytest
        with pytest.raises(ValueError):
            FeedbackMemory(max_entries=0)
        with pytest.raises(ValueError):
            FeedbackMemory(max_entries=-1)

    def test_recent_zero_returns_empty(self) -> None:
        fm = FeedbackMemory()
        fm.add("S", 85.0, "track", ["balance"], {}, [])
        assert fm.recent(0) == []

    def test_entry_fields(self) -> None:
        fm = FeedbackMemory()
        e = fm.add("Summary", 85.5, "monza",
                    ["balance", "grip"], {"balance": "understeer"},
                    ["fw: 25 -> 27"])
        assert isinstance(e, FeedbackMemoryEntry)
        assert e.summary == "Summary"
        assert e.lap_time == 85.5
        assert e.track_id == "monza"
        assert e.dimension_names == ["balance", "grip"]
        assert e.dimension_values == {"balance": "understeer"}
        assert e.setup_changes == ["fw: 25 -> 27"]
        assert e.timestamp > 0
