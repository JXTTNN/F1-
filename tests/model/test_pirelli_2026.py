"""Tests for pirelli_2026 (Iter-28)."""

from __future__ import annotations

import pytest

from f1opt.data.tracks import ALL_TRACKS
from f1opt.model.pirelli_2026 import (
    Pirelli2026Range,
    all_pirelli_compounds,
    pirelli_compound,
    tire_compound_for_track,
)


class TestPirelliRange:
    def test_returns_8_compounds(self) -> None:
        all_c = all_pirelli_compounds()
        assert len(all_c) == 8
        for code in ("C0", "C1", "C2", "C3", "C4", "C5", "intermediate", "wet"):
            assert code in all_c

    def test_compound_lookup(self) -> None:
        c3 = pirelli_compound("C3")
        assert c3.code == "C3"
        assert c3.grip_factor == 1.00  # baseline

    def test_unknown_compound_raises(self) -> None:
        with pytest.raises(ValueError):
            pirelli_compound("C99")


class TestCompoundPhysics:
    def test_c0_hardest_lowest_grip(self) -> None:
        c0 = pirelli_compound("C0")
        c5 = pirelli_compound("C5")
        assert c0.grip_factor < c5.grip_factor

    def test_c5_softest_lowest_wear_threshold(self) -> None:
        """C5 softest → earliest cliff."""
        c5 = pirelli_compound("C5")
        c0 = pirelli_compound("C0")
        assert c5.cliff_threshold_pct < c0.cliff_threshold_pct

    def test_c5_fastest_warmup(self) -> None:
        c5 = pirelli_compound("C5")
        c0 = pirelli_compound("C0")
        assert c5.warmup_laps < c0.warmup_laps

    def test_wet_lowest_temp_optimal(self) -> None:
        wet = pirelli_compound("wet")
        c3 = pirelli_compound("C3")
        assert wet.temp_optimal_c < c3.temp_optimal_c


class TestTrackSelection:
    @pytest.mark.parametrize("track", ALL_TRACKS)
    def test_every_track_has_tire_selection(self, track) -> None:
        r = tire_compound_for_track(track.track_id)
        assert isinstance(r, Pirelli2026Range)
        assert r.track_id == track.track_id

    def test_monaco_uses_softest_range(self) -> None:
        r = tire_compound_for_track("monaco")
        assert r.soft_code == "C5"
        assert r.medium_code == "C4"
        assert r.hard_code == "C3"

    def test_monza_uses_hardest_range(self) -> None:
        r = tire_compound_for_track("monza")
        assert r.soft_code == "C2"
        assert r.medium_code == "C1"
        assert r.hard_code == "C0"

    def test_color_lookup(self) -> None:
        r = tire_compound_for_track("monza")
        soft = r.compound_for_color("soft")
        medium = r.compound_for_color("medium")
        hard = r.compound_for_color("hard")
        # soft > medium > hard grip
        assert soft.grip_factor > medium.grip_factor > hard.grip_factor

    def test_intermediate_wet_always_available(self) -> None:
        r = tire_compound_for_track("monza")
        inter = r.compound_for_color("intermediate")
        wet = r.compound_for_color("wet")
        assert inter.code == "intermediate"
        assert wet.code == "wet"

    def test_unknown_color_raises(self) -> None:
        r = tire_compound_for_track("monza")
        with pytest.raises(ValueError):
            r.compound_for_color("purple")


class TestColorHierarchy:
    """soft > medium > hard in grip; reverse in durability."""

    @pytest.mark.parametrize("track_id", ["monza", "monaco", "silverstone", "spa"])
    def test_color_hierarchy(self, track_id: str) -> None:
        r = tire_compound_for_track(track_id)
        soft = r.compound_for_color("soft")
        medium = r.compound_for_color("medium")
        hard = r.compound_for_color("hard")
        # Grip: soft > medium > hard
        assert soft.grip_factor > medium.grip_factor > hard.grip_factor
        # Cliff threshold: hard > medium > soft (hard more durable)
        assert hard.cliff_threshold_pct > medium.cliff_threshold_pct > soft.cliff_threshold_pct
