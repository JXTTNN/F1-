"""Tests for the multi-layer cache infrastructure."""

from __future__ import annotations

import pytest

from f1opt.model.cache import PrecomputedLookups, SetupCache, WarmupCache


# --------------------------------------------------------------------------- #
# SetupCache
# --------------------------------------------------------------------------- #
class TestSetupCache:
    def test_get_or_compute_caches_result(self) -> None:
        c = SetupCache(maxsize=10)
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return "value"

        r1 = c.get_or_compute(("key1",), compute)
        r2 = c.get_or_compute(("key1",), compute)
        assert r1 == r2 == "value"
        assert call_count == 1  # compute only once

    def test_stats_shows_hits_and_misses(self) -> None:
        c = SetupCache(maxsize=10)
        c.get_or_compute(("k",), lambda: 1)
        c.get_or_compute(("k",), lambda: 1)  # hit
        s = c.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1

    def test_hit_rate_computed(self) -> None:
        c = SetupCache(maxsize=10)
        c.get_or_compute(("k",), lambda: 1)  # miss
        c.get_or_compute(("k",), lambda: 1)  # hit
        s = c.stats()
        assert s["hit_rate"] == pytest.approx(0.5)

    def test_invalidate_removes_entry(self) -> None:
        c = SetupCache(maxsize=10)
        c.get_or_compute(("k",), lambda: 1)
        c.invalidate(("k",))
        assert ("k",) not in c.keys()

    def test_clear_empties(self) -> None:
        c = SetupCache(maxsize=10)
        c.get_or_compute(("k",), lambda: 1)
        c.clear()
        assert c.keys() == []
        assert c.stats()["hits"] == 0

    def test_maxsize_eviction(self) -> None:
        c = SetupCache(maxsize=2)
        c.get_or_compute(("k1",), lambda: 1)
        c.get_or_compute(("k2",), lambda: 2)
        c.get_or_compute(("k3",), lambda: 3)  # evicts k1
        keys = c.keys()
        assert ("k1",) not in keys
        assert ("k3",) in keys
        assert c.stats()["evictions"] >= 1

    def test_keys_returns_list(self) -> None:
        c = SetupCache(maxsize=10)
        c.get_or_compute(("a",), lambda: 1)
        c.get_or_compute(("b",), lambda: 2)
        ks = c.keys()
        assert ("a",) in ks
        assert ("b",) in ks

    def test_lru_order_updates_on_access(self) -> None:
        """Accessing k1 should make it most-recently-used (not evicted next)."""
        c = SetupCache(maxsize=2)
        c.get_or_compute(("k1",), lambda: 1)
        c.get_or_compute(("k2",), lambda: 2)
        c.get_or_compute(("k1",), lambda: 1)  # access k1 → MRU
        c.get_or_compute(("k3",), lambda: 3)  # evicts k2 (LRU)
        assert ("k1",) in c.keys()
        assert ("k2",) not in c.keys()

    def test_maxsize_one(self) -> None:
        c = SetupCache(maxsize=1)
        c.get_or_compute(("k1",), lambda: 1)
        c.get_or_compute(("k2",), lambda: 2)
        assert c.keys() == [("k2",)]

    def test_invalid_maxsize_raises(self) -> None:
        with pytest.raises(ValueError):
            SetupCache(maxsize=0)

    def test_determinism(self) -> None:
        c1 = SetupCache(maxsize=10)
        c2 = SetupCache(maxsize=10)
        assert c1.get_or_compute(("k",), lambda: 42) == c2.get_or_compute(("k",), lambda: 42)


# --------------------------------------------------------------------------- #
# PrecomputedLookups
# --------------------------------------------------------------------------- #
class TestPrecomputedLookups:
    def test_track_reference_lap_times_returns_24_tracks(self) -> None:
        pl = PrecomputedLookups()
        refs = pl.track_reference_lap_times()
        assert len(refs) == 24
        assert all(isinstance(v, float) for v in refs.values())

    def test_track_reference_lap_times_cached(self) -> None:
        pl = PrecomputedLookups()
        r1 = pl.track_reference_lap_times()
        r2 = pl.track_reference_lap_times()
        assert r1 is r2  # same object (cached)

    def test_setup_field_ranges_returns_dict(self) -> None:
        pl = PrecomputedLookups()
        ranges = pl.setup_field_ranges()
        assert len(ranges) > 0
        # Each entry has min/max/step/Unit.
        for _name, r in ranges.items():
            assert "min" in r and "max" in r

    def test_compound_params_returns_dict(self) -> None:
        pl = PrecomputedLookups()
        cp = pl.compound_params()
        assert "soft" in cp
        assert "mu_peak" in cp["soft"]

    def test_invalidate_all_clears(self) -> None:
        pl = PrecomputedLookups()
        pl.track_reference_lap_times()
        pl.invalidate_all()
        assert pl._track_refs is None


# --------------------------------------------------------------------------- #
# WarmupCache
# --------------------------------------------------------------------------- #
class TestWarmupCache:
    def test_warmup_surrogate_returns_dict(self) -> None:
        w = WarmupCache()
        result = w.warmup_surrogate(track_ids=["melbourne"])
        assert "tracks_warmed" in result
        assert "time_s" in result
        assert "cache_hits_after" in result
        assert result["tracks_warmed"] >= 0

    def test_warmup_all_returns_summary(self) -> None:
        w = WarmupCache()
        result = w.warmup_all()
        assert "surrogate" in result
        assert result["lookups_loaded"] is True
        assert result["track_refs_count"] == 24
        assert result["compounds_count"] >= 1

    def test_warmup_surrogate_unknown_track_skipped(self) -> None:
        w = WarmupCache()
        result = w.warmup_surrogate(track_ids=["nonexistent_track"])
        # Should not crash; surrogate may still produce a fallback prediction.
        assert "tracks_warmed" in result
        assert result["tracks_warmed"] >= 0
