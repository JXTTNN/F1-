"""Multi-layer caching infrastructure for the F1 setup optimizer.

Provides a bounded LRU cache, precomputed lookup tables, and a warmup helper
that pre-populates the surrogate's prediction cache for fast cold-start
behaviour.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from f1opt.observability.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "SetupCache",
    "PrecomputedLookups",
    "WarmupCache",
]


# --------------------------------------------------------------------------- #
# SetupCache — bounded LRU
# --------------------------------------------------------------------------- #
class SetupCache:
    """Bounded LRU cache keyed by an arbitrary hashable tuple."""

    def __init__(self, maxsize: int = 1000) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = int(maxsize)
        self._store: OrderedDict[Any, Any] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get_or_compute(self, key: Any, compute_fn: Callable[[], Any]) -> Any:
        """Return cached value for ``key`` or compute + store it."""
        if key in self._store:
            self._store.move_to_end(key)
            self._hits += 1
            return self._store[key]
        # Miss: compute + insert.
        value = compute_fn()
        self._misses += 1
        self._store[key] = value
        # Evict oldest if over capacity.
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)
            self._evictions += 1
        return value

    def invalidate(self, key: Any) -> None:
        if key in self._store:
            del self._store[key]

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (self._hits / total) if total > 0 else 0.0,
            "size": len(self._store),
            "maxsize": self.maxsize,
            "evictions": self._evictions,
        }

    def keys(self) -> list[Any]:
        return list(self._store.keys())


# --------------------------------------------------------------------------- #
# PrecomputedLookups
# --------------------------------------------------------------------------- #
class PrecomputedLookups:
    """Precompute expensive lookup tables (lazy, cached)."""

    def __init__(self) -> None:
        self._track_refs: dict[str, float] | None = None
        self._setup_ranges: dict[str, Any] | None = None
        self._compound_params: dict[str, Any] | None = None

    def track_reference_lap_times(self) -> dict[str, float]:
        """Per-track reference lap times, computed once."""
        if self._track_refs is not None:
            return self._track_refs
        from f1opt.data.tracks import ALL_TRACKS

        refs: dict[str, float] = {}
        # Heuristic reference: ~ 36 m/s avg * (length/1000) km → seconds.
        # Tuned so a 5 km track ≈ 95 s (typical F1 medium-length circuit).
        for t in ALL_TRACKS:
            refs[t.track_id] = (t.length_m / 1000.0) * 19.0
        self._track_refs = refs
        return refs

    def setup_field_ranges(self) -> dict[str, Any]:
        """Cached setup field ranges from the schema."""
        if self._setup_ranges is not None:
            return self._setup_ranges
        from f1opt.data.setup_schema import _FIELD_DEFS

        ranges: dict[str, Any] = {}
        for fd in _FIELD_DEFS:
            # _FIELD_DEFS tuple: (name, group, type, min, max, step, unit, desc)
            name = fd[0]
            ranges[name] = {"min": fd[3], "max": fd[4], "step": fd[5], "unit": fd[6]}
        self._setup_ranges = ranges
        return ranges

    def compound_params(self) -> dict[str, Any]:
        """Cached tire compound parameters."""
        if self._compound_params is not None:
            return self._compound_params
        try:
            from f1opt.model.tire_dynamics import COMPOUND_PARAMS

            params = {
                name: {
                    "name": p.name,
                    "mu_peak": p.mu_peak,
                    "b_long": p.b_long,
                    "b_lat": p.b_lat,
                    "peak_temp_c": p.peak_temp_c,
                    "thermal_window_c": p.thermal_window_c,
                }
                for name, p in COMPOUND_PARAMS.items()
            }
        except ImportError:
            params = {}
        self._compound_params = params
        return params

    def invalidate_all(self) -> None:
        self._track_refs = None
        self._setup_ranges = None
        self._compound_params = None


# --------------------------------------------------------------------------- #
# WarmupCache
# --------------------------------------------------------------------------- #
class WarmupCache:
    """Pre-warm caches at startup for fast cold-start behaviour."""

    def __init__(self) -> None:
        self._lookups = PrecomputedLookups()

    def warmup_surrogate(
        self, track_ids: list[str] | None = None
    ) -> dict[str, Any]:
        """Run predict_lap_time for DEFAULT_SETUP on each track, populating cache."""
        from f1opt.data.setup_schema import DEFAULT_SETUP

        if track_ids is None:
            from f1opt.data.tracks import ALL_TRACKS

            track_ids = [t.track_id for t in ALL_TRACKS]

        start = time.perf_counter()
        warmed = 0
        for tid in track_ids:
            try:
                from f1opt.model.surrogate import predict_lap_time

                predict_lap_time(DEFAULT_SETUP, tid, None)
                warmed += 1
            except Exception:
                # Skip tracks that fail (unknown / model unavailable).
                log.debug("warmup_surrogate skipped track", track_id=tid, exc_info=True)

        elapsed = time.perf_counter() - start

        # Count cache hits after warmup by re-querying one track.
        cache_hits = 0
        if warmed > 0 and track_ids:
            try:
                from f1opt.model.surrogate import predict_lap_time

                # Second call should hit the cache.
                cache_info_before = getattr(predict_lap_time, "cache_info", None)
                if cache_info_before is not None:
                    before = cache_info_before().hits
                    predict_lap_time(DEFAULT_SETUP, track_ids[0], None)
                    after = cache_info_before().hits
                    cache_hits = after - before
            except Exception:
                log.debug("warmup_surrogate cache-hit probe failed", exc_info=True)

        return {
            "tracks_warmed": warmed,
            "time_s": elapsed,
            "cache_hits_after": cache_hits,
        }

    def warmup_all(self) -> dict[str, Any]:
        """Warmup surrogate + precomputed lookups."""
        surrogate_result = self.warmup_surrogate()
        # Populate lookups.
        self._lookups.track_reference_lap_times()
        self._lookups.setup_field_ranges()
        self._lookups.compound_params()
        return {
            "surrogate": surrogate_result,
            "lookups_loaded": True,
            "track_refs_count": len(self._lookups.track_reference_lap_times()),
            "setup_fields_count": len(self._lookups.setup_field_ranges()),
            "compounds_count": len(self._lookups.compound_params()),
        }
