"""Tests for :mod:`f1opt.model.lr_finder` (Iter-148)."""
from __future__ import annotations

import numpy as np

from f1opt.model.lr_finder import LRFinderResult, lr_find


class TestLRFinder:
    def test_basic_run(self) -> None:
        """LR finder should complete and return a valid result."""
        result = lr_find(n_steps=30, n_samples=500, batch_size=32, seed=42)
        assert isinstance(result, LRFinderResult)
        assert len(result.lrs) > 0
        assert len(result.losses) == len(result.lrs)
        assert result.suggested_lr > 0
        assert result.steps_run > 0

    def test_lr_monotonically_increasing(self) -> None:
        """LRs should increase monotonically across steps."""
        result = lr_find(n_steps=30, n_samples=500, batch_size=32, seed=42)
        for i in range(1, len(result.lrs)):
            assert result.lrs[i] > result.lrs[i - 1], (
                f"LR not increasing at step {i}: {result.lrs[i]} <= {result.lrs[i-1]}"
            )

    def test_suggested_lr_in_range(self) -> None:
        """Suggested LR should be within the tested range."""
        result = lr_find(
            n_steps=50, start_lr=1e-7, end_lr=1.0,
            n_samples=500, batch_size=32, seed=42,
        )
        assert result.suggested_lr >= 1e-7
        assert result.suggested_lr <= 1.0

    def test_fewer_samples_than_batch(self) -> None:
        """Should handle case where n_samples < batch_size."""
        result = lr_find(
            n_steps=10, n_samples=16, batch_size=64, seed=42,
        )
        assert result.steps_run > 0
        assert result.suggested_lr > 0

    def test_no_smoothing(self) -> None:
        """Should work with smoothing=0.0."""
        result = lr_find(
            n_steps=20, smoothing=0.0, n_samples=500, batch_size=32, seed=42,
        )
        assert len(result.losses) == len(result.lrs)

    def test_min_loss_tracked(self) -> None:
        """min_loss should be finite and positive."""
        result = lr_find(n_steps=30, n_samples=500, batch_size=32, seed=42)
        assert result.min_loss > 0
        assert np.isfinite(result.min_loss)
        assert result.min_loss_lr > 0

    def test_result_reproducible(self) -> None:
        """Same seed should produce similar loss trajectories."""
        r1 = lr_find(n_steps=20, n_samples=500, batch_size=32, seed=42)
        r2 = lr_find(n_steps=20, n_samples=500, batch_size=32, seed=42)
        # Losses should be identical (same seed, same data, same model init)
        assert r1.losses == r2.losses
        assert r1.lrs == r2.lrs

    def test_different_seeds_different(self) -> None:
        """Different seeds may produce different results (stochastic training)."""
        r1 = lr_find(n_steps=20, n_samples=500, batch_size=32, seed=1)
        r2 = lr_find(n_steps=20, n_samples=500, batch_size=32, seed=2)
        # At least the loss sequences should differ
        assert r1.losses != r2.losses or r1.suggested_lr != r2.suggested_lr

    def test_custom_model(self) -> None:
        """Should accept a pre-existing model."""
        from f1opt.model.surrogate import SurrogateModel
        model = SurrogateModel()
        result = lr_find(model=model, n_steps=20, n_samples=500, batch_size=32, seed=42)
        assert result.steps_run > 0
        assert result.suggested_lr > 0