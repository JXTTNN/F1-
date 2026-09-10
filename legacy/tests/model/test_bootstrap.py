"""Bootstrap confidence interval tests for :mod:`f1opt.model.diagnostics` (Iter-161)."""
from __future__ import annotations

import numpy as np
import pytest

from f1opt.model.diagnostics import (
    BootstrapReport,
    ModelComparisonReport,
    bootstrap_metrics,
    compare_models,
)


def _train_small_model(seed: int = 0) -> object:
    """Train a small model for testing."""
    from f1opt.model.train import train
    return train(iterations=50, n_samples=200, seed=seed, log=False, save=False)


class TestBootstrapMetrics:
    def test_basic_run(self) -> None:
        """bootstrap_metrics returns a populated BootstrapReport."""
        rng = np.random.RandomState(0)
        y_true = rng.rand(100) * 5 + 80
        y_pred = y_true + rng.randn(100) * 0.5
        r = bootstrap_metrics(y_true, y_pred, n_bootstrap=200, seed=0)
        assert isinstance(r, BootstrapReport)
        assert r.n_samples == 100
        assert r.n_bootstrap == 200
        assert r.confidence == 0.95

    def test_lower_le_upper(self) -> None:
        """All confidence intervals satisfy lower <= mean <= upper (approximately)."""
        rng = np.random.RandomState(1)
        y_true = rng.rand(200) * 5 + 80
        y_pred = y_true + rng.randn(200) * 0.3
        r = bootstrap_metrics(y_true, y_pred, n_bootstrap=500, seed=1)
        assert r.mae_lower <= r.mae_mean <= r.mae_upper
        assert r.rmse_lower <= r.rmse_mean <= r.rmse_upper
        assert r.max_error_lower <= r.max_error_mean <= r.max_error_upper

    def test_mae_le_rmse(self) -> None:
        """MAE <= RMSE for any residual distribution (Jensen's inequality)."""
        rng = np.random.RandomState(2)
        y_true = rng.rand(150) * 5 + 80
        y_pred = y_true + rng.randn(150) * 0.5
        r = bootstrap_metrics(y_true, y_pred, n_bootstrap=200, seed=2)
        assert r.mae_mean <= r.rmse_mean

    def test_perfect_predictions_zero_metric(self) -> None:
        """When predictions exactly match true, MAE / RMSE / max-error are 0."""
        y = np.linspace(80, 90, 50)
        r = bootstrap_metrics(y, y, n_bootstrap=200, seed=3)
        assert r.mae_mean == pytest.approx(0.0, abs=1e-12)
        assert r.rmse_mean == pytest.approx(0.0, abs=1e-12)
        assert r.max_error_mean == pytest.approx(0.0, abs=1e-12)
        assert r.mae_std == pytest.approx(0.0, abs=1e-12)

    def test_higher_confidence_wider_interval(self) -> None:
        """Higher confidence => wider (or equal) MAE CI."""
        rng = np.random.RandomState(4)
        y_true = rng.rand(200) * 5 + 80
        y_pred = y_true + rng.randn(200) * 0.4
        r90 = bootstrap_metrics(y_true, y_pred, n_bootstrap=500,
                                confidence=0.90, seed=4)
        r99 = bootstrap_metrics(y_true, y_pred, n_bootstrap=500,
                                confidence=0.99, seed=4)
        assert (r99.mae_upper - r99.mae_lower) >= (r90.mae_upper - r90.mae_lower)

    def test_seed_reproducibility(self) -> None:
        """Same seed produces identical results."""
        rng = np.random.RandomState(5)
        y_true = rng.rand(100) * 5 + 80
        y_pred = y_true + rng.randn(100) * 0.3
        r1 = bootstrap_metrics(y_true, y_pred, n_bootstrap=200, seed=42)
        r2 = bootstrap_metrics(y_true, y_pred, n_bootstrap=200, seed=42)
        assert r1.mae_mean == r2.mae_mean
        assert r1.mae_lower == r2.mae_lower
        assert r1.mae_upper == r2.mae_upper

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce (slightly) different results."""
        rng = np.random.RandomState(6)
        y_true = rng.rand(100) * 5 + 80
        y_pred = y_true + rng.randn(100) * 0.3
        r1 = bootstrap_metrics(y_true, y_pred, n_bootstrap=200, seed=1)
        r2 = bootstrap_metrics(y_true, y_pred, n_bootstrap=200, seed=2)
        # Means converge to the same population value but bootstrap samples differ
        assert r1.mae_lower != r2.mae_lower or r1.mae_upper != r2.mae_upper

    def test_invalid_confidence_zero(self) -> None:
        """confidence=0 raises ValueError."""
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="confidence must be in"):
            bootstrap_metrics(y, y, confidence=0.0)

    def test_invalid_confidence_one(self) -> None:
        """confidence=1 raises ValueError."""
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="confidence must be in"):
            bootstrap_metrics(y, y, confidence=1.0)

    def test_invalid_n_bootstrap(self) -> None:
        """n_bootstrap < 2 raises ValueError."""
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="n_bootstrap must be >= 2"):
            bootstrap_metrics(y, y, n_bootstrap=1)

    def test_too_few_samples(self) -> None:
        """Less than 2 samples raises ValueError."""
        y = np.array([1.0])
        with pytest.raises(ValueError, match="need at least 2 samples"):
            bootstrap_metrics(y, y)

    def test_shape_mismatch(self) -> None:
        """Mismatched y_true / y_pred shapes raise ValueError."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="shape mismatch"):
            bootstrap_metrics(y_true, y_pred)

    def test_rejects_2d_input(self) -> None:
        """2-D inputs raise ValueError (must be 1-D)."""
        y_true = np.ones((5, 2))
        y_pred = np.ones((5, 2))
        with pytest.raises(ValueError, match="expected 1-D arrays"):
            bootstrap_metrics(y_true, y_pred)

    def test_point_estimate_matches_full_sample(self) -> None:
        """Bootstrap mean MAE ≈ MAE computed on the full sample (large n_bootstrap)."""
        rng = np.random.RandomState(7)
        y_true = rng.rand(500) * 5 + 80
        y_pred = y_true + rng.randn(500) * 0.4
        full_mae = float(np.mean(np.abs(y_pred - y_true)))
        r = bootstrap_metrics(y_true, y_pred, n_bootstrap=2000, seed=7)
        assert abs(r.mae_mean - full_mae) < 0.02  # close to full-sample MAE

    def test_constant_offset(self) -> None:
        """Constant prediction offset of `c` => MAE = |c| exactly."""
        y_true = np.linspace(80, 90, 100)
        y_pred = y_true + 0.5  # constant offset
        r = bootstrap_metrics(y_true, y_pred, n_bootstrap=200, seed=8)
        assert r.mae_mean == pytest.approx(0.5, abs=1e-12)
        assert r.rmse_mean == pytest.approx(0.5, abs=1e-12)
        assert r.max_error_mean == pytest.approx(0.5, abs=1e-12)


class TestCompareModels:
    def test_basic_run(self) -> None:
        """compare_models returns a populated ModelComparisonReport."""
        model_a = _train_small_model(seed=0)
        model_b = _train_small_model(seed=1)
        rng = np.random.RandomState(0)
        x = rng.randn(50, 41).astype(np.float32)
        y = rng.rand(50) * 5 + 80
        r = compare_models(model_a, model_b, x, y, n_bootstrap=200, seed=0)
        assert isinstance(r, ModelComparisonReport)
        assert r.n_samples == 50
        assert r.n_bootstrap == 200
        assert r.confidence == 0.95

    def test_delta_lower_le_upper(self) -> None:
        """Confidence interval on the delta satisfies lower <= upper."""
        model_a = _train_small_model(seed=0)
        model_b = _train_small_model(seed=1)
        rng = np.random.RandomState(1)
        x = rng.randn(80, 41).astype(np.float32)
        y = rng.rand(80) * 5 + 80
        r = compare_models(model_a, model_b, x, y, n_bootstrap=300, seed=1)
        assert r.delta_mae_lower <= r.delta_mae_upper

    def test_p_better_in_range(self) -> None:
        """p_b_better_mae is in [0, 1]."""
        model_a = _train_small_model(seed=0)
        model_b = _train_small_model(seed=1)
        rng = np.random.RandomState(2)
        x = rng.randn(60, 41).astype(np.float32)
        y = rng.rand(60) * 5 + 80
        r = compare_models(model_a, model_b, x, y, n_bootstrap=200, seed=2)
        assert 0.0 <= r.p_b_better_mae <= 1.0

    def test_self_comparison_undecided(self) -> None:
        """Comparing a model to itself: delta ≈ 0, p ≈ 0.5."""
        model_a = _train_small_model(seed=0)
        rng = np.random.RandomState(3)
        x = rng.randn(100, 41).astype(np.float32)
        y = rng.rand(100) * 5 + 80
        r = compare_models(model_a, model_a, x, y, n_bootstrap=500, seed=3)
        # MAE_A == MAE_B exactly, so delta == 0 in expectation.
        assert r.mae_a == pytest.approx(r.mae_b, abs=1e-12)
        assert abs(r.delta_mae_mean) < 1e-9
        # When the two models are identical, MAE_A == MAE_B for every draw, so
        # the strict-less test always fails and p_b_better_mae == 0.
        assert r.p_b_better_mae == pytest.approx(0.0, abs=1e-12)

    def test_point_estimates_match_full_sample(self) -> None:
        """mae_a / mae_b point estimates match direct computation."""
        model_a = _train_small_model(seed=0)
        model_b = _train_small_model(seed=1)
        rng = np.random.RandomState(4)
        x = rng.randn(60, 41).astype(np.float32)
        y = rng.rand(60) * 5 + 80
        r = compare_models(model_a, model_b, x, y, n_bootstrap=200, seed=4)

        from f1opt.model.diagnostics import _predict_lap
        mae_a_direct = float(np.mean(np.abs(_predict_lap(model_a, x) - y)))
        mae_b_direct = float(np.mean(np.abs(_predict_lap(model_b, x) - y)))
        assert r.mae_a == pytest.approx(mae_a_direct, abs=1e-9)
        assert r.mae_b == pytest.approx(mae_b_direct, abs=1e-9)

    def test_seed_reproducibility(self) -> None:
        """Same seed => identical reports."""
        model_a = _train_small_model(seed=0)
        model_b = _train_small_model(seed=1)
        rng = np.random.RandomState(5)
        x = rng.randn(40, 41).astype(np.float32)
        y = rng.rand(40) * 5 + 80
        r1 = compare_models(model_a, model_b, x, y, n_bootstrap=200, seed=99)
        r2 = compare_models(model_a, model_b, x, y, n_bootstrap=200, seed=99)
        assert r1.delta_mae_mean == r2.delta_mae_mean
        assert r1.p_b_better_mae == r2.p_b_better_mae

    def test_invalid_confidence(self) -> None:
        """confidence out of (0, 1) raises ValueError."""
        model_a = _train_small_model(seed=0)
        model_b = _train_small_model(seed=1)
        x = np.zeros((10, 41), dtype=np.float32)
        y = np.ones(10) * 80
        with pytest.raises(ValueError, match="confidence must be in"):
            compare_models(model_a, model_b, x, y, confidence=0.0)
        with pytest.raises(ValueError, match="confidence must be in"):
            compare_models(model_a, model_b, x, y, confidence=1.5)

    def test_invalid_n_bootstrap(self) -> None:
        """n_bootstrap < 2 raises ValueError."""
        model_a = _train_small_model(seed=0)
        model_b = _train_small_model(seed=1)
        x = np.zeros((10, 41), dtype=np.float32)
        y = np.ones(10) * 80
        with pytest.raises(ValueError, match="n_bootstrap must be >= 2"):
            compare_models(model_a, model_b, x, y, n_bootstrap=1)

    def test_too_few_samples(self) -> None:
        """Fewer than 2 samples raises ValueError."""
        model_a = _train_small_model(seed=0)
        model_b = _train_small_model(seed=1)
        x = np.zeros((1, 41), dtype=np.float32)
        y = np.ones(1) * 80
        with pytest.raises(ValueError, match="need at least 2 samples"):
            compare_models(model_a, model_b, x, y)

    def test_x_y_length_mismatch(self) -> None:
        """Mismatched x and y_lap lengths raise ValueError."""
        model_a = _train_small_model(seed=0)
        model_b = _train_small_model(seed=1)
        x = np.zeros((10, 41), dtype=np.float32)
        y = np.ones(8) * 80
        with pytest.raises(ValueError, match="x.shape"):
            compare_models(model_a, model_b, x, y)
