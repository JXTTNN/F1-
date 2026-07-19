"""Prediction interval tests for :mod:`f1opt.model.diagnostics` (Iter-157)."""
from __future__ import annotations

import numpy as np
import pytest

from f1opt.model.diagnostics import (
    PredictionInterval,
    prediction_intervals,
)


def _train_small_model():
    """Train a small model for testing."""
    from f1opt.model.train import train
    return train(iterations=50, n_samples=200, seed=0, log=False, save=False)


class TestPredictionIntervals:
    def test_basic_run(self) -> None:
        """prediction_intervals 返回有效的 PredictionInterval."""
        model = _train_small_model()
        rng = np.random.RandomState(0)
        x_cal = rng.randn(50, 37).astype(np.float32)
        y_cal = rng.rand(50).astype(np.float32) * 5 + 80
        x_new = rng.randn(10, 37).astype(np.float32)
        result = prediction_intervals(model, x_cal, y_cal, x_new, confidence=0.90)
        assert isinstance(result, PredictionInterval)
        assert result.confidence == 0.90
        assert result.n_calibration == 50
        assert len(result.lower) == 10
        assert len(result.upper) == 10
        assert len(result.center) == 10

    def test_lower_le_center_le_upper(self) -> None:
        """lower <= center <= upper."""
        model = _train_small_model()
        rng = np.random.RandomState(1)
        x_cal = rng.randn(40, 37).astype(np.float32)
        y_cal = rng.rand(40).astype(np.float32) * 5 + 80
        x_new = rng.randn(10, 37).astype(np.float32)
        result = prediction_intervals(model, x_cal, y_cal, x_new)
        assert np.all(result.lower <= result.center)
        assert np.all(result.center <= result.upper)

    def test_margin_positive(self) -> None:
        """margin 为正数 (除非残差全为零)."""
        model = _train_small_model()
        rng = np.random.RandomState(2)
        x_cal = rng.randn(40, 37).astype(np.float32)
        y_cal = rng.rand(40).astype(np.float32) * 5 + 80
        x_new = rng.randn(5, 37).astype(np.float32)
        result = prediction_intervals(model, x_cal, y_cal, x_new)
        assert result.margin > 0

    def test_margin_symmetric(self) -> None:
        """margin 对称: upper - center == center - lower == margin."""
        model = _train_small_model()
        rng = np.random.RandomState(3)
        x_cal = rng.randn(40, 37).astype(np.float32)
        y_cal = rng.rand(40).astype(np.float32) * 5 + 80
        x_new = rng.randn(5, 37).astype(np.float32)
        result = prediction_intervals(model, x_cal, y_cal, x_new)
        np.testing.assert_array_almost_equal(
            result.upper - result.center,
            np.full(5, result.margin),
            decimal=4,
        )
        np.testing.assert_array_almost_equal(
            result.center - result.lower,
            np.full(5, result.margin),
            decimal=4,
        )

    def test_empirical_coverage_reasonable(self) -> None:
        """经验覆盖率应接近 confidence (允许一定偏差)."""
        model = _train_small_model()
        rng = np.random.RandomState(4)
        x_cal = rng.randn(100, 37).astype(np.float32)
        y_cal = rng.rand(100).astype(np.float32) * 5 + 80
        x_new = rng.randn(5, 37).astype(np.float32)
        result = prediction_intervals(model, x_cal, y_cal, x_new, confidence=0.90)
        # Coverage should be >= 0.90 - tolerance (conformal guarantee)
        # or at least reasonable
        assert result.empirical_coverage >= 0.80

    def test_higher_confidence_wider_interval(self) -> None:
        """更高 confidence → 更宽 margin."""
        model = _train_small_model()
        rng = np.random.RandomState(5)
        x_cal = rng.randn(50, 37).astype(np.float32)
        y_cal = rng.rand(50).astype(np.float32) * 5 + 80
        x_new = rng.randn(5, 37).astype(np.float32)
        result_90 = prediction_intervals(model, x_cal, y_cal, x_new, confidence=0.90)
        result_99 = prediction_intervals(model, x_cal, y_cal, x_new, confidence=0.99)
        assert result_99.margin >= result_90.margin

    def test_invalid_confidence_zero(self) -> None:
        """confidence=0 抛出 ValueError."""
        model = _train_small_model()
        x_cal = np.zeros((10, 37), dtype=np.float32)
        y_cal = np.ones(10, dtype=np.float32) * 80
        x_new = np.zeros((5, 37), dtype=np.float32)
        with pytest.raises(ValueError, match="confidence must be in"):
            prediction_intervals(model, x_cal, y_cal, x_new, confidence=0.0)

    def test_invalid_confidence_one(self) -> None:
        """confidence=1 抛出 ValueError."""
        model = _train_small_model()
        x_cal = np.zeros((10, 37), dtype=np.float32)
        y_cal = np.ones(10, dtype=np.float32) * 80
        x_new = np.zeros((5, 37), dtype=np.float32)
        with pytest.raises(ValueError, match="confidence must be in"):
            prediction_intervals(model, x_cal, y_cal, x_new, confidence=1.0)

    def test_too_few_calibration_samples(self) -> None:
        """校准集 < 2 样本抛出 ValueError."""
        model = _train_small_model()
        x_cal = np.zeros((1, 37), dtype=np.float32)
        y_cal = np.ones(1, dtype=np.float32) * 80
        x_new = np.zeros((5, 37), dtype=np.float32)
        with pytest.raises(ValueError, match="calibration set must have"):
            prediction_intervals(model, x_cal, y_cal, x_new)

    def test_empty_new_set(self) -> None:
        """x_new 为空时返回空数组."""
        model = _train_small_model()
        rng = np.random.RandomState(6)
        x_cal = rng.randn(20, 37).astype(np.float32)
        y_cal = rng.rand(20).astype(np.float32) * 5 + 80
        x_new = np.zeros((0, 37), dtype=np.float32)
        result = prediction_intervals(model, x_cal, y_cal, x_new)
        assert len(result.lower) == 0
        assert len(result.upper) == 0

    def test_coverage_with_perfect_predictions(self) -> None:
        """当模型预测完美时 (残差=0), margin=0, coverage=1."""
        # Use a model and make y_cal exactly equal to predictions
        model = _train_small_model()
        rng = np.random.RandomState(7)
        x_cal = rng.randn(20, 37).astype(np.float32)
        # Make y_cal equal to model predictions
        from f1opt.model.diagnostics import _predict_lap
        y_cal = _predict_lap(model, x_cal)
        x_new = rng.randn(5, 37).astype(np.float32)
        result = prediction_intervals(model, x_cal, y_cal, x_new, confidence=0.90)
        assert result.margin == pytest.approx(0.0, abs=1e-6)
        assert result.empirical_coverage == pytest.approx(1.0)

    def test_single_new_point(self) -> None:
        """单个新点的区间."""
        model = _train_small_model()
        rng = np.random.RandomState(8)
        x_cal = rng.randn(20, 37).astype(np.float32)
        y_cal = rng.rand(20).astype(np.float32) * 5 + 80
        x_new = rng.randn(1, 37).astype(np.float32)
        result = prediction_intervals(model, x_cal, y_cal, x_new)
        assert len(result.lower) == 1
        assert result.lower[0] <= result.center[0] <= result.upper[0]
