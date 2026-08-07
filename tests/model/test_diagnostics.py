"""Tests for :mod:`f1opt.model.diagnostics` (Iter-141)."""
from __future__ import annotations

import numpy as np

from f1opt.model.diagnostics import (
    CalibrationReport,
    calibration_curve,
    feature_importance,
    per_track_error_breakdown,
    prediction_uncertainty,
    residual_analysis,
)


class TestCalibrationCurve:
    def test_perfect_prediction(self) -> None:
        from f1opt.model.train import train
        m = train(iterations=50, n_samples=200, seed=0, log=False, save=False)
        x = np.random.randn(100, 39).astype(np.float32)
        y = np.random.rand(100).astype(np.float32) * 5 + 80
        r = calibration_curve(m, x, y)
        assert isinstance(r, CalibrationReport)
        assert r.n_samples == 100
        assert r.mae >= 0
        assert r.rmse >= r.mae
        assert isinstance(r.r_squared, float)  # R² can be negative for poor models
        assert len(r.bin_edges) == 11
        assert len(r.bin_pred_means) == 10
        assert r.bin_counts.sum() == 100

    def test_small_dataset(self) -> None:
        from f1opt.model.train import train
        m = train(iterations=20, n_samples=50, seed=1, log=False, save=False)
        x = np.random.randn(10, 39).astype(np.float32)
        y = np.array([80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0])
        r = calibration_curve(m, x, y, n_bins=5)
        assert r.bin_counts.sum() == 10

    def test_constant_y(self) -> None:
        from f1opt.model.train import train
        m = train(iterations=20, n_samples=50, seed=2, log=False, save=False)
        x = np.random.randn(20, 39).astype(np.float32)
        y = np.ones(20) * 85.0
        r = calibration_curve(m, x, y)
        assert r.r_squared <= 1.0

    def test_heteroscedasticity_detected(self) -> None:
        from f1opt.model.train import train
        m = train(iterations=50, n_samples=200, seed=3, log=False, save=False)
        x = np.random.randn(100, 39).astype(np.float32)
        y = np.linspace(75, 95, 100)
        r = calibration_curve(m, x, y, n_bins=5)
        assert r.heteroscedasticity_ratio >= 1.0


class TestResAnalysis:
    def test_zero_residuals(self) -> None:
        r = residual_analysis(np.zeros(50))
        assert r.mean == 0.0
        assert r.std == 0.0
        assert r.outlier_count == 0

    def test_normal_residuals(self) -> None:
        np.random.seed(42)
        res = np.random.randn(1000) * 0.1
        r = residual_analysis(res)
        assert abs(r.mean) < 0.02
        assert abs(r.skewness) < 0.5
        assert 2.0 < r.kurtosis < 5.0

    def test_skewed_residuals(self) -> None:
        res = np.array([-0.5] * 20 + [0.1] * 80)
        r = residual_analysis(res)
        assert r.skewness < 0

    def test_outliers_flagged(self) -> None:
        res = np.array([0.0] * 48 + [10.0, -10.0])
        r = residual_analysis(res)
        assert r.outlier_count >= 2
        assert len(r.outlier_indices) >= 2

    def test_percentiles_ordered(self) -> None:
        res = np.random.randn(200) * 0.5
        r = residual_analysis(res)
        assert r.q_01 <= r.q_05 <= r.q_25 <= r.q_50 <= r.q_75 <= r.q_95 <= r.q_99

    def test_single_sample(self) -> None:
        r = residual_analysis(np.array([0.5]))
        assert r.mean == 0.5
        assert r.std == 0.0


class TestPerTrackBreakdown:
    def test_basic(self) -> None:
        from f1opt.model.train import train
        m = train(iterations=30, n_samples=100, seed=7, log=False, save=False)
        x = np.random.randn(60, 39).astype(np.float32)
        y = np.random.rand(60).astype(np.float32) * 5 + 80
        tracks = ["melbourne"] * 30 + ["monza"] * 30
        bd = per_track_error_breakdown(m, x, y, tracks)
        assert "melbourne" in bd
        assert "monza" in bd
        assert bd["melbourne"]["n"] == 30
        assert bd["monza"]["n"] == 30
        assert bd["melbourne"]["mae"] >= 0

class TestPredictionUncertainty:
    def test_single_model_basic(self) -> None:
        from f1opt.model.train import train
        m = train(iterations=50, n_samples=200, seed=0, log=False, save=False)
        x = np.random.randn(20, 39).astype(np.float32)
        result = prediction_uncertainty(m, x, n_samples=30, noise_std=0.01, seed=42)
        assert 'mean' in result
        assert 'std' in result
        assert 'q05' in result
        assert 'q95' in result
        assert result['mean'].shape == (20,)
        assert result['std'].shape == (20,)
        assert np.all(result['std'] >= 0)
        assert np.all(result['q05'] <= result['mean'])
        assert np.all(result['mean'] <= result['q95'])

    def test_single_model_larger_noise(self) -> None:
        from f1opt.model.train import train
        m = train(iterations=50, n_samples=200, seed=1, log=False, save=False)
        x = np.random.randn(10, 39).astype(np.float32)
        r1 = prediction_uncertainty(m, x, n_samples=20, noise_std=0.001, seed=42)
        r2 = prediction_uncertainty(m, x, n_samples=20, noise_std=0.05, seed=42)
        assert np.mean(r2['std']) >= np.mean(r1['std']) * 0.5

    def test_ensemble_model(self) -> None:
        from f1opt.model.train import train_ensemble
        ens = train_ensemble(n_members=3, base_seed=0, iterations=50,
                              n_samples=200, save=False, log=False)
        x = np.random.randn(10, 39).astype(np.float32)
        result = prediction_uncertainty(ens, x, seed=42)
        assert result['mean'].shape == (10,)
        assert result['std'].shape == (10,)
        assert np.all(result['std'] >= 0)

    def test_single_point(self) -> None:
        from f1opt.model.train import train
        m = train(iterations=30, n_samples=100, seed=3, log=False, save=False)
        x = np.random.randn(1, 39).astype(np.float32)
        result = prediction_uncertainty(m, x, n_samples=20, noise_std=0.01, seed=42)
        assert result['mean'].shape == (1,)
        assert result['std'].shape == (1,)

class TestFeatureImportance:
    def test_basic_importance(self) -> None:
        from f1opt.model.train import _build_tensors, generate_dataset, train
        m = train(iterations=100, n_samples=500, seed=0, log=False, save=False)
        data = generate_dataset(n_samples=200, seed=1, label_source='physics')
        x, sec_y, resp_y, sp, rp = _build_tensors(data)
        y_lap = np.asarray(sec_y.sum(dim=1))
        result = feature_importance(m, np.asarray(x), y_lap, n_repeats=3, seed=42)
        assert 'names' in result
        assert 'importance_mean' in result
        assert 'importance_std' in result
        assert len(result['names']) == 39
        assert len(result['importance_mean']) == 39
        assert len(result['importance_std']) == 39
        # Importance should be sorted descending
        imps = result['importance_mean']
        assert np.all(imps[:-1] >= imps[1:])

    def test_custom_feature_names(self) -> None:
        from f1opt.model.train import _build_tensors, generate_dataset, train
        m = train(iterations=50, n_samples=200, seed=2, log=False, save=False)
        data = generate_dataset(n_samples=100, seed=3, label_source='physics')
        x, sec_y, resp_y, sp, rp = _build_tensors(data)
        y_lap = np.asarray(sec_y.sum(dim=1))
        names = ['fw_angle', 'rw_angle'] + [f'f{i}' for i in range(2, 39)]
        result = feature_importance(m, np.asarray(x), y_lap, feature_names=names, n_repeats=2, seed=42)
        assert result['names'][0] in names

    def test_invalid_feature_names_raises(self) -> None:
        import pytest

        from f1opt.model.train import _build_tensors, generate_dataset, train
        m = train(iterations=30, n_samples=100, seed=4, log=False, save=False)
        data = generate_dataset(n_samples=50, seed=5, label_source='physics')
        x, sec_y, resp_y, sp, rp = _build_tensors(data)
        y_lap = np.asarray(sec_y.sum(dim=1))
        with pytest.raises(ValueError):
            feature_importance(m, np.asarray(x), y_lap, feature_names=['a', 'b'], seed=42)

    def test_importance_non_negative(self) -> None:
        from f1opt.model.train import _build_tensors, generate_dataset, train
        m = train(iterations=100, n_samples=500, seed=6, log=False, save=False)
        data = generate_dataset(n_samples=200, seed=7, label_source='physics')
        x, sec_y, resp_y, sp, rp = _build_tensors(data)
        y_lap = np.asarray(sec_y.sum(dim=1))
        result = feature_importance(m, np.asarray(x), y_lap, n_repeats=3, seed=42)
        # Top features should have positive importance
        assert result['importance_mean'][0] > 0
        assert np.all(result['importance_std'] >= 0)
