"""Cross-validation tests for :mod:`f1opt.model.diagnostics` (Iter-153)."""
from __future__ import annotations

import numpy as np
import pytest

from f1opt.model.diagnostics import (
    CrossValidationReport,
    cross_validate,
)


class TestCrossValidate:
    def test_basic_run(self) -> None:
        """cross_validate 返回有效的 CrossValidationReport."""
        rng = np.random.RandomState(0)
        x = rng.randn(60, 39).astype(np.float32)
        y = rng.rand(60).astype(np.float32) * 5 + 80
        report = cross_validate(x, y, n_folds=3, iterations=30, seed=0)
        assert isinstance(report, CrossValidationReport)
        assert report.n_folds == 3
        assert report.n_total == 60

    def test_fold_sizes_sum_to_total(self) -> None:
        """各 fold 大小之和等于总样本数."""
        rng = np.random.RandomState(1)
        x = rng.randn(50, 39).astype(np.float32)
        y = rng.rand(50).astype(np.float32) * 5 + 80
        report = cross_validate(x, y, n_folds=5, iterations=20, seed=0)
        assert report.fold_sizes.sum() == 50

    def test_metrics_non_negative(self) -> None:
        """所有 MAE/RMSE/max_error 均非负."""
        rng = np.random.RandomState(2)
        x = rng.randn(40, 39).astype(np.float32)
        y = rng.rand(40).astype(np.float32) * 5 + 80
        report = cross_validate(x, y, n_folds=4, iterations=20, seed=0)
        assert np.all(report.fold_maes >= 0)
        assert np.all(report.fold_rmses >= 0)
        assert np.all(report.fold_max_errors >= 0)
        assert report.mean_mae >= 0
        assert report.mean_rmse >= 0

    def test_rmse_geq_mae_per_fold(self) -> None:
        """每个 fold 的 RMSE >= MAE (数学性质)."""
        rng = np.random.RandomState(3)
        x = rng.randn(40, 39).astype(np.float32)
        y = rng.rand(40).astype(np.float32) * 5 + 80
        report = cross_validate(x, y, n_folds=4, iterations=20, seed=0)
        for mae, rmse in zip(report.fold_maes, report.fold_rmses, strict=True):
            assert rmse >= mae - 1e-9

    def test_max_error_geq_mae_per_fold(self) -> None:
        """每个 fold 的 max_error >= MAE (max(|r|) >= mean(|r|))."""
        rng = np.random.RandomState(4)
        x = rng.randn(40, 39).astype(np.float32)
        y = rng.rand(40).astype(np.float32) * 5 + 80
        report = cross_validate(x, y, n_folds=4, iterations=20, seed=0)
        for mae, max_err in zip(report.fold_maes, report.fold_max_errors, strict=True):
            assert max_err >= mae - 1e-9

    def test_reproducible_with_same_seed(self) -> None:
        """相同 seed 产生相同结果."""
        rng = np.random.RandomState(5)
        x = rng.randn(40, 39).astype(np.float32)
        y = rng.rand(40).astype(np.float32) * 5 + 80
        r1 = cross_validate(x, y, n_folds=4, iterations=20, seed=42)
        r2 = cross_validate(x, y, n_folds=4, iterations=20, seed=42)
        np.testing.assert_array_almost_equal(r1.fold_maes, r2.fold_maes)

    def test_different_seeds_differ(self) -> None:
        """不同 seed 产生不同 fold 划分 (大概率)."""
        rng = np.random.RandomState(6)
        x = rng.randn(40, 39).astype(np.float32)
        y = rng.rand(40).astype(np.float32) * 5 + 80
        r1 = cross_validate(x, y, n_folds=4, iterations=20, seed=0)
        r2 = cross_validate(x, y, n_folds=4, iterations=20, seed=99)
        # fold 划分不同, MAE 应该有差异
        assert not np.allclose(r1.fold_maes, r2.fold_maes)

    def test_n_folds_too_small_raises(self) -> None:
        """n_folds < 2 抛出 ValueError."""
        x = np.zeros((10, 39), dtype=np.float32)
        y = np.ones(10, dtype=np.float32) * 80
        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            cross_validate(x, y, n_folds=1, iterations=10)

    def test_n_folds_too_large_raises(self) -> None:
        """n_folds > n_samples 抛出 ValueError."""
        x = np.zeros((5, 39), dtype=np.float32)
        y = np.ones(5, dtype=np.float32) * 80
        with pytest.raises(ValueError, match="n_folds.* > n_samples"):
            cross_validate(x, y, n_folds=10, iterations=10)

    def test_no_shuffle(self) -> None:
        """shuffle=False 时不打乱顺序 (仍能运行)."""
        rng = np.random.RandomState(7)
        x = rng.randn(30, 39).astype(np.float32)
        y = rng.rand(30).astype(np.float32) * 5 + 80
        report = cross_validate(x, y, n_folds=3, iterations=15, seed=0, shuffle=False)
        assert report.n_total == 30
        assert report.fold_sizes.sum() == 30

    def test_uneven_fold_sizes(self) -> None:
        """n_samples 不能被 n_folds 整除时, 前面 fold 多一个样本."""
        rng = np.random.RandomState(8)
        x = rng.randn(32, 39).astype(np.float32)
        y = rng.rand(32).astype(np.float32) * 5 + 80
        report = cross_validate(x, y, n_folds=5, iterations=10, seed=0)
        # 32 / 5 = 6 余 2, 所以前 2 个 fold 有 7 个样本, 后 3 个有 6 个
        expected = np.array([7, 7, 6, 6, 6])
        np.testing.assert_array_equal(report.fold_sizes, expected)

    def test_std_with_multiple_folds(self) -> None:
        """n_folds >= 2 时 std_mae 为非负浮点数."""
        rng = np.random.RandomState(9)
        x = rng.randn(40, 39).astype(np.float32)
        y = rng.rand(40).astype(np.float32) * 5 + 80
        report = cross_validate(x, y, n_folds=4, iterations=20, seed=0)
        assert isinstance(report.std_mae, float)
        assert report.std_mae >= 0.0
        assert isinstance(report.std_rmse, float)
        assert report.std_rmse >= 0.0
