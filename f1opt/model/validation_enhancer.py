from __future__ import annotations

import numpy as np
from typing import Sequence


class ValidationEnhancer:

    @staticmethod
    def compute_learning_curve(
        model_fn,
        X: np.ndarray,
        y: np.ndarray,
        train_sizes: np.ndarray | None = None,
        n_splits: int = 5,
        seed: int = 42,
    ) -> dict:
        rng = np.random.default_rng(seed)
        n_samples = len(X)

        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        train_scores = []
        test_scores = []
        sizes_used = []

        from f1opt.model.cross_validation import k_fold_split

        for frac in train_sizes:
            n_train = max(int(n_samples * frac), 5)
            if n_train >= n_samples:
                continue

            idx = rng.choice(n_samples, n_train, replace=False)
            X_sub = X[idx]
            y_sub = y[idx]

            folds = k_fold_split(len(X_sub), k=n_splits, shuffle=True, seed=seed)

            fold_train = []
            fold_test = []

            for train_idx, test_idx in folds:
                model_instance = model_fn()
                model_instance.fit(X_sub[train_idx], y_sub[train_idx])

                train_pred = model_instance.predict(X_sub[train_idx])
                test_pred = model_instance.predict(X_sub[test_idx])

                fold_train.append(-np.mean(np.abs(y_sub[train_idx] - train_pred)))
                fold_test.append(-np.mean(np.abs(y_sub[test_idx] - test_pred)))

            train_scores.append((np.mean(fold_train), np.std(fold_train)))
            test_scores.append((np.mean(fold_test), np.std(fold_test)))
            sizes_used.append(n_train)

        return {
            "train_sizes": sizes_used,
            "train_scores": train_scores,
            "test_scores": test_scores,
        }

    @staticmethod
    def check_overfit(
        train_score: float,
        test_score: float,
        threshold: float = 0.1,
    ) -> tuple[bool, float]:
        gap = train_score - test_score
        is_overfit = gap > threshold
        return is_overfit, gap

    @staticmethod
    def ensemble_predict(
        models: list,
        X: np.ndarray,
        weights: list[float] | None = None,
    ) -> np.ndarray:
        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        preds = np.zeros(len(X), dtype=np.float64)
        for model, w in zip(models, weights):
            p = model.predict(X)
            preds += w * np.asarray(p, dtype=np.float64)

        return preds

    @staticmethod
    def calibration_curve(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10,
    ) -> dict:
        bins = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
        bin_means = np.zeros(n_bins)
        bin_true = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins, dtype=np.int32)

        for i in range(n_bins):
            mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
            if i == n_bins - 1:
                mask = (y_pred >= bins[i]) & (y_pred <= bins[i + 1])
            bin_counts[i] = np.sum(mask)
            if bin_counts[i] > 0:
                bin_means[i] = np.mean(y_pred[mask])
                bin_true[i] = np.mean(y_true[mask])

        return {
            "bin_means_pred": bin_means.tolist(),
            "bin_means_true": bin_true.tolist(),
            "bin_counts": bin_counts.tolist(),
        }
