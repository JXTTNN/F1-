from __future__ import annotations

import numpy as np
from typing import Sequence


class FeatureImportance:

    @staticmethod
    def permutation_importance(
        model,
        X: np.ndarray,
        y: np.ndarray,
        metric_fn=None,
        n_repeats: int = 5,
        seed: int = 42,
    ) -> dict[str, np.ndarray]:
        if metric_fn is None:
            def metric_fn(a, b):
                return -np.mean(np.abs(a - b))

        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        baseline_pred = model.predict(X)
        baseline_score = metric_fn(y, baseline_pred)

        n_features = X.shape[1]
        importances = np.zeros((n_repeats, n_features), dtype=np.float64)

        for rep in range(n_repeats):
            X_permuted = X.copy()
            for j in range(n_features):
                rng.shuffle(X_permuted[:, j])
                perm_pred = model.predict(X_permuted)
                perm_score = metric_fn(y, perm_pred)
                importances[rep, j] = baseline_score - perm_score
                X_permuted[:, j] = X[:, j]

        mean_imp = np.mean(importances, axis=0)
        std_imp = np.std(importances, axis=0)

        return {
            "importances_mean": mean_imp,
            "importances_std": std_imp,
            "baseline_score": baseline_score,
        }

    @staticmethod
    def top_features(
        importance_result: dict[str, np.ndarray],
        feature_names: list[str] | None = None,
        top_k: int = 10,
    ) -> list[tuple[str, float, float]]:
        mean_imp = importance_result["importances_mean"]
        std_imp = importance_result.get("importances_std", np.zeros_like(mean_imp))

        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(mean_imp))]

        ranked = sorted(
            zip(feature_names, mean_imp, std_imp),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked[:top_k]

    @staticmethod
    def correlation_matrix(
        X: np.ndarray,
        threshold: float = 0.9,
    ) -> list[tuple[int, int, float]]:
        corr = np.corrcoef(X.T)
        n = corr.shape[0]
        high_corr = []
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr[i, j]) > threshold:
                    high_corr.append((int(i), int(j), float(corr[i, j])))
        return high_corr
