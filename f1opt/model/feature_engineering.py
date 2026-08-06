from __future__ import annotations

import numpy as np
from typing import Sequence


class FeatureEngineer:

    @staticmethod
    def normalize(X: np.ndarray, method: str = "minmax") -> tuple[np.ndarray, dict]:
        if method == "minmax":
            x_min = X.min(axis=0)
            x_max = X.max(axis=0)
            denom = np.where(x_max - x_min == 0, 1.0, x_max - x_min)
            X_scaled = (X - x_min) / denom
            params = {"min": x_min, "max": x_max}

        elif method == "zscore":
            x_mean = X.mean(axis=0)
            x_std = X.std(axis=0, ddof=1)
            x_std = np.where(x_std == 0, 1.0, x_std)
            X_scaled = (X - x_mean) / x_std
            params = {"mean": x_mean, "std": x_std}

        elif method == "robust":
            q25 = np.percentile(X, 25, axis=0)
            q75 = np.percentile(X, 75, axis=0)
            iqr = q75 - q25
            iqr = np.where(iqr == 0, 1.0, iqr)
            X_scaled = (X - q25) / iqr
            params = {"q25": q25, "q75": q75, "iqr": iqr}

        else:
            X_scaled = X.copy()
            params = {}

        return X_scaled, params

    def apply_normalize(
        self,
        X: np.ndarray,
        params: dict,
        method: str = "minmax",
    ) -> np.ndarray:
        if method == "minmax":
            denom = np.where(params["max"] - params["min"] == 0, 1.0, params["max"] - params["min"])
            return (X - params["min"]) / denom
        elif method == "zscore":
            return (X - params["mean"]) / params["std"]
        elif method == "robust":
            return (X - params["q25"]) / params["iqr"]
        return X.copy()

    @staticmethod
    def polynomial_features(
        X: np.ndarray,
        degree: int = 2,
        interaction_only: bool = False,
    ) -> np.ndarray:
        n_samples, n_features = X.shape

        combos = []
        for d in range(1, degree + 1):
            if d == 1:
                combos.append(X)
            elif interaction_only:
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        combos.append((X[:, i] * X[:, j]).reshape(-1, 1))
            else:
                for i in range(n_features):
                    for j in range(i, n_features):
                        combos.append((X[:, i] * X[:, j]).reshape(-1, 1))

        return np.hstack(combos)

    @staticmethod
    def lag_features(
        X: np.ndarray,
        lags: list[int] | None = None,
    ) -> np.ndarray:
        if lags is None:
            lags = [1, 2, 3, 5]

        n_samples, n_features = X.shape
        max_lag = max(lags)

        if n_samples <= max_lag:
            return X

        result = X[max_lag:].copy()
        lag_arrays = []

        for lag in lags:
            lagged = X[max_lag - lag : n_samples - lag]
            lag_arrays.append(lagged)

        return np.hstack([result] + lag_arrays)

    @staticmethod
    def rolling_statistics(
        X: np.ndarray,
        window: int = 5,
    ) -> np.ndarray:
        n = len(X)
        if n <= window:
            return X

        result = X[window:].copy()
        mean_win = np.array([
            np.mean(X[i - window:i], axis=0)
            for i in range(window, n)
        ])
        std_win = np.array([
            np.std(X[i - window:i], axis=0, ddof=1)
            for i in range(window, n)
        ])

        return np.hstack([result, mean_win, std_win])

    @staticmethod
    def select_k_best(
        X: np.ndarray,
        y: np.ndarray,
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_features = X.shape[1]
        correlations = np.array([
            np.corrcoef(X[:, i], y)[0, 1] for i in range(n_features)
        ])

        top_indices = np.argsort(np.abs(correlations))[-k:]
        return X[:, top_indices], top_indices
