from __future__ import annotations

import numpy as np
from typing import Sequence


class OutlierDetector:

    @staticmethod
    def iqr_outliers(
        X: np.ndarray,
        factor: float = 1.5,
    ) -> np.ndarray:
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        iqr = q75 - q25
        lower = q25 - factor * iqr
        upper = q75 + factor * iqr
        return ((X < lower) | (X > upper)).any(axis=1)

    @staticmethod
    def zscore_outliers(
        X: np.ndarray,
        threshold: float = 3.0,
    ) -> np.ndarray:
        z = np.abs((X - X.mean(axis=0)) / np.where(X.std(axis=0, ddof=1) == 0, 1.0, X.std(axis=0, ddof=1)))
        return (z > threshold).any(axis=1)

    @staticmethod
    def isolation_forest_outliers(
        X: np.ndarray,
        contamination: float = 0.1,
        n_estimators: int = 100,
        seed: int = 42,
    ) -> np.ndarray:
        from sklearn.ensemble import isolationforest

        clf = isolationforest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1,
        )
        pred = clf.fit_predict(X)
        return pred == -1

    @staticmethod
    def mahalanobis_distance(
        X: np.ndarray,
    ) -> np.ndarray:
        mean_vec = X.mean(axis=0)
        X_centered = X - mean_vec
        cov = np.cov(X_centered.T)

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_reg = cov + np.eye(cov.shape[0]) * 1e-6
            cov_inv = np.linalg.inv(cov_reg)

        m_dist = np.zeros(len(X))
        for i in range(len(X)):
            m_dist[i] = np.sqrt(X_centered[i].T @ cov_inv @ X_centered[i])

        return m_dist

    @staticmethod
    def detect(
        X: np.ndarray,
        method: str = "iqr",
        **kwargs,
    ) -> np.ndarray:
        methods = {
            "iqr": lambda: OutlierDetector.iqr_outliers(X, **kwargs),
            "zscore": lambda: OutlierDetector.zscore_outliers(X, **kwargs),
            "isolation_forest": lambda: OutlierDetector.isolation_forest_outliers(X, **kwargs),
        }
        if method not in methods:
            return np.zeros(len(X), dtype=bool)
        return methods[method]()
