from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class PredictionInterval:
    mean: float
    lower: float
    upper: float
    confidence: float = 0.95


class BootstrapInterval:

    def __init__(
        self,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: int | None = 42,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self.rng = np.random.default_rng(seed)

    def compute_intervals(
        self,
        predictions: np.ndarray,
        residuals: np.ndarray | None = None,
    ) -> np.ndarray:
        predictions = np.asarray(predictions, dtype=np.float64)
        n = len(predictions)

        if residuals is None:
            residuals = predictions - np.mean(predictions)
        residuals = np.asarray(residuals, dtype=np.float64)

        alpha = 1.0 - self.confidence
        lower_percentile = alpha / 2.0
        upper_percentile = 1.0 - lower_percentile

        boot_means = np.empty(self.n_bootstrap, dtype=np.float64)
        for i in range(self.n_bootstrap):
            idx = self.rng.choice(n, size=n, replace=True)
            boot_means[i] = float(np.mean(predictions[idx] + self.rng.choice(residuals, size=n, replace=True)))

        interval_center = np.mean(predictions)
        residual_std = np.std(residuals, ddof=1) if len(residuals) > 1 else 0.0

        from scipy import stats as sp_stats
        z_score = sp_stats.norm.ppf(upper_percentile)

        intervals = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            margin = z_score * residual_std
            intervals[i] = [predictions[i], predictions[i] - margin, predictions[i] + margin]

        return intervals

    def prediction_interval(
        self,
        prediction: float,
        residual_std: float,
    ) -> PreditionInterval:
        from scipy import stats as sp_stats
        alpha = 1.0 - self.confidence
        z_score = sp_stats.norm.ppf(1.0 - alpha / 2.0)
        margin = z_score * residual_std

        return PreditionInterval(
            mean=prediction,
            lower=prediction - margin,
            upper=prediction + margin,
            confidence=self.confidence,
        )
