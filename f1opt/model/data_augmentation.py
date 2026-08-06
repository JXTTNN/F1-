from __future__ import annotations

import numpy as np
from typing import Sequence
from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    noise_std: float = 0.01
    jitter_scale: float = 0.005
    mixup_alpha: float = 0.2
    shift_range: float = 0.02


class TimeSeriesAugmenter:

    def __init__(self, config: AugmentationConfig | None = None, seed: int = 42) -> None:
        self.config = config or AugmentationConfig()
        self.rng = np.random.default_rng(seed)

    def add_noise(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        noise = self.rng.normal(0, self.config.noise_std, X.shape)
        X_aug = X + noise
        return X_aug, y

    def jitter(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        jitter = self.rng.uniform(-self.config.jitter_scale, self.config.jitter_scale, X.shape)
        return X + jitter, y

    def scaling(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        scales = self.rng.uniform(0.9, 1.1, (X.shape[0], 1))
        return X * scales, y

    def shift(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        shift = self.rng.uniform(-self.config.shift_range, self.config.shift_range, (1, X.shape[1]))
        return X + shift, y

    def mixup(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        n = len(X)
        lam = self.rng.beta(self.config.mixup_alpha, self.config.mixup_alpha, n)
        idx = self.rng.permutation(n)
        X_mix = lam.reshape(-1, 1) * X + (1 - lam.reshape(-1, 1)) * X[idx]
        if y is not None:
            y_mix = lam * y + (1 - lam) * y[idx]
            return X_mix, y_mix
        return X_mix, None

    def augment(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        methods: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if methods is None:
            methods = ["noise", "jitter", "scaling", "shift"]

        augmented = []
        labels = [] if y is not None else None

        for method in methods:
            if method == "noise":
                x_a, y_a = self.add_noise(X, y)
            elif method == "jitter":
                x_a, y_a = self.jitter(X, y)
            elif method == "scaling":
                x_a, y_a = self.scaling(X, y)
            elif method == "shift":
                x_a, y_a = self.shift(X, y)
            elif method == "mixup":
                if y is not None:
                    x_a, y_a = self.mixup(X, y)
                else:
                    x_a, y_a = self.mixup_self(X)
            else:
                continue

            augmented.append(x_a)
            if y is not None:
                labels.append(y_a)

        X_aug = np.vstack([X] + augmented)
        y_aug = np.concatenate([y] + labels) if y is not None else None

        return X_aug, y_aug

    def mixup_self(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        return self.mixup(X, None)
