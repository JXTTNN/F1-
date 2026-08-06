from __future__ import annotations

import numpy as np
from typing import Callable


def k_fold_split(
    n_samples: int,
    k: int = 5,
    shuffle: bool = True,
    seed: int | None = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    fold_size = n_samples // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n_samples
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_idx, test_idx))

    return folds


def stratified_split(
    y: np.ndarray,
    k: int = 5,
    seed: int | None = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(y)
    rng = np.random.default_rng(seed)

    n_bins = min(k * 2, n // 4)
    if n_bins < 2:
        return r_fold_split(np.arange(n), k, False, seed)

    bin_edges = np.percentile(y, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(y, bin_edges[:-1])

    folds = [[] for _ in range(k)]
    for bin_id in range(1, n_bins + 1):
        bin_members = np.where(bin_indices == bin_id)[0]
        if len(bin_members) == 0:
            continue
        rng.shuffle(bin_members)
        for j, idx in enumerate(bin_members):
            folds[j % k].append(idx)

    result = []
    fold_arrs = [np.array(f, dtype=np.intp) for f in folds]
    for i in range(k):
        test_idx = fold_arrs[i]
        train_idx = np.concatenate([fold_arrs[j] for j in range(k) if j != i])
        result.append((train_idx, test_idx))

    return result


def time_series_split(
    n_samples: int,
    k: int = 5,
    gap: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(n_samples)
    folds = []
    for i in range(k):
        split_point = n_samples - (k - i) * (n_samples // (k + 1))
        test_size = n_samples // (k + 1)
        train_end = split_point - gap
        test_start = split_point
        test_end = min(test_start + test_size, n_samples)

        if train_end <= 0 or test_start >= n_samples:
            continue

        folds.append((indices[:train_end], indices[test_start:test_end]))

    return folds
