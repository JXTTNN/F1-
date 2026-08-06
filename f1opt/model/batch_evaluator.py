from __future__ import annotations

import torch
import numpy as np
from typing import Sequence
from dataclasses import dataclass


@dataclass
class EvalBatch:
    X: torch.Tensor
    pop_idx: np.ndarray


BATCH_SIZE = 256


def batched_inference(
    surrogate: torch.nn.Module,
    population: np.ndarray,
    device: str = "cuda",
) -> np.ndarray:
    if len(population) == 0:
        return np.array([], dtype=np.float32)

    if hasattr(surrogate, "to"):
        surrogate = surrogate.to(device)

    x = torch.from_numpy(population.astype(np.float32)).to(device)
    n = x.shape[0]
    preds = np.empty(n, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            batch = x[start:end]
            y = surrogate(batch).squeeze(-1)
            preds[start:end] = y.cpu().numpy()

    return preds


def evaluate_population(
    surrogate: torch.nn.Module,
    population: np.ndarray,
    constraint_fn=None,
    device: str = "cuda",
) -> tuple[np.ndarray, float, float]:
    fitness = batched_inference(surrogate, population, device)

    if constraint_fn is not None:
        penalty = constraint_fn(population)
        fitness = fitness - penalty

    best = np.min(fitness)
    mean = np.mean(fitness)
    return fitness, best, mean


def compute_pairwise_distances(
    population: np.ndarray,
    subset_size: int = 128,
) -> np.ndarray:
    from scipy.spatial.distance import pdist, squareform

    if len(population) <= subset_size:
        return squareform(pdist(population))

    rng = np.random.default_rng()
    idx = rng.choice(len(population), subset_size, replace=False)
    return squareform(pdist(population[idx]))
