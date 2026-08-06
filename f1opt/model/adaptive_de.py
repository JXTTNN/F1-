from __future__ import annotations

import numpy as np


def compute_population_diversity(population: np.ndarray, bounds: np.ndarray) -> float:
    range_widths = bounds[:, 1] - bounds[:, 0]
    range_widths = np.where(range_widths == 0, 1.0, range_widths)

    pop_range = np.max(population, axis=0) - np.min(population, axis=0)
    normalized = np.clip(pop_range / range_widths, 0.0, 1.0)

    return float(np.mean(normalized))


def adaptive_mutation_factor(
    generation: int,
    max_generations: int,
    diversity: float,
    f_min: float = 0.2,
    f_max: float = 1.0,
) -> float:
    progress = generation / max(max_generations, 1)

    base_f = f_max + (f_min - f_max) * progress ** 0.5

    if diversity < 0.1:
        base_f = min(base_f + 0.3, f_max)
    elif diversity > 0.7:
        base_f = max(base_f - 0.1, f_min)

    return float(np.clip(base_f, f_min, f_max))


def adaptive_crossover_rate(
    generation: int,
    max_generations: int,
    diversity: float,
    cr_min: float = 0.1,
    cr_max: float = 0.9,
) -> float:
    progress = min(generation / max(max_generations, 1), 1.0)

    base_cr = cr_min + (cr_max - cr_min) * (1.0 - progress) ** 0.3

    if diversity < 0.1:
        base_cr = min(base_cr + 0.2, cr_max)
    elif diversity > 0.7:
        base_cr = max(base_cr - 0.15, cr_min)

    return float(np.clip(base_cr, cr_min, cr_max))


def adaptive_strategy(
    generation: int,
    max_generations: int,
    population: np.ndarray,
    bounds: np.ndarray,
    pop_size: int,
) -> tuple[float, float, str]:
    diversity = compute_population_diversity(population, bounds)
    F = adaptive_mutation_factor(generation, max_generations, diversity)
    CR = adaptive_crossover_rate(generation, max_generations, diversity)

    if diversity < 0.05:
        strategy = "restart"
    elif diversity < 0.15:
        strategy = "exploit"
    elif diversity > 0.6 or generation < max_generations * 0.2:
        strategy = "explore"
    else:
        strategy = "balance"

    return F, CR, strategy
