from __future__ import annotations

import numpy as np
from typing import Callable


def clip_population(population: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.clip(population, bounds[:, 0], bounds[:, 1])


def mutate_rand_1(
    population: np.ndarray,
    bounds: np.ndarray,
    F: float,
    rng: np.random.Generator,
) -> np.ndarray:
    pop_size, dim = population.shape
    idx_a = rng.choice(pop_size, size=pop_size, replace=True)
    idx_b = rng.choice(pop_size, size=pop_size, replace=True)
    idx_c = rng.choice(pop_size, size=pop_size, replace=True)

    mask_eq_a = (idx_b == idx_a) | (idx_c == idx_a) | (idx_b == idx_c)
    while mask_eq_a.any():
        idx_b[mask_eq_a] = rng.choice(pop_size, size=mask_eq_a.sum(), replace=True)
        idx_c[mask_eq_a] = rng.choice(pop_size, size=mask_eq_a.sum(), replace=True)
        mask_eq_a = (idx_b == idx_a) | (idx_c == idx_a) | (idx_b == idx_c)

    donor = (
        population[idx_a]
        + F * (population[idx_b] - population[idx_c])
    )
    return clip_population(donor, bounds)


def binomial_crossover(
    population: np.ndarray,
    donor: np.ndarray,
    CR: float,
    rng: np.random.Generator,
) -> np.ndarray:
    pop_size, dim = population.shape
    cross_points = rng.random((pop_size, dim)) < CR
    j_rand = rng.integers(0, dim, size=pop_size)
    cross_points[np.arange(pop_size), j_rand] = True
    trial = np.where(cross_points, donor, population)
    return trial


def tournament_selection(
    population: np.ndarray,
    trial: np.ndarray,
    fitness: np.ndarray,
    trial_fitness: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    improved = trial_fitness < fitness
    pop_new = population.copy()
    fit_new = fitness.copy()
    pop_new[improved] = trial[improved]
    fit_new[improved] = trial_fitness[improved]
    return pop_new, fit_new


def compute_constraint_penalty(
    population: np.ndarray,
    bounds: np.ndarray,
    penalty_weight: float = 1e6,
) -> np.ndarray:
    lower_violation = np.maximum(0, bounds[:, 0] - population).sum(axis=1)
    upper_violation = np.maximum(0, population - bounds[:, 1]).sum(axis=1)
    return penalty_weight * (lower_violation + upper_violation)


def de_one_generation(
    population: np.ndarray,
    bounds: np.ndarray,
    fitness: np.ndarray,
    F: float,
    CR: float,
    rng: np.random.Generator,
    evaluate_fn,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    donor = mutate_rand_1(population, bounds, F, rng)
    trial = binomial_crossover(population, donor, CR, rng)

    trial_fitness = evaluate_fn(trial)
    if isinstance(trial_fitness, list):
        trial_fitness = np.array(trial_fitness, dtype=np.float64)

    new_pop, new_fit = tournament_selection(
        population=population,
        trial=trial,
        fitness=fitness,
        trial_fitness=trial_fitness,
    )

    best_val = float(np.min(new_fit))
    mean_val = float(np.mean(new_fit))

    return new_pop, new_fit, best_val, mean_val
