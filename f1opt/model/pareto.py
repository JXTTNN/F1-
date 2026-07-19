"""Multi-objective Pareto optimization + NSGA-II-style setup search.

Provides:

- :class:`ParetoFront` — compute / inspect a Pareto front from multi-objective
  samples (non-dominated sort, hypervolume, NSGA-II crowding distance, knee
  point, summary).
- :class:`MultiObjectiveOptimizer` — multi-objective setup search over the
  normalized ``CarSetup`` space (Latin-hypercube init + tournament / crowding
  selection + SBX crossover + polynomial mutation). Deterministic given seed.
- :func:`lap_time_objective` / :func:`tire_wear_objective` /
  :func:`compound_objective` / :func:`constraint_penalty` — objective helpers
  wrapping the DNN surrogate (:func:`predict_lap_time` / :func:`predict_full`).

The tire-wear proxy mirrors :mod:`f1opt.model.optimizer`:
``(tyre_temp - 90)/30 + slip_angle/5 + tyre_load_spread`` (larger = faster wear).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
    "ParetoFront",
    "MultiObjectiveOptimizer",
    "lap_time_objective",
    "tire_wear_objective",
    "compound_objective",
    "constraint_penalty",
]

# Tire-wear proxy constants (mirror optimizer.py for consistency).
_TYRE_TEMP_REF = 90.0
_TYRE_TEMP_SPAN = 30.0
_SLIP_REF = 5.0


# --------------------------------------------------------------------------- #
# ParetoFront
# --------------------------------------------------------------------------- #
class ParetoFront:
    """Compute and inspect a Pareto front from multi-objective samples.

    Parameters
    ----------
    objectives
        Objective names, e.g. ``["lap_time", "tire_wear"]``.
    maximize
        Per-objective direction flag (``True`` = maximize). ``None`` defaults
        to all-minimize (``False``). Length must match ``objectives``.
    """

    def __init__(
        self,
        objectives: list[str],
        maximize: list[bool] | None = None,
    ) -> None:
        if not objectives:
            raise ValueError("objectives must be non-empty")
        self.objectives: list[str] = list(objectives)
        if maximize is None:
            self.maximize: list[bool] = [False] * len(self.objectives)
        else:
            if len(maximize) != len(objectives):
                raise ValueError("maximize length must match objectives")
            self.maximize = [bool(m) for m in maximize]
        # Each sample: (values: list[float], metadata: dict | None).
        self._samples: list[tuple[list[float], dict | None]] = []

    # ------------------------------------------------------------------ #
    def add_sample(self, values: list[float], metadata: dict | None = None) -> None:
        """Add a sample (objective values + optional metadata)."""
        if len(values) != len(self.objectives):
            raise ValueError(
                f"values length {len(values)} != objectives {len(self.objectives)}"
            )
        self._samples.append(
            ([float(v) for v in values], dict(metadata) if metadata else None)
        )

    # ------------------------------------------------------------------ #
    def _dominates(self, i: int, j: int) -> bool:
        """Return True if sample *i* dominates sample *j*."""
        a = self._samples[i][0]
        b = self._samples[j][0]
        better_all = True
        strict = False
        for k, max_k in enumerate(self.maximize):
            if max_k:
                if a[k] < b[k]:
                    better_all = False
                    break
                if a[k] > b[k]:
                    strict = True
            else:
                if a[k] > b[k]:
                    better_all = False
                    break
                if a[k] < b[k]:
                    strict = True
        return better_all and strict

    # ------------------------------------------------------------------ #
    def compute_front(self) -> list[int]:
        """Return indices of non-dominated samples."""
        n = len(self._samples)
        front: list[int] = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(j, i):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        return front

    # ------------------------------------------------------------------ #
    def compute_front_with_metadata(self) -> list[dict]:
        """Return front members enriched with ``dominated_by_count`` + metadata."""
        front = self.compute_front()
        out: list[dict] = []
        for i in front:
            dominated_by = 0
            for j in range(len(self._samples)):
                if j != i and self._dominates(j, i):
                    dominated_by += 1
            out.append(
                {
                    "index": i,
                    "values": list(self._samples[i][0]),
                    "metadata": self._samples[i][1],
                    "dominated_by_count": dominated_by,
                }
            )
        return out

    # ------------------------------------------------------------------ #
    def hypervolume(self, reference_point: list[float]) -> float:
        """2D/3D hypervolume indicator w.r.t. ``reference_point``.

        For minimization objectives the dominated region is the union of
        hyper-rectangles ``[point, reference]``; for maximization objectives
        the values are negated internally so the same minimization routine
        applies. Points that fail to dominate the reference are ignored.
        """
        if len(self.objectives) not in (2, 3):
            raise ValueError("hypervolume supports only 2D or 3D fronts")
        if len(reference_point) != len(self.objectives):
            raise ValueError("reference_point length must match objectives")
        # Transform to minimization space (negate maximize dims).
        ref = [
            (-float(reference_point[k]) if self.maximize[k] else float(reference_point[k]))
            for k in range(len(self.objectives))
        ]
        pts: list[list[float]] = []
        for vals, _ in self._samples:
            p = [
                (-float(vals[k]) if self.maximize[k] else float(vals[k]))
                for k in range(len(self.objectives))
            ]
            # Point must dominate the reference (<= in all dims) to contribute.
            if all(p[k] <= ref[k] for k in range(len(self.objectives))):
                pts.append(p)
        if not pts:
            return 0.0
        if len(self.objectives) == 2:
            return _hv_2d(pts, ref)
        return _hv_3d(pts, ref)

    # ------------------------------------------------------------------ #
    def crowding_distance(self, front_indices: list[int]) -> dict[int, float]:
        """NSGA-II crowding distance for diversity preservation.

        Edge points (extremes in any objective) receive ``inf``. Returns
        ``{index: distance}``.
        """
        n = len(front_indices)
        if n == 0:
            return {}
        distances: dict[int, float] = {i: 0.0 for i in front_indices}
        if n <= 2:
            for i in front_indices:
                distances[i] = math.inf
            return distances
        for k in range(len(self.objectives)):
            ordered = sorted(front_indices, key=lambda i: self._samples[i][0][k])
            distances[ordered[0]] = math.inf
            distances[ordered[-1]] = math.inf
            vals = [self._samples[i][0][k] for i in ordered]
            span = vals[-1] - vals[0]
            if span <= 0.0:
                continue
            for pos in range(1, n - 1):
                idx = ordered[pos]
                if math.isinf(distances[idx]):
                    continue
                distances[idx] += (vals[pos + 1] - vals[pos - 1]) / span
        return distances

    # ------------------------------------------------------------------ #
    def knee_point(self, front_indices: list[int]) -> int:
        """Identify the knee of the Pareto front (max trade-off point).

        Uses the first two objectives: points are normalized on the front,
        sorted by objective 0, and the interior point with the largest
        perpendicular distance to the line connecting the two extremes is the
        knee. Returns the sample index.
        """
        if not front_indices:
            raise ValueError("cannot find knee of empty front")
        if len(front_indices) == 1:
            return front_indices[0]
        if len(self.objectives) == 1:
            # Single objective: return the best (sort by objective direction).
            ordered = sorted(
                front_indices,
                key=lambda i: self._samples[i][0][0],
                reverse=self.maximize[0],
            )
            return ordered[0]

        k0, k1 = 0, 1
        xs = [self._samples[i][0][k0] for i in front_indices]
        ys = [self._samples[i][0][k1] for i in front_indices]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_span = x_max - x_min if x_max > x_min else 1.0
        y_span = y_max - y_min if y_max > y_min else 1.0
        # Normalize and sort by objective 0 ascending.
        norm = []
        for i in front_indices:
            v = self._samples[i][0]
            nx = (v[k0] - x_min) / x_span
            ny = (v[k1] - y_min) / y_span
            norm.append((nx, ny, i))
        norm.sort(key=lambda t: t[0])
        if len(norm) <= 2:
            return norm[0][2]
        x1, y1, _ = norm[0]
        x2, y2, _ = norm[-1]
        best_idx = norm[0][2]
        best_dist = -1.0
        # Perpendicular distance to the line through the two extremes.
        dx, dy = x2 - x1, y2 - y1
        seg_len = math.hypot(dx, dy)
        for pos in range(1, len(norm) - 1):
            x0, y0, idx = norm[pos]
            if seg_len <= 1e-12:
                continue
            dist = abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / seg_len
            if dist > best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    # ------------------------------------------------------------------ #
    def summary(self) -> dict:
        """Return ``{front_size, total_samples, objectives, knee_index, spread}``."""
        front = self.compute_front()
        spread: dict[str, tuple[float, float]] = {}
        for k, name in enumerate(self.objectives):
            if front:
                vals = [self._samples[i][0][k] for i in front]
                spread[name] = (float(min(vals)), float(max(vals)))
            else:
                spread[name] = (0.0, 0.0)
        knee = self.knee_point(front) if front else -1
        return {
            "front_size": len(front),
            "total_samples": len(self._samples),
            "objectives": list(self.objectives),
            "knee_index": knee,
            "spread": spread,
        }


# --------------------------------------------------------------------------- #
# Pure-python hypervolume helpers (minimization space)
# --------------------------------------------------------------------------- #
def _hv_2d(points: list[list[float]], ref: list[float]) -> float:
    """2D hypervolume (minimization): union of [point, ref] rectangles."""
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    hv = 0.0
    prev_x = pts[0][0]
    min_y = pts[0][1]
    for i, p in enumerate(pts):
        if i > 0:
            hv += (p[0] - prev_x) * (ref[1] - min_y)
        if p[1] < min_y:
            min_y = p[1]
        prev_x = p[0]
    hv += (ref[0] - prev_x) * (ref[1] - min_y)
    return float(hv)


def _hv_3d(points: list[list[float]], ref: list[float]) -> float:
    """3D hypervolume (minimization) via z-sweep with 2D slices."""
    pts = sorted(points, key=lambda p: (p[2], p[0], p[1]))
    ref2d = [ref[0], ref[1]]
    hv = 0.0
    prev_z = pts[0][2]
    front2d: list[list[float]] = [[pts[0][0], pts[0][1]]]
    for i in range(1, len(pts)):
        z_i = pts[i][2]
        hv += _hv_2d(front2d, ref2d) * (z_i - prev_z)
        front2d.append([pts[i][0], pts[i][1]])
        prev_z = z_i
    hv += _hv_2d(front2d, ref2d) * (ref[2] - prev_z)
    return float(hv)


# --------------------------------------------------------------------------- #
# MultiObjectiveOptimizer
# --------------------------------------------------------------------------- #
class MultiObjectiveOptimizer:
    """NSGA-II-style multi-objective setup search over a bounded space.

    Operates in the normalized ``CarSetup`` space (e.g. ``[0,1]^19``). The
    objective evaluation lazy-imports :func:`predict_full` from the surrogate
    module and returns ``[lap_time, tire_wear_proxy]``.
    """

    def __init__(
        self,
        bounds: np.ndarray,
        objectives: list[str],
        n_iterations: int = 20,
        seed: int = 42,
    ) -> None:
        self.bounds = np.asarray(bounds, dtype=np.float64)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must be shape (n_dim, 2)")
        self.objectives = list(objectives)
        self.n_iterations = int(n_iterations)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._pop_size = 12

    # ------------------------------------------------------------------ #
    def evaluate(self, setup_vec: np.ndarray, track_id: str) -> list[float]:
        """Evaluate objectives via the surrogate. Returns ``[lap_time, wear]``."""
        from f1opt.data.setup_schema import CarSetup
        from f1opt.model.surrogate import predict_full

        vec = np.clip(
            np.asarray(setup_vec, dtype=np.float64),
            self.bounds[:, 0],
            self.bounds[:, 1],
        )
        try:
            setup = CarSetup.from_vector(vec.tolist())
            pred = predict_full(setup, track_id, None)
            lap = float(pred["lap_time"])
            resp = pred["responses"]
            wear = (
                (float(resp["tyre_temp"]) - _TYRE_TEMP_REF) / _TYRE_TEMP_SPAN
                + float(resp["slip_angle"]) / _SLIP_REF
                + float(resp["tyre_load_spread"])
            )
        except Exception:
            lap, wear = 90.0, 0.7
        return [lap, wear]

    # ------------------------------------------------------------------ #
    def _lhs_init(self, n_dim: int, n_samples: int) -> list[np.ndarray]:
        """Latin hypercube initialization within ``self.bounds``."""
        perms = [self._rng.permutation(n_samples) for _ in range(n_dim)]
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        span = ub - lb
        samples: list[np.ndarray] = []
        for s in range(n_samples):
            u = self._rng.random(n_dim)
            val = (np.array([perms[d][s] for d in range(n_dim)], dtype=np.float64) + u)
            val /= n_samples
            samples.append(lb + val * span)
        return samples

    # ------------------------------------------------------------------ #
    def _tournament_select(
        self,
        front: ParetoFront,
        population: list,
        k: int = 2,
    ) -> int:
        """Tournament selection: best of ``k`` random members by rank/crowding."""
        n = len(population)
        if n == 0:
            raise ValueError("population is empty")
        contenders = [int(self._rng.integers(0, n)) for _ in range(min(k, n))]

        def rank_of(idx: int) -> tuple[int, float]:
            m = population[idx]
            if isinstance(m, dict):
                return (int(m.get("rank", 0)), -float(m.get("crowding", 0.0)))
            return (0, 0.0)

        return min(contenders, key=rank_of)

    # ------------------------------------------------------------------ #
    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Simplified simulated binary crossover (SBX)."""
        p1 = np.asarray(p1, dtype=np.float64)
        p2 = np.asarray(p2, dtype=np.float64)
        eta = 15.0
        u = self._rng.random(p1.shape[0])
        beta = np.where(
            u <= 0.5,
            (2.0 * u) ** (1.0 / (eta + 1.0)),
            (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0)),
        )
        child = 0.5 * ((p1 + p2) - beta * (p2 - p1))
        return np.clip(child, self.bounds[:, 0], self.bounds[:, 1])

    # ------------------------------------------------------------------ #
    def _mutate(self, vec: np.ndarray, prob: float = 0.1) -> np.ndarray:
        """Simplified polynomial mutation. Guarantees >=1 change when prob>0."""
        vec = np.asarray(vec, dtype=np.float64).copy()
        n = vec.shape[0]
        if n == 0:
            return vec
        mask = self._rng.random(n) < prob
        if prob > 0.0:
            if not mask.any():
                mask[int(self._rng.integers(0, n))] = True
            if mask.all() and n > 1:
                mask[int(self._rng.integers(0, n))] = False
        eta = 20.0
        span = self.bounds[:, 1] - self.bounds[:, 0]
        for i in range(n):
            if mask[i]:
                delta = 2.0 * self._rng.random() - 1.0
                # Floor the magnitude so a masked entry always actually changes.
                mag = abs(delta) ** (1.0 / (eta + 1.0))
                if mag < 0.05:
                    mag = 0.05
                delta_q = np.sign(delta) * mag
                vec[i] = vec[i] + delta_q * span[i] * 0.1
        return np.clip(vec, self.bounds[:, 0], self.bounds[:, 1])

    # ------------------------------------------------------------------ #
    def search(self, track_id: str, driver_profile: Any = None) -> dict:
        """Run NSGA-II-style search. Returns dict with setups + history."""
        from f1opt.data.setup_schema import CarSetup

        n_dim = self.bounds.shape[0]
        population = self._lhs_init(n_dim, self._pop_size)
        history: list[dict] = []
        best_lap_vec: np.ndarray | None = None
        best_lap_val = math.inf
        best_wear_vec: np.ndarray | None = None
        best_wear_val = math.inf

        def update_bests(vec: np.ndarray, vals: list[float]) -> None:
            nonlocal best_lap_vec, best_lap_val, best_wear_vec, best_wear_val
            if vals[0] < best_lap_val:
                best_lap_val = vals[0]
                best_lap_vec = vec.copy()
            if vals[1] < best_wear_val:
                best_wear_val = vals[1]
                best_wear_vec = vec.copy()

        for it in range(self.n_iterations):
            values_list = [self.evaluate(v, track_id) for v in population]
            for v, vals in zip(population, values_list, strict=True):
                update_bests(v, vals)
            front = ParetoFront(self.objectives)
            for vals in values_list:
                front.add_sample(vals)
            front_idx = front.compute_front()
            cd = front.crowding_distance(front_idx)
            front_set = set(front_idx)
            members: list[dict] = []
            for i, (v, vals) in enumerate(zip(population, values_list, strict=True)):
                members.append(
                    {
                        "vec": v,
                        "values": vals,
                        "rank": 0 if i in front_set else 1,
                        "crowding": cd.get(i, 0.0),
                    }
                )
            history.append(
                {
                    "iter": it,
                    "front_size": len(front_idx),
                    "best_lap": float(best_lap_val),
                    "best_wear": float(best_wear_val),
                }
            )
            offspring: list[np.ndarray] = []
            while len(offspring) < self._pop_size:
                i1 = self._tournament_select(front, members, k=2)
                i2 = self._tournament_select(front, members, k=2)
                child = self._crossover(members[i1]["vec"], members[i2]["vec"])
                offspring.append(self._mutate(child, prob=0.1))
            population = offspring

        # Final evaluation of the last population.
        values_list = [self.evaluate(v, track_id) for v in population]
        for v, vals in zip(population, values_list, strict=True):
            update_bests(v, vals)
        final_front = ParetoFront(self.objectives)
        for vals in values_list:
            final_front.add_sample(vals)
        front_idx = final_front.compute_front()
        knee = final_front.knee_point(front_idx) if front_idx else 0
        knee_vec = population[knee] if knee < len(population) else population[0]

        if best_lap_vec is None:
            best_lap_vec = population[0]
            best_lap_val = self.evaluate(population[0], track_id)[0]
        if best_wear_vec is None:
            best_wear_vec = population[0]
            best_wear_val = self.evaluate(population[0], track_id)[1]

        def to_setup(arr: np.ndarray) -> CarSetup:
            clipped = np.clip(np.asarray(arr, dtype=np.float64), 0.0, 1.0)
            return CarSetup.from_vector(clipped.tolist())

        return {
            "pareto_front": final_front,
            "best_lap_time_setup": to_setup(best_lap_vec),
            "best_tire_wear_setup": to_setup(best_wear_vec),
            "knee_setup": to_setup(knee_vec),
            "history": history,
            "iterations": self.n_iterations,
        }


# --------------------------------------------------------------------------- #
# Objective function helpers
# --------------------------------------------------------------------------- #
def lap_time_objective(setup: Any, track_id: str, driver_profile: Any = None) -> float:
    """Lap-time objective (minimize): wraps :func:`predict_lap_time`."""
    from f1opt.model.surrogate import predict_lap_time

    return float(predict_lap_time(setup, track_id, driver_profile))


def tire_wear_objective(setup: Any, track_id: str, driver_profile: Any = None) -> float:
    """Tire-wear proxy objective (minimize): wraps :func:`predict_full` responses."""
    from f1opt.model.surrogate import predict_full

    pred = predict_full(setup, track_id, driver_profile)
    resp = pred["responses"]
    return float(
        (float(resp["tyre_temp"]) - _TYRE_TEMP_REF) / _TYRE_TEMP_SPAN
        + float(resp["slip_angle"]) / _SLIP_REF
        + float(resp["tyre_load_spread"])
    )


def compound_objective(
    setup: Any,
    track_id: str,
    driver_profile: Any,
    weights: list[float],
) -> float:
    """Weighted-sum compound objective: ``w0*lap + w1*wear``."""
    lap = lap_time_objective(setup, track_id, driver_profile)
    wear = tire_wear_objective(setup, track_id, driver_profile)
    return float(weights[0]) * lap + float(weights[1]) * wear


def _parse_constraint(spec: Any) -> tuple[str, float]:
    """Parse a constraint spec into ``(op, limit)``.

    A bare number implies ``"lt"`` (must stay below the limit). A
    ``(op, limit)`` tuple is accepted with ``op`` in
    ``{"<","<=","lt","le",">",">=","gt","ge"}``.
    """
    if isinstance(spec, tuple | list) and len(spec) == 2:
        op = str(spec[0]).lower()
        limit = float(spec[1])
        return op, limit
    return "lt", float(spec)


def _violation(actual: float, op: str, limit: float) -> float:
    """Return non-negative violation magnitude (0 if satisfied)."""
    if op in ("<", "<=", "lt", "le"):
        return max(0.0, actual - limit)
    if op in (">", ">=", "gt", "ge"):
        return max(0.0, limit - actual)
    raise ValueError(f"unsupported constraint op: {op!r}")


def constraint_penalty(setup: Any, track_id: str, constraints: dict) -> float:
    """Sum of constraint violations against surrogate responses.

    ``constraints`` maps a response name (e.g. ``"tyre_temp"``) to either a
    number (implied ``< limit``) or an ``(op, limit)`` tuple. Returns 0 if all
    constraints are satisfied, else the total violation magnitude.
    """
    if not constraints:
        return 0.0
    from f1opt.model.surrogate import predict_full

    pred = predict_full(setup, track_id, None)
    resp = pred["responses"]
    total = 0.0
    for key, spec in constraints.items():
        actual = float(resp.get(key, 0.0))
        op, limit = _parse_constraint(spec)
        total += _violation(actual, op, limit)
    return float(total)
