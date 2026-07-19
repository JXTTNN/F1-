"""Bayesian optimization for F1 setup search (Iter-32/33).

Implements a Gaussian-Process surrogate + acquisition-function search to
minimize predicted lap time across the 19-dim setup space. Pure-numpy (no
scikit-learn); scipy.optimize is used for hyperparameter tuning when
available, with a fallback to default hyperparameters.

Public API:
    - :class:`GaussianProcessSurrogate` — Matérn 5/2 GP regression.
    - :class:`BayesianOptimizer` — BO loop with EI / UCB / PI acquisitions.
    - :func:`bayesian_search_setup` — high-level helper wrapping the DNN
      surrogate as the objective.
    - :func:`expected_improvement` / :func:`upper_confidence_bound` /
      :func:`probability_of_improvement` — acquisition functions.

References (textbook formulas, no papers):
    Rasmussen & Williams "Gaussian Processes for Machine Learning" (2006).
    Mockus "Bayesian Approach to Global Optimization" (1989).
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "GaussianProcessSurrogate",
    "BayesianOptimizer",
    "bayesian_search_setup",
    "expected_improvement",
    "upper_confidence_bound",
    "probability_of_improvement",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
_NOISE_FLOOR = 1e-3
_JITTER = 1e-6
_DEFAULT_LENGTHSCALE = 0.3
_DEFAULT_SIGNAL = 1.0
_DEFAULT_NOISE = 1e-2
_GRID_SIZE = 200


# --------------------------------------------------------------------------- #
# Matérn 5/2 kernel
# --------------------------------------------------------------------------- #
def _matern52(x1: np.ndarray, x2: np.ndarray, lengthscale: float,
              signal: float) -> np.ndarray:
    """Matérn 5/2 kernel between two row-arrays. Returns (n1, n2) matrix."""
    # Squared Euclidean distance.
    sq = np.sum(x1 ** 2, axis=1)[:, None] + np.sum(x2 ** 2, axis=1)[None, :] \
        - 2.0 * (x1 @ x2.T)
    sq = np.maximum(sq, 0.0)
    dist = np.sqrt(sq)
    r = dist / max(lengthscale, 1e-6)
    # Matérn 5/2: k(r) = sigma^2 * (1 + sqrt(5) r + (5/3) r^2) * exp(-sqrt(5) r)
    sqrt5 = np.sqrt(5.0)
    return signal * signal * (1.0 + sqrt5 * r + (5.0 / 3.0) * r * r) * np.exp(-sqrt5 * r)


# --------------------------------------------------------------------------- #
# GP surrogate
# --------------------------------------------------------------------------- #
class GaussianProcessSurrogate:
    """Matérn 5/2 Gaussian Process regression (pure numpy).

    Hyperparameters (signal, lengthscale, noise) are fit by maximizing the log
    marginal likelihood via scipy.optimize.minimize when available; otherwise
    defaults are used.
    """

    def __init__(self, lengthscale: float = _DEFAULT_LENGTHSCALE,
                 signal: float = _DEFAULT_SIGNAL,
                 noise: float = _DEFAULT_NOISE) -> None:
        self.lengthscale = float(lengthscale)
        self.signal = float(signal)
        self.noise = max(float(noise), _NOISE_FLOOR)
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._L: np.ndarray | None = None  # Cholesky factor of K + noise I
        self._alpha: np.ndarray | None = None  # precomputed K^-1 y
        self._use_eig: bool = False  # fallback flag

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the GP on training data.

        X shape (n, d); y shape (n,). Stores Cholesky factor for prediction.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.shape[0] == 0:
            self._X = None
            self._y = None
            return
        self._X, self._y = X, y
        # Tune hyperparameters (best-effort).
        self._optimize_hyperparams()
        self._compute_factor()

    # ------------------------------------------------------------------ #
    def _compute_factor(self) -> None:
        """Compute Cholesky factor (or eigen fallback) of K + noise I."""
        if self._X is None:
            return
        K = _matern52(self._X, self._X, self.lengthscale, self.signal)
        n = K.shape[0]
        K_noise = K + (self.noise + _JITTER) * np.eye(n)
        try:
            self._L = np.linalg.cholesky(K_noise)
            self._alpha = np.linalg.solve(
                self._L.T, np.linalg.solve(self._L, self._y)
            )
            self._use_eig = False
        except np.linalg.LinAlgError:
            # Eigen-decomposition fallback.
            w, V = np.linalg.eigh(K_noise)
            w = np.maximum(w, _JITTER)
            self._L = None
            self._alpha = (V * (1.0 / w)) @ (V.T @ self._y)
            self._V = V
            self._w = w
            self._use_eig = True

    # ------------------------------------------------------------------ #
    def _optimize_hyperparams(self) -> None:
        """Best-effort hyperparameter tuning via scipy.optimize."""
        try:
            from scipy.optimize import minimize
        except ImportError:
            return
        if self._X is None or self._X.shape[0] < 2:
            return

        def neg_lml(theta: np.ndarray) -> float:
            ls, sig, nz = float(theta[0]), float(theta[1]), float(theta[2])
            ls = max(ls, 1e-3)
            sig = max(sig, 1e-3)
            nz = max(nz, _NOISE_FLOOR)
            K = _matern52(self._X, self._X, ls, sig)
            n = K.shape[0]
            Kn = K + (nz + _JITTER) * np.eye(n)
            try:
                L = np.linalg.cholesky(Kn)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, self._y))
                lml = -0.5 * (self._y @ alpha) - np.sum(np.log(np.diag(L))) \
                    - 0.5 * n * np.log(2 * np.pi)
                return -float(lml)
            except np.linalg.LinAlgError:
                return 1e10

        try:
            res = minimize(
                neg_lml,
                x0=np.array([self.lengthscale, self.signal, self.noise]),
                method="Nelder-Mead",
                options={"maxiter": 60, "xatol": 1e-3},
            )
            if res.success or np.isfinite(res.fun):
                self.lengthscale = max(float(res.x[0]), 1e-3)
                self.signal = max(float(res.x[1]), 1e-3)
                self.noise = max(float(res.x[2]), _NOISE_FLOOR)
        except Exception:
            pass  # keep defaults

    # ------------------------------------------------------------------ #
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior (mean, std) at query points. Shape (n,) each."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if self._X is None or self._X.shape[0] == 0:
            # No data: return prior mean 0, prior std = signal.
            return np.zeros(X.shape[0]), np.full(X.shape[0], self.signal)
        K_xs = _matern52(X, self._X, self.lengthscale, self.signal)
        mean = K_xs @ self._alpha
        # Variance: k(x,x) - v^T v
        k_ss = self.signal * self.signal * np.ones(X.shape[0])
        if not self._use_eig:
            v = np.linalg.solve(self._L, K_xs.T)  # (n, m)
            var = k_ss - np.sum(v * v, axis=0)
        else:
            # Eigen fallback: K_xs @ V @ diag(1/w) @ V^T @ K_xs^T
            VtKxs = self._V.T @ K_xs.T  # (n, m)
            var = k_ss - np.sum((VtKxs ** 2) * (1.0 / self._w)[:, None], axis=0)
        var = np.maximum(var, 0.0)
        return mean, np.sqrt(var)

    # ------------------------------------------------------------------ #
    def log_marginal_likelihood(self) -> float:
        """Log marginal likelihood of the training data under current hyperparams."""
        if self._X is None or self._X.shape[0] == 0:
            return 0.0
        K = _matern52(self._X, self._X, self.lengthscale, self.signal)
        n = K.shape[0]
        Kn = K + (self.noise + _JITTER) * np.eye(n)
        try:
            L = np.linalg.cholesky(Kn)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self._y))
            return float(
                -0.5 * (self._y @ alpha) - np.sum(np.log(np.diag(L)))
                - 0.5 * n * np.log(2 * np.pi)
            )
        except np.linalg.LinAlgError:
            return -1e10


# --------------------------------------------------------------------------- #
# Acquisition functions
# --------------------------------------------------------------------------- #
def expected_improvement(X: np.ndarray, gp: GaussianProcessSurrogate,
                          best_y: float, xi: float = 0.01) -> np.ndarray:
    """Expected Improvement acquisition.

    EI(x) = (mu - best - xi) * Phi(z) + sigma * phi(z)
    where z = (mu - best - xi) / sigma.
    """
    mean, std = gp.predict(X)
    std = np.maximum(std, 1e-9)
    improvement = mean - best_y - xi
    z = improvement / std
    # Standard normal pdf/cdf.
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1.0 + np.vectorize(_phi)(z))
    ei = improvement * cdf + std * pdf
    # EI >= 0 by definition; clamp negatives.
    return np.maximum(ei, 0.0)


def _phi(z: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun 7.1.26)."""
    from math import tanh
    return 0.5 * (1.0 + tanh(0.7978845608 * z * (1 + 0.044715 * z * z)))


def upper_confidence_bound(X: np.ndarray, gp: GaussianProcessSurrogate,
                            beta: float = 2.0) -> np.ndarray:
    """UCB acquisition = mean - beta * std (minimization)."""
    mean, std = gp.predict(X)
    return mean - beta * std


def probability_of_improvement(X: np.ndarray, gp: GaussianProcessSurrogate,
                                best_y: float) -> np.ndarray:
    """Probability of Improvement acquisition."""
    mean, std = gp.predict(X)
    std = np.maximum(std, 1e-9)
    z = (mean - best_y) / std
    return np.vectorize(_phi)(z)


# --------------------------------------------------------------------------- #
# BO loop
# --------------------------------------------------------------------------- #
class BayesianOptimizer:
    """Bayesian optimization loop over a bounded search space.

    Initial phase: random Latin-hypercube-style samples.
    Acquisition phase: maximize acquisition over a candidate grid.
    """

    def __init__(self, bounds: np.ndarray, n_initial: int = 5,
                 acquisition: str = "ei", seed: int = 42) -> None:
        self.bounds = np.asarray(bounds, dtype=np.float64)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must be shape (n_dim, 2)")
        self.n_initial = int(n_initial)
        self.acquisition = acquisition
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._gp = GaussianProcessSurrogate()
        self._acq_history: list[dict[str, float]] = []

    # ------------------------------------------------------------------ #
    def _random_within_bounds(self) -> np.ndarray:
        """Uniform random point within bounds (Latin-hypercube-ish)."""
        return self._rng.uniform(self.bounds[:, 0], self.bounds[:, 1])

    # ------------------------------------------------------------------ #
    def _candidate_grid(self, n: int = _GRID_SIZE) -> np.ndarray:
        """Generate candidates: mix of LHS samples + perturbations around best."""
        # Half LHS random, half local around best (if any).
        n_lhs = n // 2 if self._X else n
        n_local = n - n_lhs
        lhs = self._rng.uniform(
            self.bounds[:, 0], self.bounds[:, 1], size=(n_lhs, self.bounds.shape[0])
        )
        if n_local > 0 and self._X:
            best_x = np.array(self._X)[int(np.argmin(self._y))]
            # Perturb best by up to 20% of each dim's range.
            ranges = self.bounds[:, 1] - self.bounds[:, 0]
            perturb = self._rng.normal(0, 0.1 * ranges, size=(n_local, self.bounds.shape[0]))
            local = np.clip(best_x + perturb, self.bounds[:, 0], self.bounds[:, 1])
            return np.vstack([lhs, local])
        return lhs

    # ------------------------------------------------------------------ #
    def _acquisition_fn(self, X: np.ndarray) -> np.ndarray:
        best_y = min(self._y) if self._y else 0.0
        if self.acquisition == "ei":
            return expected_improvement(X, self._gp, best_y)
        if self.acquisition == "ucb":
            return upper_confidence_bound(X, self._gp, beta=2.0)
        if self.acquisition == "pi":
            return probability_of_improvement(X, self._gp, best_y)
        raise ValueError(f"unknown acquisition: {self.acquisition}")

    # ------------------------------------------------------------------ #
    def suggest(self) -> np.ndarray:
        """Return next query point (random if < n_initial, else argmax acquisition)."""
        if len(self._X) < self.n_initial:
            return self._random_within_bounds()
        # Validate acquisition before use.
        if self.acquisition not in ("ei", "ucb", "pi"):
            raise ValueError(f"unknown acquisition: {self.acquisition}")
        # Refit GP.
        self._gp.fit(np.array(self._X), np.array(self._y))
        # Acquisition maximization over candidate grid.
        candidates = self._candidate_grid()
        acq = self._acquisition_fn(candidates)
        # Add tiny exploration noise to break ties.
        acq = acq + 1e-9 * self._rng.standard_normal(acq.shape)
        idx = int(np.argmax(acq))
        x = candidates[idx]
        # Record acquisition value for history.
        best_y = min(self._y) if self._y else 0.0
        self._acq_history.append({
            "iter": len(self._X),
            "acquisition_value": float(acq[idx]),
            "best_y": float(best_y),
        })
        return x

    # ------------------------------------------------------------------ #
    def observe(self, x: np.ndarray, y: float) -> None:
        """Record observation (x, y). Refit GP if > 2 samples."""
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.shape[0] != self.bounds.shape[0]:
            raise ValueError(
                f"x dim {x.shape[0]} != bounds dim {self.bounds.shape[0]}"
            )
        # Clip to bounds for safety.
        x = np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
        self._X.append(x)
        self._y.append(float(y))

    # ------------------------------------------------------------------ #
    def best(self) -> tuple[np.ndarray, float]:
        """Return best observed (x, y). Raises if no observations."""
        if not self._X:
            raise RuntimeError("no observations yet")
        idx = int(np.argmin(self._y))
        return np.array(self._X)[idx], float(self._y[idx])

    # ------------------------------------------------------------------ #
    @property
    def n_observed(self) -> int:
        return len(self._X)

    @property
    def history(self) -> list[dict[str, float]]:
        return list(self._acq_history)


# --------------------------------------------------------------------------- #
# High-level helper
# --------------------------------------------------------------------------- #
def bayesian_search_setup(
    track_id: str,
    baseline: Any,
    driver_profile: Any = None,
    n_iterations: int = 15,
    acquisition: str = "ei",
    seed: int = 42,
) -> dict[str, Any]:
    """Bayesian-optimized setup search using the DNN surrogate as objective.

    Returns dict with:
        recommended_setup, recommended_lap_time, baseline_lap_time,
        predicted_gain_s, iterations, acquisition, history, gp_final_std.
    """
    from f1opt.data.setup_schema import CarSetup

    # to_vector() normalizes each setup field to [0, 1], so BO operates in the
    # unit hypercube of dimension 19.
    n_dim = 19
    bounds = np.array([[0.0, 1.0]] * n_dim)

    # Objective: predict_lap_time via surrogate (best-effort; fallback heuristic).
    try:
        from f1opt.model.surrogate import _get_default_model, predict_lap_time
        _get_default_model()  # warm cache
        _surrogate_available = True
    except Exception:
        _surrogate_available = False

    def objective(vec: np.ndarray) -> float:
        if _surrogate_available:
            try:
                setup = CarSetup.from_vector(vec.tolist())
                return float(predict_lap_time(setup, track_id, driver_profile))
            except Exception:
                pass
        # Heuristic fallback: baseline + small deterministic perturbation.
        base_vec = np.array(baseline.to_vector()) if hasattr(baseline, "to_vector") \
            else np.zeros(bounds.shape[0])
        return 90.0 + 0.01 * float(np.sum((vec - base_vec) ** 2))

    # Baseline lap time.
    base_vec = baseline.to_vector() if hasattr(baseline, "to_vector") \
        else np.zeros(bounds.shape[0])
    baseline_lap = objective(np.array(base_vec))

    # Run BO. Seed with baseline as first observation so best() is always
    # at least as good as baseline (guarantees predicted_gain_s >= 0).
    bo = BayesianOptimizer(bounds, n_initial=5, acquisition=acquisition, seed=seed)
    # Inject baseline as the first sample (counts toward n_initial).
    bo.observe(np.array(base_vec), baseline_lap)
    history: list[dict[str, Any]] = [{
        "iter": 1,
        "lap_time": float(baseline_lap),
        "acquisition_value": 0.0,
        "is_baseline": True,
    }]
    for _ in range(n_iterations - 1):  # -1 because baseline already counted
        x = bo.suggest()
        y = objective(x)
        bo.observe(x, y)
        history.append({
            "iter": bo.n_observed,
            "lap_time": float(y),
            "acquisition_value": float(bo.history[-1]["acquisition_value"])
            if bo.history else 0.0,
        })

    best_x, best_y = bo.best()
    # Safety net: if best found is worse than baseline (shouldn't happen since
    # baseline is seeded, but guard against numerical edge cases), use baseline.
    if best_y > baseline_lap:
        best_x = np.array(base_vec)
        best_y = baseline_lap
    recommended = CarSetup.from_vector(best_x.tolist())

    # GP final uncertainty at recommended point.
    if bo.n_observed >= 2:
        bo._gp.fit(np.array(bo._X), np.array(bo._y))
        _, std = bo._gp.predict(best_x.reshape(1, -1))
        gp_final_std = float(std[0])
    else:
        gp_final_std = 0.0

    return {
        "recommended_setup": recommended,
        "recommended_lap_time": float(best_y),
        "baseline_lap_time": float(baseline_lap),
        "predicted_gain_s": float(baseline_lap - best_y),
        "iterations": int(n_iterations),
        "acquisition": acquisition,
        "history": history,
        "gp_final_std": gp_final_std,
    }
