"""Global feature importance analysis for the surrogate model (Iter-137).

EA F1 2026 professional team workflow: engineers need to understand *which*
input dimensions the surrogate model relies on most — globally, across the
whole input space — not just at a single setup point (that local view is
:mod:`f1opt.model.setup_analysis`). This module answers two questions:

1. **Gradient-based**: How sensitive is the predicted lap time to each input
   feature, averaged over a representative sample of setups/tracks/drivers?
   Fast (one forward + one backward over a batch), differentiable, but only
   captures local linear sensitivity at each sample point.
2. **Permutation-based**: How much does held-out lap-time MAE degrade when a
   single feature is randomly shuffled (breaking its relationship with the
   target)? Model-agnostic, captures non-linear interactions, but slower
   (one forward per feature).

The 37-dim input vector layout (see :func:`surrogate.build_input_vector`):

* ``[0:19]``  — 19 setup fields (front_wing … fuel_load)
* ``[19:29]`` — 10 track-context dims (length, corners, is_sprint,
  5 one-hot track_type, elevation, unknown_flag)
* ``[29:37]`` — 8 driver-profile dims (brake_point_norm … drs_usage_efficiency)

Public API:

* :data:`FEATURE_NAMES` — 37-dim ordered feature name list.
* :data:`FEATURE_GROUPS` — per-index group label (``"setup"`` / ``"track"`` /
  ``"driver"``).
* :func:`gradient_feature_importance` — mean ``|∂lap/∂x_i|`` over samples.
* :func:`permutation_feature_importance` — MAE increase when shuffling a column.
* :func:`rank_features` — sort features by importance, return ranked list.
* :func:`feature_importance_summary` — combined gradient + permutation report.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from f1opt.data.setup_schema import ALL_SETUP_FIELDS
from f1opt.model.surrogate import (
    DRIVER_DIM,
    INPUT_DIM,
    SETUP_DIM,
    TRACK_CONTEXT_DIM,
    TRACK_TYPES,
    SurrogateModel,
)

# --- 37-dim feature name layout -------------------------------------------- #
# Must stay in lock-step with surrogate.build_input_vector / track_context.
_SETUP_NAMES: list[str] = [f.name for f in ALL_SETUP_FIELDS()]
_TRACK_NAMES: list[str] = [
    "track_length",
    "track_corners",
    "is_sprint",
    *[f"track_type_{t}" for t in TRACK_TYPES],
    "track_elevation",
    "track_unknown_flag",
]
_DRIVER_NAMES: list[str] = [
    "drv_brake_point_norm",
    "drv_throttle_smoothness",
    "drv_steer_smoothness",
    "drv_corner_balance_pref",
    "drv_aggression_score",
    "drv_consistency_score",
    "drv_ers_usage_intensity",
    "drv_drs_usage_efficiency",
]

FEATURE_NAMES: tuple[str, ...] = tuple(
    _SETUP_NAMES + _TRACK_NAMES + _DRIVER_NAMES
)
"""Ordered 37-dim feature names matching :func:`surrogate.build_input_vector`."""

if len(FEATURE_NAMES) != INPUT_DIM:
    raise RuntimeError(  # pragma: no cover - defensive layout invariant
        f"FEATURE_NAMES length {len(FEATURE_NAMES)} != INPUT_DIM {INPUT_DIM}; "
        "feature_importance layout is out of sync with surrogate."
    )

#: Per-index group label, parallel to :data:`FEATURE_NAMES`.
FEATURE_GROUPS: tuple[str, ...] = tuple(
    ["setup"] * SETUP_DIM
    + ["track"] * TRACK_CONTEXT_DIM
    + ["driver"] * DRIVER_DIM
)


@dataclass
class FeatureRanking:
    """One row of a feature-importance ranking.

    - ``index``: position in the 37-dim input vector.
    - ``name``: feature name (see :data:`FEATURE_NAMES`).
    - ``group``: ``"setup"`` / ``"track"`` / ``"driver"``.
    - ``importance``: non-negative importance score (method-dependent units).
    - ``rank``: 1-based rank (1 = most important).
    - ``share``: importance / total importance (0..1).
    """

    index: int
    name: str
    group: str
    importance: float
    rank: int
    share: float


def _to_model(model: Any) -> SurrogateModel:
    """Accept a SurrogateModel or an EnsembleSurrogateModel (use its mean)."""
    # EnsembleSurrogateModel exposes .members; use the first member for gradient
    # analysis (per-member gradients are averaged by the ensemble at predict time
    # but a single member's gradient is a representative sensitivity profile).
    # For permutation, the ensemble's predict_lap_time is used directly.
    if isinstance(model, SurrogateModel):
        return model
    members = getattr(model, "members", None)
    if members:
        return members[0]
    if hasattr(model, "model"):
        return model.model
    raise TypeError(
        f"Unsupported model type {type(model).__name__}; expected SurrogateModel "
        "or EnsembleSurrogateModel."
    )


def gradient_feature_importance(
    model: Any,
    x: np.ndarray | torch.Tensor,
    *,
    normalize: bool = True,
) -> dict[str, float]:
    """Mean absolute gradient of predicted lap time w.r.t. each input feature.

    For each sample in ``x``, computes ``|∂lap_time / ∂x_i|`` via torch autograd
    (one forward + one backward over the whole batch), then averages over
    samples. Captures local linear sensitivity at each sample point.

    Args:
        model: A :class:`SurrogateModel` or :class:`EnsembleSurrogateModel`.
            For an ensemble, the first member's gradient is used (a
            representative sensitivity profile; averaging per-member gradients
            would 3x the cost for marginal accuracy gain).
        x: Input batch ``(N, 37)`` (numpy array or torch tensor), as produced
            by :func:`surrogate.build_input_vector` stacked row-wise.
        normalize: When True (default), divide each feature's importance by the
            sum of all importances so values sum to 1.0 (relative shares).
            When False, return raw mean |gradient| (seconds per unit input).

    Returns:
        ``{feature_name: importance}`` dict (37 entries, matching
        :data:`FEATURE_NAMES`).
    """
    if x.ndim != 2 or x.shape[1] != INPUT_DIM:
        raise ValueError(
            f"x must be (N, {INPUT_DIM}), got shape {tuple(x.shape)}"
        )
    m = _to_model(model)
    m.eval()
    xt = torch.as_tensor(x, dtype=torch.float32, device=next(m.parameters()).device)
    xt.requires_grad_(True)
    sec, _resp = m(xt)
    lap = sec.sum(dim=1)  # (N,)
    # Sum to get a scalar whose gradient is the sum of per-sample gradients;
    # we then divide by N to get the mean.
    lap.sum().backward()
    grad = xt.grad  # (N, 37)
    if grad is None:
        raise RuntimeError("Gradient is None — model may not be differentiable.")
    mean_abs = grad.abs().mean(dim=0).detach().cpu().numpy()  # (37,)
    if normalize:
        total = float(mean_abs.sum())
        if total > 0.0:
            mean_abs = mean_abs / total
    return {name: float(v) for name, v in zip(FEATURE_NAMES, mean_abs, strict=True)}


def permutation_feature_importance(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_repeats: int = 5,
    seed: int = 0,
    normalize: bool = True,
) -> dict[str, float]:
    """Mean MAE increase when a single feature column is randomly shuffled.

    Model-agnostic permutation importance (Breiman 2001). For each feature,
    shuffle its column in ``x`` (breaking the feature-target relationship),
    recompute lap-time MAE, and report the increase over the baseline MAE.
    Averaged over ``n_repeats`` shuffles for stability.

    Args:
        model: A :class:`SurrogateModel` or :class:`EnsembleSurrogateModel`
            (must expose ``predict_lap_time(setup, track_id, driver)`` OR a
            forward returning ``(sectors, responses)``). When the model is a
            raw :class:`SurrogateModel`, this function calls ``model(x)`` and
            sums sectors directly (no setup/track/driver reconstruction needed).
        x: Input batch ``(N, 37)``.
        y: True lap times ``(N,)`` (seconds).
        n_repeats: Number of shuffle repeats per feature (default 5).
        seed: RNG seed for reproducible shuffles.
        normalize: When True (default), divide each feature's importance by the
            sum of all importances (relative shares).

    Returns:
        ``{feature_name: importance}`` dict (37 entries). Values are MAE
        increases in seconds; ``normalize=True`` converts to shares summing
        to 1.0.
    """
    if x.ndim != 2 or x.shape[1] != INPUT_DIM:
        raise ValueError(
            f"x must be (N, {INPUT_DIM}), got shape {tuple(x.shape)}"
        )
    if y.shape != (x.shape[0],):
        raise ValueError(
            f"y must be (N,) matching x rows; got {tuple(y.shape)} vs "
            f"{x.shape[0]} rows."
        )
    m = _to_model(model)
    m.eval()
    rng = np.random.default_rng(seed)
    xt = torch.as_tensor(x, dtype=torch.float32, device=next(m.parameters()).device)

    @torch.no_grad()
    def _mae(batch: torch.Tensor) -> float:
        sec, _resp = m(batch)
        lap = sec.sum(dim=1).cpu().numpy()
        return float(np.mean(np.abs(lap - y)))

    baseline = _mae(xt)
    importances = np.zeros(INPUT_DIM, dtype=np.float64)
    for i in range(INPUT_DIM):
        deltas = np.empty(n_repeats, dtype=np.float64)
        for r in range(n_repeats):
            perm = rng.permutation(x.shape[0])
            x_perm = x.copy()
            x_perm[:, i] = x[perm, i]
            xt_perm = torch.as_tensor(
                x_perm, dtype=torch.float32, device=xt.device
            )
            mae_perm = _mae(xt_perm)
            # Clamp at 0: shuffling a feature should not *improve* MAE in
            # expectation; negative deltas are noise around zero.
            deltas[r] = max(0.0, mae_perm - baseline)
        importances[i] = float(deltas.mean())
    if normalize:
        total = float(importances.sum())
        if total > 0.0:
            importances = importances / total
    return {name: float(v) for name, v in zip(FEATURE_NAMES, importances, strict=True)}


def rank_features(
    importance: dict[str, float],
    *,
    top_k: int | None = None,
) -> list[FeatureRanking]:
    """Sort features by importance (descending) and return ranked entries.

    Args:
        importance: ``{feature_name: score}`` dict (e.g. from
            :func:`gradient_feature_importance`).
        top_k: When set, return only the top-K entries (default: all).

    Returns:
        List of :class:`FeatureRanking` (rank 1 = most important). ``share``
        is computed against the sum of the *provided* importance values.
        Names not in :data:`FEATURE_NAMES` get ``index=-1`` and
        ``group="unknown"`` (defensive — supports unit testing of the sort
        logic with arbitrary keys).
    """
    total = float(sum(importance.values()))
    name_to_idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    if top_k is not None:
        items = items[: max(0, top_k)]
    out: list[FeatureRanking] = []
    for rank, (name, imp) in enumerate(items, start=1):
        idx = name_to_idx.get(name, -1)
        group = FEATURE_GROUPS[idx] if idx >= 0 else "unknown"
        share = (imp / total) if total > 0.0 else 0.0
        out.append(
            FeatureRanking(
                index=idx,
                name=name,
                group=group,
                importance=float(imp),
                rank=rank,
                share=float(share),
            )
        )
    return out


@dataclass
class FeatureImportanceSummary:
    """Combined gradient + permutation importance report."""

    gradient: dict[str, float]
    permutation: dict[str, float]
    gradient_ranked: list[FeatureRanking]
    permutation_ranked: list[FeatureRanking]
    top_k_agreement: float
    """Fraction of the top-K features shared between the two methods (0..1)."""


def feature_importance_summary(
    model: Any,
    x: np.ndarray,
    y: np.ndarray | None = None,
    *,
    method: str = "both",
    n_repeats: int = 5,
    seed: int = 0,
    top_k: int = 10,
) -> FeatureImportanceSummary:
    """Compute gradient (and optionally permutation) importance + rankings.

    Args:
        model: Surrogate model.
        x: Input batch ``(N, 37)``.
        y: True lap times ``(N,)``. Required when ``method`` includes
            ``"permutation"``; ignored for gradient-only.
        method: ``"gradient"``, ``"permutation"``, or ``"both"`` (default).
        n_repeats: Permutation shuffle repeats.
        seed: RNG seed for permutation shuffles.
        top_k: K for the top-K agreement metric.

    Returns:
        :class:`FeatureImportanceSummary`.
    """
    if method not in ("gradient", "permutation", "both"):
        raise ValueError(f"method must be gradient/permutation/both, got {method!r}")
    grad: dict[str, float] = {}
    perm: dict[str, float] = {}
    if method in ("gradient", "both"):
        grad = gradient_feature_importance(model, x, normalize=True)
    if method in ("permutation", "both"):
        if y is None:
            raise ValueError("y is required for permutation importance.")
        perm = permutation_feature_importance(
            model, x, y, n_repeats=n_repeats, seed=seed, normalize=True
        )
    grad_ranked = rank_features(grad) if grad else []
    perm_ranked = rank_features(perm) if perm else []
    # Top-K agreement: fraction of top-K gradient features also in top-K perm.
    agreement = 0.0
    if grad_ranked and perm_ranked:
        g_top = {r.name for r in grad_ranked[:top_k]}
        p_top = {r.name for r in perm_ranked[:top_k]}
        agreement = len(g_top & p_top) / max(1, len(g_top))
    return FeatureImportanceSummary(
        gradient=grad,
        permutation=perm,
        gradient_ranked=grad_ranked,
        permutation_ranked=perm_ranked,
        top_k_agreement=agreement,
    )
