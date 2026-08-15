"""Model diagnostics: calibration curves and residual analysis (Iter-141).

EA F1 2026 engineering workflow: after training a surrogate model, the team
needs to calibrate and diagnose errors.
"""
from __future__ import annotations

from dataclasses import dataclass as _dataclass

import numpy as _np
import torch as _torch

from f1opt.model.surrogate import SurrogateModel

__all__ = [
    'BootstrapReport', 'CalibrationReport', 'CrossValidationReport',
    'ModelComparisonReport', 'PredictionInterval', 'ResidualReport',
    'bootstrap_metrics', 'calibration_curve', 'compare_models',
    'cross_validate', 'feature_importance',
    'per_track_error_breakdown', 'prediction_intervals',
    'prediction_uncertainty', 'residual_analysis',
]


@_dataclass
class CalibrationReport:
    n_samples: int
    mean_bias: float
    mae: float
    rmse: float
    r_squared: float
    max_error: float
    bin_edges: _np.ndarray
    bin_pred_means: _np.ndarray
    bin_actual_means: _np.ndarray
    bin_counts: _np.ndarray
    bin_std: _np.ndarray
    heteroscedasticity_ratio: float


@_dataclass
class CrossValidationReport:
    """K-fold cross-validation report (Iter-153).

    Aggregates per-fold metrics (MAE, RMSE, max error) plus the overall
    mean and standard deviation across folds, giving a robust estimate of
    model generalization performance and its variability.
    """
    n_folds: int
    n_total: int
    fold_maes: _np.ndarray
    fold_rmses: _np.ndarray
    fold_max_errors: _np.ndarray
    fold_sizes: _np.ndarray
    mean_mae: float
    std_mae: float
    mean_rmse: float
    std_rmse: float
    mean_max_error: float
    std_max_error: float


@_dataclass
class ResidualReport:
    mean: float
    std: float
    skewness: float
    kurtosis: float
    outlier_count: int
    outlier_indices: _np.ndarray
    q_01: float
    q_05: float
    q_25: float
    q_50: float
    q_75: float
    q_95: float
    q_99: float


@_dataclass
class PredictionInterval:
    """Prediction interval report for a single confidence level (Iter-157).

    Uses the conformal prediction approach: calibrate residual quantiles on a
    held-out set, then apply them as additive margins to new predictions.

    Attributes:
        confidence: Confidence level (e.g. 0.90 for 90% intervals).
        lower: Lower bound predictions ``(n_points,)``.
        upper: Upper bound predictions ``(n_points,)``.
        center: Point predictions ``(n_points,)``.
        margin: Half-width of the interval (same for all points in the
            basic conformal approach; per-point margins require ensemble
            or quantile models).
        empirical_coverage: Fraction of calibration points that fall within
            the interval (should be ≈ ``confidence``).
        n_calibration: Number of calibration samples used.
    """
    confidence: float
    lower: _np.ndarray
    upper: _np.ndarray
    center: _np.ndarray
    margin: float
    empirical_coverage: float
    n_calibration: int


@_dataclass
class BootstrapReport:
    """Bootstrap confidence intervals for evaluation metrics (Iter-161).

    Resamples the residual array with replacement ``n_bootstrap`` times and
    recomputes MAE / RMSE / max-error for each resample, producing a
    distribution-free estimate of the uncertainty around each metric.

    Attributes:
        n_samples: Number of (pred, true) pairs in the original sample.
        n_bootstrap: Number of bootstrap resamples drawn.
        mae_mean: Mean MAE across resamples (the bootstrap point estimate).
        mae_std: Std of MAE across resamples.
        mae_lower: Lower bound of the MAE confidence interval (``alpha/2`` quantile).
        mae_upper: Upper bound of the MAE confidence interval (``1-alpha/2`` quantile).
        rmse_mean / rmse_std / rmse_lower / rmse_upper: Same for RMSE.
        max_error_mean / max_error_std / max_error_lower / max_error_upper:
            Same for the maximum absolute error.
        confidence: Confidence level (e.g. 0.95 for 95% intervals).
        seed: Random seed used for reproducibility.
    """
    n_samples: int
    n_bootstrap: int
    mae_mean: float
    mae_std: float
    mae_lower: float
    mae_upper: float
    rmse_mean: float
    rmse_std: float
    rmse_lower: float
    rmse_upper: float
    max_error_mean: float
    max_error_std: float
    max_error_lower: float
    max_error_upper: float
    confidence: float
    seed: int


@_dataclass
class ModelComparisonReport:
    """Paired-bootstrap model comparison report (Iter-161).

    Compares two models (A and B) on the *same* evaluation set using paired
    bootstrap resampling: on each draw, the same index set is used to compute
    MAE for both models, so the per-resample delta reflects only the model
    difference (not sampling noise). The fraction of draws where B beats A on
    MAE gives a non-parametric significance proxy.

    Convention: ``delta_mae = mae_A - mae_B`` (positive => B is better).

    Attributes:
        n_samples: Number of evaluation points.
        n_bootstrap: Number of paired bootstrap resamples.
        mae_a: Point-estimate MAE of model A on the full set.
        mae_b: Point-estimate MAE of model B on the full set.
        delta_mae_mean: Mean of ``mae_A - mae_B`` across resamples.
        delta_mae_std: Std of the per-resample delta.
        delta_mae_lower: Lower bound of the delta confidence interval.
        delta_mae_upper: Upper bound of the delta confidence interval.
        p_b_better_mae: Fraction of resamples where ``mae_B < mae_A``
            (i.e. B is better). Values close to 1.0 => B significantly better;
            close to 0.0 => A significantly better; ~0.5 => no significant
            difference.
        confidence: Confidence level for the delta interval.
        seed: Random seed used.
    """
    n_samples: int
    n_bootstrap: int
    mae_a: float
    mae_b: float
    delta_mae_mean: float
    delta_mae_std: float
    delta_mae_lower: float
    delta_mae_upper: float
    p_b_better_mae: float
    confidence: float
    seed: int


def calibration_curve(
    model: SurrogateModel, x: _np.ndarray, y_lap: _np.ndarray,
    *, n_bins: int = 10,
) -> CalibrationReport:
    y_pred = _predict_lap(model, x)
    residuals = y_pred - y_lap
    n = len(y_lap)
    mean_bias = float(_np.mean(residuals))
    mae = float(_np.mean(_np.abs(residuals)))
    rmse = float(_np.sqrt(_np.mean(residuals ** 2)))
    max_error = float(_np.max(_np.abs(residuals)))
    ss_res = float(_np.sum(residuals ** 2))
    ss_tot = float(_np.sum((y_lap - _np.mean(y_lap)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    y_min, y_max = float(_np.min(y_lap)), float(_np.max(y_lap))
    if y_max - y_min < 1e-6:
        y_max = y_min + 1.0
    bin_edges = _np.linspace(y_min, y_max, n_bins + 1)
    bin_indices = _np.digitize(y_lap, bin_edges) - 1
    bin_indices = _np.clip(bin_indices, 0, n_bins - 1)
    bin_pred_means = _np.zeros(n_bins)
    bin_actual_means = _np.zeros(n_bins)
    bin_counts = _np.zeros(n_bins, dtype=int)
    bin_std = _np.zeros(n_bins)
    for i in range(n_bins):
        mask = bin_indices == i
        cnt = _np.sum(mask)
        bin_counts[i] = cnt
        if cnt > 0:
            bin_pred_means[i] = float(_np.mean(y_pred[mask]))
            bin_actual_means[i] = float(_np.mean(y_lap[mask]))
            bin_std[i] = float(_np.std(residuals[mask]))
    nonzero_std = bin_std[bin_counts > 0]
    if len(nonzero_std) >= 2 and nonzero_std.min() > 1e-9:
        het_ratio = float(nonzero_std.max() / nonzero_std.min())
    else:
        het_ratio = 1.0
    return CalibrationReport(
        n_samples=n, mean_bias=mean_bias, mae=mae, rmse=rmse,
        r_squared=r_squared, max_error=max_error,
        bin_edges=bin_edges, bin_pred_means=bin_pred_means,
        bin_actual_means=bin_actual_means, bin_counts=bin_counts,
        bin_std=bin_std, heteroscedasticity_ratio=het_ratio,
    )


def residual_analysis(residuals: _np.ndarray) -> ResidualReport:
    n = len(residuals)
    mean = float(_np.mean(residuals))
    std = float(_np.std(residuals, ddof=1) if n > 1 else 0.0)
    qs = _np.percentile(residuals, [1, 5, 25, 50, 75, 95, 99])
    q_01, q_05, q_25, q_50, q_75, q_95, q_99 = map(float, qs)
    if std > 1e-9:
        skewness = float(_np.mean(((residuals - mean) / std) ** 3))
        kurtosis = float(_np.mean(((residuals - mean) / std) ** 4))
    else:
        skewness = 0.0
        kurtosis = 0.0
    iqr = q_75 - q_25
    lower = q_25 - 3.0 * iqr
    upper = q_75 + 3.0 * iqr
    outlier_mask = (residuals < lower) | (residuals > upper)
    outlier_indices = _np.where(outlier_mask)[0]
    return ResidualReport(
        mean=mean, std=std, skewness=skewness, kurtosis=kurtosis,
        outlier_count=int(_np.sum(outlier_mask)),
        outlier_indices=outlier_indices,
        q_01=q_01, q_05=q_05, q_25=q_25, q_50=q_50,
        q_75=q_75, q_95=q_95, q_99=q_99,
    )


def per_track_error_breakdown(
    model: SurrogateModel, x: _np.ndarray, y_lap: _np.ndarray,
    track_ids: list[str],
) -> dict[str, dict[str, float]]:
    y_pred = _predict_lap(model, x)
    residuals = y_pred - y_lap
    result: dict[str, dict[str, float]] = {}
    unique_tracks = sorted(set(track_ids))
    for tid in unique_tracks:
        mask = _np.array([t == tid for t in track_ids])
        n = int(_np.sum(mask))
        if n == 0:
            continue
        res = residuals[mask]
        result[tid] = {
            "mae": float(_np.mean(_np.abs(res))),
            "rmse": float(_np.sqrt(_np.mean(res ** 2))),
            "bias": float(_np.mean(res)),
            "n": n,
        }
    return result


def prediction_uncertainty(
    model: SurrogateModel,
    x: _np.ndarray,
    n_samples: int = 30,
    noise_std: float = 0.01,
    seed: int = 0,
) -> dict[str, _np.ndarray]:
    """Prediction uncertainty via ensemble variance or input perturbation (Iter-145).

    Two strategies depending on model type:

    - **EnsembleSurrogateModel**: each member predicts independently;
      variance across members measures model uncertainty.
    - **Single SurrogateModel**: adds small Gaussian noise to input features
      and runs ``n_samples`` forward passes to estimate sensitivity.

    Args:
        model: Trained surrogate model (single or ensemble).
        x: Input features ``(n_points, INPUT_DIM)``.
        n_samples: Number of perturbation passes (single-model only).
        noise_std: Std of Gaussian noise added to inputs (single-model only).
        seed: Random seed for reproducibility.

    Returns:
        ``{"mean": ndarray, "std": ndarray, "q05": ndarray, "q95": ndarray}``
        each of shape ``(n_points,)``.
    """
    from f1opt.model.surrogate import EnsembleSurrogateModel

    _torch.manual_seed(seed)
    _np.random.seed(seed)
    xt = _torch.as_tensor(x, dtype=_torch.float32)
    all_preds: list[_np.ndarray] = []

    if isinstance(model, EnsembleSurrogateModel):
        model.eval()
        with _torch.no_grad():
            for member in model.models:
                member.eval()
                sectors, _ = member(xt)
                lap = _np.asarray(sectors.sum(dim=1))
                all_preds.append(lap)
    else:
        model.eval()
        with _torch.no_grad():
            for _ in range(n_samples):
                noise = _torch.randn_like(xt) * noise_std
                sectors, _ = model(xt + noise)
                lap = _np.asarray(sectors.sum(dim=1))
                all_preds.append(lap)

    stacked = _np.stack(all_preds, axis=0)  # (n_samples_or_n_members, n_points)
    return {
        "mean": _np.mean(stacked, axis=0),
        "std": _np.std(stacked, axis=0),
        "q05": _np.percentile(stacked, 5, axis=0),
        "q95": _np.percentile(stacked, 95, axis=0),
    }


def _predict_lap(model: SurrogateModel, x: _np.ndarray) -> _np.ndarray:
    """Helper: predict lap time from input features."""
    xt = _torch.as_tensor(x, dtype=_torch.float32)
    model.eval()
    with _torch.no_grad():
        sectors, _ = model(xt)
    return _np.asarray(sectors.sum(dim=1))


def feature_importance(
    model: SurrogateModel,
    x: _np.ndarray,
    y_lap: _np.ndarray,
    *,
    feature_names: list[str] | None = None,
    n_repeats: int = 5,
    seed: int = 0,
) -> dict[str, _np.ndarray]:
    """Permutation-based feature importance (Iter-149).

    Measures each input feature's importance by randomly shuffling that
    feature's column and measuring the increase in prediction error. A
    large error increase means the feature is important.

    Args:
        model: Trained surrogate model.
        x: Input features ``(n_points, INPUT_DIM)``.
        y_lap: True lap times ``(n_points,)`` in seconds.
        feature_names: Optional list of 37 feature names. If ``None``,
            defaults to ``["f0", "f1", ...]``.
        n_repeats: Number of shuffle repetitions per feature (higher = more
            stable estimates).
        seed: Random seed for shuffle reproducibility.

    Returns:
        ``{"names": ndarray, "importance_mean": ndarray, "importance_std": ndarray}``
        sorted by importance (descending).
    """
    _np.random.seed(seed)
    n_features = x.shape[1]

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]
    elif len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length {len(feature_names)} != x.shape[1]={n_features}"
        )

    # Baseline error
    baseline_pred = _predict_lap(model, x)
    baseline_error = _np.mean(_np.abs(baseline_pred - y_lap))

    importances = _np.zeros((n_features, n_repeats))

    for fi in range(n_features):
        for ri in range(n_repeats):
            x_perm = x.copy()
            _np.random.shuffle(x_perm[:, fi])
            perm_pred = _predict_lap(model, x_perm)
            perm_error = _np.mean(_np.abs(perm_pred - y_lap))
            importances[fi, ri] = perm_error - baseline_error

    mean_imp = _np.mean(importances, axis=1)
    std_imp = _np.std(importances, axis=1, ddof=1)

    # Sort by importance (descending)
    order = _np.argsort(-mean_imp)
    names_arr = _np.array(feature_names, dtype=str)

    return {
        "names": names_arr[order],
        "importance_mean": mean_imp[order],
        "importance_std": std_imp[order],
    }


def cross_validate(
    x: _np.ndarray,
    y_lap: _np.ndarray,
    *,
    n_folds: int = 5,
    iterations: int = 200,
    seed: int = 0,
    shuffle: bool = True,
) -> CrossValidationReport:
    """K-fold cross-validation of the surrogate model (Iter-153).

    Trains ``n_folds`` models, each on ``n_folds - 1`` folds and evaluated on
    the held-out fold. This gives a robust estimate of generalization
    performance and its variability across data splits, which is critical
    for detecting overfitting or unstable training.

    Args:
        x: Input features ``(n_points, INPUT_DIM)``.
        y_lap: True lap times ``(n_points,)`` in seconds.
        n_folds: Number of cross-validation folds (default 5). Must be >= 2.
        iterations: Training iterations per fold (default 200 — small for
            speed; increase for production-quality estimates).
        seed: Random seed for fold assignment reproducibility.
        shuffle: If True, shuffle indices before splitting (default).

    Returns:
        :class:`CrossValidationReport` with per-fold and aggregate metrics.

    Raises:
        ValueError: If ``n_folds < 2`` or ``n_folds > n_samples``.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    n = len(y_lap)
    if n_folds > n:
        raise ValueError(f"n_folds={n_folds} > n_samples={n}")

    rng = _np.random.RandomState(seed)
    indices = _np.arange(n)
    if shuffle:
        rng.shuffle(indices)

    # Split indices into n_folds roughly-equal chunks.
    fold_sizes = _np.full(n_folds, n // n_folds, dtype=int)
    fold_sizes[: n % n_folds] += 1  # distribute remainder

    fold_maes: list[float] = []
    fold_rmses: list[float] = []
    fold_max_errors: list[float] = []
    fold_n: list[int] = []

    current = 0
    for fi in range(n_folds):
        fs = int(fold_sizes[fi])
        val_idx = indices[current : current + fs]
        current += fs
        train_idx = _np.setdiff1d(indices, val_idx, assume_unique=False)

        x_train = x[train_idx]
        y_train = y_lap[train_idx]
        x_val = x[val_idx]
        y_val = y_lap[val_idx]

        # Build a minimal SurrogateModel trained on this fold's train set.
        # We use the low-level train() with n_samples=len(train_idx) but
        # since train() generates its own data, we instead train the model
        # directly on the provided tensors for a true CV evaluation.
        _torch.manual_seed(seed + fi)  # seed before model init for reproducibility
        model = SurrogateModel()
        _train_model_on_data(model, x_train, y_train, iterations=iterations,
                             seed=seed + fi)

        # Evaluate on held-out fold.
        y_pred = _predict_lap(model, x_val)
        residuals = y_pred - y_val
        mae = float(_np.mean(_np.abs(residuals)))
        rmse = float(_np.sqrt(_np.mean(residuals ** 2)))
        max_err = float(_np.max(_np.abs(residuals)))

        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_max_errors.append(max_err)
        fold_n.append(len(val_idx))

    maes = _np.array(fold_maes)
    rmses = _np.array(fold_rmses)
    max_errs = _np.array(fold_max_errors)
    sizes = _np.array(fold_n, dtype=int)

    return CrossValidationReport(
        n_folds=n_folds,
        n_total=n,
        fold_maes=maes,
        fold_rmses=rmses,
        fold_max_errors=max_errs,
        fold_sizes=sizes,
        mean_mae=float(_np.mean(maes)),
        std_mae=float(_np.std(maes, ddof=1)) if n_folds > 1 else 0.0,
        mean_rmse=float(_np.mean(rmses)),
        std_rmse=float(_np.std(rmses, ddof=1)) if n_folds > 1 else 0.0,
        mean_max_error=float(_np.mean(max_errs)),
        std_max_error=float(_np.std(max_errs, ddof=1)) if n_folds > 1 else 0.0,
    )


def _train_model_on_data(
    model: SurrogateModel,
    x: _np.ndarray,
    y_lap: _np.ndarray,
    *,
    iterations: int = 200,
    seed: int = 0,
) -> None:
    """Train a SurrogateModel in-place on provided (x, y_lap) data (Iter-153).

    Uses AdamW + cosine LR schedule + gradient clipping, consistent with
    :func:`f1opt.model.train.train` but operating directly on the provided
    arrays rather than generating synthetic data. The model is trained to
    predict sector times that sum to ``y_lap``; response targets are set
    to zero (not used in CV lap-time evaluation).
    """
    import torch

    torch.manual_seed(seed)
    model.train()

    x_t = torch.as_tensor(x, dtype=torch.float32)
    # Distribute lap time equally across 3 sectors as a simple target.
    n = x_t.shape[0]
    sec_y = torch.zeros(n, 3, dtype=torch.float32)
    sec_y[:, :] = torch.as_tensor(y_lap, dtype=torch.float32).unsqueeze(1) / 3.0
    resp_y = torch.zeros(n, 7, dtype=torch.float32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, iterations), eta_min=1e-5,
    )
    loss_fn = torch.nn.MSELoss()

    for _ in range(iterations):
        optimizer.zero_grad()
        sec_pred, resp_pred = model(x_t)
        lap_pred = sec_pred.sum(dim=1)
        lap_tgt = sec_y.sum(dim=1)
        loss = (
            loss_fn(sec_pred, sec_y)
            + 0.3 * loss_fn(resp_pred, resp_y)
            + 0.1 * loss_fn(lap_pred, lap_tgt)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    model.eval()


def prediction_intervals(
    model: SurrogateModel,
    x_cal: _np.ndarray,
    y_cal: _np.ndarray,
    x_new: _np.ndarray,
    confidence: float = 0.90,
) -> PredictionInterval:
    """Conformal prediction intervals for lap-time predictions (Iter-157).

    Uses the split conformal prediction approach:

    1. Compute absolute residuals ``|y_cal - pred(x_cal)|`` on the
       calibration set.
    2. Find the ``ceil((1 - alpha) * (n + 1)) / n`` quantile of these
       residuals, where ``alpha = 1 - confidence``.
    3. Apply this quantile as an additive margin to new predictions:
       ``[pred - margin, pred + margin]``.

    This provides a distribution-free, finite-sample coverage guarantee:
    the true value falls within the interval with probability ≥
    ``confidence`` (assuming exchangeability of calibration and test data).

    Args:
        model: Trained surrogate model.
        x_cal: Calibration input features ``(n_cal, INPUT_DIM)``.
        y_cal: Calibration true lap times ``(n_cal,)`` in seconds.
        x_new: New input features to compute intervals for ``(n_new, INPUT_DIM)``.
        confidence: Desired coverage probability in ``(0, 1)`` (default 0.90).

    Returns:
        :class:`PredictionInterval` with lower/upper/center arrays and
        empirical coverage on the calibration set.

    Raises:
        ValueError: If ``confidence`` is not in ``(0, 1)`` or calibration
            set has fewer than 2 samples.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"confidence must be in (0, 1), got {confidence}"
        )
    n_cal = len(y_cal)
    if n_cal < 2:
        raise ValueError(
            f"calibration set must have >= 2 samples, got {n_cal}"
        )

    # Step 1: absolute residuals on calibration set
    cal_pred = _predict_lap(model, x_cal)
    abs_residuals = _np.abs(cal_pred - y_cal)

    # Step 2: conformal quantile
    # The corrected quantile level for finite-sample coverage:
    #   q_level = ceil((1 - alpha) * (n + 1)) / n
    alpha = 1.0 - confidence
    q_level = _np.ceil((1.0 - alpha) * (n_cal + 1)) / n_cal
    q_level = min(q_level, 1.0)  # cap at 1.0 for small n_cal
    margin = float(_np.quantile(abs_residuals, q_level))

    # Step 3: apply margin to new predictions
    center = _predict_lap(model, x_new)
    lower = center - margin
    upper = center + margin

    # Empirical coverage on calibration set
    cal_lower = cal_pred - margin
    cal_upper = cal_pred + margin
    covered = _np.sum((y_cal >= cal_lower) & (y_cal <= cal_upper))
    coverage = float(covered) / n_cal

    return PredictionInterval(
        confidence=confidence,
        lower=lower,
        upper=upper,
        center=center,
        margin=margin,
        empirical_coverage=coverage,
        n_calibration=n_cal,
    )


def bootstrap_metrics(
    y_true: _np.ndarray,
    y_pred: _np.ndarray,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapReport:
    """Bootstrap confidence intervals for MAE / RMSE / max-error (Iter-161).

    Resamples the residuals with replacement ``n_bootstrap`` times and
    recomputes the three error metrics on each resample. The empirical
    distribution of each metric yields a distribution-free confidence
    interval — useful for comparing models without assuming Gaussian
    residuals.

    Args:
        y_true: True lap times ``(n_samples,)`` in seconds.
        y_pred: Predicted lap times ``(n_samples,)`` in seconds.
        n_bootstrap: Number of bootstrap resamples (default 1000). Must be >= 2.
        confidence: Confidence level in ``(0, 1)`` (default 0.95 for 95% CIs).
        seed: Random seed for reproducibility.

    Returns:
        :class:`BootstrapReport` with mean / std / lower / upper for each
        metric.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` have mismatched lengths,
            if there are fewer than 2 samples, if ``n_bootstrap < 2``, or
            if ``confidence`` is not in ``(0, 1)``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"confidence must be in (0, 1), got {confidence}"
        )
    if n_bootstrap < 2:
        raise ValueError(f"n_bootstrap must be >= 2, got {n_bootstrap}")
    y_true = _np.asarray(y_true, dtype=_np.float64)
    y_pred = _np.asarray(y_pred, dtype=_np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if y_true.ndim != 1:
        raise ValueError(
            f"expected 1-D arrays, got y_true.ndim={y_true.ndim}"
        )
    n = len(y_true)
    if n < 2:
        raise ValueError(f"need at least 2 samples, got {n}")

    abs_err = _np.abs(y_pred - y_true)
    sq_err = abs_err ** 2

    rng = _np.random.RandomState(seed)
    # Draw all resample indices up front: shape (n_bootstrap, n).
    idx = rng.randint(0, n, size=(n_bootstrap, n))

    mae_samples = abs_err[idx].mean(axis=1)
    rmse_samples = _np.sqrt(sq_err[idx].mean(axis=1))
    max_samples = abs_err[idx].max(axis=1)

    alpha = 1.0 - confidence
    q_lo = 100.0 * (alpha / 2.0)
    q_hi = 100.0 * (1.0 - alpha / 2.0)

    def _summarize(samples: _np.ndarray) -> tuple[float, float, float, float]:
        return (
            float(_np.mean(samples)),
            float(_np.std(samples, ddof=1)),
            float(_np.percentile(samples, q_lo)),
            float(_np.percentile(samples, q_hi)),
        )

    mae_m, mae_s, mae_lo, mae_hi = _summarize(mae_samples)
    rmse_m, rmse_s, rmse_lo, rmse_hi = _summarize(rmse_samples)
    max_m, max_s, max_lo, max_hi = _summarize(max_samples)

    return BootstrapReport(
        n_samples=n,
        n_bootstrap=int(n_bootstrap),
        mae_mean=mae_m, mae_std=mae_s, mae_lower=mae_lo, mae_upper=mae_hi,
        rmse_mean=rmse_m, rmse_std=rmse_s, rmse_lower=rmse_lo, rmse_upper=rmse_hi,
        max_error_mean=max_m, max_error_std=max_s,
        max_error_lower=max_lo, max_error_upper=max_hi,
        confidence=float(confidence),
        seed=int(seed),
    )


def compare_models(
    model_a: SurrogateModel,
    model_b: SurrogateModel,
    x: _np.ndarray,
    y_lap: _np.ndarray,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> ModelComparisonReport:
    """Paired-bootstrap comparison of two surrogate models (Iter-161).

    Both models are evaluated on the *same* ``x``; on each bootstrap draw the
    same resampled indices are used to compute MAE for both, so the per-draw
    delta isolates the model difference from sampling noise. The fraction of
    draws where model B has lower MAE than model A is a non-parametric
    significance indicator.

    Convention: ``delta_mae = mae_A - mae_B``. Positive delta means B wins.

    Args:
        model_a: Reference model A.
        model_b: Candidate model B (the one we hope is better).
        x: Evaluation inputs ``(n_points, INPUT_DIM)``.
        y_lap: True lap times ``(n_points,)`` in seconds.
        n_bootstrap: Number of paired bootstrap resamples (default 1000).
        confidence: Confidence level for the delta interval (default 0.95).
        seed: Random seed for reproducibility.

    Returns:
        :class:`ModelComparisonReport` with point estimates and paired
        bootstrap statistics.

    Raises:
        ValueError: If ``x`` and ``y_lap`` have inconsistent lengths, if
            there are fewer than 2 samples, if ``n_bootstrap < 2``, or if
            ``confidence`` is not in ``(0, 1)``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"confidence must be in (0, 1), got {confidence}"
        )
    if n_bootstrap < 2:
        raise ValueError(f"n_bootstrap must be >= 2, got {n_bootstrap}")
    y_lap = _np.asarray(y_lap, dtype=_np.float64)
    n = len(y_lap)
    if n < 2:
        raise ValueError(f"need at least 2 samples, got {n}")
    if x.shape[0] != n:
        raise ValueError(
            f"x.shape[0]={x.shape[0]} != len(y_lap)={n}"
        )

    pred_a = _np.asarray(_predict_lap(model_a, x), dtype=_np.float64)
    pred_b = _np.asarray(_predict_lap(model_b, x), dtype=_np.float64)
    abs_a = _np.abs(pred_a - y_lap)
    abs_b = _np.abs(pred_b - y_lap)

    mae_a_pt = float(abs_a.mean())
    mae_b_pt = float(abs_b.mean())

    rng = _np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_bootstrap, n))
    mae_a_boot = abs_a[idx].mean(axis=1)
    mae_b_boot = abs_b[idx].mean(axis=1)
    delta = mae_a_boot - mae_b_boot  # positive => B is better

    alpha = 1.0 - confidence
    q_lo = 100.0 * (alpha / 2.0)
    q_hi = 100.0 * (1.0 - alpha / 2.0)

    p_b_better = float(_np.mean(mae_b_boot < mae_a_boot))

    return ModelComparisonReport(
        n_samples=n,
        n_bootstrap=int(n_bootstrap),
        mae_a=mae_a_pt,
        mae_b=mae_b_pt,
        delta_mae_mean=float(_np.mean(delta)),
        delta_mae_std=float(_np.std(delta, ddof=1)),
        delta_mae_lower=float(_np.percentile(delta, q_lo)),
        delta_mae_upper=float(_np.percentile(delta, q_hi)),
        p_b_better_mae=p_b_better,
        confidence=float(confidence),
        seed=int(seed),
    )

