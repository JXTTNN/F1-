"""Learning Rate Finder for surrogate model training (Iter-148).

EA F1 2026 professional standard: before training a surrogate model, the
optimal learning rate should be automatically determined rather than using
a fixed default. The :class:`LRFinder` runs a short exploratory training pass
with exponentially increasing learning rates and identifies the LR range
where the loss decreases most rapidly.

Algorithm (Leslie Smith, 2015 / fast.ai adaptation):

1. Start with a very small LR (``start_lr``, default 1e-7).
2. Run mini-batches, multiplying LR by a constant factor each step.
3. Record (lr, smoothed_loss) for each step.
4. Stop when loss diverges (exceeds ``divergence_threshold`` × min_loss)
   or ``max_steps`` is reached.
5. Compute the steepest-descent LR (where loss decreases fastest) and
   recommend it as ``suggested_lr``.

Usage::

    from f1opt.model.train import train
    from f1opt.model.lr_finder import lr_find

    model = SurrogateModel()
    result = lr_find(model, n_steps=200, start_lr=1e-7, end_lr=10.0)
    print(f"Suggested LR: {result.suggested_lr}")
    # Then train with the suggested LR:
    # model = train(iterations=3000, lr=result.suggested_lr)
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass

import numpy as _np
import torch as _torch

from f1opt.model.surrogate import SurrogateModel
from f1opt.model.train import _build_tensors, generate_dataset


@_dataclass
class LRFinderResult:
    """Results of a learning-rate range test (Iter-148)."""

    lrs: list[float]
    """Learning rates tested (one per step)."""

    losses: list[float]
    """Smoothed loss at each step."""

    suggested_lr: float
    """Recommended learning rate (steepest-descent point)."""

    min_loss: float
    """Minimum smoothed loss observed."""

    min_loss_lr: float
    """Learning rate at which minimum loss was observed."""

    diverged: bool
    """True if loss diverged before all steps were run."""

    steps_run: int
    """Number of steps actually executed."""


def lr_find(
    model: SurrogateModel | None = None,
    *,
    n_steps: int = 200,
    start_lr: float = 1e-7,
    end_lr: float = 10.0,
    n_samples: int = 2000,
    batch_size: int = 64,
    seed: int = 0,
    smoothing: float = 0.95,
    divergence_threshold: float = 4.0,
) -> LRFinderResult:
    """Run a learning-rate range test (Iter-148).

    Trains with exponentially increasing LR and records the smoothed loss
    at each step. The recommended LR is the point where loss decreases
    fastest (steepest descent), typically 10× smaller than the LR at the
    minimum loss.

    Args:
        model: Model to test. If ``None``, creates a fresh ``SurrogateModel``.
        n_steps: Maximum number of LR-increase steps.
        start_lr: Initial learning rate.
        end_lr: Final learning rate.
        n_samples: Number of synthetic training samples.
        batch_size: Mini-batch size.
        seed: Random seed for reproducibility.
        smoothing: EMA smoothing factor for loss (0.0 = no smoothing).
        divergence_threshold: Stop when loss > divergence_threshold × min_loss.

    Returns:
        :class:`LRFinderResult` with the LR history and recommendation.
    """
    if model is None:
        _torch.manual_seed(seed)
        model = SurrogateModel()

    data = generate_dataset(n_samples=n_samples, seed=seed, label_source="physics")
    x, sec_y, resp_y, _sec_priors, _resp_priors = _build_tensors(data)

    n = x.shape[0]
    if n < batch_size:
        batch_size = n

    # Multiplier to go from start_lr to end_lr in n_steps
    multiplier = (end_lr / start_lr) ** (1.0 / max(n_steps - 1, 1))

    _torch.manual_seed(seed)
    model.train()
    optimizer = _torch.optim.AdamW(model.parameters(), lr=start_lr, weight_decay=1e-5)

    lrs: list[float] = []
    losses: list[float] = []
    smoothed_loss: float | None = None
    min_loss = float("inf")
    min_loss_lr = start_lr
    diverged = False
    steps_run = 0

    for step in range(n_steps):
        # Sample a mini-batch
        indices = _torch.randint(0, n, (batch_size,))
        xb = x[indices]
        ysb = sec_y[indices]
        yrb = resp_y[indices]

        optimizer.zero_grad()
        sectors, responses = model(xb)
        lap_pred = sectors.sum(dim=1)
        lap_target = ysb.sum(dim=1)
        loss = (
            _torch.nn.functional.mse_loss(sectors, ysb)
            + 0.3 * _torch.nn.functional.mse_loss(responses, yrb)
            + 0.1 * _torch.nn.functional.mse_loss(lap_pred, lap_target)
        )
        loss.backward()
        optimizer.step()

        current_lr = start_lr * (multiplier ** step)
        lrs.append(current_lr)

        raw_loss = float(loss.item())
        if smoothed_loss is None:
            smoothed_loss = raw_loss
        else:
            smoothed_loss = smoothing * smoothed_loss + (1.0 - smoothing) * raw_loss
        losses.append(smoothed_loss)

        if smoothed_loss < min_loss:
            min_loss = smoothed_loss
            min_loss_lr = current_lr

        # Check divergence
        if smoothed_loss > divergence_threshold * min_loss and step > 10:
            diverged = True
            steps_run = step + 1
            break

        # Update LR
        for group in optimizer.param_groups:
            group["lr"] = start_lr * (multiplier ** (step + 1))

        steps_run = step + 1

    if steps_run == 0:
        steps_run = n_steps

    model.eval()

    # Compute suggested LR: steepest descent point (where loss decreases fastest).
    # Use the LR at which the gradient of (smoothed) loss is most negative.
    suggested_lr = _compute_suggested_lr(lrs, losses)

    return LRFinderResult(
        lrs=lrs[:steps_run],
        losses=losses[:steps_run],
        suggested_lr=suggested_lr,
        min_loss=min_loss,
        min_loss_lr=min_loss_lr,
        diverged=diverged,
        steps_run=steps_run,
    )


def _compute_suggested_lr(lrs: list[float], losses: list[float]) -> float:
    """Find the LR with the steepest negative loss gradient.

    Uses a moving average of the loss gradient (5-point window) to find
    the LR where loss decreases fastest. Falls back to 1/10 of min_loss_lr
    if no good descent region is found.
    """
    if len(losses) < 5:
        return lrs[-1] * 0.1 if lrs else 1e-3

    losses_arr = _np.array(losses, dtype=_np.float64)
    # Compute gradient of smoothed loss
    grad = _np.gradient(losses_arr)
    # Smooth gradient with a 5-point moving average
    from numpy import convolve as _convolve

    kernel = _np.ones(5) / 5.0
    grad_smooth = _convolve(grad, kernel, mode="same")

    # Find the index with the most negative gradient (steepest descent)
    best_idx = int(_np.argmin(grad_smooth))
    if best_idx < len(lrs) and grad_smooth[best_idx] < 0:
        return lrs[best_idx]

    # Fallback: 1/10 of the LR at minimum loss
    min_idx = int(_np.argmin(losses_arr))
    if min_idx < len(lrs):
        return lrs[min_idx] * 0.1

    return 1e-3


__all__ = [
    "LRFinderResult",
    "lr_find",
]