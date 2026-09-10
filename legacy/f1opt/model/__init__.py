"""``f1opt.model`` — Performance model and optimization.

Public:

- :class:`f1opt.model.surrogate.SurrogateModel` — Multi-task performance model
- :func:`f1opt.model.surrogate.predict_lap_time` — Lap time prediction (backward-compatible)
- :func:`f1opt.model.surrogate.predict_full` — Rich prediction (lap/sector/response)
- :data:`f1opt.model.surrogate.MODEL_VERSION` — Public version constant
- :func:`f1opt.model.train.train` — Synthetic data training
"""

from __future__ import annotations

from f1opt.model.lr_finder import LRFinderResult, lr_find
from f1opt.model.optimizer import SearchOptimizer, SearchResult, search_setup
from f1opt.model.surrogate import (
    MODEL_VERSION,
    SurrogateModel,
    predict_full,
    predict_lap_time,
)
from f1opt.model.train import train

__all__ = [
    "LRFinderResult",
    "MODEL_VERSION",
    "SearchOptimizer",
    "SearchResult",
    "SurrogateModel",
    "lr_find",
    "predict_full",
    "predict_lap_time",
    "search_setup",
    "train",
]
