"""Shared numeric helpers.

Small, dependency-light numeric utilities that were previously duplicated
across several modules (clamping, safe float coercion, coefficient of
variation). Keeping a single implementation avoids drift between copies.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["clamp", "clamp01", "safe_float", "coefficient_of_variation"]


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to the closed interval ``[low, high]``."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def clamp01(x: float) -> float:
    """Clamp ``x`` to ``[0, 1]`` and return it as a ``float``."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce ``value`` to ``float``; return ``default`` on ``None`` / failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coefficient_of_variation(values: list[float] | np.ndarray) -> float:
    """Coefficient of variation ``std / |mean|``.

    Returns ``0.0`` for empty input, when all values are non-finite, or when
    the mean is (effectively) zero. Non-finite values are ignored.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    mean = float(np.mean(arr))
    if abs(mean) < 1e-12 or not math.isfinite(mean):
        return 0.0
    return float(np.std(arr) / abs(mean))
