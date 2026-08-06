"""Rust-style zero-cost error type for non-exceptional failure paths.

Avoids expensive Python exception stack frame construction for common
soft-failure patterns (missing data, out-of-range values, parse warnings).

Iter-14: Performance optimization — Result/Either pattern to avoid
throwing exceptions in hot paths like packet parsing and telemetry validation.
"""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E")


class Ok(Generic[T]):
    __slots__ = ("_value",)

    def __init__(self, value: T) -> None:
        self._value = value

    @property
    def is_ok(self) -> bool:
        return True

    @property
    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self._value

    def unwrap_or(self, _default: T) -> T:
        return self._value


class Err(Generic[T, E]):
    __slots__ = ("_error",)

    def __init__(self, error: E) -> None:
        self._error = error

    @property
    def is_ok(self) -> bool:
        return False

    @property
    def is_err(self) -> bool:
        return True

    def unwrap(self) -> T:
        raise ValueError(str(self._error))

    def unwrap_or(self, default: T) -> T:
        return default

    def error(self) -> E:
        return self._error
