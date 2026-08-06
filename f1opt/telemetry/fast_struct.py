"""F1 25 / 2026 UDP telemetry: packet optimizations for low-latency parsing.

Iter-13: pre-compiled struct module helpers and cached field-name maps
for faster packet field access during high-frequency (60 Hz) parsing.
"""

from __future__ import annotations

import struct
from typing import ClassVar


class FastStruct:
    """Pre-compiled struct wrapper with cached unpack and field lookup."""

    _fmt: str
    _struct: struct.Struct
    _names: tuple[str, ...]
    _size: int

    __slots__ = ("_fmt", "_struct", "_names", "_size")

    def __init__(self, fmt: str, names: tuple[str, ...]) -> None:
        self._fmt = fmt
        self._struct = struct.Struct(fmt)
        self._names = names
        self._size = self._struct.size

    @property
    def size(self) -> int:
        return self._size

    def unpack(self, data: bytes) -> dict[str, int | float]:
        vals = self._struct.unpack(data[: self._size])
        return dict(zip(self._names, vals))

    def unpack_into_numpy(self, data: bytes, arr):
        vals = self._struct.unpack(data[: self._size])
        arr[:] = vals


# Pre-compiled common formats
_F32 = struct.Struct("<f")
_H16 = struct.Struct("<H")
_B8 = struct.Struct("<B")
_I32 = struct.Struct("<I")


def parse_float_le(data: bytes, offset: int = 0) -> float:
    return _F32.unpack(data[offset : offset + 4])[0]


def parse_uint16_le(data: bytes, offset: int = 0) -> int:
    return _H16.unpack(data[offset : offset + 2])[0]


def parse_uint32_le(data: bytes, offset: int = 0) -> int:
    return _I32.unpack(data[offset : offset + 4])[0]


def parse_uint8(data: bytes, offset: int = 0) -> int:
    return _B8.unpack(data[offset : offset + 1])[0]
