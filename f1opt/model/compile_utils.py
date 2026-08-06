from __future__ import annotations

import os
import torch
import functools


_is_windows = os.name == "nt"


def _should_use_compile() -> bool:
    return (
        not _is_windows
        and hasattr(torch, "compile")
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 7
    )


def compile_if_available(model: torch.nn.Module) -> torch.nn.Module:
    if not _should_use_compile():
        return model

    try:
        compiled = torch.compile(
            model,
            mode="reduce-overhead",
            fullgraph=False,
            dynamic=True,
        )
        compiled._original = model
        return compiled
    except Exception:
        return model


@functools.lru_cache(maxsize=1)
def get_device() -> str:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_device(model: torch.nn.Module, device: str | None = None) -> torch.nn.Module:
    if device is None:
        device = get_device()
    if device == "cpu":
        return model
    return model.to(device)


def ensure_tensor(x, device: str | None = None, dtype=torch.float32):
    if not isinstance(x, torch.Tensor):
        import numpy as np
        x = torch.tensor(np.asarray(x, dtype=np.float32), dtype=dtype)
    if device is not None and x.device.type != device:
        x = x.to(device)
    return x
