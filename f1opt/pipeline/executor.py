"""Thread-safe background task execution for long-running model operations.

Provides a simple thread-pool executor for CPU-bound tasks (model training,
optimization) that should not block the asyncio event loop. Used by the API
server to offload heavy computation.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
from collections.abc import Callable
from typing import Any, TypeVar

from f1opt.observability.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")

_max_workers = 4

executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=_max_workers,
    thread_name_prefix="f1opt-worker",
)


async def run_in_thread(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    loop = asyncio.get_running_loop()
    bound = functools.partial(func, *args, **kwargs)
    try:
        return await loop.run_in_executor(executor, bound)
    except Exception:
        log.exception("background task failed: %s", func.__name__)
        raise
