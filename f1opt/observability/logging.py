"""structlog configuration for F1OPT (Iter-06).

Provides :func:`configure_structlog` (called once at API startup) and
:func:`get_logger` (a drop-in replacement for ``logging.getLogger`` that
returns a structlog-bound logger compatible with ``log.debug/info/warning/
exception`` calls).
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_structlog(json: bool = True) -> None:
    """Configure structlog + stdlib logging integration.

    JSON renderer for production, console renderer for dev. Called once at
    startup (e.g. by :func:`f1opt.api.runner.run`). After this call,
    :func:`get_logger` returns loggers whose output is structured JSON
    (or a human-readable console dump when ``json=False``).
    """
    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    # Also route stdlib logging through structlog (message-only format so the
    # JSON/console-rendered event string is emitted verbatim).
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to *name*.

    Drop-in for ``logging.getLogger(__name__)``: the returned object supports
    ``log.debug/info/warning/error/exception`` with the same call shapes used
    by the existing stdlib call sites.
    """
    return structlog.get_logger(name)


__all__ = ["configure_structlog", "get_logger"]
