"""OpenTelemetry tracing integration (Iter-170, best-effort).

Wraps the OpenTelemetry API so the rest of the codebase can call
:func:`span` / :func:`get_tracer` unconditionally — when OTel is not
installed or no exporter is configured, the calls are no-ops.

Configuration:
- ``F1OPT_OTEL_EXPORT``: OTLP exporter endpoint (e.g.
  ``http://otel-collector:4317``). When unset, tracing is disabled
  (no-op spans).
- ``F1OPT_OTEL_SERVICE_NAME``: service name reported to the collector
  (default: ``f1opt``).
- ``F1OPT_OTEL_RESOURCE_ATTRS``: comma-separated ``key=value`` pairs
  appended to the resource (e.g. ``env=prod,team=f1opt``).

When OpenTelemetry is installed AND ``F1OPT_OTEL_EXPORT`` is set, a
real :class:`TracerProvider` is configured with an OTLP exporter and
batch span processor. Otherwise, a no-op tracer is used (the
``span`` context manager still works as a ``nullcontext``).

Usage::

    from f1opt.observability.tracing import span, get_tracer

    with span("predict_lap_time", track_id="suzuka"):
        result = predict_lap_time(setup, track_id)

The ``span`` context manager accepts arbitrary keyword attributes
which are recorded as span attributes when OTel is active.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

__all__ = ["span", "get_tracer", "is_tracing_enabled", "shutdown_tracing"]


def is_tracing_enabled() -> bool:
    """Return True if OpenTelemetry export is configured."""
    return bool(os.environ.get("F1OPT_OTEL_EXPORT"))


def _otel_available() -> bool:
    """Return True if the ``opentelemetry`` package is importable."""
    try:
        import opentelemetry  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def get_tracer() -> Any:
    """Return a tracer — real OTel tracer if configured, else a no-op.

    The returned object's ``start_as_current_span(name, attributes=...)``
    method is always safe to call (returns a context manager).
    """
    if not is_tracing_enabled() or not _otel_available():
        return _NoopTracer()
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]
        return trace.get_tracer("f1opt")
    except Exception:
        return _NoopTracer()


class _NoopSpan:
    """No-op span — does nothing, supports ``set_attribute``/``record_exception``."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def record_exception(self, exc: BaseException) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(self, *exc: Any) -> None:
        pass


class _NoopTracer:
    """No-op tracer — returns :class:`_NoopSpan` for all calls."""

    def start_as_current_span(
        self, name: str, **kwargs: Any
    ) -> _NoopSpan:
        return _NoopSpan()

    def start_span(self, name: str, **kwargs: Any) -> _NoopSpan:
        return _NoopSpan()


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Context manager that opens a tracing span.

    When OTel is active, attributes are recorded on the span. When
    inactive, this is a no-op ``nullcontext``. Always safe to use.

    Example::

        with span("search_setup", track="suzuka", iterations=80):
            result = search_setup("suzuka", iterations=80)
    """
    tracer = get_tracer()
    if isinstance(tracer, _NoopTracer):
        # Fast path: no-op
        with _NoopSpan() as s:
            yield s
        return
    # Real OTel path
    try:
        cm = tracer.start_as_current_span(name, attributes=attributes)
        with cm as s:
            yield s
    except Exception:
        # Best-effort: never raise from tracing.
        with _NoopSpan() as s:
            yield s


def shutdown_tracing() -> None:
    """Flush pending spans and shut down the tracer provider.

    Safe to call multiple times. Called automatically on process exit
    via :mod:`atexit` (registered on first import when OTel is active).
    """
    if not is_tracing_enabled() or not _otel_available():
        return
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            provider.shutdown()
    except Exception:
        pass


# Auto-configure on import when OTel export is set.
if is_tracing_enabled() and _otel_available():
    try:
        import atexit

        from opentelemetry import trace  # type: ignore[import-not-found]
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import-not-found]
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource  # type: ignore[import-not-found]
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
        from opentelemetry.sdk.trace.export import (  # type: ignore[import-not-found]
            BatchSpanProcessor,
        )

        service_name = os.environ.get("F1OPT_OTEL_SERVICE_NAME", "f1opt")
        resource_attrs: dict[str, str] = {"service.name": service_name}
        extra = os.environ.get("F1OPT_OTEL_RESOURCE_ATTRS", "")
        for pair in extra.split(","):
            pair = pair.strip()
            if "=" in pair:
                k, v = pair.split("=", 1)
                resource_attrs[k.strip()] = v.strip()
        resource = Resource.create(resource_attrs)
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(
            endpoint=os.environ["F1OPT_OTEL_EXPORT"],
            insecure=True,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        atexit.register(shutdown_tracing)
    except Exception:
        # Best-effort: if OTel setup fails, fall back to no-op.
        pass
