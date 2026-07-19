"""Factory building a FastAPI app augmented with the extended router.

Wraps :func:`f1opt.api.app.create_app` and mounts
:mod:`f1opt.api.extended` so the core app module stays unmodified. Tests build
the app via :func:`create_extended_app` (listener disabled by default).
"""

from __future__ import annotations

from fastapi import FastAPI

from f1opt.api.app import create_app
from f1opt.api.extended import router as extended_router

__all__ = ["create_extended_app"]


def create_extended_app(start_listener: bool = False) -> FastAPI:
    """Build the core app and include the extended API router.

    ``start_listener`` defaults to ``False`` so importing the factory (e.g. in
    tests) never binds a real UDP port; pass ``True`` to enable live telemetry.

    The core app mounts ``StaticFiles`` at ``/`` as its *last* route (best-effort
    UI). ``include_router`` appends after that mount, which would shadow the
    extended ``/api`` routes (POST -> 405, GET -> 404). To preserve the
    "static-mount-last" contract we temporarily lift that mount, include the
    extended router, then re-append the mount so it stays last.
    """
    app = create_app(start_listener=start_listener)
    routes = app.router.routes
    static_mounts = [r for r in routes if getattr(r, "name", None) == "static"]
    for mount in static_mounts:
        routes.remove(mount)
    app.include_router(extended_router)
    routes.extend(static_mounts)
    return app
