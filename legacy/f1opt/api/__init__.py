"""FastAPI backend (REST + WebSocket) for F1OPT."""

from f1opt.api.app import app, create_app

__all__ = ["app", "create_app"]
