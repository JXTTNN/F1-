"""Console entrypoint: boot the F1OPT API server with uvicorn.

Run with::

    python -m f1opt.api.runner

or set ``F1OPT_API_HOST`` / ``F1OPT_API_PORT`` to override the bind address.
"""

from __future__ import annotations

import uvicorn

from f1opt.config import get_settings
from f1opt.observability.logging import configure_structlog


def run() -> None:
    """Boot ``uvicorn f1opt.api.app:app`` using the configured host/port."""
    configure_structlog()
    settings = get_settings()
    uvicorn.run(
        "f1opt.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
    )


if __name__ == "__main__":
    run()
