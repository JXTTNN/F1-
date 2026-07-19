"""F1 2026 Setup Optimization System.

Sub-packages by responsibility:

- ``telemetry``: UDP listener and packet parsers
- ``data``: Track database, setup schema, reference telemetry
- ``model``: Performance model and optimizers
- ``driver``: Driver profiles and driving style features
- ``feedback``: Driver feedback engine
- ``api``: FastAPI backend (REST + WebSocket)
- ``ui``: Dashboard frontend
- ``pipeline``: Training / inference orchestration
"""

__version__ = "0.1.0"
