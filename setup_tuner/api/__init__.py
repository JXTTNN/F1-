"""API 服务层 - REST 端点与 WebSocket 网关。

导出：
    - :data:`router`     — REST 路由（统一前缀 /api/v1）
    - :data:`ws_router`  — WebSocket 路由
    - :class:`WSManager` — WebSocket 连接管理器
    - :func:`ok` / :func:`fail` / :func:`error` — 响应信封工具
    - :class:`Envelope` / :class:`ApiError` — 信封模型与业务异常
"""

from __future__ import annotations

from .envelope import ApiError, Envelope, error, fail, ok
from .routes import router
from .ws import WSManager, ws_router

__all__ = [
    # 路由
    "router",
    "ws_router",
    "WSManager",
    # 信封
    "Envelope",
    "ApiError",
    "ok",
    "fail",
    "error",
]
