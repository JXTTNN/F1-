"""FastAPI 应用工厂 —— 创建并配置 FastAPI 应用实例。

对齐 design.md 2.1.2 / 2.8 与 spec FR-API-* 需求：

    - 创建 FastAPI 实例；
    - 挂载静态资源（ui/ 目录，含 SVG/HTML/JS/CSS）；
    - 注册 REST 路由（from api.routes import router）；
    - 注册 WebSocket 路由（from api.ws import ws_router）；
    - 生命周期管理：启动时初始化 Store + TelemetryListener，停止时清理；
    - 全局异常处理（统一 {code, message, data} 信封）。

使用方式::

    from setup_tuner.app import create_app
    app = create_app()
    # 然后 uvicorn.run(app, host=..., port=...)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from setup_tuner.config import Config, load_config
from setup_tuner.db.store import Store
from setup_tuner.feedback.iteration import IterationService
from setup_tuner.feedback.service import FeedbackService
from setup_tuner.telemetry.listener import TelemetryListener
from setup_tuner.telemetry.stream import TelemetryStream

from setup_tuner.api.routes import router
from setup_tuner.api.ws import WSManager, ws_router

logger = logging.getLogger(__name__)

# UI 静态资源目录（setup_tuner/ui/）
_UI_DIR = Path(__file__).resolve().parent / "ui"


# =========================================================================== #
# 生命周期管理
# =========================================================================== #
@asynccontextmanager
async def _lifespan(app: FastAPI):
    """应用生命周期：启动时初始化服务，停止时清理。

    初始化：
        - Store（SQLite）
        - FeedbackService / IterationService
        - TelemetryStream（最新帧缓存）
        - TelemetryListener（UDP 监听，后台线程）
        - WSManager（WebSocket 连接管理器）

    清理：
        - 停止 TelemetryListener
        - 关闭 Store
    """
    config: Config = app.state.config

    # ① Store
    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "f1opt.db"
    store = Store(str(db_path))
    app.state.store = store

    # ② 业务服务
    app.state.feedback_service = FeedbackService(store)
    app.state.iteration_service = IterationService(store)

    # ③ 遥测
    app.state.telemetry_stream = TelemetryStream()
    listener = TelemetryListener(
        host=config.udp_host, port=config.udp_port
    )
    # 注册 handler：将解析结果写入 TelemetryStream
    def _on_packet(parsed: dict[str, Any]) -> None:
        packet_id = parsed.get("packet_id")
        if packet_id is not None:
            app.state.telemetry_stream.update(int(packet_id), parsed)

    listener.add_handler(_on_packet)
    app.state.telemetry_listener = listener

    # ④ WS 管理器
    app.state.ws_manager = WSManager()

    # ⑤ 当前赛道（None 表示未选择）
    app.state.current_track_id = None
    app.state.current_track_source = "manual"

    # 启动 UDP 监听（后台线程，容错不崩溃）
    try:
        listener.start()
        logger.info(
            "telemetry listener started on %s:%d",
            config.udp_host,
            config.udp_port,
        )
    except OSError as e:
        # UDP 端口占用不阻止 API 启动（降级运行）
        logger.warning(
            "telemetry listener failed to start on %s:%d: %s "
            "(API will run without telemetry)",
            config.udp_host,
            config.udp_port,
            e,
        )

    logger.info("F1OPT app started, data_dir=%s", config.data_dir)

    try:
        yield
    finally:
        # 清理
        try:
            listener.stop()
        except Exception:
            logger.exception("telemetry listener stop failed")
        try:
            store.close()
        except Exception:
            logger.exception("store close failed")
        logger.info("F1OPT app stopped")


# =========================================================================== #
# 全局异常处理（统一 {code, message, data} 信封）
# =========================================================================== #
async def _http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """将 HTTPException 转为统一信封响应。

    FastAPI 的 HTTPException.detail 可能是字符串或信封 dict（由 envelope.fail 构造）。
    """
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        # 已经是信封格式（由 envelope.fail/error 构造）
        body = exc.detail
    else:
        # 原生 HTTPException（如 404 路由不存在）
        body = {
            "code": exc.status_code,
            "message": str(exc.detail) if exc.detail else "请求失败",
            "data": None,
        }
    return JSONResponse(status_code=exc.status_code, content=body)


async def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """将请求参数校验异常转为统一信封响应（400）。"""
    body = {
        "code": 4000,
        "message": "请求参数校验失败",
        "data": {"errors": exc.errors()},
    }
    return JSONResponse(status_code=422, content=body)


async def _generic_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """将未捕获异常转为统一信封响应（500）。"""
    logger.exception("unhandled exception on %s %s", request.method, request.url.path)
    body = {
        "code": -1,
        "message": f"服务内部错误：{exc}",
        "data": None,
    }
    return JSONResponse(status_code=500, content=body)


# =========================================================================== #
# 应用工厂
# =========================================================================== #
def create_app(config: Config | None = None) -> FastAPI:
    """创建并配置 FastAPI 应用实例。

    Args:
        config: 运行时配置；None 时自动加载 .env。

    Returns:
        配置好的 FastAPI 实例。
    """
    if config is None:
        config = load_config()

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = FastAPI(
        title="F1OPT 赛车调教优化助手",
        description="EA F1 2026 Setup Tuner — 纯确定性规则引擎",
        version="0.1.0",
        lifespan=_lifespan,
    )

    # 挂载配置到 app.state
    app.state.config = config

    # 注册路由
    app.include_router(router)
    app.include_router(ws_router)

    # 挂载静态资源（ui/ 目录，含 SVG/HTML/JS/CSS）
    if _UI_DIR.exists():
        from fastapi.staticfiles import StaticFiles

        # 挂载 SVG 赛道图子目录
        tracks_dir = _UI_DIR / "tracks"
        if tracks_dir.exists():
            app.mount(
                "/static/tracks",
                StaticFiles(directory=str(tracks_dir)),
                name="static-tracks",
            )
        # 挂载 UI 根目录（index.html / app.js / style.css）
        app.mount(
            "/static",
            StaticFiles(directory=str(_UI_DIR)),
            name="static-ui",
        )

    # 注册全局异常处理器
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    app.add_exception_handler(Exception, _generic_exception_handler)

    # 根路径 → 返回 API 信息
    @app.get("/", tags=["root"])
    async def root() -> dict[str, Any]:
        """根路径 —— 返回 API 信息与前端入口。"""
        return {
            "name": "F1OPT 赛车调教优化助手",
            "version": "0.1.0",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "static": "/static/",
            "websocket": "/api/v1/ws",
        }

    return app