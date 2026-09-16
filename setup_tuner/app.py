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
import queue
import sys
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from setup_tuner.api.routes import router
from setup_tuner.api.ws import WSManager, ws_router
from setup_tuner.config import Config, load_config
from setup_tuner.db.store import Store
from setup_tuner.feedback.iteration import IterationService
from setup_tuner.feedback.service import FeedbackService
from setup_tuner.telemetry.lap_aggregator import LapAggregator
from setup_tuner.telemetry.listener import TelemetryListener
from setup_tuner.telemetry.stream import TelemetryStream
from setup_tuner.telemetry.style_extractor import StyleExtractor

logger = logging.getLogger(__name__)

# UI 静态资源目录（setup_tuner/ui/）
# 多候选路径探测，兼容开发模式与 Nuitka onefile/standalone 模式
def _find_ui_dir() -> Path:
    """查找 UI 静态资源目录，兼容开发模式和 Nuitka 打包模式。

    在 Nuitka onefile 模式下，__file__ 可能指向临时解压目录中的虚拟路径，
    数据文件实际位于 <temp_dir>/setup_tuner/ui/。
    通过多候选路径探测确保在所有模式下都能正确定位。
    """
    candidates = [
        # 1. 基于 __file__（开发模式：setup_tuner/app.py → setup_tuner/ui/）
        Path(__file__).resolve().parent / "ui",
        # 2. 基于 sys.executable + setup_tuner/ui（Nuitka onefile/standalone）
        Path(sys.executable).resolve().parent / "setup_tuner" / "ui",
        # 3. 基于 sys.executable + ui（Nuitka onefile 根目录极端情况）
        Path(sys.executable).resolve().parent / "ui",
    ]
    for candidate in candidates:
        if (candidate / "index.html").exists():
            logger.debug("UI 目录定位成功: %s", candidate)
            return candidate
    logger.warning(
        "UI 目录未找到，尝试过的路径: %s",
        [str(c) for c in candidates],
    )
    return candidates[0]


_UI_DIR = _find_ui_dir()


# =========================================================================== #
# 生命周期管理
# =========================================================================== #
def _init_app_services(app: FastAPI, config: Config) -> tuple[Store, TelemetryListener]:
    """初始化 Store、业务服务、遥测流与监听器，返回 (store, listener)。"""
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
    app.state.lap_aggregator = LapAggregator()
    app.state.style_extractor = StyleExtractor()
    app.state.lap_write_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    app.state.lap_writer = threading.Thread(
        target=_lap_writer_loop, args=(app,), name="f1opt-lap-writer",
        daemon=True,
    )
    app.state.lap_writer.start()
    listener = TelemetryListener(
        host=config.udp_host, port=config.udp_port,
    )
    # 包处理函数同时存入 state：遥测模拟器（无真实 F1 游戏时）需要复用同一
    # 处理链路，否则模拟帧无人消费，/suggest 永远拿不到整圈统计。
    app.state.packet_handler = _make_packet_handler(app)
    listener.add_handler(app.state.packet_handler)
    listener.add_raw_handler(_make_raw_packet_handler(app))
    app.state.telemetry_listener = listener

    # ④ WS 管理器
    app.state.ws_manager = WSManager()

    # ⑤ 当前赛道（None 表示未选择）
    app.state.current_track_id = None
    app.state.current_track_source = "manual"
    return store, listener


def _lap_writer_loop(app: FastAPI) -> None:
    """整圈落库线程：把队列里的整圈快照写入 lap_record + driver_style（EMA）。

    质量门槛（不达标不入库）：样本帧数 ≥ 3000（≈50 秒）。
    第 12 维（圈速一致性）由同车手同赛道的圈史计算后追加到向量。
    EMA：新向量 = 0.8 × 旧向量 + 0.2 × 本圈向量（样本数 +1）。
    """
    MIN_FRAMES = 3000
    EMA_ALPHA = 0.2
    while True:
        try:
            job = app.state.lap_write_queue.get(timeout=1.0)
        except queue.Empty:
            # 正常的 1 秒轮询超时，继续等待
            continue
        except Exception:
            # 队列被替换/关闭等异常：记录后继续，避免本线程静默死循环
            logger.warning("整圈落库队列读取异常", exc_info=True)
            continue
        try:
            snapshot = job.get("snapshot") or {}
            track_id = job.get("track_id")
            if not track_id or snapshot.get("lap_frames", 0) < MIN_FRAMES:
                continue
            store = getattr(app.state, "store", None)
            if store is None:
                continue
            driver_id = store.get_or_create_driver("默认车手")
            lap_time_ms = snapshot.get("lap_time_ms")
            records = store.get_lap_records(track_id, driver_id, limit=10)
            times = [r["lap_time_ms"] for r in records
                     if r.get("lap_time_ms")]
            mean_t = sum(times) / len(times) if times else 0.0
            var = (sum((t - mean_t) ** 2 for t in times) / len(times)
                   if times else 0.0)
            cv = (var ** 0.5) / mean_t if mean_t else 1.0
            consistency = max(0.0, min(1.0, 1.0 - cv / 0.05))

            old = store.get_driver_style(driver_id, track_id)
            new_vec = list(job.get("style_vector") or [])
            if len(new_vec) >= 11:
                base = old["vector"] if old and len(old["vector"]) >= 11                     else new_vec[:11]
                blended = [
                    round((1 - EMA_ALPHA) * b + EMA_ALPHA * n, 4)
                    for b, n in zip(base, new_vec[:11], strict=True)
                ]
                blended.append(round(consistency, 4))
                sample_count = (old["sample_count"] + 1) if old else 1
                store.save_driver_style(driver_id, track_id, blended, sample_count)

            snapshot["lap_consistency"] = round(consistency, 4)
            snapshot["driver_id"] = driver_id
            store.save_lap_record(
                driver_id, track_id, snapshot,
                lap_number=snapshot.get("lap_number"),
                lap_time_ms=lap_time_ms,
                is_valid=True,
                session_uid=None,
                setup_id=None,
            )
        except Exception:
            logger.exception("整圈落库失败，跳过该圈")


def _make_packet_handler(app: FastAPI) -> Callable[[dict[str, Any]], None]:
    """构造 UDP 包处理函数：写入 TelemetryStream，并喂给整圈聚合器。

    - Packet 6 (CarTelemetry)  → ``LapAggregator.on_telemetry``
    - Packet 2 (LapData)       → ``LapAggregator.on_lap_data``
    - Packet 13 (MotionEx)     → ``LapAggregator.on_motion_ex``
    这样 ``/suggest`` 才能拿到 ``max_speed`` / ``avg_steer`` / ``max_steer`` /
    ``on_straight`` 等整圈统计（此前只喂单帧，导致 5 条遥测规则永不触发），
    以及底板离地高度（规则9 刮底检测的唯一信号源）。
    """
    def _on_packet(parsed: dict[str, Any]) -> None:
        packet_id = parsed.get("packet_id")
        if packet_id is None:
            return
        app.state.telemetry_stream.update(int(packet_id), parsed)
        aggregator = getattr(app.state, "lap_aggregator", None)
        if aggregator is None:
            return
        if packet_id == 6:
            aggregator.on_telemetry(parsed)
            extractor = getattr(app.state, "style_extractor", None)
            if extractor is not None:
                extractor.on_telemetry(parsed)
        elif packet_id == 2:
            aggregator.on_lap_data(parsed)
            extractor = getattr(app.state, "style_extractor", None)
            if extractor is not None:
                extractor.on_lap_data(parsed)
        elif packet_id == 13:
            aggregator.on_motion_ex(parsed)
        elif packet_id == 7:
            # Packet 7 (CarStatus)：轮胎配方/胎龄/燃油/ERS/刹车平衡。
            # parse_car_status 早已实现但此前从未分发，导致这些状态量
            # 解析出来即丢弃（配方相关的阈值区分、刹车平衡核对全部失效）。
            aggregator.on_car_status(parsed)
        # task-62 M1：一圈结束时把整圈快照与风格向量交给落库线程
        completed = aggregator.take_completed_lap()
        if completed is not None:
            writer = getattr(app.state, "lap_write_queue", None)
            if writer is not None:
                writer.put({
                    "snapshot": completed,
                    "track_id": getattr(app.state, "current_track_id", None),
                    "style_vector": (
                        getattr(app.state, "style_extractor", None).
                        take_completed()
                    ),
                })
    return _on_packet


def _make_raw_packet_handler(app: FastAPI) -> Callable[[bytes, dict[str, Any] | None], None]:
    """构造原始字节处理函数：将原始字节 + 解析结果传给录制器。"""
    def _on_raw_packet(data: bytes, parsed: dict[str, Any] | None) -> None:
        recorder = getattr(app.state, "telemetry_recorder", None)
        if recorder is not None and recorder.is_recording:
            recorder.on_raw_packet(data, parsed)
    return _on_raw_packet


def _start_telemetry_listener(listener: TelemetryListener, config: Config) -> None:
    """启动 UDP 监听（后台线程，端口占用时降级运行不崩溃）。"""
    try:
        listener.start()
        logger.info(
            "telemetry listener started on %s:%d",
            config.udp_host, config.udp_port,
        )
    except OSError as e:
        logger.warning(
            "telemetry listener failed to start on %s:%d: %s "
            "(API will run without telemetry)",
            config.udp_host, config.udp_port, e,
        )


def _cleanup_app_services(listener: TelemetryListener, store: Store) -> None:
    """停止 UDP 监听并关闭 Store（容错，失败仅记日志）。"""
    try:
        listener.stop()
    except Exception:
        logger.exception("telemetry listener stop failed")
    try:
        store.close()
    except Exception:
        logger.exception("store close failed")
    logger.info("F1OPT app stopped")


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
    store, listener = _init_app_services(app, config)
    _start_telemetry_listener(listener, config)
    logger.info("F1OPT app started, data_dir=%s", config.data_dir)

    try:
        yield
    finally:
        _cleanup_app_services(listener, store)


# =========================================================================== #
# 全局异常处理（统一 {code, message, data} 信封）
# =========================================================================== #
async def _http_exception_handler(
    request: Request, exc: StarletteHTTPException,
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
    request: Request, exc: RequestValidationError,
) -> JSONResponse:
    """将请求参数校验异常转为统一信封响应（400）。"""
    body = {
        "code": 4000,
        "message": "请求参数校验失败",
        "data": {"errors": exc.errors()},
    }
    return JSONResponse(status_code=422, content=body)


async def _generic_exception_handler(
    request: Request, exc: Exception,
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
def _mount_static_assets(app: FastAPI) -> None:
    """挂载 UI 静态资源（SVG 赛道图 + index.html/app.js/style.css）。"""
    if not _UI_DIR.exists():
        return
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


def _register_exception_handlers(app: FastAPI) -> None:
    """注册全局异常处理器（统一 {code, message, data} 信封）。"""
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _generic_exception_handler)


def _register_root_route(app: FastAPI) -> None:
    """注册根路径路由，返回前端 index.html。"""
    @app.get("/", tags=["root"])
    async def root() -> HTMLResponse:
        """根路径 —— 返回前端 index.html，用户打开浏览器即见界面。"""
        index_path = _UI_DIR / "index.html"
        if index_path.exists():
            return HTMLResponse(index_path.read_text(encoding="utf-8"))
        # UI 文件不存在时回退到提示页
        return HTMLResponse(
            "<html><body><h1>F1OPT</h1><p>UI 未找到，请检查安装。</p></body></html>",
            status_code=404,
        )


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
        description="F1 25 Setup Tuner — 纯确定性规则引擎",
        version="0.1.0",
        lifespan=_lifespan,
    )

    # 挂载配置到 app.state
    app.state.config = config

    # 注册路由
    app.include_router(router)
    app.include_router(ws_router)

    _mount_static_assets(app)
    _register_exception_handlers(app)
    _register_root_route(app)
    return app