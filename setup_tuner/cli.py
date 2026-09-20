"""一键启动入口 —— 加载配置 → 端口检测 → 启动服务 → 打开浏览器。

对齐 design.md 2.2 cli.py 职责与 spec FR-DEP-01 ~ FR-DEP-04：

    1. 加载配置（config.load_config）；
    2. 端口占用检测（检查 api_port 是否被占用）；
    3. 启动 uvicorn 服务（create_app）；
    4. 自动打开浏览器（webbrowser.open）；
    5. 优雅退出处理（KeyboardInterrupt → 停止服务）。

入口点：``python -m setup_tuner.cli`` 或 ``f1opt``（pyproject.scripts）。

命令行参数（2026-09-20 补全）：此前 ``main(argv)`` 忽略全部参数 ——
``f1opt --help`` 会**直接启动服务**（挂住终端），README 里写的
``f1opt feedback/search`` 等子命令也从未存在。现在提供真实、最小、
不撒谎的参数集：``--host/--port/--no-browser/--keep-telemetry/--version/-h``。
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import threading
import webbrowser
from time import sleep
from typing import Any

from setup_tuner.app import create_app
from setup_tuner.config import load_config

logger = logging.getLogger(__name__)


# =========================================================================== #
# 端口占用检测
# =========================================================================== #
def is_port_in_use(host: str, port: int) -> bool:
    """检测指定端口是否被占用。

    Args:
        host: 监听地址。
        port: 端口号。

    Returns:
        True 表示端口已被占用。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


# =========================================================================== #
# 浏览器自动打开
# =========================================================================== #
def _open_browser_delayed(url: str, delay: float = 1.5) -> None:
    """延迟打开浏览器（等待服务启动）。

    Args:
        url: 要打开的 URL。
        delay: 延迟秒数（等待 uvicorn 启动完成）。
    """
    sleep(delay)
    try:
        webbrowser.open(url)
        logger.info("浏览器已打开：%s", url)
    except Exception:
        logger.warning("无法自动打开浏览器，请手动访问：%s", url)


# =========================================================================== #
# 主入口
# =========================================================================== #
def _check_port_available(host: str, port: int) -> int:
    """检测端口占用，被占用时返回 1，否则返回 0。"""
    if is_port_in_use(host, port):
        print(
            f"\n❌ 端口 {port} 已被占用，请关闭占用程序或修改 .env 中的 API_PORT\n"
            f"   提示：可执行 `netstat -ano | findstr :{port}` 查看占用进程\n",
            file=sys.stderr,
        )
        return 1
    return 0


def _start_browser_thread(url: str) -> threading.Thread:
    """启动延迟打开浏览器的后台线程。"""
    browser_thread = threading.Thread(
        target=_open_browser_delayed,
        args=(url,),
        daemon=True,
        name="f1opt-browser",
    )
    browser_thread.start()
    return browser_thread


def _print_startup_banner(url: str, config: Any) -> None:
    """打印启动横幅。"""
    print("\n🏁 F1OPT 赛车调教优化助手已启动")
    print(f"   服务地址：{url}")
    print(f"   API 文档：{url}/docs")
    print(f"   WebSocket：{url}/api/v1/ws")
    print(f"   遥测监听：{config.udp_host}:{config.udp_port}")
    print("   按 Ctrl+C 退出\n")


def _run_uvicorn(app: Any, host: str, port: int, log_level: str) -> int:
    """启动 uvicorn 服务，返回退出码。"""
    try:
        import uvicorn

        uvicorn.run(app, host=host, port=port, log_level=log_level.lower())
    except KeyboardInterrupt:
        print("\n\n正在停止服务...")
        logger.info("received KeyboardInterrupt, shutting down")
    except Exception as e:
        logger.exception("uvicorn run failed")
        print(f"\n❌ 启动失败：{e}\n", file=sys.stderr)
        return 2
    print("服务已停止。")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器（真实存在的参数；不虚构子命令）。"""
    from setup_tuner import __version__ as _version

    parser = argparse.ArgumentParser(
        prog="f1opt",
        description=(
            "F1OPT — F1 2026 赛车调校助手：接收游戏 UDP 遥测 → 车手反馈 → "
            "参数矩阵给方向 + 神经网络模拟优化 → 给出整体性调教建议。"
        ),
        epilog=(
            "启动后浏览器打开实时面板；Ctrl+C 退出。"
            "退出时默认清理派生遥测数据（录制永久保留），"
            "加 --keep-telemetry 可保留。"
        ),
    )
    parser.add_argument("--host", default=None, help="API 监听地址（默认取配置 api_host）")
    parser.add_argument("--port", type=int, default=None, help="API 端口（默认取配置 api_port）")
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    parser.add_argument(
        "--keep-telemetry", action="store_true",
        help="退出时保留派生遥测数据（等价 F1OPT_KEEP_TELEMETRY=1）",
    )
    parser.add_argument(
        "--version", action="version", version=f"f1opt {_version}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """一键启动主入口。

    流程：
        1. 解析参数（--help/--version 直接返回，不启动服务）；
        2. 加载配置（命令行参数优先于环境变量）；
        3. 端口占用检测；
        4. 启动 uvicorn + 自动开浏览器；
        5. 优雅退出（KeyboardInterrupt）→ 清理遥测数据（录制保留）。

    Args:
        argv: 命令行参数；None 时取 sys.argv[1:]。

    Returns:
        退出码：0=正常退出（含 --help/--version），1=端口占用，2=其他错误。
    """
    from dataclasses import replace

    args = _build_parser().parse_args(argv)

    # ① 加载配置（命令行覆盖环境变量）
    config = load_config()
    if args.host or args.port or args.keep_telemetry:
        config = replace(
            config,
            api_host=args.host or config.api_host,
            api_port=args.port or config.api_port,
            keep_telemetry=config.keep_telemetry or args.keep_telemetry,
        )
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    host = config.api_host
    port = config.api_port

    # ② 端口占用检测
    if _check_port_available(host, port):
        return 1

    # ③ 创建应用
    app = create_app(config)

    # ④ 延迟打开浏览器（后台线程）
    url = f"http://{host}:{port}"
    if not args.no_browser:
        _start_browser_thread(url)

    # ⑤ 启动 uvicorn
    _print_startup_banner(url, config)
    return _run_uvicorn(app, host, port, config.log_level)


if __name__ == "__main__":
    sys.exit(main())