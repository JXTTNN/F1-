"""一键启动入口 —— 加载配置 → 端口检测 → 启动服务 → 打开浏览器。

对齐 design.md 2.2 cli.py 职责与 spec FR-DEP-01 ~ FR-DEP-04：

    1. 加载配置（config.load_config）；
    2. 端口占用检测（检查 api_port 是否被占用）；
    3. 启动 uvicorn 服务（create_app）；
    4. 自动打开浏览器（webbrowser.open）；
    5. 优雅退出处理（KeyboardInterrupt → 停止服务）。

入口点：``python -m setup_tuner.cli`` 或 ``f1opt``（pyproject.scripts）。
"""

from __future__ import annotations

import logging
import socket
import sys
import threading
import webbrowser
from time import sleep

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
def main(argv: list[str] | None = None) -> int:
    """一键启动主入口。

    流程：
        1. 加载配置；
        2. 端口占用检测；
        3. 启动 uvicorn + 自动开浏览器；
        4. 优雅退出（KeyboardInterrupt）。

    Args:
        argv: 命令行参数（未使用，保留以备扩展）。

    Returns:
        退出码：0=正常退出，1=端口占用，2=其他错误。
    """
    # ① 加载配置
    config = load_config()
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    host = config.api_host
    port = config.api_port

    # ② 端口占用检测
    if is_port_in_use(host, port):
        print(
            f"\n❌ 端口 {port} 已被占用，请关闭占用程序或修改 .env 中的 API_PORT\n"
            f"   提示：可执行 `netstat -ano | findstr :{port}` 查看占用进程\n",
            file=sys.stderr,
        )
        return 1

    # ③ 创建应用
    app = create_app(config)

    # ④ 延迟打开浏览器（后台线程）
    url = f"http://{host}:{port}"
    browser_thread = threading.Thread(
        target=_open_browser_delayed,
        args=(url,),
        daemon=True,
        name="f1opt-browser",
    )
    browser_thread.start()

    # ⑤ 启动 uvicorn
    print("\n🏁 F1OPT 赛车调教优化助手已启动")
    print(f"   服务地址：{url}")
    print(f"   API 文档：{url}/docs")
    print(f"   WebSocket：{url}/api/v1/ws")
    print(f"   遥测监听：{config.udp_host}:{config.udp_port}")
    print("   按 Ctrl+C 退出\n")

    try:
        import uvicorn

        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=config.log_level.lower(),
        )
    except KeyboardInterrupt:
        print("\n\n正在停止服务...")
        logger.info("received KeyboardInterrupt, shutting down")
    except Exception as e:
        logger.exception("uvicorn run failed")
        print(f"\n❌ 启动失败：{e}\n", file=sys.stderr)
        return 2

    print("服务已停止。")
    return 0


if __name__ == "__main__":
    sys.exit(main())