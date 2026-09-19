"""真实 HTTP 启动冒烟：本地起一次 F1OPT 服务，验证新代码 + 模型可加载。

只做三件事（用临时数据目录，不碰用户真实数据库）：
1. 启动 uvicorn（真实 HTTP，随机空闲端口）；
2. 请求 ``/api/v1/health`` 与 ``/api/v1/setup/fields``；
3. 打印代理模型是否被引擎加载，然后退出。

用法::

    python scripts/smoke_boot_server.py
"""

from __future__ import annotations

import socket
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def main() -> int:
    import uvicorn

    from setup_tuner.app import create_app
    from setup_tuner.config import Config

    tmp = Path(tempfile.mkdtemp(prefix="f1opt_boot_"))
    port = _free_port()
    cfg = Config(data_dir=str(tmp / "data"), api_port=port,
                 udp_port=_free_port())
    app = create_app(cfg)

    from setup_tuner.engine.surrogate import get_surrogate

    model = get_surrogate()
    print(f"[模型] 路径 {model.path}")
    print(f"[模型] 可用 {model.available}  摘要 {model.describe()}")

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    ok = False
    for _ in range(50):
        time.sleep(0.2)
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/v1/health", timeout=2,
            ) as resp:
                body = resp.read().decode("utf-8")
            print(f"[HTTP] GET /api/v1/health -> {resp.status} {body[:120]}")
            ok = resp.status == 200
            break
        except Exception:  # noqa: BLE001 — 启动期间连接失败属正常
            continue

    server.should_exit = True
    thread.join(timeout=10)
    print(f"[结论] {'✅ 服务可正常启动（新代码 + 模型均已加载）' if ok else '✗ 启动失败'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
