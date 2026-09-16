"""前后端契约一致性测试。

防止两类退化：

1. **前端调用了不存在的后端路由** —— 用户点击按钮必然报错。
2. **后端注册了却完全无人可达的端点** —— 静默的"半成品功能"。

第 2 类不强制清零：有些端点先于 UI 落地。但必须在
``_KNOWN_UNWIRED`` 中显式登记并写明原因，新增未接线端点会直接失败，
迫使作者做出"接线 or 登记"的决定，而不是默默留下。
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from setup_tuner.app import create_app

_APP_JS = Path(__file__).resolve().parents[1] / "setup_tuner" / "ui" / "app.js"
_INDEX_HTML = Path(__file__).resolve().parents[1] / "setup_tuner" / "ui" / "index.html"

#: 后端已注册但前端暂无入口的端点 → 原因。
#: 每一条都是"有意识留下"的，不是遗漏。
_KNOWN_UNWIRED: dict[str, str] = {
    "/api/v1/health": "部署健康探测，供外部脚本/监控使用，不需要 UI 入口",
    "/api/v1/telemetry/simulate": "无真实 F1 游戏时的测试通道，供 e2e 测试与调试使用",
    "/api/v1/telemetry/experiment": "A/B 实验批量入口，属 CLI/脚本用途",
    "/api/v1/telemetry/import-file": "从磁盘导入遥测文件，属 CLI/脚本用途",
    "/api/v1/tracks/{track_id}": "单赛道详情：前端用 /tracks 全量列表即可满足",
    "/api/v1/ws": "WebSocket，前端经 WS_URL 常量连接，非 fetchJSON 路径",
}


def _backend_paths() -> set[str]:
    """收集后端实际注册的 API 路径。"""
    app = create_app()
    paths: set[str] = set()
    for route in app.routes:
        if type(route).__name__ == "_IncludedRouter":
            orig = getattr(route, "original_router", None)
            for sub in getattr(orig, "routes", []):
                p = getattr(sub, "path", None)
                if p and p.startswith("/"):
                    paths.add(p)
        else:
            p = getattr(route, "path", None)
            if p and p.startswith("/api"):
                paths.add(p)
    return paths


def _frontend_paths() -> set[str]:
    """抽取前端经 ``fetchJSON(API_BASE + path)`` 实际请求的路径。"""
    src = _APP_JS.read_text(encoding="utf-8")
    return {
        m.group(1)
        for m in re.finditer(r"""fetchJSON\(\s*[`'"]([^`'"]+)[`'"]""", src)
    }


def _normalize(path: str) -> str:
    """统一路径参数与查询串：/tracks/${id}?x=1 → /tracks/{p}。"""
    path = path.split("?")[0]
    path = re.sub(r"\$\{[^}]*\}", "{p}", path)
    path = re.sub(r"\{[^}]*\}", "{p}", path)
    return path.rstrip("/") or "/"


class TestFrontendBackendContract:
    """前端请求路径必须被后端满足。"""

    def test_frontend_calls_exist_in_backend(self) -> None:
        """前端每一个 fetchJSON 路径都必须有对应后端路由。"""
        backend = {_normalize(p) for p in _backend_paths()}
        missing = []
        for raw in _frontend_paths():
            full = _normalize("/api/v1" + raw)
            if full not in backend:
                missing.append(raw)
        assert not missing, (
            "前端调用了后端不存在的路由（点击即报错）："
            + ", ".join(sorted(missing))
        )

    def test_no_silently_unwired_endpoints(self) -> None:
        """后端新增端点必须接线，或在 _KNOWN_UNWIRED 中显式登记。"""
        backend = _backend_paths()
        frontend = {_normalize("/api/v1" + p) for p in _frontend_paths()}
        unwired = {
            p for p in backend if _normalize(p) not in frontend
        }
        unregistered = sorted(unwired - set(_KNOWN_UNWIRED))
        assert not unregistered, (
            "发现未接线且未登记的端点（请接线，或加入 _KNOWN_UNWIRED 并写明原因）："
            + ", ".join(unregistered)
        )

    def test_known_unwired_entries_still_exist(self) -> None:
        """_KNOWN_UNWIRED 不得残留已接线或已删除的端点（防止清单腐化）。"""
        backend = _backend_paths()
        stale = sorted(set(_KNOWN_UNWIRED) - backend)
        assert not stale, "以下端点已不存在，请从 _KNOWN_UNWIRED 移除：" + ", ".join(stale)


class TestUnwiredEndpointsAreReachable:
    """登记在册的"未接线"端点至少自身可用（不能是坏的）。"""

    def test_health_ok(self) -> None:
        app = create_app()
        with TestClient(app) as c:
            r = c.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["code"] == 0

    def test_recordings_list_empty_ok(self) -> None:
        app = create_app()
        with TestClient(app) as c:
            r = c.get("/api/v1/telemetry/recordings")
        assert r.status_code == 200
        assert r.json()["code"] == 0
        assert isinstance(r.json()["data"], list)

    def test_replay_status_ok(self) -> None:
        app = create_app()
        with TestClient(app) as c:
            r = c.get("/api/v1/telemetry/replay/status")
        assert r.status_code == 200
        assert r.json()["data"]["replaying"] is False

    def test_track_detail_ok(self) -> None:
        app = create_app()
        with TestClient(app) as c:
            r = c.get("/api/v1/tracks/suzuka")
        assert r.status_code == 200
        assert r.json()["data"]["track"]["track_id"] == "suzuka"

    def test_track_detail_unknown_returns_error(self) -> None:
        app = create_app()
        with TestClient(app) as c:
            r = c.get("/api/v1/tracks/__no_such_track__")
        assert r.status_code >= 400
        assert r.json()["code"] != 0


class TestIndexHtmlReferencesExistingAssets:
    """index.html 引用的静态资源必须存在（防止 404 死链）。"""

    def test_script_and_style_assets_exist(self) -> None:
        html = _INDEX_HTML.read_text(encoding="utf-8")
        ui_dir = _INDEX_HTML.parent
        refs = re.findall(r"""(?:src|href)\s*=\s*["'](/static/[^"']+)["']""", html)
        assert refs, "index.html 未引用任何 /static 资源，选择器可能已失效"
        missing = []
        for ref in refs:
            rel = ref[len("/static/"):].split("?")[0]
            if not (ui_dir / rel).exists():
                missing.append(ref)
        assert not missing, "index.html 引用了不存在的资源：" + ", ".join(missing)


class TestDomIdConsistency:
    """app.js 经 ``$("id")`` 取用的元素必须在 index.html 中存在。

    这类错配的表现是"按钮点了没反应"——运行期静默失败，
    静态检查能在提交前拦住。
    """

    def test_all_referenced_ids_exist(self) -> None:
        js = _APP_JS.read_text(encoding="utf-8")
        html = _INDEX_HTML.read_text(encoding="utf-8")
        ids_used = set(re.findall(r'\$\("([^"]+)"\)', js))
        ids_defined = set(re.findall(r'id="([^"]+)"', html))
        ids_dynamic = set(re.findall(r'id="([^"]+)"', js))  # JS 动态注入
        missing = sorted(ids_used - ids_defined - ids_dynamic)
        assert not missing, (
            "app.js 引用了 index.html 中不存在的元素 id（点击将无反应）："
            + ", ".join(missing)
        )
