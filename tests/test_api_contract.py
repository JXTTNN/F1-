"""前后端契约一致性测试。

防止四类退化：

1. **前端调用了不存在的后端路由** —— 用户点击按钮必然报错。
2. **后端注册了却完全无人可达的端点** —— 静默的"半成品功能"。
3. **前端 POST 缺少后端声明的必填字段** —— 路径对、调用必 422。
4. **装饰器挂在错误的函数上** —— 端点名义存在，实际注册的是别的东西。

第 2 类不强制清零：有些端点先于 UI 落地。但必须在
``_KNOWN_UNWIRED`` 中显式登记并写明原因，新增未接线端点会直接失败，
迫使作者做出"接线 or 登记"的决定，而不是默默留下。

第 3、4 类来自两次真实事故（见 ``TestPostBodyContracts`` 与
``TestRouteHandlersAreRealEndpoints`` 的文档字符串）。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from setup_tuner.app import create_app

_ROOT = Path(__file__).resolve().parents[1]
_APP_JS = _ROOT / "setup_tuner" / "ui" / "app.js"
_INDEX_HTML = _ROOT / "setup_tuner" / "ui" / "index.html"

_API_PREFIX = "/api/v1"
_QUOTES = "\"'`"

#: 后端已注册但前端暂无入口的端点 → 原因。
#: 每一条都是"有意识留下"的，不是遗漏。
#: 注意：登记"未接线"**不等于**免除可用性 —— 见
#: ``TestUnwiredEndpointsAreReachable``，每个登记项都必须自身可调通。
_KNOWN_UNWIRED: dict[str, str] = {
    "/api/v1/health": "部署健康探测，供外部脚本/监控使用，不需要 UI 入口",
    "/api/v1/telemetry/simulate": "无真实 F1 游戏时的测试通道，供 e2e 测试与调试使用",
    "/api/v1/telemetry/experiment": "A/B 调教对比，供脚本/分析使用；已由本文件端到端覆盖",
    "/api/v1/telemetry/import-file": "从磁盘导入遥测文件，属 CLI/脚本用途",
    "/api/v1/tracks/{track_id}": "单赛道详情：前端用 /tracks 全量列表即可满足",
    "/api/v1/ws": "WebSocket，前端经 WS_URL 常量连接，非 fetchJSON 路径",
}


# --------------------------------------------------------------------------- #
# 路径收集
# --------------------------------------------------------------------------- #
def _api_routes() -> list[tuple[str, str, str]]:
    """列出后端实际注册的 API 路由 → ``(method, path, 处理函数名)``。

    WebSocket 路由没有 ``methods`` 属性，统一记为 ``WS``，
    以便 ``_backend_paths()`` 也能覆盖到它。
    """
    app = create_app()
    out: list[tuple[str, str, str]] = []

    def _collect(routes: Any) -> None:
        for r in routes:
            path = getattr(r, "path", None)
            endpoint = getattr(r, "endpoint", None)
            if not path or endpoint is None or not path.startswith("/api"):
                continue
            name = getattr(endpoint, "__name__", "?")
            methods = getattr(r, "methods", None)
            for method in methods or {"WS"}:
                out.append((method, path, name))

    for route in app.routes:
        if type(route).__name__ == "_IncludedRouter":
            orig = getattr(route, "original_router", None)
            _collect(getattr(orig, "routes", []))
        else:
            _collect([route])
    return out


def _backend_paths() -> set[str]:
    """收集后端实际注册的 API 路径。"""
    return {path for _, path, _ in _api_routes()}


# --------------------------------------------------------------------------- #
# app.js 静态解析（不执行 JS，只用括号配平扫描）
# --------------------------------------------------------------------------- #
def _match_close(src: str, open_idx: int) -> int:
    """从 ``open_idx``（指向 ``(``/``{``/``[``）返回配平闭合下标；失败 -1。"""
    depth = 0
    i = open_idx
    n = len(src)
    while i < n:
        c = src[i]
        if c in _QUOTES:
            quote = c
            i += 1
            while i < n:
                if src[i] == "\\":
                    i += 2
                    continue
                if src[i] == quote:
                    break
                i += 1
        elif c in "({[":
            depth += 1
        elif c in ")}]":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _js_source() -> str:
    """读取前端源码。"""
    return _APP_JS.read_text(encoding="utf-8")


def _fetch_calls() -> list[dict[str, Any]]:
    """解析 app.js 中所有 ``fetchJSON(path, opts)`` 调用。

    返回每项的 ``path``（首参字符串字面量）、``method``、``opts``（选项源码）、
    ``pos``（调用起始偏移，用于回溯局部变量定义）。
    """
    src = _js_source()
    calls: list[dict[str, Any]] = []
    for m in re.finditer(r"fetchJSON\s*\(", src):
        lp = m.end() - 1
        rp = _match_close(src, lp)
        if rp < 0:
            continue
        args = src[lp + 1 : rp]
        pm = re.match(r"\s*([\"'`])([^\"'`]*)\1", args)
        if not pm:
            continue  # 首参不是字面量（动态拼接），跳过
        opts = args[pm.end() :]
        method = "GET"
        mm = re.search(r"method:\s*([\"'])([A-Za-z]+)\1", opts)
        if mm:
            method = mm.group(2).upper()
        calls.append({
            "path": pm.group(2),
            "method": method,
            "opts": opts,
            "pos": m.start(),
        })
    return calls


def _frontend_paths() -> set[str]:
    """前端实际请求的路径集合。"""
    return {c["path"] for c in _fetch_calls()}


def _object_keys(inner: str) -> set[str]:
    """取对象字面量内部文本的顶层键名（跳过字符串/嵌套/spread）。"""
    keys: set[str] = set()
    depth = 0
    token_start = 0
    i = 0
    n = len(inner)
    while i <= n:
        c = inner[i] if i < n else ","
        if c in _QUOTES:
            quote = c
            i += 1
            while i < n:
                if inner[i] == "\\":
                    i += 2
                    continue
                if inner[i] == quote:
                    break
                i += 1
            i += 1
            continue
        if c in "({[":
            depth += 1
        elif c in ")}]":
            depth -= 1
        elif c == "," and depth == 0:
            seg = inner[token_start:i].strip()
            token_start = i + 1
            if seg and not seg.startswith("..."):
                km = re.match(r"([A-Za-z_$][\w$]*)\s*(?::|$)", seg)
                if km:
                    keys.add(km.group(1))
        i += 1
    return keys


def _resolve_body_keys(expr: str, src: str, call_pos: int) -> set[str]:
    """把 body 表达式解析为顶层键集合。

    支持对象字面量 ``{ a, b: 1 }`` 与标识符（回溯 ``const x = {...}``
    定义，并叠加随后的 ``x.k = ...`` 赋值）。
    """
    expr = expr.strip()
    keys: set[str] = set()

    ident = re.fullmatch(r"([A-Za-z_$][\w$]*)", expr)
    if ident:
        name = ident.group(1)
        pat = re.compile(r"\b(?:const|let|var)\s+" + re.escape(name) + r"\s*=\s*")
        best = None
        for dm in pat.finditer(src, 0, call_pos):
            best = dm
        if best is None:
            return set()
        brace = src.find("{", best.end())
        if brace < 0 or src[best.end() : brace].strip():
            return set()
        close = _match_close(src, brace)
        if close < 0:
            return set()
        keys |= _object_keys(src[brace + 1 : close])
        assign = (
            re.escape(name)
            + r"(?:\.([A-Za-z_$][\w$]*)|\[\s*([\"'])([^\"']+)\2\s*\])\s*=[^=]"
        )
        for am in re.finditer(assign, src[best.end() : call_pos]):
            keys.add(am.group(1) or am.group(3))
        return keys

    if expr.startswith("{"):
        close = _match_close(expr, 0)
        if close > 0:
            keys |= _object_keys(expr[1:close])
    return keys


def _extract_body_expr(opts: str) -> str:
    """从 fetch 选项源码里取出 body 表达式。"""
    m = re.search(r"body:\s*JSON\.stringify\s*\(", opts)
    if m:
        lp = m.end() - 1
        rp = _match_close(opts, lp)
        if rp > 0:
            return opts[lp + 1 : rp].strip()
    m2 = re.search(r"body:\s*([A-Za-z_$][\w$]*)", opts)
    return m2.group(1) if m2 else ""


def _body_schemas() -> dict[tuple[str, str], tuple[str, list[str]]]:
    """``(METHOD, path) -> (模型名, 必填字段)``，仅含带 JSON body 的端点。"""
    spec = create_app().openapi()
    schemas = spec.get("components", {}).get("schemas", {})
    out: dict[tuple[str, str], tuple[str, list[str]]] = {}
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            rb = op.get("requestBody")
            if not rb:
                continue
            sch = rb["content"].get("application/json", {}).get("schema", {})
            ref = sch.get("$ref", "")
            if ref:
                name = ref.rsplit("/", 1)[-1]
                required = list(schemas.get(name, {}).get("required", []))
            else:
                name, required = "(inline)", list(sch.get("required", []))
            out[(method.upper(), path)] = (name, required)
    return out


def _normalize(path: str) -> str:
    """统一路径参数与查询串：/tracks/${id}?x=1 → /tracks/{p}。"""
    path = path.split("?")[0]
    path = re.sub(r"\$\{[^}]*\}", "{p}", path)
    path = re.sub(r"\{[^}]*\}", "{p}", path)
    return path.rstrip("/") or "/"


def _isolated_app(tmp_path: Path):
    """使用隔离数据目录构建 app（测试不得读写仓库 ./data）。"""
    from setup_tuner.config import Config

    return create_app(Config(data_dir=str(tmp_path / "data")))


# --------------------------------------------------------------------------- #
# 1) 前端 → 后端：路径与参数
# --------------------------------------------------------------------------- #
class TestFrontendBackendContract:
    """前端请求必须被后端正确满足。"""

    def test_frontend_calls_exist_in_backend(self) -> None:
        """前端每一个 fetchJSON 路径都必须有对应后端路由。"""
        backend = {_normalize(p) for p in _backend_paths()}
        missing = []
        for raw in _frontend_paths():
            if _normalize(_API_PREFIX + raw) not in backend:
                missing.append(raw)
        assert not missing, (
            "前端调用了后端不存在的路由（点击即报错）：" + ", ".join(sorted(missing))
        )

    def test_no_silently_unwired_endpoints(self) -> None:
        """后端新增端点必须接线，或在 _KNOWN_UNWIRED 中显式登记。"""
        backend = _backend_paths()
        frontend = {_normalize(_API_PREFIX + p) for p in _frontend_paths()}
        unwired = {p for p in backend if _normalize(p) not in frontend}
        unregistered = sorted(unwired - set(_KNOWN_UNWIRED))
        assert not unregistered, (
            "发现未接线且未登记的端点（请接线，或加入 _KNOWN_UNWIRED 并写明原因）："
            + ", ".join(unregistered)
        )

    def test_known_unwired_entries_still_exist(self) -> None:
        """_KNOWN_UNWIRED 不得残留已接线或已删除的端点（防止清单腐化）。"""
        stale = sorted(set(_KNOWN_UNWIRED) - _backend_paths())
        assert not stale, "以下端点已不存在，请从 _KNOWN_UNWIRED 移除：" + ", ".join(stale)


class TestPostBodyContracts:
    """前端 POST 必须提供后端声明的必填 body 字段。

    历史事故：``/telemetry/record/toggle`` 的 ``action`` 为必填，前端却发
    ``{}`` —— 路径存在、契约测试全绿，但录制按钮点击**必然 422**，
    用户完全无法录制。这类"路径对、参数错"的失败必须由静态检查拦住。
    """

    def test_frontend_posts_satisfy_required_body_fields(self) -> None:
        """逐条比对前端 POST body 键与后端 OpenAPI required。"""
        schemas = _body_schemas()
        src = _js_source()
        problems: list[str] = []
        checked = 0

        for call in _fetch_calls():
            raw = call["path"]
            if not raw.startswith("/"):
                continue
            key = (call["method"], _API_PREFIX + raw.split("?")[0])
            if key not in schemas:
                continue
            _, required = schemas[key]
            if not required:
                continue
            checked += 1
            expr = _extract_body_expr(call["opts"])
            provided = _resolve_body_keys(expr, src, call["pos"]) if expr else set()
            missing = [r for r in required if r not in provided]
            if missing:
                problems.append(
                    f"{key[1]} 缺 {missing}（body={expr or '无'}）"
                )

        assert checked, "未匹配到任何带必填字段的前端 POST 调用，解析器可能已失效"
        assert not problems, (
            "前端 POST 缺少后端必填字段（点击必然 422）：" + "; ".join(problems)
        )

    def test_all_backend_post_bodies_are_covered(self) -> None:
        """带必填字段的后端 POST 端点，要么被前端调用，要么在名册中。

        防止"新增端点忘了接线"再次静默发生。
        """
        schemas = _body_schemas()
        required_posts = {
            path for (method, path), (_, req) in schemas.items() if method == "POST" and req
        }
        called = {
            _API_PREFIX + c["path"].split("?")[0]
            for c in _fetch_calls()
            if c["method"] == "POST" and c["path"].startswith("/")
        }
        uncovered = sorted(required_posts - called - set(_KNOWN_UNWIRED))
        assert not uncovered, "以下必需 body 的 POST 端点既无前端调用也未登记：" + ", ".join(
            uncovered
        )


# --------------------------------------------------------------------------- #
# 2) 路由注册形态
# --------------------------------------------------------------------------- #
class TestRouteHandlersAreRealEndpoints:
    """防止"装饰器挂在错误的函数上"。

    历史事故：``@router.post("/telemetry/experiment")`` 误挂在内部辅助函数
    ``_generate_experiment_suggestion`` 上，导致 FastAPI 把该辅助函数的形参
    当成契约（``setup``/``telemetry`` 视为 body、``track_id`` 视为 query），
    真正的端点 ``telemetry_experiment`` 沦为**死代码** —— 端点名义存在、
    路径检查全绿，但按文档契约调用必定 422。

    判据：注册到路由的处理函数不应是私有函数（下划线开头），
    也不应产生 ``Body__<func>_...`` 这类由"裸形参"推断出的内联模型。
    """

    def test_no_private_function_is_registered_as_route(self) -> None:
        """路由处理函数不得是私有函数。"""
        offenders = sorted(
            {(method, path, name) for method, path, name in _api_routes() if name.startswith("_")}
        )
        assert not offenders, (
            "以下路由由私有函数（下划线开头）处理，装饰器很可能挂错了函数："
            + "; ".join(f"{m} {p} -> {n}" for m, p, n in offenders)
        )

    def test_no_inline_body_schema_leak(self) -> None:
        """不应存在 ``Body__`` 内联模型（裸形参被误当契约的信号）。"""
        leaked = sorted(
            name for name, _ in _body_schemas().values() if name.startswith("Body__")
        )
        assert not leaked, (
            "检测到由裸形参推断出的内联请求体模型，说明某个端点的签名不是 "
            "强类型 Request 模型（装饰器可能挂错函数）：" + ", ".join(leaked)
        )

    def test_key_endpoints_use_their_declared_models(self) -> None:
        """关键端点的请求体必须是各自声明的模型。"""
        schemas = _body_schemas()
        expected = {
            "/api/v1/telemetry/experiment": "TelemetryExperimentRequest",
            "/api/v1/feedback": "FeedbackRequest",
            "/api/v1/suggest": "SuggestRequest",
            "/api/v1/telemetry/record/toggle": "RecordToggleRequest",
            "/api/v1/telemetry/listener/toggle": "ListenerToggleRequest",
            "/api/v1/telemetry/replay/start": "ReplayStartRequest",
            "/api/v1/setup/manual": "ManualSetupRequest",
            "/api/v1/tracks/current": "SelectTrackRequest",
        }
        mismatched = [
            f"{path}: 期望 {model}，实际 {schemas[(m, path)][0]}"
            for path, model in expected.items()
            for m in ("POST",)
            if (m, path) in schemas and schemas[(m, path)][0] != model
        ]
        missing = [
            path for path in expected if ("POST", path) not in schemas
        ]
        assert not missing, "以下端点的 POST 请求体缺失：" + ", ".join(missing)
        assert not mismatched, "; ".join(mismatched)


# --------------------------------------------------------------------------- #
# 3) 登记在册的"未接线"端点必须自身可用
# --------------------------------------------------------------------------- #
class TestUnwiredEndpointsAreReachable:
    """登记在册的"未接线"端点至少自身可用（不能是坏的）。"""

    def test_health_ok(self) -> None:
        app = create_app()
        with TestClient(app) as c:
            r = c.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["code"] == 0

    def test_recordings_list_empty_ok(self, tmp_path: Path) -> None:
        """空数据目录下录制列表返回空数组。

        必须隔离数据目录：默认 ``./data`` 会被本机录制内容填充，
        断言"空"在这里就会变成依赖环境的状态（本地绿 / CI 红）。
        """
        app = _isolated_app(tmp_path)
        with TestClient(app) as c:
            r = c.get("/api/v1/telemetry/recordings")
        assert r.status_code == 200
        assert r.json()["code"] == 0
        assert isinstance(r.json()["data"], list)
        assert r.json()["data"] == []

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


class TestExperimentEndpointWorks:
    """``POST /api/v1/telemetry/experiment`` 端到端契约。

    该端点曾因装饰器错位而不可达（见 ``TestRouteHandlersAreRealEndpoints``），
    因此必须按**文档契约**真实呼叫一次，而不是只检查路径存在。
    """

    _PATH = "/api/v1/telemetry/experiment"

    @staticmethod
    def _setup(**overrides: float) -> dict[str, float]:
        from setup_tuner.domain.setup import CarSetup

        setup = CarSetup().to_dict()
        setup.update(overrides)
        return setup

    def test_documented_contract_is_accepted(self) -> None:
        """按文档提供的 body 必须返回 200 + 完整对比视图。"""
        app = create_app()
        with TestClient(app) as c:
            r = c.post(
                self._PATH,
                json={
                    "track_id": "suzuka",
                    "setup_a": self._setup(front_wing=1.0),
                    "setup_b": self._setup(front_wing=11.0),
                    "telemetry": {"avg_speed": 210.0},
                },
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["code"] == 0
        assert set(body["data"]) == {
            "track_id",
            "suggestion_a",
            "suggestion_b",
            "dx_diff",
            "setup_delta_diff",
            "confidence_diff",
        }
        assert body["data"]["track_id"] == "suzuka"

    def test_telemetry_is_optional(self) -> None:
        """省略 telemetry 仍应成功（无遥测的纯规则路径）。"""
        app = create_app()
        with TestClient(app) as c:
            r = c.post(
                self._PATH,
                json={
                    "track_id": "suzuka",
                    "setup_a": self._setup(),
                    "setup_b": self._setup(),
                },
            )
        assert r.status_code == 200, r.text
        assert r.json()["code"] == 0

    def test_dx_diff_equals_a_minus_b(self) -> None:
        """dx_diff 必须恒等于 dx_A − dx_B（A 为正表示 A 需求更强）。"""
        app = create_app()
        with TestClient(app) as c:
            r = c.post(
                self._PATH,
                json={
                    "track_id": "suzuka",
                    "setup_a": self._setup(),
                    "setup_b": self._setup(),
                    "telemetry": {"avg_speed": 200.0, "avg_throttle": 0.8},
                },
            )
        data = r.json()["data"]
        for dim, diff in data["dx_diff"].items():
            expected = data["suggestion_a"]["dx"].get(dim, 0.0) - data["suggestion_b"][
                "dx"
            ].get(dim, 0.0)
            assert abs(diff - expected) < 1e-9, f"{dim} 的 dx_diff 与两侧 Dx 不自洽"

    def test_split_telemetry_makes_dx_diff_informative(self) -> None:
        """提供分侧遥测后 dx_diff 必须能反映方案差异（不再恒为空）。

        600°C 刹车温度：软胎（severe=550）触发降功率 + 更强制动稳定需求，
        硬胎（severe=660）只触发基础告警 —— 两侧 Dx 应出现差异。
        """
        app = create_app()
        temps = [600, 600, 600, 600]
        with TestClient(app) as c:
            r = c.post(
                self._PATH,
                json={
                    "track_id": "suzuka",
                    "setup_a": self._setup(),
                    "setup_b": self._setup(),
                    "telemetry_a": {"m_brakesTemperature": temps, "is_soft_compound": True},
                    "telemetry_b": {"m_brakesTemperature": temps, "is_hard_compound": True},
                },
            )
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        assert data["suggestion_a"]["dx"]["brake_stab_req"] > data["suggestion_b"]["dx"]["brake_stab_req"]
        assert data["dx_diff"]["brake_stab_req"] > 0
        assert len([v for v in data["dx_diff"].values() if v]) >= 1

    def test_identical_setups_give_zero_delta_diff(self) -> None:
        """两份相同调教 → SetupDelta 差异必须全零。"""
        app = create_app()
        same = self._setup()
        with TestClient(app) as c:
            r = c.post(
                self._PATH,
                json={
                    "track_id": "suzuka",
                    "setup_a": same,
                    "setup_b": same,
                    "telemetry": {"avg_speed": 200.0},
                },
            )
        diff = r.json()["data"]["setup_delta_diff"]
        assert not [k for k, v in diff.items() if v], f"相同调教不应有差异：{diff}"

    def test_unknown_track_returns_404(self) -> None:
        app = create_app()
        with TestClient(app) as c:
            r = c.post(
                self._PATH,
                json={
                    "track_id": "__no_such_track__",
                    "setup_a": self._setup(),
                    "setup_b": self._setup(),
                },
            )
        assert r.status_code == 404
        assert r.json()["code"] == 4040

    def test_missing_setup_b_returns_422(self) -> None:
        """缺必填字段必须 422（而非静默成功）。"""
        app = create_app()
        with TestClient(app) as c:
            r = c.post(
                self._PATH,
                json={"track_id": "suzuka", "setup_a": self._setup()},
            )
        assert r.status_code == 422
        assert r.json()["code"] == 4000

    def test_incomplete_setup_returns_400(self) -> None:
        """调教参数不全必须被拒（不能拿残缺参数去生成建议）。"""
        app = create_app()
        with TestClient(app) as c:
            r = c.post(
                self._PATH,
                json={
                    "track_id": "suzuka",
                    "setup_a": {"front_wing": 5.0},
                    "setup_b": self._setup(),
                },
            )
        assert r.status_code == 400
        assert r.json()["code"] == 4005

    def test_out_of_range_param_returns_400(self) -> None:
        app = create_app()
        with TestClient(app) as c:
            r = c.post(
                self._PATH,
                json={
                    "track_id": "suzuka",
                    "setup_a": self._setup(front_wing=999.0),
                    "setup_b": self._setup(),
                },
            )
        assert r.status_code == 400
        assert r.json()["code"] == 4007


# --------------------------------------------------------------------------- #
# 4) 静态资源与 DOM 一致性
# --------------------------------------------------------------------------- #
class TestIndexHtmlReferencesExistingAssets:
    """index.html 引用的静态资源必须存在（防止 404 死链）。"""

    def test_script_and_style_assets_exist(self) -> None:
        html = _INDEX_HTML.read_text(encoding="utf-8")
        ui_dir = _INDEX_HTML.parent
        refs = re.findall(r"""(?:src|href)\s*=\s*["'](/static/[^"']+)["']""", html)
        assert refs, "index.html 未引用任何 /static 资源，选择器可能已失效"
        missing = []
        for ref in refs:
            rel = ref[len("/static/") :].split("?")[0]
            if not (ui_dir / rel).exists():
                missing.append(ref)
        assert not missing, "index.html 引用了不存在的资源：" + ", ".join(missing)


class TestDomIdConsistency:
    """app.js 经 ``$("id")`` 取用的元素必须在 index.html 中存在。

    这类错配的表现是"按钮点了没反应"——运行期静默失败，
    静态检查能在提交前拦住。
    """

    def test_all_referenced_ids_exist(self) -> None:
        js = _js_source()
        html = _INDEX_HTML.read_text(encoding="utf-8")
        ids_used = set(re.findall(r'\$\("([^"]+)"\)', js))
        ids_defined = set(re.findall(r'id="([^"]+)"', html))
        ids_dynamic = set(re.findall(r'id="([^"]+)"', js))  # JS 动态注入
        missing = sorted(ids_used - ids_defined - ids_dynamic)
        assert not missing, (
            "app.js 引用了 index.html 中不存在的元素 id（点击将无反应）："
            + ", ".join(missing)
        )


# --------------------------------------------------------------------------- #
# 录制训练样本导出（系统内导出，替代独立桌面接收器）
# --------------------------------------------------------------------------- #
def _hdr(pid: int) -> bytes:
    import struct

    from setup_tuner.telemetry.packets import HEADER_FORMAT

    return struct.pack(
        HEADER_FORMAT, 2026, 26, 1, 0, 1, pid,
        0x1234_5678_9ABC_DEF0, 1.0, 1, 1, 0, 255,
    )


def _session_pkt(track_id: int = 3) -> bytes:
    import struct

    return (
        _hdr(1)
        + struct.pack(
            "<BbbBHBbBHHBBBBBB",
            0, 25, 22, 58, 5807, 0, track_id, 1, 1800, 3600, 60, 0, 0, 0, 0, 21,
        )
        + b"".join(struct.pack("<fb", 0.0, 0) for _ in range(21))
        + struct.pack("<BBB", 0, 0, 0)
    )


def _lap_pkt(lap_num: int, last_ms: int = 0, invalid: int = 0) -> bytes:
    import struct

    from setup_tuner.telemetry.packets import _LAP_PER_STRUCT

    body = bytearray(_LAP_PER_STRUCT.size)
    struct.pack_into("<I", body, 0, last_ms)
    body[33] = lap_num
    body[37] = invalid
    return _hdr(2) + bytes(body)


def _setup_pkt(brake_bias: int = 55) -> bytes:
    from setup_tuner.telemetry.packets import _SETUP_PER_STRUCT

    body = bytearray(_SETUP_PER_STRUCT.size)
    body[27] = brake_bias
    return _hdr(5) + bytes(body)


def _tele_pkt(speed: int = 180) -> bytes:
    import struct

    body = bytearray(59)
    struct.pack_into("<H", body, 0, speed)
    return _hdr(6) + bytes(body)


def _make_recording(rec_dir: Path) -> str:
    """在 rec_dir 下录一段两圈的合成遥测，返回 session_id。"""
    from setup_tuner.telemetry.recorder import TelemetryRecorder

    recorder = TelemetryRecorder(str(rec_dir))
    recorder.start()
    raws = [
        _session_pkt(track_id=3),
        _setup_pkt(brake_bias=55),
        _lap_pkt(1),
        _tele_pkt(180), _tele_pkt(250),
        _lap_pkt(1),                     # 同圈：不固化
        _lap_pkt(2, last_ms=91_234),     # 圈号变化 → 固化圈 1
        _tele_pkt(200),
        _lap_pkt(3, last_ms=88_500),     # 固化圈 2
    ]
    for raw in raws:
        recorder.on_raw_packet(raw, None)
    return recorder.stop()["session_id"]


class TestRecordingTrainingExport:
    """``POST /telemetry/recordings/{id}/export`` —— 系统内导出逐圈训练样本。"""

    def test_export_produces_samples(self, tmp_path: Path) -> None:
        import json

        data_dir = tmp_path / "data"
        session_id = _make_recording(data_dir / "recordings")

        app = _isolated_app(tmp_path)  # 同一 data_dir 布局，且隔离环境
        with TestClient(app) as c:
            r = c.post(f"/api/v1/telemetry/recordings/{session_id}/export")

        assert r.status_code == 200, r.text
        d = r.json()["data"]
        assert d["laps"] == 2, d
        assert d["packets"] == 9
        assert d["parse_errors"] == 0

        out = Path(d["out_path"])
        assert out.exists()
        rows = [json.loads(line) for line in out.read_text("utf-8").splitlines()]
        assert len(rows) == 2

        lap1 = rows[0]
        assert lap1["track_id"] == 3
        assert lap1["lap_number"] == 1
        assert lap1["lap_time_ms"] == 91_234
        assert lap1["setup"]["m_brakeBias"] == 55
        assert isinstance(lap1["style"], list) and lap1["style"]

    def test_export_unknown_session_404(self, tmp_path: Path) -> None:
        app = _isolated_app(tmp_path)
        with TestClient(app) as c:
            r = c.post("/api/v1/telemetry/recordings/__nope__/export")
        assert r.status_code == 404
        assert r.json()["code"] == 4044

    def test_export_rejects_path_traversal(self, tmp_path: Path) -> None:
        """负向：含 ``..`` 的会话 ID 必须 400，不能拼出目录外的路径。"""
        app = _isolated_app(tmp_path)
        with TestClient(app) as c:
            r = c.post("/api/v1/telemetry/recordings/..foo/export")
        assert r.status_code == 400
        assert r.json()["code"] == 4004
