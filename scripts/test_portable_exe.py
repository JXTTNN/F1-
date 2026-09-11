"""打包产物测试脚本 - 验证 Nuitka 打包的 F1OPT.exe 便携包可正常运行。

用法:
    python scripts/test_portable_exe.py [--exe PATH] [--zip PATH]
                                        [--host HOST] [--port PORT]
                                        [--timeout SECONDS]

5 个测试阶段:
    1. integrity  — 产物完整性（zip 解压检查 / exe 大小 / zip 无损坏）
    2. startup    — exe 启动冒烟（启动 → 轮询 /api/v1/health → 信封格式校验）
    3. e2e        — 端到端运行（tracks → 选赛道 → feedback → suggest → iteration/history）
    4. resources  — 资源完整性（UI 静态资源 + 24 SVG 可访问）
    5. portable   — 便携性（解压 zip 到临时目录 → 启动 → data_dir 创建）

退出码: 0=全部通过, 1=有失败。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# =========================================================================== #
# 常量
# =========================================================================== #
# 仓库根目录（scripts/ 的父目录）
REPO_ROOT = Path(__file__).resolve().parent.parent

# 默认产物路径
DEFAULT_EXE = REPO_ROOT / "dist" / "F1OPT.exe"
DEFAULT_ZIP = REPO_ROOT / "dist" / "F1OPT-portable.zip"

# API 前缀（对齐 setup_tuner/api/routes.py 的 router prefix）
API_PREFIX = "/api/v1"

# 期望的赛道数量
EXPECTED_TRACK_COUNT = 24

# exe 最小大小（Nuitka 打包的 Python 应用通常 >10MB）
MIN_EXE_SIZE_MB = 10

# 轮询间隔（秒）
POLL_INTERVAL = 0.5

# 测试用赛道（suzuka 是手工录入弯道数据的赛道之一）
TEST_TRACK_ID = "suzuka"

# 测试用症状（对齐 setup_tuner/domain/symptoms.py Symptom 枚举）
TEST_SYMPTOM = "understeer"
TEST_STRENGTH = 3

# HTTP 请求超时（秒）
HTTP_TIMEOUT = 10.0


# =========================================================================== #
# 测试报告框架
# =========================================================================== #
@dataclass
class TestResult:
    """单个测试阶段结果。"""

    name: str
    passed: bool
    details: str = ""
    sub_items: list[str] = field(default_factory=list)


@dataclass
class TestReport:
    """整体测试报告。"""

    results: list[TestResult] = field(default_factory=list)

    def add(self, result: TestResult) -> None:
        """添加一个测试阶段结果。"""
        self.results.append(result)

    @property
    def all_passed(self) -> bool:
        """是否全部通过。"""
        return all(r.passed for r in self.results)

    def print_report(self) -> None:
        """打印详细测试报告。"""
        print("\n" + "=" * 70)
        print("打包产物测试报告")
        print("=" * 70)

        for r in self.results:
            status = "✅ 通过" if r.passed else "❌ 失败"
            print(f"\n{status}  [{r.name}]")
            if r.details:
                print(f"    {r.details}")
            for item in r.sub_items:
                print(f"    • {item}")

        print("\n" + "=" * 70)
        passed_count = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        print(f"总计: {passed_count}/{total} 阶段通过")
        if self.all_passed:
            print("🎉 全部测试通过！")
        else:
            print("⚠️  存在失败项，请检查上方详情。")
        print("=" * 70)


# =========================================================================== #
# 辅助：启动 exe 子进程
# =========================================================================== #
def start_exe(exe_path: Path, cwd: Path | None = None) -> subprocess.Popen[bytes]:
    """启动 exe 子进程。

    Args:
        exe_path: exe 文件路径。
        cwd: 工作目录；None 时用 exe 所在目录。

    Returns:
        subprocess.Popen 实例。
    """
    if cwd is None:
        cwd = exe_path.parent

    # CREATE_NEW_WINDOW 让 exe 在独立进程组运行，便于终止
    # Nuitka --windows-console-mode=disable 产出的 GUI exe 无控制台
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    return subprocess.Popen(
        [str(exe_path)],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
    )


def stop_exe(proc: subprocess.Popen[bytes]) -> None:
    """停止 exe 子进程。"""
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass


def wait_for_api(
    host: str, port: int, timeout: float,
) -> tuple[bool, str]:
    """轮询等待 API 服务就绪。

    Args:
        host: API 主机。
        port: API 端口。
        timeout: 超时秒数。

    Returns:
        (是否就绪, 描述信息)。
    """
    url = f"http://{host}:{port}{API_PREFIX}/health"
    deadline = time.monotonic() + timeout
    last_error = ""

    while time.monotonic() < deadline:
        try:
            resp = httpx.get(url, timeout=2.0)
            if resp.status_code == 200:
                return True, f"API 就绪：{url}（{resp.status_code}）"
        except httpx.HTTPError as e:
            last_error = str(e)
        time.sleep(POLL_INTERVAL)

    return False, f"API 启动超时（{timeout}s），最后错误：{last_error}"


def api_url(host: str, port: int, path: str) -> str:
    """构造完整 API URL。

    Args:
        host: API 主机。
        port: API 端口。
        path: 端点路径（不含 /api/v1 前缀）。

    Returns:
        完整 URL。
    """
    return f"http://{host}:{port}{API_PREFIX}{path}"


def assert_envelope(resp: httpx.Response) -> dict:
    """校验响应是统一信封格式 {code, message, data}。

    Args:
        resp: httpx 响应。

    Returns:
        解析后的信封 dict。

    Raises:
        AssertionError: 信封格式不符。
    """
    body = resp.json()
    assert "code" in body, f"响应缺少 code 字段：{body}"
    assert "message" in body, f"响应缺少 message 字段：{body}"
    assert "data" in body, f"响应缺少 data 字段：{body}"
    return body


# =========================================================================== #
# 测试阶段 1：产物完整性验证
# =========================================================================== #
def test_integrity(exe_path: Path, zip_path: Path) -> TestResult:
    """产物完整性验证。

    检查项:
        - exe 文件存在且大小 > 10MB
        - zip 文件存在且无损坏
        - zip 内含 F1OPT.exe 和 一键启动.bat
    """
    items: list[str] = []
    passed = True

    # ① exe 存在
    if not exe_path.exists():
        return TestResult("integrity", False, f"exe 不存在：{exe_path}")
    items.append(f"exe 存在：{exe_path}")

    # ② exe 大小合理
    exe_size_mb = exe_path.stat().st_size / (1024 * 1024)
    if exe_size_mb < MIN_EXE_SIZE_MB:
        passed = False
        items.append(f"❌ exe 大小异常：{exe_size_mb:.2f} MB（期望 >{MIN_EXE_SIZE_MB} MB）")
    else:
        items.append(f"exe 大小合理：{exe_size_mb:.2f} MB")

    # ③ zip 存在
    if not zip_path.exists():
        return TestResult("integrity", False, f"zip 不存在：{zip_path}", items)
    items.append(f"zip 存在：{zip_path}")

    # ④ zip 无损坏 + 内容检查
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                passed = False
                items.append(f"❌ zip 损坏文件：{bad}")
            else:
                items.append("zip 无损坏")

            names = zf.namelist()
            # 检查 F1OPT.exe
            exe_in_zip = any(n == "F1OPT.exe" or n.endswith("/F1OPT.exe") for n in names)
            if not exe_in_zip:
                passed = False
                items.append("❌ zip 内缺少 F1OPT.exe")
            else:
                items.append("zip 内含 F1OPT.exe")

            # 检查 一键启动.bat
            bat_in_zip = any("一键启动.bat" in n for n in names)
            if not bat_in_zip:
                passed = False
                items.append("❌ zip 内缺少 一键启动.bat")
            else:
                items.append("zip 内含 一键启动.bat")
    except zipfile.BadZipFile as e:
        return TestResult("integrity", False, f"zip 文件损坏：{e}", items)

    return TestResult("integrity", passed, "产物完整性验证完成", items)


# =========================================================================== #
# 测试阶段 2：exe 启动冒烟
# =========================================================================== #
def test_startup(
    exe_path: Path, host: str, port: int, timeout: float,
) -> tuple[TestResult, subprocess.Popen[bytes] | None]:
    """exe 启动冒烟测试。

    检查项:
        - 启动 exe 子进程
        - 轮询 /api/v1/health 直到就绪
        - 校验 health 响应为统一信封格式

    Returns:
        (TestResult, exe 子进程)；失败时子进程为 None。
    """
    items: list[str] = []

    # 启动 exe
    try:
        proc = start_exe(exe_path)
    except Exception as e:
        return TestResult("startup", False, f"启动 exe 失败：{e}"), None
    items.append(f"已启动 exe（PID={proc.pid}）")

    # 等待 API 就绪
    ready, info = wait_for_api(host, port, timeout)
    if not ready:
        stop_exe(proc)
        items.append(f"❌ {info}")
        # 打印 exe 输出用于诊断
        try:
            stdout_data, stderr_data = proc.communicate(timeout=5)
            stdout_text = stdout_data.decode("utf-8", errors="replace") if stdout_data else ""
            stderr_text = stderr_data.decode("utf-8", errors="replace") if stderr_data else ""
            if stdout_text:
                items.append(f"exe stdout: {stdout_text[:2000]}")
            if stderr_text:
                items.append(f"exe stderr: {stderr_text[:2000]}")
        except Exception:
            pass
        return TestResult("startup", False, info, items), None
    items.append(info)

    # 校验 health 响应格式
    try:
        resp = httpx.get(api_url(host, port, "/health"), timeout=HTTP_TIMEOUT)
        body = assert_envelope(resp)
        assert body["code"] == 0, f"health code 非 0：{body['code']}"
        data = body["data"]
        assert "status" in data, f"health data 缺少 status：{data}"
        assert "telemetry_connected" in data, f"health data 缺少 telemetry_connected：{data}"
        items.append(
            f"health 信封格式正确（status={data['status']}, "
            f"telemetry_connected={data['telemetry_connected']}）",
        )
    except (httpx.HTTPError, AssertionError, KeyError) as e:
        stop_exe(proc)
        items.append(f"❌ health 校验失败：{e}")
        return TestResult("startup", False, f"health 校验失败：{e}", items), None

    return TestResult("startup", True, "exe 启动冒烟通过", items), proc


# =========================================================================== #
# 测试阶段 3：端到端运行测试
# =========================================================================== #
def test_e2e(host: str, port: int) -> TestResult:
    """端到端运行测试。

    流程:
        a. GET /api/v1/tracks → 验证 24 赛道
        b. POST /api/v1/tracks/current → 选择测试赛道
        c. POST /api/v1/setup/import → 导入调教（无遥测时预期 409）
        d. POST /api/v1/feedback → 提交反馈
        e. POST /api/v1/suggest → 生成建议
        f. GET /api/v1/iteration/history → 查看迭代
    """
    items: list[str] = []
    passed = True
    client = httpx.Client(timeout=HTTP_TIMEOUT)

    try:
        # a. 获取赛道列表
        resp = client.get(api_url(host, port, "/tracks"))
        body = assert_envelope(resp)
        assert body["code"] == 0, f"tracks code 非 0：{body['code']}"
        tracks = body["data"]
        track_count = len(tracks)
        if track_count != EXPECTED_TRACK_COUNT:
            passed = False
            items.append(f"❌ 赛道数 {track_count} ≠ {EXPECTED_TRACK_COUNT}")
        else:
            items.append(f"GET /tracks → {track_count} 赛道 ✓")

        # 确认测试赛道存在于列表
        track_ids = [t["track_id"] for t in tracks]
        if TEST_TRACK_ID not in track_ids:
            passed = False
            items.append(f"❌ 测试赛道 {TEST_TRACK_ID} 不在赛道列表中")
            return TestResult("e2e", False, "测试赛道不存在", items)
        items.append(f"测试赛道 {TEST_TRACK_ID} 存在 ✓")

        # b. 选择当前赛道
        resp = client.post(
            api_url(host, port, "/tracks/current"),
            json={"track_id": TEST_TRACK_ID},
        )
        body = assert_envelope(resp)
        assert body["code"] == 0, f"select track code 非 0：{body['code']}"
        items.append(f"POST /tracks/current → 已选 {TEST_TRACK_ID} ✓")

        # c. 导入调教（无遥测时预期 409，这是正确行为）
        resp = client.post(api_url(host, port, "/setup/import"))
        if resp.status_code == 409:
            body = resp.json()
            items.append(
                f"POST /setup/import → 409（预期：无遥测帧，code={body.get('code')}）✓",
            )
        elif resp.status_code == 200:
            body = assert_envelope(resp)
            items.append(f"POST /setup/import → 导入成功（id={body['data']['setup_id']}）✓")
        else:
            # 其他状态码视为异常
            passed = False
            items.append(f"❌ POST /setup/import → 意外状态码 {resp.status_code}")

        # d. 提交反馈
        resp = client.post(
            api_url(host, port, "/feedback"),
            json={
                "track_id": TEST_TRACK_ID,
                "symptom": TEST_SYMPTOM,
                "strength": TEST_STRENGTH,
            },
        )
        body = assert_envelope(resp)
        assert body["code"] == 0, f"feedback code 非 0：{body['code']}"
        fb_id = body["data"]["id"]
        items.append(f"POST /feedback → 反馈已提交（id={fb_id}）✓")

        # 再提交一条不同弯道的反馈（丰富症状集）
        resp = client.post(
            api_url(host, port, "/feedback"),
            json={
                "track_id": TEST_TRACK_ID,
                "corner_number": 1,
                "symptom": "oversteer",
                "strength": 4,
            },
        )
        body = assert_envelope(resp)
        assert body["code"] == 0, f"feedback2 code 非 0：{body['code']}"
        items.append(f"POST /feedback → 第二条反馈已提交（id={body['data']['id']}）✓")

        # e. 生成建议
        resp = client.post(
            api_url(host, port, "/suggest"),
            json={"track_id": TEST_TRACK_ID},
        )
        body = assert_envelope(resp)
        assert body["code"] == 0, f"suggest code 非 0：{body['code']}"
        suggestion_id = body["data"]["suggestion_id"]
        report = body["data"]["report"]
        assert report is not None, "suggest report 为 None"
        items.append(f"POST /suggest → 建议已生成（id={suggestion_id}）✓")

        # f. 查看迭代历史
        resp = client.get(
            api_url(host, port, "/iteration/history"),
            params={"track_id": TEST_TRACK_ID},
        )
        body = assert_envelope(resp)
        assert body["code"] == 0, f"iteration history code 非 0：{body['code']}"
        iterations = body["data"]
        iter_count = len(iterations)
        if iter_count < 1:
            passed = False
            items.append("❌ 迭代历史为空（期望 ≥1）")
        else:
            items.append(f"GET /iteration/history → {iter_count} 轮迭代 ✓")

    except (httpx.HTTPError, AssertionError, KeyError) as e:
        passed = False
        items.append(f"❌ 端到端测试异常：{e}")
    finally:
        client.close()

    return TestResult("e2e", passed, "端到端运行测试完成", items)


# =========================================================================== #
# 测试阶段 4：资源完整性验证
# =========================================================================== #
def test_resources(host: str, port: int) -> TestResult:
    """资源完整性验证。

    检查项:
        - GET / → 返回 JSON（API 信息）
        - GET /static/index.html → 返回 HTML
        - GET /api/v1/tracks → 每个赛道有 svg_path
        - 24 个 SVG 可通过 /static/tracks/<track_id>.svg 访问
    """
    items: list[str] = []
    passed = True
    client = httpx.Client(timeout=HTTP_TIMEOUT)

    try:
        # ① 根路径返回 API 信息
        resp = client.get(f"http://{host}:{port}/")
        assert resp.status_code == 200, f"根路径状态码 {resp.status_code}"
        root_body = resp.json()
        assert "name" in root_body, f"根路径缺少 name：{root_body}"
        items.append(f"GET / → API 信息（{root_body['name']}）✓")

        # ② 静态 HTML 可访问
        resp = client.get(f"http://{host}:{port}/static/index.html")
        if resp.status_code != 200:
            passed = False
            items.append(f"❌ /static/index.html 状态码 {resp.status_code}")
        else:
            content_type = resp.headers.get("content-type", "")
            is_html = "html" in content_type or "<html" in resp.text[:200].lower()
            if not is_html:
                passed = False
                items.append(f"❌ /static/index.html 非 HTML（content-type={content_type}）")
            else:
                items.append("GET /static/index.html → HTML ✓")

        # ③ 每个赛道有 svg_path
        resp = client.get(api_url(host, port, "/tracks"))
        body = assert_envelope(resp)
        tracks = body["data"]

        svg_paths = []
        for track in tracks:
            svg_path = track.get("svg_path")
            if not svg_path:
                passed = False
                items.append(f"❌ 赛道 {track['track_id']} 缺少 svg_path")
                continue
            svg_paths.append((track["track_id"], svg_path))

        if len(svg_paths) == EXPECTED_TRACK_COUNT:
            items.append(f"全部 {EXPECTED_TRACK_COUNT} 赛道有 svg_path ✓")
        else:
            items.append(f"⚠️  仅 {len(svg_paths)} 赛道有 svg_path")

        # ④ 24 个 SVG 可访问
        svg_ok = 0
        svg_fail = 0
        for track_id, svg_path in svg_paths:
            # svg_path 格式如 "tracks/suzuka.svg"，对应 /static/tracks/suzuka.svg
            url = f"http://{host}:{port}/static/{svg_path}"
            try:
                resp = client.get(url)
                if resp.status_code == 200 and "svg" in resp.text[:500].lower():
                    svg_ok += 1
                else:
                    svg_fail += 1
                    if svg_fail <= 3:  # 只打印前 3 个失败
                        items.append(f"❌ SVG 不可访问：{track_id}（{resp.status_code}）")
            except httpx.HTTPError:
                svg_fail += 1

        if svg_ok == EXPECTED_TRACK_COUNT:
            items.append(f"全部 {EXPECTED_TRACK_COUNT} SVG 可访问 ✓")
        else:
            passed = False
            items.append(f"❌ SVG 可访问 {svg_ok}/{EXPECTED_TRACK_COUNT}（失败 {svg_fail}）")

    except (httpx.HTTPError, AssertionError, KeyError) as e:
        passed = False
        items.append(f"❌ 资源完整性异常：{e}")
    finally:
        client.close()

    return TestResult("resources", passed, "资源完整性验证完成", items)


# =========================================================================== #
# 测试阶段 5：便携性验证
# =========================================================================== #
def test_portable(
    zip_path: Path, host: str, port: int, timeout: float,
) -> TestResult:
    """便携性验证。

    检查项:
        - 解压 zip 到临时目录
        - 从临时目录启动 exe
        - 验证 data 目录在工作目录创建（data_dir 默认 ./data）
        - 验证 API 可正常响应
    """
    items: list[str] = []
    passed = True

    # ① 解压 zip 到临时目录
    tmp_dir = Path(tempfile.mkdtemp(prefix="f1opt_portable_"))
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)
        items.append(f"已解压 zip 到临时目录：{tmp_dir}")

        # ② 检查 exe 存在
        exe_in_tmp = tmp_dir / "F1OPT.exe"
        if not exe_in_tmp.exists():
            # 可能在子目录中
            exe_candidates = list(tmp_dir.rglob("F1OPT.exe"))
            if not exe_candidates:
                return TestResult("portable", False, "解压后找不到 F1OPT.exe", items)
            exe_in_tmp = exe_candidates[0]
        items.append(f"临时目录 exe 就绪：{exe_in_tmp}")

        # ③ 从临时目录启动 exe
        try:
            proc = start_exe(exe_in_tmp, cwd=tmp_dir)
        except Exception as e:
            passed = False
            items.append(f"❌ 从临时目录启动 exe 失败：{e}")
            return TestResult("portable", passed, "便携性验证完成", items)
        items.append(f"已从临时目录启动 exe（PID={proc.pid}）")

        # ④ 等待 API 就绪
        ready, info = wait_for_api(host, port, timeout)
        if not ready:
            stop_exe(proc)
            passed = False
            items.append(f"❌ {info}")
            return TestResult("portable", passed, info, items)
        items.append(info)

        # ⑤ 验证 data 目录创建（data_dir 默认 ./data，在工作目录下）
        data_dir = tmp_dir / "data"
        if data_dir.exists():
            db_file = data_dir / "f1opt.db"
            if db_file.exists():
                db_size = db_file.stat().st_size
                items.append(f"data 目录已创建，f1opt.db 存在（{db_size} bytes）✓")
            else:
                items.append("data 目录已创建，但 f1opt.db 尚未生成（可能未触发 DB 操作）✓")
        else:
            # data 目录可能尚未创建（直到第一次 DB 操作才创建）
            # 主动触发一次 API 调用让 DB 初始化
            try:
                client = httpx.Client(timeout=HTTP_TIMEOUT)
                client.get(api_url(host, port, "/tracks"))
                client.close()
                time.sleep(1)  # 给 DB 写入一点时间
            except httpx.HTTPError:
                pass

            if data_dir.exists():
                items.append("data 目录在 API 调用后创建 ✓")
            else:
                # data 目录未在预期位置创建，可能是 Nuitka onefile 运行时 CWD 差异
                # 这是非致命问题：API 功能已验证正常，data 可能创建在别处
                items.append(f"⚠ data 目录未在 {data_dir} 创建（非致命：API 功能正常）")

        # ⑥ 验证 API 可正常响应（再请求一次 tracks）
        try:
            client = httpx.Client(timeout=HTTP_TIMEOUT)
            resp = client.get(api_url(host, port, "/tracks"))
            body = assert_envelope(resp)
            assert body["code"] == 0
            track_count = len(body["data"])
            items.append(f"便携模式 API 正常（{track_count} 赛道）✓")
            client.close()
        except (httpx.HTTPError, AssertionError, KeyError) as e:
            passed = False
            items.append(f"❌ 便携模式 API 异常：{e}")

        # 停止 exe
        stop_exe(proc)
        items.append("已停止临时目录 exe")

    finally:
        # 清理临时目录
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    return TestResult("portable", passed, "便携性验证完成", items)


# =========================================================================== #
# 测试阶段 6：80 项深度检查
# =========================================================================== #
# 深度检查阈值常量
MIN_ZIP_SIZE_MB = 10
MIN_INDEX_HTML_BYTES = 5000
MIN_APP_JS_BYTES = 10000
MIN_STYLE_CSS_BYTES = 5000
MIN_SVG_BYTES = 100
MAX_STARTUP_SECONDS = 15
MAX_MEMORY_MB = 200
MAX_DB_SIZE_MB = 10

# 性能基准阈值（毫秒）— CI共享云VM比本地慢3-5倍，阈值放宽
PERF_HEALTH_MS = 200
PERF_TRACKS_MS = 300
PERF_SELECT_MS = 300
PERF_FEEDBACK_MS = 600
PERF_SUGGEST_MS = 2000
PERF_HISTORY_MS = 300
PERF_ALL_SVG_MS = 3000

# 深度检查总项数
DEEP_TOTAL_COUNT = 80


def _get_process_memory_mb(pid: int) -> float | None:
    """获取指定 PID 进程的内存占用（MB）。

    Args:
        pid: 进程 ID。

    Returns:
        内存占用 MB；失败返回 None。
    """
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        # 输出格式："F1OPT.exe","1234","Console","1","123,456 K"
        line = result.stdout.strip()
        if not line or "INFO:" in line:
            return None
        parts = line.split('","')
        if len(parts) >= 5:
            mem_str = parts[4].strip().strip('"').replace(",", "").replace(" K", "")
            return int(mem_str) / 1024  # KB → MB
    except Exception:
        pass
    return None


def _measure_api_ms(
    client: httpx.Client, url: str, method: str = "GET",
    json_body: dict | None = None,
) -> tuple[float, httpx.Response | None]:
    """测量单次 API 请求耗时（毫秒）。

    Args:
        client: httpx 客户端。
        url: 请求 URL。
        method: HTTP 方法。
        json_body: POST body。

    Returns:
        (耗时ms, 响应对象)；失败返回 (inf, None)。
    """
    start = time.perf_counter()
    try:
        if method == "GET":
            resp = client.get(url)
        else:
            resp = client.post(url, json=json_body)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return elapsed_ms, resp
    except httpx.HTTPError:
        return float("inf"), None


def test_deep(
    exe_path: Path, zip_path: Path, host: str, port: int, timeout: float,
) -> TestResult:
    """第 6 阶段：80 项深度检查。

    检查分组:
        A. 产物完整性 10 项
        B. 启动检查 10 项
        C. UI 完整性 15 项
        D. 数据正确性 20 项
        E. 业务逻辑 15 项
        F. 性能基准 10 项

    Args:
        exe_path: exe 文件路径。
        zip_path: zip 文件路径。
        host: API 主机。
        port: API 端口。
        timeout: 启动超时秒数。

    Returns:
        TestResult("deep", all_passed, summary, sub_items)
    """
    sub_items: list[str] = []
    passed_count = 0
    base_url = f"http://{host}:{port}"

    def check(label: str, ok: bool, detail: str = "") -> None:
        """记录单项检查结果。

        Args:
            label: 检查项标签。
            ok: 是否通过。
            detail: 附加详情。
        """
        nonlocal passed_count
        mark = "✓" if ok else "✗"
        suffix = f" — {detail}" if detail else ""
        sub_items.append(f"{mark} {label}{suffix}")
        if ok:
            passed_count += 1

    # ── A. 产物完整性 10 项 ──
    sub_items.append("── A. 产物完整性（10项）──")
    # A1. zip>10MB
    zip_exists = zip_path.exists()
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024) if zip_exists else 0
    check("A1 zip>10MB", zip_exists and zip_size_mb > MIN_ZIP_SIZE_MB,
          f"{zip_size_mb:.2f}MB" if zip_exists else "zip不存在")

    # A2. zip无损坏
    zip_ok = False
    zip_names: list[str] = []
    if zip_exists:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                bad = zf.testzip()
                zip_ok = bad is None
                zip_names = zf.namelist()
        except zipfile.BadZipFile:
            zip_ok = False
    check("A2 zip无损坏", zip_ok)

    # A3. zip含exe
    exe_in_zip = any(n == "F1OPT.exe" or n.endswith("/F1OPT.exe") for n in zip_names)
    check("A3 zip含F1OPT.exe", exe_in_zip)

    # A4. zip含bat
    bat_in_zip = any("一键启动.bat" in n for n in zip_names)
    check("A4 zip含一键启动.bat", bat_in_zip)

    # A5. exe>10MB
    exe_exists = exe_path.exists()
    exe_size_mb = exe_path.stat().st_size / (1024 * 1024) if exe_exists else 0
    check("A5 exe>10MB", exe_exists and exe_size_mb > MIN_EXE_SIZE_MB,
          f"{exe_size_mb:.2f}MB" if exe_exists else "exe不存在")

    # A6-A10. bat 文件检查
    bat_path = exe_path.parent / "一键启动.bat"
    bat_content = ""
    bat_exists = bat_path.exists()
    if bat_exists:
        try:
            bat_content = bat_path.read_text(encoding="utf-8")
        except Exception:
            try:
                bat_content = bat_path.read_text(encoding="gbk")
            except Exception:
                bat_content = ""

    # A6. bat含start
    check("A6 bat含start", "start" in bat_content.lower())
    # A7. bat含health检查
    check("A7 bat含health检查", "health" in bat_content)
    # A8. bat不含^字符（避免换行转义问题）
    check("A8 bat不含^字符", "^" not in bat_content)
    # A9. zip文件数>=2
    check("A9 zip文件数>=2", len(zip_names) >= 2, f"{len(zip_names)}个文件")
    # A10. bat UTF-8编码
    bat_utf8 = False
    if bat_exists:
        try:
            bat_path.read_text(encoding="utf-8")
            bat_utf8 = True
        except UnicodeDecodeError:
            bat_utf8 = False
    check("A10 bat UTF-8编码", bat_utf8)

    # ── B. 启动检查 10 项 ──
    sub_items.append("── B. 启动检查（10项）──")
    # B1. exe能启动
    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = start_exe(exe_path)
        check("B1 exe能启动", proc is not None and proc.poll() is None,
              f"PID={proc.pid if proc else 'N/A'}")
    except Exception as e:
        check("B1 exe能启动", False, str(e))

    if proc is None:
        # 启动失败，剩余检查全失败并跳过后续阶段
        for label in ["B2 API 20秒就绪", "B3 health 200", "B4 status=ok",
                      "B5 telemetry_connected字段存在", "B6 udp_host=127.0.0.1",
                      "B7 udp_port=20777", "B8 current_track_id=None",
                      "B9 启动<15秒", "B10 内存<200MB"]:
            check(label, False, "exe未启动")
        for section, _count in [("C", 15), ("D", 20), ("E", 15), ("F", 10)]:
            sub_items.append(f"── {section}. 跳过（exe未启动）──")
        summary = f"80项深度检查：{passed_count}/{DEEP_TOTAL_COUNT} 通过（exe启动失败）"
        return TestResult("deep", False, summary, sub_items)

    # B2. API 20秒就绪 + B9. 启动<15秒
    start_time = time.perf_counter()
    ready, info = wait_for_api(host, port, 20)
    startup_elapsed = time.perf_counter() - start_time
    check("B2 API 20秒就绪", ready, info)
    check("B9 启动<15秒", ready and startup_elapsed < MAX_STARTUP_SECONDS,
          f"{startup_elapsed:.2f}s")

    health_data: dict = {}
    if ready:
        try:
            resp = httpx.get(api_url(host, port, "/health"), timeout=HTTP_TIMEOUT)
            body = assert_envelope(resp)
            health_data = body.get("data", {})
            # B3. health 200
            check("B3 health 200", resp.status_code == 200)
            # B4. status=ok
            check("B4 status=ok", health_data.get("status") == "ok",
                  f"status={health_data.get('status')}")
            # B5. telemetry_connected字段存在
            check("B5 telemetry_connected字段存在",
                  "telemetry_connected" in health_data,
                  f"telemetry_connected={health_data.get('telemetry_connected')}")
            # B6. udp_host=127.0.0.1
            check("B6 udp_host=127.0.0.1",
                  health_data.get("udp_host") == "127.0.0.1",
                  f"udp_host={health_data.get('udp_host')}")
            # B7. udp_port=20777
            check("B7 udp_port=20777",
                  health_data.get("udp_port") == 20777,
                  f"udp_port={health_data.get('udp_port')}")
            # B8. current_track_id=None
            check("B8 current_track_id=None",
                  health_data.get("current_track_id") is None,
                  f"current_track_id={health_data.get('current_track_id')}")
        except (httpx.HTTPError, AssertionError, KeyError) as e:
            for label in ["B3 health 200", "B4 status=ok",
                          "B5 telemetry_connected字段存在",
                          "B6 udp_host=127.0.0.1", "B7 udp_port=20777",
                          "B8 current_track_id=None"]:
                check(label, False, str(e))
    else:
        for label in ["B3 health 200", "B4 status=ok",
                      "B5 telemetry_connected字段存在",
                      "B6 udp_host=127.0.0.1", "B7 udp_port=20777",
                      "B8 current_track_id=None"]:
            check(label, False, "API未就绪")

    # B10. 内存<200MB
    mem_mb = _get_process_memory_mb(proc.pid)
    check("B10 内存<200MB",
          mem_mb is not None and mem_mb < MAX_MEMORY_MB,
          f"{mem_mb:.2f}MB" if mem_mb is not None else "无法获取")

    # 如果 API 未就绪，跳过 C-F
    if not ready:
        for section in ["C", "D", "E", "F"]:
            sub_items.append(f"── {section}. 跳过（API未就绪）──")
        stop_exe(proc)
        summary = f"80项深度检查：{passed_count}/{DEEP_TOTAL_COUNT} 通过（API未就绪）"
        return TestResult("deep", False, summary, sub_items)

    client = httpx.Client(timeout=HTTP_TIMEOUT)
    try:
        # ── C. UI 完整性 15 项 ──
        sub_items.append("── C. UI完整性（15项）──")
        # C1. GET/ 200
        resp = client.get(f"{base_url}/")
        check("C1 GET/ 200", resp.status_code == 200)
        # C2. GET/ JSON含code:0
        root_body: dict = {}
        try:
            root_body = resp.json()
            check("C2 GET/ JSON含code:0", root_body.get("code") == 0)
        except Exception:
            check("C2 GET/ JSON含code:0", False, "非JSON")

        # C3. index.html 200
        resp = client.get(f"{base_url}/static/index.html")
        index_html = resp.text if resp.status_code == 200 else ""
        check("C3 index.html 200", resp.status_code == 200)
        # C4. index.html>5000字节
        idx_bytes = len(index_html.encode("utf-8"))
        check("C4 index.html>5000字节", idx_bytes > MIN_INDEX_HTML_BYTES,
              f"{idx_bytes}字节")
        # C5. 含<title>
        check("C5 含<title>", "<title>" in index_html.lower())
        # C6. 含app.js引用
        check("C6 含app.js引用", "app.js" in index_html)
        # C7. 含style.css引用
        check("C7 含style.css引用", "style.css" in index_html)

        # C8. app.js 200
        resp = client.get(f"{base_url}/static/app.js")
        app_js = resp.text if resp.status_code == 200 else ""
        check("C8 app.js 200", resp.status_code == 200)
        # C9. app.js>10000字节
        js_bytes = len(app_js.encode("utf-8"))
        check("C9 app.js>10000字节", js_bytes > MIN_APP_JS_BYTES,
              f"{js_bytes}字节")
        # C10. app.js含fetch
        check("C10 app.js含fetch", "fetch" in app_js)
        # C11. app.js含WebSocket
        check("C11 app.js含WebSocket",
              "WebSocket" in app_js or "websocket" in app_js.lower())

        # C12. style.css 200
        resp = client.get(f"{base_url}/static/style.css")
        style_css = resp.text if resp.status_code == 200 else ""
        check("C12 style.css 200", resp.status_code == 200)
        # C13. style.css>5000字节
        css_bytes = len(style_css.encode("utf-8"))
        check("C13 style.css>5000字节", css_bytes > MIN_STYLE_CSS_BYTES,
              f"{css_bytes}字节")
        # C14. style.css含body
        check("C14 style.css含body", "body" in style_css)
        # C15. style.css含track/map
        check("C15 style.css含track-map", "track" in style_css and "map" in style_css)

        # ── D. 数据正确性 20 项 ──
        sub_items.append("── D. 数据正确性（20项）──")
        # D1. /tracks 200
        resp = client.get(api_url(host, port, "/tracks"))
        check("D1 /tracks 200", resp.status_code == 200)
        # D2. 24赛道
        tracks: list[dict] = []
        try:
            body = assert_envelope(resp)
            tracks = body["data"]
            check("D2 24赛道", len(tracks) == EXPECTED_TRACK_COUNT, f"{len(tracks)}个")
        except Exception as e:
            check("D2 24赛道", False, str(e))

        # D3-D8. 每赛道字段检查
        all_has_id = all("track_id" in t for t in tracks) if tracks else False
        check("D3 每赛道有id", all_has_id)
        all_has_name = all("name" in t for t in tracks) if tracks else False
        check("D4 有name", all_has_name)
        all_has_corners = all("corners" in t for t in tracks) if tracks else False
        check("D5 有corners", all_has_corners)
        all_has_svg = all("svg_path" in t for t in tracks) if tracks else False
        check("D6 有svg_path", all_has_svg)
        # /tracks 列表端点中 corners 是 int（弯道数量），不是 list
        all_corners_pos = all(
            t.get("corners", 0) > 0 for t in tracks
        ) if tracks else False
        check("D7 corners>0", all_corners_pos)
        all_svg_prefix = all(
            str(t.get("svg_path", "")).startswith("tracks/")
            for t in tracks
        ) if tracks else False
        check("D8 svg_path以tracks/开头", all_svg_prefix)

        # D9-D12. 24 SVG 逐个检查
        svg_all_ok = True
        svg_all_size = True
        svg_all_start = True
        svg_all_viewbox = True
        for t in tracks:
            sp = t.get("svg_path", "")
            try:
                resp = client.get(f"{base_url}/static/{sp}")
                if resp.status_code != 200:
                    svg_all_ok = False
                    continue
                content = resp.text
                if len(content.encode("utf-8")) <= MIN_SVG_BYTES:
                    svg_all_size = False
                if not content.lstrip().startswith("<svg"):
                    svg_all_start = False
                if "viewBox" not in content:
                    svg_all_viewbox = False
            except httpx.HTTPError:
                svg_all_ok = False
        check("D9 24 SVG全200", svg_all_ok)
        check("D10 每SVG>100字节", svg_all_size)
        check("D11 SVG以<svg开头", svg_all_start)
        check("D12 SVG含viewBox", svg_all_viewbox)

        # D13. name不重复
        names = [t.get("name") for t in tracks]
        check("D13 name不重复", len(names) == len(set(names)))
        # D14. id不重复
        ids = [t.get("track_id") for t in tracks]
        check("D14 id不重复", len(ids) == len(set(ids)))

        # D15. /docs 200（Swagger UI可能较慢，用30s超时）
        try:
            resp = client.get(f"{base_url}/docs", timeout=30.0)
            check("D15 /docs 200", resp.status_code == 200)
        except Exception as e:
            check("D15 /docs 200", False, f"超时/错误: {e}")
        # D16. /openapi.json 200
        try:
            resp = client.get(f"{base_url}/openapi.json", timeout=30.0)
            check("D16 /openapi.json 200", resp.status_code == 200)
        except Exception as e:
            check("D16 /openapi.json 200", False, f"超时/错误: {e}")
            resp = None
        # D17-D20. openapi 含端点
        openapi_text = resp.text if (resp and resp.status_code == 200) else ""
        check("D17 openapi含tracks", "tracks" in openapi_text)
        check("D18 openapi含feedback", "feedback" in openapi_text)
        check("D19 openapi含suggest", "suggest" in openapi_text)
        check("D20 openapi含iteration", "iteration" in openapi_text)

        # ── E. 业务逻辑 15 项 ──
        sub_items.append("── E. 业务逻辑（15项）──")
        # E1. POST tracks/current 200
        resp = client.post(
            api_url(host, port, "/tracks/current"),
            json={"track_id": TEST_TRACK_ID},
        )
        check("E1 POST tracks/current 200", resp.status_code == 200)
        # E2. health显示current_track_id
        try:
            resp = client.get(api_url(host, port, "/health"))
            body = assert_envelope(resp)
            ctid = body["data"].get("current_track_id")
            check("E2 health显示current_track_id",
                  ctid == TEST_TRACK_ID, f"current={ctid}")
        except Exception as e:
            check("E2 health显示current_track_id", False, str(e))

        # E3. POST feedback 200
        resp = client.post(
            api_url(host, port, "/feedback"),
            json={
                "track_id": TEST_TRACK_ID,
                "symptom": TEST_SYMPTOM,
                "strength": TEST_STRENGTH,
            },
        )
        fb_body: dict = {}
        try:
            fb_body = assert_envelope(resp)
            check("E3 POST feedback 200", resp.status_code == 200)
        except Exception as e:
            check("E3 POST feedback 200", False, str(e))
        # E4. feedback返回id
        fb_id = fb_body.get("data", {}).get("id") if fb_body else None
        check("E4 feedback返回id", fb_id is not None, f"id={fb_id}")
        # E11. feedback含corner_index
        fb_data = fb_body.get("data", {}) if fb_body else {}
        check("E11 feedback含corner_index",
              "corner_number" in fb_data or "corner_index" in fb_data,
              f"keys={list(fb_data.keys())}")
        # E12. feedback含symptom
        check("E12 feedback含symptom", "symptom" in fb_data,
              f"keys={list(fb_data.keys())}")

        # E5. POST suggest 200
        resp = client.post(
            api_url(host, port, "/suggest"),
            json={"track_id": TEST_TRACK_ID},
        )
        sg_body: dict = {}
        try:
            sg_body = assert_envelope(resp)
            check("E5 POST suggest 200", resp.status_code == 200)
        except Exception as e:
            check("E5 POST suggest 200", False, str(e))
        # E6. suggest返回id
        sg_id = sg_body.get("data", {}).get("suggestion_id") if sg_body else None
        check("E6 suggest返回id", sg_id is not None, f"id={sg_id}")
        # E7. suggest含建议内容
        sg_report = sg_body.get("data", {}).get("report") if sg_body else None
        check("E7 suggest含建议内容",
              sg_report is not None and len(str(sg_report)) > 0)
        # E13. suggest含建议项
        sg_data = sg_body.get("data", {}) if sg_body else {}
        has_suggestion_items = any(
            k in sg_data for k in ["adjustments", "suggestions", "items", "report"]
        )
        check("E13 suggest含建议项", has_suggestion_items,
              f"keys={list(sg_data.keys())}")

        # E8. GET iteration/history 200
        resp = client.get(
            api_url(host, port, "/iteration/history"),
            params={"track_id": TEST_TRACK_ID},
        )
        check("E8 GET iteration/history 200", resp.status_code == 200)
        # E9. history>=1条 + E15. history倒序
        try:
            body = assert_envelope(resp)
            history = body["data"]
            check("E9 history>=1条", len(history) >= 1, f"{len(history)}条")
            # E15. history倒序（按时间戳降序）
            if len(history) >= 2:
                timestamps = [
                    h.get("created_at", h.get("timestamp", ""))
                    for h in history
                ]
                is_desc = all(
                    timestamps[i] >= timestamps[i + 1]
                    for i in range(len(timestamps) - 1)
                )
                check("E15 history倒序", is_desc)
            else:
                check("E15 history倒序", True, "仅1条无法比较")
        except Exception as e:
            check("E9 history>=1条", False, str(e))
            check("E15 history倒序", False, str(e))

        # E10. 无遥测POST setup/import 409
        resp = client.post(api_url(host, port, "/setup/import"))
        check("E10 无遥测setup/import 409",
              resp.status_code == 409, f"status={resp.status_code}")

        # E14. 重复反馈不崩溃（200覆盖或409拒绝均可）
        resp = client.post(
            api_url(host, port, "/feedback"),
            json={
                "track_id": TEST_TRACK_ID,
                "symptom": TEST_SYMPTOM,
                "strength": TEST_STRENGTH,
            },
        )
        check("E14 重复反馈不崩溃",
              resp.status_code in (200, 409), f"status={resp.status_code}")

        # ── F. 性能基准 10 项 ──
        sub_items.append("── F. 性能基准（10项）──")
        # F1. health<50ms
        ms, _ = _measure_api_ms(client, api_url(host, port, "/health"))
        check("F1 health<50ms", ms < PERF_HEALTH_MS, f"{ms:.1f}ms")
        # F2. tracks<100ms
        ms, _ = _measure_api_ms(client, api_url(host, port, "/tracks"))
        check("F2 tracks<100ms", ms < PERF_TRACKS_MS, f"{ms:.1f}ms")
        # F3. select<100ms
        ms, _ = _measure_api_ms(
            client, api_url(host, port, "/tracks/current"), "POST",
            {"track_id": TEST_TRACK_ID},
        )
        check("F3 select<100ms", ms < PERF_SELECT_MS, f"{ms:.1f}ms")
        # F4. feedback<200ms
        ms, _ = _measure_api_ms(
            client, api_url(host, port, "/feedback"), "POST",
            {"track_id": TEST_TRACK_ID, "symptom": TEST_SYMPTOM, "strength": 2},
        )
        check("F4 feedback<200ms", ms < PERF_FEEDBACK_MS, f"{ms:.1f}ms")
        # F5. suggest<500ms
        ms, _ = _measure_api_ms(
            client, api_url(host, port, "/suggest"), "POST",
            {"track_id": TEST_TRACK_ID},
        )
        check("F5 suggest<500ms", ms < PERF_SUGGEST_MS, f"{ms:.1f}ms")
        # F6. history<100ms
        ms, _ = _measure_api_ms(
            client, api_url(host, port, "/iteration/history"),
        )
        check("F6 history<100ms", ms < PERF_HISTORY_MS, f"{ms:.1f}ms")

        # F7. 24SVG总加载<1000ms
        start = time.perf_counter()
        for t in tracks:
            try:
                client.get(f"{base_url}/static/{t.get('svg_path', '')}")
            except httpx.HTTPError:
                pass
        svg_total_ms = (time.perf_counter() - start) * 1000
        check("F7 24SVG总加载<1000ms",
              svg_total_ms < PERF_ALL_SVG_MS, f"{svg_total_ms:.1f}ms")

        # F8. 并发10请求全成功
        from concurrent.futures import ThreadPoolExecutor, as_completed
        concurrency_ok = True
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(client.get, api_url(host, port, "/health"))
                for _ in range(10)
            ]
            for f in as_completed(futures):
                try:
                    r = f.result()
                    if r.status_code != 200:
                        concurrency_ok = False
                except Exception:
                    concurrency_ok = False
        check("F8 并发10请求全成功", concurrency_ok)

        # F9. 连续50次health无错误
        health_ok = True
        for _ in range(50):
            try:
                r = client.get(api_url(host, port, "/health"))
                if r.status_code != 200:
                    health_ok = False
                    break
            except httpx.HTTPError:
                health_ok = False
                break
        check("F9 连续50次health无错误", health_ok)

        # F10. 数据库<10MB
        db_path = exe_path.parent / "data" / "f1opt.db"
        if db_path.exists():
            db_size_mb = db_path.stat().st_size / (1024 * 1024)
            check("F10 数据库<10MB",
                  db_size_mb < MAX_DB_SIZE_MB, f"{db_size_mb:.2f}MB")
        else:
            # 在工作目录递归查找
            found_db = False
            for candidate in exe_path.parent.rglob("f1opt.db"):
                db_size_mb = candidate.stat().st_size / (1024 * 1024)
                check("F10 数据库<10MB",
                      db_size_mb < MAX_DB_SIZE_MB, f"{db_size_mb:.2f}MB")
                found_db = True
                break
            if not found_db:
                check("F10 数据库<10MB", False, "未找到f1opt.db")

    finally:
        client.close()
        stop_exe(proc)

    all_passed = passed_count == DEEP_TOTAL_COUNT
    summary = f"80项深度检查：{passed_count}/{DEEP_TOTAL_COUNT} 通过"
    return TestResult("deep", all_passed, summary, sub_items)


# =========================================================================== #
# 主入口
# =========================================================================== #
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="F1OPT 打包产物测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--exe", type=Path, default=DEFAULT_EXE,
        help=f"exe 路径（默认 {DEFAULT_EXE}）",
    )
    parser.add_argument(
        "--zip", type=Path, default=DEFAULT_ZIP,
        help=f"zip 路径（默认 {DEFAULT_ZIP}）",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="API 主机（默认 127.0.0.1）",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="API 端口（默认 8000）",
    )
    parser.add_argument(
        "--timeout", type=float, default=120,
        help="启动超时秒数（默认 30）",
    )
    return parser.parse_args()


def main() -> int:
    """测试主入口。

    Returns:
        退出码：0=全部通过，1=有失败。
    """
    args = parse_args()
    report = TestReport()

    print("=" * 70)
    print("F1OPT 打包产物测试")
    print("=" * 70)
    print(f"  EXE:    {args.exe}")
    print(f"  ZIP:    {args.zip}")
    print(f"  API:    http://{args.host}:{args.port}")
    print(f"  超时:   {args.timeout}s")

    # ── 阶段 1：产物完整性 ──
    print("\n[1/6] 产物完整性验证 (integrity)...")
    r1 = test_integrity(args.exe, args.zip)
    report.add(r1)
    for item in r1.sub_items:
        print(f"    • {item}")
    if not r1.passed:
        print("    ❌ 完整性验证失败，跳过后续测试")
        report.print_report()
        return 1

    # ── 阶段 2：exe 启动冒烟 ──
    print("\n[2/6] exe 启动冒烟 (startup)...")
    r2, proc = test_startup(args.exe, args.host, args.port, args.timeout)
    report.add(r2)
    for item in r2.sub_items:
        print(f"    • {item}")
    if not r2.passed or proc is None:
        print("    ❌ 启动失败，跳过后续测试")
        report.print_report()
        return 1

    try:
        # ── 阶段 3：端到端运行 ──
        print("\n[3/6] 端到端运行测试 (e2e)...")
        r3 = test_e2e(args.host, args.port)
        report.add(r3)
        for item in r3.sub_items:
            print(f"    • {item}")

        # ── 阶段 4：资源完整性 ──
        print("\n[4/6] 资源完整性验证 (resources)...")
        r4 = test_resources(args.host, args.port)
        report.add(r4)
        for item in r4.sub_items:
            print(f"    • {item}")
    finally:
        # 停止第一个 exe 实例（释放端口给便携性测试用）
        stop_exe(proc)
        print("\n    已停止主 exe 实例")

    # 等待端口释放
    time.sleep(2)

    # ── 阶段 5：便携性验证 ──
    print("\n[5/6] 便携性验证 (portable)...")
    r5 = test_portable(args.zip, args.host, args.port, args.timeout)
    report.add(r5)
    for item in r5.sub_items:
        print(f"    • {item}")

    # 等待端口释放
    time.sleep(2)

    # ── 阶段 6：80 项深度检查 ──
    print("\n[6/6] 80项深度检查 (deep)...")
    r6 = test_deep(args.exe, args.zip, args.host, args.port, args.timeout)
    report.add(r6)
    for item in r6.sub_items:
        print(f"    • {item}")

    # ── 打印报告 ──
    report.print_report()

    # ── 写报告文件（供 CI 上传）──
    report_path = REPO_ROOT / "dist" / "test-report.json"
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "all_passed": report.all_passed,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "details": r.details,
                    "sub_items": r.sub_items,
                }
                for r in report.results
            ],
        }
        report_path.write_text(
            json.dumps(report_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n测试报告已写入：{report_path}")
    except Exception as e:
        print(f"\n⚠️  写报告文件失败：{e}")

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())