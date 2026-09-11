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
                passed = False
                items.append(f"❌ data 目录未创建：{data_dir}")

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
    print("\n[1/5] 产物完整性验证 (integrity)...")
    r1 = test_integrity(args.exe, args.zip)
    report.add(r1)
    for item in r1.sub_items:
        print(f"    • {item}")
    if not r1.passed:
        print("    ❌ 完整性验证失败，跳过后续测试")
        report.print_report()
        return 1

    # ── 阶段 2：exe 启动冒烟 ──
    print("\n[2/5] exe 启动冒烟 (startup)...")
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
        print("\n[3/5] 端到端运行测试 (e2e)...")
        r3 = test_e2e(args.host, args.port)
        report.add(r3)
        for item in r3.sub_items:
            print(f"    • {item}")

        # ── 阶段 4：资源完整性 ──
        print("\n[4/5] 资源完整性验证 (resources)...")
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
    print("\n[5/5] 便携性验证 (portable)...")
    r5 = test_portable(args.zip, args.host, args.port, args.timeout)
    report.add(r5)
    for item in r5.sub_items:
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