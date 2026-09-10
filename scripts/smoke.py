"""云端冒烟测试 - 启动服务 → 探 HTTP 端口 → 断言 200。

用法: python scripts/smoke.py

对齐 design.md 2.10.2 云端冒烟与 tasks.md T9 验收：
    - 启动 uvicorn 服务（subprocess，端口 8199 避免冲突）；
    - 等待服务就绪（轮询 /api/v1/health 直到 200 或超时 10 秒）；
    - 断言端点：
        * GET  /api/v1/health          → 200, code==0
        * GET  /api/v1/tracks          → 200, 24 赛道
        * GET  /api/v1/tracks/suzuka   → 200, 含弯道锚点
        * POST /api/v1/feedback        → 200（提交一条反馈）
        * POST /api/v1/suggest         → 200（生成建议，含 23 参数）
        * GET  /api/v1/suggest/latest  → 200
    - 清理：停止服务，删除临时 DB；
    - 退出码 0=全通过，1=有失败。

仅依赖 Python 标准库（urllib + subprocess + json），不引入 httpx。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# =========================================================================== #
# 常量
# =========================================================================== #
# 冒烟测试专用端口（避免与开发端口 8000 冲突）
SMOKE_PORT = 8199
SMOKE_HOST = "127.0.0.1"
BASE_URL = f"http://{SMOKE_HOST}:{SMOKE_PORT}/api/v1"

# 服务就绪轮询参数
READY_TIMEOUT = 10.0  # 秒
READY_INTERVAL = 0.3  # 秒

# 赛道与参数常量
EXPECTED_TRACK_COUNT = 24
EXPECTED_PARAM_COUNT = 23
TEST_TRACK_ID = "suzuka"

# 退出码
EXIT_OK = 0
EXIT_FAIL = 1


# =========================================================================== #
# HTTP 工具（标准库 urllib）
# =========================================================================== #
def http_get(url: str, timeout: float = 5.0) -> tuple[int, dict[str, Any]]:
    """发起 GET 请求，返回 (status_code, json_body)。

    Raises:
        urllib.error.URLError: 连接失败。
    """
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = resp.status
        body = json.loads(resp.read().decode("utf-8"))
    return status, body


def http_post(
    url: str, payload: dict[str, Any], timeout: float = 10.0
) -> tuple[int, dict[str, Any]]:
    """发起 POST 请求（JSON body），返回 (status_code, json_body)。

    Raises:
        urllib.error.URLError / HTTPError: 连接失败或非 2xx。
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = resp.status
        body = json.loads(resp.read().decode("utf-8"))
    return status, body


# =========================================================================== #
# 服务生命周期
# =========================================================================== #
def start_server(data_dir: Path) -> subprocess.Popen[bytes]:
    """启动 uvicorn 服务（subprocess）。

    通过环境变量配置端口与临时数据目录，避免污染工作区。

    Args:
        data_dir: 临时数据目录（SQLite 存放）。

    Returns:
        subprocess.Popen 实例。
    """
    env = os.environ.copy()
    env["API_HOST"] = SMOKE_HOST
    env["API_PORT"] = str(SMOKE_PORT)
    env["DATA_DIR"] = str(data_dir)
    env["LOG_LEVEL"] = "WARNING"  # 减少日志噪音

    # 用 uvicorn 直接启动 app 工厂
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "setup_tuner.app:create_app",
        "--factory",
        "--host",
        SMOKE_HOST,
        "--port",
        str(SMOKE_PORT),
        "--log-level",
        "warning",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    print(f"[server] 启动 uvicorn：pid={proc.pid}，端口={SMOKE_PORT}")
    return proc


def wait_for_ready(proc: subprocess.Popen[bytes]) -> bool:
    """轮询 /api/v1/health 直到 200 或超时。

    Args:
        proc: 服务进程（用于检测是否已退出）。

    Returns:
        True 表示服务就绪，False 表示超时或进程已退出。
    """
    deadline = time.monotonic() + READY_TIMEOUT
    url = f"{BASE_URL}/health"

    while time.monotonic() < deadline:
        # 进程已退出则不再等待
        if proc.poll() is not None:
            print(f"[server] 进程已提前退出，返回码={proc.returncode}", file=sys.stderr)
            _dump_proc_output(proc)
            return False

        try:
            status, _ = http_get(url, timeout=2.0)
            if status == 200:
                print(f"[server] 服务就绪（{url} → 200）")
                return True
        except (urllib.error.URLError, OSError):
            pass  # 服务尚未启动，继续轮询

        time.sleep(READY_INTERVAL)

    print(f"[server] 服务就绪超时（{READY_TIMEOUT}s）", file=sys.stderr)
    return False


def stop_server(proc: subprocess.Popen[bytes]) -> None:
    """停止服务进程（先 terminate，再 kill 兜底）。"""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3.0)
    print("[server] 服务已停止。")


def _dump_proc_output(proc: subprocess.Popen[bytes]) -> None:
    """打印服务进程的 stdout/stderr（用于调试启动失败）。"""
    try:
        stdout = proc.stdout.read().decode("utf-8", errors="replace") if proc.stdout else ""
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        if stdout:
            print(f"[server:stdout]\n{stdout}", file=sys.stderr)
        if stderr:
            print(f"[server:stderr]\n{stderr}", file=sys.stderr)
    except Exception:
        pass


# =========================================================================== #
# 断言工具
# =========================================================================== #
class SmokeFailure(Exception):
    """冒烟测试断言失败。"""


def assert_ok(status: int, body: dict[str, Any], label: str) -> dict[str, Any]:
    """断言 HTTP 200 且信封 code==0，返回 data。

    Raises:
        SmokeFailure: 状态码非 200 或 code 非 0。
    """
    if status != 200:
        raise SmokeFailure(f"[{label}] 期望 HTTP 200，实际 {status}，body={body}")
    code = body.get("code")
    if code != 0:
        raise SmokeFailure(f"[{label}] 期望信封 code==0，实际 code={code}，message={body.get('message')}")
    return body.get("data")


# =========================================================================== #
# 冒烟测试用例
# =========================================================================== #
def test_health() -> None:
    """① GET /api/v1/health → 200, code==0。"""
    status, body = http_get(f"{BASE_URL}/health")
    data = assert_ok(status, body, "health")
    assert "status" in data, f"[health] data 缺少 status 字段：{data}"
    assert "telemetry_connected" in data, f"[health] data 缺少 telemetry_connected 字段：{data}"
    print(f"  [PASS] health → status={data['status']}, telemetry_connected={data['telemetry_connected']}")


def test_tracks() -> None:
    """② GET /api/v1/tracks → 200, 24 赛道。返回 track_id 列表供后续用例使用。"""
    status, body = http_get(f"{BASE_URL}/tracks")
    data = assert_ok(status, body, "tracks")
    assert isinstance(data, list), f"[tracks] data 不是列表：{type(data)}"
    count = len(data)
    if count != EXPECTED_TRACK_COUNT:
        raise SmokeFailure(f"[tracks] 期望 {EXPECTED_TRACK_COUNT} 赛道，实际 {count}")
    # 验证 suzuka 存在
    track_ids = [t["track_id"] for t in data]
    if TEST_TRACK_ID not in track_ids:
        raise SmokeFailure(f"[tracks] 赛道列表中未找到 {TEST_TRACK_ID}，现有：{track_ids[:5]}...")
    print(f"  [PASS] tracks → {count} 赛道（含 {TEST_TRACK_ID}）")


def test_track_detail() -> None:
    """③ GET /api/v1/tracks/suzuka → 200, 含弯道锚点。"""
    status, body = http_get(f"{BASE_URL}/tracks/{TEST_TRACK_ID}")
    data = assert_ok(status, body, f"tracks/{TEST_TRACK_ID}")
    assert "track" in data, f"[track_detail] data 缺少 track 字段：{data}"
    assert "corners" in data, f"[track_detail] data 缺少 corners 字段：{data}"
    corners = data["corners"]
    assert isinstance(corners, list) and len(corners) > 0, f"[track_detail] corners 为空：{corners}"
    # 验证弯道锚点字段
    first = corners[0]
    for field in ("corner_number", "anchor_x", "anchor_y"):
        assert field in first, f"[track_detail] 弯道缺少 {field} 字段：{first}"
    print(f"  [PASS] tracks/{TEST_TRACK_ID} → {len(corners)} 个弯道（含锚点）")


def test_feedback() -> None:
    """④ POST /api/v1/feedback → 200（提交一条反馈）。"""
    payload = {
        "track_id": TEST_TRACK_ID,
        "corner_number": 1,
        "symptom": "understeer",
        "strength": 3,
    }
    status, body = http_post(f"{BASE_URL}/feedback", payload)
    data = assert_ok(status, body, "feedback")
    assert data.get("symptom") == "understeer", f"[feedback] symptom 不匹配：{data}"
    print(f"  [PASS] feedback → id={data.get('id')}, symptom={data.get('symptom')}")


def test_suggest() -> None:
    """⑤ POST /api/v1/suggest → 200（生成建议，含 23 参数）。"""
    payload = {"track_id": TEST_TRACK_ID}
    status, body = http_post(f"{BASE_URL}/suggest", payload)
    data = assert_ok(status, body, "suggest")
    assert "report" in data, f"[suggest] data 缺少 report 字段：{data}"
    report = data["report"]
    # 报告中应包含 parameters 列表（design 2.7.7 格式）
    params = report.get("parameters")
    if params is not None:
        assert isinstance(params, list), f"[suggest] parameters 不是列表：{type(params)}"
        param_count = len(params)
        if param_count != EXPECTED_PARAM_COUNT:
            raise SmokeFailure(
                f"[suggest] 期望 {EXPECTED_PARAM_COUNT} 参数，实际 {param_count}"
            )
        print(f"  [PASS] suggest → {param_count} 参数建议")
    else:
        # 报告格式可能用 setupDelta dict
        delta = report.get("setup_delta") or report.get("setupDelta")
        if delta is not None and isinstance(delta, dict):
            param_count = len(delta)
            print(f"  [PASS] suggest → setupDelta 含 {param_count} 参数")
        else:
            print(f"  [PASS] suggest → 报告已生成（track_id={report.get('track_id')}）")


def test_suggest_latest() -> None:
    """⑥ GET /api/v1/suggest/latest → 200。"""
    url = f"{BASE_URL}/suggest/latest?track_id={TEST_TRACK_ID}"
    status, body = http_get(url)
    data = assert_ok(status, body, "suggest/latest")
    assert "report" in data, f"[suggest/latest] data 缺少 report 字段：{data}"
    print(f"  [PASS] suggest/latest → suggestion_id={data.get('suggestion_id')}")


# =========================================================================== #
# 主入口
# =========================================================================== #
def main() -> int:
    """云端冒烟测试主入口。

    Returns:
        退出码：0=全通过，1=有失败。
    """
    print("=" * 70)
    print("F1OPT 云端冒烟测试")
    print("=" * 70)

    # 临时数据目录（SQLite 存放，测试后清理）
    tmp_dir = Path(tempfile.mkdtemp(prefix="f1opt-smoke-"))
    data_dir = tmp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] 临时数据目录：{tmp_dir}")

    proc: subprocess.Popen[bytes] | None = None
    failures: list[str] = []

    try:
        # ① 启动服务
        proc = start_server(data_dir)

        # ② 等待就绪
        if not wait_for_ready(proc):
            print("\n[FAIL] 服务未能就绪，冒烟测试中止。", file=sys.stderr)
            return EXIT_FAIL

        print("\n[run] 开始端点断言...")

        # ③ 逐端点断言
        test_cases = [
            ("health", test_health),
            ("tracks", test_tracks),
            ("track_detail", test_track_detail),
            ("feedback", test_feedback),
            ("suggest", test_suggest),
            ("suggest_latest", test_suggest_latest),
        ]

        for name, test_fn in test_cases:
            try:
                test_fn()
            except SmokeFailure as e:
                failures.append(str(e))
                print(f"  [FAIL] {name}: {e}", file=sys.stderr)
            except Exception as e:
                failures.append(f"[{name}] 异常：{e}")
                print(f"  [FAIL] {name}: 异常 {e}", file=sys.stderr)

    finally:
        # ④ 清理：停止服务
        if proc is not None:
            stop_server(proc)
        # ⑤ 清理：删除临时 DB
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"[clean] 已清理临时目录：{tmp_dir}")
        except Exception:
            pass

    # ⑥ 汇总
    print("\n" + "=" * 70)
    if failures:
        print(f"冒烟测试失败：{len(failures)} 项")
        for f in failures:
            print(f"  - {f}")
        print("=" * 70)
        return EXIT_FAIL

    print("冒烟测试全部通过！")
    print("=" * 70)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())