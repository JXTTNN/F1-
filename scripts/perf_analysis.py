"""F1OPT 性能分析脚本 —— 测量 API 响应时间 / 规则引擎耗时 / SQLite 查询耗时 / 内存占用。

用法::

    python scripts/perf_analysis.py --label baseline --out scripts/perf_baseline.json
    python scripts/perf_analysis.py --label optimized --out scripts/perf_optimized.json

工作流程：
    1. 用临时 data_dir 启动 uvicorn 子进程（避免污染生产数据）；
    2. 等待 /api/v1/health 就绪；
    3. 预置：选赛道 suzuka → 提交 1 条 feedback（为 suggest 端点提供前置）；
    4. 对 6 个端点各测 N 次响应时间，计算 mean / P50 / P95 / min / max；
    5. 直接调用规则引擎纯函数，测量 compute_dx / 矩阵乘法 / compute_setup_delta /
       generate_suggestion 各 M 次耗时；
    6. 直接调用 Store 各查询方法，测量 SQLite 查询耗时；
    7. 采样当前进程 RSS 内存；
    8. 输出 JSON 报告到指定路径。

仅依赖标准库 + httpx（dev 依赖），不引入新依赖。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

# 确保可导入 setup_tuner
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# =========================================================================== #
# 统计工具
# =========================================================================== #
def _stats(samples: list[float]) -> dict[str, float]:
    """计算样本统计量（单位：毫秒）。"""
    if not samples:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0,
                "min_ms": 0.0, "max_ms": 0.0}
    sorted_s = sorted(samples)
    n = len(sorted_s)
    # P95 插值
    p95_idx = max(0, min(n - 1, int(round(0.95 * (n - 1)))))
    return {
        "count": n,
        "mean_ms": round(statistics.fmean(samples), 4),
        "p50_ms": round(sorted_s[n // 2], 4),
        "p95_ms": round(sorted_s[p95_idx], 4),
        "min_ms": round(sorted_s[0], 4),
        "max_ms": round(sorted_s[-1], 4),
    }


def _now_ms() -> float:
    """当前高精度时间戳（毫秒）。"""
    return time.perf_counter() * 1000.0


# =========================================================================== #
# 内存采样
# =========================================================================== #
def _rss_kb() -> int:
    """返回当前进程 RSS（KB）。跨平台兼容。"""
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except ImportError:
        # Windows：用 psutil? 不能引入新依赖。退而求其次用标准库。
        # 读取 /proc 不适用于 Windows；用 tasklist 采样当前进程。
        try:
            out = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {os.getpid()}",
                 "/FO", "CSV", "/NH"],
                text=True, stderr=subprocess.DEVNULL,
            )
            # CSV: "ImageName","PID","SessionName","Session#","Mem"
            line = out.strip().splitlines()[0]
            parts = [p.strip().strip('"') for p in line.split(",")]
            if len(parts) >= 5:
                mem_str = parts[4].replace(" K", "").replace(",", "").strip()
                return int(mem_str)
        except Exception:
            pass
        return 0


# =========================================================================== #
# API 端点测量
# =========================================================================== #
# 端点定义：(method, path, body, name)
# body 为 None 表示 GET；否则为 POST JSON body。
ENDPOINTS: list[tuple[str, str, dict | None, str]] = [
    ("GET", "/api/v1/health", None, "health"),
    ("GET", "/api/v1/tracks", None, "tracks_list"),
    ("POST", "/api/v1/tracks/current", {"track_id": "suzuka"}, "select_track"),
    ("POST", "/api/v1/feedback",
     {"track_id": "suzuka", "corner_number": 1,
      "symptom": "understeer", "strength": 3}, "submit_feedback"),
    ("POST", "/api/v1/suggest", {"track_id": "suzuka"}, "suggest"),
    ("GET", "/api/v1/iteration/history?track_id=suzuka", None, "iteration_history"),
]


def measure_api(client: httpx.Client, base_url: str, n: int = 10) -> dict[str, dict]:
    """对每个端点测 n 次响应时间，返回 {name: stats}。"""
    results: dict[str, dict] = {}
    for method, path, body, name in ENDPOINTS:
        url = base_url + path
        samples: list[float] = []
        status_codes: list[int] = []
        for _ in range(n):
            t0 = _now_ms()
            try:
                if method == "GET":
                    resp = client.get(url, timeout=10.0)
                else:
                    resp = client.post(url, json=body, timeout=10.0)
                dt = _now_ms() - t0
                samples.append(dt)
                status_codes.append(resp.status_code)
            except Exception as e:
                samples.append(_now_ms() - t0)
                status_codes.append(-1)
                print(f"  [WARN] {name} 请求异常: {e}", file=sys.stderr)
        results[name] = {
            "method": method,
            "path": path,
            **_stats(samples),
            "status_codes": status_codes[:3],  # 仅记前 3 个供核对
        }
        print(f"  API {name}: mean={results[name]['mean_ms']:.2f}ms "
              f"p50={results[name]['p50_ms']:.2f}ms "
              f"p95={results[name]['p95_ms']:.2f}ms")
    return results


# =========================================================================== #
# 规则引擎纯函数测量
# =========================================================================== #
def measure_engine(n: int = 200) -> dict[str, dict]:
    """直接调用规则引擎纯函数，测量各阶段耗时。"""
    from setup_tuner.domain.setup import CarSetup
    from setup_tuner.engine.coupling import COUPLING_MATRIX
    from setup_tuner.engine.diagnostic import DIAG_DIMS, compute_dx
    from setup_tuner.engine.engine import (
        compute_setup_delta,
        generate_suggestion,
    )

    default_setup = CarSetup.default().to_dict()
    symptoms = [("understeer", 3), ("exit_wheelspin", 2), ("tyre_wear", 3)]

    # ① compute_dx
    dx_samples: list[float] = []
    for _ in range(n):
        t0 = _now_ms()
        compute_dx(symptoms)
        dx_samples.append(_now_ms() - t0)

    # ② 矩阵乘法（compute_setup_delta 内部主循环）
    dx = compute_dx(symptoms)
    delta_samples: list[float] = []
    for _ in range(n):
        t0 = _now_ms()
        compute_setup_delta(dx, default_setup)
        delta_samples.append(_now_ms() - t0)

    # ③ 完整 generate_suggestion（含报告组装）
    gen_samples: list[float] = []
    for _ in range(n):
        t0 = _now_ms()
        generate_suggestion(
            symptoms=symptoms,
            current_setup=default_setup,
            track_id="suzuka",
            telemetry=None,
        )
        gen_samples.append(_now_ms() - t0)

    # ④ 单次矩阵乘法裸测（隔离报告组装开销）
    # 模拟 compute_setup_delta 步骤1的核心循环
    from setup_tuner.domain.setup import ALL_SETUP_FIELDS
    matmul_samples: list[float] = []
    for _ in range(n):
        t0 = _now_ms()
        for spec in ALL_SETUP_FIELDS:
            raw = 0.0
            for dim in DIAG_DIMS:
                dx_val = dx.get(dim, 0.0)
                if dx_val == 0.0:
                    continue
                cell = COUPLING_MATRIX[dim][spec.name]
                if cell is None:
                    continue
                raw += dx_val * cell.value
        matmul_samples.append(_now_ms() - t0)

    return {
        "compute_dx": _stats(dx_samples),
        "matrix_mul_only": _stats(matmul_samples),
        "compute_setup_delta": _stats(delta_samples),
        "generate_suggestion_full": _stats(gen_samples),
    }


# =========================================================================== #
# SQLite 查询测量
# =========================================================================== #
def measure_sqlite(n: int = 200) -> dict[str, dict]:
    """直接调用 Store 各查询方法，测量耗时。"""
    import tempfile as _tf

    from setup_tuner.db.store import Store

    # 用临时文件库，预置数据
    tmp_dir = _tf.mkdtemp(prefix="f1opt_perf_sql_")
    db_path = os.path.join(tmp_dir, "perf.db")
    store = Store(db_path)

    # 预置：1 个赛道 + 50 条 feedback + 20 条 suggestion + 20 条 iteration
    store.upsert_track("suzuka", "Japanese GP", "Suzuka", "high_downforce",
                       5807.0, 18, "tracks/suzuka.svg", udp_track_id=2)
    for i in range(50):
        store.add_feedback("suzuka", (i % 18) + 1, "understeer", "entry",
                           strength=3, setup_id=None)
    for i in range(20):
        sid = store.save_suggestion("suzuka", f'{{"i": {i}}}', setup_id=None)
        store.save_iteration("suzuka", i + 1, None, None, sid)

    # ① get_feedbacks
    fb_samples: list[float] = []
    for _ in range(n):
        t0 = _now_ms()
        store.get_feedbacks("suzuka")
        fb_samples.append(_now_ms() - t0)

    # ② has_feedback
    has_samples: list[float] = []
    for _ in range(n):
        t0 = _now_ms()
        store.has_feedback("suzuka")
        has_samples.append(_now_ms() - t0)

    # ③ get_latest_suggestion
    ls_samples: list[float] = []
    for _ in range(n):
        t0 = _now_ms()
        store.get_latest_suggestion("suzuka")
        ls_samples.append(_now_ms() - t0)

    # ④ get_iterations
    it_samples: list[float] = []
    for _ in range(n):
        t0 = _now_ms()
        store.get_iterations("suzuka")
        it_samples.append(_now_ms() - t0)

    # ⑤ get_latest_setup（无数据，测空查询）
    lsetup_samples: list[float] = []
    for _ in range(n):
        t0 = _now_ms()
        store.get_latest_setup("suzuka")
        lsetup_samples.append(_now_ms() - t0)

    store.close()
    try:
        os.remove(db_path)
        # WAL/SHM 文件
        for suffix in ("-wal", "-shm"):
            p = db_path + suffix
            if os.path.exists(p):
                os.remove(p)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    return {
        "get_feedbacks_50rows": _stats(fb_samples),
        "has_feedback": _stats(has_samples),
        "get_latest_suggestion": _stats(ls_samples),
        "get_iterations_20rows": _stats(it_samples),
        "get_latest_setup_empty": _stats(lsetup_samples),
    }


# =========================================================================== #
# 服务启动
# =========================================================================== #
def start_server(data_dir: str, port: int) -> subprocess.Popen:
    """启动 uvicorn 子进程。"""
    env = os.environ.copy()
    env["DATA_DIR"] = data_dir
    env["LOG_LEVEL"] = "WARNING"  # 减少日志 I/O 噪声
    cmd = [
        sys.executable, "-m", "uvicorn",
        "setup_tuner.app:create_app", "--factory",
        "--host", "127.0.0.1", "--port", str(port),
        "--log-level", "warning",
    ]
    proc = subprocess.Popen(
        cmd, env=env, cwd=str(_REPO_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return proc


def wait_ready(base_url: str, timeout: float = 30.0) -> bool:
    """等待 /api/v1/health 就绪。"""
    deadline = time.time() + timeout
    with httpx.Client() as client:
        while time.time() < deadline:
            try:
                r = client.get(base_url + "/api/v1/health", timeout=2.0)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(0.2)
    return False


# =========================================================================== #
# 主流程
# =========================================================================== #
def main() -> int:
    parser = argparse.ArgumentParser(description="F1OPT 性能分析")
    parser.add_argument("--label", default="run", help="运行标签")
    parser.add_argument("--out", required=True, help="输出 JSON 路径")
    parser.add_argument("--port", type=int, default=18001, help="服务端口")
    parser.add_argument("--api-n", type=int, default=10, help="每端点测量次数")
    parser.add_argument("--engine-n", type=int, default=200, help="引擎测量次数")
    parser.add_argument("--sqlite-n", type=int, default=200, help="SQLite 测量次数")
    parser.add_argument("--skip-server", action="store_true",
                        help="跳过服务启动（仅测引擎+SQLite）")
    args = parser.parse_args()

    print(f"=== 性能分析 [{args.label}] ===")
    report: dict = {
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "repo_root": str(_REPO_ROOT),
    }

    # 内存基线（脚本进程自身）
    rss_before = _rss_kb()
    report["process_rss_kb_before_engine"] = rss_before

    # ① 规则引擎纯函数测量（不需要服务）
    print("[1/3] 测量规则引擎纯函数...")
    report["engine"] = measure_engine(n=args.engine_n)
    for k, v in report["engine"].items():
        print(f"  engine.{k}: mean={v['mean_ms']:.4f}ms p95={v['p95_ms']:.4f}ms")

    # ② SQLite 查询测量（不需要服务）
    print("[2/3] 测量 SQLite 查询...")
    report["sqlite"] = measure_sqlite(n=args.sqlite_n)
    for k, v in report["sqlite"].items():
        print(f"  sqlite.{k}: mean={v['mean_ms']:.4f}ms p95={v['p95_ms']:.4f}ms")

    # ③ API 端点测量（需要启动服务）
    if not args.skip_server:
        print("[3/3] 启动服务并测量 API 端点...")
        tmp_data = tempfile.mkdtemp(prefix="f1opt_perf_data_")
        proc = start_server(tmp_data, args.port)
        base_url = f"http://127.0.0.1:{args.port}"
        try:
            if not wait_ready(base_url):
                # 读取 stderr 辅助排查
                proc.terminate()
                err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
                print(f"[ERROR] 服务未就绪。stderr:\n{err}", file=sys.stderr)
                return 2
            print(f"  服务就绪: {base_url}")

            # 预置：先选赛道 + 提交 1 条 feedback，确保 suggest 端点可工作
            with httpx.Client() as client:
                client.post(base_url + "/api/v1/tracks/current",
                            json={"track_id": "suzuka"}, timeout=5.0)
                client.post(base_url + "/api/v1/feedback",
                            json={"track_id": "suzuka", "corner_number": 1,
                                  "symptom": "understeer", "strength": 3},
                            timeout=5.0)

            # 测量 API
            with httpx.Client() as client:
                report["api"] = measure_api(client, base_url, n=args.api_n)

            # 服务进程内存（Windows tasklist）
            try:
                out = subprocess.check_output(
                    ["tasklist", "/FI", f"PID eq {proc.pid}",
                     "/FO", "CSV", "/NH"],
                    text=True, stderr=subprocess.DEVNULL,
                )
                line = out.strip().splitlines()[0]
                parts = [p.strip().strip('"') for p in line.split(",")]
                if len(parts) >= 5:
                    mem_str = parts[4].replace(" K", "").replace(",", "").strip()
                    report["server_rss_kb"] = int(mem_str)
            except Exception:
                pass

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            # 清理临时 data_dir
            import shutil
            try:
                shutil.rmtree(tmp_data, ignore_errors=True)
            except Exception:
                pass
    else:
        print("[3/3] 跳过服务启动。")

    report["process_rss_kb_after"] = _rss_kb()

    # 输出 JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n报告已写入: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())