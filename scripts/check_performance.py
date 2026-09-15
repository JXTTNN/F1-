"""F1OPT 性能检查脚本。

直接以 Python 脚本方式启动（不打包 exe），测量：
1. 冷启动时间（import create_app 到 uvicorn 就绪）
2. API 端到端响应延迟（health / tracks / svg / suggest）
3. 内存占用
4. 功能完整性（24 条赛道 SVG + 关键 API 端点）

不打开浏览器，直接启动 uvicorn 服务做测量。
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, "D:/F1OPT-Test")

HOST = "127.0.0.1"
PORT = 8000
BASE = f"http://{HOST}:{PORT}"


def wait_health(timeout=30.0):
    """轮询 health 接口直到就绪，返回就绪耗时。"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = urllib.request.urlopen(BASE + "/api/v1/health", timeout=1)
            if r.status == 200:
                return time.time() - start
        except Exception:
            pass
        time.sleep(0.1)
    return None


def timed_get(path, timeout=5):
    """计时 GET 请求，返回 (耗时秒, 状态码, 内容长度)。"""
    start = time.time()
    try:
        r = urllib.request.urlopen(BASE + path, timeout=timeout)
        data = r.read()
        return time.time() - start, r.status, len(data)
    except Exception as e:
        return time.time() - start, None, 0


def measure():
    print("=" * 70)
    print("F1OPT 性能检查（Python 脚本直跑，非 exe）")
    print("=" * 70)

    # 1. 冷启动时间
    print("\n[1] 测量冷启动时间（import + create_app + uvicorn 就绪）...")
    t0 = time.time()
    from setup_tuner.app import create_app
    from setup_tuner.config import load_config

    config = load_config()
    app = create_app(config)

    import threading
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=PORT, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    ready = wait_health()
    if ready is None:
        print("  ❌ 服务未在 30s 内就绪")
        server.should_exit = True
        return
    import_time = time.time() - t0
    print(f"  ✅ import+create_app 耗时: {import_time:.3f}s")
    print(f"  ✅ 服务就绪耗时: {ready:.3f}s")
    print(f"  ✅ 总冷启动时间: {import_time + ready:.3f}s")

    # 2. API 响应延迟
    print("\n[2] API 端到端响应延迟...")
    endpoints = [
        ("GET /api/v1/health", "/api/v1/health"),
        ("GET /api/v1/tracks", "/api/v1/tracks"),
        ("GET /api/v1/setup/fields", "/api/v1/setup/fields"),

        ("GET /static/tracks/suzuka.svg", "/static/tracks/suzuka.svg"),
        ("GET /static/tracks/baku.svg", "/static/tracks/baku.svg"),
        ("GET /static/tracks/jeddah.svg", "/static/tracks/jeddah.svg"),
    ]
    latencies = []
    for label, path in endpoints:
        dt, status, size = timed_get(path)
        latencies.append(dt)
        status_str = f"{status}" if status else "FAIL"
        print(f"  {label:45s} {dt*1000:7.1f}ms  [{status_str}] {size}B")
    avg = sum(latencies) / len(latencies)
    max_lat = max(latencies)
    print(f"  平均延迟: {avg*1000:.1f}ms   最大延迟: {max_lat*1000:.1f}ms")

    # 3. 内存占用
    print("\n[3] 内存占用...")
    import psutil

    pid = os.getpid()
    proc = psutil.Process(pid)
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    print(f"  当前进程 RSS 内存: {rss_mb:.1f} MB")

    # 4. 功能完整性
    print("\n[4] 功能完整性检查...")
    # 4a. 24 条赛道 SVG
    track_ids = [
        "melbourne", "shanghai", "suzuka", "sakhir", "jeddah",
        "miami", "monaco", "montreal", "barcelona", "silverstone",
        "spa", "hungaroring", "zandvoort", "monza", "austin",
        "mexico_city", "sao_paulo", "las_vegas", "baku", "singapore",
        "madrid", "spielberg", "lusail", "yas_marina",
    ]
    svg_ok = 0
    svg_fail = []
    for tid in track_ids:
        dt, status, _ = timed_get(f"/static/tracks/{tid}.svg")
        if status == 200:
            svg_ok += 1
        else:
            svg_fail.append(tid)
    print(f"  赛道SVG: {svg_ok}/{len(track_ids)} 加载成功")
    if svg_fail:
        print(f"  失败: {svg_fail}")

    # 4b. tracks API 数量
    dt, status, size = timed_get("/api/v1/tracks")
    n_tracks = 0
    if status == 200:
        data = json.loads(urllib.request.urlopen(BASE + "/api/v1/tracks", timeout=5).read())
        n_tracks = len(data.get("data", []))
    print(f"  tracks API: {n_tracks} 条赛道")

    # 4c. validate SVG 结构
    print("\n[5] SVG 结构验证（圆点/标注/防重叠）...")
    def validate_svg(tid):
        r = urllib.request.urlopen(f"{BASE}/static/tracks/{tid}.svg", timeout=5)
        root = ET.fromstring(r.read().decode("utf-8"))
        ns = {"svg": "http://www.w3.org/2000/svg"}
        circles = root.findall(".//svg:circle", ns)
        texts = root.findall(".//svg:text", ns)
        below = sum(1 for c, t in zip(circles, texts) if float(t.get("y")) > float(c.get("cy")))
        return len(circles), len(texts), below

    for tid in ["baku", "jeddah", "singapore"]:
        n_c, n_t, below = validate_svg(tid)
        print(f"  {tid:12s}: {n_c}圆点 {n_t}标注 {below}个下方标注(防重叠)")

    # 停止服务
    server.should_exit = True
    thread.join(timeout=5)

    print("\n" + "=" * 70)
    print("性能检查完成")
    print(f"  冷启动总时间: {import_time + ready:.3f}s")
    print(f"  API 平均延迟: {avg*1000:.1f}ms")
    print(f"  内存占用: {rss_mb:.1f} MB")
    print(f"  赛道SVG: {svg_ok}/24")
    print("=" * 70)


if __name__ == "__main__":
    measure()