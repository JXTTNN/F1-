"""本地快速测试脚本 —— 用Python直接启动app跑B-F区65项检查。

用法:
    python scripts/run_local_test.py [--host HOST] [--port PORT]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

# 仓库根目录（本脚本位于 <root>/scripts/），避免硬编码绝对路径
ROOT = pathlib.Path(__file__).resolve().parents[1]
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    api = f"{base}/api/v1"

    passed = 0
    failed = 0
    items: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, failed
        mark = "✓" if ok else "✗"
        suffix = f" — {detail}" if detail else ""
        items.append(f"{mark} {label}{suffix}")
        if ok:
            passed += 1
        else:
            failed += 1

    client = httpx.Client(timeout=10.0)

    # ── B. 启动检查 10 项 ──
    items.append("── B. 启动检查（10项）──")
    try:
        r = client.get(f"{api}/health")
        body = r.json()
        d = body.get("data", {})
        check("B1 API就绪", r.status_code == 200)
        check("B2 health 200", r.status_code == 200)
        check("B3 status=ok", d.get("status") == "ok", f"status={d.get('status')}")
        check("B4 telemetry_connected字段", "telemetry_connected" in d)
        check("B5 udp_host=0.0.0.0", d.get("udp_host") == "0.0.0.0")
        check("B6 udp_port=20777", d.get("udp_port") == 20777)
        check("B7 current_track_id=None", d.get("current_track_id") is None)
        check("B8 code=0", body.get("code") == 0)
        check("B9 message存在", "message" in body)
        check("B10 data存在", "data" in body)
    except Exception as e:
        for i in range(1, 11):
            check(f"B{i}", False, str(e))

    # ── C. UI完整性 15 项 ──
    items.append("── C. UI完整性（15项）──")
    root_r = client.get(f"{base}/")
    idx_r = client.get(f"{base}/static/index.html")
    js_r = client.get(f"{base}/static/app.js")
    css_r = client.get(f"{base}/static/style.css")
    check("C1 GET/ 200", root_r.status_code == 200)

    check("C2 GET/ 返回HTML", "<html" in root_r.text[:500].lower())
    check("C3 index.html 200", idx_r.status_code == 200)
    idx_html = idx_r.text
    idx_bytes = len(idx_html.encode("utf-8"))
    check("C4 index.html>5000字节", idx_bytes > 5000, f"{idx_bytes}字节")
    check("C5 含<title>", "<title>" in idx_html.lower())
    check("C6 含app.js引用", "app.js" in idx_html)
    check("C7 含style.css引用", "style.css" in idx_html)
    app_js = js_r.text
    check("C8 app.js 200", js_r.status_code == 200)
    js_bytes = len(app_js.encode("utf-8"))
    check("C9 app.js>10000字节", js_bytes > 10000, f"{js_bytes}字节")
    check("C10 app.js含fetch", "fetch" in app_js)
    check("C11 app.js含WebSocket", "WebSocket" in app_js or "websocket" in app_js.lower())
    css_text = css_r.text
    check("C12 style.css 200", css_r.status_code == 200)
    css_bytes = len(css_text.encode("utf-8"))
    check("C13 style.css>1000字节", css_bytes > 1000, f"{css_bytes}字节")
    check("C14 css含body", "body" in css_text)
    check("C15 css含color", "color" in css_text or "#" in css_text)

    # ── D. 数据正确性 20 项 ──
    items.append("── D. 数据正确性（20项）──")
    r = client.get(f"{api}/tracks")
    body = r.json()
    tracks = body["data"]
    check("D1 /tracks 200", r.status_code == 200)
    check("D2 赛道数=24", len(tracks) == 24, f"{len(tracks)}")
    check("D3 code=0", body.get("code") == 0)
    field_ok = all("track_id" in t and "official_name" in t for t in tracks)
    check("D4 含track_id+official_name", field_ok)
    svg_ok = sum(1 for t in tracks if t.get("svg_path"))
    check("D5 全部有svg_path", svg_ok == 24, f"{svg_ok}/24")
    r2 = client.get(f"{api}/tracks/suzuka")
    check("D6 /tracks/suzuka 200", r2.status_code == 200)
    detail = r2.json()["data"]
    track_detail = detail.get("track", detail)  # 嵌套在track字段里
    check("D7 suzuka含corners", "corners" in detail)
    check("D8 suzuka含length_m", "length_m" in track_detail)
    check("D9 suzuka含track_type", "track_type" in track_detail)
    svg_access_ok = 0
    for t in tracks:
        sp = t.get("svg_path", "")
        if sp:
            sr = client.get(f"{base}/static/{sp}")
            if sr.status_code == 200 and "svg" in sr.text[:500].lower():
                svg_access_ok += 1
    check("D10 24SVG可访问", svg_access_ok == 24, f"{svg_access_ok}/24")
    names = [t.get("official_name") for t in tracks]
    check("D11 official_name不重复", len(names) == len(set(names)))
    ids = [t.get("track_id") for t in tracks]
    check("D12 id不重复", len(ids) == len(set(ids)))
    docs_r = client.get(f"{base}/docs")
    check("D13 /docs 200", docs_r.status_code == 200)
    oa_r = client.get(f"{base}/openapi.json")
    check("D14 /openapi.json 200", oa_r.status_code == 200)
    oa_text = oa_r.text
    check("D15 openapi含tracks", "tracks" in oa_text)
    check("D16 openapi含feedback", "feedback" in oa_text)
    check("D17 openapi含suggest", "suggest" in oa_text)
    check("D18 openapi含iteration", "iteration" in oa_text)
    check("D19 openapi含health", "health" in oa_text)
    check("D20 openapi含setup", "setup" in oa_text)

    # ── E. 业务逻辑 15 项 ──
    items.append("── E. 业务逻辑（15项）──")
    try:
        r = client.post(f"{api}/tracks/current", json={"track_id": "suzuka"}, timeout=30.0)
        check("E1 POST tracks/current 200", r.status_code == 200)
    except Exception as e:
        check("E1 POST tracks/current 200", False, str(e))
    r = client.get(f"{api}/health")
    d = r.json()["data"]
    ctid = d.get("current_track_id")
    check("E2 health显示current_track_id", ctid == "suzuka", f"current={ctid}")
    fb: dict = {}
    try:
        r = client.post(
            f"{api}/feedback",
            json={"track_id": "suzuka", "symptom": "understeer", "strength": 3},
            timeout=30.0,
        )
        fb = r.json()
        check("E3 POST feedback 200", r.status_code == 200)
    except Exception as e:
        check("E3 POST feedback 200", False, str(e))
    fb_id = fb.get("data", {}).get("id") if fb else None
    check("E4 feedback返回id", fb_id is not None, f"id={fb_id}")
    fb_data = fb.get("data", {}) if fb else {}
    check("E5 feedback含symptom", "symptom" in fb_data, f"keys={list(fb_data.keys())}")
    sg: dict = {}
    try:
        r = client.post(f"{api}/suggest", json={"track_id": "suzuka"}, timeout=30.0)
        sg = r.json()
        check("E6 POST suggest 200", r.status_code == 200)
    except Exception as e:
        check("E6 POST suggest 200", False, str(e))
    sg_id = sg.get("data", {}).get("suggestion_id") if sg else None
    check("E7 suggest返回id", sg_id is not None, f"id={sg_id}")
    sg_report = sg.get("data", {}).get("report") if sg else None
    check("E8 suggest含建议内容", sg_report is not None and len(str(sg_report)) > 0)
    try:
        r = client.get(f"{api}/iteration/history", params={"track_id": "suzuka"}, timeout=30.0)
        check("E9 GET iteration/history 200", r.status_code == 200)
        hist = r.json()["data"]
        check("E10 history>=1条", len(hist) >= 1, f"{len(hist)}条")
    except Exception as e:
        check("E9 GET iteration/history 200", False, str(e))
        check("E10 history>=1条", False, str(e))
    try:
        r = client.post(f"{api}/setup/import", timeout=30.0)
        check("E11 setup/import 409", r.status_code == 409, f"status={r.status_code}")
    except Exception as e:
        check("E11 setup/import 409", False, str(e))
    try:
        r = client.post(
            f"{api}/feedback",
            json={"track_id": "suzuka", "corner_number": 1, "symptom": "oversteer", "strength": 3},
            timeout=30.0,
        )
        check("E12 第二条feedback 200", r.status_code == 200)
    except Exception as e:
        check("E12 第二条feedback 200", False, str(e))
    sg_data = sg.get("data", {}) if sg else {}
    has_items = any(k in sg_data for k in ["adjustments", "suggestions", "items", "report"])
    check("E13 suggest含建议项", has_items, f"keys={list(sg_data.keys())}")
    try:
        r = client.post(
            f"{api}/feedback",
            json={"track_id": "suzuka", "symptom": "understeer", "strength": 3},
            timeout=30.0,
        )
        check("E14 重复反馈不崩溃", r.status_code in (200, 409), f"status={r.status_code}")
    except Exception as e:
        check("E14 重复反馈不崩溃", False, str(e))
    try:
        r = client.get(f"{api}/iteration/history", params={"track_id": "suzuka"}, timeout=30.0)
        hist = r.json()["data"]
        if len(hist) >= 2:
            ts = [h.get("created_at", h.get("timestamp", "")) for h in hist]
            is_desc = all(ts[i] >= ts[i + 1] for i in range(len(ts) - 1))
            check("E15 history倒序", is_desc)
        else:
            check("E15 history倒序", True, "仅1条")
    except Exception as e:
        check("E15 history倒序", False, str(e))

    # ── F. 性能基准 10 项 ──
    items.append("── F. 性能基准（10项）──")

    def measure(url: str, method: str = "GET", json_data: dict | None = None):
        t0 = time.perf_counter()
        try:
            if method == "GET":
                r = client.get(url)
            else:
                r = client.post(url, json=json_data)
            return (time.perf_counter() - t0) * 1000, r
        except Exception:
            return float("inf"), None

    ms, _ = measure(f"{api}/health")
    check("F1 health<200ms", ms < 200, f"{ms:.1f}ms")
    ms, _ = measure(f"{api}/tracks")
    check("F2 tracks<300ms", ms < 300, f"{ms:.1f}ms")
    ms, _ = measure(f"{api}/tracks/current", "POST", {"track_id": "suzuka"})
    check("F3 select<300ms", ms < 300, f"{ms:.1f}ms")
    ms, _ = measure(f"{api}/feedback", "POST", {"track_id": "suzuka", "symptom": "understeer", "strength": 2})
    check("F4 feedback<600ms", ms < 600, f"{ms:.1f}ms")
    ms, _ = measure(f"{api}/suggest", "POST", {"track_id": "suzuka"})
    check("F5 suggest<2000ms", ms < 2000, f"{ms:.1f}ms")
    ms, _ = measure(f"{api}/iteration/history")
    check("F6 history<300ms", ms < 300, f"{ms:.1f}ms")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda sp: client.get(f"{base}/static/{sp}"), [t.get("svg_path", "") for t in tracks]))
    svg_ms = (time.perf_counter() - t0) * 1000
    check("F7 24SVG总加载<3000ms", svg_ms < 3000, f"{svg_ms:.1f}ms")
    con_ok = True
    with ThreadPoolExecutor(max_workers=10) as pool:
        futs = [pool.submit(client.get, f"{api}/health") for _ in range(10)]
        for f in futs:
            try:
                if f.result().status_code != 200:
                    con_ok = False
            except Exception:
                con_ok = False
    check("F8 并发10请求全成功", con_ok)
    h_ok = True
    for _ in range(50):
        try:
            if client.get(f"{api}/health").status_code != 200:
                h_ok = False
                break
        except Exception:
            h_ok = False
            break
    check("F9 连续50次health无错误", h_ok)
    db = pathlib.ROOT / "data" / "f1opt.db"
    if db.exists():
        db_mb = db.stat().st_size / (1024 * 1024)
        check("F10 数据库<10MB", db_mb < 10, f"{db_mb:.2f}MB")
    else:
        check("F10 数据库<10MB", False, "未找到")

    client.close()

    # 打印结果
    total = passed + failed
    print(f"\n{'=' * 70}")
    print(f"B-F区深度检查: {passed}通过 / {failed}失败 (共{total}项)")
    print(f"{'=' * 70}")
    for item in items:
        print(f"  {item}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
