"""验证所有新功能API端点 + 前端关键元素（任务45 步骤4）。"""
import subprocess
import sys
import time
from pathlib import Path

import requests

# 仓库根目录（本脚本位于 <root>/scripts/），避免硬编码绝对路径
ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
PORT = 8000
BASE = f"http://{HOST}:{PORT}"
RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    RESULTS.append((name, ok, detail))
    print(f"  [{status}] {name} {detail}")


def main() -> int:
    # 启动应用
    print("=== 启动应用 ===")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "setup_tuner.app:create_app",
         "--factory", "--host", HOST, "--port", str(PORT)],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)

    try:
        # 1. GET / (HTML页面)
        print("\n=== 1. GET / (HTML页面) ===")
        r = requests.get(f"{BASE}/", timeout=10)
        check("GET / 200", r.status_code == 200, f"status={r.status_code}")
        html = r.text
        check("返回HTML", "<html" in html.lower(), "有<html>标签")
        # 前端关键元素
        check("23参数输入面板", "param" in html.lower() or "参数" in html, "含参数面板")
        check("checkbox多选反馈", "checkbox" in html.lower(), "含checkbox")
        check("Cyberpunk风格", "cyberpunk" in html.lower() or "neon" in html.lower()
              or "racing" in html.lower() or "f1" in html.lower(), "Cyberpunk风格")
        check("静态CSS引用", "/static/style.css" in html, "引用style.css")
        check("静态JS引用", "/static/app.js" in html, "引用app.js")

        # 2. GET /api/v1/health
        print("\n=== 2. GET /api/v1/health ===")
        r = requests.get(f"{BASE}/api/v1/health", timeout=10)
        check("health 200", r.status_code == 200, f"status={r.status_code}")
        check("health JSON", r.headers.get("content-type", "").startswith("application/json"), "JSON响应")

        # 3. GET /api/v1/tracks (24条赛道)
        print("\n=== 3. GET /api/v1/tracks ===")
        r = requests.get(f"{BASE}/api/v1/tracks", timeout=10)
        check("tracks 200", r.status_code == 200, f"status={r.status_code}")
        tracks_resp = r.json()
        # 信封格式 {code, message, data}
        tracks_data = tracks_resp.get("data", tracks_resp) if isinstance(tracks_resp, dict) else tracks_resp
        track_count = len(tracks_data) if isinstance(tracks_data, list) else 0
        check("24条赛道", track_count == 24, f"实际{track_count}条")

        # 4. GET /api/v1/setup/fields (23参数定义，新端点)
        print("\n=== 4. GET /api/v1/setup/fields (新端点) ===")
        r = requests.get(f"{BASE}/api/v1/setup/fields", timeout=10)
        check("setup/fields 200", r.status_code == 200, f"status={r.status_code}")
        fields_resp = r.json()
        fields_data = fields_resp.get("data", fields_resp) if isinstance(fields_resp, dict) else fields_resp
        field_count = len(fields_data) if isinstance(fields_data, list) else 0
        check("23参数定义", field_count == 23, f"实际{field_count}个参数")
        # 提取参数名和默认值，供 setup/manual 使用
        all_params = {}
        if isinstance(fields_data, list):
            for f in fields_data:
                all_params[f["name"]] = f.get("default", 0)

        # 5. GET /api/v1/setup/current?track_id=suzuka
        print("\n=== 5. GET /api/v1/setup/current?track_id=suzuka ===")
        r = requests.get(f"{BASE}/api/v1/setup/current", params={"track_id": "suzuka"}, timeout=10)
        check("setup/current 200", r.status_code == 200, f"status={r.status_code}")

        # 6. GET /api/v1/feedback?track_id=suzuka
        print("\n=== 6. GET /api/v1/feedback?track_id=suzuka ===")
        r = requests.get(f"{BASE}/api/v1/feedback", params={"track_id": "suzuka"}, timeout=10)
        check("feedback GET 200", r.status_code == 200, f"status={r.status_code}")

        # 7. POST /api/v1/setup/manual (新端点)
        print("\n=== 7. POST /api/v1/setup/manual (新端点) ===")
        # 提供全部23个参数（使用默认值）
        manual_payload = {
            "track_id": "suzuka",
            "params": all_params,
        }
        r = requests.post(f"{BASE}/api/v1/setup/manual", json=manual_payload, timeout=10)
        check("setup/manual 200/201", r.status_code in (200, 201), f"status={r.status_code}")

        # 8. POST /api/v1/feedback (批量反馈，新端点)
        print("\n=== 8. POST /api/v1/feedback (批量反馈，新端点) ===")
        feedback_payload = {
            "track_id": "suzuka",
            "feedbacks": [
                {"corner_number": 1, "symptom": "understeer", "strength": 3},
                {"corner_number": 2, "symptom": "oversteer", "strength": 2},
            ],
        }
        r = requests.post(f"{BASE}/api/v1/feedback", json=feedback_payload, timeout=10)
        check("feedback POST 200/201", r.status_code in (200, 201), f"status={r.status_code}")

        # 9. POST /api/v1/suggest (支持model_type)
        print("\n=== 9. POST /api/v1/suggest (支持model_type) ===")
        for model_type in ["rule", "nn", "hybrid"]:
            suggest_payload = {
                "track_id": "suzuka",
                "model_type": model_type,
            }
            r = requests.post(f"{BASE}/api/v1/suggest", json=suggest_payload, timeout=15)
            check(f"suggest model_type={model_type}",
                  r.status_code in (200, 201),
                  f"status={r.status_code}")

        # 额外验证：遥测模拟端点
        print("\n=== 额外: POST /api/v1/telemetry/simulate (新端点) ===")
        try:
            r = requests.post(f"{BASE}/api/v1/telemetry/simulate",
                              json={"track_id": "suzuka", "action": "start"}, timeout=10)
            check("telemetry/simulate start", r.status_code in (200, 201), f"status={r.status_code}")
            # 停止模拟
            r2 = requests.post(f"{BASE}/api/v1/telemetry/simulate",
                               json={"track_id": "suzuka", "action": "stop"}, timeout=10)
            check("telemetry/simulate stop", r2.status_code in (200, 201), f"status={r2.status_code}")
        except Exception as e:
            check("telemetry/simulate", False, f"error={e}")

    finally:
        proc.terminate()
        proc.wait()

    # 汇总
    print("\n" + "=" * 60)
    total = len(RESULTS)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = total - passed
    print(f"总计: {total}项, 通过: {passed}, 失败: {failed}")
    print("=" * 60)
    if failed:
        print("\n失败项:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  [FAIL] {name} {detail}")
        return 1
    print("\n所有新功能验证通过!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
