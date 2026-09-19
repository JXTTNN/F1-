#!/usr/bin/env python3
"""端到端API验证：用FastAPI TestClient验证赛道图API。"""
import sys
from pathlib import Path

# 仓库根目录（本脚本位于 <root>/scripts/），避免硬编码绝对路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient

from setup_tuner.app import create_app

app = create_app()
client = TestClient(app)

all_ok = True

# 1. 验证 /tracks 返回赛道数据
resp = client.get("/tracks")
if resp.status_code == 200:
    data = resp.json()
    print(f"GET /tracks: OK ({len(data)} tracks)")
else:
    print(f"GET /tracks: FAILED (status={resp.status_code})")
    all_ok = False

# 2. 验证单条赛道数据
print("\n--- 弯道数据验证 ---")
for tid in ["melbourne", "monaco", "spa", "silverstone", "singapore", "mexico_city"]:
    resp = client.get(f"/tracks/{tid}")
    if resp.status_code == 200:
        data = resp.json()
        corners = data.get("corners", [])
        print(f"  {tid}: {len(corners)} corners")
    else:
        print(f"  {tid}: FAILED (status={resp.status_code})")
        all_ok = False

# 3. 验证SVG静态文件
print("\n--- SVG静态文件验证 ---")
for tid in ["melbourne", "monaco", "spa", "silverstone", "singapore", "mexico_city"]:
    resp = client.get(f"/static/tracks/{tid}.svg")
    if resp.status_code == 200:
        content = resp.text
        has_svg = "<svg" in content
        has_circle = "<circle" in content
        has_path = "<path" in content
        n_circles = content.count("<circle")
        print(f"  {tid}.svg: OK ({len(content)} bytes, svg={has_svg}, paths={has_path}, circles={n_circles})")
    else:
        print(f"  {tid}.svg: FAILED (status={resp.status_code})")
        all_ok = False

# 4. 验证首页加载（包含前端HTML）
print("\n--- 首页加载验证 ---")
resp = client.get("/")
if resp.status_code == 200:
    content = resp.text
    print(f"  GET /: OK ({len(content)} bytes)")
else:
    print(f"  GET /: FAILED (status={resp.status_code})")
    all_ok = False

print(f"\n{'=' * 60}")
if all_ok:
    print("端到端API验证通过！")
else:
    print("端到端API验证失败！")
