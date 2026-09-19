#!/usr/bin/env python3
"""端到端启动验证：启动F1OPT.exe，等待API就绪，验证赛道图API返回正确数据。"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# 仓库根目录（本脚本位于 <root>/scripts/），避免硬编码绝对路径
ROOT = Path(__file__).resolve().parents[1]
exe_path = ROOT / "dist" / "f1opt" / "f1opt.exe"
work_dir = exe_path.parent

print(f"启动 {exe_path}...")
print(f"工作目录: {work_dir}")

# 用subprocess.Popen启动（CreateProcess方式，继承标准流）
proc = subprocess.Popen(
    [str(exe_path)],
    cwd=str(work_dir),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# 等待API就绪（最多30秒）
api_url = "http://127.0.0.1:8000"
max_wait = 30
ready = False

for i in range(max_wait):
    time.sleep(1)
    try:
        resp = urllib.request.urlopen(f"{api_url}/api/tracks", timeout=2)
        if resp.status == 200:
            ready = True
            print(f"API就绪（等待{i+1}秒）")
            break
    except Exception:
        pass

if not ready:
    print(f"API未就绪（等待{max_wait}秒超时）")
    proc.terminate()
    sys.exit(1)

# 验证赛道图API
try:
    resp = urllib.request.urlopen(f"{api_url}/api/tracks", timeout=5)
    data = resp.read().decode("utf-8")
    print(f"API /api/tracks 返回 {len(data)} bytes")

    # 检查是否包含赛道数据
    if '"track_id"' in data or '"corners"' in data:
        print("赛道数据验证: OK")
    else:
        print("赛道数据验证: 可能有问题（未找到关键字段）")

    # 尝试获取单条赛道的SVG
    resp2 = urllib.request.urlopen(f"{api_url}/tracks/melbourne.svg", timeout=5)
    svg_data = resp2.read().decode("utf-8")
    if "<svg" in svg_data and "</svg>" in svg_data:
        print(f"SVG加载验证: OK (melbourne.svg, {len(svg_data)} bytes)")
    else:
        print("SVG加载验证: FAILED")

except Exception as e:
    print(f"API验证失败: {e}")
finally:
    # 关闭进程
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("\n进程已关闭。")
