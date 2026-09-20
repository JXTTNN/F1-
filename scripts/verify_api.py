#!/usr/bin/env python3
"""端到端API验证：直接用uvicorn启动app，验证赛道图API。"""
import subprocess
import sys

# 写一个临时启动脚本
import tempfile
import time
import urllib.request
from pathlib import Path

# 仓库根目录（本脚本位于 <root>/scripts/），避免硬编码绝对路径
ROOT = Path(__file__).resolve().parents[1]
startup_script = """
import sys
import tempfile

sys.path.insert(0, str(ROOT))
import uvicorn

from setup_tuner.app import create_app
from setup_tuner.config import Config

# 测试脚本必须隔离数据目录：应用启动/关闭会按约定清理派生遥测数据
# （data/training/、data/sim_telemetry/），用默认 ./data 会删掉开发数据集
uvicorn.run(
    create_app(Config(data_dir=tempfile.mkdtemp(prefix="f1opt_test_"))),
    host="127.0.0.1", port=8000, log_level="error",
)
"""

tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
tmp.write(startup_script)
tmp.close()

proc = subprocess.Popen(
    [sys.executable, tmp.name],
    cwd=str(ROOT),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

try:
    for i in range(20):
        time.sleep(1)
        try:
            resp = urllib.request.urlopen("http://127.0.0.1:8000/api/tracks", timeout=2)
            if resp.status == 200:
                data = resp.read().decode("utf-8")
                print(f"API就绪（等待{i+1}秒），返回{len(data)} bytes")

                # 验证SVG
                resp2 = urllib.request.urlopen("http://127.0.0.1:8000/tracks/melbourne.svg", timeout=2)
                svg = resp2.read().decode("utf-8")
                print(f"SVG加载: OK ({len(svg)} bytes)")
                if "<svg" in svg:
                    print("SVG内容验证: 包含<svg>标签")

                # 验证多条赛道SVG
                for tid in ["monaco", "spa", "silverstone", "singapore"]:
                    resp3 = urllib.request.urlopen(f"http://127.0.0.1:8000/tracks/{tid}.svg", timeout=2)
                    svg3 = resp3.read().decode("utf-8")
                    has_svg = "<svg" in svg3
                    has_circle = "<circle" in svg3
                    print(f"  {tid}.svg: {len(svg3)} bytes, svg={has_svg}, circles={has_circle}")

                break
        except Exception:
            if i == 19:
                err = proc.stderr.read().decode("utf-8", errors="replace")[:500]
                print(f"API未就绪。stderr: {err}")
    else:
        print("API未就绪（20秒超时）")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("进程已关闭")

import os

os.unlink(tmp.name)
