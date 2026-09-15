"""启动简单HTTP服务器展示24条赛道SVG，供浏览器直观检查。"""
import http.server
import socketserver
import threading
import time
import webbrowser
from pathlib import Path

PORT = 8899
TRACK_DIR = Path("D:/F1OPT-Test/setup_tuner/ui/tracks")

HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>F1赛道SVG检查</title>
<style>
body { background: #1a1a2e; color: #eee; font-family: monospace; margin: 20px; }
.grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }
.card { background: #0d0d12; border-radius: 8px; padding: 10px; }
.card h3 { color: #3B9EFF; font-size: 12px; margin: 0 0 8px 0; }
.card svg { width: 100%; height: auto; }
</style></head><body>
<h1>F1OPT 24条赛道SVG正确性检查</h1>
<div class="grid">
"""

svgs = sorted(TRACK_DIR.glob("*.svg"))
for svg in svgs:
    content = svg.read_text(encoding="utf-8")
    # 提取svg标签内容（去掉xml声明）
    svg_tag = content[content.find("<svg"):]
    name = svg.stem.upper()
    HTML += f'<div class="card"><h3>{name}</h3>{svg_tag}</div>\n'

HTML += "</div></body></html>"

# 写HTML文件
html_path = Path("D:/F1OPT-Test/scripts/track_check.html")
html_path.write_text(HTML, encoding="utf-8")
print(f"HTML已生成: {html_path}")
print(f"用浏览器打开: file:///{html_path}")