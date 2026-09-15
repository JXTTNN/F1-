"""用Edge无头模式将SVG转为PNG，供analyzeImage分析。"""
import subprocess
from pathlib import Path

TRACK_DIR = Path("D:/F1OPT-Test/setup_tuner/ui/tracks")
PNG_DIR = Path("D:/F1OPT-Test/scripts/track_pngs")
PNG_DIR.mkdir(exist_ok=True)

# 关键赛道：形状复杂的代表性赛道
KEY_TRACKS = ["yas_marina", "suzuka", "monaco", "baku", "spa", "jeddah", "singapore", "melbourne"]

edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not Path(edge).exists():
    edge = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

for tid in KEY_TRACKS:
    svg_path = TRACK_DIR / f"{tid}.svg"
    png_path = PNG_DIR / f"{tid}.png"
    # Edge无头模式截图
    cmd = [
        edge,
        "--headless",
        "--disable-gpu",
        "--screenshot=" + str(png_path),
        "--window-size=800,600",
        "file:///" + str(svg_path).replace("\\", "/"),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=15)
    if png_path.exists():
        sz = png_path.stat().st_size / 1024
        print(f"  {tid}: {sz:.0f}KB ✅")
    else:
        print(f"  {tid}: 失败 ❌")

print(f"\nPNG文件在: {PNG_DIR}")