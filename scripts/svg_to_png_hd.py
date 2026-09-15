"""用高分辨率渲染所有24条赛道SVG，供analyzeImage检查。"""
import subprocess
from pathlib import Path

TRACK_DIR = Path("D:/F1OPT-Test/setup_tuner/ui/tracks")
PNG_DIR = Path("D:/F1OPT-Test/scripts/track_pngs_hd")
PNG_DIR.mkdir(exist_ok=True)

edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not Path(edge).exists():
    edge = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

for svg_path in sorted(TRACK_DIR.glob("*.svg")):
    tid = svg_path.stem
    png_path = PNG_DIR / f"{tid}.png"
    cmd = [
        edge, "--headless", "--disable-gpu",
        "--screenshot=" + str(png_path),
        "--window-size=1600,1200",
        "file:///" + str(svg_path).replace("\\", "/"),
    ]
    subprocess.run(cmd, capture_output=True, timeout=15)
    if png_path.exists():
        sz = png_path.stat().st_size / 1024
        print(f"  {tid}: {sz:.0f}KB ✅")
    else:
        print(f"  {tid}: 失败 ❌")