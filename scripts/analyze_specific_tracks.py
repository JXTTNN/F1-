"""检查特定赛道的弯道圆圈与赛道线path的精确关系。"""
import math
import re
import sys
from pathlib import Path

# 仓库根目录（本脚本位于 <root>/scripts/），避免硬编码绝对路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_track_svgs import path_to_points

TRACK_DIR = ROOT / "setup_tuner" / "ui" / "tracks"


def analyze_track(track_id):
    svg_path = TRACK_DIR / f"{track_id}.svg"
    content = svg_path.read_text(encoding="utf-8")

    # 提取赛道线path
    match = re.search(r'<path class="track-line" d="([^"]+)"', content)
    d = match.group(1)
    track_points = path_to_points(d)

    # 提取起点标记
    start_match = re.search(r'<rect x="([\d.]+)" y="([\d.]+)" width="10"', content)
    start = (float(start_match.group(1)) + 5, float(start_match.group(2)) + 5)

    # 提取弯道圆圈
    pattern = r'<circle cx="([\d.]+)" cy="([\d.]+)" r="6"'
    corners = [(float(cx), float(cy)) for cx, cy in re.findall(pattern, content)]

    # 计算赛道线bbox
    xs = [p[0] for p in track_points]
    ys = [p[1] for p in track_points]
    print(f"\n=== {track_id} ===")
    print(f"  赛道线: {len(track_points)}点, x=[{min(xs):.1f}, {max(xs):.1f}], y=[{min(ys):.1f}, {max(ys):.1f}]")
    print(f"  起点: ({start[0]:.1f}, {start[1]:.1f})")

    # 起点到赛道线的距离
    min_d = min(math.sqrt((start[0]-p[0])**2 + (start[1]-p[1])**2) for p in track_points)
    print(f"  起点到赛道线距离: {min_d:.1f}px")

    # 每个弯道到赛道线的距离
    for i, (cx, cy) in enumerate(corners, 1):
        min_d = min(math.sqrt((cx-p[0])**2 + (cy-p[1])**2) for p in track_points)
        # 检查弯道是否在画布范围内
        in_canvas = 0 <= cx <= 800 and 0 <= cy <= 600
        print(f"  弯道{i:2d}: ({cx:6.1f}, {cy:6.1f}) dist={min_d:4.1f}px {'✅' if min_d < 5 else '❌'} {'(画布外!)' if not in_canvas else ''}")


for tid in ["yas_marina", "monaco", "baku", "suzuka"]:
    analyze_track(tid)
