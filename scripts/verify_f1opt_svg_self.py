"""检查F1OPT SVG自身的赛道线path是否与弯道圆圈匹配。

关键：verify_corners_on_track.py检查的是原始SVG的path经scale_points后的结果，
但F1OPT SVG中的赛道线path可能与之不同。本脚本直接检查F1OPT SVG自身的path。
"""
import re
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_track_svgs import path_to_points

TRACK_DIR = Path("D:/F1OPT-Test/setup_tuner/ui/tracks")


def extract_f1opt_track_points(svg_path):
    """从F1OPT SVG中提取赛道线path的采样点。"""
    content = svg_path.read_text(encoding="utf-8")
    # 找 track-line 的 path d
    match = re.search(r'<path class="track-line" d="([^"]+)"', content)
    if not match:
        return []
    d = match.group(1)
    return path_to_points(d)


def extract_corners(svg_path):
    """提取弯道圆圈坐标。"""
    content = svg_path.read_text(encoding="utf-8")
    pattern = r'<circle cx="([\d.]+)" cy="([\d.]+)" r="6"'
    return [(float(cx), float(cy)) for cx, cy in re.findall(pattern, content)]


def min_dist_to_track(pt, track_points):
    """计算点到赛道线的最小距离。"""
    min_d = float("inf")
    for tp in track_points:
        d = math.sqrt((pt[0] - tp[0]) ** 2 + (pt[1] - tp[1]) ** 2)
        if d < min_d:
            min_d = d
    return min_d


print("=" * 70)
print("F1OPT SVG自身赛道线path vs 弯道圆圈位置检查")
print("=" * 70)

all_ok = True
for svg_path in sorted(TRACK_DIR.glob("*.svg")):
    tid = svg_path.stem
    track_points = extract_f1opt_track_points(svg_path)
    corners = extract_corners(svg_path)

    if not track_points:
        print(f"  {tid:15s}: 无法提取赛道线path ❌")
        all_ok = False
        continue

    max_dist = 0
    for cx, cy in corners:
        d = min_dist_to_track((cx, cy), track_points)
        if d > max_dist:
            max_dist = d

    status = "OK" if max_dist < 5.0 else "OFF_TRACK"
    if max_dist >= 5.0:
        all_ok = False
    print(f"  {tid:15s}: track_pts={len(track_points):4d} corners={len(corners):2d} max_dist={max_dist:6.1f}px [{status}]")

print(f"\n{'=' * 70}")
if all_ok:
    print("✅ 全部通过！弯道圆圈落在F1OPT SVG自身的赛道线上。")
else:
    print("❌ 部分赛道弯道圆圈偏离F1OPT SVG自身的赛道线！")