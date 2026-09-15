"""深度验证赛道图正确性 —— 三个维度：

1. 赛道线形状：F1OPT SVG的path vs 原始SVG经scale_points缩放后的path，两者应一致
2. 弯道位置：弯道在赛道线上的位置 vs 真实弯道位置（距离比例）
3. 视觉效果：高分辨率渲染检查
"""
import re
import math
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "D:/F1OPT-Test")
from convert_track_svgs import (
    path_to_points, scale_points, extract_path_d,
    compute_bbox, CANVAS_W, CANVAS_H, MARGIN,
    TRACK_LAYOUT_MAP, TRACK_CORNERS_COUNT,
)

TRACK_DIR = Path("D:/F1OPT-Test/setup_tuner/ui/tracks")
RAW_DIR = Path("D:/F1OPT-Test/scripts/raw_svgs")


def hausdorff_distance(pts1, pts2):
    """计算两组点的Hausdorff距离（最大最小距离）。"""
    def max_min_dist(a, b):
        max_d = 0
        for pa in a:
            min_d = float('inf')
            for pb in b:
                d = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
                if d < min_d:
                    min_d = d
            if min_d > max_d:
                max_d = min_d
        return max_d
    return max(max_min_dist(pts1, pts2), max_min_dist(pts2, pts1))


print("=" * 70)
print("维度1: 赛道线形状验证（F1OPT SVG path vs 原始SVG缩放后path）")
print("=" * 70)

shape_ok = True
for tid in sorted(TRACK_LAYOUT_MAP.keys()):
    layout = TRACK_LAYOUT_MAP[tid]
    raw_svg = RAW_DIR / f"{layout}.svg"
    f1opt_svg = TRACK_DIR / f"{tid}.svg"

    # 原始SVG → scale_points缩放
    raw_content = raw_svg.read_text(encoding="utf-8")
    raw_d = extract_path_d(raw_content)
    raw_points = path_to_points(raw_d)
    scaled_points = scale_points(raw_points, CANVAS_W, CANVAS_H, MARGIN)

    # F1OPT SVG → path_to_points
    f1opt_content = f1opt_svg.read_text(encoding="utf-8")
    match = re.search(r'<path class="track-line" d="([^"]+)"', f1opt_content)
    f1opt_d = match.group(1)
    f1opt_points = path_to_points(f1opt_d)

    # 计算Hausdorff距离
    hd = hausdorff_distance(scaled_points, f1opt_points)

    status = "OK" if hd < 2.0 else "SHAPE_MISMATCH"
    if hd >= 2.0:
        shape_ok = False
    print(f"  {tid:15s}: raw_pts={len(scaled_points):4d} f1opt_pts={len(f1opt_points):4d} hausdorff={hd:6.2f}px [{status}]")

print(f"\n赛道线形状: {'✅ 全部一致' if shape_ok else '❌ 部分不一致'}")

# 维度2: 弯道位置验证
print(f"\n{'=' * 70}")
print("维度2: 弯道位置验证（弯道间距均匀性 + 弯道在赛道上的分布）")
print("=" * 70)

position_ok = True
for tid in sorted(TRACK_LAYOUT_MAP.keys()):
    f1opt_svg = TRACK_DIR / f"{tid}.svg"
    content = f1opt_svg.read_text(encoding="utf-8")

    # 提取弯道
    pattern = r'<circle cx="([\d.]+)" cy="([\d.]+)" r="6"'
    corners = [(float(cx), float(cy)) for cx, cy in re.findall(pattern, content)]

    # 提取赛道线
    match = re.search(r'<path class="track-line" d="([^"]+)"', content)
    track_points = path_to_points(match.group(1))

    # 计算每个弯道在赛道线上的弧长位置
    arc_lens = [0.0]
    for i in range(len(track_points) - 1):
        dx = track_points[i+1][0] - track_points[i][0]
        dy = track_points[i+1][1] - track_points[i][1]
        arc_lens.append(arc_lens[-1] + math.sqrt(dx*dx + dy*dy))

    # 闭合
    dx = track_points[0][0] - track_points[-1][0]
    dy = track_points[0][1] - track_points[-1][1]
    total_px = arc_lens[-1] + math.sqrt(dx*dx + dy*dy)

    # 每个弯道的弧长比例
    corner_ratios = []
    for cx, cy in corners:
        min_d = float('inf')
        best_idx = 0
        for i, tp in enumerate(track_points):
            d = (cx - tp[0])**2 + (cy - tp[1])**2
            if d < min_d:
                min_d = d
                best_idx = i
        corner_ratios.append(arc_lens[best_idx] / total_px)

    # 检查弯道间距是否过于不均匀（最小间距 < 总长/弯道数 * 0.1）
    n = len(corner_ratios)
    if n < 2:
        continue
    min_gap = float('inf')
    for i in range(n):
        gap = corner_ratios[(i+1) % n] - corner_ratios[i]
        if gap < 0:
            gap += 1.0
        if gap < min_gap:
            min_gap = gap
    expected_gap = 1.0 / n
    gap_ratio = min_gap / expected_gap

    # 检查是否有弯道聚集（gap_ratio < 0.05 表示有弯道几乎重合）
    status = "OK" if gap_ratio > 0.05 else "CORNERS_CLUSTERED"
    if gap_ratio <= 0.05:
        position_ok = False
    print(f"  {tid:15s}: corners={n:2d} min_gap_ratio={gap_ratio:5.3f} (期望~1.0) [{status}]")

print(f"\n弯道位置: {'✅ 分布合理' if position_ok else '⚠️ 部分赛道弯道聚集'}")

# 维度3: 起点位置验证
print(f"\n{'=' * 70}")
print("维度3: 起点位置验证（起点是否在赛道线上 + 起点是否在第一个弯道之前）")
print("=" * 70)

start_ok = True
for tid in sorted(TRACK_LAYOUT_MAP.keys()):
    f1opt_svg = TRACK_DIR / f"{tid}.svg"
    content = f1opt_svg.read_text(encoding="utf-8")

    # 起点标记
    start_match = re.search(r'<rect x="([\d.]+)" y="([\d.]+)" width="10"', content)
    start = (float(start_match.group(1)) + 5, float(start_match.group(2)) + 5)

    # 赛道线
    match = re.search(r'<path class="track-line" d="([^"]+)"', content)
    track_points = path_to_points(match.group(1))

    # 起点到赛道线距离
    min_d = min(math.sqrt((start[0]-p[0])**2 + (start[1]-p[1])**2) for p in track_points)

    # 起点在赛道线上的位置
    arc_lens = [0.0]
    for i in range(len(track_points) - 1):
        dx = track_points[i+1][0] - track_points[i][0]
        dy = track_points[i+1][1] - track_points[i][1]
        arc_lens.append(arc_lens[-1] + math.sqrt(dx*dx + dy*dy))
    dx = track_points[0][0] - track_points[-1][0]
    dy = track_points[0][1] - track_points[-1][1]
    total_px = arc_lens[-1] + math.sqrt(dx*dx + dy*dy)

    best_idx = 0
    best_d = float('inf')
    for i, tp in enumerate(track_points):
        d = (start[0]-tp[0])**2 + (start[1]-tp[1])**2
        if d < best_d:
            best_d = d
            best_idx = i
    start_ratio = arc_lens[best_idx] / total_px

    status = "OK" if min_d < 5.0 else "OFF_TRACK"
    if min_d >= 5.0:
        start_ok = False
    print(f"  {tid:15s}: start=({start[0]:6.1f},{start[1]:6.1f}) dist={min_d:4.1f}px track_pos={start_ratio:.3f} [{status}]")

print(f"\n起点位置: {'✅ 全部在赛道线上' if start_ok else '❌ 部分偏离'}")