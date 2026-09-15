#!/usr/bin/env python3
"""测试曲率分析方案：直接在SVG path上检测弯道，确保弯道标注落在赛道线上。"""
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_track_svgs import (
    path_to_points, scale_points, download_svg, extract_path_d,
    detect_corners, CANVAS_W, CANVAS_H, MARGIN, TRACK_LAYOUT_MAP, TRACK_CORNERS_COUNT,
)

def min_dist_to_track(pt, scaled_points):
    min_d = float('inf')
    for sp in scaled_points:
        dx = pt[0] - sp[0]
        dy = pt[1] - sp[1]
        d = math.sqrt(dx*dx + dy*dy)
        if d < min_d:
            min_d = d
    return min_d

all_ok = True

for tid in sorted(TRACK_LAYOUT_MAP.keys()):
    layout_id = TRACK_LAYOUT_MAP[tid]
    n_corners = TRACK_CORNERS_COUNT[tid]

    svg_content = download_svg(layout_id)
    raw_d = extract_path_d(svg_content)
    raw_points = path_to_points(raw_d)
    scaled_points = scale_points(raw_points, CANVAS_W, CANVAS_H, MARGIN)

    # 用曲率分析检测弯道
    corner_points = detect_corners(scaled_points, n_corners)

    # 验证弯道数量
    n_detected = len(corner_points)

    # 验证弯道是否落在赛道线上
    max_dist = 0
    for pt in corner_points:
        d = min_dist_to_track(pt, scaled_points)
        if d > max_dist:
            max_dist = d

    # 验证弯道间距
    min_spacing = float('inf')
    for i in range(1, len(corner_points)):
        dx = corner_points[i][0] - corner_points[i-1][0]
        dy = corner_points[i][1] - corner_points[i-1][1]
        d = (dx*dx + dy*dy)**0.5
        if d < min_spacing:
            min_spacing = d

    count_ok = n_detected == n_corners
    track_ok = max_dist < 5.0  # 曲率分析的弯道一定在赛道线上
    spacing_ok = min_spacing >= 10.0

    status = "OK" if (count_ok and track_ok) else "ISSUE"
    if not count_ok:
        status += f" count={n_detected}/{n_corners}"
    if not track_ok:
        status += f" off_track={max_dist:.1f}"
    if not spacing_ok:
        status += f" spacing={min_spacing:.1f}"

    if not count_ok or not track_ok:
        all_ok = False

    print(f"  {tid:15s}: {n_detected}/{n_corners} corners, max_dist={max_dist:.1f}px, min_spacing={min_spacing:.1f}px  [{status}]")

print(f"\n{'=' * 60}")
if all_ok:
    print("曲率分析方案验证通过！")
else:
    print("部分赛道有问题，需要进一步调优。")