#!/usr/bin/env python3
"""验证所有24条赛道的弯道坐标是否落在赛道线上。"""
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "D:/F1OPT-Test/legacy")
from convert_track_svgs import (
    path_to_points, scale_points, download_svg, extract_path_d,
    CANVAS_W, CANVAS_H, MARGIN, TRACK_LAYOUT_MAP, TRACK_CORNERS_COUNT, OUTPUT_DIR,
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "setup_tuner" / "domain"))
from _track_anchors import TRACK_ANCHORS

def min_dist_to_track(pt, scaled_points):
    """计算点到赛道线的最小距离。"""
    min_d = float('inf')
    for sp in scaled_points:
        dx = pt[0] - sp[0]
        dy = pt[1] - sp[1]
        d = math.sqrt(dx*dx + dy*dy)
        if d < min_d:
            min_d = d
    return min_d

all_ok = True
max_overall = 0

for tid in sorted(TRACK_ANCHORS.keys()):
    layout_id = TRACK_LAYOUT_MAP[tid]
    svg_content = download_svg(layout_id)
    raw_d = extract_path_d(svg_content)
    raw_points = path_to_points(raw_d)
    scaled_points = scale_points(raw_points, CANVAS_W, CANVAS_H, MARGIN)

    corners = TRACK_ANCHORS[tid]
    max_dist = 0
    for cn, (cx, cy) in corners.items():
        d = min_dist_to_track((cx, cy), scaled_points)
        if d > max_dist:
            max_dist = d

    if max_dist > max_overall:
        max_overall = max_dist

    status = "OK" if max_dist < 10.0 else "OFF_TRACK"
    if max_dist >= 10.0:
        all_ok = False
    print(f"  {tid:15s}: max_dist_to_track={max_dist:5.1f}px  [{status}]")

print(f"\n{'=' * 60}")
print(f"全局最大偏离: {max_overall:.1f}px")
if all_ok:
    print("验证通过！所有弯道坐标都落在赛道线上（偏离<10px）。")
else:
    print("验证失败！部分弯道坐标偏离赛道线>10px。")