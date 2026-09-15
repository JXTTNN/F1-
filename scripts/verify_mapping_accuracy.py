#!/usr/bin/env python3
"""验证距离比例映射的弯道坐标是否落在赛道线上。
计算每个映射弯道点到最近赛道采样点的距离。"""
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "D:/F1OPT-Test/legacy")
from convert_track_svgs import (
    path_to_points, scale_points, download_svg, extract_path_d,
    CANVAS_W, CANVAS_H, MARGIN, TRACK_LAYOUT_MAP, TRACK_CORNERS_COUNT,
)
from f1opt.data.track_maps import TRACK_MAPS

def compute_arc_lengths(points):
    arc_lens = [0.0]
    for i in range(len(points) - 1):
        dx = points[i+1][0] - points[i][0]
        dy = points[i+1][1] - points[i][1]
        arc_lens.append(arc_lens[-1] + math.sqrt(dx*dx + dy*dy))
    dx = points[0][0] - points[-1][0]
    dy = points[0][1] - points[-1][1]
    total = arc_lens[-1] + math.sqrt(dx*dx + dy*dy)
    return arc_lens, total

def find_point_at_ratio(points, arc_lens, total_len, ratio):
    target = ratio * total_len
    lo, hi = 0, len(arc_lens) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if arc_lens[mid] < target:
            lo = mid
        else:
            hi = mid
    seg_len = arc_lens[hi] - arc_lens[lo]
    if seg_len < 1e-9:
        return points[lo]
    t = (target - arc_lens[lo]) / seg_len
    x = points[lo][0] + t * (points[hi][0] - points[lo][0])
    y = points[lo][1] + t * (points[hi][1] - points[lo][1])
    return (x, y)

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

# 测试多条赛道
for tid in ['melbourne', 'monaco', 'spa', 'silverstone', 'singapore']:
    tm = TRACK_MAPS[tid]
    layout_id = TRACK_LAYOUT_MAP[tid]

    svg_content = download_svg(layout_id)
    raw_d = extract_path_d(svg_content)
    raw_points = path_to_points(raw_d)
    scaled_points = scale_points(raw_points, CANVAS_W, CANVAS_H, MARGIN)

    arc_lens, total_len = compute_arc_lengths(scaled_points)
    legacy_total_m = tm.control_points[-1][0]

    corners = sorted(tm.corners, key=lambda c: c.corner_id)
    print(f"\n=== {tid} ({len(corners)} legacy corners, {TRACK_CORNERS_COUNT[tid]} expected) ===")

    # 检查SVG path起点是否接近legacy第一个control_point
    legacy_start = (tm.control_points[0][1], tm.control_points[0][2])  # (x_px, y_px) at distance=0
    svg_start = scaled_points[0]
    print(f"  legacy start: ({legacy_start[0]}, {legacy_start[1]})")
    print(f"  svg start:    ({svg_start[0]:.1f}, {svg_start[1]:.1f})")

    # 距离比例映射
    max_dist_to_track = 0
    for c in corners:
        corner_mid_dist = (c.distance_start + c.distance_end) / 2
        ratio = corner_mid_dist / legacy_total_m
        mapped_pt = find_point_at_ratio(scaled_points, arc_lens, total_len, ratio)
        dist_to_track = min_dist_to_track(mapped_pt, scaled_points)
        if dist_to_track > max_dist_to_track:
            max_dist_to_track = dist_to_track

    print(f"  距离比例映射 — 最大偏离赛道线距离: {max_dist_to_track:.1f}px")

    # 对比：legacy像素坐标偏离赛道线的距离
    max_legacy_dist = 0
    for c in corners:
        legacy_pt = (c.x_px, c.y_px)
        dist_to_track = min_dist_to_track(legacy_pt, scaled_points)
        if dist_to_track > max_legacy_dist:
            max_legacy_dist = dist_to_track

    print(f"  legacy像素坐标 — 最大偏离赛道线距离: {max_legacy_dist:.1f}px")