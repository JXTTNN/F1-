#!/usr/bin/env python3
"""诊断6条偏离赛道的弯道映射问题。"""
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "D:/F1OPT-Test/legacy")
from convert_track_svgs import (
    path_to_points, scale_points, download_svg, extract_path_d,
    CANVAS_W, CANVAS_H, MARGIN, TRACK_LAYOUT_MAP,
)
from f1opt.data.track_maps import TRACK_MAPS

problem_tracks = ['austin', 'baku', 'madrid', 'sakhir', 'shanghai', 'spielberg']

for tid in problem_tracks:
    tm = TRACK_MAPS[tid]
    layout_id = TRACK_LAYOUT_MAP[tid]

    svg_content = download_svg(layout_id)
    raw_d = extract_path_d(svg_content)
    raw_points = path_to_points(raw_d)
    scaled_points = scale_points(raw_points, CANVAS_W, CANVAS_H, MARGIN)

    # legacy数据
    legacy_total_m = tm.control_points[-1][0]
    legacy_start = (tm.control_points[0][1], tm.control_points[0][2])
    legacy_end = (tm.control_points[-1][1], tm.control_points[-1][2])

    # SVG path起点和终点
    svg_start = scaled_points[0]
    svg_end = scaled_points[-1]

    # legacy第一个弯道和最后一个弯道
    corners = sorted(tm.corners, key=lambda c: c.corner_id)
    first_corner = corners[0]
    last_corner = corners[-1]

    print(f"\n=== {tid} ===")
    print(f"  legacy总长度: {legacy_total_m}m")
    print(f"  legacy起点(px): ({legacy_start[0]}, {legacy_start[1]})")
    print(f"  legacy终点(px): ({legacy_end[0]}, {legacy_end[1]})")
    print(f"  SVG起点(px): ({svg_start[0]:.1f}, {svg_start[1]:.1f})")
    print(f"  SVG终点(px): ({svg_end[0]:.1f}, {svg_end[1]:.1f})")
    print(f"  legacy第一个弯道: C{first_corner.corner_id} dist={first_corner.distance_start}m px=({first_corner.x_px},{first_corner.y_px})")
    print(f"  legacy最后一个弯道: C{last_corner.corner_id} dist={last_corner.distance_start}m px=({last_corner.x_px},{last_corner.y_px})")

    # 计算SVG path的总弧长（像素）
    total_px = 0.0
    for i in range(len(scaled_points) - 1):
        dx = scaled_points[i+1][0] - scaled_points[i][0]
        dy = scaled_points[i+1][1] - scaled_points[i][1]
        total_px += math.sqrt(dx*dx + dy*dy)
    dx = scaled_points[0][0] - scaled_points[-1][0]
    dy = scaled_points[0][1] - scaled_points[-1][1]
    total_px += math.sqrt(dx*dx + dy*dy)
    print(f"  SVG path总弧长: {total_px:.1f}px")

    # 检查SVG path起点与legacy起点的距离
    start_dist = math.sqrt((svg_start[0]-legacy_start[0])**2 + (svg_start[1]-legacy_start[1])**2)
    print(f"  SVG起点与legacy起点距离: {start_dist:.1f}px")

    # 检查第一个弯道映射后的位置
    ratio = first_corner.distance_start / legacy_total_m
    target_px = ratio * total_px
    # 找到target_px对应的采样点
    arc_lens = [0.0]
    for i in range(len(scaled_points) - 1):
        dx = scaled_points[i+1][0] - scaled_points[i][0]
        dy = scaled_points[i+1][1] - scaled_points[i][1]
        arc_lens.append(arc_lens[-1] + math.sqrt(dx*dx + dy*dy))

    # 找到最接近target_px的采样点
    best_idx = 0
    best_diff = abs(arc_lens[0] - target_px)
    for i in range(1, len(arc_lens)):
        diff = abs(arc_lens[i] - target_px)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    mapped_first = scaled_points[best_idx]
    print(f"  第一个弯道映射: ratio={ratio:.3f}, target={target_px:.1f}px, mapped=({mapped_first[0]:.1f},{mapped_first[1]:.1f})")
    print(f"  第一个弯道legacy px: ({first_corner.x_px},{first_corner.y_px})")