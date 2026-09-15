#!/usr/bin/env python3
"""验证距离比例映射方案：用弯道distance_start在SVG path采样点上定位弯道。"""
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "D:/F1OPT-Test/legacy")
from convert_track_svgs import (
    path_to_points, scale_points, compute_bbox, download_svg, extract_path_d,
    CANVAS_W, CANVAS_H, MARGIN, TRACK_LAYOUT_MAP,
)
from f1opt.data.track_maps import TRACK_MAPS

def compute_arc_lengths(points):
    """计算采样点的累积弧长。"""
    arc_lens = [0.0]
    for i in range(len(points) - 1):
        dx = points[i+1][0] - points[i][0]
        dy = points[i+1][1] - points[i][1]
        arc_lens.append(arc_lens[-1] + math.sqrt(dx*dx + dy*dy))
    # 闭合路径
    dx = points[0][0] - points[-1][0]
    dy = points[0][1] - points[-1][1]
    total = arc_lens[-1] + math.sqrt(dx*dx + dy*dy)
    return arc_lens, total

def find_point_at_ratio(points, arc_lens, total_len, ratio):
    """根据比例(0-1)在采样点上找到对应位置的点（插值）。"""
    target = ratio * total_len
    # 二分查找
    lo, hi = 0, len(arc_lens) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if arc_lens[mid] < target:
            lo = mid
        else:
            hi = mid
    # 在arc_lens[lo]和arc_lens[hi]之间插值
    seg_len = arc_lens[hi] - arc_lens[lo]
    if seg_len < 1e-9:
        return points[lo]
    t = (target - arc_lens[lo]) / seg_len
    x = points[lo][0] + t * (points[hi][0] - points[lo][0])
    y = points[lo][1] + t * (points[hi][1] - points[lo][1])
    return (x, y)

# 测试melbourne
tid = 'melbourne'
tm = TRACK_MAPS[tid]
layout_id = TRACK_LAYOUT_MAP[tid]

# 1. 下载SVG并提取path
svg_content = download_svg(layout_id)
raw_d = extract_path_d(svg_content)

# 2. 解析为采样点
raw_points = path_to_points(raw_d)
print(f"raw_points: {len(raw_points)} points")

# 3. 缩放到800x600画布
scaled_points = scale_points(raw_points, CANVAS_W, CANVAS_H, MARGIN)
print(f"scaled_points: {len(scaled_points)} points")
print(f"  x range: [{min(p[0] for p in scaled_points):.1f}, {max(p[0] for p in scaled_points):.1f}]")
print(f"  y range: [{min(p[1] for p in scaled_points):.1f}, {max(p[1] for p in scaled_points):.1f}]")

# 4. 计算累积弧长
arc_lens, total_len = compute_arc_lengths(scaled_points)
print(f"total arc length: {total_len:.1f}px")

# 5. legacy赛道总长度
legacy_total_m = tm.control_points[-1][0]  # 最后一个control_point的distance_m
print(f"legacy track length: {legacy_total_m}m")

# 6. 用距离比例定位弯道
corners = sorted(tm.corners, key=lambda c: c.corner_id)
print(f"\n=== 弯道映射结果 ===")
for c in corners[:5]:
    # 用弯道中点的距离比例
    corner_mid_dist = (c.distance_start + c.distance_end) / 2
    ratio = corner_mid_dist / legacy_total_m
    mapped_pt = find_point_at_ratio(scaled_points, arc_lens, total_len, ratio)
    print(f"  C{c.corner_id:2d} ({c.name}): dist={corner_mid_dist:.0f}m, ratio={ratio:.3f}, "
          f"mapped=({mapped_pt[0]:.1f}, {mapped_pt[1]:.1f}), "
          f"legacy_px=({c.x_px}, {c.y_px})")

print(f"\n  ... (showing first 5 of {len(corners)} corners)")

# 7. 对比：所有弯道的映射坐标 vs legacy坐标
print(f"\n=== 全部弯道对比 ===")
for c in corners:
    corner_mid_dist = (c.distance_start + c.distance_end) / 2
    ratio = corner_mid_dist / legacy_total_m
    mapped_pt = find_point_at_ratio(scaled_points, arc_lens, total_len, ratio)
    dx = mapped_pt[0] - c.x_px
    dy = mapped_pt[1] - c.y_px
    dist_diff = math.sqrt(dx*dx + dy*dy)
    print(f"  C{c.corner_id:2d}: mapped=({mapped_pt[0]:6.1f}, {mapped_pt[1]:6.1f})  "
          f"legacy=({c.x_px:6.1f}, {c.y_px:6.1f})  diff={dist_diff:6.1f}px")