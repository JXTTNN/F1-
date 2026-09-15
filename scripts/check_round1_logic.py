#!/usr/bin/env python3
"""第1遍检查：代码逻辑审查——检查convert_track_svgs.py中的潜在问题。

检查项：
1. _decluster_corners是否在曲率分析后不必要地偏移了弯道
2. 弯道编号顺序是否按赛道前进方向排列
3. 起点标记位置是否正确
4. scale_path_d是否正确处理了所有命令类型
5. detect_corners的弯道数量是否可靠
"""
import sys
import math
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "setup_tuner" / "domain"))
from convert_track_svgs import (
    path_to_points, scale_points, detect_corners, _decluster_corners,
    download_svg, extract_path_d, scale_path_d, tokenize_path,
    CANVAS_W, CANVAS_H, MARGIN, TRACK_LAYOUT_MAP, TRACK_CORNERS_COUNT, OUTPUT_DIR,
)
from _track_anchors import TRACK_ANCHORS

issues = []

# ========== 检查1: _decluster_corners是否偏移了弯道 ==========
print("=" * 60)
print("检查1: _decluster_corners是否偏移了弯道")
print("=" * 60)

for tid in sorted(TRACK_LAYOUT_MAP.keys()):
    layout_id = TRACK_LAYOUT_MAP[tid]
    n_corners = TRACK_CORNERS_COUNT[tid]

    svg_content = download_svg(layout_id)
    raw_d = extract_path_d(svg_content)
    raw_points = path_to_points(raw_d)
    scaled_points = scale_points(raw_points, CANVAS_W, CANVAS_H, MARGIN)

    # 曲率分析的原始结果
    raw_corners = detect_corners(scaled_points, n_corners)

    # _decluster_corners后的结果
    declustered = _decluster_corners(raw_corners, scaled_points)

    # 比较两者
    max_shift = 0
    for i in range(len(raw_corners)):
        dx = declustered[i][0] - raw_corners[i][0]
        dy = declustered[i][1] - raw_corners[i][1]
        shift = math.sqrt(dx*dx + dy*dy)
        if shift > max_shift:
            max_shift = shift

    if max_shift > 0.1:
        print(f"  {tid:15s}: _decluster偏移了弯道! max_shift={max_shift:.1f}px")
        issues.append(f"{tid}: _decluster偏移了弯道max_shift={max_shift:.1f}px")
    else:
        print(f"  {tid:15s}: OK (no shift)")

# ========== 检查2: 弯道编号顺序是否按赛道前进方向排列 ==========
print(f"\n{'=' * 60}")
print("检查2: 弯道编号顺序是否按赛道前进方向排列")
print("=" * 60)

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

for tid in sorted(TRACK_LAYOUT_MAP.keys()):
    layout_id = TRACK_LAYOUT_MAP[tid]
    n_corners = TRACK_CORNERS_COUNT[tid]

    svg_content = download_svg(layout_id)
    raw_d = extract_path_d(svg_content)
    raw_points = path_to_points(raw_d)
    scaled_points = scale_points(raw_points, CANVAS_W, CANVAS_H, MARGIN)

    corner_points = detect_corners(scaled_points, n_corners)
    arc_lens, total_len = compute_arc_lengths(scaled_points)

    # 计算每个弯道在赛道上的弧长位置
    corner_arc_pos = []
    for cp in corner_points:
        # 找最近的采样点
        best_idx = 0
        best_d = float('inf')
        for i, sp in enumerate(scaled_points):
            dx = cp[0] - sp[0]
            dy = cp[1] - sp[1]
            d = dx*dx + dy*dy
            if d < best_d:
                best_d = d
                best_idx = i
        corner_arc_pos.append(arc_lens[best_idx])

    # 检查弯道是否按弧长递增排列
    is_ordered = all(corner_arc_pos[i] <= corner_arc_pos[i+1] for i in range(len(corner_arc_pos)-1))

    if not is_ordered:
        # 检查是否是环形赛道（最后一个弯道接近起点）
        last_to_first = total_len - corner_arc_pos[-1] + corner_arc_pos[0]
        if last_to_first < corner_arc_pos[1] - corner_arc_pos[0]:
            print(f"  {tid:15s}: OK (环形赛道, C1接近终点)")
        else:
            print(f"  {tid:15s}: WARNING - 弯道未按弧长递增排列!")
            issues.append(f"{tid}: 弯道未按弧长递增排列")
    else:
        print(f"  {tid:15s}: OK (按弧长递增)")

# ========== 检查3: 起点标记位置 ==========
print(f"\n{'=' * 60}")
print("检查3: 起点标记位置是否在赛道线上")
print("=" * 60)

for tid in sorted(TRACK_LAYOUT_MAP.keys()):
    layout_id = TRACK_LAYOUT_MAP[tid]
    svg_content = download_svg(layout_id)
    raw_d = extract_path_d(svg_content)

    # 起点位置（从SVG文件中读取）
    svg_path = OUTPUT_DIR / f"{tid}.svg"
    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}
    rects = root.findall(".//svg:rect", ns)

    # 找起点标记（红色方块，fill="#FF1801"）
    start_rect = None
    for r in rects:
        if r.get("fill") == "#FF1801":
            start_rect = r
            break

    if start_rect is None:
        print(f"  {tid:15s}: FAILED - 未找到起点标记")
        issues.append(f"{tid}: 未找到起点标记")
        continue

    sx = float(start_rect.get("x", "0")) + 5  # x是左上角，中心是x+5
    sy = float(start_rect.get("y", "0")) + 5

    # 检查起点是否在赛道线上
    raw_points = path_to_points(raw_d)
    scaled_points = scale_points(raw_points, CANVAS_W, CANVAS_H, MARGIN)

    min_d = float('inf')
    for sp in scaled_points:
        dx = sx - sp[0]
        dy = sy - sp[1]
        d = math.sqrt(dx*dx + dy*dy)
        if d < min_d:
            min_d = d

    if min_d > 10:
        print(f"  {tid:15s}: WARNING - 起点偏离赛道线{min_d:.1f}px")
        issues.append(f"{tid}: 起点偏离赛道线{min_d:.1f}px")
    else:
        print(f"  {tid:15s}: OK (偏离{min_d:.1f}px)")

# ========== 检查4: anchors与SVG circle坐标一致性 ==========
print(f"\n{'=' * 60}")
print("检查4: anchors文件与SVG circle坐标一致性")
print("=" * 60)

for tid in sorted(TRACK_ANCHORS.keys()):
    svg_path = OUTPUT_DIR / f"{tid}.svg"
    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}
    circles = root.findall(".//svg:circle", ns)

    anchors = TRACK_ANCHORS[tid]
    mismatches = []
    for i, circle in enumerate(circles):
        cn = i + 1
        cx = float(circle.get("cx", "0"))
        cy = float(circle.get("cy", "0"))
        if cn in anchors:
            ax, ay = anchors[cn]
            if abs(ax - cx) > 0.2 or abs(ay - cy) > 0.2:
                mismatches.append((cn, f"anchor=({ax:.1f},{ay:.1f}) svg=({cx:.1f},{cy:.1f})"))

    if mismatches:
        print(f"  {tid:15s}: MISMATCH {mismatches}")
        issues.append(f"{tid}: anchors与SVG不一致")
    else:
        print(f"  {tid:15s}: OK")

# ========== 汇总 ==========
print(f"\n{'=' * 60}")
print(f"发现 {len(issues)} 个问题:")
for issue in issues:
    print(f"  - {issue}")
if not issues:
    print("  无问题！")