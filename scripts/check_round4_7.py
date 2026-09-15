#!/usr/bin/env python3
"""第4-7轮检查：SVG文件完整性、弯道数量、坐标范围、元素完整性。"""
import sys
import re
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "setup_tuner" / "domain"))
from _track_anchors import TRACK_ANCHORS, TRACK_CANVAS
from convert_track_svgs import TRACK_CORNERS_COUNT, OUTPUT_DIR, CANVAS_W, CANVAS_H

all_ok = True

# ========== 第4轮：SVG文件XML完整性 ==========
print("=" * 60)
print("第4轮检查：SVG文件XML完整性")
print("=" * 60)
svg_dir = OUTPUT_DIR
svg_files = sorted(svg_dir.glob("*.svg"))
print(f"找到 {len(svg_files)} 个SVG文件")

for svg_path in svg_files:
    try:
        ET.parse(str(svg_path))
        print(f"  {svg_path.name}: OK")
    except ET.ParseError as e:
        print(f"  {svg_path.name}: FAILED - {e}")
        all_ok = False

# ========== 第5轮：弯道数量匹配 ==========
print("\n" + "=" * 60)
print("第5轮检查：弯道数量匹配")
print("=" * 60)
for tid in sorted(TRACK_ANCHORS.keys()):
    expected = TRACK_CORNERS_COUNT.get(tid, 0)
    actual = len(TRACK_ANCHORS[tid])
    status = "OK" if actual == expected else "MISMATCH"
    if actual != expected:
        all_ok = False
    print(f"  {tid:15s}: expected={expected:2d}, actual={actual:2d}  [{status}]")

# ========== 第6轮：弯道坐标在画布范围内 ==========
print("\n" + "=" * 60)
print("第6轮检查：弯道坐标在画布范围内")
print("=" * 60)
for tid in sorted(TRACK_ANCHORS.keys()):
    corners = TRACK_ANCHORS[tid]
    out_of_range = []
    for cn, (x, y) in corners.items():
        if x < 10 or x > 790 or y < 10 or y > 590:
            out_of_range.append((cn, x, y))
    if out_of_range:
        all_ok = False
        print(f"  {tid:15s}: OUT_OF_RANGE {out_of_range}")
    else:
        print(f"  {tid:15s}: OK (all within 10-790, 10-590)")

# ========== 第7轮：SVG元素完整性 ==========
print("\n" + "=" * 60)
print("第7轮检查：SVG元素完整性")
print("=" * 60)
for svg_path in svg_files:
    tid = svg_path.stem
    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    paths = root.findall(".//svg:path", ns)
    circles = root.findall(".//svg:circle", ns)
    texts = root.findall(".//svg:text", ns)
    rects = root.findall(".//svg:rect", ns)

    expected_corners = TRACK_CORNERS_COUNT.get(tid, 0)
    issues = []
    if len(paths) < 3:
        issues.append(f"paths={len(paths)} (<3)")
    if len(circles) != expected_corners:
        issues.append(f"circles={len(circles)} (!={expected_corners})")
    if len(texts) < expected_corners + 1:  # +1 for track name
        issues.append(f"texts={len(texts)} (<{expected_corners + 1})")
    if len(rects) < 2:  # background + start marker
        issues.append(f"rects={len(rects)} (<2)")

    if issues:
        all_ok = False
        print(f"  {tid:15s}: ISSUES {issues}")
    else:
        print(f"  {tid:15s}: OK (paths={len(paths)}, circles={len(circles)}, texts={len(texts)}, rects={len(rects)})")

# ========== 汇总 ==========
print("\n" + "=" * 60)
if all_ok:
    print("第4-7轮检查全部通过！")
else:
    print("部分检查失败，请查看上方详细信息。")