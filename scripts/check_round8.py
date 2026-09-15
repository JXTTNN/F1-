#!/usr/bin/env python3
"""第8轮检查：验证_track_anchors.py中的弯道坐标与SVG文件中的circle坐标一致。"""
import sys
import re
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "setup_tuner" / "domain"))
from _track_anchors import TRACK_ANCHORS
from convert_track_svgs import OUTPUT_DIR

svg_dir = OUTPUT_DIR
all_ok = True

for tid in sorted(TRACK_ANCHORS.keys()):
    svg_path = svg_dir / f"{tid}.svg"
    if not svg_path.exists():
        print(f"  {tid:15s}: SVG FILE NOT FOUND")
        all_ok = False
        continue

    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}
    circles = root.findall(".//svg:circle", ns)

    # 从SVG中提取circle坐标
    svg_corners = {}
    for circle in circles:
        cx = float(circle.get("cx", "0"))
        cy = float(circle.get("cy", "0"))
        # circle的顺序就是弯道编号（从1开始）
        idx = len(svg_corners) + 1
        svg_corners[idx] = (cx, cy)

    # 比较anchors和SVG中的坐标
    anchors = TRACK_ANCHORS[tid]
    mismatches = []
    for cn in sorted(anchors.keys()):
        if cn not in svg_corners:
            mismatches.append((cn, "missing in SVG"))
            continue
        ax, ay = anchors[cn]
        sx, sy = svg_corners[cn]
        # 允许小数精度差异（anchors文件保留1位小数）
        if abs(ax - sx) > 0.2 or abs(ay - sy) > 0.2:
            mismatches.append((cn, f"anchor=({ax:.1f},{ay:.1f}) svg=({sx:.1f},{sy:.1f})"))

    if mismatches:
        all_ok = False
        print(f"  {tid:15s}: MISMATCH {mismatches}")
    else:
        print(f"  {tid:15s}: OK ({len(anchors)} corners match)")

print(f"\n{'=' * 60}")
if all_ok:
    print("第8轮检查通过！anchors与SVG坐标完全一致。")
else:
    print("第8轮检查失败！部分坐标不一致。")