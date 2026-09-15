#!/usr/bin/env python3
"""重新处理6条需要更新的赛道：mexico_city, monaco, silverstone, spa, singapore。
同时更新SVG和anchors文件。
"""
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_track_svgs import (
    process_track, TRACK_LAYOUT_MAP, TRACK_CORNERS_COUNT,
    OUTPUT_DIR, ANCHORS_FILE,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 需要重新处理的赛道
tracks_to_reprocess = ['mexico_city', 'monaco', 'silverstone', 'spa', 'singapore']

# 处理赛道
results = []
for tid in tracks_to_reprocess:
    layout_id = TRACK_LAYOUT_MAP[tid]
    n_corners = TRACK_CORNERS_COUNT[tid]
    print(f"处理: {tid} ({layout_id}) -- {n_corners}弯道...", end=" ")
    try:
        result = process_track(tid, layout_id, n_corners)
        results.append(result)
        n_detected = len(result["corner_points"])
        print(f"OK ({n_detected}/{n_corners}弯道)")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

# 读取现有的anchors文件，解析出其他赛道的数据
existing = ANCHORS_FILE.read_text(encoding="utf-8")

# 解析 TRACK_CANVAS
canvas_info = {}
canvas_re = re.compile(r'"(\w+)":\s*\((\d+),\s*(\d+)\)')
for m in canvas_re.finditer(existing):
    track_id = m.group(1)
    w = int(m.group(2))
    h = int(m.group(3))
    canvas_info[track_id] = (w, h)

# 解析 TRACK_ANCHORS
track_anchors = {}
track_block_re = re.compile(r'"(\w+)":\s*\{([^}]+)\}')
corner_re = re.compile(r'(\d+):\s*\(([\d.]+),\s*([\d.]+)\)')

for m in track_block_re.finditer(existing):
    track_id = m.group(1)
    block = m.group(2)
    corners = {}
    for cm in corner_re.finditer(block):
        corner_num = int(cm.group(1))
        x = float(cm.group(2))
        y = float(cm.group(3))
        corners[corner_num] = (x, y)
    track_anchors[track_id] = corners

print(f"\n从现有anchors文件解析出 {len(track_anchors)} 条赛道")

# 用新数据替换旧数据
for r in results:
    tid = r["track_id"]
    corners = {}
    for i, (cx, cy) in enumerate(r["corner_points"], 1):
        corners[i] = (cx, cy)
    track_anchors[tid] = corners
    canvas_info[tid] = (800, 600)
    print(f"  更新 {tid}: {len(corners)}弯道")

# 重新生成anchors文件
lines = []
lines.append('"""从 julesr0y/f1-circuits-svg 真实赛道SVG提取的弯道像素坐标。')
lines.append("")
lines.append("由 scripts/convert_track_svgs.py 自动生成，请勿手工编辑。")
lines.append('"""')
lines.append("")
lines.append("from __future__ import annotations")
lines.append("")
lines.append("# (canvas_width, canvas_height)")
lines.append("TRACK_CANVAS: dict[str, tuple[int, int]] = {")
for tid in sorted(canvas_info.keys()):
    w, h = canvas_info[tid]
    lines.append(f'    "{tid}": ({w}, {h}),')
lines.append("}")
lines.append("")
lines.append("# {track_id: {corner_number: (x_px, y_px)}}")
lines.append("TRACK_ANCHORS: dict[str, dict[int, tuple[float, float]]] = {")
for tid in sorted(track_anchors.keys()):
    corners = track_anchors[tid]
    lines.append(f'    "{tid}": {{')
    for cn in sorted(corners.keys()):
        cx, cy = corners[cn]
        lines.append(f'        {cn}: ({cx:.1f}, {cy:.1f}),')
    lines.append("    },")
lines.append("}")
lines.append("")

anchors_content = "\n".join(lines)
ANCHORS_FILE.write_text(anchors_content, encoding="utf-8")
print(f"\n已更新: {ANCHORS_FILE}")
print(f"总赛道数: {len(track_anchors)}")

# 验证更新后的6条赛道弯道间距
print(f"\n验证更新后的弯道间距:")
for r in results:
    tid = r["track_id"]
    corners = track_anchors[tid]
    min_d = float('inf')
    min_pair = None
    for i in sorted(corners.keys()):
        for j in sorted(corners.keys()):
            if j <= i:
                continue
            dx = corners[i][0] - corners[j][0]
            dy = corners[i][1] - corners[j][1]
            d = (dx*dx + dy*dy)**0.5
            if d < min_d:
                min_d = d
                min_pair = (i, j)
    status = "OK" if min_d >= 15.0 else "CLUSTER"
    print(f"  {tid:15s}: min_dist={min_d:6.1f}px (C{min_pair[0]}-C{min_pair[1]})  [{status}]")