#!/usr/bin/env python3
"""重新处理有聚集问题的赛道"""
import sys
import re
from pathlib import Path

sys.path.insert(0, "D:/F1OPT-Test/scripts")
sys.path.insert(0, "D:/F1OPT-Test/legacy")
from convert_track_svgs import (
    process_track, generate_anchors_file,
    TRACK_LAYOUT_MAP, TRACK_CORNERS_COUNT,
    ANCHORS_FILE, OUTPUT_DIR,
)

problem_tracks = [
    "austin", "hungaroring", "mexico_city", "monaco",
    "sao_paulo", "silverstone", "spa", "suzuka",
    "yas_marina", "zandvoort",
]

results = []
for tid in problem_tracks:
    layout_id = TRACK_LAYOUT_MAP[tid]
    n = TRACK_CORNERS_COUNT[tid]
    print(f"重新处理: {tid} ({layout_id}) — {n}弯道...", end=" ")
    try:
        r = process_track(tid, layout_id, n)
        results.append(r)
        print(f"OK ({len(r['corner_points'])}/{n}弯道)")
    except Exception as e:
        print(f"FAILED: {e}")

# 从已生成的SVG中提取其他赛道的弯道数据
all_results = []
order = {tid: i for i, tid in enumerate(TRACK_LAYOUT_MAP.keys())}
for tid in TRACK_LAYOUT_MAP:
    if tid in problem_tracks:
        continue
    svg_path = OUTPUT_DIR / f"{tid}.svg"
    if not svg_path.exists():
        continue
    content = svg_path.read_text(encoding="utf-8")
    circles = re.findall(r'<circle cx="([\d.]+)" cy="([\d.]+)" r="6"', content)
    corner_points = [(float(x), float(y)) for x, y in circles]
    start_match = re.search(r'<rect x="([\d.]+)" y="([\d.]+)" width="10"', content)
    start_point = (float(start_match.group(1)) + 5, float(start_match.group(2)) + 5) if start_match else (0, 0)
    all_results.append({
        "track_id": tid, "layout_id": TRACK_LAYOUT_MAP[tid],
        "corner_points": corner_points, "start_point": start_point,
        "display_name": tid.upper(), "n_points": 0,
    })

all_results.extend(results)
all_results.sort(key=lambda r: order.get(r["track_id"], 999))

anchors_content = generate_anchors_file(all_results)
ANCHORS_FILE.write_text(anchors_content, encoding="utf-8")
print(f"\n已更新: {ANCHORS_FILE}")
print(f"总计: {len(all_results)} 条赛道")