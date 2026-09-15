#!/usr/bin/env python3
"""第10轮检查：端到端验证——赛道图加载和弯道锚点归一化。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "setup_tuner"))
from domain._track_anchors import TRACK_ANCHORS, TRACK_CANVAS
from domain.track import ALL_TRACKS, _apply_real_anchors

all_ok = True

# 构建track_id -> Track映射
track_map = {t.track_id: t for t in ALL_TRACKS}

for tid in sorted(track_map.keys()):
    track = track_map[tid]
    expected_corners = len(TRACK_ANCHORS.get(tid, {}))

    # 验证anchors数据存在
    if tid not in TRACK_ANCHORS:
        print(f"  {tid:15s}: NO ANCHORS DATA")
        all_ok = False
        continue

    # 验证canvas尺寸
    canvas = TRACK_CANVAS.get(tid, (800, 600))
    if canvas != (800, 600):
        print(f"  {tid:15s}: CANVAS SIZE MISMATCH {canvas}")
        all_ok = False
        continue

    # 验证锚点归一化：用_apply_real_anchors处理Track的corners
    normalized_corners = _apply_real_anchors(tid, track.corners)

    if not normalized_corners:
        print(f"  {tid:15s}: NORMALIZATION FAILED")
        all_ok = False
        continue

    # 验证归一化后的锚点数量
    if len(normalized_corners) != len(track.corners):
        print(f"  {tid:15s}: COUNT MISMATCH ({len(normalized_corners)}/{len(track.corners)})")
        all_ok = False
        continue

    # 验证归一化后的坐标在0-1范围内
    out_of_range = []
    for c in normalized_corners:
        ax = c.anchor.anchor_x
        ay = c.anchor.anchor_y
        if ax <= 0 or ax >= 1 or ay <= 0 or ay >= 1:
            out_of_range.append((c.number, ax, ay))

    if out_of_range:
        all_ok = False
        print(f"  {tid:15s}: OUT OF RANGE {out_of_range}")
    else:
        print(f"  {tid:15s}: OK ({len(normalized_corners)} corners normalized)")

# 验证SVG文件能被正确读取
print(f"\n--- SVG文件加载验证 ---")
svg_dir = Path(__file__).resolve().parent.parent / "setup_tuner" / "ui" / "tracks"
for tid in sorted(track_map.keys()):
    svg_path = svg_dir / f"{tid}.svg"
    if not svg_path.exists():
        print(f"  {tid:15s}: SVG NOT FOUND")
        all_ok = False
        continue
    content = svg_path.read_text(encoding="utf-8")
    if len(content) < 100:
        print(f"  {tid:15s}: SVG TOO SMALL ({len(content)} bytes)")
        all_ok = False
        continue
    if "<svg" not in content or "</svg>" not in content:
        print(f"  {tid:15s}: SVG MALFORMED")
        all_ok = False
        continue
    print(f"  {tid:15s}: OK ({len(content)} bytes)")

print(f"\n{'=' * 60}")
if all_ok:
    print("第10轮检查通过！赛道图加载和锚点归一化全部正常。")
else:
    print("第10轮检查失败！部分验证未通过。")
