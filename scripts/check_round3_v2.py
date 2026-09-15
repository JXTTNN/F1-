#!/usr/bin/env python3
"""第3轮检查：验证所有24条赛道的弯道坐标分布，确保没有聚集问题。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "setup_tuner" / "domain"))
from _track_anchors import TRACK_ANCHORS, TRACK_CANVAS

MIN_SPACING = 15.0  # 弯道之间最小间距(px)

all_ok = True
problems = []

for tid in sorted(TRACK_ANCHORS.keys()):
    corners = TRACK_ANCHORS[tid]
    n = len(corners)
    if n < 2:
        continue
    
    # 检查所有弯道对之间的距离
    min_d = float('inf')
    min_pair = None
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if i not in corners or j not in corners:
                continue
            dx = corners[i][0] - corners[j][0]
            dy = corners[i][1] - corners[j][1]
            d = (dx * dx + dy * dy) ** 0.5
            if d < min_d:
                min_d = d
                min_pair = (i, j)
    
    status = "OK" if min_d >= MIN_SPACING else "CLUSTER"
    if min_d < MIN_SPACING:
        all_ok = False
        problems.append((tid, min_pair, min_d))
    
    print(f"  {tid:15s}: {n:2d} corners, min_dist={min_d:6.1f}px (C{min_pair[0]}-C{min_pair[1]})  [{status}]")

print(f"\n{'=' * 60}")
if all_ok:
    print(f"第3轮检查通过！所有24条赛道弯道间距 >= {MIN_SPACING}px")
else:
    print(f"第3轮检查失败！{len(problems)}条赛道仍有聚集问题：")
    for tid, pair, d in problems:
        print(f"  {tid}: C{pair[0]}-C{pair[1]} 间距={d:.1f}px < {MIN_SPACING}px")