#!/usr/bin/env python3
"""检查legacy弯道坐标中的聚集问题"""
import sys
sys.path.insert(0, "D:/F1OPT-Test/legacy")
from f1opt.data.track_maps import TRACK_MAPS

# 检查legacy坐标中相邻弯道的间距
for tid in sorted(TRACK_MAPS.keys()):
    tm = TRACK_MAPS[tid]
    corners = sorted(tm.corners, key=lambda c: c.corner_id)
    close_pairs = []
    for i in range(1, len(corners)):
        dx = corners[i].x_px - corners[i-1].x_px
        dy = corners[i].y_px - corners[i-1].y_px
        dist = (dx*dx + dy*dy)**0.5
        if dist < 15:
            close_pairs.append((corners[i-1].corner_id, corners[i].corner_id, dist))
    if close_pairs:
        print(f"{tid:15s} legacy聚集:")
        for a, b, d in close_pairs:
            print(f"  弯{a}→弯{b}: {d:.1f}px")