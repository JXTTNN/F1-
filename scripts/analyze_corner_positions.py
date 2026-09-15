#!/usr/bin/env python3
"""分析赛道SVG中弯道标注位置是否正确"""
import re
from pathlib import Path

track_dir = Path("D:/F1OPT-Test/setup_tuner/ui/tracks")

for track_id in ["melbourne", "monaco", "suzuka", "jeddah"]:
    svg_path = track_dir / f"{track_id}.svg"
    content = svg_path.read_text(encoding="utf-8")
    
    # 提取弯道圆点坐标
    circles = re.findall(r'<circle cx="([\d.]+)" cy="([\d.]+)"', content)
    
    print(f"\n=== {track_id} ({len(circles)}个弯道) ===")
    
    # 检查相邻弯道间距
    min_dist = 999
    max_dist = 0
    close_pairs = []
    for i in range(1, len(circles)):
        dx = float(circles[i][0]) - float(circles[i-1][0])
        dy = float(circles[i][1]) - float(circles[i-1][1])
        dist = (dx**2 + dy**2)**0.5
        if dist < 20:
            close_pairs.append((i, i+1, dist))
        min_dist = min(min_dist, dist)
        max_dist = max(max_dist, dist)
    
    print(f"  相邻弯道间距: min={min_dist:.1f}px, max={max_dist:.1f}px")
    if close_pairs:
        print(f"  ⚠ {len(close_pairs)}对弯道间距过近(<20px):")
        for a, b, d in close_pairs:
            print(f"    弯{a}→弯{b}: {d:.1f}px ← 聚集!")
    else:
        print(f"  ✓ 无聚集")
    
    # 打印所有弯道坐标
    for i, (x, y) in enumerate(circles, 1):
        print(f"  弯{i}: ({x}, {y})")