#!/usr/bin/env python3
"""第3轮：弯道坐标分布验证——检查聚集、越界、间距"""
import re
from pathlib import Path

track_dir = Path("D:/F1OPT-Test/setup_tuner/ui/tracks")
expected = {
    "melbourne": 14, "shanghai": 16, "suzuka": 18, "sakhir": 15,
    "jeddah": 27, "miami": 19, "montreal": 14, "monaco": 19,
    "barcelona": 14, "spielberg": 10, "silverstone": 18, "spa": 19,
    "hungaroring": 14, "zandvoort": 14, "monza": 11, "madrid": 22,
    "baku": 20, "singapore": 19, "austin": 20, "mexico_city": 17,
    "sao_paulo": 15, "las_vegas": 17, "lusail": 16, "yas_marina": 16,
}

all_ok = True
for tid, n_exp in sorted(expected.items()):
    svg = track_dir / f"{tid}.svg"
    content = svg.read_text(encoding="utf-8")
    circles = re.findall(r'<circle cx="([\d.]+)" cy="([\d.]+)" r="6"', content)
    n = len(circles)
    pts = [(float(x), float(y)) for x, y in circles]
    
    issues = []
    
    # 1. 数量匹配
    if n != n_exp:
        issues.append(f"数量{n}/{n_exp}")
    
    # 2. 越界检查（0-800, 0-600）
    for i, (x, y) in enumerate(pts, 1):
        if x < 0 or x > 800 or y < 0 or y > 600:
            issues.append(f"弯{i}越界({x:.0f},{y:.0f})")
    
    # 3. 聚集检查（相邻间距<15px）
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        dist = (dx*dx + dy*dy)**0.5
        if dist < 15:
            issues.append(f"弯{i}→{i+1}聚集({dist:.1f}px)")
    
    # 4. 重复坐标检查
    seen = {}
    for i, (x, y) in enumerate(pts, 1):
        key = (round(x, 1), round(y, 1))
        if key in seen:
            issues.append(f"弯{i}与弯{seen[key]}重复")
        seen[key] = i
    
    status = "OK" if not issues else "FAIL: " + "; ".join(issues)
    if issues:
        all_ok = False
    print(f"  {tid:15s} {n:2d}/{n_exp:2d} [{status}]")

print()
print("全部OK!" if all_ok else "有问题!")