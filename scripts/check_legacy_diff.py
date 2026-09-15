#!/usr/bin/env python3
"""检查legacy弯道坐标与补充弯道的具体问题"""
import sys
sys.path.insert(0, "D:/F1OPT-Test/legacy")
sys.path.insert(0, "D:/F1OPT-Test/scripts")
from f1opt.data.track_maps import TRACK_MAPS
from convert_track_svgs import TRACK_CORNERS_COUNT

print("Legacy弯道数 vs 预期弯道数:")
for tid in sorted(TRACK_CORNERS_COUNT.keys()):
    legacy_n = len(TRACK_MAPS[tid].corners) if tid in TRACK_MAPS else 0
    expected = TRACK_CORNERS_COUNT[tid]
    diff = expected - legacy_n
    status = "OK" if diff == 0 else f"缺{diff}弯"
    print(f"  {tid:15s} legacy={legacy_n:2d} expected={expected:2d} [{status}]")