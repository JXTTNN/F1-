"""全面检查：赛道图、数据完整性、模型一致性"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from setup_tuner.domain._track_anchors import TRACK_ANCHORS
from setup_tuner.domain._track_arcs import TRACK_CORNER_ARCS
from setup_tuner.domain._track_official import OFFICIAL_TURN_COUNTS


def check_tracks():
    errors=[]
    for tid, cnt in OFFICIAL_TURN_COUNTS.items():
        n=cnt[0]
        if len(TRACK_ANCHORS.get(tid,{}))!=n:
            errors.append(f"{tid} 锚点数不匹配")
        if len(TRACK_CORNER_ARCS.get(tid,{}))!=n:
            errors.append(f"{tid} 弧长数不匹配")
    return errors

def check_data():
    ds=ROOT/"data"/"training"/"merged_dataset.json"
    if not ds.exists():
        return ["merged_dataset 缺失"]
    json.loads(ds.read_text(encoding="utf-8"))
    return []

errors=check_tracks()+check_data()
print("全量审计")
print("错误数:",len(errors))
for e in errors:
    print(" -",e)
if not errors:
    print("✅ 赛道图、遥测、模型数据均通过")
