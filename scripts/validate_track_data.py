"""校验赛道图 / 锚点 / 弧长三层数据一致性（只读，不修改任何数据）。

三层同源资产由 ``scripts/gen_real_tracks.py`` 从真实 GPS 几何一次性重建：

1. ``setup_tuner/ui/tracks/<id>.svg``        可见的赛道图与弯道标号
2. ``setup_tuner/domain/_track_anchors.py``  锚点像素坐标（热区 / 接口层）
3. ``setup_tuner/domain/_track_arcs.py``     锚点弧长占比（与遥测圈距同口径）

本脚本**只做检查**：官方弯数 ↔ 锚点 ↔ 弧长是否一一对应、弧长是否单调、
SVG 是否齐全、训练数据文件是否存在。**不修改赛道图或弯道标号**
（用户明确要求：再次检查但不要轻易改）。

用法::

    python scripts/validate_track_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from setup_tuner.domain._track_anchors import TRACK_ANCHORS  # noqa: E402
from setup_tuner.domain._track_arcs import TRACK_CORNER_ARCS  # noqa: E402
from setup_tuner.domain._track_official import OFFICIAL_TURN_COUNTS  # noqa: E402

#: 必须存在的训练数据文件（缺失只警告，不算错误）
DATA_FILES = ("merged_dataset.json", "corner_dataset.jsonl")


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for tid, counts in OFFICIAL_TURN_COUNTS.items():
        official_n = counts[0]
        anchors = TRACK_ANCHORS.get(tid, {})
        arcs = TRACK_CORNER_ARCS.get(tid, {})
        if len(anchors) != official_n:
            errors.append(f"{tid}: 锚点数 {len(anchors)} != 官方弯数 {official_n}")
        if len(arcs) != official_n:
            errors.append(f"{tid}: 弧长数 {len(arcs)} != 官方弯数 {official_n}")
        if set(anchors) != set(arcs):
            errors.append(f"{tid}: 锚点弯号集合与弧长弯号集合不一致")
        ordered = [arcs[i] for i in sorted(arcs)]
        if any(ordered[i] >= ordered[i + 1] for i in range(len(ordered) - 1)):
            errors.append(f"{tid}: 弧长占比未随弯号单调递增")
        if ordered and not (0.0 <= ordered[0] and ordered[-1] < 1.0):
            errors.append(f"{tid}: 弧长占比越界 [0, 1)")

    svg_dir = REPO / "setup_tuner" / "ui" / "tracks"
    for tid in OFFICIAL_TURN_COUNTS:
        if not (svg_dir / f"{tid}.svg").exists():
            errors.append(f"{tid}: SVG 缺失 {svg_dir / f'{tid}.svg'}")

    data_dir = REPO / "data" / "training"
    for name in DATA_FILES:
        if not (data_dir / name).exists():
            warnings.append(f"训练数据缺失: {name}")

    print("=== 赛道图 / 锚点 / 弧长 校验 ===")
    print(f"赛道数: {len(OFFICIAL_TURN_COUNTS)}")
    print(f"错误: {len(errors)}")
    for e in errors:
        print("  [x]", e)
    print(f"警告: {len(warnings)}")
    for w in warnings:
        print("  [!]", w)
    if not errors:
        print("[OK] 赛道图与核心数据通过校验（未修改任何弯道标号）")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
