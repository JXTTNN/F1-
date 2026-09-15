#!/usr/bin/env python3
"""用距离比例映射方案重新生成全部24条赛道的SVG和anchors文件。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_track_svgs import (
    process_track, TRACK_LAYOUT_MAP, TRACK_CORNERS_COUNT,
    OUTPUT_DIR, ANCHORS_FILE, generate_anchors_file,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results = []
failed = []

for track_id, layout_id in TRACK_LAYOUT_MAP.items():
    n_corners = TRACK_CORNERS_COUNT[track_id]
    print(f"处理: {track_id} ({layout_id}) -- {n_corners}弯道...", end=" ")

    try:
        result = process_track(track_id, layout_id, n_corners)
        results.append(result)
        n_detected = len(result["corner_points"])
        print(f"OK ({n_detected}/{n_corners}弯道)")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed.append((track_id, str(e)))

# 生成anchors文件
if results:
    anchors_content = generate_anchors_file(results)
    ANCHORS_FILE.write_text(anchors_content, encoding="utf-8")
    print(f"\n已更新: {ANCHORS_FILE}")

# 汇总
print(f"\n{'=' * 70}")
print(f"完成: {len(results)}/{len(TRACK_LAYOUT_MAP)} 条赛道")
if failed:
    print(f"失败: {len(failed)} 条")
    for tid, err in failed:
        print(f"  {tid}: {err}")
else:
    print("全部成功！")

# 验证弯道数量
print(f"\n弯道检测验证:")
all_match = True
for r in results:
    expected = TRACK_CORNERS_COUNT[r["track_id"]]
    detected = len(r["corner_points"])
    status = "OK" if detected == expected else f"MISMATCH ({detected}/{expected})"
    if detected != expected:
        all_match = False
    print(f"  {r['track_id']}: {status}")

if all_match:
    print("\n所有弯道检测数量匹配！")
else:
    print("\n部分弯道检测数量不匹配。")