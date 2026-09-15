#!/usr/bin/env python3
"""处理剩余4条赛道并生成完整的_track_anchors.py"""

import sys
sys.path.insert(0, "D:/F1OPT-Test/scripts")
from convert_track_svgs import (
    TRACK_LAYOUT_MAP, TRACK_CORNERS_COUNT, TRACK_DISPLAY_NAMES,
    OUTPUT_DIR, ANCHORS_FILE, CANVAS_W, CANVAS_H,
    process_track, generate_anchors_file,
)
from pathlib import Path
import json

REMAINING = ["sao_paulo", "las_vegas", "lusail", "yas_marina"]

# 读取已成功赛道的结果（从SVG文件中提取弯道信息）
# 由于anchors文件还没生成，我们需要重新处理所有赛道
# 但为了节省时间，只处理剩余4条，然后从已有SVG中提取已处理的弯道数据

def extract_corners_from_svg(svg_path: Path) -> list[tuple[float, float]]:
    """从已生成的SVG文件中提取弯道圆点坐标"""
    content = svg_path.read_text(encoding="utf-8")
    import re
    # 匹配 <circle cx="..." cy="..." r="6" ...>
    pattern = r'<circle cx="([\d.]+)" cy="([\d.]+)" r="6"'
    matches = re.findall(pattern, content)
    return [(float(x), float(y)) for x, y in matches]

def main():
    print("=" * 70)
    print("处理剩余赛道 + 生成完整anchors文件")
    print("=" * 70)

    results = []

    # 1. 从已生成的SVG中提取已成功赛道的弯道数据
    all_track_ids = list(TRACK_LAYOUT_MAP.keys())
    for track_id in all_track_ids:
        if track_id in REMAINING:
            continue
        svg_path = OUTPUT_DIR / f"{track_id}.svg"
        if not svg_path.exists():
            print(f"  跳过 {track_id}: SVG文件不存在")
            continue
        corner_points = extract_corners_from_svg(svg_path)
        n_expected = TRACK_CORNERS_COUNT[track_id]
        if len(corner_points) != n_expected:
            print(f"  警告 {track_id}: 弯道数 {len(corner_points)}/{n_expected}")
        # 提取起点
        import re
        content = svg_path.read_text(encoding="utf-8")
        start_match = re.search(r'<rect x="([\d.]+)" y="([\d.]+)" width="10"', content)
        start_point = (float(start_match.group(1)) + 5, float(start_match.group(2)) + 5) if start_match else (0, 0)

        results.append({
            "track_id": track_id,
            "layout_id": TRACK_LAYOUT_MAP[track_id],
            "corner_points": corner_points,
            "start_point": start_point,
            "display_name": TRACK_DISPLAY_NAMES.get(track_id, track_id.upper()),
            "n_points": 0,
        })
        print(f"  已提取 {track_id}: {len(corner_points)}/{n_expected}弯道")

    # 2. 处理剩余4条赛道
    print(f"\n处理剩余 {len(REMAINING)} 条赛道:")
    for track_id in REMAINING:
        layout_id = TRACK_LAYOUT_MAP[track_id]
        n_corners = TRACK_CORNERS_COUNT[track_id]
        print(f"\n  {track_id} ({layout_id}) — {n_corners}弯道...", end=" ")
        try:
            result = process_track(track_id, layout_id, n_corners)
            results.append(result)
            n_detected = len(result["corner_points"])
            print(f"OK ({n_detected}/{n_corners}弯道)")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    # 3. 生成anchors文件
    print(f"\n生成anchors文件...")
    # 按TRACK_LAYOUT_MAP的顺序排序
    order = {tid: i for i, tid in enumerate(TRACK_LAYOUT_MAP.keys())}
    results.sort(key=lambda r: order.get(r["track_id"], 999))

    anchors_content = generate_anchors_file(results)
    ANCHORS_FILE.write_text(anchors_content, encoding="utf-8")
    print(f"已更新: {ANCHORS_FILE}")

    # 4. 验证
    print(f"\n{'=' * 70}")
    print("弯道检测验证:")
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

    print(f"\n总计: {len(results)}/{len(TRACK_LAYOUT_MAP)} 条赛道")


if __name__ == "__main__":
    main()