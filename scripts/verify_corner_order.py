"""检查弯道编号顺序是否沿赛道行驶方向递增。

原理：
1. 从SVG path中提取赛道线采样点
2. 从SVG中提取弯道圆圈坐标
3. 将弯道按编号排序，计算每个弯道在赛道线上的最近点索引
4. 检查索引是否单调递增（允许小范围回绕）
"""
import math
import re
import sys
from pathlib import Path

# 仓库根目录（本脚本位于 <root>/scripts/），避免硬编码绝对路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_track_svgs import (
    CANVAS_H,
    CANVAS_W,
    MARGIN,
    extract_path_d,
    path_to_points,
    scale_points,
)

TRACK_DIR = ROOT / "setup_tuner" / "ui" / "tracks"
RAW_DIR = ROOT / "scripts" / "raw_svgs"

# layout映射
LAYOUT_MAP = {
    "melbourne": "melbourne-2", "shanghai": "shanghai-1", "suzuka": "suzuka-2",
    "sakhir": "bahrain-1", "jeddah": "jeddah-1", "miami": "miami-1",
    "montreal": "montreal-6", "monaco": "monaco-6", "barcelona": "catalunya-6",
    "spielberg": "spielberg-3", "silverstone": "silverstone-8",
    "spa": "spa-francorchamps-4", "hungaroring": "hungaroring-3",
    "zandvoort": "zandvoort-5", "monza": "monza-7", "madrid": "madring-1",
    "baku": "baku-1", "singapore": "marina-bay-4", "austin": "austin-1",
    "mexico_city": "mexico-city-3", "sao_paulo": "interlagos-2",
    "las_vegas": "las-vegas-1", "lusail": "lusail-1", "yas_marina": "yas-marina-2",
}


def extract_corners_from_svg(svg_path):
    """从F1OPT SVG中提取弯道圆圈坐标和编号。"""
    content = svg_path.read_text(encoding="utf-8")
    # 匹配 circle + 紧随其后的 text
    pattern = r'<circle cx="([\d.]+)" cy="([\d.]+)" r="6"[^>]*/>\s*<text[^>]*>(\d+)</text>'
    matches = re.findall(pattern, content)
    corners = []
    for cx, cy, num in matches:
        corners.append((int(num), float(cx), float(cy)))
    return corners


def find_nearest_index(pt, track_points):
    """找到pt在track_points中最近点的索引。"""
    min_d = float("inf")
    min_i = 0
    for i, tp in enumerate(track_points):
        d = (pt[0] - tp[0]) ** 2 + (pt[1] - tp[1]) ** 2
        if d < min_d:
            min_d = d
            min_i = i
    return min_i, math.sqrt(min_d)


def check_track(track_id):
    """检查单条赛道的弯道编号顺序。"""
    # 从原始SVG提取赛道线
    layout = LAYOUT_MAP[track_id]
    raw_svg = RAW_DIR / f"{layout}.svg"
    raw_content = raw_svg.read_text(encoding="utf-8")
    raw_d = extract_path_d(raw_content)
    raw_points = path_to_points(raw_d)
    track_points = scale_points(raw_points, CANVAS_W, CANVAS_H, MARGIN)

    # 从F1OPT SVG提取弯道
    f1opt_svg = TRACK_DIR / f"{track_id}.svg"
    corners = extract_corners_from_svg(f1opt_svg)

    # 按编号排序
    corners.sort(key=lambda c: c[0])

    # 计算每个弯道在赛道线上的索引
    indices = []
    for num, cx, cy in corners:
        idx, dist = find_nearest_index((cx, cy), track_points)
        indices.append((num, idx, dist))

    # 检查索引是否单调递增（允许回绕）
    n = len(indices)
    violations = []
    for i in range(1, n):
        prev_idx = indices[i - 1][1]
        curr_idx = indices[i][1]
        if curr_idx <= prev_idx:
            # 检查是否是回绕（赛道是环形）
            track_len = len(track_points)
            if curr_idx < track_len * 0.1 and prev_idx > track_len * 0.9:
                pass  # 回绕，OK
            else:
                violations.append(
                    f"  弯道{indices[i][0]} idx={curr_idx} <= 弯道{indices[i-1][0]} idx={prev_idx}"
                )

    max_dist = max(d for _, _, d in indices)
    status = "OK" if not violations else "ORDER_ISSUE"
    return track_id, n, max_dist, violations, status


# 检查所有24条赛道
print("=" * 70)
print("弯道编号顺序验证（沿赛道行驶方向递增）")
print("=" * 70)

all_ok = True
for tid in sorted(LAYOUT_MAP.keys()):
    track_id, n, max_dist, violations, status = check_track(tid)
    if status != "OK":
        all_ok = False
    print(f"  {track_id:15s} corners={n:2d} max_dist={max_dist:4.1f}px [{status}]")
    for v in violations:
        print(v)

print(f"\n{'=' * 70}")
if all_ok:
    print("✅ 全部通过！弯道编号沿赛道方向单调递增。")
else:
    print("⚠️ 部分赛道弯道编号顺序有问题。")
