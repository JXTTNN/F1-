"""从 pits-n-giggles 项目下载真实F1 25赛道SVG，转换为我们的格式。

改进版 v2：
1. 使用真实F1 25赛道polyline数据（来自 pits-n-giggles）
2. 使用legacy track_maps中的弯道距离信息精确定位弯道位置
3. 将弯道距离(米)映射到polyline上的弧长比例位置
4. 生成我们的SVG格式（深色背景 + 弯道圆圈 + 起点红块 + 赛道名称）
5. 导出 _track_anchors.py

输出：setup_tuner/ui/tracks/<track_id>.svg + setup_tuner/domain/_track_anchors.py
"""
from __future__ import annotations

import math
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "legacy"))

OUT_DIR = ROOT / "setup_tuner" / "ui" / "tracks"
ANCHORS_PATH = ROOT / "setup_tuner" / "domain" / "_track_anchors.py"

# pits-n-giggles 文件名 → 我们的 track_id
TRACK_FILE_MAP = {
    "Melbourne": "melbourne",
    "Shanghai": "shanghai",
    "Sakhir_Bahrain": "sakhir",
    "Jeddah": "jeddah",
    "Miami": "miami",
    "Texas": "austin",
    "Monaco": "monaco",
    "Montreal": "montreal",
    "Catalunya": "barcelona",
    "Silverstone": "silverstone",
    "Hungaroring": "hungaroring",
    "Spa": "spa",
    "Zandvoort": "zandvoort",
    "Monza": "monza",
    "Singapore": "singapore",
    "Brazil": "sao_paulo",
    "Las_Vegas": "las_vegas",
    "Losail": "lusail",
    "Baku_Azerbaijan": "baku",
    "Abu_Dhabi": "yas_marina",
    "Austria": "spielberg",
    "Suzuka": "suzuka",
    "Mexico": "mexico_city",
}

# 赛道显示名称
TRACK_LABELS = {
    "melbourne": "MELBOURNE",
    "shanghai": "SHANGHAI",
    "sakhir": "BAHRAIN",
    "jeddah": "JEDDAH",
    "miami": "MIAMI",
    "austin": "AUSTIN",
    "monaco": "MONACO",
    "montreal": "MONTREAL",
    "barcelona": "BARCELONA",
    "silverstone": "SILVERSTONE",
    "hungaroring": "HUNGARORING",
    "spa": "SPA",
    "zandvoort": "ZANDVOORT",
    "monza": "MONZA",
    "singapore": "SINGAPORE",
    "sao_paulo": "SAO PAULO",
    "las_vegas": "LAS VEGAS",
    "lusail": "LOSAIL",
    "baku": "BAKU",
    "yas_marina": "ABU DHABI",
    "spielberg": "AUSTRIA",
    "suzuka": "SUZUKA",
    "mexico_city": "MEXICO CITY",
    "madrid": "MADRID",
}

# F1 Broadcast 深色主题
BG = "#0d0d12"
TRACK_LINE = "#3a3e48"
TRACK_ACCENT = "rgba(59,158,255,0.20)"
CORNER_RING = "#3B9EFF"
CORNER_TEXT = "#9494a8"
START_LINE = "#FF1801"
LABEL_COLOR = "#5c5c70"

# 画布尺寸
CANVAS_W = 800
CANVAS_H = 600
PADDING = 40

BASE_URL = "https://raw.githubusercontent.com/ashwin-nat/pits-n-giggles/main/assets/track-maps/f1_2025/"


# --------------------------------------------------------------------------- #
# SVG 下载与解析
# --------------------------------------------------------------------------- #

def download_svg(track_file: str) -> str:
    """从 pits-n-giggles 下载赛道 SVG。"""
    url = BASE_URL + track_file + ".svg"
    print(f"  下载: {url}")
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read().decode("utf-8")


def parse_polyline_points(svg_text: str) -> list[tuple[float, float]]:
    """从 SVG 文本中提取 polyline 的 points。"""
    match = re.search(r'<polyline\s+points="([^"]+)"', svg_text)
    if not match:
        raise ValueError("未找到 polyline points")
    raw = match.group(1)
    points = []
    for pair in raw.split():
        parts = pair.split(",")
        if len(parts) == 2:
            points.append((float(parts[0]), float(parts[1])))
    return points


def parse_start_finish_line(svg_text: str) -> tuple[float, float, float, float] | None:
    """从 SVG 文本中提取起点/终点线坐标。"""
    match = re.search(
        r'<line\s+x1="([\d.]+)"\s+y1="([\d.]+)"\s+x2="([\d.]+)"\s+y2="([\d.]+)"',
        svg_text,
    )
    if not match:
        return None
    return (
        float(match.group(1)),
        float(match.group(2)),
        float(match.group(3)),
        float(match.group(4)),
    )


# --------------------------------------------------------------------------- #
# 坐标变换：将 pits-n-giggles 坐标 (1000×600) 映射到我们的画布 (800×600)
# --------------------------------------------------------------------------- #

def compute_bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def transform_points(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """将原始坐标变换到 800×600 画布，保持纵横比并居中。"""
    min_x, min_y, max_x, max_y = compute_bbox(points)
    track_w = max_x - min_x
    track_h = max_y - min_y

    if track_w < 1 or track_h < 1:
        return [(p[0] * CANVAS_W / 1000, p[1] * CANVAS_H / 600) for p in points]

    avail_w = CANVAS_W - 2 * PADDING
    avail_h = CANVAS_H - 2 * PADDING
    scale = min(avail_w / track_w, avail_h / track_h)

    offset_x = (CANVAS_W - track_w * scale) / 2 - min_x * scale
    offset_y = (CANVAS_H - track_h * scale) / 2 - min_y * scale

    return [(p[0] * scale + offset_x, p[1] * scale + offset_y) for p in points]


def transform_start_line(
    line: tuple[float, float, float, float],
    points: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    """使用与 transform_points 相同的变换参数变换起点线。"""
    min_x, min_y, max_x, max_y = compute_bbox(points)
    track_w = max_x - min_x
    track_h = max_y - min_y

    if track_w < 1 or track_h < 1:
        scale = 1.0
        offset_x = 0.0
        offset_y = 0.0
    else:
        avail_w = CANVAS_W - 2 * PADDING
        avail_h = CANVAS_H - 2 * PADDING
        scale = min(avail_w / track_w, avail_h / track_h)
        offset_x = (CANVAS_W - track_w * scale) / 2 - min_x * scale
        offset_y = (CANVAS_H - track_h * scale) / 2 - min_y * scale

    return (
        line[0] * scale + offset_x,
        line[1] * scale + offset_y,
        line[2] * scale + offset_x,
        line[3] * scale + offset_y,
    )


# --------------------------------------------------------------------------- #
# 弯道定位：使用legacy track_maps的弯道距离信息
# --------------------------------------------------------------------------- #

def compute_arc_lengths(points: list[tuple[float, float]]) -> list[float]:
    """计算沿polyline的累积弧长。"""
    dists = [0.0]
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        dists.append(dists[-1] + math.sqrt(dx * dx + dy * dy))
    return dists


def find_point_at_fraction(
    points: list[tuple[float, float]],
    arc_lengths: list[float],
    fraction: float,
) -> tuple[float, float]:
    """找到polyline上指定弧长比例处的点坐标。

    fraction: 0.0 = 起点, 1.0 = 终点
    """
    total = arc_lengths[-1]
    target = fraction * total

    # 二分查找
    lo, hi = 0, len(arc_lengths) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if arc_lengths[mid] < target:
            lo = mid + 1
        else:
            hi = mid

    # 线性插值
    if lo == 0:
        return points[0]

    prev_dist = arc_lengths[lo - 1]
    curr_dist = arc_lengths[lo]
    segment_len = curr_dist - prev_dist

    if segment_len < 1e-6:
        return points[lo]

    t = (target - prev_dist) / segment_len
    x = points[lo - 1][0] + t * (points[lo][0] - points[lo - 1][0])
    y = points[lo - 1][1] + t * (points[lo][1] - points[lo - 1][1])
    return (x, y)


def find_start_index(
    points: list[tuple[float, float]],
    start_line: tuple[float, float, float, float] | None,
) -> int:
    """找到起点/终点线在polyline上的最近点索引。"""
    if start_line is None:
        return 0
    mid_x = (start_line[0] + start_line[2]) / 2
    mid_y = (start_line[1] + start_line[3]) / 2
    min_dist = float("inf")
    best_idx = 0
    for i, (px, py) in enumerate(points):
        d = (px - mid_x) ** 2 + (py - mid_y) ** 2
        if d < min_dist:
            min_dist = d
            best_idx = i
    return best_idx


def reorder_points_from_start(
    points: list[tuple[float, float]],
    start_idx: int,
) -> list[tuple[float, float]]:
    """将polyline点重新排序，使起点索引为0。"""
    if start_idx == 0:
        return points[:]
    return points[start_idx:] + points[:start_idx]


def get_corner_positions_from_legacy(
    track_id: str,
    points: list[tuple[float, float]],
    start_idx: int,
) -> dict[int, tuple[float, float]] | None:
    """使用legacy track_maps中的弯道距离信息定位弯道。

    返回 {corner_number: (x_px, y_px)} 或 None（如果legacy数据不可用）。
    """
    try:
        from f1opt.data.track_maps import TRACK_MAPS
    except ImportError:
        return None

    tm = TRACK_MAPS.get(track_id)
    if tm is None:
        return None

    # 重新排序点，使起点为索引0
    reordered = reorder_points_from_start(points, start_idx)

    # 计算累积弧长
    arc_lengths = compute_arc_lengths(reordered)
    total_arc = arc_lengths[-1]
    if total_arc < 1e-6:
        return None

    # 获取赛道总长度（米）
    total_length_m = tm.control_points[-1][0] if tm.control_points else 0
    if total_length_m < 1:
        return None

    # 对每个弯道，计算其在赛道上的距离比例
    corner_positions: dict[int, tuple[float, float]] = {}
    for corner in tm.corners:
        # 弯道中心距离 = (distance_start + distance_end) / 2
        center_dist_m = (corner.distance_start + corner.distance_end) / 2
        # 映射到弧长比例
        fraction = center_dist_m / total_length_m
        # 处理环形（fraction可能接近1.0或略超）
        fraction = fraction % 1.0

        # 在polyline上找到对应位置的点
        x, y = find_point_at_fraction(reordered, arc_lengths, fraction)
        corner_positions[corner.corner_id] = (x, y)

    return corner_positions


# --------------------------------------------------------------------------- #
# 曲率分析（fallback：当legacy数据不可用时）
# --------------------------------------------------------------------------- #

def compute_curvature(
    points: list[tuple[float, float]],
    window: int = 5,
) -> list[float]:
    n = len(points)
    if n < 3:
        return [0.0] * n
    curvature = [0.0] * n
    for i in range(n):
        i_prev = max(0, i - window)
        i_next = min(n - 1, i + window)
        if i_next - i_prev < 2:
            curvature[i] = 0.0
            continue
        dx1 = points[i][0] - points[i_prev][0]
        dy1 = points[i][1] - points[i_prev][1]
        dx2 = points[i_next][0] - points[i][0]
        dy2 = points[i_next][1] - points[i][1]
        len1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
        len2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
        if len1 < 1e-6 or len2 < 1e-6:
            curvature[i] = 0.0
            continue
        dot = (dx1 * dx2 + dy1 * dy2) / (len1 * len2)
        dot = max(-1.0, min(1.0, dot))
        angle_change = math.acos(dot)
        arc_len = len1 + len2
        if arc_len < 1e-6:
            curvature[i] = 0.0
            continue
        curvature[i] = angle_change / arc_len
    return curvature


def find_corner_positions_curvature(
    points: list[tuple[float, float]],
    num_corners: int,
    start_idx: int = 0,
) -> dict[int, tuple[float, float]]:
    """使用曲率分析识别弯道位置（fallback方法）。"""
    n = len(points)
    if n < 5 or num_corners == 0:
        return {}

    reordered = reorder_points_from_start(points, start_idx)
    curvature = compute_curvature(reordered, window=7)

    # 使用非极大值抑制找峰值
    min_sep = max(5, n // (num_corners * 2))
    peaks: list[tuple[float, int]] = []
    for i in range(n):
        is_peak = True
        for j in range(max(0, i - min_sep), min(n, i + min_sep + 1)):
            if j != i and curvature[j] > curvature[i]:
                is_peak = False
                break
        if is_peak and curvature[i] > 0.0008:
            peaks.append((curvature[i], i))

    peaks.sort(key=lambda x: x[0], reverse=True)
    selected = peaks[:num_corners]
    selected.sort(key=lambda x: x[1])

    result: dict[int, tuple[float, float]] = {}
    for i, (_, idx) in enumerate(selected):
        result[i + 1] = reordered[idx]
    return result


# --------------------------------------------------------------------------- #
# SVG 生成
# --------------------------------------------------------------------------- #

def offset_polyline(
    points: list[tuple[float, float]],
    offset: float,
) -> list[tuple[float, float]]:
    n = len(points)
    out: list[tuple[float, float]] = []
    for i in range(n):
        prev = points[(i - 1) % n]
        nxt = points[(i + 1) % n]
        tx = nxt[0] - prev[0]
        ty = nxt[1] - prev[1]
        length = math.sqrt(tx * tx + ty * ty)
        if length < 1e-6:
            out.append(points[i])
            continue
        nx = -ty / length
        ny = tx / length
        out.append((points[i][0] + nx * offset, points[i][1] + ny * offset))
    return out


def points_to_path(
    points: list[tuple[float, float]],
    closed: bool = True,
) -> str:
    if not points:
        return ""
    parts = [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
    for i in range(1, len(points)):
        parts.append(f"L {points[i][0]:.1f} {points[i][1]:.1f}")
    if closed:
        parts.append("Z")
    return " ".join(parts)


def build_svg(
    track_id: str,
    points: list[tuple[float, float]],
    corners: dict[int, tuple[float, float]],
    start_line_px: tuple[float, float] | None,
    label: str,
) -> str:
    """生成我们的 SVG 格式。"""
    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {CANVAS_W} {CANVAS_H}" '
        f'width="{CANVAS_W}" height="{CANVAS_H}">'
    )
    lines.append("  <defs>")
    lines.append("    <style>")
    lines.append(f"      .track-bg {{ fill: {BG}; }}")
    lines.append(
        f"      .track-line {{ fill: none; stroke: {TRACK_LINE}; "
        f"stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }}"
    )
    lines.append(
        f"      .track-line-accent {{ fill: none; stroke: {TRACK_ACCENT}; "
        f"stroke-width: 1.5; stroke-linecap: round; stroke-dasharray: 6,4; }}"
    )
    lines.append(
        f"      .track-edge {{ fill: none; stroke: {TRACK_LINE}; "
        f"stroke-width: 1; stroke-linecap: round; stroke-linejoin: round; opacity: 0.6; }}"
    )
    lines.append("    </style>")
    lines.append("  </defs>")
    lines.append(f'  <rect class="track-bg" width="{CANVAS_W}" height="{CANVAS_H}" rx="8"/>')

    # 赛道轮廓
    if points:
        outer_pts = offset_polyline(points, 5.0)
        inner_pts = offset_polyline(points, -5.0)

        outer_path = points_to_path(outer_pts, closed=True)
        inner_path = points_to_path(inner_pts, closed=True)
        center_path = points_to_path(points, closed=True)

        lines.append(f'  <path class="track-edge" d="{outer_path}"/>')
        lines.append(f'  <path class="track-edge" d="{inner_path}"/>')
        lines.append(f'  <path class="track-line" d="{center_path}"/>')
        lines.append(f'  <path class="track-line-accent" d="{center_path}"/>')

    # 起点/终点线
    if start_line_px is not None:
        sx, sy = start_line_px
        lines.append(
            f'  <rect x="{sx - 5:.1f}" y="{sy - 5:.1f}" width="10" height="10" '
            f'fill="{START_LINE}" rx="1"/>'
        )
    elif points:
        sx, sy = points[0]
        lines.append(
            f'  <rect x="{sx - 5:.1f}" y="{sy - 5:.1f}" width="10" height="10" '
            f'fill="{START_LINE}" rx="1"/>'
        )

    # 弯道标记
    for corner_num in sorted(corners.keys()):
        cx, cy = corners[corner_num]
        lines.append(
            f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="none" '
            f'stroke="{CORNER_RING}" stroke-width="1.5" opacity="0.85"/>'
        )
        lines.append(
            f'  <text x="{cx:.1f}" y="{cy - 9:.1f}" text-anchor="middle" '
            f'fill="{CORNER_TEXT}" font-size="9" font-family="monospace">{corner_num}</text>'
        )

    # 赛道名称
    lines.append(
        f'  <text x="{CANVAS_W / 2}" y="{CANVAS_H - 12}" text-anchor="middle" '
        f'fill="{LABEL_COLOR}" font-size="11" font-family="monospace" '
        f'letter-spacing="2">{label}</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Madrid 赛道（2026新增，无真实数据）
# --------------------------------------------------------------------------- #

MADRID_CONTROL_POINTS = [
    (400, 50), (420, 80), (440, 120), (430, 160), (400, 180),
    (360, 170), (340, 140), (330, 110), (340, 80), (370, 60),
    (400, 50), (440, 50), (480, 70), (510, 100), (530, 140),
    (520, 180), (490, 210), (450, 230), (410, 250), (380, 280),
    (360, 320), (380, 360), (420, 380), (460, 370), (500, 350),
    (540, 330), (570, 300), (580, 260), (570, 220), (550, 190),
    (520, 170), (490, 150), (470, 120), (450, 90), (430, 70),
    (410, 55), (400, 50),
]

MADRID_CORNERS = {
    1: (440, 120), 2: (400, 180), 3: (340, 140), 4: (340, 80),
    5: (510, 100), 6: (520, 180), 7: (450, 230), 8: (380, 280),
    9: (380, 360), 10: (460, 370), 11: (540, 330), 12: (580, 260),
    13: (550, 190), 14: (490, 150), 15: (450, 90),
}


def build_madrid_svg() -> tuple[str, dict[int, tuple[float, float]]]:
    points = MADRID_CONTROL_POINTS
    corners = MADRID_CORNERS
    svg = build_svg("madrid", points, corners, None, "MADRID")
    return svg, corners


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def process_track(
    track_file: str,
    track_id: str,
) -> tuple[str, dict[int, tuple[float, float]]] | None:
    """处理一条赛道：下载 → 解析 → 变换 → 定位弯道 → 生成 SVG。"""
    try:
        svg_text = download_svg(track_file)
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        return None

    raw_points = parse_polyline_points(svg_text)
    if len(raw_points) < 10:
        print(f"  ❌ polyline 点太少: {len(raw_points)}")
        return None
    print(f"  原始 polyline: {len(raw_points)} 个点")

    raw_start_line = parse_start_finish_line(svg_text)

    # 变换坐标到 800×600 画布
    points = transform_points(raw_points)
    if raw_start_line is not None:
        start_line = transform_start_line(raw_start_line, raw_points)
        start_line_mid = ((start_line[0] + start_line[2]) / 2, (start_line[1] + start_line[3]) / 2)
    else:
        start_line = None
        start_line_mid = None

    # 找到起点索引（在原始坐标中）
    start_idx = find_start_index(raw_points, raw_start_line)

    # 优先使用legacy数据定位弯道
    corner_positions = get_corner_positions_from_legacy(track_id, points, start_idx)

    if corner_positions is not None:
        print(f"  ✅ 使用legacy距离映射定位 {len(corner_positions)} 个弯道")
    else:
        # Fallback: 曲率分析
        print("  ⚠️ legacy数据不可用，使用曲率分析")
        # 获取已知弯道数量
        try:
            from f1opt.data.track_maps import TRACK_MAPS
            num_corners = len(TRACK_MAPS[track_id].corners) if track_id in TRACK_MAPS else 15
        except (ImportError, KeyError):
            num_corners = 15
        corner_positions = find_corner_positions_curvature(points, num_corners, start_idx)
        print(f"  曲率分析找到 {len(corner_positions)} 个弯道")

    # 生成 SVG
    label = TRACK_LABELS.get(track_id, track_id.upper())
    svg = build_svg(track_id, points, corner_positions, start_line_mid, label)

    return svg, corner_positions


def export_anchors_module(
    all_anchors: dict[str, dict[int, tuple[float, float]]],
) -> str:
    """生成 _track_anchors.py 源码。"""
    lines: list[str] = []
    lines.append('"""真实F1赛道弯道像素坐标（基于 pits-n-giggles 遥测数据 + legacy距离映射）。')
    lines.append("")
    lines.append("由 scripts/convert_real_track_svgs.py 自动生成，请勿手工编辑。")
    lines.append('"""')
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("# (canvas_width, canvas_height)")
    lines.append("TRACK_CANVAS: dict[str, tuple[int, int]] = {")
    for tid in sorted(all_anchors.keys()):
        lines.append(f'    "{tid}": ({CANVAS_W}, {CANVAS_H}),')
    lines.append("}")
    lines.append("")
    lines.append("# {track_id: {corner_number: (x_px, y_px)}}")
    lines.append("TRACK_ANCHORS: dict[str, dict[int, tuple[float, float]]] = {")
    for tid in sorted(all_anchors.keys()):
        anchors = all_anchors[tid]
        lines.append(f'    "{tid}": {{')
        for corner_num in sorted(anchors.keys()):
            x, y = anchors[corner_num]
            lines.append(f"        {corner_num}: ({x:.1f}, {y:.1f}),")
        lines.append("    },")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_anchors: dict[str, dict[int, tuple[float, float]]] = {}
    success_count = 0
    fail_count = 0

    for track_file, track_id in sorted(TRACK_FILE_MAP.items()):
        print(f"\n处理: {track_file} → {track_id}")
        result = process_track(track_file, track_id)
        if result is None:
            fail_count += 1
            continue

        svg, corner_anchors = result
        out_path = OUT_DIR / f"{track_id}.svg"
        out_path.write_text(svg, encoding="utf-8")
        all_anchors[track_id] = corner_anchors
        success_count += 1
        print(f"  ✅ 已保存: {out_path}")

    # Madrid（无真实数据）
    print("\n处理: Madrid → madrid (手动绘制)")
    svg, corners = build_madrid_svg()
    out_path = OUT_DIR / "madrid.svg"
    out_path.write_text(svg, encoding="utf-8")
    all_anchors["madrid"] = corners
    success_count += 1
    print(f"  ✅ 已保存: {out_path}")

    # 导出锚点模块
    anchors_code = export_anchors_module(all_anchors)
    ANCHORS_PATH.write_text(anchors_code, encoding="utf-8")
    print(f"\n导出锚点数据 → {ANCHORS_PATH}")

    print(f"\n完成: {success_count} 成功, {fail_count} 失败")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
