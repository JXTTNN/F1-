"""从 legacy/f1opt/data/track_maps 重新生成 24 条赛道的高质量 SVG。

改进点：
1. 用 SVG path + 三次贝塞尔（C 命令）代替 polyline，得到平滑赛道轮廓
2. 画外/内两条平行轮廓线（间距 ~10px）呈现赛道宽度
3. 用红色方块标注 start_finish_line_px
4. 在 corners 的 (x_px, y_px) 位置画圆圈 + 编号
5. 底部标注赛道名称
6. F1 深色主题：背景 #111317 / 赛道线 #2a2e36 / 弯道标记 #00e5ff

输出：setup_tuner/ui/tracks/<track_id>.svg，viewBox="0 0 800 600"。

同时导出 _track_anchors.py 模块供 track.py 使用。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "legacy"))

from f1opt.data.track_maps import TRACK_MAPS, TrackMapData  # noqa: E402
from setup_tuner.domain.track import ALL_TRACKS  # noqa: E402

# track.py 中 Track 对象的 corners 类型
from setup_tuner.domain.track import Corner as DomainCorner  # noqa: E402

OUT_DIR = ROOT / "setup_tuner" / "ui" / "tracks"

# F1 Broadcast 深色主题
BG = "#0d0d12"
TRACK_LINE = "#3a3e48"
TRACK_ACCENT = "rgba(59,158,255,0.20)"
CORNER_RING = "#3B9EFF"
CORNER_TEXT = "#9494a8"
START_LINE = "#FF1801"
LABEL_COLOR = "#5c5c70"


# --------------------------------------------------------------------------- #
# 几何辅助
# --------------------------------------------------------------------------- #

def _catmull_rom_to_bezier(pts: list[tuple[float, float]]) -> str:
    """把控制点序列转为闭合的三次贝塞尔 SVG path。

    用 Catmull-Rom 样条转换到三次贝塞尔（每段 4 控制点 → 1 个 C 命令），
    生成 C1 连续的平滑闭合曲线。张力因子 0.5（标准 Catmull-Rom）。
    """
    n = len(pts)
    if n < 3:
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        return d + " Z"

    pts_ext = pts[-1:] + pts + pts[:2]
    path_parts: list[str] = [f"M {pts[0][0]:.1f} {pts[0][1]:.1f}"]
    for i in range(n):
        p0 = pts_ext[i]
        p1 = pts_ext[i + 1]
        p2 = pts_ext[i + 2]
        p3 = pts_ext[i + 3]
        c1x = p1[0] + (p2[0] - p0[0]) / 6.0
        c1y = p1[1] + (p2[1] - p0[1]) / 6.0
        c2x = p2[0] - (p3[0] - p1[0]) / 6.0
        c2y = p2[1] - (p3[1] - p1[1]) / 6.0
        path_parts.append(
            f"C {c1x:.1f} {c1y:.1f} {c2x:.1f} {c2y:.1f} {p2[0]:.1f} {p2[1]:.1f}"
        )
    path_parts.append("Z")
    return " ".join(path_parts)


def _offset_path(pts: list[tuple[float, float]], offset: float) -> list[tuple[float, float]]:
    """沿法线方向偏移点序列（用于生成内/外赛道线）。

    对每个点取前后相邻点切向，法向偏移 offset 像素。
    """
    n = len(pts)
    out: list[tuple[float, float]] = []
    for i in range(n):
        prev = pts[(i - 1) % n]
        nxt = pts[(i + 1) % n]
        tx = nxt[0] - prev[0]
        ty = nxt[1] - prev[1]
        length = (tx * tx + ty * ty) ** 0.5
        if length < 1e-6:
            out.append(pts[i])
            continue
        nx = -ty / length
        ny = tx / length
        out.append((pts[i][0] + nx * offset, pts[i][1] + ny * offset))
    return out


# --------------------------------------------------------------------------- #
# 赛道轮廓插值（替代椭圆估算）
# --------------------------------------------------------------------------- #

def _interpolate_position_on_track(
    control_points: list[tuple[float, float, float]],
    corner_number: int,
    total_corners: int,
) -> tuple[float, float]:
    """沿赛道轮廓 control_points 插值估算弯道像素坐标。

    用弯道编号在赛道上的相对位置（均匀分布假设）映射到 control_points
    的 distance→pixel 曲线上，得到比椭圆估算更贴合赛道形状的坐标。

    Args:
        control_points: [(distance_m, x_px, y_px), ...] 按 distance 升序排列。
        corner_number: 弯道编号（1-based）。
        total_corners: 赛道弯道总数。

    Returns:
        (x_px, y_px) 估算的弯道像素坐标。
    """
    if not control_points:
        return (400.0, 300.0)

    # 弯道在赛道上的相对位置（均匀分布假设）
    t = (corner_number - 0.5) / total_corners
    max_dist = control_points[-1][0]
    target_dist = t * max_dist

    # 在 control_points 上线性插值
    for i in range(len(control_points) - 1):
        d0, x0, y0 = control_points[i]
        d1, x1, y1 = control_points[i + 1]
        if d0 <= target_dist <= d1:
            if d1 - d0 < 1e-6:
                return (x0, y0)
            ratio = (target_dist - d0) / (d1 - d0)
            return (x0 + ratio * (x1 - x0), y0 + ratio * (y1 - y0))

    # 超出范围：返回最后一个 control_point
    return (control_points[-1][1], control_points[-1][2])


# --------------------------------------------------------------------------- #
# SVG 生成
# --------------------------------------------------------------------------- #

def _build_svg(
    track: TrackMapData,
    track_label: str,
    domain_corners: list[DomainCorner] | None = None,
) -> str:
    canvas_w = track.canvas_width
    canvas_h = track.canvas_height
    pts = [(cp[1], cp[2]) for cp in track.control_points]
    dedup: list[tuple[float, float]] = []
    for p in pts:
        if not dedup or (abs(dedup[-1][0] - p[0]) > 0.5 or abs(dedup[-1][1] - p[1]) > 0.5):
            dedup.append(p)
    if len(dedup) < 3:
        dedup = pts

    outer = _offset_path(dedup, 5.0)
    inner = _offset_path(dedup, -5.0)
    center_path = _catmull_rom_to_bezier(dedup)
    outer_path = _catmull_rom_to_bezier(outer)
    inner_path = _catmull_rom_to_bezier(inner)

    # 构建 track_maps 真实坐标查找表: {corner_id: (x_px, y_px)}
    real_corner_map: dict[int, tuple[float, float]] = {
        c.corner_id: (c.x_px, c.y_px) for c in track.corners
    }

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {canvas_w} {canvas_h}" '
        f'width="{canvas_w}" height="{canvas_h}">'
    )
    lines.append("  <defs>")
    lines.append("    <style>")
    lines.append(f"      .track-bg {{ fill: {BG}; }}")
    lines.append(
        "      .track-line { fill: none; stroke: " + TRACK_LINE + "; "
        "stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }"
    )
    lines.append(
        "      .track-line-accent { fill: none; stroke: " + TRACK_ACCENT + "; "
        "stroke-width: 1.5; stroke-linecap: round; stroke-dasharray: 6,4; }"
    )
    lines.append(
        "      .track-edge { fill: none; stroke: " + TRACK_LINE + "; "
        "stroke-width: 1; stroke-linecap: round; stroke-linejoin: round; opacity: 0.6; }"
    )
    lines.append("    </style>")
    lines.append("  </defs>")
    lines.append(f'  <rect class="track-bg" width="{canvas_w}" height="{canvas_h}" rx="8"/>')

    lines.append(f'  <path class="track-edge" d="{outer_path}"/>')
    lines.append(f'  <path class="track-edge" d="{inner_path}"/>')
    lines.append(f'  <path class="track-line" d="{center_path}"/>')
    lines.append(f'  <path class="track-line-accent" d="{center_path}"/>')

    sx, sy = track.start_finish_line_px
    lines.append(
        f'  <rect x="{sx - 5:.1f}" y="{sy - 5:.1f}" width="10" height="10" '
        f'fill="{START_LINE}" rx="1"/>'
    )

    # 确定要绘制的弯道列表：优先用 track.py 的完整弯道列表
    if domain_corners is not None:
        total = len(domain_corners)
        for dc in domain_corners:
            if dc.number in real_corner_map:
                cx, cy = real_corner_map[dc.number]
            else:
                # 缺失弯道：用赛道轮廓插值估算坐标（比椭圆更贴合赛道形状）
                cx, cy = _interpolate_position_on_track(
                    track.control_points, dc.number, total,
                )
            lines.append(
                f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="none" '
                f'stroke="{CORNER_RING}" stroke-width="1.5" opacity="0.85"/>'
            )
            lines.append(
                f'  <text x="{cx:.1f}" y="{cy - 9:.1f}" text-anchor="middle" '
                f'fill="{CORNER_TEXT}" font-size="9" font-family="monospace">{dc.number}</text>'
            )
    else:
        # fallback: 只用 track_maps 的 corners
        for c in track.corners:
            cx, cy = c.x_px, c.y_px
            lines.append(
                f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="none" '
                f'stroke="{CORNER_RING}" stroke-width="1.5" opacity="0.85"/>'
            )
            lines.append(
                f'  <text x="{cx:.1f}" y="{cy - 9:.1f}" text-anchor="middle" '
                f'fill="{CORNER_TEXT}" font-size="9" font-family="monospace">{c.corner_id}</text>'
            )

    lines.append(
        f'  <text x="{canvas_w / 2}" y="{canvas_h - 12}" text-anchor="middle" '
        f'fill="{LABEL_COLOR}" font-size="11" font-family="monospace" '
        f'letter-spacing="2">{track_label}</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 锚点数据导出
# --------------------------------------------------------------------------- #

def _export_anchors_module() -> str:
    """生成 setup_tuner/domain/_track_anchors.py 源码。

    对 track_maps 中缺失的弯道，用椭圆估算坐标补充，
    确保 _track_anchors.py 包含 track.py 中定义的所有弯道。
    """
    domain_by_id = {t.track_id: t for t in ALL_TRACKS}

    lines: list[str] = []
    lines.append('"""从 legacy track_maps 提取的真实弯道像素坐标。')
    lines.append("")
    lines.append("由 scripts/generate_track_svgs.py 自动生成，请勿手工编辑。")
    lines.append('"""')
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("# (canvas_width, canvas_height)")
    lines.append("TRACK_CANVAS: dict[str, tuple[int, int]] = {")
    for tid in sorted(TRACK_MAPS):
        tm = TRACK_MAPS[tid]
        lines.append(f'    "{tid}": ({tm.canvas_width}, {tm.canvas_height}),')
    lines.append("}")
    lines.append("")
    lines.append("# {track_id: {corner_number: (x_px, y_px)}}")
    lines.append("TRACK_ANCHORS: dict[str, dict[int, tuple[float, float]]] = {")
    for tid in sorted(TRACK_MAPS):
        tm = TRACK_MAPS[tid]
        lines.append(f'    "{tid}": {{')
        real_map = {c.corner_id: (c.x_px, c.y_px) for c in tm.corners}
        # 从 track.py 获取完整弯道列表，为缺失弯道补充插值坐标
        domain_track = domain_by_id.get(tid)
        if domain_track is not None:
            total = len(domain_track.corners)
            for dc in domain_track.corners:
                if dc.number in real_map:
                    lines.append(f"        {dc.number}: ({real_map[dc.number][0]}, {real_map[dc.number][1]}),")
                else:
                    cx, cy = _interpolate_position_on_track(
                        tm.control_points, dc.number, total,
                    )
                    lines.append(f"        {dc.number}: ({cx:.1f}, {cy:.1f}),")
        else:
            for c in tm.corners:
                lines.append(f"        {c.corner_id}: ({c.x_px}, {c.y_px}),")
        lines.append("    },")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    domain_by_id = {t.track_id: t for t in ALL_TRACKS}
    count = 0
    for tid, tm in TRACK_MAPS.items():
        label = tid.replace("_", " ").upper()
        domain_track = domain_by_id.get(tid)
        domain_corners = domain_track.corners if domain_track is not None else None
        svg = _build_svg(tm, label, domain_corners)
        out_path = OUT_DIR / f"{tid}.svg"
        out_path.write_text(svg, encoding="utf-8")
        count += 1
    print(f"生成 {count} 个 SVG -> {OUT_DIR}")

    anchors_path = ROOT / "setup_tuner" / "domain" / "_track_anchors.py"
    anchors_path.write_text(_export_anchors_module(), encoding="utf-8")
    print(f"导出锚点数据 -> {anchors_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())