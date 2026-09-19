"""由真实 GPS 几何重建全部赛道资产（单一数据源，一次性重建工具）。

产物（全部由 ``scripts/real_geometry.py`` 的官方 N 锚点选择驱动）：
1. ``setup_tuner/ui/tracks/<id>.svg`` —— 真实几何渲染的赛道图（可见圆点层）
2. ``setup_tuner/domain/_track_anchors.py`` —— 锚点像素坐标（热区/API 层）
3. ``setup_tuner/domain/_track_arcs.py`` —— 锚点弧长占比（真实圈距占比，
   与遥测 lap_distance 同一口径）

三层同源生成，结构上不可能脱节（task-79 根因的根治）。
运行::

    python scripts/gen_real_tracks.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import real_geometry as rg  # noqa: E402

from setup_tuner.domain._track_official import OFFICIAL_TURN_COUNTS  # noqa: E402

TRACKS_DIR = REPO / "setup_tuner" / "ui" / "tracks"
ANCHORS_PATH = REPO / "setup_tuner" / "domain" / "_track_anchors.py"
ARCS_PATH = REPO / "setup_tuner" / "domain" / "_track_arcs.py"

SVG_HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="800" height="600">
  <defs>
    <style>
      .track-bg { fill: #0d0d12; }
      .track-line { fill: none; stroke: #3a3e48; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }
      .track-line-accent { fill: none; stroke: rgba(59,158,255,0.20); stroke-width: 1.5; stroke-linecap: round; stroke-dasharray: 6,4; }
      .track-edge { fill: none; stroke: #3a3e48; stroke-width: 1; stroke-linecap: round; stroke-linejoin: round; opacity: 0.6; }
    </style>
  </defs>
  <rect class="track-bg" width="800" height="600" rx="8"/>"""

CIRCLE_FMT = (
    '  <circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="none" stroke="#3B9EFF" '
    'stroke-width="1.5" opacity="0.85"/>'
)
TEXT_FMT = (
    '  <text x="{x:.2f}" y="{y:.2f}" text-anchor="middle" fill="#9494a8" '
    'font-size="9" font-family="monospace">{num}</text>'
)

TEXT_Y_OFFSET = 9.0
PATH_EVERY = 6  # 路径抽稀步长（×1m 采样）


def old_caption(track_id: str) -> str:
    """从旧 SVG 提取标题文字（如 MONZA / BAHRAIN），缺失回退 track_id。"""
    path = TRACKS_DIR / f"{track_id}.svg"
    if path.exists():
        m = re.search(r'letter-spacing="2">([A-Z0-9 ]+)</text>', path.read_text(encoding="utf-8"))
        if m:
            return m.group(1).strip()
    return track_id.upper()


def path_d(pts: list[tuple[float, float]]) -> str:
    """抽稀折线路径（真实几何等比投影，首尾闭合）。"""
    sel = pts[::PATH_EVERY]
    if sel[-1] != pts[0]:
        sel.append(pts[0])
    return "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in sel) + " Z"


def build_svg(track_id: str, pts: list[tuple[float, float]], anchors: list[tuple[float, float]]) -> str:
    d = path_d(pts)
    parts = [SVG_HEAD, f'  <path class="track-edge" d="{d}"/>']
    parts.append(f'  <path class="track-line" d="{d}"/>')
    parts.append(f'  <path class="track-line-accent" d="{d}"/>')
    for i, (x, y) in enumerate(anchors, start=1):
        parts.append(CIRCLE_FMT.format(x=x, y=y))
        parts.append(TEXT_FMT.format(x=x, y=y - TEXT_Y_OFFSET, num=i))
    parts.append(
        f'  <text x="400" y="588" text-anchor="middle" fill="#5c5c70" '
        f'font-size="11" font-family="monospace" letter-spacing="2">'
        f'{old_caption(track_id)}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_anchors_module(data: dict[str, dict[int, tuple[float, float]]]) -> str:
    lines = [
        '"""各赛道弯道锚点像素坐标 —— 由真实 GPS 几何自动定位（自动生成，勿手工编辑）。',
        "",
        "数据源：``scripts/real_circuits/*.geojson``（GPS 实测圈，起点=发车线，",
        "方向=官方行驶方向）。锚点 = 官方弯数 N 个真实转弯特征点，由",
        "``scripts/real_geometry.py::select_anchors`` 选出：每个锚点都落在",
        "实测转弯上（硬门 >= 12°/±40m），官方每个弯都有锚点、无凭空多标。",
        "坐标为 800×600 画布像素；由 ``scripts/gen_real_tracks.py`` 生成。",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "# (canvas_width, canvas_height)",
        "TRACK_CANVAS: dict[str, tuple[int, int]] = {",
    ]
    for tid in sorted(data):
        lines.append(f'    "{tid}": (800, 600),')
    lines.append("}")
    lines.append("")
    lines.append("# {track_id: {corner_number: (x_px, y_px)}}")
    lines.append("TRACK_ANCHORS: dict[str, dict[int, tuple[float, float]]] = {")
    for tid, anchors in sorted(data.items()):
        lines.append(f'    "{tid}": {{')
        for num, (x, y) in anchors.items():
            lines.append(f"        {num}: ({x:.2f}, {y:.2f}),")
        lines.append("    },")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def render_arcs_module(arcs: dict[str, dict[int, float]]) -> str:
    lines = [
        '"""各赛道弯道锚点的圈内距离占比（真实 GPS 圈距口径，自动生成，勿手工编辑）。',
        "",
        "由 ``scripts/gen_real_tracks.py`` 生成：锚点来自真实几何（起点=发车线），",
        "占比 = 锚点真实弧长 / 真实圈长，与遥测 ``lap_distance`` 同一口径。",
        "``tests/test_track_arcs.py`` 在 CI 中用同一算法复算比对，防止数据脱节。",
        "",
        "用途：``api/ws.py::_map_corner`` 依据圈内距离 (m) → 占比，取最近弯道。",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "# {track_id: {corner_number: arc_fraction}}，arc_fraction ∈ [0, 1)",
        "TRACK_CORNER_ARCS: dict[str, dict[int, float]] = {",
    ]
    for tid, corners in arcs.items():
        lines.append(f'    "{tid}": {{')
        lines.extend(f"        {num}: {frac}," for num, frac in corners.items())
        lines.append("    },")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    all_anchors: dict[str, dict[int, tuple[float, float]]] = {}
    all_arcs: dict[str, dict[int, float]] = {}
    for tid in sorted(rg.TRACK_GEOJSON):
        n_off = OFFICIAL_TURN_COUNTS[tid][0]
        rt = rg.RealTrack(tid)
        idxs = rg.select_anchors(rt, n_off)
        pts, anchor_px = rg.to_canvas(rt, idxs)
        TRACKS_DIR.mkdir(parents=True, exist_ok=True)
        (TRACKS_DIR / f"{tid}.svg").write_text(
            build_svg(tid, pts, anchor_px), encoding="utf-8", newline="\n",
        )
        all_anchors[tid] = {i + 1: p for i, p in enumerate(anchor_px)}
        all_arcs[tid] = {
            pos + 1: round(rt.cum[idx] / rt.total_m, 6)
            for pos, idx in enumerate(idxs)
        }
    ANCHORS_PATH.write_text(
        render_anchors_module(all_anchors), encoding="utf-8", newline="\n",
    )
    ARCS_PATH.write_text(
        render_arcs_module(all_arcs), encoding="utf-8", newline="\n",
    )
    total = sum(len(v) for v in all_anchors.values())
    print(f"重建 {len(all_anchors)} 条赛道 / {total} 个官方弯锚点")
    print(f"  {TRACKS_DIR}/<id>.svg × 24")
    print(f"  {ANCHORS_PATH}")
    print(f"  {ARCS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
