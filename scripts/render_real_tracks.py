"""把 24 条赛道的真实 GPS 几何渲染成带标记的小图，供人工核对起点/方向/弯道。

每张图：真实轨迹 + 红方块(frac0 起点) + 方向箭头 + 弯道簇(编号+累计转角)。
输出：scripts/real_circuits/render/<track_id>.svg
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from real_corner_analysis import analyze, load_coords, to_local_m, TRACK_GEOJSON

OUT_DIR = Path(__file__).resolve().parent / "real_circuits" / "render"

W, H = 720, 520
PAD = 46


def render(track_id: str) -> Path:
    r = analyze(track_id)
    pts = to_local_m(load_coords(track_id))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    scale = min((W - 2 * PAD) / (maxx - minx), (H - 2 * PAD) / (maxy - miny))
    ox = (W - (maxx - minx) * scale) / 2
    oy = (H - (maxy - miny) * scale) / 2

    def px(p: tuple[float, float]) -> tuple[float, float]:
        return (ox + (p[0] - minx) * scale, H - (oy + (p[1] - miny) * scale))

    # 环加密用于画线（简单直线段即可）
    ring = pts + [pts[0]]
    path_pts = [px(p) for p in ring]
    d_attr = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in path_pts) + " Z"

    # 方向箭头：frac 0.5% 处的切向
    import real_corner_analysis as ra
    dense = ra.densify_closed(pts, 2.0)
    n = len(dense)
    ai = int(0.005 * n)
    ax0, ay0 = px(dense[ai])
    ax1, ay1 = px(dense[(ai + 12) % n])
    ang = math.degrees(math.atan2(ay1 - ay0, ax1 - ax0))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}">',
        f'<rect width="{W}" height="{H}" fill="#111318"/>',
        f'<path d="{d_attr}" fill="none" stroke="#8a93a6" stroke-width="4" '
        f'stroke-linejoin="round" stroke-linecap="round"/>',
        # 起点标记（红方块）
    ]
    sx, sy = px(dense[0])
    parts.append(
        f'<rect x="{sx-6:.1f}" y="{sy-6:.1f}" width="12" height="12" '
        f'fill="#FF1801" stroke="#fff" stroke-width="1"/>'
    )
    # 方向箭头
    parts.append(
        f'<g transform="translate({ax0:.1f},{ay0:.1f}) rotate({ang:.0f})">'
        f'<path d="M 10 0 L -4 -6 L -4 6 Z" fill="#FFD21E"/></g>'
    )
    # 弯道簇
    for idx, c in enumerate(r["corners"], start=1):
        cx_, cy_ = dense[int(c["frac"] * n) % n]
        X, Y = px((cx_, cy_))
        parts.append(
            f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="7" fill="none" '
            f'stroke="#3B9EFF" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{X:.1f}" y="{Y-11:.1f}" text-anchor="middle" '
            f'fill="#e8ecf4" font-size="12" font-family="monospace" '
            f'font-weight="bold">{idx}</text>'
        )
        parts.append(
            f'<text x="{X:.1f}" y="{Y+22:.1f}" text-anchor="middle" '
            f'fill="#9aa3b5" font-size="9" font-family="monospace">'
            f'{c["deg"]:.0f}°</text>'
        )
    parts.append(
        f'<text x="12" y="20" fill="#e8ecf4" font-size="14" '
        f'font-family="monospace">{track_id} · 官方{r["corners"] and ""}'
        f'{len(r["corners"])}簇 / {r["total_m"]:.0f}m · ■=frac0(geojson起点) ▶=行驶向</text>'
    )
    parts.append("</svg>")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{track_id}.svg"
    out.write_text("\n".join(parts), encoding="utf-8", newline="\n")
    return out


def main() -> None:
    ids = sys.argv[1:] or sorted(TRACK_GEOJSON)
    for tid in ids:
        print("render", render(tid))


if __name__ == "__main__":
    main()
