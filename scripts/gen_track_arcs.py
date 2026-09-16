"""生成 ``setup_tuner/domain/_track_arcs.py`` —— 各弯道锚点在赛道 SVG 路径上的弧长占比。

为什么需要：
    ``api/ws.py::_map_corner`` 早期用「弯角沿赛道均匀分布」的近似判定「当前弯」，
    但弯道在真实赛道上并不等距（例如 Suzuka：T1 在 0%、T7 在 26%、T14-15 在 76-79%）。
    云端实测该近似的判定错误率高达 69%（24 赛道 × 200 采样点 = 3313/4800）。
    本脚本把「锚点 → 路径弧长占比」一次性算好落成数据文件，运行时二分查找即可，
    单次查询约 0.2–0.3 µs（与错误近似同量级），零运行时 SVG 解析开销。

用法：
    python scripts/gen_track_arcs.py

产物：
    ``setup_tuner/domain/_track_arcs.py``：``TRACK_CORNER_ARCS[track_id][corner] = fraction``
    数值含义：弯道锚点最近点在该赛道 SVG 路径累计弧长上的占比 ∈ [0, 1)。

正确性保障：
    ``tests/test_track_arcs.py`` 在 CI 中调用本模块的 :func:`compute_arcs` 重新计算，
    与已提交的数据文件逐条比对（容差 1e-3），确保数据与 ``ui/tracks/*.svg`` 不脱节。
"""

from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SVG_DIR = REPO_ROOT / "setup_tuner" / "ui" / "tracks"
ANCHORS_PATH = REPO_ROOT / "setup_tuner" / "domain" / "_track_anchors.py"
OUT_PATH = REPO_ROOT / "setup_tuner" / "domain" / "_track_arcs.py"

_TOK = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|(-?\d*\.?\d+(?:[eE][-+]?\d+)?)")
_CUBIC_SAMPLES = 10
_QUAD_SAMPLES = 8
_ARC_SAMPLES = 10


def load_anchors() -> dict[str, dict[int, tuple[float, float]]]:
    """加载 ``_track_anchors.TRACK_ANCHORS``（按文件路径导入，不触发包导入副作用）。"""
    spec = importlib.util.spec_from_file_location("_gen_anchors", ANCHORS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TRACK_ANCHORS


# --------------------------------------------------------------------------- #
# 曲线采样
# --------------------------------------------------------------------------- #
def _cubic_points(
    p0: tuple[float, float], p1: tuple[float, float],
    p2: tuple[float, float], p3: tuple[float, float],
) -> list[tuple[float, float]]:
    out = []
    for k in range(1, _CUBIC_SAMPLES + 1):
        t = k / _CUBIC_SAMPLES
        mt = 1 - t
        out.append((
            mt ** 3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t ** 3 * p3[0],
            mt ** 3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t ** 3 * p3[1],
        ))
    return out


def _quad_points(
    p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float],
) -> list[tuple[float, float]]:
    out = []
    for k in range(1, _QUAD_SAMPLES + 1):
        t = k / _QUAD_SAMPLES
        mt = 1 - t
        out.append((
            mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0],
            mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1],
        ))
    return out


def _arc_points(
    p0: tuple[float, float], rx: float, ry: float, phi_deg: float,
    large_arc: int, sweep: int, p1: tuple[float, float],
) -> list[tuple[float, float]]:
    """SVG 椭圆弧（端点参数化）→ 采样点。"""
    x0, y0 = p0
    x, y = p1
    rx, ry = abs(rx), abs(ry)
    if rx == 0.0 or ry == 0.0 or (x0 == x and y0 == y):
        return [p1]
    phi = math.radians(phi_deg)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    dx2, dy2 = (x0 - x) / 2.0, (y0 - y) / 2.0
    x1p = cos_p * dx2 + sin_p * dy2
    y1p = -sin_p * dx2 + cos_p * dy2
    lam = x1p * x1p / (rx * rx) + y1p * y1p / (ry * ry)
    if lam > 1.0:
        scale = math.sqrt(lam)
        rx *= scale
        ry *= scale
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coef = math.sqrt(max(0.0, num / den)) if den else 0.0
    if large_arc == sweep:
        coef = -coef
    cxp = coef * rx * y1p / ry
    cyp = -coef * ry * x1p / rx
    cx = cos_p * cxp - sin_p * cyp + (x0 + x) / 2.0
    cy = sin_p * cxp + cos_p * cyp + (y0 + y) / 2.0

    def _angle(ux: float, uy: float, vx: float, vy: float) -> float:
        dot = ux * vx + uy * vy
        norm = math.hypot(ux, uy) * math.hypot(vx, vy)
        value = 0.0 if norm == 0 else math.acos(max(-1.0, min(1.0, dot / norm)))
        return -value if (ux * vy - uy * vx) < 0 else value

    ux, uy = (x1p - cxp) / rx, (y1p - cyp) / ry
    vx, vy = (-x1p - cxp) / rx, (-y1p - cyp) / ry
    theta1 = _angle(1.0, 0.0, ux, uy)
    dtheta = _angle(ux, uy, vx, vy)
    if not sweep and dtheta > 0:
        dtheta -= 2 * math.pi
    elif sweep and dtheta < 0:
        dtheta += 2 * math.pi
    out = []
    for k in range(1, _ARC_SAMPLES + 1):
        t = theta1 + dtheta * k / _ARC_SAMPLES
        out.append((
            cx + rx * math.cos(t) * cos_p - ry * math.sin(t) * sin_p,
            cy + rx * math.cos(t) * sin_p + ry * math.sin(t) * cos_p,
        ))
    return out


# --------------------------------------------------------------------------- #
# path → 折线
# --------------------------------------------------------------------------- #
def parse_path(d: str) -> list[tuple[float, float]]:
    """把 SVG path 的 ``d`` 展平为折线点集。

    支持 M/m L/l H/h V/v C/c S/s Q/q T/t A/a Z/z（本项目 24 条赛道资产实际用到的全集）。
    """
    toks = [(m.group(1), m.group(2)) for m in _TOK.finditer(d)]
    pts: list[tuple[float, float]] = []
    i = 0
    cur: tuple[float, float] = (0.0, 0.0)
    start: tuple[float, float] = (0.0, 0.0)
    prev_cubic: tuple[float, float] | None = None
    prev_quad: tuple[float, float] | None = None
    cmd: str | None = None

    def num() -> float:
        nonlocal i
        while i < len(toks) and toks[i][0] is not None:
            i += 1
        value = float(toks[i][1])
        i += 1
        return value

    def more() -> bool:
        return i < len(toks) and toks[i][0] is None

    while i < len(toks):
        if toks[i][0] is not None:
            cmd = toks[i][0]
            i += 1
        c = cmd
        # 平滑曲线（S/s、T/t）的反射控制点只在同族命令连续时有效
        if c not in ("C", "c", "S", "s"):
            prev_cubic = None
        if c not in ("Q", "q", "T", "t"):
            prev_quad = None
        if c in "Mm":
            x, y = num(), num()
            if c == "m":
                x += cur[0]
                y += cur[1]
            cur = (x, y)
            start = cur
            pts.append(cur)
            cmd = "l" if c == "m" else "L"
        elif c in "Ll":
            while more():
                x, y = num(), num()
                if c == "l":
                    x += cur[0]
                    y += cur[1]
                cur = (x, y)
                pts.append(cur)
        elif c in "Hh":
            while more():
                x = num()
                if c == "h":
                    x += cur[0]
                cur = (x, cur[1])
                pts.append(cur)
        elif c in "Vv":
            while more():
                y = num()
                if c == "v":
                    y += cur[1]
                cur = (cur[0], y)
                pts.append(cur)
        elif c in "CcSs":
            while more():
                if c in "Cc":
                    x1, y1, x2, y2, x, y = (
                        num(), num(), num(), num(), num(), num(),
                    )
                    if c == "c":
                        x1 += cur[0]
                        y1 += cur[1]
                        x2 += cur[0]
                        y2 += cur[1]
                        x += cur[0]
                        y += cur[1]
                else:
                    x2, y2, x, y = num(), num(), num(), num()
                    if c == "s":
                        x2 += cur[0]
                        y2 += cur[1]
                        x += cur[0]
                        y += cur[1]
                    if prev_cubic is not None:
                        x1 = 2 * cur[0] - prev_cubic[0]
                        y1 = 2 * cur[1] - prev_cubic[1]
                    else:
                        x1, y1 = cur
                pts.extend(_cubic_points(cur, (x1, y1), (x2, y2), (x, y)))
                prev_cubic = (x2, y2)
                cur = (x, y)
        elif c in "QqTt":
            while more():
                if c in "Qq":
                    x1, y1, x, y = num(), num(), num(), num()
                    if c == "q":
                        x1 += cur[0]
                        y1 += cur[1]
                        x += cur[0]
                        y += cur[1]
                else:
                    x, y = num(), num()
                    if c == "t":
                        x += cur[0]
                        y += cur[1]
                    if prev_quad is not None:
                        x1 = 2 * cur[0] - prev_quad[0]
                        y1 = 2 * cur[1] - prev_quad[1]
                    else:
                        x1, y1 = cur
                pts.extend(_quad_points(cur, (x1, y1), (x, y)))
                prev_quad = (x1, y1)
                cur = (x, y)
        elif c in "Aa":
            while more():
                rx, ry, rot = num(), num(), num()
                large_arc, sweep = int(num()), int(num())
                x, y = num(), num()
                if c == "a":
                    x += cur[0]
                    y += cur[1]
                pts.extend(_arc_points(cur, rx, ry, rot, large_arc, sweep, (x, y)))
                cur = (x, y)
        elif c in "Zz":
            pts.append(start)
            cur = start
        else:
            i += 1
    return pts


def _cumulative(poly: list[tuple[float, float]]) -> tuple[list[float], float]:
    cum = [0.0]
    for k in range(len(poly) - 1):
        cum.append(cum[-1] + math.hypot(poly[k + 1][0] - poly[k][0],
                                        poly[k + 1][1] - poly[k][1]))
    return cum, cum[-1] or 1.0


def nearest_arc_fraction(
    point: tuple[float, float], poly: list[tuple[float, float]],
    cum: list[float], total: float,
) -> float:
    """锚点投影到折线后的累计弧长占比。"""
    px, py = point
    best_d, best_s = float("inf"), 0.0
    for k in range(len(poly) - 1):
        ax, ay = poly[k]
        bx, by = poly[k + 1]
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
        dist = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
        if dist < best_d:
            best_d = dist
            best_s = cum[k] + t * math.hypot(dx, dy)
    return best_s / total


def compute_arcs() -> dict[str, dict[int, float]]:
    """重新计算全部赛道的弯道弧长占比（生成与 CI 校验共用同一实现）。"""
    anchors = load_anchors()
    result: dict[str, dict[int, float]] = {}
    for track_id, corners in sorted(anchors.items()):
        svg = (SVG_DIR / f"{track_id}.svg").read_text(encoding="utf-8")
        paths = re.findall(r'<path[^>]*\sd="([^"]+)"', svg)
        poly = parse_path(max(paths, key=len))
        cum, total = _cumulative(poly)
        result[track_id] = {
            number: round(nearest_arc_fraction(point, poly, cum, total), 6)
            for number, point in sorted(corners.items())
        }
    return result


def render_module(arcs: dict[str, dict[int, float]]) -> str:
    """渲染 ``_track_arcs.py`` 内容。"""
    lines = [
        '"""各赛道弯道锚点在 SVG 路径上的累计弧长占比（自动生成，请勿手工编辑）。',
        "",
        "由 ``scripts/gen_track_arcs.py`` 生成；``tests/test_track_arcs.py`` 会在 CI 中用同一",
        "算法重新计算并逐条比对，保证本文件与 ``ui/tracks/*.svg`` 不脱节。",
        "",
        "用途：``api/ws.py::_map_corner`` 依据圈内距离 (m) → 占比，取弧长最近的弯道，",
        "取代早期「弯道均匀分布」的近似（云端实测错误率 69%）。",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "# {track_id: {corner_number: arc_fraction}}，arc_fraction ∈ [0, 1)",
        "TRACK_CORNER_ARCS: dict[str, dict[int, float]] = {",
    ]
    for track_id, corners in arcs.items():
        lines.append(f'    "{track_id}": {{')
        lines.extend(f"        {number}: {fraction}," for number, fraction in corners.items())
        lines.append("    },")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    arcs = compute_arcs()
    # newline 固定为 LF，避免 Windows 文本模式写出 CRLF 导致整文件 diff
    OUT_PATH.write_text(render_module(arcs), encoding="utf-8", newline="\n")
    total = sum(len(v) for v in arcs.values())
    print(f"wrote {OUT_PATH} ({len(arcs)} tracks, {total} corners)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
