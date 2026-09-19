"""弧长表复算与 SVG 交叉校验 —— 真实几何口径（task-80 重写）。

2026-09-17 起 ``setup_tuner/domain/_track_arcs.py`` 由
``scripts/gen_real_tracks.py`` 用真实 GPS 几何（``scripts/real_circuits/*.geojson``，
起点 = 发车线、方向 = 官方行驶方向）同源生成：

- 锚点 = 官方弯数 N 个实测转弯特征点（``scripts/real_geometry.py::select_anchors``，
  每点 ±40m 累计转角 >= 12°、官方每个 >= 60° 弯段必有锚点）；
- 弧长占比 = 锚点真实圈内距离 / 真实圈长，与遥测 ``lap_distance`` 同一口径。

本脚本保留两项职责：

1. ``compute_arcs()`` —— 用与生成器**完全相同**的算法复算全部占比，供
   ``tests/test_track_arcs.py`` 在 CI 中与已提交数据逐条比对（容差 1e-3），
   防止数据文件与生成逻辑脱节；
2. ``verify_svg_consistency()`` —— 读取 ``ui/tracks/*.svg`` 的**实际路径**，
   把 ``_track_anchors`` 的像素坐标投影回 SVG 路径反算占比，与真实几何
   口径交叉验证。SVG 是真实几何的等比投影，两条独立计算路径应当给出
   一致占比 —— 这是「SVG 可见层 / 锚点层 / 弧长层」三层同源的独立证据。

已废弃（2026-09-17 删除）：旧「SVG 曲率峰值法」（在示意 SVG 上找转角局部
极大、凑官方弯数）。示意 SVG 采样噪声大、需按赛道调参凑峰数，街道赛
缓弯经常漏检，``_KNOWN_WEAK_ANCHORS`` 17 项人工豁免即其产物；真实几何
口径下豁免整体不再需要。

用法::

    python scripts/gen_track_arcs.py      # 复算 + 交叉验证（只读校验器）
    python scripts/gen_real_tracks.py     # 重建 SVG/锚点/弧长三件套（唯一写入者）
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SVG_DIR = REPO_ROOT / "setup_tuner" / "ui" / "tracks"

# real_geometry 与本文件同目录；tests 用 importlib 从路径加载本模块，
# scripts/ 不在 pytest 的 sys.path 上
sys.path.insert(0, str(Path(__file__).resolve().parent))

#: SVG 反算占比与数据表的最大允许偏差（SVG 为真实几何等比投影，
#: 偏差仅来自路径抽稀与像素取整）
SVG_CONSISTENCY_TOL = 0.01

_TOK = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|(-?\d*\.?\d+(?:[eE][-+]?\d+)?)")
_CUBIC_SAMPLES = 10
_QUAD_SAMPLES = 8
_ARC_SAMPLES = 10

# 闭合回路起跑线归一化容差（见 :func:`nearest_arc_fraction`）：
#   CLOSURE_ANCHOR_TOL —— 锚点与回路闭合点的像素距离上限（锚点坐标保留 2 位小数）
#   CLOSURE_ARC_TOL    —— 最近点弧长与回路总长的相对偏差上限
CLOSURE_ANCHOR_TOL = 0.5
CLOSURE_ARC_TOL = 5e-4


# --------------------------------------------------------------------------- #
# path → 折线（供测试从 SVG 反算闭合点/累计转角；新 SVG 为 M/L/Z 折线）
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
            mt ** 3 * p0[1] + 3 * mt * t * t * p2[1] + t ** 3 * p3[1],
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


def parse_path(d: str) -> list[tuple[float, float]]:
    """把 SVG path 的 ``d`` 展平为折线点集。

    支持 M/m L/l H/h V/v C/c S/s Q/q T/t A/a Z/z（SVG 全集）。
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
    """锚点投影到折线后的累计弧长占比。

    .. note::
        赛道 SVG 是**闭合回路**（折线终点 == 起点）。当锚点恰好落在
        起跑线/终点线（即折线首尾重合点）时，最近点在数值上会命中
        最后一段的末端，返回 1.0 而非 0.0 —— 这会让 T1 被判定为全圈
        最后一个弯，造成整条赛道的弯号整体错位。

        因此这里对"投影到回路闭合点"的情况做归一化：若最近点落在
        折线末端且与起点重合（闭合回路），视其弧长为 0。
    """
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

    # 闭合回路归一化：锚点落在起跑线（回路闭合点）时，最近点在数值上
    # 会命中最后一段末端，返回 ≈1.0。此时应视其弧长为 0（起跑线）。
    is_closed = math.hypot(poly[0][0] - poly[-1][0], poly[0][1] - poly[-1][1]) < 1e-6
    if is_closed:
        d_start = math.hypot(px - poly[0][0], py - poly[0][1])
        near_closure = (
            (d_start <= CLOSURE_ANCHOR_TOL and best_s <= CLOSURE_ARC_TOL * total)
            or best_s >= (1.0 - CLOSURE_ARC_TOL) * total
        )
        if near_closure and d_start <= CLOSURE_ANCHOR_TOL:
            return 0.0
    return best_s / total


def _dense_path(track_id: str) -> tuple[list[tuple[float, float]], list[float], float]:
    """读取赛道主 path 并采样为折线 + 累计弧长。"""
    svg = (SVG_DIR / f"{track_id}.svg").read_text(encoding="utf-8")
    paths = re.findall(r'<path[^>]*\sd="([^"]+)"', svg)
    poly = parse_path(max(paths, key=len))
    cum, total = _cumulative(poly)
    return poly, cum, total


def _turning_series(
    poly: list[tuple[float, float]],
) -> list[float]:
    """逐点转角幅值（0..π）：相邻两段方向的夹角，越大越接近弯心。"""
    n = len(poly)
    out = [0.0] * n
    for i in range(1, n - 1):
        ax = poly[i][0] - poly[i - 1][0]
        ay = poly[i][1] - poly[i - 1][1]
        bx = poly[i + 1][0] - poly[i][0]
        by = poly[i + 1][1] - poly[i][1]
        na = math.hypot(ax, ay) or 1.0
        nb = math.hypot(bx, by) or 1.0
        cosang = (ax * bx + ay * by) / (na * nb)
        out[i] = math.acos(max(-1.0, min(1.0, cosang)))
    out[0] = out[-1] = 0.0
    return out


def _cumulative_turn(
    raw: list[float], cum: list[float], total: float, window: float,
) -> list[float]:
    """每个顶点处、弧长窗口内的**累计转角**（弧度，可取 >2π）。

    弯的正确度量：一段弧上所有单点转角之和。直道≈0，发夹≈π 以上。
    """
    import bisect
    half = window * total / 2.0
    out: list[float] = []
    for x in cum:
        lo = bisect.bisect_left(cum, x - half)
        hi = bisect.bisect_right(cum, x + half)
        out.append(sum(raw[lo:hi]))
    return out


# --------------------------------------------------------------------------- #
# 真实几何复算（与 scripts/gen_real_tracks.py 完全同一算法）
# --------------------------------------------------------------------------- #
def compute_arcs() -> dict[str, dict[int, float]]:
    """复算全部赛道的弯道弧长占比（与生成器共用 ``real_geometry`` 实现）。

    CI 中 ``tests/test_track_arcs.py`` 用本函数重算并与已提交的
    ``_track_arcs.py`` 逐条比对（容差 1e-3），保证数据与生成逻辑不脱节。
    选不满官方弯数时 ``select_anchors`` 抛 ``RuntimeError`` —— 宁可失败
    不可编造。
    """
    import real_geometry as rg

    from setup_tuner.domain._track_official import OFFICIAL_TURN_COUNTS

    result: dict[str, dict[int, float]] = {}
    for tid in sorted(rg.TRACK_GEOJSON):
        n_off = OFFICIAL_TURN_COUNTS[tid][0]
        rt = rg.RealTrack(tid)
        idxs = rg.select_anchors(rt, n_off)
        result[tid] = {
            pos + 1: round(rt.cum[idx] / rt.total_m, 6)
            for pos, idx in enumerate(idxs)
        }
    return result


# --------------------------------------------------------------------------- #
# 三层同源交叉验证：SVG 实际路径反算 vs 数据表
# --------------------------------------------------------------------------- #
def verify_svg_consistency() -> tuple[float, list[str]]:
    """把 ``_track_anchors`` 像素坐标投影回 ``ui/tracks/*.svg`` 实际路径，
    反算占比并与 ``_track_arcs`` 数据表比对。

    Returns:
        ``(worst_diff, issues)``：最大偏差与超差清单（空 = 三层一致）。
    """
    from setup_tuner.domain._track_anchors import TRACK_ANCHORS
    from setup_tuner.domain._track_arcs import TRACK_CORNER_ARCS

    issues: list[str] = []
    worst = 0.0
    for tid, anchors in sorted(TRACK_ANCHORS.items()):
        poly, cum, total = _dense_path(tid)
        for number, (x, y) in anchors.items():
            frac = nearest_arc_fraction((x, y), poly, cum, total)
            committed = TRACK_CORNER_ARCS[tid][number]
            diff = abs(frac - committed)
            worst = max(worst, diff)
            if diff > SVG_CONSISTENCY_TOL:
                issues.append(
                    f"{tid} T{number}: SVG 反算 {frac:.4f} vs 数据 {committed:.4f}"
                )
    return worst, issues


def main() -> int:
    """只读校验：复算一致 + SVG 交叉一致（写盘职责归 gen_real_tracks.py）。"""
    from setup_tuner.domain._track_arcs import TRACK_CORNER_ARCS

    arcs = compute_arcs()
    drift = [
        f"{tid} T{num}: 提交 {TRACK_CORNER_ARCS[tid][num]:.4f} vs 复算 {frac:.4f}"
        for tid, corners in arcs.items() for num, frac in corners.items()
        if abs(TRACK_CORNER_ARCS[tid][num] - frac) > 1e-3
    ]
    worst, issues = verify_svg_consistency()
    if drift or issues:
        print("数据脱节：")
        for line in drift + issues:
            print(" ", line)
        return 1
    print(
        f"OK: {len(arcs)} 条赛道 / {sum(len(v) for v in arcs.values())} 个锚点，"
        f"复算与提交数据一致；SVG 实际路径反算最大偏差 {worst:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
