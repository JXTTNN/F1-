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

# 闭合回路起跑线归一化容差（见 :func:`nearest_arc_fraction`）：
#   CLOSURE_ANCHOR_TOL —— 锚点与回路闭合点的像素距离上限（锚点坐标保留 1 位小数）
#   CLOSURE_ARC_TOL    —— 最近点弧长与回路总长的相对偏差上限
CLOSURE_ANCHOR_TOL = 0.5
CLOSURE_ARC_TOL = 5e-4


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
    #
    # 判据：
    #   - 折线首尾重合（闭合回路）
    #   - 投影点距离闭合点足够近（像素级容差，锚点本身有小数截断）
    #   - 弧长落在回路末端附近（而非起点附近的普通弯）
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


def compute_arcs() -> dict[str, dict[int, float]]:
    """重新计算全部赛道的弯道弧长占比（生成与 CI 校验共用同一实现）。

    2026-09-17 起**改用曲率峰值法**：弯心 = 转角局部极大点，由赛道 SVG 几何
    自动定位，并以官方弯数作硬校验（见 :func:`detect_apex_fractions`）。

    为什么不再用人工锚点投影（用户实测指出蒙扎）：锚点摆错则全表皆错 ——
    蒙扎 T1（第一减速弯，主直道末端 ≈1100 m）的锚点被摆在起跑线上，弧长
    占比 0.0000，整条赛道弯号**整体前移 ≈19%**，所有"归因到某弯"的遥测实际
    都属于前一个弯。几何法不依赖人工摆放，且峰数必须等于官方弯数。
    """
    from setup_tuner.domain.track import get_all_tracks

    counts = {t.track_id: len(t.corners) for t in get_all_tracks()}
    result, failed = compute_arcs_curvature(counts)
    if failed:
        raise RuntimeError(
            "以下赛道无法用曲率峰值定位出恰好 N 个弯心，"
            "需检查其 SVG 路径几何：" + ", ".join(sorted(failed))
        )
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


# --------------------------------------------------------------------------- #
# 曲率峰值法：由几何自动定位弯心（2026-09-17 起，替代人工锚点投影）
# --------------------------------------------------------------------------- #
# 为什么改（用户实测指出）：
#   人工锚点会放错 —— 蒙扎 T1（第一减速弯，主直道末端 ≈1100 m）的锚点被摆在
#   起跑线上，弧长占比 0.0000，导致整条赛道弯号**整体前移 ≈19%**；凡"遥测归因
#   到某弯"的数据实际都属于前一个弯。受同样影响的还有 melbourne / mexico_city /
#   spielberg（T1 锚点压在起跑线）。
#   改为**曲率峰值自动定位**：弯心 = 转角局部极大点，纯几何决定，不依赖人工摆放；
#   并以**官方弯数**作硬校验（峰数必须等于官方弯数，对不上就报错，不许编造）。

_APEX_WINDOWS = (0.010, 0.015, 0.020, 0.030, 0.045, 0.060, 0.080, 0.110)
_APEX_MIN_SEPS = (0.030, 0.022, 0.015, 0.010, 0.007)
_APEX_THRESH_FRACS = (0.50, 0.42, 0.35, 0.30, 0.25, 0.20, 0.16, 0.12)


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


def _smooth_by_arc(
    values: list[float], cum: list[float], total: float, window: float,
) -> list[float]:
    """按弧长窗口做滑动平均（抑制 SVG 采样噪声造成的锯齿峰）。"""
    import bisect
    half = window * total / 2.0
    out: list[float] = []
    for x in cum:
        lo = bisect.bisect_left(cum, x - half)
        hi = bisect.bisect_right(cum, x + half)
        seg = values[lo:hi]
        out.append(sum(seg) / len(seg) if seg else 0.0)
    return out


def _cyclic_dist(a: float, b: float, total: float) -> float:
    """占比环回距离（赛道是闭合回路，首尾相邻）。"""
    d = abs(a - b) % total
    return min(d, total - d)


def _peaks_by_arc(
    values: list[float], cum: list[float], total: float,
    min_sep: float, threshold: float, n_want: int,
) -> list[float] | None:
    """取**恰好 ``n_want`` 个**最强且相互间隔 ≥ ``min_sep`` 的局部极大。

    关键（早期版本的错误）：不是"调参让候选峰数恰好等于 n"——那样任何一组
    参数都很难命中；而是**按强度从大到小贪心选取，凑满 n 个即停**。
    蒙zilla 实测：转角聚集在 0.10–0.20（=物理上的 T1/T2 第一减速弯）、
    0.35–0.45（T3 Curva Grande）、0.55–0.65（Lesmo）、0.80–0.90（Ascari）、
    0.95–1.00（Parabolica），与真实赛历剖面吻合 —— 几何里本来就有正确答案，
    是选峰方式不对。

    候选不足 ``n_want`` 个时返回 ``None``（调用方须回退，不得编造）。
    """
    n = len(values)
    idxs = [
        i for i in range(1, n - 1)
        if values[i] >= threshold
        and values[i] >= values[i - 1] and values[i] >= values[i + 1]
    ]
    idxs.sort(key=lambda i: (-values[i], cum[i]))
    chosen: list[int] = []
    for i in idxs:
        x = cum[i]
        if all(_cyclic_dist(x, cum[j], total) >= min_sep for j in chosen):
            chosen.append(i)
            if len(chosen) == n_want:
                return sorted(round(cum[k] / total, 6) for k in chosen)
    return None


def detect_apex_fractions(track_id: str, n_expected: int) -> list[float] | None:
    """在赛道几何上定位 ``n_expected`` 个弯心，返回升序弧长占比。

    确定性标定：按「窗口由大到小（越平滑越可信）→ 最小间距由大到小 →
    阈值由高到低」的固定顺序搜索，取**第一个**能产出恰好 ``n_expected``
    个峰的组合。找不到返回 ``None`` —— 调用方**必须**回退，不得编造。
    """
    poly, cum, total = _dense_path(track_id)
    if total <= 0 or n_expected <= 0:
        return None
    raw = _turning_series(poly)
    for min_sep_frac in _APEX_MIN_SEPS:
        for window in _APEX_WINDOWS:
            smoothed = _smooth_by_arc(raw, cum, total, window)
            peak_max = max(smoothed) or 1.0
            for thr_frac in _APEX_THRESH_FRACS:
                fr = _peaks_by_arc(
                    smoothed, cum, total, min_sep_frac * total,
                    thr_frac * peak_max, n_expected,
                )
                if fr is not None:
                    return fr
    return None


def compute_arcs_curvature(
    corner_counts: dict[str, int],
) -> tuple[dict[str, dict[int, float]], list[str]]:
    """对全部赛道做曲率峰值定位。

    Returns:
        ``(arcs, failed)``：``arcs[track_id][corner] = fraction``；
        ``failed`` 为未能用几何定位出恰好 N 个弯心的赛道 —— 调用方
        **必须**对这些赛道保留人工锚点结果并明确报告，不得编造。
    """
    result: dict[str, dict[int, float]] = {}
    failed: list[str] = []
    for track_id, n in sorted(corner_counts.items()):
        fr = detect_apex_fractions(track_id, n)
        if fr is None:
            failed.append(track_id)
            continue
        result[track_id] = {i + 1: f for i, f in enumerate(fr)}
    return result, failed
