"""真实赛道几何单一数据源：加载 → 投影 → 转向率 → 官方 N 锚点选择。

数据：``scripts/real_circuits/*.geojson``（bacinger/f1-circuits，GPS 实测，
24 条赛道圈长与 F1 官方口径一致；frac0 = 发车线，+frac = 官方行驶方向，
已对全部赛道程序化校验）。

保证（select_anchors 的硬门）：
1. 恰好选出官方弯数 N 个锚点；
2. 每个锚点 ±WINDOW_M 米内累计转角 >= MIN_ANCHOR_DEG（点必落在真弯上）；
3. 每个累计转角 >= REGION_COVER_DEG 的真实转弯段至少含 1 个锚点
   （官方弯没有漏标）；
4. 锚点沿 +frac 排序即官方编号 1..N。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REAL_DIR = Path(__file__).resolve().parent / "real_circuits"

#: track_id -> geojson 文件名（不含扩展名）
TRACK_GEOJSON: dict[str, str] = {
    "austin": "us-2012", "baku": "az-2016", "barcelona": "es-1991",
    "hungaroring": "hu-1986", "jeddah": "sa-2021", "las_vegas": "us-2023",
    "lusail": "qa-2004", "madrid": "es-2026", "melbourne": "au-1953",
    "mexico_city": "mx-1962", "miami": "us-2022", "monaco": "mc-1929",
    "montreal": "ca-1978", "monza": "it-1922", "sakhir": "bh-2002",
    "sao_paulo": "br-1940", "shanghai": "cn-2004", "silverstone": "gb-1948",
    "singapore": "sg-2008", "spa": "be-1925", "spielberg": "at-1969",
    "suzuka": "jp-1962", "yas_marina": "ae-2009", "zandvoort": "nl-1948",
}

# ---- 参数（单位：米 / 度）----
DENSIFY_STEP = 1.0        # 采样步长
SMOOTH_M = 12.0           # 转向速率去噪窗口
RATE_FLOOR = 0.30         # 转弯段速率下限（°/m），≈半径 191m
REGION_MERGE_GAP = 25.0   # 转弯段合并间隔
REGION_MIN_DEG = 15.0     # 记为转弯段的累计转角下限
ANCHOR_SEP_M = 35.0       # 锚点最小间距（减速弯相邻 apex 约 35-60m）
ANCHOR_SEP_MIN_M = 18.0   # 求不满时的放宽下限
WINDOW_M = 40.0           # 锚点强度评估窗口（±米）
MIN_ANCHOR_DEG = 12.0     # 锚点强度硬门（官方口径含高速缓弯，如 Spa/Zandvoort）
REGION_COVER_DEG = 60.0   # 必须含锚点的转弯段下限

CANVAS_W = 800
CANVAS_H = 600
CANVAS_MARGIN = 40


def load_coords(track_id: str) -> list[tuple[float, float]]:
    path = REAL_DIR / f"{TRACK_GEOJSON[track_id]}.geojson"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(c[0], c[1]) for c in data["features"][0]["geometry"]["coordinates"]]


def to_local_m(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """经纬度 → 本地平面坐标（米），y 向北。"""
    lat0 = sum(c[1] for c in coords) / len(coords)
    kx = 111320.0 * math.cos(math.radians(lat0))
    ky = 110540.0
    return [(c[0] * kx, c[1] * ky) for c in coords]


def densify_closed(
    pts: list[tuple[float, float]], step: float = DENSIFY_STEP,
) -> list[tuple[float, float]]:
    """闭合环等距加密采样（首点即 frac0）。"""
    ring = pts + [pts[0]]
    out: list[tuple[float, float]] = []
    for (x1, y1), (x2, y2) in zip(ring, ring[1:], strict=False):
        d = math.hypot(x2 - x1, y2 - y1)
        if d < 1e-9:
            continue
        n = max(1, int(round(d / step)))
        for i in range(n):
            t = i / n
            out.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
    return out


def _bearing_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def _norm180(d: float) -> float:
    while d > 180.0:
        d -= 360.0
    while d <= -180.0:
        d += 360.0
    return d


class RealTrack:
    """真实几何的派生数据（加密点、转向率、转弯段）。"""

    def __init__(self, track_id: str) -> None:
        self.track_id = track_id
        self.pts = densify_closed(to_local_m(load_coords(track_id)))
        n = len(self.pts)
        self.n = n
        seg_len: list[float] = []
        brg: list[float] = []
        self.total_m = 0.0
        for i in range(n):
            x1, y1 = self.pts[i]
            x2, y2 = self.pts[(i + 1) % n]
            d = math.hypot(x2 - x1, y2 - y1)
            seg_len.append(d)
            self.total_m += d
            brg.append(_bearing_deg(x1, y1, x2, y2))
        self.seg_len = seg_len
        self.cum = [0.0] * (n + 1)
        for i in range(n):
            self.cum[i + 1] = self.cum[i] + seg_len[i]
        # 逐点转向速率（°/m）+ 滑动平均去噪
        rate = [
            abs(_norm180(brg[(i + 1) % n] - brg[i])) / max(seg_len[i], 1e-6)
            for i in range(n)
        ]
        w = max(1, int(SMOOTH_M / DENSIFY_STEP))
        self.rate = [
            sum(rate[(i + k) % n] for k in range(-(w // 2), w // 2 + 1)) / w
            for i in range(n)
        ]
        self.regions = self._regions()

    # ---- 转弯段 ----
    def _regions(self) -> list[dict]:
        n = self.n
        in_c = [self.rate[i] >= RATE_FLOOR for i in range(n)]
        segs: list[list[int]] = []
        i = 0
        while i < n:
            if in_c[i]:
                j = i
                while j < n and in_c[j]:
                    j += 1
                segs.append([i, j - 1])
                i = j
            else:
                i += 1
        if len(segs) >= 2:
            gap_wrap = (self.cum[n] - self.cum[segs[-1][1]]) + self.cum[segs[0][0]]
            if gap_wrap < REGION_MERGE_GAP:
                segs[0][0] = segs[-1][0]
                segs.pop()
        merged: list[list[int]] = []
        for s in segs:
            if merged and self.cum[s[0]] - self.cum[merged[-1][1]] < REGION_MERGE_GAP:
                merged[-1][1] = s[1]
            else:
                merged.append(s)
        regions = []
        for a, b in merged:
            deg = self._turn_between(a, b)
            if deg >= REGION_MIN_DEG:
                regions.append({"a": a, "b": b, "deg": deg})
        return regions

    def _turn_between(self, a: int, b: int) -> float:
        """[a,b] 内累计转角绝对值（过零按左右弯分段累加）。"""
        total = 0.0
        run = 0.0
        last_sign = 0
        for i in range(a, b + 1):
            d = _norm180(
                _bearing_deg(*self.pts[i], *self.pts[(i + 1) % self.n])
                - _bearing_deg(*self.pts[i - 1], *self.pts[i])
            )
            sign = 1 if d > 0 else -1
            if last_sign == 0:
                last_sign = sign
            if sign != last_sign and abs(run) > 8.0:
                total += abs(run)
                run = 0.0
                last_sign = sign
            run += d
        return total + abs(run)

    def turn_at(self, idx: int, half_window: float = WINDOW_M / 2.0) -> float:
        """idx 处 ±half_window 弧长窗口内的累计转角（度）。"""
        n = self.n
        lo = idx
        while self.cum[idx] - self.cum[lo] < half_window and lo > 0:
            lo -= 1
        hi = idx
        while self.cum[hi] - self.cum[idx] < half_window and hi < n - 1:
            hi += 1
        return self._turn_between(lo, hi)

    def region_of(self, idx: int) -> int | None:
        for k, r in enumerate(self.regions):
            if r["a"] <= idx <= r["b"]:
                return k
        return None


def select_anchors(rt: RealTrack, n_expected: int) -> list[int]:
    """在真实几何上选出恰好 n_expected 个官方弯锚点（采样点下标）。

    硬门见模块 docstring；选不满直接 RuntimeError（宁可失败不可编造）。
    """
    n = rt.n
    # 候选：转向率局部极大（±3m 内最大）
    cands: list[int] = []
    for i in range(n):
        if rt.rate[i] < 1e-6:
            continue
        win = int(3.0 / DENSIFY_STEP)
        if all(rt.rate[i] >= rt.rate[(i + k) % n] for k in range(-win, win + 1)):
            cands.append(i)
    # 去重相邻（环上取等值平台只留一个）
    cands.sort()
    dedup: list[int] = []
    for i in cands:
        if dedup and i - dedup[-1] <= 2:
            continue
        dedup.append(i)
    if len(dedup) >= 2 and dedup[0] == 0 and dedup[-1] >= n - 3:
        dedup.pop()
    cands = dedup

    def try_fill(sep: float) -> list[int]:
        order = sorted(cands, key=lambda i: -rt.turn_at(i))
        picked: list[int] = []
        for i in order:
            if len(picked) >= n_expected:
                break
            if all(rt._cyc_m(i, j) >= sep for j in picked):
                picked.append(i)
        return picked

    picked = try_fill(ANCHOR_SEP_M)
    if len(picked) < n_expected:
        picked = try_fill(ANCHOR_SEP_MIN_M)
    if len(picked) < n_expected:
        raise RuntimeError(
            f"{rt.track_id}: 真实几何只能提供 {len(picked)} 个候选锚点，"
            f"官方需要 {n_expected} —— 拒绝编造"
        )

    # 超额裁剪：同段多锚保最强，全局删最弱；>=60° 的段永不空
    def region_deg_covered(r: dict) -> bool:
        return any(r["a"] <= p <= r["b"] for p in picked)

    def strongest_in(r: dict) -> int:
        members = [p for p in picked if r["a"] <= p <= r["b"]]
        return max(members, key=lambda i: rt.turn_at(i))

    while len(picked) > n_expected:
        droppable = []
        for r in rt.regions:
            members = [p for p in picked if r["a"] <= p <= r["b"]]
            if len(members) >= 2:
                keep = strongest_in(r)
                droppable.extend(p for p in members if p != keep)
        if not droppable:
            droppable = [
                p for p in picked
                if not any(
                    r["a"] <= p <= r["b"] and r["deg"] >= REGION_COVER_DEG
                    and len([q for q in picked if r["a"] <= q <= r["b"]]) == 1
                    for r in rt.regions
                )
            ]
        if not droppable:
            break
        victim = min(droppable, key=lambda i: rt.turn_at(i))
        picked.remove(victim)

    # 覆盖门：>=60° 的段必须有锚点，没有就强制补段内 argmax
    for r in rt.regions:
        if r["deg"] < REGION_COVER_DEG:
            continue
        if not region_deg_covered(r):
            if len(picked) >= n_expected:
                weak = min(picked, key=lambda i: rt.turn_at(i))
                picked.remove(weak)
            arg = max(range(r["a"], r["b"] + 1), key=lambda i: rt.rate[i])
            picked.append(arg)

    if len(picked) != n_expected:
        raise RuntimeError(
            f"{rt.track_id}: 锚点数 {len(picked)} != 官方 {n_expected}"
        )
    # 硬门：每个锚点必须落在真弯上
    weak = [(i, rt.turn_at(i)) for i in picked if rt.turn_at(i) < MIN_ANCHOR_DEG]
    if weak:
        raise RuntimeError(
            f"{rt.track_id}: 锚点强度不足（< {MIN_ANCHOR_DEG}°）："
            + ", ".join(f"@{rt.cum[i]:.0f}m={d:.0f}°" for i, d in weak)
        )
    return sorted(picked, key=lambda i: i)


def to_canvas(rt: RealTrack, idxs: list[int]) -> tuple[
    list[tuple[float, float]], list[tuple[float, float]],
]:
    """本地米坐标 → 800×600 画布像素（等比、居中、y 翻转朝上）。"""
    xs = [p[0] for p in rt.pts]
    ys = [p[1] for p in rt.pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    scale = min(
        (CANVAS_W - 2 * CANVAS_MARGIN) / max(maxx - minx, 1e-6),
        (CANVAS_H - 2 * CANVAS_MARGIN) / max(maxy - miny, 1e-6),
    )
    ox = (CANVAS_W - (maxx - minx) * scale) / 2.0
    oy = (CANVAS_H - (maxy - miny) * scale) / 2.0

    def px(p: tuple[float, float]) -> tuple[float, float]:
        return (
            round(ox + (p[0] - minx) * scale, 2),
            round(CANVAS_H - (oy + (p[1] - miny) * scale), 2),
        )

    canvas_pts = [px(p) for p in rt.pts]
    anchor_px = [canvas_pts[i] for i in idxs]
    return canvas_pts, anchor_px


def _cyc_m(self, a: int, b: int) -> float:  # noqa: PLW0211
    d = abs(self.cum[a] - self.cum[b])
    return min(d, self.total_m - d)


RealTrack._cyc_m = _cyc_m  # type: ignore[attr-defined]
