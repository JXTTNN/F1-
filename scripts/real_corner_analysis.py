"""真实赛道几何弯道分析（对照官方弯数，验证每个编号弯道都是真弯）。

数据源：bacinger/f1-circuits（GPS 实测几何，24 条赛道与官方圈长一致）。

方法（转向速率分段，避免固定窗口偏袒急弯、掩盖大半径快弯）：
1. 经纬度 → 本地平面坐标（米），闭合环加密采样；
2. 逐点转向速率（°/m，15m 滑动平均去噪）；
3. 速率超过 RATE_FLOOR 的连续区段 = 转弯段；相邻段间隔 < MERGE_GAP 则合并；
4. 累计转角 >= CORNER_MIN_DEG 的段记为弯道（官方口径逐 apex 计数）。

用法::

    python scripts/real_corner_analysis.py            # 全部赛道报表
    python scripts/real_corner_analysis.py monza      # 单条调试
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from setup_tuner.domain._track_official import OFFICIAL_TURN_COUNTS  # noqa: E402

REAL_DIR = Path(__file__).resolve().parent / "real_circuits"

#: track_id -> geojson 文件名
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

# 分析参数
DENSIFY_STEP = 2.0        # 采样步长（米）
SMOOTH_M = 15.0           # 转向速率去噪窗口（米）
RATE_FLOOR = 0.35         # 转向速率下限（°/m），≈半径 164m 以内才计为弯内
MERGE_GAP = 25.0          # 转弯段间隔小于此（米）合并为同一弯
CORNER_MIN_DEG = 25.0     # 弯道累计转角下限（度）


def load_coords(track_id: str) -> list[tuple[float, float]]:
    path = REAL_DIR / f"{TRACK_GEOJSON[track_id]}.geojson"
    d = json.loads(path.read_text(encoding="utf-8"))
    return [(c[0], c[1]) for c in d["features"][0]["geometry"]["coordinates"]]


def to_local_m(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    lat0 = sum(c[1] for c in coords) / len(coords)
    kx = 111320.0 * math.cos(math.radians(lat0))
    ky = 110540.0
    return [(c[0] * kx, c[1] * ky) for c in coords]


def densify_closed(pts: list[tuple[float, float]], step: float) -> list[tuple[float, float]]:
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


def _bearing(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def _norm180(d: float) -> float:
    while d > 180.0:
        d -= 360.0
    while d <= -180.0:
        d += 360.0
    return d


def analyze(track_id: str) -> dict:
    """返回真实几何的弯道簇列表与圈长。"""
    pts = densify_closed(to_local_m(load_coords(track_id)), DENSIFY_STEP)
    n = len(pts)
    total = 0.0
    seg_len: list[float] = []
    brg: list[float] = []
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        d = math.hypot(x2 - x1, y2 - y1)
        seg_len.append(d)
        total += d
        brg.append(_bearing(x1, y1, x2, y2))
    # 逐采样点的方向变化率（°/m），平滑去噪
    rate: list[float] = [0.0] * n
    for i in range(n):
        rate[i] = abs(_norm180(brg[(i + 1) % n] - brg[i])) / max(seg_len[i], 1e-6)
    smooth_w = max(1, int(SMOOTH_M / DENSIFY_STEP))
    rate_s = [0.0] * n
    for i in range(n):
        s = 0.0
        for k in range(-smooth_w // 2, smooth_w // 2 + 1):
            s += rate[(i + k) % n]
        rate_s[i] = s / smooth_w
    # 转弯段：速率超阈值（允许段内 <1m 级别的瞬时低于阈值，靠 MERGE_GAP 合并）
    in_corner = [rate_s[i] >= RATE_FLOOR for i in range(n)]
    # 弧长位置
    cum = [0.0] * (n + 1)
    for i in range(n):
        cum[i + 1] = cum[i] + seg_len[i]
    # 提取连续段并合并
    segments: list[list[int]] = []
    i = 0
    while i < n:
        if in_corner[i]:
            j = i
            while j < n and in_corner[j]:
                j += 1
            segments.append([i, j - 1])
            i = j
        else:
            i += 1
    # 环形合并（首尾段相邻）
    if len(segments) >= 2:
        first, last = segments[0], segments[-1]
        gap = (cum[n] - cum[last[1]]) + cum[first[0]]
        if gap < MERGE_GAP:
            segments[0][0] = last[0]
            segments.pop()
    merged: list[list[int]] = []
    for s in segments:
        if merged and cum[s[0]] - cum[merged[-1][1]] < MERGE_GAP:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    # 弯道：段累计转角达阈值；段内转角过零（左-右）时按符号切开
    corners: list[dict] = []
    for a, b in merged:
        total_deg = 0.0
        parts: list[float] = []
        run = 0.0
        last_sign = 0
        for i in range(a, b + 1):
            diff = _norm180(brg[(i + 1) % n] - brg[i])
            sign = 1 if diff > 0 else -1
            if last_sign == 0:
                last_sign = sign
            if sign != last_sign and abs(run) > 8.0:
                parts.append(run)
                run = 0.0
                last_sign = sign
            run += diff
        if abs(run) > 1e-9:
            parts.append(run)
        for pdeg in parts:
            total_deg += abs(pdeg)
        if total_deg < CORNER_MIN_DEG:
            continue
        mid = (a + b) // 2
        corners.append({
            "frac_start": cum[a] / total,
            "frac": cum[mid] / total,
            "frac_end": cum[b] / total,
            "deg": total_deg,
            "len_m": cum[b] - cum[a],
        })
    corners.sort(key=lambda c: c["frac_start"])
    return {"track_id": track_id, "total_m": total, "corners": corners}


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    ids = [only] if only else sorted(TRACK_GEOJSON)
    print(f"{'track':14s} {'官方':>4s} {'真实':>4s}  状态   逐弯累计转角（°）与位置")
    bad: list[str] = []
    for tid in ids:
        r = analyze(tid)
        m = len(r["corners"])
        n_off = OFFICIAL_TURN_COUNTS.get(tid, (0, None))[0]
        mark = "OK" if m == n_off else "!!"
        if m != n_off:
            bad.append(f"{tid}: 官方{n_off} vs 真实{m}")
        detail = "  ".join(
            f"[{c['frac']*100:4.1f}%]{c['deg']:>3.0f}°" for c in r["corners"]
        )
        print(f"{tid:14s} {n_off:>4d} {m:>4d}  {mark:4s}  {detail}")
    print()
    if bad:
        print("数量不一致：")
        for b in bad:
            print(" -", b)
    else:
        print("全部赛道：真实弯道数 == 官方弯数 ✓")


if __name__ == "__main__":
    main()
