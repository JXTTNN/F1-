"""把 2026 真实遥测按官方弯道锚点切成**逐弯样本**（调教模型的训练数据来源）。

为什么按弯切而不是整圈平均
--------------------------
用户要求"训练逻辑不是机械覆盖赛道，要针对弯道、轮胎、天气、赛车调教、
车手风格"。整圈平均只有一个样本/圈，且把慢弯与快弯混在一起；按官方弯道
锚点把每一切成 N 个逐弯样本后：

- 样本量放大一个量级（每圈 N 个弯）；
- 每个样本带**弯道类别**（慢/中/快），模型才能学到"同样的胎/天气下，
  不同弯型的表现差异"；
- 目标量是**该弯的通过时间**，是可从遥测直接测量的物理量，不是编出来的标签。

数据来源（全部 2026，年检由 fetch_official_2026.py 保证）
--------------------------------------------------------
``.ref/tracing_2026/<Race>/<Session>/<DRV>/<lap>_tel.json``
    ``tel`` 内含逐点 time / speed / throttle / brake / drs / gear / rpm /
    distance / acc_x,y,z。
``.ref/tracing_2026/<Race>/<Session>/session_laptimes.json``
    逐圈配方 / 天气 / 圈速（列式结构）。

切弯口径
--------
弯的参考点由 ``domain/_track_arcs.TRACK_CORNER_ARCS`` 给出（真实 GPS 弧长
占比，与遥测 ``distance`` 同源）。第 i 个弯的窗口 = 与相邻弯参考点的**中点**
之间（首尾环绕），因此弯窗口连续且不重叠。

产出
----
``data/training/corner_dataset.jsonl``  逐弯样本（每行一个 JSON）
``data/training/corner_dataset_meta.json``  统计与特征口径

用法::

    python scripts/build_corner_dataset.py                 # 每赛道每会话取 120 圈
    python scripts/build_corner_dataset.py --max-laps 400
    python scripts/build_corner_dataset.py --races "Italian Grand Prix"
"""

from __future__ import annotations

import argparse
import bisect
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / ".ref" / "tracing_2026"
OUT_JSONL = ROOT / "data" / "training" / "corner_dataset.jsonl"
OUT_META = ROOT / "data" / "training" / "corner_dataset_meta.json"

#: 弯窗口解析所需的最少采样点数（低于此视为脏数据，丢弃该弯）
MIN_POINTS_IN_WINDOW = 4


def _race_to_track() -> dict[str, Any]:
    from setup_tuner.domain.track import ALL_TRACKS

    return {t.official_name: t for t in ALL_TRACKS}


def _num_list(vals: Any) -> list[float]:
    """把遥测列转 float 列表，剔除 None / 'None' / 非数值。"""
    out: list[float] = []
    if not isinstance(vals, list):
        return out
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and v.strip().lower() == "none":
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def _corner_windows(
    arcs: dict[int, float], min_width_frac: float = 0.025,
) -> list[tuple[int, float, float]]:
    """由锚点弧长占比算出每个弯的 `(弯号, 起点占比, 终点占比)`（首尾环绕）。

    窗口边界取相邻锚点的中点；但**必须再保证最小宽度**：
    有些赛道的相邻锚点弧长极其接近（Monaco T1 紧贴 T19/T2），纯中点会把
    T1 切成约 0.4% 圈长（十几米）—— 实测该窗口算出的"T1 用时 0.28s"完全不物理。
    因此对过窄的窗口做**对称扩张**（从两侧邻居各借一点宽度），迭代若干轮后
    仍不满足的用下限兜底，保证每个弯都有足够采样点。

    Args:
        arcs: ``{弯号: 该弯参考点的弧长占比}``。
        min_width_frac: 每个弯窗口的最小宽度（占圈长比例），默认 2.5%。

    Returns:
        ``[(弯号, 起点占比, 终点占比), ...]``，按弯号升序；窗口连续、不重叠。
    """
    if not arcs:
        return []
    nums = sorted(arcs)
    if len(nums) == 1:
        return [(nums[0], 0.0, 1.0)]
    n = len(nums)
    # 相邻锚点中点作为边界。**必须保持"弯号顺序"**（不能排序！）：
    # mids[i] 是"第 i 个弯的终点边界"，排序会把中点与弯号的对应关系打乱，
    # 实测会让首个弯的窗口宽度变成负值（整圈被算成 -96%）。
    # 锚点弧长本身随弯号单调递增，因此未展开坐标下的中点序列天然单调，
    # 只需对浮点误差做一次单调性兜底。
    mids: list[float] = []
    for i, num in enumerate(nums):
        nxt = nums[(i + 1) % n]
        a, b = arcs[num], arcs[nxt]
        if i == n - 1:
            b += 1.0  # 跨过发车线（最后一个弯 → 第一个弯）
        mids.append((a + b) / 2.0)
    bounds = list(mids)
    for i in range(1, n):
        if bounds[i] <= bounds[i - 1]:
            bounds[i] = bounds[i - 1] + 1e-6
    # 迭代扩张过窄窗口：把两侧边界各外推 need/2，同时保证邻居不被压到过窄
    floor_frac = min_width_frac * 0.5
    for _ in range(8):
        changed = False
        for i in range(n):
            start = bounds[i - 1] if i > 0 else bounds[-1] - 1.0
            end = bounds[i]
            span = end - start
            if span >= min_width_frac:
                continue
            need = (min_width_frac - span) / 2.0
            if i > 0:
                prev_start = bounds[i - 2] if i > 1 else bounds[-1] - 1.0
                move = min(need, max(0.0, (bounds[i - 1] - prev_start) - floor_frac))
                bounds[i - 1] -= move
            else:
                move = min(need, max(0.0, (bounds[-1] - bounds[-2]) - floor_frac))
                bounds[-1] += move
            nxt_span = (bounds[i + 1] - bounds[i]) if i + 1 < n \
                else (bounds[0] + 1.0 - bounds[i])
            move2 = min(need, max(0.0, nxt_span - floor_frac))
            bounds[i] += move2
            changed = changed or move > 0 or move2 > 0
        if not changed:
            break
    out: list[tuple[int, float, float]] = []
    for i, num in enumerate(nums):
        start = bounds[i - 1] if i > 0 else bounds[-1] - 1.0
        out.append((num, start, bounds[i]))
    return out


def _lap_meta(laptimes: dict[str, Any], driver: str, lap: int) -> dict[str, Any]:
    """取某车某圈的配方/天气/圈速元数据（外部列式结构）。"""
    drv = laptimes.get("drv") or []
    laps = laptimes.get("lap") or []
    idx = next(
        (i for i, (d, n) in enumerate(zip(drv, laps, strict=False))
         if d == driver and n == lap),
        None,
    )
    if idx is None:
        return {}

    def col(name: str) -> Any:
        v = laptimes.get(name)
        raw = v[idx] if isinstance(v, list) and idx < len(v) else None
        if isinstance(raw, str):
            return None if raw.strip().lower() in ("none", "") else raw
        return raw

    return {
        "compound": col("compound"),
        "tyre_life": col("life"),
        "lap_time_s": col("time"),
        "air_temp": col("wAT"),
        "track_temp": col("wTT"),
        "rainfall": col("wR"),
    }


#: 配方字符串 → 归类（软/中/硬/湿）
_COMPOUND_CLASS: dict[str, str] = {
    "SOFT": "soft", "MEDIUM": "medium", "HARD": "hard",
    "INTERMEDIATE": "inter", "WET": "wet",
}


def _compound_class(raw: Any) -> tuple[str, float]:
    """配方 → (类别, 软度 0..1)。未知配方返回 ('unknown', 0.5)。"""
    if not isinstance(raw, str):
        return "unknown", 0.5
    key = raw.strip().upper()
    cls = _COMPOUND_CLASS.get(key, "unknown")
    softness = {"soft": 1.0, "medium": 0.6, "hard": 0.3, "inter": 0.2, "wet": 0.1}
    return cls, softness.get(cls, 0.5)


def _corner_features(
    win: tuple[int, float, float],
    dist: list[float], time: list[float], speed: list[float],
    throttle: list[float], brake: list[float], drs: list[float],
    gear: list[float], acc_x: list[float], acc_y: list[float],
    lap_dist: float,
) -> dict[str, Any] | None:
    """从一圈逐点数据里裁出单个弯的特征。"""
    n = len(dist)
    if n < MIN_POINTS_IN_WINDOW or lap_dist <= 0:
        return None
    number, start_frac, end_frac = win
    # 窗口可能跨过发车线（起点为负、终点 > 1）：按**本圈**范围夹取。
    # 跨线部分属于相邻两圈，本圈数据里不存在，直接截断即可（物理上等价于
    # "这个弯在发车线附近的可见部分"）。
    d0 = max(0.0, start_frac * lap_dist)
    d1 = min(lap_dist, end_frac * lap_dist)
    if d1 <= d0:
        return None
    i0 = bisect.bisect_left(dist, d0)
    i1 = bisect.bisect_right(dist, d1)
    if i1 - i0 < MIN_POINTS_IN_WINDOW:
        return None
    seg = slice(i0, i1)

    def _slice(vals: list[float]) -> list[float]:
        return vals[seg] if len(vals) >= i1 else []

    spd = _slice(speed)
    if len(spd) < MIN_POINTS_IN_WINDOW:
        return None
    thr = _slice(throttle)
    brk = _slice(brake)
    drs_s = _slice(drs)
    gr = _slice(gear)
    t = _slice(time)
    ax = _slice(acc_x)
    ay = _slice(acc_y)

    corner_time = (t[-1] - t[0]) if len(t) >= 2 else None
    if corner_time is None or corner_time <= 0:
        return None
    m = len(spd)
    return {
        "corner": number,
        "corner_time_s": round(corner_time, 4),
        "entry_speed": round(spd[0], 2),
        "min_speed": round(min(spd), 2),
        "exit_speed": round(spd[-1], 2),
        "avg_speed": round(sum(spd) / m, 2),
        "speed_range": round(max(spd) - min(spd), 2),
        "throttle_full_pct": round(100.0 * sum(1 for v in thr if v >= 99) / max(1, len(thr)), 2),
        "brake_pct": round(100.0 * sum(1 for v in brk if v > 0) / max(1, len(brk)), 2),
        "drs_pct": round(100.0 * sum(1 for v in drs_s if v > 0) / max(1, len(drs_s)), 2),
        "gear_min": min(gr) if gr else None,
        "gear_max": max(gr) if gr else None,
        "max_lateral_g": round(max((abs(v) for v in ax), default=0.0), 3),
        "max_long_g": round(max((abs(v) for v in ay), default=0.0), 3),
        "points": m,
    }


def build(max_laps: int, races_filter: list[str] | None) -> dict[str, Any]:
    from setup_tuner.domain._track_arcs import TRACK_CORNER_ARCS

    race_to_track = _race_to_track()
    samples: list[dict[str, Any]] = []
    per_track: dict[str, int] = {}
    skipped: dict[str, int] = {}

    def bump(key: str) -> None:
        skipped[key] = skipped.get(key, 0) + 1

    if not CACHE.exists():
        raise SystemExit(f"缓存不存在：{CACHE}（先跑 fetch_official_2026.py）")

    for race_dir in sorted(p for p in CACHE.iterdir() if p.is_dir()):
        race = race_dir.name
        if races_filter and race not in races_filter:
            continue
        track = race_to_track.get(race)
        if track is None:
            bump(f"{race}: 无本地赛道映射")
            continue
        arcs = TRACK_CORNER_ARCS.get(track.track_id)
        if not arcs:
            bump(f"{race}: 无弯道弧长数据")
            continue
        windows = _corner_windows(arcs)
        class_by_number = {c.number: c.corner_type for c in track.corners}

        from setup_tuner.engine.holistic import track_demand

        demand = track_demand(track.track_id)

        for sess_dir in sorted(p for p in race_dir.iterdir() if p.is_dir()):
            lt_path = sess_dir / "session_laptimes.json"
            if not lt_path.exists():
                bump(f"{race}/{sess_dir.name}: 缺 session_laptimes")
                continue
            laptimes = json.loads(lt_path.read_text(encoding="utf-8"))
            # 计数单位是**圈**（早期误按"弯"计数 → 60 个弯就截断，只取到约 4 圈）
            taken_laps = 0
            for drv_dir in sorted(p for p in sess_dir.iterdir() if p.is_dir()):
                if taken_laps >= max_laps:
                    break
                driver = drv_dir.name
                for f in sorted(drv_dir.glob("*_tel.json")):
                    if taken_laps >= max_laps:
                        break
                    try:
                        lap = int(f.name.split("_", 1)[0])
                    except ValueError:
                        continue
                    meta = _lap_meta(laptimes, driver, lap)
                    lap_time = meta.get("lap_time_s")
                    if not isinstance(lap_time, (int, float)) or lap_time <= 0:
                        bump(f"{race}/{sess_dir.name}/{driver}/{lap}: 圈速缺失")
                        continue
                    try:
                        tel = json.loads(f.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        bump(f"{race}/{driver}/{lap}: 遥测不可读")
                        continue
                    body = tel.get("tel") or {}
                    dist = _num_list(body.get("distance"))
                    if len(dist) < 50:
                        bump(f"{race}/{driver}/{lap}: distance 过短")
                        continue
                    lap_dist = max(dist)
                    time = _num_list(body.get("time"))
                    speed = _num_list(body.get("speed"))
                    thr = _num_list(body.get("throttle"))
                    brk = _num_list(body.get("brake"))
                    drs = _num_list(body.get("drs"))
                    gear = _num_list(body.get("gear"))
                    ax = _num_list(body.get("acc_x"))
                    ay = _num_list(body.get("acc_y"))
                    if len(speed) < 50 or len(time) < 50:
                        bump(f"{race}/{driver}/{lap}: time/speed 过短")
                        continue
                    comp_cls, softness = _compound_class(meta.get("compound"))
                    wet = 1.0 if comp_cls in ("inter", "wet") else 0.0
                    track_temp = meta.get("track_temp")
                    air_temp = meta.get("air_temp")
                    lap_emitted = 0
                    for win in windows:
                        feat = _corner_features(
                            win, dist, time, speed, thr, brk, drs, gear, ax, ay,
                            lap_dist,
                        )
                        if feat is None:
                            continue
                        cclass = str(class_by_number.get(win[0], "medium")).lower()
                        samples.append({
                            "source": "external_2026",
                            "race": race,
                            "session": sess_dir.name,
                            "driver": driver,
                            "lap": lap,
                            "track_id": track.track_id,
                            "corner": feat["corner"],
                            "corner_class": cclass,
                            "tyre_class": comp_cls,
                            "tyre_softness": softness,
                            "wet": wet,
                            "track_temp": track_temp if isinstance(track_temp, (int, float)) else None,
                            "air_temp": air_temp if isinstance(air_temp, (int, float)) else None,
                            "track_traction_index": demand.traction_index,
                            "track_aero_index": demand.aero_index,
                            "track_braking_index": demand.braking_index,
                            "track_slow_share": demand.slow_share,
                            "track_fast_share": demand.fast_share,
                            "lap_time_s": float(lap_time),
                            **{k: v for k, v in feat.items() if k != "corner"},
                        })
                        lap_emitted += 1
                    if lap_emitted:
                        taken_laps += 1
            if taken_laps:
                per_track[track.track_id] = per_track.get(track.track_id, 0) + taken_laps
    return {
        "samples": samples,
        "per_track": per_track,
        "skipped": skipped,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="2026 遥测 → 逐弯训练样本")
    ap.add_argument("--max-laps", type=int, default=120,
                    help="每赛道每会话最多取多少圈（默认 120）")
    ap.add_argument("--races", nargs="*", default=None, help="只处理指定比赛")
    ap.add_argument("--out", default=str(OUT_JSONL))
    args = ap.parse_args(argv)

    res = build(args.max_laps, args.races)
    samples = res["samples"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for row in samples:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    classes = {}
    for s in samples:
        classes[s["corner_class"]] = classes.get(s["corner_class"], 0) + 1
    times = [s["corner_time_s"] for s in samples]
    meta = {
        "schema": "f1opt-corner-dataset/1",
        "source": "TracingInsights 2026 (2026-only, year-checked)",
        "count": len(samples),
        "per_track": res["per_track"],
        "by_corner_class": classes,
        "corner_time_s": {
            "min": round(min(times), 3) if times else None,
            "median": round(statistics.median(times), 3) if times else None,
            "max": round(max(times), 3) if times else None,
        },
        "skipped": res["skipped"],
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== 逐弯样本构建 ===")
    print(f"样本数: {len(samples)}")
    print(f"赛道: {len(res['per_track'])}")
    for tid, cnt in sorted(res["per_track"].items(), key=lambda kv: -kv[1]):
        print(f"  {tid:>14}: {cnt:>6} 弯样本")
    print(f"弯型分布: {classes}")
    print(f"写出: {out}")
    print(f"      {OUT_META}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
