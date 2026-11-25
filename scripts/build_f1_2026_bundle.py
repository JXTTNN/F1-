"""构建 F1 2026 遥测 + 调教数据包（data/f1_2026/）。

设计约束（与 .gitignore 的隐私口径一致）
----------------------------------------
仓库明确规定「车手真实遥测与训练数据（个人数据，绝不入库）」：

- 原始逐点遥测（``.ref/tracing_2026``）**不入库**，只保留在本地缓存；
- 入库的只有**去标识化的聚合统计**（按 赛道 × 弯道 取中位数，不含车手/圈号）；
- 调教数据来自 F1Laps 公开页面（``data/setup_laps/f1laps_setups.jsonl``），
  只筛 2026 游戏年版，保留作者署名与来源链接。

产物
----
``data/f1_2026/telemetry_corners.jsonl``
    赛道 × 弯道 的真实 2026 遥测聚合（n / 中位数 / p10 / p90），无车手信息。
``data/f1_2026/setups_f1_26.jsonl``
    F1 26 调教 + 圈速配对（899 条级），20 项参数与项目单位一致。
``data/f1_2026/track_profile.jsonl``
    每站一行：遥测弯速画像 + 调教参数分布 + 最优/典型圈速（把两边 join 起来）。
``data/f1_2026/manifest.json``
    来源、许可、抓取时间、覆盖度、行数与 sha256。

用法::

    python scripts/build_f1_2026_bundle.py
    python scripts/build_f1_2026_bundle.py --max-setups 0   # 不限条数
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORNER_SRC = ROOT / "data" / "training" / "corner_dataset.jsonl"
SETUP_SRC = ROOT / "data" / "setup_laps" / "f1laps_setups.jsonl"
OUT_DIR = ROOT / "data" / "f1_2026"

OUT_CORNERS = OUT_DIR / "telemetry_corners.jsonl"
OUT_SETUPS = OUT_DIR / "setups_f1_26.jsonl"
OUT_PROFILE = OUT_DIR / "track_profile.jsonl"
OUT_MANIFEST = OUT_DIR / "manifest.json"

#: 入库（可提交）的文件：只保留不含车手/圈号的赛道级聚合；
#: 逐弯明细与逐条调教留在本地（见 data/f1_2026/.gitignore），可随时重建。
COMMITTED_FILES: frozenset[str] = frozenset({OUT_PROFILE.name})

#: 遥测聚合取这些数值列的 p10 / 中位数 / p90
CORNER_METRICS: tuple[str, ...] = (
    "entry_speed", "min_speed", "exit_speed", "avg_speed",
    "throttle_full_pct", "brake_pct", "gear_min", "gear_max",
    "max_lateral_g", "max_long_g", "corner_time_s",
)


def _percentile(values: list[float], q: float) -> float:
    """线性插值分位数（与 numpy 默认口径一致，避免额外依赖）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- #
# 1. 遥测 → 赛道 × 弯道聚合
# --------------------------------------------------------------------------- #
def build_corner_stats(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """把逐圈逐弯样本压成 (track, corner) 的聚合统计，去除车手/圈号。

    ``drop_speed_zero``：上游 build_corner_dataset 的 1 号弯窗口会包含
    起跑/记圈点的零速帧（实测 142/6380 条，其中 135 条在 1 号弯），
    会把 entry/min/avg speed 的 p10 拉成 0。聚合时直接剔除这类无效样本，
    并在 manifest 里如实记数（不改上游数据）。
    """
    buckets: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    corner_class: dict[tuple[str, int], str] = {}
    tracks: set[str] = set()
    dropped_zero = 0
    for row in rows:
        if row.get("source") != "external_2026":
            continue
        track = row.get("track_id")
        corner = row.get("corner")
        if not track or not isinstance(corner, int):
            continue
        speeds = [row.get("min_speed"), row.get("avg_speed")]
        if any(not isinstance(v, (int, float)) or float(v) <= 0.0 for v in speeds):
            dropped_zero += 1
            continue
        key = (track, corner)
        tracks.add(track)
        corner_class[key] = row.get("corner_class") or corner_class.get(key, "")
        for metric in CORNER_METRICS:
            value = row.get(metric)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                buckets[key][metric].append(float(value))

    out: list[dict[str, Any]] = []
    for (track, corner), metrics in sorted(buckets.items()):
        n = len(metrics["corner_time_s"])
        if n == 0:
            continue
        row: dict[str, Any] = {
            "track_id": track,
            "corner": corner,
            "corner_class": corner_class.get((track, corner), ""),
            "n_laps": n,
        }
        for metric in CORNER_METRICS:
            values = metrics[metric]
            if not values:
                continue
            row[metric] = {
                "median": _round(statistics.median(values)),
                "p10": _round(_percentile(values, 0.10)),
                "p90": _round(_percentile(values, 0.90)),
            }
        out.append(row)
    return out, dropped_zero


# --------------------------------------------------------------------------- #
# 2. 调教 → 2026 子集与分布
# --------------------------------------------------------------------------- #
def build_setups_2026(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """筛 2026 游戏年版调教，按赛道 + 圈速升序；limit<=0 不截断。"""
    picked = [
        r for r in rows
        if int(r.get("game_year") or 0) == 2026 and r.get("setup")
    ]
    picked.sort(key=lambda r: (str(r.get("track_id")), int(r.get("lap_time_ms") or 0)))
    if limit > 0:
        picked = picked[:limit]
    keep = (
        "track_id", "author", "setup", "setup_field_count", "lap_time_ms",
        "team", "session", "conditions", "surface", "game_year",
        "assists", "recorded_date", "source_url", "fetched_at",
    )
    return [{k: r.get(k) for k in keep} for r in picked]


def setup_distribution(setups: list[dict[str, Any]]) -> dict[str, Any]:
    """按赛道汇总调教参数分布与圈速分位。"""
    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in setups:
        by_track[str(row.get("track_id"))].append(row)

    profiles: list[dict[str, Any]] = []
    for track, rows in sorted(by_track.items()):
        laps = [int(r["lap_time_ms"]) for r in rows if r.get("lap_time_ms")]
        dry = [r for r in rows if (r.get("conditions") or "dry") == "dry"]
        wet = [r for r in rows if (r.get("conditions") or "") == "wet"]
        params: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            for name, value in (r.get("setup") or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    params[name].append(float(value))
        profiles.append({
            "track_id": track,
            "n_setups": len(rows),
            "n_dry": len(dry),
            "n_wet": len(wet),
            "lap_time_best_ms": min(laps) if laps else None,
            "lap_time_p10_ms": int(_percentile(laps, 0.10)) if laps else None,
            "lap_time_median_ms": int(statistics.median(laps)) if laps else None,
            "params": {
                name: {
                    "median": _round(statistics.median(values), 2),
                    "min": _round(min(values), 2),
                    "max": _round(max(values), 2),
                }
                for name, values in sorted(params.items())
            },
        })
    return {"by_track": profiles}


def build_track_profile(
    corner_stats: list[dict[str, Any]], setups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """把遥测弯速画像与调教分布按赛道 join 成一行。"""
    telemetry: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "corners": 0, "n_laps": 0, "min_speed_medians": [], "corner_time_medians": [],
        "lateral_g_medians": [],
    })
    for row in corner_stats:
        track = row["track_id"]
        entry = telemetry[track]
        entry["corners"] += 1
        entry["n_laps"] = max(entry["n_laps"], int(row.get("n_laps") or 0))
        for metric, target in (
            ("min_speed", "min_speed_medians"),
            ("corner_time_s", "corner_time_medians"),
            ("max_lateral_g", "lateral_g_medians"),
        ):
            value = (row.get(metric) or {}).get("median")
            if isinstance(value, (int, float)):
                entry[target].append(float(value))

    dist = {p["track_id"]: p for p in setup_distribution(setups)["by_track"]}
    all_tracks = sorted(set(telemetry) | set(dist))
    out: list[dict[str, Any]] = []
    for track in all_tracks:
        tel = telemetry.get(track)
        sp = dist.get(track)
        row: dict[str, Any] = {"track_id": track}
        if tel:
            row["telemetry"] = {
                "n_corners": tel["corners"],
                "n_laps_max": tel["n_laps"],
                "corner_min_speed_median_kmh": _round(
                    statistics.median(tel["min_speed_medians"]), 1,
                ) if tel["min_speed_medians"] else None,
                "corner_time_median_s": _round(
                    statistics.median(tel["corner_time_medians"]), 2,
                ) if tel["corner_time_medians"] else None,
                "corner_lateral_g_median": _round(
                    statistics.median(tel["lateral_g_medians"]), 2,
                ) if tel["lateral_g_medians"] else None,
            }
        if sp:
            row["setups"] = {
                k: sp[k] for k in (
                    "n_setups", "n_dry", "n_wet",
                    "lap_time_best_ms", "lap_time_p10_ms", "lap_time_median_ms",
                )
            }
            row["setup_params"] = sp["params"]
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="构建 F1 2026 遥测+调教数据包")
    ap.add_argument("--max-setups", type=int, default=0,
                    help="最多写入多少条调教（0=全部 2026）")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    out_dir = Path(args.out)

    if not CORNER_SRC.exists():
        print(f"[ERR] 缺少 {CORNER_SRC}；先运行 scripts/build_corner_dataset.py", file=sys.stderr)
        return 2
    if not SETUP_SRC.exists():
        print(f"[ERR] 缺少 {SETUP_SRC}；先运行 scripts/fetch_setup_laps.py", file=sys.stderr)
        return 2

    corner_rows = _read_jsonl(CORNER_SRC)
    setup_rows = _read_jsonl(SETUP_SRC)
    corner_stats, dropped_zero = build_corner_stats(corner_rows)
    setups = build_setups_2026(setup_rows, args.max_setups)
    profile = build_track_profile(corner_stats, setups)

    _write_jsonl(out_dir / OUT_CORNERS.name, corner_stats)
    _write_jsonl(out_dir / OUT_SETUPS.name, setups)
    _write_jsonl(out_dir / OUT_PROFILE.name, profile)

    manifest = {
        "name": "f1_2026_telemetry_setups",
        "built_at": datetime.now(UTC).isoformat(),
        "privacy": (
            "原始逐点遥测留在本地 .ref/ 缓存，不入库；本目录只含去标识化聚合统计，"
            "不含车手姓名/车号/圈号。"
        ),
        "sources": [
            {
                "id": "tracinginsights_2026",
                "kind": "real 2026 telemetry (per-lap, per-point)",
                "url": "https://github.com/TracingInsights/2026",
                "license": "Apache-2.0",
                "access": "GitHub raw + scripts/fetch_official_2026.py",
            },
            {
                "id": "f1laps_f1_26",
                "kind": "setup → lap time pairs (game F1 26)",
                "url": "https://www.f1laps.com/f1-26/setups/",
                "license": "public pages, robots.txt allows crawling; attribution kept",
                "access": "scripts/fetch_setup_laps.py",
            },
        ],
        "counts": {
            "corner_rows": len(corner_rows),
            "corner_rows_dropped_zero_speed": dropped_zero,
            "corner_stats_rows": len(corner_stats),
            "setup_rows_2026": len(setups),
            "profile_tracks": len(profile),
        },
        "coverage": {
            "telemetry_tracks": sorted({r["track_id"] for r in corner_stats}),
            "setup_tracks": sorted({r["track_id"] for r in setups}),
        },
        "files": {},
    }
    for name in (OUT_CORNERS.name, OUT_SETUPS.name, OUT_PROFILE.name):
        path = out_dir / name
        manifest["files"][name] = {
            "rows": sum(1 for _ in path.open(encoding="utf-8")),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "committed": name in COMMITTED_FILES,
        }
    (out_dir / OUT_MANIFEST.name).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"遥测聚合: {len(corner_stats)} 行 / "
          f"{len(manifest['coverage']['telemetry_tracks'])} 站")
    print(f"2026 调教: {len(setups)} 行 / {len(manifest['coverage']['setup_tracks'])} 站")
    print(f"赛道画像: {len(profile)} 行")
    print(f"输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
