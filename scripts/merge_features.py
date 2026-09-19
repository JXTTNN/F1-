"""外部 2026 专业遥测 × 本地 UDP 训练样本 融合。

数据流::

    .ref/tracing_2026/<Race>/<Session>/<DRV>/<lap>_tel.json   （外部 2026 逐点遥测）
    data/training/telemetry_dataset.json                      （本地游戏 UDP 逐圈样本）
        → 外部逐圈特征聚合（每赛道专业基准画像）
        → 本地样本附上 ext_track_ref（专业基准）
        → data/training/merged_2026.jsonl

**严格约束**：
- 外部数据必须全部来自 2026（由 fetch_official_2026.py 的年检保证；
  本脚本再次校验 session_laptimes 的 lSD 时间戳，发现非 2026 直接报错）。
- 只做**特征对齐**，不做跨物理域标签混用：
  外部数据来自真实 F1，本地数据来自游戏，圈速绝对值不可直接比较；
  外部只提供「赛道画像 / 驾驶输入基准」层面的先验特征。
- 本地独有特征（损伤/主动空力/超车/底板触地/路肩）保持原样，不被外部覆盖。

用法::

    python scripts/merge_features.py                 # 融合全部已抓取数据
    python scripts/merge_features.py --dry-run       # 只报告，不写文件
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / ".ref" / "tracing_2026"
LOCAL_DATASET = ROOT / "data" / "training" / "telemetry_dataset.json"
OUT = ROOT / "data" / "training" / "merged_2026.jsonl"


def _race_to_track_id() -> dict[str, str]:
    """外部比赛名 → 本地 track_id（以官方大奖赛名为键，缺失的忽略）。"""
    from setup_tuner.domain.track import ALL_TRACKS

    return {t.official_name: t.track_id for t in ALL_TRACKS}


def _iter_tel_files() -> list[tuple[str, str, str, int, Path]]:
    """遍历缓存中的逐点遥测文件 → (race, session, driver, lap, path)。"""
    out: list[tuple[str, str, str, int, Path]] = []
    if not CACHE.exists():
        return out
    for race_dir in sorted(p for p in CACHE.iterdir() if p.is_dir()):
        for sess_dir in sorted(p for p in race_dir.iterdir() if p.is_dir()):
            for drv_dir in sorted(p for p in sess_dir.iterdir() if p.is_dir()):
                for f in sorted(drv_dir.glob("*_tel.json")):
                    try:
                        lap = int(f.name.split("_", 1)[0])
                    except ValueError:
                        continue
                    out.append((race_dir.name, sess_dir.name, drv_dir.name, lap, f))
    return out


def _verify_2026(laptimes: dict[str, Any], where: str) -> None:
    stamps = [str(s) for s in (laptimes.get("lSD") or []) if s is not None]
    real = [s for s in stamps if s != "None"]
    if not real or any(not s.startswith("2026-") for s in real):
        raise RuntimeError(f"{where}: 非 2026 数据（年检失败）→ 拒绝融合")


def _lap_meta(laptimes: dict[str, Any], driver: str, lap: int) -> dict[str, Any]:
    """从 session_laptimes 列式结构取某车某圈的官方圈速/配方/天气元数据。"""
    drv, laps = laptimes.get("drv") or [], laptimes.get("lap") or []
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
        # 该数据源用字符串 'None' 表示缺失（未完成圈/无数据），统一转 None
        if isinstance(raw, str):
            return None if raw.strip().lower() in ("none", "") else raw
        return raw
    return {
        "lap_time_s": col("time"),
        "compound": col("compound"),
        "stint": col("stint"),
        "tyre_life": col("life"),
        "position": col("pos"),
        "sector1_s": col("s1"), "sector2_s": col("s2"), "sector3_s": col("s3"),
        "air_temp": col("wAT"), "track_temp": col("wTT"),
        "humidity": col("wH"), "wind_speed": col("wWS"), "rainfall": col("wR"),
    }


def _to_floats(vals):
    out = []
    for v in vals or []:
        if v is None:
            continue
        if isinstance(v, str) and v.strip().lower() == "none":
            continue
        try:
            out.append(float(v))
        except Exception:
            continue
    return out

def _tel_features(tel: dict[str, Any]) -> dict[str, Any]:
    """逐点遥测 → 单圈特征（全部来自 2026 真实遥测点）。"""
    t = tel.get("tel") or {}
    speed = _to_floats(t.get("speed", []))
    throttle = _to_floats(t.get("throttle", []))
    brake = _to_floats(t.get("brake", []))
    drs = _to_floats(t.get("drs", []))
    rpm = _to_floats(t.get("rpm", []))
    gear = _to_floats(t.get("gear", []))
    ax = _to_floats(t.get("acc_x", []))
    ay = _to_floats(t.get("acc_y", []))
    n = len(speed) or 1
    return {
        "points": len(speed),
        "speed_max": max(speed) if speed else None,
        "speed_avg": round(sum(speed) / n, 2) if speed else None,
        "throttle_full_pct": round(100.0 * sum(1 for v in throttle if v >= 99) / n, 2),
        "brake_pct": round(100.0 * sum(1 for v in brake if v > 0) / n, 2),
        "drs_open_pct": round(100.0 * sum(1 for v in drs if v > 0) / n, 2),
        "rpm_max": max(rpm) if rpm else None,
        "rpm_avg": round(sum(rpm) / n, 1) if rpm else None,
        "gear_max": max(gear) if gear else None,
        "max_lateral_g": round(max((abs(v) for v in ax), default=0.0), 3),
        "max_long_g": round(max((abs(v) for v in ay), default=0.0), 3),
    }


def build_external_reference(
    race_filter: str | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """聚合外部逐圈数据 → {track_id: [逐圈特征...]} + 汇总统计。"""
    race_to_track = _race_to_track_id()
    per_track: dict[str, list[dict[str, Any]]] = {}
    stats: dict[str, Any] = {"files": 0, "laps": 0, "tracks": {}, "skipped": []}

    # 会话级 laptimes 缓存（同目录多个遥测文件共用）
    lt_cache: dict[Path, dict[str, Any]] = {}
    for race, session, driver, lap, path in _iter_tel_files():
        if race_filter and race != race_filter:
            continue
        track_id = race_to_track.get(race)
        if track_id is None:
            stats["skipped"].append(f"{race}: 无本地赛道映射")
            continue
        lt_path = path.parents[1] / "session_laptimes.json"
        if lt_path not in lt_cache:
            if not lt_path.exists():
                stats["skipped"].append(f"{race}/{session}: 缺 session_laptimes")
                continue
            lt = json.loads(lt_path.read_text(encoding="utf-8"))
            _verify_2026(lt, f"{race}/{session}")
            lt_cache[lt_path] = lt
        meta = _lap_meta(lt_cache[lt_path], driver, lap)
        if not meta.get("lap_time_s"):
            continue
        tel = json.loads(path.read_text(encoding="utf-8"))
        feat = _tel_features(tel)
        # 圈速必须是数值（字符串 'None'/异常值一律剔除，避免训练标签被污染）
        if not isinstance(meta.get("lap_time_s"), (int, float)):
            stats["skipped"].append(f"{race}/{session}/{driver}/{lap}: 圈速非数值")
            continue
        row = {
            "race": race, "session": session, "driver": driver, "lap": lap,
            "source_year": 2026, **meta, **feat,
        }
        per_track.setdefault(track_id, []).append(row)
        stats["files"] += 1
        stats["laps"] += 1

    for track_id, rows in per_track.items():
        times = [r["lap_time_s"] for r in rows if r.get("lap_time_s")]
        stats["tracks"][track_id] = {
            "laps": len(rows),
            "best_lap_s": round(min(times), 3) if times else None,
            "median_lap_s": round(statistics.median(times), 3) if times else None,
        }
    return per_track, stats


#: 外部 2026 专业基准特征列（并入训练特征向量；无外部数据时补 0）
EXT_FEATURE_KEYS: tuple[str, ...] = (
    "ext_best_lap_s", "ext_median_lap_s", "ext_n_laps_log",
    "ext_median_speed_max", "ext_median_throttle_full_pct",
    "ext_median_brake_pct", "ext_median_drs_open_pct",
    "ext_median_max_lateral_g",
)


def _flatten_ext(prior: dict[str, Any] | None) -> dict[str, float]:
    """外部基准画像 → 数值特征列（无外部数据时全 0，训练侧可据 has_external_ref 区分）。"""
    if not prior:
        return dict.fromkeys(EXT_FEATURE_KEYS, 0.0)
    n = prior.get("n_laps") or 0
    return {
        "ext_best_lap_s": float(prior.get("best_lap_s") or 0.0),
        "ext_median_lap_s": float(prior.get("median_lap_s") or 0.0),
        "ext_n_laps_log": round(math.log1p(n), 4),
        "ext_median_speed_max": float(prior.get("median_speed_max") or 0.0),
        "ext_median_throttle_full_pct": float(prior.get("median_throttle_full_pct") or 0.0),
        "ext_median_brake_pct": float(prior.get("median_brake_pct") or 0.0),
        "ext_median_drs_open_pct": float(prior.get("median_drs_open_pct") or 0.0),
        "ext_median_max_lateral_g": float(prior.get("median_max_lateral_g") or 0.0),
    }


def _track_prior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """把某赛道的外部逐圈数据压成"专业基准画像"（供本地样本引用）。"""
    def med(key: str) -> float | None:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(statistics.median(vals), 3) if vals else None

    times = [r["lap_time_s"] for r in rows if r.get("lap_time_s")]
    return {
        "source": "TracingInsights 2026 (official feed via FastF1)",
        "source_year": 2026,
        "n_laps": len(rows),
        "n_drivers": len({r["driver"] for r in rows}),
        "best_lap_s": round(min(times), 3) if times else None,
        "median_lap_s": round(statistics.median(times), 3) if times else None,
        "median_speed_max": med("speed_max"),
        "median_throttle_full_pct": med("throttle_full_pct"),
        "median_brake_pct": med("brake_pct"),
        "median_drs_open_pct": med("drs_open_pct"),
        "median_max_lateral_g": med("max_lateral_g"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="2026 外部专业遥测 × 本地样本融合")
    ap.add_argument("--local", default=str(LOCAL_DATASET))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--race", default=None, help="只融合某场比赛（默认全部）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    local_path = Path(args.local)
    if not local_path.exists():
        print(f"本地数据集不存在: {local_path}（先跑 build_training_dataset.py）")
        return 2

    per_track, stats = build_external_reference(args.race)
    priors = {tid: _track_prior(rows) for tid, rows in per_track.items()}

    local = json.loads(local_path.read_text(encoding="utf-8"))
    rows = local.get("rows") or []
    # 本地 track_id 是 udp 编号（int），外部映射是短 id（str）——按 udp→短 id 对齐
    from setup_tuner.domain.track import ALL_TRACKS
    udp_to_short = {t.udp_track_id: t.track_id for t in ALL_TRACKS}

    merged: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    covered = 0
    for r in rows:
        meta = r.get("meta") or {}
        # 本地 track_id 是短 id（str，如 'monza'）；udp_track_id 是官方枚举（int）
        short = meta.get("track_id") or r.get("track_id")
        udp_id = meta.get("track_udp_id")
        if short not in priors and isinstance(udp_id, int):
            short = udp_to_short.get(udp_id, short)
        prior = priors.get(short) if short else None
        if prior:
            covered += 1
        merged.append({
            "source": "local_game_udp_2026",
            "track_id": udp_id,
            "track_key": short,
            "ext_track_ref": prior,
            "features": r.get("features"),
            "target": r.get("target"),
        })
        # 训练用行：本地 29 维特征 + 外部 2026 专业基准特征（缺失补 0 并标记）
        feats = dict(r.get("features") or {})
        feats.update(_flatten_ext(prior))
        training_rows.append({
            "track_id": short,
            "target": r.get("target"),
            "has_external_ref": prior is not None,
            "features": feats,
        })

    # 外部基准本身也作为独立行（赛道先验样本，供画像层使用）
    for tid, prior in sorted(priors.items()):
        merged.append({"source": "external_2026_reference",
                       "track_id": None, "track_key": tid,
                       "ext_track_ref": prior, "features": None, "target": None})

    print("=== 融合报告 ===")
    print(f"外部文件: {stats['files']}  外部圈: {stats['laps']}")
    print(f"覆盖赛道: {list(stats['tracks'].keys())}")
    for tid, s in sorted(stats["tracks"].items()):
        print(f"  {tid:>12}: {s['laps']:>4} 圈  best={s['best_lap_s']}s  median={s['median_lap_s']}s")
    print(f"本地样本: {len(rows)}  其中 {covered} 条匹配到外部赛道基准")
    if stats["skipped"]:
        print(f"跳过: {stats['skipped'][:5]}")
    if args.dry_run:
        print("(dry-run，未写文件)")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"写出: {out}（{len(merged)} 行，全部 2026）")

    # 训练用矩阵：本地特征 + 外部 2026 基准特征（直接喂训练脚本）
    base_keys = list((rows[0].get("features") or {}).keys()) if rows else []
    ds_out = out.with_name("merged_dataset.json")
    dataset = {
        "schema": "f1opt-merged-training/1",
        "feature_keys": base_keys + list(EXT_FEATURE_KEYS),
        "ext_feature_keys": list(EXT_FEATURE_KEYS),
        "external_source": "TracingInsights 2026 (only 2026, year-checked)",
        "count": len(training_rows),
        "rows": training_rows,
    }
    ds_out.write_text(json.dumps(dataset, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    with_ext = sum(1 for r in training_rows if r["has_external_ref"])
    print(f"写出: {ds_out}（{len(training_rows)} 行，其中 {with_ext} 行带外部 2026 基准）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
