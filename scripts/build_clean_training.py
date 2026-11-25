"""从全量 Race 缓存构建**干净圈**训练集（data/f1_2026/train/）。

背景：``fetch_official_2026.py --max-laps 10`` 只取每位车手**前 10 圈**，
其中包含发车圈、进站圈、黄旗/安全车圈，直接训练会把「脏圈」当成正常圈速。
本脚本用上游 ``session_laptimes`` 的逐圈元数据筛出干净圈：

- 排除第 1 圈（发车）
- ``status`` 必须为 "1"（整圈绿旗）
- ``pin`` / ``pout`` 为空（非进站/出站圈）
- ``del`` 为 False（未被赛会删除）
- ``iacc`` 为 True（计时准确）
- ``ff1G`` 为 False（非 FastF1 补造圈）

并输出：

- ``laps_clean.jsonl``：每圈一行（圈速、胎、天气、弯速极值、stint/life/position）
- ``corner_samples_clean.jsonl.gz``：逐弯训练样本（剔除零速帧）
- ``setups_f1_26.jsonl``：调教表（独立任务，**不与真实遥测逐圈绑定**）
- ``splits.json``：按分站的 train/val/test 切分（防同场泄漏）
- ``manifest.json`` / ``DATASET_CARD.md``

用法::

    python scripts/build_clean_training.py
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from f1_2026_extras import build_lap_summary  # noqa: E402

CACHE = ROOT / ".ref" / "tracing_2026"
CORNER_SRC = ROOT / "data" / "training" / "corner_dataset.jsonl"
OUT_DIR = ROOT / "data" / "f1_2026" / "train"
SETUP_SRC = ROOT / "data" / "f1_2026" / "setups_f1_26.jsonl"

# 按分站切分，避免同一场比赛的圈同时出现在 train/test
SPLIT_TEST = {"melbourne", "monza", "singapore"}
SPLIT_VAL = {"spa", "madrid", "suzuka"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _none(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() == "none")


def _clean_row(row: dict[str, Any]) -> bool:
    lap = row.get("lap")
    t = row.get("time")
    if not isinstance(lap, int) or lap <= 1:
        return False
    if not isinstance(t, (int, float)) or t <= 0:
        return False
    if str(row.get("status")) != "1":
        return False
    if not _none(row.get("pin")) or not _none(row.get("pout")):
        return False
    if row.get("del") is True:
        return False
    if row.get("iacc") is not True:
        return False
    if row.get("ff1G") is True:
        return False
    return True


def collect_clean(cache: Path) -> tuple[set[tuple[str, str, int]], dict[tuple[str, str, int], dict[str, Any]], Counter]:
    keys: set[tuple[str, str, int]] = set()
    meta: dict[tuple[str, str, int], dict[str, Any]] = {}
    stats: Counter = Counter()
    for race_dir in sorted(p for p in cache.iterdir() if p.is_dir()):
        lt_path = race_dir / "Race" / "session_laptimes.json"
        if not lt_path.exists():
            stats["race_without_laptimes"] += 1
            continue
        data = json.loads(lt_path.read_text(encoding="utf-8"))
        for i in range(len(data.get("lap") or [])):
            stats["timed_rows"] += 1
            rec = {k: (data[k][i] if isinstance(data.get(k), list) and i < len(data[k]) else None)
                   for k in ("drv", "lap", "time", "status", "pin", "pout", "del", "iacc", "ff1G",
                             "compound", "stint", "life", "pos", "fresh", "wAT", "wTT", "wR")}
            if not _clean_row(rec):
                stats["dirty_rows"] += 1
                continue
            key = (race_dir.name, str(rec["drv"]), int(rec["lap"]))
            keys.add(key)
            meta[key] = rec
    return keys, meta, stats


def filter_corners(keys: set[tuple[str, str, int]]) -> tuple[list[dict[str, Any]], Counter]:
    rows: list[dict[str, Any]] = []
    stats: Counter = Counter()
    with CORNER_SRC.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            stats["input_rows"] += 1
            if row.get("source") != "external_2026":
                stats["not_external"] += 1
                continue
            key = (row.get("race"), row.get("driver"), row.get("lap"))
            if key not in keys:
                stats["dirty_lap_rows"] += 1
                continue
            speeds = [row.get("min_speed"), row.get("avg_speed")]
            if any(not isinstance(v, (int, float)) or float(v) <= 0.0 for v in speeds):
                stats["zero_speed_rows"] += 1
                continue
            rows.append(row)
    return rows, stats


def _gzip_deterministic(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as out:
        for row in rows:
            out.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))


def augment_laps(laps: list[dict[str, Any]], meta: dict[tuple[str, str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    for row in laps:
        m = meta.get((row.get("race"), row.get("driver"), row.get("lap")), {})
        row["stint"] = m.get("stint")
        row["tyre_life"] = m.get("life")
        row["position"] = m.get("pos")
        row["fresh_tyre"] = bool(m.get("fresh")) if m.get("fresh") is not None else None
    laps.sort(key=lambda r: (str(r.get("race")), str(r.get("driver")), int(r.get("lap") or 0)))
    return laps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    keys, meta, cstats = collect_clean(args.cache)
    rows, fstats = filter_corners(keys)
    laps = augment_laps(build_lap_summary(rows), meta)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    laps_path = out / "laps_clean.jsonl"
    with laps_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in laps:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    corners_path = out / "corner_samples_clean.jsonl.gz"
    _gzip_deterministic(corners_path, rows)

    setups_path = out / "setups_f1_26.jsonl"
    if SETUP_SRC.exists():
        setups_path.write_bytes(SETUP_SRC.read_bytes())

    races = sorted({str(r.get("track_id")) for r in laps})
    splits = {
        "unit": "race (track_id); 同一分站的圈只出现在一个 split",
        "train": [t for t in races if t not in SPLIT_TEST | SPLIT_VAL],
        "val": [t for t in races if t in SPLIT_VAL],
        "test": [t for t in races if t in SPLIT_TEST],
    }
    (out / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    wet = sum(1 for r in laps if (r.get("wet") or 0) > 0)
    tyre = Counter(str(r.get("tyre_class")) for r in laps)
    per_track = Counter(str(r.get("track_id")) for r in laps)
    setup_rows = 0
    if setups_path.exists():
        with setups_path.open(encoding="utf-8") as fh:
            setup_rows = sum(1 for line in fh if line.strip())

    manifest: dict[str, Any] = {
        "name": "f1_2026_train_clean",
        "built_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "filter": {
            "lap_gt": 1, "track_status": "1 only", "pit_in_out": "excluded",
            "deleted": "excluded", "accurate_timing": "iacc=true only", "fastf1_generated": "excluded",
        },
        "counts": {
            "timed_rows": cstats["timed_rows"], "dirty_rows": cstats["dirty_rows"],
            "clean_laps": len(laps), "clean_corner_rows": len(rows),
            "dropped_zero_speed_corners": fstats["zero_speed_rows"],
            "setups_2026": setup_rows, "tracks": len(per_track),
            "dry_laps": len(laps) - wet, "wet_laps": wet,
        },
        "tyre_class": dict(tyre),
        "per_track_laps": dict(per_track),
        "splits": splits,
        "domain": {
            "telemetry": "real F1 2026 Race telemetry (TracingInsights, Apache-2.0)",
            "setups": "F1 26 game setups + lap times (F1Laps public pages)",
            "pairing": "NO per-lap setup in public real telemetry; setups are a separate supervised task",
        },
        "files": {},
    }
    for p in (laps_path, corners_path, setups_path):
        if p.exists():
            manifest["files"][p.name] = {"rows": len(laps) if p == laps_path else (len(rows) if p == corners_path else setup_rows),
                                         "bytes": p.stat().st_size, "sha256": _sha256(p)}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    card = f"""# F1 2026 训练集（干净圈）

## 规模

- 可计时圈：{cstats['timed_rows']:,}；脏圈剔除：{cstats['dirty_rows']:,}
- **干净圈：{len(laps):,}**（{len(per_track)} 站 Race）
- **逐弯样本：{len(rows):,}**（零速帧再剔 {fstats['zero_speed_rows']:,}）
- 干圈 {len(laps)-wet:,} / 湿圈 {wet:,}
- 调教：{setup_rows:,} 条（24 站，独立任务）

## 清洗口径

排除第 1 圈、非绿旗（`status != "1"`）、进站/出站（`pin/pout`）、赛会删除（`del`）、计时不准（`iacc=false`）、FastF1 补造（`ff1G=true`）。

## 两个训练任务（不要混成一个）

1. **遥测代理模型**：`corner_samples_clean.jsonl.gz` → 弯速/弯时/油门刹车等，输入赛道画像 + 弯号 + 胎/天气。
2. **调教→圈速模型**：`setups_f1_26.jsonl` → 20 项调教 + 赛道/工况 → 圈速。

> 公开的真实 F1 遥测**不含车队调教**（车队不公开）。因此不存在「同一圈既有真实遥测又有真实调教」的数据；调教表来自 F1 26 游戏，不能与真实圈速强行逐圈配对。

## 切分

按分站切分（`splits.json`），同一场比赛只进一个 split，避免同场泄漏：

- train：{len([t for t in races if t not in SPLIT_TEST | SPLIT_VAL])} 站
- val：{len([t for t in races if t in SPLIT_VAL])} 站
- test：{len([t for t in races if t in SPLIT_TEST])} 站

## 注意

- `max_*_g` 是原始峰值，建议截尾（如 99 分位）后再训练。
- 本目录的 `laps_clean` / `corner_samples_clean` 是全量 Race 干净圈；父目录的 `raw_telemetry/*.tar.gz` 仍是「每位车手前 10 圈」的原始抽样，用于点级核查。
- 若要「原始逐点全量」也入库，体积约 1.2GB（压缩后），需单独确认。
"""
    (out / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    print(f"out: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
