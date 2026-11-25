"""在 ``build_f1_2026_bundle.py`` 之后补齐逐圈汇总与逐弯样本，并刷新 manifest。

``build_f1_2026_bundle.py``（仓库既有版本）只产出
``telemetry_corners.jsonl`` / ``setups_f1_26.jsonl`` / ``track_profile.jsonl``；
本脚本在同一目录追加：

- ``telemetry_laps.jsonl``：逐圈汇总（圈速/胎/天气/弯速极值/油门刹车占比）；
- ``corner_samples.jsonl.gz``：``data/training/corner_dataset.jsonl`` 的确定性 gzip；
- 刷新 ``manifest.json``：补两项 counts/files，标记全部文件 ``committed=true``。

用法::

    python scripts/build_f1_2026_bundle.py
    python scripts/f1_2026_extras.py
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORNER_SRC = ROOT / "data" / "training" / "corner_dataset.jsonl"
OUT_DIR = ROOT / "data" / "f1_2026"
OUT_LAPS = OUT_DIR / "telemetry_laps.jsonl"
OUT_SAMPLES = OUT_DIR / "corner_samples.jsonl.gz"
OUT_MANIFEST = OUT_DIR / "manifest.json"

METRICS = (
    "corner_time_s", "min_speed", "exit_speed",
    "throttle_full_pct", "brake_pct", "max_lateral_g", "max_long_g",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gzip_deterministic(src: Path, dst: Path) -> None:
    with src.open("rb") as fin, dst.open("wb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9,
    ) as fout:
        fout.write(fin.read())


def build_lap_summary(corner_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把逐弯样本回卷成「每圈一行」；剔除零速帧污染的弯（与聚合同口径）。"""
    buckets: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in corner_rows:
        if row.get("source") != "external_2026":
            continue
        race, driver, lap = row.get("race"), row.get("driver"), row.get("lap")
        if not race or not driver or not isinstance(lap, int):
            continue
        speeds = [row.get("min_speed"), row.get("avg_speed")]
        if any(not isinstance(v, (int, float)) or float(v) <= 0.0 for v in speeds):
            continue
        buckets[(race, driver, lap)].append(row)

    out: list[dict[str, Any]] = []
    for (race, driver, lap), rows in sorted(buckets.items()):
        first = rows[0]
        cols: dict[str, list[float]] = {key: [] for key in METRICS}
        for row in rows:
            for key in METRICS:
                value = row.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    cols[key].append(float(value))
        out.append({
            "race": race,
            "track_id": first.get("track_id"),
            "session": first.get("session"),
            "driver": driver,
            "lap": lap,
            "lap_time_s": first.get("lap_time_s"),
            "tyre_class": first.get("tyre_class"),
            "tyre_softness": first.get("tyre_softness"),
            "wet": first.get("wet"),
            "track_temp": first.get("track_temp"),
            "air_temp": first.get("air_temp"),
            "n_corners": len(rows),
            "corner_time_sum_s": _round(sum(cols["corner_time_s"])),
            "min_speed_kmh": _round(min(cols["min_speed"])) if cols["min_speed"] else None,
            "max_speed_kmh": _round(max(cols["exit_speed"])) if cols["exit_speed"] else None,
            "throttle_full_pct": _round(statistics.median(cols["throttle_full_pct"]))
            if cols["throttle_full_pct"] else None,
            "brake_pct": _round(statistics.median(cols["brake_pct"]))
            if cols["brake_pct"] else None,
            "max_lateral_g": _round(max(cols["max_lateral_g"])) if cols["max_lateral_g"] else None,
            "max_long_g": _round(max(cols["max_long_g"])) if cols["max_long_g"] else None,
        })
    return out


def refresh_manifest(
    manifest_path: Path, out_dir: Path, laps: int, samples_rows: int,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["privacy"] = (
        "全部赛站的逐弯/逐圈明细、聚合统计与逐点原始遥测"
        "（raw_telemetry/*.tar.gz，上游原文件未裁剪）均入库；.ref/ 仅为本机抓取缓存。"
    )
    manifest.setdefault("counts", {})["lap_rows"] = laps
    manifest["counts"]["corner_samples_rows"] = samples_rows
    for info in manifest.get("files", {}).values():
        info["committed"] = True
    manifest.setdefault("files", {})
    for name in (OUT_LAPS.name, OUT_SAMPLES.name):
        path = out_dir / name
        rows = laps if name == OUT_LAPS.name else samples_rows
        manifest["files"][name] = {
            "rows": rows,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "committed": True,
        }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="补齐逐圈/逐弯明细并刷新 manifest")
    ap.add_argument("--corner-src", default=str(CORNER_SRC))
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    corner_path = Path(args.corner_src)
    if not corner_path.exists():
        print(f"[ERR] 缺少 {corner_path}；先运行 scripts/build_corner_dataset.py",
              file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    corner_rows = _read_jsonl(corner_path)
    laps = build_lap_summary(corner_rows)
    _write_jsonl(out_dir / OUT_LAPS.name, laps)
    _gzip_deterministic(corner_path, out_dir / OUT_SAMPLES.name)
    manifest = refresh_manifest(
        out_dir / OUT_MANIFEST.name, out_dir, len(laps), len(corner_rows),
    )

    print(f"逐圈汇总: {len(laps)} 行；逐弯样本: {len(corner_rows)} 行（已压缩）")
    print(f"manifest 覆盖: {len(manifest['coverage']['telemetry_tracks'])} 站遥测 / "
          f"{len(manifest['coverage']['setup_tracks'])} 站调教")
    print(f"输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
