"""把逐点原始遥测按赛道打包进 ``data/f1_2026/raw_telemetry/``（完整、可入库）。

为什么按赛道打包
----------------
``.ref/tracing_2026/<Race>/<Session>/<DRV>/<lap>_tel.json`` 是上游 TracingInsights
2026 的原始逐点文件（每圈 600+ 采样点，14 个通道）。单个文件 ~100 KB，全部赛站
4000+ 个文件、约 640 MB，直接入库既超 GitHub 单文件硬限也无法浏览。

按赛道打成 ``<track_id>.tar.gz`` 后：每包 8–14 MB（< 50 MB 警告线），
内容与上游**逐字节一致**（不做任何裁剪/降采样），解压即可还原成
``.ref/tracing_2026/<Race>/Race/...`` 目录结构。

产物
----
``data/f1_2026/raw_telemetry/<track_id>.tar.gz``
``data/f1_2026/raw_telemetry/manifest.json``（文件数 / 原始字节 / 包字节 / sha256）

用法::

    python scripts/pack_raw_telemetry.py
    python scripts/pack_raw_telemetry.py --out data/f1_2026/raw_telemetry
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / ".ref" / "tracing_2026"
OUT_DIR = ROOT / "data" / "f1_2026" / "raw_telemetry"

#: 上游目录名 → track_id（official_name 对不上的特例）
RACE_ALIASES: dict[str, str] = {
    "Barcelona Grand Prix": "barcelona",
    "São Paulo Grand Prix": "sao_paulo",
}


def _race_to_track() -> dict[str, str]:
    from setup_tuner.domain.track import ALL_TRACKS

    mapping = {t.official_name: t.track_id for t in ALL_TRACKS}
    mapping.update(RACE_ALIASES)
    return mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack(out_dir: Path) -> dict[str, Any]:
    mapping = _race_to_track()
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks: list[dict[str, Any]] = []
    unknown: list[str] = []

    for race_dir in sorted(p for p in CACHE.iterdir() if p.is_dir()):
        session_dir = race_dir / "Race"
        if not session_dir.is_dir():
            continue
        track_id = mapping.get(race_dir.name)
        if track_id is None:
            unknown.append(race_dir.name)
            continue
        files = sorted(f for f in session_dir.rglob("*") if f.is_file())
        if not files:
            continue
        tar_path = out_dir / f"{track_id}.tar.gz"
        with tarfile.open(tar_path, "w:gz", compresslevel=9) as tf:
            for f in files:
                tf.add(f, arcname=str(f.relative_to(CACHE)))
        raw_bytes = sum(f.stat().st_size for f in files)
        tracks.append({
            "track_id": track_id,
            "race": race_dir.name,
            "files": len(files),
            "raw_bytes": raw_bytes,
            "tar_bytes": tar_path.stat().st_size,
            "sha256": _sha256(tar_path),
        })
        print(f"  {track_id:14s} {race_dir.name:28s} "
              f"{raw_bytes / 1e6:6.1f} MB → {tar_path.stat().st_size / 1e6:5.1f} MB")

    manifest = {
        "name": "f1_2026_raw_telemetry",
        "built_at": datetime.now(UTC).isoformat(),
        "source": "https://github.com/TracingInsights/2026",
        "license": "Apache-2.0",
        "structure": "tar.gz 内为 <Race>/Race/<DRV>/<lap>_tel.json（上游原文件，未裁剪）",
        "counts": {
            "tracks": len(tracks),
            "files": sum(t["files"] for t in tracks),
            "raw_bytes": sum(t["raw_bytes"] for t in tracks),
            "tar_bytes": sum(t["tar_bytes"] for t in tracks),
        },
        "tracks": tracks,
        "unknown_races": unknown,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[raw] {len(tracks)} 站 / {manifest['counts']['files']} 文件 / "
          f"{manifest['counts']['raw_bytes'] / 1e6:.1f} MB → "
          f"{manifest['counts']['tar_bytes'] / 1e6:.1f} MB")
    if unknown:
        print(f"[WARN] 未映射的赛站（未打包）: {unknown}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="打包 F1 2026 逐点原始遥测")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    if not CACHE.exists():
        print(f"[ERR] 缺少 {CACHE}；先运行 scripts/fetch_official_2026.py", file=sys.stderr)
        return 2
    pack(Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
