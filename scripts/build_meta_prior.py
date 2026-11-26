"""从 899 套 F1 26 游戏调教生成「每站 meta 包络」，并给出与当前建议的落差。

产物
----
- ``setup_tuner/resources/models/meta_prior.json``：24 站 × 20 项 的 P25/P50/P75
- 终端打印若干赛道的落差（当前建议 vs meta 中位数）

为什么存在
----------
调教语料本身带圈速，按「到该站 meta 中位数的距离」分组，干地条件下远组比近组
中位慢 206 ms（19/24 站为正）。这说明「向该站 meta 靠拢」在游戏里确实更快，
而这个信号比现有物理模型对翼角的敏感度强得多。

用法::

    python scripts/build_meta_prior.py --report monaco monza spa silverstone
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from setup_tuner.domain.setup import ALL_SETUP_FIELDS  # noqa: E402

SETUPS = ROOT / "data" / "f1_2026" / "train" / "setups_f1_26.jsonl"
OUT = ROOT / "setup_tuner" / "resources" / "models" / "meta_prior.json"

#: 至少这么多套才出包络，否则该站不给先验（避免 2 套调教当"meta"）
MIN_SAMPLES = 8


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(data) - 1)
    frac = pos - lo
    return data[lo] + (data[hi] - data[lo]) * frac


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", nargs="*", default=[])
    args = ap.parse_args(argv)

    if not SETUPS.exists():
        raise SystemExit(f"缺少调教语料 {SETUPS}")
    rows = [json.loads(line) for line in SETUPS.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_track: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        setup = r.get("setup") or {}
        if len(setup) >= 20:
            by_track[r["track_id"]].append(r)

    names = [f.name for f in ALL_SETUP_FIELDS]
    prior: dict[str, dict] = {}
    skipped: list[str] = []
    for track, items in sorted(by_track.items()):
        if len(items) < MIN_SAMPLES:
            skipped.append(track)
            continue
        entry: dict[str, dict] = {}
        for name in names:
            vals = [float(i["setup"][name]) for i in items if name in i["setup"]]
            if not vals:
                continue
            entry[name] = {
                "p25": round(quantile(vals, 0.25), 3),
                "p50": round(quantile(vals, 0.50), 3),
                "p75": round(quantile(vals, 0.75), 3),
                "n": len(vals),
            }
        entry["_n_setups"] = len(items)
        prior[track] = entry

    payload = {
        "schema": "f1opt-meta-prior/1",
        "source": "f1laps F1 26 setups (899 条，24 站，含圈速)",
        "note": "P25–P75 = 该站快调教的主要区间；用于先验与落差提示，不直接等于最优解",
        "min_samples": MIN_SAMPLES,
        "tracks": prior,
        "skipped_tracks": skipped,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {OUT}（{len(prior)} 站，跳过 {len(skipped)} 站：{skipped}）")

    if args.report:
        from setup_tuner.engine.engine import generate_suggestion

        base = {f.name: f.default for f in ALL_SETUP_FIELDS}
        tel = {
            "kerb_corners": [{"corner": 3, "ratio": 2.2}],
            "plank_bottoming": True, "plank_bottoming_ratio": 1.3,
            "m_tyresSurfaceTemperature": [98, 99, 101, 100],
            "m_tyresPressure": [23.0, 23.2, 22.0, 22.1],
            "tyres_wear": [12, 14, 10, 11],
            "m_brakesTemperature": [420, 430, 380, 390],
            "is_soft_compound": True, "avg_speed": 210.0,
            "max_steer": 0.42, "avg_steer": 0.11,
        }
        keys = ("front_wing", "rear_wing", "front_suspension", "brake_pressure")
        for track in args.report:
            if track not in prior:
                print(f"{track}: 无先验")
                continue
            res = generate_suggestion([], dict(base), track, telemetry=tel, model_type="hybrid")
            delta = res.get("setup_delta") or {}
            final = {k: base[k] + float(delta.get(k, 0.0)) for k in base}
            print(f"\n{track}（{prior[track]['_n_setups']} 套）")
            for k in keys:
                p = prior[track].get(k)
                if not p:
                    continue
                gap = p["p50"] - final[k]
                flag = "  ← 差距大" if abs(gap) >= 5 else ""
                print(f"  {k:18s} 建议 {final[k]:7.1f}   meta P50 {p['p50']:7.1f}"
                      f"  [{p['p25']:.1f}–{p['p75']:.1f}]  差 {gap:+6.1f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
