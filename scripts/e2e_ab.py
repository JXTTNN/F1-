"""端到端 A/B：现役遥测代理 vs 新版（基准+残差），同一模拟遥测输入。

判定口径（不依赖模型自报的收益）：
- 独立物理仿真 `simulate_lap_delta_ms` 对**建议后的调教**打分（正=更慢）
- 覆盖：有多少站真正用上了遥测学出的逐弯重要度（而不是 120/speed 启发式）
- 行为：改动条数、总改动量、耗时

用法：
    python scripts/e2e_ab.py --old /tmp/telemetry_surrogate.old.json \
                              --new setup_tuner/resources/models/telemetry_surrogate.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from setup_tuner.domain.setup import ALL_SETUP_FIELDS  # noqa: E402
from setup_tuner.domain.track import ALL_TRACKS, get_track_by_id  # noqa: E402
from setup_tuner.engine import surrogate as surrogate_mod  # noqa: E402
from setup_tuner.engine import setup_sim as sim_mod  # noqa: E402
from setup_tuner.engine.engine import generate_suggestion  # noqa: E402
from setup_tuner.engine.setup_sim import normalize_setup  # noqa: E402
from build_setup_sim_dataset import load_corner_stats, simulate_lap_delta_ms  # noqa: E402

STATS = load_corner_stats(None)
BASE = {f.name: f.default for f in ALL_SETUP_FIELDS}


def fake_telemetry(track_id: str, wet: bool) -> dict:
    track = get_track_by_id(track_id)
    corners = sorted(c.number for c in track.corners)
    slow = [c.number for c in track.corners if str(c.corner_type).lower().startswith("s")]
    kerb = (slow or corners)[:2]
    speeds = [c.speed_kmh for c in track.corners] or [200.0]
    return {
        "kerb_corners": [{"corner": kerb[0], "ratio": 2.2},
                         {"corner": kerb[-1], "ratio": 1.7}],
        "plank_bottoming": True,
        "plank_bottoming_ratio": 1.3,
        "plank_front_height_min": 18.0,
        "plank_rear_height_min": 44.0,
        "m_tyresSurfaceTemperature": [96, 97, 101, 100] if not wet else [78, 79, 82, 81],
        "m_tyresInnerTemperature": [104, 105, 108, 107] if not wet else [85, 86, 88, 87],
        "m_tyresPressure": [23.0, 23.2, 22.0, 22.1],
        "tyres_wear": [12, 14, 10, 11] if not wet else [30, 32, 28, 29],
        "m_brakesTemperature": [420, 430, 380, 390],
        "is_soft_compound": not wet,
        "is_hard_compound": False,
        "avg_speed": round(sum(speeds) / len(speeds), 1),
        "max_steer": 0.42,
        "avg_steer": 0.11,
        "wind": 0,
    }


def run(model_path: Path, sim_path: str | None = None) -> dict:
    # 必须先 reset：逐弯评价按赛道缓存，热替换模型后不清缓存会读到上一版的重要度
    surrogate_mod.reset_surrogate()
    surrogate_mod._SURROGATE = surrogate_mod.SurrogateModel(model_path)
    if sim_path:
        # 调教性能模型同样是单例：换模型必须重建，否则仍用上一版权重
        sim_mod.reset_setup_sim()
        sim_mod._SIM = sim_mod.SetupSimModel(sim_path)
    sur = surrogate_mod._SURROGATE
    out = {"model": str(model_path), "available": sur.available,
           "tracks_with_importance": len(sur.importance),
           "samples": sur.trained_samples, "rows": []}
    for track in sorted(t.track_id for t in ALL_TRACKS):
        if track not in STATS:
            continue
        for wet in (False, True):
            tel = fake_telemetry(track, wet)
            t0 = time.perf_counter()
            try:
                r = generate_suggestion([], dict(BASE), track, telemetry=tel,
                                        model_type="hybrid")
                dt = (time.perf_counter() - t0) * 1000.0
                delta = r.get("setup_delta") or {}
                final = {k: BASE[k] + float(delta.get(k, 0.0)) for k in BASE}
                norm = normalize_setup(final)
                sim_ms = simulate_lap_delta_ms(STATS[track], norm)
                learned = sur.corner_importance(track) is not None
                out["rows"].append({
                    "track": track, "wet": wet, "ms": round(dt, 1),
                    "n_changes": sum(1 for v in delta.values() if abs(float(v)) > 1e-9),
                    "abs_delta": round(sum(abs(float(v)) for v in delta.values()), 3),
                    "sim_delta_ms": round(sim_ms, 2),
                    "learned_importance": learned,
                    "gain_ms": (r.get("holistic", {}).get("sim") or {}).get("gain_ms"),
                })
            except Exception as exc:  # noqa: BLE001
                out["rows"].append({"track": track, "wet": wet, "error": f"{type(exc).__name__}: {exc}"})
    ok = [r for r in out["rows"] if "sim_delta_ms" in r]
    out["summary"] = {
        "runs": len(out["rows"]), "ok": len(ok), "errors": len(out["rows"]) - len(ok),
        "mean_sim_delta_ms": round(statistics.mean(r["sim_delta_ms"] for r in ok), 2) if ok else None,
        "median_sim_delta_ms": round(statistics.median(r["sim_delta_ms"] for r in ok), 2) if ok else None,
        "mean_changes": round(statistics.mean(r["n_changes"] for r in ok), 2) if ok else None,
        "mean_ms": round(statistics.mean(r["ms"] for r in ok), 1) if ok else None,
        "learned_runs": sum(1 for r in ok if r["learned_importance"]),
        "negative_runs": sum(1 for r in ok if r["sim_delta_ms"] < 0),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=str(ROOT / "work" / "telemetry_surrogate.old.json"))
    ap.add_argument("--new", default=str(REPO / "setup_tuner" / "resources" / "models" / "telemetry_surrogate.json"))
    ap.add_argument("--old-sim", default=None, help="旧调教性能模型（setup_sim_nn.json）")
    ap.add_argument("--new-sim", default=None, help="新调教性能模型（setup_sim_nn.json）")
    args = ap.parse_args()
    old = run(Path(args.old), args.old_sim)
    new = run(Path(args.new), args.new_sim)
    result = {"old": old, "new": new}

    paired = {}
    for a, b in zip(old["rows"], new["rows"], strict=False):
        if "sim_delta_ms" in a and "sim_delta_ms" in b and a["track"] == b["track"] and a["wet"] == b["wet"]:
            paired.setdefault(a["track"], {})["wet" if a["wet"] else "dry"] = {
                "old": a["sim_delta_ms"], "new": b["sim_delta_ms"],
                "old_changes": a["n_changes"], "new_changes": b["n_changes"],
            }
    result["paired"] = paired
    better = sum(1 for t in paired for k in paired[t] if paired[t][k]["new"] < paired[t][k]["old"])
    worse = sum(1 for t in paired for k in paired[t] if paired[t][k]["new"] > paired[t][k]["old"])
    result["verdict"] = {"new_better": better, "new_worse": worse,
                         "same": sum(1 for t in paired for k in paired[t]) - better - worse}
    (ROOT / "work" / "e2e_ab_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("old", "new"):
        print(name, json.dumps(result[name]["summary"], ensure_ascii=False))
    print("verdict", result["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
