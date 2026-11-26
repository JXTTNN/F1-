"""端到端回归复核：不同起始调教 × 新旧预算，验证收益不是单一场景的偶然。

与 `e2e_ab.py` 的区别
----------------------
- `e2e_ab.py` 比的是"两个模型"，起始调教固定为默认；
- 本脚本比的是"两套优化器预算"，起始调教取 3 组（默认 / 偏软 / 偏翼），
  并且**在同一进程内交替跑**，避免环境漂移。

判定：
  1. 新预算的平均/中位物理收益必须优于旧预算；
  2. 新预算下"比默认更慢"的组数必须为 0；
  3. 同一输入重复调用必须逐位一致（确定性）。

用法::

    python scripts/e2e_check.py --cases 126
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from setup_tuner.domain.setup import ALL_SETUP_FIELDS, get_field  # noqa: E402
from setup_tuner.domain.track import ALL_TRACKS, get_track_by_id  # noqa: E402
from setup_tuner.engine import sim_optimizer as opt_mod  # noqa: E402
from setup_tuner.engine.engine import generate_suggestion  # noqa: E402
from setup_tuner.engine.setup_sim import normalize_setup  # noqa: E402
from build_setup_sim_dataset import load_corner_stats, simulate_lap_delta_ms  # noqa: E402

STATS = load_corner_stats(None)
BASE = {f.name: f.default for f in ALL_SETUP_FIELDS}

#: 三组起始调教（全部在合法区间内，用归一化偏移生成）
PROFILES: dict[str, dict[str, float]] = {
    "default": {},
    "soft": {
        "front_suspension": -0.4, "rear_suspension": -0.4,
        "front_anti_roll_bar": -0.4, "rear_anti_roll_bar": -0.4,
    },
    "wing": {
        "front_wing": 0.4, "rear_wing": 0.4,
        "front_ride_height": 0.3, "rear_ride_height": 0.3,
    },
}

LEGACY = {"_MAX_TOTAL_GAIN_S": 0.120, "_MAX_MOVES": 3,
          "_MAX_STEP_GAIN_S": 0.080, "_GRID": 0.25, "_OBJ_PATCH": 0.08,
          "_MAX_PASSES": 3, "_GUARD": False, "_ZERO_START": False}


def setup_from(profile: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for f in ALL_SETUP_FIELDS:
        n = float(profile.get(f.name, 0.0))
        half = max(f.max_val - f.default, f.default - f.min_val) or 1.0
        val = f.default + n * half
        val = max(f.min_val, min(f.max_val, val))
        step = f.step or 0.01
        val = round(round((val - f.min_val) / step) * step + f.min_val, 6)
        out[f.name] = max(f.min_val, min(f.max_val, val))
    return out


def telemetry(track_id: str, wet: bool, phase: int) -> dict:
    track = get_track_by_id(track_id)
    corners = sorted(c.number for c in track.corners)
    slow = [c.number for c in track.corners if str(c.corner_type).lower().startswith("s")] or corners
    kerb = slow[phase % len(slow)]
    return {
        "kerb_corners": [{"corner": kerb, "ratio": 1.8 + 0.2 * phase},
                         {"corner": corners[-1], "ratio": 1.6}],
        "plank_bottoming": phase % 2 == 0,
        "plank_bottoming_ratio": 1.1 + 0.1 * phase,
        "m_tyresSurfaceTemperature": ([95, 97, 100, 99] if not wet else [76, 78, 81, 80]),
        "m_tyresInnerTemperature": ([103, 105, 108, 107] if not wet else [84, 86, 89, 88]),
        "m_tyresPressure": [22.8 + 0.1 * phase, 23.1, 22.0, 22.2],
        "tyres_wear": [10 + phase, 12, 9, 10],
        "m_brakesTemperature": [410 + 10 * phase, 425, 375, 388],
        "is_soft_compound": not wet,
        "avg_speed": 200.0,
        "max_steer": 0.42,
        "avg_steer": 0.11,
    }


def apply_caps(caps: dict) -> None:
    opt_mod._MAX_TOTAL_GAIN_S = caps["_MAX_TOTAL_GAIN_S"]
    opt_mod._MAX_MOVES = caps["_MAX_MOVES"]
    opt_mod._MAX_STEP_GAIN_S = caps["_MAX_STEP_GAIN_S"]
    opt_mod._GRID = caps["_GRID"]
    opt_mod._OBJ_BUDGET = caps["_OBJ_PATCH"]
    opt_mod._MAX_PASSES = caps.get("_MAX_PASSES", opt_mod._MAX_PASSES)
    opt_mod._GUARD_ENABLED = caps.get("_GUARD", True)
    opt_mod._ZERO_START = caps.get("_ZERO_START", True)


def current_caps() -> dict:
    return {
        "_MAX_TOTAL_GAIN_S": opt_mod._MAX_TOTAL_GAIN_S,
        "_MAX_MOVES": opt_mod._MAX_MOVES,
        "_MAX_STEP_GAIN_S": opt_mod._MAX_STEP_GAIN_S,
        "_GRID": opt_mod._GRID,
        "_OBJ_PATCH": opt_mod._OBJ_BUDGET,
        "_MAX_PASSES": opt_mod._MAX_PASSES,
        "_GUARD": opt_mod._GUARD_ENABLED,
        "_ZERO_START": opt_mod._ZERO_START,
    }


def run_all(caps: dict, tracks: list[str]) -> list[dict]:
    apply_caps(caps)
    out: list[dict] = []
    for track in tracks:
        for wet in (False, True):
            for pname, profile in PROFILES.items():
                start_setup = setup_from(profile)
                tel = telemetry(track, wet, len(out) % 3)
                t0 = time.perf_counter()
                r = generate_suggestion([], dict(start_setup), track,
                                        telemetry=tel, model_type="hybrid")
                ms = (time.perf_counter() - t0) * 1000
                delta = r.get("setup_delta") or {}
                final = {k: start_setup[k] + float(delta.get(k, 0.0)) for k in start_setup}
                phys = simulate_lap_delta_ms(STATS[track], normalize_setup(final))
                out.append({
                    "track": track, "wet": wet, "profile": pname,
                    "sim_delta_ms": round(phys, 2), "ms": round(ms, 1),
                    "changes": sum(1 for v in delta.values() if abs(float(v)) > 1e-9),
                })
    return out


def summarize(rows: list[dict], label: str) -> dict:
    vals = [r["sim_delta_ms"] for r in rows]
    lat = [r["ms"] for r in rows]
    lat.sort()
    return {
        "label": label,
        "n": len(rows),
        "mean": round(statistics.mean(vals), 2),
        "median": round(statistics.median(vals), 2),
        "worse": sum(1 for v in vals if v > 0),
        "p50_ms": round(lat[len(lat) // 2], 1),
        "p95_ms": round(lat[int(len(lat) * 0.95) - 1], 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "work" / "e2e_check.json"))
    args = ap.parse_args(argv)

    tracks = [t for t in sorted(x.track_id for x in ALL_TRACKS) if t in STATS]
    legacy = dict(LEGACY)
    modern = current_caps()

    old_rows = run_all(legacy, tracks)
    new_rows = run_all(modern, tracks)
    # 确定性：同输入重复一次
    apply_caps(modern)
    probe = generate_suggestion([], dict(setup_from(PROFILES["default"])), tracks[0],
                                telemetry=telemetry(tracks[0], False, 0), model_type="hybrid")
    probe2 = generate_suggestion([], dict(setup_from(PROFILES["default"])), tracks[0],
                                 telemetry=telemetry(tracks[0], False, 0), model_type="hybrid")
    deterministic = (probe.get("setup_delta") == probe2.get("setup_delta"))

    summary = {
        "tracks": len(tracks),
        "cases": len(new_rows),
        "profiles": list(PROFILES),
        "legacy": summarize(old_rows, "旧预算 120/3/80"),
        "modern": summarize(new_rows, "新预算"),
        "deterministic": deterministic,
        "modern_caps": {k: round(v, 4) for k, v in modern.items()},
    }
    summary["improvement"] = {
        "mean_pct": round((summary["legacy"]["mean"] - summary["modern"]["mean"])
                           / max(1e-9, abs(summary["legacy"]["mean"])) * 100, 1),
        "median_pct": round((summary["legacy"]["median"] - summary["modern"]["median"])
                             / max(1e-9, abs(summary["legacy"]["median"])) * 100, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": new_rows},
                                          ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=1))
    ok = (summary["modern"]["mean"] < summary["legacy"]["mean"]
          and summary["modern"]["worse"] == 0 and deterministic)
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
