"""气动模型回归验证：把关键敏感度与公开工程量级对齐，改动常数后必须重跑。

输出 ``docs/reports/aero_validation.json``，并在终端打印对照表。

判定口径（公开量级，见 f1_aero 模块头注释）：
- 前后翼各加满（CL 约 +8%）在高速赛道量级 0.2–0.6 s；
- 离地间隙在峰值附近小幅下调（归一化 −0.4）量级 0.03–0.25 s；
- 再往下降到失速区必须**变慢**（旧线性模型做不到这一点）；
- 默认调教必须严格返回 0（增量模型的自洽性）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import f1_aero  # noqa: E402
import build_setup_sim_dataset as B  # noqa: E402

OUT = ROOT / "docs" / "reports" / "aero_validation.json"


def main() -> int:
    stats = B.load_corner_stats(None)
    f1_aero.AERO_V2 = True
    tracks = [t for t in ("monza", "spa", "monaco", "silverstone") if t in stats]

    def delta(track: str, n: dict[str, float]) -> float:
        return round(B.simulate_lap_delta_ms(stats[track], n), 1)

    wing_up = {t: delta(t, {"front_wing": 1.0, "rear_wing": 1.0}) for t in tracks}
    rh_peak = {t: delta(t, {"front_ride_height": -0.4, "rear_ride_height": -0.4}) for t in tracks}
    rh_stall = {t: delta(t, {"front_ride_height": -1.0, "rear_ride_height": -1.0}) for t in tracks}
    default_zero = {t: delta(t, {}) for t in tracks}

    checks = {
        "default_is_zero": all(abs(v) < 1e-6 for v in default_zero.values()),
        "wings_full_range_in_band": all(-600.0 <= v <= 200.0 for v in wing_up.values()),
        "ride_height_peak_gain": all(-250.0 <= v <= -30.0 for v in rh_peak.values()),
        "stall_must_be_slower": all(v > 200.0 for v in rh_stall.values()),
        "stall_worse_than_peak": all(rh_stall[t] > rh_peak[t] for t in tracks),
    }

    floor_curve = [
        {"h": round(-1.0 + i * 0.1, 1), "floor": round(f1_aero.floor_downforce(-1.0 + i * 0.1)[0], 3),
         "stall": round(f1_aero.floor_downforce(-1.0 + i * 0.1)[1], 2)}
        for i in range(21)
    ]
    payload = {
        "aero_v2": True,
        "constants": {
            "FLOOR_PEAK_H": f1_aero.FLOOR_PEAK_H,
            "FLOOR_STALL_H": f1_aero.FLOOR_STALL_H,
            "FLOOR_CL_SHARE": f1_aero.FLOOR_CL_SHARE,
            "INDUCED_DRAG_K": f1_aero.INDUCED_DRAG_K,
            "FLOOR_DRAG_K": f1_aero.FLOOR_DRAG_K,
            "DRAG_MS_PER_CD": f1_aero.DRAG_MS_PER_CD,
            "ACTIVE_AERO_DRAG": f1_aero.ACTIVE_AERO_DRAG,
            "AERO_GAIN_SCALE": f1_aero.AERO_GAIN_SCALE,
        },
        "floor_curve": floor_curve,
        "wings_full_range_ms": wing_up,
        "ride_height_minus_0_4_ms": rh_peak,
        "ride_height_minus_1_0_ms": rh_stall,
        "default_zero_ms": default_zero,
        "checks": checks,
        "passed": all(checks.values()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'track':12s}{'wings+1':>10s}{'rh-0.4':>10s}{'rh-1.0':>10s}")
    for t in tracks:
        print(f"{t:12s}{wing_up[t]:>10.1f}{rh_peak[t]:>10.1f}{rh_stall[t]:>10.1f}")
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
