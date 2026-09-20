"""从真实遥测构建**普世化**训练数据集（task-82）。

目标：让模型学到「工况 → 调教 → 圈速」的**可迁移**关系，
而不是记住某条赛道——因此：

- 赛道特征用 **画像**（慢/中/快弯占比、牵引/空力指数、连弯段长度），
  不用赛道 ID 独热——没见过的新赛道只要有弯道画像即可推理；
- 目标用 **归一化节奏** ``lap_time_ms / track_length_m``（秒/公里），
  与赛道长度解耦，跨赛道可比；
- 工况特征：轮胎配方类（软/中/硬/雨）、天气、赛道温度、气温；
- 风格：12 维车手风格向量（不同车手 → 不同最优解）；
- 路肩：按弯路肩检出（ratio 最大值 + 弯数）——路肩不得不压的工况特征。

用法::

    python scripts/build_training_dataset.py            # 全部 *_laps.jsonl
    python scripts/build_training_dataset.py --out data/training/telemetry_dataset.json

纯标准库（调教性能 NN 训练见 scripts/train_setup_sim_nn.py），可在 CI 无依赖环境运行。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from setup_tuner.domain.setup import ALL_SETUP_FIELDS  # noqa: E402
from setup_tuner.domain.track import get_track_by_udp_id  # noqa: E402
from setup_tuner.engine.holistic import track_demand  # noqa: E402
from setup_tuner.telemetry.packets import (  # noqa: E402
    is_wet_weather_code,
    weather_label,
)

#: 归一化用：特征取值域（保证跨数据集一致）
_COMPOUND_CLASS = {"soft": 0.0, "medium": 0.5, "hard": 1.0, "wet": 0.75}


def compound_class(code: int | None) -> str:
    """轮胎配方代码 → 类别（软/中/硬/雨）。"""
    if code in (16, 17, 22, 9):
        return "soft"
    if code in (18, 10):
        return "medium"
    if code in (19, 20, 21, 11):
        return "hard"
    if code in (7, 8):
        return "wet"
    return "unknown"


def build_row(sample: dict[str, Any]) -> dict[str, Any] | None:
    """把一条逐圈样本转成训练行（特征 + 目标）；不合格返回 None。

    普世化要点：
    - 赛道特征来自 ``track_demand``（弯道画像），**不含赛道 ID**；
    - 目标 = ``lap_time_ms / track_length_m``（归一化节奏，跨赛道可比）。
    """
    track_id = sample.get("track_id")
    lap_time = sample.get("lap_time_ms")
    valid = sample.get("lap_valid")
    if not isinstance(track_id, int) or not isinstance(lap_time, (int, float)):
        return None
    if lap_time <= 0 or valid is False:
        return None

    track = get_track_by_udp_id(int(track_id))
    if track is None:
        return None  # 未知赛道无法取画像，跳过（不臆造）

    demand = track_demand(track.track_id)
    agg = sample.get("lap_agg") or {}
    track_len = float(track.length_m)

    # ── 目标：归一化节奏（跨赛道可比）──
    # lap_time_ms / track_length_m 的数值 = ms/m；而 1 ms/m 恰好 ≡ 1 s/km
    # （85000ms 跑 5807m ≈ 14.64 s/km），无需换算系数。
    pace_s_per_km = float(lap_time) / track_len

    # ── 工况 ──
    cc = compound_class(agg.get("tyre_compound"))
    weather_raw = sample.get("weather")
    # 官方 6 档枚举：只有 ≥3（小雨/大雨/暴雨）才是湿地；
    # 1=轻云 / 2=阴 仍是干地（此前误按 4 档把轻云当湿地）。
    weather_wet = 1.0 if is_wet_weather_code(weather_raw) else 0.0

    # ── 路肩特征 ──
    kerb = agg.get("kerb_corners") or []
    kerb_ratios = [
        float(k.get("ratio", 0.0)) for k in kerb
        if isinstance(k, dict) and isinstance(k.get("ratio"), (int, float))
    ]
    features: dict[str, Any] = {
        # 赛道画像（普世：新赛道只要画像即可推理）
        "track_slow_share": round(demand.slow_share, 4),
        "track_medium_share": round(demand.medium_share, 4),
        "track_fast_share": round(demand.fast_share, 4),
        "track_traction_index": round(demand.traction_index, 4),
        "track_aero_index": round(demand.aero_index, 4),
        "track_avg_speed": round(demand.avg_speed, 1),
        "track_max_speed": round(demand.max_speed, 1),
        "track_longest_sequence": demand.longest_sequence,
        # 工况
        "tyre_class_soft": 1.0 if cc == "soft" else 0.0,
        "tyre_class_medium": 1.0 if cc == "medium" else 0.0,
        "tyre_class_hard": 1.0 if cc == "hard" else 0.0,
        "tyre_class_wet": 1.0 if cc == "wet" else 0.0,
        "weather_wet": weather_wet,
        # 原始档位也保留（0-5），让模型能区分轻云/阴/小雨/暴雨的强度
        "weather_code": (
            float(weather_raw)
            if isinstance(weather_raw, int) and not isinstance(weather_raw, bool)
            else -1.0
        ),
        "track_temp": sample.get("track_temp"),
        "air_temp": sample.get("air_temp"),
        # 路肩
        "kerb_corner_count": float(len(kerb)),
        "kerb_max_ratio": round(max(kerb_ratios), 3) if kerb_ratios else 0.0,
        # 驾驶风格（12 维）
        **{
            f"style_{i}": round(float(v), 4)
            for i, v in enumerate(sample.get("style") or [])
        },
        # 当前调教（归一化到 [0,1]，跨参数可比）
        **{
            f"setup_{f.name}": round(
                (float(sample["setup"][f.name]) - f.min_val)
                / (f.max_val - f.min_val), 4,
            )
            for f in ALL_SETUP_FIELDS
            if isinstance(sample.get("setup"), dict)
            and isinstance(sample["setup"].get(f.name), (int, float))
        },
    }

    return {
        "features": features,
        "target": round(pace_s_per_km, 4),       # 归一化节奏（s/km）
        "meta": {
            "track_id": track.track_id,
            "track_udp_id": int(track_id),
            "track_length_m": track_len,
            "lap_number": sample.get("lap_number"),
            "lap_time_ms": int(lap_time),
            "tyre_class": cc,
            "tyre_compound_raw": agg.get("tyre_compound"),
            "weather_code": weather_raw,
            "weather_label": weather_label(weather_raw),
            "weather_wet": bool(weather_wet),
        },
    }


def collect_lap_files(recordings_dir: Path) -> list[Path]:
    """收集全部逐圈训练样本文件（``*_laps.jsonl``）。"""
    if not recordings_dir.is_dir():
        return []
    return sorted(recordings_dir.glob("*_laps.jsonl"))


def build_dataset(recordings_dir: Path) -> dict[str, Any]:
    """构建训练数据集：全部有效圈 → 普世化特征 + 归一化目标。"""
    rows: list[dict[str, Any]] = []
    skipped = 0
    for path in collect_lap_files(recordings_dir):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            row = build_row(sample)
            if row is None:
                skipped += 1
                continue
            rows.append(row)

    by_track: dict[str, int] = {}
    for r in rows:
        tid = r["meta"]["track_id"]
        by_track[tid] = by_track.get(tid, 0) + 1

    return {
        "schema": "f1opt-telemetry-training-dataset/1",
        "target": "lap_time_ms / track_length_m (s per km)",
        "count": len(rows),
        "skipped": skipped,
        "by_track": by_track,
        "feature_keys": sorted(rows[0]["features"].keys()) if rows else [],
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="构建普世化遥测训练数据集")
    parser.add_argument(
        "--recordings", default="data/recordings",
        help="录制目录（默认 data/recordings）",
    )
    parser.add_argument(
        "--out", default="data/training/telemetry_dataset.json",
        help="数据集输出路径",
    )
    args = parser.parse_args()

    recordings_dir = Path(args.recordings)
    dataset = build_dataset(recordings_dir)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=1), encoding="utf-8",
    )

    print("=" * 60)
    print("普世化遥测训练数据集")
    print("=" * 60)
    print(f"样本数: {dataset['count']}（跳过 {dataset['skipped']}）")
    print(f"赛道分布: {dataset['by_track']}")
    print(f"特征数: {len(dataset['feature_keys'])}")
    print(f"输出: {out}")
    if dataset["count"] < 50:
        print()
        print("提示：样本数 < 50，仅供管线验证；建议每条赛道每工况 ≥20 圈")
    return 0


if __name__ == "__main__":
    sys.exit(main())
