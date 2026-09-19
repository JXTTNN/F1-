"""训练数据覆盖度分析 + 下一步采集清单生成。

读取本地数据（`data/training/telemetry_dataset.json` + `data/recordings/*_laps.jsonl`），
按六个维度量化缺口，并输出「下一次该跑哪条赛道、什么配方、几圈」的任务清单。

维度：
    1. 赛道覆盖   —— 距 2026 赛历 24 站还差多少
    2. setup 多样性 —— 每赛道不同 setup 套数（模型学"改动方向"的关键）
    3. 配方        —— 每赛道是否覆盖软/硬两档
    4. 天气        —— 干/湿分离
    5. 损伤工况    —— 有无损伤标签样本（训练混淆控制）
    6. 2026 空力  —— aero_straight_ratio / overtake_active_ratio 是否有非零样本

用法::

    python scripts/coverage_analyzer.py
    python scripts/coverage_analyzer.py --out data/training/coverage_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: 每赛道目标：2 配方 × 3 套 setup × 5 圈（best-of-5）
TARGET_LAPS_PER_TRACK = 30
TARGET_SETUPS_PER_TRACK = 3
TARGET_COMPOUNDS_PER_TRACK = 2


def _load_local_rows() -> list[dict[str, Any]]:
    p = ROOT / "data" / "training" / "telemetry_dataset.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("rows") or []


def _load_recording_samples() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in sorted((ROOT / "data" / "recordings").glob("*_laps.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _track_names() -> dict[int, str]:
    from setup_tuner.domain.track import ALL_TRACKS
    return {t.udp_track_id: t.track_id for t in ALL_TRACKS}


def _external_coverage() -> dict[str, int]:
    """外部 2026 专业遥测的每赛道逐圈文件数（.ref/tracing_2026）。"""
    from setup_tuner.domain.track import ALL_TRACKS

    race_to_track = {t.official_name: t.track_id for t in ALL_TRACKS}
    cache = ROOT / ".ref" / "tracing_2026"
    out: dict[str, int] = {}
    if not cache.exists():
        return out
    for race_dir in cache.iterdir():
        if not race_dir.is_dir():
            continue
        tid = race_to_track.get(race_dir.name)
        if tid is None:
            continue
        n = sum(1 for _ in race_dir.glob("*/*/*_tel.json"))
        if n:
            out[tid] = n
    return out


def analyze() -> dict[str, Any]:
    rows = _load_local_rows()
    samples = _load_recording_samples()
    names = _track_names()

    by_track: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"laps": 0, "setups": set(), "compounds": set(),
                 "weather": set(), "damage_laps": 0,
                 "aero_laps": 0, "overtake_laps": 0}
    )
    def _norm_tid(v: Any) -> int | None:
        """训练来源多样（int/str），统一成 int，非数字则丢弃。"""
        if isinstance(v, bool) or v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    for r in rows:
        meta = r.get("meta") or {}
        # 本地数据集：track_id 是短 id（str），track_udp_id 才是官方枚举（int）
        tid = _norm_tid(meta.get("track_udp_id"))
        if tid is None:
            tid = _norm_tid(meta.get("track_id"))
        if tid is None:
            continue
        g = by_track[tid]
        g["laps"] += 1
        if meta.get("tyre_class"):
            g["compounds"].add(meta["tyre_class"])
        if meta.get("weather_label"):
            g["weather"].add(meta["weather_label"])
        if meta.get("setup"):
            g["setups"].add(tuple(sorted(meta["setup"].items())))

    # 记录样本里的损伤/空力/超车覆盖（来自 .f1rec 逐圈导出）
    for s in samples:
        tid = _norm_tid(s.get("track_id"))
        if tid is None:
            continue
        g = by_track[tid]
        # setup 多样性来自 .f1rec 逐圈样本（数据集 features 不含 setup 明细）
        setup = s.get("setup")
        if isinstance(setup, dict) and setup:
            g["setups"].add(tuple(sorted(setup.items())))
        if s.get("damage_severe") or (s.get("damage_max") or 0) > 0:
            g["damage_laps"] += 1
        aero = s.get("aero_straight_ratio")
        if isinstance(aero, (int, float)) and aero > 0:
            g["aero_laps"] += 1
        ot = s.get("overtake_active_ratio")
        if isinstance(ot, (int, float)) and ot > 0:
            g["overtake_laps"] += 1

    total_laps = sum(g["laps"] for g in by_track.values())
    all_setups: set[tuple] = set()
    for g in by_track.values():
        all_setups |= g["setups"]
    wet_laps = sum(
        1 for r in rows if (r.get("meta") or {}).get("weather_label") == "wet"
    )

    tasks: list[dict[str, Any]] = []
    for tid, g in sorted(by_track.items()):
        name = names.get(tid, str(tid))
        need_laps = max(0, TARGET_LAPS_PER_TRACK - g["laps"])
        n_setups = len(g["setups"])
        need_setups = max(0, TARGET_SETUPS_PER_TRACK - n_setups)
        need_compounds = max(0, TARGET_COMPOUNDS_PER_TRACK - len(g["compounds"]))
        reasons = []
        if need_laps:
            reasons.append(f"缺 {need_laps} 圈")
        if need_setups:
            reasons.append(f"setup 仅 {n_setups} 套，需 +{need_setups}")
        if need_compounds:
            reasons.append(f"配方仅 {sorted(g['compounds'])}，需 +{need_compounds}")
        if g["damage_laps"] == 0:
            reasons.append("无损伤样本")
        if g["aero_laps"] == 0:
            reasons.append("无主动空力样本")
        if g["overtake_laps"] == 0:
            reasons.append("无超车模式样本")
        if reasons:
            tasks.append({
                "track_id": tid, "track": name,
                "current_laps": g["laps"], "need_laps": need_laps,
                "setups": n_setups, "compounds": sorted(g["compounds"]),
                "weather": sorted(g["weather"]),
                "reasons": reasons,
            })

    tasks.sort(key=lambda t: (-t["need_laps"], -len(t["reasons"])))
    external = _external_coverage()
    return {
        "summary": {
            "tracks_covered": len(by_track),
            "tracks_total_2026": 24,
            "total_valid_laps": total_laps,
            "unique_setups": len(all_setups),
            "wet_laps": wet_laps,
            "target_laps_t2": 8 * TARGET_LAPS_PER_TRACK,
            "external_2026_cache": str(ROOT / ".ref" / "tracing_2026"),
            "external_tracks": len(external),
            "external_laps": sum(external.values()),
        },
        "external_by_track": external,
        "by_track": {
            names.get(tid, str(tid)): {
                "laps": g["laps"],
                "setups": len(g["setups"]),
                "compounds": sorted(g["compounds"]),
                "weather": sorted(g["weather"]),
                "damage_laps": g["damage_laps"],
                "aero_laps": g["aero_laps"],
                "overtake_laps": g["overtake_laps"],
            }
            for tid, g in sorted(by_track.items())
        },
        "tasks": tasks,
    }


def _render_markdown(rep: dict[str, Any]) -> str:
    s = rep["summary"]
    lines = [
        "# 训练数据覆盖度报告（F1 2026）",
        "",
        f"- 赛道覆盖：**{s['tracks_covered']} / {s['tracks_total_2026']}**",
        f"- 有效圈：**{s['total_valid_laps']}**（T2 泛化目标 ≥ {s['target_laps_t2']} 圈）",
        f"- 不同 setup：**{s['unique_setups']}**（关键短板：模型靠它学「改动方向」）",
        f"- 湿地圈：**{s['wet_laps']}**",
        f"- 外部 2026 专业基准：**{s.get('external_tracks', 0)}** 条赛道 / "
        f"**{s.get('external_laps', 0)}** 圈（TracingInsights 2026，年检通过）",
        "",
        "## 各赛道明细",
        "",
        "| 赛道 | 圈数 | setup 数 | 配方 | 天气 | 损伤样本 | 主动空力 | 超车模式 | 外部 2026 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    ext = rep.get("external_by_track", {})
    for name, g in rep["by_track"].items():
        lines.append(
            f"| {name} | {g['laps']} | {g['setups']} | {','.join(g['compounds']) or '-'} "
            f"| {','.join(g['weather']) or '-'} | {g['damage_laps']} | {g['aero_laps']} "
            f"| {g['overtake_laps']} | {ext.get(name, 0)} |"
        )
    lines += ["", "## 下一步采集清单（按缺口优先级）", ""]
    for i, t in enumerate(rep["tasks"][:10], 1):
        lines.append(f"{i}. **{t['track']}** —— 已有 {t['current_laps']} 圈 / "
                     f"{t['setups']} 套 setup / {t['compounds'] or '无配方记录'}；"
                     f"建议补 {t['need_laps']} 圈。原因：{'；'.join(t['reasons'])}")
    if not rep["tasks"]:
        lines.append("(无缺口)")
    lines += ["", "## 采集协议（每赛道）", "",
              f"- 2 配方 × {TARGET_SETUPS_PER_TRACK} 套 setup × 5 圈 best-of-5 "
              f"= {TARGET_LAPS_PER_TRACK} 圈",
              "- 每套 setup 只改 1-2 个参数，形成可辨识的对照",
              "- 刻意采集：损伤前后、主动空力直道模式、超车模式激活、湿地",
              ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="训练数据覆盖度分析")
    ap.add_argument("--out", default=str(ROOT / "data" / "training" / "coverage_report.md"))
    ap.add_argument("--json", action="store_true", help="同时输出 JSON")
    args = ap.parse_args(argv)

    rep = analyze()
    md = _render_markdown(rep)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    if args.json:
        out.with_suffix(".json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    print(md)
    print(f"\n写出: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
