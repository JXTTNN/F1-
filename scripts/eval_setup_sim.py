"""评测调教性能模型「能不能真的用来调车」。

为什么需要这个脚本
------------------
``train_setup_sim_nn.py`` 打印的 ``val_mae`` 是在训练分布（全量程随机调教）
上的插值误差，回答的是「能不能估出这套调教有多慢」。但调教优化的真实操作是：

    从当前调教出发，比较两套**已经很接近**的调教，判断哪套快几十毫秒。

这两件事不是一回事。一个 90 ms 的全量程 MAE 可能对应「相邻候选排序 60% 正确」，
那就等于优化器在掷硬币 —— 这正是「模型训练好了却没在调优里发挥作用」的根因。

因此本脚本用**训练时没见过的全新样本**（同一物理仿真器生成，不同随机种子）
测四个决策相关指标：

1. ``fresh_local_mae_ms``：最优点 ±0.30 归一化单位内重新采样的绝对误差；
2. ``pairwise_acc``：同赛道内真实差 10~100 ms 的样本对，模型排序正确的比例
   （优化器每次「接受/拒绝」候选，本质就是一次这样的比较）；
3. ``direction_agreement``：最优点处逐个参数 ±1 档，模型与仿真器的方向是否一致
   （决定优化器能否走对方向；近乎平坦的参数不计入分母）；
4. ``zero_bias_ms``：默认调教的预测偏差（模型应以 0 为基准）。

诚实边界
--------
标签仍来自 ``simulate_lap_delta_ms``（遥测锚定的准静态物理仿真），**不是真实
跑圈**。本脚本证明的是「模型对物理仿真器的保真度是否足够支撑调优决策」，
不证明「对真实赛道的预测准确」。``labels.real_setup_laps`` 恒为 0。

用法::

    python scripts/eval_setup_sim.py --strict
    python scripts/eval_setup_sim.py --json out.json --md out.md
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_setup_sim_dataset import (  # noqa: E402
    _half_range,
    coarse_optimum,
    load_corner_stats,
    simulate_lap_delta_ms,
    snap_to_step,
)
from setup_tuner.domain.setup import ALL_SETUP_FIELDS, get_field  # noqa: E402
from setup_tuner.engine.setup_sim import SetupSimModel  # noqa: E402

#: 默认模型路径（与打包位置一致）
DEFAULT_MODEL = ROOT / "setup_tuner" / "resources" / "models" / "setup_sim_nn.json"
DEFAULT_JSON = ROOT / "setup_tuner" / "resources" / "models" / "setup_sim_eval.json"
DEFAULT_MD = ROOT / "setup_tuner" / "resources" / "models" / "setup_sim_eval.md"

#: 每个赛道的新鲜评测样本数（局部 / 全量程）
N_LOCAL = 50
N_WIDE = 30
#: 局部采样半宽（归一化单位）—— 与数据集构建的最内两档对齐
LOCAL_HALF_WIDTH = 0.30
#: 成对比较的「真实差」区间（ms）
PAIR_LO_MS = 10.0
PAIR_HI_MS = 100.0
#: 方向一致性只统计真实增益超过该阈值的参数（更小的视为平坦，方向无意义）
DIR_MIN_MS = 3.0

#: 门禁阈值（``--strict`` 时生效）。全量程 MAE 不作门禁 —— 优化器不会去那里。
GATES = {
    "fresh_local_mae_ms": ("<=", 120.0),
    "pairwise_acc": (">=", 0.75),
    "direction_agreement": (">=", 0.80),
    "zero_bias_ms": ("<=", 40.0),
}


def _neighbor_norm(name: str, norm: float, direction: int) -> float | None:
    """把参数推动一个**真实档位**（``step``）；越界返回 None。"""
    spec = get_field(name)
    real = spec.default + float(norm) * _half_range(spec)
    moved = real + direction * spec.step
    if moved < spec.min_val - 1e-9 or moved > spec.max_val + 1e-9:
        return None
    return (moved - spec.default) / _half_range(spec)


def _mae(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return 0.0
    return sum(abs(a - b) for a, b in pairs) / len(pairs)


def evaluate(model_path: Path, seed: int = 20261102) -> dict[str, Any]:
    """跑全部指标，返回机器可读字典。"""
    model = SetupSimModel(path=str(model_path.resolve()))
    if not model.available:
        raise SystemExit(f"模型不可用（{model.reason}）：{model_path}")

    specs = list(ALL_SETUP_FIELDS)
    stats = load_corner_stats(None)
    if not stats:
        raise SystemExit("没有可用的赛道锚点数据（先跑 build_corner_dataset.py）")

    rng = random.Random(seed)
    per_track: dict[str, Any] = {}
    local_pairs: list[tuple[float, float]] = []
    wide_pairs: list[tuple[float, float]] = []
    pair_total = 0
    pair_ok = 0
    pair_total_30 = 0
    pair_ok_30 = 0
    dir_total = 0
    dir_ok = 0
    zero_bias: list[float] = []
    raw_zero_bias: list[float] = []
    skipped: list[str] = []

    for tid in sorted(stats):
        if not model.covers(tid):
            skipped.append(tid)
            continue
        track = stats[tid]
        optimum, opt_ms = coarse_optimum(track, specs)

        # ---- 新鲜样本：路径管（与训练同分布）+ 全量程 --------------------- #
        samples: list[dict[str, float]] = []
        for _ in range(N_LOCAL):
            # 优化器从当前调教出发逐步走向最优，活动区是「默认→最优」路径的
            # 管状邻域；评测样本必须落在同一区域，否则测的不是真实决策场景。
            t = rng.uniform(0.0, 1.0)
            samples.append({
                s.name: snap_to_step(
                    t * optimum[s.name]
                    + rng.uniform(-LOCAL_HALF_WIDTH, LOCAL_HALF_WIDTH),
                    s,
                )
                for s in specs
            })
        fresh_local = samples[:]
        fresh_wide: list[dict[str, float]] = []
        for _ in range(N_WIDE):
            raw = {
                s.name: s.min_val + rng.randint(
                    0, max(1, int(round((s.max_val - s.min_val) / s.step)))
                ) * s.step
                for s in specs
            }
            fresh_wide.append({s.name: snap_to_step(_norm(raw[s.name], s), s) for s in specs})

        t_loc: list[float] = []
        p_loc: list[float] = []
        for sn in fresh_local:
            truth = simulate_lap_delta_ms(track, sn)
            pred = model.predict_delta_s(tid, sn)
            if pred is None:
                continue
            t_loc.append(truth)
            p_loc.append(pred * 1000.0)
            local_pairs.append((truth, pred * 1000.0))
        t_wide: list[float] = []
        p_wide: list[float] = []
        for sn in fresh_wide:
            truth = simulate_lap_delta_ms(track, sn)
            pred = model.predict_delta_s(tid, sn)
            if pred is None:
                continue
            t_wide.append(truth)
            p_wide.append(pred * 1000.0)
            wide_pairs.append((truth, pred * 1000.0))

        # ---- 成对排序：优化器的「接受/拒绝」就是这个问题 ------------------ #
        for i in range(len(t_loc)):
            for j in range(i + 1, len(t_loc)):
                dy_true = t_loc[i] - t_loc[j]
                ady = abs(dy_true)
                if not (PAIR_LO_MS <= ady <= PAIR_HI_MS):
                    continue
                dy_pred = p_loc[i] - p_loc[j]
                pair_total += 1
                if (dy_true > 0) == (dy_pred > 0):
                    pair_ok += 1
                if ady >= 30.0:
                    pair_total_30 += 1
                    if (dy_true > 0) == (dy_pred > 0):
                        pair_ok_30 += 1

        # ---- 方向一致性：优化器能否朝正确方向移动 ------------------------ #
        base_pred = model.predict_delta_s(tid, optimum)
        for spec in specs:
            for direction in (+1, -1):
                cand = _neighbor_norm(spec.name, optimum[spec.name], direction)
                if cand is None:
                    continue
                cand_delta = dict(optimum)
                cand_delta[spec.name] = cand
                d_true = simulate_lap_delta_ms(track, cand_delta) - opt_ms
                if abs(d_true) < DIR_MIN_MS:
                    continue
                pred = model.predict_delta_s(tid, cand_delta)
                if pred is None or base_pred is None:
                    continue
                d_pred = (pred - base_pred) * 1000.0
                dir_total += 1
                if (d_true > 0) == (d_pred > 0):
                    dir_ok += 1

        # ---- 零偏差：默认调教的预测应接近 0 ------------------------------ #
        # ``predict_delta_s`` 是**上线口径**（含每赛道差分校准），默认点必为 0；
        # 同时记录未校准的原始偏置，用来观察模型本身的系统性误差（透明度）。
        zero_setup = {s.name: 0.0 for s in specs}
        z = model.predict_delta_s(tid, zero_setup)
        if z is not None:
            zero_bias.append(abs(z * 1000.0))
        zr = model.predict_raw_delta_s(tid, zero_setup)
        if zr is not None:
            raw_zero_bias.append(abs(zr * 1000.0))

        per_track[tid] = {
            "optimum_delta_ms": round(opt_ms, 2),
            "n_local": len(t_loc),
            "fresh_local_mae_ms": round(_mae(list(zip(t_loc, p_loc))), 2),
            "fresh_wide_mae_ms": round(_mae(list(zip(t_wide, p_wide))), 2),
        }

    def _acc(ok: int, total: int) -> float:
        return round(ok / total, 4) if total else 0.0

    result = {
        "schema": "f1opt-setup-sim-eval/1",
        "model": str(model_path.name),
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": seed,
        "n_tracks": len(per_track),
        "skipped_tracks": skipped,
        "labels": {
            "source": "simulate_lap_delta_ms（遥测锚定准静态物理仿真）",
            "real_setup_laps": 0,
            "note": "以下指标衡量对物理仿真器的保真度，不构成真实跑圈验证",
        },
        "metrics": {
            "fresh_local_mae_ms": round(_mae(local_pairs), 2),
            "fresh_wide_mae_ms": round(_mae(wide_pairs), 2),
            "pairwise_acc": _acc(pair_ok, pair_total),
            "pairwise_acc_30_100ms": _acc(pair_ok_30, pair_total_30),
            "n_pairs": pair_total,
            "n_pairs_30_100ms": pair_total_30,
            "direction_agreement": _acc(dir_ok, dir_total),
            "n_direction_checks": dir_total,
            "zero_bias_ms": round(sum(zero_bias) / len(zero_bias), 2) if zero_bias else 0.0,
            "raw_zero_bias_ms": (
                round(sum(raw_zero_bias) / len(raw_zero_bias), 2) if raw_zero_bias else 0.0
            ),
            "train_val_mae_ms": model.metrics.get("val_mae_ms"),
            "train_val_mae_ms_local": model.metrics.get("val_mae_ms_local"),
            "model_tracks": len(model.track_ids),
            "model_samples": model.n_samples,
        },
        "per_track": per_track,
    }
    result["gate"] = _check_gates(result["metrics"])
    return result


def _norm(value: float, spec: Any) -> float:
    """真实值 → 归一化（与数据集构建同口径）。"""
    return (float(value) - spec.default) / _half_range(spec)


def _check_gates(metrics: dict[str, Any]) -> dict[str, Any]:
    """按 :data:`GATES` 判定，返回逐项结果与总判定。"""
    checks: dict[str, Any] = {}
    passed = True
    for key, (op, threshold) in GATES.items():
        value = metrics.get(key)
        if value is None:
            checks[key] = {"value": None, "threshold": f"{op}{threshold}", "ok": False}
            passed = False
            continue
        ok = (value <= threshold) if op == "<=" else (value >= threshold)
        checks[key] = {
            "value": value, "threshold": f"{op}{threshold}", "ok": bool(ok),
        }
        passed = passed and bool(ok)
    return {"passed": passed, "checks": checks}


def render_md(result: dict[str, Any]) -> str:
    """人类可读报告。"""
    m = result["metrics"]
    g = result["gate"]
    lines = [
        "# 调教性能模型评测",
        "",
        f"- 模型：`{result['model']}`",
        f"- 评测时间：{result['evaluated_at']}",
        f"- 赛道：{result['n_tracks']} 条（跳过 {len(result['skipped_tracks'])} 条未覆盖）",
        f"- 标签来源：{result['labels']['source']}",
        f"- **真实跑圈样本：{result['labels']['real_setup_laps']}**",
        "",
        "## 决策相关指标",
        "",
        "| 指标 | 值 | 门槛 | 结论 |",
        "|---|---|---|---|",
    ]
    for key, chk in g["checks"].items():
        mark = "PASS" if chk["ok"] else "FAIL"
        lines.append(f"| `{key}` | {chk['value']} | {chk['threshold']} | {mark} |")
    lines += [
        "",
        "## 全部指标",
        "",
        "| 指标 | 值 |",
        "|---|---|",
    ]
    for key, value in m.items():
        lines.append(f"| `{key}` | {value} |")
    lines += [
        "",
        "## 说明",
        "",
        "- `pairwise_acc` 是优化器每次「接受/拒绝候选」的实际上限：低于 0.5 等于掷硬币。",
        "- `direction_agreement` 决定坐标上升能否朝正确方向走；只统计真实增益 ≥3 ms 的参数。",
        "- 上表全部由**训练时未见过**的样本（不同随机种子）测得，不是训练集指标。",
        f"- {result['labels']['note']}。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="评测调教性能模型的决策能力")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--json", default=str(DEFAULT_JSON))
    ap.add_argument("--md", default=str(DEFAULT_MD))
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--strict", action="store_true", help="门禁不通过则退出码 1")
    args = ap.parse_args(argv)

    t0 = time.time()
    result = evaluate(Path(args.model), seed=args.seed)
    elapsed = time.time() - t0

    m = result["metrics"]
    print(f"评测完成（{elapsed:.1f}s，{result['n_tracks']} 赛道）")
    print(f"  局部新鲜样本 MAE : {m['fresh_local_mae_ms']:.2f} ms")
    print(f"  全量程新鲜样本 MAE: {m['fresh_wide_mae_ms']:.2f} ms")
    print(f"  成对排序准确率   : {m['pairwise_acc']:.3f} "
          f"（10~100ms, n={m['n_pairs']}）| ≥30ms: {m['pairwise_acc_30_100ms']:.3f}")
    print(f"  方向一致率       : {m['direction_agreement']:.3f} "
          f"（n={m['n_direction_checks']}）")
    print(f"  默认调教零偏差   : {m['zero_bias_ms']:.2f} ms")
    print(f"  门禁             : {'PASS' if result['gate']['passed'] else 'FAIL'}")
    for key, chk in result["gate"]["checks"].items():
        if not chk["ok"]:
            print(f"    - {key} = {chk['value']} 未达 {chk['threshold']}")

    out_json = Path(args.json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    out_md = Path(args.md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_md(result), encoding="utf-8")
    print(f"写出: {out_json} / {out_md}")

    if args.strict and not result["gate"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
