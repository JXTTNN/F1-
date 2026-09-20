"""用真实遥测训练调教优化用的代理模型（surrogate）。

两个头
------
**头 A：弯速代理（telemetry-trained，样本来自 2026 真实逐点遥测）**
    输入 = 弯型 / 轮胎 / 天气 / 温度 / 赛道需求画像 / 弯号
    输出 = 该弯的通过时间 ``corner_time_s``
    用途（这就是"遥测数据训练出来的东西到底怎么用"）：
      1. 给出**数据驱动的逐弯重要度** —— 取代 ``lap_model`` 里
         ``importance = 120 / 该弯参考速度`` 这条纯经验启发式；
      2. 给出轮胎/温度/天气对弯速的**实测敏感度**，供需求倾斜使用。

**头 B：整圈配速代理（含 setup 与车手风格，样本来自本地真实跑圈）**
    输入 = 22 项调教参数 + 轮胎/天气/温度 + 11 维车手风格 + 赛道需求
    输出 = ``lap_time_ms``
    用途：作为"NN 模拟"的目标函数分量。样本量小是事实，因此**不假装可用**：
    用留一交叉验证算出 ``skill = 1 - MAE_model / MAE_baseline``，只有当它
    真的优于"取均值"这个基线时才 > 0；样本少时 skill 自然接近 0，
    模型不会去覆盖物理方向（C 矩阵）—— 这比编造一个"高精度模型"诚实得多。

两个学习器
----------
- **岭回归**（闭式解，纯标准库）：全量样本上精确、秒级完成；
- **MLP**（:mod:`setup_tuner.engine.pure_nn`）：在分层子样本上训练，
  结构 + Adam 反传全部纯标准库实现。

两者都在**同一留出集**上评估，谁 MAE 低就由引擎优先使用；两个都存进
``data/models/telemetry_surrogate.json``，指标一并落盘，便于复核。

用法::

    python scripts/train_telemetry_surrogate.py
    python scripts/train_telemetry_surrogate.py --corner-samples 4000 --epochs 40
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORNER_JSONL = ROOT / "data" / "training" / "corner_dataset.jsonl"
RECORDINGS = ROOT / "data" / "recordings"
# 默认输出到**包内资源**（随包分发；pip 安装后引擎仍能找到）
OUT = ROOT / "setup_tuner" / "resources" / "models" / "telemetry_surrogate.json"

#: 头 A 特征列（顺序即模型输入顺序，必须与引擎侧一致）
CORNER_FEATURE_KEYS: tuple[str, ...] = (
    "cls_slow", "cls_medium", "cls_fast",
    "tyre_softness", "tyre_medium", "tyre_wet",
    "wet", "track_temp", "air_temp",
    "traction_index", "aero_index", "braking_index",
    "slow_share", "fast_share",
)

#: 头 B 特征列（在 setup 之外追加工况与风格）
LAP_CONDITION_KEYS: tuple[str, ...] = (
    "tyre_softness", "wet", "track_temp", "air_temp",
    "traction_index", "aero_index", "braking_index",
    "slow_share", "fast_share",
)

#: 本地遥测 setup 字段名 → 本项目的规范参数名
SETUP_KEY_MAP: dict[str, str] = {
    "m_frontWing": "front_wing",
    "m_rearWing": "rear_wing",
    "m_onThrottleDiff": "on_throttle_diff",
    "m_offThrottleDiff": "off_throttle_diff",
    "m_frontCamber": "front_camber",
    "m_rearCamber": "rear_camber",
    "m_frontToe": "front_toe",
    "m_rearToe": "rear_toe",
    "m_frontSuspension": "front_suspension",
    "m_rearSuspension": "rear_suspension",
    "m_frontAntiRollBar": "front_anti_roll_bar",
    "m_rearAntiRollBar": "rear_anti_roll_bar",
    "m_frontSuspensionHeight": "front_ride_height",
    "m_rearSuspensionHeight": "rear_ride_height",
    "m_brakePressure": "brake_pressure",
    "m_brakeBias": "brake_bias",
    "m_frontLeftTyrePressure": "front_left_tyre_pressure",
    "m_frontRightTyrePressure": "front_right_tyre_pressure",
    "m_rearLeftTyrePressure": "rear_left_tyre_pressure",
    "m_rearRightTyrePressure": "rear_right_tyre_pressure",
}


# --------------------------------------------------------------------------- #
# 岭回归（闭式解，纯标准库）
# --------------------------------------------------------------------------- #
def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """高斯消元解线性方程组（带部分主元），返回解向量。"""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            continue
        m[col], m[pivot] = m[pivot], m[col]
        pv = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                m[r][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]


class Ridge:
    """岭回归（含截距与按列标准化）。"""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.coef: list[float] = []
        self.intercept: float = 0.0
        self.mean: list[float] = []
        self.std: list[float] = []

    def fit(self, x: list[list[float]], y: list[float]) -> Ridge:
        n, d = len(x), len(x[0])
        self.mean = [sum(row[j] for row in x) / n for j in range(d)]
        self.std = []
        for j in range(d):
            var = sum((row[j] - self.mean[j]) ** 2 for row in x) / n
            self.std.append(math.sqrt(var) if var > 1e-12 else 1.0)
        z = [[(row[j] - self.mean[j]) / self.std[j] for j in range(d)] for row in x]
        ybar = sum(y) / n
        yc = [v - ybar for v in y]
        xtx = [[sum(z[i][a] * z[i][b] for i in range(n)) for b in range(d)]
               for a in range(d)]
        for j in range(d):
            xtx[j][j] += self.alpha
        xty = [sum(z[i][j] * yc[i] for i in range(n)) for j in range(d)]
        self.coef = _solve(xtx, xty)
        self.intercept = ybar
        return self

    def predict(self, x: list[list[float]]) -> list[float]:
        out: list[float] = []
        for row in x:
            s = self.intercept
            for j, c in enumerate(self.coef):
                s += c * (row[j] - self.mean[j]) / self.std[j]
            out.append(s)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "ridge", "alpha": self.alpha,
            "coef": self.coef, "intercept": self.intercept,
            "mean": self.mean, "std": self.std,
        }


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def _mae(y: list[float], p: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y, p, strict=True)) / max(1, len(y))


def _r2(y: list[float], p: list[float]) -> float:
    if not y:
        return 0.0
    m = sum(y) / len(y)
    ss_res = sum((a - b) ** 2 for a, b in zip(y, p, strict=True))
    ss_tot = sum((a - m) ** 2 for a in y)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


# --------------------------------------------------------------------------- #
# 头 A：弯速代理
# --------------------------------------------------------------------------- #
def _corner_row(
    s: dict[str, Any], pair_index: dict[str, int],
) -> list[float]:
    """弯速特征：(赛道×弯号) one-hot + 弯型/轮胎/天气/温度/赛道需求。

    为什么必须带 (赛道×弯号) 身份：弯时主要由"哪个赛道的哪个弯"决定
    （Monaco T1 与 Monza T1 完全不是一个量级）。只给赛道身份时实测 R²≈0.11；
    给到弯级身份后，模型才能给出**逐弯**的期望值与残差。
    条件项（轮胎/天气/温度）则从"同一弯在不同工况下"的差异里学到敏感度。
    """
    cls = s.get("corner_class")
    tyre = s.get("tyre_class")
    onehot = [0.0] * len(pair_index)
    key = f"{s.get('track_id')}#{s.get('corner')}"
    pi = pair_index.get(key)
    if pi is not None:
        onehot[pi] = 1.0
    return onehot + [
        1.0 if cls == "slow" else 0.0,
        1.0 if cls == "medium" else 0.0,
        1.0 if cls == "fast" else 0.0,
        float(s.get("tyre_softness") or 0.5),
        1.0 if tyre == "medium" else 0.0,
        1.0 if tyre in ("wet", "inter") else 0.0,
        float(s.get("wet") or 0.0),
        float(s.get("track_temp") or 35.0),
        float(s.get("air_temp") or 24.0),
        float(s.get("track_traction_index") or 0.0),
        float(s.get("track_aero_index") or 0.0),
        float(s.get("track_braking_index") or 0.0),
        float(s.get("track_slow_share") or 0.0),
        float(s.get("track_fast_share") or 0.0),
    ]


def train_corner_head(
    samples: list[dict[str, Any]],
    mlp_samples: int,
    epochs: int,
    seed: int,
    ridge_samples: int = 20000,
    force_model: str = "auto",
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for s in samples:
        tid = s["track_id"]
        counts[tid] = max(counts.get(tid, 0), int(s.get("corner") or 0))
    track_ids = sorted({str(s["track_id"]) for s in samples})
    pair_keys = sorted({f"{s['track_id']}#{s['corner']}" for s in samples})
    pair_index = {k: i for i, k in enumerate(pair_keys)}
    x = [_corner_row(s, pair_index) for s in samples]
    # 目标用**弯时占圈速的比例**（无量纲）：同一弯在慢圈里会整体变慢，
    # 若直接回归绝对弯时，整圈配速噪声会吃掉全部解释力（实测 R²≈0.01）。
    # 用占比后，"这个弯在该赛道的圈速里占多少"才是可学的稳定量。
    y = [float(s["corner_time_s"]) / float(s["lap_time_s"]) for s in samples]

    rng = random.Random(seed)
    idx = list(range(len(x)))
    rng.shuffle(idx)
    n_val = max(200, int(0.2 * len(x)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    x_va = [x[i] for i in val_idx]
    y_va = [y[i] for i in val_idx]

    # 岭回归在 127 维特征上做闭式解，全量 79k 会使 XᵀX 累加变成纯 Python 瓶颈
    # （约 1.3G 次乘加）。统计上 2 万样本对 127 个参数已远超需求，故设上限。
    r_tr = train_idx[:ridge_samples] if ridge_samples > 0 else train_idx
    ridge = Ridge(alpha=1.0).fit([x[i] for i in r_tr], [y[i] for i in r_tr])
    r_pred = ridge.predict(x_va)

    mlp_metrics: dict[str, Any] | None = None
    mlp = None
    if mlp_samples > 0:
        sub = train_idx[:mlp_samples]
        from setup_tuner.engine.pure_nn import MLP

        mlp = MLP([len(pair_keys) + len(CORNER_FEATURE_KEYS), 32, 16, 1], seed=seed)
        mlp.fit([x[i] for i in sub], [[y[i]] for i in sub],
                epochs=epochs, lr=0.01, batch_size=64)
        m_pred = [v[0] for v in mlp.predict(x_va)]
        mlp_metrics = {
            "val_mae": round(_mae(y_va, m_pred), 4),
            "val_r2": round(_r2(y_va, m_pred), 4),
            "train_samples": len(sub),
        }

    ridge_metrics = {
        "val_mae": round(_mae(y_va, r_pred), 4),
        "val_r2": round(_r2(y_va, r_pred), 4),
        "train_samples": len(r_tr),
        "n_samples_total": len(x),
    }
    winner = "mlp"
    if mlp_metrics is None or mlp_metrics["val_mae"] > ridge_metrics["val_mae"]:
        winner = "ridge"
    if force_model in ("ridge", "mlp"):
        if force_model == "mlp" and mlp_metrics is None:
            raise SystemExit("--force-model mlp 需要同时训练 MLP（--corner-samples > 0）")
        winner = force_model

    # 数据驱动的逐弯重要度（在**全量**样本上重拟合，避免只用训练集）
    f_idx = idx[:ridge_samples] if ridge_samples > 0 else idx
    full_ridge = Ridge(alpha=1.0).fit([x[i] for i in f_idx], [y[i] for i in f_idx])
    median_lap: dict[str, float] = {}
    for tid in track_ids:
        vals = [float(s["lap_time_s"]) for s in samples if str(s["track_id"]) == tid]
        median_lap[tid] = statistics.median(vals) if vals else 0.0
    share_acc: dict[str, dict[str, list[float]]] = {}
    for s in samples:
        tid = str(s["track_id"])
        pred = full_ridge.predict([_corner_row(s, pair_index)])[0]
        share_acc.setdefault(tid, {}).setdefault(str(s["corner"]), []).append(pred)
    share_by_corner: dict[str, dict[str, float]] = {}
    for tid, corners in share_acc.items():
        share_by_corner[tid] = {
            num: statistics.median(vals) for num, vals in corners.items()
        }
    # 归一化为重要度（同一赛道内各弯占比之和 = 1）
    importance: dict[str, dict[str, float]] = {}
    for tid, shares in share_by_corner.items():
        total = sum(shares.values()) or 1.0
        importance[tid] = {num: round(v / total, 6) for num, v in shares.items()}
    expected_times = {
        tid: {
            num: round(share * median_lap.get(tid, 0.0), 4)
            for num, share in shares.items()
        }
        for tid, shares in share_by_corner.items()
    }
    corner_class_share: dict[str, float] = {}
    for s in samples:
        key = str(s.get("corner_class"))
        corner_class_share[key] = corner_class_share.get(key, 0.0) + float(s["corner_time_s"])
    total_share = sum(corner_class_share.values()) or 1.0
    corner_class_share = {k: round(v / total_share, 4)
                          for k, v in corner_class_share.items()}

    out: dict[str, Any] = {
        "feature_keys": [f"pair={k}" for k in pair_keys] + list(CORNER_FEATURE_KEYS),
        "pair_keys": pair_keys,
        "track_ids": track_ids,
        "target": "corner_time_share",
        "n_samples": len(samples),
        "tracks": sorted(share_by_corner),
        "corner_count_by_track": counts,
        "family": {
            "ridge": {"model": ridge.to_dict(), "metrics": ridge_metrics,
                      "full_model": full_ridge.to_dict()},
            "winner": winner,
        },
        "mlp": None,
        "expected_corner_time_s": expected_times,
        "corner_importance": importance,
        "corner_class_time_share": corner_class_share,
    }
    if mlp is not None:
        out["mlp"] = {"model": mlp.to_dict(), "metrics": mlp_metrics}
    # 让引擎可以直接按 winner 取模型
    if winner == "mlp" and mlp is not None:
        out["active"] = {"kind": "mlp", "metrics": mlp_metrics}
    else:
        out["active"] = {"kind": "ridge", "metrics": ridge_metrics}
    return out


# --------------------------------------------------------------------------- #
# 头 B：整圈配速代理（含 setup / 风格）
# --------------------------------------------------------------------------- #
def _load_local_laps() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not RECORDINGS.exists():
        return rows
    for f in sorted(RECORDINGS.glob("*_laps.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def train_lap_head(seed: int) -> dict[str, Any]:
    from setup_tuner.domain.setup import ALL_SETUP_FIELDS, get_field
    from setup_tuner.domain.track import ALL_TRACKS
    from setup_tuner.engine.holistic import track_demand

    udp_to_short = {t.udp_track_id: t.track_id for t in ALL_TRACKS}
    rows = _load_local_laps()
    setup_names = [f.name for f in ALL_SETUP_FIELDS]
    feats: list[list[float]] = []
    targets: list[float] = []
    tids: list[str] = []
    used = 0
    times_by_track: dict[str, list[float]] = {}

    for r in rows:
        if not r.get("lap_valid", True):
            continue
        lt = r.get("lap_time_ms")
        if not isinstance(lt, (int, float)) or lt <= 0:
            continue
        tid = udp_to_short.get(r.get("track_id"))
        if tid is None:
            continue
        times_by_track.setdefault(tid, []).append(float(lt))
        setup = r.get("setup") or {}
        style = r.get("style") or [0.0] * 11
        weights = r.get("tyres_wear") or []
        rd = track_demand(tid)
        wet = 1.0 if int(r.get("weather") or 0) >= 3 else 0.0
        comp = None
        agg = r.get("lap_agg") or {}
        if agg.get("is_soft_compound"):
            comp = "soft"
        elif agg.get("is_hard_compound"):
            comp = "hard"
        else:
            comp = "medium"
        tyre_map = {"soft": 1.0, "medium": 0.6, "hard": 0.3}
        row: list[float] = []
        for name in setup_names:
            tel_name = next((k for k, v in SETUP_KEY_MAP.items() if v == name), None)
            val = setup.get(tel_name) if tel_name else None
            spec = get_field(name)
            if isinstance(val, (int, float)):
                span = spec.max_val - spec.min_val or 1.0
                row.append((float(val) - spec.min_val) / span)
            else:
                row.append((spec.default - spec.min_val) / (spec.max_val - spec.min_val or 1.0))
        row.append(sum(weights) / len(weights) / 100.0 if weights else 0.0)
        row.extend([
            tyre_map.get(comp, 0.6), wet,
            float(r.get("track_temp") or 35.0), float(r.get("air_temp") or 24.0),
            rd.traction_index, rd.aero_index, rd.braking_index,
            rd.slow_share, rd.fast_share,
        ])
        row.extend([float(v) for v in list(style)[:11]] + [0.0] * max(0, 11 - len(style)))
        feats.append(row)
        targets.append(float(lt))
        tids.append(tid)
        used += 1

    if used < 5:
        return {"available": False, "reason": f"本地有效带 setup 的圈数不足（{used}）",
                "n_samples": used}

    # 剔除异常慢圈（出场圈/进站圈）：超过全体中位数的 2 倍即视为非推进圈
    med_all = statistics.median(targets)
    keep = [i for i, t in enumerate(targets) if t <= med_all * 2.0]

    x = [feats[i] for i in keep]
    y = [targets[i] for i in keep]
    y_tid = [tids[i] for i in keep]

    # 留一交叉验证 + **同赛道均值**基线（赛道已知，用全局均值当基线会虚高 skill）
    preds: list[float] = []
    for i in range(len(x)):
        tr_x = [x[j] for j in range(len(x)) if j != i]
        tr_y = [y[j] for j in range(len(x)) if j != i]
        m = Ridge(alpha=10.0).fit(tr_x, tr_y)
        preds.append(m.predict([x[i]])[0])
    loo_mae = _mae(y, preds)
    mean_pred: list[float] = []
    for i in range(len(y)):
        same = [y[j] for j in range(len(y)) if j != i and y_tid[j] == y_tid[i]]
        pool = same or [y[j] for j in range(len(y)) if j != i]
        mean_pred.append(sum(pool) / max(1, len(pool)))
    base_mae = _mae(y, mean_pred)
    skill = max(0.0, 1.0 - loo_mae / base_mae) if base_mae > 1e-9 else 0.0
    model = Ridge(alpha=10.0).fit(x, y)

    feature_keys = list(setup_names) + ["tyre_wear_avg"] + list(LAP_CONDITION_KEYS) \
        + [f"style_{i}" for i in range(11)]
    return {
        "available": True,
        "feature_keys": feature_keys,
        "target": "lap_time_ms",
        "n_samples": len(x),
        "setup_names": setup_names,
        "metrics": {
            "loo_mae_ms": round(loo_mae, 2),
            "baseline_mae_ms": round(base_mae, 2),
            "skill": round(skill, 4),
            "loo_r2": round(_r2(y, preds), 4),
        },
        "model": model.to_dict(),
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="训练遥测代理模型")
    ap.add_argument("--corner-samples", type=int, default=4000,
                    help="MLP 头使用的分层子样本数（0 = 跳过 MLP）")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--ridge-samples", type=int, default=20000,
                    help="岭回归使用的样本上限（0 = 全量；全量在纯 Python 下很慢）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-model", choices=("auto", "ridge", "mlp"), default="auto",
                    help="强制引擎采纳哪个学习器（默认按留出集 MAE 自动择优）")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    if not CORNER_JSONL.exists():
        raise SystemExit(f"缺少逐弯数据集 {CORNER_JSONL}（先跑 build_corner_dataset.py）")
    samples: list[dict[str, Any]] = []
    for line in CORNER_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            samples.append(json.loads(line))
    print(f"逐弯样本: {len(samples)}（{len({s['track_id'] for s in samples})} 赛道）")

    print("[1/2] 训练弯速代理（遥测数据）...")
    corner_head = train_corner_head(
        samples, args.corner_samples, args.epochs, args.seed, args.ridge_samples,
        args.force_model,
    )
    _rm = corner_head["family"]["ridge"]["metrics"]
    print(f"  ridge val_MAE = {_rm['val_mae']} (占圈速比) / R² = {_rm['val_r2']}"
          f"（训练 {_rm['train_samples']} / 共 {_rm['n_samples_total']} 样本）")
    if corner_head.get("mlp"):
        print(f"  mlp   val_MAE = {corner_head['mlp']['metrics']['val_mae']} s"
              f" / R² = {corner_head['mlp']['metrics']['val_r2']}")
    print(f"  采纳: {corner_head['active']['kind']}")

    print("[2/2] 训练整圈配速代理（本地 setup + 风格）...")
    lap_head = train_lap_head(args.seed)
    if lap_head.get("available"):
        mm = lap_head["metrics"]
        print(f"  样本 {lap_head['n_samples']}，LOO MAE {mm['loo_mae_ms']} ms，"
              f"基线 {mm['baseline_mae_ms']} ms，skill={mm['skill']}")
    else:
        print(f"  跳过：{lap_head.get('reason')}")

    payload = {
        "schema": "f1opt-telemetry-surrogate/1",
        "source": "TracingInsights 2026 (external, 2026-only) + local game UDP laps",
        "corner_head": corner_head,
        "lap_head": lap_head,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"写出: {out}（{out.stat().st_size / 1e6:.2f} MB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
