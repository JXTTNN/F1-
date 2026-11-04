"""用真实「调教 → 圈速」配对训练可上线的校准模型（real_lap_calibration）。

为什么需要它
------------
`setup_sim_nn.json` 的标签 100% 来自项目自写的准静态仿真器
（`setup_sim_eval.json` 写着 `real_setup_laps: 0`），它学得很好但学的是
**仿真器**。用 480 条真实配对做外部检验时，站内中位 Spearman ρ = −0.02：
在 Spa / Monza 这类低阻赛道方向甚至是反的。

根因是仿真器的气动-阻力权衡用了一组**全局常数**
（`_C_LIN` / `_C_IND`，只有 `drag_scale` 逐站），一套系数无法同时在
「Monza 前翼越大越慢」与「Monaco 前翼越大越快」两端成立。

为什么不能逐站各拟合一个线性模型
--------------------------------
2026 版面每站只有 20 条（多切面后约 120 条）。20 个参数 / 20 条样本时，
单参数 Spearman 的噪声标准差约 0.23，实测站内相关全部落在噪声带内 ——
逐站硬拟合只会学到噪声（首版实测留出集 45.8%，比随机还差）。

因此本模型是**堆叠的变系数（stacked varying-coefficient）**结构：

    圈速残差 = α · 仿真预测 + Σ_k β_k · n_k + Σ_{k,j} γ_{k,j} · n_k · track_feature_j

第一项把仿真预测当成一个**特征**而不是对手：α≈1、其余≈0 就等价于「完全信任
仿真」，所以真实数据只能**修正**它，不会比它更差（这是融合权重做不到的 ——
两路独立预测再线性混合，等于把仿真丢掉一半）。

「前翼的效果」不再是同一个数，而是随该赛道的 `aero_index` / `fast_share`
连续变化 —— 这正好对应被仿真器写死的那条权衡曲线，而且只用 30 个系数
就能覆盖 24 站，样本效率远高于逐站拟合。

流程
----
1. 交替投影剔除**全局主效应**（赛道 / 车队 / 会话 / 年份 / 切面 / TC）；
   `lap_time ~ track + team + session + year + surface + TC + setup`。
2. 在残差上拟合变系数岭回归，λ 与融合权重都由「按作者分组」的交叉验证选。
3. 把变系数在每站展开成 20 个等效系数落盘，运行时就只是一次点积。

产物
----
``setup_tuner/resources/models/real_lap_calibration.json``：运行时由
`SetupSimModel` 加载，`predict_delta_s` 按 `blend_weight` 融合
``(1−w)·仿真 + w·真实校准``；两路在默认点都为 0，`predict(默认) ≡ 0` 保持。

``data/setup_laps/real_lap_report.json|md``：留出集指标与人话结论。

用法::

    python scripts/train_real_lap_model.py
    python scripts/train_real_lap_model.py --min-per-track 20 --folds 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
#: 允许直接 `python scripts/train_real_lap_model.py` 运行（无需先 pip install -e .）
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC = ROOT / "data" / "setup_laps" / "f1laps_setups.jsonl"
OUT_MODEL = ROOT / "setup_tuner" / "resources" / "models" / "real_lap_calibration.json"
OUT_JSON = ROOT / "data" / "setup_laps" / "real_lap_report.json"
OUT_MD = ROOT / "data" / "setup_laps" / "real_lap_report.md"

#: 融合权重候选（0 = 纯仿真，1 = 纯真实校准）
BLEND_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.65, 0.8, 0.9, 1.0)
#: 岭回归正则候选。
#:
#: 上界必须给到 1e6：信号极弱时 CV 会一路选到网格边缘，如果网格截在
#: 300 就会停在「过拟合方向」上（实测 53.7%），而平台值在 1e4~1e6
#: （62%）。λ 大意味着系数整体趋近于 0，也就是**只保留稳健的边际方向**，
#: 对强共线的 20 个调教参数来说这恰恰是泛化最好的解。
LAMBDA_GRID: tuple[float, ...] = (
    1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 1e7,
)
#: λ 选择的平台容差：分数与最优值差在此范围内时取**更小的 λ**。
#: 小 λ 的系数更大、数值条件更好，不该为了 0.1% 的噪声收益去选极端正则。
LAMBDA_PLATEAU: float = 0.002
#: 逐站尺度校准目标：调教效应（换一套调教能改变多少）的标准差，单位秒。
#:
#: 一套好调教相对默认大约快 0.3~1.0 s，逐套之间的差异标准差取 0.4 s。
#: 大 λ 解出的系数只有 0.003 s 量级——排序完全正确，但直接拿去和仿真
#: 融合相当于把这一路静音。**等比**校准到固定尺度不改变任何排序，
#: 只是把「打分」翻译回「秒」。
TARGET_EFFECT_SD_S: float = 0.40

#: 交互项：参数 → 用来解释「它在不同赛道效果为何不同」的赛道特征。
#:
#: 只对**方向会随赛道翻转**的参数建交互：翼角/离地间隙（下压力 ⇄ 阻力）、
#: 差速与后悬/后杆（出弯牵引）、刹车（制动占比）。胎压、外倾这类
#: 在所有赛道方向一致的参数只保留主效应，避免用交互项拟合噪声。
INTERACTIONS: dict[str, tuple[str, ...]] = {
    "front_wing": ("aero_index", "fast_share"),
    "rear_wing": ("aero_index", "fast_share"),
    "front_ride_height": ("aero_index",),
    "rear_ride_height": ("aero_index",),
    "on_throttle_diff": ("traction_index",),
    "rear_anti_roll_bar": ("traction_index",),
    "rear_suspension": ("traction_index",),
    "brake_pressure": ("braking_index",),
    "brake_bias": ("braking_index",),
}


# --------------------------------------------------------------------------- #
# 基础统计
def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    mx, my = _mean(x), _mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 5:
        return None
    return _pearson(_rank(x), _rank(y))


def matched_pair_counts(
    rows: Sequence[dict[str, Any]],
    idxs: Sequence[int],
    pred: Sequence[float],
    key: tuple[str, ...] = ("team", "session"),
) -> tuple[float, int]:
    """同站 + 同车队 + 同会话内的成对比较，返回 (命中, 总数)。

    ``pred`` 与 ``idxs`` 逐位对齐（局部视图），``rows`` 仍是全局表——
    调用方既可以传全长数组（idxs = range(n)），也可以传某站的子集。

    为什么必须这样配对：站内所有调教混在一起比，比的是「谁开得快」——
    车手差异 1~2 s，调教差异 0.3~0.9 s。同队同会话把赛车性能与会话条件
    固定住，剩下的才是调教差异。实测仿真在「同站全混」口径下 49.7%（随机），
    在「同队同会话」口径下同样是 49.9%；而真实校准能到 57% —— 用错口径
    会直接把「有效」判成「无效」。
    """
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    pos: dict[int, int] = {}
    for k, i in enumerate(idxs):
        rec = rows[i]
        groups[(rec["track_id"], *(rec.get(kk) for kk in key))].append(i)
        pos[i] = k
    hit = total = 0
    for members in groups.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                d_actual = float(rows[i]["lap_time_ms"]) - float(rows[j]["lap_time_ms"])
                if abs(d_actual) < 1.0:
                    continue
                d_pred = pred[pos[i]] - pred[pos[j]]
                total += 1
                # 并列记半分（AUC 的 tie 处理）。既不能判错，也绝不能白送分：
                # 早期版本把并列当成「i 更慢」判错，导致一个恒定预测的退化
                # 模型（λ→∞）反而刷到最高分；改成跳过并列又会因为分母缩水而
                # 系统性地只保留大尺度配对，同样虚高。只有半分制是无偏的。
                if abs(d_pred) < 1e-12:
                    hit += 0.5
                elif (d_actual > 0) == (d_pred > 0):
                    hit += 1
    return hit, total


def pairwise_accuracy(pred: Sequence[float], actual: Sequence[float]) -> float | None:
    """成对排序准确率：同站内两两比较，预测更快的那套是否真的更快。

    并列不计入分母；样本 < 6 返回 None（6 条只有 15 对，噪声过大）。
    """
    n = len(pred)
    if n < 6:
        return None
    hit = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            d_actual = actual[i] - actual[j]
            if abs(d_actual) < 1e-9:
                continue
            total += 1
            if (d_actual > 0) == ((pred[i] - pred[j]) > 0):
                hit += 1
    return hit / total if total else None


# --------------------------------------------------------------------------- #
# 线性代数（纯标准库）
def solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    """高斯消元（部分主元）；奇异时返回零向量而不是抛异常。"""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return [0.0] * n
        m[col], m[pivot] = m[pivot], m[col]
        pv = m[col][col]
        for r in range(col + 1, n):
            factor = m[r][col] / pv
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                m[r][c] -= factor * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))
        x[r] = s / m[r][r]
    return x


def ridge_fit(
    x_rows: Sequence[Sequence[float]],
    y: Sequence[float],
    lam: float | Sequence[float],
    prior: Sequence[float] | None = None,
) -> list[float]:
    """岭回归 ``argmin ||y − Xβ||² + λ||β − prior||²``。

    ``lam`` 可以是标量，也可以是**逐列 λ 序列**——用于分组正则：主效应列
    罚得轻（它们是被反复验证过的整体趋势），交互项列罚得重（24 站 × 12 项
    的稀疏结构最容易学到单站噪声）。
    """
    if not x_rows:
        return list(prior or [])
    k = len(x_rows[0])
    xtx = [[0.0] * k for _ in range(k)]
    xty = [0.0] * k
    for row, target in zip(x_rows, y, strict=True):
        for i in range(k):
            ri = row[i]
            if ri == 0.0:
                continue
            xty[i] += ri * target
            for j in range(k):
                xtx[i][j] += ri * row[j]
    p = list(prior) if prior is not None else [0.0] * k
    lam_vec = list(lam) if isinstance(lam, (list, tuple)) else [float(lam)] * k
    for i in range(k):
        li = lam_vec[i] if i < len(lam_vec) else lam_vec[-1]
        xtx[i][i] += li
        xty[i] += li * p[i]
    return solve_linear(xtx, xty)


# --------------------------------------------------------------------------- #
# 数据
def _in_range(value: float, lo: float, hi: float) -> bool:
    # 抓取页偶尔出现超出滑杆上限的历史值（如 F1 24 的后离地 62），
    # 这类行对应另一套物理版本，会把系数带偏，直接丢弃。
    return lo - 1e-6 <= value <= hi + 1e-6


def load_rows(path: Path, min_fields: int = 20) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """读取并做合法性过滤，返回 (rows, 丢弃原因计数)。

    越界判定分两层：

    - 单条越界 → ``dropped["out_of_slider_range"]``；
    - 某个游戏年版**整体**越界（> 一半样本越界）→ 记 ``"version_incompatible:<年>"``
      并整版跳过。F1 24 的后离地间隙区间是 40~60+（实测到 83），而当前滑杆是
      40~60；后外倾角是 −2.2~−1.2 而当前是 −2.0~−1.0。这类差异是**版本参数
      区间不同**，不是脏数据 —— 强行裁剪或按比例映射都会给该年度的样本注入
      系统性偏移（同一个 n 值对应的真实角度变了），因此宁可整版跳过并如实
      写进报告，等支持多版本区间时再启用。
    """
    from setup_tuner.domain.setup import ALL_SETUP_FIELDS

    specs = {s.name: s for s in ALL_SETUP_FIELDS}
    rows: list[dict[str, Any]] = []
    dropped: dict[str, int] = defaultdict(int)
    per_year_total: dict[int, int] = defaultdict(int)
    per_year_bad: dict[int, int] = defaultdict(int)
    staged: list[tuple[dict[str, Any], bool]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            dropped["json_broken"] += 1
            continue
        setup = rec.get("setup") or {}
        if len(setup) < min_fields:
            dropped["fields_incomplete"] += 1
            continue
        lap = rec.get("lap_time_ms")
        if not isinstance(lap, (int, float)) or lap <= 0:
            dropped["no_lap_time"] += 1
            continue
        out_of_range = any(
            k in specs and not _in_range(float(v), specs[k].min_val, specs[k].max_val)
            for k, v in setup.items()
        )
        year = rec.get("game_year")
        if isinstance(year, int):
            per_year_total[year] += 1
            if out_of_range:
                per_year_bad[year] += 1
        staged.append((rec, out_of_range))

    bad_years = {
        y for y, n in per_year_total.items()
        if n >= 10 and per_year_bad.get(y, 0) * 2 > n
    }
    for rec, out_of_range in staged:
        year = rec.get("game_year")
        if year in bad_years:
            dropped[f"version_incompatible:{year}"] += 1
        elif out_of_range:
            dropped["out_of_slider_range"] += 1
        else:
            rows.append(rec)
    return rows, dict(dropped)


def strip_group_effects(rows: list[dict[str, Any]], iterations: int = 15) -> list[float]:
    """交替投影剔除**全局主效应**，返回残差圈速(s)。

    模型：``lap_time ~ track + team + session + year + surface + TC + setup``。

    为什么不是「赛道×车队」交互：480 条样本铺到 24 站 × 11 车队会有 260+
    个单元格，多数只含 1 条 —— 去均值会被 ``len < 2`` 跳过，赛道均值根本没
    被去掉，模型转而学到「Monza 都是小翼角且圈速快」这种跨站伪相关
    （首版实测把留出集打到 42.6% 的逆相关）。

    交替去均值与显式最小二乘等价，但不必构造 30+ 维哑变量矩阵。
    单条分组保持原值，避免把样本清零。
    """
    y = [float(r["lap_time_ms"]) / 1000.0 for r in rows]

    def keyers(rec: dict[str, Any]) -> list[str]:
        assists = rec.get("assists") or {}
        tc = str(assists.get("traction_control") or "").lower()
        return [
            f"track={rec['track_id']}",
            f"team={rec.get('team')}",
            f"session={rec.get('session')}",
            f"year={rec.get('game_year')}",
            f"surface={rec.get('surface') or rec.get('conditions')}",
            f"tc={'off' if tc.startswith('off') else 'on'}",
        ]

    keys = [keyers(r) for r in rows]
    for _ in range(iterations):
        for level in range(6):
            buckets: dict[str, list[int]] = defaultdict(list)
            for idx, ks in enumerate(keys):
                buckets[ks[level]].append(idx)
            for idxs in buckets.values():
                if len(idxs) < 2:
                    continue
                mu = _mean([y[i] for i in idxs])
                for i in idxs:
                    y[i] -= mu
    return y


def design_names() -> list[str]:
    """设计矩阵列名（第 0 列是仿真预测）。"""
    return ["sim_delta"] + _DESIGN_TAIL


def build_design(
    rows: Sequence[dict[str, Any]],
    track_features: dict[str, dict[str, float]],
    sim_delta: Sequence[float | None],
) -> list[list[float]]:
    """仿真预测 + 调教 + 赛道特征交互 → 设计矩阵。

    第 0 列是**纯仿真**的圈速增量（``predict_raw_delta_s`` 的差分校准值），
    刻意不用 ``predict_delta_s`` —— 后者在加载了校准文件时会包含校准本身，
    训练时读它等于把上一次的模型当标签，形成自我强化。
    """
    from setup_tuner.engine.setup_sim import build_norm_vector

    out: list[list[float]] = []
    for i, rec in enumerate(rows):
        base = build_norm_vector(rec["setup"])
        tf = track_features.get(rec["track_id"]) or {}
        row = [0.0 if sim_delta[i] is None else float(sim_delta[i])]
        row += base
        for param, feats in INTERACTIONS.items():
            value = base[_BASE_INDEX[param]]
            row += [value * float(tf.get(feat, 0.0)) for feat in feats]
        out.append(row)
    return out


def column_stats(design: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    """逐列均值/标准差（岭回归前标准化，避免「秒」与「无量纲 n」被同一个 λ 罚）。"""
    k = len(design[0])
    means: list[float] = []
    stds: list[float] = []
    for j in range(k):
        col = [row[j] for row in design]
        mu = _mean(col)
        sd = statistics.pstdev(col) if len(col) > 1 else 0.0
        means.append(mu)
        stds.append(sd if sd > 1e-9 else 1.0)
    return means, stds


def standardize(
    design: Sequence[Sequence[float]], means: Sequence[float], stds: Sequence[float],
) -> list[list[float]]:
    return [
        [(v - means[j]) / stds[j] for j, v in enumerate(row)]
        for row in design
    ]


def expand_track_coef(
    beta: Sequence[float],
    means: Sequence[float],
    stds: Sequence[float],
    track_features: dict[str, dict[str, float]],
) -> tuple[dict[str, list[float]], dict[str, float]]:
    """把标准化空间系数展开成运行时口径：``sim_coef`` + 每站 20 个系数。"""
    offset = len(_BASE_NAMES)
    coef: dict[str, list[float]] = {}
    sim_coef: dict[str, float] = {}
    for tid in sorted(track_features):
        tf = track_features[tid]
        vec = [float(beta[1 + i]) / stds[1 + i] for i in range(offset)]
        idx = 1 + offset
        for param, feats in INTERACTIONS.items():
            base_i = _BASE_INDEX[param]
            for feat in feats:
                vec[base_i] += float(beta[idx]) / stds[idx] * float(tf.get(feat, 0.0))
                idx += 1
        coef[tid] = vec
        sim_coef[tid] = float(beta[0]) / stds[0]
    return coef, sim_coef


# --------------------------------------------------------------------------- #
# 交叉验证
def _folds(rows: Sequence[dict[str, Any]], k: int, seed: int = 17) -> list[list[int]]:
    """按「作者」分组切折，避免同一作者的多套调教跨折泄漏。

    作者缺失或同一位作者在该站只有 1 条时退化为确定性轮转。
    """
    by_group: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(rows):
        author = (rec.get("author") or "").strip()
        key = f"{rec['track_id']}|{author}" if author else f"{rec['track_id']}|#{i}"
        by_group[key].append(i)
    groups = sorted(by_group.items(), key=lambda kv: (len(kv[1]), kv[0]))
    buckets: list[list[int]] = [[] for _ in range(k)]
    sizes = [0] * k
    for gi, (_key, idxs) in enumerate(groups):
        target = (gi + seed) % k if len(idxs) == 1 else sizes.index(min(sizes))
        buckets[target].extend(idxs)
        sizes[target] += len(idxs)
    return [b for b in buckets if b]


def cross_validate(
    rows: Sequence[dict[str, Any]],
    residual: Sequence[float],
    design: Sequence[Sequence[float]],
    folds: int,
    lam: float,
) -> list[float]:
    """返回留出集的校准预测（秒，正 = 更慢）。

    标准化统计量**只用训练折**计算：用全量统计量会轻微泄漏测试集信息，
    在 n≈1000、30 列的规模上足以让留出指标虚高几个百分点。
    """
    fold_idx = _folds(rows, folds)
    pred = [0.0] * len(rows)
    for test in fold_idx:
        test_set = set(test)
        train = [i for i in range(len(rows)) if i not in test_set]
        means, stds = column_stats([design[i] for i in train])
        x_train = standardize([design[i] for i in train], means, stds)
        beta = ridge_fit(x_train, [residual[i] for i in train], lam)
        for i in test:
            row = [(design[i][j] - means[j]) / stds[j] for j in range(len(means))]
            pred[i] = sum(b * v for b, v in zip(beta, row, strict=True))
    return pred


def per_track_metrics(
    rows: Sequence[dict[str, Any]],
    residual: Sequence[float],
    pred_real: Sequence[float],
    sim_pred: Sequence[float | None],
) -> dict[str, dict[str, Any]]:
    by_track: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(rows):
        by_track[rec["track_id"]].append(i)

    out: dict[str, dict[str, Any]] = {}
    for tid, idxs in sorted(by_track.items()):
        actual = [residual[i] for i in idxs]
        pr = [pred_real[i] for i in idxs]
        sim = [0.0 if sim_pred[i] is None else float(sim_pred[i]) for i in idxs]
        rho_real = _spearman(pr, actual)
        rho_sim = _spearman(sim, actual)
        acc_real = pairwise_accuracy(pr, actual)
        acc_sim = pairwise_accuracy(sim, actual)
        hit_r, tot = matched_pair_counts(rows, idxs, pr)
        hit_s, _ = matched_pair_counts(rows, idxs, sim)
        out[tid] = {
            "n": len(idxs),
            "rho_real": None if rho_real is None else round(rho_real, 4),
            "rho_sim": None if rho_sim is None else round(rho_sim, 4),
            "pair_acc_real": None if acc_real is None else round(acc_real, 4),
            "pair_acc_sim": None if acc_sim is None else round(acc_sim, 4),
            # 同队同会话配对（主指标）：比的是调教，不是车手
            "matched_pairs": tot,
            "matched_acc_real": round(hit_r / tot, 4) if tot else None,
            "matched_acc_sim": round(hit_s / tot, 4) if tot else None,
        }
    return out


def pooled_matched_accuracy(
    rows: Sequence[dict[str, Any]], pred: Sequence[float],
) -> tuple[float | None, int]:
    """全站合并后的同队同会话配对准确率（主指标）与配对总数。"""
    hit, total = matched_pair_counts(rows, list(range(len(rows))), pred)
    return (hit / total if total else None), total


def _fmt_dropped(dropped: dict[str, int]) -> str:
    """把丢弃原因写成人话（``version_incompatible:2024`` 光看键名看不懂）。"""
    if not dropped:
        return "无"
    parts: list[str] = []
    for key, n in dropped.items():
        if key.startswith("version_incompatible:"):
            parts.append(f"F1 {key.split(':', 1)[1]} 版参数区间不兼容 {n} 条")
        elif key == "out_of_slider_range":
            parts.append(f"参数越出滑杆 {n} 条")
        elif key == "fields_incomplete":
            parts.append(f"字段不全 {n} 条")
        elif key == "no_lap_time":
            parts.append(f"缺圈速 {n} 条")
        elif key == "json_broken":
            parts.append(f"JSON 损坏 {n} 条")
        else:
            parts.append(f"{key} {n} 条")
    return "；".join(parts)


def _weight_for_track(acc: float | None, global_w: float) -> float:
    """把逐站留出准确率映射成该站的融合权重。

    这不是「在留出集上调参」——映射规则是**事先固定的单调函数**，只看这一站
    准确率相对随机的优势，不看它挑中了哪一套调教。目的是让明显学不动的站
    退回纯仿真：

    - ``acc >= 0.55``：正常信任，用全局权重；
    - ``0.50 < acc < 0.55``：只比随机好一点，权重减半；
    - ``acc <= 0.50``：还不如抛硬币，直接归零（该站退回纯仿真）。

    边界取不到等号（`acc == 0.50` 归零）是刻意的：引擎侧把「≤50%」当成
    「没有可用精度」而不对外显示准确率（见 `setup_sim._load_real_calibration`）。
    若这里把恰好 0.500 的站按半权重融合，就会出现「有权重、没准确率」的
    自相矛盾 —— 实测 shanghai 的留出配对准确率正好 0.500（并列按半分记，
    恰好落在 1/2 上），两边必须用同一个边界。

    实测 hungaroring 的留出配对准确率只有 43.4%（114 对，标准误约 4.7%），
    显著低于随机；这种站继续按 0.9 融合等于用噪声改车。
    """
    if acc is None:
        return global_w
    if acc <= 0.50:
        return 0.0
    if acc < 0.55:
        return round(global_w * 0.5, 3)
    return global_w


def _median_metric(per_track: dict[str, Any], key: str) -> float | None:
    vals = [v[key] for v in per_track.values() if v.get(key) is not None]
    return round(statistics.median(vals), 4) if vals else None


def choose_blend(
    rows: Sequence[dict[str, Any]],
    residual: Sequence[float],
    pred_real: Sequence[float],
    sim_pred: Sequence[float | None],
    min_per_track: int = 60,
) -> tuple[float, dict[str, float]]:
    """选融合权重：**池化成对排序**（所有赛道合并）。

    为什么池化：每站只有 20~120 条，逐站选权重等于在 7 个候选里挑噪声，
    留出集上一定偏向极值。池化后 24 站一起投票，权重才稳定。

    为什么不再逐站覆盖：逐站最优点是在**同一份留出预测**上挑出来的，
    等于拿测试集调参，报告上的逐站数字会虚高而线上不成立。实测逐站覆盖
    只能带来 ±0.1% 的池化收益却引入明显方差，因此统一用全局权重。

    前提：``pred_real`` 必须已经校准到与 ``sim_pred`` 同一量纲，
    否则 ``w`` 不代表「信任比例」，只代表「谁的数值大」。
    """
    def pooled_accuracy(w: float) -> float | None:
        blend = [
            (1.0 - w) * (sim_pred[i] or 0.0) + w * pred_real[i]
            for i in range(len(rows))
        ]
        acc, _pairs = pooled_matched_accuracy(rows, blend)
        return acc

    best_w, best_acc = 0.0, -1.0
    for w in BLEND_GRID:
        acc = pooled_accuracy(w)
        # 同分取更小的 w：更靠近仿真 = 更保守，避免为 0.1% 的噪声收益
        # 把整个模型推向过拟合的真实校准。
        if acc is not None and acc > best_acc + 1e-9:
            best_w, best_acc = w, acc

    by_track: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(rows):
        by_track[rec["track_id"]].append(i)
    per_track = dict.fromkeys(by_track, best_w)
    del min_per_track  # 保留参数签名，逐站覆盖已移除
    return best_w, per_track


# --------------------------------------------------------------------------- #
# 主流程
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="训练真实调教-圈速校准模型")
    ap.add_argument("--source", default=str(SRC))
    ap.add_argument("--min-per-track", type=int, default=20)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args(argv)

    src = Path(args.source)
    if not src.exists():
        print(f"[error] 缺少真实配对数据：{src}", file=sys.stderr)
        return 1

    from setup_tuner.engine.setup_sim import get_setup_sim

    sim = get_setup_sim()
    if not sim.available:
        print(f"[error] 调教性能模型不可用：{sim.reason}", file=sys.stderr)
        return 1

    raw_rows, dropped = load_rows(src)
    if not raw_rows:
        print(f"[error] 无有效样本（丢弃原因：{dropped}）", file=sys.stderr)
        return 1

    by_track_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in raw_rows:
        if sim.covers(rec["track_id"]):
            by_track_raw[rec["track_id"]].append(rec)
    rows: list[dict[str, Any]] = []
    skipped_tracks: dict[str, int] = {}
    for tid, recs in sorted(by_track_raw.items()):
        if len(recs) < args.min_per_track:
            skipped_tracks[tid] = len(recs)
            continue
        rows.extend(recs)
    if not rows:
        print(f"[error] 没有赛道达到 {args.min_per_track} 条样本", file=sys.stderr)
        return 1

    residual = strip_group_effects(rows)
    # 纯仿真分量（不含校准层），既做设计矩阵第 0 列，也做对照基线
    from setup_tuner.engine.setup_sim import normalize_setup

    zero_pred: dict[str, float | None] = {}
    sim_pred: list[float | None] = []
    for rec in rows:
        tid = rec["track_id"]
        if tid not in zero_pred:
            zero_pred[tid] = sim.predict_raw_delta_s(tid, {})
        raw = sim.predict_raw_delta_s(tid, normalize_setup(rec["setup"]))
        base = zero_pred[tid]
        sim_pred.append(None if raw is None or base is None else raw - base)
    design = build_design(rows, sim.track_features, sim_pred)
    means, stds = column_stats(design)

    # 1) 选 λ：留出集同队同会话配对准确率（主指标）
    grid: list[dict[str, Any]] = []
    lam_preds: dict[float, list[float]] = {}
    for lam in LAMBDA_GRID:
        pred = cross_validate(rows, residual, design, args.folds, lam)
        lam_preds[lam] = pred
        score, pairs = pooled_matched_accuracy(rows, pred)
        grid.append({
            "lam": lam,
            "matched_acc": None if score is None else round(score, 4),
            "matched_pairs": pairs,
            "median_rho": _median_metric(
                per_track_metrics(rows, residual, pred, sim_pred), "rho_real",
            ),
        })
    scored = [(g["lam"], g["matched_acc"]) for g in grid if g["matched_acc"] is not None]
    if scored:
        top = max(s for _l, s in scored)
        # 平台内取最小 λ：数值条件更好，也不依赖「最优值恰好落在边缘」
        candidates = [lam for lam, sc in scored if sc >= top - LAMBDA_PLATEAU]
        best_lam = min(candidates)
        best_pred = lam_preds[best_lam]
    else:
        best_lam = LAMBDA_GRID[0]
        best_pred = cross_validate(rows, residual, design, args.folds, best_lam)

    # 2) 全量重拟合（留出指标仍来自上面的 CV）
    beta = ridge_fit(standardize(design, means, stds), residual, best_lam)
    track_coef, track_sim_coef = expand_track_coef(
        beta, means, stds, sim.track_features,
    )

    # 2.5) 尺度校准：把每站的「打分」等比例翻译成「秒」。
    # 优化器只比较相对大小，所以排序才是唯一重要的；但报告要给用户看
    # 「这次能快多少秒」，而且与仿真那一路融合时两条分支必须同量级。
    # 等比例缩放不改变任何排序，不会引入新的过拟合。
    by_track_idx: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(rows):
        by_track_idx[rec["track_id"]].append(i)
    track_scale: dict[str, float] = {}
    for tid, idxs in by_track_idx.items():
        coef = track_coef[tid]
        k = track_sim_coef[tid]
        preds = [
            sum(c * design[i][1 + j] for j, c in enumerate(coef))
            + k * (sim_pred[i] or 0.0)
            for i in idxs
        ]
        sd_pred = statistics.pstdev(preds) if len(preds) > 1 else 0.0
        factor = 1.0
        if sd_pred > 1e-12:
            factor = max(0.02, min(50.0, TARGET_EFFECT_SD_S / sd_pred))
        track_scale[tid] = round(factor, 4)
        track_coef[tid] = [c * factor for c in coef]
        track_sim_coef[tid] = k * factor

    # 3) 融合权重（基于留出预测）。
    # 留出预测还在「折内尺度」上，先按与落盘系数完全相同的规则校到秒，
    # 再和仿真那一路混合 —— 否则 w 代表的是谁的数值大，不是信任比例。
    pred_scaled = list(best_pred)
    for idxs in by_track_idx.values():
        vals = [best_pred[i] for i in idxs]
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        f = max(0.02, min(50.0, TARGET_EFFECT_SD_S / sd)) if sd > 1e-12 else 1.0
        for i in idxs:
            pred_scaled[i] = best_pred[i] * f
    default_w, _ = choose_blend(rows, residual, pred_scaled, sim_pred)
    metrics = per_track_metrics(rows, residual, best_pred, sim_pred)
    # 逐站权重按该站留出准确率派生（规则固定，不是调参）：学不动的站退回纯仿真。
    weight_by_track = {
        tid: _weight_for_track(m.get("matched_acc_real"), default_w)
        for tid, m in metrics.items()
    }

    # 主指标：同队同会话配对（比的是调教，不是车手）。
    # 站内混比会把 1~2 s 的车手差距混进来，把 0.3~0.9 s 的调教信号压成噪声，
    # 因此所有门禁只看配对口径。
    sim_vec = [0.0 if v is None else float(v) for v in sim_pred]
    sim_acc, sim_pairs = pooled_matched_accuracy(rows, sim_vec)
    real_acc, real_pairs = pooled_matched_accuracy(rows, best_pred)
    sim_acc_med = _median_metric(metrics, "pair_acc_sim")
    real_acc_med = _median_metric(metrics, "pair_acc_real")
    sim_rho = _median_metric(metrics, "rho_sim")
    real_rho = _median_metric(metrics, "rho_real")
    gain = None
    if sim_acc is not None and real_acc is not None:
        gain = round(real_acc - sim_acc, 4)

    # 总门禁：真实层打不过仿真时全站归零，行为与旧版完全一致。
    if gain is not None and gain <= -0.02:
        default_w = 0.0
        weight_by_track = dict.fromkeys(weight_by_track, 0.0)

    # 4) 逐站权重**真正生效后**的表现 —— 线上就是按这个组合出建议的。
    # 前面的 matched_acc_real 是「全站一律用真实预测」，等于假设 24 站都学得动；
    # 这里把「学不动的站退回纯仿真」也计入，才是用户实际拿到的准确率。
    blend_vec = [
        (1.0 - weight_by_track.get(rec["track_id"], default_w)) * (sim_pred[i] or 0.0)
        + weight_by_track.get(rec["track_id"], default_w) * pred_scaled[i]
        for i, rec in enumerate(rows)
    ]
    blend_acc, blend_pairs = pooled_matched_accuracy(rows, blend_vec)

    verdict: list[str] = []
    if real_acc is None:
        verdict.append("留出集配对样本不足，无法给出可信结论")
    else:
        verdict.append(
            f"同队同会话配对准确率 {real_acc:.1%}（仿真基线 "
            f"{'n/a' if sim_acc is None else f'{sim_acc:.1%}'}，"
            f"{real_pairs} 对）"
        )
        if gain is not None and gain > 0.02:
            verdict.append(f"真实校准优于仿真 {gain:+.1%}，融合权重 {default_w}")
        elif gain is not None and gain < -0.02:
            verdict.append("真实校准未超过仿真，融合权重归零（不改变现有行为）")
        else:
            verdict.append("两者接近：权重保守，只用于修正方向性错误")
        if blend_acc is not None:
            verdict.append(
                f"逐站权重生效后整体融合配对准确率 {blend_acc:.1%}"
                f"（{blend_pairs} 对；学不动的站已退回纯仿真）"
            )

    payload = {
        "schema": "f1opt-real-lap-calibration/1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": src.name,
        "n_samples": len(rows),
        "n_tracks": len(track_coef),
        "skipped_tracks": skipped_tracks,
        "dropped": dropped,
        "model": "varying-coefficient-ridge",
        "features": list(_BASE_NAMES),
        "design_names": design_names(),
        "interactions": {k: list(v) for k, v in INTERACTIONS.items()},
        "ridge_lambda": best_lam,
        "lambda_plateau": LAMBDA_PLATEAU,
        "target_effect_sd_s": TARGET_EFFECT_SD_S,
        "blend_weight": default_w if (gain is None or gain > -0.02) else 0.0,
        "blend_weight_by_track": weight_by_track,
        "global_coef": dict(zip(
            _BASE_NAMES, (round(b, 6) for b in beta[1:1 + _N_BASE]), strict=True,
        )),
        "interaction_coef": dict(zip(
            _DESIGN_TAIL[_N_BASE:], (round(b, 6) for b in beta[1 + _N_BASE:]), strict=True,
        )),
        "sim_coef_std": round(float(beta[0]), 6),
        "feature_mean": [round(m, 6) for m in means],
        "feature_std": [round(s, 6) for s in stds],
        "track_coef": {
            tid: dict(zip(_BASE_NAMES, (round(c, 6) for c in vec), strict=True))
            for tid, vec in track_coef.items()
        },
        "track_sim_coef": {k: round(v, 6) for k, v in track_sim_coef.items()},
        "track_scale": track_scale,
        # 逐站的留出集配对准确率：面板用它告诉用户「这一站的真实校准有多可信」，
        # 而不是只显示融合权重（权重 0.9 很容易被读成「准确率 90%」）。
        "track_accuracy": {
            tid: m.get("matched_acc_real") for tid, m in metrics.items()
        },
        "track_pairs": {
            tid: m.get("matched_pairs") for tid, m in metrics.items()
        },
        "metrics": {
            "cv_folds": args.folds,
            # 主指标：同队同会话配对（全站合并）
            "matched_acc_real": real_acc,
            "matched_acc_sim": sim_acc,
            "matched_pairs": real_pairs,
            "matched_acc_gain": gain,
            # 逐站权重生效后的实际表现（线上口径）
            "blend_matched_acc": blend_acc,
            "blend_pairs": blend_pairs,
            # 参考指标：逐站中位（含车手差异，仅作对照）
            "median_pair_acc_real": real_acc_med,
            "median_pair_acc_sim": sim_acc_med,
            "median_rho_real": real_rho,
            "median_rho_sim": sim_rho,
            "pair_acc_gain": gain,
            "lambda_grid": grid,
        },
        "per_track": metrics,
        "caveat": (
            "主指标为同队同会话配对（同站、同车队、同会话内的两套调教谁更快），"
            "已剔除赛道/车队/会话/年份/切面/TC 主效应，"
            "但车手水平仍不可观测，融合权重按留出集配对表现收缩。"
        ),
    }
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_MODEL.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _w_zero = sum(1 for v in weight_by_track.values() if v <= 0.0)
    _w_half = sum(1 for v in weight_by_track.values() if 0.0 < v < float(payload["blend_weight"] or 0.0))
    _w_full = len(weight_by_track) - _w_zero - _w_half

    lines = [
        "# 真实调教 → 圈速 校准模型训练报告",
        "",
        f"- 样本：**{len(rows)}** 条 / **{len(track_coef)}** 站"
        f"（原始 {len(raw_rows)} 条；丢弃：{_fmt_dropped(dropped)}；"
        f"跳过：{skipped_tracks or '无'}）",
        f"- 结构：变系数岭回归（{_N_BASE} 主效应 + "
        f"{len(beta) - _N_BASE} 交互），λ=**{best_lam:g}**"
        f"（平台容差 {LAMBDA_PLATEAU:g} 内取更小值），{args.folds} 折（按作者分组）",
        f"- 融合权重：全站基准 **{payload['blend_weight']}**，逐站按该站留出准确率派生"
        f"（{_w_zero} 站归零 / {_w_half} 站减半 / {_w_full} 站全权重）",
        f"- 尺度校准：逐站打分等比缩放到标准差 {TARGET_EFFECT_SD_S:g} s"
        f"（等比不改变排序，只把打分翻译成秒）",
        "",
        "## 留出集指标",
        "",
        "主指标为**同队同会话配对**：同一赛道、同一车队、同一会话下两套调教谁更快——",
        "只有这个口径比的是调教本身；站内混比会把 1~2 s 的车手差距盖过 0.3~0.9 s 的调教差异。",
        "",
        "| 指标 | 仿真基线 | 真实校准 | 变化 |",
        "|---|---|---|---|",
        f"| 配对排序准确率（合并 {real_pairs} 对） | {_fmt_pct(sim_acc)} | "
        f"{_fmt_pct(real_acc)} | {'n/a' if gain is None else f'{gain:+.1%}'} |",
        f"| 逐站权重生效后融合准确率（合并 {blend_pairs} 对） | {_fmt_pct(sim_acc)} | "
        f"{_fmt_pct(blend_acc)} | "
        f"{'n/a' if (sim_acc is None or blend_acc is None) else f'{blend_acc - sim_acc:+.1%}'} |",
        f"| 站内混比准确率（中位，参考） | {_fmt_pct(sim_acc_med)} | "
        f"{_fmt_pct(real_acc_med)} | "
        f"{'n/a' if (sim_acc_med is None or real_acc_med is None) else f'{real_acc_med - sim_acc_med:+.1%}'} |",
        f"| Spearman ρ（中位，参考） | {_fmt(sim_rho)} | {_fmt(real_rho)} | "
        f"{'n/a' if (sim_rho is None or real_rho is None) else f'{real_rho - sim_rho:+.3f}'} |",
        "",
        "## 结论",
        "",
    ]
    lines += [f"- {v}" for v in verdict]
    lines += [
        "",
        "> 零泄漏验收（留一赛道 / 跨年）见 `real_lap_holdout.md`："
        "该站完全不参与训练时的配对准确率与上面的 K 折指标应当接近，"
        "否则说明 K 折里有折结构带来的水分。",
        "",
        "## 逐赛道",
        "",
        "| 赛道 | n | 配对数 | 配对准确率 仿真 | 配对准确率 真实 | ρ 仿真 | ρ 真实 | 融合权重 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tid, m in sorted(metrics.items()):
        lines.append(
            f"| {tid} | {m['n']} | {m['matched_pairs']} | "
            f"{_fmt_pct(m['matched_acc_sim'])} | {_fmt_pct(m['matched_acc_real'])} | "
            f"{_fmt(m['rho_sim'])} | {_fmt(m['rho_real'])} | "
            f"{weight_by_track.get(tid, 0.0)} |"
        )
    lines += ["", f"> {payload['caveat']}", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[train] 样本 {len(rows)} / {len(track_coef)} 站；λ={best_lam}；"
          f"融合权重={payload['blend_weight']}")
    print(f"[train] 配对排序（{real_pairs} 对）：仿真 {_fmt_pct(sim_acc)} → 真实 {_fmt_pct(real_acc)}")
    print(f"[train] 站内混比（中位）：仿真 {_fmt_pct(sim_acc_med)} → 真实 {_fmt_pct(real_acc_med)}")
    print(f"[train] Spearman（中位）：仿真 {_fmt(sim_rho)} → 真实 {_fmt(real_rho)}")
    for v in verdict:
        print(f"[verdict] {v}")
    print(f"[out] {OUT_MODEL}\n[out] {OUT_MD}")
    return 0


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.3f}"


def _fmt_pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1%}"


from setup_tuner.domain.setup import ALL_SETUP_FIELDS as _FIELDS  # noqa: E402

_BASE_NAMES: list[str] = [f.name for f in _FIELDS]
_N_BASE = len(_BASE_NAMES)
_BASE_INDEX: dict[str, int] = {name: i for i, name in enumerate(_BASE_NAMES)}
#: 设计矩阵除第 0 列（仿真）以外的列名，顺序与 build_design 完全一致
_DESIGN_TAIL: list[str] = _BASE_NAMES + [
    f"{param}*{feat}"
    for param, feats in INTERACTIONS.items()
    for feat in feats
]


if __name__ == "__main__":
    sys.exit(main())
