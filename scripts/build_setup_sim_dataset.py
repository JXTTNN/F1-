"""构建「调教 → 圈速影响」仿真数据集（遥测锚定，供神经网络训练）。

为什么需要这一层
----------------
公开世界**不存在**「F1 赛车里每个调教组合对应圈速」的监督数据：
- 外部 2026 真遥测（TracingInsights，80k+ 逐弯样本）有弯速/抓地/工况，
  但**没有调教信息**（车队调教不公开）；
- 本地游戏录制有调教（14 圈），但样本量不可能支撑监督学习。

因此本文件采取业界通行的「遥测锚定的准稳态仿真」路线：

    真实遥测（锚点）            +   车辆动力学结构（文档化的相对灵敏度）
    ─────────────────────           ──────────────────────────────────
    每赛道×每弯的中位速度剖面          v² = a_grip · R      （弯心）
    （入弯/弯心/出弯速度、弯时）        相位分解：入弯/弯心/出弯
    赛道中位圈速、弯时占比            直道：阻力项 Δt = f(ΔD, 直道占比)
    工况（胎温/气温/干湿）            灵敏度系数：F1 车辆动力学量级（见 EFFECTS）
                                      （注：工况当前不进入标签，见下文）

    产出 = 每（赛道, 调教, 工况）→ 圈速增量（ms）
    → 训练纯标准库 MLP（scripts/train_setup_sim_nn.py）
    → 引擎用它做「神经网络不断模拟优化」（engine/sim_optimizer.py）

灵敏度系数的口径（诚实声明）
--------------------------
EFFECTS 表里全部是**本项目标定的相对权重**，不是官方数值，也不宣称绝对精度：
- 方向（符号）来自 F1 车辆动力学共识与公开调教指南；
- 幅度标定到公开工程量级的**范围**（例如：前后翼 ±10 档 ≈ 0.2–0.5 s/圈；
  外倾 ±0.5° ≈ 0.05–0.15 s/圈；悬挂软硬整档 ≈ 0.05–0.15 s/圈），
  由 build 脚本末尾的标定表打印出来复核；
- 弯速对抓地的平方根响应（v ∝ √a）、气动力随速度平方增长（w_aero 按弯型）
  是结构性的，不是拟合参数。

工况特征的真实状态（不要误解）
----------------------------
``simulate_lap_delta_ms()`` **不接受工况参数**：湿、胎温、气温、胎软度在数据集
里只是随行记录，不影响 ``lap_delta_ms``。而 ``track_features`` 每赛道只有一组
常量，因此在特征向量里这 4 维与赛道 one-hot 完全共线 —— 模型学不到任何
「湿地让尾翼更重要」的知识。要给工况真实权重，必须先给仿真器加湿地方程，
那是独立的一期。当前行为记为 "> 仅记录"，不在文档/界面里声称已建模。

采样策略（决定模型“准在哪里”）
-------------------------------
模型的精度只能分布在它见过的区域。旧版每赛道 400 个样本在**全量程均匀**采样，
标签均值约 2.8 s、标准差约 1.3 s —— 容量全花在区分「糟糕/极糟糕」上，而调教
优化只在当前调教附近 ±``max_delta`` 内比较「哪套快几十毫秒」。现改为三档混合：

    40%  沿「默认→最优」路径随机取位 t∈[0,1]，逐参数抖动 ±0.15（主力区）
    30%  同上，t∈[0,1.1]，抖动 ±0.40（跨档位组合、略越过最优）
    30%  全量程均匀（粗形状 + 极端区域，保证远离最优点时预测不发散）

该策略由 ``scripts/eval_setup_sim.py`` 的决策指标（局部 MAE / 成对排序准确率 /
方向一致率）验证 —— 训练集的 ``val_mae`` 不作为「能不能调车」的依据。

赛道覆盖：遥测锚点 + 几何推定锚点
---------------------------------
遥测只能锚定「已经跑过的分站」。2026 赛历上还有几站没有遥测语料（后段分站
尚未开赛、已取消分站），用户却可以在 UI 里选中它们练习 —— 那几站过去直接显示
「模型未启用」，等于 24 条赛道里有 5 条没有模型。

本文件因此增加**几何推定锚点**（``authored_anchor``）：用 19 站真实遥测拟合出
「每类弯的等效弯长 / 制动占比」与「圈速 = f(长度, Σ1/v)」三个全局常数，再套到
仓库里已人工维护的赛道几何（``domain.track``：每弯类别 + 弯心速度 + 赛道长度）
上，得到该站的弯剖面与圈速锚点，进而用**同一个仿真器**生成标签。

诚实边界（必须记住）：
- 这类锚点**不是遥测**，标签是「几何 + 全局常数」推出来的，精度低于遥测锚点；
- 因此 ``meta["anchor_sources"]`` 逐站标注 ``telemetry`` / ``authored``，
  模型与报告如实展示，不把推定说成实测；
- 拟合质量有门禁：19 站上圈速拟合的中位相对误差 > ``_AUTHORED_FIT_MAX_ERR``
  时**放弃**几何推定（覆盖回到纯遥测），避免用坏常数污染模型。

用法::

    python scripts/build_setup_sim_dataset.py                 # 默认 400 调教/赛道
    python scripts/build_setup_sim_dataset.py --setups 800
    python scripts/build_setup_sim_dataset.py --tracks monza,suzuka
    python scripts/build_setup_sim_dataset.py --sampling legacy  # 复现旧行为/消融
    python scripts/build_setup_sim_dataset.py --no-authored-anchors  # 只用遥测
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORNER_JSONL = ROOT / "data" / "training" / "corner_dataset.jsonl"
OUT_JSONL = ROOT / "data" / "training" / "setup_sim_dataset.jsonl"
OUT_META = ROOT / "data" / "training" / "setup_sim_meta.json"

# --------------------------------------------------------------------------- #
# 1. 调教 → 能力系数（相对权重表；符号 = 增大该参数时的方向）
# --------------------------------------------------------------------------- #
# 能力维度：
#   A = 气动抓地（随速度平方生效，按弯型加权）
#   M = 机械抓地（与速度无关，全弯型生效）
#   T = 出弯牵引（慢弯出弯相位为主）
#   B = 制动效能（入弯相位为主）
#   D = 阻力（直道时间，越大越慢）
#
# 数值 = 「归一化整幅摆动 [-1,+1]」对应的能力相对变化。
# 归一化 n(p) = (value - default) / half_range，其中
# half_range = max(max-default, default-min)，n ∈ [-1, 1]。
EFFECTS: dict[str, dict[str, float]] = {
    # 前/后翼：下压力↑（气动抓地，按弯型加权）；阻力代价走平方项（见下方阻力模型）
    "front_wing": {"A": +0.035},
    "rear_wing": {"A": +0.035, "M": +0.010},
    # 差速器：油门锁止率↑ → 出弯牵引↑、转向略钝；松油锁止率↑ → 入弯稳、旋转少
    "on_throttle_diff": {"T": +0.040, "M": -0.005},
    "off_throttle_diff": {"M": +0.010, "T": -0.010},
    # 外倾：负值越大 → 弯中侧向抓地越强（前轮区间 -3.5..-2.5，默认 -3.5）
    # 注意默认在最负端 → n>0 表示「减少负外倾」→ 机械抓地下降（系数取负）
    "front_camber": {"M": -0.022, "D": +0.005},
    "rear_camber": {"M": -0.022, "T": -0.010, "D": +0.005},
    # 前束：束角越大 → 阻力越大；前束大转向响应钝
    "front_toe": {"M": -0.005, "D": +0.030},
    "rear_toe": {"T": +0.010, "D": +0.030},
    # 悬挂：数值小 = 更软 → 机械抓地↑（低速相位）；后悬更软还利于出弯牵引
    "front_suspension": {"M": -0.020},
    "rear_suspension": {"M": -0.015, "T": -0.025},
    # 防倾杆：数值小 = 更软 → 单侧抓地↑；后杆更软利于出弯牵引
    "front_anti_roll_bar": {"M": -0.012},
    "rear_anti_roll_bar": {"M": -0.008, "T": -0.018},
    # 离地间隙：数值小 = 更低 → 地效下压力↑（阻力代价走平方项，见阻力模型）
    "front_ride_height": {"A": -0.035},
    "rear_ride_height": {"A": -0.035},
    # 刹车：压力大 → 制动效能↑；平衡偏离默认（58）→ 效能下降（双侧罚，见 |n|²）
    "brake_pressure": {"B": +0.030},
    "brake_bias": {"B": 0.0},  # 单独处理：B -= 0.010 * n²
    # 胎压：偏离默认 → 抓地下降（轮胎工作窗口，双侧罚）；
    # 胎压偏高 → 略增制动响应
    "front_left_tyre_pressure": {"M": 0.0, "B": +0.005},
    "front_right_tyre_pressure": {"M": 0.0, "B": +0.005},
    "rear_left_tyre_pressure": {"M": 0.0, "B": +0.005},
    "rear_right_tyre_pressure": {"M": 0.0, "B": +0.005},
}

#: 胎压工作窗口罚：M -= WINDOW_PENALTY * n²（n 为归一化偏离）
_PRESSURE_WINDOW_PENALTY = 0.020
#: 刹车平衡偏离罚：B -= BIAS_PENALTY * n²
_BIAS_PENALTY = 0.010
#: 刮底阈值与罚（离地间隙归一化 < 阈值 时，圈速罚 = K * (阈值 - n)）
_BOTTOMING_THRESHOLD = -0.70
_BOTTOMING_K = 0.06   # 秒/单位（对整圈时间贡献，仅当前/后都极低时）

# --------------------------------------------------------------------------- #
# 阻力项（内点最优的来源）
# --------------------------------------------------------------------------- #
# 关键结构：**诱导阻力随下压力平方增长**（CD = CD0 + k·CL²，空气动力学常识）。
# 若阻力对翼角线性，则「下压力收益（线性）− 阻力代价（线性）」单调无内点，
# 仿真最优解会永远落在翼角极值上（实测：Monaco 判成"翼角拉满"）——
# 与真实调教行为（每条赛道有各自最优翼角）矛盾。
# 因此翼角/离地间隙的阻力代价 = C_LIN·df + C_IND·df²：
#   df > 0（增加下压力）→ 线性+平方双升；
#   df < 0（减少下压力）→ 线性项下降、平方项仍为正，净效应温和（减翼省阻有限）。
# 权重 = 直道占比 × 速度因子（均速/200）²——Monza 长直道均速高 → 阻力代价大；
#       Monaco 低速窄街 → 阻力几乎不罚。
_C_LIN = 41638.0   # ms / 单位 df（×drag_scale）—— 仅作用于**翼角**
_C_IND = 455685.0  # ms / 单位 df²（×drag_scale）—— 诱导阻力，仅作用于翼角
#   标定目标（df* = 各赛道最优翼角，来自 S_track/drag_scale 排序）：
#   monaco ≈ +12 档（最想加翼）、spa ≈ -6 档（最想减翼）、monza ≈ -1、
#   hungaroring ≈ +11；标定依据：真实调教行为（Monaco 最高下压力，
#   Monza/Spa/Silverstone 低阻配置）+ 逐赛道方向校验收敛表。
#: 离地间隙的线性阻力收益（更低 = 更省阻，负系数）：全幅 -4 档 ≈ 30-70ms
_RIDE_DRAG_LIN = -2500.0  # ms / 单位 df_rh（×drag_scale）
#: 其余参数（前束等）的线性阻力：Δt = 0.5 · dD_other · straight_share · t_lap
_DRAG_K = 1.0

#: 弯型 → 气动抓地权重（aero 载荷 ∝ v²，快弯权重高）
_AERO_WEIGHT = {"slow": 0.08, "medium": 0.30, "fast": 0.65}
#: 相位权重（入弯/弯心/出弯）
_PHASE_IN, _PHASE_MID, _PHASE_OUT = 0.30, 0.40, 0.30
#: 速度对抓地平方根响应：Δv/v = 0.5 · Δcap/cap
_SPEED_HALF = 0.5
#: 入弯制动相位的效能权重（制动占比高的弯受益更大）
_BRAKE_PHASE_W = 0.6
#: 出弯牵引相位的机械抓地混合权重
_EXIT_MECH_MIX = 0.4
#: 出弯相位的气动权重折减（低速时气动效率低）
_EXIT_AERO_SCALE = 0.3


def _half_range(spec: Any) -> float:
    """参数归一化的半幅（到较远一端），保证 n ∈ [-1, +1]。"""
    return max(abs(spec.max_val - spec.default), abs(spec.default - spec.min_val)) or 1.0


def _norm(value: float, spec: Any) -> float:
    """参数值 → 归一化 n ∈ [-1, +1]（相对默认值）。"""
    return (float(value) - spec.default) / _half_range(spec)


def capability_deltas(setup_norm: dict[str, float]) -> dict[str, float]:
    """归一化调教 → 五项能力增量 {A, M, T, B, D}。"""
    caps = {"A": 0.0, "M": 0.0, "T": 0.0, "B": 0.0, "D": 0.0}
    for name, row in EFFECTS.items():
        n = setup_norm.get(name, 0.0)
        for cap, coef in row.items():
            caps[cap] += coef * n
    # 胎压工作窗口（双侧罚，非线性的来源之一）
    for name in (
        "front_left_tyre_pressure", "front_right_tyre_pressure",
        "rear_left_tyre_pressure", "rear_right_tyre_pressure",
    ):
        n = setup_norm.get(name, 0.0)
        caps["M"] -= _PRESSURE_WINDOW_PENALTY * n * n
    # 刹车平衡偏离（双侧罚）
    n_bias = setup_norm.get("brake_bias", 0.0)
    caps["B"] -= _BIAS_PENALTY * n_bias * n_bias
    return caps


def bottoming_penalty(setup_norm: dict[str, float]) -> float:
    """离地间隙过低 → 刮底圈速罚（秒）。"""
    penalty = 0.0
    for name, scale in (("front_ride_height", 1.0), ("rear_ride_height", 1.0)):
        n = setup_norm.get(name, 0.0)
        if n < _BOTTOMING_THRESHOLD:
            penalty += _BOTTOMING_K * (_BOTTOMING_THRESHOLD - n) * scale
    return penalty


# --------------------------------------------------------------------------- #
# 2. 遥测锚点：从真实逐弯数据提取每赛道速度剖面
# --------------------------------------------------------------------------- #
def load_corner_stats(
    tracks_filter: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """读取 80k+ 真实逐弯样本 → 每赛道（弯剖面 + 工况 + 圈速）。

    Returns:
        {track_id: {
            "corners": [{corner, klass, v_entry, v_min, v_exit, t_corner,
                          brake_pct, throttle_pct, n}],
            "lap_time_s": float,       # 中位圈速（锚点）
            "straight_share": float,   # 直道时间占比（1 - 弯时和/圈速，下限 0.15）
            "traction_index": float, "aero_index": float, "braking_index": float,
            "slow_share": float, "fast_share": float,
            "tyre_softness": float, "wet": float,
            "track_temp": float, "air_temp": float,
            "n_samples": int,
        }}
    """
    if not CORNER_JSONL.exists():
        raise SystemExit(f"缺少逐弯数据集 {CORNER_JSONL}（先跑 build_corner_dataset.py）")

    # 分组累积
    acc: dict[str, dict[int, dict[str, list[float]]]] = {}
    track_meta: dict[str, dict[str, Any]] = {}
    lap_times: dict[str, list[float]] = {}

    with CORNER_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = str(s.get("track_id") or "")
            if not tid or (tracks_filter and tid not in tracks_filter):
                continue
            corner = int(s.get("corner") or 0)
            if corner <= 0:
                continue
            bucket = acc.setdefault(tid, {}).setdefault(corner, {
                "v_entry": [], "v_min": [], "v_exit": [], "t_corner": [],
                "brake": [], "throttle": [], "lat_g": [],
            })
            for key, src in (
                ("v_entry", "entry_speed"), ("v_min", "min_speed"),
                ("v_exit", "exit_speed"), ("t_corner", "corner_time_s"),
                ("brake", "brake_pct"), ("throttle", "throttle_full_pct"),
                ("lat_g", "max_lateral_g"),
            ):
                v = s.get(src)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    bucket[key].append(float(v))
            # 弯型（取该弯最常见值）
            if "klass" not in bucket:
                bucket["klass"] = str(s.get("corner_class") or "medium")
            lt = s.get("lap_time_s")
            if isinstance(lt, (int, float)) and lt > 0:
                lap_times.setdefault(tid, []).append(float(lt))
            # 赛道级特征（任一非空样本）
            tm = track_meta.setdefault(tid, {})
            for key, src in (
                ("traction_index", "track_traction_index"),
                ("aero_index", "track_aero_index"),
                ("braking_index", "track_braking_index"),
                ("slow_share", "track_slow_share"),
                ("fast_share", "track_fast_share"),
                ("tyre_softness", "tyre_softness"),
                ("wet", "wet"),
                ("track_temp", "track_temp"),
                ("air_temp", "air_temp"),
            ):
                if key not in tm:
                    v = s.get(src)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        tm[key] = float(v)

    out: dict[str, dict[str, Any]] = {}
    for tid, corners in acc.items():
        rows: list[dict[str, Any]] = []
        for corner, b in sorted(corners.items()):
            if len(b["t_corner"]) < 3:
                continue
            rows.append({
                "corner": corner,
                "klass": b.get("klass", "medium"),
                "v_entry": statistics.median(b["v_entry"]) if b["v_entry"] else 0.0,
                "v_min": statistics.median(b["v_min"]) if b["v_min"] else 0.0,
                "v_exit": statistics.median(b["v_exit"]) if b["v_exit"] else 0.0,
                "t_corner": statistics.median(b["t_corner"]),
                "brake_pct": statistics.median(b["brake"]) if b["brake"] else 0.0,
                "lat_g": statistics.median(b["lat_g"]) if b["lat_g"] else 0.0,
                "n": len(b["t_corner"]),
            })
        if not rows:
            continue
        lap_med = statistics.median(lap_times.get(tid, [0.0])) or 0.0
        tm = track_meta.get(tid, {})
        # 直道时间占比（拖曳项的权重）：外部弯数据窗口含加减速段，
        # 「1 - 弯时和/圈速」会被压在低位且无赛道差异（实测全部 0.15），
        # 因此改用**可区分的信号组合**（本项目的相对权重，标定到合理区间）：
        #   重刹车指数（重刹前必然有长直道）+ 气动依赖（快弯多=高速段多）
        #   + 快弯占比 − 慢弯占比；clamp 到 [0.18, 0.55]。
        # 方向校验：monaco ≈ 0.18（最低）< monza ≈ 0.35 < spa ≈ 0.55（最高）。
        braking_index = float(tm.get("braking_index", 0.0))
        aero_index = float(tm.get("aero_index", 0.0))
        slow_share = float(tm.get("slow_share", 0.0))
        fast_share = float(tm.get("fast_share", 0.0))
        straight_share = min(0.55, max(
            0.18,
            0.15 + 0.7 * braking_index + 0.5 * aero_index
            + 0.3 * fast_share - 0.3 * slow_share,
        ))
        # 均速（真实赛道长度 × 遥测中位圈速）→ 阻力代价的速度因子
        # （气动阻力功率 ∝ v³，取均速平方近似；Monza 均速高 → 阻力代价大）
        avg_speed_kmh = 0.0
        try:
            from setup_tuner.domain.track import get_track_by_id

            track_obj = get_track_by_id(tid)
            if track_obj is not None and lap_med > 1.0:
                avg_speed_kmh = float(track_obj.length_m) / lap_med * 3.6
        except Exception:  # noqa: BLE001 — 未知赛道中性降级
            avg_speed_kmh = 0.0
        drag_scale = (
            straight_share * (avg_speed_kmh / 200.0) ** 2 * (lap_med / 90.0)
            if avg_speed_kmh > 0.0 else 0.0
        )
        out[tid] = {
            "corners": rows,
            "lap_time_s": lap_med,
            "straight_share": straight_share,
            "avg_speed_kmh": avg_speed_kmh,
            "drag_scale": drag_scale,
            "traction_index": float(tm.get("traction_index", 0.0)),
            "aero_index": aero_index,
            "braking_index": braking_index,
            "slow_share": slow_share,
            "fast_share": fast_share,
            "tyre_softness": float(tm.get("tyre_softness", 0.6)),
            "wet": float(tm.get("wet", 0.0)),
            "track_temp": float(tm.get("track_temp", 35.0)),
            "air_temp": float(tm.get("air_temp", 24.0)),
            "n_samples": sum(r["n"] for r in rows),
        }
    return out


# --------------------------------------------------------------------------- #
# 2b. 几何推定锚点（无遥测赛道）
# --------------------------------------------------------------------------- #
#: 圈速拟合在 19 站上的中位相对误差门限；超过则放弃几何推定（宁缺勿滥）
_AUTHORED_FIT_MAX_ERR = 0.08
#: 至少要有这么多站遥测才允许拟合全局常数
_AUTHORED_MIN_TRACKS = 8
def fit_authored_constants(stats: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """从遥测锚点拟合「几何推定」所需的全局常数。

    拟合三样东西（都是**全局单值 / 每弯类一个值**，参数极少，不容易过拟合）：

    1. 每类弯的**等效弯长** ``eff_len = t_corner · v_min``（米）—— 中位数；
    2. 每类弯的**制动占比** ``brake_pct`` —— 中位数；
    3. **圈速模型** ``lap = c0 + c1·(length/1000) + c2·Σ(1/v_i)``，
       最小二乘（3 个系数），并在 19 站上做留一交叉验证式的误差统计。

    Returns:
        ``{"corner": {...}, "lap": {...}, "share": float}``；数据不足或拟合
        质量不达标时返回 ``None``（调用方退化为纯遥测覆盖）。
    """
    if len(stats) < _AUTHORED_MIN_TRACKS:
        print(f"[warn] 只有 {len(stats)} 站遥测，不足 {_AUTHORED_MIN_TRACKS} 站，"
              f"跳过几何推定锚点")
        return None

    per_class_len: dict[str, list[float]] = {}
    per_class_brake: dict[str, list[float]] = {}
    shares: list[float] = []
    design: list[list[float]] = []
    targets: list[float] = []
    unknown_klass: dict[str, int] = {}

    for tid, track in stats.items():
        corner_time_sum = 0.0
        inv_v_sum = 0.0
        for c in track["corners"]:
            klass = str(c.get("klass") or "medium").lower()
            # 只接受 slow/medium/fast 三档：几何数据侧的类别名必须与之一致，
            # 否则 authored 查找会全部落空、退化成同一个兜底弯长（静默劣化）。
            if klass not in ("slow", "medium", "fast"):
                unknown_klass[klass] = unknown_klass.get(klass, 0) + 1
                klass = "medium"
            v = max(30.0, float(c.get("v_min") or 0.0)) / 3.6
            t = float(c.get("t_corner") or 0.0)
            corner_time_sum += t
            inv_v_sum += 1.0 / v if v > 0 else 0.0
            per_class_len.setdefault(klass, []).append(t * v)
            per_class_brake.setdefault(klass, []).append(float(c.get("brake_pct") or 0.0))
        lap = float(track["lap_time_s"] or 0.0)
        if lap <= 1.0:
            continue
        if corner_time_sum > 0:
            shares.append(min(0.9, corner_time_sum / lap))
        try:
            from setup_tuner.domain.track import get_track_by_id

            tobj = get_track_by_id(tid)
            length_m = float(tobj.length_m) if tobj is not None else 0.0
        except Exception:  # noqa: BLE001
            length_m = 0.0
        if length_m <= 0:
            continue
        design.append([1.0, length_m / 1000.0, inv_v_sum])
        targets.append(lap)

    if unknown_klass:
        print(f"[warn] 遥测弯类出现未知取值，已按 medium 处理：{unknown_klass}")
    if not design:
        print("[warn] 没有可用于圈速拟合的赛道（缺长度或圈速），跳过几何推定锚点")
        return None
    corner_cal = {
        klass: {
            "eff_len_m": round(statistics.median(vals), 1),
            "brake_pct": round(statistics.median(per_class_brake.get(klass, [30.0])), 1),
            "n": len(vals),
        }
        for klass, vals in per_class_len.items()
    }
    share = round(statistics.median(shares), 4) if shares else 0.62

    # 最小二乘：解 (XᵀX)c = Xᵀy（3×3，纯标准库高斯消元）
    coeffs = _least_squares(design, targets)
    if coeffs is None:
        print("[warn] 圈速模型拟合失败（奇异矩阵），跳过几何推定锚点")
        return None
    rel_err = []
    for feat, lap in zip(design, targets, strict=True):
        pred = sum(c * f for c, f in zip(coeffs, feat, strict=True))
        if lap > 1.0 and pred > 1.0:
            rel_err.append(abs(pred - lap) / lap)
    rel_err.sort()
    med_err = statistics.median(rel_err) if rel_err else 1.0
    print(f"几何推定常数：弯类 {len(corner_cal)} 个 | 弯时占比中位 {share:.3f} | "
          f"圈速模型 c0={coeffs[0]:+.2f}s c1={coeffs[1]:+.3f}s/km c2={coeffs[2]:+.3f}")
    print(f"圈速模型回代 {len(targets)} 站：中位相对误差 {med_err:.2%} | "
          f"最差 {rel_err[-1]:.2%}" if rel_err else "圈速模型无有效回代样本")
    if med_err > _AUTHORED_FIT_MAX_ERR:
        print(f"[warn] 圈速拟合中位误差 {med_err:.2%} > 门限 "
              f"{_AUTHORED_FIT_MAX_ERR:.0%}，放弃几何推定锚点")
        return None
    return {"corner": corner_cal, "lap": {"coeffs": coeffs, "median_rel_err": med_err},
            "share": share, "n_tracks": len(targets)}


def _least_squares(x: list[list[float]], y: list[float]) -> list[float] | None:
    """最小二乘（正规方程 + 高斯消元，带部分主元）；奇异返回 None。"""
    n = len(x[0]) if x else 0
    ata = [[0.0] * n for _ in range(n)]
    aty = [0.0] * n
    for row, target in zip(x, y, strict=True):
        for i in range(n):
            aty[i] += row[i] * target
            for j in range(n):
                ata[i][j] += row[i] * row[j]
    # 高斯消元
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(ata[r][col]))
        if abs(ata[pivot][col]) < 1e-12:
            return None
        ata[col], ata[pivot] = ata[pivot], ata[col]
        aty[col], aty[pivot] = aty[pivot], aty[col]
        for r in range(col + 1, n):
            factor = ata[r][col] / ata[col][col]
            if factor == 0.0:
                continue
            for c in range(col, n):
                ata[r][c] -= factor * ata[col][c]
            aty[r] -= factor * aty[col]
    out = [0.0] * n
    for r in range(n - 1, -1, -1):
        acc = aty[r] - sum(ata[r][c] * out[c] for c in range(r + 1, n))
        out[r] = acc / ata[r][r]
    return out


def authored_anchor(
    track_id: str, calib: dict[str, Any], condition: dict[str, float],
) -> dict[str, Any] | None:
    """用**赛道几何 + 全局常数**推一条赛道的锚点（无遥测时使用）。

    与 :func:`load_corner_stats` 返回值同构，因此下游（采样 / 仿真 / 元信息）
    完全复用同一套代码 —— 差别只在 ``anchor`` 字段标注为 ``authored``。

    Args:
        track_id: 赛道标识（必须在 ``domain.track`` 里，且有人工维护的弯数据）。
        calib: :func:`fit_authored_constants` 的结果。
        condition: 该站工况（无遥测时用已覆盖赛道的中位数）。

    Returns:
        锚点字典；赛道不存在/无弯数据/推不出合理圈速时返回 ``None``。
    """
    from setup_tuner.domain.track import get_track_by_id
    from setup_tuner.engine.holistic import corner_class, track_demand

    track = get_track_by_id(track_id)
    if track is None or not track.corners or not track.length_m:
        return None
    demand = track_demand(track_id)
    corner_cal = calib["corner"]
    fallback_len = statistics.median(
        [v["eff_len_m"] for v in corner_cal.values()]
    ) if corner_cal else 180.0

    rows: list[dict[str, Any]] = []
    inv_v_sum = 0.0
    raw_corner_time = 0.0
    for c in track.corners:
        klass = corner_class(c.corner_type)
        v_kmh = max(30.0, float(c.speed_kmh))
        v = v_kmh / 3.6
        eff = float(corner_cal.get(klass, {}).get("eff_len_m", fallback_len))
        t_corner = eff / v
        raw_corner_time += t_corner
        inv_v_sum += 1.0 / v
        rows.append({
            "corner": int(c.number),
            "klass": klass,
            "v_entry": v_kmh, "v_min": v_kmh, "v_exit": v_kmh,
            "t_corner": t_corner,
            "brake_pct": float(corner_cal.get(klass, {}).get("brake_pct", 30.0)),
            "lat_g": 0.0,
            "n": 0,
        })

    c0, c1, c2 = calib["lap"]["coeffs"]
    lap = c0 + c1 * (float(track.length_m) / 1000.0) + c2 * inv_v_sum
    share = float(calib["share"])
    # 弯时占比归一：让该站「弯时和 / 圈速」等于遥测站的中位占比，避免用类目
    # 中位弯长逐个叠加后把整圈时间挤爆（Σt_corner 可能超过圈速本身）。
    target_corner_time = max(1.0, share * lap)
    if raw_corner_time > 1e-6:
        scale = target_corner_time / raw_corner_time
        for r in rows:
            r["t_corner"] = round(r["t_corner"] * scale, 4)
    if lap <= 10.0:
        return None

    straight_share = min(0.55, max(
        0.18,
        0.15 + 0.7 * demand.braking_index + 0.5 * demand.aero_index
        + 0.3 * demand.fast_share - 0.3 * demand.slow_share,
    ))
    avg_speed_kmh = float(track.length_m) / lap * 3.6
    drag_scale = straight_share * (avg_speed_kmh / 200.0) ** 2 * (lap / 90.0)
    return {
        "corners": rows,
        "lap_time_s": round(lap, 3),
        "straight_share": round(straight_share, 4),
        "avg_speed_kmh": round(avg_speed_kmh, 2),
        "drag_scale": round(drag_scale, 4),
        "traction_index": demand.traction_index,
        "aero_index": demand.aero_index,
        "braking_index": demand.braking_index,
        "slow_share": demand.slow_share,
        "fast_share": demand.fast_share,
        "tyre_softness": condition["tyre_softness"],
        "wet": condition["wet"],
        "track_temp": condition["track_temp"],
        "air_temp": condition["air_temp"],
        "n_samples": 0,
        "anchor": "authored",
    }


# --------------------------------------------------------------------------- #
# 3. 仿真：调教 → 圈速增量
# --------------------------------------------------------------------------- #
def simulate_lap_delta_ms(
    track: dict[str, Any], setup_norm: dict[str, float],
) -> float:
    """仿真该调教相对**默认调教**的圈速增量（ms；正 = 更慢）。

    结构（全部来自遥测锚点 + 文档化灵敏度）：
        每个弯：Δt = -t_corner · Σ_phase φ · 0.5 · Δcap/cap
            入弯：dM + w_aero·dA + 0.6·dB·(brake_pct/100)
            弯心：dM + w_aero·dA
            出弯：dT + 0.4·dM + 0.3·w_aero·dA
        下压力-阻力权衡（内点最优）：
            Δt_drag = (C_LIN·df + C_IND·df²) · drag_scale   （df = dA）
            drag_scale = 直道占比 × (均速/200)² × (t_lap/90)
        其余阻力（前束等）：+0.5 · dD_other · straight_share · t_lap
        刮底：+ bottoming_penalty（离地间隙过低罚）
    """
    caps = capability_deltas(setup_norm)
    d_m, d_a, d_t, d_b, d_d = (
        caps["M"], caps["A"], caps["T"], caps["B"], caps["D"],
    )
    delta_s = 0.0
    for c in track["corners"]:
        w_aero = _AERO_WEIGHT.get(str(c["klass"]).lower(), 0.30)
        brake_frac = min(1.0, max(0.0, float(c["brake_pct"]) / 100.0))
        # 各相位的相对速度变化（Δv/v = 0.5·Δcap）
        v_in = _SPEED_HALF * (d_m + w_aero * d_a + _BRAKE_PHASE_W * d_b * brake_frac)
        v_mid = _SPEED_HALF * (d_m + w_aero * d_a)
        v_out = _SPEED_HALF * (
            d_t + _EXIT_MECH_MIX * d_m + _EXIT_AERO_SCALE * w_aero * d_a
        )
        speed_gain = _PHASE_IN * v_in + _PHASE_MID * v_mid + _PHASE_OUT * v_out
        # 时间变化 = -t · Δv/v（速度提升 → 时间下降）
        delta_s += -float(c["t_corner"]) * speed_gain
    # 下压力-阻力权衡项（诱导阻力 ∝ df²，给出一族内点最优）
    # 只作用于**翼角**（翼角增加阻力）；离地间隙更低反而省阻（小线性项）。
    wings_df = 0.035 * (
        setup_norm.get("front_wing", 0.0) + setup_norm.get("rear_wing", 0.0)
    )
    ride_df = -0.035 * (
        setup_norm.get("front_ride_height", 0.0) + setup_norm.get("rear_ride_height", 0.0)
    )
    drag_scale = float(track.get("drag_scale", 0.0))
    if drag_scale > 0.0:
        delta_s += (_C_LIN * wings_df + _C_IND * wings_df * wings_df) * drag_scale / 1000.0
        delta_s += _RIDE_DRAG_LIN * ride_df * drag_scale / 1000.0
    # 其余线性阻力（前束等）
    t_lap = float(track["lap_time_s"]) or 90.0
    delta_s += _SPEED_HALF * d_d * float(track["straight_share"]) * _DRAG_K * t_lap
    # 刮底罚
    delta_s += bottoming_penalty(setup_norm)
    return round(delta_s * 1000.0, 3)


# --------------------------------------------------------------------------- #
# 4. 采样与写出
# --------------------------------------------------------------------------- #
def _sample_setup(rng: random.Random, specs: list[Any]) -> dict[str, float]:
    """在合法区间内均匀采样一组调教（对齐步长）。"""
    setup: dict[str, float] = {}
    for spec in specs:
        n_steps = int(round((spec.max_val - spec.min_val) / spec.step)) or 1
        value = spec.min_val + rng.randint(0, n_steps) * spec.step
        if spec.step >= 1.0 and float(spec.step).is_integer():
            value = float(round(value))
        setup[spec.name] = value
    return setup


#: 采样分档：``(抖动半宽 | None=全量程均匀, 占比, 路径位置上限)``。
#:
#: 为什么不是全量程均匀：优化器只在当前调教附近 ±``max_delta`` 内移动，
#: 决策尺度是「10~50 ms 谁更快」。全量程均匀采样给出的标签均值约 2.8 s、
#: 标准差约 1.3 s，模型容量全部花在区分「糟糕/极糟糕」上，恰恰在最需要
#: 精度的邻域精度最差。
#:
#: 为什么是「路径管」而不是「最优解邻域」：坐标上升从当前调教（通常接近
#: 默认）出发，一步步摸到最优，实际活动区是**默认→最优这条路径的管状
#: 邻域**，而不是最优解一个点。先沿路径随机取位置 ``t``，再在 20 维上加
#: ±half_width 抖动；t 上限允许略超过 1（越过最优一点点），避免把最优
#: 解压成边界。
_SAMPLE_TIERS: tuple[tuple[float | None, float, float], ...] = (
    (0.15, 0.40, 1.00),   # 优化器实际走的路径 ±0.15（主力样本）
    (0.40, 0.30, 1.10),   # 更宽邻域/略越最优（跨档位组合）
    (None, 0.30, 0.00),   # 全量程均匀（粗形状 + 极端区域，保证不发散）
)
#: 近似最优点的坐标下降：步长（归一化单位）与轮数
_OPT_GRID = 0.25
_OPT_PASSES = 4


def snap_to_step(value_norm: float, spec: Any) -> float:
    """归一化值 → 先夹进合法区间、对齐到该参数 ``step`` 档位，再归一化。

    采样的候选值必须落在**车手真的能设出来**的离散档位上，否则模型会在
    不存在的中间值上学到虚假曲率，优化器又会给出设不出来的建议。
    """
    real = spec.default + float(value_norm) * _half_range(spec)
    real = max(spec.min_val, min(spec.max_val, real))
    n_steps = round((real - spec.min_val) / spec.step)
    real = spec.min_val + n_steps * spec.step
    if spec.step >= 1.0 and float(spec.step).is_integer():
        real = float(round(real))
    real = max(spec.min_val, min(spec.max_val, real))
    return (real - spec.default) / _half_range(spec)


def coarse_optimum(
    track: dict[str, Any], specs: list[Any],
    grid: float = _OPT_GRID, passes: int = _OPT_PASSES,
) -> tuple[dict[str, float], float]:
    """用仿真器做确定性坐标下降，求该赛道的近似最优调教（归一化）。

    Returns:
        ``(最优调教, 该调教相对默认的圈速增量 ms)``。完全确定性（无随机）。
        这是「训练样本该落在哪里」的答案：调教优化需要精度的区域是最优点
        邻域，而不是全量程。
    """
    cur: dict[str, float] = {spec.name: 0.0 for spec in specs}
    best = simulate_lap_delta_ms(track, cur)
    for _ in range(passes):
        moved = False
        for spec in specs:
            base = cur[spec.name]
            for direction in (+1, -1):
                cand = snap_to_step(base + direction * grid, spec)
                if abs(cand - base) < 1e-9:
                    continue
                trial = dict(cur)
                trial[spec.name] = cand
                value = simulate_lap_delta_ms(track, trial)
                if value < best - 1e-6:
                    cur, best, moved = trial, value, True
        if not moved:
            break
    return cur, best


def _tier_counts(total: int) -> list[int]:
    """按分档占比把 ``total`` 拆成整数（余数补给最后一档，和恒等于 total）。"""
    counts: list[int] = []
    used = 0
    for idx, tier in enumerate(_SAMPLE_TIERS):
        if idx == len(_SAMPLE_TIERS) - 1:
            counts.append(max(0, total - used))
        else:
            n = int(round(total * tier[1]))
            counts.append(n)
            used += n
    return counts

def build(
    n_setups: int, seed: int, tracks_filter: set[str] | None, out: Path,
    sampling: str = "tube", authored_anchors: bool = True,
) -> dict[str, Any]:
    from setup_tuner.domain.setup import ALL_SETUP_FIELDS

    specs = list(ALL_SETUP_FIELDS)
    stats = load_corner_stats(tracks_filter)
    if not stats:
        raise SystemExit("没有可用的赛道锚点数据")
    for track in stats.values():
        track.setdefault("anchor", "telemetry")
    print(f"遥测锚点：{len(stats)} 赛道 | 总逐弯样本 "
          f"{sum(t['n_samples'] for t in stats.values())}")

    # ---- 无遥测赛道的几何推定锚点（见模块文档「赛道覆盖」）---------------- #
    anchor_sources = {tid: "telemetry" for tid in stats}
    calibration_block: dict[str, Any] = {}
    if authored_anchors:
        try:
            from setup_tuner.domain.track import ALL_TRACKS

            calib = fit_authored_constants(stats)
            if calib is not None:
                covered = set(stats)
                condition = {
                    key: round(statistics.median(
                        [float(t.get(key, 0.0)) for t in stats.values()]
                    ), 4)
                    for key in ("tyre_softness", "wet", "track_temp", "air_temp")
                }
                added = 0
                for tobj in ALL_TRACKS:
                    tid = tobj.track_id
                    if tid in covered:
                        continue
                    if tracks_filter and tid not in tracks_filter:
                        continue
                    anchor = authored_anchor(tid, calib, condition)
                    if anchor is None:
                        print(f"  [skip] {tid}: 赛道几何不足，无法推定锚点")
                        continue
                    stats[tid] = anchor
                    anchor_sources[tid] = "authored"
                    added += 1
                calibration_block = {
                    "corner": calib["corner"],
                    "lap_model": calib["lap"],
                    "corner_time_share": calib["share"],
                    "n_tracks_fitted": calib["n_tracks"],
                }
                print(f"几何推定锚点：+{added} 赛道（无遥测）→ 覆盖 {len(stats)} 赛道")
        except Exception:  # noqa: BLE001 — 推定失败绝不能拖垮遥测主路径
            import traceback

            print("[warn] 几何推定锚点构建失败，本次只用遥测锚点：")
            traceback.print_exc()
    if len(anchor_sources) > len(stats):  # 理论上不会发生，防御性断言
        anchor_sources = {tid: anchor_sources.get(tid, "telemetry") for tid in stats}

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    optimal_delta_ms: dict[str, float] = {}
    default_norm = {spec.name: 0.0 for spec in specs}

    for tid in sorted(stats):
        track = stats[tid]
        # 锚点自检：默认调教增量应为 0
        anchor = simulate_lap_delta_ms(track, default_norm)
        assert abs(anchor) < 1e-6, f"{tid} 默认调教增量非零：{anchor}"
        # 采样（默认、两端极值、近似最优点，其余按分档密集/均匀混合）
        # 每条样本带 tier 标记：训练侧据此对「路径管」样本加权，让容量落在
        # 优化器真正做决策的区域（见 train_setup_sim_nn.py --tube-weight）。
        tagged: list[tuple[dict[str, float], str]] = [(default_norm, "seed_default")]
        tagged.append(({spec.name: 1.0 for spec in specs}, "seed_extreme"))
        tagged.append(({spec.name: -1.0 for spec in specs}, "seed_extreme"))
        optimum, opt_ms = coarse_optimum(track, specs)
        optimal_delta_ms[tid] = round(opt_ms, 3)
        if sampling == "legacy":
            # 旧行为：全量程均匀（复现历史结果 / 消融对照）。
            # 注意不追加最优点，以保持与旧版逐行一致。
            while len(tagged) < n_setups:
                raw = _sample_setup(rng, specs)
                tagged.append(({s.name: _norm(raw[s.name], s) for s in specs}, "wide"))
        else:
            tagged.append((dict(optimum), "seed_optimum"))
            # 路径管采样：先在「默认→最优」路径上取位置 t，再逐参数加抖动。
            for (half_width, _share, t_max), count in zip(
                _SAMPLE_TIERS, _tier_counts(max(0, n_setups - len(tagged)))
            ):
                for _ in range(count):
                    if half_width is None:
                        raw = _sample_setup(rng, specs)
                        tagged.append((
                            {s.name: _norm(raw[s.name], s) for s in specs}, "wide",
                        ))
                    else:
                        t = rng.uniform(0.0, t_max)
                        tagged.append(({
                            s.name: snap_to_step(
                                t * optimum[s.name]
                                + rng.uniform(-half_width, half_width), s,
                            )
                            for s in specs
                        }, "tube"))
        for sn, tier in tagged:
            delta_ms = simulate_lap_delta_ms(track, sn)
            row = {
                "track_id": tid,
                "setup_norm": {k: round(v, 5) for k, v in sn.items()},
                "lap_delta_ms": delta_ms,
                "tier": tier,
                "conditions": {
                    "tyre_softness": track["tyre_softness"],
                    "wet": track["wet"],
                    "track_temp": track["track_temp"],
                    "air_temp": track["air_temp"],
                },
            }
            rows.append(row)
        n_tube = sum(1 for _, tier in tagged if tier == "tube")
        mark = "" if track.get("anchor") == "telemetry" else "（几何推定）"
        print(f"  {tid:12s} 弯数 {len(track['corners']):2d} | 圈速锚点 "
              f"{track['lap_time_s']:.2f}s{mark} | 直道占比 {track['straight_share']:.2f} "
              f"| 采样 {len(tagged)}（路径管 {n_tube}）"
              f" | 近似最优 Δ={opt_ms:+7.1f} ms")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "schema": "f1opt-setup-sim/1",
        "source": "TracingInsights 2026 corner telemetry anchors + documented QSS physics",
        "tracks": sorted(stats),
        # 逐站锚点来源：telemetry = 真实遥测；authored = 赛道几何 + 全局常数推定。
        # 模型与报告按此如实展示，不把推定说成实测。
        "anchor_sources": {tid: anchor_sources.get(tid, "telemetry") for tid in sorted(stats)},
        "n_rows": len(rows),
        "n_setups_per_track": n_setups,
        "effects_version": "1.0",
        "authored_calibration": calibration_block,
        # 采样策略（可追溯）：调教优化需要精度的区域是最优点邻域，
        # 全量程均匀采样会把模型容量消耗在区分「糟糕/极糟糕」上。
        "sampling": {
            "strategy": sampling,
            "tiers": [
                {"jitter_half_width": hw, "share": share, "path_t_max": t_max}
                for hw, share, t_max in _SAMPLE_TIERS
            ],
            "n_per_track": n_setups,
            "optimal_delta_ms": optimal_delta_ms,
        },
        # 每赛道聚合特征 + 工况（训练与推理共用；推理侧按此建特征向量）
        "track_features": {
            tid: {
                "traction_index": t["traction_index"],
                "aero_index": t["aero_index"],
                "braking_index": t["braking_index"],
                "slow_share": t["slow_share"],
                "fast_share": t["fast_share"],
                "tyre_softness": t["tyre_softness"],
                "wet": t["wet"],
                "track_temp": t["track_temp"],
                "air_temp": t["air_temp"],
                "lap_time_s": t["lap_time_s"],
                "straight_share": t["straight_share"],
                "drag_scale": t["drag_scale"],
            }
            for tid, t in stats.items()
        },
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写出 {out}（{len(rows)} 行）")
    _print_wing_table(stats, anchor_sources)
    return meta


def _print_wing_table(
    stats: dict[str, dict[str, Any]], anchor_sources: dict[str, str],
) -> None:
    """打印逐赛道「最优翼角」方向校验表（含几何推定站）。

    这是几何推定锚点唯一有效的**方向性**自检：估算出来的圈速/直道占比若把
    Las Vegas 判成「该拉满下压力」或把 Monaco 判成「该减翼」，说明常数拟合
    已经跑偏，必须人工介入 —— 所以放进日志，每轮重训都能看见。
    """
    from setup_tuner.domain.setup import ALL_SETUP_FIELDS

    base = {f.name: 0.0 for f in ALL_SETUP_FIELDS}
    rows: list[tuple[str, str, int, float, float]] = []
    for tid, track in sorted(stats.items()):
        best_w, best_v = 0, None
        for w_clicks in range(-12, 13):
            n = dict(base)
            n["front_wing"] = w_clicks / 25.0
            n["rear_wing"] = w_clicks / 25.0
            v = simulate_lap_delta_ms(track, n)
            if best_v is None or v < best_v:
                best_w, best_v = w_clicks, v
        rows.append((
            tid, anchor_sources.get(tid, "telemetry"), best_w,
            float(track["drag_scale"]), float(track["lap_time_s"]),
        ))
    print("\n逐赛道最优翼角（方向自检：Monaco 应最想加翼、Monza/Spa/Las Vegas 应想减翼）：")
    for tid, src, w, ds, lap in sorted(rows, key=lambda r: -r[2]):
        tag = "遥测" if src == "telemetry" else "推定"
        print(f"  {tid:12s} [{tag}] 最优翼角 {w:+3d} 档 | drag_scale {ds:.3f} "
              f"| 圈速锚点 {lap:.2f}s")


def print_calibration() -> None:
    """打印标定表：常见调教改动的仿真圈速影响（供人工复核量级）。"""
    from setup_tuner.domain.setup import ALL_SETUP_FIELDS

    # 用一个中性赛道（弯型混合）演示
    demo_track = {
        "corners": [
            {"corner": 1, "klass": "slow", "t_corner": 4.5, "brake_pct": 55.0},
            {"corner": 2, "klass": "medium", "t_corner": 5.0, "brake_pct": 30.0},
            {"corner": 3, "klass": "fast", "t_corner": 4.0, "brake_pct": 12.0},
            {"corner": 4, "klass": "medium", "t_corner": 5.5, "brake_pct": 35.0},
            {"corner": 5, "klass": "fast", "t_corner": 4.2, "brake_pct": 10.0},
        ],
        "lap_time_s": 90.0,
        "straight_share": 0.40,
        "drag_scale": 0.40 * (200.0 / 200.0) ** 2 * 1.0,
    }
    print("\n标定表（中性混合赛道，默认调教为基准）：")

    def _delta(mods: dict[str, float]) -> float:
        n = {f.name: 0.0 for f in ALL_SETUP_FIELDS}
        for name, dn in mods.items():
            n[name] = dn
        return simulate_lap_delta_ms(demo_track, n)

    cases = [
        ("前后翼 +10 档（25→35）", {"front_wing": +0.4, "rear_wing": +0.4}),
        ("前后翼 -10 档（25→15）", {"front_wing": -0.4, "rear_wing": -0.4}),
        ("前后离地 -4 档（25→21 / 50→46）",
         {"front_ride_height": -0.4, "rear_ride_height": -0.4}),
        ("前外倾 -3.5→-3.0（减少负外倾）", {"front_camber": +0.5}),
        ("前后悬挂软化 4 档（6→2）",
         {"front_suspension": -0.125, "rear_suspension": -0.125}),
        ("前后防倾杆软化 4 档（6→2）",
         {"front_anti_roll_bar": -0.2, "rear_anti_roll_bar": -0.2}),
        ("油门差速锁止 +30（50→80）", {"on_throttle_diff": +0.6}),
        ("刹车压力 +5（90→95）", {"brake_pressure": +0.5}),
        ("胎压 +2 psi（四轮）",
         {"front_left_tyre_pressure": +0.28, "front_right_tyre_pressure": +0.28,
          "rear_left_tyre_pressure": +0.33, "rear_right_tyre_pressure": +0.33}),
    ]
    for label, mods in cases:
        ms = _delta(mods)
        print(f"  {label:36s} → {ms:+8.1f} ms/圈")

    # 逐赛道最优翼角（方向校验：Monaco 最想加翼、Monza/Spa 最想减翼）
    try:
        stats = load_corner_stats(None)
    except SystemExit:
        return
    if not stats:
        return
    print("\n逐赛道最优翼角（仿真内点最优；+ 表示比默认 25 档更想加翼）：")
    base = {f.name: 0.0 for f in ALL_SETUP_FIELDS}
    rows = []
    for tid, track in stats.items():
        best_w, best_v = 0.0, None
        for w_clicks in range(-12, 13, 1):
            n = dict(base)
            n["front_wing"] = w_clicks / 25.0
            n["rear_wing"] = w_clicks / 25.0
            v = simulate_lap_delta_ms(track, n)
            if best_v is None or v < best_v:
                best_w, best_v = w_clicks, v
        rows.append((tid, best_w, best_v, track["drag_scale"]))
    for tid, best_w, best_v, ds in sorted(rows, key=lambda r: r[1]):
        print(f"  {tid:12s} 最优翼角 = 默认 {best_w:+3d} 档 "
              f"（Δ={best_v:+7.1f} ms，drag_scale={ds:.3f}）")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="构建调教仿真数据集（遥测锚定）")
    ap.add_argument("--setups", type=int, default=400, help="每赛道采样调教数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tracks", default="", help="逗号分隔的赛道过滤（空=全部）")
    ap.add_argument("--out", default=str(OUT_JSONL))
    ap.add_argument(
        "--sampling", choices=("tube", "legacy"), default="tube",
        help="采样策略：tube=路径管密采样（默认，面向调优决策）；"
             "legacy=旧的全量程均匀（复现历史结果/消融对照）",
    )
    ap.add_argument(
        "--no-authored-anchors", dest="authored_anchors", action="store_false",
        help="只用遥测锚点（不给无遥测赛道补几何推定锚点；用于消融/回滚）",
    )
    ap.set_defaults(authored_anchors=True)
    ap.add_argument("--calibrate-only", action="store_true", help="只打印标定表")
    args = ap.parse_args(argv)

    if args.calibrate_only:
        print_calibration()
        return 0

    tracks = {t.strip() for t in args.tracks.split(",") if t.strip()} or None
    build(args.setups, args.seed, tracks, Path(args.out), args.sampling,
          authored_anchors=args.authored_anchors)
    print_calibration()
    return 0


if __name__ == "__main__":
    sys.exit(main())
