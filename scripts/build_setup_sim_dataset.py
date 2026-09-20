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

用法::

    python scripts/build_setup_sim_dataset.py                 # 默认 400 调教/赛道
    python scripts/build_setup_sim_dataset.py --setups 800
    python scripts/build_setup_sim_dataset.py --tracks monza,suzuka
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


def build(
    n_setups: int, seed: int, tracks_filter: set[str] | None, out: Path,
) -> dict[str, Any]:
    from setup_tuner.domain.setup import ALL_SETUP_FIELDS

    specs = list(ALL_SETUP_FIELDS)
    stats = load_corner_stats(tracks_filter)
    if not stats:
        raise SystemExit("没有可用的赛道锚点数据")
    print(f"遥测锚点：{len(stats)} 赛道 | 总逐弯样本 "
          f"{sum(t['n_samples'] for t in stats.values())}")

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    default_norm = {spec.name: 0.0 for spec in specs}

    for tid in sorted(stats):
        track = stats[tid]
        # 锚点自检：默认调教增量应为 0
        anchor = simulate_lap_delta_ms(track, default_norm)
        assert abs(anchor) < 1e-6, f"{tid} 默认调教增量非零：{anchor}"
        # 采样（包含默认与两端极值）
        setups = [default_norm]
        setups.append({spec.name: 1.0 for spec in specs})
        setups.append({spec.name: -1.0 for spec in specs})
        while len(setups) < n_setups:
            raw = _sample_setup(rng, specs)
            setups.append({spec.name: _norm(raw[spec.name], spec) for spec in specs})
        for sn in setups:
            delta_ms = simulate_lap_delta_ms(track, sn)
            row = {
                "track_id": tid,
                "setup_norm": {k: round(v, 5) for k, v in sn.items()},
                "lap_delta_ms": delta_ms,
                "conditions": {
                    "tyre_softness": track["tyre_softness"],
                    "wet": track["wet"],
                    "track_temp": track["track_temp"],
                    "air_temp": track["air_temp"],
                },
            }
            rows.append(row)
        print(f"  {tid:12s} 弯数 {len(track['corners']):2d} | 圈速锚点 "
              f"{track['lap_time_s']:.2f}s | 直道占比 {track['straight_share']:.2f} "
              f"| 采样 {len(setups)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "schema": "f1opt-setup-sim/1",
        "source": "TracingInsights 2026 corner telemetry anchors + documented QSS physics",
        "tracks": sorted(stats),
        "n_rows": len(rows),
        "n_setups_per_track": n_setups,
        "effects_version": "1.0",
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
    return meta


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
    ap.add_argument("--calibrate-only", action="store_true", help="只打印标定表")
    args = ap.parse_args(argv)

    if args.calibrate_only:
        print_calibration()
        return 0

    tracks = {t.strip() for t in args.tracks.split(",") if t.strip()} or None
    build(args.setups, args.seed, tracks, Path(args.out))
    print_calibration()
    return 0


if __name__ == "__main__":
    sys.exit(main())
