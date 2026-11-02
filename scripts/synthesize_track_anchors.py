"""为「无真实遥测锚点」的赛道合成几何迁移锚点，保证 24/24 覆盖。

问题
----
``scripts/build_setup_sim_dataset.py`` 只遍历 ``corner_dataset.jsonl`` 里
**实际有遥测**的赛道。2026 赛季后半段（Austin / Mexico / São Paulo /
Las Vegas / Lusail / Yas Marina）以及 Jeddah 在上游尚无可用逐点数据，
结果这些赛道整条被跳过 → 训练出的模型 ``track_ids`` 只有 17 条。

后果不是"少几条"，而是**静默降级**：``sim_optimizer.sim_refine`` 对未覆盖
赛道直接返回 ``available=False``，用户在面板上点「生成建议」得到的是纯规则
结果，报告 ``model_type`` 显示 ``rule`` —— 训练好的模型对 7 条赛道完全无效。

做法
----
对缺失赛道，用**几何迁移**合成一份等价结构的锚点：

1. 在已覆盖赛道中选一条最相似的参照赛道（确定性打分：赛道长度、弯数、
   弯型分布 slow/medium/fast、赛道类型 street/high_downforce/...）；
2. 逐弯映射：目标第 i 弯 → 参照第 round(i·N_ref/N_t) 弯；
3. 时间结构迁移：参考赛道"该弯时间占整圈比例" × 目标赛道估算圈速
   （圈速按赛道长度比例缩放，再用弯速比修正）；
4. 速度按两赛道同弯 domain 标称速度之比缩放；
5. 赛道级特征（slow_share / fast_share）由目标赛道自身弯道统计，
   牵引/气动/制动指数继承参照赛道；
6. 每份合成锚点标记 ``anchor_source="geometry_transfer"`` 与
   ``transferred_from``，在数据集 meta 与模型内**如实可查**，绝不伪装成
   真实遥测。

诚实边界
--------
这是工程近近似，不是真实遥测。合成赛道上的模型预测可信度低于遥测赛道，
因此报告侧应显式标注来源（见 ``anchor_sources``）。它的价值是消除
「7 条赛道完全没有模型」这一**断崖式降级**，而不是宣称精度等同。

用法
----
    from synthesize_track_anchors import fill_missing_tracks
    stats, sources = fill_missing_tracks(stats)

作为独立模块存在的原因：``build_setup_sim_dataset.py`` 已有 28 KB，
把迁移逻辑塞进去会让本已偏大的脚本更难维护；独立模块也便于单测。
"""

from __future__ import annotations

from typing import Any

#: 锚点来源：真实遥测
ANCHOR_TELEMETRY = "telemetry"
#: 锚点来源：几何迁移（合成）
ANCHOR_TRANSFER = "geometry_transfer"

#: 目标/参照速度比的合理区间（防止 domain 标称速度异常放大）
_RATIO_CLAMP = (0.55, 2.2)
#: 单弯时间占比的合理区间（防止映射到极弯时出现离谱值）
_SHARE_CLAMP = (0.002, 0.25)
#: 赛道类型 → 经验均速（km/h）。用于估算**圈速量级**，只依赖目标赛道自身属性，
#: 不受参照池影响。
#:
#: 为什么不直接用「参照赛道圈速 × 长度比」：参照池里可能只有 Monza（全赛季
#: 最快之一，均速 254 km/h），按比例缩放会把 Jeddah 算成 72s、Monaco 51s
#: （实际 ~88s / ~72s），偏差 20-30%，进而污染 lap_time_s 与 drag_scale。
#: 这些值按 2026 赛历各赛道公开中位圈速反推的均速量级设定，允许 ±10% 误差。
_V_TYPE_KMH: dict[str, float] = {
    "street": 158.0,
    "high_downforce": 180.0,
    "mixed": 208.0,
    "medium": 205.0,
    "high_speed_low_downforce": 238.0,
}
#: 校准因子合理区间（参照赛道真实圈速 / 经验估算）
_CALIB_CLAMP = (0.85, 1.15)


def _baseline_lap(length_m: float, track_type: str) -> float:
    """按赛道类型的经验均速估算圈时（秒）。"""
    v = _V_TYPE_KMH.get(str(track_type), 205.0)
    return float(length_m) * 3.6 / v


def _corner_class_share(corners: list[Any]) -> dict[str, float]:
    """弯道类型分布（slow / medium / fast 占比）。"""
    total = len(corners) or 1
    acc: dict[str, float] = {"slow": 0.0, "medium": 0.0, "fast": 0.0}
    for c in corners:
        key = str(getattr(c, "corner_type", "medium"))
        acc[key] = acc.get(key, 0.0) + 1.0
    return {k: v / total for k, v in acc.items()}


def _similarity(target: Any, ref: Any) -> float:
    """相似度打分（越小越像）：长度 / 弯数 / 弯型分布 / 赛道类型。"""
    d_len = abs(ref.length_m - target.length_m) / max(1.0, float(target.length_m))
    d_cnt = abs(len(ref.corners) - len(target.corners)) / max(
        1.0, float(len(target.corners)),
    )
    t_share = _corner_class_share(target.corners)
    r_share = _corner_class_share(ref.corners)
    d_share = sum(abs(r_share[k] - t_share[k]) for k in ("slow", "medium", "fast"))
    d_type = 0.0 if ref.track_type == target.track_type else 0.35
    # 权重：长度与弯数最能决定圈时结构，弯型分布决定气动/机械权重
    return 1.2 * d_len + 0.8 * d_cnt + 0.9 * d_share + d_type


def _pick_reference(target: Any, stats: dict[str, dict[str, Any]]) -> tuple[str, Any] | None:
    """在**已覆盖**赛道里选最相似的参照赛道（同分时按 track_id 字典序，保证确定性）。"""
    from setup_tuner.domain.track import get_track_by_id

    best: tuple[float, str, Any] | None = None
    for tid in sorted(stats):
        if tid == target.track_id:
            continue
        ref = get_track_by_id(tid)
        if ref is None or not ref.corners:
            continue
        score = _similarity(target, ref)
        if best is None or score < best[0]:
            best = (score, tid, ref)
    if best is None:
        return None
    return best[1], best[2]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def synthesize_track(
    track_id: str, stats: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """为单个缺失赛道合成锚点；无法合成时返回 ``None``。

    Args:
        track_id: 目标赛道 id（必须在 ``ALL_TRACKS`` 中）。
        stats: 已覆盖赛道的锚点表（``load_corner_stats`` 的产物）。

    Returns:
        与 ``load_corner_stats`` 同结构的锚点字典，额外带
        ``anchor_source`` / ``transferred_from`` / ``transfer_score``。
    """
    from setup_tuner.domain.track import get_track_by_id

    target = get_track_by_id(track_id)
    if target is None or not target.corners:
        return None
    picked = _pick_reference(target, stats)
    if picked is None:
        return None
    ref_id, ref_track = picked
    ref = stats[ref_id]
    ref_corners: list[dict[str, Any]] = ref.get("corners") or []
    if not ref_corners:
        return None

    n_t, n_r = len(target.corners), len(ref_track.corners)
    ref_lap = float(ref.get("lap_time_s") or 0.0) or 90.0
    ref_len = float(ref_track.length_m) or 1.0

    # ① 逐弯速度比（domain 标称速度，两侧同源，比值可靠）
    ratios: list[float] = []
    pairs: list[tuple[int, int]] = []
    for i in range(1, n_t + 1):
        j = int(round(i * n_r / n_t))
        j = max(1, min(n_r, j))
        pairs.append((i, j))
        v_t = float(target.corners[i - 1].speed_kmh) or 1.0
        v_r = float(ref_track.corners[j - 1].speed_kmh) or 1.0
        ratios.append(_clamp(v_t / v_r, *_RATIO_CLAMP))
    # ② 目标圈速：类型经验均速估算 + 参照赛道的偏差校准。
    # 不直接用「参照圈速 × 长度比」：参照池可能只有 Monza（均速 254 km/h），
    # 比例缩放会把所有赛道拉快 20-30%。这里改为
    #   圈速 = 目标赛道经验圈时 × (参照真实圈速 / 参照经验圈时)
    # 前半段只依赖目标赛道自身长度与类型，后半段只用来校准经验表的系统偏差。
    ref_base = _baseline_lap(ref_len, getattr(ref_track, "track_type", "medium"))
    calib = _clamp(ref_lap / ref_base, *_CALIB_CLAMP) if ref_base > 0 else 1.0
    lap_t = _baseline_lap(float(target.length_m), target.track_type) * calib

    # ③ 逐弯迁移：时间结构按"参照弯时占圈比例"，速度按弯速比
    corners: list[dict[str, Any]] = []
    for (i, j), ratio in zip(pairs, ratios, strict=True):
        src = ref_corners[j - 1]
        share = _clamp(
            float(src.get("t_corner") or 0.0) / ref_lap, *_SHARE_CLAMP,
        )
        klass = str(getattr(target.corners[i - 1], "corner_type", "medium")).lower()
        corners.append({
            "corner": i,
            "klass": klass if klass in ("slow", "medium", "fast") else "medium",
            "v_entry": round(float(src.get("v_entry") or 0.0) * ratio, 2),
            "v_min": round(float(src.get("v_min") or 0.0) * ratio, 2),
            "v_exit": round(float(src.get("v_exit") or 0.0) * ratio, 2),
            "t_corner": round(share * lap_t, 4),
            "brake_pct": round(float(src.get("brake_pct") or 0.0), 2),
            "lat_g": round(float(src.get("lat_g") or 0.0), 3),
            # 真实样本数记 0：合成锚点不是遥测，不能冒充
            "n": 0,
        })

    # ④ 赛道级特征：分布由目标自身弯道统计，动力学指数继承参照
    share = _corner_class_share(target.corners)
    straight_share = float(ref.get("straight_share") or 0.3)
    avg_speed_kmh = float(target.length_m) / lap_t * 3.6 if lap_t > 0 else 0.0
    drag_scale = (
        straight_share * (avg_speed_kmh / 200.0) ** 2 * (lap_t / 90.0)
        if avg_speed_kmh > 0.0 else 0.0
    )
    return {
        "corners": corners,
        "lap_time_s": round(lap_t, 3),
        "straight_share": straight_share,
        "avg_speed_kmh": round(avg_speed_kmh, 2),
        "drag_scale": drag_scale,
        "traction_index": float(ref.get("traction_index") or 0.0),
        "aero_index": float(ref.get("aero_index") or 0.0),
        "braking_index": float(ref.get("braking_index") or 0.0),
        "slow_share": round(share["slow"], 4),
        "fast_share": round(share["fast"], 4),
        "tyre_softness": float(ref.get("tyre_softness") or 0.6),
        "wet": float(ref.get("wet") or 0.0),
        "track_temp": float(ref.get("track_temp") or 35.0),
        "air_temp": float(ref.get("air_temp") or 24.0),
        "n_samples": 0,
        # ---- 溯源标记（绝不隐藏合成事实）----
        "anchor_source": ANCHOR_TRANSFER,
        "transferred_from": ref_id,
        "transfer_score": round(_similarity(target, ref_track), 4),
    }


def fill_missing_tracks(
    stats: dict[str, dict[str, Any]],
    tracks_filter: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """补齐缺失赛道，返回 ``(补全后的 stats, {track_id: anchor_source})``。

    Args:
        stats: 真实遥测锚点表（**不会被就地修改**）。
        tracks_filter: 只处理指定赛道（``None`` = 全部 24 条）。

    Returns:
        补全后的新字典；来源表覆盖所有 24 条赛道，值取自
        :data:`ANCHOR_TELEMETRY` / :data:`ANCHOR_TRANSFER`。
    """
    from setup_tuner.domain.track import ALL_TRACKS

    out: dict[str, dict[str, Any]] = dict(stats)
    sources: dict[str, str] = {}
    for tid in out:
        out[tid] = dict(out[tid])
        out[tid].setdefault("anchor_source", ANCHOR_TELEMETRY)
        out[tid].setdefault("transferred_from", None)
        sources[tid] = ANCHOR_TELEMETRY

    for track in ALL_TRACKS:
        tid = track.track_id
        if tracks_filter and tid not in tracks_filter:
            continue
        if tid in out:
            continue
        # 参照池固定为**真实遥测**赛道：若用 out，后续合成赛道会参照前面的
        # 合成结果，误差逐级放大（链式漂移）。
        synth = synthesize_track(tid, stats)
        if synth is None:
            continue
        out[tid] = synth
        sources[tid] = ANCHOR_TRANSFER

    return out, sources
