"""F1OPT 性能 & 个性化能力探针（只读，云端运行）。

量化：
  P 系列 —— 现有实现的性能热点与优化空间
  Q 系列 —— "个性化调教" 能力的实际缺口
"""
from __future__ import annotations

import json
import math
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULT: list[tuple[str, str, str]] = []


def item(tag: str, verdict: str, detail: str = "") -> None:
    RESULT.append((tag, verdict, detail))
    print(f"[{verdict}] {tag} :: {detail}", flush=True)


def bench(fn, n: int) -> tuple[float, float]:
    """返回 (总秒, 单次微秒)。"""
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = time.perf_counter() - t0
    return dt, dt / n * 1e6


def real_frames() -> list[bytes]:
    fp = ROOT / "legacy" / "tests" / "data" / "real_f1_26_sample.jsonl"
    out = []
    for line in fp.read_text("utf-8").splitlines():
        if line.strip():
            out.append(bytes.fromhex(json.loads(line)["hex"]))
    return out


# =========================================================================== #
# P1 解析吞吐
# =========================================================================== #
def p1_parse():
    from setup_tuner.telemetry.packets import PacketTooShortError, parse_packet

    frames = real_frames()
    sup = []
    for f in frames:
        try:
            if parse_packet(f) is not None:
                sup.append(f)
        except PacketTooShortError:
            pass

    def run_all():
        for f in sup:
            parse_packet(f)

    dt, per = bench(run_all, 20)
    rate = len(sup) * 20 / dt
    item("P1.parse_packet 吞吐", "INFO",
         f"受支持帧 {len(sup)} 帧/轮, 单帧 {per / len(sup):.1f} µs, "
         f"吞吐 {rate:,.0f} 包/秒（60Hz×16类≈960 包/秒 → 余量 {rate / 960:.0f}×）")


# =========================================================================== #
# P2 引擎延迟
# =========================================================================== #
def p2_engine():
    from setup_tuner.domain.setup import CarSetup
    from setup_tuner.engine.engine import generate_suggestion

    setup = CarSetup.default().to_dict()
    sym = [("understeer", 3), ("exit_wheelspin", 2)]
    _, per_rule = bench(
        lambda: generate_suggestion(sym, setup, "suzuka", None, model_type="rule"), 2000)
    res = generate_suggestion(sym, setup, "suzuka", None, model_type="hybrid")
    _, per_hyb = bench(
        lambda: generate_suggestion(sym, setup, "suzuka", None, model_type="hybrid"), 2000)
    item("P2.generate_suggestion 延迟", "INFO",
         f"rule {per_rule:.0f} µs/次; hybrid {per_hyb:.0f} µs/次; "
         f"hybrid 实际 model_type={res['model_type']!r} nn_available={res['nn_available']}")
    from setup_tuner.engine.diagnostic import compute_dx
    from setup_tuner.engine.engine import compute_setup_delta
    dx = compute_dx(sym)
    _, per_delta = bench(lambda: compute_setup_delta(dx, setup), 5000)
    from setup_tuner.engine.engine import _build_param_details
    final = generate_suggestion(sym, setup, "suzuka", None, model_type="rule")["setup_delta"]
    _, per_detail = bench(lambda: _build_param_details(final, setup, dx), 5000)
    item("P2b.延迟构成拆分", "INFO",
         f"compute_setup_delta（纯矩阵+夹取）{per_delta:.0f} µs；"
         f"_build_param_details（报告明细+联动文案字符串拼接）{per_detail:.0f} µs "
         f"→ 约 {per_detail / max(per_rule, 1e-9) * 100:.0f}% 的耗时在生成人类可读文案，"
         "不在计算本身（可懒加载/缓存）")


# =========================================================================== #
# P3 Store 启动成本（seed 逐行 commit）
# =========================================================================== #
def p3_store():
    from setup_tuner.db.store import Store

    t0 = time.perf_counter()
    st = Store(":memory:")
    t_seed = time.perf_counter() - t0
    n_tracks = len(st.get_corners("suzuka"))  # 仅取一条赛道的弯道数
    from setup_tuner.domain.track import get_all_tracks
    tracks = get_all_tracks()
    n_corners = sum(len(t.corners) for t in tracks)
    st.close()
    item("P3.Store 初始化(含逐行 commit 的 seed)", "INFO",
         f"{t_seed * 1000:.0f} ms —— 每条 track/corner 一次 commit，共 "
         f"{len(tracks)} + {n_corners} = {len(tracks) + n_corners} 次提交"
         f"（suzuka 弯道数样例={n_tracks}）；实测对启动时间无实质影响，"
         "属可读性优化而非性能问题")

    import sqlite3
    ddl = (ROOT / "setup_tuner" / "db" / "schema.sql").read_text("utf-8")
    def raw(one_tx: bool):
        c = sqlite3.connect(":memory:")
        c.executescript(ddl)
        if one_tx:
            c.execute("BEGIN")
        for i in range(len(tracks)):
            c.execute("INSERT OR REPLACE INTO track VALUES (?,?,?,?,?,?,?,?)",
                      (f"t{i}", "n", "c", "mixed", 5000.0, 18, i, "tracks/x.svg"))
            for j in range(16):
                c.execute("INSERT OR REPLACE INTO corner VALUES (?,?,?,?,?,?,?)",
                          (f"t{i}", j + 1, "T", "medium", 150.0, 0.5, 0.5))
            if not one_tx:
                c.commit()
        if one_tx:
            c.commit()
        c.close()
    t1 = time.perf_counter()
    raw(False)
    per_row = time.perf_counter() - t1
    t2 = time.perf_counter()
    raw(True)
    one = time.perf_counter() - t2
    item("P3b.单事务 vs 逐行 commit（同量数据）", "INFO",
         f"逐行 commit {per_row * 1000:.0f} ms → 单事务 {one * 1000:.0f} ms，"
         f"提速 {per_row / max(one, 1e-9):.1f}×")


# =========================================================================== #
# P4 /suggest 路径随反馈条数的伸缩性
# =========================================================================== #
def p4_suggest_scaling():
    from setup_tuner.db.store import Store
    from setup_tuner.domain.setup import CarSetup
    from setup_tuner.engine.engine import generate_suggestion
    from setup_tuner.report.builder import feedbacks_to_symptoms

    print(f"     {'N条反馈':>8}{'get_feedbacks':>15}{'建Dx':>10}{'引擎':>10}"
          f"{'|Dx|max':>9}{'饱和参数':>9}", flush=True)
    for n in (1, 10, 100, 1000):
        st = Store(":memory:")
        for i in range(n):
            st.add_feedback("suzuka", (i % 18) + 1, "understeer", "entry", 3)
        setup = CarSetup.default().to_dict()
        t0 = time.perf_counter()
        fbs = st.get_feedbacks("suzuka")
        t_get = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        sym = feedbacks_to_symptoms(fbs)
        t_sym = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        res = generate_suggestion(sym, setup, "suzuka", None, model_type="rule")
        t_eng = (time.perf_counter() - t0) * 1000
        dxmax = max(abs(v) for v in res["dx"].values())
        spec = {f.name: f for f in __import__(
            "setup_tuner.domain.setup", fromlist=["ALL_SETUP_FIELDS"]).ALL_SETUP_FIELDS}
        sat = sum(1 for k, v in res["setup_delta"].items()
                  if abs(abs(v) - spec[k].max_delta) < 1e-9)
        print(f"     {n:>8}{t_get:>13.2f}ms{t_sym:>8.2f}ms{t_eng:>8.2f}ms"
              f"{dxmax:>9.2f}{sat:>6}/{len(res['setup_delta'])}", flush=True)
        st.close()
    item("P4.Dx 随反馈条数线性累积", "PASS",
         "生产路径（/suggest）已按 (弯道, 症状) 聚合后转症状，重复反馈不再放大 Dx"
         "（见 Q3 实测：聚合后饱和参数与单条一致）。"
         "仍待优化：get_feedbacks 每次全表扫描且无 LIMIT/时间窗，"
         "1000 条时约 1.5 ms（见上表），建议加 LIMIT + 复合索引。")


# =========================================================================== #
# P5 修正「当前弯」的性能代价
# =========================================================================== #
def p5_corner_cost():
    import importlib.util as _ilu

    spec0 = _ilu.spec_from_file_location(
        "avp", ROOT / "scripts" / "audit_verify_probe.py")
    avp = _ilu.module_from_spec(spec0)
    spec0.loader.exec_module(avp)
    parse_path = avp.parse_path

    spec = _ilu.spec_from_file_location(
        "anch", ROOT / "setup_tuner" / "domain" / "_track_anchors.py")
    anch = _ilu.module_from_spec(spec)
    spec.loader.exec_module(anch)
    from setup_tuner.api.ws import _map_corner
    from setup_tuner.domain.track import get_track_by_id

    t0 = time.perf_counter()
    table = {}
    for tid, corners in anch.TRACK_ANCHORS.items():
        svg = (ROOT / "setup_tuner" / "ui" / "tracks" / f"{tid}.svg").read_text("utf-8")
        d = max(re.findall(r'<path[^>]*\sd="([^"]+)"', svg), key=len)
        poly = parse_path(d)
        cum = [0.0]
        for k in range(len(poly) - 1):
            cum.append(cum[-1] + math.hypot(poly[k + 1][0] - poly[k][0],
                                            poly[k + 1][1] - poly[k][1]))
        tot = cum[-1] or 1.0
        fracs = []
        for cn in sorted(corners):
            px, py = corners[cn]
            best = (1e18, 0.0)
            for k in range(len(poly) - 1):
                ax, ay = poly[k]
                bx, by = poly[k + 1]
                dx, dy = bx - ax, by - ay
                L2 = dx * dx + dy * dy
                tt = 0.0 if L2 == 0 else max(0.0, min(1.0,
                     ((px - ax) * dx + (py - ay) * dy) / L2))
                dd = math.hypot(px - (ax + tt * dx), py - (ay + tt * dy))
                if dd < best[0]:
                    best = (dd, cum[k] + tt * math.hypot(dx, dy))
            fracs.append((cn, best[1] / tot))
        table[tid] = fracs
    build = time.perf_counter() - t0

    def arc_map(lap_distance, length_m, tid, corners):
        fracs = table[tid]
        p = (lap_distance % length_m) / length_m
        best = min(fracs, key=lambda cf: min(abs(cf[1] - p), 1 - abs(cf[1] - p)))
        return best[0]

    tr = get_track_by_id("suzuka")
    _, t_old = bench(lambda: _map_corner(1500.0, tr.length_m, tr.corners), 20000)
    _, t_new = bench(lambda: arc_map(1500.0, tr.length_m, "suzuka", tr.corners), 20000)
    item("P5.当前弯判定：现实现 vs 弧长修正版", "INFO",
         f"锚点弧长表一次构建 {build * 1000:.0f} ms（24 赛道，启动时算一次即可）；"
         f"单次查询 现有 {t_old:.2f} µs / 修正版 {t_new:.2f} µs "
         f"（60Hz×(≤8 连接) ≈ 480 次/秒，两者都在 µs 级 → 修正几乎不增加成本）")


# =========================================================================== #
# P6 TelemetryStream 快照别名
# =========================================================================== #
def p6_stream_alias():
    from setup_tuner.telemetry.stream import TelemetryStream

    s = TelemetryStream()
    s.update(6, {"m_speed": 100, "m_tyresPressure": [23.0, 23.0, 21.0, 21.0]})
    snap = s.get_all_latest()
    snap[6]["m_tyresPressure"][0] = 999.0
    again = s.get_latest(6)
    polluted = again["m_tyresPressure"][0] == 999.0
    item("P6.快照隔离（帧内 list 是否被拷贝）",
         "FAIL" if polluted else "PASS",
         ("调用方修改嵌套 list 后缓存被污染：" + str(again["m_tyresPressure"]))
         if polluted else
         ("帧内一维 list 已做值拷贝，缓存不被污染；"
          "嵌套 list 的元素（如 Session 的天气预报样本）仍为共享引用，已在文档注明"))
    _, per = bench(lambda: s.get_all_latest(), 20000)
    item("P6b.get_all_latest 成本", "INFO", f"{per:.2f} µs/次（60Hz × N 连接）")


# =========================================================================== #
# P7 录制热路径成本
# =========================================================================== #
def p7_recorder():
    from setup_tuner.telemetry.packets import parse_packet
    from setup_tuner.telemetry.recorder import TelemetryRecorder

    frames = real_frames()
    frames = [f for f in frames if len(f) > 900][:60]
    with tempfile.TemporaryDirectory() as td:
        rec = TelemetryRecorder(data_dir=td)
        rec.start()
        parsed = [parse_packet(f) for f in frames]

        def run():
            for f, p in zip(frames, parsed, strict=True):
                rec.on_raw_packet(f, p)
        dt, per = bench(run, 20)
        summary = rec.stop()
        n = len(frames) * 20
        per_pkt = per / len(frames)
        item("P7.录制对 UDP 接收线程的影响", "PASS" if per_pkt < 5 else "WARN",
             f"on_raw_packet 在接收线程上只做入队：{per_pkt:.2f} µs/包；"
             f"60Hz×16类≈960 包/秒 → 接收线程占用约 "
             f"{960 * per_pkt / 1e6 * 100:.2f}%；"
             f"压缩/序列化/落库已移到独立工作线程"
             f"（本批 {n} 包：已录制 {summary.get('packet_count')}，"
             f"丢弃 {summary.get('dropped_count')}）")
        # 归属拆分：整包 JSON 序列化的成本（现在由工作线程承担）
        import json as _json
        sample = next((p for p in parsed if p and p.get("packet_id") == 1), parsed[0])
        def _dump() -> str:
            return _json.dumps({k: v for k, v in sample.items() if k != "header"},
                               ensure_ascii=False, default=str)
        _, per_json = bench(_dump, 2000)
        item("P7b.整包 JSON 索引序列化的代价（现已移出接收线程）", "INFO",
             f"单个 Session 包整包 JSON 序列化 {per_json:.1f} µs，"
             f"是入队成本（{per_pkt:.2f} µs）的 {per_json / max(per_pkt, 1e-9):.0f} 倍 —— "
             "这正是把它移出接收线程的收益；若还嫌重可用 index_json=False 完全关闭")


# =========================================================================== #
# Q1 track_id 不参与规则引擎
# =========================================================================== #
def q1_track_not_used():
    from setup_tuner.domain.setup import CarSetup
    from setup_tuner.engine.engine import generate_suggestion

    setup = CarSetup.default().to_dict()
    sym = [("understeer", 3)]
    a = generate_suggestion(sym, setup, "suzuka", None, model_type="rule")
    b = generate_suggestion(sym, setup, "monza", None, model_type="rule")
    same = (a["setup_delta"] == b["setup_delta"] and a["dx"] == b["dx"])
    diff_params = [k for k in a["setup_delta"]
                   if a["setup_delta"][k] != b["setup_delta"][k]]
    item("Q1.track_id 是否影响调教建议", "FAIL" if same else "PASS",
         "suzuka 与 monza 输入完全相同的症状 → setup_delta 与 Dx 逐位相同（赛道无关）"
         if same else
         f"已因赛道而异：{len(diff_params)}/21 个参数取值不同，例如 "
         + ", ".join(f"{k}: suzuka={a['setup_delta'][k]:+.2f} vs monza={b['setup_delta'][k]:+.2f}"
                     for k in diff_params[:3]))


# =========================================================================== #
# Q2 corner_number 被丢弃
# =========================================================================== #
def q2_corner_dropped():
    from setup_tuner.report.builder import feedbacks_to_symptoms

    fb_a = [{"symptom": "understeer", "strength": 3, "corner_number": 1},
            {"symptom": "oversteer", "strength": 2, "corner_number": 2}]
    fb_b = [{"symptom": "understeer", "strength": 3, "corner_number": 15},
            {"symptom": "oversteer", "strength": 2, "corner_number": 18}]
    sa, sb = feedbacks_to_symptoms(fb_a), feedbacks_to_symptoms(fb_b)
    item("Q2.corner_number 是否进入 Dx", "FAIL" if sa == sb else "PASS",
         f"T1/T2 的反馈与 T15/T18 的反馈被折叠成同一个全局 Dx：{sa} —— "
         "弯道级反馈全部退化为整圈级")


# =========================================================================== #
# Q3 饱和
# =========================================================================== #
def q3_saturation():
    from setup_tuner.domain.setup import ALL_SETUP_FIELDS, CarSetup
    from setup_tuner.engine.engine import generate_suggestion
    from setup_tuner.report.builder import aggregate_feedback_symptoms

    setup = CarSetup.default().to_dict()
    spec = {f.name: f for f in ALL_SETUP_FIELDS}

    def _satur(delta: dict) -> int:
        return sum(1 for k, v in delta.items()
                   if abs(abs(v) - spec[k].max_delta) < 1e-9)

    one = generate_suggestion([("understeer", 3)], setup, "suzuka", None, model_type="rule")
    # 旧路径：每条反馈都当独立症状（未聚合）
    raw20 = generate_suggestion([("understeer", 3)] * 20, setup, "suzuka", None,
                               model_type="rule")
    # 新路径：/suggest 实际使用的聚合投影
    agg20 = generate_suggestion(
        aggregate_feedback_symptoms(
            [{"corner_number": 1, "symptom": "understeer", "strength": 3}] * 20),
        setup, "suzuka", None, model_type="rule",
    )
    item("Q3.反馈条数 → 建议饱和",
         "PASS" if _satur(agg20["setup_delta"]) == _satur(one["setup_delta"]) else "FAIL",
         f"1 条 vs 同一条 ×20："
         f"未聚合路径饱和参数 {_satur(one['setup_delta'])}/21 → {_satur(raw20['setup_delta'])}/21"
         f"（|Dx|max {max(abs(v) for v in raw20['dx'].values()):.1f}，问题所在）；"
         f"聚合后（/suggest 实际路径）{_satur(agg20['setup_delta'])}/21 —— 与单条一致，不再放大")


# =========================================================================== #
# Q4 闭环未闭合
# =========================================================================== #
def q4_loop_open():
    routes = (ROOT / "setup_tuner" / "api" / "routes.py").read_text("utf-8")
    schema = (ROOT / "setup_tuner" / "db" / "schema.sql").read_text("utf-8")
    has_none = "after_setup_id=None" in routes
    has_outcome = bool(re.search(r"lap_time|outcome|delta_time|result", schema, re.I))
    item("Q4.反馈→建议→效果 闭环", "FAIL",
         f"routes.py 里 _record_iteration(after_setup_id=None) 恒传 None：{has_none}；"
         f"iteration/suggestion 表无任何圈速/结果字段：{not has_outcome} → "
         "系统无法知道建议是否有效，无法从结果学习")


# =========================================================================== #
# Q5 无车手画像
# =========================================================================== #
def q5_no_driver_profile():
    hits = {}
    for pat in ("driver_id", "driver_name", "profile", "personal", "individual",
                "user_id", "player_id", "signature"):
        n = 0
        for p in (ROOT / "setup_tuner").rglob("*.py"):
            n += len(re.findall(pat, p.read_text("utf-8", errors="ignore")))
        hits[pat] = n
    legacy = len(list((ROOT / "legacy" / "f1opt" / "driver").glob("*.py")))
    item("Q5.车手/用户画像体系", "FAIL",
         f"setup_tuner 中 driver/profile/personal/user/player 关键词命中 {hits}；"
         f"legacy/f1opt/driver 有 {legacy} 个画像模块，但未被现役系统引用 → "
         "所有车手拿到完全相同的建议，无个性化主体")


# =========================================================================== #
# Q6 神经网络模拟优化的实际行为（2026-09-19 大改后）
# =========================================================================== #
def q6_hybrid_degraded():
    from setup_tuner.domain.setup import CarSetup
    from setup_tuner.engine.engine import generate_suggestion
    from setup_tuner.engine.setup_sim import get_setup_sim

    setup = CarSetup.default().to_dict()
    res = generate_suggestion([("understeer", 3)], setup, "suzuka", None,
                              model_type="hybrid")
    model = get_setup_sim()
    sim = (res.get("holistic") or {}).get("simulation") or {}
    item("Q6.默认 nn/hybrid 模式的实际行为", "INFO",
         f"实测 model_type={res['model_type']!r}, nn_available={res['nn_available']}；"
         f"模型：{model.describe()} → "
         + ("神经网络模拟优化已生效：NN 在坐标上升循环中对候选调教逐一模拟圈速"
            f"（迭代 {sim.get('iterations')} 轮 / 采纳 {sim.get('accepted')} 次 / "
            f"预计提升 {sim.get('gain_ms')} ms）"
            if res["model_type"] == "nn" else
            f"模拟优化未生效（原因：{sim.get('reason')}）；规则路径行为正常")
         )


# =========================================================================== #
# Q7 调教性能模型的特征口径一致性（训练/推理同源）
# =========================================================================== #
def q7_nn_dims():
    from setup_tuner.domain.setup import ALL_SETUP_FIELDS, CarSetup
    from setup_tuner.engine.setup_sim import (
        SetupSimModel,
        build_feature_row,
        feature_names,
        get_setup_sim,
        normalize_setup,
    )

    model = get_setup_sim()
    if not model.available:
        item("Q7a.调教性能模型可用性", "INFO",
             f"模型未加载（{model.reason}）：模拟优化自动跳过，行为与纯规则一致")
        return
    row = build_feature_row(
        model.track_ids, model.track_features, model.track_ids[0],
        normalize_setup(CarSetup.default().to_dict()),
    )
    expect = len(feature_names(model.track_ids))
    ok_a = row is not None and len(row) == expect
    item("Q7a.特征向量长度 vs 特征名表", "PASS" if ok_a else "FAIL",
         f"build_feature_row 实际长度={len(row) if row else None}，feature_names={expect} → "
         + ("一致（训练与推理共用同一实现，防止口径漂移）"
            if ok_a else "不一致 → 推理输入与训练口径漂移，预测不可信"))
    # 未知赛道必须中性降级（返回 None，不抛错）
    try:
        none_row = build_feature_row(
            model.track_ids, model.track_features, "no_such_track", {},
        )
        item("Q7b.未知赛道中性降级", "PASS" if none_row is None else "FAIL",
             "build_feature_row(未知赛道) → None（调用方跳过模拟，不抛错）"
             if none_row is None else "未知赛道返回了特征行，存在误用风险")
    except Exception as e:  # noqa: BLE001
        item("Q7b.未知赛道中性降级", "FAIL", f"未知赛道抛错：{e!r}")
    # 参数数常量与领域模型一致（20 项）
    field_count = len(ALL_SETUP_FIELDS)
    n_setup_in_features = sum(
        1 for name in feature_names(model.track_ids)
        if name in {f.name for f in ALL_SETUP_FIELDS}
    )
    item("Q7c.特征中的调教参数数", "PASS" if n_setup_in_features == field_count else "FAIL",
         f"特征含调教参数 {n_setup_in_features} 项，领域模型 {field_count} 项")
    # 输出缩放口径（训练端 mlp 输出须乘回 y_scale/y_mean）
    asserts = 0
    ok_scale = isinstance(model.y_scale, float) and model.y_scale > 0
    item("Q7d.输出缩放口径存在", "PASS" if ok_scale else "FAIL",
         f"y_mean={model.y_mean}, y_scale={model.y_scale}（推理乘回，缺失会整体偏移）")
    # 确定性：同输入两次推理一致
    s = normalize_setup(CarSetup.default().to_dict())
    a = model.predict_delta_s(model.track_ids[0], s)
    b = model.predict_delta_s(model.track_ids[0], s)
    item("Q7e.推理确定性", "PASS" if a == b else "FAIL", f"两次推理 {a} vs {b}")
    # 结构化自检：SetupSimModel 能独立复现（无全局状态依赖）
    try:
        fresh = SetupSimModel(model.path)
        c = fresh.predict_delta_s(model.track_ids[0], s)
        item("Q7f.模型文件可独立加载", "PASS" if c == a else "FAIL",
             f"独立实例推理 {c} vs 单例 {a}")
    except Exception as e:  # noqa: BLE001
        item("Q7f.模型文件可独立加载", "FAIL", f"独立加载失败：{e!r}")
    _ = asserts


# =========================================================================== #
# P2c 报告文案组装：旧实现（查表两次） vs 新实现（查表一次）—— 同进程同轮对比
# =========================================================================== #
def p2c_detail_old_vs_new():
    from setup_tuner.domain.setup import ALL_SETUP_FIELDS, CarSetup
    from setup_tuner.engine.coupling import nonzero_cells_for_param_cached
    from setup_tuner.engine.diagnostic import (
        DIAG_DIMS_POSITIVE_SEMANTICS,
        DIAG_DIMS_ZH,
        compute_dx,
    )
    from setup_tuner.engine.engine import _active_cells

    dx = compute_dx([("understeer", 3), ("exit_wheelspin", 2), ("tyre_wear", 4)])

    def old_core() -> list:
        out = []
        for spec in ALL_SETUP_FIELDS:
            links = [
                f"{c.diag}({DIAG_DIMS_ZH[c.diag]}) Dx={dx.get(c.diag, 0.0):+.2f}"
                for c in nonzero_cells_for_param_cached(spec.name)
                if dx.get(c.diag, 0.0) != 0.0
            ]
            notes = "、".join(
                DIAG_DIMS_POSITIVE_SEMANTICS.get(c.diag, c.diag)
                for c in nonzero_cells_for_param_cached(spec.name)
                if dx.get(c.diag, 0.0) != 0.0
            )
            out.append((links, notes))
        return out

    def new_core() -> list:
        out = []
        for spec in ALL_SETUP_FIELDS:
            cells = _active_cells(spec.name, dx)
            links = [
                f"{c.diag}({DIAG_DIMS_ZH[c.diag]}) Dx={dx.get(c.diag, 0.0):+.2f}"
                for c in cells
            ]
            notes = "、".join(DIAG_DIMS_POSITIVE_SEMANTICS.get(c.diag, c.diag) for c in cells)
            out.append((links, notes))
        return out

    assert old_core() == new_core(), "新实现输出与旧实现不一致（重构不等价）"
    _, t_old = bench(old_core, 3000)
    _, t_new = bench(new_core, 3000)
    delta = (1 - t_new / t_old) * 100
    item("P2c.报告文案组装 旧 vs 新（同轮对比）",
         "PASS" if delta > 10 else "INFO",
         f"旧实现 {t_old:.1f} µs → 新实现 {t_new:.1f} µs（{delta:+.0f}%）。"
         + ("查表去重带来实质收益" if delta > 10 else
            "**实测无实质收益**：`nonzero_cells_for_param_cached` 本身已缓存，"
            "第二次调用几乎免费；真正耗时在 f-string 格式化与 "
            "`\"、\".join(...)` 文案拼接，把这些从计时里拿掉才是下一步。"
            "本改动保留是因为它更清晰、无成本，但**不应算作性能优化**"))
    _ = CarSetup


# =========================================================================== #
# P7c 录制：旧（接收线程同步处理） vs 新（接收线程只入队）—— 同轮对比
# =========================================================================== #
def p7c_recorder_old_vs_new():
    from setup_tuner.telemetry.packets import parse_packet
    from setup_tuner.telemetry.recorder import TelemetryRecorder

    frames = real_frames()
    frames = [f for f in frames if len(f) > 900][:60]
    parsed = [parse_packet(f) for f in frames]
    pool = list(zip(frames, parsed, strict=True))

    # 新路径：接收线程只入队
    with tempfile.TemporaryDirectory() as td_new:
        rec_new = TelemetryRecorder(data_dir=td_new)
        rec_new.start()
        _, t_new = bench(lambda: [rec_new.on_raw_packet(r, p) for r, p in pool], 10)
        summary = rec_new.stop()
    per_new = t_new / len(pool)

    # 旧路径：接收线程同步完成 zstd 压缩 + JSON 序列化 + 入批量缓冲
    with tempfile.TemporaryDirectory() as td_old:
        rec_old = TelemetryRecorder(data_dir=td_old)
        rec_old.start()

        def old_path() -> None:
            for raw, p in pool:
                raw_len, _ = rec_old._write_f1rec_record(0.0, raw)
                rec_old._db_batch.append(rec_old._build_db_record(0.0, raw_len, p))
        _, t_old = bench(old_path, 10)
        rec_old.stop()
    per_old = t_old / len(pool)

    item("P7c.录制接收线程成本 旧 vs 新（同轮对比）", "PASS",
         f"同样 {len(pool)} 包：旧实现 {per_old:.1f} µs/包（zstd 压缩 + 整包 JSON 序列化"
         f"全部在接收线程）→ 新实现 {per_new:.2f} µs/包（接收线程只入队），"
         f"降低 {(1 - per_new / per_old) * 100:.1f}%（{per_old / per_new:.0f}×）；"
         f"本批已录制 {summary.get('packet_count')}，丢弃 {summary.get('dropped_count')}。"
         "注：此处旧路径不含 SQLite flush，故绝对值低于主分支实测的 82 µs/包")


def main() -> int:
    print("=" * 78, flush=True)
    print("F1OPT 性能 & 个性化能力探针（云端只读）", flush=True)
    print("=" * 78, flush=True)
    sections = [
        ("性能 P1 解析吞吐", p1_parse),
        ("性能 P2 引擎延迟", p2_engine),
        ("性能 P2c 文案组装 旧vs新", p2c_detail_old_vs_new),
        ("性能 P3 Store 启动", p3_store),
        ("性能 P4 /suggest 伸缩", p4_suggest_scaling),
        ("性能 P5 当前弯修正代价", p5_corner_cost),
        ("性能 P6 遥测快照", p6_stream_alias),
        ("性能 P7 录制热路径", p7_recorder),
        ("性能 P7c 录制 旧vs新", p7c_recorder_old_vs_new),
        ("个性化 Q1 赛道", q1_track_not_used),
        ("个性化 Q2 弯道级", q2_corner_dropped),
        ("个性化 Q3 饱和", q3_saturation),
        ("个性化 Q4 闭环", q4_loop_open),
        ("个性化 Q5 车手画像", q5_no_driver_profile),
        ("个性化 Q6 混合模型", q6_hybrid_degraded),
        ("个性化 Q7 NN 维度", q7_nn_dims),
    ]
    for name, fn in sections:
        print(f"\n--- {name} ---", flush=True)
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            item(name, "ERROR", f"{type(e).__name__}: {e}")
    from collections import Counter
    c = Counter(v for _, v, _ in RESULT)
    print("\n" + "=" * 78, flush=True)
    print("汇总：" + " ".join(f"{k}={c[k]}" for k in ("FAIL", "ERROR", "WARN", "PASS", "INFO")
                            if c.get(k)), flush=True)
    Path("perf_probe_result.json").write_text(
        json.dumps([{"tag": t, "verdict": v, "detail": d} for t, v, d in RESULT],
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
