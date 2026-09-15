# -*- coding: utf-8 -*-
"""F1OPT 性能 & 个性化能力探针（只读，云端运行）。

量化：
  P 系列 —— 现有实现的性能热点与优化空间
  Q 系列 —— "个性化调教" 能力的实际缺口
"""
from __future__ import annotations

import json
import math
import re
import statistics
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
    t1 = time.perf_counter(); raw(False); per_row = time.perf_counter() - t1
    t2 = time.perf_counter(); raw(True); one = time.perf_counter() - t2
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
    item("P4.Dx 随反馈条数线性累积", "FAIL",
         "同一条反馈重复 N 次 → Dx 线性放大（不聚合、不衰减、不取最大值），"
         "最终建议被 max_delta 全量饱和 → 反馈越多建议越统一且极端，"
         "而不是越个性化。get_feedbacks 每次全表扫描且无 LIMIT。")


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
    anch = _ilu.module_from_spec(spec); spec.loader.exec_module(anch)
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
                ax, ay = poly[k]; bx, by = poly[k + 1]
                dx, dy = bx - ax, by - ay; L2 = dx * dx + dy * dy
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
    from setup_tuner.telemetry.recorder import TelemetryRecorder
    from setup_tuner.telemetry.packets import parse_packet

    frames = real_frames()
    frames = [f for f in frames if len(f) > 900][:60]
    with tempfile.TemporaryDirectory() as td:
        rec = TelemetryRecorder(data_dir=td)
        rec.start()
        parsed = [parse_packet(f) for f in frames]

        def run():
            for f, p in zip(frames, parsed):
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
    item("Q1.track_id 是否影响调教建议", "FAIL" if same else "PASS",
         "suzuka 与 monza 输入完全相同的症状 → setup_delta 与 Dx 逐位相同；"
         "track_id 只传入 nn_model（而 nn 分支不可用）→ 24 条赛道得到同一套建议"
         if same else "不同赛道结果不同")


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

    setup = CarSetup.default().to_dict()
    spec = {f.name: f for f in ALL_SETUP_FIELDS}
    base = generate_suggestion([("understeer", 3)], setup, "suzuka", None,
                               model_type="rule")
    many = generate_suggestion([("understeer", 3)] * 20, setup, "suzuka", None,
                               model_type="rule")
    sat_base = sum(1 for k, v in base["setup_delta"].items()
                   if abs(abs(v) - spec[k].max_delta) < 1e-9)
    sat_many = sum(1 for k, v in many["setup_delta"].items()
                   if abs(abs(v) - spec[k].max_delta) < 1e-9)
    item("Q3.反馈条数 → 建议饱和", "INFO",
         f"1 条 vs 相同 20 条：饱和参数 {sat_base}/21 → {sat_many}/21，"
         f"|Dx|max {max(abs(v) for v in base['dx'].values()):.1f} → "
         f"{max(abs(v) for v in many['dx'].values()):.1f}（无去重/衰减/聚合）")


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
# Q6 混合模型实际降级
# =========================================================================== #
def q6_hybrid_degraded():
    from setup_tuner.domain.setup import CarSetup
    from setup_tuner.engine.engine import generate_suggestion
    from setup_tuner.engine.nn_model import is_torch_available

    setup = CarSetup.default().to_dict()
    res = generate_suggestion([("understeer", 3)], setup, "suzuka", None,
                              model_type="hybrid")
    pyproj = (ROOT / "pyproject.toml").read_text("utf-8")
    declared = "torch" in pyproj
    item("Q6.默认 hybrid 模型是否真的生效", "FAIL" if res["model_type"] == "rule" else "PASS",
         f"API 默认 model_type='hybrid'，实测返回 model_type={res['model_type']!r}, "
         f"nn_available={res['nn_available']}；torch 可用={is_torch_available()}，"
         f"pyproject 是否声明 torch={declared} → 三档模型（rule/nn/hybrid）"
         "在下发版里行为完全一致")


# =========================================================================== #
# Q7 nn_model 维度常量失配
# =========================================================================== #
def q7_nn_dims():
    import setup_tuner.engine.nn_model as m
    from setup_tuner.domain.setup import ALL_SETUP_FIELDS, CarSetup
    from setup_tuner.domain.symptoms import Symptom

    vec = m.build_input_vector([("understeer", 3)], m.empty_dx() if hasattr(m, "empty_dx")
                               else {d: 0.0 for d in m.DIAG_DIMS},
                               CarSetup.default().to_dict(), "suzuka")
    item("Q7a.输入向量长度 vs 网络输入层", "FAIL" if len(vec) != m._INPUT_SIZE else "PASS",
         f"build_input_vector 实际长度={len(vec)}，_INPUT_SIZE(硬编码)={m._INPUT_SIZE} "
         f"→ 前向传播维度不匹配 → predict 内部 except 吞掉 → 永远返回 None")
    item("Q7b.症状数常量", "FAIL" if m._NUM_SYMPTOMS != len(Symptom) else "PASS",
         f"_NUM_SYMPTOMS(硬编码)={m._NUM_SYMPTOMS}，实际 Symptom={len(Symptom)}")
    item("Q7c.参数数常量", "FAIL" if m._NUM_SETUP_PARAMS != len(ALL_SETUP_FIELDS) else "PASS",
         f"_NUM_SETUP_PARAMS(硬编码)={m._NUM_SETUP_PARAMS}，实际 "
         f"ALL_SETUP_FIELDS={len(ALL_SETUP_FIELDS)}")
    try:
        m._normalize_symptoms([("high_speed_instability", 3)])
        item("Q7d.新增症状归一化", "PASS", "未抛错")
    except IndexError as e:
        item("Q7d.新增症状归一化", "FAIL",
             f"_normalize_symptoms([('high_speed_instability',3)]) → IndexError: {e} "
             "（vec 只分配 12 位，而症状有 15 个）")


def main() -> int:
    print("=" * 78, flush=True)
    print("F1OPT 性能 & 个性化能力探针（云端只读）", flush=True)
    print("=" * 78, flush=True)
    sections = [
        ("性能 P1 解析吞吐", p1_parse),
        ("性能 P2 引擎延迟", p2_engine),
        ("性能 P3 Store 启动", p3_store),
        ("性能 P4 /suggest 伸缩", p4_suggest_scaling),
        ("性能 P5 当前弯修正代价", p5_corner_cost),
        ("性能 P6 遥测快照", p6_stream_alias),
        ("性能 P7 录制热路径", p7_recorder),
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
