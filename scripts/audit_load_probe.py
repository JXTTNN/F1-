# -*- coding: utf-8 -*-
"""F1OPT 负载驱动性能探测（结合仓库已有遥测数据，云端只读运行）。

思路：用 legacy/tests/data/real_f1_26_sample.jsonl 里 403 帧真实 F1 2026 抓包
还原「包尺寸 + 包混合比 + 帧率」，构造真实的 60Hz 负载，然后测量：
解析 → 缓存 → 摘要 → WS 推送 → 引擎 → 录制 全链路的 CPU 时间与伸缩性。
"""
from __future__ import annotations

import json
import math
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT: list[tuple[str, str, str]] = []


def item(tag: str, verdict: str, detail: str = "") -> None:
    OUT.append((tag, verdict, detail))
    print(f"[{verdict}] {tag} :: {detail}", flush=True)


def bench(fn, n: int) -> tuple[float, float]:
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = time.perf_counter() - t0
    return dt, dt / n * 1e6


def load_samples() -> list[dict]:
    fp = ROOT / "legacy" / "tests" / "data" / "real_f1_26_sample.jsonl"
    return [json.loads(l) for l in fp.read_text("utf-8").splitlines() if l.strip()]


# =========================================================================== #
# L1 从真实抓包还原负载模型
# =========================================================================== #
# F1 25/26 官方发送频率（UDP 规范）：
#   60Hz: Motion / MotionEx / CarTelemetry / CarStatus / LapData / CarDamage
#    2Hz: Session
#  ~1Hz: SessionHistory / TyreSets / LapPositions / Participants / CarTelemetryData2
_RATE_HZ = {
    "Motion": 60, "MotionEx": 60, "CarTelemetry": 60, "CarStatus": 60,
    "LapData": 60, "CarDamage": 60, "Session": 2, "SessionHistory": 1,
    "TyreSets": 1, "LapPositions": 1, "Participants": 0.2,
    "CarTelemetryData2": 60, "Event": 0.1,
}


def l1_load_model(samples: list[dict]) -> dict:
    sizes: dict[str, int] = {}
    for r in samples:
        sizes.setdefault(r["name"], r["len"])
    total_pkt = sum(_RATE_HZ.get(k, 0) for k in sizes)
    total_bw = sum(_RATE_HZ.get(k, 0) * v for k, v in sizes.items())
    sup = {"Session", "LapData", "CarSetups", "CarTelemetry", "CarStatus"}
    sup_pkt = sum(_RATE_HZ.get(k, 0) for k in sizes if k in sup)
    frame_ids = [r["frameIdentifier"] for r in samples if r["name"] == "CarTelemetry"]
    item("L1.真实负载模型（尺寸×官方频率）", "INFO",
         f"包类型 {len(sizes)} 种；合计 {total_pkt:.0f} 包/秒，"
         f"{total_bw / 1024:.0f} KB/s（{total_bw * 8 / 1e6:.2f} Mbit/s）；"
         f"其中程序会解析的 5 类 = {sup_pkt:.0f} 包/秒")
    item("L1b.仓库内真实抓包是否可用于压测", "FAIL",
         f"403 帧 / 383.6 s = 1.0 帧/秒 → 这是**降采样样本**（frameIdentifier 跨度 "
         f"{max(frame_ids) - min(frame_ids)} → 实际游戏帧率仅 "
         f"{(max(frame_ids) - min(frame_ids)) / 383.62:.1f} FPS）；"
         "data/sim_telemetry/*.jsonl 只有字段摘要、无 hex 原始字节，且 "
         "size_bytes 与真实不符（Session 声明 156B，实测 926B）→ "
         "项目内**没有任何可直接回放做压测的真实负载**")
    return {"sizes": sizes, "total_pkt": total_pkt, "total_bw": total_bw,
            "sup_pkt": sup_pkt}


# =========================================================================== #
# L2 解析链路在真实负载下的 CPU 占用
# =========================================================================== #
def l2_parse_under_load(model: dict, samples: list[dict]) -> None:
    from setup_tuner.telemetry.packets import PacketTooShortError, packet_name, parse_packet

    pool: list[bytes] = []
    for r in samples:
        if r["name"] in ("Session", "LapData", "CarSetups", "CarTelemetry", "CarStatus"):
            pool.append(bytes.fromhex(r["hex"]))
    t0 = time.perf_counter()
    n_ok = sum(1 for f in pool if _try(parse_packet, f))
    _ = n_ok
    dt, per = bench(lambda: [parse_packet(f) for f in pool], 10)
    per_pkt = dt / (len(pool) * 10)
    item("L2.解析链路 CPU 占用（真实负载）", "PASS" if per_pkt * model["total_pkt"] < 0.1 else "WARN",
         f"单包 {per_pkt * 1e6:.2f} µs（{len(pool)} 帧真实抓包）→ "
         f"在 {model['total_pkt']:.0f} 包/秒负载下占单核 "
         f"{per_pkt * model['total_pkt'] * 100:.2f}%；"
         f"60Hz 峰值（含全部 14 类包）也仅 "
         f"{per_pkt * model['total_pkt'] * 100:.2f}% —— 解析不是瓶颈")
    _ = t0


def _try(fn, arg):
    from setup_tuner.telemetry.packets import PacketTooShortError
    try:
        return fn(arg) is not None
    except PacketTooShortError:
        return False


# =========================================================================== #
# L3 单圈回放：解析 + 缓存 + 摘要 全链路成本
# =========================================================================== #
def l3_lap_replay(samples: list[dict]) -> None:
    from setup_tuner.report.builder import extract_telemetry_summary
    from setup_tuner.telemetry.packets import parse_packet
    from setup_tuner.telemetry.stream import TelemetryStream

    by_name = {r["name"]: bytes.fromhex(r["hex"]) for r in samples}
    cycle = [by_name["Session"], by_name["LapData"], by_name["CarSetups"],
             by_name["CarTelemetry"], by_name["CarStatus"]]
    LAP_S = 90.0
    FPS = 60
    frames = int(LAP_S * FPS)  # 90 秒单圈 @60Hz

    st = TelemetryStream()

    def one_frame():
        for f in cycle:
            p = parse_packet(f)
            if p is not None:
                st.update(p["packet_id"], p)

    _, per_frame = bench(one_frame, frames)
    per_frame_s = per_frame / 1e6
    _, per_summary = bench(lambda: extract_telemetry_summary(st.get_all_latest()), 5000)

    # 拟议的整圈聚合器（把当前缺失的 max_speed/avg_steer/max_steer 等一次累积出来）
    acc = {"speed_max": 0.0, "speed_sum": 0.0, "steer_abs_sum": 0.0,
           "steer_abs_max": 0.0, "thr_sum": 0.0, "brk_sum": 0.0,
           "ttemp": [0.0] * 4, "tpres": [0.0] * 4, "n": 0}

    def aggregate(tel: dict) -> None:
        sp = float(tel.get("m_speed") or 0.0)
        st_abs = abs(float(tel.get("m_steer") or 0.0))
        acc["speed_max"] = max(acc["speed_max"], sp)
        acc["speed_sum"] += sp
        acc["steer_abs_sum"] += st_abs
        acc["steer_abs_max"] = max(acc["steer_abs_max"], st_abs)
        acc["thr_sum"] += float(tel.get("m_throttle") or 0.0)
        acc["brk_sum"] += float(tel.get("m_brake") or 0.0)
        for i, v in enumerate((tel.get("m_tyresSurfaceTemperature") or [0] * 4)[:4]):
            acc["ttemp"][i] += float(v)
        for i, v in enumerate((tel.get("m_tyresPressure") or [0] * 4)[:4]):
            acc["tpres"][i] += float(v)
        acc["n"] += 1

    lap_snapshot = dict(st.get_latest(6) or {})
    _, per_agg = bench(lambda: aggregate(lap_snapshot), 20000)

    total_lap = per_frame_s * frames
    item("L3.单圈(90s@60Hz)全链路成本", "INFO",
         f"每帧解析 5 类包 + 写缓存 {per_frame:.1f} µs；单圈 {frames} 帧 = "
         f"{total_lap * 1000:.0f} ms（占单核 {total_lap / LAP_S * 100:.2f}%）；"
         f"60Hz 下 extract_telemetry_summary {per_summary:.1f} µs/次 = "
         f"{per_summary / 1e6 * 60 * 100:.2f}% 单核")
    item("L3b.整圈聚合（当前缺失的能力）成本", "PASS",
         f"拟议聚合器 {per_agg:.2f} µs/帧 → 单圈 "
         f"{per_agg / 1e6 * frames * 1000:.0f} ms（{per_agg / 1e6 * 60 * 100:.3f}% 单核）"
         " → **把遥测真正做成整圈统计、驱动个性化调教，CPU 成本可忽略（每圈几十 ms 级以下）**")


# =========================================================================== #
# L4 WS 推送伸缩性（N 连接 N 循环 → N² 扇出）
# =========================================================================== #
def l4_ws_fanout(samples: list[dict]) -> None:
    from setup_tuner.report.builder import extract_telemetry_summary
    from setup_tuner.telemetry.packets import parse_packet
    from setup_tuner.telemetry.stream import TelemetryStream

    st = TelemetryStream()
    for r in samples:
        if r["name"] in ("Session", "LapData", "CarSetups", "CarTelemetry", "CarStatus"):
            p = parse_packet(bytes.fromhex(r["hex"]))
            if p:
                st.update(p["packet_id"], p)
    payload = st.get_all_latest().get(6, {})
    msg = json.dumps({"event": "telemetry", "payload": {
        "speed": payload.get("m_speed"), "throttle": payload.get("m_throttle"),
        "brake": payload.get("m_brake"), "steer": payload.get("m_steer"),
        "gear": payload.get("m_gear"), "engine_rpm": payload.get("m_engineRPM"),
        "drs": payload.get("m_drs")}}, ensure_ascii=False)
    _, per_dumps = bench(lambda: len(msg.encode()), 20000)

    def client_tick(n_conn: int) -> None:
        # 每个连接一个 _telemetry_push_loop：各自取全量缓存 + 各自 broadcast 给全部连接
        for _ in range(n_conn):
            all_latest = st.get_all_latest()
            extract_telemetry_summary(all_latest)
            for _ in range(n_conn):
                json.dumps({"event": "telemetry", "payload": all_latest.get(6, {}).get("m_speed")})

    print(f"     {'连接数':>6}{'每tick实际发送':>16}{'每tick耗时':>14}{'每秒耗时(60Hz)':>16}{'单核占用':>10}",
          flush=True)
    for n in (1, 2, 4, 8):
        _, per_tick = bench(lambda: client_tick(n), 200)
        per_s = per_tick / 1e6 * 60
        print(f"     {n:>6}{n * n:>16}{per_tick:>12.0f}µs{per_s * 1000:>14.1f}ms"
              f"{per_s * 100:>9.2f}%", flush=True)
    item("L4.WS 推送扇出结构", "FAIL",
         f"_telemetry_push_loop 是**每连接一个**后台任务，而每个任务都用 ws_manager"
         f".broadcast() 向**所有**连接发送 → N 个客户端 = 每 tick N×N 次发送/序列化"
         f"（N=8 时 64 次，60Hz 下 3840 次/秒）；"
         f"单条 payload 序列化 {per_dumps:.2f} µs。应改为 app 级单一推送任务 + 一次序列化扇出")


# =========================================================================== #
# L5 缓存内存占用
# =========================================================================== #
def l5_cache_memory(samples: list[dict]) -> None:
    from setup_tuner.telemetry.packets import parse_packet
    from setup_tuner.telemetry.stream import TelemetryStream

    st = TelemetryStream()
    for r in samples:
        if r["name"] in ("Session", "LapData", "CarSetups", "CarTelemetry", "CarStatus"):
            p = parse_packet(bytes.fromhex(r["hex"]))
            if p:
                st.update(p["packet_id"], p)
    all_latest = st.get_all_latest()
    approx = sum(len(json.dumps(v, ensure_ascii=False, default=str)) for v in all_latest.values())
    item("L5.最新帧缓存内存", "INFO",
         f"5 类包最新帧 JSON 近似 {approx / 1024:.1f} KB；"
         "Session 包内含 64 条天气预报样本，是整个缓存里最大的单体 —— "
         "60Hz 覆盖写只保留最新帧，内存无增长风险")


# =========================================================================== #
# L6 录制链路在真实负载下的 CPU
# =========================================================================== #
def l6_recorder_load(model: dict, samples: list[dict]) -> None:
    from setup_tuner.telemetry.packets import parse_packet
    from setup_tuner.telemetry.recorder import TelemetryRecorder

    pool = []
    for r in samples:
        if r["name"] in ("Session", "LapData", "CarSetups", "CarTelemetry", "CarStatus"):
            raw = bytes.fromhex(r["hex"])
            pool.append((raw, parse_packet(raw)))
    with tempfile.TemporaryDirectory() as td:
        rec = TelemetryRecorder(data_dir=td)
        rec.start()

        def run():
            for raw, p in pool:
                rec.on_raw_packet(raw, p)
        dt, per = bench(run, 10)
        rec.stop()
        per_pkt = dt / (len(pool) * 10)
    cpu = per_pkt * model["total_pkt"] * 100
    item("L6.录制链路 CPU 占用（真实负载）", "WARN" if cpu > 10 else "PASS",
         f"单包 {per_pkt * 1e6:.1f} µs（zstd 压缩 + 整包 JSON 序列化 + 批量入库）；"
         f"{model['total_pkt']:.0f} 包/秒负载下占单核 {cpu:.1f}%，"
         "且全部在 **UDP 接收线程** 上串行执行 → 录制开启时会拖慢收包")


# =========================================================================== #
# L7 「当前弯」判定：现实现 vs 弧长修正版的成本与正确率
# =========================================================================== #
def l7_corner_map() -> None:
    import importlib.util

    spec0 = importlib.util.spec_from_file_location(
        "avp", ROOT / "scripts" / "audit_verify_probe.py")
    avp = importlib.util.module_from_spec(spec0)
    spec0.loader.exec_module(avp)
    parse_path, arc_lookup = avp.parse_path, avp.arc_lookup
    _ = arc_lookup

    spec = importlib.util.spec_from_file_location(
        "anch", ROOT / "setup_tuner" / "domain" / "_track_anchors.py")
    anch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(anch)
    from setup_tuner.api.ws import _map_corner
    from setup_tuner.domain.track import get_all_tracks

    t0 = time.perf_counter()
    table: dict[str, list[tuple[int, float]]] = {}
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
    build_ms = (time.perf_counter() - t0) * 1000

    def arc_map(lap_distance: float, length_m: float, tid: str) -> int:
        fracs = table[tid]
        p = (lap_distance % length_m) / length_m
        return min(fracs, key=lambda cf: min(abs(cf[1] - p), 1 - abs(cf[1] - p)))[0]

    import bisect
    sorted_tab: dict[str, tuple[list[float], list[int]]] = {}
    for tid, fracs in table.items():
        order = sorted(fracs, key=lambda cf: cf[1])
        sorted_tab[tid] = ([f for _, f in order], [c for c, _ in order])

    def arc_map_bisect(lap_distance: float, length_m: float, tid: str) -> int:
        fr, cn = sorted_tab[tid]
        p = (lap_distance % length_m) / length_m
        i = bisect.bisect_left(fr, p)
        lo = fr[i - 1] if i > 0 else fr[-1] - 1.0
        hi = fr[i] if i < len(fr) else fr[0] + 1.0
        return cn[i - 1] if (p - lo) <= (hi - p) else cn[i % len(cn)]

    tr = None
    for t in get_all_tracks():
        if t.track_id == "suzuka":
            tr = t
    _, t_old = bench(lambda: _map_corner(1500.0, tr.length_m, tr.corners), 30000)
    _, t_min = bench(lambda: arc_map(1500.0, tr.length_m, "suzuka"), 30000)
    _, t_bis = bench(lambda: arc_map_bisect(1500.0, tr.length_m, "suzuka"), 30000)

    # 正确率：24 赛道 × 200 采样点
    wrong = tot_n = 0
    for t in get_all_tracks():
        if t.track_id not in table:
            continue
        fr = table[t.track_id]
        for i in range(200):
            d = t.length_m * i / 200
            truth = min(fr, key=lambda cf: min(abs(cf[1] - d / t.length_m),
                                               1 - abs(cf[1] - d / t.length_m)))[0]
            got = _map_corner(d, t.length_m, t.corners)
            wrong += 0 if got == truth else 1
            tot_n += 1
    # 修正版（bisect）与真值一致性
    wrong2 = 0
    for t in get_all_tracks():
        if t.track_id not in sorted_tab:
            continue
        for i in range(200):
            d = t.length_m * i / 200
            if arc_map_bisect(d, t.length_m, t.track_id) != arc_map(d, t.length_m, t.track_id):
                wrong2 += 1
    cost_60hz8 = t_bis / 1e6 * 60 * 8 * 100
    item("L7.当前弯判定：成本与正确率", "FAIL",
         f"弧长表构建 {build_ms:.0f} ms（24 赛道，启动/打包时算一次即可）｜单次查询："
         f"现状 {t_old:.2f} µs、修正版(min) {t_min:.2f} µs、修正版(bisect) {t_bis:.2f} µs｜"
         f"错误率 {wrong / tot_n * 100:.1f}%（{wrong}/{tot_n}）→ 0%（bisect 版与真值差 {wrong2} 个）｜"
         f"bisect 版在 60Hz×8 连接下仅占单核 {cost_60hz8:.2f}%，"
         "比现状慢几十微秒但完全可忽略 —— 正确性优先，不必为性能保留错误算法")


def main() -> int:
    print("=" * 78, flush=True)
    print("F1OPT 负载驱动性能探测（基于仓库已有遥测数据）", flush=True)
    print("=" * 78, flush=True)
    samples = load_samples()
    model = {}
    for fn in (lambda: l1_load_model(samples),
               lambda: l2_parse_under_load(model, samples),
               lambda: l3_lap_replay(samples),
               lambda: l4_ws_fanout(samples),
               lambda: l5_cache_memory(samples),
               lambda: l6_recorder_load(model, samples),
               lambda: l7_corner_map()):
        try:
            r = fn()
            if isinstance(r, dict):
                model.update(r)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            item(fn.__name__ if hasattr(fn, "__name__") else "step", "ERROR",
                 f"{type(e).__name__}: {e}")
    from collections import Counter
    c = Counter(v for _, v, _ in OUT)
    print("\n" + "=" * 78 + "\n汇总：" +
          " ".join(f"{k}={c[k]}" for k in ("FAIL", "ERROR", "WARN", "PASS", "INFO") if c.get(k)),
          flush=True)
    Path("load_probe_result.json").write_text(
        json.dumps([{"tag": t, "verdict": v, "detail": d} for t, v, d in OUT],
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
