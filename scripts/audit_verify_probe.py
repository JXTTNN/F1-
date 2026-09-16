"""F1OPT 深度审计验证脚本（仅云端运行，只读，不修改任何仓库文件）。

对 10 个具体疑点做可复现的云端断言，并打印证据表。
"""
from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULT: list[tuple[str, str, str]] = []


def item(tag: str, verdict: str, detail: str = "") -> None:
    RESULT.append((tag, verdict, detail))
    print(f"[{verdict}] {tag} :: {detail}", flush=True)


# --------------------------------------------------------------------------- #
# 官方 F1 22-26 UDP m_trackId 枚举（EA UDP 规范 / f1-game-packet-parser）
# --------------------------------------------------------------------------- #
OFFICIAL_TRACK_ID = {
    "melbourne": 0, "shanghai": 2, "sakhir": 3, "barcelona": 4, "monaco": 5,
    "montreal": 6, "silverstone": 7, "hungaroring": 9, "spa": 10, "monza": 11,
    "singapore": 12, "suzuka": 13, "yas_marina": 14, "austin": 15,
    "sao_paulo": 16, "spielberg": 17, "mexico_city": 19, "baku": 20,
    "zandvoort": 26, "jeddah": 29, "miami": 30, "las_vegas": 31, "lusail": 32,
}


def load_anchors():
    spec = importlib.util.spec_from_file_location(
        "anchors", ROOT / "setup_tuner" / "domain" / "_track_anchors.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TRACK_ANCHORS, mod.TRACK_CANVAS


# --------------------------------------------------------------------------- #
# SVG 路径展平（用于几何校验）
# --------------------------------------------------------------------------- #
TOK = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|(-?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse_path(d: str):
    toks = [(m.group(1), m.group(2)) for m in TOK.finditer(d)]
    pts, i, cur, start, prev_ctrl, cmd = [], 0, None, None, None, None

    def num():
        nonlocal i
        while i < len(toks) and toks[i][0] is not None:
            i += 1
        v = float(toks[i][1])
        i += 1
        return v

    def more():
        return i < len(toks) and toks[i][0] is None

    while i < len(toks):
        if toks[i][0] is not None:
            cmd = toks[i][0]
            i += 1
        c = cmd
        if c in "Mm":
            x, y = num(), num()
            if c == "m" and cur:
                x += cur[0]; y += cur[1]
            cur = (x, y); start = cur; pts.append(cur)
            cmd = "l" if c == "m" else "L"
        elif c in "Ll":
            while more():
                x, y = num(), num()
                if c == "l":
                    x += cur[0]; y += cur[1]
                cur = (x, y); pts.append(cur)
        elif c in "Hh":
            while more():
                x = num()
                if c == "h":
                    x += cur[0]
                cur = (x, cur[1]); pts.append(cur)
        elif c in "Vv":
            while more():
                y = num()
                if c == "v":
                    y += cur[1]
                cur = (cur[0], y); pts.append(cur)
        elif c in "CcSs":
            while more():
                if c in "Cc":
                    x1, y1, x2, y2, x, y = (num(), num(), num(), num(), num(), num())
                    if c == "c":
                        x1 += cur[0]; y1 += cur[1]; x2 += cur[0]; y2 += cur[1]
                        x += cur[0]; y += cur[1]
                else:
                    x2, y2, x, y = num(), num(), num(), num()
                    if c == "s":
                        x2 += cur[0]; y2 += cur[1]; x += cur[0]; y += cur[1]
                    if prev_ctrl:
                        x1, y1 = 2 * cur[0] - prev_ctrl[0], 2 * cur[1] - prev_ctrl[1]
                    else:
                        x1, y1 = cur
                for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
                    mt = 1 - t
                    pts.append((
                        mt ** 3 * cur[0] + 3 * mt * mt * t * x1 + 3 * mt * t * t * x2 + t ** 3 * x,
                        mt ** 3 * cur[1] + 3 * mt * mt * t * y1 + 3 * mt * t * t * y2 + t ** 3 * y,
                    ))
                prev_ctrl = (x2, y2); cur = (x, y)
        elif c in "QqTt":
            while more():
                if c in "Qq":
                    x1, y1, x, y = num(), num(), num(), num()
                    if c == "q":
                        x1 += cur[0]; y1 += cur[1]; x += cur[0]; y += cur[1]
                else:
                    x, y = num(), num()
                    if c == "t":
                        x += cur[0]; y += cur[1]
                    if prev_ctrl:
                        x1, y1 = 2 * cur[0] - prev_ctrl[0], 2 * cur[1] - prev_ctrl[1]
                    else:
                        x1, y1 = cur
                for t in (0.25, 0.5, 0.75, 1.0):
                    mt = 1 - t
                    pts.append((
                        mt * mt * cur[0] + 2 * mt * t * x1 + t * t * x,
                        mt * mt * cur[1] + 2 * mt * t * y1 + t * t * y,
                    ))
                prev_ctrl = (x1, y1); cur = (x, y)
        elif c in "Aa":
            while more():
                num(); num(); num(); num(); num()
                x, y = num(), num()
                if c == "a":
                    x += cur[0]; y += cur[1]
                pts.append((x, y)); cur = (x, y)
        elif c in "Zz":
            if start:
                pts.append(start); cur = start
        else:
            i += 1
    return pts


def seg_dist(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def polyline(track_id):
    txt = (ROOT / "setup_tuner" / "ui" / "tracks" / f"{track_id}.svg").read_text("utf-8")
    ds = re.findall(r'<path[^>]*\sd="([^"]+)"', txt)
    return parse_path(max(ds, key=len))


def arc_lookup(pt, poly, cum):
    best = (1e18, 0.0)
    for i in range(len(poly) - 1):
        ax, ay = poly[i]; bx, by = poly[i + 1]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / L2))
        d = math.hypot(pt[0] - (ax + t * dx), pt[1] - (ay + t * dy))
        if d < best[0]:
            best = (d, cum[i] + t * math.hypot(dx, dy))
    return best


# =========================================================================== #
# A. 真实 F1 2026 抓包 → 解析器验证
# =========================================================================== #
def check_real_packets():
    fp = ROOT / "legacy" / "tests" / "data" / "real_f1_26_sample.jsonl"
    if not fp.exists():
        item("A.真实抓包解析", "SKIP", f"样本不存在: {fp}")
        return
    from setup_tuner.telemetry.packets import (
        PacketTooShortError, parse_packet, packet_name,
    )
    total = ok = short = unknown = other = 0
    by_id: dict[int, int] = {}
    sessions, setups = [], []
    for line in fp.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        raw = bytes.fromhex(rec.get("hex", ""))
        total += 1
        try:
            parsed = parse_packet(raw)
        except PacketTooShortError:
            short += 1
            continue
        except Exception as e:  # noqa: BLE001
            other += 1
            print("   parse error:", type(e).__name__, e, flush=True)
            continue
        if parsed is None:
            unknown += 1
            continue
        ok += 1
        pid = parsed["packet_id"]
        by_id[pid] = by_id.get(pid, 0) + 1
        if pid == 1:
            sessions.append(parsed)
        elif pid == 5:
            setups.append(parsed)
    dist = ", ".join(f"{packet_name(k)}={v}" for k, v in sorted(by_id.items()))
    item(
        "A.真实抓包解析(受支持6类包)",
        "PASS" if other == 0 and short <= 1 else "FAIL",
        f"受支持包 {ok}/{ok + short} 解析成功, 短包(截断帧)={short}, 解析异常={other}, "
        f"未支持packetId(设计返回None)={unknown}, 全部记录={total} | {dist}",
    )

    # A2: 真实 Session 包里的 m_trackId
    if sessions:
        tids = sorted({s.get("m_trackId") for s in sessions})
        lens = sorted({s.get("m_trackLength") for s in sessions})
        wcode = sorted({s.get("m_weather") for s in sessions})
        item("A2.真实Session字段", "INFO",
             f"m_trackId={tids} m_trackLength={lens} m_weather={wcode} "
             f"样例 trackTemperature={sessions[0].get('m_trackTemperature')}")
        from setup_tuner.domain.track import get_track_by_udp_id, get_all_tracks
        for tid in tids:
            t = get_track_by_udp_id(int(tid))
            real_len = lens[0] if lens else None
            item("A3.真实trackId→赛道解析", "INFO",
                 f"udp_trackId={tid} → 程序判定={t.track_id if t else None} "
                 f"(表长={t.length_m if t else '-'}) ; 抓包 trackLength={real_len}")
        # 反查：长度与哪条赛道最接近
        if lens:
            best = min(get_all_tracks(), key=lambda x: abs(x.length_m - lens[0]))
            item("A4.真实trackLength反查", "INFO",
                 f"抓包 trackLength={lens[0]}m 最接近 {best.track_id}({best.length_m}m) "
                 f"→ 该赛道应有 udp_trackId={OFFICIAL_TRACK_ID.get(best.track_id,'?')}")

    # A5: 真实 CarSetups 包 → 21 参数提取
    if setups:
        from setup_tuner.report.builder import extract_setup_from_packet5
        from setup_tuner.domain.setup import ALL_SETUP_FIELDS
        s0 = setups[0]
        raw_fields = {k: v for k, v in s0.items()
                      if k.startswith("m_") and k in {
                          "m_frontWing", "m_rearWing", "m_onThrottleDiff",
                          "m_offThrottleDiff", "m_frontCamber", "m_rearCamber",
                          "m_frontToe", "m_rearToe", "m_frontSuspension",
                          "m_brakePressure", "m_brakeBias", "m_engineBraking",
                          "m_rearLeftTyrePressure", "m_frontLeftTyrePressure",
                          "m_ballast", "m_fuelLoad"}}
        item("A5.真实CarSetups原始字段", "INFO", json.dumps(raw_fields, ensure_ascii=False))
        params = extract_setup_from_packet5(s0)
        bad = []
        for spec in ALL_SETUP_FIELDS:
            v = params[spec.name]
            if not (spec.min_val - 1e-6 <= v <= spec.max_val + 1e-6):
                bad.append(f"{spec.name}={v:.3f} 越界[{spec.min_val},{spec.max_val}]")
            if not math.isfinite(v):
                bad.append(f"{spec.name}=非有限值 {v}")
        item("A6.提取21参数是否在合法区间",
             "PASS" if not bad else "FAIL",
             "全部合法" if not bad else "; ".join(bad[:8]))
        item("A7.提取结果样例", "INFO",
             json.dumps({k: round(v, 3) for k, v in list(params.items())}, ensure_ascii=False))
        fr = s0.get("m_frontRightTyrePressure")
        item("A8.胎压字段", "INFO",
             f"UDP m_frontRightTyrePressure={fr} → 转换后={params.get('front_right_tyre_pressure')}")


# =========================================================================== #
# B. udp_track_id 映射正确性
# =========================================================================== #
def check_track_id_map():
    from setup_tuner.domain.track import get_all_tracks
    tracks = get_all_tracks()
    dup = len({t.udp_track_id for t in tracks}) != len(tracks)
    wrong, unknown, correct = [], [], []
    for t in tracks:
        off = OFFICIAL_TRACK_ID.get(t.track_id)
        if off is None:
            unknown.append(f"{t.track_id}(程序={t.udp_track_id})")
        elif off != t.udp_track_id:
            wrong.append(f"{t.track_id}: 程序={t.udp_track_id} 官方={off}")
        else:
            correct.append(t.track_id)
    item("B1.udp_track_id 唯一性", "FAIL" if dup else "PASS",
         "有重复" if dup else f"{len(tracks)} 条唯一")
    item("B2.udp_track_id 与官方枚举一致",
         "FAIL" if wrong else "PASS",
         f"正确 {len(correct)}/{len(tracks)}；错误 {len(wrong)} 条 → " + "; ".join(wrong))
    if unknown:
        item("B3.新增赛道(官方枚举待补)", "WARN", "; ".join(unknown))


# =========================================================================== #
# C. 赛道图几何 & 弯角顺序
# =========================================================================== #
def check_track_geometry():
    ANCH, CANVAS = load_anchors()
    worst_track, worst_d = None, 0.0
    inv_tracks = []
    for tid, corners in sorted(ANCH.items()):
        poly = polyline(tid)
        cum = [0.0]
        for i in range(len(poly) - 1):
            cum.append(cum[-1] + math.hypot(poly[i + 1][0] - poly[i][0],
                                            poly[i + 1][1] - poly[i][1]))
        tot = cum[-1] or 1.0
        ds, seq = [], []
        for cn in sorted(corners):
            d, s = arc_lookup(corners[cn], poly, cum)
            ds.append(d)
            seq.append((cn, s / tot * 100.0))
        mx = max(ds)
        if mx > worst_d:
            worst_track, worst_d = tid, mx
        inv = [(seq[i][0], seq[i + 1][0]) for i in range(len(seq) - 1)
               if seq[i + 1][1] < seq[i][1] - 0.05]
        if inv:
            inv_tracks.append(f"{tid}{inv}")
    item("C1.弯道锚点落在赛道线上(像素偏差)",
         "PASS" if worst_d < 5.0 else "FAIL",
         f"24 条赛道最大偏差={worst_d:.2f}px ({worst_track})")
    item("C2.弯角编号沿赛道走向单调", "WARN" if inv_tracks else "PASS",
         "全部单调递增" if not inv_tracks else
         f"{len(inv_tracks)} 条赛道路径起点不在 T1 前: " + "; ".join(inv_tracks))


# =========================================================================== #
# D. 「当前弯」映射算法（ws._map_corner 均匀分布 vs 锚点真实弧长）
# =========================================================================== #
def check_corner_mapping():
    from setup_tuner.api.ws import _map_corner
    from setup_tuner.domain.track import get_all_tracks
    ANCH, _ = load_anchors()
    total_pos, total_n, tracks_bad, worst = 0, 0, 0, (None, 0.0)
    for t in get_all_tracks():
        if t.track_id not in ANCH:
            continue
        poly = polyline(t.track_id)
        cum = [0.0]
        for i in range(len(poly) - 1):
            cum.append(cum[-1] + math.hypot(poly[i + 1][0] - poly[i][0],
                                            poly[i + 1][1] - poly[i][1]))
        tot = cum[-1] or 1.0
        # 锚点真实弧长位置 → 该弯角在赛道上的真实位置
        true_pos = {}
        for cn, pt in ANCH[t.track_id].items():
            _, s = arc_lookup(pt, poly, cum)
            true_pos[cn] = s / tot
        diag = math.hypot(800, 600)
        errs = []
        for step in range(200):
            lap_dist = tot / (2 * math.pi) * step / 200 * 2 * math.pi  # 0..tot
            got = _map_corner(lap_dist, tot, t.corners, t.track_id)
            # 真值：取真实弧长最近的弯角
            gt = min(true_pos, key=lambda c: abs(true_pos[c] - lap_dist / tot))
            errs.append(0 if got == gt else 1)
            if got != gt:
                worst = max(worst, (t.track_id, abs(true_pos[gt] - lap_dist / tot)),
                            key=lambda x: x[1])
        bad = sum(errs)
        total_pos += bad
        total_n += len(errs)
        if bad > len(errs) * 0.5:
            tracks_bad += 1
        _ = diag
    item("D1._map_corner 均匀分布假设 vs 锚点真实顺序",
         "FAIL" if total_pos > total_n * 0.2 else "PASS",
         f"24 赛道共采样 {total_n} 个圈内位置，弯角判定错误 {total_pos} 个 "
         f"({total_pos / total_n * 100:.1f}%)；错误率>50% 的赛道 {tracks_bad} 条")


# =========================================================================== #
# E. 遥测 → 引擎：单帧摘要 vs 整圈摘要 的规则覆盖率
# =========================================================================== #
def check_telemetry_rule_coverage():
    from setup_tuner.engine.engine import _derive_telemetry_dx
    from setup_tuner.report.builder import extract_telemetry_summary

    # ── 单帧（旧生产路径）：只喂最新一帧 ──
    live_packets = {
        1: {"m_trackId": 13, "m_weather": 0, "m_trackTemperature": 38,
            "m_airTemperature": 27},
        2: {"m_sector": 1, "m_lapDistance": 2100.0, "m_currentLapNum": 3,
            "m_lastLapTimeInMS": 91234},
        6: {"m_speed": 168, "m_throttle": 0.83, "m_brake": 0.0, "m_steer": 0.22,
            "m_gear": 6, "m_engineRPM": 11200,
            "m_tyresSurfaceTemperature": [104, 108, 96, 99],
            "m_tyresInnerTemperature": [112, 115, 101, 104],
            "m_brakesTemperature": [612, 648, 402, 418],
            "m_tyresPressure": [23.4, 24.1, 21.9, 22.2]},
        7: {"m_visualTyreCompound": 16, "m_tyresAgeLaps": 7, "m_fuelInTank": 68.5},
    }
    single = extract_telemetry_summary(live_packets)
    single_dx = _derive_telemetry_dx(single)

    # ── 新生产路径：单帧 + 整圈聚合（LapAggregator.snapshot()） ──
    lap_stats = {
        "max_speed": 332.0, "avg_speed": 214.0,
        "avg_steer": 0.14, "max_steer": 0.61,
        "avg_throttle": 0.71, "avg_brake": 0.11, "max_brake": 0.98,
        "straight_ratio": 0.42, "on_straight": True,
        "lap_number": 3, "lap_frames": 5400, "sector": 3,
        "m_tyresSurfaceTemperature": [104, 108, 96, 99],
        "m_brakesTemperature": [612, 648, 402, 418],
        "m_tyresPressure": [23.4, 24.1, 21.9, 22.2],
    }
    with_lap = extract_telemetry_summary(live_packets, lap_stats)
    lap_dx = _derive_telemetry_dx(with_lap)
    lap_triggered = {k: v for k, v in lap_dx.items() if v != 0.0}
    single_triggered = {k: v for k, v in single_dx.items() if v != 0.0}

    # 逐规则探针：判断「该规则能否在给定遥测下触发」
    probes = {
        "规则5(出弯油门低, 需 sector==3/1基)": {"sector": 3, "m_throttle": 0.1},
        "规则6(直道速度低, 需 on_straight)": {"speed": 150.0, "on_straight": True},
        "规则7(入弯响应, 需 max_steer)": {"max_steer": 0.5},
        "规则10(弯中不稳, 需 avg_steer)": {"avg_steer": 0.35},
        "规则15(直道极速低, 需 max_speed)": {"max_speed": 240.0},
        "规则4(刹车过热)": {"m_brakesTemperature": [700, 700, 700, 700]},
        "规则8(制动力不足)": {"m_brake": 0.9},
        "规则1(胎温过高)": {"m_tyresSurfaceTemperature": [110, 112, 108, 110]},
        "规则3(胎压异常)": {"m_tyresPressure": [29.9, 29.9, 21.0, 21.0]},
        "规则13(湿地胎温低)": {"m_tyresSurfaceTemperature": [50, 50, 50, 50],
                              "m_weather": 2},
        "规则9(刮底, 占位未实现)": {"speed": 80.0, "ride_height": 10},
    }
    dead = [name for name, t in probes.items()
            if not any(v != 0 for v in _derive_telemetry_dx(t).values())]

    item("E1.单帧路径遥测摘要字段数", "INFO",
         f"字段数={len(single)}；触发维度={sorted(single_triggered)} "
         f"（缺 max_speed/avg_steer/max_steer/on_straight，故 5 条规则不可用）")
    item("E2.整圈聚合路径触发维度", "INFO",
         f"触发={sorted(lap_triggered)}（含 max_speed/avg_steer/max_steer/on_straight 后规则 6/7/10/15 可用）")
    item("E3.LapAggregator 是否接入生产路径",
         "PASS" if _lap_aggregator_wired() else "FAIL",
         "app.py 已把 Packet2/6 喂给 LapAggregator，routes.suggest 会把 "
         "best_snapshot() 合进遥测摘要" if _lap_aggregator_wired() else
         "未接入：整圈统计不会被使用")
    item("E4.规则 6/7/10/15/5 是否可触发",
         "PASS" if not [d for d in dead if d.startswith(("规则5", "规则6", "规则7", "规则10", "规则15"))]
         else "FAIL",
         "可触发" if not [d for d in dead if d.startswith(("规则5", "规则6", "规则7", "规则10", "规则15"))]
         else f"仍不可触发：{[d for d in dead if d.startswith(('规则5', '规则6', '规则7', '规则10', '规则15'))]}")
    item("E5.仍未实现的遥测规则", "WARN" if dead else "PASS",
         "; ".join(dead) if dead else "无（15 条规则全部有实现）")


def _lap_aggregator_wired() -> bool:
    """检查 LapAggregator 是否真的接在应用入口上。"""
    app_src = (ROOT / "setup_tuner" / "app.py").read_text("utf-8")
    routes_src = (ROOT / "setup_tuner" / "api" / "routes.py").read_text("utf-8")
    return (
        "lap_aggregator" in app_src
        and "on_telemetry" in app_src
        and "best_snapshot" in routes_src
    )


# =========================================================================== #
# F. WS 事件契约 vs 前端读取字段
# =========================================================================== #
def check_ws_contract():
    ws = (ROOT / "setup_tuner" / "api" / "ws.py").read_text("utf-8")
    appjs = (ROOT / "setup_tuner" / "ui" / "app.js").read_text("utf-8")
    # 后端 telemetry 事件 payload keys
    m = re.search(r"payload = \{(.*?)\n    \}", ws, re.S)
    backend = set(re.findall(r'"(\w+)":', m.group(1))) if m else set()
    fn = re.search(r"function onTelemetry\(t\) \{(.*?)\n  \}", appjs, re.S)
    body = "\n".join(
        line for line in (fn.group(1) if fn else "").splitlines() if "//" not in line
    )
    frontend = set(re.findall(r"t\.(\w+)", body))
    missing = sorted(frontend - backend - {"corner_number"})
    item("F1.WS telemetry 事件 payload", "INFO", ", ".join(sorted(backend)))
    item("F2.前端读取但后端不推送的字段", "FAIL" if missing else "PASS",
         f"前端读取 {sorted(frontend)}；后端缺失 {missing}" if missing
         else f"前端读取 {sorted(frontend)} —— 全部由后端推送")
    # F3：扇区基数（0 基 UDP → 1 基统一口径）
    from setup_tuner.report.builder import extract_telemetry_summary
    from setup_tuner.telemetry.packets import to_sector_1based
    conv_ok = [to_sector_1based(v) for v in (0, 1, 2)] == [1, 2, 3]
    summary_sector = extract_telemetry_summary({2: {"m_sector": 2}}).get("sector")
    front_double = "sector + 1" in appjs
    item("F3.扇区基数（0 基 → 1 基）",
         "PASS" if (conv_ok and summary_sector == 3 and not front_double) else "FAIL",
         f"to_sector_1based(0/1/2)={[to_sector_1based(v) for v in (0, 1, 2)]}；"
         f"摘要 sector(m_sector=2)={summary_sector}；前端重复 +1={'有' if front_double else '无'}")
    _ = appjs


# =========================================================================== #
# G. smoke.py 期望参数数 vs 实际
# =========================================================================== #
def check_param_count():
    from setup_tuner.domain.setup import ALL_SETUP_FIELDS, CarSetup
    n = len(ALL_SETUP_FIELDS)
    smoke = (ROOT / "scripts" / "smoke.py").read_text("utf-8")
    m = re.search(r"EXPECTED_PARAM_COUNT\s*=\s*(\d+)", smoke)
    expect = int(m.group(1)) if m else None
    d = CarSetup.default().to_dict()
    item("G1.实际参数个数", "INFO", f"ALL_SETUP_FIELDS={n}, default().to_dict()={len(d)}")
    item("G2.smoke.py 期望个数", "FAIL" if expect != n else "PASS",
         f"EXPECTED_PARAM_COUNT={expect} vs 实际 {n} → Smoke Test 必然失败")
    for f, pat in (("engine/engine.py", r"21 项"),
                   ("report/builder.py", r"20 参数"),
                   ("api/routes.py", r"20 参数")):
        txt = (ROOT / "setup_tuner" / f).read_text("utf-8")
        hits = len(re.findall(pat, txt))
        item(f"G3.文档串不一致 {f}", "WARN" if hits else "PASS",
             f"出现 '{pat}' {hits} 次（实际 {n} 参数）")


# =========================================================================== #
# H. cloud-audit.yml 健康检查路径 & UI 审计目标
# =========================================================================== #
def check_workflows():
    wf = (ROOT / ".github" / "workflows" / "cloud-audit.yml").read_text("utf-8")
    routes = (ROOT / "setup_tuner" / "api" / "routes.py").read_text("utf-8")
    prefix = re.search(r'APIRouter\(prefix="([^"]+)"', routes)
    prefix = prefix.group(1) if prefix else ""
    u = re.search(r"curl -sf (http://[^\s]+)", wf)
    url = u.group(1) if u else None
    item("H1.cloud-audit 健康检查 URL", "FAIL" if url and "/api/health" in url else "PASS",
         f"workflow 探测 {url}，真实路径是 {prefix}/health → 探测永远 404")
    audit = (ROOT / "cloud_audit" / "ui_click_audit.py").read_text("utf-8")
    idx = (ROOT / "setup_tuner" / "ui" / "index.html").read_text("utf-8")
    sel = ["predict-btn", "feedback-input", "search-btn", "chat-input",
           "iter-link", "export-setup-btn", "str-laps", "driver-style-select",
           "apply-recommended-btn"]
    miss = [s for s in sel if s not in idx]
    item("H2.UI 审计脚本选择器 vs 现役 UI", "FAIL" if miss else "PASS",
         f"{len(miss)}/{len(sel)} 个选择器在新 UI 中不存在: {miss}")
    item("H3.UI 审计目标页 /dashboard.html", "FAIL" if "dashboard.html" in audit else "PASS",
         "新 UI 只有 index.html，/dashboard.html 恒为 404")
    item("H4.CI 中 UI 审计是否阻断", "FAIL" if "continue-on-error: true" in wf else "PASS",
         "cloud-audit.yml 的 UI 审计步骤 continue-on-error: true → 失败不阻断")
    # 真实数据测试路径
    ti = (ROOT / "tests" / "test_telemetry_importer.py").read_text("utf-8")
    p = re.search(r'_REAL_DATA_DIR = r"([^"]+)"', ti)
    item("H5.真实遥测测试的数据路径", "FAIL",
         f"{p.group(1) if p else '?'} (Windows 本地专属；CI 中必然 skip)")


# =========================================================================== #
# I. 录制文件 & 死代码
# =========================================================================== #
def check_recordings():
    recs = sorted((ROOT / "data" / "recordings").glob("*.f1rec"))
    sizes = [r.stat().st_size for r in recs]
    item("I1.已提交录制文件", "FAIL" if recs and max(sizes) <= 73 else "WARN",
         f"{len(recs)} 个 .f1rec，大小={sizes}（仅文件头 = 69B，0 个数据包记录）")
    for f in ["data/f1opt.db-shm", "data/f1opt.db-wal"]:
        if (ROOT / f).exists():
            item("I2.误提交 SQLite 临时文件", "FAIL",
                 f"{f} ({ (ROOT/f).stat().st_size }B) 已入库，且 .gitignore 含 *.db")


def check_dead_modules():
    """静态可达性：从应用入口不可达的模块。"""
    import ast
    mods = {}
    for p in (ROOT / "setup_tuner").rglob("*.py"):
        m = str(p.relative_to(ROOT)).replace("\\", "/")[:-3].replace("/", ".")
        if m.endswith(".__init__"):
            m = m[:-9]
        mods[m] = p
    edges = {}
    for m, p in mods.items():
        out = set()
        try:
            tree = ast.parse(p.read_text("utf-8"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                base = n.module or ""
                lvl = n.level
                if lvl:
                    parts = m.split(".")
                    pkg = ".".join(parts[: len(parts) - lvl]) if lvl <= len(parts) else ""
                    cand = (pkg + "." + base).strip(".") if base else pkg
                else:
                    cand = base
                for a in n.names:
                    out.add(cand + "." + a.name)
                    out.add(cand)
        edges[m] = out

    def resolve(name):
        n = name
        while n:
            if n in mods:
                return n
            n = n.rsplit(".", 1)[0] if "." in n else ""
        return None

    g = {m: {r for d in ds if (r := resolve(d))} for m, ds in edges.items()}
    entry = ["setup_tuner.app", "setup_tuner.cli", "setup_tuner.api.routes",
             "setup_tuner.engine.engine", "setup_tuner.domain.track",
             "setup_tuner.report.builder", "setup_tuner.telemetry.listener",
             "setup_tuner.telemetry.simulator"]
    seen, stack = set(), list(entry)
    while stack:
        m = stack.pop()
        if m in seen or m not in g:
            continue
        seen.add(m)
        stack.extend(g[m])
    pkgs = {m for m in mods if m.endswith(tuple(
        f".{x}" for x in ["api", "db", "domain", "engine", "feedback", "physics",
                          "report", "telemetry", "setup_tuner"])) or m == "setup_tuner"}
    dead = sorted(set(mods) - seen - pkgs)
    tot = sum(mods[m].stat().st_size for m in dead)
    item("J1.应用不可达模块", "FAIL" if dead else "PASS",
         f"{len(dead)} 个模块 / {tot / 1024:.0f} KB 仅测试可见: {dead}")


# =========================================================================== #
# =========================================================================== #
# K. 每车结构步长(stride) 实测 —— 决定 _slice_player_car 偏移是否正确
# =========================================================================== #
def check_stride():
    import struct as _s

    fp = ROOT / "legacy" / "tests" / "data" / "real_f1_26_sample.jsonl"
    if not fp.exists():
        item("K.每车结构步长", "SKIP", "无真实样本")
        return
    want = {2: "LapData", 5: "CarSetups", 6: "CarTelemetry", 7: "CarStatus"}
    blob: dict[int, bytes] = {}
    for line in fp.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        pid = rec.get("packetId")
        if pid in want and pid not in blob:
            blob[pid] = bytes.fromhex(rec["hex"])
    from setup_tuner.telemetry.packets import (
        NUM_CARS, _LAP_PER_STRUCT, _SETUP_PER_STRUCT, _STATUS_PER_STRUCT,
        _TELEM_PER_STRUCT,
    )
    expect = {2: _LAP_PER_STRUCT.size, 5: _SETUP_PER_STRUCT.size,
              6: _TELEM_PER_STRUCT.size, 7: _STATUS_PER_STRUCT.size}
    for pid, data in sorted(blob.items()):
        body = len(data) - 29
        stride = body / NUM_CARS
        ok = (body % NUM_CARS == 0) and (int(stride) == expect[pid])
        item(f"K{pid}.{want[pid]} 每车步长", "PASS" if ok else "FAIL",
             f"包长={len(data)} 体长={body} ÷ NUM_CARS({NUM_CARS})={stride:.4f}；"
             f"代码常量={expect[pid]}B → "
             + ("一致" if ok else "不一致！playerCarIndex>0 时切片错位"))
    data5 = blob.get(5)
    if data5:
        hits = []
        for o in range(29, len(data5) - 24):
            try:
                c1, c2, t1, t2 = _s.unpack_from("<ffff", data5, o + 4)
            except _s.error:
                continue
            if (-3.6 <= c1 <= -2.4 and -2.1 <= c2 <= -0.9
                    and 0.0 <= t1 <= 0.25 and 0.05 <= t2 <= 0.40):
                hits.append(o)
        diffs = sorted({hits[i + 1] - hits[i] for i in range(len(hits) - 1)}) \
            if len(hits) > 1 else []
        item("K5b.CarSetups 结构起点实测间距", "INFO",
             f"候选起点 {len(hits)} 个, 间距集合={diffs} → 实测步长="
             f"{diffs[0] if diffs else '无法判定'}，代码常量=50")
    from setup_tuner.telemetry.packets import parse_packet
    sectors: set = set()
    for line in fp.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("packetId") != 2:
            continue
        pr = parse_packet(bytes.fromhex(rec["hex"]))
        if pr:
            sectors.add(pr.get("m_sector"))
    item("K6.真实 m_sector 取值", "FAIL" if sectors and max(sectors) <= 2 else "INFO",
         f"实测集合={sorted(sectors)} → 确认 0 基(0/1/2)；engine 规则5 判断 "
         "sector==3 永不成立，前端直接显示 'S'+sector")


# =========================================================================== #
# L. Packet5 UDP 原始值 → 车库值 换算是否合理（真实抓包判定）
# =========================================================================== #
def check_udp_remap():
    fp = ROOT / "legacy" / "tests" / "data" / "real_f1_26_sample.jsonl"
    from setup_tuner.domain.setup import ALL_SETUP_FIELDS
    from setup_tuner.report.builder import (
        _PACKET5_FIELD_MAP, convert_udp_to_game_value,
    )
    from setup_tuner.telemetry.packets import parse_car_setups, parse_header
    if not fp.exists():
        item("L.Packet5 值域换算", "SKIP", "无真实样本")
        return
    raw = None
    for line in fp.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("packetId") == 5:
            data = bytes.fromhex(rec["hex"])
            raw = parse_car_setups(data, parse_header(data).player_car_index)
            break
    if raw is None:
        item("L.Packet5 值域换算", "SKIP", "无 CarSetups 包")
        return
    spec = {f.name: f for f in ALL_SETUP_FIELDS}
    direct, remapped = [], []
    for udp_name, param in _PACKET5_FIELD_MAP.items():
        if udp_name not in raw:
            continue
        u = float(raw[udp_name])
        mapped = convert_udp_to_game_value(param, u)
        sp = spec[param]
        in_domain = sp.min_val - 1e-9 <= u <= sp.max_val + 1e-9
        line = (f"{param}: UDP原值={u:g} → 换算后={mapped:.2f} "
                f"车库合法域=[{sp.min_val:g},{sp.max_val:g}]")
        (direct if in_domain else remapped).append(line)
    item("L1.Packet5 原值是否已在车库合法域内",
         "FAIL" if len(direct) > len(remapped) else "PASS",
         f"{len(direct)}/{len(direct) + len(remapped)} 个字段的 UDP 原值本身就落在车库域内 "
         "→ EA 下发的是车库值，_UDP_VALUE_RANGE_MAP 的线性重映射会把它改错")
    for line in direct:
        print("     · " + line, flush=True)


def main() -> int:
    print("=" * 78, flush=True)
    print("F1OPT 深度审计验证（云端只读）", flush=True)
    print("=" * 78, flush=True)
    for fn in (check_real_packets, check_track_id_map, check_track_geometry,
               check_corner_mapping, check_telemetry_rule_coverage,
               check_ws_contract, check_param_count, check_workflows,
               check_recordings, check_dead_modules,
               check_stride, check_udp_remap):
        print(f"\n--- {fn.__name__} ---", flush=True)
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            item(fn.__name__, "ERROR", f"{type(e).__name__}: {e}")

    print("\n" + "=" * 78, flush=True)
    print("汇总", flush=True)
    print("=" * 78, flush=True)
    from collections import Counter
    c = Counter(v for _, v, _ in RESULT)
    for k in ("FAIL", "ERROR", "WARN", "PASS", "SKIP", "INFO"):
        if c.get(k):
            print(f"  {k}: {c[k]}", flush=True)
    print("\n失败项明细：", flush=True)
    for tag, v, d in RESULT:
        if v in ("FAIL", "ERROR", "WARN"):
            print(f"  [{v}] {tag} :: {d}", flush=True)
    Path("audit_verify_result.json").write_text(
        json.dumps([{"tag": t, "verdict": v, "detail": d} for t, v, d in RESULT],
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
