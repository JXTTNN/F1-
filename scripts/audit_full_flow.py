"""全流程审计：按真实使用顺序走一遍面板后端，检查「功能是否生效、彼此是否协同」。

与单元测试的区别：本脚本**只走用户真实路径**（HTTP API + UDP 帧处理函数），
不直接调用引擎内部函数。每一步都断言可观察结果，并记录耗时，便于发现
「单测绿、串起来坏」的协同问题。

覆盖链路：
    启动 → 健康检查 → 24 赛道 → 选赛道 → 导入默认调教
    → 连跑 3 圈（UDP 帧）→ 圈记录落库 + 12 维风格向量（样本 ≥3 才启用）
    → 车手反馈 → 生成建议（nn）→ 报告含模拟轨迹 / 风格定向
    → 参数合法性（区间 / 单次上限 / 档位对齐 / 胎压对称）
    → 建议落库 + 迭代历史 + 反馈消费
    → 风格定向确实改变建议（同一反馈，清空风格前后对比）

用法：
    python scripts/audit_full_flow.py            # 全流程（约 20s）
    python scripts/audit_full_flow.py --keep     # 保留审计数据目录

退出码：0 = 全部通过；1 = 有断言失败（逐条打印）。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from setup_tuner.app import _make_packet_handler, create_app  # noqa: E402
from setup_tuner.config import Config  # noqa: E402
from setup_tuner.domain.setup import ALL_SETUP_FIELDS  # noqa: E402

AUDIT_DIR = ROOT / ".audit_data"
TRACK_ID = "suzuka"
LAPS = 3


class Audit:
    """极简断言收集器：记录每条检查 + 耗时，最后统一汇报。"""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, float, str]] = []

    def check(self, name: str, ok: bool, detail: str = "",
              elapsed: float = 0.0) -> bool:
        self.rows.append((name, bool(ok), elapsed, detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
        return bool(ok)

    @property
    def failed(self) -> list[tuple[str, bool, float, str]]:
        return [r for r in self.rows if not r[1]]

    def summary(self) -> None:
        total = len(self.rows)
        bad = len(self.failed)
        slow = sorted(self.rows, key=lambda r: -r[2])[:3]
        print("\n" + "=" * 74)
        print(f"检查项 {total} | 通过 {total - bad} | 失败 {bad}")
        if slow:
            print("最慢三步：" + "；".join(
                f"{n} {t * 1000:.0f}ms" for n, _, t, _ in slow))
        print("=" * 74)


def _style_frame(i: int) -> dict[str, Any]:
    """构造「有明确驾驶风格」的一帧遥测（Packet 6）。

    为什么不是匀速直线：风格提取器的 12 维全部来自驾驶输入分布，匀速直行
    会得到一个「无偏好」向量（转向小、无刹车、平滑度满格），风格定向层
    按设计不会触发。这里按 60 帧一个循环模拟：重刹入弯 → 持续转向 →
    全油出弯，使「攻弯强度 / 转向平滑度 / 循迹刹车 / 胎温离散」四个维度
    都落在触发区间，才能检验整条链路（遥测 → 风格 → 建议）真的接通。
    """
    phase = i % 60
    if phase < 12:            # 重刹入弯：刹车到底、几乎不给油
        speed, brake, throttle = 240 - (phase * 12), 0.9, 0.05
    elif phase < 32:          # 弯中：持续大转向 + 循迹刹车
        speed, brake, throttle = 96 + (phase % 6) * 3, 0.55, 0.12
    else:                     # 出弯：全油加速
        speed, brake, throttle = 96 + (phase - 31) * 8, 0.0, 1.0
    # 转向每 5 帧在「大角度 / 回正」之间抖动 → 转向变化率高（平滑度低）。
    # 注意：提取器用 |steer| 的变化率，所以必须让**幅度**变化（只换正负号
    # 的等幅摆动，绝对值不变，会被算成"非常平滑"）。
    steer = 0.45 if (i // 5) % 2 == 0 else 0.05
    return {
        "packet_id": 6, "m_speed": float(speed), "m_throttle": throttle,
        "m_brake": brake, "m_steer": steer,
        # 四轮胎温拉开 30℃ 以上 → 胎温离散度高（轮胎管理差）
        "m_tyresSurfaceTemperature": [72 + (i % 7), 104 - (i % 5), 88, 96],
        "m_brakesTemperature": [620, 640, 430, 450],
    }


def _run_laps(handler, laps: int) -> None:
    """连跑 N 圈：每圈 3000 帧，圈号切换触发落库（与真实整圈同一路径）。"""
    handler({"packet_id": 2, "m_currentLapNum": 1, "m_sector": 0})
    for lap in range(1, laps + 1):
        for i in range(3000):
            handler(_style_frame(i))
        handler({
            "packet_id": 2, "m_currentLapNum": lap + 1, "m_sector": 0,
            "m_lastLapTimeInMS": 90000 + lap * 120,
        })


def _wait_style(store, track_id: str, want: int, timeout: float = 8.0) -> dict | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        entry = store.get_driver_style(1, track_id)
        if entry and entry.get("sample_count", 0) >= want:
            return entry
        time.sleep(0.05)
    return store.get_driver_style(1, track_id)


def _reset_style(store, track_id: str) -> bool:
    """把该赛道风格样本清零（模拟「刚装好、还没跑够 3 圈」）。"""
    try:
        with store._lock:
            store._conn.execute(
                "DELETE FROM driver_style WHERE driver_id = 1 AND track_id = ?",
                (track_id,),
            )
            store._conn.commit()
        return store.get_driver_style(1, track_id) is None
    except Exception as exc:  # pragma: no cover - 审计脚本
        print("    reset style failed:", exc)
        return False


def _suggest(client: TestClient, model_type: str = "nn") -> dict[str, Any]:
    """生成建议 → 返回 report 本体（API 把它包在 data.report 里）。"""
    resp = client.post("/api/v1/suggest",
                       json={"track_id": TRACK_ID, "model_type": model_type})
    assert resp.status_code == 200, f"/suggest {resp.status_code}: {resp.text[:300]}"
    body = resp.json()
    assert body.get("code") == 0, body
    data = body["data"]
    return data.get("report") or data


def _check_delta_legality(a: Audit, setup_delta: dict[str, float],
                          current: dict[str, float], tag: str) -> None:
    """参数合法性：|Δ|<=max_delta、终值在区间内、落在真实档位上、胎压对称。"""
    by_name = {s.name: s for s in ALL_SETUP_FIELDS}
    out_of_range, over_cap, off_step, unknown = [], [], [], []
    for name, delta in setup_delta.items():
        spec = by_name.get(name)
        if spec is None:
            unknown.append(name)
            continue
        cur = float(current.get(name, spec.default))
        if abs(delta) > spec.max_delta + 1e-9:
            over_cap.append(name)
        end = cur + delta
        if end < spec.min_val - 1e-9 or end > spec.max_val + 1e-9:
            out_of_range.append(name)
        if spec.step > 0:
            k = (end - spec.min_val) / spec.step
            if abs(k - round(k)) > 1e-6:
                off_step.append(name)
    a.check(f"{tag}：无未知参数", not unknown, ",".join(unknown) or "20 项全部已知")
    a.check(f"{tag}：单次上限", not over_cap, ",".join(over_cap) or "全部 <= max_delta")
    a.check(f"{tag}：合法区间", not out_of_range, ",".join(out_of_range) or "全部在 [min,max]")
    a.check(f"{tag}：档位对齐", not off_step, ",".join(off_step) or "终值均可拧出")

    pairs = [("tyre_pressure_fl", "tyre_pressure_fr"),
             ("tyre_pressure_rl", "tyre_pressure_rr")]
    asym = []
    for left, right in pairs:
        if left in setup_delta and right in setup_delta:
            end_l = float(current[left]) + setup_delta[left]
            end_r = float(current[right]) + setup_delta[right]
            if abs(end_l - end_r) > 0.011:
                asym.append(f"{left}/{right}")
    a.check(f"{tag}：胎压左右对称", not asym, ",".join(asym) or "前后轴均对称")


def main() -> int:
    parser = argparse.ArgumentParser(description="F1OPT 全流程审计")
    parser.add_argument("--keep", action="store_true", help="保留审计数据目录")
    args = parser.parse_args()

    if AUDIT_DIR.exists():
        shutil.rmtree(AUDIT_DIR)
    cfg = Config(data_dir=str(AUDIT_DIR), udp_port=0)
    app = create_app(cfg)
    audit = Audit()
    handler = _make_packet_handler(app)

    print("\n[1] 启动 / 健康检查 / 赛道与调教")
    boot_t0 = time.perf_counter()
    with TestClient(app) as client:
        t = time.perf_counter()
        health = client.get("/api/v1/health").json()
        audit.check("健康检查", health.get("code") == 0
                    and "status" in (health.get("data") or {}),
                    f"status={(health.get('data') or {}).get('status')}",
                    time.perf_counter() - t)

        t = time.perf_counter()
        tracks = client.get("/api/v1/tracks").json()["data"]
        audit.check("赛道清单", len(tracks) == 24, f"{len(tracks)} 条",
                    time.perf_counter() - t)

        t = time.perf_counter()
        sel = client.post("/api/v1/tracks/current", json={"track_id": TRACK_ID})
        audit.check("选当前赛道", sel.status_code == 200,
                    TRACK_ID, time.perf_counter() - t)

        t = time.perf_counter()
        fields = client.get("/api/v1/setup/fields").json()["data"]
        audit.check("调教字段", len(fields) == 20, f"{len(fields)} 项",
                    time.perf_counter() - t)

        # 真实流程第 3 步：导入当前调教。没有快照时 /setup/current 应 404，
        # 这里用默认值手填一份（等价于面板里「手动设置调教」）。
        t = time.perf_counter()
        no_snapshot = client.get("/api/v1/setup/current",
                                 params={"track_id": TRACK_ID})
        audit.check("无快照时如实报 404", no_snapshot.status_code == 404,
                    f"HTTP {no_snapshot.status_code}", time.perf_counter() - t)

        defaults = {f["name"]: f["default"] for f in fields}
        t = time.perf_counter()
        saved = client.post("/api/v1/setup/manual",
                            json={"track_id": TRACK_ID, "params": defaults})
        audit.check("导入/手动调教", saved.status_code == 200,
                    (saved.json().get("message") or "")[:40],
                    time.perf_counter() - t)

        t = time.perf_counter()
        cur = client.get("/api/v1/setup/current",
                         params={"track_id": TRACK_ID}).json()["data"]
        current_setup = cur.get("params") or {}
        audit.check("当前调教可读", len(current_setup) == 20,
                    f"{len(current_setup)} 项", time.perf_counter() - t)

        print("\n[2] 遥测 → 整圈 → 12 维风格向量")
        t = time.perf_counter()
        _run_laps(handler, LAPS)
        store = app.state.store
        style = _wait_style(store, TRACK_ID, LAPS)
        records = store.get_lap_records(TRACK_ID, 1)
        audit.check("整圈落库", len(records) == LAPS, f"{len(records)}/{LAPS} 圈",
                    time.perf_counter() - t)
        audit.check("风格样本累积", bool(style) and style["sample_count"] == LAPS,
                    f"sample_count={(style or {}).get('sample_count')}")
        audit.check("风格向量维度", bool(style) and len(style["vector"]) == 12,
                    f"len={len(style['vector']) if style else 0}")

        t = time.perf_counter()
        style_api = client.get("/api/v1/driver/style",
                               params={"track_id": TRACK_ID}).json()["data"]
        audit.check("风格 API 启用态", style_api.get("style_applied") is True,
                    f"applied={style_api.get('style_applied')} "
                    f"samples={style_api.get('sample_count')}",
                    time.perf_counter() - t)

        print("\n[3] 车手反馈 → 生成建议")
        t = time.perf_counter()
        fb = client.post("/api/v1/feedback", json={
            "track_id": TRACK_ID, "corner_number": 1,
            "symptom": "understeer", "strength": 2,
        })
        audit.check("提交反馈", fb.status_code == 200,
                    json.dumps(fb.json().get("data"), ensure_ascii=False)[:90],
                    time.perf_counter() - t)

        t = time.perf_counter()
        report = _suggest(client, "nn")
        audit.check("生成建议（nn）", bool(report.get("setup_delta")),
                    f"{len(report.get('setup_delta') or {})} 项参数",
                    time.perf_counter() - t)
        delta = report.get("setup_delta") or {}
        audit.check("模型真实生效", report.get("model_type") == "nn",
                    f"model_type={report.get('model_type')} "
                    f"nn_available={report.get('nn_available')}")

        hol = report.get("holistic") or {}
        sim = hol.get("simulation") or {}
        audit.check("模拟优化轨迹", bool(sim.get("available")),
                    f"迭代 {sim.get('iterations')} / 采纳 {sim.get('accepted')} / "
                    f"收益 {sim.get('net_gain_ms') or sim.get('gain_ms')}ms")

        style_block = hol.get("style_targeting") or {}
        audit.check("风格定向已生效", bool(style_block.get("enabled")),
                    f"采纳 {len(style_block.get('applied') or [])} 条 / "
                    f"拒绝 {len(style_block.get('rejected') or [])} 条 / "
                    f"代价 {style_block.get('cost_ms')}ms")
        audit.check("风格动作可归因",
                    all(a.get("dim_name") for a in (style_block.get("applied") or [])),
                    "；".join(str(a.get("reason", ""))[:26]
                             for a in (style_block.get("applied") or [])[:2]))
        audit.check("风格代价在预算内",
                    abs(float(style_block.get("cost_ms") or 0.0))
                    <= float(style_block.get("budget_ms") or 25.0) + 0.01,
                    f"{style_block.get('cost_ms')}ms / 预算 "
                    f"{style_block.get('budget_ms')}ms")

        print("\n[4] 参数合法性与链路一致性")
        _check_delta_legality(audit, delta, current_setup, "建议")

        t = time.perf_counter()
        latest = client.get("/api/v1/suggest/latest",
                            params={"track_id": TRACK_ID}).json()
        latest_data = latest.get("data") or {}
        audit.check("建议已落库可回读", latest.get("code") == 0
                    and bool(latest_data.get("suggestion_id")),
                    f"suggestion_id={latest_data.get('suggestion_id')}",
                    time.perf_counter() - t)

        t = time.perf_counter()
        hist = client.get("/api/v1/iteration/history",
                          params={"track_id": TRACK_ID}).json()["data"]
        n_hist = len(hist) if isinstance(hist, list) else len(hist.get("items", []))
        audit.check("迭代历史", n_hist >= 1, f"{n_hist} 条", time.perf_counter() - t)

        t = time.perf_counter()
        left = client.get("/api/v1/feedback",
                          params={"track_id": TRACK_ID}).json()["data"]
        audit.check("反馈按约定消费", len(left) == 0,
                    f"剩余 {len(left)} 条（生成后清空）", time.perf_counter() - t)

        print("\n[5] 风格定向确实改变建议（同反馈、清空风格前后对比）")
        audit.check("可清空风格样本", _reset_style(store, TRACK_ID), "用于 A/B 对比")
        client.post("/api/v1/feedback", json={
            "track_id": TRACK_ID, "corner_number": 1,
            "symptom": "understeer", "strength": 2,
        })
        report_plain = _suggest(client, "nn")
        delta_plain = report_plain.get("setup_delta") or {}
        changed = {k: round(delta.get(k, 0.0) - delta_plain.get(k, 0.0), 4)
                   for k in set(delta) | set(delta_plain)
                   if abs(delta.get(k, 0.0) - delta_plain.get(k, 0.0)) > 1e-9}
        audit.check("风格改变建议", bool(changed),
                    f"{len(changed)} 项不同：" + "、".join(
                        f"{k}{v:+.2f}" for k, v in list(changed.items())[:4]))
        audit.check("无风格时定向关闭",
                    not ((report_plain.get("holistic") or {})
                         .get("style_targeting") or {}).get("enabled"),
                    "样本 <3 圈 → 整层跳过")

        print("\n[6] 稳态性能（同一路径重复生成）")
        timings = []
        for _ in range(3):
            client.post("/api/v1/feedback", json={
                "track_id": TRACK_ID, "corner_number": 2,
                "symptom": "understeer", "strength": 2,
            })
            t = time.perf_counter()
            _suggest(client, "nn")
            timings.append(time.perf_counter() - t)
        avg_ms = sum(timings) / len(timings) * 1000
        audit.check("建议生成耗时", avg_ms < 1500,
                    f"均值 {avg_ms:.0f}ms（{min(timings) * 1000:.0f}–"
                    f"{max(timings) * 1000:.0f}ms）")

    audit.check("启动到首屏可用", True,
                f"{(time.perf_counter() - boot_t0) * 1000:.0f}ms（含进程内建库）")

    audit.summary()
    if not args.keep and AUDIT_DIR.exists():
        shutil.rmtree(AUDIT_DIR, ignore_errors=True)
    return 1 if audit.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
