"""在**真实 API 链路**上验证：车手零反馈时，遥测自动发现的问题仍能出方案。

为什么必须做这一层验证
----------------------
引擎内部的单元验证（``scripts/verify_model_in_engine.py``）只能证明
"引擎函数会这么做"。用户实际用的是**跑起来的 F1OPT 服务**：
    ``POST /suggest`` → 路由 → 遥测摘要 → 引擎 → 报告 → 落库
原来这条路被一个闸门挡住了 —— 没有车手反馈时 `/suggest` 直接 400，
所以遥测自动发现的问题**永远到不了用户面前**（这就是"没生效"的根因）。

本脚本用**真实的 app 实例 + 真实的 UDP 包处理链**（``_make_packet_handler``）
喂进一圈合成遥测，然后走 HTTP 调用 `/suggest`，断言：

1. 零车手反馈 + 有遥测发现 → HTTP 200（不再是 400）；
2. 报告里 ``telemetry_only`` 为真、``holistic.implicit_feedbacks`` 非空；
3. 报告里 ``holistic.implicit_plan`` 给出「问题 → 参数改动」的落地映射；
4. 建议里出现悬挂/几何类改动（整体性）；
5. 回归：零反馈 + 零遥测 → 仍然 400（不能变成"什么都出方案"）。

用法::

    python scripts/verify_api_telemetry_only.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_MECH = (
    "front_camber", "rear_camber", "front_toe", "rear_toe",
    "front_suspension", "rear_suspension",
    "front_anti_roll_bar", "rear_anti_roll_bar",
    "front_ride_height", "rear_ride_height",
)

#: Monza：UDP 赛道号 11，长度 5793 m
_TRACK_ID = "monza"
_TRACK_UDP = 11
_TRACK_LEN = 5793.0

#: 压路肩的弯窗口（弧长占比 → 米），取自官方锚点表
_KERB_WINDOWS = ((0.045, 0.075), (0.33, 0.36))   # T2 Rettifilo / T5 Roggia 附近


def _feed_lap(handler, lap_no: int) -> None:
    """通过真实包处理链喂进一圈遥测（含路肩/刮底/胎耗/车损/ABS/TC）。"""
    handler({
        "packet_id": 1,
        "m_trackId": _TRACK_UDP,
        "m_trackLength": _TRACK_LEN,
        "m_sessionUID": 12345,
    })
    steps = 720                       # 每圈 720 帧（≈4 帧/25 m），保证每弯 ≥15 帧
    for i in range(steps):
        frac = i / steps
        dist = frac * _TRACK_LEN
        is_kerb = any(a <= frac < b for a, b in _KERB_WINDOWS)
        rough = 6.0 if is_kerb else 0.4

        handler({"packet_id": 2, "m_currentLapNum": lap_no,
                 "m_lapDistance": dist, "m_sector": i * 3 // steps,
                 "m_lastLapTimeInMS": 95000})
        handler({
            "packet_id": 6,
            "m_speed": 300.0 if frac < 0.05 else 120.0,
            "m_throttle": 0.9 if frac < 0.5 else 0.4,
            "m_brake": 0.2 if frac > 0.5 else 0.0,
            "m_steer": 0.30 if is_kerb else 0.10,
            # 官方车轮序 RL, RR, FL, FR：[2][3]=前轮更热 → 前轴在滑
            "m_tyresSurfaceTemperature": [88, 89, 106, 107],
            "m_tyresInnerTemperature": [95, 96, 113, 114],
            "m_brakesTemperature": [420, 425, 690, 700],
            "m_tyresPressure": [22.0, 22.1, 23.2, 23.3],
        })
        handler({
            "packet_id": 13,
            # 底板离地落入触地带（上限 12 mm）
            "m_frontAeroHeight": 0.006 if is_kerb else 0.030,
            "m_rearAeroHeight": 0.008 if is_kerb else 0.035,
            "m_suspensionPosition": [0.02 + rough / 100.0] * 4,
            "m_suspensionAcceleration": [rough, rough, rough, rough],
        })
        handler({"packet_id": 0, "m_gForceLateral": 3.2,
                 "m_gForceLongitudinal": -4.1})
    # 车损 / 胎耗（Packet 10）与状态（Packet 7）
    handler({"packet_id": 10, "m_tyresWear": [12.0, 12.0, 26.0, 26.0],
             "m_floorDamage": 45, "m_frontLeftWingDamage": 30,
             "m_frontRightWingDamage": 28, "m_rearWingDamage": 5})
    handler({"packet_id": 7, "m_antiLockBrakes": 1, "m_tractionControl": 2,
             "m_actualTyreCompound": 16, "m_frontBrakeBias": 57,
             "m_tyresAgeLaps": 6})


def main() -> int:
    from fastapi.testclient import TestClient

    from setup_tuner.app import _make_packet_handler, create_app
    from setup_tuner.config import Config

    tmp = Path(tempfile.mkdtemp(prefix="f1opt_api_verify_"))
    app = create_app(Config(data_dir=str(tmp / "data")))
    handler = _make_packet_handler(app)
    client = TestClient(app)
    ok = True

    print("=" * 74)
    print("真实 API 链路验证：车手零反馈 + 遥测自动发现")
    print("=" * 74)

    # TestClient 的上下文管理器才会触发 startup（app.state 的服务在此装配）
    with client:
        _feed_lap(handler, lap_no=1)
        _feed_lap(handler, lap_no=2)
        resp = client.post("/api/v1/suggest", json={"track_id": _TRACK_ID,
                                             "model_type": "hybrid"})
    print(f"[1] POST /suggest（无任何车手反馈）→ HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(f"    ✗ 仍未放行：{resp.text[:300]}")
        return 1
    body = resp.json()
    report = body["data"]["report"]

    print(f"    telemetry_only       = {report.get('telemetry_only')}")
    print(f"    telemetry_discovered = {report.get('telemetry_discovered')}")
    print(f"    summary              = {str(report.get('summary'))[:80]}")
    ok = ok and report.get("telemetry_only") is True

    holistic = report.get("holistic") or {}
    findings = holistic.get("implicit_feedbacks") or []
    print(f"\n[2] 报告里的遥测自动发现：{len(findings)} 条")
    for f in findings[:10]:
        where = f"T{f['corner']}" if f.get("corner") else "全局"
        print(f"    - {where:<6}{f['symptom']:<24}强度{f['strength']}  [{f['source']}]")
        print(f"      {f['evidence']}")
    ok = ok and len(findings) > 0

    plan = holistic.get("implicit_plan") or []
    resolved = [p for p in plan if p.get("resolved")]
    print(f"\n[3] 问题 → 优化方案 映射：{len(resolved)}/{len(plan)} 条已落地")
    for p in plan[:8]:
        tag = "✓" if p.get("resolved") else "✗"
        chg = "、".join(
            f"{c['label']}{c['delta']:+.2f}" for c in (p.get("changes") or [])[:3]
        ) or "无对应改动"
        print(f"    {tag} {p['symptom']:<24}→ {chg}")
    ok = ok and len(resolved) > 0

    delta = report.get("setup_delta") or {}
    nonzero = {k: v for k, v in delta.items() if v}
    mech = {k: v for k, v in nonzero.items() if k in _MECH}
    print(f"\n[4] 建议共 {len(nonzero)} 项非零改动，其中悬挂/几何 {len(mech)} 项")
    for k, v in mech.items():
        print(f"    ★ {k:<28}{v:+.2f}")
    ok = ok and len(mech) > 0

    print("\n[5] 回归：清空遥测与反馈后仍应被拒绝")
    app2 = create_app(Config(data_dir=str(tmp / "data2")))
    with TestClient(app2) as client2:
        resp2 = client2.post("/api/v1/suggest", json={"track_id": _TRACK_ID})
    print(f"    零反馈 + 零遥测 → HTTP {resp2.status_code}")
    ok = ok and resp2.status_code == 400

    print("\n" + "=" * 74)
    print("结论：" + ("✅ 五项全部通过 —— 遥测自动发现已在真实 API 链路上生效"
                    if ok else "✗ 存在未通过项"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
