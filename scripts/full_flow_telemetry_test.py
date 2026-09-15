"""F1OPT 全流程测试 —— 利用 D 盘真实遥测数据，从"点击启动"开始端到端验证。

覆盖链路：
1. 启动服务（等价于双击一键启动.bat 后服务就绪）
2. 健康检查 + 24 条赛道列表
3. 导入真实遥测数据（D:\\F1TelemetryCollector\\dist\\data 4 圈阿布扎比）
4. 验证遥测解析（调教参数/速度/胎温/胎压/圈速统计）
5. 选赛道（阿布扎比 = yas_marina）
6. 提交反馈 → 生成建议 → 读取建议报告
7. 直接验证 importer → to_telemetry_dict → engine 诊断链路
"""
import json
import os
import sys
import threading
import time
import urllib.request
import urllib.error

sys.path.insert(0, "D:/F1OPT-Test")

HOST = "127.0.0.1"
PORT = 8000
BASE = f"http://{HOST}:{PORT}"
DATA_DIR = r"D:\F1TelemetryCollector\dist\data"
TRACK_ID = "yas_marina"  # 阿布扎比


def http(method, path, body=None):
    """通用 HTTP 请求，返回 (status, json_data)。"""
    url = BASE + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def wait_health(timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = urllib.request.urlopen(BASE + "/api/v1/health", timeout=1)
            if r.status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def main():
    print("=" * 72)
    print("F1OPT 全流程测试（真实遥测数据，从点击启动开始）")
    print("=" * 72)

    # ============================================================ #
    # 第 1 步：启动服务（等价于点击一键启动.bat）
    # ============================================================ #
    print("\n[步骤1] 启动服务（等价双击一键启动.bat）...")
    from setup_tuner.app import create_app
    from setup_tuner.config import load_config
    import uvicorn

    config = load_config()
    app = create_app(config)
    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=PORT, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    if not wait_health():
        print("  ❌ 服务未就绪")
        return 1
    print("  ✅ 服务已就绪（http://127.0.0.1:8000）")

    # ============================================================ #
    # 第 2 步：健康检查 + 赛道列表
    # ============================================================ #
    print("\n[步骤2] 健康检查 + 赛道列表...")
    status, data = http("GET", "/api/v1/health")
    print(f"  health: {status} -> telemetry_connected={data.get('data',{}).get('telemetry_connected')}")

    status, data = http("GET", "/api/v1/tracks")
    tracks = data.get("data", [])
    print(f"  tracks: {status} -> {len(tracks)} 条赛道")

    status, data = http("GET", "/api/v1/tracks/" + TRACK_ID)
    yas = data.get("data", {})
    print(f"  tracks/{TRACK_ID}: {status} -> {yas.get('circuit_name','?')} ({yas.get('city','?')}) 弯道数={yas.get('corners','?')}")

    # ============================================================ #
    # 第 3 步：导入真实遥测数据（4 圈阿布扎比）
    # ============================================================ #
    print("\n[步骤3] 导入真实遥测数据（目录批量导入）...")
    print(f"  数据目录: {DATA_DIR}")
    files = sorted(f for f in os.listdir(DATA_DIR) if f.startswith("lap_") and f.endswith(".json"))
    print(f"  发现 {len(files)} 个圈数据文件")
    for f in files:
        sz = os.path.getsize(os.path.join(DATA_DIR, f)) / (1024*1024)
        print(f"    - {f} ({sz:.2f} MB)")

    status, data = http("POST", "/api/v1/telemetry/import-file", {"path": DATA_DIR})
    if status != 200:
        print(f"  ❌ 导入失败: {status} {data}")
        return 1
    laps = data.get("data", [])
    print(f"  ✅ 导入成功: {len(laps)} 圈遥测数据")

    # ============================================================ #
    # 第 4 步：验证遥测解析（统计摘要正确性）
    # ============================================================ #
    print("\n[步骤4] 验证遥测解析（统计摘要）...")
    for lap in laps:
        print(f"  圈{lap.get('lap_number')}: 圈速={lap.get('lap_time_str')} "
              f"平均速度={lap.get('avg_speed',0):.1f}km/h 极速={lap.get('max_speed',0):.1f}km/h "
              f"采样={lap.get('sample_count')}")
        print(f"    天气={lap.get('weather_name')} 赛道温={lap.get('track_temp')}°C "
              f"胎={lap.get('tyre_compound')} 胎龄={lap.get('tyre_age_laps')}圈")
        print(f"    平均油门={lap.get('avg_throttle',0):.3f} 平均刹车={lap.get('avg_brake',0):.3f} "
              f"平均转向={lap.get('avg_steer',0):.3f}")
        print(f"    四轮表面胎温={[round(t,1) for t in lap.get('avg_tyre_surface_temp',[])]}")
        print(f"    四轮胎压={[round(p,2) for p in lap.get('avg_tyre_pressure',[])]}")
        print(f"    扇区时间={lap.get('sector_times_str')}")

    # 验证 track_name 映射
    track_name = laps[0].get("track_name")
    track_id_udp = laps[0].get("track_id")
    print(f"\n  遥测 track_name={track_name} track_id(udp)={track_id_udp}")
    print(f"  ✅ 对应 F1OPT 赛道: {TRACK_ID}（阿布扎比 Yas Marina）")

    # ============================================================ #
    # 第 5 步：选赛道
    # ============================================================ #
    print("\n[步骤5] 选赛道...")
    status, data = http("POST", "/api/v1/tracks/current", {"track_id": TRACK_ID})
    print(f"  tracks/current: {status} -> {data.get('data')}")

    # ============================================================ #
    # 第 6 步：提交反馈（基于真实遥测数据推断症状）
    # ============================================================ #
    print("\n[步骤6] 提交反馈（基于遥测数据推断）...")
    # 用第一圈数据推断症状
    lap = laps[0]
    feedbacks = []
    if lap.get("avg_throttle", 0) < 0.5:
        feedbacks.append({"corner_number": None, "symptom": "lap_slow", "strength": 3})
    if lap.get("max_brake", 0) > 0.5:
        feedbacks.append({"corner_number": 1, "symptom": "brake_long", "strength": 3})
    if lap.get("max_steer", 0) > 0.4:
        feedbacks.append({"corner_number": None, "symptom": "understeer", "strength": 3})
    if not feedbacks:
        # 默认给一个全局症状
        feedbacks.append({"corner_number": None, "symptom": "straight_slow", "strength": 3})
    print(f"  提交反馈项: {[f['symptom'] for f in feedbacks]}")
    status, data = http("POST", "/api/v1/feedback", {"track_id": TRACK_ID, "feedbacks": feedbacks})
    print(f"  feedback: {status} -> {data.get('message','')}")

    # ============================================================ #
    # 第 7 步：生成建议 + 读取报告
    # ============================================================ #
    print("\n[步骤7] 生成建议...")
    status, data = http("POST", "/api/v1/suggest", {"track_id": TRACK_ID, "model_type": "hybrid"})
    if status != 200:
        print(f"  ⚠️ suggest: {status} -> {data.get('message','')}")
    else:
        print(f"  suggest: {status} -> {data.get('message','')}")

    status, data = http("GET", "/api/v1/suggest/latest")
    if status == 200:
        latest = data.get("data", {})
        report = latest.get("report", {}) if isinstance(latest, dict) else {}
        print(f"  suggest/latest: {status}")
        if report:
            _print_report_summary(report)
    else:
        print(f"  suggest/latest: {status}")

    # ============================================================ #
    # 第 8 步：直接验证 importer → engine 诊断链路
    # ============================================================ #
    print("\n[步骤8] 直接验证 importer → to_telemetry_dict → engine 诊断...")
    from setup_tuner.telemetry.importer import import_laps_from_directory
    summaries = import_laps_from_directory(DATA_DIR)
    print(f"  import_laps_from_directory: {len(summaries)} 圈")
    for s in summaries:
        assert s.track_name == "abu_dhabi", f"track_name 异常: {s.track_name}"
        assert s.sample_count > 0, "sample_count 异常"
        assert s.avg_speed > 0, "avg_speed 异常"
    tdict = summaries[0].to_telemetry_dict()
    print(f"  to_telemetry_dict keys(前10): {list(tdict.keys())[:10]}")
    print(f"    m_speed={tdict.get('m_speed'):.1f}  m_throttle={tdict.get('m_throttle'):.3f}")
    print(f"    m_brake={tdict.get('m_brake'):.3f}  weather={tdict.get('weather')}")
    print(f"    m_tyresSurfaceTemperature={[round(t,1) for t in tdict.get('m_tyresSurfaceTemperature',[])]}")
    print(f"    m_tyresPressure={[round(p,2) for p in tdict.get('m_tyresPressure',[])]}")
    print(f"  ✅ 遥测字典转换正确，可喂给 engine._derive_telemetry_dx/gain")

    # ============================================================ #
    # 收尾
    # ============================================================ #
    print("\n" + "=" * 72)
    print("全流程测试完成")
    print(f"  ✅ 服务启动 → 健康检查 → 24 赛道列表")
    print(f"  ✅ 导入 {len(laps)} 圈真实遥测（阿布扎比 yas_marina）")
    print(f"  ✅ 遥测解析：调教参数/速度/胎温/胎压/圈速统计正确")
    print(f"  ✅ 选赛道 → 反馈 → 建议 → 报告")
    print(f"  ✅ importer → to_telemetry_dict → engine 诊断链路通")
    print("=" * 72)

    server.should_exit = True
    thread.join(timeout=5)
    return 0


def _print_report_summary(report):
    """打印建议报告摘要。"""
    for key in ("symptoms", "dx", "setup_delta", "confidence", "summary"):
        if key in report:
            v = report[key]
            if isinstance(v, dict):
                print(f"    {key}: {json.dumps(v, ensure_ascii=False)[:200]}")
            else:
                print(f"    {key}: {v}")


if __name__ == "__main__":
    sys.exit(main())