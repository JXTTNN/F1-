"""F1OPT 性能基准测试 —— 测量 API/引擎/遥测关键路径耗时。

运行方式：
    python -m scripts.perf_benchmark

测量项：
    1. API 端点响应时间（health/tracks/feedback/suggest）
    2. 规则引擎计算时间（Dx → SetupDelta → 报告 完整链路）
    3. 遥测解析性能（6 类 UDP 包解析速度）
    4. 耦合矩阵查询性能
    5. SQLite Store 操作性能

输出：每项测量的 min/mean/max/p95 耗时（ms）与吞吐（ops/s）。
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _stats(samples: list[float]) -> dict[str, float]:
    """计算样本统计量（ms）。"""
    if not samples:
        return {"count": 0}
    sorted_s = sorted(samples)
    n = len(sorted_s)
    p95_idx = min(n - 1, int(n * 0.95))
    return {
        "count": n,
        "min_ms": round(sorted_s[0] * 1000, 4),
        "mean_ms": round(statistics.mean(samples) * 1000, 4),
        "median_ms": round(statistics.median(samples) * 1000, 4),
        "max_ms": round(sorted_s[-1] * 1000, 4),
        "p95_ms": round(sorted_s[p95_idx] * 1000, 4),
        "p99_ms": round(sorted_s[min(n - 1, int(n * 0.99))] * 1000, 4),
        "ops_per_sec": round(n / sum(samples), 1) if sum(samples) > 0 else 0,
    }


def bench_engine() -> dict[str, dict]:
    """维度 2：规则引擎计算性能。"""
    from setup_tuner.domain.setup import CarSetup
    from setup_tuner.domain.symptoms import Symptom
    from setup_tuner.engine.engine import generate_suggestion

    default_setup = CarSetup.default().to_dict()
    all_symptoms = [s.value for s in Symptom]

    # 单症状建议生成
    samples_single: list[float] = []
    for _ in range(2000):
        for sym in all_symptoms:
            t0 = time.perf_counter()
            generate_suggestion([(sym, 3)], default_setup, "perf_track", None)
            samples_single.append(time.perf_counter() - t0)

    # 多症状叠加（4 症状）
    multi_symptoms = [("understeer", 3), ("oversteer", 2), ("brake_long", 4), ("exit_wheelspin", 3)]
    samples_multi: list[float] = []
    for _ in range(5000):
        t0 = time.perf_counter()
        generate_suggestion(multi_symptoms, default_setup, "perf_track", None)
        samples_multi.append(time.perf_counter() - t0)

    # 带遥测的建议生成
    telemetry = {"m_weather": 1, "m_tyresAgeLaps": 5, "m_lastLapTimeInMS": 90000}
    samples_telemetry: list[float] = []
    for _ in range(5000):
        t0 = time.perf_counter()
        generate_suggestion(multi_symptoms, default_setup, "perf_track", telemetry)
        samples_telemetry.append(time.perf_counter() - t0)

    return {
        "single_symptom_12x": _stats(samples_single),
        "multi_symptom_4x": _stats(samples_multi),
        "with_telemetry": _stats(samples_telemetry),
    }


def bench_telemetry_parse() -> dict[str, dict]:
    """维度 3：遥测包解析性能。"""
    from setup_tuner.telemetry import packets

    # 构造 6 类包的样本数据
    header = bytes([
        0xE8, 0x07,  # packetFormat=2026
        26, 1, 0, 1,  # gameYear, major, minor, packetVersion
        0,  # packetId 占位（下面按包设置）
        0, 0, 0, 0, 0, 0, 0, 0,  # sessionUID
        0, 0, 0, 0,  # sessionTime
        0, 0, 0, 0,  # frameIdentifier
        0, 0, 0, 0,  # overallFrameIdentifier
        0,  # playerCarIndex
        255,  # secondaryPlayerCarIndex
    ])
    # 修正：用 struct 构造 header
    import struct
    def make_header(packet_id: int, player_idx: int = 0) -> bytes:
        return struct.pack(
            "<HBBBBBQfIIBB",
            2026, 26, 1, 0, 1, packet_id,
            0, 0.0, 0, 0, player_idx, 255,
        )

    # Packet 1 Session — 构造最小合法包
    # 16 字段 + 21 marshal zones (21*5=105) + 3 字段
    # 格式 BbbBHBbBHHBBBBBB = 16 字段：
    #   weather, track_temp, air_temp, total_laps, track_len,
    #   session_type, track_id, formula, session_time_left, session_duration,
    #   pit_speed_limit, game_paused, is_spectating, spectator_car_idx,
    #   sli_pro, num_marshal
    session_body = struct.pack(
        "<BbbBHBbBHHBBBBBB",
        0, 25, 22, 58, 5303, 5, 0, 1, 0, 0, 100, 0, 0, 0, 0, 0,
    )
    # 21 marshal zones (float + uint8 = 5 bytes each)
    for _ in range(21):
        session_body += struct.pack("<fb", 0.0, 0)
    session_body += struct.pack("<BBB", 0, 0, 0)  # sc, network, numWfs
    packet1 = make_header(1) + session_body

    # Packet 2 LapData — 玩家车在 index 0
    lap_per = struct.pack(
        "<IIHBHBHBHBfffBBBBBBBBBBBBBBBHHBfB",
        90000, 45000, 30000, 0, 30000, 0, 30000, 0, 30000, 0,
        1500.0, 1500.0, 0.0,
        1, 5, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0.0, 0,
    )
    packet2 = make_header(2, 0) + lap_per

    # Packet 5 CarSetups — 玩家车在 index 0
    setup_per = struct.pack(
        "<BBBBffffBBBBBBBBBffffBf",
        5, 3, 50, 80,
        -3.5, -2.0, 0.05, 0.10,
        6, 6, 11, 11, 3, 3, 100, 53, 12,
        27.0, 27.0, 27.0, 27.0,
        50, 30.0,
    )
    packet5 = make_header(5, 0) + setup_per

    # Packet 6 CarTelemetry — 玩家车在 index 0
    telem_per = struct.pack(
        "<HfffBbHBBH4H4B4BB4f4B",
        250, 0.8, 0.0, 0.2, 0, 5, 11000, 0, 50, 0xAAAA,
        400, 450, 430, 440,
        90, 92, 91, 93,
        110, 112, 111, 113,
        105,
        27.0, 27.0, 27.0, 27.0,
        0, 0, 0, 0,
    )
    packet6 = make_header(6, 0) + telem_per

    # Packet 7 CarStatus — 玩家车在 index 0
    # 格式 BBBBBfffHHBBHBBBbfffBffffB = 26 字段
    status_per = struct.pack(
        "<BBBBBfffHHBBHBBBbfffBffffB",
        1, 1, 2, 53, 0,
        30.0, 100.0, 30.0,
        12000, 4000,
        8, 1, 0,
        1, 1, 5,
        0,
        800.0, 60.0, 1000000.0,
        0,
        0.0, 0.0, 1000000.0, 0.0,
        0,
    )
    packet7 = make_header(7, 0) + status_per

    # Packet 16 CarTelemetryData2 — 玩家车在 index 0
    ct2_per = struct.pack("<BBHBBHBB", 0, 1, 50, 1, 0, 100, 1, 0)
    packet16 = make_header(16, 0) + ct2_per

    test_packets = {
        "packet1_session": packet1,
        "packet2_lapdata": packet2,
        "packet5_setups": packet5,
        "packet6_telemetry": packet6,
        "packet7_status": packet7,
        "packet16_ct2": packet16,
    }

    results: dict[str, dict] = {}
    for name, data in test_packets.items():
        samples: list[float] = []
        for _ in range(10000):
            t0 = time.perf_counter()
            packets.parse_packet(data)
            samples.append(time.perf_counter() - t0)
        results[name] = _stats(samples)

    # 混合解析（模拟 60Hz 流）
    all_packets = list(test_packets.values())
    samples_mixed: list[float] = []
    for _ in range(2000):
        for data in all_packets:
            t0 = time.perf_counter()
            packets.parse_packet(data)
            samples_mixed.append(time.perf_counter() - t0)
    results["mixed_60hz_sim"] = _stats(samples_mixed)

    return results


def bench_coupling_matrix() -> dict[str, dict]:
    """耦合矩阵查询性能。"""
    from setup_tuner.engine.coupling import (
        COUPLING_MATRIX,
        get_coupling,
        get_column,
        get_row,
        matrix_stats,
        nonzero_cells_for_diag,
        nonzero_cells_for_param,
    )
    from setup_tuner.engine.diagnostic import DIAG_DIMS
    from setup_tuner.domain.setup import ALL_SETUP_FIELDS

    param_names = [f.name for f in ALL_SETUP_FIELDS]

    # 单格查询
    samples: list[float] = []
    for _ in range(20000):
        for d in DIAG_DIMS:
            for p in param_names:
                t0 = time.perf_counter()
                get_coupling(d, p)
                samples.append(time.perf_counter() - t0)

    # 整行查询
    samples_row: list[float] = []
    for _ in range(20000):
        for d in DIAG_DIMS:
            t0 = time.perf_counter()
            get_row(d)
            samples_row.append(time.perf_counter() - t0)

    # 整列查询
    samples_col: list[float] = []
    for _ in range(20000):
        for p in param_names:
            t0 = time.perf_counter()
            get_column(p)
            samples_col.append(time.perf_counter() - t0)

    # 非零单元查询
    samples_nz: list[float] = []
    for _ in range(20000):
        for d in DIAG_DIMS:
            t0 = time.perf_counter()
            nonzero_cells_for_diag(d)
            samples_nz.append(time.perf_counter() - t0)

    return {
        "single_cell": _stats(samples),
        "full_row": _stats(samples_row),
        "full_column": _stats(samples_col),
        "nonzero_for_diag": _stats(samples_nz),
        "matrix_stats": matrix_stats(),
    }


def bench_store() -> dict[str, dict]:
    """SQLite Store 操作性能。"""
    import tempfile

    from setup_tuner.db.store import Store
    from setup_tuner.domain.setup import CarSetup

    with tempfile.TemporaryDirectory() as tmpdir:
        store = Store(str(Path(tmpdir) / "perf.db"))
        default_params = CarSetup.default().to_dict()

        # import_setup
        samples_import: list[float] = []
        for i in range(500):
            t0 = time.perf_counter()
            store.import_setup("suzuka", default_params)
            samples_import.append(time.perf_counter() - t0)

        # get_latest_setup
        samples_get: list[float] = []
        for _ in range(5000):
            t0 = time.perf_counter()
            store.get_latest_setup("suzuka")
            samples_get.append(time.perf_counter() - t0)

        # save_suggestion
        report = {"track_id": "suzuka", "summary": "perf test"}
        report_json = json.dumps(report)
        samples_save: list[float] = []
        for _ in range(500):
            t0 = time.perf_counter()
            store.save_suggestion("suzuka", report_json, setup_id=1)
            samples_save.append(time.perf_counter() - t0)

        # get_latest_suggestion
        samples_get_sug: list[float] = []
        for _ in range(5000):
            t0 = time.perf_counter()
            store.get_latest_suggestion("suzuka")
            samples_get_sug.append(time.perf_counter() - t0)

        store.close()

    return {
        "import_setup": _stats(samples_import),
        "get_latest_setup": _stats(samples_get),
        "save_suggestion": _stats(samples_save),
        "get_latest_suggestion": _stats(samples_get_sug),
    }


def bench_api_endpoints() -> dict[str, dict]:
    """API 端点响应时间（使用 TestClient，避免启动真实服务器）。"""
    import tempfile

    from fastapi.testclient import TestClient

    from setup_tuner.app import create_app
    from setup_tuner.config import Config

    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(data_dir=tmpdir, udp_host="127.0.0.1", udp_port=12077)
        app = create_app(config)
        client = TestClient(app)

        with client:
            # health
            samples_health: list[float] = []
            for _ in range(1000):
                t0 = time.perf_counter()
                client.get("/api/v1/health")
                samples_health.append(time.perf_counter() - t0)

            # tracks
            samples_tracks: list[float] = []
            for _ in range(1000):
                t0 = time.perf_counter()
                client.get("/api/v1/tracks")
                samples_tracks.append(time.perf_counter() - t0)

            # track detail
            samples_track_detail: list[float] = []
            for _ in range(1000):
                t0 = time.perf_counter()
                client.get("/api/v1/tracks/suzuka")
                samples_track_detail.append(time.perf_counter() - t0)

            # select track + feedback + suggest
            client.post("/api/v1/tracks/current", json={"track_id": "suzuka"})
            for _ in range(5):
                client.post(
                    "/api/v1/feedback",
                    json={"track_id": "suzuka", "corner_number": 1, "symptom": "understeer", "strength": 3},
                )

            samples_suggest: list[float] = []
            for _ in range(200):
                t0 = time.perf_counter()
                client.post("/api/v1/suggest", json={"track_id": "suzuka"})
                samples_suggest.append(time.perf_counter() - t0)

            samples_feedback: list[float] = []
            for _ in range(1000):
                t0 = time.perf_counter()
                client.post(
                    "/api/v1/feedback",
                    json={"track_id": "suzuka", "corner_number": 2, "symptom": "oversteer", "strength": 3},
                )
                samples_feedback.append(time.perf_counter() - t0)

            samples_get_feedback: list[float] = []
            for _ in range(1000):
                t0 = time.perf_counter()
                client.get("/api/v1/feedback?track_id=suzuka")
                samples_get_feedback.append(time.perf_counter() - t0)

    return {
        "GET /health": _stats(samples_health),
        "GET /tracks": _stats(samples_tracks),
        "GET /tracks/suzuka": _stats(samples_track_detail),
        "POST /feedback": _stats(samples_feedback),
        "GET /feedback": _stats(samples_get_feedback),
        "POST /suggest": _stats(samples_suggest),
    }


def main() -> None:
    """运行所有基准测试并输出报告。"""
    print("=" * 80)
    print("F1OPT 性能基准测试报告")
    print("=" * 80)

    print("\n[1/5] 规则引擎计算性能 ...")
    engine_results = bench_engine()
    for name, stats in engine_results.items():
        print(f"  {name}: {stats}")

    print("\n[2/5] 遥测包解析性能 ...")
    telemetry_results = bench_telemetry_parse()
    for name, stats in telemetry_results.items():
        print(f"  {name}: {stats}")

    print("\n[3/5] 耦合矩阵查询性能 ...")
    coupling_results = bench_coupling_matrix()
    for name, stats in coupling_results.items():
        print(f"  {name}: {stats}")

    print("\n[4/5] SQLite Store 操作性能 ...")
    store_results = bench_store()
    for name, stats in store_results.items():
        print(f"  {name}: {stats}")

    print("\n[5/5] API 端点响应时间 ...")
    api_results = bench_api_endpoints()
    for name, stats in api_results.items():
        print(f"  {name}: {stats}")

    print("\n" + "=" * 80)
    print("基准测试完成。")

    # 汇总输出 JSON 供后续分析
    report = {
        "engine": engine_results,
        "telemetry": telemetry_results,
        "coupling": coupling_results,
        "store": store_results,
        "api": api_results,
    }
    out_path = ROOT / "scripts" / "perf_benchmark_result.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"详细结果已写入：{out_path}")


if __name__ == "__main__":
    main()