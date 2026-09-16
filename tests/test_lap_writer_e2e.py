# -*- coding: utf-8 -*-
"""task-62 落库线程端到端验收：喂 3000+ 帧 → 圈结束 → lap_record/driver_style 落库。"""
from __future__ import annotations

import time

from setup_tuner.app import _init_app_services, _make_packet_handler, create_app
from setup_tuner.config import Config


def _frame(speed: float = 200.0) -> dict:
    return {
        "packet_id": 6, "m_speed": speed, "m_throttle": 0.7, "m_brake": 0.0,
        "m_steer": 0.1,
        "m_tyresSurfaceTemperature": [95, 97, 92, 94],
        "m_brakesTemperature": [420, 430, 400, 410],
    }


def _boot(tmp_path) -> tuple:
    app = create_app(Config(data_dir=str(tmp_path), udp_port=0))
    _init_app_services(app, Config(data_dir=str(tmp_path), udp_port=0))
    app.state.current_track_id = "suzuka"
    return app, _make_packet_handler(app)


def test_lap_persistence_end_to_end(tmp_path) -> None:
    """整圈跑完 → lap_record + driver_style 落库（12 维），含质量门槛。"""
    app, handler = _boot(tmp_path)
    handler({"packet_id": 2, "m_currentLapNum": 1, "m_sector": 0})
    for i in range(3000):
        handler(_frame(speed=150 + (i % 200)))
    handler({"packet_id": 2, "m_currentLapNum": 2, "m_sector": 0,
             "m_lastLapTimeInMS": 90000})

    store = app.state.store
    for _ in range(30):
        if store.get_lap_records("suzuka", 1):
            break
        time.sleep(0.1)

    records = store.get_lap_records("suzuka", 1)
    assert len(records) == 1, "整圈应落库 1 条"
    assert records[0]["lap_time_ms"] == 90000
    import json
    assert json.loads(records[0]["telemetry_json"])["lap_frames"] == 3000

    style = store.get_driver_style(1, "suzuka")
    assert style is not None, "风格向量应落库"
    assert len(style["vector"]) == 12, "风格向量应 12 维（含圈速一致性）"
    assert style["sample_count"] == 1


def test_lap_below_threshold_skipped(tmp_path) -> None:
    """样本帧数不足门槛 → 不入库（防垃圾圈）。"""
    app, handler = _boot(tmp_path)
    handler({"packet_id": 2, "m_currentLapNum": 1})
    for _ in range(100):
        handler(_frame())
    handler({"packet_id": 2, "m_currentLapNum": 2})
    time.sleep(0.5)
    assert app.state.store.get_lap_records("suzuka", 1) == []
    assert app.state.store.get_driver_style(1, "suzuka") is None
