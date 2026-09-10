"""T7 WebSocket 端点冒烟测试 —— 验证 WS 连接 + 事件推送。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from setup_tuner.app import create_app
from setup_tuner.config import Config

TEST_DATA_DIR = Path("./data_test_t7_ws")


def test_ws_connect():
    """WebSocket 能连接并接收 telemetry_status 事件。"""
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)
    config = Config(data_dir=str(TEST_DATA_DIR))
    app = create_app(config)
    client = TestClient(app)

    with client.websocket_connect("/api/v1/ws") as ws:
        # 发送一个 select_track 消息
        ws.send_text(json.dumps({"action": "select_track", "track_id": "suzuka"}))
        # 接收响应
        msg = ws.receive_text()
        data = json.loads(msg)
        print(f"  WS received: event={data.get('event')}, payload={data.get('payload')}")
        assert data.get("event") == "track_selected"
        assert data["payload"]["track_id"] == "suzuka"
    print("✅ WebSocket 连接 + 消息收发 OK")


if __name__ == "__main__":
    test_ws_connect()
    print("\n🎉 WebSocket 测试通过！")