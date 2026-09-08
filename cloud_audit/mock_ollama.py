"""内置 LLM 端到端审计的 mock 服务器（Ollama 兼容, OpenAI chat 接口格式）。

模拟跑在 ``127.0.0.1:11434`` 的 Ollama OpenAI 兼容端点:

- ``POST /v1/chat/completions`` — ``stream=false`` 返回整段 JSON;
  ``stream=true`` 返回 SSE ``data:`` 分片流 (含最终 usage 块与 [DONE])。
- ``GET /healthz`` — 探活。

固定返回带 ``[MOCK-LLM]`` 标记的内容, 供 UI 审计断言「LLM 增强确实生效」。
仅服务本地回环, 由云端审计工作流以子进程方式拉起/终止。
"""

from __future__ import annotations

import json
import os
import pathlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = "127.0.0.1"
PORT = int(os.environ.get("MOCK_LLM_PORT", "11434"))
MARKER = "[MOCK-LLM]"
CONTENT = (
    MARKER + " 内置LLM链路正常。针对该弯道推头: 建议前翼 -1、后翼 +1、"
    "入弯更晚刹车, 并将差速锁适当调低以改善转向初段响应。"
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _chat_json(payload: dict) -> dict:
    return {
        "id": "chatcmpl-mock-001",
        "object": "chat.completion",
        "model": payload.get("model", "llama3.1"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": CONTENT},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 128, "completion_tokens": 64, "total_tokens": 192},
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # 静默访问日志
        pass

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._send(200, b"ok")
            return
        self._send(404, b"not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._send(404, b"not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._send(400, b"bad json")
            return
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            # 三个内容分片 + 最终 usage 块 + [DONE], 与 OpenAI 流式一致。
            parts = [MARKER, " 内置LLM链路正常。", "推头对策: 前翼-1, 后翼+1。"]
            for p in parts:
                chunk = {
                    "id": "chatcmpl-mock-001",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            final = {
                "id": "chatcmpl-mock-001",
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": {"prompt_tokens": 128, "completion_tokens": 64, "total_tokens": 192},
            }
            self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        body = json.dumps(_chat_json(payload)).encode()
        self._send(200, body, ctype="application/json")

    def _send(self, code: int, body: bytes, ctype: str = "text/plain") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    # 写 PID 供审计脚本精准终止 (作为测试夹具生命周期管理)。
    pid_file = ROOT / "reports" / "mock_ollama.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()), encoding="ascii")
    print(f"mock ollama listening on http://{HOST}:{PORT} (pid {os.getpid()})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
