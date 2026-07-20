"""
server.py — Minimal OpenAI-compatible API server for the browser LLM agent.

One browser session. One request at a time. No streaming, no polling hacks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
import uuid
import yaml
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from browser import BrowserBridge
from tools import ToolExecutor
from agent_core import AgentLoop, extract_hermes_user_message, build_system_prompt
from privacy import redact_text, workspace_label


# ── globals ────────────────────────────────────────────────────────

_loop: asyncio.AbstractEventLoop | None = None
_bridge: BrowserBridge | None = None
_agent: AgentLoop | None = None
_busy = threading.Lock()


def _run(coro, timeout=300):
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result(timeout=timeout)


def _start_loop():
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop.run_forever()


# ── init ────────────────────────────────────────────────────────────

async def _init(config: dict):
    global _bridge, _agent
    _bridge = BrowserBridge(config)
    _agent = AgentLoop(_bridge, ToolExecutor(config), config)
    await _bridge.start()
    await _bridge.ensure_logged_in()
    if config.get("model"):
        await _bridge.select_model(config["model"])
    prompt = build_system_prompt(config)
    prompt = redact_text(prompt)
    await _bridge.send_message(prompt)
    print("[server] Agent ready.")


# ── handler ─────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/v1/chat/completions/"):
            self._json(404, {"error": {"message": "Not found"}})
            return

        try:
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            body = json.loads(raw)
            # Log what Hermes actually sends (model name, stream flag)
            stream = body.get("stream", False)
            model_req = body.get("model", "?")
            # Only print this once per session to avoid spam
            if not hasattr(Handler, "_logged_request"):
                print(f"[debug] Hermes requests: model={model_req} stream={stream}")
                Handler._logged_request = True
        except Exception:
            self._json(400, {"error": {"message": "Invalid JSON"}})
            return

        user_msg, _ = extract_hermes_user_message(body.get("messages", []))
        if not user_msg:
            self._json(400, {"error": {"message": "No user message"}})
            return

        print(f"\n[request] {user_msg[:100].replace(chr(10), ' ')}...")

        if not _busy.acquire(blocking=False):
            self._json(503, {"error": {"message": "Server busy"}})
            return

        try:
            response = _run(_run_turn(user_msg), timeout=600)
        except Exception as e:
            print(f"[error] {e}")
            self._json(500, {"error": {"message": str(e)}})
            _busy.release()
            return

        print(f"[response] {response[:120].replace(chr(10), ' ')}...")
        self._json_response(model_req, response)
        _busy.release()

    def do_GET(self):
        if self.path in ("/v1/models", "/v1/models/"):
            self._json(200, {"object": "list", "data": [{"id": "browser-agent", "object": "model"}]})
        elif self.path in ("/health", "/v1/health"):
            self._json(200, {"status": "ok" if _bridge else "starting"})
        else:
            self._json(404, {"error": {"message": "Not found"}})

    def _json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_response(self, model_name, text):
        self._json(200, {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion", "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(text)//4, "completion_tokens": len(text)//4, "total_tokens": len(text)//2},
        })

    def _stream_response(self, model_name, text):
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                  "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]}
        self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
        final = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                  "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        self.wfile.write(f"data: {json.dumps(final, ensure_ascii=False)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *args): pass


# ── agent turn ──────────────────────────────────────────────────────

async def _run_turn(user_msg: str) -> str:
    """Run the agent loop and return the final report text."""
    parts: list[str] = []
    last_text = ""
    async for ev in _agent.run_turn(user_msg):
        if ev.kind == "status":
            print(f"  [{ev.kind}] {ev.text}")
        elif ev.kind == "tool_start":
            print(f"  [tool>] {ev.text}")
            parts.append(f"\n$ {ev.text}")
        elif ev.kind == "tool_result":
            ok = ev.data.get("ok", True)
            print(f"  [tool<{'ok' if ok else 'ERR'}] {ev.text[:120].replace(chr(10), ' ')}")
        elif ev.kind == "response":
            print(f"  [llm] {ev.text[:100].replace(chr(10), ' ')}...")
            if ev.text.strip():
                parts.append(ev.text)
        elif ev.kind in ("done", "report", "error"):
            last_text = ev.text
    return last_text or "\n".join(parts)


# ── main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", "-p", help="chatgpt, claude, perplexity")
    parser.add_argument("--model", "-m")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workspace", "-w")
    args = parser.parse_args()

    config = yaml.safe_load(open(Path(__file__).parent / "config.yaml"))
    if args.provider: config["provider"] = args.provider
    if args.model: config["model"] = args.model
    if args.workspace:
        import os; os.chdir(Path(args.workspace).expanduser().resolve())

    threading.Thread(target=_start_loop, daemon=True).start()
    time.sleep(0.2)
    _run(_init(config), timeout=180)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n[server] http://localhost:{args.port}/v1")
    print("[server] hermes config set model.provider custom")
    print(f"[server] hermes config set model.base_url http://localhost:{args.port}/v1")
    print("[server] hermes config set model.api_key noop\n")

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        _run(_bridge.close(), timeout=10)
        _loop.call_soon_threadsafe(_loop.stop)


if __name__ == "__main__":
    main()
