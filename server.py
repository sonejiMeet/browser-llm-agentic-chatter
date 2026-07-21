"""
server.py — Minimal OpenAI-compatible API server for the browser LLM agent.

One browser session. One request at a time. Streaming SSE with live agent events.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
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
from agent_core import AgentLoop, AgentEvent, extract_hermes_user_message, build_system_prompt
from privacy import redact_text, workspace_label


# ── globals ────────────────────────────────────────────────────────

_loop: asyncio.AbstractEventLoop | None = None
_bridge: BrowserBridge | None = None
_agent: AgentLoop | None = None
_busy = threading.Lock()
_last_user_msg: str | None = None
_last_response: str | None = None


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
    # Discard the LLM's initial acknowledgement so it doesn't
    # leak into the first user turn.
    try:
        await _bridge.wait_for_response()
    except Exception:
        pass
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
            stream = body.get("stream", False)
            model_req = body.get("model", "?")
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

        global _last_user_msg, _last_response
        if user_msg == _last_user_msg and _last_response is not None:
            print(f"  [cached] duplicate user message — returning last response")
            if stream:
                self._stream_response(model_req, _last_response)
            else:
                self._json_response(model_req, _last_response)
            return

        if not _busy.acquire(blocking=False):
            self._json(503, {"error": {"message": "Server busy"}})
            return

        try:
            if stream:
                self._stream_agent_loop(model_req, user_msg)
            else:
                response = _run(_run_turn(user_msg), timeout=600)
                print(f"[response] {response[:120].replace(chr(10), ' ')}...")
                _last_user_msg = user_msg
                _last_response = response
                self._json_response(model_req, response)
        except Exception as e:
            print(f"[error] {e}")
            try:
                self._json(500, {"error": {"message": str(e)}})
            except Exception:
                pass
        finally:
            _busy.release()

    def do_GET(self):
        if self.path in ("/v1/models", "/v1/models/"):
            self._json(200, {"object": "list", "data": [{"id": "browser-agent", "object": "model"}]})
        elif self.path in ("/health", "/v1/health"):
            self._json(200, {"status": "ok" if _bridge else "starting"})
        else:
            self._json(404, {"error": {"message": "Not found"}})

    # ── streaming agent loop ────────────────────────────────────

    def _stream_agent_loop(self, model_req: str, user_msg: str):
        """Run the agent turn, pushing live events as SSE deltas to Hermes."""
        q: queue.Queue = queue.Queue()

        # Schedule the agent on the asyncio event loop
        asyncio.run_coroutine_threadsafe(_run_turn(user_msg, q), _loop)

        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        full_text = ""
        first = True
        try:
            while True:
                try:
                    chunk = q.get(timeout=600)
                except queue.Empty:
                    break
                if chunk is None:       # sentinel — turn complete
                    break
                full_text += chunk
                delta = {"content": chunk}
                if first:
                    delta["role"] = "assistant"
                    first = False
                sse = {
                    "id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": model_req,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(sse, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # Hermes disconnected

        # Cache for dedup
        global _last_user_msg, _last_response
        _last_user_msg = user_msg
        _last_response = full_text.strip()

        # Final chunk
        final = {
            "id": cid, "object": "chat.completion.chunk",
            "created": created, "model": model_req,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        try:
            self.wfile.write(f"data: {json.dumps(final, ensure_ascii=False)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception:
            pass

    # ── response helpers ────────────────────────────────────────

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
        """One-shot SSE (for cached responses — still fast)."""
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

def _fmt_event(ev: AgentEvent) -> str:
    """Format an agent event for Hermes display.
    Uses plain ASCII — no markdown syntax that shows as garbage if Hermes
    renders streaming deltas raw. Indentation and [OK]/[ERR] tags give
    visual hierarchy without needing markdown rendering."""
    kind = ev.kind

    if kind == "status":
        return f"  ... {ev.text}\n"

    if kind == "tool_start":
        # ev.text already has prefix ($ cmd, write path, read path)
        return f"\n{ev.text}\n"

    if kind == "tool_result":
        ok = ev.data.get("ok", True)
        text = ev.text.strip()
        if ok:
            if "\n" in text:
                # Multi-line: indent each line, show [OK] header
                lines = text.split("\n")[:60]
                body = "\n".join(f"    {l}" for l in lines)
                return f"  [OK]\n{body}\n"
            return f"  [OK] {text[:250]}\n"
        else:
            return f"  [ERR] {text[:400]}\n"

    if kind == "response":
        text = ev.text.strip()
        if not text:
            return ""
        return f"\n{text}\n"

    if kind == "report":
        if ev.data.get("results"):
            return f"\n{ev.text}\n"
        return ""

    if kind == "done":
        return ""

    if kind == "error":
        return f"\n[ERROR] {ev.text}\n"

    return ""


async def _run_turn(user_msg: str, q: queue.Queue | None = None) -> str:
    """Run the agent loop. If *q* is given, push formatted string chunks
    for live SSE streaming. Always returns the final text for non-streaming
    callers (and for the server console log)."""
    parts: list[str] = []
    last_text = ""
    async for ev in _agent.run_turn(user_msg):
        # Server console log
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

        # Push to streaming queue
        if q is not None:
            chunk = _fmt_event(ev)
            if chunk:
                q.put(chunk)

    if q is not None:
        q.put(None)  # sentinel

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
