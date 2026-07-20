"""
server.py — OpenAI-compatible API server wrapping the browser LLM agent.

Talks to Hermes (or any OpenAI-compatible client). Streams agent activity:
running commands, git-like file diffs, cleaned chat responses, and status.

Uses a persistent asyncio event loop in a background thread so all Playwright
operations share the same loop (Playwright objects are loop-bound).

Usage:
    python server.py
    python server.py --provider chatgpt --port 8765

Hermes:
    hermes config set model.provider custom
    hermes config set model.base_url http://localhost:8765/v1
    hermes config set model.api_key noop
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
from queue import Queue, Empty
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from browser import BrowserBridge
from tools import ToolExecutor
from agent_core import (
    AgentLoop,
    AgentEvent,
    extract_hermes_user_message,
)
from privacy import workspace_label


# ── persistent event loop (Playwright objects are loop-bound) ──────

_loop: asyncio.AbstractEventLoop | None = None
_bridge: BrowserBridge | None = None
_tools: ToolExecutor | None = None
_agent: AgentLoop | None = None
_config: dict = {}
_lock = threading.Lock()
_last_request: tuple[str, float, str] = ("", 0, "")  # (user_msg, timestamp, cached_response)


def _run_in_loop(coro, timeout: int = 600):
    """Submit a coroutine to the persistent event loop and wait for result."""
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result(timeout=timeout)


def _start_event_loop():
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop.run_forever()


# ── agent init ─────────────────────────────────────────────────────

async def _init_agent(config: dict):
    global _bridge, _tools, _agent, _config
    _config = config
    _bridge = BrowserBridge(config)
    _tools = ToolExecutor(config)
    _agent = AgentLoop(_bridge, _tools, config)

    print("[server] Launching browser...")
    await _bridge.start()
    await _bridge.ensure_logged_in()

    model = config.get("model", "")
    if model:
        await _bridge.select_model(model)

    print("[server] Priming chat with system prompt...")
    await _agent.prime()
    print("[server] Agent ready. Browser session active.")
    # Local log only — never send full path to the browser LLM
    print(f"[server] Workspace label: {workspace_label()}")


async def _collect_turn(user_message: str, extra_context: str = "") -> str:
    """Run a full agent turn and return the final report for non-streaming."""
    final = ""
    async for event in _agent.run_turn(user_message, extra_context=extra_context):
        _log_event(event)
        if event.kind in ("done", "error", "report"):
            final = event.text
    return final or "(no response)"


async def _stream_turn(user_message: str, extra_context: str, out_queue: Queue):
    """Run agent turn and push text chunks + control messages onto a queue."""
    try:
        final_text = ""
        had_tools = False
        async for event in _agent.run_turn(user_message, extra_context=extra_context):
            _log_event(event)
            if event.kind in ("tool_start", "tool_result"):
                had_tools = True
            chunk = _event_to_stream_text(event)
            if chunk:
                out_queue.put(("chunk", chunk))
            if event.kind in ("done", "error"):
                final_text = event.text or ""
                # Append compact change-log footer so Hermes always sees a summary
                if had_tools and _tools and _tools.change_log:
                    footer_lines = ["\n\n## Change Log"]
                    for entry in _tools.change_log[-40:]:
                        kind = entry.get("kind", "?")
                        if kind == "create":
                            footer_lines.append(
                                f"+ created  {entry.get('path')}  ({entry.get('bytes', 0)} bytes)"
                            )
                        elif kind == "modify":
                            footer_lines.append(
                                f"~ modified {entry.get('path')}  "
                                f"(+{entry.get('added', 0)} / -{entry.get('removed', 0)} lines)"
                            )
                        elif kind == "shell":
                            footer_lines.append(f"$ {entry.get('command', '')[:120]}")
                    out_queue.put(("chunk", "\n".join(footer_lines) + "\n"))
                out_queue.put(("final", final_text))
                return
        out_queue.put(("final", final_text))
    except Exception as e:
        out_queue.put(("error", str(e)))
    finally:
        out_queue.put(("end", None))


def _log_event(event: AgentEvent):
    if event.kind == "status":
        print(f"  [status] {event.text}")
    elif event.kind == "tool_start":
        print(f"  [tool>] {event.text}")
    elif event.kind == "tool_result":
        preview = event.text[:120].replace("\n", " ")
        ok = event.data.get("ok", True)
        mark = "ok" if ok else "ERR"
        print(f"  [tool<{mark}] {preview}")
    elif event.kind == "response":
        preview = event.text[:100].replace("\n", " ")
        print(f"  [llm] {preview}...")
    elif event.kind == "error":
        print(f"  [error] {event.text}")
    elif event.kind == "done":
        print(f"  [done] {len(event.text)} chars")


def _event_to_stream_text(event: AgentEvent) -> str:
    """Convert an agent event into text Hermes should see while streaming."""
    if event.kind == "status":
        return f"\n*{event.text}*\n"
    if event.kind == "tool_start":
        tool = event.data.get("tool", "tool")
        if tool == "shell":
            return f"\n```\n$ {event.data.get('command', event.text)}\n```\n"
        if tool == "file_write":
            return f"\n*Writing `{event.data.get('path', '?')}`...*\n"
        if tool == "file_read":
            return f"\n*Reading `{event.data.get('path', '?')}`...*\n"
        return f"\n*Running {tool}...*\n"
    if event.kind == "tool_result":
        tool = event.data.get("tool", "tool")
        if tool == "shell":
            body = event.text
            if len(body) > 2500:
                body = body[:2500] + "\n... [truncated]"
            return f"```\n{body}\n```\n"
        if tool == "file_write":
            # Prefer diff-style output
            return f"```diff\n{event.text[:4000]}\n```\n"
        if tool == "file_read":
            return f"```\n{event.text[:2000]}\n```\n"
        return f"{event.text[:1500]}\n"
    if event.kind == "response":
        # Cleaned LLM prose between tool rounds
        return f"\n{event.text}\n"
    if event.kind == "error":
        return f"\n**Error:** {event.text}\n"
    # 'report' and 'done' are handled as the final payload — avoid double-send
    # during stream by only emitting incremental events above.
    return ""


# ── HTTP handler ───────────────────────────────────────────────────


class AgentHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/v1/chat/completions/"):
            self._send_json(404, {"error": {"message": "Not found", "type": "invalid_request_error"}})
            return

        try:
            body_len = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(body_len)
            body = json.loads(raw)
        except Exception:
            self._send_json(400, {"error": {"message": "Invalid JSON", "type": "invalid_request_error"}})
            return

        stream = bool(body.get("stream", False))
        messages = body.get("messages", [])
        # Only the current user task — no Hermes history/system/private paths
        user_content, _extra = extract_hermes_user_message(messages)

        if not user_content:
            self._send_json(400, {"error": {"message": "No user message found", "type": "invalid_request_error"}})
            return

        model_req = body.get("model", "browser-agent")
        print(f"\n[request] model={model_req} stream={stream}")
        print(f"  user: {user_content[:120].replace(chr(10), ' ')}...")

        # Dedup: Hermes sends stream then non-stream for the same prompt.
        # Return the cached response instead of hitting the browser LLM twice.
        global _last_request
        now = time.time()
        if user_content == _last_request[0] and (now - _last_request[1]) < 30:
            print("  dedup: returning cached response (same prompt within 30s)")
            if stream:
                self._stream_cached(chat_id, model_req, created, _last_request[2])
            else:
                self._send_cached(chat_id, model_req, created, _last_request[2])
            return

        chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if stream:
            self._handle_stream(chat_id, model_req, created, user_content, "")
        else:
            self._handle_sync(chat_id, model_req, created, user_content, "")

    def _cache_response(self, user_msg: str, response_text: str):
        global _last_request
        _last_request = (user_msg, time.time(), response_text)

    def _send_cached(self, chat_id, model_name, created, text):
        self._send_json(200, {
            "id": chat_id, "object": "chat.completion", "created": created,
            "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    def _stream_cached(self, chat_id, model_name, created, text):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "keep-alive"); self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        chunk = text[:5000]
        delta = {"role": "assistant", "content": chunk}
        self.wfile.write(f"data: {json.dumps({'id':chat_id,'object':'chat.completion.chunk','created':created,'model':model_name,'choices':[{'index':0,'delta':delta,'finish_reason':None}]})}\n\n".encode())
        self.wfile.write(f"data: {json.dumps({'id':chat_id,'object':'chat.completion.chunk','created':created,'model':model_name,'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()

    def _handle_sync(self, chat_id, model_name, created, user_content, extra_context):
        try:
            with _lock:
                response_text = _run_in_loop(
                    _collect_turn(user_content, extra_context),
                    timeout=600,
                )
        except Exception as e:
            print(f"[error] {e}")
            self._send_json(500, {"error": {"message": str(e), "type": "server_error"}})
            return

        print(f"[response] {response_text[:120].replace(chr(10), ' ')}...")
        self._cache_response(user_content, response_text)
        self._send_json(200, {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        })

    def _handle_stream(self, chat_id, model_name, created, user_content, extra_context):
        """Stream OpenAI SSE chunks as the agent works."""
        q: Queue = Queue()

        def runner():
            try:
                with _lock:
                    _run_in_loop(
                        _stream_turn(user_content, extra_context, q),
                        timeout=600,
                    )
            except Exception as e:
                q.put(("error", str(e)))
                q.put(("end", None))

        t = threading.Thread(target=runner, daemon=True)
        t.start()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        sent_role = False
        final_text = ""

        try:
            while True:
                try:
                    kind, payload = q.get(timeout=620)
                except Empty:
                    self._write_sse_chunk(chat_id, model_name, created, "\n[timeout]\n", role=not sent_role)
                    sent_role = True
                    break

                if kind == "chunk":
                    text = payload or ""
                    if not text:
                        continue
                    self._write_sse_chunk(
                        chat_id, model_name, created, text, role=not sent_role
                    )
                    sent_role = True
                elif kind == "final":
                    final_text = payload or ""
                    # If nothing was streamed incrementally, dump the final report
                    if not sent_role and final_text:
                        self._write_sse_chunk(
                            chat_id, model_name, created, final_text, role=True
                        )
                        sent_role = True
                    elif final_text and not self._streamed_substantial():
                        # Append a compact change-log footer if we only streamed partials
                        pass
                elif kind == "error":
                    self._write_sse_chunk(
                        chat_id, model_name, created,
                        f"\n**Error:** {payload}\n",
                        role=not sent_role,
                    )
                    sent_role = True
                elif kind == "end":
                    break
        except (BrokenPipeError, ConnectionResetError):
            print("[stream] client disconnected")
            return

        # finish
        final = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        try:
            self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

        print(f"[stream done] final={len(final_text)} chars")
        # Cache for dedup — Hermes often resends the same prompt
        if final_text:
            global _last_request
            _last_request = (user_content, time.time(), final_text)

    def _streamed_substantial(self) -> bool:
        return True  # incremental events are the primary payload

    def _write_sse_chunk(self, chat_id, model, created, text, role: bool = False):
        delta = {"content": text}
        if role:
            delta["role"] = "assistant"
        payload = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        self.wfile.write(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def do_GET(self):
        if self.path in ("/v1/models", "/v1/models/"):
            models = [{"id": "browser-agent", "object": "model", "owned_by": "local"}]
            # Also advertise configured model name if set
            if _config.get("model"):
                models.append({
                    "id": _config["model"],
                    "object": "model",
                    "owned_by": "local",
                })
            self._send_json(200, {"object": "list", "data": models})
        elif self.path in ("/health", "/v1/health"):
            self._send_json(200, {
                "status": "ok" if _bridge else "starting",
                "provider": _config.get("provider"),
                # Basename only — no full path / username
                "workspace": workspace_label(),
            })
        else:
            self._send_json(404, {"error": {"message": "Not found"}})

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Quieter access log — we print our own request lines
        pass


# ── entry point ────────────────────────────────────────────────────


def load_config(path: str | None = None) -> dict:
    search = [path, Path(__file__).parent / "config.yaml", Path("config.yaml")]
    for p in search:
        if p and Path(p).exists():
            with open(p, encoding="utf-8") as f:
                return yaml.safe_load(f)
    return {
        "provider": "chatgpt",
        "urls": {"chatgpt": "https://chatgpt.com/"},
        "user_data_dir": "./browser_profile",
        "browser": "chromium",
        "headless": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Browser LLM Agent — API Server for Hermes")
    parser.add_argument("--provider", "-p", help="chatgpt, claude, gemini, perplexity")
    parser.add_argument("--model", "-m", help="Model name (for Perplexity)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument("--workspace", "-w", help="Working directory for tools")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.provider:
        config["provider"] = args.provider
    if args.model:
        config["model"] = args.model

    if args.workspace:
        ws = Path(args.workspace).expanduser().resolve()
        ws.mkdir(parents=True, exist_ok=True)
        import os
        os.chdir(ws)

    print("Starting browser agent server...")
    print(f"  Provider:  {config.get('provider', 'chatgpt')}")
    if config.get("model"):
        print(f"  Model:     {config['model']}")
    print(f"  Workspace: {workspace_label()}  (paths redacted for browser LLM)")

    loop_thread = threading.Thread(target=_start_event_loop, daemon=True)
    loop_thread.start()
    time.sleep(0.2)

    try:
        _run_in_loop(_init_agent(config), timeout=180)
    except Exception as e:
        print(f"[fatal] Failed to initialize agent: {e}")
        sys.exit(1)

    # ThreadingHTTPServer so stream writers don't block health checks
    server = ThreadingHTTPServer((args.host, args.port), AgentHandler)
    print(f"\n[server] Listening on http://{args.host}:{args.port}/v1")
    print("[server] Hermes setup:")
    print("         hermes config set model.provider custom")
    print(f"         hermes config set model.base_url http://{args.host}:{args.port}/v1")
    print("         hermes config set model.api_key noop")
    print("[server] Streams: commands, file diffs, cleaned chat text, status.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] Shutting down...")
        try:
            if _bridge:
                _run_in_loop(_bridge.close(), timeout=10)
        except Exception:
            pass
        _loop.call_soon_threadsafe(_loop.stop)
        server.shutdown()


if __name__ == "__main__":
    main()
