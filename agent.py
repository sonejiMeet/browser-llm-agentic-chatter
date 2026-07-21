"""
agent.py — Standalone autonomous agent (single-task mode).

Drives a web-based LLM (ChatGPT, Claude, etc.) as an autonomous agent.
No API key — uses your browser session / subscription.

Usage:
    python agent.py "Build a Flask TODO app in ./myapp/"
    python agent.py --provider claude "Refactor all Python files to use pathlib"

For interactive REPL:  python cli.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import yaml
from pathlib import Path

from browser import BrowserBridge
from tools import ToolExecutor
from agent_core import AgentLoop, build_system_prompt


def load_config(path: str | None = None) -> dict:
    search_paths = [
        path,
        Path(__file__).parent / "config.yaml",
        Path("config.yaml"),
    ]
    for p in search_paths:
        if p and Path(p).exists():
            with open(p, encoding="utf-8") as f:
                return yaml.safe_load(f)
    return {
        "provider": "chatgpt",
        "urls": {"chatgpt": "https://chatgpt.com/"},
        "user_data_dir": "./browser_profile",
        "browser": "chromium",
        "headless": False,
        "max_turns_before_summary": 15,
        "tools": {"shell": {"enabled": True}, "file_write": {"enabled": True}},
        "system_prompt": "You are an autonomous agent. Use [[[SHELL]]]...[[[END]]] to run commands.",
    }


async def run_agent(task: str, config: dict):
    tools = ToolExecutor(config)
    bridge = BrowserBridge(config)
    agent = AgentLoop(bridge, tools, config)

    print(f"[*] Browser opening at {config.get('urls', {}).get(config.get('provider', 'chatgpt'), '?')}")
    await bridge.start()
    print("[*] Waiting for login...")
    await bridge.ensure_logged_in(timeout_seconds=120)

    model = config.get("model") or ""
    if model:
        await bridge.select_model(model)

    print("[*] Priming system prompt...")
    await agent.prime()
    print("[*] Starting agent loop.\n")

    turn = 0
    async for event in agent.run_turn(task):
        if event.kind == "status":
            print(f"  · {event.text}")
        elif event.kind == "response":
            preview = event.text[:500]
            print(f"\n── LLM ──\n{preview}{'...' if len(event.text) > 500 else ''}\n")
        elif event.kind == "tool_start":
            print(f"  ▶ {event.text}")
        elif event.kind == "tool_result":
            preview = event.text[:400].replace("\n", "\n    ")
            print(f"  ✓ {preview}")
        elif event.kind == "error":
            print(f"\n[!] {event.text}")
        elif event.kind == "done":
            print("\n[✓] Agent turn complete.")
            if event.data.get("complete"):
                print("[✓] LLM signaled TASK_COMPLETE.")
            # Show change log
            if tools.change_log:
                print("\n── Change log ──")
                for e in tools.change_log:
                    kind = e.get("kind")
                    if kind in ("create", "modify"):
                        print(f"  {kind:8} {e.get('path')}")
                    elif kind == "shell":
                        print(f"  shell    {e.get('command', '')[:80]}")
            turn += 1

    print("\n[*] Browser stays open — close it when you're done reviewing.")
    print("[*] Press Enter to close browser and exit...")
    await asyncio.get_event_loop().run_in_executor(None, input)
    await bridge.close()


def main():
    parser = argparse.ArgumentParser(
        description="Browser LLM Agent — use subscription-based LLMs as autonomous agents."
    )
    parser.add_argument("task", nargs="?", help="Task description (or read from stdin)")
    parser.add_argument("--provider", default=None, help="chatgpt, claude, gemini, perplexity")
    parser.add_argument("--model", "-m", help="Model name (Perplexity etc.)")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--workspace", "-w", help="Working directory for tools")
    parser.add_argument("--stdin", action="store_true", help="Read task from stdin")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.provider:
        config["provider"] = args.provider
    if args.model:
        config["model"] = args.model

    if args.workspace:
        import os
        ws = Path(args.workspace).expanduser().resolve()
        ws.mkdir(parents=True, exist_ok=True)
        os.chdir(ws)

    task = args.task
    if args.stdin or not task:
        print("Enter task description (Ctrl+Z then Enter on Windows, Ctrl+D on Unix):")
        task = sys.stdin.read().strip()

    if not task:
        print("Error: no task provided.")
        sys.exit(1)

    asyncio.run(run_agent(task, config))


if __name__ == "__main__":
    main()
