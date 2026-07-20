"""
agent_core.py — Shared agent loop used by server.py, cli.py, and agent.py.

Yields structured events so callers (especially the Hermes-facing server)
can stream git-like change logs, running commands, and cleaned chat text.

Privacy: nothing sent to the browser chat LLM includes the user's name,
home directory, unrelated local projects, or prior local chat history.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from browser import BrowserBridge
from privacy import redact_text, workspace_label
from tools import ToolExecutor


# ── events ─────────────────────────────────────────────────────────

@dataclass
class AgentEvent:
    """A single unit of agent progress for streaming / logging."""
    kind: str  # status | thinking | response | tool_start | tool_result | report | done | error
    text: str = ""
    data: dict = field(default_factory=dict)


def clean_llm_text(text: str) -> str:
    """Strip tool markers and TASK_COMPLETE for human/Hermes display."""
    if not text:
        return ""
    clean = re.sub(r"\[\[\[SHELL\]\]\].*?\[\[\[END\]\]\]", "", text, flags=re.DOTALL)
    clean = re.sub(
        r'\[\[\[FILE\s+path=["\']?.*?["\']?\]\]\].*?\[\[\[END\]\]\]',
        "",
        clean,
        flags=re.DOTALL,
    )
    clean = re.sub(r'\[\[\[READ\s+path=["\']?.*?["\']?\]\]\]', "", clean)
    clean = re.sub(r"\bTASK_COMPLETE\b", "", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return clean.strip()


def build_system_prompt(config: dict, workspace: Optional[str] = None) -> str:
    """System prompt for the browser chat LLM — no private local identity."""
    base = (config.get("system_prompt") or "").rstrip()
    shell_name = "PowerShell" if os.name == "nt" else "bash"
    # OS kind only — no hostname, release string, or username
    if os.name == "nt":
        os_kind = "Windows"
    elif hasattr(os, "uname"):
        sysname = os.uname().sysname.lower()
        os_kind = "macOS" if "darwin" in sysname else "Linux"
    else:
        os_kind = "Unix"

    # Only the workspace folder name — never the full path (path contains username)
    ws_name = workspace_label(workspace)

    env = f"""
ENVIRONMENT
-----------
OS: {os_kind}
Shell: {shell_name}
Workspace: {ws_name}  (use relative paths like ./file.py — never absolute paths)
Privacy: Do not ask for or repeat the user's real name, home directory, or
paths outside this workspace. Stay on the current task only.

RULES
-----
1. You control a local computer through tool markers. The local agent executes them.
2. Use EXACT markers on their own lines (plain text, no markdown fences around markers):

[[[SHELL]]]
command here
[[[END]]]

[[[FILE path="./relative/path"]]]
file contents here
[[[END]]]

[[[READ path="./relative/path"]]]

3. After you emit tool markers, STOP and wait. The agent pastes [TOOL OUTPUT] back.
4. Prefer [[[FILE ...]]] for creating/editing files. Do NOT use shell heredocs.
5. One focused step at a time when debugging; batch independent file writes when safe.
6. When the full task is done, write TASK_COMPLETE on its own line.
7. Output plain text. Tool markers must appear exactly as shown.
8. Paths: relative only (./...). Do not explore or list folders outside the workspace.
9. Do not request personal data, other project trees, or unrelated chat history.
"""

    if os.name == "nt":
        env += """
WINDOWS / POWERSHELL NOTES
--------------------------
- Use PowerShell syntax: dir, Get-ChildItem, New-Item -ItemType Directory -Force
- Do not use bash-only constructs (ls -la, cat, mkdir -p, && bash chaining)
- Chain with ;  or separate [[[SHELL]]] blocks
- Write files with [[[FILE path="..."]]] not Set-Content heredocs
- Stay inside the workspace; do not cd to the user home or other drives
"""

    if base:
        return f"{base}\n\n{env}".strip()
    return env.strip()


def extract_hermes_user_message(messages: list[dict]) -> tuple[str, str]:
    """Extract only the current user task for the browser LLM.

    Intentionally drops:
      - Hermes/system developer prompts (may include local paths, identity)
      - Multi-turn chat history (local conversations)
      - Embedded Assistant/User history blobs beyond the latest user turn

    Returns (user_task, extra_context) where extra_context is always empty
    so private client context is never forwarded to the cloud chat.
    """
    user_parts: list[str] = []

    for m in messages:
        role = (m.get("role") or "").lower()
        content = m.get("content") or ""
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif isinstance(part, str):
                    texts.append(part)
            content = "\n".join(texts)
        if not isinstance(content, str):
            content = str(content)

        # Ignore system/developer — do not forward to browser chat
        if role == "user":
            user_parts.append(content.strip())

    user_content = user_parts[-1] if user_parts else ""

    # Hermes sometimes embeds multi-turn as "User: ...\n\nAssistant: ...\n\nUser: ..."
    # Keep ONLY the last User: segment — never prior turns or assistant text.
    if user_content and (
        "\nUser:" in user_content
        or user_content.startswith("User:")
        or "\nAssistant:" in user_content
    ):
        parts = re.split(r"(?:^|\n)(?=User:\s?)", user_content)
        last_user = None
        for p in parts:
            p = p.strip()
            if p.startswith("User:"):
                last_user = p[5:].strip()
        if last_user:
            user_content = last_user

    # Redact any absolute paths / usernames that snuck into the task text
    user_content = redact_text(user_content.strip())

    # Never forward extra client context (history, system, projects)
    return user_content, ""


class AgentLoop:
    """Run the browser-LLM ↔ local-tools loop, yielding AgentEvents."""

    def __init__(
        self,
        bridge: BrowserBridge,
        tools: ToolExecutor,
        config: dict,
        max_tool_rounds: int = 25,
    ):
        self.bridge = bridge
        self.tools = tools
        self.config = config
        self.max_tool_rounds = max_tool_rounds
        self._primed = False

    async def prime(self) -> str:
        """Send system prompt once and wait for acknowledgement."""
        if self._primed:
            return ""
        prompt = build_system_prompt(self.config)
        # Final privacy pass on anything leaving the machine toward the chat LLM
        prompt = redact_text(prompt, workspace=self.tools.workspace)
        await self.bridge.send_message(prompt)
        resp = await self.bridge.wait_for_response()
        self._primed = True
        return resp

    async def run_turn(
        self,
        user_message: str,
        extra_context: str = "",
    ) -> AsyncIterator[AgentEvent]:
        """Process one user turn: send to chat LLM, execute tools, loop until done.

        extra_context is ignored for privacy — only the current task is sent.
        """
        self.tools.reset_change_log()
        all_results: list[dict] = []
        last_raw = ""

        # Privacy: never attach Hermes history / system / unrelated projects
        outbound = redact_text(user_message or "", workspace=self.tools.workspace)

        yield AgentEvent("status", "Sending request to browser LLM...")
        try:
            await self.bridge.send_message(outbound)
        except Exception as e:
            yield AgentEvent("error", f"Failed to send message: {e}")
            return

        for round_i in range(self.max_tool_rounds):
            yield AgentEvent("status", f"Waiting for LLM response (round {round_i + 1})...")
            try:
                response = await self.bridge.wait_for_response()
            except Exception as e:
                yield AgentEvent("error", f"Failed waiting for response: {e}")
                return

            last_raw = response or ""
            cleaned = clean_llm_text(last_raw)
            if cleaned:
                yield AgentEvent("response", cleaned, {"round": round_i + 1})

            results = self.tools.execute_tool_calls(last_raw)

            if not results:
                final_clean = cleaned or last_raw.strip()
                report = self.tools.format_agent_report(all_results, final_clean)
                yield AgentEvent(
                    "report",
                    report,
                    {
                        "results": all_results,
                        "cleaned": final_clean,
                        "raw": last_raw,
                        "complete": "TASK_COMPLETE" in last_raw,
                    },
                )
                yield AgentEvent(
                    "done",
                    report if report else final_clean,
                    {
                        "results": all_results,
                        "cleaned": final_clean,
                        "complete": "TASK_COMPLETE" in last_raw,
                    },
                )
                return

            for r in results:
                all_results.append(r)
                if r["type"] == "shell":
                    yield AgentEvent(
                        "tool_start",
                        f"$ {r.get('command', '')}",
                        {"tool": "shell", "command": r.get("command", "")},
                    )
                elif r["type"] == "file_write":
                    yield AgentEvent(
                        "tool_start",
                        f"write {r.get('path', '?')}",
                        {"tool": "file_write", "path": r.get("path")},
                    )
                elif r["type"] == "file_read":
                    yield AgentEvent(
                        "tool_start",
                        f"read {r.get('path', '?')}",
                        {"tool": "file_read", "path": r.get("path")},
                    )
                else:
                    yield AgentEvent("tool_start", r.get("type", "tool"), {"tool": r.get("type")})

                if r.get("error"):
                    body = f"ERROR: {r['error']}"
                elif r["type"] == "file_write" and r.get("diff"):
                    body = f"{r.get('result', '')}\n{r['diff']}"
                else:
                    body = r.get("result", "")[:2000]
                yield AgentEvent(
                    "tool_result",
                    body,
                    {
                        "tool": r.get("type"),
                        "ok": not bool(r.get("error")),
                        **{
                            k: r[k]
                            for k in (
                                "command",
                                "path",
                                "exit_code",
                                "duration_ms",
                                "mode",
                                "bytes",
                            )
                            if k in r
                        },
                    },
                )

            # format_results already redacts private paths for the browser LLM
            feedback = self.tools.format_results(results)
            yield AgentEvent("status", "Feeding tool output back to LLM...")
            try:
                await self.bridge.send_message(feedback)
            except Exception as e:
                yield AgentEvent("error", f"Failed to send tool feedback: {e}")
                return

        report = self.tools.format_agent_report(all_results, clean_llm_text(last_raw))
        yield AgentEvent(
            "done",
            report + "\n\n[Agent stopped: max tool rounds reached]",
            {"results": all_results, "cleaned": clean_llm_text(last_raw), "complete": False},
        )
