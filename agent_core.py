"""
agent_core.py — Shared agent loop used by cli.py and agent.py.

Yields structured events so callers can stream git-like change logs,
running commands, and cleaned chat text.

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
    """Strip tool markers, markdown, and TASK_COMPLETE for display."""
    if not text:
        return ""
    # Strip fenced code blocks — keep content, remove fences+lang
    clean = re.sub(r"```[^\n]*\n?(.*?)```", r"\1", text, flags=re.DOTALL)
    # Strip inline backticks
    clean = re.sub(r"`([^`\n]+)`", r"\1", clean)
    # Strip bold/italic
    clean = re.sub(r"\*\*(.+?)\*\*", r"\1", clean)
    clean = re.sub(r"\*([^*\n]+)\*", r"\1", clean)
    # Strip tool markers
    clean = re.sub(r"\[\[\[SHELL\]{2,}.*?\[\[\[END\]{2,}", "", clean, flags=re.DOTALL)
    clean = re.sub(
        r'\[\[\[FILE\s+path=["\']?.*?["\']?\]{2,}.*?\[\[\[END\]{2,}',
        "",
        clean,
        flags=re.DOTALL,
    )
    clean = re.sub(r'\[\[\[READ\s+path=["\']?.*?["\']?\]{2,}', "", clean)
    clean = re.sub(r"\bTASK_COMPLETE\b", "", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return clean.strip()


def build_system_prompt(config: dict, workspace: Optional[str] = None) -> str:
    """System prompt for the browser chat LLM — no private local identity.
    The config.yaml system_prompt already contains all rules. We only append
    a short environment hint."""
    base = (config.get("system_prompt") or "").rstrip()
    shell_name = "PowerShell" if os.name == "nt" else "bash"
    ws_name = workspace_label(workspace)

    # Perplexity's models are system-prompted to use built-in tools and
    # reject external marker protocols.  We don't fight that — we give a
    # minimal output-format instruction instead of an agent persona.
    prefix = ""
    if config.get("provider") == "perplexity":
        prefix = (
            "OUTPUT FORMAT — follow exactly:\n"
            "\n"
            "Shell command:\n"
            "[[[SHELL]]]\n"
            "command\n"
            "[[[END]]]\n"
            "\n"
            "Write file:\n"
            "[[[FILE path=\"name\"]]]\n"
            "content\n"
            "[[[END]]]\n"
            "\n"
            "Read file:\n"
            "[[[READ path=\"name\"]]]\n"
            "\n"
            "No markdown.  No backticks.  No explanations outside markers.\n"
            "Output ONLY the markers above with the requested content.\n"
            "\n"
        )

    env = (
        f"\n\nENVIRONMENT: {shell_name} shell. "
        f"Workspace folder: {ws_name}. "
        f"Use relative paths (./file.py). Stay in the workspace."
    )
    if os.name == "nt":
        env += " Use PowerShell commands (dir, New-Item, not ls or mkdir -p)."

    return f"{prefix}{base}{env}".strip()


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
        """Send system prompt. Do NOT wait for a response — the LLM's
        acknowledgement text would confuse subsequent turns."""
        if self._primed:
            return ""
        prompt = build_system_prompt(self.config)
        prompt = redact_text(prompt, workspace=self.tools.workspace)
        await self.bridge.send_message(prompt)
        self._primed = True
        return ""

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

        # Only forward the current task — no prior history or system context
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
            # Strip markdown fences that could hide tool markers
            parseable = re.sub(r"```[^\n]*\n?(.*?)```", r"\1", last_raw, flags=re.DOTALL)
            parseable = re.sub(r"`([^`\n]+)`", r"\1", parseable)
            parseable = re.sub(r"\*\*(.+?)\*\*", r"\1", parseable)
            parseable = re.sub(r"\*([^*\n]+)\*", r"\1", parseable)
            cleaned = clean_llm_text(last_raw)
            if cleaned:
                yield AgentEvent("response", cleaned, {"round": round_i + 1})

            results = self.tools.execute_tool_calls(parseable)

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
