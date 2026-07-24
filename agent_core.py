"""
agent_core.py — Shared agent loop used by cli.py and agent.py.

Yields structured events so callers can stream git-like change logs,
running commands, and cleaned chat text.

v2: Rich initial context — gathers workspace state before first LLM call
    so the LLM doesn't waste turns on discovery. Structured turn feedback
    with progress tracking and error recovery guidance.

Privacy: nothing sent to the browser chat LLM includes the user's name,
home directory, unrelated local projects, or prior local chat history.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from browser import BrowserBridge
from context import (
    WorkspaceContext,
    TurnState,
    gather_workspace_context,
    build_initial_prompt,
    build_turn_feedback,
    build_system_prompt,
)
from privacy import redact_text
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
        r'\[\[\[FILE\s+path=[\"\']?.*?[\"\']?\]{2,}.*?\[\[\[END\]{2,}',
        "",
        clean,
        flags=re.DOTALL,
    )
    clean = re.sub(r'\[\[\[READ\s+path=[\"\']?.*?[\"\']?\]{2,}', "", clean)
    clean = re.sub(r"\bTASK_COMPLETE\b", "", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return clean.strip()


def _looks_like_code(text: str) -> bool:
    """Heuristic: does this text look like source code rather than prose?
    Used to detect when the LLM outputs raw code without markers."""
    if not text or len(text) < 20:
        return False
    lines = text.split("\n")
    if len(lines) < 2:
        return False
    # Strong code signals
    code_signals = [
        r"#include\s*[<\"]",    # C/C++ include
        r"\bint\s+main\s*\(",   # C main
        r"\bdef\s+\w+\s*\(",    # Python/Ruby function
        r"\bfn\s+\w+\s*\(",     # Rust function
        r"\bfunc\s+\w+\s*\(",   # Go function
        r"\bfunction\s+\w+\s*\(", # JS function
        r"\bclass\s+\w+",       # class definition
        r"\bimport\s+",         # import statement
        r"\bfrom\s+\w+\s+import", # Python from-import
        r"\bconst\s+\w+\s*=",   # JS/TS const
        r"\blet\s+\w+\s*=",     # JS let
        r"\bvar\s+\w+\s*=",     # JS var
        r"^\s*\}?\s*$",         # closing braces are common in code
    ]
    for signal in code_signals:
        if re.search(signal, text, re.MULTILINE):
            return True
    # Heuristic: high ratio of punctuation/symbols to words
    alpha_chars = sum(1 for c in text if c.isalpha())
    punct_chars = sum(1 for c in text if c in "{}()[];=+-*/<>!&|^~")
    if alpha_chars > 0 and punct_chars / max(alpha_chars, 1) > 0.05:
        # Also check for natural language signals
        common_words = ["the", "and", "that", "have", "for", "you", "with", "this"]
        word_count = sum(1 for w in common_words if re.search(rf"\b{w}\b", text, re.IGNORECASE))
        if word_count < 3:  # few common English words → probably code
            return True
    return False


def _guess_code_filename(text: str) -> str:
    """Guess a reasonable filename from code content.
    Used when auto-wrapping raw code that the LLM refused to mark up."""
    # Check for language-specific signals
    if re.search(r"#include\s*[<\"]", text) or re.search(r"\bint\s+main\s*\(", text):
        return "agent_output.c"
    # C-like fragments: for-loops with type declarations, printf, type casts
    if re.search(r"\bfor\s*\(\s*(int|char|float|double|size_t)\s", text):
        return "agent_output.c"
    if re.search(r"\bprintf\s*\(", text) or re.search(r"\bscanf\s*\(", text):
        return "agent_output.c"
    if re.search(r"\bdef\s+\w+\s*\(", text) or re.search(r"\bimport\s+\w", text):
        return "agent_output.py"
    if re.search(r"\bfn\s+\w+\s*\(", text) or re.search(r"\buse\s+\w+::", text):
        return "agent_output.rs"
    if re.search(r"\bfunc\s+\w+\s*\(", text) and re.search(r"\bpackage\s+\w", text):
        return "agent_output.go"
    if re.search(r"\bfunction\s+\w+\s*\(", text) or re.search(r"\bconst\s+\w+\s*=", text):
        return "agent_output.js"
    if re.search(r"<\w+>", text) or re.search(r"</\w+>", text):
        return "agent_output.html"
    if re.search(r"^[.#]\w+\s*\{", text, re.MULTILINE):
        return "agent_output.css"
    return "agent_output.txt"


class AgentLoop:
    """Run the browser-LLM <-> local-tools loop, yielding AgentEvents."""

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
        self._state: Optional[TurnState] = None

    async def prime(self) -> str:
        """Send system prompt and WAIT for the LLM to finish processing it.
        Consumes the acknowledgment so it doesn't pollute the first turn.
        Without this wait, the LLM may not have absorbed the marker protocol
        before the user task arrives, causing it to output raw code."""
        if self._primed:
            return ""
        prompt = build_system_prompt(self.config)
        prompt = redact_text(prompt, workspace=self.tools.workspace)
        await self.bridge.send_message(prompt)
        # Wait for the LLM to finish its acknowledgment response.
        # We consume and discard it — "OK, I understand" text would
        # confuse subsequent turns if left unread.
        try:
            await self.bridge.wait_for_response(timeout_seconds=60)
        except Exception:
            pass
        self._primed = True
        return ""

    def gather_context(self) -> WorkspaceContext:
        """Gather workspace state before the first LLM call.
        Call this once per task — it's deterministic and fast."""
        return gather_workspace_context(self.tools.workspace)

    def build_first_message(self, task: str, ctx: WorkspaceContext) -> str:
        """Build the rich first message with task + workspace context.
        The LLM sees the project structure before it takes any action."""
        prompt = build_initial_prompt(task, ctx)
        return redact_text(prompt, workspace=self.tools.workspace)

    async def run_turn(
        self,
        user_message: str,
        workspace_context: Optional[WorkspaceContext] = None,
    ) -> AsyncIterator[AgentEvent]:
        """Process one user turn: send to chat LLM, execute tools, loop until done.

        If workspace_context is provided, builds a rich initial prompt
        with file tree, git status, and project configs so the LLM
        doesn't waste turns on discovery.
        """
        self.tools.reset_change_log()
        all_results: list[dict] = []

        # Initialize turn state for progress tracking
        self._state = TurnState(task=user_message)

        # Build the first message
        if workspace_context:
            outbound = self.build_first_message(user_message, workspace_context)
            yield AgentEvent("status", "Sending task + workspace context to LLM...")
        else:
            outbound = redact_text(user_message or "", workspace=self.tools.workspace)
            yield AgentEvent("status", "Sending task to LLM...")

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

            # ── DIAG: show first 200 chars of raw browser response ──
            # Remove after confirming [[[ markers arrive intact.
            if last_raw and ("[[[" not in last_raw[:500] or True):  # always show
                preview = last_raw[:200].replace("\n", "\\n")
                yield AgentEvent("status", f"[raw preview] {preview}...")

            # ── Parse tool calls on RAW response first ──
            # Backtick/markdown stripping can mangle [[[END]]] markers when
            # the LLM wraps code in ``` inside a [[[FILE ...]]] block.
            # Parse raw first; only fall back to stripped if raw finds nothing.
            results = self.tools.execute_tool_calls(last_raw)

            # If raw parse found nothing but [[[ appears, the response may be
            # truncated (LLM still rendering). Check for unbalanced markers
            # (more openings than closings) and poll until complete.
            if not results and "[[[" in last_raw:
                opens = len(re.findall(r"\[\[\[(?:FILE|SHELL|READ)", last_raw))
                closes = len(re.findall(r"\[\[\[END\]{2,}", last_raw))
                if opens > closes:
                    yield AgentEvent(
                        "status",
                        f"Response incomplete ({opens} open, {closes} close) — waiting for LLM to finish..."
                    )
                    import asyncio
                    deadline = asyncio.get_event_loop().time() + 30
                    while asyncio.get_event_loop().time() < deadline:
                        await asyncio.sleep(1.5)
                        try:
                            response2 = await self.bridge._read_last_response()
                        except Exception:
                            continue
                        if response2 and response2 != last_raw:
                            last_raw = response2
                            results = self.tools.execute_tool_calls(last_raw)
                            if results:
                                yield AgentEvent("status", "Response completed, tools detected.")
                                break
                            # Check if balanced now
                            opens2 = len(re.findall(r"\[\[\[(?:FILE|SHELL|READ)", last_raw))
                            closes2 = len(re.findall(r"\[\[\[END\]{2,}", last_raw))
                            if opens2 <= closes2 and opens2 > 0:
                                # Balanced but still no parse — try stripped fallback
                                break
                    else:
                        yield AgentEvent(
                            "status",
                            "Response still incomplete after 30s — proceeding with partial content."
                        )

            # Fallback: strip markdown and retry parsing.
            # If the LLM output code (in ``` fences or raw), auto-wrap it in
            # [[[FILE ...]]] markers immediately — no warnings, no delays.
            # BUT: never auto-wrap text that already contains [[[ markers —
            # that means the LLM used proper markers and parsing should have
            # worked. Wrapping marker-bearing text would create nested/mangled
            # markers and write to the wrong file.
            code_to_wrap: Optional[str] = None

            if not results:
                parseable = re.sub(r"```[^\n]*\n?(.*?)```", r"\1", last_raw, flags=re.DOTALL)
                parseable = re.sub(r"`([^`\n]+)`", r"\1", parseable)
                parseable = re.sub(r"\*\*(.+?)\*\*", r"\1", parseable)
                parseable = re.sub(r"\*([^*\n]+)\*", r"\1", parseable)
                if parseable != last_raw:
                    results = self.tools.execute_tool_calls(parseable)
                    # Still no markers — stripped content is code without markers?
                    if not results and _looks_like_code(parseable) and "[[[" not in parseable:
                        code_to_wrap = parseable

            if not results and code_to_wrap is None:
                # Raw output (no fences) that looks like code — auto-wrap it too.
                # Guard: don't wrap if markers are present (LLM used them, parser
                # should have caught it — wrapping would create nested markers).
                if (_looks_like_code(last_raw) and "[[[" not in last_raw
                        and round_i < self.max_tool_rounds - 1):
                    code_to_wrap = last_raw

            if code_to_wrap:
                fname = _guess_code_filename(code_to_wrap)
                wrapped = f'[[[FILE path="./{fname}"]]]\n{code_to_wrap}\n[[[END]]]'
                yield AgentEvent("status", f"Auto-wrapped code as ./{fname}")
                results = self.tools.execute_tool_calls(wrapped)

            # Display cleaned text (strip markers for user)
            cleaned = clean_llm_text(last_raw)
            if cleaned:
                yield AgentEvent("response", cleaned, {"round": round_i + 1})

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
                                "command", "path", "exit_code",
                                "duration_ms", "mode", "bytes",
                            )
                            if k in r
                        },
                    },
                )

            # Build structured turn feedback with progress tracking
            tool_output = self.tools.format_results(results)
            feedback = build_turn_feedback(results, self._state)

            # Combine: tool output + structured guidance
            combined = f"{tool_output}\n\n{feedback}"
            yield AgentEvent("status", "Feeding tool output + progress back to LLM...")
            try:
                await self.bridge.send_message(combined)
            except Exception as e:
                yield AgentEvent("error", f"Failed to send tool feedback: {e}")
                return

        report = self.tools.format_agent_report(all_results, clean_llm_text(last_raw))
        yield AgentEvent(
            "done",
            report + "\n\n[Agent stopped: max tool rounds reached]",
            {"results": all_results, "cleaned": clean_llm_text(last_raw), "complete": False},
        )
