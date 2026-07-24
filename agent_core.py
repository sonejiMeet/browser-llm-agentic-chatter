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
    """Heuristic: does text look like source code?"""
    if not text or len(text) < 20 or text.count("\n") < 1:
        return False
    signals = [
        r"#include\s*[<\"]", r"\bint\s+main\s*\(", r"\bdef\s+\w+\s*\(",
        r"\bfn\s+\w+\s*\(", r"\bfunc\s+\w+\s*\(", r"\bfunction\s+\w+\s*\(",
        r"\bclass\s+\w+", r"\bimport\s+", r"\bfrom\s+\w+\s+import",
        r"\bconst\s+\w+\s*=", r"\blet\s+\w+\s*=", r"\bvar\s+\w+\s*=",
    ]
    if any(re.search(s, text, re.MULTILINE) for s in signals):
        return True
    a = sum(1 for c in text if c.isalpha())
    p = sum(1 for c in text if c in "{}()[];=+-*/<>!&|^~")
    if a > 0 and p / a > 0.05:
        common = ["the", "and", "that", "have", "for", "you", "with", "this"]
        if sum(1 for w in common if re.search(rf"\b{w}\b", text, re.IGNORECASE)) < 3:
            return True
    return False


def _extract_path_from_text(text: str) -> Optional[str]:
    """Find a file path mentioned in the LLM's response.
    Matches: ./path/file.ext, path/file.ext, game_of_life/game.c
    Also catches paths without extension if they look like file paths."""
    # Try with extension first
    m = re.search(r'(?:\.?/)?[\w-]+(?:/[\w-]+)*\.[a-zA-Z]{1,6}', text)
    if m:
        return m.group(0)
    # Try paths in quotes: "game_of_life/game_of_life" or './path/file'
    m = re.search(r'["\']((?:\.?/)?[\w-]+(?:/[\w-]+){1,})["\']', text)
    if m:
        return m.group(1)
    # Try bare paths with slashes that look file-like (no extension)
    m = re.search(r'(?:\.?/)?[\w-]+(?:/[\w-]+){1,}', text)
    if m:
        return m.group(0)
    return None


def _build_turn_summary(results: list[dict], complete: bool) -> str:
    """Build a concise summary of what was done and how to run things."""
    if not results:
        return ""
    files_created: list[str] = []
    files_modified: list[str] = []
    binaries: list[str] = []
    scripts: list[str] = []

    for r in results:
        if r.get("type") == "file_write" and not r.get("error"):
            path = r.get("path", "")
            if r.get("mode") == "create":
                files_created.append(path)
            else:
                files_modified.append(path)
            # Detect runnable artifacts
            if path.endswith((".exe", ".bin", ".out", ".app")):
                binaries.append(f"./{path}")
            elif path.endswith((".py", ".sh", ".bash", ".ps1")):
                scripts.append(f"python ./{path}" if path.endswith(".py") else f"./{path}")
        elif r.get("type") == "shell" and not r.get("error"):
            cmd = r.get("command", "")
            # Detect compilation commands that produce executables
            m = re.search(r'(?:gcc|g\+\+|clang|rustc|go build|javac)\s+.+?\s+-o\s+(\S+)', cmd)
            if m:
                binaries.append(m.group(1))
            # pip install, npm install, etc.
            m = re.search(r'(?:pip|npm|cargo)\s+install\s+(\S+)', cmd)
            if m:
                pass  # installed packages, not runnable

    parts: list[str] = []
    if complete:
        parts.append("Task completed.")
    else:
        parts.append("Task stopped.")

    if files_created:
        parts.append(f"Created: {', '.join(files_created[:10])}")
    if files_modified:
        parts.append(f"Modified: {', '.join(files_modified[:10])}")

    # Deduplicate binaries and scripts
    binaries = list(dict.fromkeys(binaries))
    scripts = list(dict.fromkeys(scripts))
    if binaries:
        parts.append("Run: " + "  |  ".join(binaries))
    if scripts:
        parts.append("Run: " + "  |  ".join(scripts))
    return "\n".join(parts)


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
        self._asked_for_path: bool = False  # prevent ask-loop

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
        self._asked_for_path = False
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
            if not results:
                parseable = re.sub(r"```[^\n]*\n?(.*?)```", r"\1", last_raw, flags=re.DOTALL)
                parseable = re.sub(r"`([^`\n]+)`", r"\1", parseable)
                parseable = re.sub(r"\*\*(.+?)\*\*", r"\1", parseable)
                parseable = re.sub(r"\*([^*\n]+)\*", r"\1", parseable)
                if parseable != last_raw:
                    results = self.tools.execute_tool_calls(parseable)

            # Code without markers? Try to extract the intended path from the
            # LLM's response. If found, wrap and execute with THAT path.
            # If not found, feed the cleaned code back and ask for a path.
            # Never create default filenames like agent_output.*.
            if not results and _looks_like_code(last_raw):
                code = parseable if parseable != last_raw else last_raw
                path = _extract_path_from_text(last_raw)
                # Also check task description for path hints
                if not path and self._state:
                    path = _extract_path_from_text(self._state.task)

                if path:
                    wrapped = f'[[[FILE path="{path}"]]]\n{code}\n[[[END]]]'
                    yield AgentEvent("status", f"Extracted path from response -> {path}")
                    results = self.tools.execute_tool_calls(wrapped)
                elif not self._asked_for_path and round_i < self.max_tool_rounds - 1:
                    self._asked_for_path = True
                    ask = (
                        "I found code in your response but no [[[FILE path=\"...\"]]] "
                        "marker. Where should I save this?\n\n"
                        f"The code I extracted:\n```\n{code[:2000]}\n```\n\n"
                        "Reply with [[[FILE path=\"./your/path.ext\"]]] ... [[[END]]] "
                        "to tell me where to save it."
                    )
                    yield AgentEvent("status", "Code without path — asking LLM where to save...")
                    try:
                        await self.bridge.send_message(ask)
                        continue
                    except Exception as e:
                        yield AgentEvent("error", f"Failed to ask for path: {e}")
                        return
                # else: already asked, no path found → fall through to display + end turn

            # Display cleaned text (strip markers for user)
            cleaned = clean_llm_text(last_raw)
            if cleaned:
                yield AgentEvent("response", cleaned, {"round": round_i + 1})

            if not results:
                if "TASK_COMPLETE" in last_raw:
                    # Ask LLM for a final summary before ending
                    if not self._asked_for_path and round_i < self.max_tool_rounds - 1:
                        self._asked_for_path = True  # reuse flag to prevent looping
                        yield AgentEvent("status", "TASK_COMPLETE received — asking for summary...")
                        try:
                            await self.bridge.send_message(
                                "Task marked complete. Please provide a brief summary: "
                                "what files were created/modified, and how to run or use them."
                            )
                            continue
                        except Exception:
                            pass  # fall through to done if send fails
                    complete = True
                    summary = _build_turn_summary(all_results, complete)
                    report = self.tools.format_agent_report(all_results, cleaned or last_raw.strip())
                    yield AgentEvent("report", report, {
                        "results": all_results, "cleaned": cleaned or last_raw.strip(),
                        "raw": last_raw, "complete": complete, "summary": summary,
                    })
                    yield AgentEvent("done", report, {
                        "results": all_results, "cleaned": cleaned or last_raw.strip(),
                        "complete": complete, "summary": summary,
                    })
                    return
                # LLM wrote text but no tool calls and no TASK_COMPLETE —
                # feed its message back as context so it can continue.
                if cleaned:
                    yield AgentEvent("status", "No tool calls in response — feeding back to LLM...")
                    try:
                        await self.bridge.send_message(
                            f"[No tool calls detected in your response. "
                            f"Continue with [[[FILE ...]]], [[[SHELL]]], or [[[READ ...]]] "
                            f"if you need to take action.]\n\n"
                            f"Your message:\n{cleaned[:1000]}"
                        )
                        continue
                    except Exception as e:
                        yield AgentEvent("error", f"Failed to send prose feedback: {e}")
                        return
                # Empty response? Just continue
                continue

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
        summary = _build_turn_summary(all_results, False)
        yield AgentEvent(
            "done",
            report + "\n\n[Agent stopped: max tool rounds reached]",
            {"results": all_results, "cleaned": clean_llm_text(last_raw),
             "complete": False, "summary": summary},
        )
