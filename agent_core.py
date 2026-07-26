"""
agent_core.py — Transaction-aware shared agent loop used by cli.py and agent.py.

The loop never adds arbitrary waiting between transactions. BrowserBridge owns
completion detection and resolves wait_for_response() as soon as generation is
actually complete. A transaction is:

    send -> continuously watch response -> parse tools -> execute -> send feedback

Tool commands are only executed after a complete response, preventing partial
[[[FILE]]]/[[[SHELL]]] blocks from being acted on.
"""

from __future__ import annotations

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


@dataclass
class AgentEvent:
    """A single streamable unit of agent progress."""
    kind: str  # status | thinking | response | tool_start | tool_result | report | done | error
    text: str = ""
    data: dict = field(default_factory=dict)


def clean_llm_text(text: str) -> str:
    """Remove tool calls and lightweight markup from display text."""
    if not text:
        return ""
    clean = re.sub(r"```[^\n]*\n?(.*?)```", r"\1", text, flags=re.DOTALL)
    clean = re.sub(r"`([^`\n]+)`", r"\1", clean)
    clean = re.sub(r"\*\*(.+?)\*\*", r"\1", clean)
    clean = re.sub(r"\*([^*\n]+)\*", r"\1", clean)
    clean = re.sub(r"\[\[\[SHELL\]{2,}.*?\[\[\[END\]{2,}", "", clean, flags=re.DOTALL)
    clean = re.sub(
        r'\[\[\[FILE\s+path=["\']?.*?["\']?\]{2,}.*?\[\[\[END\]{2,}',
        "",
        clean,
        flags=re.DOTALL,
    )
    clean = re.sub(r'\[\[\[READ\s+path=["\']?.*?["\']?\]{2,}', "", clean)
    clean = re.sub(r"\bTASK_COMPLETE\b", "", clean)
    return re.sub(r"\n{3,}", "\n\n", clean).strip()


def _strip_markdown_for_parser(text: str) -> str:
    """Only use this after trying raw text: markdown can wrap marker blocks."""
    text = re.sub(r"```[^\n]*\n?(.*?)```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    return re.sub(r"\*([^*\n]+)\*", r"\1", text)


def _looks_like_code(text: str) -> bool:
    if not text or len(text) < 20 or "\n" not in text:
        return False
    signals = (
        r"#include\s*[<\"]", r"\bint\s+main\s*\(", r"\bdef\s+\w+\s*\(",
        r"\bfn\s+\w+\s*\(", r"\bfunc\s+\w+\s*\(", r"\bfunction\s+\w+\s*\(",
        r"\bclass\s+\w+", r"\bimport\s+", r"\bfrom\s+\w+\s+import",
        r"\bconst\s+\w+\s*=", r"\blet\s+\w+\s*=", r"\bvar\s+\w+\s*=",
    )
    if any(re.search(pattern, text, re.MULTILINE) for pattern in signals):
        return True
    alpha = sum(c.isalpha() for c in text)
    punct = sum(c in "{}()[];=+-*/<>!&|^~" for c in text)
    if alpha and punct / alpha > 0.05:
        common = ("the", "and", "that", "have", "for", "you", "with", "this")
        return sum(bool(re.search(rf"\b{word}\b", text, re.I)) for word in common) < 3
    return False


def _extract_path_from_text(text: str) -> Optional[str]:
    m = re.search(r'(?:\.?/)?[\w-]+(?:/[\w-]+)*\.[A-Za-z]{1,12}', text)
    if m:
        return m.group(0)
    m = re.search(r'["\']((?:\.?/)?[\w-]+(?:/[\w-]+){1,})["\']', text)
    if m:
        return m.group(1)
    m = re.search(r'(?:\.?/)?[\w-]+(?:/[\w-]+){1,}', text)
    return m.group(0) if m else None


def _build_turn_summary(results: list[dict], complete: bool) -> str:
    if not results:
        return "Task completed." if complete else "Task stopped."
    created: list[str] = []
    modified: list[str] = []
    binaries: list[str] = []
    scripts: list[str] = []
    for result in results:
        if result.get("type") == "file_write" and not result.get("error"):
            path = result.get("path", "")
            (created if result.get("mode") == "create" else modified).append(path)
            if path.endswith((".exe", ".bin", ".out", ".app")):
                binaries.append(f"./{path}")
            elif path.endswith((".py", ".sh", ".bash", ".ps1")):
                scripts.append(f"python ./{path}" if path.endswith(".py") else f"./{path}")
        elif result.get("type") == "shell" and not result.get("error"):
            match = re.search(r'(?:gcc|g\+\+|clang|rustc|go build|javac)\s+.+?\s+-o\s+(\S+)', result.get("command", ""))
            if match:
                binaries.append(match.group(1))
    parts = ["Task completed." if complete else "Task stopped."]
    if created:
        parts.append("Created: " + ", ".join(created[:10]))
    if modified:
        parts.append("Modified: " + ", ".join(modified[:10]))
    if binaries:
        parts.append("Run: " + "  |  ".join(dict.fromkeys(binaries)))
    if scripts:
        parts.append("Run: " + "  |  ".join(dict.fromkeys(scripts)))
    return "\n".join(parts)


class AgentLoop:
    """Browser-LLM/local-tool transaction loop."""

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
        self._path_requested = False

    async def prime(self) -> str:
        """Send protocol setup without blocking the first useful task.

        The acknowledgement is intentionally not awaited. Browser chats retain
        ordering: the task is queued after the system message. This removes the
        old mandatory 60-second prime transaction.
        """
        if self._primed:
            return ""
        prompt = redact_text(build_system_prompt(self.config), workspace=self.tools.workspace)
        await self.bridge.send_message(prompt)
        self._primed = True
        return ""

    def gather_context(self) -> WorkspaceContext:
        return gather_workspace_context(self.tools.workspace)

    def build_first_message(self, task: str, ctx: WorkspaceContext) -> str:
        return redact_text(build_initial_prompt(task, ctx), workspace=self.tools.workspace)

    async def _send_and_wait(self, message: str) -> str:
        """One browser transaction; bridge completes exactly when UI completes."""
        await self.bridge.send_message(message)
        return await self.bridge.wait_for_response(
            timeout_seconds=int(self.config.get("response_timeout_seconds", 300))
        )

    def _parse_tools(self, raw: str) -> tuple[list[dict], str]:
        """Parse raw first, then a markdown-normalized fallback."""
        results = self.tools.execute_tool_calls(raw)
        if results:
            return results, raw
        parseable = _strip_markdown_for_parser(raw)
        if parseable != raw:
            results = self.tools.execute_tool_calls(parseable)
        return results, parseable

    async def run_turn(
        self,
        user_message: str,
        workspace_context: Optional[WorkspaceContext] = None,
    ) -> AsyncIterator[AgentEvent]:
        self.tools.reset_change_log()
        self._path_requested = False
        self._state = TurnState(task=user_message)
        all_results: list[dict] = []
        last_raw = ""

        if workspace_context:
            outbound = self.build_first_message(user_message, workspace_context)
            yield AgentEvent("status", "Sending task + workspace context...")
        else:
            outbound = redact_text(user_message or "", workspace=self.tools.workspace)
            yield AgentEvent("status", "Sending task...")

        try:
            last_raw = await self._send_and_wait(outbound)
        except Exception as exc:
            yield AgentEvent("error", f"Initial transaction failed: {exc}")
            return

        for round_i in range(self.max_tool_rounds):
            results, parseable = self._parse_tools(last_raw)

            # Raw code recovery remains guarded: never invent a filename.
            if not results and _looks_like_code(last_raw):
                code = parseable if parseable != last_raw else last_raw
                path = _extract_path_from_text(last_raw) or _extract_path_from_text(self._state.task)
                if path:
                    yield AgentEvent("status", f"Recovered unmarked code for {path}")
                    results = self.tools.execute_tool_calls(
                        f'[[[FILE path="{path}"]]]\n{code}\n[[[END]]]'
                    )
                elif not self._path_requested:
                    self._path_requested = True
                    question = (
                        "Your answer contains code but no destination marker. Reply with "
                        "[[[FILE path=\"relative/path.ext\"]]] followed by the complete "
                        "file contents and [[[END]]]."
                    )
                    yield AgentEvent("status", "Code has no path; requesting a destination...")
                    try:
                        last_raw = await self._send_and_wait(question)
                    except Exception as exc:
                        yield AgentEvent("error", f"Path transaction failed: {exc}")
                        return
                    continue

            cleaned = clean_llm_text(last_raw)
            if cleaned:
                yield AgentEvent("response", cleaned, {"round": round_i + 1})

            if not results:
                if "TASK_COMPLETE" in last_raw:
                    summary = _build_turn_summary(all_results, True)
                    report = self.tools.format_agent_report(all_results, cleaned or last_raw.strip())
                    data = {
                        "results": all_results,
                        "cleaned": cleaned or last_raw.strip(),
                        "raw": last_raw,
                        "complete": True,
                        "summary": summary,
                    }
                    yield AgentEvent("report", report, data)
                    yield AgentEvent("done", report, data)
                    return

                # Do not create a useless feedback transaction for a final
                # conversational reply. Ask for continuation only once per
                # response, and only when it says it still needs action.
                if cleaned:
                    continuation = (
                        "Continue the task now. If an action is needed, emit only valid "
                        "[[[FILE ...]]], [[[SHELL]]], or [[[READ ...]]] markers. "
                        "Use TASK_COMPLETE when finished."
                    )
                    yield AgentEvent("status", "No actions found; requesting next action...")
                    try:
                        last_raw = await self._send_and_wait(continuation)
                    except Exception as exc:
                        yield AgentEvent("error", f"Continuation transaction failed: {exc}")
                        return
                    continue

                yield AgentEvent("error", "LLM returned an empty response.")
                return

            # Emit tool events and execute_tool_calls results are already
            # materialized by ToolExecutor. There is no extra wait here.
            for result in results:
                all_results.append(result)
                type_ = result.get("type", "tool")
                if type_ == "shell":
                    label, meta = f"$ {result.get('command', '')}", {"tool": type_, "command": result.get("command", "")}
                elif type_ in ("file_write", "file_read"):
                    verb = "write" if type_ == "file_write" else "read"
                    label, meta = f"{verb} {result.get('path', '?')}", {"tool": type_, "path": result.get("path")}
                else:
                    label, meta = type_, {"tool": type_}
                yield AgentEvent("tool_start", label, meta)

                if result.get("error"):
                    body = f"ERROR: {result['error']}"
                elif type_ == "file_write" and result.get("diff"):
                    body = f"{result.get('result', '')}\n{result['diff']}"
                else:
                    body = result.get("result", "")[:2000]
                yield AgentEvent(
                    "tool_result",
                    body,
                    {
                        "tool": type_,
                        "ok": not bool(result.get("error")),
                        **{key: result[key] for key in ("command", "path", "exit_code", "duration_ms", "mode", "bytes") if key in result},
                    },
                )

            tool_output = self.tools.format_results(results)
            feedback = build_turn_feedback(results, self._state)
            yield AgentEvent("status", "Sending tool results...")
            try:
                last_raw = await self._send_and_wait(f"{tool_output}\n\n{feedback}")
            except Exception as exc:
                yield AgentEvent("error", f"Tool-feedback transaction failed: {exc}")
                return

        cleaned = clean_llm_text(last_raw)
        report = self.tools.format_agent_report(all_results, cleaned)
        yield AgentEvent(
            "done",
            report + "\n\n[Agent stopped: max tool rounds reached]",
            {
                "results": all_results,
                "cleaned": cleaned,
                "complete": False,
                "summary": _build_turn_summary(all_results, False),
            },
        )
