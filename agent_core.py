"""
agent_core.py - Transaction-aware shared agent loop used by cli.py and agent.py.

A transaction is:

    send -> observe complete response -> parse tools -> execute -> send feedback

Tool commands run only after BrowserBridge reports a complete response, so
partial FILE and SHELL blocks are never executed.

Important file-payload rule:
- Tool parsing always uses the raw browser response.
- A FILE payload may contain a normal fenced code block.
- Generic Markdown stripping must never run before file parsing because it can
  alter code payloads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from browser import BrowserBridge
from context import (
    WorkspaceContext,
    TurnState,
    build_initial_prompt,
    build_system_prompt,
    build_turn_feedback,
    gather_workspace_context,
)
from privacy import redact_text
from tools import ToolExecutor


@dataclass
class AgentEvent:
    """A single streamable unit of agent progress."""

    kind: str
    text: str = ""
    data: dict = field(default_factory=dict)


def clean_llm_text(text: str) -> str:
    """
    Remove tool blocks from text displayed to the user.

    This function is display-only. Its output is never passed to
    ToolExecutor.execute_tool_calls(), so Markdown cleanup here cannot affect
    written source files.
    """
    if not text:
        return ""

    clean = re.sub(
        r"\[\[\[SHELL\]\]\].*?\[\[\[END\]\]\]",
        "",
        text,
        flags=re.DOTALL,
    )
    clean = re.sub(
        r"""
        \[\[\[FILE
        \s+path=
        (?:
            "[^"]+"
            |
            '[^']+'
            |
            [^\]\s]+
        )
        \]\]\]
        .*?
        \[\[\[END\]\]\]
        """,
        "",
        clean,
        flags=re.DOTALL | re.VERBOSE,
    )
    clean = re.sub(
        r"""
        \[\[\[READ
        \s+path=
        (?:
            "[^"]+"
            |
            '[^']+'
            |
            [^\]\s]+
        )
        \]\]\]
        """,
        "",
        clean,
        flags=re.VERBOSE,
    )
    clean = re.sub(r"\bTASK_COMPLETE\b", "", clean)

    return re.sub(r"\n{3,}", "\n\n", clean).strip()


def _looks_like_code(text: str) -> bool:
    """
    Detect likely source code only for a guarded recovery path.

    Normal file creation must use a FILE marker. This helper exists only for
    cases where a model gave complete source but omitted the file marker.
    """
    if not text or len(text) < 20 or "\n" not in text:
        return False

    signals = (
        r"#include\s*[<\"]",
        r"\bint\s+main\s*\(",
        r"\bdef\s+\w+\s*\(",
        r"\bfn\s+\w+\s*\(",
        r"\bfunc\s+\w+\s*\(",
        r"\bfunction\s+\w+\s*\(",
        r"\bclass\s+\w+",
        r"\bimport\s+",
        r"\bfrom\s+\w+\s+import",
        r"\bconst\s+\w+\s*=",
        r"\blet\s+\w+\s*=",
        r"\bvar\s+\w+\s*=",
    )

    if any(re.search(pattern, text, re.MULTILINE) for pattern in signals):
        return True

    alpha = sum(char.isalpha() for char in text)
    punctuation = sum(char in "{}()[];=+-*/<>!&|^~" for char in text)

    if alpha and punctuation / alpha > 0.05:
        common_words = (
            "the",
            "and",
            "that",
            "have",
            "for",
            "you",
            "with",
            "this",
        )
        return (
            sum(
                bool(re.search(rf"\b{word}\b", text, re.IGNORECASE))
                for word in common_words
            )
            < 3
        )

    return False


def _extract_path_from_text(text: str) -> Optional[str]:
    """Find a plausible relative path in normal prose or a task message."""
    match = re.search(
        r"(?:\.?/)?[\w-]+(?:/[\w-]+)*\.[A-Za-z]{1,12}",
        text,
    )
    if match:
        return match.group(0)

    match = re.search(
        r"""["']((?:\.?/)?[\w-]+(?:/[\w-]+){1,})["']""",
        text,
    )
    if match:
        return match.group(1)

    match = re.search(
        r"(?:\.?/)?[\w-]+(?:/[\w-]+){1,}",
        text,
    )
    return match.group(0) if match else None


def _build_turn_summary(results: list[dict], complete: bool) -> str:
    """Build a compact terminal summary from executed tool results."""
    if not results:
        return "Task completed." if complete else "Task stopped."

    created: list[str] = []
    modified: list[str] = []
    binaries: list[str] = []
    scripts: list[str] = []

    for result in results:
        if (
            result.get("type") == "file_write"
            and not result.get("error")
        ):
            path = result.get("path", "")
            mode = result.get("mode")

            if mode == "create":
                created.append(path)
            else:
                modified.append(path)

            if path.endswith((".exe", ".bin", ".out", ".app")):
                binaries.append(f"./{path}")
            elif path.endswith((".py", ".sh", ".bash", ".ps1")):
                if path.endswith(".py"):
                    scripts.append(f"python ./{path}")
                else:
                    scripts.append(f"./{path}")

        elif (
            result.get("type") == "shell"
            and not result.get("error")
        ):
            match = re.search(
                r"""
                (?:gcc|g\+\+|clang|rustc|go\ build|javac)
                \s+.+?
                \s+-o\s+
                (\S+)
                """,
                result.get("command", ""),
                re.VERBOSE,
            )
            if match:
                binaries.append(match.group(1))

    parts = ["Task completed." if complete else "Task stopped."]

    if created:
        parts.append("Created: " + ", ".join(created[:10]))
    if modified:
        parts.append("Modified: " + ", ".join(modified[:10]))
    if binaries:
        parts.append("Run: " + " | ".join(dict.fromkeys(binaries)))
    if scripts:
        parts.append("Run: " + " | ".join(dict.fromkeys(scripts)))

    return "\n".join(parts)


class AgentLoop:
    """Browser-LLM/local-tool transaction loop."""

    def __init__(
        self,
        bridge: BrowserBridge,
        tools: ToolExecutor,
        config: dict,
        max_tool_rounds: int = 25,
        confirm=None,
    ):
        self.bridge = bridge
        self.tools = tools
        self.config = config
        self.max_tool_rounds = max_tool_rounds
        # Optional async confirmation gate. When provided, it is awaited with
        # the parsed plan before any file write or shell command runs. It must
        # return the (possibly reduced) plan to execute. A truthy return value
        # that is not a list means "approve all". None/empty list means "reject
        # all local changes for this turn". Read-only actions are never gated.
        self.confirm = confirm
        self._primed = False
        self._state: Optional[TurnState] = None
        self._path_requested = False

    async def prime(self) -> str:
        """
        Send protocol setup without blocking the first useful task.

        The acknowledgement is intentionally not awaited. Browser chats retain
        message order, so the first task is queued behind this setup prompt.
        """
        if self._primed:
            return ""

        prompt = redact_text(
            build_system_prompt(self.config),
            workspace=self.tools.workspace,
        )
        await self.bridge.send_message(prompt)
        self._primed = True

        return ""

    def gather_context(self) -> WorkspaceContext:
        return gather_workspace_context(self.tools.workspace)

    def build_first_message(
        self,
        task: str,
        context: WorkspaceContext,
    ) -> str:
        return redact_text(
            build_initial_prompt(task, context),
            workspace=self.tools.workspace,
        )

    async def _send_and_wait(self, message: str) -> str:
        """Run one full browser transaction."""
        await self.bridge.send_message(message)

        return await self.bridge.wait_for_response(
            timeout_seconds=int(
                self.config.get("response_timeout_seconds", 300)
            )
        )

    def _parse_tools(self, raw: str) -> list[dict]:
        """
        Parse only the exact raw browser response into an execution plan.

        Returns a plan (no side effects). ToolExecutor owns FILE code-fence
        removal and execution. Do not run Markdown cleanup here: that risks
        changing code before it reaches the file writer.
        """
        return self.tools.plan_tool_calls(raw)

    async def _confirm_plan(self, plan: list[dict]) -> list[dict]:
        """
        Ask the user to approve mutating actions before they run locally.

        Read-only actions (file_read) always pass through. When no `confirm`
        callback is configured, the plan is executed as-is (auto-approve).
        """
        if not plan:
            return plan

        mutating = [
            entry
            for entry in plan
            if entry.get("action") in ("file_write", "shell")
        ]

        if not mutating:
            return plan

        if self.confirm is None:
            return plan

        decision = await self.confirm(plan)

        # Truthy non-list => approve everything. List => caller-vetted plan.
        if decision is True:
            return plan
        if decision is False or decision is None:
            # Reject all local changes but keep reads.
            return [
                entry
                for entry in plan
                if entry.get("action") == "file_read"
            ]
        if isinstance(decision, list):
            return decision

        return plan

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
            outbound = self.build_first_message(
                user_message,
                workspace_context,
            )
            yield AgentEvent(
                "status",
                "Sending task + workspace context...",
            )
        else:
            outbound = redact_text(
                user_message or "",
                workspace=self.tools.workspace,
            )
            yield AgentEvent("status", "Sending task...")

        try:
            last_raw = await self._send_and_wait(outbound)
        except Exception as exc:
            yield AgentEvent(
                "error",
                f"Initial transaction failed: {exc}",
            )
            return

        for round_index in range(self.max_tool_rounds):
            plan = self._parse_tools(last_raw)

            # Guarded recovery for a model response that contains source code
            # but omitted the FILE marker. It never invents a filename.
            if not plan and _looks_like_code(last_raw):
                path = (
                    _extract_path_from_text(last_raw)
                    or _extract_path_from_text(self._state.task)
                )

                if path:
                    yield AgentEvent(
                        "status",
                        f"Recovered unmarked code for {path}",
                    )
                    plan = self.tools.plan_tool_calls(
                        f'[[[FILE path="{path}"]]]\n'
                        f"{last_raw}\n"
                        f"[[[END]]]"
                    )

                elif not self._path_requested:
                    self._path_requested = True

                    question = (
                        "Your response includes source code but no file "
                        "destination. Reply with exactly one FILE block in "
                        "this format:\n\n"
                        '[[[FILE path="relative/path.ext"]]]\n'
                        "```language\n"
                        "complete source code\n"
                        "```\n"
                        "[[[END]]]"
                    )

                    yield AgentEvent(
                        "status",
                        "Code has no path; requesting a destination...",
                    )

                    try:
                        last_raw = await self._send_and_wait(question)
                    except Exception as exc:
                        yield AgentEvent(
                            "error",
                            f"Path transaction failed: {exc}",
                        )
                        return

                    continue

            # Gate mutating actions (file writes, shell) behind confirmation
            # before anything touches the local filesystem. Read-only actions
            # are always safe and execute unconditionally.
            requested = [
                entry
                for entry in plan
                if entry.get("action") in ("file_write", "shell")
            ]
            plan = await self._confirm_plan(plan)
            approved = [
                entry
                for entry in plan
                if entry.get("action") in ("file_write", "shell")
            ]

            results = self.tools.execute_planned(plan)

            # All mutating actions were declined. Tell the LLM so it can adapt
            # instead of blindly retrying the same blocked actions.
            if requested and not approved:
                yield AgentEvent(
                    "status",
                    "Local changes declined; asking for an updated plan...",
                )

                decline = (
                    "The user reviewed your proposed FILE and SHELL actions and "
                    "declined them, so nothing was written or run on the local "
                    "machine. Explain what you intended, or propose a different "
                    "approach the user may approve. Use TASK_COMPLETE only when "
                    "the user has clearly accepted the outcome."
                )

                try:
                    last_raw = await self._send_and_wait(decline)
                except Exception as exc:
                    yield AgentEvent(
                        "error",
                        f"Decline transaction failed: {exc}",
                    )
                    return

                continue

            cleaned = clean_llm_text(last_raw)

            if cleaned:
                yield AgentEvent(
                    "response",
                    cleaned,
                    {"round": round_index + 1},
                )

            if not results:
                if "TASK_COMPLETE" in last_raw:
                    summary = _build_turn_summary(all_results, True)
                    report = self.tools.format_agent_report(
                        all_results,
                        cleaned or last_raw.strip(),
                    )
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

                if cleaned:
                    continuation = (
                        "Continue the task. If an action is required, use "
                        "valid [[[FILE ...]]], [[[SHELL]]], or [[[READ ...]]] "
                        "markers. FILE source must be wrapped in a normal "
                        "triple-backtick code fence inside its FILE block. "
                        "Use TASK_COMPLETE when finished."
                    )

                    yield AgentEvent(
                        "status",
                        "No actions found; requesting next action...",
                    )

                    try:
                        last_raw = await self._send_and_wait(
                            continuation
                        )
                    except Exception as exc:
                        yield AgentEvent(
                            "error",
                            f"Continuation transaction failed: {exc}",
                        )
                        return

                    continue

                yield AgentEvent(
                    "error",
                    "LLM returned an empty response.",
                )
                return

            for result in results:
                all_results.append(result)
                result_type = result.get("type", "tool")

                if result_type == "shell":
                    label = f"$ {result.get('command', '')}"
                    meta = {
                        "tool": result_type,
                        "command": result.get("command", ""),
                    }
                elif result_type in ("file_write", "file_read"):
                    verb = (
                        "write"
                        if result_type == "file_write"
                        else "read"
                    )
                    label = f"{verb} {result.get('path', '?')}"
                    meta = {
                        "tool": result_type,
                        "path": result.get("path"),
                    }
                else:
                    label = result_type
                    meta = {"tool": result_type}

                yield AgentEvent("tool_start", label, meta)

                if result.get("error"):
                    body = f"ERROR: {result['error']}"
                elif (
                    result_type == "file_write"
                    and result.get("diff")
                ):
                    body = (
                        f"{result.get('result', '')}\n"
                        f"{result['diff']}"
                    )
                else:
                    body = result.get("result", "")[:2000]

                yield AgentEvent(
                    "tool_result",
                    body,
                    {
                        "tool": result_type,
                        "ok": not bool(result.get("error")),
                        **{
                            key: result[key]
                            for key in (
                                "command",
                                "path",
                                "exit_code",
                                "duration_ms",
                                "mode",
                                "bytes",
                            )
                            if key in result
                        },
                    },
                )

            tool_output = self.tools.format_results(results)
            feedback = build_turn_feedback(results, self._state)

            yield AgentEvent("status", "Sending tool results...")

            try:
                last_raw = await self._send_and_wait(
                    f"{tool_output}\n\n{feedback}"
                )
            except Exception as exc:
                yield AgentEvent(
                    "error",
                    f"Tool-feedback transaction failed: {exc}",
                )
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