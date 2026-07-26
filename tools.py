"""
tools.py - Local tool execution for Browser LLM Agent.

Protocol accepted from browser-hosted models:

[[[SHELL]]]
command
[[[END]]]

[[[FILE relative/path.py]]]
file contents
[[[END]]]

[[[FILE path="relative/path.py"]]]
file contents
[[[END]]]

[[[READ relative/path.py]]]
[[[END]]]
"""

from __future__ import annotations

import difflib
import re
import subprocess
import time
from pathlib import Path
from typing import Any


_FILE_MARKER_RE = re.compile(
    r"""
    \[\[\[\s*FILE\s+
    (?P<path>[^\]\r\n]+?)
    \s*\]\]\]
    (?P<content>.*?)
    \[\[\[\s*END\s*\]\]\]
    """,
    flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
)

_SHELL_MARKER_RE = re.compile(
    r"""
    \[\[\[\s*SHELL\s*\]\]\]
    (?P<command>.*?)
    \[\[\[\s*END\s*\]\]\]
    """,
    flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
)

_READ_MARKER_RE = re.compile(
    r"""
    \[\[\[\s*READ\s+
    (?P<path>[^\]\r\n]+?)
    \s*\]\]\]
    (?:.*?\[\[\[\s*END\s*\]\]\])?
    """,
    flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
)

_OPEN_FENCE_RE = re.compile(
    r"^\s*```[A-Za-z0-9_+#.-]*\s*\n?",
)

_CLOSE_FENCE_RE = re.compile(
    r"\n?\s*```\s*$",
)

_LANGUAGE_LABELS = {
    "asm",
    "assembly",
    "bash",
    "batch",
    "c",
    "c#",
    "c++",
    "clojure",
    "cmake",
    "cpp",
    "csharp",
    "css",
    "csv",
    "dart",
    "dockerfile",
    "elixir",
    "fish",
    "fortran",
    "go",
    "graphql",
    "groovy",
    "haskell",
    "html",
    "ini",
    "java",
    "javascript",
    "jinja",
    "js",
    "json",
    "jsonc",
    "jsx",
    "julia",
    "kotlin",
    "latex",
    "less",
    "lisp",
    "lua",
    "makefile",
    "markdown",
    "md",
    "nginx",
    "objective-c",
    "objc",
    "perl",
    "php",
    "plaintext",
    "powershell",
    "properties",
    "proto",
    "ps1",
    "py",
    "python",
    "r",
    "regex",
    "ruby",
    "rs",
    "rust",
    "sass",
    "scala",
    "scss",
    "shell",
    "sh",
    "sql",
    "swift",
    "text",
    "toml",
    "ts",
    "tsx",
    "typescript",
    "vb",
    "vue",
    "xml",
    "yaml",
    "yml",
    "zsh",
}


def clean_file_content(content: str) -> str:
    """
    Remove browser-chat formatting artifacts immediately before file writing.

    ChatGPT, DeepSeek, Perplexity, Claude, and Gemini may expose a code block's
    language label in copied response text:

        Python
        import os

    The standalone language line is only removed when it is the first non-empty
    line, so normal source content is not modified elsewhere.
    """
    if not content:
        return ""

    content = content.replace("\r\n", "\n").replace("\r", "\n")
    content = content.lstrip("\ufeff")

    # Remove outer Markdown fences if present.
    content = _OPEN_FENCE_RE.sub("", content, count=1)
    content = _CLOSE_FENCE_RE.sub("", content, count=1)

    lines = content.split("\n")

    # Remove blank lines directly inside FILE markers.
    while lines and not lines[0].strip():
        lines.pop(0)

    # Remove a standalone code language label from browser UI extraction.
    if lines and lines[0].strip().lower() in _LANGUAGE_LABELS:
        lines.pop(0)

        while lines and not lines[0].strip():
            lines.pop(0)

    cleaned = "\n".join(lines)
    cleaned = _CLOSE_FENCE_RE.sub("", cleaned)

    # Text source files should end with one final newline.
    return cleaned.rstrip() + "\n" if cleaned.strip() else ""


class ToolExecutor:
    """Parse and execute local FILE, READ, and SHELL tool markers."""

    def __init__(self, config: dict):
        configured_workspace = (
            config.get("workspace")
            or config.get("workspace_dir")
        )

        if configured_workspace:
            self.workspace = Path(
                configured_workspace
            ).expanduser().resolve()
        else:
            self.workspace = Path.cwd().resolve()

        self.workspace.mkdir(parents=True, exist_ok=True)

        tools_config = config.get("tools", {})
        shell_config = tools_config.get("shell", {})
        write_config = tools_config.get("file_write", {})
        read_config = tools_config.get("file_read", {})

        self.shell_enabled = bool(shell_config.get("enabled", True))
        self.file_write_enabled = bool(
            write_config.get("enabled", True)
        )
        self.file_read_enabled = bool(
            read_config.get("enabled", True)
        )

        self.shell_timeout = int(
            config.get(
                "shell_timeout_seconds",
                shell_config.get("timeout_seconds", 120),
            )
        )
        self.max_read_bytes = int(
            config.get(
                "max_read_bytes",
                read_config.get("max_bytes", 100_000),
            )
        )

        self.change_log: list[dict[str, Any]] = []

    def reset_change_log(self) -> None:
        """Clear this task's accumulated file and shell changes."""
        self.change_log.clear()

    def execute_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """
        Parse all complete markers and execute them in their original order.

        A response can mix FILE, SHELL, and READ blocks; preserving their order
        allows models to create files, then compile or inspect them.
        """
        if not text:
            return []

        calls: list[tuple[int, str, re.Match[str]]] = []

        for match in _FILE_MARKER_RE.finditer(text):
            calls.append((match.start(), "file_write", match))

        for match in _SHELL_MARKER_RE.finditer(text):
            calls.append((match.start(), "shell", match))

        for match in _READ_MARKER_RE.finditer(text):
            calls.append((match.start(), "file_read", match))

        calls.sort(key=lambda item: item[0])

        results: list[dict[str, Any]] = []

        for _, kind, match in calls:
            if kind == "file_write":
                results.append(
                    self.write_file(
                        match.group("path"),
                        match.group("content"),
                    )
                )

            elif kind == "shell":
                results.append(self.run_shell(match.group("command")))

            elif kind == "file_read":
                results.append(self.read_file(match.group("path")))

        return results

    def write_file(
        self,
        relative_path: str,
        content: str,
    ) -> dict[str, Any]:
        """Write cleaned text to a workspace-relative file."""
        started = time.monotonic()
        relative_path = self._normalize_tool_path(relative_path)

        if not self.file_write_enabled:
            return self._error_result(
                "file_write",
                "File writing is disabled by configuration.",
                path=relative_path,
                started=started,
            )

        try:
            target = self._resolve_path(relative_path)
        except ValueError as exc:
            return self._error_result(
                "file_write",
                str(exc),
                path=relative_path,
                started=started,
            )

        content = clean_file_content(content)

        if not content.strip():
            return self._error_result(
                "file_write",
                "Refusing to write an empty file after content cleanup.",
                path=relative_path,
                started=started,
            )

        existed = target.exists()
        old_content = ""

        if existed:
            try:
                old_content = target.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return self._error_result(
                    "file_write",
                    (
                        "Refusing to overwrite a non-UTF-8 file: "
                        f"{relative_path}"
                    ),
                    path=relative_path,
                    started=started,
                )
            except OSError as exc:
                return self._error_result(
                    "file_write",
                    f"Could not read existing file: {exc}",
                    path=relative_path,
                    started=started,
                )

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            return self._error_result(
                "file_write",
                f"Could not write {relative_path}: {exc}",
                path=relative_path,
                started=started,
            )

        mode = "modify" if existed else "create"
        byte_count = len(content.encode("utf-8"))
        duration_ms = self._duration_ms(started)
        added, removed = self._diff_counts(old_content, content)

        log_entry: dict[str, Any] = {
            "kind": mode,
            "path": relative_path,
            "bytes": byte_count,
        }

        if existed:
            log_entry["added"] = added
            log_entry["removed"] = removed

        self.change_log.append(log_entry)

        action = "Updated" if existed else "Created"

        return {
            "type": "file_write",
            "path": relative_path,
            "mode": mode,
            "bytes": byte_count,
            "result": f"{action} {relative_path} ({byte_count} bytes).",
            "diff": self._make_diff(
                old_content,
                content,
                relative_path,
            ),
            "duration_ms": duration_ms,
        }

    def read_file(self, relative_path: str) -> dict[str, Any]:
        """Read one UTF-8 text file from the workspace."""
        started = time.monotonic()
        relative_path = self._normalize_tool_path(relative_path)

        if not self.file_read_enabled:
            return self._error_result(
                "file_read",
                "File reading is disabled by configuration.",
                path=relative_path,
                started=started,
            )

        try:
            target = self._resolve_path(relative_path)
        except ValueError as exc:
            return self._error_result(
                "file_read",
                str(exc),
                path=relative_path,
                started=started,
            )

        if not target.exists():
            return self._error_result(
                "file_read",
                f"File does not exist: {relative_path}",
                path=relative_path,
                started=started,
            )

        if not target.is_file():
            return self._error_result(
                "file_read",
                f"Path is not a file: {relative_path}",
                path=relative_path,
                started=started,
            )

        try:
            raw = target.read_bytes()
        except OSError as exc:
            return self._error_result(
                "file_read",
                f"Could not read {relative_path}: {exc}",
                path=relative_path,
                started=started,
            )

        truncated = len(raw) > self.max_read_bytes
        raw = raw[: self.max_read_bytes]

        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            return self._error_result(
                "file_read",
                f"File is not UTF-8 text: {relative_path}",
                path=relative_path,
                started=started,
            )

        if truncated:
            content += (
                f"\n\n... [truncated at {self.max_read_bytes} bytes]"
            )

        return {
            "type": "file_read",
            "path": relative_path,
            "result": content,
            "bytes": len(raw),
            "truncated": truncated,
            "duration_ms": self._duration_ms(started),
        }

    def run_shell(self, command: str) -> dict[str, Any]:
        """Run a shell command with the workspace as its working directory."""
        started = time.monotonic()
        command = command.strip()

        if not self.shell_enabled:
            return self._error_result(
                "shell",
                "Shell execution is disabled by configuration.",
                command=command,
                started=started,
            )

        if not command:
            return self._error_result(
                "shell",
                "Refusing to run an empty command.",
                command=command,
                started=started,
            )

        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=self.shell_timeout,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return self._error_result(
                "shell",
                (
                    f"Command timed out after "
                    f"{self.shell_timeout} seconds."
                ),
                command=command,
                started=started,
            )
        except OSError as exc:
            return self._error_result(
                "shell",
                f"Could not run command: {exc}",
                command=command,
                started=started,
            )

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        output = "\n".join(
            part
            for part in (stdout, stderr)
            if part
        )

        if not output:
            output = (
                "Command completed successfully."
                if completed.returncode == 0
                else (
                    "Command failed with exit code "
                    f"{completed.returncode}."
                )
            )

        duration_ms = self._duration_ms(started)

        self.change_log.append(
            {
                "kind": "shell",
                "command": command,
                "exit_code": completed.returncode,
                "duration_ms": duration_ms,
            }
        )

        result: dict[str, Any] = {
            "type": "shell",
            "command": command,
            "result": output,
            "exit_code": completed.returncode,
            "duration_ms": duration_ms,
        }

        if completed.returncode != 0:
            result["error"] = (
                f"Command exited with code {completed.returncode}."
            )

        return result

    def format_results(self, results: list[dict[str, Any]]) -> str:
        """
        Format bounded tool feedback for the next model transaction.

        Terminal output can be huge, especially `git diff`, build logs, or
        generated files. The browser chat only needs enough context to decide
        the next action, so each result and the combined feedback are capped.
        """
        if not results:
            return "No tool actions were executed."

        max_item_chars = 8_000
        max_total_chars = 20_000
        parts: list[str] = []

        for result in results:
            kind = result.get("type", "tool")
            error = result.get("error")

            if kind == "file_write":
                path = result.get("path", "?")

                if error:
                    item = f"FILE ERROR {path}: {error}"
                else:
                    item = f"FILE OK {path}: {result.get('result', '')}"

            elif kind == "file_read":
                path = result.get("path", "?")
                content = str(result.get("result", ""))

                if error:
                    item = f"READ ERROR {path}: {error}"
                else:
                    item = f"READ OK {path}:\n{content}"

            elif kind == "shell":
                command = str(result.get("command", ""))
                output = str(result.get("result", ""))

                if error:
                    item = (
                        f"SHELL ERROR ({command}):\n"
                        f"{output or error}"
                    )
                else:
                    item = f"SHELL OK ({command}):\n{output}"

            else:
                item = str(result)

            if len(item) > max_item_chars:
                omitted = len(item) - max_item_chars
                item = (
                    item[:max_item_chars]
                    + f"\n\n[output truncated: {omitted} characters omitted]"
                )

            parts.append(item)

        feedback = "\n\n".join(parts)

        if len(feedback) > max_total_chars:
            omitted = len(feedback) - max_total_chars
            feedback = (
                feedback[:max_total_chars]
                + f"\n\n[tool feedback truncated: {omitted} characters omitted]"
            )

        return feedback
    
    
    def format_agent_report(
        self,
        results: list[dict[str, Any]],
        final_text: str = "",
    ) -> str:
        """Build a short structured report for AgentLoop's done event."""
        changed = [
            result.get("path", "?")
            for result in results
            if result.get("type") == "file_write"
            and not result.get("error")
        ]

        failures = [
            result.get("error", "Unknown tool error")
            for result in results
            if result.get("error")
        ]

        lines: list[str] = []

        if changed:
            lines.append("Files changed:")
            lines.extend(f"- {path}" for path in changed)

        if failures:
            lines.append("Errors:")
            lines.extend(f"- {error}" for error in failures)

        if final_text.strip():
            lines.append("Model returned a final response.")

        return "\n".join(lines) if lines else "No workspace changes."

    @staticmethod
    def _normalize_tool_path(value: str) -> str:
        """
        Normalize file paths emitted by different LLM protocol styles.

        Accepts:
            src/main.py
            ./src/main.py
            path="src/main.py"
            path='./src/main.py'
            file="src/main.py"
            filename="src/main.py"
        """
        value = value.strip()

        match = re.match(
            r"^(?:path|file|filename)\s*=\s*(.+?)\s*$",
            value,
            flags=re.IGNORECASE,
        )

        if match:
            value = match.group(1).strip()

        if (
            len(value) >= 2
            and value[0] in ('"', "'")
            and value[-1] == value[0]
        ):
            value = value[1:-1].strip()

        while value.startswith("./") or value.startswith(".\\"):
            value = value[2:]

        return value

    def _resolve_path(self, relative_path: str) -> Path:
        """
        Resolve a safe workspace-relative path.

        Absolute paths and traversal outside the workspace are rejected.
        """
        relative_path = self._normalize_tool_path(relative_path)

        if not relative_path:
            raise ValueError("File path is empty.")

        candidate = Path(relative_path)

        if candidate.is_absolute():
            raise ValueError(
                f"Absolute paths are not allowed: {relative_path}"
            )

        resolved = (self.workspace / candidate).resolve()

        try:
            resolved.relative_to(self.workspace)
        except ValueError as exc:
            raise ValueError(
                f"Path escapes workspace: {relative_path}"
            ) from exc

        return resolved

    @staticmethod
    def _make_diff(
        old_content: str,
        new_content: str,
        relative_path: str,
    ) -> str:
        """Create a bounded unified diff for file-write UI output."""
        if old_content == new_content:
            return f"No content changes in {relative_path}."

        lines = list(
            difflib.unified_diff(
                old_content.splitlines(),
                new_content.splitlines(),
                fromfile=f"a/{relative_path}",
                tofile=f"b/{relative_path}",
                lineterm="",
            )
        )

        limit = 400

        if len(lines) > limit:
            lines = lines[:limit]
            lines.append(f"... diff truncated after {limit} lines")

        return "\n".join(lines)

    @staticmethod
    def _diff_counts(
        old_content: str,
        new_content: str,
    ) -> tuple[int, int]:
        """Return added and removed line counts."""
        added = 0
        removed = 0

        for line in difflib.ndiff(
            old_content.splitlines(),
            new_content.splitlines(),
        ):
            if line.startswith("+ "):
                added += 1
            elif line.startswith("- "):
                removed += 1

        return added, removed

    @staticmethod
    def _duration_ms(started: float) -> int:
        """Return elapsed monotonic time in milliseconds."""
        return int((time.monotonic() - started) * 1000)

    def _error_result(
        self,
        kind: str,
        error: str,
        *,
        path: str | None = None,
        command: str | None = None,
        started: float,
    ) -> dict[str, Any]:
        """Return a consistently shaped failed tool result."""
        result: dict[str, Any] = {
            "type": kind,
            "error": error,
            "result": f"ERROR: {error}",
            "duration_ms": self._duration_ms(started),
        }

        if path is not None:
            result["path"] = path

        if command is not None:
            result["command"] = command
            result["exit_code"] = -1

        return result