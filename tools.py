"""
tools.py - Local tool execution.
Runs shell commands, reads/writes files. Returns structured results with
git-like change logs for agent feedback to Hermes / the chat LLM.

Outputs destined for the browser chat LLM are privacy-redacted (no home
paths, usernames, or absolute local identity).
"""

import difflib
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from privacy import redact_path_for_llm, redact_text


def _truncate(text: str, limit: int = 6000) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return (
        text[:half]
        + f"\n\n... [{omitted} chars omitted] ...\n\n"
        + text[-half:]
    )


def _unified_diff(path: str, before: str, after: str, max_lines: int = 80) -> str:
    """Produce a compact unified diff of file changes."""
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=2,
        )
    )
    if not diff:
        return f"(no textual diff for {path})"
    if len(diff) > max_lines:
        head = diff[: max_lines - 2]
        head.append(f"... [{len(diff) - max_lines + 2} more diff lines omitted]\n")
        diff = head
    return "".join(diff).rstrip()


class ToolExecutor:
    def __init__(self, config: dict):
        self.shell_enabled = config.get("tools", {}).get("shell", {}).get("enabled", True)
        self.shell_executable = (
            config.get("tools", {}).get("shell", {}).get("executable", None) or None
        )
        self.dangerous_patterns = (
            config.get("tools", {}).get("shell", {}).get("dangerous_patterns", [])
        )
        self.allowed_write_paths = (
            config.get("tools", {}).get("file_write", {}).get("allowed_paths", ["~"])
        )
        self.shell_timeout = int(
            config.get("tools", {}).get("shell", {}).get("timeout", 60)
        )
        self.workspace = Path.cwd()
        # Accumulated change log for the current session / turn
        self.change_log: list[dict] = []

    def reset_change_log(self):
        self.change_log = []

    # ── public API ───────────────────────────────────────────────

    def execute_tool_calls(self, llm_response: str) -> list[dict]:
        """Parse an LLM response for tool-call blocks and execute each one.

        Uses plain-text markers that survive browser DOM rendering
        (markdown fences often lose backticks in inner_text()).
        """
        results: list[dict] = []

        # Collect all tool calls with positions so we execute in document order
        calls: list[tuple[int, str, tuple]] = []

        for match in re.finditer(
            r"\[\[\[SHELL\]{2,}\s*\n?(.*?)\[\[\[END\]{2,}", llm_response, re.DOTALL
        ):
            calls.append((match.start(), "shell", (match.group(1).strip(),)))

        for match in re.finditer(
            r'\[\[\[FILE\s+path=["\']?(.*?)["\']?\]{2,}\s*\n?(.*?)\[\[\[END\]{2,}',
            llm_response,
            re.DOTALL,
        ):
            calls.append((match.start(), "file", (match.group(1).strip(), match.group(2))))

        for match in re.finditer(
            r'\[\[\[READ\s+path=["\']?(.*?)["\']?\]{2,}', llm_response
        ):
            calls.append((match.start(), "read", (match.group(1).strip(),)))

        calls.sort(key=lambda c: c[0])

        for _, kind, args in calls:
            if kind == "shell":
                results.append(self._run_shell(args[0]))
            elif kind == "file":
                results.append(self._write_file(args[0], args[1]))
            elif kind == "read":
                results.append(self._read_file(args[0]))

        return results

    def format_results(self, results: list[dict]) -> str:
        """Format execution results for the browser LLM (privacy-redacted).
        Explicitly flags errors so the LLM knows to FIX, not repeat."""
        if not results:
            return ""
        ws = self.workspace
        lines = ["[TOOL OUTPUT]"]
        has_error = False
        for r in results:
            if r.get("error"):
                has_error = True
                err = redact_text(str(r["error"]), workspace=ws)
                lines.append(f"FAILED ({r['type']}): {err}")
            else:
                body = redact_text(r.get("result", "") or "", workspace=ws)
                if r["type"] == "file_write" and r.get("diff"):
                    path = redact_path_for_llm(r.get("path", "?"), workspace=ws)
                    diff = redact_text(r["diff"], workspace=ws)
                    lines.append(
                        f"OK file_write: wrote {r.get('bytes', '?')} bytes to {path}\n"
                        f"{diff}"
                    )
                else:
                    lines.append(f"OK {r['type']}:\n{_truncate(body, 5000)}")
        lines.append("[/TOOL OUTPUT]")

        if has_error:
            lines.append(
                "The command(s) above FAILED. Read the error, CHANGE your approach, "
                "and try a DIFFERENT fix. Do NOT repeat the same failing command or code. "
                "Use [[[READ ...]]] to inspect files before editing them."
            )
        else:
            lines.append(
                "Continue the task. Issue more [[[...]]] tool calls if needed, "
                "or write SUMMARY + TASK_COMPLETE when done."
            )
        return "\n".join(lines)

    def format_agent_report(self, results: list[dict], cleaned_response: str = "") -> str:
        """Rich report for Hermes / external clients — agent-style activity log."""
        sections: list[str] = []

        if results:
            sections.append("Agent Actions:")
            for i, r in enumerate(results, 1):
                if r["type"] == "shell":
                    cmd = r.get("command", "")
                    status = "ERROR" if r.get("error") else "OK"
                    exit_code = r.get("exit_code", "?")
                    duration = r.get("duration_ms", "?")
                    sections.append(
                        f"\n### [{i}] shell  ({status}, exit={exit_code}, {duration}ms)\n"
                        f"$ {cmd}\n"
                    )
                    body = r.get("error") or r.get("result", "")
                    sections.append(f"```\n{_truncate(body, 3000)}\n```")
                elif r["type"] == "file_write":
                    path = r.get("path", "?")
                    status = "ERROR" if r.get("error") else "OK"
                    if r.get("error"):
                        sections.append(f"\n### [{i}] file_write  ({status})\n{r['error']}")
                    else:
                        mode = r.get("mode", "write")
                        sections.append(
                            f"\n### [{i}] file_write  ({status}, {mode})\n"
                            f"path: {path}  bytes: {r.get('bytes', '?')}"
                        )
                        if r.get("diff"):
                            sections.append(f"```diff\n{r['diff']}\n```")
                        else:
                            sections.append(f"Wrote {r.get('bytes', '?')} bytes.")
                elif r["type"] == "file_read":
                    path = r.get("path", "?")
                    status = "ERROR" if r.get("error") else "OK"
                    sections.append(f"\n### [{i}] file_read  ({status})\npath: {path}")
                    if r.get("error"):
                        sections.append(r["error"])
                    else:
                        sections.append(
                            f"```\n{_truncate(r.get('result', ''), 2000)}\n```"
                        )
                else:
                    sections.append(
                        f"\n### [{i}] {r.get('type', 'tool')}\n"
                        f"{r.get('error') or r.get('result', '')}"
                    )

        # Session change log summary (git-like)
        if self.change_log:
            sections.append("\nChange Log:")
            for entry in self.change_log[-30:]:
                kind = entry.get("kind", "?")
                path = entry.get("path", "")
                if kind == "create":
                    sections.append(f"+ created  {path}  ({entry.get('bytes', 0)} bytes)")
                elif kind == "modify":
                    sections.append(
                        f"~ modified {path}  "
                        f"(+{entry.get('added', 0)} / -{entry.get('removed', 0)} lines)"
                    )
                elif kind == "shell":
                    sections.append(f"$ {entry.get('command', '')[:120]}")
                else:
                    sections.append(f"* {kind} {path}")

        if cleaned_response and cleaned_response.strip():
            sections.append("\n")
            sections.append(cleaned_response.strip())

        return "\n".join(sections).strip()

    # ── internal ─────────────────────────────────────────────────

    def _run_shell(self, command: str) -> dict:
        if not self.shell_enabled:
            return {
                "type": "shell",
                "command": command,
                "error": "Shell execution disabled in config",
            }

        for pattern in self.dangerous_patterns:
            if pattern in command:
                return {
                    "type": "shell",
                    "command": command,
                    "error": f"Command matches dangerous pattern '{pattern}'. Refusing.",
                }

        t0 = time.time()
        try:
            if self.shell_executable:
                result = subprocess.run(
                    [self.shell_executable, "-Command", command],
                    capture_output=True,
                    text=True,
                    timeout=self.shell_timeout,
                    cwd=str(self.workspace),
                )
            else:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.shell_timeout,
                    cwd=str(self.workspace),
                )
            duration_ms = int((time.time() - t0) * 1000)
            output = result.stdout or ""
            if result.stderr:
                output += ("\n[stderr]\n" if output else "[stderr]\n") + result.stderr
            if not output.strip():
                output = f"(exit code {result.returncode}, no output)"
            # Redact home/username paths before any downstream consumer
            output = redact_text(output.strip(), workspace=self.workspace)

            entry = {
                "type": "shell",
                "command": command,
                "result": output,
                "exit_code": result.returncode,
                "duration_ms": duration_ms,
            }
            self.change_log.append({"kind": "shell", "command": command, "exit_code": result.returncode})
            return entry
        except subprocess.TimeoutExpired:
            return {
                "type": "shell",
                "command": command,
                "error": f"Command timed out ({self.shell_timeout}s)",
                "duration_ms": int((time.time() - t0) * 1000),
            }
        except Exception as e:
            return {"type": "shell", "command": command, "error": str(e)}

    def _path_allowed(self, p: Path) -> bool:
        return any(
            str(p).startswith(str(Path(a).expanduser().resolve()))
            for a in self.allowed_write_paths
        )

    def _write_file(self, path_str: str, content: str) -> dict:
        # Normalize content: strip a single trailing newline mismatch noise
        if content.startswith("\n") and not content.startswith("\n\n"):
            # Marker capture sometimes keeps a leading newline — keep content as-is
            pass
        # Drop trailing [[[END]]] leakage if model botched markers
        content = re.sub(r"\n?\[\[\[END\]{2,}\s*$", "", content)

        p = Path(path_str).expanduser()
        if not p.is_absolute():
            p = (self.workspace / p).resolve()
        else:
            p = p.resolve()

        if not self._path_allowed(p):
            safe = redact_path_for_llm(p, workspace=self.workspace)
            return {
                "type": "file_write",
                "path": safe,
                "error": f"Path {safe} is outside the allowed workspace.",
            }

        try:
            before = ""
            existed = p.exists()
            if existed:
                try:
                    before = p.read_text(encoding="utf-8")
                except Exception:
                    before = ""

            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

            rel = self._rel(p)
            diff = _unified_diff(rel, before, content)
            before_lines = before.splitlines()
            after_lines = content.splitlines()
            # Approximate added/removed
            sm = difflib.SequenceMatcher(None, before_lines, after_lines)
            added = removed = 0
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "insert":
                    added += j2 - j1
                elif tag == "delete":
                    removed += i2 - i1
                elif tag == "replace":
                    removed += i2 - i1
                    added += j2 - j1

            mode = "create" if not existed else "modify"
            self.change_log.append(
                {
                    "kind": mode,
                    "path": rel,
                    "bytes": len(content),
                    "added": added,
                    "removed": removed,
                }
            )
            return {
                "type": "file_write",
                "path": rel,
                "bytes": len(content),
                "mode": mode,
                "diff": diff,
                "result": f"Wrote {len(content)} bytes to {rel} ({mode})",
            }
        except Exception as e:
            return {"type": "file_write", "path": path_str, "error": str(e)}

    def _read_file(self, path_str: str) -> dict:
        p = Path(path_str).expanduser()
        if not p.is_absolute():
            p = (self.workspace / p).resolve()
        else:
            p = p.resolve()
        try:
            text = p.read_text(encoding="utf-8")
            # Add line numbers so the LLM can reference specific lines
            lines = text.split("\n")
            numbered = "\n".join(f"{i+1:4d}|{line}" for i, line in enumerate(lines))
            numbered = _truncate(numbered, 8000)
            rel = self._rel(p)
            return {
                "type": "file_read",
                "path": rel,
                "result": numbered,
            }
        except Exception as e:
            return {"type": "file_read", "path": path_str, "error": str(e)}

    def _rel(self, p: Path) -> str:
        try:
            return p.relative_to(self.workspace).as_posix()
        except ValueError:
            # Outside workspace — never return raw absolute path (leaks username)
            return redact_path_for_llm(p, workspace=self.workspace)
