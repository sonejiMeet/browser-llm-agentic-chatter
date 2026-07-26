"""
context.py - Workspace-context gathering and structured prompt construction.

Before the first LLM call, this module gathers deterministic workspace state:
the file tree, Git status, recent commits, project type, and selected config
files. It also builds the protocol and turn-feedback prompts sent to the LLM.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from privacy import workspace_label


@dataclass
class WorkspaceContext:
    """Snapshot of the workspace before the agent starts working."""

    root_name: str
    file_tree: str
    git_branch: str
    git_status: str
    git_recent: str
    project_type: str
    config_files: dict[str, str]
    total_files: int


@dataclass
class TurnState:
    """Tracks completed work and errors within one user task."""

    task: str = ""
    plan: list[str] = field(default_factory=list)
    steps_done: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    turn_count: int = 0

    def record_step(self, description: str) -> None:
        self.steps_done.append(description)

    def record_file(self, path: str, mode: str) -> None:
        if mode == "create":
            if path not in self.files_created:
                self.files_created.append(path)
        elif path not in self.files_modified:
            self.files_modified.append(path)

    def record_error(self, message: str) -> None:
        self.errors.append(message)

    def summary(self) -> str:
        lines: list[str] = []

        if self.steps_done:
            lines.append("COMPLETED:")
            lines.extend(f"  + {step}" for step in self.steps_done[-8:])

        if self.files_created:
            lines.append(
                "Files created: "
                + ", ".join(self.files_created[-10:])
            )

        if self.files_modified:
            lines.append(
                "Files modified: "
                + ", ".join(self.files_modified[-10:])
            )

        if self.errors:
            lines.append(f"Last error: {self.errors[-1][:200]}")

        return "\n".join(lines)


_SKIP_DIRS = {
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
    ".env",
    "browser_profile",
    ".browser_profile",
    ".playwright",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".eggs",
    ".idea",
    ".vscode",
}

_KEEP_HIDDEN = {
    ".gitignore",
    ".env.example",
    ".editorconfig",
    ".prettierrc",
    ".eslintrc",
    ".eslintrc.json",
    ".babelrc",
    ".npmrc",
    ".nvmrc",
}


def _should_skip_dir(name: str) -> bool:
    return name in _SKIP_DIRS or name.endswith(".egg-info")


def _should_skip_file(name: str) -> bool:
    return name.startswith(".") and name not in _KEEP_HIDDEN


def _run_cmd(cmd: str, cwd: Path, timeout: float = 5.0) -> str:
    """Run a quick local command and return stdout, or an empty string."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd),
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _count_files(workspace: Path) -> int:
    """Count visible source files while skipping caches and build outputs."""
    try:
        total = 0

        for path in workspace.rglob("*"):
            if not path.is_file():
                continue

            parts = path.relative_to(workspace).parts

            if any(_should_skip_dir(part) for part in parts[:-1]):
                continue

            if _should_skip_file(path.name):
                continue

            total += 1

        return total
    except Exception:
        return 0


def _get_file_tree(
    workspace: Path,
    max_entries: int = 50,
) -> str:
    """Build a compact, one-level-deep workspace tree."""
    lines = [f"{workspace.name}/"]

    try:
        entries = sorted(
            workspace.iterdir(),
            key=lambda path: (path.is_file(), path.name.lower()),
        )
    except PermissionError:
        return f"{workspace.name}/  [permission denied]"

    count = 0

    for entry in entries:
        if count >= max_entries:
            lines.append(
                "  ... "
                f"({_count_files(workspace)} source files total, "
                f"showing first {max_entries} entries)"
            )
            break

        name = entry.name

        if _should_skip_file(name):
            continue

        if entry.is_dir():
            if _should_skip_dir(name):
                continue

            try:
                children = sorted(
                    entry.iterdir(),
                    key=lambda path: (
                        path.is_file(),
                        path.name.lower(),
                    ),
                )
            except PermissionError:
                lines.append(f"  {name}/")
                count += 1
                continue

            lines.append(f"  {name}/")
            shown_children = 0

            for child in children:
                if shown_children >= 6:
                    lines.append("    ...")
                    break

                child_name = child.name

                if _should_skip_file(child_name):
                    continue

                if (
                    child.is_dir()
                    and _should_skip_dir(child_name)
                ):
                    continue

                suffix = "/" if child.is_dir() else ""
                lines.append(f"    {child_name}{suffix}")
                shown_children += 1

            count += 1
        else:
            lines.append(f"  {name}")
            count += 1

    return "\n".join(lines)


def _detect_project(
    workspace: Path,
) -> tuple[str, dict[str, str]]:
    """Detect the primary project type and selected configuration files."""
    config_files: dict[str, str] = {}
    project_type = "unknown"

    detectors = [
        (
            ["requirements.txt", "setup.py", "pyproject.toml", "Pipfile"],
            "python",
        ),
        (
            ["package.json", "yarn.lock", "pnpm-lock.yaml"],
            "node",
        ),
        (["Cargo.toml", "Cargo.lock"], "rust"),
        (["go.mod", "go.sum"], "go"),
        (["Gemfile"], "ruby"),
        (["composer.json"], "php"),
        (["CMakeLists.txt", "Makefile"], "c/c++"),
        (
            ["index.html", "styles.css", "tailwind.config.js"],
            "web",
        ),
    ]

    for filenames, detected_type in detectors:
        if any((workspace / filename).exists() for filename in filenames):
            project_type = detected_type
            break

    key_files = [
        "requirements.txt",
        "package.json",
        "pyproject.toml",
        "Cargo.toml",
        "go.mod",
        "Makefile",
        "CMakeLists.txt",
        "setup.py",
        "config.yaml",
        ".gitignore",
    ]

    for filename in key_files:
        path = workspace / filename

        if not path.exists():
            continue

        try:
            content = path.read_text(encoding="utf-8")
            config_files[filename] = "\n".join(
                content.splitlines()[:20]
            )
        except Exception:
            config_files[filename] = "[could not read]"

    return project_type, config_files


def gather_workspace_context(
    workspace: Optional[Path] = None,
) -> WorkspaceContext:
    """Collect deterministic workspace context before the first LLM call."""
    resolved_workspace = (
        Path(workspace).resolve()
        if workspace
        else Path.cwd().resolve()
    )

    project_type, config_files = _detect_project(resolved_workspace)

    return WorkspaceContext(
        root_name=resolved_workspace.name,
        file_tree=_get_file_tree(resolved_workspace),
        git_branch=_run_cmd(
            "git branch --show-current",
            resolved_workspace,
        ),
        git_status=(
            _run_cmd("git status --short", resolved_workspace)
            or "(clean or not a git repo)"
        ),
        git_recent=(
            _run_cmd("git log --oneline -5", resolved_workspace)
            or "(no commits or not a git repo)"
        ),
        project_type=project_type,
        config_files=config_files,
        total_files=_count_files(resolved_workspace),
    )


def build_initial_prompt(task: str, ctx: WorkspaceContext) -> str:
    """
    Build the initial task message.

    The system prompt defines the marker syntax. This message adds the task and
    workspace snapshot without repeating a large protocol on every turn.
    """
    parts = [
        "=== TASK ===",
        task,
        "",
        "=== WORKSPACE ===",
        (
            f"Project: {ctx.root_name} | Type: {ctx.project_type} | "
            f"Files: ~{ctx.total_files}"
        ),
        "",
        "File tree:",
        ctx.file_tree,
    ]

    if ctx.git_branch:
        parts.append(f"\nGit branch: {ctx.git_branch}")

    if ctx.git_status != "(clean or not a git repo)":
        parts.append(f"Git status:\n{ctx.git_status}")

    if ctx.git_recent != "(no commits or not a git repo)":
        parts.append(f"Recent commits:\n{ctx.git_recent}")

    if ctx.config_files:
        parts.append("\n=== KEY CONFIG FILES ===")

        for filename, content in ctx.config_files.items():
            parts.append(f"\n--- {filename} ---\n{content}")

    parts.append(
        "\n=== WORKFLOW ===\n"
        "1. OBSERVE: READ only files needed to understand the task.\n"
        "2. PLAN: state a short plan in ordinary text.\n"
        "3. ACT: perform one coherent step using a tool marker.\n"
        "4. VERIFY: use the TOOL OUTPUT before deciding the next action.\n"
        "5. COMPLETE: write TASK_COMPLETE on its own line when finished.\n\n"
        "For every file write, output the entire file in one FILE block. "
        "Place normal fenced source code inside the FILE block. Do not use "
        "arrow indentation, indentation markers, escaped line breaks, or "
        "raw code outside a FILE block."
    )

    return "\n".join(parts)


def build_turn_feedback(
    results: list[dict],
    state: TurnState,
) -> str:
    """Build deterministic feedback after a group of tool calls completes."""
    state.turn_count += 1

    for result in results:
        if (
            result.get("type") == "shell"
            and not result.get("error")
        ):
            command = result.get("command", "")[:100]
            state.record_step(f"Ran: {command}")

        elif (
            result.get("type") == "file_write"
            and not result.get("error")
        ):
            state.record_file(
                result.get("path", "?"),
                result.get("mode", "modify"),
            )

        elif result.get("error"):
            state.record_error(str(result["error"]))

    parts = ["=== TURN FEEDBACK ==="]

    progress = state.summary()
    if progress:
        parts.append(progress)

    parts.append(f"Turn: {state.turn_count}")

    errors = [result for result in results if result.get("error")]

    if errors:
        parts.append(
            "\nERRORS DETECTED: do not repeat the same failed action."
        )
        parts.append(
            "Read the relevant file or command output, identify the cause, "
            "then issue a changed corrective action."
        )

        for result in errors:
            parts.append(
                f"FAILED {result.get('type', 'tool')}: "
                f"{str(result['error'])[:200]}"
            )

        return "\n".join(parts)

    parts.append("\nStatus: all tool calls succeeded.")

    last_type = results[-1].get("type", "") if results else ""

    if last_type == "file_read":
        parts.append(
            "NEXT: If you need to change or create source, write a complete "
            "FILE block now. Put source in a normal ```language fenced code "
            "block inside [[[FILE ...]]] and [[[END]]]."
        )
    elif last_type == "file_write":
        parts.append(
            "NEXT: Verify the changed file with SHELL when practical, or "
            "write the next required file using a complete FILE block."
        )
    elif last_type == "shell":
        parts.append(
            "NEXT: Inspect the command output. Continue with the next "
            "necessary action, or write TASK_COMPLETE if the task is done."
        )
    else:
        parts.append(
            "NEXT: Use FILE, SHELL, or READ markers for actions. Use "
            "TASK_COMPLETE only when all requested work is finished."
        )

    return "\n".join(parts)


def build_system_prompt(
    config: dict,
    workspace: Optional[str] = None,
) -> str:
    """
    Build the stable agent protocol.

    Source code is requested in an ordinary Markdown code fence because LLMs
    preserve whitespace inside fenced blocks much more reliably than in prose.
    ToolExecutor removes that outer fence before writing the file.
    """
    base = (config.get("system_prompt") or "").rstrip()
    shell_name = "PowerShell" if os.name == "nt" else "bash"
    workspace_name = workspace_label(workspace)

    protocol = r'''
TOOL PROTOCOL

Use these exact markers only when taking an action.

Run a shell command:

[[[SHELL]]]
command here
[[[END]]]

Read a file:

[[[READ path="relative/path.ext"]]]

Write a complete text file:

[[[FILE path="relative/path.ext"]]]
```language
exact complete file contents
```
[[[END]]]

FILE RULES

- Write the whole target file in a single FILE block.
- Inside FILE, use a normal triple-backtick code fence, with a language label
  when known, for example ```python, ```c, ```json, or ```text.
- The code fence is part of the transport format; it is removed before the
  file is written.
- Preserve source exactly inside the code fence: indentation, tabs/spaces,
  blank lines, quotes, backslashes, Unicode, and identifiers.
- Never use arrows, indentation prefixes, visible whitespace markers, escaped
  newline sequences, or Markdown lists to represent code indentation.
- Preserve Python dunder identifiers exactly, including __name__, __main__,
  __file__, __init__, __all__, and magic methods such as __enter__.
- Do not put explanations, comments about the protocol, or extra text inside
  a FILE block.
- Do not output raw source code outside a FILE block.
- Use relative paths only. Never attempt to write outside the workspace.

After receiving TOOL OUTPUT:
- Fix failures with a meaningfully changed action.
- Do not repeat a failed command or identical invalid file content.
- When the task is fully complete, write TASK_COMPLETE on its own line.
'''.strip()

    environment = (
        f"ENVIRONMENT: {shell_name} shell. "
        f"Workspace: {workspace_name}/. "
        "Use relative paths and remain inside the workspace."
    )

    if os.name == "nt":
        environment += (
            " Use PowerShell syntax, such as dir and New-Item, rather than "
            "Unix-only commands."
        )

    pieces = [piece for piece in (base, environment, protocol) if piece]

    return "\n\n".join(pieces)