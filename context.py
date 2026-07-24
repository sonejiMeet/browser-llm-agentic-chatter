"""
context.py — Workspace context gathering + structured prompt construction.

Before the first LLM call, gathers the workspace state (file tree, git status,
project configs) so the LLM doesn't waste turns on discovery. Builds structured
turn prompts with task progress tracking and error recovery guidance.

No LLM required — all deterministic, fast information gathering.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from privacy import workspace_label


# ── data classes ──────────────────────────────────────────────────

@dataclass
class WorkspaceContext:
    """Snapshot of the workspace before the agent starts working."""
    root_name: str          # folder name only
    file_tree: str          # top-level directory listing (tree view)
    git_branch: str         # current git branch
    git_status: str         # git status --short
    git_recent: str         # recent commits (last 5, --oneline)
    project_type: str       # "python", "node", "rust", "web", "unknown"
    config_files: dict      # {filename: first_20_lines} for key configs
    total_files: int        # rough file count


@dataclass
class TurnState:
    """Tracks what's happened so far in the current task."""
    task: str = ""
    plan: list[str] = field(default_factory=list)  # steps the LLM outlined
    steps_done: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    turn_count: int = 0

    def record_step(self, description: str):
        self.steps_done.append(description)

    def record_file(self, path: str, mode: str):
        if mode == "create":
            if path not in self.files_created:
                self.files_created.append(path)
        else:
            if path not in self.files_modified:
                self.files_modified.append(path)

    def record_error(self, msg: str):
        self.errors.append(msg)

    def summary(self) -> str:
        lines = []
        if self.steps_done:
            lines.append("COMPLETED:")
            for s in self.steps_done[-8:]:
                lines.append(f"  + {s}")
        if self.files_created:
            lines.append(f"Files created: {', '.join(self.files_created[-10:])}")
        if self.files_modified:
            lines.append(f"Files modified: {', '.join(self.files_modified[-10:])}")
        if self.errors:
            lines.append(f"Last error: {self.errors[-1][:200]}")
        return "\n".join(lines)


# ── workspace gathering ───────────────────────────────────────────

# Directories to skip in file tree + counting (build artifacts, caches, profiles)
_SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv", ".env",
    "browser_profile", ".browser_profile", ".playwright",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".eggs", "*.egg-info", ".idea", ".vscode",
}

# Hidden files worth showing (skip the rest)
_KEEP_HIDDEN = {".gitignore", ".env.example", ".editorconfig", ".prettierrc",
                ".eslintrc", ".eslintrc.json", ".babelrc", ".npmrc", ".nvmrc"}

def _should_skip_dir(name: str) -> bool:
    return name in _SKIP_DIRS or name.endswith(".egg-info")

def _should_skip_file(name: str) -> bool:
    return name.startswith(".") and name not in _KEEP_HIDDEN


def _run_cmd(cmd: str, cwd: Path, timeout: float = 5.0) -> str:
    """Run a quick command, return stripped stdout or '' on failure."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=str(cwd),
        )
        return r.stdout.strip()
    except Exception:
        return ""


def _get_file_tree(ws: Path, max_entries: int = 50) -> str:
    """Build a compact tree view of the workspace (top 3 levels),
    skipping build artifacts, caches, and hidden noise."""
    lines = [ws.name + "/"]
    try:
        entries = sorted(ws.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return ws.name + "/  [permission denied]"

    count = 0
    for entry in entries:
        if count >= max_entries:
            lines.append(f"  ... ({_count_files(ws)} source files total, showing first {max_entries} entries)")
            break
        name = entry.name
        if _should_skip_file(name):
            continue
        if entry.is_dir():
            if _should_skip_dir(name):
                continue
            # Show one level deep
            try:
                children = sorted(entry.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            except PermissionError:
                lines.append(f"  {name}/")
                count += 1
                continue
            sub_count = 0
            child_lines = []
            for child in children:
                if sub_count >= 6:
                    child_lines.append("    ...")
                    break
                cn = child.name
                if _should_skip_file(cn):
                    continue
                if child.is_dir() and _should_skip_dir(cn):
                    continue
                marker = "/" if child.is_dir() else ""
                child_lines.append(f"    {cn}{marker}")
                sub_count += 1
            lines.append(f"  {name}/")
            lines.extend(child_lines)
            count += 1
        else:
            lines.append(f"  {name}")
            count += 1
    return "\n".join(lines)


def _count_files(ws: Path) -> int:
    """Count source files, skipping skip-list dirs and hidden files."""
    try:
        total = 0
        for p in ws.rglob("*"):
            if not p.is_file():
                continue
            # Check if any parent dir is in skip list
            parts = p.relative_to(ws).parts
            if any(_should_skip_dir(d) for d in parts):
                continue
            if _should_skip_file(p.name):
                continue
            total += 1
        return total
    except Exception:
        return 0


def _detect_project(ws: Path) -> tuple[str, dict[str, str]]:
    """Detect project type and read key config files."""
    config_files: dict[str, str] = {}
    project_type = "unknown"

    detectors = [
        (["requirements.txt", "setup.py", "pyproject.toml", "Pipfile"], "python"),
        (["package.json", "yarn.lock", "pnpm-lock.yaml"], "node"),
        (["Cargo.toml", "Cargo.lock"], "rust"),
        (["go.mod", "go.sum"], "go"),
        (["Gemfile"], "ruby"),
        (["composer.json"], "php"),
        (["CMakeLists.txt", "Makefile"], "c/c++"),
        (["index.html", "styles.css", "tailwind.config.js"], "web"),
    ]

    for filenames, ptype in detectors:
        for fn in filenames:
            fp = ws / fn
            if fp.exists():
                project_type = ptype
                break
        if project_type != "unknown":
            break

    # Read up to 20 lines from each key config
    key_files = ["requirements.txt", "package.json", "pyproject.toml",
                 "Cargo.toml", "go.mod", "Makefile", "CMakeLists.txt",
                 "setup.py", "config.yaml", ".gitignore"]
    for fn in key_files:
        fp = ws / fn
        if fp.exists():
            try:
                content = fp.read_text(encoding="utf-8")
                lines = content.split("\n")[:20]
                config_files[fn] = "\n".join(lines)
            except Exception:
                config_files[fn] = "[could not read]"

    return project_type, config_files


def gather_workspace_context(workspace: Optional[Path] = None) -> WorkspaceContext:
    """Collect workspace state before the first LLM call.
    Deterministic — no AI, no network, sub-200ms on any project."""
    ws = Path(workspace).resolve() if workspace else Path.cwd()

    file_tree = _get_file_tree(ws)
    git_branch = _run_cmd("git branch --show-current", ws)
    git_status = _run_cmd("git status --short", ws)
    git_recent = _run_cmd("git log --oneline -5", ws)
    project_type, config_files = _detect_project(ws)
    total_files = _count_files(ws)

    return WorkspaceContext(
        root_name=ws.name,
        file_tree=file_tree,
        git_branch=git_branch.strip(),
        git_status=git_status.strip() or "(clean or not a git repo)",
        git_recent=git_recent.strip() or "(no commits or not a git repo)",
        project_type=project_type,
        config_files=config_files,
        total_files=total_files,
    )


# ── prompt builders ───────────────────────────────────────────────

def build_initial_prompt(task: str, ctx: WorkspaceContext) -> str:
    """Build the FIRST message to the LLM with full workspace context.
    The LLM sees the project structure, git state, and key configs before
    it takes any action — no discovery turns wasted."""

    parts = [
        "═══ TASK ═══",
        task,
        "",
        "═══ WORKSPACE ═══",
        f"Project: {ctx.root_name}  |  Type: {ctx.project_type}  |  Files: ~{ctx.total_files}",
        "",
        "File tree:",
        ctx.file_tree,
    ]

    if ctx.git_branch:
        parts.append(f"\nGit: branch={ctx.git_branch}")
    if ctx.git_status and ctx.git_status != "(clean or not a git repo)":
        parts.append(f"Git status:\n{ctx.git_status}")
    if ctx.git_recent and ctx.git_recent != "(no commits or not a git repo)":
        parts.append(f"Recent commits:\n{ctx.git_recent}")

    if ctx.config_files:
        parts.append("\n═══ KEY CONFIG FILES ═══")
        for fn, content in ctx.config_files.items():
            parts.append(f"\n--- {fn} ---\n{content}")

    parts.append("\n═══ APPROACH ═══")
    parts.append(
        "1. OBSERVE — READ any existing files you need to understand the codebase.\n"
        "2. PLAN — outline the steps before you start (write them as plain text).\n"
        "3. ACT — execute ONE step at a time with [[[SHELL]]] or [[[FILE ...]]].\n"
        "4. VERIFY — after each action, check the [TOOL OUTPUT] before the next step.\n"
        "5. COMPLETE — when done, write TASK_COMPLETE on its own line.\n"
        "\n"
        "CRITICAL: Each turn is independent. After a tool result comes back,\n"
        "your response MUST include markers. If you were writing code when\n"
        "interrupted by a tool result, re-output the COMPLETE [[[FILE ...]]]\n"
        "block. Raw code without markers WILL BE IGNORED."
    )

    return "\n".join(parts)


def build_turn_feedback(
    results: list[dict],
    state: TurnState,
) -> str:
    """Build structured feedback after tool execution.
    Gives the LLM: what happened, progress, errors, and what to do next."""

    state.turn_count += 1

    # Record what happened
    for r in results:
        if r.get("type") == "shell" and not r.get("error"):
            cmd = r.get("command", "")[:100]
            state.record_step(f"Ran: {cmd}")
        elif r.get("type") == "file_write" and not r.get("error"):
            state.record_file(r.get("path", "?"), r.get("mode", "modify"))
        elif r.get("error"):
            state.record_error(str(r["error"]))

    parts = ["═══ TURN FEEDBACK ═══"]

    # Progress summary
    progress = state.summary()
    if progress:
        parts.append(progress)
    parts.append(f"Turn: {state.turn_count}")

    # Error context
    has_errors = any(r.get("error") for r in results)
    if has_errors:
        parts.append("\n[!] ERRORS DETECTED — do NOT repeat the same command.")
        parts.append("    Instead: READ the relevant file → find the issue → fix it.")
        for r in results:
            if r.get("error"):
                parts.append(f"    FAILED: {r.get('type')}: {str(r['error'])[:200]}")
        parts.append("    → Your next action MUST be different from what just failed.")
    else:
        parts.append("\nStatus: All commands succeeded.")
        # Directive next-step guidance based on what just happened
        last_type = results[-1].get("type", "") if results else ""
        if last_type == "file_read":
            parts.append(
                'NEXT: Output the file using [[[FILE path="..."]]] markers NOW.\n'
                '      Wrap ALL code in [[[FILE path="name"]]] ... [[[END]]].\n'
                "      Do NOT output raw code without markers — it will be IGNORED."
            )
        elif last_type == "file_write":
            parts.append(
                'NEXT: Verify the file works — run it with [[[SHELL]]], or\n'
                '      write the next file with [[[FILE path="..."]]].'
            )
        elif last_type == "shell":
            parts.append(
                "NEXT: Check the output above. If the task is complete, write\n"
                "      TASK_COMPLETE. Otherwise, continue with the next step."
            )
        else:
            parts.append(
                "NEXT: Use [[[FILE ...]]] to write files, [[[SHELL]]] to run\n"
                "      commands, or TASK_COMPLETE if done. NO raw code without markers."
            )

    return "\n".join(parts)


# ── improved system prompt ────────────────────────────────────────

def build_system_prompt(config: dict, workspace: Optional[str] = None) -> str:
    """Improved system prompt with reasoning framework + marker reference."""

    base = (config.get("system_prompt") or "").rstrip()
    shell_name = "PowerShell" if os.name == "nt" else "bash"
    ws_name = workspace_label(workspace)

    # Perplexity prefix for output format
    prefix = ""
    if config.get("provider") == "perplexity":
        prefix = (
            "OUTPUT FORMAT — follow exactly:\n\n"
            "Shell command:\n"
            "[[[SHELL]]]\n"
            "command\n"
            "[[[END]]]\n\n"
            "Write file:\n"
            '[[[FILE path="name"]]]\n'
            "content\n"
            "[[[END]]]\n\n"
            "Read file:\n"
            '[[[READ path="name"]]]\n\n'
            "No markdown. No backticks. No explanations outside markers.\n"
            "Output ONLY the markers above with the requested content.\n\n"
        )

    env = (
        f"ENVIRONMENT: {shell_name} shell. "
        f"Workspace: {ws_name}/. "
        f"Use relative paths (./file.py). Stay in the workspace."
    )
    if os.name == "nt":
        env += " Use PowerShell commands (dir, New-Item, not ls or mkdir -p)."

    return f"{prefix}{base}\n\n{env}".strip()
