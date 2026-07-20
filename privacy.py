"""
privacy.py — Redact private local identity before anything is sent to the
browser chat LLM (ChatGPT / Claude / etc.).

Strips:
  - Windows/macOS/Linux home paths and usernames
  - Absolute paths outside the active workspace (shown as relative or redacted)
  - Unrelated conversation history / local project dumps when not needed
"""

from __future__ import annotations

import getpass
import os
import re
from pathlib import Path
from typing import Optional


def _home() -> Path:
    try:
        return Path.home().resolve()
    except Exception:
        return Path(os.path.expanduser("~")).resolve()


def _username() -> str:
    try:
        return getpass.getuser() or ""
    except Exception:
        return ""


def workspace_label(workspace: Optional[Path | str] = None) -> str:
    """Public name for the workspace — folder basename only, never full path."""
    try:
        p = Path(workspace) if workspace else Path.cwd()
        p = p.resolve()
        name = p.name or "workspace"
        # Never include parent path segments (they can contain the username)
        return name
    except Exception:
        return "workspace"


def redact_text(text: str, workspace: Optional[Path | str] = None) -> str:
    """Remove home dirs, usernames, and absolute local paths from text."""
    if not text:
        return text

    try:
        ws = Path(workspace).resolve() if workspace else Path.cwd().resolve()
    except Exception:
        ws = Path.cwd()

    home = _home()
    user = _username()
    out = text

    # 1) Replace workspace absolute path with relative-friendly token first
    ws_str = str(ws)
    ws_posix = ws.as_posix()
    for form in (ws_str, ws_posix):
        if form and form in out:
            out = out.replace(form, ".")

    # Also handle workspace with trailing separators
    for sep in ("\\", "/"):
        prefix = ws_str.rstrip("\\/") + sep
        if prefix in out:
            out = out.replace(prefix, "./" if sep == "/" else ".\\")

    # 2) Replace home directory (and common variants)
    home_str = str(home)
    home_posix = home.as_posix()
    for form in (home_str, home_posix):
        if form and form in out:
            out = out.replace(form, "~")

    # Windows short / mixed-case variants of home
    if os.name == "nt":
        # C:\Users\<name> and /c/Users/<name>
        if user:
            out = re.sub(
                rf"(?i)[A-Z]:\\Users\\{re.escape(user)}",
                r"~",
                out,
            )
            out = re.sub(
                rf"(?i)/([A-Za-z])/Users/{re.escape(user)}",
                r"~",
                out,
            )
            out = re.sub(
                rf"(?i)\\Users\\{re.escape(user)}",
                r"\\Users\\USER",
                out,
            )

    # 3) Generic absolute path patterns that still leak location
    #    Windows: C:\...  Unix: /home/... /Users/...
    out = re.sub(
        r"(?i)[A-Z]:\\Users\\[^\\/\s\"']+",
        r"~",
        out,
    )
    out = re.sub(
        r"(?i)/Users/[^/\s\"']+",
        r"~",
        out,
    )
    out = re.sub(
        r"(?i)/home/[^/\s\"']+",
        r"~",
        out,
    )

    # 4) Drop bare username tokens only when they look like path components
    #    (avoid rewriting normal English if username is a common word)
    if user and len(user) >= 3:
        
        out = re.sub(
            rf"(?i)([\\/]){re.escape(user)}([\\/])",
            r"\1USER\2",
            out,
        )

    return out


def redact_path_for_llm(path: str | Path, workspace: Optional[Path | str] = None) -> str:
    """Prefer a workspace-relative path; otherwise redact absolute private parts."""
    try:
        ws = Path(workspace).resolve() if workspace else Path.cwd().resolve()
        p = Path(path)
        if not p.is_absolute():
            return str(path).replace("\\", "/")
        p = p.resolve()
        try:
            rel = p.relative_to(ws)
            return rel.as_posix() or "."
        except ValueError:
            return redact_text(str(p), workspace=ws)
    except Exception:
        return redact_text(str(path), workspace=workspace)


def is_path_inside_workspace(path: Path | str, workspace: Optional[Path | str] = None) -> bool:
    try:
        ws = Path(workspace).resolve() if workspace else Path.cwd().resolve()
        p = Path(path).expanduser().resolve()
        p.relative_to(ws)
        return True
    except Exception:
        return False
