"""Validated local tools, exact edits, bounded logs and async process control."""
import asyncio
import difflib
import fnmatch
import hashlib
import os
import shutil
import signal
import stat
import tempfile
import time
import uuid
from pathlib import Path
from protocol import validate_action

SKIP = {".git", ".venv", "venv", "node_modules", "__pycache__", "browser_profile", ".browser-agent", "build", "dist"}
SENSITIVE = {".env", ".npmrc", ".pypirc", "credentials", "id_rsa", "id_ed25519"}

def digest(data):
    return hashlib.sha256(data).hexdigest()

class ToolExecutor:
    def __init__(self, config):
        self.config = config
        self.workspace = Path(config.get("workspace") or config.get("workspace_dir") or Path.cwd()).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.state_dir = Path(config.get("state_dir") or self.workspace / ".browser-agent").resolve()
        self.logs = self.state_dir / "outputs"
        self.logs.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.change_log = []
        configured = config.get("tools", {}).get("shell", {}).get("executable")
        self.shell = configured or ("powershell.exe" if os.name == "nt" else "/bin/bash")
        self.max_output = int(config.get("max_output_bytes", 2_000_000))

    def reset_change_log(self):
        self.change_log.clear()

    def _resolve_path(self, path=".", disclose=True):
        p = Path(path)
        target = (self.workspace / p).resolve()
        if p.is_absolute() or not target.is_relative_to(self.workspace):
            raise ValueError("Path must remain inside workspace")
        parts = target.relative_to(self.workspace).parts
        if target == self.state_dir or target.is_relative_to(self.state_dir) or any(x in SKIP for x in parts):
            raise ValueError("Internal/cache directory denied")
        if disclose and any(x in SENSITIVE or (x.startswith(".env.") and x != ".env.example") or x.endswith((".pem", ".key")) for x in parts):
            raise ValueError("Sensitive file disclosure denied")
        return target

    def _files(self, base=".", glob="*"):
        basepath = self._resolve_path(base)
        count = 0
        for parent, dirs, names in os.walk(basepath, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP and not (Path(parent) / d).is_symlink())
            for name in sorted(names):
                count += 1
                if count > 50_000:
                    raise ValueError("File enumeration limit reached; narrow the path")
                path = Path(parent) / name
                rel = path.relative_to(self.workspace).as_posix()
                try:
                    self._resolve_path(rel)
                except ValueError:
                    continue
                if not path.is_symlink() and fnmatch.fnmatch(rel, glob):
                    yield rel, path

    def _enabled(self, tool):
        key = "shell" if tool.startswith("run_") else "file_write" if tool in {"write_file", "edit_file"} else "file_read"
        if not self.config.get("tools", {}).get(key, {}).get("enabled", True):
            raise ValueError(f"{key} disabled")

    def prepare(self, action):
        validate_action(action)
        self._enabled(action["tool"])
        tool, args = action["tool"], action["arguments"]
        preview = {"action": "shell" if tool.startswith("run_") else "file_write" if tool in {"edit_file", "write_file"} else "file_read", "id": action["id"], "tool": tool, **args}
        if tool not in {"write_file", "edit_file"}:
            return preview
        path = self._resolve_path(args["path"])
        old = path.read_bytes() if path.exists() else None
        if old is not None and len(old) > 2_000_000:
            raise ValueError("File too large for editing")
        expected = args["expected_sha256"]
        if (digest(old) if old is not None else None) != expected:
            raise ValueError("Stale file version; read the file again")
        before = old.decode("utf-8") if old is not None else ""
        after = args.get("content")
        if tool == "edit_file":
            spans = []
            for edit in args["replacements"]:
                needle = edit["old_text"]
                if before.count(needle) != 1:
                    raise ValueError("old_text must match exactly once")
                start = before.index(needle)
                spans.append((start, start + len(needle), edit["new_text"]))
            spans.sort()
            if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
                raise ValueError("Overlapping replacements")
            after = before
            for start, end, replacement in reversed(spans):
                after = after[:start] + replacement + after[end:]
        if len(after.encode("utf-8")) > 2_000_000:
            raise ValueError("Candidate file exceeds size limit")
        diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile="a/" + args["path"], tofile="b/" + args["path"]))
        return {**preview, "content": after, "diff": diff, "base_sha256": expected}

    def _write(self, args, prepared):
        path = self._resolve_path(args["path"])
        old = path.read_bytes() if path.exists() else None
        if (digest(old) if old is not None else None) != prepared["base_sha256"]:
            raise ValueError("File changed after approval")
        # Local backups support manual file recovery, not shell rollback.
        backup = self.state_dir / "backups" / uuid.uuid4().hex
        backup.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if old is not None:
            backup.write_bytes(old)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = prepared["content"].encode("utf-8")
        fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".agent-")
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
            if path.exists():
                os.chmod(temp, stat.S_IMODE(path.stat().st_mode))
            current = path.read_bytes() if path.exists() else None
            if (digest(current) if current is not None else None) != prepared["base_sha256"]:
                raise ValueError("File changed during write preparation")
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        self.change_log.append({"kind": "create" if old is None else "modify", "path": args["path"]})
        return {"path": args["path"], "sha256": digest(data), "diff": prepared["diff"], "backup_id": backup.name if old is not None else None}

    async def _process(self, tool, args, emit):
        cwd = self._resolve_path(args.get("cwd", "."))
        timeout = min(args.get("timeout", int(self.config.get("shell_timeout_seconds", 90))), 3600)
        argv = args.get("argv")
        if tool == "run_shell":
            shell = str(self.shell)
            name = Path(shell).name.lower()
            flags = ["/d", "/s", "/c"] if name in {"cmd", "cmd.exe"} else ["-NoProfile", "-NonInteractive", "-Command"] if name in {"pwsh", "pwsh.exe", "powershell", "powershell.exe"} else ["-c"]
            argv = [shell, *flags, args["command"]]
        prefix = self.config.get("execution_prefix", [])
        if not isinstance(prefix, list) or not all(isinstance(x, str) for x in prefix):
            raise ValueError("execution_prefix must be a trusted argv list")
        argv = [*prefix, *argv]
        output_id = uuid.uuid4().hex
        output = self.logs / (output_id + ".log")
        options = {"start_new_session": True} if os.name != "nt" else {"creationflags": 0x00000200}
        proc = await asyncio.create_subprocess_exec(*argv, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, **options)
        written = 0
        discarded = 0
        async def drain():
            nonlocal written, discarded
            with output.open("wb") as log:
                while chunk := await proc.stdout.read(4096):
                    kept = chunk[:max(0, self.max_output - written)]
                    log.write(kept)
                    written += len(kept)
                    discarded += len(chunk) - len(kept)
                    if emit and kept:
                        await emit(kept.decode("utf-8", "replace"))
        async def kill_tree():
            if os.name == "nt":
                killer = await asyncio.create_subprocess_exec("taskkill", "/PID", str(proc.pid), "/T", "/F", stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                await killer.wait()
            else:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            await proc.wait()
        reader = asyncio.create_task(drain())
        timed_out = False
        try:
            await asyncio.wait_for(asyncio.gather(proc.wait(), asyncio.shield(reader)), timeout)
        except TimeoutError:
            timed_out = True
            await kill_tree()
            await reader
        except BaseException:
            await kill_tree()
            await reader
            raise
        raw = output.read_bytes()
        preview = raw if len(raw) <= 8000 else raw[:4000] + b"\n... use read_output ...\n" + raw[-4000:]
        result = {"output_id": output_id, "output": preview.decode("utf-8", "replace"), "exit_code": proc.returncode, "stored_bytes": written, "discarded_bytes": discarded, "timed_out": timed_out}
        if timed_out or proc.returncode:
            result["error"] = "Command timed out" if timed_out else f"Exit code {proc.returncode}"
        self.change_log.append({"kind": "shell", "command": str(argv), "exit_code": proc.returncode})
        return result

    async def execute(self, action, prepared=None, emit=None):
        started = time.monotonic()
        try:
            self._enabled(action["tool"])
            validate_action(action)
            tool, args = action["tool"], action["arguments"]
            if tool in {"write_file", "edit_file"}:
                result = self._write(args, prepared or self.prepare(action))
            elif tool == "read_file":
                path = self._resolve_path(args["path"])
                if path.stat().st_size > 5_000_000:
                    raise ValueError("File too large")
                raw = path.read_bytes()
                lines = raw.decode("utf-8").splitlines(keepends=True)
                start = args.get("start_line", 1)
                end = min(args.get("end_line", start + 119), len(lines), start + 399)
                selection = "".join(f"{i + 1}|{lines[i]}" for i in range(start - 1, end))
                if len(selection) > 8_000:
                    raise ValueError("Selected lines too long; narrow the range")
                result = {"path": args["path"], "sha256": digest(raw), "total_lines": len(lines), "start_line": start, "end_line": end, "content": selection, "next_line": end + 1 if end < len(lines) else None}
            elif tool == "list_files":
                limit = min(args.get("limit", 100), 1000)
                found = []
                for rel, _ in self._files(args.get("path", "."), args.get("glob", "*")):
                    found.append(rel)
                    if len(found) > limit:
                        break
                result = {"files": found[:limit], "truncated": len(found) > limit}
            elif tool == "search_text":
                hits = []
                limit = min(args.get("limit", 30), 100)
                context = min(args.get("context_lines", 2), 5)
                for rel, path in self._files(glob=args.get("glob", "*")):
                    if path.stat().st_size > 1_000_000:
                        continue
                    try:
                        lines = path.read_text(encoding="utf-8").splitlines()
                    except (UnicodeError, OSError):
                        continue
                    for i, line in enumerate(lines):
                        if args["pattern"] in line:
                            hits.append({"path": rel, "line": i + 1, "context": [f"{j + 1}|{lines[j][:500]}" for j in range(max(0, i-context), min(len(lines), i+context+1))]})
                            if len(hits) >= limit:
                                break
                    if len(hits) >= limit:
                        break
                result = {"matches": hits, "limit_reached": len(hits) >= limit}
            elif tool == "read_output":
                oid = args["output_id"]
                if len(oid) != 32 or any(c not in "0123456789abcdef" for c in oid):
                    raise ValueError("Invalid output_id")
                with (self.logs / (oid + ".log")).open("rb") as stream:
                    stream.seek(args.get("offset", 0))
                    data = stream.read(min(args.get("limit", 8000), 24000))
                    result = {"output": data.decode("utf-8", "replace"), "next_offset": stream.tell(), "bytes": len(data)}
            else:
                result = await self._process(tool, args, emit)
            return {"id": action["id"], "type": tool, **result, "duration_ms": int((time.monotonic()-started)*1000)}
        except (ValueError, OSError, UnicodeError) as exc:
            return {"id": action["id"], "type": action["tool"], "error": str(exc)}
