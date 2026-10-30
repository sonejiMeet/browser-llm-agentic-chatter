"""Single outbound privacy policy; it is not a complete DLP system."""
import re
from pathlib import Path

SECRET = re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|password|secret)\s*[=:]\s*[\"']?([^\s\"',;]+)")
TOKEN = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,})\b|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")

def workspace_label(workspace=None):
    return Path(workspace or Path.cwd()).resolve().name

def redact_text(text, workspace=None):
    ws = Path(workspace or Path.cwd()).resolve()
    out = str(text)
    for form in {str(ws), ws.as_posix()}:
        out = out.replace(form, ".")
    for form in {str(Path.home()), Path.home().as_posix()}:
        out = out.replace(form, "~")
    return re.sub(r"(?i)(?:[A-Z]:\\Users\\|/home/|/Users/)[^/\\\s\"']+", "~", out)

def outbound_text(text, workspace=None):
    if TOKEN.search(text) or any(v.lower() not in {"none", "null", "false", "true", "example", "placeholder", "..."} for v in SECRET.findall(text)):
        raise ValueError("Outbound message may contain a credential. Review locally; nothing was submitted.")
    return redact_text(text, workspace)

def redact_path_for_llm(path, workspace=None):
    p = Path(path)
    try:
        return p.resolve().relative_to(Path(workspace or Path.cwd()).resolve()).as_posix()
    except ValueError:
        return redact_text(str(path), workspace)

def is_path_inside_workspace(path, workspace=None):
    return Path(path).resolve().is_relative_to(Path(workspace or Path.cwd()).resolve())
