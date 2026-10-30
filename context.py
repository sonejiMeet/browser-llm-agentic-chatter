"""Small deterministic workspace snapshot; source is retrieved through tools."""
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from protocol import build_system_prompt

SKIP = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', 'browser_profile', '.browser-agent', 'dist', 'build'}

@dataclass
class WorkspaceContext:
    root_name: str
    file_tree: str
    git_branch: str
    git_status: str
    project_type: str
    total_files: int

def gather_workspace_context(workspace=None):
    root = Path(workspace or Path.cwd()).resolve()
    entries = []
    total = 0
    for parent, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not (Path(parent) / d).is_symlink())
        for name in sorted(names):
            if name.startswith('.') and name not in {'.gitignore', '.editorconfig'}:
                continue
            total += 1
            if len(entries) < 100:
                entries.append((Path(parent) / name).relative_to(root).as_posix())
        if total >= 10000:
            break
    def git(*args):
        try:
            return subprocess.run(['git', *args], cwd=root, capture_output=True, text=True, timeout=3).stdout[:4000].strip()
        except (OSError, subprocess.TimeoutExpired):
            return ''
    markers = {'pyproject.toml': 'python', 'requirements.txt': 'python', 'package.json': 'node', 'Cargo.toml': 'rust', 'go.mod': 'go', 'CMakeLists.txt': 'c/c++'}
    kind = next((v for k, v in markers.items() if (root/k).exists()), 'unknown')
    return WorkspaceContext(root.name, '\n'.join(entries), git('branch', '--show-current'), git('status', '--short'), kind, total)
