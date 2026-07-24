#!/usr/bin/env python3
"""Browser LLM Agent — one-command setup for Linux, WSL2, and Windows.

Usage:
  python setup.py        # creates .venv, installs deps, sets up Playwright
  python setup.py --help

After setup, run:
  python cli.py          # interactive REPL
  python agent.py "task" # single-shot
"""

import os
import platform
import subprocess
import sys
import venv
from pathlib import Path

# ── Color codes ──────────────────────────────────────────────────
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
CYAN = '\033[0;36m'
BOLD = '\033[1m'
NC = '\033[0m'  # No Color

ROOT = Path(__file__).resolve().parent
VENV_DIR = Path(".venv")
IS_WIN = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

PYTHON = sys.executable
if IS_WIN:
    VENV_PYTHON = str(VENV_DIR / "Scripts" / "python.exe")
    PIP = str(VENV_DIR / "Scripts" / "pip.exe")
    ACTIVATE = str(VENV_DIR / "Scripts" / "activate")
else:
    VENV_PYTHON = str(VENV_DIR / "bin" / "python")
    PIP = str(VENV_DIR / "bin" / "pip")
    ACTIVATE = f"source {VENV_DIR}/bin/activate"


def run(cmd: str, **kw) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"  {CYAN}$ {cmd[:100]}{'...' if len(cmd) > 100 else ''}{NC}")
    return subprocess.run(cmd, shell=True, check=False, **kw)


def step(msg: str):
    """Print a step header."""
    print(f"\n{BOLD}{CYAN}{'═' * 60}{NC}")
    print(f"  {BOLD}{msg}{NC}")
    print(f"{BOLD}{CYAN}{'═' * 60}{NC}")


def success(msg: str):
    """Print a success message in green."""
    print(f"  {GREEN}✓ {msg}{NC}")


def error(msg: str):
    """Print an error message in red."""
    print(f"  {RED}✗ {msg}{NC}")


def warning(msg: str):
    """Print a warning message in yellow."""
    print(f"  {YELLOW}⚠ {msg}{NC}")


def info(msg: str):
    """Print an info message in cyan."""
    print(f"  {CYAN}ℹ {msg}{NC}")


def main():
    os.chdir(ROOT)

    # ── 1. Python version ────────────────────────────────────────
    step("Checking Python version")
    v = sys.version_info
    if v < (3, 11):
        error(f"Python 3.11+ required, found {v.major}.{v.minor}")
        print(f"  {RED}Please upgrade Python to 3.11 or newer.{NC}")
        sys.exit(1)
    success(f"Python {v.major}.{v.minor}.{v.micro}")

    # ── 2. Virtual environment ───────────────────────────────────
    if not VENV_DIR.exists():
        step("Creating virtual environment")
        try:
            venv.create(VENV_DIR, with_pip=True, clear=False)
            success(f"Created {VENV_DIR}")
        except Exception as e:
            error(f"Failed to create virtual environment: {e}")
            sys.exit(1)
    else:
        info(f"Using existing venv: {VENV_DIR}")

    # ── 3. Install Python dependencies ───────────────────────────
    step("Installing Python packages")
    
    # Upgrade pip in the venv
    result = run(f'"{VENV_PYTHON}" -m pip install --upgrade pip', cwd=ROOT)
    if result.returncode == 0:
        success("pip upgraded")
    else:
        error("Failed to upgrade pip")
    
    # Install requirements using venv's pip
    result = run(f'"{PIP}" install -r requirements.txt', cwd=ROOT)
    if result.returncode == 0:
        success("Python packages installed")
    else:
        error("Failed to install Python packages")
        print(f"  {RED}Check requirements.txt and try again.{NC}")
        sys.exit(1)

    # ── 4. Install Playwright browsers ───────────────────────────
    step("Installing Playwright Chromium")
    result = run(f'"{VENV_PYTHON}" -m playwright install chromium', cwd=ROOT)
    if result.returncode == 0:
        success("Playwright Chromium installed")
    else:
        error("Playwright install failed — retrying with explicit path...")
        run(f'"{PIP}" install --force-reinstall playwright', cwd=ROOT)
        result = run(f'"{VENV_PYTHON}" -m playwright install chromium', cwd=ROOT)
        if result.returncode == 0:
            success("Playwright Chromium installed (after retry)")
        else:
            error("Playwright Chromium installation failed")
            sys.exit(1)

    # ── 5. System dependencies (Linux / WSL2 only) ───────────────
    if IS_LINUX:
        step("Installing system dependencies (Linux/WSL2)")
        result = run(f'"{VENV_PYTHON}" -m playwright install-deps chromium', cwd=ROOT)
        if result.returncode == 0:
            success("System dependencies installed")
        else:
            error("System dependencies installation failed")
            print(f"  {YELLOW}You may need to run manually:{NC}")
            print(f"  {YELLOW}sudo apt-get install -y \\{NC}")
            print(f"  {YELLOW}  libnspr4 libnss3 libatk-bridge2.0-0 libcups2 libdrm2 \\{NC}")
            print(f"  {YELLOW}  libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 \\{NC}")
            print(f"  {YELLOW}  libgbm1 libpango-1.0-0 libcairo2 libasound2t64{NC}")
            print(f"  {YELLOW}playwright install-deps chromium{NC}")

    # ── 6. Verify ────────────────────────────────────────────────
    step("Verifying installation")
    ok = True
    for mod in ["playwright", "yaml", "prompt_toolkit", "rich"]:
        # Use VENV_PYTHON to check packages in the virtual environment
        r = run(f'"{VENV_PYTHON}" -c "import {mod}; print(\'{mod} OK\')"', cwd=ROOT,
                capture_output=True, text=True)
        if r.returncode != 0:
            error(f"{mod} MISSING")
            ok = False
        else:
            success(f"{mod} verified")
    
    if ok:
        success("All packages verified")
    else:
        error("Some packages are missing")
        sys.exit(1)

    # ── 7. Done ──────────────────────────────────────────────────
    print(f"""
{BOLD}{GREEN}{'═' * 60}{NC}
  {BOLD}{GREEN}✅ Setup complete!{NC}

  To run the REPL:
    {CYAN}{ACTIVATE}{NC}
    {CYAN}python cli.py{NC}

  Single-shot:
    {CYAN}{ACTIVATE}{NC}
    {CYAN}python agent.py "your task here"{NC}

  {YELLOW}Providers: chatgpt (default), claude, gemini, perplexity{NC}
  {YELLOW}Edit config.yaml to change provider or model.{NC}
  {YELLOW}Login in the browser window on first run.{NC}
  {YELLOW}Profile persists in ./browser_profile/{NC}
{BOLD}{GREEN}{'═' * 60}{NC}
""")


if __name__ == "__main__":
    main()