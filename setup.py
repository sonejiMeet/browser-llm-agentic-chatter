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

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
IS_WIN = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

PYTHON = sys.executable
if IS_WIN:
    PIP = str(VENV_DIR / "Scripts" / "pip.exe")
    ACTIVATE = str(VENV_DIR / "Scripts" / "activate")
else:
    PIP = str(VENV_DIR / "bin" / "pip")
    ACTIVATE = f"source {VENV_DIR}/bin/activate"


def run(cmd: str, **kw) -> subprocess.CompletedProcess:
    print(f"  $ {cmd[:100]}{'...' if len(cmd) > 100 else ''}")
    return subprocess.run(cmd, shell=True, check=False, **kw)


def step(msg: str):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def main():
    os.chdir(ROOT)

    # ── 1. Python version ────────────────────────────────────────
    step("Checking Python version")
    v = sys.version_info
    if v < (3, 11):
        sys.exit(f"Python 3.11+ required, found {v.major}.{v.minor}")
    print(f"  Python {v.major}.{v.minor}.{v.micro} ✓")

    # ── 2. Virtual environment ───────────────────────────────────
    if not VENV_DIR.exists():
        step("Creating virtual environment")
        venv.create(VENV_DIR, with_pip=True, clear=False)
        print(f"  Created {VENV_DIR}")
    else:
        print(f"\n  Using existing venv: {VENV_DIR}")

    # ── 3. Install Python dependencies ───────────────────────────
    step("Installing Python packages")
    run(f'"{PYTHON}" -m pip install --upgrade pip', cwd=ROOT)
    run(f'"{PIP}" install -r requirements.txt', cwd=ROOT)

    # ── 4. Install Playwright browsers ───────────────────────────
    step("Installing Playwright Chromium")
    result = run(f'"{PYTHON}" -m playwright install chromium', cwd=ROOT)
    if result.returncode != 0:
        print("  [!] Playwright install failed — retrying with explicit path...")
        run(f'"{PIP}" install --force-reinstall playwright', cwd=ROOT)
        run(f'"{PYTHON}" -m playwright install chromium', cwd=ROOT)

    # ── 5. System dependencies (Linux / WSL2 only) ───────────────
    if IS_LINUX:
        step("Installing system dependencies (Linux/WSL2)")
        result = run(f'"{PYTHON}" -m playwright install-deps chromium', cwd=ROOT)
        if result.returncode != 0:
            print("\n  [!] System deps failed. You may need to run manually:")
            print("      sudo apt-get install -y \\")
            print("        libnspr4 libnss3 libatk-bridge2.0-0 libcups2 libdrm2 \\")
            print("        libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 \\")
            print("        libgbm1 libpango-1.0-0 libcairo2 libasound2t64")
            print("      playwright install-deps chromium")

    # ── 6. Verify ────────────────────────────────────────────────
    step("Verifying installation")
    ok = True
    for mod in ["playwright", "yaml", "prompt_toolkit", "rich"]:
        r = run(f'"{PYTHON}" -c "import {mod}; print(\"{mod} OK\")"', cwd=ROOT,
                capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [!] {mod} MISSING")
            ok = False
    if ok:
        print("  All packages OK ✓")

    # ── 7. Done ──────────────────────────────────────────────────
    print(f"""
{'='*60}
  Setup complete!

  To run the REPL:
    {ACTIVATE}
    python cli.py

  Single-shot:
    {ACTIVATE}
    python agent.py "your task here"

  Providers: chatgpt (default), claude, gemini, perplexity
  Edit config.yaml to change provider or model.
  Login in the browser window on first run.
  Profile persists in ./browser_profile/
{'='*60}
""")


if __name__ == "__main__":
    main()
