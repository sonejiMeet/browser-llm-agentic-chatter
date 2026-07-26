#!/usr/bin/env python3
"""
Browser LLM Agent — one-command setup for Linux, WSL2, and Windows.

Usage:
    python setup.py

After setup, open a NEW terminal in any project folder and run:

    orbit

The orbit command uses this installation's virtual environment, but keeps
the terminal's current directory as the agent workspace.
"""

from __future__ import annotations

import os
import platform
import shlex
import subprocess
import sys
import venv
from pathlib import Path


# ── Color codes ──────────────────────────────────────────────────

GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
NC = "\033[0m"


# ── Installation paths ───────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"

IS_WIN = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

PYTHON = sys.executable

if IS_WIN:
    VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
    PIP = VENV_DIR / "Scripts" / "pip.exe"
    ACTIVATE = VENV_DIR / "Scripts" / "activate"

    USER_LOCAL_BIN = (
        Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        / "BrowserLLMAgent"
        / "bin"
    )
    ORBIT_LAUNCHER = USER_LOCAL_BIN / "orbit.cmd"
else:
    VENV_PYTHON = VENV_DIR / "bin" / "python"
    PIP = VENV_DIR / "bin" / "pip"
    ACTIVATE = f"source {VENV_DIR / 'bin' / 'activate'}"

    USER_LOCAL_BIN = Path.home() / ".local" / "bin"
    ORBIT_LAUNCHER = USER_LOCAL_BIN / "orbit"


def run(cmd: str, **kwargs) -> subprocess.CompletedProcess:
    """Run a shell command and return its result."""
    print(
        f"  {CYAN}$ {cmd[:100]}"
        f"{'...' if len(cmd) > 100 else ''}{NC}"
    )
    return subprocess.run(
        cmd,
        shell=True,
        check=False,
        **kwargs,
    )


def step(message: str) -> None:
    """Print a step header."""
    print(f"\n{BOLD}{CYAN}{'═' * 60}{NC}")
    print(f"  {BOLD}{message}{NC}")
    print(f"{BOLD}{CYAN}{'═' * 60}{NC}")


def success(message: str) -> None:
    """Print a success message."""
    print(f"  {GREEN}✓ {message}{NC}")


def error(message: str) -> None:
    """Print an error message."""
    print(f"  {RED}✗ {message}{NC}")


def warning(message: str) -> None:
    """Print a warning message."""
    print(f"  {YELLOW}⚠ {message}{NC}")


def info(message: str) -> None:
    """Print an informational message."""
    print(f"  {CYAN}ℹ {message}{NC}")


def path_contains(path_value: str, directory: Path) -> bool:
    """Return whether PATH already contains directory."""
    wanted = os.path.normcase(
        os.path.normpath(str(directory))
    )

    for entry in path_value.split(os.pathsep):
        if not entry:
            continue

        current = os.path.normcase(
            os.path.normpath(os.path.expandvars(entry))
        )

        if current == wanted:
            return True

    return False


def add_windows_user_path(directory: Path) -> bool:
    """
    Add directory to the current user's persistent Windows PATH.

    This does not affect the terminal currently running setup.py. Open a new
    PowerShell, Command Prompt, or Windows Terminal tab after setup.
    """
    try:
        import winreg

        environment_key = r"Environment"

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            environment_key,
            0,
            winreg.KEY_READ | winreg.KEY_WRITE,
        ) as key:
            try:
                old_path, path_type = winreg.QueryValueEx(
                    key,
                    "Path",
                )
            except FileNotFoundError:
                old_path = ""
                path_type = winreg.REG_EXPAND_SZ

            if path_contains(old_path, directory):
                success("orbit command directory is already on PATH")
                return True

            new_path = (
                f"{old_path}{os.pathsep}"
                f"{directory}"
                if old_path
                else str(directory)
            )

            winreg.SetValueEx(
                key,
                "Path",
                0,
                path_type,
                new_path,
            )

        try:
            import ctypes

            hwnd_broadcast = 0xFFFF
            wm_settingchange = 0x001A
            smto_abortifhung = 0x0002

            result = ctypes.c_ulong()
            ctypes.windll.user32.SendMessageTimeoutW(
                hwnd_broadcast,
                wm_settingchange,
                0,
                "Environment",
                smto_abortifhung,
                3000,
                ctypes.byref(result),
            )
        except Exception:
            pass

        success("Added orbit command directory to your user PATH")
        return True

    except Exception as exc:
        warning(
            "Could not update the Windows user PATH automatically: "
            f"{exc}"
        )
        return False


def add_unix_user_path(directory: Path) -> bool:
    """
    Add ~/.local/bin to ~/.profile when it is not already present.

    A new terminal or a new login session is required before the shell sees
    this PATH change.
    """
    profile = Path.home() / ".profile"
    marker_start = "# Browser LLM Agent orbit command"
    marker_end = "# End Browser LLM Agent orbit command"

    try:
        if profile.exists():
            profile_text = profile.read_text(encoding="utf-8")
        else:
            profile_text = ""

        export_line = (
            f'export PATH="{directory}:$PATH"'
        )

        if (
            str(directory) in profile_text
            or marker_start in profile_text
        ):
            success("orbit command directory is already configured")
            return True

        block = (
            f"\n{marker_start}\n"
            f"{export_line}\n"
            f"{marker_end}\n"
        )

        profile.write_text(
            profile_text.rstrip() + block,
            encoding="utf-8",
        )

        success(f"Added {directory} to ~/.profile")
        return True

    except Exception as exc:
        warning(
            "Could not update ~/.profile automatically: "
            f"{exc}"
        )
        return False


def install_orbit_command() -> bool:
    """
    Create the global orbit launcher.

    The launcher deliberately does not change directory. Therefore:

        cd C:\\some\\project
        orbit

    runs cli.py with C:\\some\\project as its current workspace.
    """
    step("Installing orbit command")

    try:
        USER_LOCAL_BIN.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        error(
            f"Could not create command directory "
            f"{USER_LOCAL_BIN}: {exc}"
        )
        return False

    cli_file = ROOT / "cli.py"

    if not cli_file.exists():
        error(f"Cannot create orbit command: missing {cli_file}")
        return False

    if not VENV_PYTHON.exists():
        error(
            "Cannot create orbit command: "
            f"missing virtual-environment Python at {VENV_PYTHON}"
        )
        return False

    try:
        if IS_WIN:
            launcher_text = (
                "@echo off\r\n"
                "setlocal\r\n"
                f'"{VENV_PYTHON}" "{cli_file}" %*\r\n'
                "exit /b %ERRORLEVEL%\r\n"
            )

            ORBIT_LAUNCHER.write_text(
                launcher_text,
                encoding="utf-8",
                newline="\r\n",
            )

        else:
            python_path = shlex.quote(str(VENV_PYTHON))
            cli_path = shlex.quote(str(cli_file))

            launcher_text = (
                "#!/usr/bin/env sh\n"
                f"exec {python_path} {cli_path} \"$@\"\n"
            )

            ORBIT_LAUNCHER.write_text(
                launcher_text,
                encoding="utf-8",
            )

            ORBIT_LAUNCHER.chmod(
                ORBIT_LAUNCHER.stat().st_mode | 0o111
            )

        success(f"Created command launcher: {ORBIT_LAUNCHER}")

    except Exception as exc:
        error(f"Failed to create orbit command: {exc}")
        return False

    if IS_WIN:
        path_ok = add_windows_user_path(USER_LOCAL_BIN)
    else:
        path_ok = add_unix_user_path(USER_LOCAL_BIN)

    if not path_ok:
        warning(
            "The launcher exists, but PATH was not updated automatically."
        )

        if IS_WIN:
            print(
                f"  Add this directory to your user PATH manually:\n"
                f"  {YELLOW}{USER_LOCAL_BIN}{NC}"
            )
        else:
            print(
                f"  Add this to your shell profile manually:\n"
                f'  {YELLOW}export PATH="{USER_LOCAL_BIN}:$PATH"{NC}'
            )

    return True


def verify_imports() -> bool:
    """Verify required modules in the project virtual environment."""
    step("Verifying installation")

    ok = True

    for module in [
        "playwright",
        "yaml",
        "prompt_toolkit",
        "rich",
    ]:
        result = run(
            (
                f'"{VENV_PYTHON}" -c '
                f'"import {module}; print(\'{module} OK\')"'
            ),
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            error(f"{module} MISSING")
            ok = False
        else:
            success(f"{module} verified")

    return ok


def main() -> None:
    os.chdir(ROOT)

    # ── 1. Python version ────────────────────────────────────────

    step("Checking Python version")

    version = sys.version_info

    if version < (3, 11):
        error(
            "Python 3.11+ required, found "
            f"{version.major}.{version.minor}"
        )
        print(
            f"  {RED}Please upgrade Python to 3.11 or newer.{NC}"
        )
        sys.exit(1)

    success(
        f"Python {version.major}.{version.minor}.{version.micro}"
    )

    # ── 2. Virtual environment ───────────────────────────────────

    if not VENV_DIR.exists():
        step("Creating virtual environment")

        try:
            venv.create(
                VENV_DIR,
                with_pip=True,
                clear=False,
            )
            success(f"Created {VENV_DIR}")

        except Exception as exc:
            error(f"Failed to create virtual environment: {exc}")
            sys.exit(1)

    else:
        info(f"Using existing venv: {VENV_DIR}")

    # ── 3. Python dependencies ───────────────────────────────────

    step("Installing Python packages")

    result = run(
        f'"{VENV_PYTHON}" -m pip install --upgrade pip',
        cwd=ROOT,
    )

    if result.returncode == 0:
        success("pip upgraded")
    else:
        error("Failed to upgrade pip")

    requirements_file = ROOT / "requirements.txt"

    if not requirements_file.exists():
        error(f"Missing requirements file: {requirements_file}")
        sys.exit(1)

    result = run(
        f'"{PIP}" install -r requirements.txt',
        cwd=ROOT,
    )

    if result.returncode == 0:
        success("Python packages installed")
    else:
        error("Failed to install Python packages")
        print(
            f"  {RED}Check requirements.txt and try again.{NC}"
        )
        sys.exit(1)

    # ── 4. Playwright ────────────────────────────────────────────

    step("Installing Playwright Chromium")

    result = run(
        f'"{VENV_PYTHON}" -m playwright install chromium',
        cwd=ROOT,
    )

    if result.returncode == 0:
        success("Playwright Chromium installed")
    else:
        error(
            "Playwright install failed — retrying with "
            "a clean Playwright install..."
        )

        run(
            f'"{PIP}" install --force-reinstall playwright',
            cwd=ROOT,
        )

        result = run(
            f'"{VENV_PYTHON}" -m playwright install chromium',
            cwd=ROOT,
        )

        if result.returncode == 0:
            success("Playwright Chromium installed after retry")
        else:
            error("Playwright Chromium installation failed")
            sys.exit(1)

    # ── 5. Linux / WSL system dependencies ───────────────────────

    if IS_LINUX:
        step("Installing system dependencies for Linux / WSL2")

        result = run(
            f'"{VENV_PYTHON}" -m playwright install-deps chromium',
            cwd=ROOT,
        )

        if result.returncode == 0:
            success("System dependencies installed")
        else:
            warning(
                "Playwright system dependency installation failed."
            )
            print(
                f"  {YELLOW}Try manually:{NC}\n"
                f"  {YELLOW}sudo {VENV_PYTHON} -m "
                f"playwright install-deps chromium{NC}"
            )

    # ── 6. Verify Python modules ─────────────────────────────────

    if not verify_imports():
        error("Some required packages are missing")
        sys.exit(1)

    success("All Python packages verified")

    # ── 7. Global orbit command ──────────────────────────────────

    if not install_orbit_command():
        error(
            "Setup completed, but the orbit command could not be installed."
        )
        sys.exit(1)

    # ── 8. Done ──────────────────────────────────────────────────

    print(
        f"""
{BOLD}{GREEN}{'═' * 60}{NC}
  {BOLD}{GREEN}✅ Setup complete!{NC}

  Close this terminal and open a NEW terminal.

  Then, from any folder where you want the agent to work:

    {CYAN}cd path\\to\\your\\project{NC}
    {CYAN}orbit{NC}

  You can also pass normal CLI arguments:

    {CYAN}orbit "Build a Flask TODO app"{NC}
    {CYAN}orbit --provider deepseek "Fix the build error"{NC}
    {CYAN}orbit --provider perplexity --model Sonar "Review this project"{NC}

  The orbit command automatically uses:
    {YELLOW}{VENV_PYTHON}{NC}

  It does NOT change your current terminal folder, so the folder
  where you run orbit remains the agent workspace.

  Providers: chatgpt (default), claude, gemini, perplexity, deepseek
  Edit config.yaml to change provider or model.
  Log in in the browser window on first run.
  Browser profile persists in ./browser_profile/
{BOLD}{GREEN}{'═' * 60}{NC}
"""
    )


if __name__ == "__main__":
    main()