"""
cli.py - Polished Rich terminal shell for the browser LLM agent.

Features:
- Interactive browser-hosted LLM agent
- Clean task lifecycle output
- Live waiting spinner during browser transactions
- Syntax-highlighted tool results
- Compact task-result and change summaries
- Runtime browser transaction debug controls

Usage:
    python cli.py
    python cli.py "build a Flask app"
    python cli.py -p deepseek -m expert
    python cli.py --debug
"""

from __future__ import annotations

import argparse
import asyncio
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import yaml
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import clear as clear_screen
from prompt_toolkit.styles import Style as PTKStyle
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from agent_core import AgentLoop
from browser import BrowserBridge, PERPLEXITY_MODELS
from session import Session
from tools import ToolExecutor


console = Console()

PTK_STYLE = PTKStyle.from_dict(
    {
        "prompt": "bold #6ee7b7",
    }
)

_TOOL_ICONS = {
    "shell": "⚡",
    "file_write": "✎",
    "file_read": "📖",
}

_STATUS_MESSAGES = {
    "Sending task...": "Sending request",
    "Sending task + workspace context...": "Sending request",
    "Sending tool results...": "Sending tool results",
    "No actions found; requesting next action...": "Waiting for next step",
    "Code has no path; requesting a destination...": "Requesting file path",
}


class AgentShell:
    """Interactive Rich UI around AgentLoop."""

    def __init__(self, config: dict):
        self.config = config
        self.tools = ToolExecutor(config)
        self.session = Session(config)
        self.bridge: BrowserBridge | None = None
        self.agent: AgentLoop | None = None

    async def start(self, task: str | None = None) -> None:
        self._show_welcome()

        self.bridge = BrowserBridge(self.config)

        with console.status(
            "[dim]Opening browser session...[/]",
            spinner="dots",
        ):
            await self.bridge.start()
            await self.bridge.ensure_logged_in()

        model = self.config.get("model", "")
        if model:
            with console.status(
                f"[dim]Selecting model: {model}...[/]",
                spinner="dots",
            ):
                await self.bridge.select_model(model)

        self.agent = AgentLoop(self.bridge, self.tools, self.config)

        with console.status(
            "[dim]Preparing agent...[/]",
            spinner="dots",
        ):
            init_response = await self.agent.prime()

        if init_response:
            self.session.add_assistant(init_response)

        if task:
            await self._process_message(task)
            await self.bridge.close()
            return

        history_file = Path.home() / ".browser-agent-history"
        prompt_session = PromptSession(
            history=FileHistory(str(history_file)),
            style=PTK_STYLE,
            message=[("class:prompt", "› ")],
        )

        console.print(
            "[dim]Enter a task, or use /help for commands. "
            "Alt+Enter inserts a new line.[/]\n"
        )

        while True:
            try:
                user_input = await prompt_session.prompt_async()
                user_input = user_input.strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    if await self._handle_command(user_input):
                        break
                    continue

                await self._process_message(user_input)

            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Exiting...[/]")
                break

            except Exception as exc:
                console.print(
                    Panel(
                        str(exc),
                        title="[bold red]Error[/]",
                        border_style="red",
                        box=box.ROUNDED,
                    )
                )

        await self.bridge.close()
        console.print("[dim]Browser closed. Goodbye.[/]")

    def _show_welcome(self) -> None:
        """Display a compact app header."""
        provider = self.config.get("provider", "chatgpt")
        model = self.config.get("model") or "default"
        workspace = Path.cwd().name
        debug_enabled = bool(
            self.config.get("debug_transactions", False)
        )

        details = Table.grid(padding=(0, 1))
        details.add_column(style="dim", no_wrap=True)
        details.add_column()

        details.add_row("Provider", f"[cyan]{provider}[/]")
        details.add_row("Model", f"[yellow]{model}[/]")
        details.add_row("Workspace", f"[white]{workspace}[/]")
        details.add_row(
            "Diagnostics",
            "[green]On[/]" if debug_enabled else "[dim]Off[/]",
        )

        console.print()
        console.print(
            Panel(
                details,
                title="[bold bright_blue]Browser LLM Agent[/]",
                subtitle="[dim]/help for commands · /exit to quit[/]",
                border_style="bright_blue",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
        console.print()

    async def _process_message(self, text: str) -> None:
        """Run one agent task and render the resulting event stream."""
        if self.agent is None:
            console.print("[bold red]Agent is not initialized.[/]")
            return

        self.session.add_user(text)
        context = self.agent.gather_context()
        self._show_task_context(context)

        displayed_response = False
        did_work = False
        final_event: Any | None = None
        spinner_message = "Thinking"
        live: Live | None = None

        try:
            async for event in self.agent.run_turn(
                text,
                workspace_context=context,
            ):
                if event.kind == "status":
                    spinner_message = _STATUS_MESSAGES.get(
                        event.text,
                        event.text.rstrip("."),
                    )

                    if live is None:
                        live = Live(
                            self._spinner_renderable(spinner_message),
                            console=console,
                            refresh_per_second=12,
                            transient=True,
                        )
                        live.start()
                    else:
                        live.update(
                            self._spinner_renderable(spinner_message)
                        )

                elif event.kind == "response":
                    if live is not None:
                        live.stop()
                        live = None

                    clean = event.text.strip()

                    if clean:
                        displayed_response = True
                        self.session.add_assistant(clean)
                        self._show_response(clean)

                elif event.kind == "tool_start":
                    if live is not None:
                        live.stop()
                        live = None

                    did_work = True
                    self._show_tool_start(event)

                elif event.kind == "tool_result":
                    if live is not None:
                        live.stop()
                        live = None

                    did_work = True
                    self._show_tool_result(event)

                elif event.kind == "error":
                    if live is not None:
                        live.stop()
                        live = None

                    self._show_error(event.text)

                elif event.kind == "done":
                    if live is not None:
                        live.stop()
                        live = None

                    final_event = event

        finally:
            if live is not None:
                live.stop()

        if final_event is not None:
            self._show_completion(
                final_event,
                displayed_response=displayed_response,
                did_work=did_work,
            )

    def _spinner_renderable(self, message: str) -> Group:
        """Create the transient waiting indicator."""
        return Group(
            Spinner(
                "dots",
                text=Text(f" {message}...", style="dim cyan"),
            )
        )

    def _show_task_context(self, context: Any) -> None:
        """Display workspace context in one unobtrusive line."""
        parts = [
            f"[bold]{context.root_name}[/]",
            f"[dim]{context.project_type}[/]",
            f"[dim]~{context.total_files} files[/]",
        ]

        if context.git_branch:
            parts.append(f"[dim]git:{context.git_branch}[/]")

        console.print(
            Panel(
                "  [dim]·[/]  ".join(parts),
                border_style="grey37",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    def _show_response(self, text: str) -> None:
        """
        Display the LLM response exactly once.

        AgentLoop also includes this content in the final done event. This is
        the only normal response display path, preventing duplicate Agent text.
        """
        console.print()
        console.print(
            Panel(
                Markdown(text),
                title="[bold green]Assistant[/]",
                border_style="green",
                box=box.ROUNDED,
                padding=(0, 2),
            )
        )

    def _show_tool_start(self, event: Any) -> None:
        """Display an action heading before its result panel."""
        tool = event.data.get("tool", "tool")
        icon = _TOOL_ICONS.get(tool, "▶")

        label = event.text.strip()

        console.print(
            f"\n[bold yellow]{icon}[/] [bold]{label}[/]"
        )

    def _show_tool_result(self, event: Any) -> None:
        """Display a compact syntax-highlighted tool result."""
        tool = event.data.get("tool", "tool")
        ok = event.data.get("ok", True)
        output = event.text.strip() or "(no output)"

        language = self._tool_language(tool, event.data.get("path", ""))
        preview_limit = 1400
        preview = output[:preview_limit]

        if len(output) > preview_limit:
            preview += (
                f"\n\n... ({len(output) - preview_limit} more characters)"
            )

        color = "green" if ok else "red"
        symbol = "✓" if ok else "✗"
        duration = event.data.get("duration_ms")
        title = f"[bold {color}]{symbol} {tool}[/]"

        if duration is not None:
            title += f" [dim]{duration} ms[/]"

        console.print(
            Panel(
                Syntax(
                    preview,
                    language,
                    theme="monokai",
                    word_wrap=True,
                ),
                title=title,
                border_style=color,
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

        self.session.add_tool(f"{tool}: {output[:500]}")

    @staticmethod
    def _tool_language(tool: str, path: str) -> str:
        """Choose a Rich lexer appropriate to a tool result."""
        if tool == "file_write":
            return "diff"

        if tool != "file_read":
            return "powershell" if os.name == "nt" else "bash"

        suffix = Path(path).suffix.lower()

        languages = {
            ".py": "python",
            ".c": "c",
            ".h": "c",
            ".cc": "cpp",
            ".cpp": "cpp",
            ".cxx": "cpp",
            ".hpp": "cpp",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".md": "markdown",
            ".html": "html",
            ".css": "css",
            ".sh": "bash",
            ".ps1": "powershell",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
        }

        return languages.get(suffix, "text")

    def _show_error(self, message: str) -> None:
        """Render an agent failure clearly without exposing a traceback."""
        console.print(
            Panel(
                message,
                title="[bold red]Agent error[/]",
                border_style="red",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    def _show_completion(
        self,
        event: Any,
        displayed_response: bool,
        did_work: bool,
    ) -> None:
        """
        Render completion only when it adds useful information.

        The final cleaned response is not printed if a response event already
        displayed it. A simple conversational turn therefore ends cleanly after
        the Assistant panel, without redundant Agent or Summary panels.
        """
        data = event.data
        final_message = str(data.get("cleaned", "")).strip()
        summary = str(data.get("summary", "")).strip()
        complete = bool(data.get("complete", False))
        results = data.get("results") or []

        if final_message and not displayed_response:
            self.session.add_assistant(final_message)
            self._show_response(final_message)

        change_lines = self._build_change_lines()
        has_file_changes = any(
            result.get("type") == "file_write"
            and not result.get("error")
            for result in results
        )
        has_shell_actions = any(
            result.get("type") == "shell"
            and not result.get("error")
            for result in results
        )

        # Do not show "Task completed." for greetings or ordinary answers.
        if not did_work and not change_lines:
            return

        result_lines: list[str] = []

        if complete:
            result_lines.append("[green]✓ Completed[/]")
        else:
            result_lines.append("[yellow]• Stopped before completion[/]")

        if has_file_changes or has_shell_actions:
            concise_summary = self._useful_summary(summary)
            if concise_summary:
                result_lines.append(concise_summary)

        if result_lines:
            console.print(
                Panel(
                    "\n".join(result_lines),
                    title="[bold]Task result[/]",
                    border_style="green" if complete else "yellow",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )

        if change_lines:
            console.print(
                Panel(
                    "\n".join(change_lines),
                    title="[bold]Changes[/]",
                    border_style="blue",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )

    @staticmethod
    def _useful_summary(summary: str) -> str:
        """
        Remove generic completion text while retaining file/run instructions.
        """
        useful_lines = [
            line
            for line in summary.splitlines()
            if line.strip()
            and line.strip()
            not in {
                "Task completed.",
                "Task stopped.",
            }
        ]

        return "\n".join(useful_lines)

    def _build_change_lines(self) -> list[str]:
        """Turn ToolExecutor's change log into a concise user-facing list."""
        lines: list[str] = []

        for entry in self.tools.change_log:
            kind = entry.get("kind", "?")
            path = entry.get("path", "?")

            if kind == "create":
                byte_count = entry.get("bytes")
                size = f" [dim]({byte_count} bytes)[/]" if byte_count else ""
                lines.append(f"[green]+[/] Created [bold]{path}[/]{size}")

            elif kind == "modify":
                added = entry.get("added", 0)
                removed = entry.get("removed", 0)
                delta = ""

                if added or removed:
                    delta = (
                        f" [dim](+{added} / -{removed} lines)[/]"
                    )

                lines.append(
                    f"[yellow]~[/] Updated [bold]{path}[/]{delta}"
                )

            elif kind == "shell":
                command = str(entry.get("command", "")).strip()

                if command:
                    exit_code = entry.get("exit_code")
                    status = (
                        "[green]✓[/]"
                        if exit_code in (None, 0)
                        else "[red]✗[/]"
                    )
                    lines.append(
                        f"{status} [dim]Ran:[/] {command[:100]}"
                    )

        return lines

    async def _handle_command(self, raw: str) -> bool:
        """Process an interactive slash command."""
        parts = raw.split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1].strip().lower() if len(parts) > 1 else ""

        if command in ("/exit", "/quit"):
            return True

        if command == "/clear":
            clear_screen()
            self.session.clear()
            console.print("[dim]Session cleared.[/]")
            return False

        if command == "/history":
            self._show_history()
            return False

        if command == "/changes":
            lines = self._build_change_lines()

            if lines:
                console.print(
                    Panel(
                        "\n".join(lines),
                        title="[bold]Changes[/]",
                        border_style="blue",
                        box=box.ROUNDED,
                        padding=(0, 1),
                    )
                )
            else:
                console.print("[dim]No workspace changes in this session.[/]")

            return False

        if command == "/provider":
            provider = self.config.get("provider", "chatgpt")
            model = self.config.get("model") or "default"

            console.print(
                f"Provider [cyan]{provider}[/]  ·  "
                f"Model [yellow]{model}[/]"
            )
            return False

        if command == "/debug":
            self._handle_debug_command(argument)
            return False

        if command == "/help":
            self._show_help()
            return False

        console.print(
            f"[red]Unknown command:[/] {command}  "
            "[dim]Use /help to see available commands.[/]"
        )
        return False

    def _show_history(self) -> None:
        """Display short conversation-history previews."""
        if not self.session.messages:
            console.print("[dim]No conversation history yet.[/]")
            return

        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold bright_blue",
            pad_edge=False,
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("Role", width=10)
        table.add_column("Preview")

        for index, message in enumerate(self.session.messages):
            role = message.role.capitalize()
            content = message.content.replace("\n", " ").strip()

            if len(content) > 110:
                content = content[:107] + "..."

            table.add_row(str(index), role, content)

        console.print(table)

    def _show_help(self) -> None:
        """Display concise interactive help."""
        console.print(
            Panel(
                Markdown(
                    """\
### Commands

- `/help` — Show this help
- `/clear` — Clear the terminal and session history
- `/history` — Show recent conversation entries
- `/changes` — Show workspace changes from the current task
- `/provider` — Show active provider and model
- `/debug on` — Enable transaction diagnostics
- `/debug off` — Disable transaction diagnostics
- `/debug status` — Show diagnostic state
- `/debug toggle` — Toggle transaction diagnostics
- `/exit` — Close the browser and exit

### Tips

- `Alt+Enter` inserts a newline.
- Tool calls execute automatically after the model completes a response.
- Use `/debug on` when diagnosing browser response detection.
"""
                ),
                title="[bold]Help[/]",
                border_style="bright_blue",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    def _handle_debug_command(self, argument: str) -> None:
        """Toggle BrowserBridge transaction diagnostics at runtime."""
        if self.bridge is None:
            console.print(
                "[bold red]Browser bridge is not initialized.[/]"
            )
            return

        if argument in ("", "status"):
            state = "[green]ON[/]" if self.bridge.debug else "[dim]OFF[/]"
            console.print(
                f"Transaction diagnostics: {state}\n"
                "[dim]/debug on | off | toggle | status[/]"
            )
            return

        if argument in ("on", "true", "1", "yes"):
            self.bridge.debug = True
            self.config["debug_transactions"] = True
            console.print("[green]Transaction diagnostics enabled.[/]")
            return

        if argument in ("off", "false", "0", "no"):
            self.bridge.debug = False
            self.config["debug_transactions"] = False
            console.print("[dim]Transaction diagnostics disabled.[/]")
            return

        if argument == "toggle":
            self.bridge.debug = not self.bridge.debug
            self.config["debug_transactions"] = self.bridge.debug

            state = (
                "[green]enabled[/]"
                if self.bridge.debug
                else "[dim]disabled[/]"
            )

            console.print(f"Transaction diagnostics {state}.")
            return

        console.print(
            "[yellow]Usage: /debug on | off | toggle | status[/]"
        )


def load_config(path: str | None = None) -> dict:
    """Load config.yaml or return minimal defaults."""
    search_paths = [
        path,
        Path(__file__).parent / "config.yaml",
        Path("config.yaml"),
    ]

    for candidate in search_paths:
        if candidate and Path(candidate).exists():
            with open(candidate, encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}

    return {
        "provider": "chatgpt",
        "urls": {
            "chatgpt": "https://chatgpt.com/",
        },
        "user_data_dir": "./browser_profile",
        "browser": "chromium",
        "headless": False,
        "debug_transactions": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="orbit",
        description="Orbit. A Browser LLM Agent",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="Single-shot task; omit for interactive mode",
    )
    parser.add_argument(
        "--provider",
        "-p",
        metavar="P",
        help="chatgpt, claude, gemini, perplexity, deepseek",
    )
    parser.add_argument(
        "--model",
        "-m",
        metavar="M",
        help="Model name, for example expert, GPT-4o, or Sonar",
    )
    parser.add_argument(
        "--workspace",
        "-w",
        metavar="W",
        help="Working directory for local tools",
    )
    parser.add_argument(
        "--config",
        metavar="C",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List supported Perplexity models",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable browser transaction diagnostics at startup",
    )

    args = parser.parse_args()

    if args.list_models:
        console.print("[bold]Perplexity models:[/]")

        for model in PERPLEXITY_MODELS:
            console.print(f"  • {model}")

        return

    config = load_config(args.config)

    if args.provider:
        config["provider"] = args.provider

    if args.model:
        config["model"] = args.model

    if args.debug:
        config["debug_transactions"] = True

    if args.workspace:
        workspace = Path(args.workspace).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        os.chdir(workspace)

    shell = AgentShell(config)
    asyncio.run(shell.start(task=args.task))


if __name__ == "__main__":
    main()