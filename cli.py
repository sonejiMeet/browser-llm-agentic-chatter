"""
cli.py — Rich-powered terminal agent shell.

Drives a browser-based LLM (ChatGPT, Claude, Perplexity) as an interactive
terminal agent with markdown rendering, syntax highlighting, and tool execution.

Usage:
    python cli.py                              # interactive REPL
    python cli.py "build a Flask app"          # single-shot
    python cli.py --provider perplexity --model "GPT-4o"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import yaml
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PTKStyle
from prompt_toolkit.shortcuts import clear as clear_screen

from browser import BrowserBridge, PERPLEXITY_MODELS
from tools import ToolExecutor
from session import Session
from agent_core import AgentLoop, clean_llm_text

console = Console()

PTK_STYLE = PTKStyle.from_dict({
    "prompt": "bold #00ff87",
})

# Tool icons for visual distinction
_TOOL_ICONS = {
    "shell": "⚡",
    "file_write": "✎",
    "file_read": "📖",
}


class AgentShell:
    def __init__(self, config: dict):
        self.config = config
        self.tools = ToolExecutor(config)
        self.session = Session(config)
        self.bridge: BrowserBridge | None = None
        self.agent: AgentLoop | None = None

    async def start(self, task: str | None = None):
        console.print()
        console.print(Panel(
            f"[bold]Browser LLM Agent[/]\n"
            f"Provider: [cyan]{self.config['provider']}[/]"
            + (f"  Model: [yellow]{self.config.get('model', 'default')}[/]"
               if self.config.get("model") else "")
            + f"\nWorkspace: [dim]{Path.cwd().name}[/]"
            + f"\n/[bold cyan]help[/] for commands  /[bold red]exit[/] to quit",
            box=box.HEAVY, border_style="bright_blue",
        ))

        self.bridge = BrowserBridge(self.config)
        await self.bridge.start()
        await self.bridge.ensure_logged_in()

        model = self.config.get("model", "")
        if model:
            await self.bridge.select_model(model)

        self.agent = AgentLoop(self.bridge, self.tools, self.config)
        console.print("[dim]Priming system prompt...[/]")
        init_resp = await self.agent.prime()
        if init_resp:
            self.session.add_assistant(init_resp)

        if task:
            await self._process_message(task)
            await self.bridge.close()
            return

        history_file = Path.home() / ".browser-agent-history"
        prompt_session = PromptSession(
            history=FileHistory(str(history_file)),
            style=PTK_STYLE,
            message=[("class:prompt", "> ")],
        )

        console.print("\n[dim]Alt+Enter for newline. Type a prompt to begin.[/]\n")

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
                console.print("\n[dim]/exit[/]")
                break
            except Exception as e:
                console.print(f"\n[bold red]ERROR:[/] {e}")

        await self.bridge.close()
        console.print("[dim]Goodbye.[/]")

    async def _process_message(self, text: str):
        self.session.add_user(text)

        final_text = ""

        async for event in self.agent.run_turn(text):
            if event.kind == "status":
                console.print(f"    [dim italic]{event.text}[/]")

            elif event.kind == "response":
                self.session.add_assistant(event.text)
                clean = event.text.strip()
                if clean:
                    console.print()
                    console.print(Panel(
                        Markdown(clean),
                        border_style="green",
                        box=box.ROUNDED,
                        padding=(0, 2),
                    ))

            elif event.kind == "tool_start":
                tool = event.data.get("tool", "tool")
                icon = _TOOL_ICONS.get(tool, "▶")
                console.print(f"\n  [bold yellow]{icon}[/] [bold]{event.text}[/]")

            elif event.kind == "tool_result":
                lang = "powershell" if os.name == "nt" else "bash"
                tool = event.data.get("tool", "tool")
                if tool == "file_write":
                    lang = "diff"
                elif tool == "file_read":
                    path = event.data.get("path", "")
                    if path.endswith(".py"):
                        lang = "python"
                    elif any(path.endswith(e) for e in (".c", ".h", ".cpp", ".hpp")):
                        lang = "c"

                ok = event.data.get("ok", True)
                output = event.text.strip()
                preview = output[:1000]
                if len(output) > 1000:
                    preview += f"\n\n... [dim]({len(output)} total chars)[/]"

                style = "red" if not ok else "green"
                symbol = "✗" if not ok else "✓"
                duration = event.data.get("duration_ms")
                dur_str = f" [dim]{duration}ms[/]" if duration else ""

                console.print(Panel(
                    Syntax(preview, lang, theme="monokai", word_wrap=True),
                    title=f"[bold {style}]{symbol} {tool}[/]{dur_str}",
                    border_style=style,
                    box=box.ROUNDED,
                    padding=(0, 1),
                ))
                self.session.add_tool(f"{tool}: {output[:500]}")

            elif event.kind == "error":
                console.print(f"\n[bold red]✗ ERROR:[/] {event.text}")

            elif event.kind == "done":
                final_text = event.text
                if self.tools.change_log:
                    lines: list[str] = []
                    for e in self.tools.change_log:
                        kind = e.get("kind", "?")
                        if kind == "create":
                            lines.append(f"  [green]+[/] created  {e.get('path', '?')}")
                        elif kind == "modify":
                            lines.append(f"  [yellow]~[/] modified {e.get('path', '?')}")
                        elif kind == "shell":
                            lines.append(f"  [dim]$[/] {e.get('command', '')[:80]}")
                    console.print(Panel(
                        "\n".join(lines),
                        title="Changes",
                        border_style="blue",
                        box=box.ROUNDED,
                        padding=(0, 1),
                    ))

        # Fallback
        if final_text and not self.session.last_assistant():
            cleaned = clean_llm_text(final_text)
            if cleaned.strip():
                console.print()
                console.print(Panel(
                    Markdown(cleaned),
                    border_style="green",
                    box=box.ROUNDED,
                    padding=(0, 2),
                ))

    async def _handle_command(self, raw: str) -> bool:
        """Return True if the shell should exit."""
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd in ("/exit", "/quit"):
            return True
        if cmd == "/clear":
            clear_screen()
            self.session.clear()
            console.print("[dim]Session cleared.[/]")
        elif cmd == "/history":
            for i, m in enumerate(self.session.messages):
                tag = {
                    "user": "[cyan]You[/]",
                    "assistant": "[green]AI[/]",
                    "tool": "[yellow]🔧[/]",
                }.get(m.role, m.role)
                content = m.content[:120].replace("\n", " ")
                console.print(f"  {i:3d} {tag} {content}")
        elif cmd == "/changes":
            if not self.tools.change_log:
                console.print("[dim]No changes yet.[/]")
            else:
                for e in self.tools.change_log:
                    console.print(f"  {e}")
        elif cmd == "/help":
            console.print(Markdown("""\
**Commands:**
- `/help` — Show this help
- `/clear` — Clear session and screen
- `/history` — Show conversation history
- `/changes` — Show file/shell change log
- `/exit` — Quit the agent
- `/provider` — Show current provider and model

**Tips:**
- `Alt+Enter` for multi-line input
- Tool calls (`[[[SHELL]]]`, `[[[FILE]]]`, `[[[READ]]]`) execute automatically
- Messages are **pasted** into the browser (fast), not typed character-by-character
- Use `--provider perplexity --model "GPT-4o"` for Perplexity models
"""))
        elif cmd == "/provider":
            provider = self.config.get("provider", "chatgpt")
            model = self.config.get("model", "default")
            console.print(f"Provider: [cyan]{provider}[/]  Model: [yellow]{model}[/]")
        else:
            console.print(f"[red]Unknown: {cmd}[/]  Type /help")
        return False


def load_config(path: str | None = None) -> dict:
    search = [path, Path(__file__).parent / "config.yaml", Path("config.yaml")]
    for p in search:
        if p and Path(p).exists():
            with open(p, encoding="utf-8") as f:
                return yaml.safe_load(f)
    return {
        "provider": "chatgpt",
        "urls": {"chatgpt": "https://chatgpt.com/"},
        "user_data_dir": "./browser_profile",
        "browser": "chromium",
        "headless": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Browser LLM Agent")
    parser.add_argument("task", nargs="?", help="Single-shot task (skip REPL)")
    parser.add_argument("--provider", "-p", help="chatgpt, claude, gemini, perplexity")
    parser.add_argument("--model", "-m", help="Model name (Perplexity: GPT-4o, Sonar, ...)")
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument("--workspace", "-w", help="Working directory for tools")
    parser.add_argument("--list-models", action="store_true", help="List Perplexity models")
    args = parser.parse_args()

    if args.list_models:
        console.print("[bold]Perplexity models:[/]")
        for m in PERPLEXITY_MODELS:
            console.print(f"  • {m}")
        return

    config = load_config(args.config)
    if args.provider:
        config["provider"] = args.provider
    if args.model:
        config["model"] = args.model

    if args.workspace:
        ws = Path(args.workspace).expanduser().resolve()
        ws.mkdir(parents=True, exist_ok=True)
        os.chdir(ws)

    shell = AgentShell(config)
    asyncio.run(shell.start(task=args.task))


if __name__ == "__main__":
    main()
