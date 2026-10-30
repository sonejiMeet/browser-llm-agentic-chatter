"""Rich Orbit terminal UI over the strict, approval-gated agent runtime."""
import argparse
import asyncio
import json
import time
from pathlib import Path
import yaml
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from agent_core import AgentLoop
from reliable_browser import BrowserBridge
from tools import ToolExecutor

console = Console()

def load_config(path=None):
    source = Path(path) if path else Path(__file__).with_name('config.yaml')
    return yaml.safe_load(source.read_text(encoding='utf-8')) or {}

class AgentShell:
    def __init__(self, config):
        self.config = config
        self.tools = ToolExecutor(config)
        self.bridge = BrowserBridge(config)
        self.prompt = PromptSession(history=FileHistory(str(self.tools.state_dir / 'prompt-history')))
        self.agent = AgentLoop(self.bridge, self.tools, config, confirm=self.confirm)
        self.status = None
        self.approving = False
        self.label = 'Thinking'

    def panel(self, text, title, color='blue'):
        self.stop_status()
        console.print(Panel(Text(str(text)), title=title, border_style=color))

    def stop_status(self):
        if self.status is not None:
            self.status.stop()

    def welcome(self):
        table = Table.grid(padding=(0, 2))
        for key, value in [('Provider', self.config.get('provider', 'chatgpt')), ('Model', self.config.get('model') or 'Browser default'), ('Workspace', str(self.tools.workspace)), ('Approval', 'On' if self.config.get('confirm_changes', True) else 'Off'), ('Diagnostics', 'On' if self.bridge.debug else 'Off')]:
            table.add_row(Text(key, style='dim'), Text(value))
        console.print(Panel(table, title='Browser LLM Agent', subtitle='/help for commands · /exit to quit', border_style='bright_blue'))

    async def confirm(self, plan):
        self.approving = True
        self.stop_status()
        try:
            for entry in plan:
                console.print(Panel(Syntax(json.dumps(entry, ensure_ascii=False, indent=2), 'json', word_wrap=True), title='Review exact action', border_style='yellow'))
                if entry.get('diff'):
                    console.print(Panel(Syntax(entry['diff'], 'diff', word_wrap=True), title='Proposed changes', border_style='yellow'))
            answer = await self.prompt.prompt_async('Apply this action? [y/N] ')
            return answer.strip().lower() == 'y'
        finally:
            self.approving = False
            if self.status is not None:
                self.status.start()

    async def process(self, text):
        self.label = 'Preparing workspace context'
        self.status = console.status(self.label, spinner='dots')
        started = time.monotonic()
        async def heartbeat():
            while True:
                phase = getattr(self.bridge, 'phase', '')
                label = phase if phase not in {'', 'Ready', 'Response received'} else self.label
                self.status.update(Text(f'{label} · {int(time.monotonic()-started)}s', style='dim cyan'))
                await asyncio.sleep(.1)
        monitor = asyncio.create_task(heartbeat())
        self.status.start()
        try:
            context = await asyncio.to_thread(self.agent.gather_context)
            async for event in self.agent.run_turn(text, context):
                if event.kind == 'status':
                    self.label = event.text
                elif event.kind == 'tool_start':
                    self.panel(event.text, 'Action', 'yellow')
                    self.label = 'Executing ' + event.data.get('tool', 'action')
                    self.status.start()
                elif event.kind == 'tool_output':
                    self.stop_status()
                    console.print(event.text, end='', markup=False, highlight=False)
                    self.status.start()
                elif event.kind == 'tool_result':
                    try:
                        result = json.loads(event.text)
                    except ValueError:
                        result = {'output': event.text}
                    body = result.get('diff') or result.get('output') or result.get('content') or json.dumps(result, ensure_ascii=False, indent=2)
                    self.panel(body, event.data.get('tool', 'Tool result'), 'green' if event.data.get('ok', True) else 'red')
                    self.label = 'Sending tool results'
                    self.status.start()
                elif event.kind == 'response':
                    self.stop_status()
                    console.print(Panel(Markdown(event.text), title='Assistant', border_style='green'))
                elif event.kind == 'error':
                    self.panel(event.text, 'Agent error', 'red')
                elif event.kind == 'done':
                    self.stop_status()
                    if self.tools.change_log:
                        lines = [f"{item.get('kind')}: {item.get('path') or item.get('command', '')}" for item in self.tools.change_log]
                        self.panel('\n'.join(lines), 'Changes')
                    if not event.data.get('complete'):
                        self.panel('Run stopped: ' + event.data.get('run_id', '?'), 'Task status', 'yellow')
        finally:
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass
            self.stop_status()
            self.status = None

    async def command(self, text):
        parts = text.split(maxsplit=1)
        name, value = parts[0].lower(), parts[1].lower() if len(parts) > 1 else 'status'
        if name in {'/exit', '/quit'}:
            return True
        if name == '/help':
            self.panel('/help · /clear · /history · /changes · /provider · /new\n/debug on|off|toggle|status\n/confirm on|off|toggle|status\n/exit', 'Commands')
        elif name == '/clear':
            console.clear()
        elif name == '/new':
            await self.bridge.new_chat()
        elif name == '/provider':
            self.panel(self.config.get('provider', 'chatgpt') + ' / ' + (self.config.get('model') or 'Browser default'), 'Provider')
        elif name in {'/debug', '/confirm'}:
            old = self.bridge.debug if name == '/debug' else self.config.get('confirm_changes', True)
            if value not in {'on', 'off', 'toggle', 'status'}:
                self.panel('Use on, off, toggle, or status', 'Command', 'yellow')
                return False
            enabled = not old if value == 'toggle' else value == 'on' if value in {'on', 'off'} else old
            if name == '/debug':
                self.bridge.debug = enabled
            else:
                self.config['confirm_changes'] = enabled
            self.panel(('On' if enabled else 'Off') + (' — actions auto-approved' if name == '/confirm' and not enabled else ''), name[1:].capitalize())
        elif name == '/history':
            rows = self.agent.journal.db.execute('SELECT id,task,state FROM runs ORDER BY rowid DESC LIMIT 20').fetchall()
            self.panel('\n'.join(f'{r[0][:8]} · {r[2]} · {r[1]}' for r in rows) or 'No history', 'History')
        elif name == '/changes':
            self.panel(json.dumps(self.tools.change_log, ensure_ascii=False, indent=2), 'Changes')
        else:
            self.panel('Unknown command. Use /help.', 'Command', 'yellow')
        return False

    async def start(self, task=None):
        self.welcome()
        try:
            with console.status('Opening browser and waiting for login…', spinner='dots'):
                await self.bridge.start()
                await self.bridge.ensure_logged_in()
                if self.config.get('model') and not await self.bridge.select_model(self.config['model']):
                    raise RuntimeError('Could not verify requested model selection')
            while True:
                text = task if task is not None else (await self.prompt.prompt_async('› ')).strip()
                if not text:
                    if task is not None:
                        return
                    continue
                try:
                    if text.startswith('/') and task is None:
                        if await self.command(text):
                            return
                    else:
                        await self.process(text)
                except Exception as exc:
                    self.panel(exc, 'Error', 'red')
                if task is not None:
                    return
        finally:
            self.agent.close()
            await self.bridge.close()

async def run(config, task=None):
    await AgentShell(config).start(task)

def main():
    parser = argparse.ArgumentParser(prog='orbit', description='Browser LLM Agent')
    parser.add_argument('task', nargs='?')
    parser.add_argument('-p', '--provider')
    parser.add_argument('-m', '--model')
    parser.add_argument('-w', '--workspace')
    parser.add_argument('--config')
    parser.add_argument('-y', '--yes', action='store_true', help='Auto-approve actions and disclosure')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resume', help='Resume run ID with the exact original task')
    parser.add_argument('--stdin', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ('provider', 'model', 'workspace'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.debug:
        config['debug_transactions'] = True
    if args.yes:
        config['confirm_changes'] = False
    if args.stdin:
        import sys
        args.task = sys.stdin.read().strip()
        if not args.task:
            parser.error('No task on stdin')
    if args.resume:
        if not args.task:
            parser.error('--resume requires the original task')
        config['resume_run'] = args.resume
    try:
        asyncio.run(run(config, args.task))
    except (KeyboardInterrupt, EOFError):
        console.print('Cancelled. Inspect any uncertain browser submission before continuing.')

if __name__ == '__main__':
    main()
