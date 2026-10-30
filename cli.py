"""Orbit: browser-only coding agent with strict tools and explicit approval."""
import argparse
import asyncio
import json
from pathlib import Path
import yaml
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.syntax import Syntax
from agent_core import AgentLoop
from reliable_browser import BrowserBridge
from tools import ToolExecutor

console = Console()

def load_config(path=None):
    source = Path(path) if path else Path(__file__).with_name('config.yaml')
    return yaml.safe_load(source.read_text(encoding='utf-8')) or {}

async def run(config, task=None):
    tools = ToolExecutor(config)
    bridge = BrowserBridge(config)
    prompt = PromptSession()
    async def confirm(plan):
        for entry in plan:
            console.print(Syntax(json.dumps(entry, ensure_ascii=False, indent=2), 'json', word_wrap=True))
            if 'diff' in entry:
                console.print(Syntax(entry['diff'], 'diff', word_wrap=True))
        answer = await prompt.prompt_async('Approve exactly these actions? [y/N] ')
        return answer.strip().lower() == 'y'
    agent = AgentLoop(bridge, tools, config, confirm=confirm)
    try:
        await bridge.start()
        await bridge.ensure_logged_in()
        if config.get('model') and not await bridge.select_model(config['model']):
            raise RuntimeError('Could not verify requested model selection')
        while True:
            text = task if task is not None else (await prompt.prompt_async('orbit> ')).strip()
            if not text:
                continue
            if text in {'/exit', '/quit'}:
                break
            if text in {'/history', '/clear'}:
                if text == '/clear':
                    console.clear()
                else:
                    console.print_json(data=agent.journal.db.execute('SELECT id,task,state FROM runs ORDER BY rowid DESC LIMIT 20').fetchall())
                continue
            if text == '/help':
                console.print('/exit | /new | /history | /clear | /changes | /debug on | /debug off | /confirm on | /confirm off')
                continue
            if text == '/new':
                await bridge.new_chat()
                continue
            if text == '/changes':
                console.print_json(data=tools.change_log)
                continue
            if text in {'/debug on', '/debug off'}:
                bridge.debug = text.endswith('on')
                continue
            if text in {'/confirm on', '/confirm off'}:
                config['confirm_changes'] = text.endswith('on')
                console.print('Approval enabled' if config['confirm_changes'] else 'Warning: automatic approval enabled')
                continue
            with patch_stdout():
                async for event in agent.run_turn(text, agent.gather_context()):
                    if event.kind == 'tool_output':
                        console.print(event.text, end='', markup=False, highlight=False)
                    elif event.kind != 'done':
                        console.print(f'[{event.kind}] {event.text}', markup=False, highlight=False)
                    elif not event.data.get('complete'):
                        console.print('Run stopped: ' + event.data['run_id'], markup=False)
            if task is not None:
                break
    finally:
        agent.close()
        await bridge.close()

def main():
    parser = argparse.ArgumentParser(description='Browser-only coding agent; commands are not sandboxed by default')
    parser.add_argument('task', nargs='?')
    parser.add_argument('-p', '--provider')
    parser.add_argument('-m', '--model')
    parser.add_argument('-w', '--workspace')
    parser.add_argument('--config')
    parser.add_argument('-y', '--yes', action='store_true', help='Explicitly auto-approve all actions, including disclosure')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resume', help='Run ID; use the exact original task. Unknown effects require manual reconciliation')
    parser.add_argument('--stdin', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ('provider', 'model', 'workspace'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    config['debug_transactions'] = args.debug
    if args.yes:
        config['confirm_changes'] = False
    if args.resume:
        config['resume_run'] = args.resume
        if not args.task:
            parser.error('--resume requires the original task')
    if args.stdin:
        import sys
        args.task = sys.stdin.read().strip()
        if not args.task:
            parser.error('No task on stdin')
    try:
        asyncio.run(run(config, args.task))
    except (KeyboardInterrupt, EOFError):
        console.print('Cancelled. Inspect any uncertain browser submission before continuing.')

if __name__ == '__main__':
    main()
