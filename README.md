# Browser LLM Agent

Drive web-based LLMs (ChatGPT, Claude, Gemini, Perplexity) as autonomous coding agents — **no API key needed**. Uses your existing browser session/subscription via Playwright.

The agent reads your task, controls the chat LLM through text markers, executes shell commands and file operations locally, and feeds results back. It loops autonomously until the task is done.

## Quick Start

### One-command setup (Linux, WSL2, Windows)

```bash
python setup.py
```

## CLI Usage

```bash
python cli.py                          # default: chatgpt
python cli.py -p deepseek -m expert
python cli.py --p perplexity
python cli.py --task "Create a hello.py"
```

Inside the REPL:

| Command | Action |
|---------|--------|
| `/help` | Show commands |
| `/clear` | Clear session and screen |
| `/history` | Show conversation history |
| `/changes` | Show file/shell change log |
| `/provider` | Show current provider/model |
| `/exit` | Quit |

`Alt+Enter` for multi-line input.

## How It Works

The agent types into the browser chat using Playwright. It sends a system prompt teaching the LLM a plain-text marker protocol:

```
[[[SHELL]]]
dir
[[[END]]]

[[[FILE path="./script.py"]]]
print("hello")
[[[END]]]

[[[READ path="./script.py"]]]
```

Login in the browser window on first run. The profile persists in `./browser_profile/`.

## Project Structure

```
setup.py          One-command setup script
agent.py          Single-shot mode
cli.py            Interactive Rich REPL
agent_core.py     Agent loop, event system, prompt builder
browser.py        Playwright wrapper (type, submit, read responses)
tools.py          Shell execution, file read/write, change tracking
context.py        Workspace context gathering, turn feedback
privacy.py        Redacts local paths/identity before sending to cloud
session.py        Conversation history and auto-summarization
config.yaml       Provider, model, tool, and prompt configuration
```
