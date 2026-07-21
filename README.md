# Browser LLM Agent

Drive web-based LLMs (ChatGPT, Claude, Gemini, Perplexity) as autonomous coding agents — **no API key needed**. Uses your existing browser session/subscription via Playwright.

The agent reads your task, controls the chat LLM through text markers, executes shell commands and file operations locally, and feeds results back. It loops autonomously until the task is done.

## Quick Start

```bash
pip install -r requirements.txt
playwright install chromium
```

### Interactive REPL

```bash
python cli.py
```

Rich-powered terminal UI with markdown rendering, syntax highlighting, and tool execution panels.

```bash
python cli.py --provider claude
python cli.py --provider perplexity --model "GPT-4o"
```

### Single-shot

```bash
python agent.py "Build a Flask TODO app in ./myapp/"
python agent.py --provider claude "Refactor all Python files to use pathlib"
```

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

The agent parses responses, executes tools locally, and feeds `[TOOL OUTPUT]` back to the LLM. The loop continues until the LLM writes `TASK_COMPLETE`.

### Indentation

Inside `[[[FILE ...]]]` blocks, use `→` at line start for indentation:

```
[[[FILE path="./script.py"]]]
def greet():
→print("hello")
→→print("nested")
[[[END]]]
```

`→` = 4 spaces, `→→` = 8 spaces. The server converts these automatically.

## Configuration

Edit `config.yaml`:

```yaml
provider: chatgpt        # chatgpt, claude, gemini, perplexity
model: ""                # model override (for Perplexity)
headless: false           # true = hidden browser

tools:
  shell:
    enabled: true
    executable: "powershell.exe"   # blank = system default
    timeout: 90
  file_write:
    enabled: true
    allowed_paths: ["."]           # restrict to workspace
```

## Project Structure

```
agent.py          Single-shot mode
cli.py            Interactive Rich REPL
agent_core.py     Agent loop, event system, system prompt builder
browser.py        Playwright wrapper (type, submit, read responses)
tools.py          Shell execution, file read/write, change tracking
privacy.py        Redacts local paths/identity before sending to cloud
session.py        Conversation history and auto-summarization
config.yaml       Provider, model, tool, and prompt configuration
```

## Providers

| Provider | URL | Notes |
|----------|-----|-------|
| ChatGPT | chatgpt.com | Default. GPT-4o. |
| Claude | claude.ai | Sonnet/Opus. |
| Gemini | gemini.google.com | Google's models. |
| Perplexity | perplexity.ai | Multi-model: GPT-4o, Sonar, Claude, Grok. Use `--model`. |

Login in the browser window on first run. The profile persists in `./browser_profile/`.

## CLI Commands

Inside the REPL (`python cli.py`):

| Command | Action |
|---------|--------|
| `/help` | Show commands |
| `/clear` | Clear session and screen |
| `/history` | Show conversation history |
| `/changes` | Show file/shell change log |
| `/provider` | Show current provider/model |
| `/exit` | Quit |

`Alt+Enter` for multi-line input.

## Requirements

- Python 3.11+
- Playwright (Chromium)
- A subscription to at least one supported LLM provider
