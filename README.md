# Browser LLM Agent

Drive web-based LLMs (ChatGPT, Claude, Gemini, Perplexity) as autonomous coding agents — **no API key needed**. Uses your existing browser session/subscription via Playwright.

The agent reads your task, controls the chat LLM through text markers, executes shell commands and file operations locally, and feeds results back. It loops autonomously until the task is done.

## Quick Start

### One-command setup (Linux, WSL2, Windows)

```bash
python setup.py
```

This creates a `.venv`, installs all Python packages, Playwright Chromium, and (on Linux) system dependencies. After setup:

```bash
# Activate the venv (or use .venv/Scripts/activate on Windows)
source .venv/bin/activate    # Linux / WSL2
.venv\Scripts\activate       # Windows PowerShell

# Interactive REPL
python cli.py

# Single-shot
python agent.py "Build a Flask TODO app in ./myapp/"
```

### Manual setup

```bash
python -m venv .venv
source .venv/bin/activate        # Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
playwright install chromium

# Linux / WSL2 only — system libraries for Chromium:
playwright install-deps chromium
# Or manually: sudo apt-get install libnspr4 libnss3 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2t64
```

### Linux / WSL2 notes

Chromium on Linux requires system shared libraries not present in minimal installs. If `python cli.py` fails with `error while loading shared libraries`, run:

```bash
playwright install-deps chromium
```

WSL2 with GUI (WSLg) works out of the box. For WSL2 without GUI, install an X server or use `headless: true` in config.yaml.

## CLI Usage

```bash
python cli.py                          # default: chatgpt
python cli.py --provider claude
python cli.py --provider perplexity --model "GPT-4o"
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

`→` = 4 spaces, `→→` = 8 spaces. The agent converts these automatically.

## Configuration

Edit `config.yaml`:

```yaml
provider: chatgpt        # chatgpt, claude, gemini, perplexity
model: ""                # model override (for Perplexity)
headless: false          # true = hidden browser

tools:
  shell:
    enabled: true
    executable: "powershell.exe"   # blank = system default (bash on Linux)
    timeout: 90
  file_write:
    enabled: true
    allowed_paths: ["."]          # restrict to workspace
```

## Providers

| Provider | URL | Notes |
|----------|-----|-------|
| ChatGPT | chatgpt.com | Default. GPT-4o. |
| Claude | claude.ai | Sonnet/Opus. |
| Gemini | gemini.google.com | Google's models. |
| Perplexity | perplexity.ai | Multi-model. Use `--model`. |

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

## Requirements

- Python 3.11+
- Playwright (Chromium)
- A subscription to at least one supported LLM provider
- (Linux) System libraries for Chromium — `setup.py` installs these automatically
