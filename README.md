# Browser LLM Agent

Use ChatGPT, Claude, Gemini, Perplexity, or DeepSeek in your browser as a local coding agent — no API key required.

The agent uses your existing browser session, writes files, runs shell commands, and feeds results back to the LLM until the task is complete.

## Setup

```bash
git clone <repo-url>
cd browser-llm-agent
python setup.py
```

Setup creates a local `.venv`, installs dependencies and Playwright Chromium, then installs the `orbit` command.

Close and reopen your terminal after setup.

## Usage

Open a terminal in the folder where you want the agent to work:

```bash
cd path/to/your/project
orbit
```

Run a task directly:

```bash
orbit "Create a hello.py"  # one-shot 
orbit -p deepseek -m expert  # preferred, CLI chat
orbit --provider deepseek --model expert "Fix the current build"
orbit --provider perplexity "Review this project"
```

`orbit` uses the agent's installed virtual environment automatically. You do not need to activate `.venv`.

Run this for all options:

```bash
orbit --help
```

## How It Works

On first run, log in to the selected provider in the browser window.

The agent sends instructions to the browser LLM, which responds with simple action markers:

```text
[[[SHELL]]]
dir
[[[END]]]

[[[FILE path="./hello.py"]]]
print("hello")
[[[END]]]
```

The agent executes the actions locally and sends the results back to the LLM.

## Providers

- `chatgpt` — default
- `claude`
- `gemini`
- `perplexity`
- `deepseek`

Configure the default provider or model in `config.yaml`.

## Commands

Inside interactive mode:

| Command | Action |
|---|---|
| `/help` | Show commands |
| `/clear` | Clear session and screen |
| `/history` | Show conversation history |
| `/changes` | Show file and shell changes |
| `/provider` | Show provider and model |
| `/debug on` | Enable transaction diagnostics |
| `/debug off` | Disable transaction diagnostics |
| `/exit` | Quit |

Use `Alt+Enter` for multi-line input.

## Notes

- Your browser profile is stored in `./browser_profile/`.
- The agent works in the folder where you run `orbit`.
- On Linux/WSL2, if `orbit` is not found after setup, run:

```bash
source ~/.profile
```