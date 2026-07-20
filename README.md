# Browser LLM Agent

Use your ChatGPT Plus / Claude Pro / Gemini Advanced **subscription** as an
autonomous AI agent — no API key needed. The agent controls a real browser,
pastes into the web chat, reads responses, executes commands on your computer,
and feeds the output back.

```
You / Hermes → server → browser chat LLM → tool markers → local tools
                ↑______________ agent activity stream ______________|
```

## Quick start

```bash
pip install -r requirements.txt
python -m playwright install chromium

# Interactive terminal
python cli.py

# Single task
python agent.py "Create a Flask app with a /health endpoint in ./myapp/"

# OpenAI-compatible server for Hermes
python server.py
```

First run: log into ChatGPT/Claude in the browser window. Session is saved to
`./browser_profile/`.

## Hermes integration

```bash
python server.py --port 8765

hermes config set model.provider custom
hermes config set model.base_url http://localhost:8765/v1
hermes config set model.api_key noop
```

The server streams **agent-style activity** back to Hermes:

- status (`Waiting for LLM…`, `Feeding tool output…`)
- shell commands as they run (`$ Get-ChildItem`)
- command output
- file writes with **unified diffs** (git-like change log)
- cleaned assistant text (tool markers stripped)
- final report with full change log

### What Hermes sees (example)

```text
*Waiting for LLM response (round 1)*

I'll create the app structure.

*Writing `myapp/app.py`...*
```diff
--- a/myapp/app.py
+++ b/myapp/app.py
@@ -0,0 +1,8 @@
+from flask import Flask
+app = Flask(__name__)
...
```

```
$ python -c "import myapp.app"
```

## Agent Actions
### [1] file_write ...
### [2] shell ...
## Change Log
+ created  myapp/app.py
$ python -c "import myapp.app"
## Assistant
Done — Flask app is ready.
```

## Architecture

| File | Role |
|------|------|
| `server.py` | OpenAI-compatible API for Hermes (streaming agent events) |
| `cli.py` | Interactive rich terminal REPL |
| `agent.py` | Single-shot autonomous task runner |
| `agent_core.py` | Shared agent loop + full system prompt + Hermes message parsing |
| `browser.py` | Playwright bridge — **clipboard paste** (not char-by-char typing) |
| `tools.py` | Shell / file read-write + unified diffs + change log |
| `session.py` | CLI conversation memory |
| `config.yaml` | Provider, tools, system prompt |

## Speed notes

- Messages are **pasted** via clipboard into the chat input (instant for long
  system prompts and tool feedback). Falls back to chunked typing if paste fails.
- Response wait polls the stop button at ~350ms intervals (not multi-second sleeps).
- Tool rounds stream to Hermes as they happen instead of waiting for the full turn.

## Providers

```yaml
provider: chatgpt    # or claude, gemini, perplexity
headless: false
model: ""            # Perplexity multi-model only
```

## Tool format

```
[[[SHELL]]]
pip install flask
[[[END]]]

[[[FILE path="./app.py"]]]
from flask import Flask
app = Flask(__name__)
[[[END]]]

[[[READ path="./app.py"]]]
```

The executor replies with:

```
[TOOL OUTPUT]
...
[/TOOL OUTPUT]
```

## Safety

- Dangerous shell patterns are blocked (`rm -rf`, `Remove-Item -Recurse -Force`, …)
- File writes restricted to `allowed_paths` in `config.yaml`
- Browser profile is isolated from your daily browser
