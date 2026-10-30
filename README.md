# Browser LLM Agent / Orbit

Use a logged-in browser chat as the model transport for a local coding agent.
No model API or local inference adapter is required. The local executor controls
files and processes; the hosted model only proposes typed actions.

## Install and run

Python 3.11+ is required. Existing installation: switch to this branch and keep
the existing virtual environment. New installation: run `python setup.py`.

```sh
python cli.py --workspace ./project
python agent.py --workspace ./project "Fix the parser and run its tests"
orbit -p deepseek -m expert "Inspect this repository"
```

Log in manually in the browser. Provider interfaces change; live provider support
must be verified on your account. Use the service only where your account and
provider rules permit automation. There is no CAPTCHA or rate-limit bypass.

## Architecture

- `protocol.py`: one versioned JSON envelope, typed arguments, no marker scanning.
- `agent_core.py`: ordered, approval-gated orchestration and compact chat rollover.
- `journal.py`: SQLite/WAL execution journal and stable per-run action IDs.
- `tools.py`: exact file tools, async process streaming, bounded local logs.
- `reliable_browser.py`: serialized submissions with fail-closed completion checks.
- `provider_adapters.py`: DOM extraction and provider-specific rejection handling.
- `browser.py`: retained browser launch, authentication and model-selection helper.
- `context.py`: small file inventory; source is retrieved on demand.

Both entry points use the reliable runtime. The old marker protocol, guessed file
writes, indentation arrows, whole-file regeneration requirement, and ad-hoc
TASK_COMPLETE detection are removed from the active execution path. `session.py`
and the old transport internals remain for compatibility/reference but are not
used to execute actions by either entry point.

## Protocol

The executor supplies a turn ID with each request. Return exactly one object:

```json
{"protocol":"browser-agent/1","turn_id":"CURRENT_ID","kind":"actions","actions":[{"id":"read_parser","tool":"read_file","arguments":{"path":"src/parser.py","start_line":1,"end_line":120}}]}
```

Final response:

```json
{"protocol":"browser-agent/1","turn_id":"CURRENT_ID","kind":"final","message":"Explain changes and actual verification results."}
```

One optional outer `json` fence is accepted; prose, extra fields, duplicate JSON
keys, incomplete envelopes, duplicate action IDs and unknown tools are rejected.
An entire envelope is validated before any action runs. Payload strings remain
opaque, including examples of tool syntax. A batch stops at its first error or
denial. IDs identify logical operations for the whole run, not just one response.
Completed IDs replay their recorded result, and cannot be reused for new arguments.

Tools:

| Tool | Arguments |
|---|---|
| `read_file` | path; optional start_line, end_line |
| `list_files` | optional path, glob, limit |
| `search_text` | literal pattern; optional glob, context_lines, limit |
| `write_file` | path, content, expected_sha256 (null only for a new file) |
| `edit_file` | path, expected_sha256, replacements containing old_text/new_text |
| `run_process` | argv array; optional cwd, timeout |
| `run_shell` | command; optional cwd, timeout; shell selected in local config |
| `read_output` | output_id; optional byte offset and limit |

File reads return numbered lines, total line count and a hash of the complete file.
Read narrower regions if a selection exceeds the response budget. Edits require
unique, non-overlapping matches against the original file. JSON content is written
without fence removal, label stripping, whitespace normalization or newline
insertion. Empty files are allowed. Existing bytes are backed up locally before
replacement; current hashes are checked before and after preparing the temp file.
Atomic replacement is not an OS-level compare-and-swap: do not concurrently edit
the same file through another process during approval/execution.

## Approvals and privacy

By default, both mutations and reads require explicit approval. The review shows
exact arguments, candidate content and diff. Missing approval UI means denial.
`-y` / `--yes` explicitly disables all action approvals; use only in a trusted,
isolated environment. Interactive controls: `/help`, `/new`, `/changes`, `/history`,
`/clear`, `/debug on|off`, `/confirm on|off`, `/exit`. `/clear` clears the display,
not the durable journal; `/new` verifies a fresh provider conversation.

All outbound messages, including tool feedback, pass through one privacy boundary.
Home/workspace paths are redacted. Obvious credential assignments, common tokens,
and private-key headers block submission instead of silently changing source.
Sensitive file paths and internal state are excluded from normal file tools.
This is conservative heuristic protection, not complete secret detection. Commands
can still read secrets, and novel secrets can escape heuristics. Review disclosure.

## Processes and isolation

Processes use explicit argv; shell tools launch the configured executable with
PowerShell, cmd, or POSIX flags. An empty executable config selects PowerShell on
Windows and bash elsewhere. stdout/stderr are merged and streamed. Timeouts and
cancellation kill the process group (POSIX) or invoke taskkill /T /F (Windows).
Logs retain at most max_output_bytes, report discarded bytes, and expose a bounded
head/tail preview plus output paging. They do not promise unlimited output storage.

Workspace checks constrain file tools, not arbitrary commands. cwd is not a sandbox.
For stronger isolation, configure execution_prefix as a trusted argv prefix for
your own sandbox launcher; configure its mounts, network, credentials and resource
limits externally. No built-in container policy is claimed. File backups cannot
roll back arbitrary command effects. Backups/journals/logs are local, unencrypted,
and may contain sensitive source. Keep them private and delete them deliberately.

## Recovery and browser limits

State lives in workspace/.browser-agent unless state_dir is set in local config.
A run ID is printed when work begins. To recover a known-outcome run:

```sh
orbit --resume RUN_ID "the exact original task"
```

Resume starts a verified fresh chat with recorded results; it never silently
replays an interrupted tool. An EXECUTING call whose outcome was not recorded is
unknown, even if the command probably finished. Inspect the filesystem/processes
and browser first. Unknown effects require manual reconciliation; do not edit the
SQLite records just to force a retry. Start a new task only after checking effects.
The journal cannot guarantee exactly-once execution of arbitrary shell side effects.

Submission failures/timeouts keep the transport locked. No automatic resubmission
occurs. Completion normally requires observing generation and then its end plus
stable response text. allow_copy_completion_fallback is off by default because a
Copy control plus a pause is weaker evidence. If a provider's Stop selector is not
recognized, the safe default is a timeout, not execution of a partial response.
Rate limits stop the run. Provider adapters isolate selectors and extraction; old
DeepSeek errors are ignored by baseline, and assistant examples are not error UI.
An unchanged repeated error can be ambiguous and may time out conservatively.

Chat rollover uses bounded deterministic changes and recent results. It does not
claim to preserve every conversational detail. The model must re-read source when
needed. Browser message size limits are enforced, never silently truncated.

## Tests

```sh
python -m unittest discover -s tests -v
python -m playwright install chromium
# Fail rather than skip missing-browser DOM tests:
# POSIX: REQUIRE_BROWSER_TESTS=1 python -m unittest discover -s tests -v
```

Core tests need no browser login or network. DOM fixtures test extraction without
provider access; they skip locally if Chromium is missing. CI requires them and
runs on Windows/Linux with Python 3.11–3.13. Live provider sessions, model selection,
real-world crash windows and sandbox policies still need environment-specific
integration testing. Do not equate mocked/fixture tests with live certification.
