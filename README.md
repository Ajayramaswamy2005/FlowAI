# FlowAI

FlowAI is an AI agent you run in your terminal, in the same spirit as Claude
Code or OpenCode. Bring your own key from Anthropic, OpenAI, NVIDIA, Google
Gemini or OpenRouter, or run fully local with Ollama or your own NVIDIA NIM.
The agent can read, search and edit the files in your project, run shell
commands (after asking you), and search the web, while a security layer keeps
your API keys out of its reach.

It is also a Python library: the same agents, tools, workflows and memory the
CLI uses are yours to build on.

## Install

```bash
pip install flowai-cli
pipx install flowai-cli       # same, in its own isolated environment
```

Python 3.10 or newer. The package is `flowai-cli` (the name `flowai` is taken on PyPI);
the command it installs is `flowai`, and the import is `import flowai`.

## First run

```bash
flowai
```

The first time, FlowAI asks for a provider and an API key, stores the key in
your operating system's credential store, and sends one tiny request to check
it works. After that, `flowai` opens a session in the current directory.

## The interface

A FlowAI session is drawn as a current: every turn flows down one rail.

```
  █▀▀ █   █▀█ █ █ █ ▄▀█ █   v0.4.0
  █▀  █▄▄ █▄█ ▀▄▀▄▀ █▀█ █   an agent that moves with your code
  ▅▇█▇▅▂▁▁▃▆▇█▇▅▂▁▁▃▆▇█▇▅▂▁▁▃▆██▇▅▂▁▁▃▆██▇▄▂▁▁▃▆

  model     nvidia › nvidia/nemotron-3.5-lightning-30b-a3b
  workspace C:\Users\you\project
  shield    armed · key in Windows Credential Manager · approvals ask

 ◆ fix the failing test in tests/test_api.py
≋ undertow thought for 2.1s · 48 words   /think to read it
◇ read_file  tests/test_api.py                                        0.0s  ✓ 120 lines
◇ bash  pytest -q tests/test_api.py                                   1.9s  ✗ exit 1

┃ GATE  edit_file  wants to run                                          risk LOW
┃ src/api.py
┃ - return a - b
┃ + return a + b
┃ [y] allow once (default)   [a] always allow edit_file   [n] deny
● flowai
│ The bug was a sign error in add(). All 12 tests pass now.
╰─ 6.3s · 3 tools · 4.1k tok   tide ▂ 18%
```

- **◆** is your prompt. **≋ undertow** is the model's reasoning, folded to one
  line (`/think` reads it in full, `/think on` shows it live).
- **◇** is a tool call, with its timing and outcome.
- The **gate** asks before anything risky runs. Its rail is coloured by risk,
  it shows commands with syntax highlighting and edits as a diff, and it says
  why a call is risky.
- The answer is rendered markdown. While the model works, a swell of block
  characters moves along the rail.
- The **tide** gauge shows how full the model's context window is. The strip at
  the bottom of the prompt shows the model, tide, tokens spent and shield state.
- `/` completes commands. Esc then Enter adds a new line. Prompt history is
  kept, with any pasted key masked.

Piped output, CI logs and `--plain` get plain text instead. `NO_COLOR` and
`FLOWAI_NO_ANIM` are respected.

## NVIDIA

FlowAI works with every kind of NVIDIA key and endpoint:

| You have | Do this |
|---|---|
| A key from build.nvidia.com (`nvapi-...`) | `flowai configure`, pick NVIDIA, paste it |
| The key in an environment variable | Any of `NVIDIA_API_KEY`, `NVAPI_KEY`, `NGC_API_KEY`, `NIM_API_KEY` works |
| A NIM running on your own machine or cluster | `NVIDIA_BASE_URL=http://localhost:8000/v1` (or `flowai configure --provider nvidia --base-url ...`); no key needed |

The catalog's models do not all accept the same requests, and FlowAI adapts to
each one:

- **No tool support.** A model that refuses `tools` is sent the conversation
  without them, and you are told so. The same goes for `stream_options`, the
  `system` role and `max_tokens`. Any chat model on the catalog can be used.
- **Reasoning models** (Nemotron, DeepSeek, GPT-OSS) stream their thinking
  separately, whether it arrives as `reasoning_content` or inside `<think>` tags.
- **Model names.** A bare name such as `glm-5.3-flash` is resolved to its
  catalog id (`z-ai/glm-5.3-flash`).
- **Unavailable models.** Some models are listed but not available to every
  account, and retired ones answer 410. FlowAI then suggests models your key
  can call, and `flowai configure` finds a working one for you.
- **Rate limits.** The free tier's limits are retried with backoff,
  honouring `Retry-After`.
- **Key mistakes** are named: a key without the `nvapi-` prefix (an old NGC
  key cannot call the hosted API), or stray whitespace around it.

`flowai models nvidia [filter]` lists the catalog. In a session, `/models` does
the same, and `/model <id>` switches without losing the conversation.

## Security

An agent that runs commands on your machine is an obvious target. Someone
would like it to read your API key and send it somewhere, usually by hiding
instructions in a file or web page the agent reads. FlowAI defends in layers:

| Layer | What it does |
|---|---|
| Key vault | Keys live in Windows Credential Manager, the macOS Keychain or the Secret Service. Without one, they go in a file only your user can read. Keys are never copied into the environment |
| Scrubbed children | `bash`, `python_repl` and MCP servers start with every credential-looking variable removed. Python runs in a separate process, so it cannot reach into FlowAI's memory |
| Redaction | Every key FlowAI knows, plus the shapes of common keys (`nvapi-`, `sk-`, `sk-ant-`, `AIza`, `ghp_`...), is masked in tool output before the model sees it. That includes base64, hex, reversed and URL-encoded forms. Transcripts, prompt history and errors are masked too |
| Tripwires | Dumping the environment, reading `.env`, SSH keys, the credential store or FlowAI's own config always asks you, even with `--yolo`. With no terminal attached, the answer is no |
| Exfiltration block | A call that reads secrets and sends data over the network in one go is refused outright, as is any tool call whose arguments contain a key |
| Network guard | `fetch_url` refuses localhost, private networks and cloud metadata addresses, checking every redirect |
| Injection fence | Web and MCP output is wrapped as untrusted data, and the system prompt tells the model to treat it that way. Tool output that tries to give the model orders ("ignore your previous instructions", fake `<\|im_start\|>` turns) is flagged, and invisible Unicode that can hide text from you is stripped |
| Guardrails | Jailbreak and system-prompt extraction attempts are caught even when disguised (`i g n o r e`, zero-width or look-alike letters, invisible tag characters) or split across several messages. The model is reminded of its rules (`jailbreak: guard`) or the prompt is refused (`block`, the default for `flowai serve`). A key you paste into a prompt is masked before it reaches the provider, replies are redacted as they stream, and a model stuck repeating one tool call is stopped |
| Project files | A cloned repo's `flowai.yaml` may tighten security but never loosen it: `approval: auto` and weaker guardrails are ignored, and an MCP server it defines starts only after you approve that exact command |
| Terminal hygiene | Escape sequences are stripped from everything drawn, so a page cannot rewrite your clipboard or hide text |
| Server | `flowai serve` requires a bearer token, refuses non-local `Host` headers (DNS rebinding), and runs no shell tools unless started with `--allow-shell` |

`flowai shield` audits your setup and says what to fix. `flowai shield --fix`
moves plain-text keys into the keyring and locks the config directory, and
`flowai logout` deletes a saved key. These are tripwires, not a sandbox: read
[SECURITY.md](SECURITY.md) for what they do not cover.

## Commands

| Command | What it does |
|---|---|
| `flowai` | Interactive session in the current directory |
| `flowai "add type hints to utils.py"` | Run one task and exit |
| `flowai chat --resume` (or `-c`) | Continue the most recent session |
| `flowai run -q "..."` | One task, print only the final answer (good for scripts) |
| `echo "..." \| flowai run` | Read the prompt from stdin |
| `flowai configure` | Choose provider, model and API key |
| `flowai models [provider] [filter]` | List the models a provider serves |
| `flowai shield [--fix]` | Audit how your keys are protected |
| `flowai plugins` | List plugins, their tools and hooks, and any that failed to load |
| `flowai logout [provider]` | Delete a saved key |
| `flowai init [dir]` | Create a `flowai.yaml` and a starter `main.py` |
| `flowai inspect` | Show config status, saved sessions, checkpoints |
| `flowai tools` | List the built-in tools |
| `flowai serve` | HTTP API with streaming (`pip install 'flowai-cli[server]'`) |

Options for `chat` and `run`: `--provider`, `--model`, `-w/--workspace DIR`,
`--yolo` (do not ask before shell/python run; secret access still asks),
`--no-tools`, `--max-iterations N`, `--plain`, `--no-plugins`.

Inside a session: `/help`, `/model <id>`, `/models [filter]`,
`/provider <name> [model]`, `/tools`, `/think [on|off]`, `/cost`, `/shield`,
`/plugins`, `/new`, `/reset`, `/clear`, `/yolo`, `/exit`.

## What the agent can do

| Tool | Notes |
|---|---|
| `read_file`, `list_dir`, `glob_files`, `grep_files` | Read-only; never ask, except for files that hold secrets |
| `write_file`, `edit_file` | Change files inside the workspace only |
| `bash` | Asks first. Runs in the workspace, 120 s timeout |
| `python_repl` | Asks first. Runs in a separate interpreter, 60 s timeout |
| `search_web`, `fetch_url` | DuckDuckGo search and public page fetch |

Every file path is resolved against the workspace (the directory you started
in, or `-w`) and refused if it points outside.

## Plugins and hooks

A plugin adds tools and hooks. Hooks run at five points, each with a context
the hook may change or block:

| Hook | Runs | Context fields |
|---|---|---|
| `pre_prompt` | before your prompt reaches the model | `prompt`, `block(reason)` |
| `post_response` | after the final answer | `prompt`, `reply` |
| `pre_tool` | before a tool call runs | `tool`, `arguments`, `block(reason)` |
| `post_tool` | after a tool call ran | `tool`, `arguments`, `result` |
| `on_error` | when a turn fails | `error` |

```python
# ~/.flowai/plugins/house_rules.py
from flowai.plugins import Plugin
from flowai.tools import tool

class HouseRules(Plugin):
    name = "house-rules"

    def pre_tool(self, ctx):
        if ctx.tool == "bash" and "docker" in ctx.arguments.get("command", ""):
            ctx.block("docker is off limits in this project")

    def post_tool(self, ctx):
        with open("audit.log", "a") as log:
            log.write(f"{ctx.tool} {ctx.arguments}\n")

    def tools(self):
        @tool
        def word_count(text: str) -> int:
            "Count the words in a piece of text."
            return len(text.split())
        return [word_count]
```

Put the file in `~/.flowai/plugins/`, or ship it as a package with an entry
point in the `flowai.plugins` group. `flowai plugins` shows what loaded.

Plugins cannot weaken the shield, and a broken one cannot break the agent:

- `pre_tool` runs *before* the shield grades a call, so arguments a hook
  rewrites are checked like any others; `post_tool` output is redacted again.
- A `pre_*` hook that raises blocks what it guarded (fail closed). A `post_*`
  or `on_error` hook that raises is skipped. A plugin that fails to load is
  reported and the rest load anyway.
- Plugin tools cannot replace built-in tools such as `bash` or `read_file`.
- Plugins never load from the project directory, so opening a cloned repo
  cannot run its code. Turn them off with `--no-plugins`, `FLOWAI_NO_PLUGINS=1`,
  or `disabled_plugins: [name]` in your config.

In the library, pass `hooks=HookManager()` to an `Agent`, and register with
`hooks.register("pre_tool", fn)` or the `@hooks.on("pre_tool")` decorator.

## Configuration

Settings are layered, later wins:

1. `~/.flowai/config.json`, written by `flowai configure` (keys themselves go
   to the OS keyring)
2. `flowai.yaml` (or `.yml`, `.toml`, `.json`) in the current directory or any
   parent, so it works from a subdirectory of your project
3. Environment: `FLOWAI_PROVIDER`, `FLOWAI_MODEL`, and the provider key variables

See [flowai.yaml](flowai.yaml) for every option, including `mcp_servers` for
borrowing tools from Model Context Protocol servers (`pip install 'flowai-cli[mcp]'`).
Never put a key in `flowai.yaml`; `flowai shield` flags it if you do.

`FLOWAI_HOME` moves `~/.flowai`. `FLOWAI_PASS_ENV=GITHUB_TOKEN,...` hands
named variables to the agent's shell on purpose. `FLOWAI_KEYRING=0` turns
the keyring off.

## Using it as a library

```python
import asyncio
from flowai import Agent, tool

@tool
def compound_interest(principal: float, rate: float, years: int) -> float:
    """Balance after compounding yearly."""
    return round(principal * (1 + rate) ** years, 2)

agent = Agent(
    name="Advisor",
    provider="nvidia",             # anthropic, ollama, openai, gemini, openrouter
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    tools=[compound_interest],
)

print(asyncio.run(agent.run("What will $5,000 be after 10 years at 8%?")))
```

`agent.stream(prompt)` yields tokens as they arrive (and `reasoning_delta` for
reasoning models) and runs tool calls along the way. Tools you register in
your own code run without an approval prompt (you wrote them). To change that,
give a `ToolRegistry` an `approver` callback; to add the CLI's protections,
give it a `shield=Shield()` (`from flowai.security import Shield`).

Beyond a single agent the library has:

- **Specialised agents**: `PlannerAgent`, `CriticAgent`, `ToolAgent`, `SupervisorAgent`
- **Patterns**: `ReflectionPattern`, `PlanAndExecutePattern`, `DebatePattern`, `Swarm`
- **Workflows**: a graph engine with conditional edges, loops, parallel
  branches and checkpointing (`Workflow`, `FileCheckpointer`)
- **Memory**: `InMemory`, `QdrantMemory` (`flowai-cli[qdrant]`), `ChromaMemory` (`flowai-cli[chroma]`)
- **Observability**: per-span token counts and estimated cost (`agent.tracer`),
  an event bus, and an OpenTelemetry exporter (`flowai-cli[observability]`)
- **Plugins and hooks**: custom tools, providers and pre/post hooks (see above)
- **Guardrails**: `Agent(guardrails=Guardrails(jailbreak="block"))`, or `guardrails=False` to switch them off

Runnable examples are in [examples/](examples/).

## Developing FlowAI

```bash
pip install -e '.[dev]'
pytest
ruff check flowai tests
```

The tests are offline: providers are faked at the HTTP layer, so no key is
needed, and they never touch your real keyring.

## Releasing to PyPI

1. Bump `version` in `pyproject.toml` and `__version__` in `flowai/__init__.py`.
2. Push a tag `flowai-vX.Y.Z`. The `flowai-release` workflow builds the
   package and publishes it with PyPI Trusted Publishing.

## License

MIT
