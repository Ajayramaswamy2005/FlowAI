# FlowAI

FlowAI is an AI agent you run in your terminal, in the same spirit as Claude
Code or OpenCode. Bring your own key from Anthropic, OpenAI, NVIDIA, Google
Gemini or OpenRouter, or run fully local with Ollama. The agent can read, search
and edit the files in your project, run shell commands (after asking you), and
search the web.

It is also a Python library: the same agents, tools, workflows and memory the
CLI uses are yours to build on.

## Install

```bash
pip install flowai-cli        # once it is published to PyPI
pipx install flowai-cli       # same, in its own isolated environment

# from a checkout of this repository:
pip install .
```

Python 3.10 or newer. The package is `flowai-cli` (the name `flowai` is taken on PyPI);
the command it installs is `flowai`, and the import is `import flowai`.

## First run

```bash
flowai
```

The first time, FlowAI asks for a provider and an API key and saves them to
`~/.flowai/config.json`. After that, `flowai` opens a session in the current
directory:

```
------------------------------------------------------------
  FlowAI 0.2.0   anthropic / claude-sonnet-5-5
  workspace: /home/you/my-project
  session:   ses_3f9a1c2e
  /help for commands, /exit to quit
------------------------------------------------------------

> why does the test in tests/test_api.py fail?

  [tool] grep_files(pattern=def test_, path=tests)
  [done] tests/test_api.py:12: def test_create_user(): (+4 more lines)
  [tool] read_file(path=tests/test_api.py)
  ...
```

Set up non-interactively instead (for scripts or CI):

```bash
flowai configure --provider anthropic --key "$ANTHROPIC_API_KEY"
```

Or skip configuration entirely: if `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`NVIDIA_API_KEY`, `GEMINI_API_KEY` or `OPENROUTER_API_KEY` is set, FlowAI uses
it.

## Commands

| Command | What it does |
|---|---|
| `flowai` | Interactive session in the current directory |
| `flowai "add type hints to utils.py"` | Run one task and exit |
| `flowai chat --resume` (or `-c`) | Continue the most recent session |
| `flowai run -q "..."` | One task, print only the final answer (good for scripts) |
| `echo "..." \| flowai run` | Read the prompt from stdin |
| `flowai configure` | Choose provider, model and API key |
| `flowai init [dir]` | Create a `flowai.yaml` and a starter `main.py` |
| `flowai inspect` | Show config status, saved sessions, checkpoints |
| `flowai tools` | List the built-in tools |
| `flowai serve` | HTTP API with streaming (`pip install 'flowai-cli[server]'`) |

Options for `chat` and `run`: `--provider`, `--model`, `-w/--workspace DIR`,
`--yolo` (do not ask before shell/python tools run), `--no-tools`,
`--max-iterations N`.

Inside a session: `/help`, `/model <name>`, `/provider <name> [model]`,
`/tools`, `/cost`, `/new`, `/reset`, `/yolo`, `/exit`.

## What the agent can do

| Tool | Notes |
|---|---|
| `read_file`, `list_dir`, `glob_files`, `grep_files` | Read-only; never asks |
| `write_file`, `edit_file` | Changes files inside the workspace only |
| `bash` | Asks first. Runs in the workspace, 120 s timeout |
| `python_repl` | Asks first. Runs code in a fresh namespace |
| `search_web`, `fetch_url` | DuckDuckGo search and page fetch |

**Safety.** Every file path is resolved against the workspace (the directory
you started in, or `-w`) and refused if it points outside. Shell, Python and
MCP tools show you exactly what they will run and wait for `y`, `n` or `a`
(always, for that tool, this session). With no terminal attached, those tools
are refused rather than run. `--yolo` or `approval: "auto"` in `flowai.yaml`
turns the prompt off; use it only in a sandbox.

## Configuration

Settings are layered, later wins:

1. `~/.flowai/config.json`, written by `flowai configure`
2. `flowai.yaml` (or `.yml`, `.toml`, `.json`) in the current directory or any
   parent, so it works from a subdirectory of your project
3. Environment: `FLOWAI_PROVIDER`, `FLOWAI_MODEL`, and the provider key variables

See [flowai.yaml](flowai.yaml) for every option, including `mcp_servers` for
borrowing tools from Model Context Protocol servers (`pip install 'flowai-cli[mcp]'`).

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
    provider="anthropic",          # nvidia, ollama, openai, gemini, openrouter
    model="claude-sonnet-5-5",
    tools=[compound_interest],
)

print(asyncio.run(agent.run("What will $5,000 be after 10 years at 8%?")))
```

`agent.stream(prompt)` yields tokens as they arrive and runs tool calls along
the way. Tools you register in your own code run without an approval prompt
(you wrote them); give a `ToolRegistry` an `approver` callback to change that.

Beyond a single agent the library has:

- **Specialised agents**: `PlannerAgent`, `CriticAgent`, `ToolAgent`, `SupervisorAgent`
- **Patterns**: `ReflectionPattern`, `PlanAndExecutePattern`, `DebatePattern`, `Swarm`
- **Workflows**: a graph engine with conditional edges, loops, parallel
  branches and checkpointing (`Workflow`, `FileCheckpointer`)
- **Memory**: `InMemory`, `QdrantMemory` (`flowai-cli[qdrant]`), `ChromaMemory` (`flowai-cli[chroma]`)
- **Observability**: per-span token counts and estimated cost (`agent.tracer`),
  an event bus, and an OpenTelemetry exporter (`flowai-cli[observability]`)
- **Plugins**: third-party providers via the `flowai.plugins` entry point

Runnable examples are in [examples/](examples/).

## Developing FlowAI

```bash
pip install -e '.[dev]'
pytest
ruff check flowai tests
```

The tests are offline: providers are faked at the HTTP layer, so no key is needed.

## Releasing to PyPI

1. Bump `version` in `pyproject.toml` and `__version__` in `flowai/__init__.py`.
2. Push a tag `flowai-vX.Y.Z`. The `flowai-release` workflow builds this
   directory and publishes it with PyPI Trusted Publishing; configure the
   `flowai-cli` project on PyPI to trust this repository and workflow first.



## License

MIT
