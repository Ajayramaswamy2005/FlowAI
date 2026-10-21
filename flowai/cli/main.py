"""The `flowai` command.

    flowai                      start an interactive session (runs setup first if needed)
    flowai "fix the failing test"   run one prompt and exit
    flowai configure | init | run | chat | serve | inspect | tools
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

from flowai import __version__
from flowai.agents.standard import Agent
from flowai.cli import ui
from flowai.config.settings import (
    GLOBAL_CONFIG_DIR,
    GLOBAL_CONFIG_PATH,
    PROVIDER_DEFAULT_MODEL,
    PROVIDER_KEY_ENV,
    Settings,
)
from flowai.errors import FlowAIError
from flowai.providers.registry import ProviderRegistry, get_provider
from flowai.tools.builtin import default_system_prompt, default_toolset
from flowai.tools.registry import ToolRegistry

SUBCOMMANDS = {"configure", "init", "run", "chat", "serve", "inspect", "tools"}

PROVIDER_MENU = [
    ("anthropic", "Anthropic (Claude)"),
    ("openai", "OpenAI (GPT-4o, o-series)"),
    ("nvidia", "NVIDIA API Catalog (DeepSeek-R1, Llama 3.1)"),
    ("gemini", "Google Gemini"),
    ("openrouter", "OpenRouter (many models, one key)"),
    ("ollama", "Ollama (local, no key)"),
]

CHAT_HELP = """Commands:
  /help                 show this help
  /model <name>         switch model, keep the conversation
  /provider <name> [m]  switch provider (and optionally model)
  /tools                list the tools the agent can use
  /cost                 tokens used and estimated cost this session
  /new                  start a fresh session
  /reset                clear the conversation, keep the session id
  /yolo                 stop asking before shell/python tools run
  /exit                 quit (also: exit, quit, Ctrl-D)"""


# --------------------------------------------------------------- building blocks


class Approver:
    """Asks before approval-gated tools run; remembers 'always' per tool."""

    def __init__(self, mode: str = "ask"):
        self.mode = mode
        self.always: set[str] = set()

    def __call__(self, tool_name: str, arguments: dict) -> bool:
        if self.mode == "auto" or tool_name in self.always:
            return True
        if not sys.stdin.isatty():
            # Nobody can answer a prompt, so refuse rather than hang or guess.
            ui.warn(f"  Refused {tool_name}: no terminal to ask for approval (use --yolo to allow).")
            return False
        verdict = ui.ask_approval(tool_name, arguments)
        if verdict == "always":
            self.always.add(tool_name)
        return verdict in ("yes", "always")


def load_mcp_tools(cfg: Settings) -> list:
    """Start each configured MCP server and borrow its tools. Failures are reported, not fatal."""
    servers = {name: s for name, s in cfg.mcp_servers.items() if s.enabled}
    if not servers:
        return []
    from flowai.mcp.client import MCPClient

    tools: list = []
    for name, server in servers.items():
        try:
            client = MCPClient(server.command, server.args, server.env)
            found = asyncio.run(client.connect_and_discover_tools())
            tools.extend(found)
            ui.info(f"  MCP '{name}': {len(found)} tools")
        except Exception as exc:  # a broken server must not stop the session
            ui.warn(f"  MCP '{name}' failed to start: {exc}")
    return tools


def build_agent(args: argparse.Namespace, cfg: Settings, name: str = "FlowAI") -> tuple[Agent, Approver]:
    """One agent, wired the same way for `run` and `chat`."""
    if getattr(args, "workspace", None):
        os.environ["FLOWAI_WORKSPACE"] = str(Path(args.workspace).resolve())

    provider_name = getattr(args, "provider", None) or cfg.provider
    model = getattr(args, "model", None) or (
        cfg.model if provider_name == cfg.provider else PROVIDER_DEFAULT_MODEL.get(provider_name, cfg.model)
    )
    provider = get_provider(provider_name, model=model, **cfg.provider_options(provider_name))

    approver = Approver("auto" if getattr(args, "yolo", False) or cfg.approval == "auto" else "ask")
    tools = ToolRegistry([] if getattr(args, "no_tools", False) else default_toolset(), approver=approver)
    if not getattr(args, "no_tools", False):
        for t in load_mcp_tools(cfg):
            tools.register(t)

    agent = Agent(
        name=name,
        provider=provider,
        model=model,
        system_prompt=default_system_prompt(),
        tools=tools,
        max_iterations=getattr(args, "max_iterations", None) or cfg.max_iterations,
        temperature=cfg.temperature,
    )

    async def show(event):
        if event.event == "tool_start":
            ui.tool_start(event.data["tool"], event.data.get("arguments", {}))
        elif event.event == "tool_end":
            ui.tool_end(event.data["tool"], event.data.get("result", ""))

    agent.event_bus.subscribe(show)
    return agent, approver


def ensure_configured(cfg: Settings, provider: str | None = None) -> bool:
    """Tell the user how to fix a missing key instead of failing on the first request."""
    name = provider or cfg.provider
    if cfg.has_credentials(name):
        return True
    env_name = PROVIDER_KEY_ENV.get(name, "an API key")
    ui.error(f"no API key found for provider '{name}'.")
    print(f"  Run 'flowai configure' to save one, or set {env_name} in your environment.")
    return False


async def stream_reply(agent: Agent, prompt: str) -> None:
    print(ui.paint(f"\n{agent.name}: ", ui.BOLD, ui.GREEN), end="", flush=True)
    async for chunk in agent.stream(prompt):
        if chunk.text_delta:
            print(chunk.text_delta, end="", flush=True)
    print()


# ------------------------------------------------------------------- commands


def cli_configure(args: argparse.Namespace) -> int:
    """Choose a provider, model and key once; saved to ~/.flowai/config.json."""
    current: dict = {}
    if GLOBAL_CONFIG_PATH.exists():
        try:
            current = json.loads(GLOBAL_CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}

    provider = args.provider
    if not provider:
        print(ui.banner(["  FlowAI setup: pick a provider and paste your API key"]))
        for i, (_, label) in enumerate(PROVIDER_MENU, 1):
            print(f"  {i}. {label}")
        choice = input("\nChoice [1-6] (default 1): ").strip() or "1"
        if not choice.isdigit() or not 1 <= int(choice) <= len(PROVIDER_MENU):
            ui.error(f"'{choice}' is not one of the options.")
            return 1
        provider = PROVIDER_MENU[int(choice) - 1][0]
    provider = provider.lower()
    if provider not in PROVIDER_DEFAULT_MODEL:
        ui.error(f"unknown provider '{provider}'. Choose one of: {', '.join(PROVIDER_DEFAULT_MODEL)}")
        return 1

    api_key = args.key
    env_name = PROVIDER_KEY_ENV.get(provider)
    if env_name and not api_key:
        existing = current.get(env_name) or os.getenv(env_name)
        hint = " (press Enter to keep the current one)" if existing else ""
        try:
            import getpass
            api_key = getpass.getpass(f"{provider} API key{hint}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 1
        if not api_key and not existing:
            ui.error("an API key is required for this provider.")
            return 1

    model = args.model
    if not model:
        default_model = PROVIDER_DEFAULT_MODEL[provider]
        if args.provider:  # non-interactive: take the default
            model = default_model
        else:
            model = input(f"Model (Enter for {default_model}): ").strip() or default_model

    current["provider"] = provider
    current["model"] = model
    if api_key and env_name:
        current[env_name] = api_key

    GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_CONFIG_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")
    try:
        os.chmod(GLOBAL_CONFIG_PATH, 0o600)  # the file holds a secret
    except OSError:
        pass

    print(f"\nSaved to {GLOBAL_CONFIG_PATH}")
    print(f"  provider: {provider}\n  model:    {model}")
    print("\nStart with:  flowai")
    return 0


def cli_init(args: argparse.Namespace) -> int:
    """Scaffold a project with a flowai.yaml and a starter agent."""
    target_dir = Path(args.name)
    if (target_dir / "flowai.yaml").exists():
        ui.error(f"{target_dir / 'flowai.yaml'} already exists; not overwriting it.")
        return 1
    target_dir.mkdir(parents=True, exist_ok=True)
    cfg = Settings.load()

    (target_dir / "flowai.yaml").write_text(
        f"""# FlowAI project configuration. Values here override ~/.flowai/config.json.
project_name: "{target_dir.resolve().name}"
provider: "{cfg.provider}"
model: "{cfg.model}"
temperature: 0.7
max_iterations: 10

# "ask" confirms before shell/python tools run; "auto" never asks.
approval: "ask"

# Tools from MCP servers (needs: pip install 'flowai-cli[mcp]').
# mcp_servers:
#   filesystem:
#     command: "npx"
#     args: ["-y", "@modelcontextprotocol/server-filesystem", "."]

server:
  host: "127.0.0.1"
  port: 8000
""",
        encoding="utf-8",
    )
    if not (target_dir / "main.py").exists():
        (target_dir / "main.py").write_text(
            '''"""A starter FlowAI agent with one custom tool."""

import asyncio

from flowai import Agent, Settings, tool


@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


cfg = Settings.load()
agent = Agent(name="Assistant", provider=cfg.provider, model=cfg.model, tools=[greet])

if __name__ == "__main__":
    print(asyncio.run(agent.run("Use the greet tool to say hi to Alice.")))
''',
            encoding="utf-8",
        )
    print(f"Created a FlowAI project in {target_dir}")
    print(f"  cd {args.name}\n  flowai")
    return 0


def cli_run(args: argparse.Namespace) -> int:
    """Run one prompt to completion and print the answer."""
    cfg = Settings.load()
    if not ensure_configured(cfg, args.provider):
        return 1
    prompt = " ".join(args.prompt).strip()
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if not prompt:
        ui.error("no prompt given.")
        return 1

    agent, _ = build_agent(args, cfg)
    if args.quiet:
        print(asyncio.run(agent.run(prompt)))
    else:
        asyncio.run(stream_reply(agent, prompt))
        if args.cost:
            s = agent.tracer.summary()
            ui.info(f"\n{s['total_tokens']} tokens, est. ${s['total_cost_usd']:.4f}")
    return 0


def cli_chat(args: argparse.Namespace) -> int:
    """Interactive session; the conversation is saved after every turn."""
    from flowai.runtime.session import SessionStore

    cfg = Settings.load()
    if not cfg.has_credentials(args.provider) and not args.provider and sys.stdin.isatty():
        ui.warn("FlowAI is not set up yet. Let's do that now.\n")
        if cli_configure(argparse.Namespace(provider=None, key=None, model=None)) != 0:
            return 1
        cfg = Settings.load()
    if not ensure_configured(cfg, args.provider):
        return 1

    store = SessionStore()
    initial = []
    if args.session:
        session_id = args.session
        initial = store.load(session_id)
    elif args.resume:
        session_id = store.get_latest_session_id() or f"ses_{uuid.uuid4().hex[:8]}"
        initial = store.load(session_id)
    else:
        session_id = f"ses_{uuid.uuid4().hex[:8]}"

    agent, approver = build_agent(args, cfg)
    if initial:
        agent.messages = initial

    workspace = os.getenv("FLOWAI_WORKSPACE") or os.getcwd()
    lines = [
        f"  FlowAI {__version__}   {agent.provider.provider_name} / {agent.model}",
        f"  workspace: {workspace}",
        f"  session:   {session_id}" + (f" (resumed, {len(initial)} messages)" if initial else ""),
        "  /help for commands, /exit to quit",
    ]
    if approver.mode == "auto":
        lines.append("  approvals: off (--yolo). Shell and python tools run without asking.")
    print(ui.banner(lines))

    def handle_command(line: str) -> bool:
        """Run a slash command. Returns False when the session should end."""
        nonlocal session_id
        cmd, _, rest = line.partition(" ")
        rest = rest.strip()
        if cmd in ("/exit", "/quit"):
            return False
        if cmd == "/help":
            print(CHAT_HELP)
        elif cmd == "/model" and rest:
            agent.switch_model(rest)
            ui.info(f"model -> {rest} (conversation kept)")
        elif cmd == "/provider" and rest:
            name, _, model = rest.partition(" ")
            model = model.strip() or PROVIDER_DEFAULT_MODEL.get(name, agent.model)
            try:
                ProviderRegistry.get(name)
            except ValueError as exc:
                ui.error(str(exc))
                return True
            if not ensure_configured(cfg, name):
                return True
            agent.switch_model(model, provider=get_provider(name, model=model, **cfg.provider_options(name)))
            ui.info(f"provider -> {name} / {model} (conversation kept)")
        elif cmd == "/tools":
            for t in agent.tools.list_tools():
                gate = " (asks first)" if t.requires_approval and approver.mode == "ask" else ""
                print(f"  {t.name}{gate}: {t.description}")
        elif cmd == "/cost":
            s = agent.tracer.summary()
            ui.info(f"{s['total_tokens']} tokens, est. ${s['total_cost_usd']:.4f}")
        elif cmd == "/reset":
            agent.reset_session()
            store.save(session_id, agent.messages)
            ui.info("conversation cleared.")
        elif cmd == "/new":
            agent.reset_session()
            session_id = f"ses_{uuid.uuid4().hex[:8]}"
            ui.info(f"new session {session_id}")
        elif cmd == "/yolo":
            approver.mode = "auto"
            ui.warn("approvals off for this session.")
        else:
            ui.warn(f"unknown command '{line}'. Try /help.")
        return True

    while True:
        try:
            line = input(ui.paint("\n> ", ui.BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            break
        if line.startswith("/"):
            if not handle_command(line):
                break
            continue

        try:
            asyncio.run(stream_reply(agent, line))
        except KeyboardInterrupt:
            ui.warn("\n(interrupted)")
        except FlowAIError as exc:
            ui.error(str(exc))
            # Drop the unanswered prompt so the next turn is not confused by it.
            if agent.messages and agent.messages[-1].role == "user":
                agent.messages.pop()
        store.save(session_id, agent.messages)

    print(f"Session saved: {session_id}  (resume with: flowai chat --resume)")
    return 0


def cli_serve(args: argparse.Namespace) -> int:
    """Run the HTTP API."""
    try:
        import uvicorn

        from flowai.server.app import create_app
    except ImportError:
        ui.error("the server needs extra packages. Install them with: pip install 'flowai-cli[server]'")
        return 1

    cfg = Settings.load()
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port
    if host not in ("127.0.0.1", "localhost"):
        ui.warn(f"Binding to {host}: anyone who can reach this port can drive an agent with shell access.")
    print(f"FlowAI server on http://{host}:{port}")
    uvicorn.run(create_app(cfg), host=host, port=port)
    return 0


def cli_inspect(args: argparse.Namespace) -> int:
    """Show saved sessions and workflow checkpoints."""
    from flowai.runtime.session import SessionStore

    cfg = Settings.load()
    print(ui.banner([f"  FlowAI {__version__}   {cfg.provider} / {cfg.model}"]))
    print(f"Config: {GLOBAL_CONFIG_PATH if GLOBAL_CONFIG_PATH.exists() else '(not configured: run flowai configure)'}")
    print(f"Key for {cfg.provider}: {'found' if cfg.has_credentials() else 'MISSING'}")

    store = SessionStore()
    sessions = store.list_sessions()
    print(f"\nSessions ({store.storage_dir}):")
    if not sessions:
        print("  (none yet)")
    for s in sessions[:10]:
        print(f"  {s['session_id']:<15} {s['turns']:>4} messages")

    checkpoint_dir = Path(".flowai/checkpoints")
    files = sorted(checkpoint_dir.glob("*.json")) if checkpoint_dir.exists() else []
    print(f"\nWorkflow checkpoints ({checkpoint_dir}):")
    if not files:
        print("  (none)")
    for f in files[:10]:
        print(f"  {f.name}")
    return 0


def cli_tools(args: argparse.Namespace) -> int:
    """List the built-in tools."""
    for t in default_toolset():
        gate = " [asks first]" if t.requires_approval else ""
        print(f"{t.name}{gate}\n    {t.description}")
    return 0


# ---------------------------------------------------------------------- parser


def _add_agent_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", help="provider for this run (anthropic, openai, nvidia, gemini, openrouter, ollama)")
    p.add_argument("--model", help="model identifier for this run")
    p.add_argument("-w", "--workspace", help="directory the agent may read and change (default: current)")
    p.add_argument("--yolo", action="store_true", help="never ask before shell/python tools run")
    p.add_argument("--no-tools", action="store_true", help="plain chat, no tools")
    p.add_argument("--max-iterations", type=int, help="cap on tool-calling rounds per prompt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flowai",
        description="FlowAI: a terminal AI agent. Run 'flowai' to chat, or 'flowai \"<prompt>\"' for one task.",
    )
    parser.add_argument("--version", action="version", version=f"flowai {__version__}")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("configure", help="choose provider, model and API key")
    p.add_argument("--provider", help="provider name")
    p.add_argument("--key", help="API key (omit to be prompted without echo)")
    p.add_argument("--model", help="default model")
    p.set_defaults(func=cli_configure)

    p = sub.add_parser("init", help="scaffold a project with flowai.yaml")
    p.add_argument("name", nargs="?", default=".", help="directory (default: current)")
    p.set_defaults(func=cli_init)

    p = sub.add_parser("run", help="run one prompt and exit")
    p.add_argument("prompt", nargs="*", help="the prompt (or pipe it on stdin)")
    p.add_argument("-q", "--quiet", action="store_true", help="print only the final answer")
    p.add_argument("--cost", action="store_true", help="print token usage afterwards")
    _add_agent_options(p)
    p.set_defaults(func=cli_run)

    p = sub.add_parser("chat", help="interactive session (the default)")
    p.add_argument("--session", help="resume this session id")
    p.add_argument("-c", "--resume", action="store_true", help="resume the most recent session")
    _add_agent_options(p)
    p.set_defaults(func=cli_chat)

    p = sub.add_parser("serve", help="run the HTTP API (needs flowai-cli[server])")
    p.add_argument("--host", help="bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, help="port (default 8000)")
    p.set_defaults(func=cli_serve)

    p = sub.add_parser("inspect", help="show config status, sessions and checkpoints")
    p.set_defaults(func=cli_inspect)

    p = sub.add_parser("tools", help="list built-in tools")
    p.set_defaults(func=cli_tools)
    return parser


def normalise_argv(argv: list[str]) -> list[str]:
    """`flowai` alone means chat; `flowai "do X"` means run."""
    if not argv:
        return ["chat"]
    first = argv[0]
    if first in SUBCOMMANDS or first in ("-h", "--help", "--version"):
        return argv
    if first.startswith("-"):
        # `flowai --yolo` and friends: options for an interactive session.
        # One-shot runs with options use the explicit form: flowai run --model x "task".
        return ["chat", *argv]
    return ["run", *argv]


def main(argv: list[str] | None = None) -> int:
    # Model output can hold any character; a legacy Windows code page must not crash on it.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args =build_parser().parse_args(normalise_argv(sys.argv[1:] if argv is None else argv))
    try:
        return args.func(args) or 0
    except FlowAIError as exc:
        ui.error(str(exc))
        return 1
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    sys.exit(main())
