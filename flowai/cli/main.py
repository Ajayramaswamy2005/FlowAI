"""The `flowai` command.

    flowai                          start an interactive session (runs setup first if needed)
    flowai "fix the failing test"   run one prompt and exit
    flowai configure | init | run | chat | models | shield | logout | serve | inspect | tools
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import uuid
from pathlib import Path

from flowai import __version__
from flowai.agents.standard import Agent
from flowai.cli import ui
from flowai.config.settings import (
    PROVIDER_DEFAULT_MODEL,
    PROVIDER_KEY_ENV,
    Settings,
    nvidia_is_self_hosted,
    provider_key_envs,
)
from flowai.errors import FlowAIError, ProviderError
from flowai.providers.base import Message
from flowai.providers.registry import ProviderRegistry, get_provider
from flowai.security import keystore
from flowai.security.secrets import REDACTOR, redact, register_secret, sanitize_terminal
from flowai.security.shield import Shield
from flowai.tools.builtin import default_system_prompt, default_toolset
from flowai.tools.registry import ToolRegistry

SUBCOMMANDS = {"configure", "init", "run", "chat", "serve", "inspect", "tools", "models", "shield", "logout"}

PROVIDER_MENU = [
    ("anthropic", "Anthropic (Claude)"),
    ("openai", "OpenAI (GPT-4o, o-series)"),
    ("nvidia", "NVIDIA (build.nvidia.com catalog, or your own NIM)"),
    ("gemini", "Google Gemini"),
    ("openrouter", "OpenRouter (many models, one key)"),
    ("ollama", "Ollama (local, no key)"),
]

CHAT_COMMANDS = {
    "/help": "show this list",
    "/model": "switch model, keep the conversation: /model <id>",
    "/models": "list the models this provider serves: /models [filter]",
    "/provider": "switch provider: /provider <name> [model]",
    "/tools": "the tools the agent can use",
    "/think": "read the last answer's reasoning; /think on|off shows it live",
    "/cost": "tokens used and estimated cost this session",
    "/shield": "what protects your keys right now",
    "/new": "start a fresh session",
    "/reset": "clear the conversation, keep the session id",
    "/clear": "clear the screen",
    "/yolo": "stop asking before shell/python run (tripwires still ask)",
    "/exit": "leave (also: exit, quit, Ctrl-D)",
}


# --------------------------------------------------------------- building blocks


class Approver:
    """Decides whether a gated tool call may run.

    'ask' asks the user; 'auto' (--yolo) allows without asking. Either way a
    call the shield rates critical, because it touches secrets, is put to the
    user every time: --yolo is a convenience, not a way to read your keys.
    """

    def __init__(self, mode: str = "ask", ask=None):
        self.mode = mode
        self.ask = ask or (lambda name, arguments, risk: ui.ask_approval(name, arguments, risk))
        self.always: set[str] = set()

    async def __call__(self, tool_name: str, arguments: dict, risk=None) -> bool:
        critical = risk is not None and risk.level == "critical"
        if not critical and (self.mode == "auto" or tool_name in self.always):
            return True
        if not ui.stdin_is_terminal():
            # Nobody can answer a prompt, so refuse rather than hang or guess.
            why = "it touches secrets" if critical else "no terminal to ask for approval (use --yolo to allow)"
            ui.warn(f"  Refused {tool_name}: {why}.")
            return False
        verdict = self.ask(tool_name, arguments, risk)
        if asyncio.iscoroutine(verdict):
            verdict = await verdict
        if verdict == "always" and not critical:
            self.always.add(tool_name)
        return verdict in ("yes", "always")


def use_rich_interface(args: argparse.Namespace) -> bool:
    if getattr(args, "plain", False) or os.getenv("FLOWAI_PLAIN"):
        return False
    if not (sys.stdout.isatty() and ui.stdin_is_terminal()):
        return False
    try:
        import prompt_toolkit  # noqa: F401
        import rich  # noqa: F401
    except ImportError:
        return False
    return True


def make_view(args: argparse.Namespace):
    if use_rich_interface(args):
        from flowai.cli.tui import Interface

        view = Interface()
    else:
        view = ui.PlainView()
    view.commands = CHAT_COMMANDS
    return view


def load_mcp_tools(cfg: Settings, view) -> list:
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
            view.info(f"MCP '{name}': {len(found)} tools")
        except Exception as exc:  # a broken server must not stop the session
            view.notice(f"MCP '{name}' failed to start: {exc}")
    return tools


def build_agent(args: argparse.Namespace, cfg: Settings, view=None, name: str = "FlowAI") -> tuple[Agent, Approver]:
    """One agent, wired the same way for `run` and `chat`."""
    view = view or ui.PlainView()
    if getattr(args, "workspace", None):
        os.environ["FLOWAI_WORKSPACE"] = str(Path(args.workspace).resolve())

    provider_name = getattr(args, "provider", None) or cfg.provider
    model = getattr(args, "model", None) or (
        cfg.model if provider_name == cfg.provider else PROVIDER_DEFAULT_MODEL.get(provider_name, cfg.model)
    )
    provider = get_provider(provider_name, model=model, **cfg.provider_options(provider_name))
    register_secret(provider.api_key, provider.api_key_env or "api-key")

    mode = "auto" if getattr(args, "yolo", False) or cfg.approval == "auto" else "ask"
    approver = Approver(mode, ask=view.gate)
    no_tools = getattr(args, "no_tools", False)
    tools = ToolRegistry([] if no_tools else default_toolset(), approver=approver, shield=Shield())
    if not no_tools:
        for t in load_mcp_tools(cfg, view):
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
            view.tool_start(event.data["tool"], event.data.get("arguments", {}))
        elif event.event == "tool_end":
            view.tool_end(event.data["tool"], event.data.get("result", ""), event.data.get("seconds"))

    agent.event_bus.subscribe(show)
    return agent, approver


def missing_key_help(name: str) -> str:
    envs = provider_key_envs(name)
    if name == "nvidia":
        return (
            "Run 'flowai configure' to save one, or set any of "
            f"{', '.join(envs)}. Get a key at https://build.nvidia.com, "
            "or point NVIDIA_BASE_URL at your own NIM (no key needed)."
        )
    env_name = envs[0] if envs else "an API key"
    return f"Run 'flowai configure' to save one, or set {env_name} in your environment."


def ensure_configured(cfg: Settings, provider: str | None = None) -> bool:
    """Tell the user how to fix a missing key instead of failing on the first request."""
    name = provider or cfg.provider
    if cfg.has_credentials(name):
        return True
    ui.error(f"no API key found for provider '{name}'.")
    print(f"  {missing_key_help(name)}")
    return False


def key_summary(cfg: Settings, provider: str) -> str:
    if provider == "ollama":
        return "local model, no key"
    if provider == "nvidia" and nvidia_is_self_hosted(cfg.provider_options("nvidia").get("base_url")):
        return "self-hosted NIM, no key"
    source = cfg.key_source(provider)
    return f"key in {source}" if source else "no key"


async def probe(provider) -> FlowAIError | None:
    """Make the smallest possible request; return the error, or None if the key and model work."""
    try:
        await provider.generate([Message(role="user", content="Reply with: ok")], max_tokens=16, temperature=0.5)
    except FlowAIError as exc:
        return exc
    return None


def model_unavailable(exc: BaseException) -> bool:
    return isinstance(exc, ProviderError) and exc.status_code in (404, 410)


def suggest_models(view, provider) -> None:
    """After 'model not found', say which models to try instead."""
    if not hasattr(provider, "suggest_models"):
        return
    suggestions = asyncio.run(provider.suggest_models())
    if suggestions:
        view.info(f"models to try: {', '.join(suggestions)}   (switch with /model <id>)")


def find_working_model(provider_name: str, cfg: Settings) -> str | None:
    """Try the recommended models in order and return the first this key can call."""
    from flowai.providers.nvidia import RECOMMENDED_MODELS

    for candidate in RECOMMENDED_MODELS:
        print(f"  trying {candidate}...", end=" ", flush=True)
        error = asyncio.run(probe(get_provider(provider_name, model=candidate, **cfg.provider_options(provider_name))))
        if error is None:
            print(ui.paint("works.", ui.GREEN))
            return candidate
        print("not available." if model_unavailable(error) else "failed.")
    return None


# ------------------------------------------------------------------- commands


def cli_configure(args: argparse.Namespace) -> int:
    """Choose a provider, model and key once. Keys go to the OS keyring when one is available."""
    current = keystore.read_config()
    interactive = not args.provider

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

    base_url = args.base_url
    if provider == "nvidia" and interactive and not base_url:
        where = input("Hosted NVIDIA catalog, or your own NIM server? [H]osted / [s]elf-hosted: ").strip().lower()
        if where.startswith("s"):
            base_url = input("NIM base URL (e.g. http://localhost:8000/v1): ").strip()
    if base_url:
        providers = current.setdefault("providers", {})
        providers.setdefault(provider, {})["base_url"] = base_url

    env_name = PROVIDER_KEY_ENV.get(provider)
    needs_key = env_name and not (provider == "nvidia" and nvidia_is_self_hosted(base_url))
    api_key = (args.key or "").strip()
    if args.key and sys.stdin.isatty():
        ui.warn("  Note: a key given with --key stays in your shell history. Prefer the prompt: flowai configure")
    if needs_key and not api_key:
        existing = keystore.stored_key(env_name) or os.getenv(env_name)
        hint = " (press Enter to keep the current one)" if existing else ""
        try:
            import getpass
            api_key = getpass.getpass(f"{provider} API key, hidden as you type{hint}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 1
        if not api_key and not existing:
            ui.error("an API key is required for this provider.")
            return 1

    if provider == "nvidia" and api_key and not base_url:
        from flowai.providers.nvidia import NvidiaProvider

        problem = NvidiaProvider(api_key=api_key).key_problem()
        if problem:
            ui.warn(f"  Heads up: {problem}")

    model = args.model
    if not model:
        default_model = PROVIDER_DEFAULT_MODEL[provider]
        if not interactive:
            model = default_model
        else:
            model = input(f"Model (Enter for {default_model}): ").strip() or default_model

    current["provider"] = provider
    current["model"] = model
    keystore.write_config(current)
    location = None
    if api_key and env_name:
        location = keystore.store_key(env_name, api_key)
        register_secret(api_key, env_name)

    print(f"\nSaved to {keystore.config_path()}")
    print(f"  provider: {provider}\n  model:    {model}")
    if location == "keyring":
        print(f"  key:      in {keystore.keyring_name()} (the config file holds only a marker)")
    elif location == "file":
        print("  key:      in the config file, readable by your user only (no OS keyring available)")

    if interactive and not args.no_verify and (api_key or not needs_key or keystore.stored_key(env_name)):
        cfg = Settings.load()
        print("\nChecking the key with a tiny request...", end=" ", flush=True)
        prov = get_provider(provider, model=model, **cfg.provider_options(provider))
        error = asyncio.run(probe(prov))
        if error is None:
            print(ui.paint("it works.", ui.GREEN))
        elif provider == "nvidia" and model_unavailable(error):
            # The key is fine; this account just cannot call that model. Find one it can.
            print(f"the key works, but '{model}' is not available to it.")
            working = find_working_model(provider, cfg)
            if working:
                current["model"] = model = working
                keystore.write_config(current)
                print(f"  Saved {working} as your default model.")
            else:
                print("  None of the usual models answered. List the rest with: flowai models nvidia")
        else:
            print()
            ui.error(str(error))
            print("  The settings were saved anyway. Fix the key with 'flowai configure' and try again.")

    print("\nStart with:  flowai")
    return 0


def cli_logout(args: argparse.Namespace) -> int:
    """Delete a saved key from the keyring and the config file."""
    cfg = Settings.load()
    provider = (args.provider or cfg.provider).lower()
    removed = [name for name in provider_key_envs(provider) if keystore.forget_key(name)]
    if removed:
        print(f"Removed the saved {provider} key ({', '.join(removed)}).")
    else:
        print(f"No saved key for {provider}.")
    still = [n for n in provider_key_envs(provider) if os.getenv(n)]
    if still:
        ui.warn(f"  {', '.join(still)} is still set in your environment; unset it in your shell profile too.")
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
# Never put API keys in this file: it gets committed. Use 'flowai configure'.
project_name: "{target_dir.resolve().name}"
provider: "{cfg.provider}"
model: "{cfg.model}"
temperature: 0.7
max_iterations: 10

# "ask" confirms before shell/python tools run; "auto" never asks
# (calls that touch secrets still ask).
approval: "ask"

# A self-hosted NVIDIA NIM instead of the hosted catalog:
# providers:
#   nvidia:
#     base_url: "http://localhost:8000/v1"

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
from flowai.providers import get_provider


@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


cfg = Settings.load()
provider = get_provider(cfg.provider, model=cfg.model, **cfg.provider_options())
agent = Agent(name="Assistant", provider=provider, model=cfg.model, tools=[greet])

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

    if args.quiet:
        agent, _ = build_agent(args, cfg)
        print(sanitize_terminal(redact(asyncio.run(agent.run(prompt)))))
        return 0

    view = make_view(args)
    agent, _ = build_agent(args, cfg, view)
    for notice in cfg.notices:
        view.notice(notice)
    try:
        asyncio.run(view.run_turn(agent, prompt))
    except KeyboardInterrupt:
        view.abort_turn()
        return 130
    except FlowAIError as exc:
        view.error(str(exc))
        if model_unavailable(exc):
            suggest_models(view, agent.provider)
        return 1
    if args.cost:
        s = agent.tracer.summary()
        ui.info(f"\n{s['total_tokens']} tokens, est. ${s['total_cost_usd']:.4f}")
    return 0


def shield_report(cfg: Settings, approvals: str | None = None) -> list[tuple[str, str, str]]:
    """What protects the user's keys right now, line by line: (state, label, detail)."""
    rows: list[tuple[str, str, str]] = []
    keyring = keystore.keyring_name()
    for provider in PROVIDER_KEY_ENV:
        source = cfg.key_source(provider)
        if not source:
            continue
        if source.startswith("project config"):
            rows.append(("bad", f"{provider} key", "written in a project file that may be committed; move it"))
        elif "plain text" in source:
            if keyring:
                rows.append(("warn", f"{provider} key", f"plain-text file; 'flowai shield --fix' moves it into {keyring}"))
            else:
                rows.append(("info", f"{provider} key", "owner-only file (no OS keyring on this machine)"))
        elif source.startswith("environment"):
            rows.append(("info", f"{provider} key", f"{source}; hidden from every tool FlowAI runs"))
        else:
            rows.append(("ok", f"{provider} key", f"sealed in {source}"))
    if not any(r[1].endswith(" key") for r in rows):
        rows.append(("info", "keys", "none stored yet"))

    problem = keystore.permission_problem(keystore.config_path())
    if problem:
        rows.append(("bad", "config file", problem))
    for notice in cfg.notices:
        rows.append(("bad", "project file", notice))

    rows.append(("ok", "redaction", f"{REDACTOR.known_count()} secrets known; masked in tool output, transcripts, history"))
    rows.append(("ok", "child processes", "bash, python and MCP servers start without credential variables"))
    rows.append(("ok", "python", "runs in a separate process, never inside FlowAI where keys are held"))
    rows.append(("ok", "tripwires", "env dumps, key files and credential stores ask every time, even with --yolo"))
    rows.append(("ok", "exfiltration", "reading secrets and sending data in one call is blocked outright"))
    rows.append(("ok", "network guard", "fetch_url refuses localhost, private networks and cloud metadata"))
    rows.append(("ok", "injection", "web and MCP output is fenced as untrusted data"))

    mode = approvals or cfg.approval
    if mode == "ask":
        rows.append(("ok", "approvals", "shell, python and risky calls ask before running"))
    else:
        rows.append(("warn", "approvals", "off (--yolo): shell and python run unasked; tripwires still apply"))

    workspace = Path(os.getenv("FLOWAI_WORKSPACE") or os.getcwd())
    if (workspace / ".env").exists() and not _git_ignored(workspace, ".env"):
        rows.append(("warn", ".env", "this project's .env is not git-ignored; add it to .gitignore"))
    if cfg.server.host not in ("127.0.0.1", "localhost", "::1"):
        rows.append(("warn", "server", f"configured to listen on {cfg.server.host}, beyond this machine"))
    return rows


def _git_ignored(directory: Path, name: str) -> bool:
    """True when git would not commit this file (or there is no git here to commit it)."""
    try:
        result = subprocess.run(["git", "check-ignore", "-q", name], cwd=directory, capture_output=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return True
    # 0: ignored, 1: not ignored, 128: not a git repository.
    return result.returncode != 1


def cli_shield(args: argparse.Namespace) -> int:
    """Audit how FlowAI is protecting your keys, and optionally fix what it can."""
    if args.fix:
        moved = keystore.migrate_file_keys_to_keyring()
        if moved:
            print(f"Moved into {keystore.keyring_name()}: {', '.join(moved)}")
        if keystore.config_path().exists():
            keystore.harden_permissions(keystore.config_path())
            keystore.harden_permissions(keystore.flowai_home(), directory=True)
            print(f"Locked {keystore.flowai_home()} to your user only.")
    cfg = Settings.load()
    rows = shield_report(cfg)
    make_view(args).show_report("FlowAI shield", rows)
    return 1 if any(state == "bad" for state, _, _ in rows) else 0


def cli_models(args: argparse.Namespace) -> int:
    """List the models a provider serves (NVIDIA's catalog has well over a hundred)."""
    cfg = Settings.load()
    provider_name = (args.provider or cfg.provider).lower()
    provider = get_provider(provider_name, model=cfg.model, **cfg.provider_options(provider_name))
    models = asyncio.run(provider.list_models())
    if args.filter:
        models = [m for m in models if args.filter.lower() in m.lower()]
    view = make_view(args)
    view.show_models(models, cfg.model if provider_name == cfg.provider else "")
    return 0


def cli_chat(args: argparse.Namespace) -> int:
    """Interactive session; the conversation is saved after every turn."""
    from flowai.runtime.session import SessionStore

    cfg = Settings.load()
    if not cfg.has_credentials(args.provider) and not args.provider and ui.stdin_is_terminal():
        ui.warn("FlowAI is not set up yet. Let's do that now.\n")
        setup = argparse.Namespace(provider=None, key=None, model=None, base_url=None, no_verify=False)
        if cli_configure(setup) != 0:
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

    view = make_view(args)
    agent, approver = build_agent(args, cfg, view)
    if initial:
        agent.messages = initial

    def refresh_status() -> None:
        view.status.update({
            "provider": agent.provider.provider_name,
            "model": agent.model,
            "tokens": agent.tracer.total_tokens(),
            "approvals": approver.mode,
        })

    refresh_status()
    workspace = os.getenv("FLOWAI_WORKSPACE") or os.getcwd()
    provider_name = agent.provider.provider_name
    shield_line = f"armed {_dot(view)} {key_summary(cfg, provider_name)} {_dot(view)} approvals {approver.mode}"
    rows = [
        ("model", f"{provider_name} {_arrow(view)} {agent.model}"),
        ("workspace", workspace),
        ("session", session_id + (f"  (resumed, {len(initial)} messages)" if initial else "")),
        ("shield", _styled(view, shield_line, "ok" if approver.mode == "ask" else "warn")),
    ]
    view.splash(__version__, rows, f"type a task {_dot(view)} / for commands {_dot(view)} ctrl-d to leave")
    for notice in cfg.notices:
        view.notice(notice)
    if approver.mode == "auto":
        view.notice("approvals are off (--yolo): shell and python run without asking. Calls that touch secrets still ask.")

    def handle_command(line: str) -> bool:
        """Run a slash command. Returns False when the session should end."""
        nonlocal session_id
        cmd, _, rest = line.partition(" ")
        rest = rest.strip()
        if cmd in ("/exit", "/quit"):
            return False
        if cmd == "/help":
            view.show_table(list(CHAT_COMMANDS.items()))
        elif cmd == "/model" and rest:
            agent.switch_model(rest)
            view.info(f"model {_arrow(view)} {rest} (conversation kept)")
        elif cmd == "/models":
            try:
                models = asyncio.run(agent.provider.list_models())
            except FlowAIError as exc:
                view.error(str(exc))
                return True
            if rest:
                models = [m for m in models if rest.lower() in m.lower()]
            view.show_models(models, agent.model)
        elif cmd == "/provider" and rest:
            name, _, model = rest.partition(" ")
            model = model.strip() or PROVIDER_DEFAULT_MODEL.get(name, agent.model)
            try:
                ProviderRegistry.get(name)
            except ValueError as exc:
                view.error(str(exc))
                return True
            if not cfg.has_credentials(name):
                view.error(f"no API key for '{name}'. {missing_key_help(name)}")
                return True
            new_provider = get_provider(name, model=model, **cfg.provider_options(name))
            register_secret(new_provider.api_key, new_provider.api_key_env or "api-key")
            agent.switch_model(model, provider=new_provider)
            view.info(f"provider {_arrow(view)} {name} / {model} (conversation kept)")
        elif cmd == "/tools":
            rows = []
            for t in agent.tools.list_tools():
                gate = "  [asks first]" if t.requires_approval and approver.mode == "ask" else ""
                rows.append((t.name, t.description + gate))
            view.show_table(rows)
        elif cmd == "/think":
            if rest in ("on", "off"):
                view.show_thinking = rest == "on"
                view.info(f"live reasoning {'shown' if view.show_thinking else 'folded'}")
            else:
                view.show_reasoning()
        elif cmd == "/cost":
            s = agent.tracer.summary()
            view.info(f"{s['total_tokens']} tokens, est. ${s['total_cost_usd']:.4f}")
        elif cmd == "/shield":
            view.show_report("FlowAI shield", shield_report(cfg, approver.mode))
        elif cmd == "/reset":
            agent.reset_session()
            store.save(session_id, agent.messages)
            view.context_tokens = 0
            view.info("conversation cleared.")
        elif cmd == "/new":
            agent.reset_session()
            session_id = f"ses_{uuid.uuid4().hex[:8]}"
            view.context_tokens = 0
            view.info(f"new session {session_id}")
        elif cmd == "/clear":
            print("\033[2J\033[H", end="", flush=True)
        elif cmd == "/yolo":
            approver.mode = "auto"
            view.notice("approvals off for this session. Calls that touch secrets still ask.")
        else:
            view.notice(f"unknown command '{line}'. Try /help.")
        refresh_status()
        return True

    while True:
        line = view.read()
        if line is None:
            break
        line = line.strip()
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            break
        if line.startswith("/"):
            if not handle_command(line):
                break
            continue

        try:
            asyncio.run(view.run_turn(agent, line))
        except KeyboardInterrupt:
            view.abort_turn()
        except FlowAIError as exc:
            view.error(str(exc))
            if model_unavailable(exc):
                suggest_models(view, agent.provider)
            # Drop the unanswered prompt so the next turn is not confused by it.
            if agent.messages and agent.messages[-1].role == "user":
                agent.messages.pop()
        store.save(session_id, agent.messages)
        refresh_status()

    view.info(f"session saved: {session_id}  (resume with: flowai chat --resume)")
    return 0


def _dot(view) -> str:
    return getattr(getattr(view, "g", None), "dot", "-")


def _arrow(view) -> str:
    return getattr(getattr(view, "g", None), "arrow", ">")


def _styled(view, text: str, state: str):
    """Colour a splash value when the rich interface is in use."""
    if not hasattr(view, "g"):
        return text
    from rich.text import Text

    from flowai.cli.theme import MOSS, SUN

    head, _, tail = text.partition(" ")
    return Text.assemble((head, f"bold {MOSS if state == 'ok' else SUN}"), (f" {tail}", "#cbd5e1"))


def cli_serve(args: argparse.Namespace) -> int:
    """Run the HTTP API."""
    try:
        import uvicorn

        from flowai.server.app import create_app
    except ImportError:
        ui.error("the server needs extra packages. Install them with: pip install 'flowai-cli[server]'")
        return 1

    import secrets as pysecrets

    cfg = Settings.load()
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port
    token = os.getenv("FLOWAI_SERVER_TOKEN") or pysecrets.token_urlsafe(32)
    register_secret(token, "FLOWAI_SERVER_TOKEN")
    if host not in ("127.0.0.1", "localhost", "::1"):
        ui.warn(f"Binding to {host}: anyone who can reach this port and has the token can drive an agent.")
    print(f"FlowAI server on http://{host}:{port}")
    if os.getenv("FLOWAI_SERVER_TOKEN"):
        print("Clients authenticate with the token in FLOWAI_SERVER_TOKEN.")
    else:
        print(f"Clients send:  Authorization: Bearer {token}")
        print("(Set FLOWAI_SERVER_TOKEN to choose your own token.)")
    if args.allow_shell:
        ui.warn("--allow-shell: HTTP clients may run shell and python tools on this machine.")
    app = create_app(cfg, token=token, allow_shell=args.allow_shell, local_only=host in ("127.0.0.1", "localhost", "::1"))
    uvicorn.run(app, host=host, port=port)
    return 0


def cli_inspect(args: argparse.Namespace) -> int:
    """Show saved sessions and workflow checkpoints."""
    from flowai.runtime.session import SessionStore

    cfg = Settings.load()
    print(ui.banner([f"  FlowAI {__version__}   {cfg.provider} / {cfg.model}"]))
    path = keystore.config_path()
    print(f"Config: {path if path.exists() else '(not configured: run flowai configure)'}")
    print(f"Key for {cfg.provider}: {key_summary(cfg, cfg.provider) if cfg.has_credentials() else 'MISSING'}")

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
    p.add_argument("--yolo", action="store_true", help="don't ask before shell/python tools run (secret access still asks)")
    p.add_argument("--no-tools", action="store_true", help="plain chat, no tools")
    p.add_argument("--max-iterations", type=int, help="cap on tool-calling rounds per prompt")
    p.add_argument("--plain", action="store_true", help="plain text output instead of the full interface")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flowai",
        description="FlowAI: a terminal AI agent. Run 'flowai' to chat, or 'flowai \"<prompt>\"' for one task.",
    )
    parser.add_argument("--version", action="version", version=f"flowai {__version__}")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("configure", help="choose provider, model and API key")
    p.add_argument("--provider", help="provider name")
    p.add_argument("--key", help="API key (omit to be prompted without echo; safer)")
    p.add_argument("--model", help="default model")
    p.add_argument("--base-url", help="custom endpoint, e.g. a self-hosted NVIDIA NIM")
    p.add_argument("--no-verify", action="store_true", help="skip the test request after saving")
    p.set_defaults(func=cli_configure)

    p = sub.add_parser("logout", help="delete a saved API key")
    p.add_argument("provider", nargs="?", help="provider (default: the current one)")
    p.set_defaults(func=cli_logout)

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

    p = sub.add_parser("models", help="list the models a provider serves")
    p.add_argument("provider", nargs="?", help="provider (default: the current one)")
    p.add_argument("filter", nargs="?", help="only models whose id contains this")
    p.add_argument("--plain", action="store_true", help="plain text output")
    p.set_defaults(func=cli_models)

    p = sub.add_parser("shield", help="audit how your API keys are protected")
    p.add_argument("--fix", action="store_true", help="move plain-text keys into the OS keyring, lock file permissions")
    p.add_argument("--plain", action="store_true", help="plain text output")
    p.set_defaults(func=cli_shield)

    p = sub.add_parser("serve", help="run the HTTP API (needs flowai-cli[server])")
    p.add_argument("--host", help="bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, help="port (default 8000)")
    p.add_argument("--allow-shell", action="store_true", help="let API clients use shell and python tools")
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
    args = build_parser().parse_args(normalise_argv(sys.argv[1:] if argv is None else argv))
    try:
        return args.func(args) or 0
    except FlowAIError as exc:
        ui.error(redact(str(exc)))
        return 1
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    sys.exit(main())
