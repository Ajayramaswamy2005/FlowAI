# Security

FlowAI is an agent that reads files, runs commands and fetches web pages on
your machine, using your API key. This page says what FlowAI does to protect
that key and that machine, and where the protection stops.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private vulnerability
reporting on this repository (Security tab, "Report a vulnerability"). Include
the FlowAI version (`flowai --version`), your OS, and the smallest prompt,
file or command that shows the problem.

## The threat

The usual attack is **prompt injection**. An attacker cannot type into your
session, but they can write text the agent will read: a README in a cloned
repository, a comment in a web page, an issue body, the output of an MCP tool.
That text tells the model to read your API key and send it somewhere. Others
aim for persistence (git hooks, shell startup files) or for internal services
(`localhost`, your router, cloud metadata at 169.254.169.254).

## What FlowAI does

**Keys are stored out of reach.**
- `flowai configure` saves keys in the OS credential store: Windows Credential
  Manager, the macOS Keychain, or the Secret Service on Linux. The config file
  holds only a marker.
- Without a keyring, the key goes into `~/.flowai/config.json`. The file is
  created with owner-only permissions (mode 0600 on POSIX, a user-only ACL on
  Windows) and written atomically.
- Keys are never placed in `os.environ`, so nothing FlowAI starts inherits them.
- Gemini keys travel in a header, not the URL, so they do not show up in
  error messages or proxy logs.

**Tools cannot see keys.**
- `bash`, `python_repl` and MCP servers start with every credential-looking
  environment variable removed (`*_API_KEY`, `*_TOKEN`, `*_SECRET`,
  `*PASSWORD*` and similar). `FLOWAI_PASS_ENV` lets you pass named ones through
  on purpose.
- `python_repl` runs in a separate interpreter, so it cannot import FlowAI and
  read keys from its memory.
- `grep_files` never opens files that usually hold credentials (`.env`, `*.pem`,
  `id_rsa`, `credentials.json`...). Reading one with `read_file` asks you first.

**The model never learns a key.**
- Every tool result is redacted before the model sees it. Redaction covers
  each key FlowAI holds, any variable whose name looks like a credential, and
  the shapes of common keys.
- Encoded copies are caught too: base64 at any alignment, hex, reversed and
  URL-encoded.
- So even when a model is tricked into reading a key file, it sees
  `[REDACTED:NVIDIA_API_KEY]`, and it has nothing to exfiltrate.
- Session transcripts, prompt history and error messages are redacted the same way.

**Dangerous calls are stopped or confirmed.**
- A tool call whose arguments contain a known key is refused.
- A command that reads secrets and sends data over the network in the same
  call is refused (`cat .env | curl -d @- ...`).
- Commands that dump the environment, read credential files or stores, or
  touch `~/.flowai` are rated critical. They ask you every time, also in
  `--yolo` mode, and "always allow" is not offered for them.
- Uploads, remote scripts (`curl ... | sh`), destructive commands, privilege
  escalation and persistence (git hooks, cron, shell profiles, the registry)
  are rated high. The approval gate names the risk in plain words.
- With no terminal attached, nothing that needs approval runs.

**The network is fenced.**
- `fetch_url` only reaches public addresses. It resolves every host and
  re-checks every redirect, refusing loopback, private ranges, link-local and
  metadata addresses, and URLs with embedded credentials.
- Output from `fetch_url`, `search_web` and MCP tools is wrapped as untrusted
  content. The system prompt tells the model never to follow instructions
  found inside it.

**The terminal is protected.** Escape sequences and control characters are
stripped from model and tool output before they are drawn. A fetched page
cannot set your clipboard (OSC 52), retitle the window or hide text. Markdown
links are shown as text and URL, never as disguised hyperlinks.

**The HTTP server is closed by default.** `flowai serve`:
- binds to 127.0.0.1 and requires `Authorization: Bearer <token>` on every
  route except `/health`
- rejects requests whose `Host` header is not local, which defeats DNS rebinding
- refuses shell and python tools unless started with `--allow-shell`
- never runs calls that touch secrets on behalf of an HTTP client

## What FlowAI does not do

- **It is not a sandbox.** When you approve a shell command, it runs with your
  user's full permissions. The risk rules are pattern-based tripwires: a
  determined attacker can write a command they do not recognise. Read what the
  gate shows you, and keep `--yolo` for throwaway environments.
- Redaction only knows the keys FlowAI holds and the formats it recognises. A
  password typed into a file the agent reads, in a format it cannot recognise,
  will reach the model.
- Name resolution is checked before connecting, but a hostile DNS server can
  still answer differently on the second lookup (a rebinding race).
- MCP servers are programs you chose to run. FlowAI scrubs their environment
  and asks before each of their tool calls, but cannot see what they do inside.
- A key you export in your shell is visible to any program you run in that
  shell, FlowAI or not. Prefer `flowai configure`.

## Checking your setup

```bash
flowai shield          # what protects your keys right now
flowai shield --fix    # move plain-text keys into the keyring, lock permissions
flowai logout nvidia   # delete a saved key
```
