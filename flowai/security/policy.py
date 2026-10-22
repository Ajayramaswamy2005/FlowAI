"""Rules for judging what a tool call is about to do.

Nothing here runs anything. Each check reads a command, a piece of code, a
path or a URL and says how risky it looks and why. The Shield turns those
judgements into allow, ask or block.

These are tripwires, not a sandbox: a determined attacker with shell access
can always write a command no pattern anticipates. They exist so that the
common attacks (dump the environment, read the key file, post it somewhere)
are caught before they run, and so that a person approving a command is told
what it touches in plain words.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from flowai.security.keystore import flowai_home

LEVELS = ("low", "medium", "high", "critical")


@dataclass
class Risk:
    level: str = "low"
    reasons: list[str] = field(default_factory=list)
    #: Too dangerous to run even with the user's approval.
    blocked: bool = False

    def raise_to(self, level: str, reason: str) -> None:
        if LEVELS.index(level) > LEVELS.index(self.level):
            self.level = level
        if reason not in self.reasons:
            self.reasons.append(reason)

    def merge(self, other: Risk) -> None:
        for reason in other.reasons:
            self.raise_to(other.level, reason)
        self.blocked = self.blocked or other.blocked

    @property
    def needs_confirmation(self) -> bool:
        return self.level in ("high", "critical")


def _rules(*pairs: tuple[str, str]) -> list[tuple[re.Pattern[str], str]]:
    return [(re.compile(p, re.IGNORECASE), why) for p, why in pairs]


# The start of a command segment: line start, after ; & | ( or a backtick, or inside $( ).
_SEG = r"(?:^|[;&|(`]\s*|\$\(\s*)"

SECRET_ACCESS = _rules(
    (_SEG + r"(?:env|printenv|export\s+-p|declare\s+-x)\s*(?:$|[|;&>)])", "prints every environment variable"),
    (_SEG + r"set\s*(?:$|[|;&>])", "prints every environment variable"),
    (r"\b(?:Get-ChildItem|gci|ls|dir)\s+env:", "lists environment variables"),
    (r"\[(?:System\.)?Environment\]::GetEnvironmentVariables?", "reads environment variables"),
    (r"\$\{?\w*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)\w*\}?", "reads a credential variable"),
    (r"\$env:\w*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", "reads a credential variable"),
    (r"%\w*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)\w*%", "reads a credential variable"),
    (r"\bprintenv\s+\w+", "prints an environment variable"),
    (r"os\.environ|getenv\s*\(", "reads environment variables"),
    (r"/proc/(?:self|\d+)/environ", "reads a process environment"),
    (r"(?:^|[\s/\\'\"=<@])\.env(?:\.(?!example|sample|template|dist\b)\w+)?(?=$|[\s'\";|&>)])",
     "touches a .env secrets file"),
    (r"\.flowai(?:[/\\]|\s|$)", "touches FlowAI's own key store"),
    (r"(?:^|[\s/\\~])\.(?:ssh|aws|gnupg|azure|kube|docker)(?:[/\\]|\s|$)", "touches a credentials directory"),
    (r"\bid_(?:rsa|dsa|ecdsa|ed25519)\b", "touches an SSH private key"),
    (r"\.(?:netrc|npmrc|pypirc|git-credentials)\b|_netrc\b", "touches a credentials file"),
    (r"\b(?:cmdkey|vaultcmd)\b|Get-StoredCredential|security\s+find-(?:generic|internet)-password"
     r"|secret-tool\s+lookup|\bkeyring\s+get\b", "reads the system credential store"),
    (r"git\s+credential\s+fill|git\s+config\s+.*credential", "reads git credentials"),
)

EXFILTRATION = _rules(
    (r"\bcurl\b[^|;&]*\s(?:-d|--data\S*|-F|--form|-T|--upload-file|-X\s*(?:POST|PUT|PATCH))\b",
     "uploads data with curl"),
    (r"\bwget\b[^|;&]*--post-(?:data|file)", "uploads data with wget"),
    (r"\b(?:Invoke-WebRequest|Invoke-RestMethod|iwr|irm)\b[^|;&]*-(?:Method\s+(?:Post|Put|Patch)|Body|InFile)",
     "uploads data from PowerShell"),
    (r"\b(?:nc|ncat|netcat|socat|telnet)\b\s+\S", "opens a raw network connection"),
    (r"\b(?:scp|sftp)\b|\brsync\b[^|;&]*\S+:", "copies files to another machine"),
    (r"/dev/tcp/|/dev/udp/", "opens a raw network connection"),
    (r"\b(?:requests|httpx)\.(?:post|put|patch)\b|urlopen\([^)]*data=", "sends data over HTTP"),
)

REMOTE_CODE = _rules(
    (r"\b(?:curl|wget|iwr|irm|Invoke-WebRequest|Invoke-RestMethod)\b[^;&]*\|\s*(?:sudo\s+)?"
     r"(?:sh|bash|zsh|python\d?|perl|ruby|node|iex|Invoke-Expression|powershell|pwsh)\b",
     "runs code downloaded from the internet"),
    (r"\b(?:iex|Invoke-Expression)\b", "evaluates a string as PowerShell code"),
    (r"\b(?:powershell|pwsh)(?:\.exe)?\b[^|;&]*\s-e(?:nc|ncodedcommand)?\b", "runs an encoded PowerShell command"),
    (r"base64\s+(?:-d|--decode)[^|;&]*\|\s*(?:sh|bash|zsh|python)", "runs decoded hidden code"),
)

DESTRUCTIVE = _rules(
    (r"\brm\s+(?:-\w*\s+)*-\w*[rR]\w*\s+(?:-\w+\s+)*(?:/|~|\$HOME|\*|\.\.?)(?:\s|$|/\*)",
     "deletes a whole directory tree"),
    (r"\bRemove-Item\b[^|;&]*-Recurse", "deletes a directory tree"),
    (r"\b(?:rd|rmdir)\s+/s\b|\bdel\s+(?:/\w\s+)*/s\b", "deletes a directory tree"),
    (r"\bmkfs(?:\.\w+)?\b|\bformat\s+[a-z]:", "formats a disk"),
    (r"\bdd\s+[^|;&]*of=/dev/", "writes directly to a device"),
    (r":\(\)\s*\{\s*:\|:&\s*\};:", "fork bomb"),
    (r"\bgit\s+push\b[^|;&]*(?:--force\b|\s-f\b)", "force-pushes git history"),
    (r"\bgit\s+(?:reset\s+--hard|clean\s+-\w*f)", "discards uncommitted work"),
    (r"\bchmod\s+-R\s+0?777\b", "makes files writable by everyone"),
    (r"\b(?:shutdown|reboot|halt|poweroff)\b|Stop-Computer|Restart-Computer", "shuts the machine down"),
)

PERSISTENCE = _rules(
    (r"\bsudo\b|\brunas\b|Start-Process\b[^|;&]*-Verb\s+RunAs", "asks for administrator rights"),
    (r"\bcrontab\b|\bschtasks\b[^|;&]*/create|Register-ScheduledTask", "schedules a recurring task"),
    (r"\breg(?:\.exe)?\s+(?:add|delete)\b|New-ItemProperty\b[^|;&]*HK(?:LM|CU)", "changes the Windows registry"),
    (r"Set-ExecutionPolicy", "changes PowerShell's script policy"),
    (r"\.(?:bashrc|zshrc|bash_profile|profile)\b|Microsoft\.PowerShell_profile", "edits a shell startup file"),
    (r"\.git[/\\]hooks\b", "edits git hooks, which run code automatically"),
    (r"\bgit\s+config\s+--global\b", "changes global git settings"),
)

NETWORK_OR_INSTALL = _rules(
    # The lookbehind keeps paths such as ~/.ssh from counting as the ssh command.
    (r"(?<![\w./\\-])(?:curl|wget|iwr|irm|Invoke-WebRequest|Invoke-RestMethod|ssh|ftp)\b", "uses the network"),
    (r"\b(?:pip|pip3|uv|npm|pnpm|yarn|gem|cargo|go)\s+(?:install|add)\b", "installs packages"),
    (r"\bgit\s+(?:clone|fetch|pull|push)\b", "talks to a git remote"),
    (r"\bbase64\b|\bxxd\b|\bcertutil\b", "encodes or decodes data"),
    (r"\beval\b", "evaluates a string as code"),
)


def _apply(text: str, rules, level: str, risk: Risk) -> bool:
    hit = False
    for pattern, why in rules:
        if pattern.search(text):
            risk.raise_to(level, why)
            hit = True
    return hit


def assess_command(command: str) -> Risk:
    """Judge a shell command."""
    risk = Risk()
    secrets = _apply(command, SECRET_ACCESS, "critical", risk)
    sends = _apply(command, EXFILTRATION, "high", risk)
    _apply(command, REMOTE_CODE, "high", risk)
    _apply(command, DESTRUCTIVE, "high", risk)
    _apply(command, PERSISTENCE, "high", risk)
    network = _apply(command, NETWORK_OR_INSTALL, "medium", Risk())
    if not risk.reasons:
        _apply(command, NETWORK_OR_INSTALL, "medium", risk)
    if secrets and (sends or network):
        risk.raise_to("critical", "reads secrets and sends data over the network in one go")
        risk.blocked = True
    return risk


PYTHON_RULES_CRITICAL = _rules(
    (r"os\.environ|os\.getenv|getenv\s*\(|environ\.get", "reads environment variables"),
    (r"""['"][^'"]*(?:\.env|\.flowai|\.ssh|\.aws|id_rsa|id_ed25519|\.netrc|\.git-credentials|\.pypirc|\.npmrc)\b""",
     "opens a secrets file"),
    (r"\bkeyring\b", "reads the system credential store"),
)
PYTHON_RULES_HIGH = _rules(
    (r"\bsubprocess\b|os\.system|os\.popen|os\.exec|pty\.spawn", "starts other programs"),
    (r"shutil\.rmtree|os\.remove|os\.unlink|\.unlink\(", "deletes files"),
    (r"\bctypes\b", "calls native code"),
    (r"\b(?:exec|eval|compile)\s*\(", "evaluates generated code"),
)
PYTHON_RULES_NETWORK = _rules(
    (r"\b(?:requests|httpx|urllib|aiohttp|socket|http\.client|ftplib|smtplib)\b", "uses the network"),
)


def assess_python(code: str) -> Risk:
    """Judge a block of Python before it runs."""
    risk = Risk()
    secrets = _apply(code, PYTHON_RULES_CRITICAL, "critical", risk)
    _apply(code, PYTHON_RULES_HIGH, "high", risk)
    network = _apply(code, PYTHON_RULES_NETWORK, "medium", risk)
    _apply(code, EXFILTRATION, "high", risk)
    if secrets and network:
        risk.raise_to("critical", "reads secrets and sends data over the network in one go")
        risk.blocked = True
    return risk


# ---------------------------------------------------------------------- paths

SENSITIVE_NAMES = {
    ".netrc", "_netrc", ".npmrc", ".pypirc", ".git-credentials", ".pgpass",
    "credentials", "credentials.json", "credentials.yaml", "credentials.yml",
    "secrets.json", "secrets.yaml", "secrets.yml", "secrets.toml",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
    "terraform.tfstate", "terraform.tfstate.backup",
}
SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".keystore", ".jks", ".kdbx", ".ppk", ".tfstate"}
SENSITIVE_DIRS = {".ssh", ".aws", ".gnupg", ".azure", ".kube", ".docker"}
SAFE_ENV_SUFFIXES = (".example", ".sample", ".template", ".dist", ".defaults")


def sensitive_path_reason(path: Path) -> str | None:
    """Why a path probably holds credentials, or None."""
    name = path.name.lower()
    if name == ".env" or (name.startswith(".env.") and not name.endswith(SAFE_ENV_SUFFIXES)):
        return f"{path.name} is an environment file, where API keys usually live"
    if name in SENSITIVE_NAMES or (name.startswith("service-account") and name.endswith(".json")):
        return f"{path.name} is a credentials file"
    if path.suffix.lower() in SENSITIVE_SUFFIXES:
        return f"{path.name} looks like a private key or certificate store"
    if {p.lower() for p in path.parts} & SENSITIVE_DIRS:
        return f"{path} is inside a credentials directory"
    try:
        home = flowai_home().resolve()
        resolved = path.resolve()
        if resolved == home or home in resolved.parents:
            return "this is FlowAI's own key store"
    except OSError:
        pass
    return None


def write_path_reason(path: Path) -> str | None:
    """Why writing to a path deserves a second look, or None."""
    if ".git" in [p.lower() for p in path.parts]:
        return "writes inside .git, where hooks run code automatically"
    if path.name.lower() in (".bashrc", ".zshrc", ".profile", ".bash_profile"):
        return "edits a shell startup file"
    return sensitive_path_reason(path)


# ------------------------------------------------------------------- network


def blocked_address_reason(host: str) -> str | None:
    """Why an IP address must not be fetched (loopback, private, metadata), or None."""
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return None
    if ip.is_loopback:
        return "it points at this machine (loopback)"
    if ip.is_link_local:
        return "it is a link-local address, where cloud metadata services hand out credentials"
    if ip.is_private or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
        return "it is a private network address"
    return None


def url_problem(url: str) -> str | None:
    """Reject URLs that are not plain public http(s)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "only http:// and https:// URLs can be fetched"
    host = (parsed.hostname or "").lower()
    if not host:
        return "the URL has no host"
    if parsed.username or parsed.password:
        return "the URL carries credentials"
    if host in ("localhost", "metadata.google.internal") or host.endswith((".localhost", ".internal", ".local")):
        return "it points at this machine or an internal host"
    return blocked_address_reason(host)
