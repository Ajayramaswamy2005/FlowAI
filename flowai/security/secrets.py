"""Keeping secrets out of everything the model sees, says and runs.

Three jobs live here:

- `Redactor` knows the actual key values this process holds, plus the shapes
  of common provider keys, and replaces them with a marker. Every tool result
  passes through it before the model reads it, so a model tricked into reading
  a key file still never learns the key. It also catches the encodings an
  injected prompt would ask for first: base64, hex, reversed and URL-encoded.
- `scrubbed_env()` is the environment child processes get: the parent's minus
  every variable that looks like a credential.
- `sanitize_terminal()` strips escape sequences from text before it is drawn,
  so a fetched web page cannot retitle the window, rewrite the clipboard
  (OSC 52) or hide text from the user.
"""

from __future__ import annotations

import base64
import os
import re
from urllib.parse import quote

#: Shapes of well-known credentials. Order matters: specific prefixes first.
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("nvidia-key", re.compile(r"nvapi-[A-Za-z0-9_\-]{20,}")),
    ("anthropic-key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openrouter-key", re.compile(r"sk-or-v1-[A-Za-z0-9]{20,}")),
    ("openai-key", re.compile(r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}")),
    ("google-key", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("github-token", re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{50,}")),
    ("aws-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("huggingface-token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("groq-key", re.compile(r"\bgsk_[A-Za-z0-9]{40,}")),
    ("slack-token", re.compile(r"xox[abprs]-[A-Za-z0-9\-]{10,}")),
    ("private-key", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
    )),
]

#: Environment variable names that hold credentials.
SENSITIVE_ENV_RE = re.compile(
    r"(?:^|_)(?:API_?KEY|KEY|TOKEN|SECRET|PASSWORD|PASSWD|PASS|PASSPHRASE|CREDENTIALS?|PAT)(?:_|$)",
    re.IGNORECASE,
)
SENSITIVE_ENV_NAMES = {"PGPASSWORD", "NVAPI_KEY", "AWS_SESSION_TOKEN", "DATABASE_URL"}

#: Values shorter than this are too likely to collide with ordinary text.
MIN_SECRET_LENGTH = 8


def is_sensitive_env_name(name: str) -> bool:
    if name.startswith("FLOWAI_") and name not in ("FLOWAI_SERVER_TOKEN",):
        return False
    return name.upper() in SENSITIVE_ENV_NAMES or bool(SENSITIVE_ENV_RE.search(name))


def _variants(value: str) -> set[str]:
    """The forms a secret takes when someone tries to smuggle it out encoded."""
    raw = value.encode()
    found = {value, value[::-1], raw.hex(), quote(value, safe="")}
    for encode in (base64.b64encode, base64.urlsafe_b64encode):
        for offset in range(3):
            # A secret inside a larger base64 blob encodes differently depending
            # on its byte alignment; the middle is stable for each alignment.
            encoded = encode(b"\0" * offset + raw).decode().rstrip("=")
            middle = encoded[4 if offset else 0: -4]
            if len(middle) >= 12:
                found.add(middle)
    return {v for v in found if len(v) >= MIN_SECRET_LENGTH}


class Redactor:
    """Replaces known secret values, their encodings and key-shaped strings."""

    def __init__(self) -> None:
        self._known: dict[str, str] = {}  # variant -> label
        self._pattern: re.Pattern[str] | None = None

    def add(self, value: str | None, label: str = "secret") -> None:
        if not value or len(value) < MIN_SECRET_LENGTH:
            return
        for variant in _variants(value.strip()):
            self._known.setdefault(variant, label)
        # Longest first, so a full key wins over a variant inside it.
        alternatives = sorted(self._known, key=len, reverse=True)
        self._pattern = re.compile("|".join(re.escape(a) for a in alternatives))

    def known_count(self) -> int:
        return len({label for label in self._known.values()})

    def find(self, text: str) -> list[str]:
        """Labels of every secret present in text (empty when it is clean)."""
        if not text:
            return []
        labels: list[str] = []
        if self._pattern:
            labels += [self._known[m.group(0)] for m in self._pattern.finditer(text)]
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                labels.append(label)
        return sorted(set(labels))

    def redact(self, text: str) -> str:
        if not text:
            return text
        if self._pattern:
            text = self._pattern.sub(lambda m: f"[REDACTED:{self._known[m.group(0)]}]", text)
        for label, pattern in SECRET_PATTERNS:
            text = pattern.sub(f"[REDACTED:{label}]", text)
        return text


#: The process-wide redactor. Settings registers every key it resolves here.
REDACTOR = Redactor()


def register_secret(value: str | None, label: str = "secret") -> None:
    REDACTOR.add(value, label)


def register_environment_secrets(environ: dict[str, str] | None = None) -> None:
    """Teach the redactor every credential already sitting in the environment."""
    for name, value in (environ if environ is not None else os.environ).items():
        if is_sensitive_env_name(name):
            REDACTOR.add(value, name)


def redact(text: str) -> str:
    return REDACTOR.redact(text)


def scrubbed_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment for a child process: the parent's minus its credentials.

    FLOWAI_PASS_ENV=NAME1,NAME2 lets a user hand specific variables through on
    purpose (a GITHUB_TOKEN for `gh`, say). `extra` is added afterwards, so a
    configured MCP server still receives the variables it was given.
    """
    allowed = {n.strip() for n in os.getenv("FLOWAI_PASS_ENV", "").split(",") if n.strip()}
    env = {k: v for k, v in os.environ.items() if k in allowed or not is_sensitive_env_name(k)}
    env.update(extra or {})
    return env


_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?")
_ESC_RE = re.compile(r"\x1b[@-Z\\\]^_]?")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\u0080-\u009f]")


def sanitize_terminal(text: str) -> str:
    """Drop escape sequences and control characters; keep newlines and tabs."""
    if not text:
        return text
    text = text.replace("\r\n", "\n")
    text = _OSC_RE.sub("", text)
    text = _CSI_RE.sub("", text)
    text = _ESC_RE.sub("", text)
    return _CONTROL_RE.sub("", text)
