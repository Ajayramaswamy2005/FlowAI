"""Where API keys live: the OS keyring when there is one, else a locked-down file.

`flowai configure` stores a key in the operating system's credential store
(Windows Credential Manager, macOS Keychain, the Secret Service on Linux) and
writes only a marker into ~/.flowai/config.json. Without a usable keyring the
key goes into that file, which is created readable by the current user alone.

Keys are never copied into os.environ, so tools and the processes they start
cannot read them from there. Providers ask `resolve_api_key()` instead.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

KEYRING_SERVICE = "flowai"
#: Written in config.json in place of a key that lives in the keyring.
KEYRING_MARKER = "@keyring"


def flowai_home() -> Path:
    """~/.flowai, or FLOWAI_HOME when set (tests and portable installs use it)."""
    return Path(os.getenv("FLOWAI_HOME") or Path.home() / ".flowai")


def config_path() -> Path:
    return flowai_home() / "config.json"


# ----------------------------------------------------------------- file safety


def harden_permissions(path: Path, directory: bool = False) -> None:
    """Make a file or directory accessible to the current user only (best effort)."""
    try:
        if sys.platform == "win32":
            user = os.getenv("USERNAME")
            if user:
                grant = f"{user}:(OI)(CI)F" if directory else f"{user}:F"
                subprocess.run(
                    ["icacls", str(path), "/inheritance:r", "/grant:r", grant],
                    capture_output=True, check=False, timeout=10,
                )
        else:
            os.chmod(path, 0o700 if directory else 0o600)
    except (OSError, subprocess.SubprocessError):
        pass


def ensure_home() -> Path:
    home = flowai_home()
    fresh = not home.exists()
    home.mkdir(parents=True, exist_ok=True)
    if fresh:
        harden_permissions(home, directory=True)
    return home


def secure_write(path: Path, text: str, harden: bool = True) -> None:
    """Write a file that only the current user can read, atomically.

    harden=False skips the Windows ACL call for files inside a directory that
    was already locked down, since they inherit its permissions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(tmp, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    if harden:
        harden_permissions(tmp)
    # On Windows a virus scanner or indexer can hold a just-written file for a
    # moment, and the rename fails with PermissionError. Wait it out briefly.
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * (attempt + 1))


def permission_problem(path: Path) -> str | None:
    """Why other users could read this file, or None when they cannot (POSIX only)."""
    if sys.platform == "win32" or not path.exists():
        return None
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        return f"{path} is readable by other users (mode {oct(mode)}); run 'flowai shield --fix'."
    return None


def read_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_config(data: dict[str, Any]) -> Path:
    ensure_home()
    secure_write(config_path(), json.dumps(data, indent=2))
    return config_path()


# --------------------------------------------------------------------- keyring


def _keyring():
    """The keyring module if a real backend is usable here, else None."""
    if os.getenv("FLOWAI_KEYRING", "1").lower() in ("0", "false", "no", "off"):
        return None
    try:
        import keyring
        from keyring.backends import fail
    except Exception:
        return None
    try:
        backend = keyring.get_keyring()
    except Exception:
        return None
    if isinstance(backend, fail.Keyring) or getattr(backend, "priority", 1) <= 0:
        return None
    return keyring


def keyring_name() -> str | None:
    """Human name of the credential store in use, or None without one."""
    kr = _keyring()
    if not kr:
        return None
    backend = type(kr.get_keyring()).__module__
    if "Windows" in backend or sys.platform == "win32":
        return "Windows Credential Manager"
    if "macOS" in backend or sys.platform == "darwin":
        return "macOS Keychain"
    return "system keyring"


def store_key(env_name: str, value: str) -> str:
    """Save a key; return where it went ("keyring" or "file")."""
    config = read_config()
    kr = _keyring()
    if kr:
        try:
            kr.set_password(KEYRING_SERVICE, env_name, value)
            config[env_name] = KEYRING_MARKER
            write_config(config)
            return "keyring"
        except Exception:
            pass
    config[env_name] = value
    write_config(config)
    return "file"


def forget_key(env_name: str) -> bool:
    """Remove a stored key from the keyring and the config file."""
    config = read_config()
    removed = config.pop(env_name, None) is not None
    kr = _keyring()
    if kr:
        try:
            kr.delete_password(KEYRING_SERVICE, env_name)
            removed = True
        except Exception:
            pass
    if removed:
        write_config(config)
    return removed


def stored_key(env_name: str) -> str | None:
    """The key saved by `flowai configure`, wherever it lives."""
    value = read_config().get(env_name)
    if value == KEYRING_MARKER:
        kr = _keyring()
        if not kr:
            return None
        try:
            return kr.get_password(KEYRING_SERVICE, env_name)
        except Exception:
            return None
    return value if isinstance(value, str) and value else None


def key_location(env_name: str) -> str | None:
    """'environment', 'keyring', 'file' or None: where a key would be found."""
    if os.getenv(env_name):
        return "environment"
    value = read_config().get(env_name)
    if value == KEYRING_MARKER:
        return "keyring" if stored_key(env_name) else None
    return "file" if value else None


def resolve_api_key(*env_names: str) -> str:
    """First key found for any of these names: environment, then saved config."""
    for name in env_names:
        if os.getenv(name):
            return os.environ[name]
    for name in env_names:
        value = stored_key(name)
        if value:
            return value
    return ""


def migrate_file_keys_to_keyring() -> list[str]:
    """Move plain-text keys from config.json into the keyring. Returns moved names."""
    kr = _keyring()
    if not kr:
        return []
    config = read_config()
    moved = []
    for name, value in list(config.items()):
        if name.endswith(("_API_KEY", "_KEY")) and isinstance(value, str) and value and value != KEYRING_MARKER:
            try:
                kr.set_password(KEYRING_SERVICE, name, value)
            except Exception:
                continue
            config[name] = KEYRING_MARKER
            moved.append(name)
    if moved:
        write_config(config)
    return moved
