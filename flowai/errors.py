"""Error types FlowAI raises, so callers can tell apart what went wrong.

The CLI prints these as a single clear line instead of a traceback: a missing
API key is a setup problem the user can fix, not a crash.
"""

from __future__ import annotations


class FlowAIError(Exception):
    """Base class for every error FlowAI raises on purpose."""


class ConfigurationError(FlowAIError):
    """Something the user must set is missing or wrong (API key, provider name)."""


class ProviderError(FlowAIError):
    """An LLM provider rejected the request or returned an error response."""

    def __init__(self, provider: str, status_code: int | None, message: str):
        self.provider = provider
        self.status_code = status_code
        self.message = message
        where = f"{provider} API" if status_code is None else f"{provider} API (HTTP {status_code})"
        super().__init__(f"{where}: {message}")


class ToolError(FlowAIError):
    """A tool could not be run at all (as opposed to running and failing)."""
