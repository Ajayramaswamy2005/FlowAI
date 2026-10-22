"""NVIDIA: the hosted API catalog (build.nvidia.com) and self-hosted NIM servers.

Both speak the OpenAI chat-completions protocol, so this is the OpenAI
provider pointed at NVIDIA, plus what makes NVIDIA's keys and models work:

- Keys are read from NVIDIA_API_KEY, NVAPI_KEY, NGC_API_KEY or NIM_API_KEY,
  whichever is set, or from `flowai configure`.
- NVIDIA_BASE_URL (or NIM_BASE_URL) points at a NIM you run yourself, e.g.
  http://localhost:8000/v1. Self-hosted NIMs need no key.
- Catalog models differ in what they accept. The OpenAI provider's fallback
  drops `tools`, `stream_options` or the system role when a model rejects
  them, so any chat model on the catalog can be used, not only tool-calling ones.
- Reasoning models (DeepSeek-R1, QwQ, Nemotron) stream their thinking
  separately from the answer.
- Errors say what to do: a key that is not an `nvapi-` key, a model the
  catalog does not have, or the free tier's rate limit.
"""

from __future__ import annotations

from typing import Any

import httpx

from flowai.errors import ProviderError
from flowai.providers.openai import OpenAIProvider

HOSTED_BASE_URL = "https://integrate.api.nvidia.com/v1"
KEY_ENV_NAMES = ("NVIDIA_API_KEY", "NVAPI_KEY", "NGC_API_KEY", "NIM_API_KEY")

#: Chat models that called tools correctly through FlowAI on the hosted
#: catalog (checked 2026-10). The catalog changes often and not every account
#: can call every model, so these are tried in order when one is unavailable.
RECOMMENDED_MODELS = (
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "openai/gpt-oss-20b",
    "nvidia/nemotron-nano-3-30b-a3b",
    "deepseek-ai/deepseek-v4.1-flash",
)
KEY_PREFIX = "nvapi-"
KEY_HELP = "Create one at https://build.nvidia.com: open any model and choose 'Get API Key'."


class NvidiaProvider(OpenAIProvider):
    """NVIDIA API catalog and NIM provider (Llama, DeepSeek, Nemotron, Qwen, Mistral...)."""

    DEFAULT_BASE_URL = HOSTED_BASE_URL
    DEFAULT_MODEL = RECOMMENDED_MODELS[0]
    api_key_env = "NVIDIA_API_KEY"
    key_env_names = KEY_ENV_NAMES
    base_url_env_names = ("NVIDIA_BASE_URL", "NIM_BASE_URL")
    # The catalog's server-side default is short enough to cut answers off mid-sentence.
    default_max_tokens = 4096

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        **kwargs: Any,
    ):
        # Reasoning models can think for a long time before their first token.
        super().__init__(model=model, api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)
        self.self_hosted = "api.nvidia.com" not in (self.base_url or "")
        self.requires_api_key = not self.self_hosted

    def key_problem(self) -> str | None:
        """Why this key will not work on the hosted catalog, judged without a request."""
        if self.self_hosted or not self.api_key:
            return None
        key = self.api_key.strip()
        if key != self.api_key:
            return "the key has spaces or a newline around it; paste it again without them."
        if not key.startswith(KEY_PREFIX):
            return (
                f"the key does not start with '{KEY_PREFIX}'. The hosted API catalog only accepts "
                f"nvapi- keys; an older NGC key cannot call it. {KEY_HELP}"
            )
        return None

    async def _recover(self, response: httpx.Response, payload: dict[str, Any], retries: int) -> str:
        # Catalog ids carry an owner prefix ("z-ai/glm-5.3-flash"). People type the
        # bare name; when exactly one catalog model ends with it, use that one.
        if response.status_code == 404 and "/" not in self.model and not self.self_hosted:
            if not response.is_closed:
                await response.aread()
            try:
                matches = [m for m in await self.list_models() if m.endswith("/" + self.model)]
            except ProviderError:
                matches = []
            if len(matches) == 1:
                self._notices.append(f"'{self.model}' is listed as '{matches[0]}' on NVIDIA's catalog; using that.")
                self.model = matches[0]
                return "degrade"
        return await super()._recover(response, payload, retries)

    def _explain_error(self, error: ProviderError) -> ProviderError:
        status = error.status_code
        hint = ""
        if status in (401, 403):
            hint = self.key_problem() or (
                "NVIDIA did not accept this key. It may be expired or revoked, or lack access to this model. "
                + KEY_HELP
            )
        elif status == 404:
            hint = (
                f"'{self.model}' is not available to this key; the catalog lists models that not every "
                "account can call. Pick another: flowai models nvidia"
            )
        elif status == 410:
            hint = f"NVIDIA has retired '{self.model}'. Pick a current model: flowai models nvidia"
        elif status == 402:
            hint = "Your NVIDIA account is out of credits for this model."
        elif status == 429:
            hint = "NVIDIA's rate limit was hit and retries did not clear it; wait a minute and try again."
        if not hint:
            return error
        return ProviderError(error.provider, status, f"{error.message}\n  {hint}")

    async def list_models(self) -> list[str]:
        try:
            return await super().list_models()
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc

    async def suggest_models(self, limit: int = 5) -> list[str]:
        """Models worth trying instead of an unavailable one: close names first, then known-good ones."""
        import difflib

        try:
            catalog = await self.list_models()
        except ProviderError:
            return [m for m in RECOMMENDED_MODELS if m != self.model][:limit]
        chat = [m for m in catalog if not any(w in m for w in ("embed", "guard", "safety", "reward", "parse", "clip"))]
        close = difflib.get_close_matches(self.model, chat, n=3, cutoff=0.5)
        known = [m for m in RECOMMENDED_MODELS if m in catalog]
        ordered = [m for m in [*close, *known] if m != self.model]
        return list(dict.fromkeys(ordered))[:limit]

    async def embeddings(
        self,
        texts: list[str],
        model: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        url = f"{self.base_url.rstrip('/')}/embeddings"
        payload = {
            "input": texts,
            "model": model or "nvidia/nv-embedqa-e5-v5",
            # NVIDIA's retrieval embedders embed queries and passages differently.
            "input_type": kwargs.get("input_type", "query"),
            "encoding_format": "float",
            "truncate": "END",
        }
        self.require_api_key()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=self._get_headers(), json=payload)
            await self._raise_for_status(resp)
            data = resp.json()
        return [item["embedding"] for item in data.get("data", [])]
