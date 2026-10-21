"""NVIDIA API Catalog Provider implementation.

The catalog speaks the OpenAI chat-completions protocol, so this provider is
the OpenAI one pointed at NVIDIA's endpoint. Only the embeddings call differs:
NVIDIA's models require an input_type.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from flowai.providers.openai import OpenAIProvider


class NvidiaProvider(OpenAIProvider):
    """NVIDIA API Catalog Provider for models like DeepSeek-R1 and Llama 3.1."""

    DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
    api_key_env = "NVIDIA_API_KEY"

    def __init__(
        self,
        model: str = "meta/llama-3.1-70b-instruct",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            api_key=api_key or os.getenv("NVIDIA_API_KEY", ""),
            base_url=base_url or os.getenv("NVIDIA_BASE_URL", self.DEFAULT_BASE_URL),
            timeout=timeout,
            **kwargs,
        )

    async def embeddings(
        self,
        texts: list[str],
        model: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        url = f"{self.base_url.rstrip('/')}/embeddings"
        payload = {
            "input": texts,
            "model": model or "nvidia/nv-embed-v1",
            "input_type": kwargs.get("input_type", "query"),
        }
        self.require_api_key()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=self._get_headers(), json=payload)
            await self._raise_for_status(resp)
            data = resp.json()
        return [item["embedding"] for item in data.get("data", [])]
