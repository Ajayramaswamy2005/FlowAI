"""Qdrant Vector Memory backend."""

from __future__ import annotations

import uuid
from typing import Any

from flowai.memory.base import BaseMemory, MemoryRecord
from flowai.providers.base import BaseProvider


class QdrantMemory(BaseMemory):
    """Semantic vector memory using Qdrant."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection_name: str = "flowai_memory",
        vector_size: int = 1536,
        embedding_provider: BaseProvider | None = None,
    ):
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.embedding_provider = embedding_provider
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from qdrant_client import AsyncQdrantClient
                self._client = AsyncQdrantClient(url=self.url, api_key=self.api_key)
            except ImportError as exc:
                raise ImportError("qdrant-client is required for QdrantMemory. Install with: pip install 'flowai-cli[qdrant]'") from exc
        return self._client

    async def add(self, record: MemoryRecord | str, **metadata: Any) -> str:
        client = self._get_client()
        from qdrant_client.http import models

        rec = MemoryRecord(content=record, metadata=metadata) if isinstance(record, str) else record

        # Generate embedding vector if provider available
        vector = rec.embedding
        if not vector and self.embedding_provider:
            vecs = await self.embedding_provider.embeddings([rec.content])
            vector = vecs[0] if vecs else [0.0] * self.vector_size
        elif not vector:
            vector = [0.0] * self.vector_size

        point_id = str(uuid.uuid4())
        await client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"content": rec.content, "role": rec.role, "metadata": rec.metadata},
                )
            ],
        )
        return point_id

    async def search(self, query: str, limit: int = 5, **kwargs: Any) -> list[MemoryRecord]:
        client = self._get_client()
        query_vector = [0.0] * self.vector_size
        if self.embedding_provider:
            vecs = await self.embedding_provider.embeddings([query])
            if vecs:
                query_vector = vecs[0]

        try:
            results = await client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
            )
            records = []
            for hit in results:
                payload = hit.payload or {}
                records.append(
                    MemoryRecord(
                        id=str(hit.id),
                        content=payload.get("content", ""),
                        role=payload.get("role", "assistant"),
                        metadata=payload.get("metadata", {}),
                        score=hit.score,
                    )
                )
            return records
        except Exception:
            return []

    async def get_recent(self, limit: int = 10) -> list[MemoryRecord]:
        return await self.search(query="", limit=limit)

    async def clear(self) -> None:
        client = self._get_client()
        try:
            await client.delete_collection(self.collection_name)
        except Exception:
            pass
