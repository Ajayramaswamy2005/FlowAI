"""ChromaDB memory backend."""

from __future__ import annotations

import uuid
from typing import Any

from flowai.memory.base import BaseMemory, MemoryRecord
from flowai.providers.base import BaseProvider


class ChromaMemory(BaseMemory):
    """Vector memory backed by ChromaDB."""

    def __init__(
        self,
        collection_name: str = "flowai_memory",
        persist_directory: str | None = None,
        embedding_provider: BaseProvider | None = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_provider = embedding_provider
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            try:
                import chromadb
                if self.persist_directory:
                    client = chromadb.PersistentClient(path=self.persist_directory)
                else:
                    client = chromadb.Client()
                self._collection = client.get_or_create_collection(name=self.collection_name)
            except ImportError as exc:
                raise ImportError("chromadb is required for ChromaMemory. Install with: pip install 'flowai-cli[chroma]'") from exc
        return self._collection

    async def add(self, record: MemoryRecord | str, **metadata: Any) -> str:
        col = self._get_collection()
        rec = MemoryRecord(content=record, metadata=metadata) if isinstance(record, str) else record
        rec_id = rec.id or str(uuid.uuid4())

        embeddings = None
        if self.embedding_provider:
            vecs = await self.embedding_provider.embeddings([rec.content])
            if vecs:
                embeddings = vecs

        col.add(
            ids=[rec_id],
            documents=[rec.content],
            metadatas=[{"role": rec.role, **rec.metadata}],
            embeddings=embeddings,
        )
        return rec_id

    async def search(self, query: str, limit: int = 5, **kwargs: Any) -> list[MemoryRecord]:
        col = self._get_collection()
        query_embeddings = None
        if self.embedding_provider:
            vecs = await self.embedding_provider.embeddings([query])
            if vecs:
                query_embeddings = vecs

        res = col.query(
            query_texts=[query] if not query_embeddings else None,
            query_embeddings=query_embeddings,
            n_results=limit,
        )

        records = []
        if res and res.get("documents"):
            docs = res["documents"][0]
            ids = res["ids"][0]
            metadatas = res.get("metadatas", [[]])[0]
            for i, doc in enumerate(docs):
                meta = metadatas[i] if i < len(metadatas) else {}
                records.append(
                    MemoryRecord(
                        id=ids[i],
                        content=doc,
                        role=meta.get("role", "assistant"),
                        metadata=meta,
                    )
                )
        return records

    async def get_recent(self, limit: int = 10) -> list[MemoryRecord]:
        return await self.search(query="", limit=limit)

    async def clear(self) -> None:
        col = self._get_collection()
        # delete all items
        existing = col.get()
        if existing and existing.get("ids"):
            col.delete(ids=existing["ids"])
