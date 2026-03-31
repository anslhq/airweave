"""LightRAG-based knowledge graph service for Airweave.

Per-collection Knowledge Graph using LightRAG with file-based storage
(NetworkX graph + NanoVectorDB). No extra Docker services required.

Usage in the sync pipeline:
    kg = KnowledgeGraphService(collection_readable_id="my-collection")
    await kg.ingest("textual representation of entity...")
    await kg.cleanup()
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from airweave.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM adapter for LightRAG (OpenAI-compatible via settings)
# ---------------------------------------------------------------------------


async def _kg_llm_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[list] = None,
    keyword_extraction: bool = False,
    **kwargs,
) -> str:
    """LightRAG-compatible LLM function using OpenAI SDK against the configured endpoint.

    Uses settings.TOGETHER_BASE_URL and settings.TOGETHER_API_KEY to reach
    an OpenAI-compatible API (e.g. host.docker.internal:8317/v1).
    """
    from openai import AsyncOpenAI

    base_url = settings.TOGETHER_BASE_URL or "http://host.docker.internal:8317/v1"
    api_key = settings.TOGETHER_API_KEY or "airweave"

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    messages: list[dict] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history_messages:
        for msg in history_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": prompt})

    try:
        response = await client.chat.completions.create(
            model="gpt-5.2",
            messages=messages,
            temperature=0.0,
            max_tokens=4096,
        )
        result = response.choices[0].message.content or ""
        return result
    except Exception as e:
        logger.error(f"[KG LLM] Completion failed: {e}")
        raise
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Embedding adapter for LightRAG (uses local text2vec service)
# ---------------------------------------------------------------------------


async def _kg_embed(texts: list[str]) -> np.ndarray:
    """LightRAG-compatible embedding function using the local text2vec service.

    Calls the same TEXT2VEC_INFERENCE_URL that the sync pipeline uses,
    returning an np.ndarray of shape (len(texts), embedding_dim).
    """
    import httpx

    inference_url = settings.TEXT2VEC_INFERENCE_URL
    vectors = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        for text in texts:
            if not text or not text.strip():
                # LightRAG may pass empty strings; return zero vector
                dim = settings.EMBEDDING_DIMENSIONS or 384
                vectors.append([0.0] * dim)
                continue

            try:
                response = await client.post(
                    f"{inference_url}/vectors",
                    json={"text": text},
                )
                response.raise_for_status()
                data = response.json()
                vectors.append(data["vector"])
            except Exception as e:
                logger.warning(f"[KG Embed] Failed to embed text ({len(text)} chars): {e}")
                dim = settings.EMBEDDING_DIMENSIONS or 384
                vectors.append([0.0] * dim)

    return np.array(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------


class KnowledgeGraphService:
    """Per-collection Knowledge Graph service backed by LightRAG.

    Storage: file-based (NetworkX for graph, NanoVectorDB for vectors).
    Each collection gets its own working directory under
    ./local_storage/kg/{collection_readable_id}/.
    """

    def __init__(self, collection_readable_id: str) -> None:
        self.collection_readable_id = collection_readable_id
        self.working_dir = str(
            Path(settings.STORAGE_PATH) / "kg" / collection_readable_id
        )
        self._rag = None
        self._initialized = False

    async def _get_rag(self):
        """Lazy-initialize LightRAG instance."""
        if self._rag is not None and self._initialized:
            return self._rag

        from lightrag import LightRAG, QueryParam  # noqa: F811
        from lightrag.utils import wrap_embedding_func_with_attrs
        from lightrag.kg.shared_storage import initialize_pipeline_status

        os.makedirs(self.working_dir, exist_ok=True)

        embedding_dim = settings.EMBEDDING_DIMENSIONS or 384

        @wrap_embedding_func_with_attrs(
            embedding_dim=embedding_dim, max_token_size=8192
        )
        async def embedding_func(texts: list[str]) -> np.ndarray:
            return await _kg_embed(texts)

        self._rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=_kg_llm_complete,
            embedding_func=embedding_func,
            chunk_token_size=1200,
            enable_llm_cache=True,
            kv_storage="JsonKVStorage",
            vector_storage="NanoVectorDBStorage",
            graph_storage="NetworkXStorage",
            doc_status_storage="JsonDocStatusStorage",
            addon_params={
                "language": "English",
            },
        )

        await self._rag.initialize_storages()
        await initialize_pipeline_status()
        self._initialized = True

        logger.info(
            f"[KG] LightRAG initialized for collection '{self.collection_readable_id}' "
            f"(embedding_dim={embedding_dim}, working_dir={self.working_dir})"
        )
        return self._rag

    async def ingest(self, text: str) -> None:
        """Ingest a single text document into the knowledge graph.

        LightRAG extracts entities and relationships automatically
        using the configured LLM.
        """
        if not text or not text.strip():
            return

        rag = await self._get_rag()
        await rag.ainsert(text)

    async def ingest_batch(self, texts: list[str]) -> int:
        """Ingest multiple text documents into the knowledge graph.

        Args:
            texts: List of textual representations from parent entities.

        Returns:
            Number of texts successfully ingested.
        """
        if not texts:
            return 0

        rag = await self._get_rag()
        ingested = 0

        for text in texts:
            if not text or not text.strip():
                continue
            try:
                await rag.ainsert(text)
                ingested += 1
            except Exception as e:
                logger.warning(
                    f"[KG] Failed to ingest document ({len(text)} chars) "
                    f"for collection '{self.collection_readable_id}': {e}"
                )

        return ingested

    async def query(self, query: str, mode: str = "hybrid") -> str:
        """Query the knowledge graph for entities and relationships.

        Args:
            query: The search query.
            mode: LightRAG query mode — 'hybrid', 'local', 'global', or 'naive'.

        Returns:
            String with entity/relationship context, or empty string on failure.
        """
        from lightrag import QueryParam

        rag = await self._get_rag()
        result = await rag.aquery(query, param=QueryParam(mode=mode))
        return result or ""

    async def cleanup(self) -> None:
        """Finalize storages on shutdown."""
        if self._rag:
            try:
                await self._rag.finalize_storages()
                logger.info(
                    f"[KG] Storages finalized for collection "
                    f"'{self.collection_readable_id}'"
                )
            except Exception as e:
                logger.warning(
                    f"[KG] Cleanup failed for collection "
                    f"'{self.collection_readable_id}': {e}"
                )
            self._rag = None
            self._initialized = False
