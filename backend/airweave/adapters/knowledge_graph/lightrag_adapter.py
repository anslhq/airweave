"""LightRAG Knowledge Graph adapter — HTTP client for the dedicated LightRAG container.

Replaces the embedded in-process approach with HTTP calls to the LightRAG REST API
running at http://lightrag:9621. No lightrag-hku dependency needed in the backend.

Usage:
    kg = KnowledgeGraphService(collection_readable_id="test-v0titp")
    await kg.ingest("document text here")
    result = await kg.query("what is the four pixel rule?")
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

LIGHTRAG_BASE_URL = "http://lightrag:9621"
LIGHTRAG_INGEST_TIMEOUT = 300.0  # 5 min for large ingestion batches
LIGHTRAG_QUERY_TIMEOUT = 10.0  # 10s max for KG queries — fail fast, don't block search


class KnowledgeGraphService:
    """HTTP client for the dedicated LightRAG container."""

    def __init__(self, collection_readable_id: str):
        self.collection_readable_id = collection_readable_id
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=LIGHTRAG_BASE_URL,
                timeout=LIGHTRAG_QUERY_TIMEOUT,
            )
        return self._client

    async def ingest(self, text: str) -> bool:
        """Ingest a single document into the KG."""
        client = await self._get_client()
        try:
            resp = await client.post(
                "/documents/text",
                json={"text": text},
                timeout=LIGHTRAG_INGEST_TIMEOUT,
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"[KG] Ingest failed for '{self.collection_readable_id}': {e}")
            return False

    async def ingest_batch(self, texts: list[str]) -> int:
        """Ingest multiple documents. Returns count of successful ingestions."""
        if not texts:
            return 0

        client = await self._get_client()
        ingested = 0

        for text in texts:
            if not text or not text.strip():
                continue
            try:
                resp = await client.post(
                    "/documents/text",
                    json={"text": text},
                    timeout=LIGHTRAG_INGEST_TIMEOUT,
                )
                resp.raise_for_status()
                ingested += 1
            except Exception as e:
                logger.warning(
                    f"[KG] Failed to ingest doc ({len(text)} chars) "
                    f"for '{self.collection_readable_id}': {e}"
                )

        return ingested

    async def query(self, query: str, mode: str = "hybrid") -> str:
        """Query the knowledge graph. Returns entity/relationship context string."""
        client = await self._get_client()
        try:
            resp = await client.post(
                "/query",
                json={"query": query, "mode": mode},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "") or ""
        except Exception as e:
            logger.warning(
                f"[KG] Query failed for '{self.collection_readable_id}': {e}"
            )
            return ""

    async def cleanup(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
