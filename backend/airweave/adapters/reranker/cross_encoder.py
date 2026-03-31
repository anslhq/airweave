"""Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

Runs the model locally via sentence-transformers. Inference is CPU-bound,
so we offload it to a thread pool to avoid blocking the async event loop.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Optional

from airweave.adapters.reranker.exceptions import RerankerError
from airweave.adapters.reranker.types import RerankerResult
from airweave.core.protocols.reranker import RerankerProtocol

# Default model and settings
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"
CROSS_ENCODER_MAX_LENGTH = 512


class CrossEncoderReranker(RerankerProtocol):
    """Reranker using a local cross-encoder model (bge-reranker-v2-m3).

    The model is loaded lazily on first use to avoid slowing startup when
    reranking is configured but not yet needed.
    """

    def __init__(
        self,
        model_name: str = CROSS_ENCODER_MODEL,
        max_length: int = CROSS_ENCODER_MAX_LENGTH,
        device: Optional[str] = None,
    ) -> None:
        """Initialize with model name and max input length.

        Args:
            model_name: HuggingFace model ID for the cross-encoder.
            max_length: Maximum token length for query-document pairs.
            device: Device to run on (e.g. "cpu", "cuda"). None = auto-detect.
        """
        self._model_name = model_name
        self._max_length = max_length
        self._device = device
        self._model = None  # lazy-loaded

    def _ensure_model(self):
        """Load the cross-encoder model if not already loaded."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self._model_name,
                max_length=self._max_length,
                device=self._device,
            )
        except Exception as e:
            raise RerankerError(
                f"Failed to load cross-encoder model {self._model_name!r}: {e}",
                cause=e,
            ) from e

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankerResult]:
        """Rerank documents using the local cross-encoder.

        Runs inference in a thread pool so the async event loop is not blocked.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_n: Maximum number of results to return. None means all.

        Returns:
            List of RerankerResult ordered by relevance (highest first).

        Raises:
            RerankerError: If model loading or inference fails.
        """
        if not documents:
            return []

        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(
                None,
                functools.partial(self._predict_sync, query, documents),
            )
        except RerankerError:
            raise
        except Exception as e:
            raise RerankerError(
                f"Cross-encoder rerank failed: {e}",
                cause=e,
            ) from e

        # Sort by score descending
        scored = sorted(
            enumerate(results),
            key=lambda pair: pair[1],
            reverse=True,
        )

        if top_n is not None:
            scored = scored[:top_n]

        return [
            RerankerResult(index=idx, relevance_score=float(score))
            for idx, score in scored
        ]

    def _predict_sync(self, query: str, documents: list[str]) -> list[float]:
        """Run cross-encoder prediction synchronously (called from thread pool)."""
        self._ensure_model()
        assert self._model is not None

        pairs = [[query, doc] for doc in documents]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]
