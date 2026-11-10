"""Local Hugging Face sentence-transformer embeddings.

The model is loaded lazily and cached. This keeps one copy in memory across
Streamlit reruns, which matters on a 16 GB machine, and avoids repeatedly
paying the model initialisation cost.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from utils.errors import EmbeddingError

logger = logging.getLogger("researchiq.embeddings")


@lru_cache(maxsize=2)
def get_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
    batch_size: int = 32,
) -> Any:
    """Return a cached ``HuggingFaceEmbeddings`` instance.

    Args:
        model_name: Hugging Face model identifier.
        device: ``cpu`` or ``cuda``. CPU is the safe default.
        batch_size: Encoding batch size used by sentence-transformers.

    Raises:
        EmbeddingError: If the model cannot be loaded.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:  # pragma: no cover - dependency is required
        raise EmbeddingError(
            "langchain-huggingface is not installed. Run `uv sync`."
        ) from exc

    try:
        logger.info("Loading embedding model %s on %s", model_name, device)
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": batch_size,
            },
            show_progress=False,
        )
        # Fail early with a clear message if the model cannot encode.
        probe = embeddings.embed_query("ResearchIQ embedding health check")
        if not probe:
            raise ValueError("Embedding model returned an empty vector")
        logger.info("Embedding model ready (dimension=%s)", len(probe))
        return embeddings
    except Exception as exc:
        logger.exception("Failed to initialise embedding model %s", model_name)
        raise EmbeddingError(
            f"Could not load embedding model '{model_name}': {exc}. "
            "Check your internet connection for the first download and ensure "
            "the model name is correct."
        ) from exc


def clear_embedding_cache() -> None:
    """Release cached embedding models (used after settings changes)."""
    get_embeddings.cache_clear()
