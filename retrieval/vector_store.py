"""ChromaDB-backed persistent vector storage.

Wraps LangChain's Chroma integration so the rest of the application does not
depend on Chroma implementation details. Metadata filters are always applied on
``doc_id`` values that come from the local document registry.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Sequence

from langchain_core.documents import Document

from utils.config import Settings
from utils.errors import VectorStoreError

logger = logging.getLogger("researchiq.vectorstore")


class VectorStoreManager:
    """Manage the persistent Chroma collection for research papers."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
        embeddings: Any,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embeddings = embeddings
        self._lock = threading.RLock()
        self._store: Any | None = None

    @property
    def store(self) -> Any:
        """Lazily open the Chroma collection."""
        if self._store is None:
            self._store = self._create_store()
        return self._store

    def _create_store(self) -> Any:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            from langchain_chroma import Chroma

            self.persist_directory.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            store = Chroma(
                client=client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                collection_metadata={"hnsw:space": "cosine"},
                relevance_score_fn=lambda distance: max(
                    0.0, min(1.0, 1.0 - float(distance))
                ),
            )
            # Read the count from the local handle: ``_store`` is not assigned
            # yet, so calling ``self.count()`` here would recurse into
            # ``_create_store`` while the collection is still being opened.
            chunk_count = int(store._collection.count())  # noqa: SLF001
            logger.info(
                "Opened Chroma collection '%s' at %s (%s chunks)",
                self.collection_name,
                self.persist_directory,
                chunk_count,
            )
            return store
        except Exception as exc:
            logger.exception("Could not open Chroma vector store")
            raise VectorStoreError(
                f"Could not open the vector database at {self.persist_directory}: {exc}"
            ) from exc

    def add_chunks(self, documents: Sequence[Document], ids: Sequence[str]) -> int:
        """Persist document chunks with explicit, stable IDs."""
        if len(documents) != len(ids):
            raise VectorStoreError("Document and ID counts must match")
        if not documents:
            return 0
        try:
            with self._lock:
                self.store.add_documents(list(documents), ids=list(ids))
            logger.info("Added %s chunks to Chroma", len(documents))
            return len(documents)
        except Exception as exc:
            logger.exception("Failed to add chunks to Chroma")
            raise VectorStoreError(f"Could not store document chunks: {exc}") from exc

    def delete_document(self, doc_id: str) -> None:
        """Delete every chunk belonging to a document."""
        try:
            with self._lock:
                self.store.delete(where={"doc_id": doc_id})
            logger.info("Deleted chunks for document %s", doc_id)
        except Exception as exc:
            logger.exception("Failed to delete document %s from Chroma", doc_id)
            raise VectorStoreError(
                f"Could not remove document vectors for {doc_id}: {exc}"
            ) from exc

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        doc_ids: Sequence[str] | None = None,
        score_threshold: float = 0.0,
    ) -> list[tuple[Document, float]]:
        """Return semantically similar chunks, optionally filtered by paper."""
        query = (query or "").strip()
        if not query:
            return []
        where = self._build_filter(doc_ids)
        try:
            with self._lock:
                results = self.store.similarity_search_with_relevance_scores(
                    query,
                    k=max(1, int(k)),
                    filter=where,
                )
        except Exception as exc:
            logger.exception("Chroma similarity search failed")
            raise VectorStoreError(f"Vector search failed: {exc}") from exc

        normalised: list[tuple[Document, float]] = []
        for document, score in results:
            numeric_score = float(score) if score is not None else 0.0
            if score_threshold > 0.0 and numeric_score < score_threshold:
                continue
            normalised.append((document, numeric_score))
        normalised.sort(key=lambda item: item[1], reverse=True)
        return normalised

    def count(self) -> int:
        """Return the number of indexed chunks."""
        try:
            return int(self.store._collection.count())  # noqa: SLF001
        except Exception:
            return 0

    def count_for_document(self, doc_id: str) -> int:
        """Return the number of indexed chunks for one document."""
        try:
            result = self.store.get(where={"doc_id": doc_id}, include=[])
            return len(result.get("ids", []))
        except Exception as exc:
            logger.warning("Could not count chunks for %s: %s", doc_id, exc)
            return 0

    @staticmethod
    def _build_filter(doc_ids: Sequence[str] | None) -> dict[str, Any] | None:
        if not doc_ids:
            return None
        unique_ids = list(dict.fromkeys(str(doc_id) for doc_id in doc_ids if doc_id))
        if not unique_ids:
            return None
        if len(unique_ids) == 1:
            return {"doc_id": unique_ids[0]}
        return {"doc_id": {"$in": unique_ids}}


def build_vector_store(settings: Settings, embeddings: Any) -> VectorStoreManager:
    """Create the vector store manager described by application settings."""
    return VectorStoreManager(
        persist_directory=settings.vectorstore_dir,
        collection_name=settings.chroma_collection,
        embeddings=embeddings,
    )
