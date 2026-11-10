"""Semantic retrieval, context construction, and page-level citations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from langchain_core.documents import Document

from retrieval.vector_store import VectorStoreManager
from utils.config import Settings
from utils.errors import VectorStoreError
from utils.helpers import citation_label, dedupe_preserve_order, truncate_text

logger = logging.getLogger("researchiq.retriever")


@dataclass
class RetrievalResult:
    """Container for retrieved evidence and its provenance."""

    documents: list[dict[str, Any]] = field(default_factory=list)
    context: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    query: str = ""
    note: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.documents


class ResearchRetriever:
    """Retrieve evidence across one or multiple indexed research papers."""

    def __init__(self, vector_store: VectorStoreManager, settings: Settings) -> None:
        self.vector_store = vector_store
        self.settings = settings

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        doc_ids: Sequence[str] | None = None,
        per_document: bool = False,
        max_chunks: int | None = None,
    ) -> RetrievalResult:
        """Run similarity search and build grounded context.

        Args:
            query: Natural-language search query.
            top_k: Number of chunks to retrieve (defaults to settings).
            doc_ids: Restrict retrieval to these paper IDs.
            per_document: Retrieve ``top_k`` chunks for each selected paper.
                Used for comparisons and reports where coverage matters.
            max_chunks: Hard cap on evidence chunks.
        """
        query = (query or "").strip()
        if not query:
            return RetrievalResult(query=query, note="The retrieval query was empty.")

        k = int(top_k or self.settings.retrieval_top_k)
        limit = int(max_chunks or self.settings.max_context_chunks)
        selected_ids = [str(doc_id) for doc_id in (doc_ids or []) if doc_id]

        try:
            if per_document and len(selected_ids) > 1:
                raw_results: list[tuple[Document, float]] = []
                seen: set[tuple[str, int, str]] = set()
                for doc_id in selected_ids:
                    for document, score in self.vector_store.similarity_search(
                        query,
                        k=k,
                        doc_ids=[doc_id],
                        score_threshold=self.settings.similarity_threshold,
                    ):
                        key = self._chunk_key(document)
                        if key not in seen:
                            seen.add(key)
                            raw_results.append((document, score))
                raw_results.sort(key=lambda item: item[1], reverse=True)
            else:
                raw_results = self.vector_store.similarity_search(
                    query,
                    k=min(k * 2 if selected_ids and len(selected_ids) > 1 else k, 40),
                    doc_ids=selected_ids or None,
                    score_threshold=self.settings.similarity_threshold,
                )[:k]
        except VectorStoreError as exc:
            logger.exception("Retrieval failed")
            return RetrievalResult(query=query, note=str(exc))

        chunks: list[dict[str, Any]] = []
        for document, score in raw_results[:limit]:
            chunks.append(self._to_chunk(document, score))

        if not chunks:
            return RetrievalResult(
                query=query,
                note=(
                    "No chunks passed the similarity threshold. Upload or index a "
                    "research paper first, or lower the threshold in Settings."
                ),
            )

        citations = self._build_citations(chunks)
        context = self.build_context(chunks)
        logger.info(
            "Retrieved %s chunks from %s document(s) for query '%s'",
            len(chunks),
            len({chunk["doc_id"] for chunk in chunks}),
            truncate_text(query, 90),
        )
        return RetrievalResult(
            documents=chunks,
            context=context,
            citations=citations,
            query=query,
        )

    def search_for_evaluation(
        self, query: str, k: int = 5, doc_ids: Sequence[str] | None = None
    ) -> list[dict[str, Any]]:
        """Return raw chunk metadata for offline retrieval evaluation."""
        return self.retrieve(query, top_k=k, doc_ids=doc_ids).documents

    @staticmethod
    def _chunk_key(document: Document) -> tuple[str, int, str]:
        metadata = document.metadata or {}
        return (
            str(metadata.get("doc_id", "")),
            int(metadata.get("page_number", metadata.get("page", 0)) or 0),
            str(metadata.get("chunk_id", "")),
        )

    @staticmethod
    def _to_chunk(document: Document, score: float) -> dict[str, Any]:
        metadata = document.metadata or {}
        page_number = int(metadata.get("page_number", 0) or 0)
        if page_number <= 0:
            page_number = int(metadata.get("page", 0) or 0) + 1
        filename = str(
            metadata.get("filename")
            or metadata.get("source_file")
            or metadata.get("source")
            or "Unknown"
        )
        return {
            "content": document.page_content,
            "document": document.page_content,
            "filename": filename,
            "page": page_number,
            "page_number": page_number,
            "doc_id": str(metadata.get("doc_id", "")),
            "chunk_id": str(metadata.get("chunk_id", "")),
            "chunk_index": metadata.get("chunk_index"),
            "score": round(float(score), 4),
            "label": citation_label(filename, page_number),
        }

    @classmethod
    def _build_citations(cls, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for chunk in chunks:
            key = (chunk["filename"], int(chunk["page"]))
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                {
                    "filename": chunk["filename"],
                    "page": int(chunk["page"]),
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "score": chunk["score"],
                }
            )
        return citations

    @staticmethod
    def build_context(chunks: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            content = chunk["content"]
            if len(content) > 1600:
                content = content[:1599].rsplit(" ", 1)[0] + "…"
            blocks.append(
                "\n".join(
                    [
                        f"[Evidence {index}]",
                        f"Document: {chunk['filename']}",
                        f"Page: {chunk['page']}",
                        f"Chunk: {chunk['chunk_id']}",
                        "Content:",
                        content,
                    ]
                )
            )
        return "\n\n".join(blocks)

    @staticmethod
    def unique_source_labels(chunks: Sequence[dict[str, Any]]) -> list[str]:
        return dedupe_preserve_order(str(chunk.get("label", "")) for chunk in chunks)
