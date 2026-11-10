"""Application service composition for ResearchIQ.

The Streamlit layer, CLI, and tests all build the same object graph through this
module. Keeping composition here prevents UI code from constructing backend
objects directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from embeddings.embedding_model import get_embeddings
from generation.llm import TextGenerator, check_ollama_health, create_chat_model
from ingestion.document_parser import DocumentRecord, DocumentRegistry
from ingestion.ingestion_pipeline import IngestionPipeline
from retrieval.retriever import ResearchRetriever
from retrieval.vector_store import VectorStoreManager, build_vector_store
from utils.config import Settings, configure_langsmith, get_settings
from workflows.research_graph import ResearchIQWorkflow

logger = logging.getLogger("researchiq.services")


@dataclass
class ApplicationServices:
    """Fully composed backend used by the UI and CLI."""

    settings: Settings
    llm: TextGenerator
    embeddings: Any
    vector_store: VectorStoreManager
    registry: DocumentRegistry
    ingestion: IngestionPipeline
    retriever: ResearchRetriever
    workflow: ResearchIQWorkflow

    def document_metadata(self) -> list[dict[str, Any]]:
        """Return paper metadata in the shape expected by the workflow."""
        return [
            self._record_to_dict(record) for record in self.registry.list_documents()
        ]

    @staticmethod
    def _record_to_dict(record: DocumentRecord) -> dict[str, Any]:
        return {
            "doc_id": record.doc_id,
            "filename": record.filename,
            "title": record.title or record.filename,
            "page_count": record.page_count,
            "chunk_count": record.chunk_count,
            "ingested_at": record.ingested_at,
            "file_size_bytes": record.file_size_bytes,
        }

    def document_choices(self) -> dict[str, str]:
        """Map doc_id -> display label for Streamlit widgets."""
        choices: dict[str, str] = {}
        for record in self.registry.list_documents():
            label = record.filename
            if record.page_count:
                label += f" ({record.page_count} pages)"
            choices[record.doc_id] = label
        return choices

    def resolve_documents(self, doc_ids: Sequence[str]) -> list[dict[str, Any]]:
        selected = set(doc_ids)
        return [
            metadata
            for metadata in self.document_metadata()
            if metadata["doc_id"] in selected
        ]

    def health(self) -> dict[str, Any]:
        status = check_ollama_health(self.settings)
        status["embedding_model"] = self.settings.embedding_model
        status["tracing_enabled"] = bool(
            self.settings.langsmith_tracing and self.settings.langsmith_api_key
        )
        status["documents"] = self.registry.count()
        status["chunks"] = self.vector_store.count()
        return status


def create_application(
    settings: Settings | None = None,
    *,
    llm: TextGenerator | None = None,
) -> ApplicationServices:
    """Construct the complete backend. Does not verify Ollama availability."""
    resolved = settings or get_settings()
    resolved.ensure_directories()
    tracing = configure_langsmith(resolved)
    logger.info(
        "Building ResearchIQ services (model=%s, tracing=%s)",
        resolved.ollama_model,
        tracing,
    )

    embeddings = get_embeddings(
        resolved.embedding_model,
        resolved.embedding_device,
        resolved.embedding_batch_size,
    )
    vector_store = build_vector_store(resolved, embeddings)
    registry = DocumentRegistry(resolved.registry_file)
    ingestion = IngestionPipeline(
        settings=resolved,
        vector_store=vector_store,
        registry=registry,
    )
    retriever = ResearchRetriever(vector_store, resolved)
    resolved_llm = llm or create_chat_model(resolved)
    workflow = ResearchIQWorkflow(resolved, retriever, resolved_llm)

    return ApplicationServices(
        settings=resolved,
        llm=resolved_llm,
        embeddings=embeddings,
        vector_store=vector_store,
        registry=registry,
        ingestion=ingestion,
        retriever=retriever,
        workflow=workflow,
    )
