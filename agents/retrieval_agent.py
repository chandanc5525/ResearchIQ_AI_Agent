"""Document Retrieval Agent: evidence acquisition from ChromaDB."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from retrieval.retriever import ResearchRetriever
from utils.config import Settings
from utils.helpers import truncate_text
from utils.logger import get_logger

logger = get_logger("agents.retrieval")

_BROAD_INTENTS = {"comparison", "report", "literature_review"}


class RetrievalAgent:
    """Retrieve page-level evidence for the current research intent."""

    def __init__(self, retriever: ResearchRetriever, settings: Settings) -> None:
        self.retriever = retriever
        self.settings = settings

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Search the vector store and update shared state with evidence."""
        intent = str(state.get("research_intent", "general_qa"))
        query = str(
            state.get("retrieval_query") or state.get("user_query") or ""
        ).strip()
        target_ids = [
            str(doc_id) for doc_id in (state.get("target_documents") or []) if doc_id
        ]
        iteration = int(state.get("iteration_count", 0)) + 1
        paper_names = self._paper_names(state, target_ids)
        per_document = intent in _BROAD_INTENTS and len(target_ids) > 1

        search_query = self._broaden_query(query, intent, paper_names)
        logger.info(
            "Retrieval pass %s: intent=%s docs=%s query=%s",
            iteration,
            intent,
            len(target_ids),
            truncate_text(search_query, 100),
        )

        result = self.retriever.retrieve(
            search_query,
            top_k=self.settings.retrieval_top_k,
            doc_ids=target_ids or None,
            per_document=per_document,
        )

        if result.is_empty:
            logger.warning("No evidence retrieved: %s", result.note)
            return {
                "retrieved_documents": [],
                "retrieval_context": "",
                "citations": [],
                "iteration_count": iteration,
                "retrieval_note": result.note,
                "error": "",
            }

        return {
            "retrieved_documents": result.documents,
            "retrieval_context": result.context,
            "citations": result.citations,
            "iteration_count": iteration,
            "retrieval_note": result.note,
            "error": "",
        }

    def _paper_names(
        self, state: dict[str, Any], target_ids: Sequence[str]
    ) -> list[str]:
        names: list[str] = []
        target_set = set(target_ids)
        for document in state.get("document_metadata") or []:
            if target_ids and str(document.get("doc_id")) not in target_set:
                continue
            filename = str(document.get("filename", ""))
            if filename:
                names.append(filename)
        if not names:
            for chunk in state.get("retrieved_documents") or []:
                filename = str(chunk.get("filename", ""))
                if filename and filename not in names:
                    names.append(filename)
        return names

    @staticmethod
    def _broaden_query(query: str, intent: str, paper_names: Sequence[str]) -> str:
        if intent == "comparison":
            suffix = (
                "research objective methodology dataset algorithm evaluation metrics "
                "experimental results limitations"
            )
            if suffix not in query.lower():
                return f"{query} {suffix}".strip()
        if intent in {"report", "literature_review"}:
            suffix = (
                "research problem objectives methodology datasets algorithms findings "
                "limitations future work"
            )
            if suffix not in query.lower():
                return f"{query} {suffix}".strip()
        return query
