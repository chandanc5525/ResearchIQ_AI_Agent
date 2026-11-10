"""Helpers shared by evidence-consuming agents."""

from __future__ import annotations

from typing import Any, Sequence

from retrieval.retriever import ResearchRetriever


def get_state_context(state: dict[str, Any]) -> str:
    """Return the retrieval context, rebuilding it from chunks if necessary."""
    context = str(state.get("retrieval_context") or "").strip()
    if context:
        return context
    chunks = state.get("retrieved_documents") or []
    if chunks:
        return ResearchRetriever.build_context(list(chunks))
    return ""


def get_paper_names(state: dict[str, Any]) -> list[str]:
    """Return display names for the selected papers, preserving upload order."""
    target_ids = [
        str(doc_id) for doc_id in (state.get("target_documents") or []) if doc_id
    ]
    target_set = set(target_ids)
    names: list[str] = []
    for document in state.get("document_metadata") or []:
        if target_ids and str(document.get("doc_id")) not in target_set:
            continue
        filename = str(document.get("filename", "")).strip()
        if filename and filename not in names:
            names.append(filename)
    if not names:
        for chunk in state.get("retrieved_documents") or []:
            filename = str(chunk.get("filename", "")).strip()
            if filename and filename not in names:
                names.append(filename)
    return names


def get_history(state: dict[str, Any], limit: int = 6) -> Sequence[dict[str, str]]:
    return list(state.get("chat_history") or [])[-limit:]
