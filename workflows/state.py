"""Shared LangGraph state for the ResearchIQ agent workflow."""

from __future__ import annotations

from typing import Any, Sequence, TypedDict

from utils.helpers import utc_now_iso


class ResearchState(TypedDict, total=False):
    """State shared by every node in the research graph.

    ``total=False`` keeps node updates small; the workflow factory always seeds
    the required fields so agents can rely on them being present.
    """

    # Core fields required by the project specification
    user_query: str
    uploaded_documents: list[str]
    document_metadata: list[dict[str, Any]]
    retrieved_documents: list[dict[str, Any]]
    retrieval_context: str
    research_intent: str
    selected_agent: str
    analysis_result: str
    comparison_result: str
    validation_result: dict[str, Any]
    final_response: str
    citations: list[dict[str, Any]]
    iteration_count: int

    # Workflow / conversation extensions
    chat_history: list[dict[str, str]]
    target_documents: list[str]
    retrieval_query: str
    requires_retrieval: bool
    intent_hint: str | None
    report_requested: bool
    literature_review: bool
    report_result: str
    direct_response: str
    retrieval_note: str
    validation_attempts: int
    max_validation_iterations: int
    routing_reason: str
    error: str
    created_at: str
    completed_at: str


def initial_state(
    user_query: str,
    *,
    documents: Sequence[dict[str, Any]] | None = None,
    target_documents: Sequence[str] | None = None,
    chat_history: Sequence[dict[str, str]] | None = None,
    intent_hint: str | None = None,
    max_validation_iterations: int = 2,
) -> ResearchState:
    """Build a fully seeded initial state for a workflow run."""
    document_list = [dict(document) for document in (documents or [])]
    return ResearchState(
        user_query=(user_query or "").strip(),
        uploaded_documents=[
            str(document.get("filename", "")) for document in document_list
        ],
        document_metadata=document_list,
        retrieved_documents=[],
        retrieval_context="",
        research_intent="",
        selected_agent="",
        analysis_result="",
        comparison_result="",
        validation_result={},
        final_response="",
        citations=[],
        iteration_count=0,
        chat_history=[dict(message) for message in (chat_history or [])],
        target_documents=[str(doc_id) for doc_id in (target_documents or []) if doc_id],
        retrieval_query="",
        requires_retrieval=True,
        intent_hint=intent_hint,
        report_requested=False,
        literature_review=False,
        report_result="",
        direct_response="",
        retrieval_note="",
        validation_attempts=0,
        max_validation_iterations=max_validation_iterations,
        routing_reason="",
        error="",
        created_at=utc_now_iso(),
        completed_at="",
    )
