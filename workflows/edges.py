"""Conditional routing rules for the ResearchIQ LangGraph workflow."""

from __future__ import annotations

from typing import Any

# Node names (shared by nodes, edges, and the UI progress display)
SUPERVISOR = "supervisor"
RETRIEVAL = "retrieval"
ANALYSIS = "research_analysis"
COMPARISON = "comparison"
VALIDATION = "citation_validation"
REPORT = "research_report"
DIRECT = "direct_response"
FINALIZE = "finalize"

NODE_LABELS: dict[str, str] = {
    SUPERVISOR: "Supervisor is analysing the request",
    RETRIEVAL: "Retrieval agent is searching the papers",
    ANALYSIS: "Research analysis agent is synthesising evidence",
    COMPARISON: "Comparison agent is building the comparison",
    REPORT: "Report agent is drafting the report",
    VALIDATION: "Citation validation agent is auditing the answer",
    DIRECT: "Preparing response",
    FINALIZE: "Finalising grounded answer",
}


def route_from_supervisor(state: dict[str, Any]) -> str:
    """Choose retrieval, direct response, or immediate finalisation."""
    if state.get("error"):
        return FINALIZE
    if not state.get("requires_retrieval", True):
        return DIRECT
    if not state.get("document_metadata"):
        return DIRECT
    return RETRIEVAL


def route_after_retrieval(state: dict[str, Any]) -> str:
    """Route to the specialist chosen by the supervisor.

    Empty evidence short-circuits to finalisation, where the required
    insufficient-evidence message is produced without inventing content.
    """
    if state.get("error"):
        return FINALIZE
    if not state.get("retrieved_documents"):
        return FINALIZE

    selected = str(state.get("selected_agent") or "")
    intent = str(state.get("research_intent") or "")
    if selected == "comparison" or intent == "comparison":
        return COMPARISON
    if selected == "report" or intent in {"report", "literature_review"}:
        return REPORT
    if selected == "research_analysis":
        return ANALYSIS
    return ANALYSIS


def route_after_specialist(state: dict[str, Any]) -> str:
    """Skip LLM validation when a specialist already reported an error."""
    if state.get("error"):
        return FINALIZE
    return VALIDATION


def route_after_validation(state: dict[str, Any]) -> str:
    """Retry retrieval when evidence is insufficient, within a hard limit."""
    if state.get("error"):
        return FINALIZE
    validation = state.get("validation_result") or {}
    iterations = int(state.get("iteration_count", 0))
    max_iterations = int(state.get("max_validation_iterations", 2))
    if validation.get("needs_more_evidence") and iterations <= max_iterations:
        return RETRIEVAL
    return FINALIZE
