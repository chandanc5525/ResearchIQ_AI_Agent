"""LangGraph node implementations for ResearchIQ.

Nodes are thin adapters: they call a specialist agent, normalise failures into
the shared state, and never contain prompt or retrieval logic themselves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agents.citation_validation_agent import CitationValidationAgent
from agents.comparison_agent import ComparisonAgent
from agents.research_analysis_agent import ResearchAnalysisAgent
from agents.research_report_agent import ResearchReportAgent
from agents.retrieval_agent import RetrievalAgent
from agents.supervisor_agent import SupervisorAgent
from generation.response_generator import ResponseGenerator
from prompts.rag_prompt import INSUFFICIENT_EVIDENCE_MESSAGE
from utils.config import Settings
from utils.helpers import utc_now_iso
from utils.logger import get_logger

logger = get_logger("workflow.nodes")


@dataclass
class WorkflowDependencies:
    """Objects required to construct the research graph."""

    settings: Settings
    supervisor: SupervisorAgent
    retrieval: RetrievalAgent
    analysis: ResearchAnalysisAgent
    comparison: ComparisonAgent
    validation: CitationValidationAgent
    report: ResearchReportAgent
    response_generator: ResponseGenerator


class ResearchNodes:
    """Build LangGraph node callables bound to the specialist agents."""

    def __init__(self, dependencies: WorkflowDependencies) -> None:
        self.deps = dependencies
        self.settings = dependencies.settings

    def supervisor(self, state: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.deps.supervisor.run(state)
        except Exception as exc:
            logger.exception("Supervisor node failed")
            return {
                "error": f"Supervisor routing failed: {exc}",
                "requires_retrieval": False,
            }

    def retrieval(self, state: dict[str, Any]) -> dict[str, Any]:
        try:
            updates = self.deps.retrieval.run(state)
            updates["error"] = ""
            return updates
        except Exception as exc:
            logger.exception("Retrieval node failed")
            return {
                "retrieved_documents": [],
                "retrieval_context": "",
                "citations": [],
                "iteration_count": int(state.get("iteration_count", 0)) + 1,
                "error": f"Document retrieval failed: {exc}",
            }

    def analysis(self, state: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.deps.analysis.run(state)
        except Exception as exc:
            logger.exception("Analysis node failed")
            return {
                "analysis_result": "",
                "error": f"Research analysis failed: {exc}",
            }

    def comparison(self, state: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.deps.comparison.run(state)
        except Exception as exc:
            logger.exception("Comparison node failed")
            return {
                "comparison_result": "",
                "error": f"Paper comparison failed: {exc}",
            }

    def report(self, state: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.deps.report.run(state)
        except Exception as exc:
            logger.exception("Report node failed")
            return {
                "report_result": "",
                "analysis_result": "",
                "error": f"Report generation failed: {exc}",
            }

    def validation(self, state: dict[str, Any]) -> dict[str, Any]:
        if str(state.get("error") or "").strip():
            return {
                "validation_result": {
                    "is_valid": False,
                    "grounded": False,
                    "confidence": 0.0,
                    "unsupported_claims": [],
                    "fabricated_citations": [],
                    "needs_more_evidence": False,
                    "notes": "Validation skipped because an upstream agent failed.",
                    "method": "skipped",
                },
                "validation_attempts": int(state.get("validation_attempts", 0)) + 1,
            }
        try:
            updates = self.deps.validation.run(state)
            validation = updates.get("validation_result") or {}
            if validation.get("needs_more_evidence"):
                updates["retrieval_query"] = (
                    validation.get("refined_query")
                    or state.get("retrieval_query")
                    or state.get("user_query")
                )
            return updates
        except Exception as exc:
            logger.exception("Validation node failed")
            fallback = {
                "is_valid": False,
                "grounded": False,
                "confidence": 0.0,
                "unsupported_claims": ["Citation validation could not be completed."],
                "fabricated_citations": [],
                "needs_more_evidence": False,
                "notes": f"Validation error: {exc}",
            }
            return {
                "validation_result": fallback,
                "validation_attempts": int(state.get("validation_attempts", 0)) + 1,
                "error": "",
            }

    def direct(self, state: dict[str, Any]) -> dict[str, Any]:
        existing = str(state.get("direct_response") or "").strip()
        if existing:
            return {"direct_response": existing, "error": ""}
        try:
            answer = self.deps.response_generator.direct_answer(
                query=str(state.get("user_query", "")),
                history=state.get("chat_history") or [],
                has_documents=bool(state.get("document_metadata")),
                note=str(state.get("routing_reason") or ""),
            )
            return {"direct_response": answer, "error": ""}
        except Exception as exc:
            logger.exception("Direct response node failed")
            return {
                "direct_response": "",
                "error": (
                    f"The local language model could not generate a response: {exc}"
                ),
            }

    def finalize(self, state: dict[str, Any]) -> dict[str, Any]:
        """Assemble the final, citation-checked response."""
        error = str(state.get("error") or "").strip()
        if error:
            final = (
                "I could not complete this research request. "
                f"{error}\n\nCheck that Ollama is running and that the selected "
                "model is installed, then try again."
            )
            return {
                "final_response": final,
                "completed_at": utc_now_iso(),
            }

        answer = self._select_answer(state)
        citations = state.get("citations") or []
        retrieval_note = str(state.get("retrieval_note") or "")

        if not answer:
            if not state.get("retrieved_documents"):
                answer = INSUFFICIENT_EVIDENCE_MESSAGE
                if retrieval_note:
                    answer += f"\n\n> {retrieval_note}"
            else:
                answer = INSUFFICIENT_EVIDENCE_MESSAGE

        final = self.deps.response_generator.prepare_final_response(
            answer,
            citations,
            state.get("validation_result") or {},
            retrieval_note,
        )
        return {"final_response": final, "completed_at": utc_now_iso()}

    @staticmethod
    def _select_answer(state: dict[str, Any]) -> str:
        for key in (
            "report_result",
            "comparison_result",
            "analysis_result",
            "direct_response",
        ):
            value = str(state.get(key) or "").strip()
            if value:
                return value
        return str(state.get("final_response") or "").strip()
