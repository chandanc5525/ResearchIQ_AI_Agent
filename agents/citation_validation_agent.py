"""Citation Validation Agent: grounding audit and fabrication rejection."""

from __future__ import annotations

import logging
from typing import Any

from agents.context_utils import get_state_context
from generation.llm import TextGenerator, invoke_json
from prompts.rag_prompt import INSUFFICIENT_EVIDENCE_MESSAGE
from prompts.validation_prompt import build_validation_prompt
from utils.helpers import parse_citations, truncate_text
from utils.logger import get_logger

logger = get_logger("agents.validation")


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _as_string_list(value: Any, limit: int = 8) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = [str(item) for item in value]
    else:
        items = [str(value)]
    return [truncate_text(item, 300) for item in items if item.strip()][:limit]


class CitationValidationAgent:
    """Verify that answers are supported by, and cite, retrieved evidence."""

    def __init__(self, llm: TextGenerator) -> None:
        self.llm = llm

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Audit the generated answer and decide whether more evidence is needed."""
        answer = self._answer_to_validate(state)
        context = get_state_context(state)
        query = str(state.get("user_query", "")).strip()
        intent = str(state.get("research_intent", "general_qa"))
        attempts = int(state.get("validation_attempts", 0)) + 1

        available = {
            (str(chunk.get("filename", "")), int(chunk.get("page", 0)))
            for chunk in state.get("retrieved_documents") or []
        }
        parsed = parse_citations(answer)
        fabricated = [pair for pair in parsed if pair not in available]
        abstained = INSUFFICIENT_EVIDENCE_MESSAGE.lower() in (answer or "").lower()
        missing_citations = bool(available) and not parsed and not abstained
        deterministic_ok = not fabricated and not missing_citations

        citation_report = self._citation_report(
            available, parsed, fabricated, missing_citations
        )

        llm_result: dict[str, Any] = {}
        llm_failed = False
        if context and not abstained:
            try:
                llm_result = invoke_json(
                    self.llm,
                    build_validation_prompt(
                        answer=answer,
                        context=context,
                        query=query,
                        citation_report=citation_report,
                    ),
                )
            except Exception as exc:
                logger.warning(
                    "LLM validation unavailable; using deterministic checks: %s", exc
                )
                llm_failed = True

        if abstained:
            validation = {
                "is_valid": True,
                "grounded": True,
                "confidence": 1.0,
                "unsupported_claims": [],
                "fabricated_citations": [],
                "verified_citations": [],
                "needs_more_evidence": False,
                "refined_query": "",
                "notes": "The assistant correctly abstained because the evidence was insufficient.",
                "method": "abstention",
            }
        elif not context:
            validation = {
                "is_valid": False,
                "grounded": False,
                "confidence": 0.0,
                "unsupported_claims": [
                    "No retrieved evidence was available for validation."
                ],
                "fabricated_citations": [],
                "verified_citations": [],
                "needs_more_evidence": True,
                "refined_query": query or "research paper evidence",
                "notes": "No evidence was retrieved.",
                "method": "deterministic",
            }
        else:
            grounded = (
                _as_bool(llm_result.get("grounded"), default=False)
                if llm_result
                else deterministic_ok
            )
            confidence = self._confidence(
                llm_result.get("confidence"), deterministic_ok
            )
            unsupported = _as_string_list(llm_result.get("unsupported_claims"))
            llm_needs_more = _as_bool(
                llm_result.get("needs_more_evidence"), default=False
            )
            refined_query = str(llm_result.get("refined_query") or "").strip()
            notes = str(llm_result.get("notes") or "").strip()

            if fabricated:
                grounded = False
            if missing_citations:
                grounded = False

            is_valid = bool(grounded and deterministic_ok and not unsupported)
            needs_more = bool((llm_needs_more or not is_valid) and not abstained)
            validation = {
                "is_valid": is_valid,
                "grounded": bool(grounded),
                "confidence": confidence,
                "unsupported_claims": unsupported,
                "fabricated_citations": [
                    f"{filename}, p. {page}" for filename, page in fabricated
                ],
                "verified_citations": [
                    {"filename": filename, "page": page}
                    for filename, page in parsed
                    if (filename, page) in available
                ],
                "missing_citations": missing_citations,
                "needs_more_evidence": needs_more,
                "refined_query": refined_query or self._refined_query(query, intent),
                "notes": notes
                or (
                    "Deterministic citation checks passed; LLM validation was unavailable."
                    if llm_failed
                    else "Validation completed."
                ),
                "method": "llm+deterministic" if llm_result else "deterministic",
            }

        if not validation["is_valid"]:
            logger.warning(
                "Validation failed: fabricated=%s missing_citations=%s unsupported=%s",
                len(validation.get("fabricated_citations", [])),
                validation.get("missing_citations", False),
                len(validation.get("unsupported_claims", [])),
            )
        else:
            logger.info("Answer validated (%s)", validation["method"])

        return {
            "validation_result": validation,
            "validation_attempts": attempts,
            "citations": state.get("citations") or [],
            "error": "",
        }

    @staticmethod
    def _answer_to_validate(state: dict[str, Any]) -> str:
        for key in (
            "report_result",
            "comparison_result",
            "analysis_result",
            "direct_response",
            "final_response",
        ):
            value = str(state.get(key) or "").strip()
            if value:
                return value
        return ""

    @staticmethod
    def _confidence(value: Any, deterministic_ok: bool) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.75 if deterministic_ok else 0.25
        return round(max(0.0, min(1.0, confidence)), 2)

    @staticmethod
    def _refined_query(query: str, intent: str) -> str:
        base = query.strip() or "research paper evidence"
        extras = {
            "comparison": "methodology datasets results limitations",
            "literature_review": "research problem methodology findings limitations future work",
            "report": "problem methodology datasets findings limitations future work",
            "research_gaps": "limitations future work unexplored challenges",
        }.get(intent, "specific evidence methodology results limitations")
        return f"{base} {extras}"

    @staticmethod
    def _citation_report(
        available: set[tuple[str, int]],
        parsed: list[tuple[str, int]],
        fabricated: list[tuple[str, int]],
        missing_citations: bool,
    ) -> str:
        lines = ["Available evidence pages:"]
        if available:
            for filename, page in sorted(available)[:25]:
                lines.append(f"- {filename}, p. {page}")
        else:
            lines.append("- None")

        lines.append("Parsed inline citations:")
        if parsed:
            for filename, page in parsed[:25]:
                lines.append(f"- {filename}, p. {page}")
        else:
            lines.append("- None")

        if fabricated:
            lines.append("Citations NOT present in the evidence:")
            for filename, page in fabricated[:25]:
                lines.append(f"- {filename}, p. {page}")
        if missing_citations:
            lines.append("The answer contains no inline citations but evidence exists.")
        return "\n".join(lines)
