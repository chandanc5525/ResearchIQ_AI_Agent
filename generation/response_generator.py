"""Generation helpers and final answer post-processing."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from generation.llm import TextGenerator, invoke_text
from prompts.rag_prompt import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    build_direct_prompt,
    build_grounded_answer_prompt,
)
from utils.errors import AgentExecutionError
from utils.helpers import strip_thinking, truncate_text

logger = logging.getLogger("researchiq.generation")


class ResponseGenerator:
    """Generate conversational and grounded answers with one configured LLM."""

    def __init__(self, llm: TextGenerator) -> None:
        self.llm = llm

    def grounded_answer(
        self,
        query: str,
        context: str,
        history: Sequence[dict[str, str]] | None = None,
    ) -> str:
        """Answer a question strictly from retrieved evidence."""
        prompt = build_grounded_answer_prompt(query, context, history)
        try:
            return invoke_text(self.llm, prompt)
        except Exception as exc:
            logger.exception("Grounded generation failed")
            if isinstance(exc, AgentExecutionError):
                raise
            raise AgentExecutionError(str(exc)) from exc

    def direct_answer(
        self,
        query: str,
        history: Sequence[dict[str, str]] | None = None,
        has_documents: bool = True,
        note: str = "",
    ) -> str:
        """Generate a response for non-retrieval messages."""
        prompt = build_direct_prompt(query, history, has_documents, note)
        try:
            return invoke_text(self.llm, prompt)
        except Exception as exc:
            logger.exception("Direct generation failed")
            raise AgentExecutionError(str(exc)) from exc

    @staticmethod
    def prepare_final_response(
        answer: str,
        citations: Sequence[dict[str, Any]],
        validation_result: dict[str, Any] | None = None,
        retrieval_note: str = "",
    ) -> str:
        """Clean an answer and add an honest validation caveat when needed."""
        cleaned = strip_thinking(answer or "")
        if not cleaned:
            return INSUFFICIENT_EVIDENCE_MESSAGE

        notes: list[str] = []
        validation = validation_result or {}
        if validation and not validation.get("is_valid", True):
            fabricated = validation.get("fabricated_citations") or []
            unsupported = validation.get("unsupported_claims") or []
            if fabricated:
                notes.append(
                    "Some citations could not be matched to retrieved evidence: "
                    + "; ".join(str(item) for item in fabricated[:3])
                )
            if unsupported:
                notes.append(
                    "Potentially unsupported statements: "
                    + "; ".join(
                        truncate_text(str(item), 180) for item in unsupported[:3]
                    )
                )
            if validation.get("notes"):
                notes.append(truncate_text(str(validation["notes"]), 240))
            if notes:
                cleaned += (
                    "\n\n> **Validation note:** "
                    + " ".join(notes)
                    + " Please verify these points against the source PDFs."
                )
        elif retrieval_note and "threshold" in retrieval_note.lower():
            cleaned += f"\n\n> **Retrieval note:** {retrieval_note}"

        if citations and not validation:
            logger.debug("Answer generated with %s available sources", len(citations))
        return cleaned.strip()
