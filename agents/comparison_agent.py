"""Research Comparison Agent: structured multi-paper comparison."""

from __future__ import annotations

import logging
from typing import Any

from agents.context_utils import get_paper_names, get_state_context
from generation.llm import TextGenerator, invoke_text
from prompts.rag_prompt import INSUFFICIENT_EVIDENCE_MESSAGE
from utils.helpers import truncate_text
from utils.logger import get_logger

logger = get_logger("agents.comparison")


class ComparisonAgent:
    """Compare objectives, methods, datasets, results, and limitations."""

    def __init__(self, llm: TextGenerator) -> None:
        self.llm = llm

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate a cited comparison table and narrative."""
        paper_names = get_paper_names(state)
        if len(paper_names) < 2:
            message = (
                "Comparison requires at least two selected research papers. "
                "Select two or more PDFs on the Paper Comparison page."
            )
            return {
                "comparison_result": message,
                "final_response": message,
                "error": "",
            }

        context = get_state_context(state)
        if not context:
            return {
                "comparison_result": INSUFFICIENT_EVIDENCE_MESSAGE,
                "error": "",
            }

        from prompts.comparison_prompt import build_comparison_prompt

        prompt = build_comparison_prompt(
            query=str(state.get("user_query", "")),
            context=context,
            paper_names=paper_names,
        )
        try:
            answer = invoke_text(self.llm, prompt)
        except Exception as exc:
            logger.exception("Paper comparison failed")
            return {
                "comparison_result": "",
                "error": f"Paper comparison failed: {exc}",
            }

        logger.info("Compared %s papers (%s chars)", len(paper_names), len(answer))
        return {"comparison_result": answer, "error": ""}
