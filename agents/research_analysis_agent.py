"""Research Analysis Agent: summaries, methodology, findings, and gaps."""

from __future__ import annotations

import logging
from typing import Any

from agents.context_utils import get_history, get_paper_names, get_state_context
from generation.llm import TextGenerator, invoke_text
from prompts.analysis_prompt import build_analysis_prompt
from prompts.rag_prompt import INSUFFICIENT_EVIDENCE_MESSAGE
from utils.logger import get_logger

logger = get_logger("agents.analysis")


class ResearchAnalysisAgent:
    """Produce evidence-grounded analysis for a single paper or a corpus."""

    def __init__(self, llm: TextGenerator) -> None:
        self.llm = llm

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run the requested analysis task and update the shared state."""
        intent = str(state.get("research_intent", "general_qa"))
        query = str(state.get("user_query", "")).strip()
        context = get_state_context(state)

        if not context:
            return {
                "analysis_result": INSUFFICIENT_EVIDENCE_MESSAGE,
                "error": "",
            }

        prompt = build_analysis_prompt(
            intent=intent,
            query=query,
            context=context,
            history=get_history(state),
            document_names=get_paper_names(state),
        )

        try:
            answer = invoke_text(self.llm, prompt)
        except Exception as exc:
            logger.exception("Research analysis failed")
            return {
                "analysis_result": "",
                "error": f"Research analysis failed: {exc}",
            }

        logger.info("Completed analysis intent=%s (%s chars)", intent, len(answer))
        return {"analysis_result": answer, "error": ""}
