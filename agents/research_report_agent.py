"""Research Report Agent: literature reviews and structured reports."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from agents.context_utils import get_paper_names, get_state_context
from generation.llm import TextGenerator, invoke_text
from prompts.rag_prompt import INSUFFICIENT_EVIDENCE_MESSAGE
from prompts.report_prompt import build_report_prompt
from utils.logger import get_logger

logger = get_logger("agents.report")


class ResearchReportAgent:
    """Generate Markdown research reports grounded in retrieved evidence."""

    def __init__(self, llm: TextGenerator) -> None:
        self.llm = llm

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate a report and expose it through both report state fields."""
        context = get_state_context(state)
        if not context:
            return {
                "report_result": INSUFFICIENT_EVIDENCE_MESSAGE,
                "analysis_result": INSUFFICIENT_EVIDENCE_MESSAGE,
                "error": "",
            }

        paper_names = get_paper_names(state)
        literature_review = bool(state.get("literature_review")) or (
            state.get("research_intent") == "literature_review"
        )
        query = str(state.get("user_query", "")).strip() or "Synthesize the papers."
        prompt = build_report_prompt(
            query=query,
            context=context,
            paper_names=paper_names,
            locale_date=datetime.now().strftime("%Y-%m-%d"),
            literature_review=literature_review,
        )
        try:
            report = invoke_text(self.llm, prompt)
        except Exception as exc:
            logger.exception("Research report generation failed")
            return {
                "report_result": "",
                "analysis_result": "",
                "error": f"Report generation failed: {exc}",
            }

        logger.info(
            "Generated %s report from %s papers (%s chars)",
            "literature review" if literature_review else "research",
            len(paper_names),
            len(report),
        )
        return {
            "report_result": report,
            "analysis_result": report,
            "report_requested": True,
            "literature_review": literature_review,
            "error": "",
        }
