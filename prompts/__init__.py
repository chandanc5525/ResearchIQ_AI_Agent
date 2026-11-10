"""Prompt templates, kept out of business logic."""

from prompts.analysis_prompt import ANALYSIS_INSTRUCTIONS, build_analysis_prompt
from prompts.comparison_prompt import build_comparison_prompt
from prompts.rag_prompt import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    build_direct_prompt,
    build_grounded_answer_prompt,
)
from prompts.report_prompt import build_report_prompt
from prompts.supervisor_prompt import (
    INTENT_TO_AGENT,
    SUPPORTED_INTENTS,
    build_supervisor_prompt,
)
from prompts.validation_prompt import build_validation_prompt

__all__ = [
    "ANALYSIS_INSTRUCTIONS",
    "INSUFFICIENT_EVIDENCE_MESSAGE",
    "INTENT_TO_AGENT",
    "SUPPORTED_INTENTS",
    "build_analysis_prompt",
    "build_comparison_prompt",
    "build_direct_prompt",
    "build_grounded_answer_prompt",
    "build_report_prompt",
    "build_supervisor_prompt",
    "build_validation_prompt",
]
