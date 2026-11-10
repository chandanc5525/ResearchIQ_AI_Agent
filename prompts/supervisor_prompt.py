"""Supervisor intent-classification prompt and routing table."""

from __future__ import annotations

from typing import Any, Sequence

SUPPORTED_INTENTS: tuple[str, ...] = (
    "general_qa",
    "summarization",
    "methodology",
    "dataset",
    "algorithm",
    "findings",
    "limitations",
    "research_gaps",
    "comparison",
    "literature_review",
    "report",
    "direct",
)

INTENT_TO_AGENT: dict[str, str] = {
    "general_qa": "research_analysis",
    "summarization": "research_analysis",
    "methodology": "research_analysis",
    "dataset": "research_analysis",
    "algorithm": "research_analysis",
    "findings": "research_analysis",
    "limitations": "research_analysis",
    "research_gaps": "research_analysis",
    "comparison": "comparison",
    "literature_review": "report",
    "report": "report",
    "direct": "direct",
}

SUPERVISOR_SYSTEM = """You are the Research Supervisor in ResearchIQ, an agentic research assistant.
Your job is to understand the user's intent and route the request to the correct specialist.

Allowed intents:
- general_qa: a question about the uploaded papers
- summarization: summarize a paper
- methodology: explain the research methodology
- dataset: identify datasets used
- algorithm: identify algorithms / models used
- findings: extract results and contributions
- limitations: identify limitations
- research_gaps: identify gaps and future work
- comparison: compare two or more papers
- literature_review: produce a literature review across papers
- report: generate a structured research report
- direct: greeting, meta-question, or request that needs no paper retrieval

Routing rules:
- Questions about the papers must retrieve evidence.
- Comparison requires at least two target papers.
- Literature reviews and reports need broad evidence across the selected papers.
- Only choose direct for greetings, capability questions, or small talk.

Return ONLY a JSON object with exactly these keys:
{
  "research_intent": "<one allowed intent>",
  "selected_agent": "<research_analysis|comparison|report|direct>",
  "requires_retrieval": <true|false>,
  "retrieval_query": "<concise standalone search query, keeping technical terms>",
  "target_document_ids": ["<doc ids from the list, or empty for all>"],
  "reasoning": "<one short sentence>"
}
Do not include markdown fences or commentary."""


def _format_documents(documents: Sequence[dict[str, Any]]) -> str:
    if not documents:
        return "No papers are indexed."
    lines = []
    for document in documents[:20]:
        title = document.get("title") or document.get("filename", "Untitled")
        lines.append(
            f"- id={document.get('doc_id')} | file={document.get('filename')} | "
            f"title={title}"
        )
    return "\n".join(lines)


def _format_history(history: Sequence[dict[str, str]], limit: int = 6) -> str:
    if not history:
        return "No previous conversation."
    recent = list(history)[-limit:]
    lines = []
    for message in recent:
        role = str(message.get("role", "user")).capitalize()
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content[:700]}")
    return "\n".join(lines) if lines else "No previous conversation."


def build_supervisor_prompt(
    query: str,
    documents: Sequence[dict[str, Any]],
    history: Sequence[dict[str, str]] | None = None,
    intent_hint: str | None = None,
) -> str:
    """Build the classification prompt for the supervisor agent."""
    hint_block = ""
    if intent_hint:
        hint_block = (
            f"\nThe user explicitly selected the '{intent_hint}' action in the UI. "
            f"Use that intent unless it is clearly impossible."
        )
    return f"""{SUPERVISOR_SYSTEM}

Indexed papers:
{_format_documents(documents)}

Recent conversation:
{_format_history(history or [])}
{hint_block}

User request:
{query.strip()}

Now return the routing JSON."""
