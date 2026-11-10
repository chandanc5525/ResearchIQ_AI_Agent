"""Prompts for grounded question answering and direct conversation."""

from __future__ import annotations

from typing import Sequence

INSUFFICIENT_EVIDENCE_MESSAGE = "I could not find sufficient supporting information in the uploaded research papers."

GROUNDED_SYSTEM = """You are ResearchIQ, a rigorous research assistant answering questions
strictly from evidence retrieved from uploaded academic papers.

Non-negotiable rules:
1. Use ONLY the evidence supplied below. Do not use outside knowledge for factual claims.
2. Every material claim must carry an inline citation in this exact format:
   [Source: <filename>, p. <page>]
3. Never invent page numbers, paper titles, datasets, metrics, or citations.
4. If the evidence does not answer the question, reply exactly:
   "I could not find sufficient supporting information in the uploaded research papers."
5. Distinguish what the paper states from your own interpretation.
6. Prefer precise, well-structured Markdown. Be concise but complete.
7. If evidence from different papers conflicts, say so explicitly.
"""


def build_grounded_answer_prompt(
    query: str,
    context: str,
    history: Sequence[dict[str, str]] | None = None,
) -> str:
    """Build a retrieval-grounded answer prompt."""
    history_block = ""
    if history:
        lines = []
        for message in list(history)[-6:]:
            role = str(message.get("role", "user")).capitalize()
            content = str(message.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content[:600]}")
        history_block = "\nRecent conversation:\n" + "\n".join(lines) + "\n"

    return f"""{GROUNDED_SYSTEM}

Retrieved evidence:
-------------------
{context}
-------------------
{history_block}
User question: {query.strip()}

Answer in Markdown with inline citations. If the evidence is insufficient, use the
required insufficient-evidence sentence instead of guessing."""


def build_direct_prompt(
    query: str,
    history: Sequence[dict[str, str]] | None = None,
    has_documents: bool = True,
    note: str = "",
) -> str:
    """Prompt for greetings/meta-questions that do not need retrieval."""
    history_block = ""
    if history:
        lines = []
        for message in list(history)[-6:]:
            role = str(message.get("role", "user")).capitalize()
            content = str(message.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content[:500]}")
        history_block = "\nRecent conversation:\n" + "\n".join(lines) + "\n"

    availability = (
        "The user has research papers indexed."
        if has_documents
        else "The user has not uploaded any research papers yet."
    )
    note_block = f"\nContext note: {note}" if note else ""
    return f"""You are ResearchIQ, a professional research intelligence assistant.
{availability}{note_block}
Do not invent research findings or citations. For greetings and capability questions,
answer briefly and suggest useful research tasks. If a research question cannot be
answered because no papers are indexed, ask the user to upload PDFs.
{history_block}
User message: {query.strip()}

Respond in clear, friendly Markdown."""
