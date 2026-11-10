"""Prompt for the Research Report Agent."""

from __future__ import annotations

from typing import Sequence

from prompts.rag_prompt import GROUNDED_SYSTEM


def build_report_prompt(
    query: str,
    context: str,
    paper_names: Sequence[str],
    locale_date: str,
    literature_review: bool = False,
) -> str:
    """Build a prompt for a structured Markdown research report."""
    papers = ", ".join(paper_names) if paper_names else "the uploaded papers"
    document_type = "Literature Review" if literature_review else "Research Report"
    extra_section = (
        "## Thematic Synthesis\nGroup papers by theme and compare their approaches."
        if literature_review
        else "## Methodology Comparison\nCompare approaches, datasets, and evaluation."
    )

    return f"""{GROUNDED_SYSTEM}

Task: generate a {document_type} across these papers: {papers}.
User focus: {query.strip()}
Date: {locale_date}

Retrieved evidence:
-------------------
{context}
-------------------

Write a well-structured Markdown {document_type} with this structure:

# {document_type}: <specific descriptive title>

## Executive Summary
## Scope and Papers Reviewed
List each paper and its central focus.
## Research Problems and Objectives
## Methodologies and Technical Approaches
## Datasets, Algorithms, and Evaluation
{extra_section}
## Key Findings
Use a table where useful.
## Limitations
Separate author-stated limitations from evidence-based observations.
## Research Gaps and Future Directions
Label AI-generated suggestions explicitly.
## Conclusion
## References
List source papers with the pages used.

Rules:
- Cite every substantive claim with [Source: <filename>, p. <page>].
- Do not cite a page that is not covered by the evidence.
- Do not invent metrics, datasets, or findings.
- Use clear academic prose; avoid filler.
- Keep the report under roughly 1800 words."""
