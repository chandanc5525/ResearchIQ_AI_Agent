"""Prompt for structured multi-paper comparison."""

from __future__ import annotations

from typing import Sequence

from prompts.rag_prompt import GROUNDED_SYSTEM


def build_comparison_prompt(
    query: str,
    context: str,
    paper_names: Sequence[str],
) -> str:
    """Build a comparison prompt that must include a structured table."""
    papers = ", ".join(paper_names) if paper_names else "the selected papers"
    return f"""{GROUNDED_SYSTEM}

Task: compare the selected research papers: {papers}.

Retrieved evidence:
-------------------
{context}
-------------------

User comparison request: {query.strip()}

Produce a rigorous comparative analysis in Markdown:

## Comparison Overview
Briefly state what is being compared.

## Structured Comparison
A Markdown table with exactly these columns:
| Dimension | {(' | '.join(paper_names)) if paper_names else 'Paper A | Paper B'} |
Rows (use "Not reported in retrieved evidence" when absent):
- Research objective
- Problem addressed
- Methodology
- Dataset(s)
- Algorithms / models
- Evaluation metrics
- Key results
- Limitations
Each cell should be short (roughly 25 words or fewer) and include a page citation
such as [Source: filename.pdf, p. 4] where possible.

## Similarities
## Key Differences
## Comparative Observations
## Limitations of This Comparison

Only compare the provided papers. Do not add outside papers. If evidence for a
dimension is missing, say so instead of filling gaps."""


def build_comparison_evidence_query(papers: Sequence[str]) -> str:
    """Build a broad query covering the main comparison dimensions."""
    names = ", ".join(papers) if papers else "the papers"
    return (
        f"Research objective, methodology, dataset, algorithms, evaluation metrics, "
        f"experimental results, and limitations of {names}"
    )
