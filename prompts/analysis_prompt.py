"""Prompts for the Research Analysis Agent."""

from __future__ import annotations

from typing import Sequence

from prompts.rag_prompt import GROUNDED_SYSTEM

ANALYSIS_INSTRUCTIONS: dict[str, str] = {
    "summarization": """Produce a structured paper summary:
## Executive Summary
2-3 sentences.
## Research Problem
## Objectives
## Methodology
## Key Findings
## Conclusion
Keep each section evidence-led and cite the page behind each claim.""",
    "methodology": """Explain the research methodology step by step:
## Methodology Overview
## Model / System Architecture
## Algorithms and Techniques
## Dataset
## Evaluation Metrics
## Experimental Setup
## Reproducibility Notes
Cite the relevant page for every technical claim. If a detail is absent, say it is
not reported in the retrieved evidence.""",
    "dataset": """Extract dataset information:
## Datasets Used
Include name, size, domain, splits, construction, and licensing when reported.
Use a Markdown table if multiple datasets are described.
## Dataset Limitations
Cite every row/claim. State clearly when a detail is not reported.""",
    "algorithm": """Extract algorithms, models, and components:
## Algorithms and Models
For each: name, purpose, key configuration/hyperparameters.
## Training Procedure
## Baselines Compared
Cite pages. Do not infer an algorithm that is not explicitly described.""",
    "findings": """Extract findings, results, and contributions:
## Key Findings
## Quantitative Results
Use a table for metrics and scores when evidence supports it.
## Research Contributions
## Evidence Strength
Cite every number and claim. Do not calculate new numbers from the paper.""",
    "limitations": """Identify limitations reported by the authors and limitations
visible in the evidence:
## Author-Stated Limitations
## Evidence-Based Limitations
## Impact on Validity
Clearly label anything that is your inference rather than an author statement.
Cite relevant pages.""",
    "research_gaps": """Identify research gaps and opportunities:
## Evidence-Based Gaps
Gaps directly supported by limitations or missing evaluations in the papers.
## AI-Generated Research Opportunities
Potential directions inferred by you; label them explicitly as suggestions, not
findings from the papers.
## Suggested Next Steps
## Suggested Research Questions
Never present a suggestion as something the authors stated. Cite supporting evidence
for evidence-based gaps.""",
    "general_qa": """Answer the user's research question directly using the evidence.
Use headings only when they improve clarity, and cite every material claim.""",
}


def build_analysis_prompt(
    intent: str,
    query: str,
    context: str,
    history: Sequence[dict[str, str]] | None = None,
    document_names: Sequence[str] | None = None,
) -> str:
    """Build the analysis prompt for a given research intent."""
    instruction = ANALYSIS_INSTRUCTIONS.get(intent, ANALYSIS_INSTRUCTIONS["general_qa"])
    history_block = ""
    if history:
        lines = []
        for message in list(history)[-4:]:
            role = str(message.get("role", "user")).capitalize()
            content = str(message.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content[:500]}")
        history_block = "\nRecent conversation:\n" + "\n".join(lines) + "\n"

    docs_block = ""
    if document_names:
        docs_block = "Selected papers: " + ", ".join(document_names) + "\n"

    return f"""{GROUNDED_SYSTEM}

Task: {instruction}

{docs_block}Retrieved evidence:
-------------------
{context}
-------------------
{history_block}
User request: {query.strip()}

Produce the requested analysis in Markdown with page-level citations."""
