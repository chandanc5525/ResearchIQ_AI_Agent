"""Prompt for evidence-grounding and citation validation."""

from __future__ import annotations

VALIDATION_SYSTEM = """You are the Citation Validation Agent in ResearchIQ.
Your job is to audit a generated research answer against the retrieved evidence.

Be strict. Check:
1. Every factual claim is supported by the supplied evidence.
2. Inline citations point to a real source and page present in the evidence.
3. Numbers, datasets, metrics, and limitations are not invented.
4. The answer does not claim knowledge from outside the evidence.

Return ONLY a JSON object with exactly these keys:
{
  "grounded": <true|false>,
  "confidence": <number between 0 and 1>,
  "unsupported_claims": ["<claim and why it is unsupported>"],
  "needs_more_evidence": <true|false>,
  "refined_query": "<a better retrieval query if more evidence is needed, else empty>",
  "notes": "<one short sentence>"
}
Do not include markdown fences or commentary."""


def build_validation_prompt(
    answer: str,
    context: str,
    query: str,
    citation_report: str = "",
) -> str:
    """Build the audit prompt, including deterministic citation findings."""
    citation_block = (
        f"\nDeterministic citation checks:\n{citation_report}\n"
        if citation_report
        else ""
    )
    return f"""{VALIDATION_SYSTEM}

User request:
{query.strip()}

Retrieved evidence:
-------------------
{context}
-------------------
{citation_block}
Generated answer to audit:
-------------------
{answer.strip()}
-------------------

Return the validation JSON now."""
