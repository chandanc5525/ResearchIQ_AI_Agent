"""Answer grounding and citation evaluation."""

from __future__ import annotations

from typing import Any, Sequence

from prompts.rag_prompt import INSUFFICIENT_EVIDENCE_MESSAGE
from utils.helpers import parse_citations


def evaluate_answer(
    answer: str,
    retrieved_documents: Sequence[dict[str, Any]],
    validation_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one generated answer for citation coverage and grounding."""
    available = {
        (str(chunk.get("filename", "")), int(chunk.get("page", 0)))
        for chunk in retrieved_documents
    }
    parsed = parse_citations(answer)
    verified = [pair for pair in parsed if pair in available]
    fabricated = [pair for pair in parsed if pair not in available]
    abstained = INSUFFICIENT_EVIDENCE_MESSAGE.lower() in (answer or "").lower()

    coverage = len(verified) / len(parsed) if parsed else (1.0 if abstained else 0.0)
    validation = validation_result or {}
    return {
        "answer_length": len(answer or ""),
        "citation_count": len(parsed),
        "verified_citations": len(verified),
        "fabricated_citations": len(fabricated),
        "citation_coverage": round(coverage, 4),
        "abstained": abstained,
        "has_citations": bool(parsed),
        "grounded": bool(validation.get("grounded", coverage == 1.0 and bool(parsed))),
        "is_valid": bool(validation.get("is_valid", coverage == 1.0 and bool(parsed))),
        "fabricated_pairs": [f"{filename}, p. {page}" for filename, page in fabricated],
    }


def evaluate_answers(
    answers: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate answer evaluation for a batch.

    Each item must contain ``answer`` and ``retrieved_documents`` and may contain
    ``validation_result``.
    """
    if not answers:
        return {"answers": 0, "mean_citation_coverage": 0.0, "valid_rate": 0.0}
    results = [
        evaluate_answer(
            str(item.get("answer", "")),
            item.get("retrieved_documents") or [],
            item.get("validation_result"),
        )
        for item in answers
    ]
    count = len(results)
    return {
        "answers": count,
        "mean_citation_coverage": round(
            sum(item["citation_coverage"] for item in results) / count, 4
        ),
        "valid_rate": round(sum(1 for item in results if item["is_valid"]) / count, 4),
        "fabricated_citation_rate": round(
            sum(1 for item in results if item["fabricated_citations"]) / count, 4
        ),
        "abstention_rate": round(
            sum(1 for item in results if item["abstained"]) / count, 4
        ),
        "details": results,
    }
