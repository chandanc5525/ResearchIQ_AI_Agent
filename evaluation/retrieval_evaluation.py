"""Retrieval quality evaluation (hit rate, MRR, recall, precision)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from retrieval.retriever import ResearchRetriever


@dataclass
class RetrievalCase:
    """One labelled retrieval query."""

    query: str
    relevant_doc_ids: list[str]
    relevant_pages: list[int] | None = None


def load_retrieval_cases(path: Path) -> list[RetrievalCase]:
    """Load evaluation cases from a JSON file.

    Expected format::

        [{"query": "...", "relevant_doc_ids": ["doc_x"], "relevant_pages": [2]}]
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Retrieval evaluation file must contain a JSON list.")
    cases: list[RetrievalCase] = []
    for item in payload:
        if not isinstance(item, dict) or not item.get("query"):
            continue
        pages = item.get("relevant_pages")
        cases.append(
            RetrievalCase(
                query=str(item["query"]),
                relevant_doc_ids=[str(doc) for doc in item.get("relevant_doc_ids", [])],
                relevant_pages=[int(page) for page in pages] if pages else None,
            )
        )
    return cases


def evaluate_retrieval(
    retriever: ResearchRetriever,
    cases: Sequence[RetrievalCase],
    k: int = 5,
) -> dict[str, Any]:
    """Compute document-level retrieval metrics for labelled queries."""
    if not cases:
        return {
            "queries": 0,
            "hit_rate": 0.0,
            "mrr": 0.0,
            "recall": 0.0,
            "precision": 0.0,
        }

    hits = 0
    reciprocal_ranks: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []
    per_query: list[dict[str, Any]] = []

    for case in cases:
        chunks = retriever.search_for_evaluation(case.query, k=k)
        retrieved_docs = [
            str(chunk.get("doc_id", "")) for chunk in chunks if chunk.get("doc_id")
        ]
        unique_retrieved: list[str] = []
        for doc_id in retrieved_docs:
            if doc_id not in unique_retrieved:
                unique_retrieved.append(doc_id)

        relevant = set(case.relevant_doc_ids)
        first_rank = 0
        for rank, doc_id in enumerate(retrieved_docs, start=1):
            if doc_id in relevant:
                first_rank = rank
                break
        hit = first_rank > 0
        hits += int(hit)
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)

        found_relevant = [doc_id for doc_id in unique_retrieved if doc_id in relevant]
        recalls.append(len(found_relevant) / len(relevant) if relevant else 0.0)
        precisions.append(
            len(found_relevant) / len(unique_retrieved) if unique_retrieved else 0.0
        )
        per_query.append(
            {
                "query": case.query,
                "hit": hit,
                "first_relevant_rank": first_rank or None,
                "retrieved_doc_ids": unique_retrieved,
                "recall": round(recalls[-1], 4),
                "precision": round(precisions[-1], 4),
            }
        )

    count = len(cases)
    return {
        "queries": count,
        "k": k,
        "hit_rate": round(hits / count, 4),
        "mrr": round(sum(reciprocal_ranks) / count, 4),
        "recall": round(sum(recalls) / count, 4),
        "precision": round(sum(precisions) / count, 4),
        "per_query": per_query,
    }


def format_retrieval_report(metrics: dict[str, Any]) -> str:
    """Render metrics as a compact Markdown report."""
    return (
        "## Retrieval Evaluation\n\n"
        f"- Queries: {metrics.get('queries', 0)}\n"
        f"- Hit rate: {metrics.get('hit_rate', 0):.1%}\n"
        f"- MRR: {metrics.get('mrr', 0):.3f}\n"
        f"- Recall: {metrics.get('recall', 0):.1%}\n"
        f"- Precision: {metrics.get('precision', 0):.1%}\n"
    )


def cases_from_documents(
    documents: Iterable[dict[str, Any]], queries: Sequence[str]
) -> list[RetrievalCase]:
    """Build a simple smoke-evaluation set from indexed documents."""
    doc_ids = [
        str(document.get("doc_id")) for document in documents if document.get("doc_id")
    ]
    return [RetrievalCase(query=query, relevant_doc_ids=doc_ids) for query in queries]
