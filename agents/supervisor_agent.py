"""Research Supervisor Agent: intent understanding and routing."""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

from generation.llm import TextGenerator, invoke_json
from prompts.supervisor_prompt import (
    INTENT_TO_AGENT,
    SUPPORTED_INTENTS,
    build_supervisor_prompt,
)
from utils.errors import LLMResponseError
from utils.logger import get_logger

logger = get_logger("agents.supervisor")

_INTENT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "comparison",
        ("compare", "comparison", " versus ", " vs ", "differences between"),
    ),
    (
        "literature_review",
        ("literature review", "related work", "survey of the papers"),
    ),
    ("report", ("report", "white paper", "briefing", "executive brief")),
    (
        "research_gaps",
        (
            "research gap",
            "gaps",
            "future research",
            "future work",
            "unexplored",
            "opportunit",
        ),
    ),
    ("limitations", ("limitation", "weakness", "shortcoming", "drawback")),
    ("dataset", ("dataset", "data set", "corpus", "benchmark data", "training data")),
    (
        "algorithm",
        (
            "algorithm",
            "model architecture",
            "which model",
            "neural network",
            "transformer",
            "classifier",
        ),
    ),
    (
        "methodology",
        (
            "methodology",
            "method",
            "approach",
            "experimental setup",
            "how did",
            "how does",
            "pipeline",
            "architecture",
        ),
    ),
    (
        "findings",
        ("finding", "result", "performance", "accuracy", "conclusion", "contribution"),
    ),
    (
        "summarization",
        (
            "summarize",
            "summarise",
            "summary",
            "explain this paper",
            "simple language",
            "overview",
        ),
    ),
)

_GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|thanks|thank you|good morning|good evening)[!. ]*$",
    re.IGNORECASE,
)
_DIRECT_PATTERNS = re.compile(
    r"(what can you do|how do you work|who are you|help me get started|capabilities)",
    re.IGNORECASE,
)


class SupervisorAgent:
    """Classify intent with the local LLM and fall back to deterministic rules."""

    def __init__(self, llm: TextGenerator, max_history_messages: int = 8) -> None:
        self.llm = llm
        self.max_history_messages = max_history_messages

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Analyse the request and return routing state updates."""
        query = str(state.get("user_query", "")).strip()
        documents = list(state.get("document_metadata") or [])
        provided_targets = [
            str(doc_id) for doc_id in (state.get("target_documents") or []) if doc_id
        ]
        hint = str(state.get("intent_hint") or "").strip() or None
        if hint not in SUPPORTED_INTENTS:
            hint = None

        if not query:
            return self._direct_update(
                query,
                "The request was empty. Please ask a question about your uploaded papers.",
                "Empty query",
            )

        decision: dict[str, Any] = {}
        if hint is not None:
            decision = self._decision_from_hint(
                hint, query, documents, provided_targets
            )
        else:
            try:
                raw = invoke_json(
                    self.llm,
                    build_supervisor_prompt(
                        query,
                        documents,
                        (state.get("chat_history") or [])[-self.max_history_messages :],
                    ),
                )
                decision = self._validate_llm_decision(
                    raw, query, documents, provided_targets
                )
            except LLMResponseError as exc:
                logger.warning(
                    "Supervisor LLM routing failed; using heuristics: %s", exc
                )
                decision = self._heuristic_decision(
                    query, documents, provided_targets, reason="Heuristic fallback"
                )

        intent = decision["research_intent"]
        selected_agent = decision["selected_agent"]
        targets = decision["target_documents"]

        if intent == "comparison" and len(documents) < 2:
            return self._direct_update(
                query,
                "Compare mode needs at least two indexed research papers. Upload another "
                "PDF, or ask a question about the paper you have.",
                "Comparison requires two papers",
            )
        if intent == "comparison" and len(targets) < 2:
            targets = [
                str(document.get("doc_id"))
                for document in documents
                if document.get("doc_id")
            ]

        requires_retrieval = intent != "direct" and bool(documents)
        if intent != "direct" and not documents:
            return self._direct_update(
                query,
                "No research papers are indexed yet. Upload one or more PDFs on the "
                "Document Upload page, then ask your research question.",
                "No documents available",
            )

        # Explicit UI targets are a hard constraint for analysis actions.
        if provided_targets and intent != "direct":
            targets = provided_targets

        logger.info(
            "Supervisor routed intent=%s agent=%s targets=%s (%s)",
            intent,
            selected_agent,
            targets,
            decision.get("routing_reason", ""),
        )
        return {
            "research_intent": intent,
            "selected_agent": selected_agent,
            "requires_retrieval": requires_retrieval,
            "retrieval_query": decision.get("retrieval_query") or query,
            "target_documents": targets,
            "report_requested": intent in {"report", "literature_review"},
            "literature_review": intent == "literature_review",
            "routing_reason": decision.get("routing_reason", ""),
            "error": "",
        }

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------
    def _decision_from_hint(
        self,
        hint: str,
        query: str,
        documents: Sequence[dict[str, Any]],
        provided_targets: Sequence[str],
    ) -> dict[str, Any]:
        targets = list(provided_targets) or self._match_documents(query, documents)
        if hint == "comparison" and len(documents) >= 2 and len(targets) < 2:
            targets = [
                str(document.get("doc_id"))
                for document in documents
                if document.get("doc_id")
            ]
        if hint == "report" or hint == "literature_review":
            if len(targets) < len(documents):
                targets = [
                    str(document.get("doc_id"))
                    for document in documents
                    if document.get("doc_id")
                ]
        return {
            "research_intent": hint,
            "selected_agent": INTENT_TO_AGENT[hint],
            "retrieval_query": self._build_retrieval_query(hint, query),
            "target_documents": targets,
            "routing_reason": f"UI action: {hint}",
        }

    def _validate_llm_decision(
        self,
        raw: dict[str, Any],
        query: str,
        documents: Sequence[dict[str, Any]],
        provided_targets: Sequence[str],
    ) -> dict[str, Any]:
        intent = str(raw.get("research_intent", "")).strip().lower()
        if intent not in SUPPORTED_INTENTS:
            logger.warning("Unsupported supervisor intent %r; using fallback", intent)
            return self._heuristic_decision(query, documents, provided_targets)

        valid_ids = {str(document.get("doc_id")) for document in documents}
        llm_targets = [
            str(doc_id)
            for doc_id in (raw.get("target_document_ids") or [])
            if str(doc_id) in valid_ids
        ]
        targets = (
            list(provided_targets)
            or llm_targets
            or self._match_documents(query, documents)
        )
        if (
            intent in {"comparison", "report", "literature_review"}
            and len(documents) > 1
        ):
            if len(targets) < 2:
                targets = [
                    str(document.get("doc_id"))
                    for document in documents
                    if document.get("doc_id")
                ]

        retrieval_query = str(raw.get("retrieval_query", "")).strip() or query
        return {
            "research_intent": intent,
            "selected_agent": INTENT_TO_AGENT[intent],
            "retrieval_query": self._build_retrieval_query(intent, retrieval_query),
            "target_documents": targets,
            "routing_reason": str(raw.get("reasoning", "")).strip()
            or "LLM classification",
        }

    def _heuristic_decision(
        self,
        query: str,
        documents: Sequence[dict[str, Any]],
        provided_targets: Sequence[str],
        reason: str = "Keyword routing",
    ) -> dict[str, Any]:
        lowered = f" {query.lower()} "
        intent = "general_qa"
        if _GREETING_PATTERNS.match(query.strip()) or _DIRECT_PATTERNS.search(query):
            intent = "direct"
        else:
            for candidate, keywords in _INTENT_KEYWORDS:
                if any(keyword in lowered for keyword in keywords):
                    intent = candidate
                    break
            if intent == "general_qa" and not documents:
                intent = "direct"

        targets = list(provided_targets) or self._match_documents(query, documents)
        if (
            intent in {"comparison", "report", "literature_review"}
            and len(documents) > 1
        ):
            targets = targets or [
                str(document.get("doc_id"))
                for document in documents
                if document.get("doc_id")
            ]
        return {
            "research_intent": intent,
            "selected_agent": INTENT_TO_AGENT[intent],
            "retrieval_query": self._build_retrieval_query(intent, query),
            "target_documents": targets,
            "routing_reason": reason,
        }

    @staticmethod
    def _match_documents(query: str, documents: Sequence[dict[str, Any]]) -> list[str]:
        """Match filenames or titles mentioned in the query."""
        lowered = query.lower()
        matched: list[str] = []
        for document in documents:
            doc_id = str(document.get("doc_id", ""))
            if not doc_id:
                continue
            filename = str(document.get("filename", "")).lower()
            stem = (
                filename.rsplit(".pdf", 1)[0] if filename.endswith(".pdf") else filename
            )
            title = str(document.get("title", "")).lower()
            if (
                (stem and len(stem) > 8 and stem in lowered)
                or (title and len(title) > 12 and title in lowered)
                or (doc_id and doc_id.lower() in lowered)
            ):
                matched.append(doc_id)
        return matched

    @staticmethod
    def _build_retrieval_query(intent: str, query: str) -> str:
        query = query.strip()
        if intent == "comparison" and len(query) < 80:
            return (
                f"{query} research objectives methodology datasets algorithms "
                "evaluation metrics experimental results limitations"
            )
        if intent in {"report", "literature_review"} and len(query) < 80:
            return (
                f"{query} research problem methodology datasets findings "
                "limitations future work"
            )
        if intent == "research_gaps":
            return f"{query} limitations future work unexplored challenges"
        return query

    @staticmethod
    def _direct_update(query: str, message: str, reason: str) -> dict[str, Any]:
        return {
            "research_intent": "direct",
            "selected_agent": "direct",
            "requires_retrieval": False,
            "retrieval_query": query,
            "target_documents": [],
            "direct_response": message,
            "final_response": message,
            "routing_reason": reason,
            "error": "",
        }
