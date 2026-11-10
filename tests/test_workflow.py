"""Integration-style tests for the compiled LangGraph workflow."""

from __future__ import annotations

import json
from typing import Any

import pytest

from prompts.rag_prompt import INSUFFICIENT_EVIDENCE_MESSAGE
from tests.conftest import FakeLLM, FakeRetriever
from workflows.edges import (
    ANALYSIS,
    COMPARISON,
    DIRECT,
    FINALIZE,
    REPORT,
    RETRIEVAL,
    SUPERVISOR,
    VALIDATION,
    route_after_retrieval,
    route_after_specialist,
    route_after_validation,
    route_from_supervisor,
)
from workflows.research_graph import ResearchIQWorkflow

DOCUMENTS = [
    {"doc_id": "doc_alpha", "filename": "alpha.pdf", "title": "Alpha Study"},
    {"doc_id": "doc_beta", "filename": "beta.pdf", "title": "Beta Study"},
]


def _decision(
    intent: str = "general_qa",
    agent: str = "research_analysis",
    requires_retrieval: bool = True,
    targets: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "research_intent": intent,
            "selected_agent": agent,
            "requires_retrieval": requires_retrieval,
            "retrieval_query": "main findings methodology",
            "target_document_ids": targets or [],
            "reasoning": "test",
        }
    )


def _valid_validation() -> str:
    return json.dumps(
        {
            "grounded": True,
            "confidence": 0.9,
            "unsupported_claims": [],
            "needs_more_evidence": False,
            "refined_query": "",
            "notes": "Supported.",
        }
    )


def _make_handler(
    *,
    decision: str | None = None,
    analysis_text: str = "The model reaches 91% accuracy [Source: alpha.pdf, p. 2].",
    comparison_text: str = (
        "## Structured Comparison\n\n| Dimension | alpha.pdf | beta.pdf |\n"
        "| --- | --- | --- |\n"
        "| Method | Transformer [Source: alpha.pdf, p. 2] | GNN [Source: beta.pdf, p. 1] |"
    ),
    report_text: str = "# Research Report: Test\n\nFindings [Source: alpha.pdf, p. 2].",
    validation_handler: Any | None = None,
) -> Any:
    def handler(prompt: str) -> str:
        if "Research Supervisor" in prompt:
            return decision or _decision()
        if "Citation Validation Agent" in prompt:
            return (
                validation_handler(prompt)
                if validation_handler
                else _valid_validation()
            )
        if "Task: compare the selected research papers" in prompt:
            return comparison_text
        if "Task: generate a Research Report" in prompt:
            return report_text
        if "Task: generate a Literature Review" in prompt:
            return report_text.replace("Research Report", "Literature Review")
        if "Task:" in prompt and "Retrieved evidence" in prompt:
            return analysis_text
        if "professional research intelligence assistant" in prompt:
            return "Hello! Ask me anything about your indexed papers."
        return "Fallback response."

    return handler


def test_full_general_qa_workflow(settings, evidence) -> None:
    llm = FakeLLM(_make_handler())
    retriever = FakeRetriever(documents=evidence)
    workflow = ResearchIQWorkflow(settings, retriever, llm)

    state = workflow.run(
        "What accuracy did the transformer achieve?", documents=DOCUMENTS
    )

    assert state["research_intent"] == "general_qa"
    assert state["selected_agent"] == "research_analysis"
    assert state["iteration_count"] == 1
    assert state["validation_result"]["is_valid"] is True
    assert "91%" in state["analysis_result"]
    assert "91%" in state["final_response"]
    assert "[Source: alpha.pdf, p. 2]" in state["final_response"]
    assert state["citations"]
    assert state["error"] == ""


def test_streaming_exposes_every_agent_node(settings, evidence) -> None:
    llm = FakeLLM(_make_handler())
    workflow = ResearchIQWorkflow(settings, FakeRetriever(documents=evidence), llm)
    initial = workflow.build_state("What accuracy was achieved?", documents=DOCUMENTS)

    seen: list[str] = []
    merged = dict(initial)
    for node, update in workflow.stream_state(initial):
        seen.append(node)
        merged.update(update)

    assert seen[0] == SUPERVISOR
    assert RETRIEVAL in seen
    assert ANALYSIS in seen
    assert VALIDATION in seen
    assert seen[-1] == FINALIZE
    assert merged["final_response"]
    assert merged["validation_result"]["is_valid"] is True


def test_comparison_routing_and_per_document_retrieval(settings, evidence) -> None:
    llm = FakeLLM(_make_handler())
    retriever = FakeRetriever(documents=evidence)
    workflow = ResearchIQWorkflow(settings, retriever, llm)

    state = workflow.run(
        "Compare the methods used in these papers",
        documents=DOCUMENTS,
        target_documents=["doc_alpha", "doc_beta"],
        intent_hint="comparison",
    )

    assert state["research_intent"] == "comparison"
    assert state["selected_agent"] == "comparison"
    assert state["comparison_result"]
    assert "| Dimension |" in state["final_response"]
    assert retriever.calls[0]["per_document"] is True
    assert retriever.calls[0]["doc_ids"] == ["doc_alpha", "doc_beta"]


def test_report_routing(settings, evidence) -> None:
    llm = FakeLLM(_make_handler())
    workflow = ResearchIQWorkflow(settings, FakeRetriever(documents=evidence), llm)

    state = workflow.run(
        "Generate a report on the selected papers",
        documents=DOCUMENTS,
        target_documents=["doc_alpha", "doc_beta"],
        intent_hint="report",
    )

    assert state["research_intent"] == "report"
    assert state["report_requested"] is True
    assert state["report_result"].startswith("# Research Report")
    assert state["final_response"].startswith("# Research Report")


def test_direct_route_skips_retrieval(settings, evidence) -> None:
    decision = _decision(intent="direct", agent="direct", requires_retrieval=False)
    llm = FakeLLM(_make_handler(decision=decision))
    retriever = FakeRetriever(documents=evidence)
    workflow = ResearchIQWorkflow(settings, retriever, llm)

    state = workflow.run("Hello there", documents=DOCUMENTS)

    assert state["research_intent"] == "direct"
    assert state["final_response"].startswith("Hello!")
    assert retriever.calls == []
    assert state["retrieved_documents"] == []


def test_empty_retrieval_returns_required_fallback(settings) -> None:
    llm = FakeLLM(_make_handler())
    workflow = ResearchIQWorkflow(settings, FakeRetriever(documents=[]), llm)

    state = workflow.run("What is the main result?", documents=DOCUMENTS)

    assert state["final_response"].startswith(INSUFFICIENT_EVIDENCE_MESSAGE)
    assert state["retrieved_documents"] == []
    assert state["citations"] == []
    assert not any("Task:" in prompt for prompt in llm.prompts)


def test_validation_retry_routes_back_to_retrieval(settings, evidence) -> None:
    validation_calls = {"count": 0}
    analysis_calls = {"count": 0}

    def validation_handler(prompt: str) -> str:
        validation_calls["count"] += 1
        if validation_calls["count"] == 1:
            return json.dumps(
                {
                    "grounded": False,
                    "confidence": 0.3,
                    "unsupported_claims": ["Needs more evidence."],
                    "needs_more_evidence": True,
                    "refined_query": "specific benchmark accuracy evaluation",
                    "notes": "Retry retrieval.",
                }
            )
        return _valid_validation()

    base_handler = _make_handler(validation_handler=validation_handler)

    def handler(prompt: str) -> str:
        if "Task:" in prompt and "Retrieved evidence" in prompt:
            analysis_calls["count"] += 1
            return f"Draft {analysis_calls['count']} [Source: alpha.pdf, p. 2]."
        return base_handler(prompt)

    retriever = FakeRetriever(documents=evidence)
    workflow = ResearchIQWorkflow(settings, retriever, FakeLLM(handler))

    state = workflow.run("What accuracy was achieved?", documents=DOCUMENTS)

    assert len(retriever.calls) == 2
    assert analysis_calls["count"] == 2
    assert validation_calls["count"] == 2
    assert state["iteration_count"] == 2
    assert state["validation_attempts"] == 2
    assert "specific benchmark accuracy evaluation" in retriever.calls[1]["query"]
    assert "Draft 2" in state["final_response"]
    assert state["validation_result"]["is_valid"] is True


def test_validation_retry_is_capped_and_adds_caution(settings, evidence) -> None:
    def always_invalid(prompt: str) -> str:
        return json.dumps(
            {
                "grounded": False,
                "confidence": 0.2,
                "unsupported_claims": ["Unsupported claim."],
                "needs_more_evidence": True,
                "refined_query": "more evidence",
                "notes": "Still insufficient.",
            }
        )

    llm = FakeLLM(_make_handler(validation_handler=always_invalid))
    retriever = FakeRetriever(documents=evidence)
    workflow = ResearchIQWorkflow(settings, retriever, llm)

    state = workflow.run("What accuracy was achieved?", documents=DOCUMENTS)

    # settings.max_validation_iterations == 1 -> initial retrieval + one retry.
    assert len(retriever.calls) == 2
    assert state["iteration_count"] == 2
    assert state["validation_result"]["is_valid"] is False
    assert "Validation note" in state["final_response"]


def test_agent_failure_is_handled_gracefully(settings, evidence) -> None:
    llm = FakeLLM(lambda prompt: (_ for _ in ()).throw(RuntimeError("ollama down")))
    workflow = ResearchIQWorkflow(settings, FakeRetriever(documents=evidence), llm)

    state = workflow.run("What accuracy was achieved?", documents=DOCUMENTS)

    # Supervisor falls back to heuristics when its LLM call fails, then analysis
    # fails against the unavailable model. No exception escapes the graph.
    assert state["error"]
    assert "could not complete" in state["final_response"].lower()


def test_route_helpers() -> None:
    assert route_from_supervisor({"error": "x"}) == FINALIZE
    assert route_from_supervisor({"requires_retrieval": False}) == DIRECT
    assert (
        route_from_supervisor({"requires_retrieval": True, "document_metadata": []})
        == DIRECT
    )
    assert (
        route_from_supervisor({"requires_retrieval": True, "document_metadata": [{}]})
        == RETRIEVAL
    )

    assert route_after_retrieval({"error": "x"}) == FINALIZE
    assert route_after_retrieval({"retrieved_documents": []}) == FINALIZE
    assert (
        route_after_retrieval(
            {
                "retrieved_documents": [{}],
                "selected_agent": "comparison",
                "research_intent": "comparison",
            }
        )
        == COMPARISON
    )
    assert (
        route_after_retrieval(
            {
                "retrieved_documents": [{}],
                "selected_agent": "report",
                "research_intent": "report",
            }
        )
        == REPORT
    )
    assert (
        route_after_retrieval(
            {
                "retrieved_documents": [{}],
                "selected_agent": "research_analysis",
                "research_intent": "general_qa",
            }
        )
        == ANALYSIS
    )

    assert (
        route_after_validation(
            {
                "validation_result": {"needs_more_evidence": True},
                "iteration_count": 1,
                "max_validation_iterations": 1,
            }
        )
        == RETRIEVAL
    )
    assert (
        route_after_validation(
            {
                "validation_result": {"needs_more_evidence": True},
                "iteration_count": 2,
                "max_validation_iterations": 1,
            }
        )
        == FINALIZE
    )
    assert (
        route_after_validation({"validation_result": {"needs_more_evidence": False}})
        == FINALIZE
    )

    assert route_after_specialist({"error": "analysis failed"}) == FINALIZE
    assert route_after_specialist({"validation_result": {}}) == VALIDATION
