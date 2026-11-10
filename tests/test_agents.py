"""Tests for supervisor routing and every specialist agent."""

from __future__ import annotations

import json

from agents.citation_validation_agent import CitationValidationAgent
from agents.comparison_agent import ComparisonAgent
from agents.research_analysis_agent import ResearchAnalysisAgent
from agents.research_report_agent import ResearchReportAgent
from agents.retrieval_agent import RetrievalAgent
from agents.supervisor_agent import SupervisorAgent
from prompts.rag_prompt import INSUFFICIENT_EVIDENCE_MESSAGE
from tests.conftest import FakeLLM, FakeRetriever

DOCUMENTS = [
    {"doc_id": "doc_alpha", "filename": "alpha.pdf", "title": "Alpha Study"},
    {"doc_id": "doc_beta", "filename": "beta.pdf", "title": "Beta Study"},
]


def _supervisor_json(**overrides) -> str:
    payload = {
        "research_intent": "general_qa",
        "selected_agent": "research_analysis",
        "requires_retrieval": True,
        "retrieval_query": "main findings",
        "target_document_ids": [],
        "reasoning": "Paper question",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_supervisor_uses_llm_classification() -> None:
    llm = FakeLLM(
        _supervisor_json(
            research_intent="comparison",
            selected_agent="comparison",
            retrieval_query="compare methods",
        )
    )
    agent = SupervisorAgent(llm)
    updates = agent.run(
        {
            "user_query": "Compare the methods in both papers",
            "document_metadata": DOCUMENTS,
            "target_documents": [],
            "chat_history": [],
        }
    )

    assert updates["research_intent"] == "comparison"
    assert updates["selected_agent"] == "comparison"
    assert updates["requires_retrieval"] is True
    assert set(updates["target_documents"]) == {"doc_alpha", "doc_beta"}
    assert "compare" in updates["retrieval_query"].lower()


def test_supervisor_respects_explicit_ui_intent_without_llm() -> None:
    llm = FakeLLM(
        lambda prompt: (_ for _ in ()).throw(AssertionError("LLM should not run"))
    )
    agent = SupervisorAgent(llm)

    updates = agent.run(
        {
            "user_query": "Explain how they did it",
            "document_metadata": DOCUMENTS[:1],
            "target_documents": ["doc_alpha"],
            "chat_history": [],
            "intent_hint": "methodology",
        }
    )

    assert updates["research_intent"] == "methodology"
    assert updates["selected_agent"] == "research_analysis"
    assert updates["target_documents"] == ["doc_alpha"]
    assert llm.prompts == []


def test_supervisor_falls_back_to_keyword_routing_on_bad_json() -> None:
    llm = FakeLLM("This is not JSON.")
    agent = SupervisorAgent(llm)

    updates = agent.run(
        {
            "user_query": "Please summarize this paper",
            "document_metadata": DOCUMENTS[:1],
            "target_documents": [],
            "chat_history": [],
        }
    )

    assert updates["research_intent"] == "summarization"
    assert updates["selected_agent"] == "research_analysis"
    assert len(llm.prompts) >= 2  # initial attempt + JSON retry


def test_supervisor_direct_when_no_documents() -> None:
    agent = SupervisorAgent(FakeLLM("hello"))
    updates = agent.run(
        {
            "user_query": "What are the main findings?",
            "document_metadata": [],
            "target_documents": [],
            "chat_history": [],
        }
    )

    assert updates["research_intent"] == "direct"
    assert updates["requires_retrieval"] is False
    assert "No research papers" in updates["final_response"]


def test_supervisor_blocks_comparison_with_one_paper() -> None:
    agent = SupervisorAgent(FakeLLM("unused"))
    updates = agent.run(
        {
            "user_query": "Compare these findings",
            "document_metadata": DOCUMENTS[:1],
            "target_documents": ["doc_alpha"],
            "chat_history": [],
            "intent_hint": "comparison",
        }
    )

    assert updates["research_intent"] == "direct"
    assert "two indexed" in updates["final_response"].lower()


def test_retrieval_agent_updates_evidence_state(settings, evidence) -> None:
    retriever = FakeRetriever(documents=evidence)
    agent = RetrievalAgent(retriever, settings)

    updates = agent.run(
        {
            "research_intent": "general_qa",
            "retrieval_query": "transformer accuracy",
            "user_query": "What accuracy was reached?",
            "target_documents": ["doc_alpha", "doc_beta"],
            "iteration_count": 0,
        }
    )

    assert updates["iteration_count"] == 1
    assert len(updates["retrieved_documents"]) == 2
    assert updates["retrieval_context"]
    assert updates["citations"]
    assert retriever.calls[0]["doc_ids"] == ["doc_alpha", "doc_beta"]
    assert retriever.calls[0]["per_document"] is False


def test_retrieval_agent_uses_per_document_mode_for_comparison(
    settings, evidence
) -> None:
    retriever = FakeRetriever(documents=evidence)
    agent = RetrievalAgent(retriever, settings)
    agent.run(
        {
            "research_intent": "comparison",
            "retrieval_query": "compare methods",
            "user_query": "Compare",
            "target_documents": ["doc_alpha", "doc_beta"],
            "iteration_count": 0,
        }
    )
    assert retriever.calls[0]["per_document"] is True


def test_analysis_agent_generates_cited_answer(settings, evidence) -> None:
    llm = FakeLLM(
        "## Key Findings\n\n"
        "The transformer reaches 91% accuracy [Source: alpha.pdf, p. 2]."
    )
    agent = ResearchAnalysisAgent(llm)
    updates = agent.run(
        {
            "research_intent": "findings",
            "user_query": "What are the key findings?",
            "retrieved_documents": evidence,
            "document_metadata": DOCUMENTS,
            "target_documents": ["doc_alpha"],
            "chat_history": [],
        }
    )

    assert "91%" in updates["analysis_result"]
    assert updates["error"] == ""
    assert "Retrieved evidence" in llm.prompts[0]
    assert "alpha.pdf" in llm.prompts[0]


def test_analysis_agent_abstains_without_context() -> None:
    llm = FakeLLM("should not be called")
    agent = ResearchAnalysisAgent(llm)
    updates = agent.run(
        {
            "research_intent": "summarization",
            "user_query": "Summarize",
            "retrieved_documents": [],
            "document_metadata": DOCUMENTS,
        }
    )

    assert updates["analysis_result"] == INSUFFICIENT_EVIDENCE_MESSAGE
    assert llm.prompts == []


def test_comparison_agent_requires_two_papers(settings, evidence) -> None:
    llm = FakeLLM("unused")
    agent = ComparisonAgent(llm)
    updates = agent.run(
        {
            "user_query": "Compare",
            "retrieved_documents": evidence,
            "document_metadata": DOCUMENTS[:1],
            "target_documents": ["doc_alpha"],
        }
    )
    assert "at least two" in updates["comparison_result"].lower()
    assert llm.prompts == []


def test_comparison_agent_generates_structured_output(settings, evidence) -> None:
    llm = FakeLLM(
        "## Structured Comparison\n\n"
        "| Dimension | alpha.pdf | beta.pdf |\n"
        "| --- | --- | --- |\n"
        "| Methodology | Transformer [Source: alpha.pdf, p. 2] | GNN [Source: beta.pdf, p. 1] |"
    )
    agent = ComparisonAgent(llm)
    updates = agent.run(
        {
            "user_query": "Compare methodology and results",
            "retrieved_documents": evidence,
            "document_metadata": DOCUMENTS,
            "target_documents": ["doc_alpha", "doc_beta"],
        }
    )

    assert "Structured Comparison" in updates["comparison_result"]
    assert "| Dimension |" in updates["comparison_result"]
    assert "| alpha.pdf | beta.pdf |" in llm.prompts[0]


def test_report_agent_sets_report_state(settings, evidence) -> None:
    llm = FakeLLM(
        "# Research Report: Test\n\n## Executive Summary\nEvidence [Source: alpha.pdf, p. 2]."
    )
    agent = ResearchReportAgent(llm)
    updates = agent.run(
        {
            "user_query": "Generate a report",
            "research_intent": "report",
            "retrieved_documents": evidence,
            "document_metadata": DOCUMENTS,
            "target_documents": ["doc_alpha", "doc_beta"],
        }
    )

    assert updates["report_result"].startswith("# Research Report")
    assert updates["analysis_result"] == updates["report_result"]
    assert updates["report_requested"] is True


def test_validation_accepts_supported_citations(evidence) -> None:
    llm = FakeLLM(
        json.dumps(
            {
                "grounded": True,
                "confidence": 0.92,
                "unsupported_claims": [],
                "needs_more_evidence": False,
                "refined_query": "",
                "notes": "All claims are supported.",
            }
        )
    )
    agent = CitationValidationAgent(llm)
    updates = agent.run(
        {
            "user_query": "What accuracy?",
            "research_intent": "findings",
            "analysis_result": "The model reaches 91% [Source: alpha.pdf, p. 2].",
            "retrieved_documents": evidence,
            "citations": evidence,
        }
    )

    result = updates["validation_result"]
    assert result["is_valid"] is True
    assert result["fabricated_citations"] == []
    assert result["verified_citations"] == [{"filename": "alpha.pdf", "page": 2}]


def test_validation_rejects_fabricated_page(evidence) -> None:
    llm = FakeLLM(
        json.dumps(
            {
                "grounded": True,
                "confidence": 0.9,
                "unsupported_claims": [],
                "needs_more_evidence": False,
                "refined_query": "",
                "notes": "ok",
            }
        )
    )
    agent = CitationValidationAgent(llm)
    updates = agent.run(
        {
            "user_query": "What accuracy?",
            "research_intent": "findings",
            "analysis_result": "A fabricated claim [Source: ghost.pdf, p. 9].",
            "retrieved_documents": evidence,
        }
    )
    result = updates["validation_result"]

    assert result["is_valid"] is False
    assert result["fabricated_citations"] == ["ghost.pdf, p. 9"]
    assert result["needs_more_evidence"] is True
    assert result["refined_query"]


def test_validation_requests_retrieval_when_unsupported(evidence) -> None:
    llm = FakeLLM(
        json.dumps(
            {
                "grounded": False,
                "confidence": 0.42,
                "unsupported_claims": ["The answer overstates generalization."],
                "needs_more_evidence": True,
                "refined_query": "specific benchmark accuracy and evaluation setup",
                "notes": "Needs stronger evidence.",
            }
        )
    )
    agent = CitationValidationAgent(llm)
    updates = agent.run(
        {
            "user_query": "Does it generalize?",
            "research_intent": "findings",
            "analysis_result": "It generalizes to every domain [Source: alpha.pdf, p. 2].",
            "retrieved_documents": evidence,
        }
    )
    result = updates["validation_result"]

    assert result["is_valid"] is False
    assert result["needs_more_evidence"] is True
    assert result["refined_query"].startswith("specific benchmark")


def test_validation_falls_back_to_deterministic_checks(evidence) -> None:
    llm = FakeLLM("not json")
    agent = CitationValidationAgent(llm)
    updates = agent.run(
        {
            "user_query": "What accuracy?",
            "research_intent": "findings",
            "analysis_result": "The model reaches 91% [Source: alpha.pdf, p. 2].",
            "retrieved_documents": evidence,
        }
    )
    result = updates["validation_result"]

    assert result["is_valid"] is True
    assert result["method"] == "deterministic"


def test_validation_accepts_honest_abstention() -> None:
    agent = CitationValidationAgent(FakeLLM("unused"))
    updates = agent.run(
        {
            "user_query": "What is the moon made of?",
            "research_intent": "general_qa",
            "analysis_result": INSUFFICIENT_EVIDENCE_MESSAGE,
            "retrieved_documents": [
                {
                    "filename": "alpha.pdf",
                    "page": 1,
                    "content": "Unrelated paper content.",
                    "doc_id": "doc_alpha",
                    "chunk_id": "c1",
                }
            ],
        }
    )
    assert updates["validation_result"]["is_valid"] is True
    assert updates["validation_result"]["needs_more_evidence"] is False
