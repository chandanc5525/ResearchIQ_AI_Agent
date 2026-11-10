"""Compile and execute the ResearchIQ LangGraph workflow."""

from __future__ import annotations

import logging
from typing import Any, Iterator, Sequence

from langgraph.graph import END, START, StateGraph

from agents.citation_validation_agent import CitationValidationAgent
from agents.comparison_agent import ComparisonAgent
from agents.research_analysis_agent import ResearchAnalysisAgent
from agents.research_report_agent import ResearchReportAgent
from agents.retrieval_agent import RetrievalAgent
from agents.supervisor_agent import SupervisorAgent
from generation.llm import TextGenerator
from generation.response_generator import ResponseGenerator
from retrieval.retriever import ResearchRetriever
from utils.config import Settings
from utils.errors import WorkflowError
from utils.helpers import utc_now_iso
from utils.logger import get_logger
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
from workflows.nodes import ResearchNodes, WorkflowDependencies
from workflows.state import ResearchState, initial_state

logger = get_logger("workflow.graph")

RECURSION_LIMIT = 40


def build_research_graph(
    dependencies: WorkflowDependencies,
    *,
    checkpointer: Any | None = None,
) -> Any:
    """Build and compile the stateful multi-agent graph."""
    nodes = ResearchNodes(dependencies)
    graph = StateGraph(ResearchState)

    graph.add_node(SUPERVISOR, nodes.supervisor)
    graph.add_node(RETRIEVAL, nodes.retrieval)
    graph.add_node(ANALYSIS, nodes.analysis)
    graph.add_node(COMPARISON, nodes.comparison)
    graph.add_node(REPORT, nodes.report)
    graph.add_node(VALIDATION, nodes.validation)
    graph.add_node(DIRECT, nodes.direct)
    graph.add_node(FINALIZE, nodes.finalize)

    graph.add_edge(START, SUPERVISOR)

    graph.add_conditional_edges(
        SUPERVISOR,
        route_from_supervisor,
        {RETRIEVAL: RETRIEVAL, DIRECT: DIRECT, FINALIZE: FINALIZE},
    )
    graph.add_conditional_edges(
        RETRIEVAL,
        route_after_retrieval,
        {
            ANALYSIS: ANALYSIS,
            COMPARISON: COMPARISON,
            REPORT: REPORT,
            FINALIZE: FINALIZE,
        },
    )

    graph.add_conditional_edges(
        ANALYSIS,
        route_after_specialist,
        {VALIDATION: VALIDATION, FINALIZE: FINALIZE},
    )
    graph.add_conditional_edges(
        COMPARISON,
        route_after_specialist,
        {VALIDATION: VALIDATION, FINALIZE: FINALIZE},
    )
    graph.add_conditional_edges(
        REPORT,
        route_after_specialist,
        {VALIDATION: VALIDATION, FINALIZE: FINALIZE},
    )
    graph.add_edge(DIRECT, FINALIZE)

    graph.add_conditional_edges(
        VALIDATION,
        route_after_validation,
        {RETRIEVAL: RETRIEVAL, FINALIZE: FINALIZE},
    )
    graph.add_edge(FINALIZE, END)

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("Compiled ResearchIQ LangGraph workflow")
    return compiled


class ResearchIQWorkflow:
    """Application-facing wrapper around the compiled LangGraph workflow."""

    def __init__(
        self,
        settings: Settings,
        retriever: ResearchRetriever,
        llm: TextGenerator,
        *,
        graph: Any | None = None,
        dependencies: WorkflowDependencies | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.llm = llm

        if dependencies is None:
            response_generator = ResponseGenerator(llm)
            dependencies = WorkflowDependencies(
                settings=settings,
                supervisor=SupervisorAgent(llm, settings.max_history_messages),
                retrieval=RetrievalAgent(retriever, settings),
                analysis=ResearchAnalysisAgent(llm),
                comparison=ComparisonAgent(llm),
                validation=CitationValidationAgent(llm),
                report=ResearchReportAgent(llm),
                response_generator=response_generator,
            )
        self.dependencies = dependencies
        self.graph = graph or build_research_graph(dependencies)

    # ------------------------------------------------------------------
    # State construction and execution
    # ------------------------------------------------------------------
    def build_state(
        self,
        query: str,
        *,
        documents: Sequence[dict[str, Any]] | None = None,
        target_documents: Sequence[str] | None = None,
        chat_history: Sequence[dict[str, str]] | None = None,
        intent_hint: str | None = None,
    ) -> ResearchState:
        return initial_state(
            query,
            documents=documents,
            target_documents=target_documents,
            chat_history=chat_history,
            intent_hint=intent_hint,
            max_validation_iterations=self.settings.max_validation_iterations,
        )

    def _config(self, state: ResearchState) -> dict[str, Any]:
        return {
            "recursion_limit": RECURSION_LIMIT,
            "run_name": "ResearchIQ research query",
            "tags": [
                "researchiq",
                str(state.get("research_intent") or "pending"),
                str(state.get("intent_hint") or "chat"),
            ],
            "metadata": {
                "model": self.settings.ollama_model,
                "top_k": self.settings.retrieval_top_k,
                "documents": len(state.get("document_metadata") or []),
            },
        }

    def run_state(self, state: ResearchState) -> ResearchState:
        """Execute the graph once and return the final state."""
        try:
            result = self.graph.invoke(state, config=self._config(state))
            return result  # type: ignore[return-value]
        except Exception as exc:
            logger.exception("Research graph execution failed")
            failed = dict(state)
            failed.update(
                {
                    "error": f"Workflow execution failed: {exc}",
                    "final_response": (
                        "I could not complete the research workflow. "
                        f"{exc}\n\nCheck the Ollama connection and try again."
                    ),
                    "completed_at": utc_now_iso(),
                }
            )
            return failed  # type: ignore[return-value]

    def run(
        self,
        query: str,
        *,
        documents: Sequence[dict[str, Any]] | None = None,
        target_documents: Sequence[str] | None = None,
        chat_history: Sequence[dict[str, str]] | None = None,
        intent_hint: str | None = None,
    ) -> ResearchState:
        """Convenience entry point used by tests and CLI tools."""
        state = self.build_state(
            query,
            documents=documents,
            target_documents=target_documents,
            chat_history=chat_history,
            intent_hint=intent_hint,
        )
        return self.run_state(state)

    def stream_state(
        self, state: ResearchState
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield ``(node_name, update)`` pairs as the graph executes.

        The Streamlit layer merges these updates for live progress. Errors are
        surfaced as a finalize update rather than propagating to the UI.
        """
        try:
            for chunk in self.graph.stream(
                state,
                config=self._config(state),
                stream_mode="updates",
            ):
                if not isinstance(chunk, dict):
                    continue
                for node_name, update in chunk.items():
                    if node_name.startswith("__"):
                        continue
                    yield str(node_name), dict(update or {})
        except Exception as exc:
            logger.exception("Streaming research workflow failed")
            yield (
                FINALIZE,
                {
                    "error": f"Workflow execution failed: {exc}",
                    "final_response": (
                        "I could not complete the research workflow. "
                        f"{exc}\n\nCheck the Ollama connection and try again."
                    ),
                    "completed_at": utc_now_iso(),
                },
            )
