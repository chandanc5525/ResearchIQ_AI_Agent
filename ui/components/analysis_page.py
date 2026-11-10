"""Research Analysis page: summaries, methods, findings, and gaps."""

from __future__ import annotations

from typing import Any

import streamlit as st

from services import ApplicationServices
from ui.components.common import (
    execute_workflow,
    format_timestamp,
    render_answer_meta,
    render_empty_documents,
    render_page_intro,
    render_source_list,
)
from ui.session import add_activity, request_navigation
from utils.helpers import utc_now_iso

ANALYSIS_ACTIONS: list[dict[str, str]] = [
    {
        "intent": "summarization",
        "title": "Summarize research paper",
        "description": "Executive summary, problem, objectives, methodology, findings, conclusion.",
        "query": "Summarize this research paper with a structured executive summary.",
    },
    {
        "intent": "methodology",
        "title": "Analyze methodology",
        "description": "Step-by-step methodology, architecture, algorithms, setup, and metrics.",
        "query": "Explain the research methodology step by step.",
    },
    {
        "intent": "dataset",
        "title": "Extract datasets",
        "description": "Dataset names, size, splits, construction, and limitations.",
        "query": "Extract and describe all datasets used in this research.",
    },
    {
        "intent": "algorithm",
        "title": "Extract algorithms",
        "description": "Models, algorithms, components, baselines, and training procedure.",
        "query": "Identify the algorithms and models used, including baselines.",
    },
    {
        "intent": "findings",
        "title": "Identify findings",
        "description": "Key findings, quantitative results, and research contributions.",
        "query": "Extract the key findings, quantitative results, and contributions.",
    },
    {
        "intent": "limitations",
        "title": "Identify limitations",
        "description": "Author-stated limitations and evidence-based validity concerns.",
        "query": "Identify the limitations of this research.",
    },
    {
        "intent": "research_gaps",
        "title": "Research gaps",
        "description": "Evidence-based gaps, opportunities, and suggested directions.",
        "query": "Identify research gaps, unexplored areas, and future directions.",
    },
]


def _scope(services: ApplicationServices) -> tuple[list[str], str]:
    documents = services.document_metadata()
    all_ids = [document["doc_id"] for document in documents]
    labels = {document["doc_id"]: document["filename"] for document in documents}
    options = ["__all__", *all_ids]
    selected = st.selectbox(
        "Analysis scope",
        options=options,
        format_func=lambda value: (
            f"All uploaded papers ({len(all_ids)})"
            if value == "__all__"
            else labels[value]
        ),
        key="analysis_scope",
    )
    if selected == "__all__":
        return all_ids, f"all {len(all_ids)} papers"
    return [selected], labels[selected]


def _run_action(
    services: ApplicationServices,
    action: dict[str, str],
    target_ids: list[str],
    scope_label: str,
) -> None:
    query = f"{action['query']} Target: {scope_label}."
    state = execute_workflow(
        services,
        query,
        target_documents=target_ids,
        intent_hint=action["intent"],
        status_label=f"Running {action['title'].lower()}…",
    )
    content = str(state.get("final_response") or "").strip()
    if not content:
        content = (
            "The analysis did not produce a response. Check the Ollama connection."
        )
    st.session_state["analysis_result"] = {
        "content": content,
        "title": action["title"],
        "intent": action["intent"],
        "scope": scope_label,
        "citations": state.get("citations") or [],
        "retrieved_documents": state.get("retrieved_documents") or [],
        "validation_result": state.get("validation_result") or {},
        "error": state.get("error") or "",
        "at": utc_now_iso(),
    }
    add_activity(action["title"], scope_label)
    st.rerun()


def _render_result() -> None:
    result = st.session_state.get("analysis_result")
    if not result:
        return
    st.divider()
    st.markdown(f"### {result['title']}")
    st.caption(
        f"{result['scope']} · {format_timestamp(result['at'])} · "
        f"intent: {result['intent'].replace('_', ' ')}"
    )
    if result.get("error"):
        st.caption(f"Workflow note: {result['error']}")
    st.markdown(result["content"])
    render_answer_meta(result)
    if st.button("Clear analysis result", key="clear_analysis"):
        st.session_state["analysis_result"] = None
        st.rerun()


def render(services: ApplicationServices) -> None:
    """Render the research analysis action grid."""
    render_page_intro(
        "Research Analysis",
        "Run focused analyses on one paper or your whole corpus. Each action is "
        "routed through the LangGraph supervisor to the Research Analysis Agent and "
        "audited by the Citation Validation Agent.",
    )

    documents = services.document_metadata()
    if not documents:
        render_empty_documents(
            "Upload at least one research paper before running an analysis."
        )
        if st.button("Open Document Upload", type="primary", key="analysis_upload"):
            request_navigation("Document Upload")
        return

    target_ids, scope_label = _scope(services)
    render_source_list(
        [document["filename"] for document in documents]
        if len(target_ids) == len(documents)
        else [scope_label]
    )

    st.markdown("### Analysis actions")
    columns = st.columns(2)
    for index, action in enumerate(ANALYSIS_ACTIONS):
        with columns[index % 2]:
            with st.container(border=True):
                st.markdown(f"**{action['title']}**")
                st.caption(action["description"])
                if st.button(
                    "Run analysis",
                    key=f"analysis_{action['intent']}",
                    width="stretch",
                ):
                    _run_action(services, action, target_ids, scope_label)

    _render_result()
