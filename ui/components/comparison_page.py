"""Paper Comparison page: multi-paper structured comparison."""

from __future__ import annotations

import streamlit as st

from services import ApplicationServices
from ui.components.common import (
    execute_workflow,
    format_timestamp,
    render_answer_meta,
    render_page_intro,
    render_source_list,
)
from ui.session import add_activity, request_navigation
from utils.helpers import utc_now_iso


def _render_result() -> None:
    result = st.session_state.get("comparison_result")
    if not result:
        return
    st.divider()
    st.markdown("### Comparison result")
    st.caption(
        f"{', '.join(result.get('papers', []))} · {format_timestamp(result.get('at', ''))}"
    )
    if result.get("error"):
        st.caption(f"Workflow note: {result['error']}")
    st.markdown(result["content"])
    render_answer_meta(result)

    columns = st.columns(2)
    with columns[0]:
        st.download_button(
            "Download comparison (Markdown)",
            data=result["content"],
            file_name="researchiq_comparison.md",
            mime="text/markdown",
            width="stretch",
        )
    with columns[1]:
        if st.button("Clear comparison", width="stretch"):
            st.session_state["comparison_result"] = None
            st.rerun()


def render(services: ApplicationServices) -> None:
    """Render the multi-paper comparison workspace."""
    render_page_intro(
        "Paper Comparison",
        "Select two or more papers and compare objectives, methodologies, datasets, "
        "algorithms, metrics, results, and limitations in a structured evidence table.",
    )

    documents = services.document_metadata()
    if len(documents) < 2:
        st.warning(
            "Paper comparison requires at least two indexed research papers. "
            "Upload another PDF to enable comparison mode."
        )
        if st.button("Open Document Upload", type="primary"):
            request_navigation("Document Upload")
        return

    choices = {document["doc_id"]: document["filename"] for document in documents}
    default_ids = list(choices)[:2]
    previous = st.session_state.get("comparison_result") or {}
    previous_targets = previous.get("target_documents") or default_ids
    default_ids = [doc_id for doc_id in previous_targets if doc_id in choices][:6]
    if len(default_ids) < 2:
        default_ids = list(choices)[:2]

    st.markdown("### Build a comparison")
    selected = st.multiselect(
        "Papers to compare",
        options=list(choices),
        default=default_ids,
        format_func=lambda doc_id: choices[doc_id],
        max_selections=6,
        key="comparison_papers",
    )
    focus = st.text_input(
        "Optional comparison focus",
        placeholder="e.g. Compare their evaluation strategies and limitations",
        key="comparison_focus",
    )
    render_source_list([choices[doc_id] for doc_id in selected])

    if selected and len(selected) < 2:
        st.info("Select at least two papers to run a comparison.")

    if st.button(
        "Compare selected papers",
        type="primary",
        disabled=len(selected) < 2,
        width="stretch",
    ):
        query = focus.strip() or (
            "Compare the selected research papers across objectives, methodology, "
            "datasets, algorithms, evaluation metrics, results, and limitations."
        )
        state = execute_workflow(
            services,
            query,
            target_documents=selected,
            intent_hint="comparison",
            status_label="Comparing papers across evidence…",
        )
        content = str(state.get("final_response") or "").strip()
        if not content:
            content = "The comparison did not produce a response. Check the Ollama connection."
        st.session_state["comparison_result"] = {
            "content": content,
            "papers": [choices[doc_id] for doc_id in selected],
            "target_documents": list(selected),
            "citations": state.get("citations") or [],
            "retrieved_documents": state.get("retrieved_documents") or [],
            "validation_result": state.get("validation_result") or {},
            "error": state.get("error") or "",
            "at": utc_now_iso(),
        }
        add_activity(
            "Compared papers", ", ".join(choices[doc_id] for doc_id in selected)
        )
        st.rerun()

    _render_result()
