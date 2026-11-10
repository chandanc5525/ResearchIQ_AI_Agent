"""Research Reports page: report generation and Markdown export."""

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
from utils.report_export import (
    list_saved_reports,
    report_title_from_markdown,
    save_markdown_report,
)


def _render_generated() -> None:
    result = st.session_state.get("report_result")
    if not result:
        return
    st.divider()
    st.markdown("### Generated report")
    st.caption(
        f"{result.get('report_type', 'Research report')} · "
        f"{format_timestamp(result.get('at', ''))} · {len(result.get('papers', []))} paper(s)"
    )
    if result.get("saved_path"):
        st.caption(f"Saved locally to `{result['saved_path']}`")
    if result.get("error"):
        st.warning(result["error"])
    st.markdown(result["content"])
    render_answer_meta(result)

    columns = st.columns(2)
    with columns[0]:
        st.download_button(
            "Download report (Markdown)",
            data=result["content"],
            file_name=result.get("download_name", "researchiq_report.md"),
            mime="text/markdown",
            width="stretch",
        )
    with columns[1]:
        if st.button("Clear report", width="stretch"):
            st.session_state["report_result"] = None
            st.rerun()


def _render_saved(services: ApplicationServices) -> None:
    saved = list_saved_reports(services.settings)
    st.markdown("### Saved reports")
    if not saved:
        st.caption(
            "Generated reports are saved automatically to `data/processed/reports/`."
        )
        return

    selected_path = st.selectbox(
        "Saved Markdown reports",
        options=saved,
        format_func=lambda path: path.name,
        key="saved_report_select",
    )
    content = selected_path.read_text(encoding="utf-8")
    with st.expander("Preview saved report", expanded=False):
        st.markdown(content)
    st.download_button(
        "Download selected report",
        data=content,
        file_name=selected_path.name,
        mime="text/markdown",
        key="download_saved_report",
    )


def render(services: ApplicationServices) -> None:
    """Render the report generation workspace."""
    render_page_intro(
        "Research Reports",
        "Generate a cited research report or literature review across the selected "
        "papers. Reports are written in Markdown, validated, and saved locally.",
    )

    documents = services.document_metadata()
    if not documents:
        st.warning("Upload research papers before generating a report.")
        if st.button("Open Document Upload", type="primary", key="reports_upload"):
            request_navigation("Document Upload")
        return

    choices = {document["doc_id"]: document["filename"] for document in documents}
    all_ids = list(choices)

    st.markdown("### Report configuration")
    report_type = st.selectbox(
        "Report type",
        options=["Research report", "Literature review"],
        key="report_type",
    )
    selected = st.multiselect(
        "Papers to include",
        options=all_ids,
        default=all_ids,
        format_func=lambda doc_id: choices[doc_id],
        max_selections=8,
        key="report_papers",
    )
    focus = st.text_area(
        "Optional focus or research question",
        placeholder=(
            "e.g. Focus on evaluation methodology and unresolved limitations across "
            "the selected papers."
        ),
        height=90,
        key="report_focus",
    )
    if selected:
        render_source_list([choices[doc_id] for doc_id in selected])

    intent = "literature_review" if report_type == "Literature review" else "report"
    if st.button(
        "Generate report",
        type="primary",
        disabled=not selected,
        width="stretch",
    ):
        query = focus.strip() or (
            "Generate a literature review synthesising the selected research papers."
            if report_type == "Literature review"
            else "Generate a structured research report on the selected papers."
        )
        state = execute_workflow(
            services,
            query,
            target_documents=selected,
            intent_hint=intent,
            status_label=f"Drafting {report_type.lower()}…",
        )
        content = str(state.get("final_response") or "").strip()
        error = str(state.get("error") or "").strip()
        if not error and not content:
            error = "The report agent did not produce a response."
        if not content:
            content = (
                f"I could not generate the report. {error}\n\n"
                "Check the Ollama connection and try again."
            )

        saved_path = ""
        if not error and content:
            try:
                path = save_markdown_report(services.settings, content)
                saved_path = str(path)
            except Exception as exc:
                st.warning(f"Report generated but could not be saved: {exc}")

        title = report_title_from_markdown(content, fallback="researchiq_report")
        download_name = f"{title[:60].replace(' ', '_').replace('/', '_')}.md"
        st.session_state["report_result"] = {
            "content": content,
            "report_type": report_type,
            "papers": [choices[doc_id] for doc_id in selected],
            "citations": state.get("citations") or [],
            "retrieved_documents": state.get("retrieved_documents") or [],
            "validation_result": state.get("validation_result") or {},
            "error": error,
            "saved_path": saved_path,
            "download_name": download_name,
            "at": utc_now_iso(),
        }
        add_activity(report_type, ", ".join(choices[doc_id] for doc_id in selected))
        st.rerun()

    _render_generated()
    st.divider()
    _render_saved(services)
