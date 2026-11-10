"""Shared UI rendering helpers and workflow progress handling."""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any, Iterable, Sequence

import streamlit as st

from services import ApplicationServices
from ui.session import request_navigation
from utils.helpers import truncate_text
from workflows.edges import NODE_LABELS


def esc(value: Any) -> str:
    """Escape untrusted text before inserting it into HTML."""
    return html.escape(str(value or ""))


def format_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.strftime("%b %d, %Y · %H:%M")
    except (TypeError, ValueError):
        return str(value or "")


def render_hero(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="riq-hero">
            <div class="riq-eyebrow">{esc(eyebrow)}</div>
            <h1>{esc(title)}</h1>
            <p>{esc(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_intro(title: str, description: str) -> None:
    st.markdown(f"## {esc(title)}")
    st.markdown(
        f"<p class='riq-muted' style='margin-top:-0.35rem'>{esc(description)}</p>",
        unsafe_allow_html=True,
    )


def render_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="riq-brand">
            <div class="riq-brand-mark">R</div>
            <div>
                <div class="riq-brand-name">ResearchIQ</div>
                <div class="riq-brand-sub">Agentic Research Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_status(health: dict[str, Any], document_count: int) -> None:
    online = bool(health.get("reachable"))
    model_ready = bool(health.get("model_available"))
    dot = "on" if online and model_ready else "off"
    if online and model_ready:
        status_text = f"{health.get('model', 'model')} ready"
    elif online:
        status_text = f"{health.get('model', 'model')} not installed"
    else:
        status_text = "Ollama offline"

    st.sidebar.markdown(
        f"""
        <div class="riq-card" style="padding:0.75rem 0.85rem;margin-top:0.4rem">
            <div style="font-size:0.8rem;color:#e2e8f0">
                <span class="riq-status-dot {dot}"></span>{esc(status_text)}
            </div>
            <div class="riq-muted" style="margin-top:0.3rem">
                {document_count} paper(s) · local inference
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chip(text: str, kind: str = "") -> str:
    return f"<span class='riq-chip {esc(kind)}'>{esc(text)}</span>"


def render_validation_badge(validation: dict[str, Any] | None) -> None:
    """Show a compact grounding/citation status for a generated answer."""
    if not validation:
        return
    valid = bool(validation.get("is_valid"))
    confidence = validation.get("confidence")
    method = str(validation.get("method") or "")
    chips = [
        render_chip(
            "Citation validated" if valid else "Needs review", "ok" if valid else "bad"
        )
    ]
    if confidence is not None:
        try:
            chips.append(render_chip(f"Grounding confidence {float(confidence):.0%}"))
        except (TypeError, ValueError):
            pass
    if method:
        chips.append(render_chip(method))
    st.markdown(" ".join(chips), unsafe_allow_html=True)

    details = validation.get("unsupported_claims") or []
    fabricated = validation.get("fabricated_citations") or []
    if not valid and (details or fabricated or validation.get("notes")):
        with st.expander("Citation validation details", expanded=False):
            if validation.get("notes"):
                st.write(validation["notes"])
            if fabricated:
                st.markdown("**Citations not found in retrieved evidence**")
                for item in fabricated:
                    st.markdown(f"- {item}")
            if details:
                st.markdown("**Statements that may need verification**")
                for item in details[:6]:
                    st.markdown(f"- {item}")


def render_citations(citations: Sequence[dict[str, Any]] | None) -> None:
    """Render page-level source references."""
    citations = list(citations or [])
    if not citations:
        st.caption("No page-level citations were returned for this answer.")
        return
    with st.expander(f"Sources ({len(citations)} page references)", expanded=False):
        for citation in citations:
            filename = citation.get("filename", "Unknown")
            page = citation.get("page", "?")
            score = citation.get("score")
            score_text = (
                f" · similarity {float(score):.2f}" if score is not None else ""
            )
            st.markdown(
                f"<div class='riq-source'><b>{esc(filename)}</b>"
                f"<span> · page {esc(page)}{esc(score_text)}</span></div>",
                unsafe_allow_html=True,
            )


def render_retrieved_context(chunks: Sequence[dict[str, Any]] | None) -> None:
    """Expose retrieved evidence for inspection and debugging."""
    chunks = list(chunks or [])
    if not chunks:
        return
    with st.expander(f"Retrieved context ({len(chunks)} chunks)", expanded=False):
        for index, chunk in enumerate(chunks, start=1):
            filename = chunk.get("filename", "Unknown")
            page = chunk.get("page", "?")
            score = chunk.get("score")
            st.markdown(
                f"**{index}. {filename} — page {page}**"
                + (f" · score {float(score):.3f}" if score is not None else "")
            )
            st.markdown(
                f"> {truncate_text(str(chunk.get('content', '')), 700).replace(chr(10), ' ')}"
            )


def render_answer_meta(state: dict[str, Any]) -> None:
    """Render validation, citations, and evidence for a workflow result."""
    render_validation_badge(state.get("validation_result"))
    render_citations(state.get("citations"))
    render_retrieved_context(state.get("retrieved_documents"))


def render_empty_documents(message: str = "No papers are indexed yet.") -> None:
    st.info(message)
    if st.button("Upload research papers", type="primary", key="empty_upload"):
        request_navigation("Document Upload")


def history_from_messages(
    messages: Sequence[dict[str, Any]], limit: int = 8
) -> list[dict[str, str]]:
    """Convert chat messages into compact conversation history for the graph."""
    history: list[dict[str, str]] = []
    for message in list(messages)[-limit:]:
        role = str(message.get("role", "user"))
        content = str(message.get("content", "")).strip()
        if content:
            history.append({"role": role, "content": truncate_text(content, 1400)})
    return history


def execute_workflow(
    services: ApplicationServices,
    query: str,
    *,
    target_documents: Sequence[str] | None = None,
    intent_hint: str | None = None,
    history: Sequence[dict[str, str]] | None = None,
    status_label: str = "ResearchIQ agents are working…",
) -> dict[str, Any]:
    """Run the LangGraph workflow with live agent progress in Streamlit."""
    try:
        state = services.workflow.build_state(
            query,
            documents=services.document_metadata(),
            target_documents=target_documents,
            chat_history=history or [],
            intent_hint=intent_hint,
        )
    except Exception as exc:
        return {
            "error": str(exc),
            "final_response": f"I could not start the research workflow: {exc}",
            "document_metadata": services.document_metadata(),
        }

    merged: dict[str, Any] = dict(state)
    try:
        with st.status(status_label, expanded=True) as status:
            status.write("Supervisor is identifying the research intent…")
            for node, update in services.workflow.stream_state(state):
                merged.update(update)
                label = NODE_LABELS.get(node, node.replace("_", " ").title())
                pass_note = ""
                if node in {"retrieval", "citation_validation"}:
                    count = (
                        merged.get("iteration_count")
                        if node == "retrieval"
                        else merged.get("validation_attempts")
                    )
                    if count:
                        pass_note = f" (pass {count})"
                status.write(f"{label}{pass_note}")
                if update.get("error"):
                    status.write(f"⚠️ {update['error']}")
            if merged.get("error"):
                status.update(
                    label="Workflow completed with an issue",
                    state="error",
                    expanded=True,
                )
            else:
                status.update(
                    label="Agent workflow complete", state="complete", expanded=False
                )
    except Exception as exc:
        merged.update(
            {
                "error": str(exc),
                "final_response": (
                    "I could not complete the research workflow. "
                    f"{exc}\n\nCheck the Ollama connection and try again."
                ),
            }
        )

    return merged


def render_source_list(filenames: Iterable[str]) -> None:
    chips = " ".join(render_chip(name) for name in filenames)
    if chips:
        st.markdown(chips, unsafe_allow_html=True)
