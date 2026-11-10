"""Dashboard page: corpus overview, system status, and recent activity."""

from __future__ import annotations

from typing import Any

import streamlit as st

from services import ApplicationServices
from ui.backend import get_health
from ui.components.common import (
    esc,
    format_timestamp,
    render_hero,
    render_source_list,
)
from ui.session import request_navigation
from utils.helpers import human_file_size


def _metric_row(services: ApplicationServices, health: dict[str, Any]) -> None:
    stats = services.ingestion.stats()
    model_state = (
        "Ready"
        if health.get("reachable") and health.get("model_available")
        else ("Missing" if health.get("reachable") else "Offline")
    )
    columns = st.columns(4)
    columns[0].metric("Research papers", stats["documents"])
    columns[1].metric("Indexed pages", stats["pages"])
    columns[2].metric("Vector chunks", stats["chunks"])
    columns[3].metric("Local model", model_state)


def _render_health(health: dict[str, Any]) -> None:
    if not health.get("reachable"):
        st.warning(
            "**Ollama is offline.** Start it with `ollama serve`, then refresh the "
            "Settings page. PDF ingestion and retrieval still work without it."
        )
    elif not health.get("model_available"):
        st.error(
            f"**Model not installed:** `{health.get('model')}`. Run "
            f"`ollama pull {health.get('model')}` to enable analysis and reports."
        )


def _documents_table(services: ApplicationServices) -> None:
    records = services.registry.list_documents()
    st.markdown("### Research library")
    if not records:
        st.info("No papers indexed yet. Upload PDFs to start building your library.")
        if st.button("Go to Document Upload", type="primary"):
            request_navigation("Document Upload")
        return

    rows = [
        {
            "Paper": record.filename,
            "Pages": record.page_count,
            "Chunks": record.chunk_count,
            "Size": human_file_size(record.file_size_bytes),
            "Indexed": format_timestamp(record.ingested_at),
        }
        for record in records
    ]
    st.dataframe(
        rows, width="stretch", hide_index=True, height=min(420, 42 + 35 * len(rows))
    )


def _recent_activity(services: ApplicationServices) -> None:
    st.markdown("### Recent activity")
    items: list[dict[str, Any]] = []
    for record in services.registry.list_documents()[:5]:
        items.append(
            {
                "action": "Paper indexed",
                "detail": record.filename,
                "at": record.ingested_at,
            }
        )
    items.extend(st.session_state.get("activity") or [])
    items.sort(key=lambda item: str(item.get("at", "")), reverse=True)
    items = items[:8]

    if not items:
        st.caption("Your ingestion and research activity will appear here.")
        return
    for item in items:
        st.markdown(
            f"<div class='riq-card' style='padding:0.65rem 0.85rem;margin-bottom:0.45rem'>"
            f"<div style='font-size:0.86rem;color:#e2e8f0'><b>{esc(item.get('action'))}</b></div>"
            f"<div class='riq-muted'>{esc(item.get('detail'))}</div>"
            f"<div class='riq-muted' style='font-size:0.72rem'>"
            f"{esc(format_timestamp(str(item.get('at', ''))))}</div></div>",
            unsafe_allow_html=True,
        )


def _system_panel(services: ApplicationServices, health: dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown("**System status**")
        online = bool(health.get("reachable") and health.get("model_available"))
        st.markdown(
            f"<span class='riq-status-dot {'on' if online else 'off'}'></span>"
            f"{esc(health.get('model', services.settings.ollama_model))}",
            unsafe_allow_html=True,
        )
        st.caption(f"Ollama · {services.settings.ollama_base_url}")
        st.caption(f"Embeddings · {services.settings.embedding_model.split('/')[-1]}")
        st.caption(f"Retrieval · Top-{services.settings.retrieval_top_k} · ChromaDB")
        tracing = "enabled" if health.get("tracing_enabled") else "disabled"
        st.caption(f"LangSmith tracing · {tracing}")

    with st.container(border=True):
        st.markdown("**Quick actions**")
        if st.button("Upload papers", width="stretch"):
            request_navigation("Document Upload")
        if st.button("Ask ResearchIQ", width="stretch"):
            request_navigation("Research Assistant")
        if st.button("Generate a report", width="stretch"):
            request_navigation("Research Reports")


def _capabilities() -> None:
    st.markdown("### What ResearchIQ can do")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Understand papers**")
        st.caption(
            "Page-level summaries, methodology analysis, datasets, algorithms, "
            "findings, and limitations."
        )
    with col2:
        st.markdown("**Compare evidence**")
        st.caption(
            "Structured comparisons across objectives, methods, datasets, results, "
            "and constraints."
        )
    with col3:
        st.markdown("**Validate claims**")
        st.caption(
            "Every answer is audited against retrieved pages; unsupported or "
            "fabricated references are flagged."
        )
    render_source_list(
        [
            "LangGraph routing",
            "ChromaDB retrieval",
            "Hugging Face embeddings",
            "Local qwen3:4b",
            "Page-level citations",
        ]
    )


def render(services: ApplicationServices) -> None:
    """Render the dashboard."""
    health = get_health()
    render_hero(
        "Agentic research intelligence",
        "ResearchIQ",
        "Upload academic papers, interrogate their evidence, compare methodologies, "
        "and generate validated research reports with a local multi-agent workflow.",
    )
    _metric_row(services, health)
    _render_health(health)

    left, right = st.columns([2.05, 1])
    with left:
        _documents_table(services)
        st.divider()
        _recent_activity(services)
    with right:
        _system_panel(services, health)

    st.divider()
    _capabilities()
