"""ResearchIQ Streamlit application entry point.

Run from the project root with:
    streamlit run ui/streamlit_app.py
or:
    uv run python main.py app
"""

from __future__ import annotations

import logging

import streamlit as st

from ui.backend import clear_service_cache, get_health, get_services
from ui.components import (
    analysis_page,
    assistant_page,
    comparison_page,
    dashboard_page,
    reports_page,
    settings_page,
    upload_page,
)
from ui.components.common import render_brand, render_sidebar_status
from ui.session import PAGE_ICONS, PAGES, init_session_state
from ui.theme import inject_theme
from utils.config import get_settings
from utils.logger import setup_logging

logger = logging.getLogger("researchiq.ui")

_RENDERERS = {
    "Dashboard": dashboard_page.render,
    "Document Upload": upload_page.render,
    "Research Assistant": assistant_page.render,
    "Research Analysis": analysis_page.render,
    "Paper Comparison": comparison_page.render,
    "Research Reports": reports_page.render,
    "Settings": settings_page.render,
}


def _configure_page() -> None:
    settings = get_settings()
    setup_logging(settings.log_level, settings.logs_dir / "researchiq.log")
    st.set_page_config(
        page_title="ResearchIQ — Agentic Research Intelligence",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    inject_theme()


def _render_startup_error(exc: Exception) -> None:
    st.error("ResearchIQ could not initialise its local models and vector store.")
    st.markdown(
        "Check the following, then retry:\n\n"
        "1. Dependencies are installed: `uv sync`\n"
        "2. Ollama is running: `ollama serve`\n"
        "3. The model is available: `ollama pull qwen3:4b`\n"
        "4. The embedding model can be downloaded on first run."
    )
    with st.expander("Error details", expanded=True):
        st.code(f"{type(exc).__name__}: {exc}")
    if st.button("Retry initialisation", type="primary"):
        clear_service_cache()
        st.rerun()


def _render_sidebar(services, health: dict) -> str:
    render_brand()
    page = st.sidebar.radio(
        "Navigation",
        options=PAGES,
        format_func=lambda name: f"{PAGE_ICONS.get(name, '·')}  {name}",
        key="nav_page",
        label_visibility="collapsed",
    )
    render_sidebar_status(health, services.registry.count())
    st.sidebar.markdown(
        "<div class='riq-muted' style='padding:0.4rem 0.2rem;font-size:0.72rem'>"
        "Local-first · ChromaDB · LangGraph<br>No cloud API keys required"
        "</div>",
        unsafe_allow_html=True,
    )
    return page


def main() -> None:
    """Application entry point."""
    _configure_page()

    try:
        services = get_services()
    except Exception as exc:  # Initialisation failures should be actionable.
        logger.exception("Application initialisation failed")
        _render_startup_error(exc)
        return

    try:
        health = get_health()
    except Exception as exc:
        logger.warning("Health check failed: %s", exc)
        health = {
            "reachable": False,
            "model_available": False,
            "error": str(exc),
            "model": services.settings.ollama_model,
        }

    page = _render_sidebar(services, health)
    renderer = _RENDERERS.get(page)
    if renderer is None:
        st.error(f"Unknown page: {page}")
        return

    try:
        renderer(services)
    except Exception as exc:
        logger.exception("Page '%s' failed", page)
        st.error(f"The {page} page encountered an error.")
        with st.expander("Error details", expanded=True):
            st.code(f"{type(exc).__name__}: {exc}")

    st.markdown(
        "<div class='riq-muted' style='text-align:center;margin-top:2.5rem;"
        "font-size:0.72rem'>ResearchIQ · evidence-grounded research intelligence · "
        "answers are validated against uploaded papers</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
