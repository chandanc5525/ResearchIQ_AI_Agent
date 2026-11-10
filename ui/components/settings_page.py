"""Settings page: model, retrieval, ingestion, and tracing configuration."""

from __future__ import annotations

import streamlit as st

from services import ApplicationServices
from ui.backend import clear_service_cache, get_health
from ui.components.common import render_page_intro
from utils.config import RUNTIME_OVERRIDES_FILE, update_runtime_settings


def _render_health(health: dict) -> None:
    online = bool(health.get("reachable"))
    model_ready = bool(health.get("model_available"))
    columns = st.columns(3)
    columns[0].metric("Ollama", "Online" if online else "Offline")
    columns[1].metric("Selected model", "Installed" if model_ready else "Missing")
    columns[2].metric("Model name", health.get("model", "—"))

    if health.get("error"):
        st.warning(health["error"])
    if health.get("models"):
        st.caption("Installed models: " + ", ".join(health["models"]))
    else:
        st.caption(
            "No local models were detected. Start Ollama with `ollama serve` and pull "
            "a model with `ollama pull qwen3:4b`."
        )


def render(services: ApplicationServices) -> None:
    """Render the settings form and connection diagnostics."""
    render_page_intro(
        "Settings",
        "Tune retrieval and generation without leaving the app. Runtime preferences "
        "are stored locally; API keys are read only from environment variables.",
    )

    settings = services.settings
    health = get_health()
    _render_health(health)
    if st.button("Refresh connection status"):
        get_health(force=True)
        st.rerun()

    st.divider()
    st.markdown("### Runtime configuration")

    installed = list(health.get("models") or [])
    model_options = sorted(set(installed + [settings.ollama_model]))
    model_index = model_options.index(settings.ollama_model)

    with st.form("runtime_settings_form"):
        st.markdown("**Model & generation**")
        model = st.selectbox(
            "Ollama model",
            options=model_options,
            index=model_index,
            help="Models installed in your local Ollama instance.",
        )
        st.caption(
            f"Ollama endpoint: `{settings.ollama_base_url}` · context window: "
            f"{settings.ollama_num_ctx} tokens"
        )

        st.markdown("**Retrieval**")
        top_k = st.slider(
            "Top-K retrieved chunks",
            min_value=1,
            max_value=12,
            value=int(settings.retrieval_top_k),
            help="Evidence chunks pulled per query (or per paper during comparisons).",
        )
        threshold = st.slider(
            "Similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(settings.similarity_threshold),
            step=0.05,
            help="0 disables threshold filtering. Chunks below the threshold are discarded.",
        )
        iterations = st.slider(
            "Maximum validation retries",
            min_value=0,
            max_value=4,
            value=int(settings.max_validation_iterations),
            help="How many times the workflow may re-retrieve when citations fail.",
        )

        st.markdown("**Ingestion**")
        chunk_size = st.number_input(
            "Chunk size (characters)",
            min_value=300,
            max_value=4000,
            value=int(settings.chunk_size),
            step=100,
        )
        chunk_overlap = st.number_input(
            "Chunk overlap (characters)",
            min_value=0,
            max_value=1000,
            value=int(settings.chunk_overlap),
            step=50,
        )
        st.caption(
            "Chunking changes apply to newly uploaded papers. Existing vectors keep "
            "their original chunking until re-uploaded."
        )

        st.markdown("**Observability**")
        tracing = st.toggle(
            "Enable LangSmith tracing",
            value=bool(settings.langsmith_tracing),
            help="Requires LANGSMITH_API_KEY in your environment. Disabled automatically "
            "when no key is present.",
        )
        if tracing and not settings.langsmith_api_key:
            st.caption(
                "Tracing will remain disabled because `LANGSMITH_API_KEY` is not set."
            )

        submitted = st.form_submit_button(
            "Save settings", type="primary", width="stretch"
        )

    if submitted:
        if int(chunk_overlap) >= int(chunk_size):
            st.error("Chunk overlap must be smaller than chunk size.")
        else:
            try:
                update_runtime_settings(
                    {
                        "ollama_model": model,
                        "retrieval_top_k": int(top_k),
                        "similarity_threshold": float(threshold),
                        "max_validation_iterations": int(iterations),
                        "chunk_size": int(chunk_size),
                        "chunk_overlap": int(chunk_overlap),
                        "langsmith_tracing": bool(tracing),
                    }
                )
                clear_service_cache()
                st.session_state["health_cache"] = None
                st.session_state["health_checked_at"] = 0.0
                st.success("Settings saved. Backend services will reload.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save settings: {exc}")

    st.divider()
    st.markdown("### Environment")
    env_col1, env_col2 = st.columns(2)
    with env_col1:
        st.caption(f"Embedding model · {settings.embedding_model}")
        st.caption(f"Embedding device · {settings.embedding_device}")
        st.caption(f"Vector store · `{settings.vectorstore_dir}`")
        st.caption(f"Runtime overrides · `{RUNTIME_OVERRIDES_FILE}`")
    with env_col2:
        st.caption(
            f"Chunk size / overlap · {settings.chunk_size} / {settings.chunk_overlap}"
        )
        st.caption(f"Top-K · {settings.retrieval_top_k}")
        st.caption(f"Max validation retries · {settings.max_validation_iterations}")
        tracing_state = "enabled" if health.get("tracing_enabled") else "disabled"
        st.caption(f"LangSmith tracing · {tracing_state}")
    st.caption(
        f"Indexed corpus · {services.registry.count()} papers · "
        f"{services.vector_store.count()} chunks"
    )

    st.divider()
    st.markdown("### Session utilities")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear conversation history", width="stretch"):
            st.session_state["messages"] = []
            st.success("Conversation history cleared.")
    with col2:
        if st.button("Clear generated results", width="stretch"):
            st.session_state["analysis_result"] = None
            st.session_state["comparison_result"] = None
            st.session_state["report_result"] = None
            st.success("Generated analyses, comparisons, and reports cleared.")
