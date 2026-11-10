"""Research Assistant page: grounded chat with citations."""

from __future__ import annotations

import streamlit as st

from services import ApplicationServices
from ui.components.common import (
    execute_workflow,
    history_from_messages,
    render_answer_meta,
    render_empty_documents,
    render_page_intro,
)
from ui.session import add_activity, request_navigation
from utils.helpers import truncate_text, utc_now_iso

SUGGESTIONS = [
    "Explain this research paper in simple language.",
    "What is the main research problem?",
    "Explain the methodology step by step.",
    "Which machine learning algorithm was used?",
    "What dataset was used?",
    "Explain the experimental results.",
    "What are the limitations?",
    "Identify possible research gaps.",
    "Compare the findings of the uploaded papers.",
    "Generate a literature review.",
]


def _intent_label(intent: str) -> str:
    return intent.replace("_", " ").title() if intent else "Research"


def _render_messages() -> None:
    messages = st.session_state.get("messages") or []
    if not messages:
        with st.container(border=True):
            st.markdown("**Start a research conversation**")
            st.caption(
                "Ask about the research problem, methodology, datasets, algorithms, "
                "results, limitations, or gaps. Every answer is retrieved from your "
                "indexed papers and checked for citation support."
            )
        return

    for message in messages:
        role = message.get("role", "assistant")
        avatar = "🧑‍💻" if role == "user" else "🧠"
        with st.chat_message(role, avatar=avatar):
            if role == "assistant":
                chips = [
                    f"<span class='riq-chip'>{_intent_label(message.get('intent', ''))}</span>"
                ]
                if message.get("selected_agent"):
                    chips.append(
                        f"<span class='riq-chip'>{str(message['selected_agent']).replace('_', ' ').title()}</span>"
                    )
                st.markdown(" ".join(chips), unsafe_allow_html=True)
            st.markdown(str(message.get("content", "")))
            if role == "assistant":
                if message.get("error"):
                    st.caption(f"Workflow note: {message['error']}")
                render_answer_meta(message)


def _handle_prompt(services: ApplicationServices, prompt: str) -> None:
    prompt = prompt.strip()
    if not prompt:
        return
    messages: list[dict] = list(st.session_state.get("messages") or [])
    history = history_from_messages(messages)
    messages.append(
        {
            "role": "user",
            "content": prompt,
            "at": utc_now_iso(),
        }
    )

    state = execute_workflow(
        services,
        prompt,
        history=history,
        status_label="ResearchIQ is consulting your papers…",
    )

    answer = str(state.get("final_response") or "").strip()
    if not answer:
        answer = (
            "I could not produce a grounded answer for this request. "
            "Check the Ollama connection and try again."
        )
    messages.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": state.get("citations") or [],
            "retrieved_documents": state.get("retrieved_documents") or [],
            "validation_result": state.get("validation_result") or {},
            "intent": state.get("research_intent") or "",
            "selected_agent": state.get("selected_agent") or "",
            "error": state.get("error") or "",
            "at": utc_now_iso(),
        }
    )
    st.session_state["messages"] = messages
    add_activity("Research query", truncate_text(prompt, 100))


def render(services: ApplicationServices) -> None:
    """Render the conversational research assistant."""
    header_left, header_right = st.columns([4, 1])
    with header_left:
        render_page_intro(
            "Research Assistant",
            "Ask evidence-grounded questions across your indexed papers. Answers "
            "include page-level citations and a citation-validation report.",
        )
    with header_right:
        st.write("")
        if st.button("Clear chat", width="stretch"):
            st.session_state["messages"] = []
            st.rerun()

    documents = services.document_metadata()
    if not documents:
        render_empty_documents(
            "The research assistant needs at least one indexed paper before it can "
            "retrieve evidence."
        )
        st.divider()
        if st.button("Open Document Upload", type="primary"):
            request_navigation("Document Upload")
        return

    st.caption(
        f"Searching {len(documents)} paper(s) · Top-{services.settings.retrieval_top_k} "
        f"retrieval · {services.settings.ollama_model}"
    )

    with st.expander(
        "Suggested research questions", expanded=not st.session_state.get("messages")
    ):
        columns = st.columns(2)
        for index, suggestion in enumerate(SUGGESTIONS):
            if columns[index % 2].button(
                suggestion,
                key=f"suggestion_{index}",
                width="stretch",
            ):
                st.session_state["pending_query"] = suggestion
                st.rerun()

    pending = st.session_state.pop("pending_query", None)
    chat_prompt = st.chat_input("Ask a question about your research papers…")
    prompt = pending or chat_prompt
    if prompt:
        _handle_prompt(services, str(prompt))

    _render_messages()
