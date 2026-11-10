"""Document Upload page: multi-PDF ingestion, preview, and removal."""

from __future__ import annotations

from typing import Any

import streamlit as st

from services import ApplicationServices
from ui.components.common import (
    format_timestamp,
    render_page_intro,
)
from ui.session import add_activity
from utils.helpers import human_file_size


def _process_uploads(services: ApplicationServices, uploads: list[Any]) -> None:
    results: list[dict[str, Any]] = []
    progress = st.progress(0.0, text="Starting ingestion…")
    with st.status("Processing research papers", expanded=True) as status:
        for index, upload in enumerate(uploads, start=1):
            status.write(f"Reading {upload.name}…")
            try:
                data = upload.getvalue()
                result = services.ingestion.ingest_bytes(data, upload.name)
                results.append(
                    {
                        "filename": upload.name,
                        "status": "duplicate" if result.skipped else "indexed",
                        "message": result.message,
                        "pages": result.record.page_count,
                        "chunks": result.record.chunk_count,
                        "doc_id": result.record.doc_id,
                    }
                )
                status.write(
                    ("Skipped duplicate: " if result.skipped else "Indexed: ")
                    + upload.name
                )
            except Exception as exc:
                results.append(
                    {
                        "filename": upload.name,
                        "status": "error",
                        "message": str(exc),
                        "pages": 0,
                        "chunks": 0,
                        "doc_id": "",
                    }
                )
                status.write(f"Failed: {upload.name} — {exc}")
            progress.progress(
                index / max(1, len(uploads)),
                text=f"Processed {index} of {len(uploads)} PDF(s)",
            )
        indexed = sum(1 for item in results if item["status"] == "indexed")
        status.update(
            label=f"Ingestion finished · {indexed} new paper(s) indexed",
            state="complete" if indexed or results else "error",
            expanded=False,
        )

    st.session_state["last_ingestion"] = results
    indexed_items = [item for item in results if item["status"] == "indexed"]
    if indexed_items:
        add_activity(
            "Papers indexed",
            ", ".join(item["filename"] for item in indexed_items),
        )
    st.session_state["upload_nonce"] = int(st.session_state.get("upload_nonce", 0)) + 1
    st.rerun()


def _render_last_results() -> None:
    results = st.session_state.get("last_ingestion") or []
    if not results:
        return
    indexed = sum(1 for item in results if item["status"] == "indexed")
    duplicates = sum(1 for item in results if item["status"] == "duplicate")
    errors = sum(1 for item in results if item["status"] == "error")
    st.markdown(
        f"**Last ingestion:** {indexed} indexed · {duplicates} duplicate · {errors} failed"
    )
    for item in results:
        if item["status"] == "indexed":
            st.success(
                f"{item['filename']} — {item['pages']} pages, {item['chunks']} chunks. "
                f"{item['message']}"
            )
        elif item["status"] == "duplicate":
            st.info(f"{item['filename']} — {item['message']}")
        else:
            st.error(f"{item['filename']} — {item['message']}")
    if st.button("Dismiss summary", key="dismiss_ingestion"):
        st.session_state["last_ingestion"] = []
        st.rerun()
    st.divider()


def _render_library(services: ApplicationServices) -> None:
    st.markdown("### Indexed research papers")
    records = services.registry.list_documents()
    if not records:
        st.info("Your research library is empty. Upload PDFs above to get started.")
        return

    choices = {record.doc_id: record.filename for record in records}
    selected_id = st.selectbox(
        "Inspect a paper",
        options=list(choices),
        format_func=lambda doc_id: choices[doc_id],
        key="upload_inspect_doc",
    )
    record = services.registry.get(selected_id)
    if record is None:
        st.warning("That document is no longer available.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pages", record.page_count)
    col2.metric("Chunks", record.chunk_count)
    col3.metric("File size", human_file_size(record.file_size_bytes))
    col4.metric("Indexed", format_timestamp(record.ingested_at).split(" · ")[0])

    with st.container(border=True):
        st.markdown(f"**{record.filename}**")
        st.caption(
            f"Document ID: `{record.doc_id}` · SHA-256: `{record.file_hash[:16]}…`"
        )
        st.caption(f"Stored at: `{record.file_path}`")

    preview = services.ingestion.get_preview(selected_id, max_pages=3)
    if preview and preview.get("pages"):
        with st.expander("Preview extracted text (first 3 pages)", expanded=False):
            page_tabs = st.tabs([f"Page {page['page']}" for page in preview["pages"]])
            for tab, page in zip(page_tabs, preview["pages"]):
                with tab:
                    st.text_area(
                        "Extracted text",
                        value=page.get("text", ""),
                        height=260,
                        disabled=True,
                        key=f"preview_{selected_id}_{page['page']}",
                        label_visibility="collapsed",
                    )
    else:
        st.caption("No processed text preview is available for this paper.")

    with st.expander("Remove this paper and its vectors", expanded=False):
        st.warning(
            "Removing a paper deletes its stored PDF, processed text, and all "
            "ChromaDB vectors. This cannot be undone."
        )
        confirmed = st.checkbox(
            "I understand this permanently removes the paper from the knowledge base.",
            key=f"confirm_remove_{selected_id}",
        )
        if st.button(
            "Remove paper permanently",
            disabled=not confirmed,
            key=f"remove_{selected_id}",
        ):
            try:
                services.ingestion.remove_document(selected_id)
                add_activity("Paper removed", record.filename)
                st.session_state["last_ingestion"] = []
                st.success(f"Removed {record.filename} and its vectors.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not remove the paper: {exc}")


def render(services: ApplicationServices) -> None:
    """Render the upload and management page."""
    render_page_intro(
        "Document Upload",
        "Add one or more academic PDFs. Files are parsed page by page, chunked, "
        "embedded locally, and stored in ChromaDB. Duplicate uploads are skipped by "
        "content hash.",
    )
    _render_last_results()

    st.markdown("### Add research papers")
    uploads = st.file_uploader(
        "Drop PDF research papers here",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_uploader_{st.session_state.get('upload_nonce', 0)}",
        help=f"Up to {services.settings.max_upload_mb} MB per PDF. Only text-based "
        "PDFs are supported; scanned images require OCR.",
    )

    if uploads:
        st.caption(
            f"{len(uploads)} file(s) selected · "
            f"{human_file_size(sum(upload.size for u in uploads))} total"
        )
        with st.container(border=True):
            for upload in uploads:
                st.markdown(f"- **{upload.name}** · {human_file_size(upload.size)}")
        st.caption(
            f"Chunking: {services.settings.chunk_size} characters with "
            f"{services.settings.chunk_overlap} overlap · "
            f"Embeddings: {services.settings.embedding_model.split('/')[-1]}"
        )
        if st.button("Process and index selected PDFs", type="primary"):
            _process_uploads(services, uploads)
    else:
        st.caption(
            "Select PDFs above. Indexing runs locally; no documents leave this machine."
        )

    st.divider()
    _render_library(services)
