"""Tests for PDF loading, page metadata, chunking, and ingestion lifecycle."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest
from langchain_core.documents import Document

from ingestion.document_parser import DocumentRegistry, build_document_id
from ingestion.pdf_loader import PDFLoader
from ingestion.text_splitter import DocumentTextSplitter
from tests.conftest import make_pdf
from utils.errors import EmptyDocumentError, PDFIngestionError


def test_pdf_loader_preserves_page_level_metadata(tmp_path: Path) -> None:
    path = make_pdf(
        tmp_path / "paper.pdf",
        ["Page one text about retrieval.", "Page two text about evaluation."],
    )

    pages = PDFLoader().load(path)

    assert len(pages) == 2
    assert "retrieval" in pages[0].page_content.lower()
    assert pages[0].metadata["page"] == 0
    assert pages[0].metadata["page_number"] == 1
    assert pages[1].metadata["page_number"] == 2
    assert pages[0].metadata["filename"] == "paper.pdf"
    assert pages[0].metadata["source"] == "paper.pdf"
    assert Path(pages[0].metadata["source_file"]).name == "paper.pdf"


def test_pdf_loader_rejects_non_pdf_and_corrupt_files(tmp_path: Path) -> None:
    not_pdf = tmp_path / "notes.pdf"
    not_pdf.write_bytes(b"This is not a PDF at all.")
    with pytest.raises(PDFIngestionError, match="%PDF"):
        PDFLoader().load(not_pdf)

    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.4\nthis content is not a valid PDF body")
    with pytest.raises(PDFIngestionError, match="Could not read"):
        PDFLoader().load(corrupt)

    with pytest.raises(PDFIngestionError, match="File not found"):
        PDFLoader().load(tmp_path / "missing.pdf")


def test_pdf_loader_rejects_empty_pdf(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(str(path))
    document.close()

    with pytest.raises(EmptyDocumentError, match="no extractable text"):
        PDFLoader().load(path)


def test_splitter_preserves_metadata_and_chunk_ids() -> None:
    splitter = DocumentTextSplitter(chunk_size=400, chunk_overlap=50)
    long_text = ("Retrieval augmented generation improves grounding. " * 40).strip()
    document = Document(
        page_content=long_text,
        metadata={
            "page": 2,
            "page_number": 3,
            "filename": "paper.pdf",
            "source": "paper.pdf",
        },
    )

    chunks = splitter.split(
        [document], doc_id="doc_test", filename="paper.pdf", file_hash="abc"
    )

    assert len(chunks) > 1
    chunk_ids = {chunk.metadata["chunk_id"] for chunk in chunks}
    assert len(chunk_ids) == len(chunks)
    for index, chunk in enumerate(chunks):
        assert chunk.metadata["doc_id"] == "doc_test"
        assert chunk.metadata["filename"] == "paper.pdf"
        assert chunk.metadata["page_number"] == 3
        assert chunk.metadata["chunk_index"] == index
        assert chunk.metadata["chunk_id"].startswith("doc_test-c")
        assert len(chunk.page_content) <= 400
        assert chunk.page_content.strip()


def test_splitter_rejects_invalid_overlap() -> None:
    with pytest.raises(PDFIngestionError):
        DocumentTextSplitter(chunk_size=300, chunk_overlap=300)


def test_document_id_is_stable() -> None:
    first = build_document_id("hash123", "paper.pdf")
    second = build_document_id("hash123", "paper.pdf")
    different = build_document_id("hash123", "other.pdf")
    assert first == second
    assert first != different
    assert first.startswith("doc_")


def test_ingestion_pipeline_deduplicates_and_removes(
    ingestion, pdf_bytes: bytes, settings
) -> None:
    result = ingestion.ingest_bytes(pdf_bytes, "research-paper.pdf")

    assert not result.skipped
    assert result.record.page_count == 3
    assert result.record.chunk_count > 0
    assert result.chunks == result.record.chunk_count
    assert ingestion.registry.count() == 1
    assert ingestion.vector_store.count() == result.record.chunk_count
    assert Path(settings.data_dir / result.record.file_path).exists()
    processed = settings.processed_dir / f"{result.record.doc_id}.json"
    assert processed.exists()

    preview = ingestion.get_preview(result.record.doc_id, max_pages=2)
    assert preview is not None
    assert len(preview["pages"]) == 2
    assert preview["pages"][0]["page"] == 1

    duplicate = ingestion.ingest_bytes(pdf_bytes, "same-content-renamed.pdf")
    assert duplicate.skipped
    assert duplicate.record.doc_id == result.record.doc_id
    assert ingestion.registry.count() == 1
    assert ingestion.vector_store.count() == result.record.chunk_count

    stored_file = Path(settings.data_dir / result.record.file_path)
    ingestion.remove_document(result.record.doc_id)

    assert ingestion.registry.count() == 0
    assert ingestion.vector_store.count() == 0
    assert not processed.exists()
    assert not stored_file.exists()


def test_ingestion_rejects_invalid_bytes(ingestion) -> None:
    with pytest.raises(PDFIngestionError, match="empty"):
        ingestion.ingest_bytes(b"", "empty.pdf")

    with pytest.raises(PDFIngestionError, match="%PDF"):
        ingestion.ingest_bytes(b"not a pdf", "fake.pdf")


def test_ingestion_file_copies_into_managed_storage(
    ingestion, tmp_path: Path, settings
) -> None:
    external = make_pdf(
        tmp_path / "external.pdf",
        ["Graph neural networks for citation recommendation are evaluated here."],
    )
    result = ingestion.ingest_file(external, original_filename="managed.pdf")

    assert result.record.filename == "managed.pdf"
    assert ingestion.registry.count() == 1
    stored = Path(settings.data_dir / result.record.file_path)
    assert stored.exists()
    assert stored.parent == settings.uploaded_papers_dir

    duplicate = ingestion.ingest_file(external)
    assert duplicate.skipped


def test_registry_persists_across_instances(settings) -> None:
    registry = DocumentRegistry(settings.registry_file)
    from ingestion.document_parser import DocumentRecord

    record = DocumentRecord(
        doc_id="doc_persist",
        filename="persist.pdf",
        original_filename="persist.pdf",
        file_hash="hash",
        file_path="uploaded_papers/persist.pdf",
        page_count=2,
        chunk_count=5,
        file_size_bytes=100,
    )
    registry.add(record)

    reloaded = DocumentRegistry(settings.registry_file)
    assert reloaded.get("doc_persist") is not None
    assert reloaded.total_pages() == 2
    assert reloaded.total_chunks() == 5
