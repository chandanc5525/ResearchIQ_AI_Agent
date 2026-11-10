"""End-to-end PDF ingestion: validate, load, split, embed, persist, register."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ingestion.document_parser import (
    DocumentRecord,
    DocumentRegistry,
    build_document_id,
)
from ingestion.pdf_loader import PDFLoader
from ingestion.text_splitter import DocumentTextSplitter
from retrieval.vector_store import VectorStoreManager
from utils.config import Settings
from utils.errors import EmptyDocumentError, PDFIngestionError, VectorStoreError
from utils.helpers import (
    atomic_write_json,
    bytes_sha256,
    file_sha256,
    is_pdf_bytes,
    read_json,
    safe_filename,
    utc_now_iso,
)
from utils.logger import get_logger

logger = get_logger("ingestion.pipeline")


@dataclass
class IngestionResult:
    """Outcome of one ingestion attempt."""

    record: DocumentRecord
    chunks: int
    skipped: bool = False
    message: str = ""


class IngestionPipeline:
    """Coordinate PDF parsing, chunking, embedding, and registration."""

    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStoreManager,
        registry: DocumentRegistry | None = None,
        loader: PDFLoader | None = None,
        splitter: DocumentTextSplitter | None = None,
    ) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.registry = registry or DocumentRegistry(settings.registry_file)
        self.loader = loader or PDFLoader()
        self.splitter = splitter or DocumentTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        settings.ensure_directories()

    # ------------------------------------------------------------------
    # Public ingestion APIs
    # ------------------------------------------------------------------
    def ingest_bytes(self, data: bytes, filename: str) -> IngestionResult:
        """Ingest a PDF already held in memory (Streamlit uploads)."""
        if not data:
            raise PDFIngestionError("The uploaded file is empty.")
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        if len(data) > max_bytes:
            raise PDFIngestionError(
                f"'{filename}' exceeds the {self.settings.max_upload_mb} MB limit."
            )
        if not is_pdf_bytes(data):
            raise PDFIngestionError(
                f"'{filename}' is not a valid PDF (missing %PDF header)."
            )

        file_hash = bytes_sha256(data)
        duplicate = self.registry.find_by_hash(file_hash)
        if duplicate is not None:
            return IngestionResult(
                record=duplicate,
                chunks=0,
                skipped=True,
                message=f"'{filename}' is already indexed; ingestion was skipped.",
            )

        clean_name = safe_filename(filename)
        doc_id = build_document_id(file_hash, clean_name)
        stored_path = self.settings.uploaded_papers_dir / f"{doc_id}_{clean_name}"
        stored_path.write_bytes(data)
        return self._process(
            stored_path=stored_path,
            original_filename=clean_name,
            file_hash=file_hash,
            doc_id=doc_id,
            saved_new=True,
        )

    def ingest_file(
        self, path: Path, original_filename: str | None = None
    ) -> IngestionResult:
        """Ingest a PDF from disk, copying it into the managed data folder."""
        source = Path(path).expanduser().resolve()
        # Validate before doing any copy or hash work.
        PDFLoader._validate(source)  # noqa: SLF001 - shared validation logic
        file_hash = file_sha256(source)
        duplicate = self.registry.find_by_hash(file_hash)
        if duplicate is not None:
            return IngestionResult(
                record=duplicate,
                chunks=0,
                skipped=True,
                message=f"'{source.name}' is already indexed; ingestion was skipped.",
            )

        clean_name = safe_filename(original_filename or source.name)
        doc_id = build_document_id(file_hash, clean_name)
        stored_path = self.settings.uploaded_papers_dir / f"{doc_id}_{clean_name}"

        already_managed = self._is_inside(source, self.settings.uploaded_papers_dir)
        if already_managed and source == stored_path:
            saved_new = False
        else:
            shutil.copy2(source, stored_path)
            saved_new = True

        return self._process(
            stored_path=stored_path,
            original_filename=clean_name,
            file_hash=file_hash,
            doc_id=doc_id,
            saved_new=saved_new,
        )

    def ingest_directory(self, directory: Path) -> list[IngestionResult]:
        """Ingest every PDF in a directory, continuing after individual failures."""
        results: list[IngestionResult] = []
        for path in sorted(Path(directory).glob("*.pdf")):
            try:
                results.append(self.ingest_file(path))
            except (PDFIngestionError, VectorStoreError) as exc:
                logger.error("Skipping %s: %s", path.name, exc)
        return results

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------
    def remove_document(self, doc_id: str) -> DocumentRecord:
        """Delete vectors, processed text, the stored PDF, and the registry entry."""
        record = self.registry.get(doc_id)
        if record is None:
            raise PDFIngestionError(f"Document '{doc_id}' was not found.")

        # Remove vectors first. If this fails the registry stays intact so the
        # user can retry rather than leaving orphaned records.
        self.vector_store.delete_document(doc_id)

        processed_path = self.settings.processed_dir / f"{doc_id}.json"
        try:
            processed_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not delete processed file %s", processed_path)

        stored_path = (self.settings.data_dir / record.file_path).resolve()
        if self._is_inside(stored_path, self.settings.uploaded_papers_dir):
            try:
                stored_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not delete stored PDF %s", stored_path)

        removed = self.registry.remove(doc_id)
        logger.info("Removed document %s and its vectors", doc_id)
        return removed or record

    def list_documents(self) -> list[DocumentRecord]:
        return self.registry.list_documents()

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        return self.registry.get(doc_id)

    def get_preview(self, doc_id: str, max_pages: int | None = 3) -> dict | None:
        """Return stored page text for UI previews."""
        payload = read_json(self.settings.processed_dir / f"{doc_id}.json")
        if not isinstance(payload, dict):
            return None
        pages = payload.get("pages", [])
        if max_pages is not None:
            payload = {**payload, "pages": pages[:max_pages]}
        return payload

    def stats(self) -> dict[str, int]:
        """Return counts used by the dashboard."""
        return {
            "documents": self.registry.count(),
            "pages": self.registry.total_pages(),
            "chunks": self.vector_store.count(),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _process(
        self,
        *,
        stored_path: Path,
        original_filename: str,
        file_hash: str,
        doc_id: str,
        saved_new: bool,
    ) -> IngestionResult:
        processed_path = self.settings.processed_dir / f"{doc_id}.json"
        vectors_added = False
        try:
            pages = self.loader.load(stored_path)
            chunks = self.splitter.split(
                pages,
                doc_id=doc_id,
                filename=original_filename,
                file_hash=file_hash,
            )
            if not chunks:
                raise EmptyDocumentError(
                    f"'{original_filename}' produced no text chunks."
                )

            self.vector_store.add_chunks(
                chunks, ids=[chunk.metadata["chunk_id"] for chunk in chunks]
            )
            vectors_added = True

            page_payload = [
                {
                    "page": int(page.metadata.get("page_number", 0)),
                    "text": page.page_content,
                }
                for page in pages
                if page.page_content.strip()
            ]
            title = self._derive_title(pages, original_filename)

            record = DocumentRecord(
                doc_id=doc_id,
                filename=original_filename,
                original_filename=original_filename,
                file_hash=file_hash,
                file_path=str(stored_path.relative_to(self.settings.data_dir)),
                page_count=len(page_payload),
                chunk_count=len(chunks),
                file_size_bytes=stored_path.stat().st_size,
                ingested_at=utc_now_iso(),
                title=title,
            )

            atomic_write_json(
                processed_path,
                {"document": record.to_dict(), "pages": page_payload},
            )
            self.registry.add(record)
            logger.info(
                "Ingested %s: %s pages, %s chunks",
                original_filename,
                record.page_count,
                record.chunk_count,
            )
            return IngestionResult(
                record=record,
                chunks=len(chunks),
                message=f"Indexed {record.page_count} pages into {len(chunks)} chunks.",
            )
        except Exception as exc:
            logger.exception("Ingestion failed for %s", original_filename)
            if vectors_added:
                try:
                    self.vector_store.delete_document(doc_id)
                except VectorStoreError:
                    logger.exception("Could not roll back vectors for %s", doc_id)
            try:
                processed_path.unlink(missing_ok=True)
            except OSError:
                pass
            if saved_new:
                try:
                    stored_path.unlink(missing_ok=True)
                except OSError:
                    pass
            if isinstance(exc, (PDFIngestionError, VectorStoreError)):
                raise
            raise PDFIngestionError(
                f"Failed to ingest '{original_filename}': {exc}"
            ) from exc

    @staticmethod
    def _derive_title(pages: Sequence, fallback_filename: str) -> str:
        for page in pages:
            for line in page.page_content.splitlines():
                candidate = line.strip()
                if 10 <= len(candidate) <= 180 and not candidate.lower().startswith(
                    ("arxiv:", "doi:", "http")
                ):
                    return candidate
        return Path(fallback_filename).stem

    @staticmethod
    def _is_inside(path: Path, directory: Path) -> bool:
        try:
            path.resolve().relative_to(directory.resolve())
            return True
        except ValueError:
            return False
