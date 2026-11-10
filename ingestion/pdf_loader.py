"""Page-by-page PDF loading with validation and metadata normalisation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from utils.errors import EmptyDocumentError, PDFIngestionError
from utils.helpers import file_sha256

logger = logging.getLogger("researchiq.ingestion.pdf")


class PDFLoader:
    """Load PDF files into LangChain documents, one document per page."""

    def __init__(self) -> None:
        self._loader_cls: Any | None = None

    @property
    def loader_cls(self) -> Any:
        if self._loader_cls is None:
            try:
                from langchain_community.document_loaders import PyMuPDFLoader
            except ImportError as exc:  # pragma: no cover
                raise PDFIngestionError(
                    "PyMuPDFLoader is unavailable. Run `uv sync` to install "
                    "langchain-community and pymupdf."
                ) from exc
            self._loader_cls = PyMuPDFLoader
        return self._loader_cls

    def load(self, path: Path) -> list[Document]:
        """Load a PDF and attach normalised page metadata.

        Raises:
            PDFIngestionError: For missing, corrupt, or non-PDF files.
            EmptyDocumentError: When no page contains extractable text.
        """
        path = Path(path).expanduser().resolve()
        self._validate(path)

        try:
            loader = self.loader_cls(str(path))
            pages = loader.load()
        except Exception as exc:
            logger.exception("Failed to parse PDF %s", path)
            raise PDFIngestionError(
                f"Could not read '{path.name}'. The file may be corrupted, "
                f"encrypted, or unsupported: {exc}"
            ) from exc

        if not pages:
            raise EmptyDocumentError(f"'{path.name}' contains no pages.")

        normalised: list[Document] = []
        for page in pages:
            metadata = dict(page.metadata or {})
            zero_based_page = int(metadata.get("page", 0) or 0)
            metadata["page"] = zero_based_page
            metadata["page_number"] = zero_based_page + 1
            metadata["filename"] = path.name
            metadata["source"] = path.name
            metadata["source_file"] = str(path)
            page.metadata = metadata
            normalised.append(page)

        if not any(page.page_content.strip() for page in normalised):
            raise EmptyDocumentError(
                f"'{path.name}' has no extractable text. Scanned image-only PDFs "
                "need OCR before they can be indexed."
            )
        return normalised

    @staticmethod
    def _validate(path: Path) -> None:
        if not path.exists():
            raise PDFIngestionError(f"File not found: {path}")
        if not path.is_file():
            raise PDFIngestionError(f"Not a file: {path}")
        if path.suffix.lower() != ".pdf":
            raise PDFIngestionError(
                f"'{path.name}' is not a PDF. Upload only .pdf research papers."
            )
        if path.stat().st_size == 0:
            raise PDFIngestionError(f"'{path.name}' is empty.")
        with path.open("rb") as handle:
            header = handle.read(1024)
        if not header.lstrip().startswith(b"%PDF-"):
            raise PDFIngestionError(
                f"'{path.name}' does not look like a valid PDF (missing %PDF header)."
            )

    @staticmethod
    def file_hash(path: Path) -> str:
        """Return the SHA-256 hash used for duplicate detection."""
        return file_sha256(Path(path))
