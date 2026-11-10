"""Recursive, metadata-preserving text chunking."""

from __future__ import annotations

import logging
from typing import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.errors import PDFIngestionError

logger = logging.getLogger("researchiq.ingestion.splitter")


class DocumentTextSplitter:
    """Split page documents into overlapping, citable chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        if chunk_size < 100:
            raise PDFIngestionError("chunk_size must be at least 100 characters")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise PDFIngestionError("chunk_overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(
        self,
        documents: Sequence[Document],
        doc_id: str,
        filename: str,
        file_hash: str = "",
    ) -> list[Document]:
        """Split pages while preserving source provenance on every chunk."""
        raw_chunks = self._splitter.split_documents(list(documents))
        chunks: list[Document] = []
        for index, chunk in enumerate(raw_chunks):
            if not chunk.page_content.strip():
                continue
            metadata = dict(chunk.metadata or {})
            page_number = int(metadata.get("page_number", 0) or 0)
            if page_number <= 0:
                page_number = int(metadata.get("page", 0) or 0) + 1
            chunk_id = f"{doc_id}-c{index:05d}"
            metadata.update(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_index": len(chunks),
                    "page_number": page_number,
                    "page": page_number - 1,
                    "filename": filename,
                    "source": filename,
                    "file_hash": file_hash,
                }
            )
            chunk.metadata = metadata
            chunks.append(chunk)

        logger.info(
            "Split %s pages of %s into %s chunks",
            len(documents),
            filename,
            len(chunks),
        )
        return chunks
