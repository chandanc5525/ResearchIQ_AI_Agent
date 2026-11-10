"""Document records and the JSON registry used for deduplication."""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from utils.helpers import atomic_write_json, read_json, utc_now_iso
from utils.logger import get_logger

logger = get_logger("ingestion.registry")


def build_document_id(file_hash: str, filename: str) -> str:
    """Create a stable, short document identifier from content and name."""
    import hashlib

    digest = hashlib.sha256(f"{file_hash}:{filename}".encode("utf-8")).hexdigest()
    return f"doc_{digest[:16]}"


@dataclass
class DocumentRecord:
    """Metadata for one ingested research paper."""

    doc_id: str
    filename: str
    original_filename: str
    file_hash: str
    file_path: str
    page_count: int
    chunk_count: int
    file_size_bytes: int
    ingested_at: str = field(default_factory=utc_now_iso)
    title: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentRecord":
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        filtered = {key: value for key, value in data.items() if key in allowed}
        return cls(**filtered)


class DocumentRegistry:
    """Persist document metadata and prevent duplicate ingestion.

    Chroma holds vectors; this registry holds the authoritative list of
    uploaded papers plus their content hashes.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._records: dict[str, DocumentRecord] = {}
        self._load()

    def _load(self) -> None:
        raw = read_json(self.path, default=None)
        if raw is None:
            if self.path.exists():
                backup = self.path.with_name(
                    f"{self.path.stem}.corrupt-{utc_now_iso().replace(':', '-')}.json"
                )
                try:
                    self.path.replace(backup)
                    logger.warning("Corrupt registry moved to %s", backup)
                except OSError:
                    logger.warning("Corrupt document registry could not be backed up")
            self._records = {}
            return

        documents = raw.get("documents", raw) if isinstance(raw, dict) else {}
        if not isinstance(documents, dict):
            logger.warning(
                "Unexpected registry format; starting with an empty registry"
            )
            documents = {}
        for doc_id, data in documents.items():
            try:
                record = DocumentRecord.from_dict({**data, "doc_id": doc_id})
                self._records[doc_id] = record
            except TypeError:
                logger.warning("Skipping malformed registry entry %s", doc_id)

    def _save(self) -> None:
        payload = {
            "version": 1,
            "updated_at": utc_now_iso(),
            "documents": {
                doc_id: record.to_dict()
                for doc_id, record in sorted(self._records.items())
            },
        }
        atomic_write_json(self.path, payload)

    def list_documents(self) -> list[DocumentRecord]:
        with self._lock:
            return sorted(
                self._records.values(),
                key=lambda record: record.ingested_at,
                reverse=True,
            )

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            return self._records.get(doc_id)

    def find_by_hash(self, file_hash: str) -> DocumentRecord | None:
        with self._lock:
            for record in self._records.values():
                if record.file_hash == file_hash:
                    return record
        return None

    def add(self, record: DocumentRecord) -> None:
        with self._lock:
            self._records[record.doc_id] = record
            self._save()
        logger.info(
            "Registered document %s (%s pages, %s chunks)",
            record.doc_id,
            record.page_count,
            record.chunk_count,
        )

    def remove(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            record = self._records.pop(doc_id, None)
            if record is not None:
                self._save()
        return record

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def total_pages(self) -> int:
        with self._lock:
            return sum(record.page_count for record in self._records.values())

    def total_chunks(self) -> int:
        with self._lock:
            return sum(record.chunk_count for record in self._records.values())

    def is_empty(self) -> bool:
        return self.count() == 0
