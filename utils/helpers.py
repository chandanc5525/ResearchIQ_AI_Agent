"""General-purpose helpers shared by ingestion, retrieval, and agents."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

THINK_BLOCK_RE = re.compile(
    r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE
)
THINK_TAG_RE = re.compile(r"</?(?:think|thinking)>", re.IGNORECASE)
CITATION_RE = re.compile(
    r"\[Source:\s*(?P<filename>[^,\]\n]+?)\s*,\s*p\.?\s*(?P<page>\d+)\]",
    re.IGNORECASE,
)
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._ ()\-]+")


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(data: bytes) -> str:
    """Compute the SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def safe_filename(filename: str, max_length: int = 140) -> str:
    """Make a filename safe for the local filesystem without changing its suffix."""
    name = Path(filename).name.strip()
    if not name:
        return "document.pdf"
    stem = Path(name).stem
    suffix = Path(name).suffix or ".pdf"
    stem = SAFE_FILENAME_RE.sub("_", stem).strip(" ._") or "document"
    return f"{stem[:max_length]}{suffix.lower()}"


def strip_thinking(text: str) -> str:
    """Remove Qwen3-style reasoning blocks from model output."""
    if not text:
        return ""
    cleaned = THINK_BLOCK_RE.sub("", text)
    cleaned = THINK_TAG_RE.sub("", cleaned)
    return cleaned.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a possibly noisy LLM response.

    Raises:
        ValueError: If no valid JSON object can be found.
    """
    if not text:
        raise ValueError("Empty LLM response")

    cleaned = strip_thinking(text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if escape:
                escape = False
                continue
            if char == "\\" and in_string:
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : index + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break
        start = cleaned.find("{", start + 1)

    raise ValueError("No valid JSON object found in LLM response")


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    """Deduplicate strings while keeping their original order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def citation_label(filename: str, page: int | str) -> str:
    """Build the canonical inline citation label used across the app."""
    return f"[Source: {filename}, p. {page}]"


def parse_citations(text: str) -> list[tuple[str, int]]:
    """Return all (filename, page) pairs referenced in an answer."""
    pairs: list[tuple[str, int]] = []
    for match in CITATION_RE.finditer(text or ""):
        filename = match.group("filename").strip()
        page = int(match.group("page"))
        if (filename, page) not in pairs:
            pairs.append((filename, page))
    return pairs


def truncate_text(text: str, max_chars: int = 1200) -> str:
    """Truncate text on a word boundary with an ellipsis."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    clipped = text[: max_chars - 1].rsplit(" ", 1)[0]
    return f"{clipped}…"


def read_json(path: Path, default: Any = None) -> Any:
    """Read JSON, returning a default on missing or malformed files."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically so a crash cannot corrupt the registry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temp_path.replace(path)


def human_file_size(size_bytes: int) -> str:
    """Format a byte count for display."""
    size = float(max(0, size_bytes))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


def is_pdf_bytes(data: bytes) -> bool:
    """Check the PDF magic header before handing bytes to a parser."""
    return data[:1024].lstrip().startswith(b"%PDF-")


def format_sources_markdown(citations: list[dict[str, Any]]) -> str:
    """Render a readable source list for reports and chat answers."""
    if not citations:
        return ""
    lines = ["### Sources"]
    for citation in citations:
        filename = citation.get("filename", "unknown")
        page = citation.get("page", "?")
        lines.append(f"- {filename} — page {page}")
    return "\n".join(lines)
