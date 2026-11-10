"""Markdown report persistence."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from utils.config import Settings
from utils.helpers import safe_filename, utc_now_iso

logger = logging.getLogger("researchiq.reports")

_REPORT_TITLE_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def report_title_from_markdown(markdown: str, fallback: str = "Research Report") -> str:
    match = _REPORT_TITLE_RE.search(markdown or "")
    if match:
        return match.group(1).strip()[:120]
    return fallback


def save_markdown_report(
    settings: Settings, content: str, title: str | None = None
) -> Path:
    """Persist a generated report and return its path."""
    if not content.strip():
        raise ValueError("Cannot save an empty report.")
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    resolved_title = title or report_title_from_markdown(content)
    timestamp = utc_now_iso().replace(":", "-").replace("+00:00", "Z")
    filename = f"{timestamp}_{safe_filename(resolved_title)}.md"
    path = settings.reports_dir / filename
    path.write_text(content, encoding="utf-8")
    logger.info("Saved report to %s", path)
    return path


def list_saved_reports(settings: Settings) -> list[Path]:
    """Return saved Markdown reports, newest first."""
    if not settings.reports_dir.exists():
        return []
    return sorted(settings.reports_dir.glob("*.md"), reverse=True)
