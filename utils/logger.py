"""Logging configuration for ResearchIQ.

Logs go to both the console and a rotating file under ``data/processed/logs``.
The setup function is idempotent so Streamlit reruns do not add duplicate
handlers.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "researchiq"
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    *,
    console: bool = True,
) -> logging.Logger:
    """Configure the ResearchIQ logger once and return it."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(_DEFAULT_FORMAT)

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=2_000_000,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError:
            logger.warning("Could not create log file at %s", log_file)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the ResearchIQ namespace."""
    if not name or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
