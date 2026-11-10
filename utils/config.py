"""Central configuration for ResearchIQ.

Configuration is read from environment variables (and an optional ``.env``
file), then non-secret runtime preferences saved from the Settings page are
layered on top. Credentials are never persisted by the application.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from utils.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_OVERRIDES_FILE = PROJECT_ROOT / "data" / "processed" / "runtime_settings.json"
LOGGER_NAME = "researchiq"

# Only user-facing, non-secret preferences may be persisted.
OVERRIDABLE_FIELDS: frozenset[str] = frozenset(
    {
        "ollama_model",
        "retrieval_top_k",
        "chunk_size",
        "chunk_overlap",
        "similarity_threshold",
        "max_validation_iterations",
        "langsmith_tracing",
    }
)


class Settings(BaseSettings):
    """Validated application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    app_name: str = "ResearchIQ"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    # Local LLM (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    ollama_temperature: float = Field(default=0.1, ge=0.0, le=1.5)
    ollama_num_ctx: int = Field(default=8192, ge=2048, le=32768)
    ollama_num_predict: int = Field(default=2000, ge=128, le=8192)
    ollama_timeout: int = Field(default=600, ge=10, le=1800)

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=32, ge=1, le=128)

    # Ingestion
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=200, ge=0, le=4000)
    max_upload_mb: int = Field(default=50, ge=1, le=500)

    # Retrieval
    retrieval_top_k: int = Field(default=4, ge=1, le=20)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    max_context_chunks: int = Field(default=12, ge=1, le=40)

    # Agent workflow
    max_validation_iterations: int = Field(default=2, ge=0, le=5)
    max_history_messages: int = Field(default=8, ge=0, le=50)

    # Storage (relative paths are resolved against the project root)
    data_dir: Path = PROJECT_ROOT / "data"
    vectorstore_dir: Path = PROJECT_ROOT / "vectorstore"
    chroma_collection: str = "researchiq_papers"

    # Optional LangSmith tracing
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "ResearchIQ"

    @model_validator(mode="after")
    def _validate_and_resolve(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        for field_name in ("data_dir", "vectorstore_dir"):
            value = Path(getattr(self, field_name)).expanduser()
            if not value.is_absolute():
                value = (PROJECT_ROOT / value).resolve()
            object.__setattr__(self, field_name, value)
        return self

    @property
    def uploaded_papers_dir(self) -> Path:
        return self.data_dir / "uploaded_papers"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.processed_dir / "reports"

    @property
    def logs_dir(self) -> Path:
        return self.processed_dir / "logs"

    @property
    def registry_file(self) -> Path:
        return self.processed_dir / "documents.json"

    def ensure_directories(self) -> None:
        """Create every directory the application writes to."""
        for path in (
            self.data_dir,
            self.uploaded_papers_dir,
            self.processed_dir,
            self.reports_dir,
            self.logs_dir,
            self.vectorstore_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def cache_signature(self) -> tuple[Any, ...]:
        """Stable key describing backend-affecting settings.

        The Streamlit layer uses this to know when cached services must be
        rebuilt after a settings change.
        """
        return (
            self.ollama_model,
            self.ollama_base_url,
            self.embedding_model,
            self.embedding_device,
            self.chunk_size,
            self.chunk_overlap,
            self.retrieval_top_k,
            self.similarity_threshold,
            self.max_context_chunks,
            self.max_validation_iterations,
            self.ollama_num_ctx,
            self.ollama_num_predict,
            self.ollama_timeout,
            str(self.data_dir),
            str(self.vectorstore_dir),
            self.chroma_collection,
            self.langsmith_tracing,
            bool(self.langsmith_api_key),
        )


def _read_overrides() -> dict[str, Any]:
    if not RUNTIME_OVERRIDES_FILE.exists():
        return {}
    try:
        data = json.loads(RUNTIME_OVERRIDES_FILE.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if k in OVERRIDABLE_FIELDS}
    except (json.JSONDecodeError, OSError):
        logging.getLogger(LOGGER_NAME).warning(
            "Could not read runtime settings from %s; using environment values.",
            RUNTIME_OVERRIDES_FILE,
        )
        return {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton application settings object."""
    settings = Settings()
    overrides = _read_overrides()
    if overrides:
        settings = settings.model_copy(update=overrides)
    settings.ensure_directories()
    return settings


def reset_settings_cache() -> None:
    """Clear the cached settings object (primarily for tests)."""
    get_settings.cache_clear()


def validate_runtime_updates(updates: dict[str, Any]) -> dict[str, Any]:
    """Validate settings coming from the UI before persisting them."""
    unknown = set(updates) - OVERRIDABLE_FIELDS
    if unknown:
        raise ConfigurationError(f"Unsupported settings: {sorted(unknown)}")

    cleaned: dict[str, Any] = {}
    for key, value in updates.items():
        if key in {"retrieval_top_k", "chunk_size", "max_validation_iterations"}:
            value = int(value)
        elif key == "chunk_overlap":
            value = int(value)
        elif key == "similarity_threshold":
            value = float(value)
        elif key == "langsmith_tracing":
            value = bool(value)
        elif key == "ollama_model":
            value = str(value).strip()
            if not value:
                raise ConfigurationError("Ollama model name cannot be empty.")
        cleaned[key] = value

    candidate = {**get_settings().model_dump(), **cleaned}
    try:
        Settings(**candidate)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    return cleaned


def update_runtime_settings(updates: dict[str, Any]) -> Settings:
    """Persist non-secret UI preferences and return refreshed settings."""
    cleaned = validate_runtime_updates(updates)
    stored = _read_overrides()
    stored.update(cleaned)

    RUNTIME_OVERRIDES_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_file = RUNTIME_OVERRIDES_FILE.with_suffix(".tmp")
    temp_file.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    temp_file.replace(RUNTIME_OVERRIDES_FILE)

    reset_settings_cache()
    return get_settings()


def configure_langsmith(settings: Settings) -> bool:
    """Apply LangSmith environment settings; returns whether tracing is on.

    Tracing stays optional. Without an API key it is automatically disabled so
    the application never attempts to send traces to a paid service.
    """
    enabled = bool(settings.langsmith_tracing and settings.langsmith_api_key.strip())
    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if enabled else "false"
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_api_key.strip():
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.strip()
    return enabled


def create_settings(**overrides: Any) -> Settings:
    """Build an isolated settings object (useful for tests and CLI tools)."""
    return Settings(**overrides)
