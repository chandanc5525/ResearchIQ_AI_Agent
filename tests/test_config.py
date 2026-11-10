"""Tests for central settings, runtime overrides, and LangSmith opt-in."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils import config
from utils.config import (
    configure_langsmith,
    create_settings,
    get_settings,
    reset_settings_cache,
    update_runtime_settings,
    validate_runtime_updates,
)
from utils.errors import ConfigurationError


@pytest.fixture(autouse=True)
def isolated_runtime_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep override-file tests away from the developer's real settings."""
    runtime_file = tmp_path / "processed" / "runtime_settings.json"
    monkeypatch.setattr(config, "RUNTIME_OVERRIDES_FILE", runtime_file)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTORSTORE_DIR", str(tmp_path / "vectorstore"))
    reset_settings_cache()
    yield runtime_file
    reset_settings_cache()


def test_defaults_match_documented_chunking() -> None:
    settings = create_settings()

    assert settings.ollama_model == "qwen3:4b"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.retrieval_top_k == 4
    assert settings.data_dir.is_absolute()
    assert settings.vectorstore_dir.is_absolute()


def test_runtime_overrides_are_persisted_and_refreshed(
    isolated_runtime_settings: Path,
) -> None:
    updated = update_runtime_settings(
        {
            "retrieval_top_k": "6",
            "chunk_size": 1200,
            "chunk_overlap": 250,
            "similarity_threshold": "0.25",
            "langsmith_tracing": True,
        }
    )

    assert updated.retrieval_top_k == 6
    assert updated.chunk_size == 1200
    assert updated.chunk_overlap == 250
    assert updated.similarity_threshold == pytest.approx(0.25)
    assert updated.langsmith_tracing is True

    stored = json.loads(isolated_runtime_settings.read_text(encoding="utf-8"))
    assert stored["retrieval_top_k"] == 6
    assert stored["chunk_size"] == 1200
    assert "langsmith_api_key" not in stored

    assert get_settings().retrieval_top_k == 6
    assert config.OVERRIDABLE_FIELDS >= set(stored)


def test_secret_and_unknown_fields_are_rejected() -> None:
    with pytest.raises(ConfigurationError):
        validate_runtime_updates({"langsmith_api_key": "sk-should-not-be-saved"})

    with pytest.raises(ConfigurationError):
        validate_runtime_updates({"unknown_setting": 1})


def test_empty_model_name_and_invalid_chunking_are_rejected() -> None:
    with pytest.raises(ConfigurationError):
        validate_runtime_updates({"ollama_model": "   "})

    with pytest.raises(ConfigurationError):
        validate_runtime_updates({"chunk_size": 300, "chunk_overlap": 300})


def test_override_file_ignores_unknown_keys_and_survives_bad_json(
    isolated_runtime_settings: Path,
) -> None:
    isolated_runtime_settings.parent.mkdir(parents=True, exist_ok=True)
    isolated_runtime_settings.write_text(
        json.dumps(
            {
                "retrieval_top_k": 9,
                "langsmith_api_key": "sk-not-allowed",
            }
        ),
        encoding="utf-8",
    )
    reset_settings_cache()

    settings = get_settings()
    assert settings.retrieval_top_k == 9
    assert settings.langsmith_api_key == ""

    isolated_runtime_settings.write_text("not valid json", encoding="utf-8")
    reset_settings_cache()

    fallback = get_settings()
    assert fallback.retrieval_top_k == 4


def test_langsmith_requires_explicit_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable in (
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_API_KEY",
    ):
        # Seed with empty values so monkeypatch restores a clean environment even
        # though configure_langsmith writes to os.environ directly.
        monkeypatch.setenv(variable, "")

    disabled = create_settings(langsmith_tracing=True, langsmith_api_key="")
    assert configure_langsmith(disabled) is False
    assert config.os.environ["LANGSMITH_TRACING"] == "false"
    assert config.os.environ.get("LANGSMITH_API_KEY", "") == ""

    enabled = create_settings(
        langsmith_tracing=True,
        langsmith_api_key="  local-test-key  ",
    )
    assert configure_langsmith(enabled) is True
    assert config.os.environ["LANGSMITH_TRACING"] == "true"
    assert config.os.environ["LANGSMITH_API_KEY"] == "local-test-key"


def test_cache_signature_changes_with_backend_settings() -> None:
    baseline = create_settings()
    changed = create_settings(retrieval_top_k=8)

    assert baseline.cache_signature() != changed.cache_signature()
    assert baseline.cache_signature() == create_settings().cache_signature()
