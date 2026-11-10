"""Tests for embedding model caching and error handling."""

from __future__ import annotations

import os

import pytest

from embeddings import embedding_model
from utils.errors import EmbeddingError

RUN_INTEGRATION = os.getenv("RUN_INTEGRATION_TESTS") == "1"


class StubHuggingFaceEmbeddings:
    instances: int = 0

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        StubHuggingFaceEmbeddings.instances += 1

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class BrokenEmbeddings:
    def __init__(self, **kwargs) -> None:
        raise RuntimeError("simulated model failure")


def test_embedding_model_is_cached(monkeypatch, tmp_path) -> None:
    StubHuggingFaceEmbeddings.instances = 0
    monkeypatch.setattr(
        "langchain_huggingface.HuggingFaceEmbeddings",
        StubHuggingFaceEmbeddings,
    )
    embedding_model.clear_embedding_cache()

    first = embedding_model.get_embeddings("stub-model", "cpu", 8)
    second = embedding_model.get_embeddings("stub-model", "cpu", 8)

    assert first is second
    assert StubHuggingFaceEmbeddings.instances == 1
    assert first.embed_query("probe") == [0.1, 0.2, 0.3]

    third = embedding_model.get_embeddings("stub-model", "cpu", 16)
    assert third is not first
    assert StubHuggingFaceEmbeddings.instances == 2


def test_embedding_initialisation_errors_are_wrapped(monkeypatch) -> None:
    monkeypatch.setattr("langchain_huggingface.HuggingFaceEmbeddings", BrokenEmbeddings)
    embedding_model.clear_embedding_cache()

    with pytest.raises(EmbeddingError, match="Could not load embedding model"):
        embedding_model.get_embeddings("broken-model", "cpu", 4)


@pytest.mark.integration
@pytest.mark.skipif(not RUN_INTEGRATION, reason="Set RUN_INTEGRATION_TESTS=1")
def test_real_huggingface_embedding_generation() -> None:
    embedding_model.clear_embedding_cache()
    embeddings = embedding_model.get_embeddings(
        "sentence-transformers/all-MiniLM-L6-v2", "cpu", 8
    )

    vector = embeddings.embed_query("retrieval augmented generation")
    assert len(vector) == 384

    first = embeddings.embed_query("research methodology and evaluation")
    second = embeddings.embed_query("experimental results and metrics")
    third = embeddings.embed_query("banana bread recipe with cinnamon")

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    assert cosine(first, second) > cosine(first, third)
