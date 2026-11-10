"""Tests for Chroma storage, similarity retrieval, filtering, and citations."""

from __future__ import annotations

import pytest

from retrieval.retriever import ResearchRetriever
from retrieval.vector_store import VectorStoreError, VectorStoreManager


def test_vector_store_first_open_is_empty(vector_store) -> None:
    assert vector_store._store is None
    assert vector_store.count() == 0
    assert vector_store._store is not None


def test_vector_store_persists_across_managers(
    settings, fake_embeddings, sample_documents
) -> None:
    first = VectorStoreManager(
        persist_directory=settings.vectorstore_dir,
        collection_name="test_collection",
        embeddings=fake_embeddings,
    )
    first.add_chunks(
        sample_documents,
        [document.metadata["chunk_id"] for document in sample_documents],
    )
    assert first.count() == 3

    reopened = VectorStoreManager(
        persist_directory=settings.vectorstore_dir,
        collection_name="test_collection",
        embeddings=fake_embeddings,
    )
    assert reopened.count() == 3
    assert reopened.count_for_document("doc_alpha") == 2
    assert reopened.count_for_document("doc_missing") == 0
    results = reopened.similarity_search("transformer accuracy", k=2)
    assert results


def test_vector_store_add_search_and_count(vector_store, sample_documents) -> None:
    ids = [document.metadata["chunk_id"] for document in sample_documents]
    added = vector_store.add_chunks(sample_documents, ids)

    assert added == 3
    assert vector_store.count() == 3

    results = vector_store.similarity_search("transformer retrieval accuracy", k=3)
    assert results
    document, score = results[0]
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert document.metadata["filename"] in {"alpha.pdf", "beta.pdf"}
    assert "page_number" in document.metadata


def test_vector_store_metadata_filter(vector_store, sample_documents) -> None:
    vector_store.add_chunks(
        sample_documents,
        [document.metadata["chunk_id"] for document in sample_documents],
    )

    results = vector_store.similarity_search(
        "graph citation recommendation", k=3, doc_ids=["doc_beta"]
    )
    assert results
    assert all(document.metadata["doc_id"] == "doc_beta" for document, _ in results)

    missing = vector_store.similarity_search(
        "transformer retrieval", k=3, doc_ids=["doc_missing"]
    )
    assert missing == []


def test_vector_store_threshold_filters_results(vector_store, sample_documents) -> None:
    vector_store.add_chunks(
        sample_documents,
        [document.metadata["chunk_id"] for document in sample_documents],
    )

    strict = vector_store.similarity_search(
        "completely unrelated quantum banana recipe",
        k=3,
        score_threshold=0.99,
    )
    assert strict == []


def test_vector_store_delete_document(vector_store, sample_documents) -> None:
    vector_store.add_chunks(
        sample_documents,
        [document.metadata["chunk_id"] for document in sample_documents],
    )
    assert vector_store.count_for_document("doc_alpha") == 2

    vector_store.delete_document("doc_alpha")

    assert vector_store.count_for_document("doc_alpha") == 0
    assert vector_store.count() == 1

    remaining = vector_store.similarity_search("transformer accuracy", k=5)
    assert all(document.metadata["doc_id"] != "doc_alpha" for document, _ in remaining)


def test_vector_store_rejects_mismatched_ids(vector_store, sample_documents) -> None:
    with pytest.raises(VectorStoreError):
        vector_store.add_chunks(sample_documents, ["only-one-id"])


def test_retriever_builds_context_and_page_citations(
    vector_store, sample_documents, settings
) -> None:
    vector_store.add_chunks(
        sample_documents,
        [document.metadata["chunk_id"] for document in sample_documents],
    )
    retriever = ResearchRetriever(vector_store, settings)

    result = retriever.retrieve("transformer retrieval accuracy", top_k=2)

    assert not result.is_empty
    assert result.context.startswith("[Evidence 1]")
    assert "Document: alpha.pdf" in result.context
    assert result.citations
    for citation in result.citations:
        assert citation["filename"]
        assert int(citation["page"]) >= 1
    labels = {chunk["label"] for chunk in result.documents}
    assert "[Source: alpha.pdf, p. 2]" in labels


def test_retriever_empty_query_and_threshold(
    vector_store, sample_documents, settings
) -> None:
    retriever = ResearchRetriever(vector_store, settings)
    assert retriever.retrieve("").is_empty

    vector_store.add_chunks(
        sample_documents,
        [document.metadata["chunk_id"] for document in sample_documents],
    )
    settings.similarity_threshold = 0.999
    strict = retriever.retrieve("unrelated banana recipe quantum", top_k=3)
    assert strict.is_empty
    assert "threshold" in strict.note.lower()


def test_retriever_per_document_coverage(
    vector_store, sample_documents, settings
) -> None:
    vector_store.add_chunks(
        sample_documents,
        [document.metadata["chunk_id"] for document in sample_documents],
    )
    retriever = ResearchRetriever(vector_store, settings)

    result = retriever.retrieve(
        "transformer graph neural network methodology",
        top_k=1,
        doc_ids=["doc_alpha", "doc_beta"],
        per_document=True,
    )

    doc_ids = {chunk["doc_id"] for chunk in result.documents}
    assert "doc_alpha" in doc_ids
    assert "doc_beta" in doc_ids
