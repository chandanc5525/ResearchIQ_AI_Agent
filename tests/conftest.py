"""Shared fixtures for deterministic, offline-friendly tests."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Callable, Sequence

import pymupdf
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from embeddings.embedding_model import clear_embedding_cache
from ingestion.document_parser import DocumentRegistry
from ingestion.ingestion_pipeline import IngestionPipeline
from ingestion.pdf_loader import PDFLoader
from ingestion.text_splitter import DocumentTextSplitter
from retrieval.retriever import RetrievalResult
from retrieval.vector_store import VectorStoreManager
from utils.config import Settings, create_settings

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings for tests.

    No model download and no network access. Related texts share tokens and
    therefore produce higher cosine similarity.
    """

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _WORD_RE.findall((text or "").lower())
        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimension
            vector[index] += 1.0
        if not any(vector):
            vector[0] = 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeLLM:
    """A scriptable LangChain-compatible chat model test double."""

    def __init__(
        self,
        handler: Callable[[str], Any] | str = "Test response.",
    ) -> None:
        self.handler = handler
        self.prompts: list[str] = []

    def invoke(self, prompt: str, **_: Any) -> str:
        self.prompts.append(prompt)
        result = self.handler(prompt) if callable(self.handler) else self.handler
        if isinstance(result, Exception):
            raise result
        return str(result)


class FakeRetriever:
    """Retriever test double returning configured evidence."""

    def __init__(
        self,
        documents: Sequence[dict[str, Any]] | None = None,
        context: str | None = None,
        citations: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self.documents = list(documents or [])
        self.context = context or ""
        self.citations = list(citations or [])
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, query: str, **kwargs: Any) -> RetrievalResult:
        self.calls.append({"query": query, **kwargs})
        if not self.documents:
            return RetrievalResult(
                query=query, note="No evidence found in test corpus."
            )
        context = self.context or "\n\n".join(
            f"[Evidence {index}]\nDocument: {chunk['filename']}\n"
            f"Page: {chunk['page']}\nContent:\n{chunk['content']}"
            for index, chunk in enumerate(self.documents, start=1)
        )
        citations = self.citations or [
            {
                "filename": chunk["filename"],
                "page": chunk["page"],
                "doc_id": chunk.get("doc_id", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "score": chunk.get("score", 0.9),
            }
            for chunk in self.documents
        ]
        return RetrievalResult(
            documents=list(self.documents),
            context=context,
            citations=list(citations),
            query=query,
        )

    def search_for_evaluation(
        self, query: str, k: int = 5, doc_ids: Sequence[str] | None = None
    ) -> list[dict[str, Any]]:
        self.calls.append({"query": query, "k": k, "doc_ids": doc_ids})
        return self.documents[:k]


def make_pdf(path: Path, pages: Sequence[str]) -> Path:
    """Create a real text PDF for ingestion tests."""
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_textbox(
            pymupdf.Rect(50, 50, 545, 780),
            text,
            fontsize=11,
            fontname="helv",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(path))
    document.close()
    return path


def make_chunk(
    *,
    doc_id: str,
    filename: str,
    page: int,
    index: int,
    content: str,
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "doc_id": doc_id,
            "filename": filename,
            "source": filename,
            "page": page - 1,
            "page_number": page,
            "chunk_id": f"{doc_id}-c{index:05d}",
            "chunk_index": index,
        },
    )


def evidence_chunk(
    *,
    doc_id: str = "doc_alpha",
    filename: str = "alpha.pdf",
    page: int = 1,
    content: str = "The proposed transformer reaches 91% accuracy on the benchmark.",
    score: float = 0.92,
) -> dict[str, Any]:
    return {
        "content": content,
        "document": content,
        "filename": filename,
        "page": page,
        "page_number": page,
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}-p{page}",
        "chunk_index": 0,
        "score": score,
        "label": f"[Source: {filename}, p. {page}]",
    }


@pytest.fixture(autouse=True)
def _clear_embedding_singleton() -> Any:
    clear_embedding_cache()
    yield
    clear_embedding_cache()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return create_settings(
        data_dir=tmp_path / "data",
        vectorstore_dir=tmp_path / "vectorstore",
        chunk_size=500,
        chunk_overlap=80,
        retrieval_top_k=3,
        max_validation_iterations=1,
        langsmith_tracing=False,
    )


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def vector_store(
    settings: Settings, fake_embeddings: FakeEmbeddings
) -> VectorStoreManager:
    return VectorStoreManager(
        persist_directory=settings.vectorstore_dir,
        collection_name="test_collection",
        embeddings=fake_embeddings,
    )


@pytest.fixture
def registry(settings: Settings) -> DocumentRegistry:
    return DocumentRegistry(settings.registry_file)


@pytest.fixture
def ingestion(
    settings: Settings,
    vector_store: VectorStoreManager,
    registry: DocumentRegistry,
) -> IngestionPipeline:
    return IngestionPipeline(
        settings=settings,
        vector_store=vector_store,
        registry=registry,
        loader=PDFLoader(),
        splitter=DocumentTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
    )


@pytest.fixture
def pdf_bytes(tmp_path: Path) -> bytes:
    path = make_pdf(
        tmp_path / "sample.pdf",
        [
            "ResearchIQ Test Paper\n\nAbstract\nThis paper studies transformer models "
            "for document retrieval. The proposed method uses dense embeddings and "
            "achieves strong accuracy on an academic benchmark.\n\n"
            "1 Introduction\nRetrieval augmented generation combines retrieval with "
            "language models to improve factual grounding.",
            "2 Methodology\nWe use a bi-encoder transformer trained with contrastive "
            "learning. The dataset contains 12000 research abstracts. We evaluate "
            "using recall at ten and mean reciprocal rank.\n\n"
            "3 Results\nThe method reaches 91 percent accuracy and outperforms the "
            "baseline by 4 points.",
            "4 Limitations\nThe study uses English papers only and does not evaluate "
            "computational cost. Future work should study multilingual retrieval.",
        ],
    )
    return path.read_bytes()


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        make_chunk(
            doc_id="doc_alpha",
            filename="alpha.pdf",
            page=1,
            index=0,
            content=(
                "We propose a transformer retrieval model trained with contrastive "
                "learning and evaluate on a benchmark dataset."
            ),
        ),
        make_chunk(
            doc_id="doc_alpha",
            filename="alpha.pdf",
            page=2,
            index=1,
            content=(
                "The method achieves 91 percent accuracy, outperforming the baseline "
                "by four percentage points."
            ),
        ),
        make_chunk(
            doc_id="doc_beta",
            filename="beta.pdf",
            page=1,
            index=0,
            content=(
                "This study evaluates graph neural networks for citation recommendation "
                "using a large scholarly graph dataset."
            ),
        ),
    ]


@pytest.fixture
def evidence() -> list[dict[str, Any]]:
    return [
        evidence_chunk(
            doc_id="doc_alpha",
            filename="alpha.pdf",
            page=2,
            content="The transformer model achieves 91 percent accuracy on the benchmark.",
        ),
        evidence_chunk(
            doc_id="doc_beta",
            filename="beta.pdf",
            page=1,
            content="The graph neural network is evaluated on a citation dataset.",
            score=0.83,
        ),
    ]
