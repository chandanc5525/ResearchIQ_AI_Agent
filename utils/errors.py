"""Domain-specific exceptions used across ResearchIQ.

Keeping exceptions in one module makes it possible for the UI and workflow
layers to handle failures consistently without importing implementation
details from every subsystem.
"""

from __future__ import annotations


class ResearchIQError(Exception):
    """Base class for all ResearchIQ errors."""


class ConfigurationError(ResearchIQError):
    """Raised when application configuration is invalid."""


class PDFIngestionError(ResearchIQError):
    """Raised when a PDF cannot be validated, loaded, or parsed."""


class EmptyDocumentError(PDFIngestionError):
    """Raised when a PDF contains no extractable text."""


class DuplicateDocumentError(ResearchIQError):
    """Raised when a document has already been ingested."""


class EmbeddingError(ResearchIQError):
    """Raised when the embedding model cannot be initialised or used."""


class VectorStoreError(ResearchIQError):
    """Raised when the vector database cannot be read or written."""


class LLMConnectionError(ResearchIQError):
    """Raised when Ollama is unreachable or the requested model is missing."""


class LLMResponseError(ResearchIQError):
    """Raised when the LLM response cannot be parsed or is unusable."""


class AgentExecutionError(ResearchIQError):
    """Raised when an agent cannot complete its task."""


class WorkflowError(ResearchIQError):
    """Raised when the LangGraph workflow fails unexpectedly."""
