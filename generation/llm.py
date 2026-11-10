"""Ollama chat model creation, invocation helpers, and health checks.

All inference is local. LangSmith tracing is optional and configured elsewhere;
this module deliberately has no cloud dependencies.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Protocol

from utils.config import Settings
from utils.errors import LLMConnectionError, LLMResponseError
from utils.helpers import extract_json_object, strip_thinking

logger = logging.getLogger("researchiq.llm")


class TextGenerator(Protocol):
    """Minimal protocol implemented by ChatOllama and test doubles."""

    def invoke(self, input: Any, **kwargs: Any) -> Any:  # pragma: no cover - protocol
        ...


def create_chat_model(settings: Settings) -> Any:
    """Create a local ChatOllama client from settings.

    The model is not downloaded here; ``ollama pull`` must have been run first.
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:  # pragma: no cover
        raise LLMConnectionError(
            "langchain-ollama is not installed. Run `uv sync`."
        ) from exc

    try:
        llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.ollama_temperature,
            num_ctx=settings.ollama_num_ctx,
            num_predict=settings.ollama_num_predict,
            keep_alive="10m",
            # Qwen3 always reasons with this Ollama template. Separating the
            # reasoning tokens keeps them out of the final answer (Ollama's
            # `think: false` does not suppress reasoning for this model).
            reasoning=True,
            validate_model_on_init=False,
            client_kwargs={"timeout": settings.ollama_timeout},
        )
        logger.info(
            "Configured Ollama model %s at %s",
            settings.ollama_model,
            settings.ollama_base_url,
        )
        return llm
    except Exception as exc:
        raise LLMConnectionError(
            f"Could not configure Ollama model '{settings.ollama_model}': {exc}"
        ) from exc


def _extract_content(response: Any) -> str:
    """Normalise LangChain message content (str or content blocks) to text."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return strip_thinking(content)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text:
                    parts.append(str(text))
        return strip_thinking("\n".join(parts))
    return strip_thinking(str(content))


def invoke_text(llm: TextGenerator, prompt: str) -> str:
    """Invoke the local LLM and return cleaned text.

    Raises:
        LLMConnectionError: If Ollama cannot be reached or generation fails.
        LLMResponseError: If the model returns an empty response.
    """
    try:
        response = llm.invoke(prompt)
    except Exception as exc:
        logger.exception("Ollama invocation failed")
        raise LLMConnectionError(
            "The local Ollama request did not complete. Confirm that Ollama is "
            "running (`ollama serve`), that the selected model is installed "
            "(`ollama pull qwen3:4b`), and that OLLAMA_TIMEOUT is high enough for "
            "CPU inference."
        ) from exc

    text = _extract_content(response)
    if not text:
        raise LLMResponseError("The local model returned an empty response.")
    return text


def invoke_json(llm: TextGenerator, prompt: str, retries: int = 1) -> dict[str, Any]:
    """Invoke the model and parse a JSON object, retrying once if malformed."""
    last_error: Exception | None = None
    current_prompt = prompt
    for attempt in range(retries + 1):
        try:
            text = invoke_text(llm, current_prompt)
            return extract_json_object(text)
        except ValueError as exc:
            last_error = exc
            logger.warning("JSON parsing failed (attempt %s): %s", attempt + 1, exc)
            current_prompt = (
                f"{prompt}\n\nIMPORTANT: Your previous response was not valid JSON. "
                "Return ONLY one valid JSON object with no markdown or commentary."
            )
        except LLMResponseError as exc:
            last_error = exc
            logger.warning("Empty JSON response (attempt %s)", attempt + 1)
    raise LLMResponseError(
        f"The model did not return valid JSON after {retries + 1} attempt(s): {last_error}"
    )


def _model_matches(requested: str, available: str) -> bool:
    requested = requested.strip()
    available = available.strip()
    if requested == available:
        return True
    if ":" not in requested and available.split(":", 1)[0] == requested:
        return True
    return requested.replace(":latest", "") == available.replace(":latest", "")


def list_ollama_models(
    base_url: str = "http://localhost:11434", timeout: int = 5
) -> list[str]:
    """Return locally installed Ollama model names.

    Raises:
        LLMConnectionError: If the Ollama server is unreachable.
    """
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise LLMConnectionError(
            f"Ollama is not reachable at {base_url}. Start it with `ollama serve`."
        ) from exc

    models = payload.get("models", []) if isinstance(payload, dict) else []
    names: list[str] = []
    for model in models:
        if isinstance(model, dict):
            name = model.get("name") or model.get("model")
            if name:
                names.append(str(name))
    return sorted(names)


def check_ollama_health(settings: Settings) -> dict[str, Any]:
    """Return a status payload for the dashboard and Settings page."""
    status: dict[str, Any] = {
        "base_url": settings.ollama_base_url,
        "model": settings.ollama_model,
        "reachable": False,
        "model_available": False,
        "models": [],
        "error": "",
    }
    try:
        models = list_ollama_models(settings.ollama_base_url)
        status["models"] = models
        status["reachable"] = True
        status["model_available"] = any(
            _model_matches(settings.ollama_model, name) for name in models
        )
        if not status["model_available"]:
            status["error"] = (
                f"Model '{settings.ollama_model}' is not installed. "
                f"Run `ollama pull {settings.ollama_model}`."
            )
    except LLMConnectionError as exc:
        status["error"] = str(exc)
    return status


def clear_llm_state() -> None:
    """Placeholder hook for future client caching; kept for UI symmetry."""
    return None
