"""Cached backend access for Streamlit."""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from services import ApplicationServices, create_application
from utils.config import get_settings


@st.cache_resource(show_spinner="Loading ResearchIQ models and vector store…")
def _load_services(signature: tuple[Any, ...]) -> ApplicationServices:
    # The signature is only a cache key; settings are read fresh so runtime
    # preferences saved from the Settings page are always applied.
    _ = signature
    return create_application(get_settings())


def get_services() -> ApplicationServices:
    """Return the cached application services for the current settings."""
    return _load_services(get_settings().cache_signature())


def clear_service_cache() -> None:
    """Invalidate cached models, vector stores, and workflow objects."""
    _load_services.clear()


def get_health(force: bool = False, ttl_seconds: int = 45) -> dict[str, Any]:
    """Return cached Ollama/status info to avoid slow reruns when offline."""
    now = time.monotonic()
    checked = float(st.session_state.get("health_checked_at") or 0.0)
    cached = st.session_state.get("health_cache")
    if not force and cached and (now - checked) < ttl_seconds:
        return dict(cached)

    try:
        services = get_services()
        health = services.health()
    except Exception as exc:  # App services themselves may fail to load.
        health = {
            "reachable": False,
            "model_available": False,
            "models": [],
            "error": str(exc),
        }
    st.session_state["health_cache"] = health
    st.session_state["health_checked_at"] = now
    return dict(health)
