"""Streamlit session-state initialisation and navigation helpers."""

from __future__ import annotations

import streamlit as st

PAGES: list[str] = [
    "Dashboard",
    "Document Upload",
    "Research Assistant",
    "Research Analysis",
    "Paper Comparison",
    "Research Reports",
    "Settings",
]

PAGE_ICONS: dict[str, str] = {
    "Dashboard": "◈",
    "Document Upload": "⇪",
    "Research Assistant": "✦",
    "Research Analysis": "◍",
    "Paper Comparison": "⇄",
    "Research Reports": "▤",
    "Settings": "⚙",
}

_DEFAULTS: dict[str, object] = {
    "messages": [],
    "analysis_result": None,
    "comparison_result": None,
    "report_result": None,
    "activity": [],
    "upload_nonce": 0,
    "pending_query": None,
    "pending_nav": None,
    "nav_page": PAGES[0],
    "health_cache": None,
    "health_checked_at": 0.0,
}


def init_session_state() -> None:
    """Create all session keys once per browser session."""
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value

    pending_nav = st.session_state.get("pending_nav")
    if pending_nav in PAGES:
        st.session_state["nav_page"] = pending_nav
        st.session_state["pending_nav"] = None


def request_navigation(page: str) -> None:
    """Navigate to a page safely before the sidebar widget is created."""
    if page in PAGES:
        st.session_state["pending_nav"] = page
        st.rerun()


def add_activity(action: str, detail: str) -> None:
    """Record a compact activity entry visible on the dashboard."""
    from utils.helpers import utc_now_iso

    activity = list(st.session_state.get("activity") or [])
    activity.insert(0, {"action": action, "detail": detail, "at": utc_now_iso()})
    st.session_state["activity"] = activity[:30]
