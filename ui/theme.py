"""Visual theme and CSS for the ResearchIQ dashboard."""

from __future__ import annotations

import streamlit as st

CSS = """
<style>
:root {
    --riq-primary: #6366f1;
    --riq-primary-soft: rgba(99, 102, 241, 0.14);
    --riq-cyan: #22d3ee;
    --riq-border: rgba(148, 163, 184, 0.20);
    --riq-surface: rgba(20, 29, 51, 0.72);
    --riq-text-muted: #9fb0c9;
    --riq-success: #34d399;
    --riq-warning: #fbbf24;
    --riq-danger: #fb7185;
}

html, body, [class*="css"] {
    font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(900px 500px at 8% -10%, rgba(99, 102, 241, 0.20), transparent 60%),
        radial-gradient(700px 420px at 102% 0%, rgba(34, 211, 238, 0.12), transparent 55%),
        #080d19;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1528 0%, #0a101f 100%);
    border-right: 1px solid var(--riq-border);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.4rem;
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

h1, h2, h3, h4 {
    letter-spacing: -0.02em;
    color: #f4f7ff;
}

h1 { font-weight: 720; }
h2 { font-weight: 650; }
p, li, label, span { color: #d6dfef; }

a { color: #a5b4fc; }

/* Brand */
.riq-brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.25rem 0 1rem 0;
}
.riq-brand-mark {
    width: 42px;
    height: 42px;
    border-radius: 13px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, #6366f1, #22d3ee);
    color: #08101f;
    font-size: 1.25rem;
    font-weight: 800;
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.35);
}
.riq-brand-name { font-size: 1.2rem; font-weight: 750; color: #fff; line-height: 1.1; }
.riq-brand-sub { font-size: 0.72rem; color: var(--riq-text-muted); text-transform: uppercase; letter-spacing: 0.12em; }

/* Hero */
.riq-hero {
    position: relative;
    overflow: hidden;
    border: 1px solid var(--riq-border);
    border-radius: 22px;
    padding: 1.9rem 2rem;
    margin-bottom: 1.3rem;
    background:
        linear-gradient(125deg, rgba(99, 102, 241, 0.20), rgba(15, 23, 42, 0.55) 52%, rgba(34, 211, 238, 0.12)),
        rgba(14, 22, 40, 0.85);
    box-shadow: 0 24px 60px rgba(3, 7, 18, 0.45);
}
.riq-hero::after {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    right: -90px;
    top: -130px;
    background: radial-gradient(circle, rgba(34, 211, 238, 0.25), transparent 70%);
    pointer-events: none;
}
.riq-eyebrow {
    display: inline-block;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #a5b4fc;
    margin-bottom: 0.55rem;
    font-weight: 650;
}
.riq-hero h1 { margin: 0 0 0.45rem 0; font-size: 2rem; }
.riq-hero p { margin: 0; max-width: 800px; color: #b9c7de; font-size: 0.98rem; line-height: 1.55; }

/* Cards / chips */
.riq-card {
    border: 1px solid var(--riq-border);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    background: var(--riq-surface);
    margin-bottom: 0.8rem;
}
.riq-card-title { font-weight: 650; color: #f1f5ff; margin-bottom: 0.25rem; }
.riq-muted { color: var(--riq-text-muted); font-size: 0.86rem; }
.riq-chip {
    display: inline-block;
    padding: 0.2rem 0.58rem;
    margin: 0.12rem 0.2rem 0.12rem 0;
    border-radius: 999px;
    background: var(--riq-primary-soft);
    border: 1px solid rgba(129, 140, 248, 0.30);
    color: #c7d2fe;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.riq-chip.ok { background: rgba(52, 211, 153, 0.12); border-color: rgba(52, 211, 153, 0.32); color: #a7f3d0; }
.riq-chip.warn { background: rgba(251, 191, 36, 0.12); border-color: rgba(251, 191, 36, 0.32); color: #fde68a; }
.riq-chip.bad { background: rgba(251, 113, 133, 0.12); border-color: rgba(251, 113, 133, 0.32); color: #fecdd3; }

.riq-source {
    border-left: 3px solid var(--riq-primary);
    background: rgba(99, 102, 241, 0.07);
    padding: 0.55rem 0.8rem;
    border-radius: 0 10px 10px 0;
    margin: 0.45rem 0;
    font-size: 0.86rem;
}
.riq-source b { color: #e0e7ff; }
.riq-source span { color: var(--riq-text-muted); }

.riq-status-dot {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    margin-right: 0.4rem;
}
.riq-status-dot.on { background: var(--riq-success); box-shadow: 0 0 12px rgba(52, 211, 153, 0.8); }
.riq-status-dot.off { background: var(--riq-danger); box-shadow: 0 0 12px rgba(251, 113, 133, 0.75); }

/* Sidebar radio as nav */
[data-testid="stSidebar"] [role="radiogroup"] { gap: 0.18rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    background: transparent;
    border-radius: 10px;
    padding: 0.48rem 0.65rem;
    transition: background 120ms ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(99, 102, 241, 0.10); }

/* Metrics */
[data-testid="stMetric"] {
    border: 1px solid var(--riq-border);
    border-radius: 15px;
    padding: 0.9rem 1rem;
    background: var(--riq-surface);
}
[data-testid="stMetricLabel"] { color: var(--riq-text-muted); }
[data-testid="stMetricValue"] { color: #f8fafc; font-weight: 700; }

/* Chat */
[data-testid="stChatMessage"] {
    border: 1px solid var(--riq-border);
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.55);
    padding: 0.55rem 0.75rem;
    margin-bottom: 0.6rem;
}
[data-testid="stChatInput"] {
    border-radius: 14px;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px;
    border: 1px solid rgba(129, 140, 248, 0.35);
    background: rgba(99, 102, 241, 0.10);
    color: #e0e7ff;
    font-weight: 600;
    transition: all 120ms ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: rgba(129, 140, 248, 0.75);
    background: rgba(99, 102, 241, 0.20);
    color: #ffffff;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    border: none;
    color: white;
}

/* Inputs */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
    background: rgba(15, 23, 42, 0.7) !important;
    border-color: var(--riq-border) !important;
    border-radius: 10px !important;
}

/* Expanders */
[data-testid="stExpander"] {
    border: 1px solid var(--riq-border);
    border-radius: 12px;
    background: rgba(15, 23, 42, 0.35);
    overflow: hidden;
}

hr { border-color: var(--riq-border); }
.stAlert { border-radius: 12px; }

@media (max-width: 740px) {
    .riq-hero { padding: 1.3rem 1.2rem; border-radius: 16px; }
    .riq-hero h1 { font-size: 1.5rem; }
    .block-container { padding-top: 1.4rem; }
}
</style>
"""


def inject_theme() -> None:
    """Inject the ResearchIQ stylesheet."""
    st.markdown(CSS, unsafe_allow_html=True)
