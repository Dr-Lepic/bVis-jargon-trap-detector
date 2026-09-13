"""Custom CSS design theme for the bVis Streamlit application.

Provides styled cards, badge pills, verdict callout banners, and general UI polish
via injected CSS. Import and call inject_css() at app startup.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# CSS string
# ---------------------------------------------------------------------------

_CSS = """
/* ============================
   FONTS & ROOT VARIABLES
   ============================ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --color-bg:            #0f1117;
    --color-surface:       #1a1d27;
    --color-surface-2:     #22263a;
    --color-border:        #2e334d;
    --color-accent:        #7c6af7;
    --color-accent-dim:    rgba(124,106,247,0.15);
    --color-text:          #e8eaf0;
    --color-text-muted:    #8b90a7;
    --color-success:       #22c55e;
    --color-danger:        #ef4444;
    --color-warning:       #f59e0b;
    --radius:              10px;
    --shadow:              0 4px 24px rgba(0,0,0,0.4);
}

/* Global font override */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ============================
   LAYOUT CONTAINERS
   ============================ */

/* Elevate the main block area */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}

/* ============================
   CARDS
   ============================ */
.jb-card {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
}

.jb-card-accent {
    border-left: 4px solid var(--color-accent);
}

/* ============================
   BADGE PILLS
   ============================ */
.jb-badge {
    display: inline-block;
    padding: 0.2em 0.75em;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}

.jb-badge-success {
    background: rgba(34,197,94,0.15);
    color: var(--color-success);
    border: 1px solid rgba(34,197,94,0.35);
}

.jb-badge-danger {
    background: rgba(239,68,68,0.15);
    color: var(--color-danger);
    border: 1px solid rgba(239,68,68,0.35);
}

.jb-badge-warning {
    background: rgba(245,158,11,0.15);
    color: var(--color-warning);
    border: 1px solid rgba(245,158,11,0.35);
}

.jb-badge-accent {
    background: var(--color-accent-dim);
    color: var(--color-accent);
    border: 1px solid rgba(124,106,247,0.35);
}

/* ============================
   VERDICT HERO BANNERS
   ============================ */
.jb-verdict {
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
    box-shadow: var(--shadow);
}

.jb-verdict-trapped {
    background: rgba(239,68,68,0.10);
    border: 2px solid var(--color-danger);
}

.jb-verdict-correct {
    background: rgba(34,197,94,0.10);
    border: 2px solid var(--color-success);
}

.jb-verdict-unclear {
    background: rgba(245,158,11,0.10);
    border: 2px solid var(--color-warning);
}

.jb-verdict h3 {
    margin: 0 0 0.5rem 0;
    font-size: 1.15rem;
    font-weight: 700;
}

.jb-verdict-trapped h3 { color: var(--color-danger); }
.jb-verdict-correct  h3 { color: var(--color-success); }
.jb-verdict-unclear  h3 { color: var(--color-warning); }

.jb-verdict-grid {
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
    margin-top: 0.8rem;
    font-size: 0.88rem;
    color: var(--color-text-muted);
}

.jb-verdict-grid span strong {
    color: var(--color-text);
}

.jb-justification {
    margin-top: 0.9rem;
    padding: 0.8rem 1rem;
    background: rgba(255,255,255,0.04);
    border-radius: 6px;
    font-style: italic;
    font-size: 0.9rem;
    color: var(--color-text-muted);
    border-left: 3px solid var(--color-accent);
}

/* ============================
   WORD COUNT COUNTER
   ============================ */
.jb-wordcount {
    font-size: 0.78rem;
    color: var(--color-text-muted);
    text-align: right;
    margin-bottom: 0.3rem;
}

/* ============================
   SECTION HEADERS
   ============================ */
.jb-section-label {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--color-accent);
    margin-bottom: 0.4rem;
}

/* ============================
   SIDEBAR POLISH
   ============================ */
section[data-testid="stSidebar"] {
    background: var(--color-surface) !important;
    border-right: 1px solid var(--color-border) !important;
}

/* ============================
   BUTTON OVERRIDES
   ============================ */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(124,106,247,0.3) !important;
}

/* Primary action button accent colour */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c6af7, #a78bfa) !important;
    border: none !important;
    color: #fff !important;
}
"""


def inject_css() -> None:
    """Injects the custom CSS theme into the Streamlit page."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Reusable HTML helpers
# ---------------------------------------------------------------------------

def badge(text: str, kind: str = "accent") -> str:
    """Returns an HTML badge pill string.

    Args:
        text: Label text.
        kind: One of 'success', 'danger', 'warning', 'accent'.
    """
    return f'<span class="jb-badge jb-badge-{kind}">{text}</span>'


def card(content_html: str, accent: bool = False) -> None:
    """Renders a styled card with arbitrary HTML content."""
    accent_cls = " jb-card-accent" if accent else ""
    st.markdown(
        f'<div class="jb-card{accent_cls}">{content_html}</div>',
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    """Renders a small, uppercase section heading."""
    st.markdown(
        f'<div class="jb-section-label">{text}</div>',
        unsafe_allow_html=True,
    )


def word_count_line(count: int) -> None:
    """Renders a small word-count line above a textarea."""
    st.markdown(
        f'<div class="jb-wordcount">Word count: {count}</div>',
        unsafe_allow_html=True,
    )


def verdict_banner(
    status: str,
    pick: str,
    correct_label: str,
    words_plain: int,
    words_jargon: int,
    justification: str,
) -> None:
    """Renders the full verdict hero card.

    Args:
        status: 'TRAPPED', 'CORRECT', or 'UNCLEAR'.
        pick: Model's pick ('A' or 'B').
        correct_label: The label ('A' or 'B') of the plain/correct report.
        words_plain: Word count of plain report.
        words_jargon: Word count of jargon report.
        justification: Model's one-sentence justification.
    """
    if status == "TRAPPED":
        css_cls = "jb-verdict-trapped"
        icon = "🚨"
        headline = "JUDGE FELL FOR THE JARGON TRAP!"
        body = (
            f"The model selected Report <strong>{pick}</strong> "
            f"(the deceptive jargon narrative) over the factual report."
        )
    elif status == "CORRECT":
        css_cls = "jb-verdict-correct"
        icon = "✅"
        headline = "JUDGE CORRECTLY IDENTIFIED THE PLAIN REPORT!"
        body = (
            f"The model selected Report <strong>{pick}</strong> "
            f"(the factual ground-truth narrative) — resisting the jargon trap."
        )
    else:
        css_cls = "jb-verdict-unclear"
        icon = "⚠️"
        headline = "VERDICT UNCLEAR"
        body = "The model did not return a clear A/B verdict."

    word_delta = words_jargon - words_plain
    delta_str = f"+{word_delta}" if word_delta >= 0 else str(word_delta)

    grid_html = (
        f'<div class="jb-verdict-grid">'
        f'<span>Correct Report: <strong>{correct_label}</strong></span>'
        f'<span>Model Pick: <strong>{pick}</strong></span>'
        f'<span>Word Delta: <strong>{delta_str} words</strong></span>'
        f'</div>'
    )

    justification_html = ""
    if justification:
        clean_just = justification.replace("<", "&lt;").replace(">", "&gt;")
        justification_html = (
            f'<div class="jb-justification">'
            f'<strong>Model Justification:</strong> {clean_just}'
            f'</div>'
        )

    html_card = (
        f'<div class="jb-verdict {css_cls}">'
        f'<h3>{icon} VERDICT: {headline}</h3>'
        f'<p style="margin:0;font-size:0.9rem;">{body}</p>'
        f'{grid_html}'
        f'{justification_html}'
        f'</div>'
    )
    st.markdown(html_card, unsafe_allow_html=True)
