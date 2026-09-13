"""bVis — Streamlit main entrypoint.

Responsibilities (Member 2 scope):
  - st.set_page_config
  - Inject CSS theme
  - Sidebar: API key input + validation, judge/writer model selectors, quick guide
  - Mount tabs: Playground | Batch Pipeline | Results Dashboard
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from core import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_WRITER_MODEL,
    POPULAR_JUDGE_MODELS,
    POPULAR_WRITER_MODELS,
    validate_api_key,
)
from ui.theme import inject_css

load_dotenv()

# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="bVis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**bVis** — Evaluating Multimodal Vision-Language Model "
            "Susceptibility to Authoritative Falsehoods.\n\n"
            "Built with ❤️ using Streamlit and OpenRouter."
        )
    },
)

inject_css()

# ---------------------------------------------------------------------------
# Global header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="padding:0.6rem 0 1rem 0;">
        <h1 style="margin:0;font-size:1.9rem;font-weight:800;
                   background:linear-gradient(90deg,#a78bfa,#7c6af7);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            🔬 bVis
        </h1>
        <p style="margin:0.25rem 0 0 0;color:#8b90a7;font-size:0.92rem;">
            Evaluating Multimodal Vision-Language Model Susceptibility to
            Authoritative Falsehoods
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<h2 style='margin-top:0;font-size:1.1rem;font-weight:700;'>⚙️ Configuration</h2>",
        unsafe_allow_html=True,
    )

    # --- API Key ---
    st.markdown("#### 🔑 OpenRouter API Key")
    env_key = os.getenv("OPENROUTER_API_KEY", "")
    api_key: str = st.text_input(
        "OpenRouter API Key",
        value=env_key,
        type="password",
        placeholder="sk-or-v1-…",
        label_visibility="collapsed",
        key="sidebar_api_key",
    )

    # Status indicator
    if api_key and api_key.strip():
        is_valid, status_msg = validate_api_key(api_key)
        if is_valid:
            st.markdown(
                f'<p style="color:#22c55e;font-size:0.82rem;margin:0;">🟢 {status_msg}</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<p style="color:#ef4444;font-size:0.82rem;margin:0;">🔴 {status_msg}</p>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<p style="color:#ef4444;font-size:0.82rem;margin:0;">🔴 Key Missing — '
            'enter your <a href="https://openrouter.ai/keys" target="_blank">OpenRouter key</a></p>',
            unsafe_allow_html=True,
        )

    st.divider()

    # --- Model selectors ---
    st.markdown("#### 🤖 Model Selection")

    judge_options = list(POPULAR_JUDGE_MODELS) + ["✏️ Custom Model..."]
    selected_judge = st.selectbox(
        "⚖️ Judge Model (VLM)",
        options=judge_options,
        index=0,
        key="sidebar_judge_model_select",
        help="Multimodal Vision-Language Model used to evaluate which report is more accurate.",
    )
    if selected_judge == "✏️ Custom Model...":
        judge_model = st.text_input(
            "Custom Judge Model Slug",
            value="",
            placeholder="e.g. google/gemini-2.0-flash-001",
            key="sidebar_custom_judge",
        )
    else:
        judge_model = selected_judge

    writer_options = list(POPULAR_WRITER_MODELS) + ["✏️ Custom Model..."]
    selected_writer = st.selectbox(
        "✍️ Writer Model (text LLM)",
        options=writer_options,
        index=0,
        key="sidebar_writer_model_select",
        help="Text-only LLM used to generate adversarial jargon reports.",
    )
    if selected_writer == "✏️ Custom Model...":
        writer_model = st.text_input(
            "Custom Writer Model Slug",
            value="",
            placeholder="e.g. meta-llama/llama-3.3-70b-instruct",
            key="sidebar_custom_writer",
        )
    else:
        writer_model = selected_writer

    st.divider()

    # --- Quick guide ---
    with st.expander("📖 Quick Guide", expanded=False):
        st.markdown(
            """
**What is the Jargon Trap?**

Multimodal AI judges are shown a dermatology image plus two candidate reports:

| Report | Description |
|--------|-------------|
| 📄 **Plain Report** | Factual, verified clinical description *(ground truth)* |
| 🧪 **Jargon Report** | Authoritative jargon describing a **wrong** condition |

Models often prefer the wrong report simply because it **sounds more impressive**.

**Jargon Trap Rate** = % of cases where the model picked the deceptive report.
A rate > 50% indicates systematic bias toward verbosity over accuracy.
            """
        )

    st.markdown(
        "<p style='font-size:0.72rem;color:#8b90a7;margin-top:1rem;'>"
        "API calls are made via <a href='https://openrouter.ai' target='_blank'>OpenRouter</a>."
        "</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Tab routing
# ---------------------------------------------------------------------------
tab_playground, tab_batch, tab_analyzer = st.tabs(
    [
        "🔬 Single Playground",
        "⚡ Batch Pipeline",
        "📊 Results Dashboard",
    ]
)

with tab_playground:
    from ui.tab_playground import render_playground_tab
    render_playground_tab(
        api_key=api_key,
        judge_model=judge_model,
        writer_model=writer_model,
    )

with tab_batch:
    from ui.tab_batch import render_batch_tab
    render_batch_tab(
        api_key=api_key,
        judge_model=judge_model,
        writer_model=writer_model,
    )

with tab_analyzer:
    from ui.tab_analyzer import render_analyzer_tab
    render_analyzer_tab()
