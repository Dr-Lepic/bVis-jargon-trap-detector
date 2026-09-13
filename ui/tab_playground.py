"""Single-Case Evaluation Playground tab for bVis.

Implements the dual-column layout defined in plan.md §3.2:
  - Image input selector (Upload / Local path)
  - True diagnosis & wrong-target text inputs
  - Blind Test Mode toggle
  - Plain report textarea with live word counter
  - Jargon report textarea with live word counter + AI generation drawer
  - Evaluate button → Verdict hero banner

Public interface:
    render_playground_tab(api_key, judge_model, writer_model) -> None
"""

from __future__ import annotations

import io
import random
from typing import Optional

import streamlit as st
from PIL import Image

from core import (
    call_openrouter_multimodal,
    call_openrouter_text,
    evaluate_trap_result,
    format_judge_prompt,
    format_writer_prompt,
    parse_judge_verdict,
    resize_and_encode_image,
)
from ui.theme import section_label, verdict_banner, word_count_line


# ---------------------------------------------------------------------------
# Sample cases bundled for quick demo usage
# ---------------------------------------------------------------------------

_SAMPLE_CASES: list[dict] = [
    {
        "label": "Case A – Darier's Disease",
        "disease_true": "Darier's Disease",
        "disease_wrong": "Lichen Planus",
        "plain_report": (
            "The lesion shows characteristic keratotic papules with a greasy, "
            "crusted surface distributed in seborrhoeic areas. The pattern is "
            "consistent with follicular dyskeratosis as seen in Darier's disease, "
            "with typical \"cobblestone\" surface texture and tan-brown colouration."
        ),
        "jargon_report": "",  # left empty for AI generation demo
    },
    {
        "label": "Case B – Melanocytic Nevus",
        "disease_true": "Melanocytic Nevus (Mole)",
        "disease_wrong": "Dysplastic Nevus",
        "plain_report": (
            "A well-defined, uniformly pigmented brown macule with smooth, "
            "regular borders. The lesion is symmetrical with homogeneous "
            "pigmentation and no atypical features on clinical examination."
        ),
        "jargon_report": "",
    },
]


# ---------------------------------------------------------------------------
# Internal helpers (pure-logic, importable by tests)
# ---------------------------------------------------------------------------

def _count_words(text: str) -> int:
    """Returns the number of whitespace-separated words in *text*."""
    return len(text.split()) if text.strip() else 0


def _randomise_report_order(
    plain_report: str,
    jargon_report: str,
) -> tuple[str, str, str]:
    """Randomly assigns plain/jargon reports to positions A and B.

    Returns:
        (report_a, report_b, correct_label) where correct_label is 'A' or 'B'.
    """
    if random.random() < 0.5:
        return plain_report, jargon_report, "A"
    return jargon_report, plain_report, "B"


def _build_verdict_key(prefix: str = "playground") -> str:
    return f"{prefix}_verdict_result"


# ---------------------------------------------------------------------------
# Main render function (called by app.py)
# ---------------------------------------------------------------------------

def render_playground_tab(
    api_key: str,
    judge_model: str,
    writer_model: str,
) -> None:
    """Renders the Single Evaluation Playground tab.

    Args:
        api_key: OpenRouter API key from the sidebar.
        judge_model: Selected VLM judge model identifier.
        writer_model: Selected text LLM for jargon generation.
    """
    # ---- Image input section -----------------------------------------------
    section_label("🖼️ Step 1 — Image Input")

    input_mode = st.radio(
        "Image source",
        options=["📤 Upload Image", "📁 Local File Path", "🧪 Use Sample Case"],
        horizontal=True,
        label_visibility="collapsed",
        key="pg_input_mode",
    )

    image_obj: Optional[Image.Image] = None
    case_defaults = {}

    if input_mode == "📤 Upload Image":
        uploaded = st.file_uploader(
            "Upload a dermatological image",
            type=["jpg", "jpeg", "png", "webp"],
            key="pg_upload",
            label_visibility="collapsed",
        )
        if uploaded:
            image_obj = Image.open(io.BytesIO(uploaded.read()))

    elif input_mode == "📁 Local File Path":
        path_str = st.text_input(
            "Absolute path to image",
            placeholder="C:\\images\\case_001.jpg",
            key="pg_filepath",
            label_visibility="collapsed",
        )
        if path_str:
            try:
                image_obj = Image.open(path_str)
            except Exception as exc:
                st.error(f"Could not open file: {exc}")

    else:  # Sample case
        sample_idx = st.selectbox(
            "Choose a sample case",
            options=list(range(len(_SAMPLE_CASES))),
            format_func=lambda i: _SAMPLE_CASES[i]["label"],
            key="pg_sample_idx",
            label_visibility="collapsed",
        )
        case_defaults = _SAMPLE_CASES[sample_idx]
        st.info(
            "📌 No real patient image is bundled. Upload a matching dermatology image "
            "or use the plain report text only (the judge will still analyse the image "
            "you provide alongside these reports).",
            icon="ℹ️",
        )

    st.divider()

    # ---- Dual-column layout ------------------------------------------------
    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        section_label("📷 Case & Visual Context")

        if image_obj is not None:
            # Thumbnailed preview (max 400px wide in the UI)
            preview_img = image_obj.copy()
            preview_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            st.image(preview_img, use_container_width=True)

            dims = image_obj.size
            st.caption(f"Original size: {dims[0]}×{dims[1]}px → encoded at ≤768px")
        else:
            st.markdown(
                '<div style="height:180px;display:flex;align-items:center;'
                'justify-content:center;border:2px dashed #2e334d;border-radius:10px;'
                'color:#8b90a7;font-size:0.9rem;">No image selected</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Diagnosis labels (hidden in blind mode)
        blind_mode = st.toggle(
            "🙈 Blind Test Mode — hide diagnosis labels",
            key="pg_blind_mode",
            value=False,
        )

        if not blind_mode:
            true_diag = st.text_input(
                "✅ True Diagnosis (plain report describes this)",
                value=case_defaults.get("disease_true", ""),
                placeholder="e.g. Darier's Disease",
                key="pg_true_diag",
            )
            wrong_target = st.text_input(
                "🎯 Wrong Target (jargon report describes this)",
                value=case_defaults.get("disease_wrong", ""),
                placeholder="e.g. Lichen Planus",
                key="pg_wrong_target",
            )
        else:
            true_diag = st.session_state.get("pg_true_diag", case_defaults.get("disease_true", ""))
            wrong_target = st.session_state.get("pg_wrong_target", case_defaults.get("disease_wrong", ""))
            st.caption("*Labels hidden — guess before evaluating!*")

    with col_right:
        # ---- Plain report --------------------------------------------------
        section_label("📄 Plain Report (Ground Truth)")
        plain_text: str = st.session_state.get("pg_plain_report", case_defaults.get("plain_report", ""))
        word_count_line(_count_words(plain_text))
        plain_report = st.text_area(
            "Plain report",
            value=plain_text,
            height=160,
            placeholder="Factual, verified clinical description of the lesion…",
            key="pg_plain_report",
            label_visibility="collapsed",
        )

        st.divider()

        # ---- Jargon report -------------------------------------------------
        section_label("🧪 Jargon Report (Adversarial Trap)")
        jargon_text: str = st.session_state.get("pg_jargon_report", case_defaults.get("jargon_report", ""))
        word_count_line(_count_words(jargon_text))
        jargon_report = st.text_area(
            "Jargon report",
            value=jargon_text,
            height=160,
            placeholder="Confident, jargon-dense narrative describing a wrong condition…",
            key="pg_jargon_report",
            label_visibility="collapsed",
        )

        # ---- AI jargon generation drawer -----------------------------------
        with st.expander("✨ Generate Jargon Report with AI"):
            min_w = st.slider("Min words", 60, 120, 90, 10, key="pg_min_words")
            max_w = st.slider("Max words", 100, 200, 130, 10, key="pg_max_words")
            extra_ctx = st.text_input(
                "Extra context (optional)",
                placeholder="e.g. Patient is 65yo male with facial lesion",
                key="pg_extra_ctx",
            )

            gen_disabled = not api_key or not wrong_target.strip()
            if gen_disabled:
                st.caption(
                    "⚠️ Provide an API key (sidebar) and a Wrong Target diagnosis to enable AI generation."
                )

            if st.button(
                "✨ Generate Jargon Report",
                disabled=gen_disabled,
                key="pg_gen_btn",
                use_container_width=True,
            ):
                with st.spinner("Generating adversarial narrative…"):
                    try:
                        prompt = format_writer_prompt(
                            wrong_condition=wrong_target,
                            actual_condition=true_diag or "the true diagnosis",
                            min_words=min_w,
                            max_words=max_w,
                            extra_context=extra_ctx,
                        )
                        generated = call_openrouter_text(
                            api_key=api_key,
                            model=writer_model,
                            prompt_text=prompt,
                            temperature=0.7,
                        )
                        st.session_state["pg_jargon_report"] = generated
                        st.success("Jargon report generated! Scroll up to see it.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Generation failed: {exc}")

    st.divider()

    # ---- Human Guess section (blind mode) ----------------------------------
    human_guess: Optional[str] = None
    if blind_mode:
        section_label("🧠 Your Diagnosis Guess (Blind Test)")
        human_guess = st.radio(
            "Which report do you think is the accurate one?",
            options=["Report A (I think this is correct)", "Report B (I think this is correct)", "Not sure"],
            horizontal=True,
            key="pg_human_guess",
        )

    # ---- Action bar --------------------------------------------------------
    section_label("⚖️ Step 2 — Evaluate")

    eval_disabled = not api_key or not plain_report.strip() or not jargon_report.strip() or image_obj is None
    btn_label = f"⚖️ Evaluate with Judge  ({judge_model.split('/')[1] if '/' in judge_model else judge_model})"

    if eval_disabled:
        missing = []
        if not api_key:
            missing.append("API key")
        if not plain_report.strip():
            missing.append("plain report")
        if not jargon_report.strip():
            missing.append("jargon report")
        if image_obj is None:
            missing.append("image")
        st.caption(f"⚠️ Missing: {', '.join(missing)}")

    if st.button(btn_label, type="primary", disabled=eval_disabled, use_container_width=True, key="pg_eval_btn"):
        _run_evaluation(
            api_key=api_key,
            judge_model=judge_model,
            plain_report=plain_report,
            jargon_report=jargon_report,
            image_obj=image_obj,  # type: ignore[arg-type]
        )

    # ---- Verdict display ---------------------------------------------------
    verdict_key = _build_verdict_key()
    if verdict_key in st.session_state:
        result = st.session_state[verdict_key]
        st.divider()

        # Optionally reveal human guess
        if blind_mode and human_guess:
            human_pick = "A" if "Report A" in human_guess else ("B" if "Report B" in human_guess else "?")
            st.info(
                f"🧠 Your guess: **Report {human_pick}** | "
                f"Correct answer: **Report {result['correct_label']}** | "
                f"Model picked: **Report {result['pick']}**"
            )

        verdict_banner(
            status=result["status"],
            pick=result["pick"],
            correct_label=result["correct_label"],
            words_plain=result["words_plain"],
            words_jargon=result["words_jargon"],
            justification=result["justification"],
        )


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def _run_evaluation(
    api_key: str,
    judge_model: str,
    plain_report: str,
    jargon_report: str,
    image_obj: Image.Image,
) -> None:
    """Calls the judge VLM and stores the verdict in session_state."""
    with st.spinner("Encoding image and calling judge model…"):
        try:
            # Randomise A/B order to avoid position bias
            report_a, report_b, correct_label = _randomise_report_order(plain_report, jargon_report)

            prompt = format_judge_prompt(report_a=report_a, report_b=report_b)
            b64_image = resize_and_encode_image(image_obj)

            raw_response = call_openrouter_multimodal(
                api_key=api_key,
                model=judge_model,
                prompt_text=prompt,
                base64_image=b64_image,
            )

            parsed = parse_judge_verdict(raw_response)
            trap = evaluate_trap_result(parsed["pick"], correct_label)

            st.session_state[_build_verdict_key()] = {
                "pick": parsed["pick"],
                "correct_label": correct_label,
                "status": trap["status"],
                "justification": parsed["justification"],
                "words_plain": _count_words(plain_report),
                "words_jargon": _count_words(jargon_report),
                "raw_response": raw_response,
            }
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
