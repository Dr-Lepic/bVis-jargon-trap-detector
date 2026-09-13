"""Results dashboard for completed jargon-bias evaluation CSV files."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import calculate_batch_statistics
from ui.theme import section_label


def coerce_result_types(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize booleans/numbers and column aliases from CSV serialization or notebook runs."""
    clean = df.copy()

    # Column aliases support (e.g. notebook's image_file -> image)
    aliases = {
        "image_file": "image",
        "image_path": "image",
        "filename": "image",
        "true_diagnosis": "disease_true",
        "actual_condition": "disease_true",
        "wrong_condition": "disease_wrong",
        "plain_report": "report_plain",
        "jargon_report": "report_jargon",
        "ab": "correct_label",
        "label": "correct_label",
    }
    renamed = {}
    for col in clean.columns:
        canonical = aliases.get(col.strip().lower())
        if canonical and canonical not in clean.columns:
            renamed[col] = canonical
    if renamed:
        clean = clean.rename(columns=renamed)

    if "pick" in clean:
        clean["pick"] = clean["pick"].fillna("UNCLEAR").astype(str).str.upper()
    if "fell_for_jargon" in clean:
        clean["fell_for_jargon"] = clean["fell_for_jargon"].map(
            lambda value: str(value).strip().lower() in {"true", "1", "yes", "trapped"}
        )
    for column in ("words_plain", "words_jargon", "picked_words"):
        if column in clean:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")

    # Derive status if missing
    if "status" not in clean.columns and "fell_for_jargon" in clean.columns:
        def _get_status(row: pd.Series) -> str:
            p = str(row.get("pick", "")).upper()
            if p == "UNCLEAR":
                return "UNCLEAR"
            return "TRAPPED" if row.get("fell_for_jargon") else "CORRECT"
        clean["status"] = clean.apply(_get_status, axis=1)

    # Derive case_id if missing
    if "case_id" not in clean.columns:
        clean["case_id"] = [f"case-{i+1:02d}" for i in range(len(clean))]

    # Extract justification from verdict_raw if justification is missing
    if "justification" not in clean.columns and "verdict_raw" in clean.columns:
        from core import parse_judge_verdict
        clean["justification"] = clean["verdict_raw"].apply(
            lambda v: parse_judge_verdict(str(v)).get("justification", "") if pd.notna(v) else ""
        )

    # Compute word counts if reports are present but word counts are not
    if "report_plain" in clean.columns and "words_plain" not in clean.columns:
        clean["words_plain"] = clean["report_plain"].apply(lambda t: len(str(t).split()) if pd.notna(t) else None)
    if "report_jargon" in clean.columns and "words_jargon" not in clean.columns:
        clean["words_jargon"] = clean["report_jargon"].apply(lambda t: len(str(t).split()) if pd.notna(t) else None)

    return clean


def validate_results_dataframe(df: pd.DataFrame) -> list[str]:
    required = {"pick", "fell_for_jargon"}
    return [f"Missing required result column: {name}" for name in sorted(required - set(df.columns))]


def _read_results(uploaded: Any, local_path: str) -> pd.DataFrame | None:
    try:
        if uploaded is not None:
            return coerce_result_types(pd.read_csv(uploaded))
        if local_path.strip():
            return coerce_result_types(pd.read_csv(Path(local_path).expanduser()))
    except Exception as exc:
        st.error(f"Could not load results: {exc}")
    return None


def _forest_figure(metrics: dict[str, Any]) -> go.Figure:
    rate, low, high = metrics["trap_rate"], metrics["ci_low"], metrics["ci_high"]
    figure = go.Figure()
    figure.add_vline(x=0.5, line_dash="dash", line_color="#8b90a7")
    figure.add_trace(go.Scatter(
        x=[rate], y=["All scored cases"], mode="markers",
        marker={"size": 14, "color": "#8b5cf6"},
        error_x={"type": "data", "symmetric": False, "array": [high-rate], "arrayminus": [rate-low]},
        hovertemplate="Trap rate: %{x:.1%}<extra></extra>",
    ))
    figure.update_layout(title="Jargon Trap Rate with 95% Wilson CI", xaxis_tickformat=".0%", xaxis_range=[0, 1], height=330)
    return figure


def _disease_figure(breakdown: pd.DataFrame) -> go.Figure:
    ordered = breakdown.sort_values("trap_rate", ascending=True)
    figure = go.Figure(go.Bar(
        x=ordered["trap_rate"], y=ordered["disease"], orientation="h",
        marker_color="#ef4444",
        text=[f"{rate:.0%} ({trapped}/{total})" for rate, trapped, total in zip(ordered.trap_rate, ordered.trapped, ordered.total)],
        textposition="auto",
    ))
    figure.update_layout(title="Most-Trapped Conditions", xaxis_tickformat=".0%", xaxis_range=[0, 1], height=max(330, 55 * len(ordered)))
    return figure


def render_analyzer_tab() -> None:
    section_label("Results Data Source")
    source = st.radio("Result source", ["Upload CSV", "Local CSV path"], horizontal=True)
    uploaded = st.file_uploader("Evaluation result CSV", type="csv", key="analytics_upload") if source == "Upload CSV" else None
    local_path = st.text_input("Results path", value="judge_output.csv") if source == "Local CSV path" else ""
    df = _read_results(uploaded, local_path)
    if df is None:
        st.info("Upload a batch result CSV or enter its local path.")
        return
    errors = validate_results_dataframe(df)
    if errors:
        for error in errors:
            st.error(error)
        return

    metrics = calculate_batch_statistics(df)
    section_label("Statistical Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Scored", metrics["scored_cases"])
    c2.metric("Jargon Trap Rate", f"{metrics['trap_rate']:.1%}" if not math.isnan(metrics["trap_rate"]) else "—")
    ci = "—" if math.isnan(metrics["ci_low"]) else f"{metrics['ci_low']:.1%} – {metrics['ci_high']:.1%}"
    c3.metric("95% Wilson CI", ci)
    length_rate = metrics["length_confound_rate"]
    c4.metric("Length Confound", f"{length_rate:.1%}" if not math.isnan(length_rate) else "—")

    if metrics["scored_cases"]:
        left, right = st.columns(2)
        left.plotly_chart(_forest_figure(metrics), use_container_width=True)
        if not metrics["disease_breakdown"].empty:
            right.plotly_chart(_disease_figure(metrics["disease_breakdown"]), use_container_width=True)
        else:
            right.info("Add a disease_true column to see condition-level results.")

    section_label("Interactive Case Explorer")
    verdict = st.radio("Verdict", ["All", "Trapped", "Correct", "Unclear"], horizontal=True)
    search = st.text_input("Search diagnosis or image", placeholder="e.g. lichen planus")
    filtered = df.copy()
    if verdict == "Trapped":
        filtered = filtered[filtered["fell_for_jargon"]]
    elif verdict == "Correct":
        filtered = filtered[(~filtered["fell_for_jargon"]) & (filtered["pick"] != "UNCLEAR")]
    elif verdict == "Unclear":
        filtered = filtered[filtered["pick"] == "UNCLEAR"]
    if search:
        searchable = filtered.get("disease_true", pd.Series("", index=filtered.index)).astype(str)
        searchable += " " + filtered.get("image", pd.Series("", index=filtered.index)).astype(str)
        filtered = filtered[searchable.str.contains(search, case=False, na=False)]

    summary_columns = [name for name in ("case_id", "image", "disease_true", "disease_wrong", "pick", "status", "fell_for_jargon") if name in filtered]
    st.dataframe(filtered[summary_columns], use_container_width=True, hide_index=True)
    for index, row in filtered.iterrows():
        title = f"Case {row.get('case_id', index)} — {row.get('disease_true', 'Unknown diagnosis')}"
        with st.expander(title):
            st.write(f"Model pick: **{row.get('pick', 'UNCLEAR')}** · Correct position: **{row.get('correct_label', '—')}**")
            a, b = st.columns(2)
            a.markdown("**Plain report (ground truth)**")
            a.write(row.get("report_plain", "Not included"))
            b.markdown("**Jargon report (adversarial)**")
            b.write(row.get("report_jargon", "Not included"))
            st.markdown("**Model justification**")
            st.write(row.get("justification", "Not included"))
