"""Batch evaluation runner with validation and row-by-row checkpointing."""

from __future__ import annotations

import hashlib
import io
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from core import (
    calculate_batch_statistics,
    call_openrouter_multimodal,
    call_openrouter_text,
    evaluate_trap_result,
    format_judge_prompt,
    format_writer_prompt,
    get_picked_word_count,
    parse_judge_verdict,
    resize_and_encode_image,
)
from ui.theme import section_label


CHECKPOINT_NAME = "judge_output.csv"
REQUIRED_COLUMNS = {"image", "disease_true", "disease_wrong", "report_plain"}
COLUMN_ALIASES = {
    "image_path": "image",
    "image_file": "image",
    "filename": "image",
    "true_diagnosis": "disease_true",
    "actual_condition": "disease_true",
    "wrong_condition": "disease_wrong",
    "plain_report": "report_plain",
    "jargon_report": "report_jargon",
}


def normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy using the canonical batch column names."""
    renamed = {
        column: COLUMN_ALIASES[column.strip().lower()]
        for column in df.columns
        if column.strip().lower() in COLUMN_ALIASES
    }
    return df.rename(columns=renamed).copy()


def validate_batch_dataframe(df: pd.DataFrame) -> list[str]:
    """Return human-readable schema errors; an empty list means valid."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    errors = [f"Missing required column: {column}" for column in missing]
    if not df.empty:
        for column in REQUIRED_COLUMNS & set(df.columns):
            if df[column].fillna("").astype(str).str.strip().eq("").any():
                errors.append(f"Column '{column}' contains empty values.")
    else:
        errors.append("The CSV contains no rows.")
    return errors


def _case_id(row: pd.Series, index: int) -> str:
    value = row.get("case_id", row.get("id", index))
    return str(value)


def _assign_reports(plain: str, jargon: str, case_id: str) -> tuple[str, str, str]:
    """Deterministically counterbalance A/B so resumed runs keep the same order."""
    plain_first = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest(), 16) % 2 == 0
    return (plain, jargon, "A") if plain_first else (jargon, plain, "B")


def _read_image(image_name: str, image_folder: str, zip_bytes: bytes | None) -> bytes:
    if zip_bytes:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            normalized = image_name.replace("\\", "/").lstrip("/")
            names = archive.namelist()
            match = next(
                (name for name in names if name == normalized or Path(name).name == Path(normalized).name),
                None,
            )
            if not match:
                raise FileNotFoundError(f"{image_name} was not found in the ZIP archive")
            return archive.read(match)

    path = Path(image_name)
    if not path.is_absolute():
        path = Path(image_folder).expanduser() / path
    return path.read_bytes()


def _load_checkpoint(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame()


def _write_checkpoint(records: list[dict[str, Any]], path: Path) -> None:
    """Replace the checkpoint atomically to avoid a half-written CSV."""
    output = pd.DataFrame(records)
    temporary = path.with_suffix(".tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(path)


def _evaluate_row(
    row: pd.Series,
    index: int,
    api_key: str,
    judge_model: str,
    writer_model: str,
    image_folder: str,
    zip_bytes: bytes | None,
) -> dict[str, Any]:
    case_id = _case_id(row, index)
    plain = str(row["report_plain"]).strip()
    jargon = str(row.get("report_jargon", "") or "").strip()
    if not jargon or jargon.lower() == "nan":
        jargon = call_openrouter_text(
            api_key,
            writer_model,
            format_writer_prompt(
                wrong_condition=str(row["disease_wrong"]),
                actual_condition=str(row["disease_true"]),
            ),
        )

    report_a, report_b, correct_label = _assign_reports(plain, jargon, case_id)
    image_bytes = _read_image(str(row["image"]), image_folder, zip_bytes)
    response = call_openrouter_multimodal(
        api_key,
        judge_model,
        format_judge_prompt(report_a, report_b),
        resize_and_encode_image(image_bytes),
    )
    parsed = parse_judge_verdict(response)
    outcome = evaluate_trap_result(parsed["pick"], correct_label)
    words_plain, words_jargon = len(plain.split()), len(jargon.split())
    picked_words = get_picked_word_count(
        parsed["pick"], correct_label, words_plain, words_jargon
    )
    return {
        "case_id": case_id,
        "image": row["image"],
        "disease_true": row["disease_true"],
        "disease_wrong": row["disease_wrong"],
        "report_plain": plain,
        "report_jargon": jargon,
        "correct_label": correct_label,
        "pick": parsed["pick"],
        "fell_for_jargon": outcome["fell_for_jargon"],
        "status": outcome["status"],
        "words_plain": words_plain,
        "words_jargon": words_jargon,
        "picked_words": picked_words,
        "picked_longer": picked_words == max(words_plain, words_jargon) if picked_words else False,
        "justification": parsed["justification"],
        "verdict_raw": parsed["verdict_raw"],
        "judge_model": judge_model,
    }


def _load_source(path_text: str, uploaded: Any) -> pd.DataFrame | None:
    try:
        if uploaded is not None:
            return normalize_input_columns(pd.read_csv(uploaded))
        if path_text.strip():
            return normalize_input_columns(pd.read_csv(Path(path_text).expanduser()))
    except Exception as exc:
        st.error(f"Could not load CSV: {exc}")
    return None


def render_batch_tab(api_key: str, judge_model: str, writer_model: str) -> None:
    """Render data selection, resumable execution, live metrics, and download."""
    section_label("Step 1 — Select Data Sources")
    source_mode = st.radio("CSV source", ["Upload CSV", "Local CSV path"], horizontal=True)
    uploaded = st.file_uploader("Input CSV", type="csv") if source_mode == "Upload CSV" else None
    path_text = (
        st.text_input("CSV path", placeholder=r"C:\data\sample_cases.csv")
        if source_mode == "Local CSV path"
        else ""
    )
    image_folder = st.text_input("Image folder", placeholder=r"C:\data\images")
    image_zip = st.file_uploader("Or upload an image ZIP", type="zip")
    source_df = _load_source(path_text, uploaded)

    if source_df is None:
        st.info("Choose a CSV to configure the batch run.")
        return

    errors = validate_batch_dataframe(source_df)
    if errors:
        for error in errors:
            st.error(error)
        return

    zip_bytes = image_zip.getvalue() if image_zip else None
    st.success(f"Validated {len(source_df)} CSV rows.")
    section_label("Step 2 — Pipeline Configuration")
    max_rows = st.number_input("Maximum rows", 1, len(source_df), len(source_df))
    delay = st.number_input("Delay between calls (seconds)", 0.0, 30.0, 1.0, 0.5)
    checkpoint_text = st.text_input("Checkpoint path", value=CHECKPOINT_NAME)
    checkpoint_path = Path(checkpoint_text).expanduser()

    start, pause, stop, resume = st.columns(4)
    if start.button("Start Batch Run", type="primary", use_container_width=True):
        st.session_state.batch_records = []
        st.session_state.batch_position = 0
        st.session_state.batch_running = True
    if pause.button("Pause", use_container_width=True):
        st.session_state.batch_running = False
    if stop.button("Stop", use_container_width=True):
        st.session_state.batch_running = False
        st.session_state.batch_position = 0
    if resume.button("Resume Checkpoint", use_container_width=True):
        previous = _load_checkpoint(checkpoint_path)
        st.session_state.batch_records = previous.to_dict("records")
        completed = {str(item.get("case_id")) for item in st.session_state.batch_records}
        remaining_positions = [
            i for i in range(min(int(max_rows), len(source_df)))
            if _case_id(source_df.iloc[i], i) not in completed
        ]
        st.session_state.batch_position = remaining_positions[0] if remaining_positions else int(max_rows)
        st.session_state.batch_running = bool(remaining_positions)

    records = st.session_state.get("batch_records", [])
    position = int(st.session_state.get("batch_position", 0))
    limit = min(int(max_rows), len(source_df))
    st.progress(min(position / limit, 1.0), text=f"{min(position, limit)} / {limit} cases visited")

    result_df = pd.DataFrame(records)
    metrics = calculate_batch_statistics(result_df) if not result_df.empty else None
    k1, k2, k3 = st.columns(3)
    k1.metric("Evaluated", len(records))
    k2.metric("Trap Rate", f"{metrics['trap_rate']:.1%}" if metrics else "—")
    k3.metric("Length Bias", f"{metrics['length_confound_rate']:.1%}" if metrics else "—")

    if records:
        st.dataframe(result_df[["case_id", "image", "disease_true", "pick", "status"]].tail(8))
        st.download_button(
            "Download Current Results",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name=CHECKPOINT_NAME,
            mime="text/csv",
        )

    if st.session_state.get("batch_running", False) and position < limit:
        if not api_key:
            st.session_state.batch_running = False
            st.error("Enter an OpenRouter API key before starting the batch.")
            return
        try:
            record = _evaluate_row(
                source_df.iloc[position], position, api_key, judge_model, writer_model,
                image_folder, zip_bytes,
            )
        except Exception as exc:
            record = {
                "case_id": _case_id(source_df.iloc[position], position),
                "image": source_df.iloc[position]["image"],
                "disease_true": source_df.iloc[position]["disease_true"],
                "pick": "UNCLEAR",
                "fell_for_jargon": False,
                "status": "ERROR",
                "error": str(exc),
            }
        records.append(record)
        st.session_state.batch_records = records
        st.session_state.batch_position = position + 1
        _write_checkpoint(records, checkpoint_path)
        if delay:
            time.sleep(float(delay))
        st.rerun()

    if position >= limit and st.session_state.get("batch_running", False):
        st.session_state.batch_running = False
        st.success(f"Batch complete. Results saved to {checkpoint_path}.")
