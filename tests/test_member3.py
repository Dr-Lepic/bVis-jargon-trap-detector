"""Pure-logic tests for Member 3 batch and analytics modules."""

import pandas as pd

from ui.tab_analyzer import coerce_result_types, validate_results_dataframe
from ui.tab_batch import _assign_reports, normalize_input_columns, validate_batch_dataframe


def test_normalize_input_aliases():
    source = pd.DataFrame([{
        "filename": "case.png", "true_diagnosis": "A", "wrong_condition": "B",
        "plain_report": "plain",
    }])
    normalized = normalize_input_columns(source)
    assert {"image", "disease_true", "disease_wrong", "report_plain"} <= set(normalized.columns)


def test_validate_batch_dataframe_reports_missing_columns():
    errors = validate_batch_dataframe(pd.DataFrame({"image": ["one.png"]}))
    assert any("disease_true" in error for error in errors)


def test_assign_reports_is_deterministic_and_labels_plain():
    first = _assign_reports("plain", "jargon", "case-1")
    second = _assign_reports("plain", "jargon", "case-1")
    assert first == second
    assert first[0 if first[2] == "A" else 1] == "plain"


def test_coerce_csv_result_types():
    cleaned = coerce_result_types(pd.DataFrame({
        "pick": ["a", None], "fell_for_jargon": ["True", "0"], "picked_words": ["12", "bad"]
    }))
    assert cleaned["pick"].tolist() == ["A", "UNCLEAR"]
    assert cleaned["fell_for_jargon"].tolist() == [True, False]
    assert cleaned["picked_words"].isna().iloc[1]


def test_validate_results_dataframe():
    assert validate_results_dataframe(pd.DataFrame({"pick": ["A"]})) == [
        "Missing required result column: fell_for_jargon"
    ]
