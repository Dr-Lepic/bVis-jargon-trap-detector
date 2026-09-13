"""Statistical engine for bVis.

Implements:
1. Wilson Score 95% Confidence Interval for binomial proportions (trap rates).
2. Length Confound Rate measuring verbosity bias.
3. High-level batch metrics aggregation for the UI dashboard.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Calculates the Wilson Score Confidence Interval for a binomial proportion.

    Given k successes (e.g. trapped cases) out of n trials (scored cases):
        phat = k / n
        denom = 1 + z^2 / n
        center = (phat + z^2 / (2n)) / denom
        margin = (z / denom) * sqrt(phat * (1 - phat) / n + z^2 / (4n^2))
        CI = [max(0.0, center - margin), min(1.0, center + margin)]

    Args:
        k: Number of successes/trapped cases (0 <= k <= n).
        n: Total number of evaluated cases.
        z: Z-score critical value (default: 1.96 for 95% two-sided confidence).

    Returns:
        (ci_lower, ci_upper) as floats bounded in [0.0, 1.0].
        Returns (nan, nan) if n == 0.
    """
    if n <= 0:
        return (float("nan"), float("nan"))

    k = max(0, min(k, n))
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom

    variance_term = (phat * (1.0 - phat)) / n + (z * z) / (4.0 * n * n)
    margin = (z / denom) * math.sqrt(max(0.0, variance_term))

    ci_low = max(0.0, center - margin)
    ci_high = min(1.0, center + margin)

    return (float(ci_low), float(ci_high))


def length_confound_rate(df: pd.DataFrame) -> float:
    """Measures whether the model systematically prefers whichever report is longer.

    Calculates:
        Length Confound = Count(picked_words == max(words_plain, words_jargon)) / n

    Args:
        df: DataFrame containing evaluation rows. Expected columns:
            - 'picked_words' (or 'pick' + 'correct_label' + 'words_plain' + 'words_jargon')
            - 'words_plain'
            - 'words_jargon'

    Returns:
        Float rate between 0.0 and 1.0, or float('nan') if no valid cases exist.
    """
    if df is None or len(df) == 0:
        return float("nan")

    df_copy = df.copy()

    # If words_plain and words_jargon are missing, see if we can derive them from report_plain and report_jargon
    if not {"words_plain", "words_jargon"}.issubset(df_copy.columns):
        if {"report_plain", "report_jargon"}.issubset(df_copy.columns):
            def _count_words(t: Any) -> int:
                return len(str(t).split()) if pd.notna(t) else 0

            df_copy["words_plain"] = df_copy["report_plain"].apply(_count_words)
            df_copy["words_jargon"] = df_copy["report_jargon"].apply(_count_words)
        else:
            return float("nan")

    # If picked_words not directly present, compute it if necessary columns exist
    if "picked_words" not in df_copy.columns:
        if {"pick", "correct_label", "words_plain", "words_jargon"}.issubset(df_copy.columns):
            def _calc_pw(row: pd.Series) -> Any:
                pick = str(row.get("pick", "")).strip().upper()
                correct = str(row.get("correct_label", "")).strip().upper()
                if pick not in ("A", "B"):
                    return None
                return row["words_plain"] if pick == correct else row["words_jargon"]

            df_copy["picked_words"] = df_copy.apply(_calc_pw, axis=1)
        else:
            return float("nan")

    # Guard: verify all required columns exist before calling dropna
    if not {"picked_words", "words_plain", "words_jargon"}.issubset(df_copy.columns):
        return float("nan")

    # Filter to scored cases with valid word counts
    valid = df_copy.dropna(subset=["picked_words", "words_plain", "words_jargon"])
    if len(valid) == 0:
        return float("nan")

    longer_words = valid[["words_plain", "words_jargon"]].max(axis=1)
    picked_longer_matches = (valid["picked_words"] == longer_words).sum()

    return float(picked_longer_matches / len(valid))


def calculate_batch_statistics(df: pd.DataFrame, z: float = 1.96) -> Dict[str, Any]:
    """Aggregates comprehensive evaluation metrics from a batch results DataFrame.

    Args:
        df: DataFrame containing batch results.
            Expected columns typically include:
            'pick', 'fell_for_jargon', 'disease_true', 'words_plain', 'words_jargon', etc.
        z: Z-score for Wilson CI (default: 1.96 for 95% CI).

    Returns:
        Dict with keys:
            - 'total_cases': int
            - 'scored_cases': int (pick != 'UNCLEAR')
            - 'unclear_cases': int (pick == 'UNCLEAR')
            - 'trapped_cases': int
            - 'correct_cases': int
            - 'trap_rate': float
            - 'ci_low': float
            - 'ci_high': float
            - 'length_confound_rate': float
            - 'disease_breakdown': pd.DataFrame with per-disease stats
    """
    if df is None or len(df) == 0:
        return {
            "total_cases": 0,
            "scored_cases": 0,
            "unclear_cases": 0,
            "trapped_cases": 0,
            "correct_cases": 0,
            "trap_rate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "length_confound_rate": float("nan"),
            "disease_breakdown": pd.DataFrame(),
        }

    total = len(df)
    is_unclear = df["pick"].astype(str).str.upper() == "UNCLEAR"
    unclear_count = int(is_unclear.sum())

    scored = df[~is_unclear]
    n_scored = len(scored)

    if n_scored > 0:
        # fell_for_jargon can be bool or int
        fell_series = scored["fell_for_jargon"].astype(bool)
        k_trapped = int(fell_series.sum())
        correct_count = n_scored - k_trapped
        trap_rate = float(k_trapped / n_scored)
        ci_lo, ci_hi = wilson_ci(k_trapped, n_scored, z=z)
    else:
        k_trapped = 0
        correct_count = 0
        trap_rate = float("nan")
        ci_lo, ci_hi = (float("nan"), float("nan"))

    lcr = length_confound_rate(df)

    # Per-disease breakdown if disease_true is present
    disease_df = pd.DataFrame()
    if "disease_true" in df.columns and n_scored > 0:
        records = []
        for disease, group in scored.groupby("disease_true"):
            g_n = len(group)
            g_k = int(group["fell_for_jargon"].astype(bool).sum())
            g_rate = g_k / g_n if g_n > 0 else 0.0
            g_lo, g_hi = wilson_ci(g_k, g_n, z=z)
            records.append({
                "disease": disease,
                "total": g_n,
                "trapped": g_k,
                "trap_rate": g_rate,
                "ci_low": g_lo,
                "ci_high": g_hi,
            })
        if records:
            disease_df = pd.DataFrame(records).sort_values("trap_rate", ascending=False).reset_index(drop=True)

    return {
        "total_cases": total,
        "scored_cases": n_scored,
        "unclear_cases": unclear_count,
        "trapped_cases": k_trapped,
        "correct_cases": correct_count,
        "trap_rate": trap_rate,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "length_confound_rate": lcr,
        "disease_breakdown": disease_df,
    }
