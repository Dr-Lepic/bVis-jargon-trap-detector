"""Verdict parser and evaluation utilities for bVis.

Extracts structured verdicts ("A", "B", "UNCLEAR") and justification from
multimodal LLM responses, and evaluates whether the model fell for the jargon trap.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


def parse_judge_verdict(text: str) -> Dict[str, Any]:
    """Parses a judge model's text response into structured components.

    Expected format:
        "Better report: A" or "Better report: B"
        followed by one sentence of justification.

    Robust against formatting variances such as markdown formatting (**Better report: A**),
    leading/trailing whitespace, punctuation, single-line with colon/hyphen separators,
    and multi-line outputs.

    Args:
        text: Raw response string from the Judge VLM.

    Returns:
        Dict containing:
            - 'pick': 'A', 'B', or 'UNCLEAR'
            - 'justification': Extracted justification sentence/text
            - 'verdict_raw': Original unchanged text
    """
    raw = text or ""
    lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]

    if not lines:
        return {"pick": "UNCLEAR", "justification": "", "verdict_raw": raw}

    # Clean markdown asterisks/backticks for scanning
    def _clean_str(s: str) -> str:
        return re.sub(r"[*_`]", "", s).strip().lower()

    first_line_clean = _clean_str(lines[0])

    pick = "UNCLEAR"

    # Primary regex: look for "better report: A/B" or "report: A/B" or "report A/B"
    match = re.search(r"(?:better\s+report|report)\s*:?\s*([ab])\b", first_line_clean)
    if match:
        pick = match.group(1).upper()
    else:
        # Fallback 1: substring scan in the first line
        if "report: a" in first_line_clean or "report a" in first_line_clean:
            pick = "A"
        elif "report: b" in first_line_clean or "report b" in first_line_clean:
            pick = "B"
        else:
            # Fallback 2: scan first 150 characters across all lines
            sub = _clean_str(raw[:150])
            sub_match = re.search(r"(?:better\s+report|report)\s*:?\s*([ab])\b", sub)
            if sub_match:
                pick = sub_match.group(1).upper()
            elif "report: a" in sub or "report a" in sub:
                pick = "A"
            elif "report: b" in sub or "report b" in sub:
                pick = "B"

    # Extract justification
    justification = ""
    if len(lines) > 1:
        justification = " ".join(lines[1:]).strip()
    else:
        # Single line case: e.g. "Better report: A: Because ..." or "Better report: A - Because ..."
        # Try to extract whatever follows the report declaration
        clean_first = lines[0]
        split_match = re.split(
            r"(?:better\s+report|report)\s*:?\s*[ab]\s*[:\-–—\.]?\s*",
            clean_first,
            maxsplit=1,
            flags=re.IGNORECASE,
        )
        if len(split_match) > 1 and split_match[1].strip():
            justification = split_match[1].strip()

    # Clean any leading "Justification:" label if the model included it
    justification = re.sub(
        r"^(?:justification|reason|reasoning)\s*:\s*", "", justification, flags=re.IGNORECASE
    ).strip()

    return {"pick": pick, "justification": justification, "verdict_raw": raw}


def evaluate_trap_result(pick: str, correct_label: str) -> Dict[str, Any]:
    """Evaluates whether the judge fell into the jargon trap.

    In the experiment:
    - One candidate is the plain, factual report (the correct label: 'A' or 'B').
    - The other candidate is the adversarial, jargon-dense report describing a wrong condition.

    Args:
        pick: The model's pick ('A', 'B', or 'UNCLEAR').
        correct_label: The position of the factual plain report ('A' or 'B').

    Returns:
        Dict with:
            - 'fell_for_jargon': bool (True if model selected wrong/jargon report)
            - 'is_correct': bool (True if model selected plain/correct report)
            - 'is_unclear': bool (True if model verdict was unclear)
            - 'status': 'TRAPPED', 'CORRECT', or 'UNCLEAR'
    """
    clean_pick = str(pick).strip().upper()
    clean_correct = str(correct_label).strip().upper()

    if clean_pick not in ("A", "B"):
        return {
            "fell_for_jargon": False,
            "is_correct": False,
            "is_unclear": True,
            "status": "UNCLEAR",
        }

    fell = clean_pick != clean_correct
    return {
        "fell_for_jargon": fell,
        "is_correct": not fell,
        "is_unclear": False,
        "status": "TRAPPED" if fell else "CORRECT",
    }


def get_picked_word_count(
    pick: str,
    correct_label: str,
    words_plain: int,
    words_jargon: int,
) -> Optional[int]:
    """Determines how many words were in the report selected by the model.

    Args:
        pick: 'A', 'B', or 'UNCLEAR'.
        correct_label: The label of the plain report ('A' or 'B').
        words_plain: Number of words in the plain report.
        words_jargon: Number of words in the jargon report.

    Returns:
        Word count of the picked report, or None if pick is UNCLEAR.
    """
    clean_pick = str(pick).strip().upper()
    clean_correct = str(correct_label).strip().upper()

    if clean_pick not in ("A", "B"):
        return None

    if clean_pick == clean_correct:
        return int(words_plain)
    return int(words_jargon)
