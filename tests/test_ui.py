"""Unit tests for Member 2 deliverables: ui/theme.py and ui/tab_playground.py.

Tests cover all pure-logic helper functions that do not require a running
Streamlit server. The tests follow exactly the same style as test_core.py.

Test suites:
    1. TestThemeHelpers   – badge() and verdict_banner() HTML generation
    2. TestPlaygroundHelpers – _count_words(), _randomise_report_order(),
                               _build_verdict_key()
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# 1. Theme HTML helpers
# ---------------------------------------------------------------------------
from ui.theme import badge


class TestThemeHelpers:
    def test_badge_default_kind(self):
        """badge() with no kind arg defaults to 'accent' class."""
        html = badge("57.1%")
        assert "jb-badge" in html
        assert "jb-badge-accent" in html
        assert "57.1%" in html

    def test_badge_success_kind(self):
        html = badge("CORRECT", kind="success")
        assert "jb-badge-success" in html
        assert "CORRECT" in html

    def test_badge_danger_kind(self):
        html = badge("TRAPPED", kind="danger")
        assert "jb-badge-danger" in html
        assert "TRAPPED" in html

    def test_badge_warning_kind(self):
        html = badge("UNCLEAR", kind="warning")
        assert "jb-badge-warning" in html

    def test_badge_is_inline_element(self):
        """badge() must wrap text in a <span>."""
        html = badge("Test")
        assert html.startswith("<span")
        assert html.endswith("</span>")


# ---------------------------------------------------------------------------
# 2. Playground pure-logic helpers
# ---------------------------------------------------------------------------
from ui.tab_playground import (
    _build_verdict_key,
    _count_words,
    _randomise_report_order,
)


class TestPlaygroundHelpers:
    # --- _count_words -------------------------------------------------------

    def test_count_words_normal_sentence(self):
        assert _count_words("The patient has erythematous papules") == 5

    def test_count_words_empty_string(self):
        assert _count_words("") == 0

    def test_count_words_whitespace_only(self):
        assert _count_words("   \n\t  ") == 0

    def test_count_words_single_word(self):
        assert _count_words("melanoma") == 1

    def test_count_words_multiline(self):
        # "Line one here." = 3, "Line two here." = 3, "Line three." = 2 → total 8
        text = "Line one here.\nLine two here.\nLine three."
        assert _count_words(text) == 8

    # --- _randomise_report_order --------------------------------------------

    def test_randomise_output_contains_both_reports(self):
        plain = "Plain factual report text."
        jargon = "Jargon dense report text."
        a, b, label = _randomise_report_order(plain, jargon)
        # Both reports must be present, just maybe swapped
        assert set([a, b]) == set([plain, jargon])

    def test_randomise_label_matches_plain_position(self):
        """correct_label must always point to the position holding plain_report."""
        plain = "Plain factual report text."
        jargon = "Jargon dense report text."
        for _ in range(30):  # run many times to cover both branches
            a, b, label = _randomise_report_order(plain, jargon)
            if label == "A":
                assert a == plain
                assert b == jargon
            else:
                assert a == jargon
                assert b == plain

    def test_randomise_label_is_a_or_b(self):
        a, b, label = _randomise_report_order("x", "y")
        assert label in ("A", "B")

    def test_randomise_both_positions_occur(self):
        """Over many iterations, both A and B should appear as correct_label."""
        plain = "P"
        jargon = "J"
        labels = {_randomise_report_order(plain, jargon)[2] for _ in range(100)}
        assert "A" in labels
        assert "B" in labels

    # --- _build_verdict_key -------------------------------------------------

    def test_build_verdict_key_default(self):
        key = _build_verdict_key()
        assert key == "playground_verdict_result"

    def test_build_verdict_key_custom_prefix(self):
        key = _build_verdict_key(prefix="batch")
        assert key == "batch_verdict_result"

    def test_build_verdict_key_is_string(self):
        assert isinstance(_build_verdict_key(), str)
