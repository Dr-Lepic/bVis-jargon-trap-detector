"""Unit test suite for the bVis Core Engine.

Tests:
1. Image processing (Lanczos resizing, base64 encoding/decoding, dimension preservation)
2. Prompt templates and formatting utilities
3. Verdict regex parser and outcome evaluator
4. Wilson Score Confidence Interval and length confound statistics
5. OpenRouter API client (retry mechanism, error handling, mock payloads)
"""

from __future__ import annotations

import io
import math
from unittest import mock
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from core.image_utils import (
    decode_base64_to_image,
    get_image_dimensions,
    resize_and_encode_image,
)
from core.openrouter import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_WRITER_MODEL,
    OpenRouterAuthError,
    OpenRouterError,
    OpenRouterRateLimitError,
    call_openrouter_multimodal,
    call_openrouter_text,
    validate_api_key,
)
from core.parser import (
    evaluate_trap_result,
    get_picked_word_count,
    parse_judge_verdict,
)
from core.prompts import (
    JARGON_WRITER_PROMPT,
    JUDGE_PROMPT,
    format_judge_prompt,
    format_writer_prompt,
)
from core.statistics import (
    calculate_batch_statistics,
    length_confound_rate,
    wilson_ci,
)


# =====================================================================
# 1. Image Processing Utilities Tests
# =====================================================================

class TestImageUtils:
    def test_resize_and_encode_large_image(self):
        """Large images should be downscaled so max dimension is 768px, preserving aspect ratio."""
        img = Image.new("RGB", (1536, 1024), color=(30, 60, 90))
        b64 = resize_and_encode_image(img, max_dim=768)

        decoded = decode_base64_to_image(b64)
        assert decoded.width == 768
        assert decoded.height == 512
        assert max(decoded.size) == 768

    def test_resize_and_encode_small_image_not_upscaled(self):
        """Images smaller than max_dim should retain their original dimensions."""
        img = Image.new("RGB", (400, 300), color=(100, 150, 200))
        b64 = resize_and_encode_image(img, max_dim=768)

        decoded = decode_base64_to_image(b64)
        assert decoded.size == (400, 300)

    def test_resize_and_encode_from_bytes(self):
        """Raw bytes input should be accepted and processed properly."""
        img = Image.new("RGB", (800, 600), color=(10, 20, 30))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        b64 = resize_and_encode_image(raw_bytes, max_dim=768)
        decoded = decode_base64_to_image(b64)
        assert max(decoded.size) == 768

    def test_decode_base64_with_data_uri_prefix(self):
        """Decoder should strip data:image/jpeg;base64, prefix if present."""
        img = Image.new("RGB", (100, 100), color=(50, 50, 50))
        b64 = resize_and_encode_image(img)
        data_uri = f"data:image/jpeg;base64,{b64}"

        decoded = decode_base64_to_image(data_uri)
        assert decoded.size == (100, 100)

    def test_get_image_dimensions(self):
        """get_image_dimensions returns correct (width, height) across inputs."""
        img = Image.new("RGB", (640, 480))
        assert get_image_dimensions(img) == (640, 480)

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        assert get_image_dimensions(buf.getvalue()) == (640, 480)


# =====================================================================
# 2. Prompt Formatting Tests
# =====================================================================

class TestPrompts:
    def test_writer_prompt_format(self):
        prompt = format_writer_prompt(
            wrong_condition="Lichen Planus",
            actual_condition="Darier's Disease",
            min_words=90,
            max_words=130,
        )
        assert "Lichen Planus" in prompt
        assert "Darier's Disease" in prompt
        assert "90 and 130 words" in prompt
        assert "Report:" in prompt

    def test_writer_prompt_with_extra_context(self):
        prompt = format_writer_prompt(
            wrong_condition="Basal Cell Carcinoma",
            extra_context="Patient is 65 years old with facial lesion.",
        )
        assert "Additional Context: Patient is 65 years old" in prompt

    def test_judge_prompt_format(self):
        prompt = format_judge_prompt(
            report_a="Report A text factual lesion.",
            report_b="Report B text jargon wrong condition.",
        )
        assert "Report A:\nReport A text factual lesion." in prompt
        assert "Report B:\nReport B text jargon wrong condition." in prompt
        assert 'Better report: A" or "Better report: B"' in prompt


# =====================================================================
# 3. Verdict Regex Parser & Trap Evaluator Tests
# =====================================================================

class TestParser:
    def test_standard_two_line_verdict(self):
        raw = "Better report: A\nReport A correctly identifies the papular eruption."
        result = parse_judge_verdict(raw)
        assert result["pick"] == "A"
        assert "Report A correctly identifies" in result["justification"]
        assert result["verdict_raw"] == raw

    def test_markdown_bolding(self):
        raw = "**Better report: B**\nJustification: Report B exhibits superior morphological precision."
        result = parse_judge_verdict(raw)
        assert result["pick"] == "B"
        assert "superior morphological precision" in result["justification"]

    def test_single_line_with_colon_or_dash(self):
        raw1 = "Better report: A: This narrative accurately mirrors the clinical lesion."
        result1 = parse_judge_verdict(raw1)
        assert result1["pick"] == "A"
        assert "accurately mirrors" in result1["justification"]

        raw2 = "Better report: B - Describes the classic dermoscopic pigment network."
        result2 = parse_judge_verdict(raw2)
        assert result2["pick"] == "B"
        assert "classic dermoscopic" in result2["justification"]

    def test_case_insensitivity_and_variants(self):
        raw = "better report b\nDetailed explanation follows."
        result = parse_judge_verdict(raw)
        assert result["pick"] == "B"

    def test_unclear_verdict(self):
        raw = "I am an AI and cannot evaluate medical images or diagnostic reports."
        result = parse_judge_verdict(raw)
        assert result["pick"] == "UNCLEAR"
        assert result["justification"] == ""

    def test_empty_verdict(self):
        result = parse_judge_verdict("")
        assert result["pick"] == "UNCLEAR"
        assert result["justification"] == ""

    def test_evaluate_trap_result(self):
        # When correct report is at position A
        trapped = evaluate_trap_result("B", "A")
        assert trapped["fell_for_jargon"] is True
        assert trapped["is_correct"] is False
        assert trapped["status"] == "TRAPPED"

        correct = evaluate_trap_result("A", "A")
        assert correct["fell_for_jargon"] is False
        assert correct["is_correct"] is True
        assert correct["status"] == "CORRECT"

        # When correct report is at position B
        trapped_b = evaluate_trap_result("A", "B")
        assert trapped_b["fell_for_jargon"] is True

        # Unclear verdict
        unclear = evaluate_trap_result("UNCLEAR", "A")
        assert unclear["fell_for_jargon"] is False
        assert unclear["is_unclear"] is True
        assert unclear["status"] == "UNCLEAR"

    def test_get_picked_word_count(self):
        # A is correct, pick is A -> picked words = plain words
        assert get_picked_word_count("A", "A", 45, 115) == 45
        # A is correct, pick is B -> picked words = jargon words
        assert get_picked_word_count("B", "A", 45, 115) == 115
        # Unclear pick
        assert get_picked_word_count("UNCLEAR", "A", 45, 115) is None


# =====================================================================
# 4. Statistical Engine Tests
# =====================================================================

class TestStatistics:
    def test_wilson_ci_benchmark_reference(self):
        """Validates Wilson Score CI matches the research notebook benchmark reference:
        For 28 trapped out of 49 cases (57.1%), 95% CI is [43.3% - 70.0%].
        """
        lo, hi = wilson_ci(28, 49)
        assert round(lo * 100, 1) == 43.3
        assert round(hi * 100, 1) == 70.0

    def test_wilson_ci_boundaries(self):
        # n = 0 returns (nan, nan)
        lo_zero, hi_zero = wilson_ci(0, 0)
        assert math.isnan(lo_zero) and math.isnan(hi_zero)

        # k = 0 (0% observed)
        lo_k0, hi_k0 = wilson_ci(0, 50)
        assert lo_k0 == 0.0
        assert 0.0 < hi_k0 < 0.1

        # k = n (100% observed)
        lo_kn, hi_kn = wilson_ci(50, 50)
        assert 0.9 < lo_kn < 1.0
        assert hi_kn == 1.0

    def test_length_confound_rate(self):
        df = pd.DataFrame({
            "pick": ["A", "B", "A", "UNCLEAR"],
            "correct_label": ["A", "A", "B", "A"],
            "words_plain": [40, 40, 50, 40],
            "words_jargon": [120, 120, 110, 120],
            "picked_words": [40, 120, 110, None],
        })
        # Row 0: picked 40, max is 120 (No match)
        # Row 1: picked 120, max is 120 (Match)
        # Row 2: picked 110, max is 110 (Match)
        # Row 3: UNCLEAR (Excluded)
        # Confound rate: 2 matches / 3 valid = 66.7%
        lcr = length_confound_rate(df)
        assert abs(lcr - (2 / 3)) < 1e-4

    def test_length_confound_rate_empty(self):
        assert math.isnan(length_confound_rate(pd.DataFrame()))

    def test_length_confound_rate_missing_word_columns(self):
        """When words_plain/words_jargon are missing (e.g. notebook judge_*.csv), returns nan without error."""
        df_no_words = pd.DataFrame({
            "image_file": ["1.png"],
            "pick": ["B"],
            "correct_label": ["A"],
            "fell_for_jargon": [True],
            "picked_words": [120],
        })
        assert math.isnan(length_confound_rate(df_no_words))

    def test_calculate_batch_statistics(self):
        df = pd.DataFrame({
            "image_file": ["1.jpg", "2.jpg", "3.jpg", "4.jpg"],
            "disease_true": ["Lichen Planus", "Lichen Planus", "Nevi", "Nevi"],
            "correct_label": ["A", "A", "B", "A"],
            "pick": ["B", "B", "B", "UNCLEAR"],
            "fell_for_jargon": [True, True, False, False],
            "words_plain": [40, 45, 50, 40],
            "words_jargon": [110, 120, 100, 110],
            "picked_words": [110, 120, 50, None],
        })
        stats = calculate_batch_statistics(df)
        assert stats["total_cases"] == 4
        assert stats["scored_cases"] == 3
        assert stats["unclear_cases"] == 1
        assert stats["trapped_cases"] == 2
        assert stats["correct_cases"] == 1
        assert abs(stats["trap_rate"] - (2 / 3)) < 1e-4

        breakdown = stats["disease_breakdown"]
        assert len(breakdown) == 2
        # Lichen Planus: 2 trapped out of 2 = 100%
        lp_row = breakdown[breakdown["disease"] == "Lichen Planus"].iloc[0]
        assert lp_row["trapped"] == 2
        assert lp_row["trap_rate"] == 1.0


# =====================================================================
# 5. OpenRouter Client Mock Tests
# =====================================================================

class TestOpenRouter:
    def test_auth_missing_key_raises(self):
        with pytest.raises(OpenRouterAuthError):
            call_openrouter_text(api_key="", model=DEFAULT_WRITER_MODEL, prompt_text="Hello")

        with pytest.raises(OpenRouterAuthError):
            call_openrouter_multimodal(
                api_key="   ",
                model=DEFAULT_JUDGE_MODEL,
                prompt_text="Hello",
                base64_image="xyz",
            )

    @mock.patch("requests.post")
    def test_call_openrouter_multimodal_success(self, mock_post):
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Better report: A\nClinical reasoning."}}]
        }
        mock_post.return_value = mock_resp

        result = call_openrouter_multimodal(
            api_key="sk-test-key",
            model=DEFAULT_JUDGE_MODEL,
            prompt_text="Evaluate image",
            base64_image="fake_base64_string",
        )
        assert result == "Better report: A\nClinical reasoning."

        # Verify call payload format
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == DEFAULT_JUDGE_MODEL
        msg_content = kwargs["json"]["messages"][0]["content"]
        assert msg_content[0]["type"] == "text"
        assert msg_content[1]["type"] == "image_url"
        assert "fake_base64_string" in msg_content[1]["image_url"]["url"]

    @mock.patch("requests.post")
    def test_call_openrouter_text_success(self, mock_post):
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Authoritative clinical jargon report..."}}]
        }
        mock_post.return_value = mock_resp

        result = call_openrouter_text(
            api_key="sk-test-key",
            model=DEFAULT_WRITER_MODEL,
            prompt_text="Generate report",
        )
        assert "Authoritative clinical jargon report" in result

    @mock.patch("time.sleep")
    @mock.patch("requests.post")
    def test_retry_on_429_rate_limit(self, mock_post, mock_sleep):
        resp_429 = mock.MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "0.1"}
        resp_429.text = "Rate limited"

        resp_200 = mock.MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {
            "choices": [{"message": {"content": "Success after retry"}}]
        }

        mock_post.side_effect = [resp_429, resp_200]

        result = call_openrouter_text(
            api_key="sk-test-key",
            model=DEFAULT_WRITER_MODEL,
            prompt_text="Generate report",
            max_retries=2,
        )
        assert result == "Success after retry"
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once()

    @mock.patch("requests.get")
    def test_validate_api_key_success(self, mock_get):
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {"label": "Research Key", "limit": 20.0, "usage": 1.25}
        }
        mock_get.return_value = mock_resp

        is_valid, msg = validate_api_key("sk-valid-key")
        assert is_valid is True
        assert "Connected (Research Key)" in msg
        assert "$1.25/$20.00" in msg

    @mock.patch("requests.get")
    def test_validate_api_key_unauthorized(self, mock_get):
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 401
        mock_get.return_value = mock_resp

        is_valid, msg = validate_api_key("sk-bad-key")
        assert is_valid is False
        assert "Unauthorized" in msg
