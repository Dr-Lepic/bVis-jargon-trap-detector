"""Core Backend Engine for bVis.

Provides modular interfaces for:
- Multimodal and text LLM invocation via OpenRouter (with backoff retry).
- High-quality image downsampling (768px Lanczos) and base64 encoding.
- Writer and judge prompt templates.
- Robust regex verdict parsing and trap outcome evaluation.
- Wilson Score 95% Confidence Interval and Length Confound Rate calculations.
"""

from core.image_utils import (
    decode_base64_to_image,
    get_image_dimensions,
    resize_and_encode_image,
)
from core.openrouter import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_WRITER_MODEL,
    POPULAR_JUDGE_MODELS,
    POPULAR_WRITER_MODELS,
    OpenRouterAuthError,
    OpenRouterError,
    OpenRouterRateLimitError,
    call_openrouter_multimodal,
    call_openrouter_text,
    check_api_connection,
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

__all__ = [
    # Image utilities
    "resize_and_encode_image",
    "decode_base64_to_image",
    "get_image_dimensions",
    # Prompts
    "JARGON_WRITER_PROMPT",
    "JUDGE_PROMPT",
    "format_writer_prompt",
    "format_judge_prompt",
    # Parser
    "parse_judge_verdict",
    "evaluate_trap_result",
    "get_picked_word_count",
    # Statistics
    "wilson_ci",
    "length_confound_rate",
    "calculate_batch_statistics",
    # OpenRouter API client
    "call_openrouter_multimodal",
    "call_openrouter_text",
    "validate_api_key",
    "check_api_connection",
    "OpenRouterError",
    "OpenRouterAuthError",
    "OpenRouterRateLimitError",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_WRITER_MODEL",
    "POPULAR_JUDGE_MODELS",
    "POPULAR_WRITER_MODELS",
]
