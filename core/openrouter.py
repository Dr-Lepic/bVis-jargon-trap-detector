"""OpenRouter API client for multimodal Vision-Language Models and text generation.

Features:
- Robust exponential backoff retry on HTTP 429 (rate limits) and transient 5xx errors.
- Support for multimodal image payload (data URL base64).
- Clear exception hierarchy for auth errors, rate limits, and network failures.
- API key validation utility for the application UI.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple
import requests

logger = logging.getLogger(__name__)

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_URL = f"{OPENROUTER_API_BASE}/chat/completions"
OPENROUTER_AUTH_URL = f"{OPENROUTER_API_BASE}/auth/key"

DEFAULT_JUDGE_MODEL = "inclusionai/ling-3.0-flash-vl:free"
DEFAULT_WRITER_MODEL = "inclusionai/ling-3.0-flash-sante:free"

POPULAR_JUDGE_MODELS: List[str] = [
    "inclusionai/ling-3.0-flash-vl:free",
    "dots-studio/dots-3-note-preview:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "google/gemma-4-31b-it:free",
    "minimax/minimax-m3",
    "google/gemini-2.0-flash-001",
]

POPULAR_WRITER_MODELS: List[str] = [
    "inclusionai/ling-3.0-flash-sante:free",
    "liquid/lfm-2.5-2.6b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "cohere/north-mini-code:free",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.0-flash-001",
]


class OpenRouterError(Exception):
    """Base exception for OpenRouter API interactions."""
    pass


class OpenRouterAuthError(OpenRouterError):
    """Raised when authentication fails (HTTP 401 / 403)."""
    pass


class OpenRouterRateLimitError(OpenRouterError):
    """Raised when rate limits are exhausted after retries (HTTP 429)."""
    pass


def _extract_error_message(resp: requests.Response) -> str:
    """Extracts a human-readable error message from an OpenRouter response."""
    try:
        err_json = resp.json()
        if isinstance(err_json, dict) and "error" in err_json:
            err_obj = err_json["error"]
            if isinstance(err_obj, dict) and "message" in err_obj:
                return str(err_obj["message"])
            return str(err_obj)
        return resp.text
    except Exception:
        return resp.text or f"HTTP {resp.status_code}"


def _get_headers(api_key: str) -> Dict[str, str]:
    """Generates standard OpenRouter request headers."""
    clean_key = (api_key or "").strip()
    return {
        "Authorization": f"Bearer {clean_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/bVis",
        "X-Title": "bVis",
    }


def _execute_with_retry(
    payload: Dict[str, Any],
    api_key: str,
    max_retries: int = 3,
    initial_delay: float = 2.0,
    timeout: int = 60,
) -> str:
    """Executes a request to OpenRouter chat completions with exponential backoff."""
    if not api_key or not api_key.strip():
        raise OpenRouterAuthError("OpenRouter API key is missing or empty. Please enter your API key.")

    headers = _get_headers(api_key)
    delay = initial_delay
    model_name = payload.get("model", "unknown")

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                OPENROUTER_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            # Handle 4xx client configuration errors (never retry, report message immediately)
            if 400 <= response.status_code < 500 and response.status_code != 429:
                err_msg = _extract_error_message(response)
                if response.status_code in (401, 403):
                    raise OpenRouterAuthError(
                        f"Authentication failed ({response.status_code}): {err_msg}"
                    )
                if response.status_code == 404:
                    raise OpenRouterError(
                        f"Model '{model_name}' not found or unavailable on OpenRouter (HTTP 404): {err_msg}"
                    )
                if response.status_code == 402:
                    raise OpenRouterError(
                        f"Insufficient credits for '{model_name}' (HTTP 402): {err_msg}"
                    )
                raise OpenRouterError(f"OpenRouter error ({response.status_code}): {err_msg}")

            # Handle rate limiting (429)
            if response.status_code == 429:
                if attempt == max_retries:
                    err_msg = _extract_error_message(response)
                    raise OpenRouterRateLimitError(
                        f"OpenRouter rate limit reached after {max_retries} retries: {err_msg}"
                    )

                retry_after = response.headers.get("Retry-After")
                wait_time = float(retry_after) if retry_after else delay + random.uniform(0.2, 1.0)
                logger.warning(
                    "OpenRouter rate limit hit (429). Retrying in %.1fs (attempt %d/%d)...",
                    wait_time,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(wait_time)
                delay *= 2
                continue

            # Handle transient server errors (500, 502, 503, 504)
            if response.status_code in (500, 502, 503, 504):
                if attempt == max_retries:
                    err_msg = _extract_error_message(response)
                    raise OpenRouterError(f"OpenRouter server error ({response.status_code}): {err_msg}")
                wait_time = delay + random.uniform(0.1, 0.5)
                logger.warning(
                    "OpenRouter server error (%d). Retrying in %.1fs (attempt %d/%d)...",
                    response.status_code,
                    wait_time,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(wait_time)
                delay *= 2
                continue

            # Check for any other unexpected error
            if response.status_code != 200:
                err_msg = _extract_error_message(response)
                raise OpenRouterError(f"OpenRouter request failed ({response.status_code}): {err_msg}")

            data = response.json()

            # Validate response shape
            if "choices" not in data or not data["choices"]:
                raise OpenRouterError(f"Unexpected response format from OpenRouter: {data}")

            message = data["choices"][0].get("message", {})
            content = message.get("content", "")
            return content.strip()

        except (OpenRouterError, OpenRouterAuthError, OpenRouterRateLimitError):
            # Do not intercept our domain errors
            raise
        except requests.exceptions.RequestException as e:
            if attempt == max_retries:
                raise OpenRouterError(f"Network error communicating with OpenRouter: {e}") from e
            wait_time = delay + random.uniform(0.2, 0.8)
            logger.warning(
                "Network exception during OpenRouter call: %s. Retrying in %.1fs...", e, wait_time
            )
            time.sleep(wait_time)
            delay *= 2

    raise OpenRouterError(f"Failed to obtain response after {max_retries} attempts.")


def call_openrouter_multimodal(
    api_key: str,
    model: str,
    prompt_text: str,
    base64_image: str,
    temperature: float = 0.2,
    max_retries: int = 3,
    timeout: int = 60,
) -> str:
    """Invokes a multimodal Vision-Language Model via OpenRouter.

    Args:
        api_key: OpenRouter API key.
        model: Target model identifier (e.g., 'minimax/minimax-m3:free').
        prompt_text: Evaluation instructions and candidate reports.
        base64_image: Base64-encoded JPEG image string.
        temperature: Sampling temperature (default: 0.2 for deterministic judging).
        max_retries: Number of retry attempts on rate limits/server errors.
        timeout: HTTP request timeout in seconds.

    Returns:
        Generated model output string.
    """
    clean_b64 = base64_image.strip()
    if clean_b64.startswith("data:image"):
        image_url = clean_b64
    else:
        image_url = f"data:image/jpeg;base64,{clean_b64}"

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
    }

    return _execute_with_retry(
        payload=payload,
        api_key=api_key,
        max_retries=max_retries,
        timeout=timeout,
    )


def call_openrouter_text(
    api_key: str,
    model: str,
    prompt_text: str,
    temperature: float = 0.7,
    max_retries: int = 3,
    timeout: int = 60,
) -> str:
    """Invokes a text-only LLM (such as the Jargon Writer) via OpenRouter.

    Args:
        api_key: OpenRouter API key.
        model: Target model identifier (e.g., 'meta-llama/llama-3.3-70b-instruct').
        prompt_text: Prompt requesting the adversarial medical narrative.
        temperature: Sampling temperature (default: 0.7 for creative jargon drafting).
        max_retries: Number of retry attempts on rate limits/server errors.
        timeout: HTTP request timeout in seconds.

    Returns:
        Generated jargon text string.
    """
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
    }

    return _execute_with_retry(
        payload=payload,
        api_key=api_key,
        max_retries=max_retries,
        timeout=timeout,
    )


def validate_api_key(api_key: str) -> Tuple[bool, str]:
    """Validates an OpenRouter API key against the OpenRouter auth endpoint.

    Args:
        api_key: The user's API key to check.

    Returns:
        (is_valid: bool, status_message: str)
    """
    if not api_key or not api_key.strip():
        return False, "API key is required."

    headers = _get_headers(api_key)
    try:
        resp = requests.get(OPENROUTER_AUTH_URL, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            label = data.get("label", "Key active")
            limit = data.get("limit")
            usage = data.get("usage")
            info = f"Connected ({label})"
            if limit is not None and usage is not None:
                info += f" | Usage: ${usage:.2f}/${limit:.2f}"
            return True, info
        if resp.status_code in (401, 403):
            return False, "Invalid API key (Unauthorized)."
        return False, f"HTTP status {resp.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection failed: {str(e)}"


# Alias for semantic clarity
check_api_connection = validate_api_key
