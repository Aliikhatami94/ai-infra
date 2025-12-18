"""LLM error handling utilities.

This module provides error translation and handling for LLM operations:
- Translates provider SDK errors into ai-infra error classes
- Provides helpful error messages with actionable hints
- Supports all major providers (OpenAI, Anthropic, Google, xAI)

Usage:
    from ai_infra.llm.utils.error_handler import translate_provider_error, with_error_handling

    # Use decorator
    @with_error_handling(provider="openai", model="gpt-4o")
    async def my_llm_call():
        return await model.ainvoke(messages)

    # Or translate manually
    try:
        result = model.invoke(messages)
    except Exception as e:
        raise translate_provider_error(e, provider="openai", model="gpt-4o")
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable
from typing import Any, TypeVar

from ai_infra.errors import (
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    LLMError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)

__all__ = [
    "extract_retry_after",
    "translate_provider_error",
    "with_error_handling",
    "with_error_handling_async",
]

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Error Message Patterns
# =============================================================================

# OpenAI error patterns
OPENAI_PATTERNS = {
    "rate_limit": [
        r"rate limit",
        r"too many requests",
        r"rate_limit_exceeded",
    ],
    "auth": [
        r"invalid api key",
        r"incorrect api key",
        r"authentication",
        r"unauthorized",
        r"invalid_api_key",
    ],
    "model_not_found": [
        r"model.*not found",
        r"model.*does not exist",
        r"invalid model",
        r"model_not_found",
    ],
    "context_length": [
        r"context length",
        r"maximum context",
        r"token limit",
        r"context_length_exceeded",
        r"max tokens",
    ],
    "content_filter": [
        r"content filter",
        r"content management policy",
        r"flagged",
        r"content_policy_violation",
    ],
}

# Anthropic error patterns
ANTHROPIC_PATTERNS = {
    "rate_limit": [
        r"rate limit",
        r"too many requests",
        r"overloaded",
    ],
    "auth": [
        r"invalid.*api.*key",
        r"authentication",
        r"unauthorized",
        r"invalid x-api-key",
    ],
    "model_not_found": [
        r"model.*not found",
        r"unknown model",
    ],
    "context_length": [
        r"context length",
        r"maximum.*tokens",
        r"prompt is too long",
    ],
    "content_filter": [
        r"content policy",
        r"harmful content",
        r"unsafe content",
        r"blocked.*safety",
        r"safety filter",
    ],
}

# Google error patterns
GOOGLE_PATTERNS = {
    "rate_limit": [
        r"quota exceeded",
        r"rate limit",
        r"resource exhausted",
    ],
    "auth": [
        r"invalid api key",
        r"api key not valid",
        r"authentication",
        r"unauthenticated",
    ],
    "model_not_found": [
        r"model.*not found",
        r"unknown model",
        r"invalid model",
    ],
    "context_length": [
        r"token limit",
        r"input too long",
    ],
    "content_filter": [
        r"safety filter",
        r"blocked.*safety",
        r"harm category",
    ],
}

PROVIDER_PATTERNS = {
    "openai": OPENAI_PATTERNS,
    "anthropic": ANTHROPIC_PATTERNS,
    "google_genai": GOOGLE_PATTERNS,
    "xai": OPENAI_PATTERNS,  # xAI uses OpenAI-compatible API
}


# =============================================================================
# Error Translation
# =============================================================================


def extract_retry_after(error: Exception) -> float | None:
    """Extract retry-after seconds from rate limit error.

    Args:
        error: The original exception

    Returns:
        Retry-after seconds, or None if not available
    """
    # Check for retry_after attribute
    if hasattr(error, "retry_after"):
        return float(error.retry_after)

    # Check for headers attribute (httpx/requests style)
    headers = getattr(error, "headers", None) or getattr(error, "response", None)
    if headers:
        if hasattr(headers, "headers"):
            headers = headers.headers
        if hasattr(headers, "get"):
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass

    # Try to extract from error message
    msg = str(error).lower()
    match = re.search(r"retry after (\d+(?:\.\d+)?)\s*(?:second|s)?", msg)
    if match:
        return float(match.group(1))

    return None


def _match_pattern(msg: str, patterns: list[str]) -> bool:
    """Check if message matches any pattern."""
    msg_lower = msg.lower()
    return any(re.search(pattern, msg_lower) for pattern in patterns)


def _get_status_code(error: Exception) -> int | None:
    """Extract HTTP status code from error."""
    # Direct attribute
    if hasattr(error, "status_code"):
        val = error.status_code
        if isinstance(val, int):
            return val
        try:
            return int(val)
        except Exception:
            return None

    # Response attribute
    if hasattr(error, "response"):
        resp = error.response
        if hasattr(resp, "status_code"):
            val = resp.status_code
            if isinstance(val, int):
                return val
            try:
                return int(val)
            except Exception:
                return None
        if hasattr(resp, "status"):
            val = resp.status
            if isinstance(val, int):
                return val
            try:
                return int(val)
            except Exception:
                return None

    # httpx style
    if hasattr(error, "code"):
        val = error.code
        if isinstance(val, int):
            return val
        try:
            return int(val)
        except Exception:
            return None

    return None


def translate_provider_error(
    error: Exception,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> Exception:
    """Translate a provider SDK error into an ai-infra error.

    Args:
        error: The original exception
        provider: Provider name (e.g., "openai", "anthropic")
        model: Model name (e.g., "gpt-4o")

    Returns:
        Translated ai-infra error, or original error if not recognized

    Example:
        try:
            result = openai_client.chat.completions.create(...)
        except Exception as e:
            raise translate_provider_error(e, provider="openai", model="gpt-4o")
    """
    # Already an ai-infra error
    if isinstance(error, (LLMError, ProviderError)):
        return error

    msg = str(error)
    status_code = _get_status_code(error)
    error_type = type(error).__name__

    # Get provider-specific patterns
    patterns = PROVIDER_PATTERNS.get(provider or "", OPENAI_PATTERNS)

    # Try to identify error type from status code first
    if status_code == 401:
        return AuthenticationError(
            msg,
            provider=provider,
        )

    if status_code == 429:
        retry_after = extract_retry_after(error)
        return RateLimitError(
            msg,
            provider=provider,
            model=model,
            retry_after=retry_after,
        )

    if status_code == 404:
        return ModelNotFoundError(
            model=model or "unknown",
            provider=provider,
        )

    # Try to identify from message patterns
    if _match_pattern(msg, patterns.get("auth", [])):
        return AuthenticationError(
            msg,
            provider=provider,
        )

    if _match_pattern(msg, patterns.get("rate_limit", [])):
        retry_after = extract_retry_after(error)
        return RateLimitError(
            msg,
            provider=provider,
            model=model,
            retry_after=retry_after,
        )

    if _match_pattern(msg, patterns.get("model_not_found", [])):
        return ModelNotFoundError(
            model=model or "unknown",
            provider=provider,
        )

    if _match_pattern(msg, patterns.get("context_length", [])):
        # Try to extract token counts from message
        max_tokens = None
        requested = None
        msg_lower = msg.lower()

        # Match "maximum context is 8192" or "max 8192 tokens"
        max_match = re.search(r"(?:maximum|max).*?(\d+)", msg_lower)
        if max_match:
            max_tokens = int(max_match.group(1))

        # Match "requested 10000" specifically
        req_match = re.search(r"requested\s+(\d+)", msg_lower)
        if req_match:
            requested = int(req_match.group(1))
        else:
            # Fallback: find all numbers and take the larger one as requested
            # if we found a max already
            all_numbers = re.findall(r"\d+", msg)
            if max_tokens and len(all_numbers) >= 2:
                numbers = [int(n) for n in all_numbers]
                # The requested is typically larger than max
                larger = max(numbers)
                if larger > max_tokens:
                    requested = larger

        return ContextLengthError(
            msg,
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            requested_tokens=requested,
        )

    if _match_pattern(msg, patterns.get("content_filter", [])):
        return ContentFilterError(
            msg,
            provider=provider,
            model=model,
        )

    # Generic provider error for other cases
    if status_code and status_code >= 400:
        return ProviderError(
            msg,
            provider=provider,
            model=model,
            status_code=status_code,
            error_type=error_type,
        )

    # Return original error if we can't classify it
    return error


# =============================================================================
# Decorators
# =============================================================================


def with_error_handling(
    provider: str | None = None,
    model: str | None = None,
) -> Callable[[F], F]:
    """Decorator to wrap a function with LLM error handling.

    Args:
        provider: Provider name
        model: Model name

    Returns:
        Decorated function with error translation

    Example:
        @with_error_handling(provider="openai", model="gpt-4o")
        def my_llm_call():
            return model.invoke(messages)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                raise translate_provider_error(
                    e,
                    provider=provider,
                    model=model,
                ) from e

        return wrapper  # type: ignore

    return decorator


def with_error_handling_async(
    provider: str | None = None,
    model: str | None = None,
) -> Callable[[F], F]:
    """Async decorator to wrap a function with LLM error handling.

    Args:
        provider: Provider name
        model: Model name

    Returns:
        Decorated async function with error translation

    Example:
        @with_error_handling_async(provider="openai", model="gpt-4o")
        async def my_llm_call():
            return await model.ainvoke(messages)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                raise translate_provider_error(
                    e,
                    provider=provider,
                    model=model,
                ) from e

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# Provider-Specific Kwargs Documentation
# =============================================================================

COMMON_KWARGS = {
    "temperature": "Controls randomness (0.0-2.0 for most providers, 0.0-1.0 for Anthropic)",
    "max_tokens": "Maximum tokens to generate",
    "top_p": "Nucleus sampling (0.0-1.0)",
    "stop": "Stop sequences (list of strings)",
    "timeout": "Request timeout in seconds",
    "max_retries": "Number of retries on transient errors",
}

PROVIDER_KWARGS = {
    "openai": {
        "frequency_penalty": "Penalty for token frequency (-2.0 to 2.0)",
        "presence_penalty": "Penalty for new topics (-2.0 to 2.0)",
        "logprobs": "Include log probabilities (bool)",
        "top_logprobs": "Number of logprobs per token (0-20)",
        "seed": "Seed for reproducible outputs",
        "response_format": "Output format (e.g., {'type': 'json_object'})",
        "organization": "OpenAI organization ID",
        "base_url": "Custom API base URL",
    },
    "anthropic": {
        "top_k": "Top-k sampling (1-infinity)",
        "metadata": "Request metadata dict",
    },
    "google_genai": {
        "top_k": "Top-k sampling",
        "candidate_count": "Number of candidates to generate",
        "safety_settings": "Content safety settings",
    },
    "xai": {
        # xAI uses OpenAI-compatible kwargs
        "frequency_penalty": "Penalty for token frequency (-2.0 to 2.0)",
        "presence_penalty": "Penalty for new topics (-2.0 to 2.0)",
    },
}


def validate_kwargs(
    provider: str,
    kwargs: dict[str, Any],
    *,
    warn: bool = True,
) -> list[str]:
    """Validate kwargs for a provider and return warnings.

    Args:
        provider: Provider name
        kwargs: Kwargs to validate
        warn: Whether to emit warnings (default True)

    Returns:
        List of warning messages

    Example:
        warnings = validate_kwargs("anthropic", {"frequency_penalty": 0.5})
        # Returns: ["'frequency_penalty' is not supported by anthropic. Use 'top_k' instead."]
    """
    import logging

    logger = logging.getLogger(__name__)

    warnings: list[str] = []
    provider_specific = PROVIDER_KWARGS.get(provider, {})

    for key in kwargs:
        # Skip common kwargs
        if key in COMMON_KWARGS:
            continue

        # Check if it's a known provider-specific kwarg
        if key not in provider_specific:
            # Check if it's a kwarg for a different provider
            for other_provider, other_kwargs in PROVIDER_KWARGS.items():
                if other_provider != provider and key in other_kwargs:
                    msg = (
                        f"'{key}' is not supported by {provider}. "
                        f"This is a {other_provider}-specific parameter."
                    )
                    warnings.append(msg)
                    if warn:
                        logger.warning(msg)
                    break

    return warnings


def get_supported_kwargs(provider: str) -> dict[str, str]:
    """Get all supported kwargs for a provider.

    Args:
        provider: Provider name

    Returns:
        Dict mapping kwarg name to description

    Example:
        kwargs = get_supported_kwargs("openai")
        print(kwargs["temperature"])  # "Controls randomness..."
    """
    result = dict(COMMON_KWARGS)
    result.update(PROVIDER_KWARGS.get(provider, {}))
    return result
