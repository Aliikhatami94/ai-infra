"""Token counting utilities for message management.

This module provides provider-agnostic token counting for messages.
Supports exact counting (when tiktoken is available) and fast approximate counting.

Example:
    ```python
    from ai_infra.memory import count_tokens, count_tokens_approximate

    # Exact count (requires tiktoken for OpenAI models)
    tokens = count_tokens(messages, model="gpt-4o")

    # Fast approximate (works anywhere, no deps)
    tokens = count_tokens_approximate(messages)
    ```
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

# Cache for tiktoken encodings
_encoding_cache: dict = {}


def count_tokens(
    messages: Sequence[BaseMessage | dict | str],
    *,
    model: str | None = None,
    provider: str | None = None,
) -> int:
    """Count tokens in messages using the appropriate tokenizer.

    Uses tiktoken for OpenAI models, falls back to approximate counting
    for other providers.

    Args:
        messages: Messages to count (BaseMessage, dict, or raw string)
        model: Model name for tokenizer selection (e.g., "gpt-4o")
        provider: Provider name (e.g., "openai", "anthropic")

    Returns:
        Token count (exact for OpenAI, approximate otherwise)

    Example:
        >>> from langchain_core.messages import HumanMessage
        >>> count_tokens([HumanMessage(content="Hello, world!")], model="gpt-4o")
        4
    """
    # Try exact counting for OpenAI
    if model or (provider and provider.lower() == "openai"):
        try:
            return _count_tokens_tiktoken(messages, model or "gpt-4o")
        except ImportError:
            logger.debug("tiktoken not available, using approximate count")
        except Exception as e:
            logger.debug(f"tiktoken failed: {e}, using approximate count")

    # Fall back to approximate
    return count_tokens_approximate(messages)


def count_tokens_approximate(
    messages: Sequence[BaseMessage | dict | str],
) -> int:
    """Fast approximate token count without external dependencies.

    Uses a simple heuristic: ~4 characters per token (English average).
    This is intentionally conservative (slightly overestimates) to avoid
    exceeding context limits.

    The heuristic accounts for:
    - Message role overhead (~4 tokens per message)
    - Content tokenization (~1 token per 4 chars)

    Args:
        messages: Messages to count

    Returns:
        Approximate token count

    Example:
        >>> count_tokens_approximate(["Hello, world!"])
        7  # ~4 for overhead + ~3 for 13 chars
    """
    total = 0

    for msg in messages:
        # Get content
        if isinstance(msg, str):
            content = msg
            overhead = 0  # No message overhead for raw strings
        elif isinstance(msg, BaseMessage):
            content = _get_message_content(msg)
            overhead = 4  # Role + formatting overhead
        elif isinstance(msg, dict):
            content = str(msg.get("content", ""))
            overhead = 4
        else:
            content = str(msg)
            overhead = 0

        # Approximate: 1 token per 4 characters (conservative)
        content_tokens = (len(content) + 3) // 4  # Round up

        total += content_tokens + overhead

    return total


def _get_message_content(msg: BaseMessage) -> str:
    """Extract string content from a message."""
    content = msg.content

    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Multi-part content (e.g., vision messages)
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if "text" in part:
                    parts.append(part["text"])
                # Image parts don't contribute much to text token count
        return " ".join(parts)
    else:
        return str(content)


def _count_tokens_tiktoken(
    messages: Sequence[BaseMessage | dict | str],
    model: str,
) -> int:
    """Count tokens using tiktoken (OpenAI's tokenizer)."""
    import tiktoken

    # Get or create encoding
    encoding_name = _get_encoding_for_model(model)
    if encoding_name not in _encoding_cache:
        _encoding_cache[encoding_name] = tiktoken.get_encoding(encoding_name)

    encoding = _encoding_cache[encoding_name]
    total = 0

    for msg in messages:
        if isinstance(msg, str):
            total += len(encoding.encode(msg))
        elif isinstance(msg, BaseMessage):
            content = _get_message_content(msg)
            # Add overhead for message structure
            total += len(encoding.encode(content)) + 4  # role + name + etc
        elif isinstance(msg, dict):
            content = str(msg.get("content", ""))
            total += len(encoding.encode(content)) + 4

    return total


def _get_encoding_for_model(model: str) -> str:
    """Get the tiktoken encoding name for a model."""
    model_lower = model.lower()

    # GPT-4o, GPT-4-turbo, GPT-4 all use cl100k_base
    if "gpt-4" in model_lower or "gpt-3.5" in model_lower:
        return "cl100k_base"

    # O1 models
    if model_lower.startswith("o1"):
        return "cl100k_base"

    # Default to cl100k_base (most modern models)
    return "cl100k_base"


def get_context_limit(model: str, provider: str | None = None) -> int:
    """Get the context window size for a model.

    Args:
        model: Model name
        provider: Provider name (optional, helps with ambiguous names)

    Returns:
        Context window size in tokens

    Note:
        Returns conservative estimates. Actual limits may vary.
    """
    model_lower = model.lower()

    # OpenAI models
    if "gpt-4o" in model_lower:
        return 128_000
    if "gpt-4-turbo" in model_lower:
        return 128_000
    if "gpt-4-32k" in model_lower:
        return 32_768
    if "gpt-4" in model_lower:
        return 8_192
    if "gpt-3.5-turbo-16k" in model_lower:
        return 16_385
    if "gpt-3.5" in model_lower:
        return 4_096
    if model_lower.startswith("o1"):
        return 128_000

    # Anthropic models
    if "claude-3" in model_lower or "claude-sonnet" in model_lower:
        return 200_000
    if "claude-2" in model_lower:
        return 100_000
    if "claude" in model_lower:
        return 100_000

    # Google models
    if "gemini-1.5" in model_lower or "gemini-2" in model_lower:
        return 1_000_000  # 1M context
    if "gemini" in model_lower:
        return 32_768

    # xAI models
    if "grok" in model_lower:
        return 131_072

    # Default conservative limit
    return 4_096
