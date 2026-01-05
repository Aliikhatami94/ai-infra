"""Cache key generation utilities.

This module provides utilities for generating cache keys from prompts,
messages, and other inputs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CacheKeyGenerator:
    """Generates cache keys from various input types.

    Cache keys should be deterministic and capture the semantic meaning
    of the input. The generator normalizes inputs to ensure consistent
    key generation.

    Example:
        >>> gen = CacheKeyGenerator()
        >>> key = gen.generate("What is the capital of France?")
        >>> key = gen.generate_from_messages([
        ...     {"role": "user", "content": "Hello"},
        ... ])
    """

    def __init__(
        self,
        normalize_whitespace: bool = True,
        lowercase: bool = False,
        strip_punctuation: bool = False,
    ) -> None:
        """Initialize the key generator.

        Args:
            normalize_whitespace: Collapse multiple spaces to single space.
            lowercase: Convert text to lowercase for case-insensitive caching.
            strip_punctuation: Remove punctuation marks.
        """
        self.normalize_whitespace = normalize_whitespace
        self.lowercase = lowercase
        self.strip_punctuation = strip_punctuation

    def generate(self, text: str) -> str:
        """Generate a cache key from text.

        Args:
            text: The input text.

        Returns:
            Normalized text suitable for embedding and caching.

        Example:
            >>> gen = CacheKeyGenerator()
            >>> gen.generate("  What is   the capital?  ")
            'What is the capital?'
        """
        key = text

        if self.normalize_whitespace:
            key = " ".join(key.split())

        if self.lowercase:
            key = key.lower()

        if self.strip_punctuation:
            import string

            key = key.translate(str.maketrans("", "", string.punctuation))

        return key.strip()

    def generate_from_messages(
        self,
        messages: list[dict[str, str]],
        include_system: bool = False,
    ) -> str:
        """Generate a cache key from a message list.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            include_system: Whether to include system messages in the key.

        Returns:
            Normalized text combining relevant messages.

        Example:
            >>> gen = CacheKeyGenerator()
            >>> gen.generate_from_messages([
            ...     {"role": "system", "content": "Be helpful"},
            ...     {"role": "user", "content": "Hello"},
            ... ])
            'Hello'
        """
        parts = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system" and not include_system:
                continue

            if content:
                parts.append(self.generate(content))

        return " | ".join(parts)

    def generate_hash(self, text: str) -> str:
        """Generate a hash from text for storage keys.

        Args:
            text: The input text.

        Returns:
            SHA-256 hash of the normalized text.

        Example:
            >>> gen = CacheKeyGenerator()
            >>> gen.generate_hash("Hello")
            '185f8db32271fe25f561a6fc938b2e26...'
        """
        normalized = self.generate(text)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def generate_with_context(
        self,
        text: str,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a cache key including context parameters.

        This creates a key that includes model and generation parameters,
        useful when the same prompt with different parameters should
        produce different cache entries.

        Args:
            text: The input text.
            model: Model name to include in key.
            temperature: Temperature to include in key.
            **kwargs: Additional parameters to include.

        Returns:
            JSON string with normalized text and parameters.

        Example:
            >>> gen = CacheKeyGenerator()
            >>> gen.generate_with_context("Hello", model="gpt-4", temperature=0.7)
            '{"text": "Hello", "model": "gpt-4", "temperature": 0.7}'
        """
        key_data: dict[str, Any] = {
            "text": self.generate(text),
        }

        if model is not None:
            key_data["model"] = model

        if temperature is not None:
            key_data["temperature"] = temperature

        # Add any additional kwargs
        for k, v in sorted(kwargs.items()):
            if v is not None:
                key_data[k] = v

        return json.dumps(key_data, sort_keys=True)

    @staticmethod
    def extract_last_user_message(messages: list[dict[str, str]]) -> str | None:
        """Extract the last user message from a message list.

        Args:
            messages: List of message dicts.

        Returns:
            Content of the last user message, or None if not found.

        Example:
            >>> CacheKeyGenerator.extract_last_user_message([
            ...     {"role": "user", "content": "First"},
            ...     {"role": "assistant", "content": "Response"},
            ...     {"role": "user", "content": "Second"},
            ... ])
            'Second'
        """
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content")
        return None
