"""Message trimming utilities for managing conversation context.

This module provides utilities to trim message history to fit within
context limits, supporting both message count and token-based strategies.

Example:
    ```python
    from ai_infra.memory import trim_messages

    # Keep last 10 messages
    trimmed = trim_messages(messages, strategy="last", max_messages=10)

    # Keep under 4000 tokens
    trimmed = trim_messages(messages, strategy="token", max_tokens=4000)

    # Keep under 4000 tokens, always preserve system message
    trimmed = trim_messages(
        messages,
        strategy="token",
        max_tokens=4000,
        preserve_system=True,
    )

    # Keep first 5 messages (rare use case)
    trimmed = trim_messages(messages, strategy="first", max_messages=5)
    ```
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ai_infra.llm.memory.tokens import count_tokens_approximate

# Type alias for strategy
TrimStrategy = Literal["last", "first", "token"]


def trim_messages(
    messages: Sequence[BaseMessage | dict],
    *,
    strategy: TrimStrategy = "last",
    max_messages: int | None = None,
    max_tokens: int | None = None,
    preserve_system: bool = True,
    token_counter: Any | None = None,
) -> list[BaseMessage]:
    """Trim messages to fit within limits.

    Supports multiple strategies:
    - "last": Keep the last N messages (most recent)
    - "first": Keep the first N messages (least recent)
    - "token": Keep messages that fit within token limit

    Args:
        messages: List of messages to trim (BaseMessage or dict format)
        strategy: Trimming strategy ("last", "first", "token")
        max_messages: Maximum number of messages to keep (for "last"/"first")
        max_tokens: Maximum tokens to keep (for "token" strategy)
        preserve_system: If True, always keep the first system message
        token_counter: Custom token counter function. If None, uses approximate count.
                      Should accept a list of messages and return int.

    Returns:
        List of trimmed messages (as BaseMessage objects)

    Raises:
        ValueError: If strategy is invalid or required params are missing

    Examples:
        >>> from langchain_core.messages import HumanMessage, AIMessage
        >>> msgs = [HumanMessage(content="Hi"), AIMessage(content="Hello")]
        >>> trim_messages(msgs, strategy="last", max_messages=1)
        [AIMessage(content='Hello')]
    """
    if not messages:
        return []

    # Convert to BaseMessage if needed
    normalized = _normalize_messages(messages)

    # Validate parameters
    if strategy in ("last", "first") and max_messages is None:
        raise ValueError(f"max_messages required for strategy='{strategy}'")
    if strategy == "token" and max_tokens is None:
        raise ValueError("max_tokens required for strategy='token'")

    # Extract system message if preserving
    system_msg: BaseMessage | None = None
    work_messages = list(normalized)

    if preserve_system and work_messages:
        if isinstance(work_messages[0], SystemMessage):
            system_msg = work_messages[0]
            work_messages = work_messages[1:]

    # Apply strategy
    if strategy == "last":
        result = _trim_last(work_messages, max_messages)  # type: ignore
    elif strategy == "first":
        result = _trim_first(work_messages, max_messages)  # type: ignore
    elif strategy == "token":
        counter = token_counter or count_tokens_approximate
        # Account for system message tokens
        # max_tokens is guaranteed to be int here due to validation above
        assert max_tokens is not None
        available_tokens = max_tokens
        if system_msg and token_counter:
            system_tokens = counter([system_msg])
            available_tokens = max(0, available_tokens - system_tokens)
        elif system_msg:
            # Approximate: estimate system message tokens
            system_tokens = count_tokens_approximate([system_msg])
            available_tokens = max(0, available_tokens - system_tokens)
        result = _trim_by_tokens(work_messages, available_tokens, counter)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'last', 'first', or 'token'")

    # Prepend system message if preserved
    if system_msg:
        return [system_msg] + result

    return result


def _normalize_messages(
    messages: Sequence[BaseMessage | dict],
) -> list[BaseMessage]:
    """Convert dict messages to BaseMessage objects."""
    result = []
    for msg in messages:
        if isinstance(msg, BaseMessage):
            result.append(msg)
        elif isinstance(msg, dict):
            result.append(_dict_to_message(msg))
        else:
            raise TypeError(f"Expected BaseMessage or dict, got {type(msg)}")
    return result


def _dict_to_message(msg: dict) -> BaseMessage:
    """Convert a dict to the appropriate BaseMessage type."""
    role = msg.get("role", msg.get("type", "human"))
    content = msg.get("content", "")

    if role in ("system", "SystemMessage"):
        return SystemMessage(content=content)
    elif role in ("human", "user", "HumanMessage"):
        return HumanMessage(content=content)
    elif role in ("ai", "assistant", "AIMessage"):
        return AIMessage(content=content)
    elif role in ("tool", "ToolMessage"):
        return ToolMessage(
            content=content,
            tool_call_id=msg.get("tool_call_id", ""),
        )
    else:
        # Default to human message
        return HumanMessage(content=content)


def _trim_last(messages: list[BaseMessage], max_messages: int) -> list[BaseMessage]:
    """Keep the last N messages."""
    if max_messages <= 0:
        return []
    return messages[-max_messages:]


def _trim_first(messages: list[BaseMessage], max_messages: int) -> list[BaseMessage]:
    """Keep the first N messages."""
    if max_messages <= 0:
        return []
    return messages[:max_messages]


def _trim_by_tokens(
    messages: list[BaseMessage],
    max_tokens: int,
    counter: Any,
) -> list[BaseMessage]:
    """Keep messages from the end that fit within token limit.

    Works backwards from the most recent message, adding messages
    until the token limit would be exceeded.
    """
    if max_tokens <= 0:
        return []

    result: list[BaseMessage] = []
    total_tokens = 0

    # Work backwards from most recent
    for msg in reversed(messages):
        msg_tokens = counter([msg])
        if total_tokens + msg_tokens <= max_tokens:
            result.insert(0, msg)
            total_tokens += msg_tokens
        else:
            # Would exceed limit, stop
            break

    return result
