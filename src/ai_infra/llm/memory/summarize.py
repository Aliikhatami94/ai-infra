"""Message summarization utilities for managing long conversations.

This module provides:
1. summarize_messages() - One-shot summarization of old messages
2. SummarizationMiddleware - Auto-summarize during agent execution

Example:
    ```python
    from ai_infra import LLM
    from ai_infra.memory import summarize_messages, SummarizationMiddleware

    # One-shot summarization
    result = summarize_messages(
        messages,
        keep_last=5,
        llm=LLM(),
    )

    # Auto-summarization middleware for agents
    from ai_infra import Agent
    agent = Agent(
        tools=[...],
        middleware=[
            SummarizationMiddleware(
                trigger_tokens=4000,
                keep_messages=10,
            )
        ]
    )
    ```
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from ai_infra.llm.memory.tokens import count_tokens_approximate

logger = logging.getLogger(__name__)

# Default summarization prompt
DEFAULT_SUMMARIZE_PROMPT = """Summarize the following conversation history concisely.
Focus on:
- Key topics discussed
- Important decisions made
- Relevant context for continuing the conversation
- Any user preferences or facts mentioned

Keep the summary brief but comprehensive enough to maintain conversation context.

Conversation to summarize:
{conversation}

Summary:"""


@dataclass
class SummarizeResult:
    """Result from summarizing messages."""

    messages: list[BaseMessage]
    """The resulting messages (summary + kept messages)."""

    summary: str
    """The generated summary text."""

    original_count: int
    """Number of messages before summarization."""

    summarized_count: int
    """Number of messages that were summarized."""

    kept_count: int
    """Number of messages kept as-is."""


def summarize_messages(
    messages: Sequence[BaseMessage | dict],
    *,
    keep_last: int = 5,
    llm: Any | None = None,
    summarize_prompt: str | None = None,
    include_system: bool = True,
) -> SummarizeResult:
    """Summarize old messages while keeping recent ones.

    Takes a conversation history and:
    1. Keeps the last N messages unchanged
    2. Summarizes older messages into a single summary
    3. Returns the summary as a SystemMessage + recent messages

    Args:
        messages: Full conversation history
        keep_last: Number of recent messages to keep unchanged
        llm: LLM instance to use for summarization. If None, creates default LLM.
        summarize_prompt: Custom prompt template for summarization.
                         Use {conversation} placeholder for the messages.
        include_system: If True, preserve original system message if present.

    Returns:
        SummarizeResult with summary and trimmed messages

    Example:
        >>> from ai_infra import LLM
        >>> from ai_infra.memory import summarize_messages
        >>> result = summarize_messages(messages, keep_last=5, llm=LLM())
        >>> print(result.summary)
        "The user discussed Python programming..."
        >>> print(len(result.messages))
        6  # summary message + 5 kept messages
    """
    from ai_infra.llm.memory.trim import _normalize_messages

    # Normalize messages
    normalized = _normalize_messages(messages)

    if not normalized:
        return SummarizeResult(
            messages=[],
            summary="",
            original_count=0,
            summarized_count=0,
            kept_count=0,
        )

    # Extract system message if present
    system_msg: BaseMessage | None = None
    work_messages = list(normalized)

    if include_system and work_messages and isinstance(work_messages[0], SystemMessage):
        system_msg = work_messages[0]
        work_messages = work_messages[1:]

    # If not enough messages to summarize, return as-is
    if len(work_messages) <= keep_last:
        early_messages = list(normalized)
        return SummarizeResult(
            messages=early_messages,
            summary="",
            original_count=len(normalized),
            summarized_count=0,
            kept_count=len(early_messages),
        )

    # Split into messages to summarize and messages to keep
    to_summarize = work_messages[:-keep_last] if keep_last > 0 else work_messages
    to_keep = work_messages[-keep_last:] if keep_last > 0 else []

    # Generate summary
    if llm is None:
        from ai_infra import LLM

        llm = LLM()

    conversation_text = _format_messages_for_summary(to_summarize)
    prompt = (summarize_prompt or DEFAULT_SUMMARIZE_PROMPT).format(conversation=conversation_text)

    response = llm.chat(prompt)
    # Extract content from response (could be AIMessage or string)
    summary = response.content if hasattr(response, "content") else str(response)

    # Build result messages
    result_messages: list[BaseMessage] = []

    # Add original system message if present
    if system_msg:
        result_messages.append(system_msg)

    # Add summary as a system message
    summary_msg = SystemMessage(content=f"[Previous conversation summary]\n{summary}")
    result_messages.append(summary_msg)

    # Add kept messages
    result_messages.extend(to_keep)

    return SummarizeResult(
        messages=result_messages,
        summary=summary,
        original_count=len(normalized),
        summarized_count=len(to_summarize),
        kept_count=len(to_keep),
    )


async def asummarize_messages(
    messages: Sequence[BaseMessage | dict],
    *,
    keep_last: int = 5,
    llm: Any | None = None,
    summarize_prompt: str | None = None,
    include_system: bool = True,
) -> SummarizeResult:
    """Async version of summarize_messages.

    See summarize_messages for full documentation.
    """
    from ai_infra.llm.memory.trim import _normalize_messages

    # Normalize messages
    normalized = _normalize_messages(messages)

    if not normalized:
        return SummarizeResult(
            messages=[],
            summary="",
            original_count=0,
            summarized_count=0,
            kept_count=0,
        )

    # Extract system message if present
    system_msg: BaseMessage | None = None
    work_messages = list(normalized)

    if include_system and work_messages and isinstance(work_messages[0], SystemMessage):
        system_msg = work_messages[0]
        work_messages = work_messages[1:]

    # If not enough messages to summarize, return as-is
    if len(work_messages) <= keep_last:
        early_messages = list(normalized)
        return SummarizeResult(
            messages=early_messages,
            summary="",
            original_count=len(normalized),
            summarized_count=0,
            kept_count=len(early_messages),
        )

    # Split into messages to summarize and messages to keep
    to_summarize = work_messages[:-keep_last] if keep_last > 0 else work_messages
    to_keep = work_messages[-keep_last:] if keep_last > 0 else []

    # Generate summary
    if llm is None:
        from ai_infra import LLM

        llm = LLM()

    conversation_text = _format_messages_for_summary(to_summarize)
    prompt = (summarize_prompt or DEFAULT_SUMMARIZE_PROMPT).format(conversation=conversation_text)

    response = await llm.achat(prompt)
    # Extract content from response (could be AIMessage or string)
    summary = response.content if hasattr(response, "content") else str(response)

    # Build result messages
    result_messages: list[BaseMessage] = []

    if system_msg:
        result_messages.append(system_msg)

    summary_msg = SystemMessage(content=f"[Previous conversation summary]\n{summary}")
    result_messages.append(summary_msg)
    result_messages.extend(to_keep)

    return SummarizeResult(
        messages=result_messages,
        summary=summary,
        original_count=len(normalized),
        summarized_count=len(to_summarize),
        kept_count=len(to_keep),
    )


def _format_messages_for_summary(messages: list[BaseMessage]) -> str:
    """Format messages as text for summarization."""
    lines = []
    for msg in messages:
        role = _get_role_name(msg)
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _get_role_name(msg: BaseMessage) -> str:
    """Get human-readable role name for a message."""
    if isinstance(msg, SystemMessage):
        return "System"
    elif isinstance(msg, HumanMessage):
        return "User"
    elif isinstance(msg, AIMessage):
        return "Assistant"
    else:
        return msg.__class__.__name__.replace("Message", "")


# =============================================================================
# Summarization Middleware
# =============================================================================


@dataclass
class SummarizationMiddleware:
    """Middleware that auto-summarizes when conversation gets too long.

    Attach to an Agent to automatically compress conversation history
    when it approaches context limits.

    Example:
        ```python
        from ai_infra import Agent
        from ai_infra.memory import SummarizationMiddleware

        agent = Agent(
            tools=[...],
            middleware=[
                SummarizationMiddleware(
                    trigger_tokens=4000,  # Summarize when over 4000 tokens
                    keep_messages=10,     # Always keep last 10 messages
                )
            ]
        )
        ```

    Attributes:
        trigger_tokens: Summarize when token count exceeds this threshold
        trigger_messages: Summarize when message count exceeds this threshold
        keep_messages: Number of recent messages to always keep
        llm: LLM to use for summarization (uses default if None)
        summarize_prompt: Custom prompt template
    """

    trigger_tokens: int | None = None
    """Token threshold to trigger summarization."""

    trigger_messages: int | None = None
    """Message count threshold to trigger summarization."""

    keep_messages: int = 10
    """Number of recent messages to always keep."""

    llm: Any | None = None
    """LLM instance for summarization."""

    summarize_prompt: str | None = None
    """Custom summarization prompt."""

    _last_summary: str | None = field(default=None, repr=False)
    """Last generated summary (for debugging)."""

    def should_summarize(self, messages: Sequence[BaseMessage]) -> bool:
        """Check if messages should be summarized.

        Args:
            messages: Current message history

        Returns:
            True if summarization should be triggered
        """
        if not messages:
            return False

        # Check message count threshold
        if self.trigger_messages is not None:
            if len(messages) > self.trigger_messages:
                return True

        # Check token threshold
        if self.trigger_tokens is not None:
            token_count = count_tokens_approximate(messages)
            if token_count > self.trigger_tokens:
                return True

        return False

    def process(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Process messages, summarizing if needed.

        Args:
            messages: Current message history

        Returns:
            Potentially summarized message history
        """
        if not self.should_summarize(messages):
            return messages

        logger.info(f"Summarizing {len(messages)} messages (keeping last {self.keep_messages})")

        result = summarize_messages(
            messages,
            keep_last=self.keep_messages,
            llm=self.llm,
            summarize_prompt=self.summarize_prompt,
        )

        self._last_summary = result.summary
        return result.messages

    async def aprocess(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Async version of process.

        Args:
            messages: Current message history

        Returns:
            Potentially summarized message history
        """
        if not self.should_summarize(messages):
            return messages

        logger.info(f"Summarizing {len(messages)} messages (keeping last {self.keep_messages})")

        result = await asummarize_messages(
            messages,
            keep_last=self.keep_messages,
            llm=self.llm,
            summarize_prompt=self.summarize_prompt,
        )

        self._last_summary = result.summary
        return result.messages
