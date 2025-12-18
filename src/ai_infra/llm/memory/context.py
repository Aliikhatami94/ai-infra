"""Unified context management for fitting messages into token budgets.

This module provides the primary API for managing conversation context:
- `fit_context()` - Fit messages into a token budget (trim or summarize)
- `ContextResult` - Result containing processed messages and optional summary

Example:
    ```python
    from ai_infra.memory import fit_context

    # Simple: just fit messages into budget (trims oldest)
    result = fit_context(messages, max_tokens=4000)

    # With summarization: compress old messages instead of dropping
    result = fit_context(messages, max_tokens=4000, summarize=True)

    # Rolling summary: extend existing summary (for stateless APIs)
    result = fit_context(
        messages,
        max_tokens=4000,
        summarize=True,
        summary="Previous conversation summary...",
    )
    ```
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import BaseMessage, SystemMessage

from ai_infra.llm.memory.tokens import count_tokens_approximate
from ai_infra.llm.memory.trim import _normalize_messages, trim_messages

logger = logging.getLogger(__name__)

# Default prompt for creating summaries
DEFAULT_SUMMARY_PROMPT = """Summarize the following conversation concisely.
Focus on:
- Key topics discussed
- Important decisions or facts
- User preferences mentioned
- Context needed for continuing

Conversation:
{conversation}

Summary:"""

# Default prompt for extending summaries
DEFAULT_EXTEND_PROMPT = """You have an existing summary of a conversation, plus new messages.
Create an updated summary that incorporates both.

Existing summary:
{summary}

New messages:
{conversation}

Updated summary:"""


@dataclass
class ContextResult:
    """Result from fit_context().

    Attributes:
        messages: The processed messages to use (fits within token budget)
        summary: The conversation summary (if summarize=True was used)
        tokens: Approximate token count of the result messages

        action: What action was taken ("none", "trimmed", "summarized")
        original_count: Number of messages before processing
        final_count: Number of messages after processing
    """

    messages: list[BaseMessage]
    """Use these messages - they fit within the token budget."""

    summary: str | None = None
    """Store this summary for the next turn (if summarize=True)."""

    tokens: int = 0
    """Approximate token count of result messages."""

    # Metadata
    action: Literal["none", "trimmed", "summarized"] = "none"
    """What action was taken to fit the context."""

    original_count: int = 0
    """Number of messages before processing."""

    final_count: int = 0
    """Number of messages after processing."""


def fit_context(
    messages: Sequence[BaseMessage | dict],
    max_tokens: int,
    *,
    summarize: bool = False,
    summary: str | None = None,
    keep: int = 10,
    llm: Any | None = None,
) -> ContextResult:
    """Fit messages into a token budget.

    This is the primary API for context management. It handles:
    - Counting tokens
    - Trimming messages (if under budget or summarize=False)
    - Summarizing old messages (if summarize=True)
    - Rolling summaries (if summary= is provided)

    Args:
        messages: Conversation messages (BaseMessage or dict format)
        max_tokens: Maximum tokens allowed in the result
        summarize: If True, summarize old messages instead of dropping them
        summary: Existing summary to extend (for rolling summaries)
        keep: Number of recent messages to always keep (when summarizing)
        llm: LLM instance for summarization (uses default if None)

    Returns:
        ContextResult with processed messages and optional summary

    Examples:
        >>> # Simple trim
        >>> result = fit_context(messages, max_tokens=4000)
        >>> prompt = build_prompt(result.messages)

        >>> # Summarize instead of trim
        >>> result = fit_context(messages, max_tokens=4000, summarize=True)
        >>> store_summary(result.summary)

        >>> # Rolling summary (stateless API pattern)
        >>> result = fit_context(
        ...     messages,
        ...     max_tokens=4000,
        ...     summarize=True,
        ...     summary=request.summary,
        ... )
        >>> return {"response": ..., "summary": result.summary}
    """
    if not messages:
        return ContextResult(
            messages=[],
            summary=summary,
            tokens=0,
            action="none",
            original_count=0,
            final_count=0,
        )

    # Normalize messages to BaseMessage
    normalized = _normalize_messages(messages)
    original_count = len(normalized)

    # Count current tokens
    current_tokens = count_tokens_approximate(normalized)

    # If under limit, return as-is
    if current_tokens <= max_tokens:
        return ContextResult(
            messages=list(normalized),
            summary=summary,  # Pass through existing summary
            tokens=current_tokens,
            action="none",
            original_count=original_count,
            final_count=len(normalized),
        )

    # Over limit - need to reduce
    if not summarize:
        # Just trim (drop oldest messages)
        trimmed = trim_messages(
            normalized,
            strategy="token",
            max_tokens=max_tokens,
            preserve_system=True,
        )
        return ContextResult(
            messages=trimmed,
            summary=summary,  # Pass through (unchanged)
            tokens=count_tokens_approximate(trimmed),
            action="trimmed",
            original_count=original_count,
            final_count=len(trimmed),
        )

    # Summarize old messages
    if summary:
        # Extend existing summary with new messages
        new_summary = _extend_summary(summary, normalized, keep, llm)
    else:
        # Create new summary
        new_summary = _create_summary(normalized, keep, llm)

    # Build result: summary as system message + kept messages
    kept_messages = list(normalized[-keep:]) if keep > 0 else []
    summary_msg = SystemMessage(content=f"[Conversation summary]\n{new_summary}")

    # Check if we need to preserve an existing system message
    result_messages: list[BaseMessage] = []
    if normalized and isinstance(normalized[0], SystemMessage):
        # Keep original system message, add summary after
        result_messages.append(normalized[0])
        result_messages.append(summary_msg)
        # Remove system message from kept if it's there
        if kept_messages and isinstance(kept_messages[0], SystemMessage):
            kept_messages = kept_messages[1:]
    else:
        result_messages.append(summary_msg)

    result_messages.extend(kept_messages)

    return ContextResult(
        messages=result_messages,
        summary=new_summary,
        tokens=count_tokens_approximate(result_messages),
        action="summarized",
        original_count=original_count,
        final_count=len(result_messages),
    )


async def afit_context(
    messages: Sequence[BaseMessage | dict],
    max_tokens: int,
    *,
    summarize: bool = False,
    summary: str | None = None,
    keep: int = 10,
    llm: Any | None = None,
) -> ContextResult:
    """Async version of fit_context.

    See fit_context() for full documentation.
    """
    if not messages:
        return ContextResult(
            messages=[],
            summary=summary,
            tokens=0,
            action="none",
            original_count=0,
            final_count=0,
        )

    # Normalize messages to BaseMessage
    normalized = _normalize_messages(messages)
    original_count = len(normalized)

    # Count current tokens
    current_tokens = count_tokens_approximate(normalized)

    # If under limit, return as-is
    if current_tokens <= max_tokens:
        return ContextResult(
            messages=list(normalized),
            summary=summary,
            tokens=current_tokens,
            action="none",
            original_count=original_count,
            final_count=len(normalized),
        )

    # Over limit - need to reduce
    if not summarize:
        trimmed = trim_messages(
            normalized,
            strategy="token",
            max_tokens=max_tokens,
            preserve_system=True,
        )
        return ContextResult(
            messages=trimmed,
            summary=summary,
            tokens=count_tokens_approximate(trimmed),
            action="trimmed",
            original_count=original_count,
            final_count=len(trimmed),
        )

    # Summarize old messages (async)
    if summary:
        new_summary = await _aextend_summary(summary, normalized, keep, llm)
    else:
        new_summary = await _acreate_summary(normalized, keep, llm)

    # Build result
    kept_messages = list(normalized[-keep:]) if keep > 0 else []
    summary_msg = SystemMessage(content=f"[Conversation summary]\n{new_summary}")

    result_messages: list[BaseMessage] = []
    if normalized and isinstance(normalized[0], SystemMessage):
        result_messages.append(normalized[0])
        result_messages.append(summary_msg)
        if kept_messages and isinstance(kept_messages[0], SystemMessage):
            kept_messages = kept_messages[1:]
    else:
        result_messages.append(summary_msg)

    result_messages.extend(kept_messages)

    return ContextResult(
        messages=result_messages,
        summary=new_summary,
        tokens=count_tokens_approximate(result_messages),
        action="summarized",
        original_count=original_count,
        final_count=len(result_messages),
    )


# =============================================================================
# Internal helpers
# =============================================================================


def _format_messages(messages: list[BaseMessage]) -> str:
    """Format messages as text for summarization."""
    lines = []
    for msg in messages:
        role = _get_role_name(msg)
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _get_role_name(msg: BaseMessage) -> str:
    """Get human-readable role name."""
    from langchain_core.messages import AIMessage, HumanMessage

    if isinstance(msg, SystemMessage):
        return "System"
    elif isinstance(msg, HumanMessage):
        return "User"
    elif isinstance(msg, AIMessage):
        return "Assistant"
    else:
        return msg.__class__.__name__.replace("Message", "")


def _invoke_llm(llm: Any, prompt: str) -> str:
    """Invoke LLM with prompt, supporting both ai_infra.LLM and LangChain models."""
    # Try ai_infra.LLM.chat() first
    if hasattr(llm, "chat"):
        response = llm.chat(prompt)
    # Fall back to LangChain .invoke()
    elif hasattr(llm, "invoke"):
        response = llm.invoke(prompt)
    else:
        raise TypeError(f"LLM must have 'chat' or 'invoke' method, got {type(llm)}")

    return response.content if hasattr(response, "content") else str(response)


async def _ainvoke_llm(llm: Any, prompt: str) -> str:
    """Async invoke LLM with prompt, supporting both ai_infra.LLM and LangChain models."""
    # Try ai_infra.LLM.achat() first
    if hasattr(llm, "achat"):
        response = await llm.achat(prompt)
    # Fall back to LangChain .ainvoke()
    elif hasattr(llm, "ainvoke"):
        response = await llm.ainvoke(prompt)
    else:
        raise TypeError(f"LLM must have 'achat' or 'ainvoke' method, got {type(llm)}")

    return response.content if hasattr(response, "content") else str(response)


def _create_summary(
    messages: list[BaseMessage],
    keep: int,
    llm: Any | None,
) -> str:
    """Create a new summary from messages."""
    if llm is None:
        from ai_infra import LLM

        llm = LLM()

    # Messages to summarize (exclude the ones we're keeping)
    to_summarize = messages[:-keep] if keep > 0 else messages
    conversation_text = _format_messages(to_summarize)

    prompt = DEFAULT_SUMMARY_PROMPT.format(conversation=conversation_text)
    return _invoke_llm(llm, prompt)


async def _acreate_summary(
    messages: list[BaseMessage],
    keep: int,
    llm: Any | None,
) -> str:
    """Async: Create a new summary from messages."""
    if llm is None:
        from ai_infra import LLM

        llm = LLM()

    to_summarize = messages[:-keep] if keep > 0 else messages
    conversation_text = _format_messages(to_summarize)

    prompt = DEFAULT_SUMMARY_PROMPT.format(conversation=conversation_text)
    return await _ainvoke_llm(llm, prompt)


def _extend_summary(
    existing_summary: str,
    messages: list[BaseMessage],
    keep: int,
    llm: Any | None,
) -> str:
    """Extend an existing summary with new messages."""
    if llm is None:
        from ai_infra import LLM

        llm = LLM()

    # New messages to incorporate (exclude the ones we're keeping)
    new_messages = messages[:-keep] if keep > 0 else messages
    conversation_text = _format_messages(new_messages)

    prompt = DEFAULT_EXTEND_PROMPT.format(
        summary=existing_summary,
        conversation=conversation_text,
    )
    return _invoke_llm(llm, prompt)


async def _aextend_summary(
    existing_summary: str,
    messages: list[BaseMessage],
    keep: int,
    llm: Any | None,
) -> str:
    """Async: Extend an existing summary with new messages."""
    if llm is None:
        from ai_infra import LLM

        llm = LLM()

    new_messages = messages[:-keep] if keep > 0 else messages
    conversation_text = _format_messages(new_messages)

    prompt = DEFAULT_EXTEND_PROMPT.format(
        summary=existing_summary,
        conversation=conversation_text,
    )
    return await _ainvoke_llm(llm, prompt)
