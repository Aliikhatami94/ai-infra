"""Logfire integration for ai-infra evaluation framework.

This module provides helpers for integrating pydantic-evals with Pydantic Logfire
for visualization, tracing, and span-based evaluation.

When Logfire is configured, evaluation results automatically appear in the
Logfire web UI for visualization, comparison, and collaborative analysis.

Example - Basic Logfire setup:
    >>> from ai_infra.eval.logfire import configure_logfire_evals
    >>> from ai_infra.eval import evaluate_agent, Dataset, Case
    >>>
    >>> # Configure Logfire for evals
    >>> configure_logfire_evals(service_name="my-rag-pipeline-evals")
    >>>
    >>> # Run evaluations - results automatically sent to Logfire
    >>> report = evaluate_agent(agent, dataset)

Example - Span-based evaluation:
    >>> from ai_infra.eval import Dataset, Case
    >>> from ai_infra.eval.logfire import HasMatchingSpan, configure_logfire_evals
    >>>
    >>> configure_logfire_evals()
    >>>
    >>> dataset = Dataset(
    ...     cases=[Case(inputs="test query", expected_output="answer")],
    ...     evaluators=[
    ...         HasMatchingSpan(
    ...             query={"name_contains": "search_tool"},
    ...             evaluation_name="used_search",
    ...         ),
    ...     ],
    ... )

For full Logfire documentation, see:
https://ai.pydantic.dev/evals/how-to/logfire-integration/
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

# Re-export span-based evaluators from pydantic-evals
try:
    from pydantic_evals.evaluators import HasMatchingSpan
    from pydantic_evals.otel import SpanNode, SpanQuery, SpanTree

    HAS_SPAN_EVALUATORS = True
except ImportError:
    HAS_SPAN_EVALUATORS = False
    HasMatchingSpan = None  # type: ignore[assignment, misc]
    SpanNode = None  # type: ignore[assignment, misc]
    SpanQuery = None  # type: ignore[assignment, misc]
    SpanTree = None  # type: ignore[assignment, misc]

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Track if Logfire has been configured
_logfire_configured = False


def configure_logfire_evals(
    *,
    service_name: str = "ai-infra-evals",
    environment: str | None = None,
    send_to_logfire: Literal["if-token-present", "always", "never"] = "if-token-present",
    instrument_pydantic_ai: bool = True,
) -> bool:
    """Configure Logfire for evaluation visualization.

    This function sets up Logfire to receive evaluation traces from pydantic-evals.
    Once configured, all evaluation runs will automatically appear in the Logfire
    web UI.

    Note: Requires the `logfire` package to be installed:
        pip install 'pydantic-evals[logfire]'

    Args:
        service_name: Name to identify this service in Logfire. Defaults to
            "ai-infra-evals". Use descriptive names like "rag-pipeline-evals"
            or "agent-v2-evals".
        environment: Environment name (e.g., "development", "staging", "production").
            If None, uses the LOGFIRE_ENVIRONMENT env var or defaults to "development".
        send_to_logfire: When to send traces to Logfire:
            - "if-token-present": Send only if LOGFIRE_TOKEN is set (default)
            - "always": Always try to send (will fail without token)
            - "never": Never send (useful for local testing)
        instrument_pydantic_ai: Whether to also instrument pydantic-ai for full
            tracing of agent calls. Defaults to True.

    Returns:
        True if Logfire was successfully configured, False otherwise.

    Raises:
        ImportError: If logfire is not installed and send_to_logfire is "always".

    Example:
        >>> from ai_infra.eval.logfire import configure_logfire_evals
        >>>
        >>> # Basic configuration
        >>> configure_logfire_evals()
        >>>
        >>> # Production configuration
        >>> configure_logfire_evals(
        ...     service_name="rag-pipeline-evals",
        ...     environment="production",
        ...     send_to_logfire="always",
        ... )
    """
    global _logfire_configured

    try:
        import logfire
    except ImportError:
        if send_to_logfire == "always":
            msg = (
                "logfire package is required for Logfire integration. "
                "Install it with: pip install 'pydantic-evals[logfire]'"
            )
            raise ImportError(msg)
        logger.debug("logfire not installed, skipping Logfire configuration")
        return False

    try:
        # Configure Logfire
        logfire.configure(
            service_name=service_name,
            environment=environment,
            send_to_logfire=send_to_logfire,
        )

        # Instrument pydantic-ai for full agent tracing
        if instrument_pydantic_ai:
            try:
                logfire.instrument_pydantic_ai()
            except Exception as e:
                logger.debug(f"Could not instrument pydantic-ai: {e}")

        _logfire_configured = True
        logger.info(f"Logfire configured for evals: service={service_name}")
        return True

    except Exception as e:
        logger.warning(f"Failed to configure Logfire: {e}")
        return False


def is_logfire_configured() -> bool:
    """Check if Logfire has been configured for evals.

    Returns:
        True if configure_logfire_evals() was called successfully.
    """
    return _logfire_configured


def create_span_query(
    *,
    name_contains: str | None = None,
    name_equals: str | None = None,
    name_regex: str | None = None,
    attribute_contains: dict[str, str] | None = None,
    attribute_equals: dict[str, str] | None = None,
) -> dict:
    """Create a span query dict for HasMatchingSpan evaluator.

    This is a helper function to build span query dictionaries in a type-safe way.
    Use this instead of manually constructing query dicts.

    Args:
        name_contains: Match spans whose name contains this substring.
        name_equals: Match spans with exactly this name.
        name_regex: Match spans whose name matches this regex.
        attribute_contains: Match spans where attribute values contain these substrings.
        attribute_equals: Match spans where attribute values equal these values.

    Returns:
        A query dict suitable for HasMatchingSpan.

    Example:
        >>> from ai_infra.eval.logfire import create_span_query, HasMatchingSpan
        >>>
        >>> # Match any search tool call
        >>> query = create_span_query(name_contains="search")
        >>> evaluator = HasMatchingSpan(query=query, evaluation_name="used_search")
        >>>
        >>> # Match specific tool with attribute
        >>> query = create_span_query(
        ...     name_equals="get_weather",
        ...     attribute_equals={"location": "San Francisco"},
        ... )
    """
    query: dict = {}

    if name_contains is not None:
        query["name_contains"] = name_contains
    if name_equals is not None:
        query["name_equals"] = name_equals
    if name_regex is not None:
        query["name_regex"] = name_regex
    if attribute_contains is not None:
        query["attribute_contains"] = attribute_contains
    if attribute_equals is not None:
        query["attribute_equals"] = attribute_equals

    return query


def check_tool_called(
    tool_name: str,
    *,
    evaluation_name: str | None = None,
) -> HasMatchingSpan | None:
    """Create an evaluator that checks if a specific tool was called.

    This is a convenience wrapper around HasMatchingSpan for the common case
    of verifying tool usage.

    Args:
        tool_name: Name or substring of the tool to check for.
        evaluation_name: Name for this evaluation in reports. Defaults to
            "called_{tool_name}".

    Returns:
        A HasMatchingSpan evaluator, or None if span evaluators are not available.

    Example:
        >>> from ai_infra.eval import Dataset, Case
        >>> from ai_infra.eval.logfire import check_tool_called
        >>>
        >>> dataset = Dataset(
        ...     cases=[Case(inputs="What's the weather?")],
        ...     evaluators=[
        ...         check_tool_called("get_weather"),
        ...         check_tool_called("search", evaluation_name="searched_web"),
        ...     ],
        ... )
    """
    if not HAS_SPAN_EVALUATORS or HasMatchingSpan is None:
        logger.warning("Span evaluators not available. Install pydantic-evals[logfire]")
        return None

    if evaluation_name is None:
        evaluation_name = f"called_{tool_name}"

    return HasMatchingSpan(
        query={"name_contains": tool_name},
        evaluation_name=evaluation_name,
    )


def check_no_tool_called(
    tool_name: str,
    *,
    evaluation_name: str | None = None,
) -> HasMatchingSpan | None:
    """Create an evaluator that checks a specific tool was NOT called.

    This is useful for verifying that forbidden or unnecessary tools were avoided.
    Note: This returns a HasMatchingSpan that will report True if the span IS found.
    To check that a tool was NOT called, use ToolUsageEvaluator with forbidden_tools.

    Args:
        tool_name: Name or substring of the tool that should not be called.
        evaluation_name: Name for this evaluation in reports. Defaults to
            "avoided_{tool_name}".

    Returns:
        A HasMatchingSpan evaluator, or None if not available.

    Example:
        >>> from ai_infra.eval import Dataset, Case
        >>> from ai_infra.eval.logfire import check_no_tool_called
        >>>
        >>> # Note: For checking tools were NOT called, consider using
        >>> # ToolUsageEvaluator(forbidden_tools=["delete"]) instead
        >>> dataset = Dataset(
        ...     cases=[Case(inputs="Simple question")],
        ...     evaluators=[
        ...         check_no_tool_called("delete"),
        ...     ],
        ... )
    """
    if not HAS_SPAN_EVALUATORS or HasMatchingSpan is None:
        logger.warning("Span evaluators not available. Install pydantic-evals[logfire]")
        return None

    if evaluation_name is None:
        evaluation_name = f"avoided_{tool_name}"

    return HasMatchingSpan(
        query={"name_contains": tool_name},
        evaluation_name=evaluation_name,
    )


__all__ = [
    # Configuration
    "configure_logfire_evals",
    "is_logfire_configured",
    # Span-based evaluation (re-exported from pydantic-evals)
    "HasMatchingSpan",
    "SpanNode",
    "SpanQuery",
    "SpanTree",
    # Helper functions
    "create_span_query",
    "check_tool_called",
    "check_no_tool_called",
]
