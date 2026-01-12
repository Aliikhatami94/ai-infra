"""Handle failure node for executor graph.

Phase 1.2.2: Handles failures and determines retry strategy.
Phase 2.2: Simplified retry logic using repair_count and test_repair_count.
Phase 2.3.1: Adaptive replanning with failure analysis and classification.
Phase 2.4: Deprecated adaptive replanning (FailureClassification, analyze_failure_node).
Phase 2.4.1: Detailed failure category detection (granular classification).
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import TYPE_CHECKING

from ai_infra.executor.failure import FailureCategory
from ai_infra.executor.state import (
    ExecutorGraphState,
    NonRetryableErrors,
    RetryPolicy,
)
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.agent import Agent

logger = get_logger("executor.nodes.failure")


# =============================================================================
# Phase 2.3.1: Failure Classification (DEPRECATED in Phase 2.4)
# =============================================================================


class FailureClassification(str, Enum):
    """Classification of failure types for adaptive replanning.

    .. deprecated:: Phase 2.4
        Adaptive replanning has been removed. Use FailureCategory from
        ai_infra.executor.failure for logging/metrics instead.

    Phase 2.3.1: Determines whether to retry, replan, or give up.

    - TRANSIENT: Temporary failures that may succeed on retry (rate limits, timeouts)
    - WRONG_APPROACH: Fundamental approach is wrong, needs replanning
    - FATAL: Unrecoverable errors that should stop execution
    """

    TRANSIENT = "transient"
    """Temporary failure - retry same approach (rate limit, timeout, flaky test)."""

    WRONG_APPROACH = "wrong_approach"
    """Approach is fundamentally wrong - needs replanning (missing dep, wrong file)."""

    FATAL = "fatal"
    """Unrecoverable error - stop execution (permission denied, invalid config)."""


# Pattern-based classification for common errors (no LLM needed)
TRANSIENT_PATTERNS = [
    "rate limit",
    "rate_limit",
    "ratelimit",
    "timeout",
    "timed out",
    "connection refused",
    "connection reset",
    "temporary failure",
    "service unavailable",
    "503",
    "429",
    "too many requests",
    "retry after",
]

WRONG_APPROACH_PATTERNS = [
    "modulenotfounderror",
    "importerror",
    "no module named",
    "cannot find module",
    "file not found",
    "filenotfounderror",
    "no such file",
    "nameerror",
    "name.*is not defined",
    "attributeerror",
    "has no attribute",
    "typeerror",
    "expected.*got",
    "invalid syntax",
    "syntaxerror",
    "indentationerror",
    "unexpected indent",
    "command not found",
    "not recognized as",
    "package.*not installed",
]

FATAL_PATTERNS = [
    "permission denied",
    "access denied",
    "authentication failed",
    "invalid credentials",
    "api key",
    "unauthorized",
    "403 forbidden",
    "out of memory",
    "disk full",
    "no space left",
    "segmentation fault",
    "core dumped",
]


# =============================================================================
# Phase 2.4.1: Detailed Failure Category Patterns
# =============================================================================

# Maps patterns to detailed FailureCategory (more granular than classification)
CATEGORY_PATTERNS: dict[FailureCategory, list[str]] = {
    FailureCategory.SYNTAX_ERROR: [
        "syntaxerror",
        "invalid syntax",
        "unexpected eof",
        "indentationerror",
        "unexpected indent",
        "unexpected unindent",
        "expected an indented block",
        "parsing error",
    ],
    FailureCategory.IMPORT_ERROR: [
        "modulenotfounderror",
        "importerror",
        "no module named",
        "cannot find module",
        "cannot import name",
        "module has no attribute",
    ],
    FailureCategory.TYPE_ERROR: [
        "typeerror",
        "expected type",
        "incompatible type",
        "type mismatch",
        "argument of type",
        "cannot be assigned to type",
        "missing required argument",
        "unexpected keyword argument",
    ],
    FailureCategory.TEST_FAILURE: [
        "assert",
        "assertionerror",
        "test failed",
        "tests failed",
        "pytest",
        "unittest",
        "failed tests",
        "failure in test",
    ],
    FailureCategory.FILE_NOT_FOUND: [
        "filenotfounderror",
        "no such file or directory",
        "path does not exist",
        "file not found",
    ],
    FailureCategory.WRONG_APPROACH: [
        "nameerror",
        "attributeerror",
        "keyerror",
        "indexerror",
        "valueerror",
    ],
    FailureCategory.TIMEOUT: [
        "timeout",
        "timed out",
        "deadline exceeded",
        "operation timed out",
    ],
    FailureCategory.API_ERROR: [
        "rate limit",
        "rate_limit",
        "429",
        "too many requests",
        "api error",
        "connection refused",
        "connection reset",
        "service unavailable",
        "503",
    ],
    FailureCategory.TOOL_FAILURE: [
        "tool failed",
        "command not found",
        "not recognized as",
        "permission denied",
        "access denied",
    ],
}


def detect_failure_category(error_message: str) -> FailureCategory:
    """Detect detailed failure category from error message.

    Phase 2.4.1: Provides granular classification for failure analysis.

    Args:
        error_message: The error message to analyze.

    Returns:
        The most specific FailureCategory that matches.
    """
    error_lower = error_message.lower()

    # Check each category's patterns
    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if pattern in error_lower:
                return category

    # Default to UNKNOWN
    return FailureCategory.UNKNOWN


def classify_error_by_pattern(error_message: str) -> FailureClassification | None:
    """Classify error using pattern matching (no LLM call).

    Args:
        error_message: The error message to classify.

    Returns:
        Classification if pattern matched, None if LLM analysis needed.
    """
    error_lower = error_message.lower()

    # Check fatal patterns first (most specific)
    for pattern in FATAL_PATTERNS:
        if pattern in error_lower:
            return FailureClassification.FATAL

    # Check transient patterns
    for pattern in TRANSIENT_PATTERNS:
        if pattern in error_lower:
            return FailureClassification.TRANSIENT

    # Check wrong approach patterns
    for pattern in WRONG_APPROACH_PATTERNS:
        if pattern in error_lower:
            return FailureClassification.WRONG_APPROACH

    # No pattern matched - need LLM analysis
    return None


async def analyze_failure_node(
    state: ExecutorGraphState,
    *,
    analyzer_agent: Agent | None = None,
) -> ExecutorGraphState:
    """Analyze failure to determine if retry, replan, or give up.

    .. deprecated:: Phase 2.4
        Adaptive replanning has been removed. Use handle_failure_node with
        repair_code_node and repair_test_node instead. This function is kept
        for backward compatibility but is no longer used in the default graph.

    Phase 2.3.1: Classifies errors to enable adaptive replanning.
    Phase 2.4.1: Adds detailed failure_category for granular analysis.

    This node:
    1. Gets error from state
    2. Detects detailed failure_category (Phase 2.4.1)
    3. Attempts pattern-based classification (fast, no LLM)
    4. Falls back to LLM analysis for ambiguous cases
    5. Returns classification, category, and suggested fix

    Args:
        state: Current graph state with error.
        analyzer_agent: Optional agent for LLM-based analysis.

    Returns:
        Updated state with failure_classification, failure_category,
        failure_reason, suggested_fix.
    """
    # Phase 2.4: Emit deprecation warning
    warnings.warn(
        "analyze_failure_node is deprecated since Phase 2.4. "
        "Use handle_failure_node with repair flows instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    error = state.get("error")
    current_task = state.get("current_task")
    task_id = str(current_task.id) if current_task else "unknown"

    if error is None:
        logger.warning(f"analyze_failure called without error for task [{task_id}]")
        return {
            **state,
            "failure_classification": FailureClassification.TRANSIENT,
            "failure_category": FailureCategory.UNKNOWN.value,
            "failure_reason": "No error provided",
            "suggested_fix": None,
        }

    error_message = error.get("message", "") or str(error)
    error_type = error.get("error_type", "unknown")

    logger.info(f"Analyzing failure for task [{task_id}]: {error_type}")

    # Phase 2.4.1: Detect detailed failure category first
    failure_category = detect_failure_category(error_message)
    logger.info(f"Detected failure_category={failure_category.value} for task [{task_id}]")

    # Step 1: Try pattern-based classification (fast)
    classification = classify_error_by_pattern(error_message)

    if classification is not None:
        logger.info(f"Pattern-classified failure as {classification.value} for task [{task_id}]")
        return {
            **state,
            "failure_classification": classification,
            "failure_category": failure_category.value,
            "failure_reason": error_message[:500],
            "suggested_fix": _get_pattern_based_suggestion(classification, error_message),
        }

    # Step 2: Use LLM for ambiguous cases
    if analyzer_agent is not None:
        try:
            analysis_result = await analyzer_agent.arun(
                f"""Analyze this error and classify it:

Error Type: {error_type}
Error Message: {error_message[:1000]}

Classify as one of:
- TRANSIENT: Temporary failure that may succeed on retry (rate limits, timeouts, flaky tests)
- WRONG_APPROACH: Fundamental approach is wrong, needs different strategy (missing dependency, wrong file path, incorrect API usage)
- FATAL: Unrecoverable error that should stop execution (permission denied, invalid credentials)

Respond with:
CLASSIFICATION: <TRANSIENT|WRONG_APPROACH|FATAL>
REASON: <one sentence explanation>
FIX: <suggested fix if WRONG_APPROACH, otherwise "retry" or "stop">
"""
            )

            # Parse LLM response
            response_text = str(analysis_result)
            classification = _parse_llm_classification(response_text)
            reason = _parse_llm_field(response_text, "REASON")
            fix = _parse_llm_field(response_text, "FIX")

            logger.info(f"LLM-classified failure as {classification.value} for task [{task_id}]")

            return {
                **state,
                "failure_classification": classification,
                "failure_category": failure_category.value,
                "failure_reason": reason or error_message[:500],
                "suggested_fix": fix,
            }

        except Exception as e:
            logger.warning(f"LLM analysis failed, defaulting to TRANSIENT: {e}")

    # Step 3: Default to TRANSIENT if no LLM or analysis failed
    logger.info(f"Defaulting to TRANSIENT classification for task [{task_id}]")
    return {
        **state,
        "failure_classification": FailureClassification.TRANSIENT,
        "failure_category": failure_category.value,
        "failure_reason": error_message[:500],
        "suggested_fix": None,
    }


def _get_pattern_based_suggestion(
    classification: FailureClassification,
    error_message: str,
) -> str | None:
    """Get suggestion based on pattern match."""
    if classification == FailureClassification.TRANSIENT:
        return "Retry the same approach"

    if classification == FailureClassification.FATAL:
        return "Stop execution - manual intervention required"

    # WRONG_APPROACH - provide specific suggestions
    error_lower = error_message.lower()

    if "modulenotfounderror" in error_lower or "no module named" in error_lower:
        # Extract module name if possible
        import re

        match = re.search(r"no module named ['\"]?(\w+)", error_lower)
        module = match.group(1) if match else "the module"
        return f"Install missing dependency: pip install {module}"

    if "filenotfounderror" in error_lower or "no such file" in error_lower:
        return "Check file path and create missing files/directories"

    if "syntaxerror" in error_lower or "indentationerror" in error_lower:
        return "Fix syntax errors in the generated code"

    if "typeerror" in error_lower or "attributeerror" in error_lower:
        return "Review API usage and fix type/attribute errors"

    return "Revise approach based on error feedback"


def _parse_llm_classification(response: str) -> FailureClassification:
    """Parse classification from LLM response."""
    response_upper = response.upper()

    if "WRONG_APPROACH" in response_upper:
        return FailureClassification.WRONG_APPROACH
    if "FATAL" in response_upper:
        return FailureClassification.FATAL

    # Default to TRANSIENT
    return FailureClassification.TRANSIENT


def _parse_llm_field(response: str, field: str) -> str | None:
    """Parse a field from LLM response."""
    import re

    pattern = rf"{field}:\s*(.+?)(?:\n|$)"
    match = re.search(pattern, response, re.IGNORECASE)
    return match.group(1).strip() if match else None


def handle_failure_node(
    state: ExecutorGraphState,
    *,
    max_retries: int | None = None,
) -> ExecutorGraphState:
    """Handle a failure and determine if retry is appropriate.

    Phase 2.2: Simplified retry logic using repair_count and test_repair_count
    instead of generic retry_count. Now routes to repair flows instead of
    retrying the same code.

    This node:
    1. Gets error from state
    2. Classifies error as validation or test failure
    3. Checks appropriate repair limit (repair_count or test_repair_count)
    4. Marks task as failed if repair limits exceeded

    Args:
        state: Current graph state with error.
        max_retries: DEPRECATED - kept for backward compatibility.

    Returns:
        Updated state indicating whether to repair or fail.
    """
    # Phase 2.2: Get both repair counters
    repair_count = state.get("repair_count", 0)
    test_repair_count = state.get("test_repair_count", 0)
    # Keep retry_count for backward compatibility
    retry_count = state.get("retry_count", 0)

    error = state.get("error")
    current_task = state.get("current_task")
    task_id = str(current_task.id) if current_task else "unknown"

    if error is None:
        logger.warning(f"handle_failure called without error for task [{task_id}]")
        return state

    error_type = error.get("error_type", "unknown")
    error_message = error.get("message", "Unknown error")
    is_recoverable = error.get("recoverable", False)

    # Phase 2.2: Determine if this is a validation error or test failure
    is_validation_error = _is_validation_error(error_type, error_message)
    is_test_failure = _is_test_failure(error_type, error_message)

    # Select appropriate counter and max limit
    if is_validation_error:
        current_count = repair_count
        max_repairs = 2  # MAX_REPAIRS from repair.py
        counter_name = "repair_count"
    elif is_test_failure:
        current_count = test_repair_count
        max_repairs = 2  # MAX_TEST_REPAIRS for test failures
        counter_name = "test_repair_count"
    else:
        # Generic retry (backward compat) - use retry_count
        effective_max_retries = max_retries if max_retries is not None else RetryPolicy.MAX_RETRIES
        current_count = retry_count
        max_repairs = effective_max_retries
        counter_name = "retry_count"

    logger.info(
        f"Handling failure for task [{task_id}]: "
        f"{error_type} - {error_message} ({counter_name}={current_count}, max={max_repairs})"
    )

    # Check if error message indicates non-retryable error
    if NonRetryableErrors.is_non_retryable(error_message):
        logger.error(f"Non-retryable error for task [{task_id}]: {error_type}")
        return {
            **state,
            "should_continue": False,
            "error": {
                **error,
                "recoverable": False,
            },
        }

    # Check if we've exceeded repair limits
    if current_count >= max_repairs:
        logger.error(f"Max repairs ({max_repairs}) exceeded for task [{task_id}]")
        return {
            **state,
            "should_continue": False,
            "error": {
                **error,
                "message": f"{error_message} (max repairs exceeded)",
                "recoverable": False,
            },
        }

    # If error is not recoverable, don't retry
    if not is_recoverable:
        logger.error(f"Non-recoverable error for task [{task_id}]: {error_message}")
        return {
            **state,
            "should_continue": False,
        }

    # Increment appropriate counter and allow repair/retry
    new_count = current_count + 1
    logger.info(f"Will repair task [{task_id}] (attempt {new_count}/{max_repairs})")

    # Build result with incremented counter
    result = {
        **state,
        "error": None,  # Clear error for retry
        "verified": False,  # Reset verification state
        "agent_result": None,  # Clear previous result
    }

    # Increment the appropriate counter
    if is_validation_error:
        result["repair_count"] = new_count
    elif is_test_failure:
        result["test_repair_count"] = new_count
    else:
        result["retry_count"] = new_count

    return result


def _is_validation_error(error_type: str, error_message: str) -> bool:
    """Check if error is a validation/syntax error.

    Phase 2.2: Validation errors use repair_count.
    """
    error_lower = error_message.lower()
    type_lower = error_type.lower()
    return (
        "syntax" in error_lower
        or "syntaxerror" in type_lower
        or "indentationerror" in type_lower
        or "validation" in error_lower
        or "ast" in error_lower
        or type_lower == "validation_error"
    )


def _is_test_failure(error_type: str, error_message: str) -> bool:
    """Check if error is a test failure.

    Phase 2.2: Test failures use test_repair_count.
    """
    error_lower = error_message.lower()
    type_lower = error_type.lower()
    return (
        "test" in error_lower
        or "pytest" in error_lower
        or "assertion" in error_lower
        or "assertionerror" in type_lower
        or type_lower == "test_failure"
        or type_lower == "verification_failed"
    )
