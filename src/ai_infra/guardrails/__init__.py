"""Guardrails and safety module for ai-infra.

This module provides input/output validation, content moderation, and safety
checks for LLM applications. Guardrails help prevent prompt injection attacks,
PII leakage, toxic content generation, and other safety concerns.

Key Components:
- Guardrail: Abstract base class for implementing safety checks
- GuardrailResult: Result of a guardrail check
- GuardrailPipeline: Orchestrates multiple guardrails for input/output validation
- GuardrailError: Exception raised when guardrails fail

Example - Basic usage:
    >>> from ai_infra.guardrails import GuardrailPipeline
    >>>
    >>> pipeline = GuardrailPipeline(
    ...     input_guardrails=[],
    ...     output_guardrails=[],
    ...     on_failure="raise",
    ... )
    >>>
    >>> result = pipeline.check_input("user message")
    >>> if not result.passed:
    ...     print(f"Blocked: {result.failed_guardrails}")

Example - Custom guardrail:
    >>> from ai_infra.guardrails import Guardrail, GuardrailResult
    >>>
    >>> class LengthGuardrail(Guardrail):
    ...     name = "length"
    ...
    ...     def __init__(self, max_length: int = 1000):
    ...         self.max_length = max_length
    ...
    ...     def check(self, text: str, context: dict | None = None) -> GuardrailResult:
    ...         if len(text) > self.max_length:
    ...             return GuardrailResult(
    ...                 passed=False,
    ...                 reason=f"Text exceeds {self.max_length} characters",
    ...             )
    ...         return GuardrailResult(passed=True)

For full documentation on guardrails, see:
https://docs.nfrax.dev/ai-infra/guardrails/
"""

from __future__ import annotations

# Core classes
from ai_infra.guardrails.base import (
    FailureAction,
    Guardrail,
    GuardrailError,
    GuardrailPipeline,
    GuardrailResult,
    PipelineResult,
    Severity,
)

# Input guardrails
from ai_infra.guardrails.input import (
    PIIDetection,
    PIIMatch,
    PromptInjection,
    TopicFilter,
)

# Middleware
from ai_infra.guardrails.middleware import (
    GuardrailsConfig,
    GuardrailsMiddleware,
    GuardrailViolation,
    create_guardrails_middleware,
)

# Output guardrails
from ai_infra.guardrails.output import (
    Hallucination,
    PIILeakage,
    PIILeakageMatch,
    Toxicity,
)

__all__ = [
    # Core classes
    "Guardrail",
    "GuardrailResult",
    "GuardrailPipeline",
    "PipelineResult",
    "GuardrailError",
    # Enums
    "Severity",
    "FailureAction",
    # Input guardrails
    "PromptInjection",
    "PIIDetection",
    "PIIMatch",
    "TopicFilter",
    # Output guardrails
    "Toxicity",
    "PIILeakage",
    "PIILeakageMatch",
    "Hallucination",
    # Middleware
    "GuardrailsMiddleware",
    "GuardrailsConfig",
    "GuardrailViolation",
    "create_guardrails_middleware",
]
