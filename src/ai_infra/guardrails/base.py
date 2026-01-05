"""Guardrail base classes and result types.

This module provides the foundational classes for building guardrails:
- GuardrailResult: The result of a guardrail check
- Guardrail: Abstract base class for all guardrails
- GuardrailPipeline: Orchestrates multiple guardrails for input/output validation

Guardrails are safety checks that validate inputs before sending to LLMs
and outputs before returning to users. They help prevent:
- Prompt injection attacks
- PII leakage
- Toxic content generation
- Hallucinations and off-topic responses

Example:
    >>> from ai_infra.guardrails import Guardrail, GuardrailResult
    >>>
    >>> class CustomGuardrail(Guardrail):
    ...     name = "custom"
    ...
    ...     def check(self, text: str, context: dict | None = None) -> GuardrailResult:
    ...         if "forbidden" in text.lower():
    ...             return GuardrailResult(
    ...                 passed=False,
    ...                 reason="Forbidden word detected",
    ...                 severity="medium",
    ...             )
    ...         return GuardrailResult(passed=True)
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Severity levels for guardrail violations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FailureAction(str, Enum):
    """Actions to take when a guardrail fails."""

    RAISE = "raise"  # Raise an exception
    WARN = "warn"  # Log a warning but continue
    BLOCK = "block"  # Return a blocked response
    REDACT = "redact"  # Redact the offending content


@dataclass
class GuardrailResult:
    """Result of a guardrail check.

    Attributes:
        passed: Whether the check passed (True) or failed (False).
        reason: Human-readable explanation of why the check failed.
        severity: How severe the violation is (low, medium, high, critical).
        details: Additional structured details about the violation.
        guardrail_name: Name of the guardrail that produced this result.

    Example:
        >>> result = GuardrailResult(
        ...     passed=False,
        ...     reason="Prompt injection detected",
        ...     severity="high",
        ...     details={"pattern": "ignore previous instructions"},
        ... )
        >>> if not result.passed:
        ...     print(f"Blocked: {result.reason}")
    """

    passed: bool
    reason: str | None = None
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    details: dict[str, Any] | None = None
    guardrail_name: str | None = None

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.passed


@dataclass
class PipelineResult:
    """Result of running a guardrail pipeline.

    Contains aggregated results from all guardrails in the pipeline.

    Attributes:
        passed: Whether all guardrails passed.
        results: Individual results from each guardrail.
        failed_guardrails: Names of guardrails that failed.
        highest_severity: The most severe violation level.
    """

    passed: bool
    results: list[GuardrailResult] = field(default_factory=list)
    failed_guardrails: list[str] = field(default_factory=list)
    highest_severity: Literal["low", "medium", "high", "critical"] | None = None

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.passed


class GuardrailError(Exception):
    """Exception raised when a guardrail check fails.

    Attributes:
        result: The GuardrailResult that triggered this error.
        message: Human-readable error message.
    """

    def __init__(self, result: GuardrailResult | PipelineResult, message: str | None = None):
        self.result = result
        if message is None:
            if isinstance(result, PipelineResult):
                failed = ", ".join(result.failed_guardrails)
                message = f"Guardrail check failed: {failed}"
            else:
                message = f"Guardrail check failed: {result.reason}"
        super().__init__(message)


class Guardrail(ABC):
    """Abstract base class for all guardrails.

    Subclasses must implement the `check` method to perform their
    specific validation logic.

    Attributes:
        name: Unique identifier for this guardrail type.

    Example:
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
        ...                 severity="low",
        ...             )
        ...         return GuardrailResult(passed=True)
    """

    name: str = "base"

    @abstractmethod
    def check(self, text: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        """Check text against this guardrail.

        Args:
            text: The text to validate.
            context: Optional context dict with additional information
                (e.g., conversation history, user info).

        Returns:
            GuardrailResult indicating whether the check passed.
        """
        ...

    async def acheck(self, text: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        """Async version of check.

        Default implementation runs the sync check in a thread executor.
        Override this method for truly async implementations (e.g., LLM-based).

        Args:
            text: The text to validate.
            context: Optional context dict with additional information.

        Returns:
            GuardrailResult indicating whether the check passed.
        """
        return await asyncio.to_thread(self.check, text, context)


class GuardrailPipeline:
    """Orchestrates multiple guardrails for input and output validation.

    A pipeline runs guardrails in sequence, collecting results and
    taking action based on the configured failure mode.

    Attributes:
        input_guardrails: Guardrails to run on user inputs.
        output_guardrails: Guardrails to run on LLM outputs.
        on_failure: Action to take when a guardrail fails.

    Example:
        >>> from ai_infra.guardrails import GuardrailPipeline
        >>>
        >>> pipeline = GuardrailPipeline(
        ...     input_guardrails=[
        ...         LengthGuardrail(max_length=10000),
        ...     ],
        ...     output_guardrails=[
        ...         LengthGuardrail(max_length=5000),
        ...     ],
        ...     on_failure="raise",
        ... )
        >>>
        >>> # Check input
        >>> result = pipeline.check_input("user message")
        >>> if not result.passed:
        ...     print(f"Input blocked: {result.failed_guardrails}")
        >>>
        >>> # Check output
        >>> result = pipeline.check_output("assistant response")
    """

    def __init__(
        self,
        input_guardrails: list[Guardrail] | None = None,
        output_guardrails: list[Guardrail] | None = None,
        on_failure: Literal["raise", "warn", "block", "redact"] = "raise",
    ):
        """Initialize the guardrail pipeline.

        Args:
            input_guardrails: List of guardrails to apply to user inputs.
            output_guardrails: List of guardrails to apply to LLM outputs.
            on_failure: Action to take when a guardrail fails:
                - "raise": Raise a GuardrailError exception
                - "warn": Log a warning but continue
                - "block": Return a blocked response
                - "redact": Attempt to redact offending content
        """
        self.input_guardrails = input_guardrails or []
        self.output_guardrails = output_guardrails or []
        self.on_failure = FailureAction(on_failure)

    def _run_guardrails(
        self,
        guardrails: list[Guardrail],
        text: str,
        context: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Run a list of guardrails and collect results."""
        results: list[GuardrailResult] = []
        failed_guardrails: list[str] = []
        severity_order = ["low", "medium", "high", "critical"]
        highest_severity_idx = -1

        for guardrail in guardrails:
            try:
                result = guardrail.check(text, context)
                result.guardrail_name = guardrail.name
                results.append(result)

                if not result.passed:
                    failed_guardrails.append(guardrail.name)
                    severity_idx = severity_order.index(result.severity)
                    if severity_idx > highest_severity_idx:
                        highest_severity_idx = severity_idx

            except Exception as e:
                logger.error(f"Guardrail {guardrail.name} raised exception: {e}")
                # Treat exceptions as failures
                error_result = GuardrailResult(
                    passed=False,
                    reason=f"Guardrail error: {e}",
                    severity="high",
                    guardrail_name=guardrail.name,
                )
                results.append(error_result)
                failed_guardrails.append(guardrail.name)

        passed = len(failed_guardrails) == 0
        highest_severity = (
            severity_order[highest_severity_idx] if highest_severity_idx >= 0 else None
        )

        return PipelineResult(
            passed=passed,
            results=results,
            failed_guardrails=failed_guardrails,
            highest_severity=highest_severity,  # type: ignore[arg-type]
        )

    async def _run_guardrails_async(
        self,
        guardrails: list[Guardrail],
        text: str,
        context: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Run guardrails asynchronously."""
        results: list[GuardrailResult] = []
        failed_guardrails: list[str] = []
        severity_order = ["low", "medium", "high", "critical"]
        highest_severity_idx = -1

        # Run all guardrails concurrently
        async def run_one(guardrail: Guardrail) -> GuardrailResult:
            try:
                result = await guardrail.acheck(text, context)
                result.guardrail_name = guardrail.name
                return result
            except Exception as e:
                logger.error(f"Guardrail {guardrail.name} raised exception: {e}")
                return GuardrailResult(
                    passed=False,
                    reason=f"Guardrail error: {e}",
                    severity="high",
                    guardrail_name=guardrail.name,
                )

        results = await asyncio.gather(*[run_one(g) for g in guardrails])

        for result in results:
            if not result.passed:
                failed_guardrails.append(result.guardrail_name or "unknown")
                severity_idx = severity_order.index(result.severity)
                if severity_idx > highest_severity_idx:
                    highest_severity_idx = severity_idx

        passed = len(failed_guardrails) == 0
        highest_severity = (
            severity_order[highest_severity_idx] if highest_severity_idx >= 0 else None
        )

        return PipelineResult(
            passed=passed,
            results=list(results),
            failed_guardrails=failed_guardrails,
            highest_severity=highest_severity,  # type: ignore[arg-type]
        )

    def _handle_failure(self, result: PipelineResult) -> PipelineResult:
        """Handle a failed pipeline result based on on_failure setting."""
        if result.passed:
            return result

        if self.on_failure == FailureAction.RAISE:
            raise GuardrailError(result)
        elif self.on_failure == FailureAction.WARN:
            logger.warning(
                f"Guardrail check failed: {result.failed_guardrails} "
                f"(severity: {result.highest_severity})"
            )
        elif self.on_failure == FailureAction.BLOCK:
            # Result already indicates failure
            pass
        elif self.on_failure == FailureAction.REDACT:
            # Redaction would be handled by specific guardrails
            logger.warning("Redaction requested but not implemented for all guardrails")

        return result

    def check_input(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Check user input against all input guardrails.

        Args:
            text: The user input to validate.
            context: Optional context dict with additional information.

        Returns:
            PipelineResult with aggregated results from all guardrails.

        Raises:
            GuardrailError: If on_failure is "raise" and a guardrail fails.
        """
        result = self._run_guardrails(self.input_guardrails, text, context)
        return self._handle_failure(result)

    def check_output(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Check LLM output against all output guardrails.

        Args:
            text: The LLM output to validate.
            context: Optional context dict with additional information.

        Returns:
            PipelineResult with aggregated results from all guardrails.

        Raises:
            GuardrailError: If on_failure is "raise" and a guardrail fails.
        """
        result = self._run_guardrails(self.output_guardrails, text, context)
        return self._handle_failure(result)

    async def acheck_input(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Async version of check_input."""
        result = await self._run_guardrails_async(self.input_guardrails, text, context)
        return self._handle_failure(result)

    async def acheck_output(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Async version of check_output."""
        result = await self._run_guardrails_async(self.output_guardrails, text, context)
        return self._handle_failure(result)

    def add_input_guardrail(self, guardrail: Guardrail) -> None:
        """Add a guardrail to the input pipeline."""
        self.input_guardrails.append(guardrail)

    def add_output_guardrail(self, guardrail: Guardrail) -> None:
        """Add a guardrail to the output pipeline."""
        self.output_guardrails.append(guardrail)
