"""Guardrails middleware for Agent integration.

This module provides middleware that integrates guardrails with the Agent class,
enabling automatic input/output validation, content moderation, and safety checks.

Example:
    >>> from ai_infra import Agent
    >>> from ai_infra.guardrails import GuardrailPipeline, PromptInjection, Toxicity
    >>> from ai_infra.guardrails.middleware import GuardrailsMiddleware
    >>>
    >>> guardrails = GuardrailPipeline(
    ...     input_guardrails=[PromptInjection()],
    ...     output_guardrails=[Toxicity()],
    ... )
    >>>
    >>> agent = Agent(
    ...     tools=[...],
    ...     guardrails=guardrails,
    ... )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ai_infra.guardrails.base import (
    Guardrail,
    GuardrailPipeline,
    GuardrailResult,
    PipelineResult,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class GuardrailViolation(Exception):
    """Exception raised when a guardrail check fails during agent execution.

    Attributes:
        result: The PipelineResult or GuardrailResult that triggered this error.
        stage: The stage where the violation occurred ("input", "output", or "tool").
        reason: Human-readable explanation of the violation.
    """

    def __init__(
        self,
        result: PipelineResult | GuardrailResult,
        stage: Literal["input", "output", "tool"] = "input",
        reason: str | None = None,
    ):
        self.result = result
        self.stage = stage

        if reason is None:
            if isinstance(result, PipelineResult):
                reason = f"Guardrail violation at {stage}: {', '.join(result.failed_guardrails)}"
            else:
                reason = f"Guardrail violation at {stage}: {result.reason}"

        self.reason = reason
        super().__init__(reason)


@dataclass
class GuardrailsConfig:
    """Configuration for guardrails behavior in Agent.

    Attributes:
        input_guardrails: List of guardrails to apply to user input.
        output_guardrails: List of guardrails to apply to LLM output.
        on_input_failure: Action when input guardrail fails.
        on_output_failure: Action when output guardrail fails.
        check_tool_inputs: Whether to check tool inputs with input guardrails.
        check_tool_outputs: Whether to check tool outputs with output guardrails.
        log_violations: Whether to log guardrail violations.
        blocked_response: Custom response when input is blocked.
    """

    input_guardrails: list[Guardrail] = field(default_factory=list)
    output_guardrails: list[Guardrail] = field(default_factory=list)
    on_input_failure: Literal["raise", "warn", "block"] = "raise"
    on_output_failure: Literal["raise", "warn", "redact", "retry"] = "raise"
    check_tool_inputs: bool = False
    check_tool_outputs: bool = False
    log_violations: bool = True
    blocked_response: str = "I cannot process this request due to content policy."
    max_output_retries: int = 3

    def to_pipeline(self) -> GuardrailPipeline:
        """Convert config to a GuardrailPipeline instance."""
        return GuardrailPipeline(
            input_guardrails=self.input_guardrails,
            output_guardrails=self.output_guardrails,
            on_failure=self.on_input_failure if self.on_input_failure != "block" else "raise",
        )


class GuardrailsMiddleware:
    """Middleware that applies guardrails to Agent inputs and outputs.

    This middleware can be used standalone or integrated into the Agent class.
    It provides input validation before sending to LLM and output validation
    before returning to the user.

    Example:
        >>> from ai_infra.guardrails import PromptInjection, Toxicity
        >>> from ai_infra.guardrails.middleware import GuardrailsMiddleware
        >>>
        >>> middleware = GuardrailsMiddleware(
        ...     input_guardrails=[PromptInjection()],
        ...     output_guardrails=[Toxicity()],
        ...     on_input_failure="raise",
        ...     on_output_failure="redact",
        ... )
        >>>
        >>> # Check input
        >>> result = middleware.check_input("user message")
        >>> if not result.passed:
        ...     raise GuardrailViolation(result, "input")
        >>>
        >>> # Check output
        >>> output, result = middleware.check_output("llm response")
        >>> # output may be redacted if violations found
    """

    def __init__(
        self,
        input_guardrails: list[Guardrail] | None = None,
        output_guardrails: list[Guardrail] | None = None,
        *,
        on_input_failure: Literal["raise", "warn", "block"] = "raise",
        on_output_failure: Literal["raise", "warn", "redact", "retry"] = "raise",
        check_tool_inputs: bool = False,
        check_tool_outputs: bool = False,
        log_violations: bool = True,
        blocked_response: str = "I cannot process this request due to content policy.",
    ):
        """Initialize the guardrails middleware.

        Args:
            input_guardrails: Guardrails to apply to user input.
            output_guardrails: Guardrails to apply to LLM output.
            on_input_failure: Action when input guardrail fails:
                - "raise": Raise GuardrailViolation exception
                - "warn": Log warning and continue
                - "block": Return blocked_response instead
            on_output_failure: Action when output guardrail fails:
                - "raise": Raise GuardrailViolation exception
                - "warn": Log warning and continue
                - "redact": Attempt to redact offending content
                - "retry": Request new output from LLM (requires agent support)
            check_tool_inputs: Apply input guardrails to tool inputs.
            check_tool_outputs: Apply output guardrails to tool outputs.
            log_violations: Log guardrail violations.
            blocked_response: Response when input is blocked.
        """
        self.input_guardrails = input_guardrails or []
        self.output_guardrails = output_guardrails or []
        self.on_input_failure = on_input_failure
        self.on_output_failure = on_output_failure
        self.check_tool_inputs = check_tool_inputs
        self.check_tool_outputs = check_tool_outputs
        self.log_violations = log_violations
        self.blocked_response = blocked_response

        # Create pipeline for running guardrails
        self._pipeline = GuardrailPipeline(
            input_guardrails=self.input_guardrails,
            output_guardrails=self.output_guardrails,
            on_failure="warn",  # We handle failure ourselves
        )

    @classmethod
    def from_pipeline(
        cls,
        pipeline: GuardrailPipeline,
        *,
        on_input_failure: Literal["raise", "warn", "block"] = "raise",
        on_output_failure: Literal["raise", "warn", "redact", "retry"] = "raise",
        check_tool_inputs: bool = False,
        check_tool_outputs: bool = False,
        log_violations: bool = True,
        blocked_response: str = "I cannot process this request due to content policy.",
    ) -> GuardrailsMiddleware:
        """Create middleware from an existing GuardrailPipeline.

        Args:
            pipeline: Existing GuardrailPipeline instance.
            **kwargs: Additional configuration options.

        Returns:
            Configured GuardrailsMiddleware instance.
        """
        middleware = cls(
            input_guardrails=pipeline.input_guardrails,
            output_guardrails=pipeline.output_guardrails,
            on_input_failure=on_input_failure,
            on_output_failure=on_output_failure,
            check_tool_inputs=check_tool_inputs,
            check_tool_outputs=check_tool_outputs,
            log_violations=log_violations,
            blocked_response=blocked_response,
        )
        return middleware

    @classmethod
    def from_config(cls, config: GuardrailsConfig) -> GuardrailsMiddleware:
        """Create middleware from a GuardrailsConfig.

        Args:
            config: GuardrailsConfig instance.

        Returns:
            Configured GuardrailsMiddleware instance.
        """
        return cls(
            input_guardrails=config.input_guardrails,
            output_guardrails=config.output_guardrails,
            on_input_failure=config.on_input_failure,
            on_output_failure=config.on_output_failure,
            check_tool_inputs=config.check_tool_inputs,
            check_tool_outputs=config.check_tool_outputs,
            log_violations=config.log_violations,
            blocked_response=config.blocked_response,
        )

    def check_input(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, PipelineResult]:
        """Check user input against input guardrails.

        Args:
            text: User input text.
            context: Optional context for guardrails.

        Returns:
            Tuple of (should_proceed, result).
            If should_proceed is False, the caller should not send to LLM.

        Raises:
            GuardrailViolation: If on_input_failure="raise" and a check fails.
        """
        if not self.input_guardrails:
            return True, PipelineResult(passed=True)

        result = self._pipeline._run_guardrails(self.input_guardrails, text, context)

        if result.passed:
            return True, result

        # Handle failure
        if self.log_violations:
            logger.warning(
                f"Input guardrail violation: {result.failed_guardrails} "
                f"(severity: {result.highest_severity})"
            )

        if self.on_input_failure == "raise":
            raise GuardrailViolation(result, stage="input")
        elif self.on_input_failure == "block":
            return False, result
        else:  # warn
            return True, result

    async def acheck_input(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, PipelineResult]:
        """Async version of check_input."""
        if not self.input_guardrails:
            return True, PipelineResult(passed=True)

        result = await self._pipeline._run_guardrails_async(self.input_guardrails, text, context)

        if result.passed:
            return True, result

        if self.log_violations:
            logger.warning(
                f"Input guardrail violation: {result.failed_guardrails} "
                f"(severity: {result.highest_severity})"
            )

        if self.on_input_failure == "raise":
            raise GuardrailViolation(result, stage="input")
        elif self.on_input_failure == "block":
            return False, result
        else:
            return True, result

    def check_output(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, PipelineResult]:
        """Check LLM output against output guardrails.

        Args:
            text: LLM output text.
            context: Optional context for guardrails.

        Returns:
            Tuple of (possibly_modified_text, result).
            Text may be redacted if on_output_failure="redact".

        Raises:
            GuardrailViolation: If on_output_failure="raise" and a check fails.
        """
        if not self.output_guardrails:
            return text, PipelineResult(passed=True)

        result = self._pipeline._run_guardrails(self.output_guardrails, text, context)

        if result.passed:
            return text, result

        if self.log_violations:
            logger.warning(
                f"Output guardrail violation: {result.failed_guardrails} "
                f"(severity: {result.highest_severity})"
            )

        if self.on_output_failure == "raise":
            raise GuardrailViolation(result, stage="output")
        elif self.on_output_failure == "redact":
            redacted = self._apply_redactions(text, result)
            return redacted, result
        else:  # warn or retry (retry is handled at agent level)
            return text, result

    async def acheck_output(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, PipelineResult]:
        """Async version of check_output."""
        if not self.output_guardrails:
            return text, PipelineResult(passed=True)

        result = await self._pipeline._run_guardrails_async(self.output_guardrails, text, context)

        if result.passed:
            return text, result

        if self.log_violations:
            logger.warning(
                f"Output guardrail violation: {result.failed_guardrails} "
                f"(severity: {result.highest_severity})"
            )

        if self.on_output_failure == "raise":
            raise GuardrailViolation(result, stage="output")
        elif self.on_output_failure == "redact":
            redacted = self._apply_redactions(text, result)
            return redacted, result
        else:
            return text, result

    def check_tool_input(
        self,
        tool_name: str,
        args: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, PipelineResult | None]:
        """Check tool input against input guardrails.

        Args:
            tool_name: Name of the tool being called.
            args: Arguments being passed to the tool.
            context: Optional context.

        Returns:
            Tuple of (should_proceed, result).
        """
        if not self.check_tool_inputs or not self.input_guardrails:
            return True, None

        # Convert args to text for checking
        import json

        args_text = json.dumps(args, default=str)
        ctx = {"tool_name": tool_name, **(context or {})}

        result = self._pipeline._run_guardrails(self.input_guardrails, args_text, ctx)

        if result.passed:
            return True, result

        if self.log_violations:
            logger.warning(
                f"Tool input guardrail violation ({tool_name}): {result.failed_guardrails}"
            )

        if self.on_input_failure == "raise":
            raise GuardrailViolation(result, stage="tool")
        elif self.on_input_failure == "block":
            return False, result
        else:
            return True, result

    def check_tool_output(
        self,
        tool_name: str,
        output: Any,
        context: dict[str, Any] | None = None,
    ) -> tuple[Any, PipelineResult | None]:
        """Check tool output against output guardrails.

        Args:
            tool_name: Name of the tool that produced output.
            output: Output from the tool.
            context: Optional context.

        Returns:
            Tuple of (possibly_modified_output, result).
        """
        if not self.check_tool_outputs or not self.output_guardrails:
            return output, None

        # Convert output to text for checking
        output_text = str(output)
        ctx = {"tool_name": tool_name, **(context or {})}

        result = self._pipeline._run_guardrails(self.output_guardrails, output_text, ctx)

        if result.passed:
            return output, result

        if self.log_violations:
            logger.warning(
                f"Tool output guardrail violation ({tool_name}): {result.failed_guardrails}"
            )

        if self.on_output_failure == "raise":
            raise GuardrailViolation(result, stage="tool")
        elif self.on_output_failure == "redact":
            redacted = self._apply_redactions(output_text, result)
            return redacted, result
        else:
            return output, result

    def _apply_redactions(self, text: str, result: PipelineResult) -> str:
        """Apply redactions from guardrail results.

        Looks for redacted_text in guardrail result details and applies them.
        """
        for gr in result.results:
            if gr.details and "redacted_text" in gr.details:
                text = gr.details["redacted_text"]
        return text

    def wrap_tool(self, tool: Any) -> Any:
        """Wrap a tool to apply guardrails to its inputs and outputs.

        Args:
            tool: The tool to wrap.

        Returns:
            Wrapped tool with guardrails applied.
        """
        import asyncio
        from functools import wraps

        # Get the tool's callable
        if hasattr(tool, "_run"):
            # LangChain tool
            original_run = tool._run
            original_arun = getattr(tool, "_arun", None)

            @wraps(original_run)
            def wrapped_run(*args, **kwargs):
                # Check input
                should_proceed, _ = self.check_tool_input(
                    tool.name if hasattr(tool, "name") else str(tool),
                    {"args": args, "kwargs": kwargs},
                )
                if not should_proceed:
                    return "Tool call blocked by guardrails"

                result = original_run(*args, **kwargs)

                # Check output
                result, _ = self.check_tool_output(
                    tool.name if hasattr(tool, "name") else str(tool),
                    result,
                )
                return result

            tool._run = wrapped_run

            if original_arun:

                @wraps(original_arun)
                async def wrapped_arun(*args, **kwargs):
                    should_proceed, _ = self.check_tool_input(
                        tool.name if hasattr(tool, "name") else str(tool),
                        {"args": args, "kwargs": kwargs},
                    )
                    if not should_proceed:
                        return "Tool call blocked by guardrails"

                    result = await original_arun(*args, **kwargs)

                    result, _ = self.check_tool_output(
                        tool.name if hasattr(tool, "name") else str(tool),
                        result,
                    )
                    return result

                tool._arun = wrapped_arun

            return tool

        elif callable(tool):
            # Plain function
            tool_name = getattr(tool, "__name__", str(tool))

            if asyncio.iscoroutinefunction(tool):

                @wraps(tool)
                async def wrapped_async(*args, **kwargs):
                    should_proceed, _ = self.check_tool_input(
                        tool_name, {"args": args, "kwargs": kwargs}
                    )
                    if not should_proceed:
                        return "Tool call blocked by guardrails"

                    result = await tool(*args, **kwargs)

                    result, _ = self.check_tool_output(tool_name, result)
                    return result

                return wrapped_async
            else:

                @wraps(tool)
                def wrapped_sync(*args, **kwargs):
                    should_proceed, _ = self.check_tool_input(
                        tool_name, {"args": args, "kwargs": kwargs}
                    )
                    if not should_proceed:
                        return "Tool call blocked by guardrails"

                    result = tool(*args, **kwargs)

                    result, _ = self.check_tool_output(tool_name, result)
                    return result

                return wrapped_sync

        return tool


def create_guardrails_middleware(
    input_guardrails: list[Guardrail] | None = None,
    output_guardrails: list[Guardrail] | None = None,
    *,
    on_input_failure: Literal["raise", "warn", "block"] = "raise",
    on_output_failure: Literal["raise", "warn", "redact", "retry"] = "raise",
    check_tool_inputs: bool = False,
    check_tool_outputs: bool = False,
    log_violations: bool = True,
) -> GuardrailsMiddleware:
    """Factory function to create guardrails middleware.

    This is a convenience function for creating GuardrailsMiddleware
    with common configurations.

    Args:
        input_guardrails: Guardrails to apply to user input.
        output_guardrails: Guardrails to apply to LLM output.
        on_input_failure: Action when input guardrail fails.
        on_output_failure: Action when output guardrail fails.
        check_tool_inputs: Apply input guardrails to tool inputs.
        check_tool_outputs: Apply output guardrails to tool outputs.
        log_violations: Log guardrail violations.

    Returns:
        Configured GuardrailsMiddleware instance.

    Example:
        >>> from ai_infra.guardrails import PromptInjection, Toxicity
        >>> from ai_infra.guardrails.middleware import create_guardrails_middleware
        >>>
        >>> middleware = create_guardrails_middleware(
        ...     input_guardrails=[PromptInjection()],
        ...     output_guardrails=[Toxicity(method="heuristic")],
        ...     on_input_failure="block",
        ...     on_output_failure="redact",
        ... )
    """
    return GuardrailsMiddleware(
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails,
        on_input_failure=on_input_failure,
        on_output_failure=on_output_failure,
        check_tool_inputs=check_tool_inputs,
        check_tool_outputs=check_tool_outputs,
        log_violations=log_violations,
    )
