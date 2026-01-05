"""Unit tests for ai-infra guardrails base module (Phase 12.1)."""

from __future__ import annotations

from typing import Any

import pytest

from ai_infra.guardrails import (
    FailureAction,
    Guardrail,
    GuardrailError,
    GuardrailPipeline,
    GuardrailResult,
    PipelineResult,
    Severity,
)


class TestGuardrailResult:
    """Tests for GuardrailResult dataclass."""

    def test_import(self) -> None:
        """Test that GuardrailResult can be imported."""
        from ai_infra.guardrails import GuardrailResult

        assert GuardrailResult is not None

    def test_create_passed_result(self) -> None:
        """Test creating a passed result."""
        result = GuardrailResult(passed=True)
        assert result.passed is True
        assert result.reason is None
        assert result.severity == "medium"
        assert result.details is None

    def test_create_failed_result(self) -> None:
        """Test creating a failed result with details."""
        result = GuardrailResult(
            passed=False,
            reason="Prompt injection detected",
            severity="high",
            details={"pattern": "ignore instructions"},
        )
        assert result.passed is False
        assert result.reason == "Prompt injection detected"
        assert result.severity == "high"
        assert result.details == {"pattern": "ignore instructions"}

    def test_bool_conversion(self) -> None:
        """Test that result can be used in boolean context."""
        passed_result = GuardrailResult(passed=True)
        failed_result = GuardrailResult(passed=False)

        assert bool(passed_result) is True
        assert bool(failed_result) is False

        # Can be used in if statements
        if passed_result:
            pass  # Expected
        else:
            pytest.fail("Expected passed_result to be truthy")

        if failed_result:
            pytest.fail("Expected failed_result to be falsy")

    def test_severity_values(self) -> None:
        """Test all valid severity values."""
        for severity in ["low", "medium", "high", "critical"]:
            result = GuardrailResult(passed=False, severity=severity)  # type: ignore[arg-type]
            assert result.severity == severity


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_import(self) -> None:
        """Test that PipelineResult can be imported."""
        from ai_infra.guardrails import PipelineResult

        assert PipelineResult is not None

    def test_create_passed_result(self) -> None:
        """Test creating a passed pipeline result."""
        result = PipelineResult(passed=True)
        assert result.passed is True
        assert result.results == []
        assert result.failed_guardrails == []
        assert result.highest_severity is None

    def test_create_failed_result(self) -> None:
        """Test creating a failed pipeline result."""
        individual_results = [
            GuardrailResult(passed=True, guardrail_name="guard1"),
            GuardrailResult(passed=False, reason="Failed", guardrail_name="guard2"),
        ]
        result = PipelineResult(
            passed=False,
            results=individual_results,
            failed_guardrails=["guard2"],
            highest_severity="high",
        )
        assert result.passed is False
        assert len(result.results) == 2
        assert "guard2" in result.failed_guardrails
        assert result.highest_severity == "high"

    def test_bool_conversion(self) -> None:
        """Test boolean conversion."""
        passed = PipelineResult(passed=True)
        failed = PipelineResult(passed=False)

        assert bool(passed) is True
        assert bool(failed) is False


class TestSeverityEnum:
    """Tests for Severity enum."""

    def test_import(self) -> None:
        """Test that Severity can be imported."""
        from ai_infra.guardrails import Severity

        assert Severity is not None

    def test_values(self) -> None:
        """Test all severity values."""
        assert Severity.LOW.value == "low"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.HIGH.value == "high"
        assert Severity.CRITICAL.value == "critical"


class TestFailureActionEnum:
    """Tests for FailureAction enum."""

    def test_import(self) -> None:
        """Test that FailureAction can be imported."""
        from ai_infra.guardrails import FailureAction

        assert FailureAction is not None

    def test_values(self) -> None:
        """Test all failure action values."""
        assert FailureAction.RAISE.value == "raise"
        assert FailureAction.WARN.value == "warn"
        assert FailureAction.BLOCK.value == "block"
        assert FailureAction.REDACT.value == "redact"


class TestGuardrailError:
    """Tests for GuardrailError exception."""

    def test_import(self) -> None:
        """Test that GuardrailError can be imported."""
        from ai_infra.guardrails import GuardrailError

        assert GuardrailError is not None

    def test_create_with_guardrail_result(self) -> None:
        """Test creating error with GuardrailResult."""
        result = GuardrailResult(passed=False, reason="Test failure")
        error = GuardrailError(result)

        assert error.result == result
        assert "Test failure" in str(error)

    def test_create_with_pipeline_result(self) -> None:
        """Test creating error with PipelineResult."""
        result = PipelineResult(
            passed=False,
            failed_guardrails=["guard1", "guard2"],
        )
        error = GuardrailError(result)

        assert error.result == result
        assert "guard1" in str(error)
        assert "guard2" in str(error)

    def test_custom_message(self) -> None:
        """Test error with custom message."""
        result = GuardrailResult(passed=False)
        error = GuardrailError(result, message="Custom error message")

        assert str(error) == "Custom error message"


class SimpleGuardrail(Guardrail):
    """Simple test guardrail that always passes."""

    name = "simple"

    def check(self, text: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        return GuardrailResult(passed=True)


class FailingGuardrail(Guardrail):
    """Test guardrail that always fails."""

    name = "failing"

    def __init__(self, severity: str = "medium"):
        self._severity = severity

    def check(self, text: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        return GuardrailResult(
            passed=False,
            reason="Always fails",
            severity=self._severity,  # type: ignore[arg-type]
        )


class ForbiddenWordGuardrail(Guardrail):
    """Test guardrail that checks for forbidden words."""

    name = "forbidden_word"

    def __init__(self, forbidden: str = "forbidden"):
        self.forbidden = forbidden

    def check(self, text: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        if self.forbidden.lower() in text.lower():
            return GuardrailResult(
                passed=False,
                reason=f"Forbidden word '{self.forbidden}' detected",
                severity="medium",
                details={"word": self.forbidden},
            )
        return GuardrailResult(passed=True)


class ExceptionGuardrail(Guardrail):
    """Test guardrail that raises an exception."""

    name = "exception"

    def check(self, text: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        raise ValueError("Intentional test error")


class TestGuardrailBaseClass:
    """Tests for Guardrail abstract base class."""

    def test_import(self) -> None:
        """Test that Guardrail can be imported."""
        from ai_infra.guardrails import Guardrail

        assert Guardrail is not None

    def test_simple_guardrail(self) -> None:
        """Test simple guardrail implementation."""
        guardrail = SimpleGuardrail()
        result = guardrail.check("any text")

        assert result.passed is True

    def test_failing_guardrail(self) -> None:
        """Test failing guardrail implementation."""
        guardrail = FailingGuardrail()
        result = guardrail.check("any text")

        assert result.passed is False
        assert result.reason == "Always fails"

    def test_forbidden_word_guardrail(self) -> None:
        """Test forbidden word detection."""
        guardrail = ForbiddenWordGuardrail(forbidden="secret")

        # Should pass without forbidden word
        result = guardrail.check("normal text")
        assert result.passed is True

        # Should fail with forbidden word
        result = guardrail.check("this is secret information")
        assert result.passed is False
        assert "secret" in result.reason

    def test_context_passed_to_check(self) -> None:
        """Test that context is passed to check method."""

        class ContextAwareGuardrail(Guardrail):
            name = "context_aware"
            received_context: dict | None = None

            def check(self, text: str, context: dict[str, Any] | None = None) -> GuardrailResult:
                self.received_context = context
                return GuardrailResult(passed=True)

        guardrail = ContextAwareGuardrail()
        context = {"user_id": "123", "session": "abc"}
        guardrail.check("text", context=context)

        assert guardrail.received_context == context


class TestGuardrailPipeline:
    """Tests for GuardrailPipeline."""

    def test_import(self) -> None:
        """Test that GuardrailPipeline can be imported."""
        from ai_infra.guardrails import GuardrailPipeline

        assert GuardrailPipeline is not None

    def test_create_empty_pipeline(self) -> None:
        """Test creating an empty pipeline."""
        pipeline = GuardrailPipeline()

        assert pipeline.input_guardrails == []
        assert pipeline.output_guardrails == []
        assert pipeline.on_failure == FailureAction.RAISE

    def test_create_pipeline_with_guardrails(self) -> None:
        """Test creating pipeline with guardrails."""
        input_guard = SimpleGuardrail()
        output_guard = SimpleGuardrail()

        pipeline = GuardrailPipeline(
            input_guardrails=[input_guard],
            output_guardrails=[output_guard],
            on_failure="warn",
        )

        assert len(pipeline.input_guardrails) == 1
        assert len(pipeline.output_guardrails) == 1
        assert pipeline.on_failure == FailureAction.WARN

    def test_check_input_all_pass(self) -> None:
        """Test check_input when all guardrails pass."""
        pipeline = GuardrailPipeline(
            input_guardrails=[SimpleGuardrail(), SimpleGuardrail()],
        )

        result = pipeline.check_input("test message")

        assert result.passed is True
        assert len(result.results) == 2
        assert len(result.failed_guardrails) == 0

    def test_check_input_with_failure_raise(self) -> None:
        """Test check_input raises when guardrail fails."""
        pipeline = GuardrailPipeline(
            input_guardrails=[SimpleGuardrail(), FailingGuardrail()],
            on_failure="raise",
        )

        with pytest.raises(GuardrailError) as exc_info:
            pipeline.check_input("test message")

        assert "failing" in exc_info.value.result.failed_guardrails

    def test_check_input_with_failure_warn(self) -> None:
        """Test check_input logs warning when guardrail fails."""
        pipeline = GuardrailPipeline(
            input_guardrails=[FailingGuardrail()],
            on_failure="warn",
        )

        # Should not raise
        result = pipeline.check_input("test message")

        assert result.passed is False
        assert "failing" in result.failed_guardrails

    def test_check_input_with_failure_block(self) -> None:
        """Test check_input returns failed result when on_failure is block."""
        pipeline = GuardrailPipeline(
            input_guardrails=[FailingGuardrail()],
            on_failure="block",
        )

        result = pipeline.check_input("test message")

        assert result.passed is False

    def test_check_output(self) -> None:
        """Test check_output method."""
        pipeline = GuardrailPipeline(
            output_guardrails=[SimpleGuardrail()],
        )

        result = pipeline.check_output("response text")

        assert result.passed is True

    def test_highest_severity_tracking(self) -> None:
        """Test that highest severity is tracked correctly."""
        pipeline = GuardrailPipeline(
            input_guardrails=[
                FailingGuardrail(severity="low"),
                FailingGuardrail(severity="critical"),
                FailingGuardrail(severity="medium"),
            ],
            on_failure="block",
        )

        result = pipeline.check_input("test")

        assert result.highest_severity == "critical"

    def test_exception_handling(self) -> None:
        """Test that exceptions in guardrails are handled."""
        pipeline = GuardrailPipeline(
            input_guardrails=[ExceptionGuardrail()],
            on_failure="block",
        )

        result = pipeline.check_input("test")

        assert result.passed is False
        assert "exception" in result.failed_guardrails

    def test_add_input_guardrail(self) -> None:
        """Test adding guardrail to input pipeline."""
        pipeline = GuardrailPipeline()
        guardrail = SimpleGuardrail()

        pipeline.add_input_guardrail(guardrail)

        assert guardrail in pipeline.input_guardrails

    def test_add_output_guardrail(self) -> None:
        """Test adding guardrail to output pipeline."""
        pipeline = GuardrailPipeline()
        guardrail = SimpleGuardrail()

        pipeline.add_output_guardrail(guardrail)

        assert guardrail in pipeline.output_guardrails

    def test_context_passed_through_pipeline(self) -> None:
        """Test that context is passed through pipeline."""

        class ContextCheckGuardrail(Guardrail):
            name = "context_check"
            received_context: dict | None = None

            def check(self, text: str, context: dict[str, Any] | None = None) -> GuardrailResult:
                self.received_context = context
                return GuardrailResult(passed=True)

        guardrail = ContextCheckGuardrail()
        pipeline = GuardrailPipeline(input_guardrails=[guardrail])

        context = {"key": "value"}
        pipeline.check_input("test", context=context)

        assert guardrail.received_context == context


class TestGuardrailPipelineAsync:
    """Tests for async guardrail pipeline methods."""

    @pytest.mark.asyncio
    async def test_acheck_input(self) -> None:
        """Test async check_input."""
        pipeline = GuardrailPipeline(
            input_guardrails=[SimpleGuardrail()],
        )

        result = await pipeline.acheck_input("test message")

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_acheck_output(self) -> None:
        """Test async check_output."""
        pipeline = GuardrailPipeline(
            output_guardrails=[SimpleGuardrail()],
        )

        result = await pipeline.acheck_output("response text")

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_acheck_with_failure(self) -> None:
        """Test async check with failure."""
        pipeline = GuardrailPipeline(
            input_guardrails=[FailingGuardrail()],
            on_failure="raise",
        )

        with pytest.raises(GuardrailError):
            await pipeline.acheck_input("test")


class TestGuardrailModuleExports:
    """Test that all expected exports are available."""

    def test_all_exports_from_guardrails(self) -> None:
        """Test all exports from ai_infra.guardrails."""
        from ai_infra.guardrails import (
            FailureAction,
            Guardrail,
            GuardrailError,
            GuardrailPipeline,
            GuardrailResult,
            PipelineResult,
            Severity,
        )

        assert Guardrail is not None
        assert GuardrailResult is not None
        assert GuardrailPipeline is not None
        assert PipelineResult is not None
        assert GuardrailError is not None
        assert Severity is not None
        assert FailureAction is not None

    def test_module_docstring(self) -> None:
        """Test that module has docstring."""
        import ai_infra.guardrails as guardrails

        assert guardrails.__doc__ is not None
        assert "Guardrails" in guardrails.__doc__
