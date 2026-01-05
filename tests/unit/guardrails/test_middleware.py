"""Tests for guardrails middleware integration.

Tests the GuardrailsMiddleware class and its integration with Agent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ai_infra.guardrails import (
    Guardrail,
    GuardrailPipeline,
    GuardrailResult,
)
from ai_infra.guardrails.middleware import (
    GuardrailsConfig,
    GuardrailsMiddleware,
    GuardrailViolation,
    create_guardrails_middleware,
)

if TYPE_CHECKING:
    pass


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


class AlwaysPassGuardrail(Guardrail):
    """A guardrail that always passes."""

    @property
    def name(self) -> str:
        return "always_pass"

    def check(self, content: str, context: dict | None = None) -> GuardrailResult:
        return GuardrailResult(
            passed=True,
            guardrail_name=self.name,
            severity="low",
        )

    async def acheck(self, content: str, context: dict | None = None) -> GuardrailResult:
        return self.check(content, context)


class AlwaysFailGuardrail(Guardrail):
    """A guardrail that always fails."""

    @property
    def name(self) -> str:
        return "always_fail"

    def check(self, content: str, context: dict | None = None) -> GuardrailResult:
        return GuardrailResult(
            passed=False,
            guardrail_name=self.name,
            severity="high",
            reason="Content blocked for testing",
        )

    async def acheck(self, content: str, context: dict | None = None) -> GuardrailResult:
        return self.check(content, context)


@pytest.fixture
def pass_guardrail() -> AlwaysPassGuardrail:
    """Single passing guardrail."""
    return AlwaysPassGuardrail()


@pytest.fixture
def fail_guardrail() -> AlwaysFailGuardrail:
    """Single failing guardrail."""
    return AlwaysFailGuardrail()


# -----------------------------------------------------------------------------
# GuardrailsConfig Tests
# -----------------------------------------------------------------------------


class TestGuardrailsConfig:
    """Tests for GuardrailsConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GuardrailsConfig()
        assert config.on_input_failure == "raise"
        assert config.on_output_failure == "raise"
        assert config.check_tool_inputs is False
        assert config.check_tool_outputs is False
        assert config.log_violations is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = GuardrailsConfig(
            on_input_failure="block",
            on_output_failure="redact",
            blocked_response="Custom block message",
        )
        assert config.on_input_failure == "block"
        assert config.on_output_failure == "redact"
        assert config.blocked_response == "Custom block message"

    def test_to_pipeline(self, pass_guardrail):
        """Test conversion to pipeline."""
        config = GuardrailsConfig(
            input_guardrails=[pass_guardrail],
            output_guardrails=[pass_guardrail],
        )
        pipeline = config.to_pipeline()
        assert isinstance(pipeline, GuardrailPipeline)


# -----------------------------------------------------------------------------
# GuardrailViolation Tests
# -----------------------------------------------------------------------------


class TestGuardrailViolation:
    """Tests for GuardrailViolation exception."""

    def test_with_guardrail_result(self, fail_guardrail):
        """Test exception with GuardrailResult."""
        result = fail_guardrail.check("test")
        exc = GuardrailViolation(result, stage="input")
        assert "input" in str(exc)
        assert exc.stage == "input"
        assert exc.result is result

    def test_with_custom_reason(self, fail_guardrail):
        """Test exception with custom reason."""
        result = fail_guardrail.check("test")
        exc = GuardrailViolation(result, stage="output", reason="Custom reason")
        assert str(exc) == "Custom reason"
        assert exc.reason == "Custom reason"


# -----------------------------------------------------------------------------
# GuardrailsMiddleware Tests
# -----------------------------------------------------------------------------


class TestGuardrailsMiddleware:
    """Tests for GuardrailsMiddleware class."""

    def test_initialization_basic(self, pass_guardrail):
        """Test basic initialization."""
        middleware = GuardrailsMiddleware(
            input_guardrails=[pass_guardrail],
            output_guardrails=[pass_guardrail],
        )
        assert len(middleware.input_guardrails) == 1
        assert len(middleware.output_guardrails) == 1
        assert middleware.on_input_failure == "raise"

    def test_initialization_with_options(self, pass_guardrail):
        """Test initialization with custom options."""
        middleware = GuardrailsMiddleware(
            input_guardrails=[pass_guardrail],
            on_input_failure="block",
            on_output_failure="redact",
            check_tool_inputs=True,
        )
        assert middleware.on_input_failure == "block"
        assert middleware.on_output_failure == "redact"
        assert middleware.check_tool_inputs is True

    def test_check_input_passes(self, pass_guardrail):
        """Test check_input with passing guardrails."""
        middleware = GuardrailsMiddleware(input_guardrails=[pass_guardrail])
        should_proceed, result = middleware.check_input("test content")
        assert should_proceed is True
        assert result.passed is True

    def test_check_input_fails_with_raise(self, fail_guardrail):
        """Test check_input with failing guardrails and raise action."""
        middleware = GuardrailsMiddleware(
            input_guardrails=[fail_guardrail],
            on_input_failure="raise",
        )
        with pytest.raises(GuardrailViolation) as exc_info:
            middleware.check_input("test content")
        assert exc_info.value.stage == "input"

    def test_check_input_fails_with_block(self, fail_guardrail):
        """Test check_input with failing guardrails and block action."""
        middleware = GuardrailsMiddleware(
            input_guardrails=[fail_guardrail],
            on_input_failure="block",
        )
        should_proceed, result = middleware.check_input("test content")
        assert should_proceed is False
        assert result.passed is False

    def test_check_input_fails_with_warn(self, fail_guardrail):
        """Test check_input with failing guardrails and warn action."""
        middleware = GuardrailsMiddleware(
            input_guardrails=[fail_guardrail],
            on_input_failure="warn",
        )
        should_proceed, result = middleware.check_input("test content")
        assert should_proceed is True  # Still proceeds with warning
        assert result.passed is False

    def test_check_output_passes(self, pass_guardrail):
        """Test check_output with passing guardrails."""
        middleware = GuardrailsMiddleware(output_guardrails=[pass_guardrail])
        text, result = middleware.check_output("test content")
        assert text == "test content"
        assert result.passed is True

    def test_check_output_fails_with_raise(self, fail_guardrail):
        """Test check_output with failing guardrails and raise action."""
        middleware = GuardrailsMiddleware(
            output_guardrails=[fail_guardrail],
            on_output_failure="raise",
        )
        with pytest.raises(GuardrailViolation) as exc_info:
            middleware.check_output("test content")
        assert exc_info.value.stage == "output"

    def test_check_output_fails_with_warn(self, fail_guardrail):
        """Test check_output with failing guardrails and warn action."""
        middleware = GuardrailsMiddleware(
            output_guardrails=[fail_guardrail],
            on_output_failure="warn",
        )
        text, result = middleware.check_output("test content")
        assert text == "test content"  # Text unchanged with warn
        assert result.passed is False

    def test_no_input_guardrails(self):
        """Test check_input when no input guardrails configured."""
        middleware = GuardrailsMiddleware()
        should_proceed, result = middleware.check_input("test content")
        assert should_proceed is True
        assert result.passed is True

    def test_no_output_guardrails(self):
        """Test check_output when no output guardrails configured."""
        middleware = GuardrailsMiddleware()
        text, result = middleware.check_output("test content")
        assert text == "test content"
        assert result.passed is True


# -----------------------------------------------------------------------------
# Async GuardrailsMiddleware Tests
# -----------------------------------------------------------------------------


class TestGuardrailsMiddlewareAsync:
    """Tests for async GuardrailsMiddleware methods."""

    @pytest.mark.asyncio
    async def test_acheck_input_passes(self, pass_guardrail):
        """Test async check_input with passing guardrails."""
        middleware = GuardrailsMiddleware(input_guardrails=[pass_guardrail])
        should_proceed, result = await middleware.acheck_input("test content")
        assert should_proceed is True
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_acheck_input_fails_with_raise(self, fail_guardrail):
        """Test async check_input with failing guardrails and raise action."""
        middleware = GuardrailsMiddleware(
            input_guardrails=[fail_guardrail],
            on_input_failure="raise",
        )
        with pytest.raises(GuardrailViolation):
            await middleware.acheck_input("test content")

    @pytest.mark.asyncio
    async def test_acheck_input_fails_with_block(self, fail_guardrail):
        """Test async check_input with failing guardrails and block action."""
        middleware = GuardrailsMiddleware(
            input_guardrails=[fail_guardrail],
            on_input_failure="block",
        )
        should_proceed, result = await middleware.acheck_input("test content")
        assert should_proceed is False

    @pytest.mark.asyncio
    async def test_acheck_output_passes(self, pass_guardrail):
        """Test async check_output with passing guardrails."""
        middleware = GuardrailsMiddleware(output_guardrails=[pass_guardrail])
        text, result = await middleware.acheck_output("test content")
        assert text == "test content"
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_acheck_output_fails_with_raise(self, fail_guardrail):
        """Test async check_output with failing guardrails and raise action."""
        middleware = GuardrailsMiddleware(
            output_guardrails=[fail_guardrail],
            on_output_failure="raise",
        )
        with pytest.raises(GuardrailViolation):
            await middleware.acheck_output("test content")


# -----------------------------------------------------------------------------
# Factory Function Tests
# -----------------------------------------------------------------------------


class TestCreateGuardrailsMiddleware:
    """Tests for create_guardrails_middleware factory function."""

    def test_create_with_guardrails(self, pass_guardrail):
        """Test factory function with guardrail instances."""
        middleware = create_guardrails_middleware(
            input_guardrails=[pass_guardrail],
            output_guardrails=[pass_guardrail],
        )
        assert len(middleware.input_guardrails) == 1
        assert len(middleware.output_guardrails) == 1

    def test_create_with_options(self, pass_guardrail):
        """Test factory function with custom options."""
        middleware = create_guardrails_middleware(
            input_guardrails=[pass_guardrail],
            on_input_failure="block",
            on_output_failure="redact",
        )
        assert middleware.on_input_failure == "block"
        assert middleware.on_output_failure == "redact"

    def test_create_empty(self):
        """Test factory function with no guardrails."""
        middleware = create_guardrails_middleware()
        assert middleware.input_guardrails == []
        assert middleware.output_guardrails == []


# -----------------------------------------------------------------------------
# From Pipeline Tests
# -----------------------------------------------------------------------------


class TestFromPipeline:
    """Tests for creating middleware from pipeline."""

    def test_from_pipeline(self, pass_guardrail):
        """Test creating middleware from existing pipeline."""
        pipeline = GuardrailPipeline(
            input_guardrails=[pass_guardrail],
            output_guardrails=[pass_guardrail],
        )
        middleware = GuardrailsMiddleware.from_pipeline(
            pipeline,
            on_input_failure="block",
        )
        assert len(middleware.input_guardrails) == 1
        assert middleware.on_input_failure == "block"

    def test_from_config(self, pass_guardrail):
        """Test creating middleware from config."""
        config = GuardrailsConfig(
            input_guardrails=[pass_guardrail],
            on_input_failure="warn",
        )
        middleware = GuardrailsMiddleware.from_config(config)
        assert len(middleware.input_guardrails) == 1
        assert middleware.on_input_failure == "warn"


# -----------------------------------------------------------------------------
# Tool Wrapping Tests
# -----------------------------------------------------------------------------


class TestToolWrapping:
    """Tests for tool input/output guardrails."""

    def test_check_tool_input_disabled_by_default(self, pass_guardrail):
        """Test tool input checking is disabled by default."""
        middleware = GuardrailsMiddleware(input_guardrails=[pass_guardrail])
        should_proceed, result = middleware.check_tool_input("test_tool", {"arg": "value"})
        assert should_proceed is True
        assert result is None  # No result when disabled

    def test_check_tool_input_enabled(self, pass_guardrail):
        """Test tool input checking when enabled."""
        middleware = GuardrailsMiddleware(
            input_guardrails=[pass_guardrail],
            check_tool_inputs=True,
        )
        should_proceed, result = middleware.check_tool_input("test_tool", {"arg": "value"})
        assert should_proceed is True
        assert result is not None
        assert result.passed is True

    def test_check_tool_output_disabled_by_default(self, pass_guardrail):
        """Test tool output checking is disabled by default."""
        middleware = GuardrailsMiddleware(output_guardrails=[pass_guardrail])
        output, result = middleware.check_tool_output("test_tool", "tool result")
        assert output == "tool result"
        assert result is None

    def test_check_tool_output_enabled(self, pass_guardrail):
        """Test tool output checking when enabled."""
        middleware = GuardrailsMiddleware(
            output_guardrails=[pass_guardrail],
            check_tool_outputs=True,
        )
        output, result = middleware.check_tool_output("test_tool", "tool result")
        assert output == "tool result"
        assert result is not None
        assert result.passed is True

    def test_wrap_tool_sync(self, pass_guardrail):
        """Test tool wrapping functionality for sync function."""

        def my_tool(x: int) -> int:
            return x * 2

        middleware = GuardrailsMiddleware(
            input_guardrails=[pass_guardrail],
            output_guardrails=[pass_guardrail],
            check_tool_inputs=True,
            check_tool_outputs=True,
        )
        wrapped = middleware.wrap_tool(my_tool)
        result = wrapped(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_wrap_tool_async(self, pass_guardrail):
        """Test tool wrapping functionality for async function."""

        async def my_async_tool(x: int) -> int:
            return x * 2

        middleware = GuardrailsMiddleware(
            input_guardrails=[pass_guardrail],
            output_guardrails=[pass_guardrail],
            check_tool_inputs=True,
            check_tool_outputs=True,
        )
        wrapped = middleware.wrap_tool(my_async_tool)
        result = await wrapped(5)
        assert result == 10


# -----------------------------------------------------------------------------
# Mixed Guardrails Tests
# -----------------------------------------------------------------------------


class TestMixedGuardrails:
    """Tests with both passing and failing guardrails."""

    def test_mixed_guardrails_reports_failure(self, pass_guardrail, fail_guardrail):
        """Test that mixed guardrails report overall failure."""
        middleware = GuardrailsMiddleware(
            input_guardrails=[pass_guardrail, fail_guardrail],
            on_input_failure="warn",
        )
        should_proceed, result = middleware.check_input("test content")
        # Should still proceed with warn
        assert should_proceed is True
        # But overall result should be failed
        assert result.passed is False
        # Should have results from both guardrails
        assert len(result.results) == 2

    def test_failed_guardrails_list(self, pass_guardrail, fail_guardrail):
        """Test getting list of failed guardrails."""
        middleware = GuardrailsMiddleware(
            input_guardrails=[pass_guardrail, fail_guardrail],
            on_input_failure="warn",
        )
        should_proceed, result = middleware.check_input("test content")
        assert "always_fail" in result.failed_guardrails
        assert "always_pass" not in result.failed_guardrails
