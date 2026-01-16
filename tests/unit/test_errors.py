"""Tests for the ai-infra error hierarchy.

Phase 13.2 of EXECUTOR_4.md - Coverage improvement for errors module.
"""

from __future__ import annotations

import logging

import pytest

from ai_infra.errors import (
    AIInfraError,
    AuthenticationError,
    ConfigurationError,
    ContentFilterError,
    ContextLengthError,
    GraphError,
    GraphExecutionError,
    GraphValidationError,
    LLMError,
    MCPConnectionError,
    MCPError,
    MCPServerError,
    MCPTimeoutError,
    MCPToolError,
    ModelNotFoundError,
    OpenAPIError,
    OpenAPINetworkError,
    OpenAPIParseError,
    OpenAPIValidationError,
    OutputValidationError,
    ProviderError,
    RateLimitError,
    ValidationError,
    log_exception,
)

# =============================================================================
# log_exception Tests
# =============================================================================


class TestLogException:
    """Tests for log_exception helper."""

    def test_log_exception_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test logging with warning level."""
        logger = logging.getLogger("test")
        exc = ValueError("test error")

        with caplog.at_level(logging.WARNING):
            log_exception(logger, "Operation failed", exc, level="warning")

        assert "Operation failed" in caplog.text
        assert "ValueError" in caplog.text

    def test_log_exception_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test logging with error level."""
        logger = logging.getLogger("test")
        exc = RuntimeError("critical error")

        with caplog.at_level(logging.ERROR):
            log_exception(logger, "Critical failure", exc, level="error")

        assert "Critical failure" in caplog.text

    def test_log_exception_without_traceback(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test logging without traceback."""
        logger = logging.getLogger("test")
        exc = ValueError("simple error")

        with caplog.at_level(logging.WARNING):
            log_exception(logger, "Simple failure", exc, include_traceback=False)

        assert "Simple failure" in caplog.text


# =============================================================================
# AIInfraError Tests
# =============================================================================


class TestAIInfraError:
    """Tests for base AIInfraError."""

    def test_basic_creation(self) -> None:
        """Test creating error with just message."""
        error = AIInfraError("Something went wrong")

        assert error.message == "Something went wrong"
        assert error.details == {}
        assert error.hint is None
        assert error.docs_url is None

    def test_with_hint(self) -> None:
        """Test error with hint."""
        error = AIInfraError("Failed", hint="Try again")

        assert error.hint == "Try again"
        assert "Hint: Try again" in str(error)

    def test_with_docs_url(self) -> None:
        """Test error with docs URL."""
        error = AIInfraError("Failed", docs_url="https://docs.example.com")

        assert error.docs_url == "https://docs.example.com"
        assert "Docs: https://docs.example.com" in str(error)

    def test_with_details(self) -> None:
        """Test error with details."""
        error = AIInfraError("Failed", details={"code": 42})

        assert error.details == {"code": 42}

    def test_repr(self) -> None:
        """Test repr format."""
        error = AIInfraError("Test message")

        assert repr(error) == "AIInfraError('Test message')"


# =============================================================================
# LLMError Tests
# =============================================================================


class TestLLMError:
    """Tests for LLMError."""

    def test_with_provider(self) -> None:
        """Test LLMError with provider."""
        error = LLMError("API failed", provider="openai")

        assert error.provider == "openai"
        assert error.details["provider"] == "openai"

    def test_with_model(self) -> None:
        """Test LLMError with model."""
        error = LLMError("Model issue", model="gpt-4o")

        assert error.model == "gpt-4o"
        assert error.details["model"] == "gpt-4o"

    def test_with_both(self) -> None:
        """Test LLMError with provider and model."""
        error = LLMError("Issue", provider="anthropic", model="claude-3")

        assert error.provider == "anthropic"
        assert error.model == "claude-3"


# =============================================================================
# ProviderError Tests
# =============================================================================


class TestProviderError:
    """Tests for ProviderError."""

    def test_with_status_code(self) -> None:
        """Test ProviderError with status code."""
        error = ProviderError("Server error", status_code=500)

        assert error.status_code == 500
        assert "(500)" in error.message

    def test_with_error_type(self) -> None:
        """Test ProviderError with error type."""
        error = ProviderError("Error", status_code=400, error_type="Bad Request")

        assert error.error_type == "Bad Request"
        assert "Bad Request" in error.message

    def test_with_provider_prefix(self) -> None:
        """Test provider name in message prefix."""
        error = ProviderError("Issue", provider="openai")

        assert "openai API error" in error.message


# =============================================================================
# RateLimitError Tests
# =============================================================================


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_default_message(self) -> None:
        """Test default rate limit message."""
        error = RateLimitError()

        assert "Rate limit" in error.message
        assert error.status_code == 429

    def test_with_retry_after(self) -> None:
        """Test rate limit with retry-after."""
        error = RateLimitError(retry_after=60.0)

        assert error.retry_after == 60.0
        assert "60" in str(error.hint)

    def test_openai_docs_url(self) -> None:
        """Test OpenAI-specific docs URL."""
        error = RateLimitError(provider="openai")

        assert "openai.com" in error.docs_url

    def test_anthropic_docs_url(self) -> None:
        """Test Anthropic-specific docs URL."""
        error = RateLimitError(provider="anthropic")

        assert "anthropic.com" in error.docs_url


# =============================================================================
# AuthenticationError Tests
# =============================================================================


class TestAuthenticationError:
    """Tests for AuthenticationError."""

    def test_default_message(self) -> None:
        """Test default auth error message."""
        error = AuthenticationError()

        assert "Authentication" in error.message
        assert error.status_code == 401

    def test_openai_hint(self) -> None:
        """Test OpenAI-specific hint."""
        error = AuthenticationError(provider="openai")

        assert "OPENAI_API_KEY" in error.hint

    def test_anthropic_hint(self) -> None:
        """Test Anthropic-specific hint."""
        error = AuthenticationError(provider="anthropic")

        assert "ANTHROPIC_API_KEY" in error.hint


# =============================================================================
# ModelNotFoundError Tests
# =============================================================================


class TestModelNotFoundError:
    """Tests for ModelNotFoundError."""

    def test_model_in_message(self) -> None:
        """Test model name in error message."""
        error = ModelNotFoundError("gpt-5-turbo")

        assert "gpt-5-turbo" in error.message
        assert error.status_code == 404


# =============================================================================
# ContextLengthError Tests
# =============================================================================


class TestContextLengthError:
    """Tests for ContextLengthError."""

    def test_with_max_tokens(self) -> None:
        """Test with max_tokens specified."""
        error = ContextLengthError(max_tokens=128000)

        assert error.max_tokens == 128000
        assert "128000" in error.hint

    def test_with_requested_tokens(self) -> None:
        """Test with requested_tokens specified."""
        error = ContextLengthError(requested_tokens=200000)

        assert error.requested_tokens == 200000


# =============================================================================
# ContentFilterError Tests
# =============================================================================


class TestContentFilterError:
    """Tests for ContentFilterError."""

    def test_default_message(self) -> None:
        """Test default content filter message."""
        error = ContentFilterError()

        assert "blocked" in error.message.lower() or "filter" in error.message.lower()


# =============================================================================
# OutputValidationError Tests
# =============================================================================


class TestOutputValidationError:
    """Tests for OutputValidationError."""

    def test_with_schema(self) -> None:
        """Test with schema type."""

        class MySchema:
            pass

        error = OutputValidationError("Invalid output", schema=MySchema)

        assert error.schema == MySchema
        assert "MySchema" in error.details["schema"]

    def test_with_errors_list(self) -> None:
        """Test with validation errors list."""
        error = OutputValidationError("Invalid", errors=["field1 required", "field2 invalid"])

        assert len(error.errors) == 2


# =============================================================================
# MCP Error Tests
# =============================================================================


class TestMCPErrors:
    """Tests for MCP error hierarchy."""

    def test_mcp_server_error(self) -> None:
        """Test MCPServerError."""
        error = MCPServerError("Server crashed", server_name="my-server")

        assert error.server_name == "my-server"
        assert error.details["server_name"] == "my-server"

    def test_mcp_tool_error(self) -> None:
        """Test MCPToolError."""
        error = MCPToolError("Tool failed", tool_name="my-tool", server_name="server")

        assert error.tool_name == "my-tool"
        assert "Tool 'my-tool'" in error.message

    def test_mcp_connection_error_stdio(self) -> None:
        """Test MCPConnectionError with stdio transport."""
        error = MCPConnectionError("Cannot connect", transport="stdio")

        assert error.transport == "stdio"
        assert "executable" in error.hint

    def test_mcp_connection_error_sse(self) -> None:
        """Test MCPConnectionError with SSE transport."""
        error = MCPConnectionError("Cannot connect", transport="sse")

        assert "URL" in error.hint

    def test_mcp_timeout_error(self) -> None:
        """Test MCPTimeoutError."""
        error = MCPTimeoutError(operation="call_tool", timeout=30.0)

        assert error.operation == "call_tool"
        assert error.timeout == 30.0
        assert "30" in error.hint


# =============================================================================
# OpenAPI Error Tests
# =============================================================================


class TestOpenAPIErrors:
    """Tests for OpenAPI error hierarchy."""

    def test_parse_error(self) -> None:
        """Test OpenAPIParseError."""
        error = OpenAPIParseError("Invalid YAML", errors=["line 5: syntax error"])

        assert len(error.errors) == 1
        assert "JSON/YAML" in error.hint

    def test_network_error(self) -> None:
        """Test OpenAPINetworkError."""
        error = OpenAPINetworkError(
            "Failed to fetch", url="https://api.example.com", status_code=404
        )

        assert error.url == "https://api.example.com"
        assert error.status_code == 404
        assert "not found" in error.hint.lower()

    def test_network_error_401(self) -> None:
        """Test OpenAPINetworkError with 401."""
        error = OpenAPINetworkError("Unauthorized", status_code=401)

        assert "Authentication" in error.hint

    def test_validation_error(self) -> None:
        """Test OpenAPIValidationError."""
        error = OpenAPIValidationError("Missing paths", missing_fields=["paths"])

        assert "paths" in error.missing_fields


# =============================================================================
# Validation Error Tests
# =============================================================================


class TestValidationErrors:
    """Tests for validation error hierarchy."""

    def test_validation_error(self) -> None:
        """Test ValidationError."""
        error = ValidationError("Invalid input", field="name", expected="string")

        assert error.field == "name"
        assert error.expected == "string"
        assert "'name' should be string" in error.hint

    def test_configuration_error(self) -> None:
        """Test ConfigurationError."""
        error = ConfigurationError("Invalid config", config_key="timeout")

        assert "timeout" in str(error.details)


# =============================================================================
# Graph Error Tests
# =============================================================================


class TestGraphErrors:
    """Tests for graph error hierarchy."""

    def test_graph_execution_error(self) -> None:
        """Test GraphExecutionError."""
        error = GraphExecutionError("Node failed", node_id="node-1", step=5)

        assert error.node_id == "node-1"
        assert error.step == 5

    def test_graph_validation_error(self) -> None:
        """Test GraphValidationError."""
        error = GraphValidationError("Invalid graph", errors=["no entry point"])

        assert len(error.errors) == 1


# =============================================================================
# Exception Hierarchy Tests
# =============================================================================


class TestExceptionHierarchy:
    """Tests for exception inheritance."""

    def test_llm_error_is_ai_infra_error(self) -> None:
        """Test LLMError inherits from AIInfraError."""
        error = LLMError("test")

        assert isinstance(error, AIInfraError)

    def test_provider_error_is_llm_error(self) -> None:
        """Test ProviderError inherits from LLMError."""
        error = ProviderError("test")

        assert isinstance(error, LLMError)
        assert isinstance(error, AIInfraError)

    def test_rate_limit_is_provider_error(self) -> None:
        """Test RateLimitError inherits from ProviderError."""
        error = RateLimitError()

        assert isinstance(error, ProviderError)
        assert isinstance(error, LLMError)
        assert isinstance(error, AIInfraError)

    def test_mcp_error_is_ai_infra_error(self) -> None:
        """Test MCPError inherits from AIInfraError."""
        error = MCPError("test")

        assert isinstance(error, AIInfraError)

    def test_openapi_error_is_ai_infra_error(self) -> None:
        """Test OpenAPIError inherits from AIInfraError."""
        error = OpenAPIError("test")

        assert isinstance(error, AIInfraError)

    def test_graph_error_is_ai_infra_error(self) -> None:
        """Test GraphError inherits from AIInfraError."""
        error = GraphError("test")

        assert isinstance(error, AIInfraError)

    def test_catch_all_ai_infra_errors(self) -> None:
        """Test catching all errors with AIInfraError."""
        errors = [
            LLMError("test"),
            RateLimitError(),
            MCPToolError("test"),
            OpenAPIParseError("test"),
            GraphExecutionError("test"),
        ]

        for error in errors:
            try:
                raise error
            except AIInfraError:
                pass  # Should catch all
