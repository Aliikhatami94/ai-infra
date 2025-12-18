"""Unified error hierarchy for ai-infra.

This module provides a consistent error hierarchy across all ai-infra components:
- LLM errors (provider issues, rate limits, validation)
- MCP errors (server, tool, connection issues)
- OpenAPI errors (parsing, network, validation)
- Graph errors (execution, validation)

All errors provide helpful, actionable error messages with:
- Clear description of what went wrong
- Context (provider, model, tool name, etc.)
- Suggested fixes or documentation links
"""

from __future__ import annotations

import logging
from typing import Any

# =============================================================================
# Logging Helper
# =============================================================================


def log_exception(
    logger: logging.Logger,
    msg: str,
    exc: Exception,
    *,
    level: str = "warning",
    include_traceback: bool = True,
) -> None:
    """Log an exception with consistent formatting.

    Use this helper instead of bare `except Exception:` blocks to ensure
    all exceptions are properly logged with context.

    Args:
        logger: The logger instance to use
        msg: Context message describing what operation failed
        exc: The exception that was caught
        level: Log level - "debug", "info", "warning", "error", "critical"
        include_traceback: Whether to include full traceback (exc_info=True)

    Example:
        try:
            result = await llm.achat("Hello")
        except Exception as e:
            log_exception(logger, "LLM call failed", e)
            # Handle gracefully or re-raise
    """
    log_func = getattr(logger, level.lower(), logger.warning)
    log_func(f"{msg}: {type(exc).__name__}: {exc}", exc_info=include_traceback)


# =============================================================================
# Base Error
# =============================================================================


class AIInfraError(Exception):
    """Base exception for all ai-infra errors.

    All ai-infra exceptions inherit from this, allowing users to catch
    all library errors with a single except clause.

    Attributes:
        message: Human-readable error description
        details: Additional context as key-value pairs
        hint: Suggested fix or action
        docs_url: Link to relevant documentation

    Example:
        try:
            result = await llm.achat("Hello")
        except AIInfraError as e:
            print(f"Error: {e.message}")
            if e.hint:
                print(f"Hint: {e.hint}")
    """

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        hint: str | None = None,
        docs_url: str | None = None,
    ):
        self.message = message
        self.details = details or {}
        self.hint = hint
        self.docs_url = docs_url

        # Build full message
        full_msg = message
        if hint:
            full_msg += f"\n  Hint: {hint}"
        if docs_url:
            full_msg += f"\n  Docs: {docs_url}"

        super().__init__(full_msg)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r})"


# =============================================================================
# LLM Errors
# =============================================================================


class LLMError(AIInfraError):
    """Base error for LLM operations."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        details: dict[str, Any] | None = None,
        hint: str | None = None,
        docs_url: str | None = None,
    ):
        self.provider = provider
        self.model = model

        # Add provider/model to details
        full_details = details or {}
        if provider:
            full_details["provider"] = provider
        if model:
            full_details["model"] = model

        super().__init__(message, details=full_details, hint=hint, docs_url=docs_url)


class ProviderError(LLMError):
    """Error from LLM provider (OpenAI, Anthropic, etc.).

    Raised when the LLM provider returns an error response.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        error_type: str | None = None,
        details: dict[str, Any] | None = None,
        hint: str | None = None,
        docs_url: str | None = None,
    ):
        self.status_code = status_code
        self.error_type = error_type

        # Build helpful message
        prefix = f"{provider} API error" if provider else "Provider error"
        if status_code:
            prefix += f" ({status_code}"
            if error_type:
                prefix += f" {error_type}"
            prefix += ")"

        full_msg = f"{prefix}: {message}"

        full_details = details or {}
        if status_code:
            full_details["status_code"] = status_code
        if error_type:
            full_details["error_type"] = error_type

        super().__init__(
            full_msg,
            provider=provider,
            model=model,
            details=full_details,
            hint=hint,
            docs_url=docs_url,
        )


class RateLimitError(ProviderError):
    """Rate limit exceeded from LLM provider.

    Provides retry-after information when available.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        provider: str | None = None,
        model: str | None = None,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.retry_after = retry_after

        hint = "Wait and retry the request"
        if retry_after:
            hint = f"Retry after {retry_after} seconds"

        # Provider-specific docs
        docs_url = None
        if provider == "openai":
            docs_url = "https://platform.openai.com/docs/guides/rate-limits"
        elif provider == "anthropic":
            docs_url = "https://docs.anthropic.com/en/api/rate-limits"

        full_details = details or {}
        if retry_after:
            full_details["retry_after"] = retry_after

        super().__init__(
            message,
            provider=provider,
            model=model,
            status_code=429,
            error_type="Too Many Requests",
            details=full_details,
            hint=hint,
            docs_url=docs_url,
        )


class AuthenticationError(ProviderError):
    """Authentication failed with LLM provider.

    Usually means API key is invalid or missing.
    """

    def __init__(
        self,
        message: str = "Authentication failed",
        *,
        provider: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        hint = "Check your API key"
        docs_url = None

        if provider == "openai":
            hint = "Set OPENAI_API_KEY environment variable or pass api_key parameter"
            docs_url = "https://platform.openai.com/api-keys"
        elif provider == "anthropic":
            hint = "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter"
            docs_url = "https://console.anthropic.com/settings/keys"

        super().__init__(
            message,
            provider=provider,
            status_code=401,
            error_type="Unauthorized",
            details=details,
            hint=hint,
            docs_url=docs_url,
        )


class ModelNotFoundError(ProviderError):
    """Model not found or not accessible."""

    def __init__(
        self,
        model: str,
        *,
        provider: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        hint = f"Check that model '{model}' exists and you have access to it"

        super().__init__(
            f"Model '{model}' not found",
            provider=provider,
            model=model,
            status_code=404,
            error_type="Not Found",
            details=details,
            hint=hint,
        )


class ContextLengthError(ProviderError):
    """Input or output exceeded context length."""

    def __init__(
        self,
        message: str = "Context length exceeded",
        *,
        provider: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        requested_tokens: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.max_tokens = max_tokens
        self.requested_tokens = requested_tokens

        hint = "Reduce the input length or use a model with larger context window"
        if max_tokens:
            hint = f"Max context is {max_tokens} tokens. {hint}"

        full_details = details or {}
        if max_tokens:
            full_details["max_tokens"] = max_tokens
        if requested_tokens:
            full_details["requested_tokens"] = requested_tokens

        super().__init__(
            message,
            provider=provider,
            model=model,
            status_code=400,
            error_type="Bad Request",
            details=full_details,
            hint=hint,
        )


class ContentFilterError(ProviderError):
    """Content was blocked by provider's content filter."""

    def __init__(
        self,
        message: str = "Content blocked by safety filter",
        *,
        provider: str | None = None,
        model: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message,
            provider=provider,
            model=model,
            status_code=400,
            error_type="Content Filtered",
            details=details,
            hint="Modify the prompt to comply with provider content policies",
        )


class OutputValidationError(LLMError):
    """LLM output failed validation against expected schema."""

    def __init__(
        self,
        message: str,
        *,
        schema: type | None = None,
        output: Any | None = None,
        errors: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.schema = schema
        self.output = output
        self.errors = errors or []

        full_details = details or {}
        if schema:
            full_details["schema"] = schema.__name__ if hasattr(schema, "__name__") else str(schema)
        if errors:
            full_details["validation_errors"] = errors

        super().__init__(
            message,
            details=full_details,
            hint="The LLM output didn't match the expected schema. Try adjusting the prompt or using a more capable model.",
        )


# =============================================================================
# MCP Errors (re-export with enhanced messages)
# =============================================================================


class MCPError(AIInfraError):
    """Base error for MCP operations."""

    pass


class MCPServerError(MCPError):
    """Error related to MCP server operations."""

    def __init__(
        self,
        message: str,
        *,
        server_name: str | None = None,
        details: dict[str, Any] | None = None,
        hint: str | None = None,
    ):
        self.server_name = server_name

        full_details = details or {}
        if server_name:
            full_details["server_name"] = server_name

        super().__init__(message, details=full_details, hint=hint)


class MCPToolError(MCPError):
    """Error executing MCP tool."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        server_name: str | None = None,
        details: dict[str, Any] | None = None,
        hint: str | None = None,
    ):
        self.tool_name = tool_name
        self.server_name = server_name

        full_details = details or {}
        if tool_name:
            full_details["tool_name"] = tool_name
        if server_name:
            full_details["server_name"] = server_name

        prefix = f"Tool '{tool_name}'" if tool_name else "Tool"
        full_msg = f"{prefix}: {message}"

        super().__init__(full_msg, details=full_details, hint=hint)


class MCPConnectionError(MCPServerError):
    """Error connecting to MCP server."""

    def __init__(
        self,
        message: str,
        *,
        server_name: str | None = None,
        transport: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.transport = transport

        hint = "Check that the server is running and accessible"
        if transport == "stdio":
            hint = "Check that the command exists and is executable"
        elif transport == "sse":
            hint = "Check that the URL is correct and the server is running"

        full_details = details or {}
        if transport:
            full_details["transport"] = transport

        super().__init__(
            message,
            server_name=server_name,
            details=full_details,
            hint=hint,
        )


class MCPTimeoutError(MCPError):
    """Timeout during MCP operation."""

    def __init__(
        self,
        message: str = "Operation timed out",
        *,
        operation: str | None = None,
        timeout: float | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.operation = operation
        self.timeout = timeout

        full_details = details or {}
        if operation:
            full_details["operation"] = operation
        if timeout:
            full_details["timeout"] = timeout

        hint = f"Increase timeout (was {timeout}s)" if timeout else "Increase timeout"

        super().__init__(message, details=full_details, hint=hint)


# =============================================================================
# OpenAPI Errors
# =============================================================================


class OpenAPIError(AIInfraError):
    """Base error for OpenAPI operations."""

    pass


class OpenAPIParseError(OpenAPIError):
    """Error parsing OpenAPI spec."""

    def __init__(
        self,
        message: str,
        *,
        errors: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.errors = errors or []

        full_details = details or {}
        if errors:
            full_details["parse_errors"] = errors

        super().__init__(
            message,
            details=full_details,
            hint="Check that the OpenAPI spec is valid JSON/YAML",
        )


class OpenAPINetworkError(OpenAPIError):
    """Network error fetching OpenAPI spec."""

    def __init__(
        self,
        message: str,
        *,
        url: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.url = url
        self.status_code = status_code

        full_details = details or {}
        if url:
            full_details["url"] = url
        if status_code:
            full_details["status_code"] = status_code

        hint = "Check that the URL is correct and accessible"
        if status_code == 404:
            hint = "The OpenAPI spec was not found at this URL"
        elif status_code == 401 or status_code == 403:
            hint = "Authentication required - provide headers or auth"

        super().__init__(message, details=full_details, hint=hint)


class OpenAPIValidationError(OpenAPIError):
    """Validation error in OpenAPI spec."""

    def __init__(
        self,
        message: str,
        *,
        missing_fields: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.missing_fields = missing_fields or []

        full_details = details or {}
        if missing_fields:
            full_details["missing_fields"] = missing_fields

        super().__init__(
            message,
            details=full_details,
            hint="Ensure the spec has required fields: openapi, info, paths",
        )


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(AIInfraError):
    """Input or configuration validation error."""

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        value: Any | None = None,
        expected: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.field = field
        self.value = value
        self.expected = expected

        full_details = details or {}
        if field:
            full_details["field"] = field
        if expected:
            full_details["expected"] = expected

        hint = None
        if field and expected:
            hint = f"'{field}' should be {expected}"

        super().__init__(message, details=full_details, hint=hint)


class ConfigurationError(ValidationError):
    """Configuration error."""

    def __init__(
        self,
        message: str,
        *,
        config_key: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message,
            field=config_key,
            details=details,
        )


# =============================================================================
# Graph Errors
# =============================================================================


class GraphError(AIInfraError):
    """Base error for graph execution."""

    pass


class GraphExecutionError(GraphError):
    """Error during graph execution."""

    def __init__(
        self,
        message: str,
        *,
        node_id: str | None = None,
        step: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.node_id = node_id
        self.step = step

        full_details = details or {}
        if node_id:
            full_details["node_id"] = node_id
        if step is not None:
            full_details["step"] = step

        prefix = f"Node '{node_id}'" if node_id else "Graph"
        full_msg = f"{prefix}: {message}"

        super().__init__(full_msg, details=full_details)


class GraphValidationError(GraphError):
    """Graph structure is invalid."""

    def __init__(
        self,
        message: str,
        *,
        errors: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.errors = errors or []

        full_details = details or {}
        if errors:
            full_details["validation_errors"] = errors

        super().__init__(
            message,
            details=full_details,
            hint="Check graph structure: ensure all nodes are connected and no cycles exist",
        )


# =============================================================================
# Tool Errors
# =============================================================================


class ToolError(AIInfraError):
    """Base error for tool operations."""

    pass


class ToolExecutionError(ToolError):
    """Error executing a tool."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.tool_name = tool_name
        self.original_error = original_error

        full_details = details or {}
        if tool_name:
            full_details["tool_name"] = tool_name

        prefix = f"Tool '{tool_name}'" if tool_name else "Tool"
        super().__init__(f"{prefix}: {message}", details=full_details)


class ToolTimeoutError(ToolError):
    """Tool execution timed out."""

    def __init__(
        self,
        message: str = "Tool execution timed out",
        *,
        tool_name: str | None = None,
        timeout: float | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.tool_name = tool_name
        self.timeout = timeout

        full_details = details or {}
        if tool_name:
            full_details["tool_name"] = tool_name
        if timeout:
            full_details["timeout"] = timeout

        hint = f"Increase timeout (was {timeout}s)" if timeout else "Increase timeout"

        super().__init__(message, details=full_details, hint=hint)


class ToolValidationError(ToolError):
    """Tool input validation failed."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        errors: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.tool_name = tool_name
        self.errors = errors or []

        full_details = details or {}
        if tool_name:
            full_details["tool_name"] = tool_name
        if errors:
            full_details["validation_errors"] = errors

        super().__init__(
            message,
            details=full_details,
            hint="Check that tool arguments match the expected schema",
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Logging helper
    "log_exception",
    # Base
    "AIInfraError",
    # LLM
    "LLMError",
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    "ModelNotFoundError",
    "ContextLengthError",
    "ContentFilterError",
    "OutputValidationError",
    # MCP
    "MCPError",
    "MCPServerError",
    "MCPToolError",
    "MCPConnectionError",
    "MCPTimeoutError",
    # OpenAPI
    "OpenAPIError",
    "OpenAPIParseError",
    "OpenAPINetworkError",
    "OpenAPIValidationError",
    # Validation
    "ValidationError",
    "ConfigurationError",
    # Graph
    "GraphError",
    "GraphExecutionError",
    "GraphValidationError",
    # Tools
    "ToolError",
    "ToolExecutionError",
    "ToolTimeoutError",
    "ToolValidationError",
]
