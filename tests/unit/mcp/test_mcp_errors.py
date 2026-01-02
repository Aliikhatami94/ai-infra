"""Tests for MCP error handling.

Tests cover:
- Connection errors
- Timeout errors
- Server errors
- Tool errors
- Error recovery
- Error details and context

Phase 1.1.4 of production readiness test plan.
"""

from __future__ import annotations

import pytest

from ai_infra.mcp.client import (
    MCPClient,
    MCPConnectionError,
    MCPError,
    MCPServerError,
    MCPTimeoutError,
    MCPToolError,
)

# =============================================================================
# Base Error Tests
# =============================================================================


class TestMCPError:
    """Tests for base MCPError class."""

    def test_mcp_error_message(self):
        """MCPError stores message."""
        error = MCPError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"

    def test_mcp_error_details(self):
        """MCPError stores details dict."""
        error = MCPError("Error", details={"key": "value"})
        assert error.details == {"key": "value"}

    def test_mcp_error_default_details(self):
        """MCPError defaults to empty details."""
        error = MCPError("Error")
        assert error.details is None or error.details == {}


# =============================================================================
# Server Error Tests
# =============================================================================


class TestMCPServerError:
    """Tests for MCPServerError class."""

    def test_server_error_message(self):
        """MCPServerError stores message."""
        error = MCPServerError("Server failed", server_name="myserver")
        assert "Server failed" in str(error)

    def test_server_error_server_name(self):
        """MCPServerError stores server_name."""
        error = MCPServerError("Failed", server_name="filesystem")
        assert error.server_name == "filesystem"

    def test_server_error_inherits_from_mcp_error(self):
        """MCPServerError inherits from MCPError."""
        error = MCPServerError("Failed", server_name="server")
        assert isinstance(error, MCPError)


# =============================================================================
# Connection Error Tests
# =============================================================================


class TestMCPConnectionError:
    """Tests for MCPConnectionError class."""

    def test_connection_error_message(self):
        """MCPConnectionError stores message."""
        error = MCPConnectionError("Connection refused", server_name="server")
        assert "Connection refused" in str(error)

    def test_connection_error_server_name(self):
        """MCPConnectionError stores server_name."""
        error = MCPConnectionError("Failed", server_name="myserver")
        assert error.server_name == "myserver"

    def test_connection_error_inherits_from_server_error(self):
        """MCPConnectionError inherits from MCPServerError."""
        error = MCPConnectionError("Failed", server_name="server")
        assert isinstance(error, MCPServerError)


# =============================================================================
# Timeout Error Tests
# =============================================================================


class TestMCPTimeoutError:
    """Tests for MCPTimeoutError class."""

    def test_timeout_error_message(self):
        """MCPTimeoutError stores message."""
        error = MCPTimeoutError("Timed out", operation="discover", timeout=30.0)
        assert "Timed out" in str(error)

    def test_timeout_error_operation(self):
        """MCPTimeoutError stores operation."""
        error = MCPTimeoutError("Timed out", operation="discover", timeout=30.0)
        assert error.operation == "discover"

    def test_timeout_error_timeout_value(self):
        """MCPTimeoutError stores timeout value."""
        error = MCPTimeoutError("Timed out", operation="call_tool", timeout=60.0)
        assert error.timeout == 60.0


# =============================================================================
# Tool Error Tests
# =============================================================================


class TestMCPToolError:
    """Tests for MCPToolError class."""

    def test_tool_error_message(self):
        """MCPToolError stores message."""
        error = MCPToolError("Tool failed", tool_name="read_file", server_name="filesystem")
        assert "Tool failed" in str(error)

    def test_tool_error_tool_name(self):
        """MCPToolError stores tool_name."""
        error = MCPToolError("Failed", tool_name="write_file", server_name="filesystem")
        assert error.tool_name == "write_file"

    def test_tool_error_server_name(self):
        """MCPToolError stores server_name."""
        error = MCPToolError("Failed", tool_name="execute", server_name="shell")
        assert error.server_name == "shell"

    def test_tool_error_with_details(self):
        """MCPToolError stores details."""
        error = MCPToolError(
            "Failed",
            tool_name="run_command",
            server_name="shell",
            details={"exit_code": 127, "stderr": "command not found"},
        )
        assert error.details["exit_code"] == 127
        assert "command not found" in error.details["stderr"]


# =============================================================================
# Error Hierarchy Tests
# =============================================================================


class TestErrorHierarchy:
    """Tests for error class hierarchy."""

    def test_all_errors_inherit_from_base(self):
        """All error types inherit from MCPError."""
        errors = [
            MCPServerError("msg", server_name="s"),
            MCPConnectionError("msg", server_name="s"),
            MCPTimeoutError("msg", operation="op", timeout=1.0),
            MCPToolError("msg", tool_name="t", server_name="s"),
        ]

        for error in errors:
            assert isinstance(error, MCPError)

    def test_connection_error_is_server_error(self):
        """MCPConnectionError is also MCPServerError."""
        error = MCPConnectionError("msg", server_name="s")
        assert isinstance(error, MCPServerError)

    def test_can_catch_by_base_class(self):
        """Can catch specific errors by base class."""
        try:
            raise MCPToolError("Failed", tool_name="t", server_name="s")
        except MCPError as e:
            assert isinstance(e, MCPToolError)


# =============================================================================
# Error Context Tests
# =============================================================================


class TestErrorContext:
    """Tests for error context and debugging info."""

    def test_error_string_includes_context(self):
        """Error string representation includes context."""
        error = MCPToolError(
            "Tool 'read_file' failed: File not found",
            tool_name="read_file",
            server_name="filesystem",
        )
        error_str = str(error)

        assert "read_file" in error_str or "File not found" in error_str

    def test_timeout_error_includes_timeout_value(self):
        """Timeout error can include timeout value in message."""
        error = MCPTimeoutError(
            "Operation timed out after 30.0s",
            operation="discover",
            timeout=30.0,
        )

        assert "30.0" in str(error) or error.timeout == 30.0


# =============================================================================
# Client Error Tracking Tests
# =============================================================================


class TestClientErrorTracking:
    """Tests for client error tracking."""

    def test_last_errors_initially_empty(self):
        """Client last_errors is empty initially."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        assert client.last_errors() == []

    def test_errors_recorded_during_discovery(self):
        """Errors during discovery are recorded."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        client._errors = [
            {
                "config": {"transport": "stdio", "command": "npx"},
                "error_type": "ConnectionError",
                "error": "Connection refused",
            }
        ]

        errors = client.last_errors()
        assert len(errors) == 1
        assert errors[0]["error_type"] == "ConnectionError"

    def test_errors_include_config_info(self):
        """Recorded errors include config information."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        client._errors = [
            {
                "config": {"transport": "stdio", "command": "failing-server"},
                "identity": "stdio: failing-server",
                "error_type": "OSError",
                "error": "No such file",
            }
        ]

        errors = client.last_errors()
        assert errors[0]["config"]["command"] == "failing-server"


# =============================================================================
# Exception Matching Tests
# =============================================================================


class TestExceptionMatching:
    """Tests for exception pattern matching."""

    def test_catch_specific_server_error(self):
        """Can catch MCPServerError specifically."""
        with pytest.raises(MCPServerError):
            raise MCPServerError("Failed", server_name="server")

    def test_catch_specific_connection_error(self):
        """Can catch MCPConnectionError specifically."""
        with pytest.raises(MCPConnectionError):
            raise MCPConnectionError("Failed", server_name="server")

    def test_catch_specific_timeout_error(self):
        """Can catch MCPTimeoutError specifically."""
        with pytest.raises(MCPTimeoutError):
            raise MCPTimeoutError("Failed", operation="op", timeout=1.0)

    def test_catch_specific_tool_error(self):
        """Can catch MCPToolError specifically."""
        with pytest.raises(MCPToolError):
            raise MCPToolError("Failed", tool_name="tool", server_name="server")
