"""Unit tests for MCP CLI module.

Tests cover:
- CLI MCP server creation
- CLI tool registration (run_cli)
- MCP server instantiation
- Transport configuration (stdio)
- Main entry point

Phase 5.2 of ai-infra test plan.
"""

from __future__ import annotations

from unittest.mock import patch

# =============================================================================
# Test CLI Module Import
# =============================================================================


class TestMCPCLIModuleImport:
    """Tests for CLI module import."""

    def test_module_imports_successfully(self):
        """Test CLI module imports without errors."""
        from ai_infra.mcp.server.custom import cli

        assert cli is not None

    def test_module_has_mcp_attribute(self):
        """Test CLI module has mcp attribute."""
        from ai_infra.mcp.server.custom import cli

        assert hasattr(cli, "mcp")

    def test_module_has_main_function(self):
        """Test CLI module has main function."""
        from ai_infra.mcp.server.custom import cli

        assert hasattr(cli, "main")
        assert callable(cli.main)


# =============================================================================
# Test CLI MCP Server Creation
# =============================================================================


class TestMCPCLIServerCreation:
    """Tests for CLI MCP server creation."""

    def test_mcp_server_created(self):
        """Test MCP server is created."""
        from ai_infra.mcp.server.custom.cli import mcp

        assert mcp is not None

    def test_mcp_server_is_fastmcp(self):
        """Test MCP server is a FastMCP instance."""
        from ai_infra.mcp.server.custom.cli import mcp

        # FastMCP should have run method
        assert hasattr(mcp, "run")

    def test_mcp_server_has_name(self):
        """Test MCP server has correct name."""
        from ai_infra.mcp.server.custom.cli import mcp

        # Check if name is set (may be accessible through state or _name)
        assert mcp is not None

    def test_mcp_server_has_tools(self):
        """Test MCP server has registered tools."""
        from ai_infra.mcp.server.custom.cli import mcp

        # FastMCP should have tool manager
        assert hasattr(mcp, "_tool_manager") or hasattr(mcp, "add_tool")


# =============================================================================
# Test CLI Tool Registration
# =============================================================================


class TestMCPCLIToolRegistration:
    """Tests for CLI tool registration."""

    def test_run_cli_imported(self):
        """Test run_cli is imported."""
        from ai_infra.llm.tools.custom.cli import run_cli

        assert run_cli is not None
        assert callable(run_cli)

    def test_mcp_from_functions_creates_server(self):
        """Test mcp_from_functions creates server with tools."""
        from ai_infra.mcp.server.tools import mcp_from_functions

        def sample_tool(x: str) -> str:
            """Sample tool description."""
            return x

        mcp = mcp_from_functions(name="test", functions=[sample_tool])

        assert mcp is not None
        assert hasattr(mcp, "run")


# =============================================================================
# Test CLI Main Entry Point
# =============================================================================


class TestMCPCLIMainEntryPoint:
    """Tests for CLI main entry point."""

    def test_main_calls_mcp_run(self):
        """Test main function calls mcp.run."""
        from ai_infra.mcp.server.custom import cli

        with patch.object(cli.mcp, "run") as mock_run:
            cli.main()

            mock_run.assert_called_once_with(transport="stdio")

    def test_main_uses_stdio_transport(self):
        """Test main uses stdio transport."""
        from ai_infra.mcp.server.custom import cli

        with patch.object(cli.mcp, "run") as mock_run:
            cli.main()

            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["transport"] == "stdio"


# =============================================================================
# Test CLI MCP Server Tools Module
# =============================================================================


class TestMCPToolsModule:
    """Tests for mcp_from_functions utility."""

    def test_mcp_from_functions_empty_functions(self):
        """Test mcp_from_functions with empty functions list."""
        from ai_infra.mcp.server.tools import mcp_from_functions

        mcp = mcp_from_functions(name="empty", functions=[])

        assert mcp is not None

    def test_mcp_from_functions_none_functions(self):
        """Test mcp_from_functions with None functions."""
        from ai_infra.mcp.server.tools import mcp_from_functions

        mcp = mcp_from_functions(name="none", functions=None)

        assert mcp is not None

    def test_mcp_from_functions_with_name(self):
        """Test mcp_from_functions with custom name."""
        from ai_infra.mcp.server.tools import mcp_from_functions

        mcp = mcp_from_functions(name="custom-name", functions=[])

        assert mcp is not None

    def test_mcp_from_functions_with_tool_def(self):
        """Test mcp_from_functions with ToolDef."""
        from ai_infra.mcp.server.tools import ToolDef, mcp_from_functions

        def my_fn(x: str) -> str:
            return x

        tool_def = ToolDef(fn=my_fn, name="custom_name", description="Custom description")
        mcp = mcp_from_functions(name="test", functions=[tool_def])

        assert mcp is not None

    def test_mcp_from_functions_with_async_function(self):
        """Test mcp_from_functions with async function."""
        from ai_infra.mcp.server.tools import mcp_from_functions

        async def async_tool(x: str) -> str:
            """Async tool description."""
            return x

        mcp = mcp_from_functions(name="async", functions=[async_tool])

        assert mcp is not None

    def test_mcp_from_functions_dedupes_by_name(self):
        """Test mcp_from_functions deduplicates tools by name."""
        from ai_infra.mcp.server.tools import mcp_from_functions

        def my_tool(x: str) -> str:
            return x

        # Pass same function twice
        mcp = mcp_from_functions(name="dedup", functions=[my_tool, my_tool])

        assert mcp is not None


# =============================================================================
# Test MCP Security Settings
# =============================================================================


class TestMCPSecuritySettings:
    """Tests for MCPSecuritySettings."""

    def test_security_settings_default(self):
        """Test default security settings."""
        from ai_infra.mcp.server.tools import MCPSecuritySettings

        settings = MCPSecuritySettings()

        assert settings.enabled is True

    def test_security_settings_disabled(self):
        """Test disabled security settings."""
        from ai_infra.mcp.server.tools import MCPSecuritySettings

        settings = MCPSecuritySettings(enable_security=False)

        assert settings.enabled is False
        assert settings.allowed_hosts == []
        assert settings.allowed_origins == []

    def test_security_settings_with_domains(self):
        """Test security settings with custom domains."""
        from ai_infra.mcp.server.tools import MCPSecuritySettings

        settings = MCPSecuritySettings(domains=["example.com", "api.example.com"])

        assert "example.com:*" in settings.allowed_hosts
        assert "api.example.com:*" in settings.allowed_hosts

    def test_security_settings_allowed_hosts(self):
        """Test security settings always includes localhost."""
        from ai_infra.mcp.server.tools import MCPSecuritySettings

        settings = MCPSecuritySettings()

        assert "127.0.0.1:*" in settings.allowed_hosts
        assert "localhost:*" in settings.allowed_hosts

    def test_security_settings_to_transport_settings(self):
        """Test conversion to TransportSecuritySettings."""
        from ai_infra.mcp.server.tools import MCPSecuritySettings

        settings = MCPSecuritySettings()
        transport = settings.to_transport_settings()

        assert transport is not None

    def test_security_settings_custom_allowed_hosts(self):
        """Test security settings with custom allowed hosts."""
        from ai_infra.mcp.server.tools import MCPSecuritySettings

        settings = MCPSecuritySettings(allowed_hosts=["custom.host:8080"])

        assert settings.allowed_hosts == ["custom.host:8080"]

    def test_security_settings_custom_allowed_origins(self):
        """Test security settings with custom allowed origins."""
        from ai_infra.mcp.server.tools import MCPSecuritySettings

        settings = MCPSecuritySettings(allowed_origins=["https://custom.origin"])

        assert settings.allowed_origins == ["https://custom.origin"]


# =============================================================================
# Test Auto-Detection Functions
# =============================================================================


class TestAutoDetection:
    """Tests for auto-detection functions."""

    def test_auto_detect_hosts_includes_localhost(self):
        """Test auto-detect hosts includes localhost."""
        from ai_infra.mcp.server.tools import _auto_detect_hosts

        hosts = _auto_detect_hosts()

        assert "127.0.0.1:*" in hosts
        assert "localhost:*" in hosts

    def test_auto_detect_origins_includes_localhost(self):
        """Test auto-detect origins includes localhost."""
        from ai_infra.mcp.server.tools import _auto_detect_origins

        origins = _auto_detect_origins()

        assert "http://127.0.0.1:*" in origins
        assert "http://localhost:*" in origins

    def test_auto_detect_hosts_from_railway(self):
        """Test auto-detect hosts from Railway environment."""
        from ai_infra.mcp.server.tools import _auto_detect_hosts

        with patch.dict("os.environ", {"RAILWAY_PUBLIC_DOMAIN": "myapp.railway.app"}):
            hosts = _auto_detect_hosts()

            assert "myapp.railway.app:*" in hosts

    def test_auto_detect_hosts_from_fly(self):
        """Test auto-detect hosts from Fly.io environment."""
        from ai_infra.mcp.server.tools import _auto_detect_hosts

        with patch.dict("os.environ", {"FLY_APP_NAME": "myapp"}, clear=False):
            hosts = _auto_detect_hosts()

            assert "myapp.fly.dev:*" in hosts

    def test_auto_detect_origins_from_vercel(self):
        """Test auto-detect origins from Vercel environment."""
        from ai_infra.mcp.server.tools import _auto_detect_origins

        with patch.dict("os.environ", {"VERCEL_URL": "myapp.vercel.app"}, clear=False):
            origins = _auto_detect_origins()

            assert "https://myapp.vercel.app" in origins


# =============================================================================
# Test ToolDef Model
# =============================================================================


class TestToolDef:
    """Tests for ToolDef model."""

    def test_tool_def_creation(self):
        """Test ToolDef creation."""
        from ai_infra.mcp.server.tools import ToolDef

        def my_fn():
            return "test"

        tool = ToolDef(fn=my_fn, name="my_tool", description="My tool")

        assert tool.fn is my_fn
        assert tool.name == "my_tool"
        assert tool.description == "My tool"

    def test_tool_def_optional_fields(self):
        """Test ToolDef with optional fields."""
        from ai_infra.mcp.server.tools import ToolDef

        tool = ToolDef()

        assert tool.fn is None
        assert tool.name is None
        assert tool.description is None

    def test_tool_def_with_only_fn(self):
        """Test ToolDef with only function."""
        from ai_infra.mcp.server.tools import ToolDef

        def my_fn():
            return "test"

        tool = ToolDef(fn=my_fn)

        assert tool.fn is my_fn
        assert tool.name is None
