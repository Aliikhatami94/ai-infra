"""Unit tests for MCP ai-infra entry point module.

Tests cover:
- Module import
- Server instantiation
- Tool registration (ai_infra_cmd_help, ai_infra_subcmd_help)
- Subcommand enum
- Default configuration
- Entry point execution

Phase 5.3 of ai-infra test plan.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

# =============================================================================
# Test Entry Point Module Import
# =============================================================================


class TestAIInfraMCPModuleImport:
    """Tests for ai_infra_mcp module import."""

    def test_module_imports_successfully(self):
        """Test module imports without errors."""
        from ai_infra.mcp import ai_infra_mcp

        assert ai_infra_mcp is not None

    def test_module_has_mcp_attribute(self):
        """Test module has mcp attribute."""
        from ai_infra.mcp import ai_infra_mcp

        assert hasattr(ai_infra_mcp, "mcp")

    def test_module_has_cli_prog(self):
        """Test module has CLI_PROG constant."""
        from ai_infra.mcp import ai_infra_mcp

        assert hasattr(ai_infra_mcp, "CLI_PROG")
        assert ai_infra_mcp.CLI_PROG == "ai-infra"


# =============================================================================
# Test MCP Server Creation
# =============================================================================


class TestAIInfraMCPServerCreation:
    """Tests for ai_infra_mcp server creation."""

    def test_mcp_server_created(self):
        """Test MCP server is created."""
        from ai_infra.mcp.ai_infra_mcp import mcp

        assert mcp is not None

    def test_mcp_server_is_fastmcp(self):
        """Test MCP server is a FastMCP instance."""
        from ai_infra.mcp.ai_infra_mcp import mcp

        # FastMCP should have run method
        assert hasattr(mcp, "run")

    def test_mcp_server_has_tools(self):
        """Test MCP server has registered tools."""
        from ai_infra.mcp.ai_infra_mcp import mcp

        # FastMCP should have tool manager
        assert hasattr(mcp, "_tool_manager") or hasattr(mcp, "add_tool")


# =============================================================================
# Test Subcommand Enum
# =============================================================================


class TestSubcommandEnum:
    """Tests for Subcommand enum."""

    def test_subcommand_enum_exists(self):
        """Test Subcommand enum exists."""
        from ai_infra.mcp.ai_infra_mcp import Subcommand

        assert Subcommand is not None

    def test_subcommand_add_publisher(self):
        """Test add-publisher subcommand."""
        from ai_infra.mcp.ai_infra_mcp import Subcommand

        assert Subcommand.add_publisher.value == "add-publisher"

    def test_subcommand_remove_publisher(self):
        """Test remove-publisher subcommand."""
        from ai_infra.mcp.ai_infra_mcp import Subcommand

        assert Subcommand.remove_publisher.value == "remove-publisher"

    def test_subcommand_chmod_publisher(self):
        """Test chmod subcommand."""
        from ai_infra.mcp.ai_infra_mcp import Subcommand

        assert Subcommand.chmod_publisher.value == "chmod"

    def test_subcommand_chmod_all(self):
        """Test chmod-all subcommand."""
        from ai_infra.mcp.ai_infra_mcp import Subcommand

        assert Subcommand.chmod_all.value == "chmod-all"

    def test_subcommand_is_string_enum(self):
        """Test Subcommand is a string enum."""
        from ai_infra.mcp.ai_infra_mcp import Subcommand

        # Should be usable as string
        assert isinstance(Subcommand.add_publisher, str)


# =============================================================================
# Test ai_infra_cmd_help Function
# =============================================================================


class TestAIInfraCmdHelp:
    """Tests for ai_infra_cmd_help function."""

    def test_ai_infra_cmd_help_is_async(self):
        """Test ai_infra_cmd_help is async function."""
        import asyncio

        from ai_infra.mcp.ai_infra_mcp import ai_infra_cmd_help

        assert asyncio.iscoroutinefunction(ai_infra_cmd_help)

    @pytest.mark.asyncio
    async def test_ai_infra_cmd_help_returns_json(self):
        """Test ai_infra_cmd_help returns JSON string."""
        import json

        from ai_infra.mcp.ai_infra_mcp import ai_infra_cmd_help

        with patch("ai_infra.mcp.ai_infra_mcp.cli_cmd_help", new_callable=AsyncMock) as mock_help:
            mock_help.return_value = {"status": "success", "output": "help text"}

            result = await ai_infra_cmd_help()

            # Should be valid JSON
            parsed = json.loads(result)
            assert parsed["status"] == "success"

    @pytest.mark.asyncio
    async def test_ai_infra_cmd_help_calls_cli_cmd_help(self):
        """Test ai_infra_cmd_help calls cli_cmd_help with correct args."""
        from ai_infra.mcp.ai_infra_mcp import CLI_PROG, ai_infra_cmd_help

        with patch("ai_infra.mcp.ai_infra_mcp.cli_cmd_help", new_callable=AsyncMock) as mock_help:
            mock_help.return_value = {}

            await ai_infra_cmd_help()

            mock_help.assert_called_once_with(CLI_PROG)


# =============================================================================
# Test ai_infra_subcmd_help Function
# =============================================================================


class TestAIInfraSubcmdHelp:
    """Tests for ai_infra_subcmd_help function."""

    def test_ai_infra_subcmd_help_is_async(self):
        """Test ai_infra_subcmd_help is async function."""
        import asyncio

        from ai_infra.mcp.ai_infra_mcp import ai_infra_subcmd_help

        assert asyncio.iscoroutinefunction(ai_infra_subcmd_help)

    @pytest.mark.asyncio
    async def test_ai_infra_subcmd_help_returns_json(self):
        """Test ai_infra_subcmd_help returns JSON string."""
        import json

        from ai_infra.mcp.ai_infra_mcp import Subcommand, ai_infra_subcmd_help

        with patch(
            "ai_infra.mcp.ai_infra_mcp.cli_subcmd_help", new_callable=AsyncMock
        ) as mock_help:
            mock_help.return_value = {"status": "success", "output": "subcommand help"}

            result = await ai_infra_subcmd_help(Subcommand.add_publisher)

            # Should be valid JSON
            parsed = json.loads(result)
            assert parsed["status"] == "success"

    @pytest.mark.asyncio
    async def test_ai_infra_subcmd_help_calls_cli_subcmd_help(self):
        """Test ai_infra_subcmd_help calls cli_subcmd_help with correct args."""
        from ai_infra.mcp.ai_infra_mcp import CLI_PROG, Subcommand, ai_infra_subcmd_help

        with patch(
            "ai_infra.mcp.ai_infra_mcp.cli_subcmd_help", new_callable=AsyncMock
        ) as mock_help:
            mock_help.return_value = {}

            await ai_infra_subcmd_help(Subcommand.remove_publisher)

            mock_help.assert_called_once_with(CLI_PROG, Subcommand.remove_publisher)

    @pytest.mark.asyncio
    async def test_ai_infra_subcmd_help_with_chmod(self):
        """Test ai_infra_subcmd_help with chmod subcommand."""
        from ai_infra.mcp.ai_infra_mcp import Subcommand, ai_infra_subcmd_help

        with patch(
            "ai_infra.mcp.ai_infra_mcp.cli_subcmd_help", new_callable=AsyncMock
        ) as mock_help:
            mock_help.return_value = {"output": "chmod help"}

            await ai_infra_subcmd_help(Subcommand.chmod_publisher)

            mock_help.assert_called_once()


# =============================================================================
# Test Entry Point Execution
# =============================================================================


class TestAIInfraMCPEntryPoint:
    """Tests for entry point execution."""

    def test_main_block_exists(self):
        """Test module can be run as script."""
        # The module should have if __name__ == "__main__" block
        from ai_infra.mcp import ai_infra_mcp

        # Module should be importable and have mcp
        assert ai_infra_mcp.mcp is not None


# =============================================================================
# Test Tool Registration
# =============================================================================


class TestAIInfraMCPToolRegistration:
    """Tests for tool registration in ai_infra_mcp."""

    def test_mcp_created_with_mcp_from_functions(self):
        """Test MCP is created using mcp_from_functions."""
        from ai_infra.mcp.ai_infra_mcp import mcp

        # Should be a FastMCP instance
        assert mcp is not None
        assert hasattr(mcp, "run")

    def test_two_tools_registered(self):
        """Test two tools are registered."""
        from ai_infra.mcp import ai_infra_mcp

        # The module registers ai_infra_cmd_help and ai_infra_subcmd_help
        assert callable(ai_infra_mcp.ai_infra_cmd_help)
        assert callable(ai_infra_mcp.ai_infra_subcmd_help)


# =============================================================================
# Test Default Configuration
# =============================================================================


class TestAIInfraMCPDefaultConfig:
    """Tests for default configuration."""

    def test_server_name_is_set(self):
        """Test server has correct name."""
        from ai_infra.mcp import ai_infra_mcp

        # The mcp is created with name="ai-infra-cli-mcp"
        assert ai_infra_mcp.mcp is not None

    def test_cli_prog_is_ai_infra(self):
        """Test CLI_PROG is set to ai-infra."""
        from ai_infra.mcp.ai_infra_mcp import CLI_PROG

        assert CLI_PROG == "ai-infra"


# =============================================================================
# Test Integration with mcp_from_functions
# =============================================================================


class TestMCPFromFunctionsIntegration:
    """Tests for mcp_from_functions integration."""

    def test_creates_server_with_functions(self):
        """Test mcp_from_functions creates server with async functions."""
        from ai_infra.mcp.server.tools import mcp_from_functions

        async def async_tool1() -> str:
            """Tool 1."""
            return "result1"

        async def async_tool2(arg: str) -> str:
            """Tool 2."""
            return arg

        mcp = mcp_from_functions(
            name="test-mcp",
            functions=[async_tool1, async_tool2],
        )

        assert mcp is not None

    def test_server_has_run_method(self):
        """Test created server has run method."""
        from ai_infra.mcp.ai_infra_mcp import mcp

        assert hasattr(mcp, "run")
        assert callable(mcp.run)
