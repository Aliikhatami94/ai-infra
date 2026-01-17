"""Tests for shell helper functions.

Phase 2.1: CLI helper functions migrated from old cli.py module.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ai_infra.llm.shell.helpers import (
    check_command_exists,
    cli_cmd_help,
    cli_help,
    cli_subcmd_help,
    run_command,
    run_command_sync,
)

# =============================================================================
# cli_cmd_help Tests
# =============================================================================


class TestCliCmdHelp:
    """Tests for cli_cmd_help function."""

    @pytest.mark.asyncio
    async def test_cli_cmd_help_success(self) -> None:
        """Test cli_cmd_help returns help text on success."""
        with patch("ai_infra.llm.shell.helpers.run_shell") as mock_shell:
            mock_shell.ainvoke = AsyncMock(
                return_value={
                    "success": True,
                    "stdout": "Usage: poetry [options] command",
                    "stderr": "",
                }
            )

            result = await cli_cmd_help("poetry")

            assert result["ok"] is True
            assert "Usage: poetry" in result["help"]
            assert result["error"] == ""
            mock_shell.ainvoke.assert_called_once_with({"command": "poetry --help"})

    @pytest.mark.asyncio
    async def test_cli_cmd_help_failure(self) -> None:
        """Test cli_cmd_help returns error on failure."""
        with patch("ai_infra.llm.shell.helpers.run_shell") as mock_shell:
            mock_shell.ainvoke = AsyncMock(
                return_value={
                    "success": False,
                    "stdout": "",
                    "stderr": "command not found: nonexistent",
                }
            )

            result = await cli_cmd_help("nonexistent")

            assert result["ok"] is False
            assert result["help"] == ""
            assert "not found" in result["error"]


# =============================================================================
# cli_subcmd_help Tests
# =============================================================================


class TestCliSubcmdHelp:
    """Tests for cli_subcmd_help function."""

    @pytest.mark.asyncio
    async def test_cli_subcmd_help_success(self) -> None:
        """Test cli_subcmd_help returns help text on success."""
        with patch("ai_infra.llm.shell.helpers.run_shell") as mock_shell:
            mock_shell.ainvoke = AsyncMock(
                return_value={
                    "success": True,
                    "stdout": "Add a new dependency to pyproject.toml",
                    "stderr": "",
                }
            )

            result = await cli_subcmd_help("poetry", "add")

            assert result["ok"] is True
            assert "dependency" in result["help"]
            mock_shell.ainvoke.assert_called_once_with({"command": "poetry add --help"})

    @pytest.mark.asyncio
    async def test_cli_subcmd_help_with_enum(self) -> None:
        """Test cli_subcmd_help handles enum values."""
        from enum import Enum

        class TestSubcommand(str, Enum):
            add = "add"
            install = "install"

        with patch("ai_infra.llm.shell.helpers.run_shell") as mock_shell:
            mock_shell.ainvoke = AsyncMock(
                return_value={
                    "success": True,
                    "stdout": "Install dependencies",
                    "stderr": "",
                }
            )

            result = await cli_subcmd_help("poetry", TestSubcommand.install)

            assert result["ok"] is True
            mock_shell.ainvoke.assert_called_once_with({"command": "poetry install --help"})

    @pytest.mark.asyncio
    async def test_cli_subcmd_help_failure(self) -> None:
        """Test cli_subcmd_help returns error on failure."""
        with patch("ai_infra.llm.shell.helpers.run_shell") as mock_shell:
            mock_shell.ainvoke = AsyncMock(
                return_value={
                    "success": False,
                    "stdout": "",
                    "stderr": "Error: No such command 'invalid'",
                }
            )

            result = await cli_subcmd_help("poetry", "invalid")

            assert result["ok"] is False
            assert "No such command" in result["error"]


# =============================================================================
# check_command_exists Tests
# =============================================================================


class TestCheckCommandExists:
    """Tests for check_command_exists function."""

    @pytest.mark.asyncio
    async def test_command_exists(self) -> None:
        """Test check_command_exists returns True when command exists."""
        with patch("ai_infra.llm.shell.helpers.run_shell") as mock_shell:
            mock_shell.ainvoke = AsyncMock(
                return_value={
                    "success": True,
                    "stdout": "/usr/bin/python",
                    "stderr": "",
                }
            )

            result = await check_command_exists("python")

            assert result is True
            mock_shell.ainvoke.assert_called_once_with({"command": "which python"})

    @pytest.mark.asyncio
    async def test_command_not_exists(self) -> None:
        """Test check_command_exists returns False when command doesn't exist."""
        with patch("ai_infra.llm.shell.helpers.run_shell") as mock_shell:
            mock_shell.ainvoke = AsyncMock(
                return_value={
                    "success": False,
                    "stdout": "",
                    "stderr": "",
                }
            )

            result = await check_command_exists("nonexistent_command_xyz")

            assert result is False


# =============================================================================
# cli_help Tests
# =============================================================================


class TestCliHelp:
    """Tests for cli_help function."""

    @pytest.mark.asyncio
    async def test_cli_help_program_only(self) -> None:
        """Test cli_help with program only."""
        with patch("ai_infra.llm.shell.helpers.run_shell") as mock_shell:
            mock_shell.ainvoke = AsyncMock(
                return_value={
                    "success": True,
                    "stdout": "Usage: npm <command>",
                    "stderr": "",
                }
            )

            result = await cli_help("npm")

            assert result["ok"] is True
            assert result["program"] == "npm"
            assert result["subcommand"] is None
            assert "npm" in result["help"]

    @pytest.mark.asyncio
    async def test_cli_help_with_subcommand(self) -> None:
        """Test cli_help with subcommand."""
        with patch("ai_infra.llm.shell.helpers.run_shell") as mock_shell:
            mock_shell.ainvoke = AsyncMock(
                return_value={
                    "success": True,
                    "stdout": "Install a package",
                    "stderr": "",
                }
            )

            result = await cli_help("npm", "install")

            assert result["ok"] is True
            assert result["program"] == "npm"
            assert result["subcommand"] == "install"
            mock_shell.ainvoke.assert_called_once()
            call_args = mock_shell.ainvoke.call_args[0][0]
            assert call_args["command"] == "npm install --help"


# =============================================================================
# run_command Tests
# =============================================================================


class TestRunCommand:
    """Tests for run_command function."""

    @pytest.mark.asyncio
    async def test_run_command_success(self) -> None:
        """Test run_command returns result on success."""
        with patch("ai_infra.llm.shell.helpers.run_shell") as mock_shell:
            mock_shell.ainvoke = AsyncMock(
                return_value={
                    "success": True,
                    "exit_code": 0,
                    "stdout": "file1.txt\nfile2.txt",
                    "stderr": "",
                    "command": "ls",
                }
            )

            result = await run_command("ls")

            assert result["success"] is True
            assert result["exit_code"] == 0
            assert "file1.txt" in result["stdout"]

    @pytest.mark.asyncio
    async def test_run_command_failure(self) -> None:
        """Test run_command returns result on failure."""
        with patch("ai_infra.llm.shell.helpers.run_shell") as mock_shell:
            mock_shell.ainvoke = AsyncMock(
                return_value={
                    "success": False,
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": "Error: file not found",
                    "command": "cat nonexistent",
                }
            )

            result = await run_command("cat nonexistent")

            assert result["success"] is False
            assert result["exit_code"] == 1

    @pytest.mark.asyncio
    async def test_run_command_raise_on_error(self) -> None:
        """Test run_command raises when raise_on_error=True."""
        with patch("ai_infra.llm.shell.helpers.run_shell") as mock_shell:
            mock_shell.ainvoke = AsyncMock(
                return_value={
                    "success": False,
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": "Error",
                    "command": "false",
                }
            )

            with pytest.raises(RuntimeError) as exc_info:
                await run_command("false", raise_on_error=True)

            assert "failed with code 1" in str(exc_info.value)


# =============================================================================
# run_command_sync Tests
# =============================================================================


class TestRunCommandSync:
    """Tests for run_command_sync function."""

    def test_run_command_sync_returns_result(self) -> None:
        """Test run_command_sync returns result."""
        # This test actually runs a command
        result = run_command_sync("echo hello")

        assert result["success"] is True
        assert "hello" in result["stdout"]

    def test_run_command_sync_failure(self) -> None:
        """Test run_command_sync on failure."""
        result = run_command_sync("false")

        assert result["success"] is False

    def test_run_command_sync_raise_on_error(self) -> None:
        """Test run_command_sync raises when raise_on_error=True."""
        with pytest.raises(RuntimeError):
            run_command_sync("false", raise_on_error=True)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for shell helpers."""

    @pytest.mark.asyncio
    async def test_cli_cmd_help_real(self) -> None:
        """Test cli_cmd_help with a real command (if available)."""
        # Skip if echo is not available (unlikely)
        result = await cli_cmd_help("echo")
        # echo --help behavior varies, just check we got a result
        assert "ok" in result
        assert "help" in result

    @pytest.mark.asyncio
    async def test_check_command_exists_real(self) -> None:
        """Test check_command_exists with real commands."""
        # echo should exist on all platforms
        exists = await check_command_exists("echo")
        assert exists is True

        # This command should not exist
        not_exists = await check_command_exists("nonexistent_command_xyz_123")
        assert not_exists is False
