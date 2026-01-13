"""Tests for ShellMiddleware.

Phase 1.4 of EXECUTOR_CLI.md - Shell Middleware.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.llm.shell.middleware import ShellMiddleware, ShellMiddlewareConfig
from ai_infra.llm.shell.session import ShellSession
from ai_infra.llm.shell.tool import get_current_session, set_current_session
from ai_infra.llm.shell.types import DEFAULT_REDACTION_RULES, RedactionRule, ShellResult


class TestShellMiddlewareConfig:
    """Tests for ShellMiddlewareConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ShellMiddlewareConfig()

        assert config.workspace_root is None
        assert config.startup_commands == []
        assert config.shutdown_commands == []
        assert config.timeout == 120.0
        assert config.max_output_bytes == 1_000_000
        assert config.redaction_rules == DEFAULT_REDACTION_RULES
        assert config.dangerous_pattern_check is True
        assert config.custom_dangerous_patterns is None
        assert config.env is None

    def test_custom_values(self, tmp_path: Path):
        """Test custom configuration values."""
        custom_patterns = (re.compile(r"secret"),)
        custom_rules = (RedactionRule(name="test", pattern=r"test", replacement="***"),)

        config = ShellMiddlewareConfig(
            workspace_root=tmp_path,
            startup_commands=["source .venv/bin/activate"],
            shutdown_commands=["deactivate"],
            timeout=300.0,
            max_output_bytes=500_000,
            redaction_rules=custom_rules,
            dangerous_pattern_check=False,
            custom_dangerous_patterns=custom_patterns,
            env={"MY_VAR": "value"},
        )

        assert config.workspace_root == tmp_path
        assert config.startup_commands == ["source .venv/bin/activate"]
        assert config.shutdown_commands == ["deactivate"]
        assert config.timeout == 300.0
        assert config.max_output_bytes == 500_000
        assert config.redaction_rules == custom_rules
        assert config.dangerous_pattern_check is False
        assert config.custom_dangerous_patterns == custom_patterns
        assert config.env == {"MY_VAR": "value"}


class TestShellMiddlewareInit:
    """Tests for ShellMiddleware initialization."""

    def test_default_init(self):
        """Test default initialization."""
        middleware = ShellMiddleware()

        assert middleware.name == "ShellMiddleware"
        assert middleware.session is None
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "run_shell"

    def test_init_with_kwargs(self, tmp_path: Path):
        """Test initialization with keyword arguments."""
        middleware = ShellMiddleware(
            workspace_root=tmp_path,
            startup_commands=["echo start"],
            timeout=60.0,
        )

        assert middleware._config.workspace_root == tmp_path
        assert middleware._config.startup_commands == ["echo start"]
        assert middleware._config.timeout == 60.0

    def test_init_with_config(self, tmp_path: Path):
        """Test initialization with config object."""
        config = ShellMiddlewareConfig(
            workspace_root=tmp_path,
            timeout=180.0,
        )
        middleware = ShellMiddleware(config=config)

        assert middleware._config == config
        assert middleware._config.workspace_root == tmp_path
        assert middleware._config.timeout == 180.0


class TestShellMiddlewareLifecycle:
    """Tests for middleware lifecycle hooks."""

    @pytest.mark.asyncio
    async def test_abefore_agent_starts_session(self):
        """Test that abefore_agent starts a session."""
        middleware = ShellMiddleware()

        # Create mock runtime
        mock_runtime = MagicMock()
        mock_state = {}

        with patch.object(ShellSession, "start", new_callable=AsyncMock):
            await middleware.abefore_agent(mock_state, mock_runtime)

            assert middleware.session is not None
            assert get_current_session() is middleware.session

        # Cleanup
        set_current_session(None)

    @pytest.mark.asyncio
    async def test_aafter_agent_closes_session(self):
        """Test that aafter_agent closes the session."""
        middleware = ShellMiddleware()
        mock_runtime = MagicMock()
        mock_state = {}

        # Start session first
        with patch.object(ShellSession, "start", new_callable=AsyncMock):
            await middleware.abefore_agent(mock_state, mock_runtime)

        # Now close it
        with patch.object(ShellSession, "close", new_callable=AsyncMock) as mock_close:
            await middleware.aafter_agent(mock_state, mock_runtime)

            mock_close.assert_called_once()
            assert middleware.session is None
            assert get_current_session() is None

    @pytest.mark.asyncio
    async def test_sync_hooks_return_none(self):
        """Test that sync hooks return None (async versions are used)."""
        middleware = ShellMiddleware()
        mock_runtime = MagicMock()
        mock_state = {}

        assert middleware.before_agent(mock_state, mock_runtime) is None
        assert middleware.after_agent(mock_state, mock_runtime) is None


class TestShellMiddlewareTool:
    """Tests for the middleware's shell tool."""

    @pytest.mark.asyncio
    async def test_tool_rejects_dangerous_command(self):
        """Test that the tool rejects dangerous commands."""
        middleware = ShellMiddleware()
        tool = middleware.tools[0]

        result = await tool.ainvoke({"command": "rm -rf /"})

        assert result["success"] is False
        assert "rejected" in result["stderr"].lower()

    @pytest.mark.asyncio
    async def test_tool_with_custom_dangerous_patterns(self):
        """Test tool with custom dangerous patterns."""
        custom_patterns = (re.compile(r"my_dangerous_cmd"),)
        middleware = ShellMiddleware(custom_dangerous_patterns=custom_patterns)
        tool = middleware.tools[0]

        result = await tool.ainvoke({"command": "my_dangerous_cmd"})

        assert result["success"] is False
        assert "rejected" in result["stderr"].lower()

    @pytest.mark.asyncio
    async def test_tool_allows_when_dangerous_check_disabled(self):
        """Test tool allows commands when dangerous check disabled."""
        middleware = ShellMiddleware(dangerous_pattern_check=False)
        tool = middleware.tools[0]

        # This would normally be rejected
        with patch(
            "ai_infra.llm.shell.tool._execute_stateless",
            new_callable=AsyncMock,
        ) as mock_execute:
            mock_execute.return_value = ShellResult(
                success=True,
                exit_code=0,
                stdout="output",
                stderr="",
                command="rm -rf /",
                duration_ms=10.0,
                timed_out=False,
            )

            await tool.ainvoke({"command": "rm -rf /"})

            # Command was allowed through
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_validates_cwd(self):
        """Test tool validates cwd path."""
        middleware = ShellMiddleware()
        tool = middleware.tools[0]

        result = await tool.ainvoke(
            {
                "command": "ls",
                "cwd": "/nonexistent/path/xyz123",
            }
        )

        assert result["success"] is False
        assert "does not exist" in result["stderr"]

    @pytest.mark.asyncio
    async def test_tool_uses_session_when_available(self):
        """Test tool uses session when available."""
        mock_session = MagicMock()
        mock_session.is_running = True
        mock_session.execute = AsyncMock(
            return_value=ShellResult(
                success=True,
                exit_code=0,
                stdout="from session",
                stderr="",
                command="echo hello",
                duration_ms=10.0,
                timed_out=False,
            )
        )

        middleware = ShellMiddleware()
        tool = middleware.tools[0]

        # Set session in context
        set_current_session(mock_session)
        try:
            result = await tool.ainvoke({"command": "echo hello"})

            assert result["success"] is True
            assert result["stdout"] == "from session"
            mock_session.execute.assert_called_once()
        finally:
            set_current_session(None)

    @pytest.mark.asyncio
    async def test_tool_prepends_cd_for_cwd(self):
        """Test tool prepends cd when cwd is specified with session."""
        mock_session = MagicMock()
        mock_session.is_running = True
        mock_session.execute = AsyncMock(
            return_value=ShellResult(
                success=True,
                exit_code=0,
                stdout="output",
                stderr="",
                command="cd /tmp && ls",
                duration_ms=10.0,
                timed_out=False,
            )
        )

        middleware = ShellMiddleware()
        tool = middleware.tools[0]

        set_current_session(mock_session)
        try:
            await tool.ainvoke({"command": "ls", "cwd": "/tmp"})

            # Check that cd was prepended
            call_args = mock_session.execute.call_args
            assert "cd /tmp && ls" in call_args[0][0]
        finally:
            set_current_session(None)

    @pytest.mark.asyncio
    async def test_tool_falls_back_to_stateless(self):
        """Test tool falls back to stateless execution."""
        middleware = ShellMiddleware()
        tool = middleware.tools[0]

        set_current_session(None)

        with patch(
            "ai_infra.llm.shell.tool._execute_stateless",
            new_callable=AsyncMock,
        ) as mock_execute:
            mock_execute.return_value = ShellResult(
                success=True,
                exit_code=0,
                stdout="stateless",
                stderr="",
                command="ls",
                duration_ms=10.0,
                timed_out=False,
            )

            result = await tool.ainvoke({"command": "ls"})

            assert result["success"] is True
            mock_execute.assert_called_once()


class TestShellMiddlewareUtilities:
    """Tests for middleware utility methods."""

    @pytest.mark.asyncio
    async def test_restart_session(self):
        """Test restarting the session."""
        middleware = ShellMiddleware()
        mock_runtime = MagicMock()
        mock_state = {}

        # Start session
        with patch.object(ShellSession, "start", new_callable=AsyncMock):
            await middleware.abefore_agent(mock_state, mock_runtime)

        # Restart
        with patch.object(ShellSession, "restart", new_callable=AsyncMock) as mock_restart:
            await middleware.restart_session()
            mock_restart.assert_called_once()

        # Cleanup
        with patch.object(ShellSession, "close", new_callable=AsyncMock):
            await middleware.aafter_agent(mock_state, mock_runtime)

    @pytest.mark.asyncio
    async def test_execute_requires_session(self):
        """Test that execute raises if session not started."""
        middleware = ShellMiddleware()

        with pytest.raises(RuntimeError, match="Session not started"):
            await middleware.execute("echo hello")

    @pytest.mark.asyncio
    async def test_execute_on_session(self):
        """Test executing on the session directly."""
        middleware = ShellMiddleware()
        mock_runtime = MagicMock()
        mock_state = {}

        # Start session
        with patch.object(ShellSession, "start", new_callable=AsyncMock):
            await middleware.abefore_agent(mock_state, mock_runtime)

        # Execute
        with patch.object(
            middleware._session,
            "execute",
            new_callable=AsyncMock,
            return_value=ShellResult(
                success=True,
                exit_code=0,
                stdout="direct",
                stderr="",
                command="echo test",
                duration_ms=10.0,
                timed_out=False,
            ),
        ) as mock_execute:
            result = await middleware.execute("echo test")

            assert result.success is True
            assert result.stdout == "direct"
            mock_execute.assert_called_once_with("echo test")

        # Cleanup
        with patch.object(ShellSession, "close", new_callable=AsyncMock):
            await middleware.aafter_agent(mock_state, mock_runtime)


class TestShellMiddlewareIntegration:
    """Integration tests with real shell execution."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("sys").platform.startswith("win"),
        reason="Shell tests run on Unix only",
    )
    async def test_full_lifecycle(self, tmp_path: Path):
        """Test full middleware lifecycle with real shell."""
        middleware = ShellMiddleware(
            workspace_root=tmp_path,
            startup_commands=["export TEST_VAR=middleware_test"],
        )

        mock_runtime = MagicMock()
        mock_state = {}

        # Start
        await middleware.abefore_agent(mock_state, mock_runtime)

        try:
            assert middleware.session is not None
            assert middleware.session.is_running

            # Check startup command was executed
            result = await middleware.execute("echo $TEST_VAR")
            assert result.success
            assert "middleware_test" in result.stdout
        finally:
            # End
            await middleware.aafter_agent(mock_state, mock_runtime)
            assert middleware.session is None

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("sys").platform.startswith("win"),
        reason="Shell tests run on Unix only",
    )
    async def test_tool_integration(self, tmp_path: Path):
        """Test tool works within middleware context."""
        middleware = ShellMiddleware(workspace_root=tmp_path)
        tool = middleware.tools[0]

        mock_runtime = MagicMock()
        mock_state = {}

        await middleware.abefore_agent(mock_state, mock_runtime)

        try:
            # Execute via tool
            result = await tool.ainvoke({"command": "echo tool_test"})

            assert result["success"] is True
            assert "tool_test" in result["stdout"]
        finally:
            await middleware.aafter_agent(mock_state, mock_runtime)
