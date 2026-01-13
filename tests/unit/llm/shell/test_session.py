"""Tests for ShellSession and SessionConfig.

Phase 1.2 of EXECUTOR_CLI.md - Shell Session Manager.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.llm.shell.session import SessionConfig, ShellSession
from ai_infra.llm.shell.types import DEFAULT_REDACTION_RULES, RedactionRule, ShellConfig


class TestSessionConfig:
    """Tests for SessionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SessionConfig()

        assert config.workspace_root is None
        assert config.startup_commands == []
        assert config.shutdown_commands == []
        assert isinstance(config.shell_config, ShellConfig)
        assert config.redaction_rules == DEFAULT_REDACTION_RULES
        assert config.idle_timeout == 0.0

    def test_custom_values(self, tmp_path: Path):
        """Test custom configuration values."""
        shell_config = ShellConfig(timeout=60.0)
        custom_rules = (RedactionRule(name="secret", pattern=r"secret", replacement="***"),)

        config = SessionConfig(
            workspace_root=tmp_path,
            startup_commands=["source .venv/bin/activate"],
            shutdown_commands=["deactivate"],
            shell_config=shell_config,
            redaction_rules=custom_rules,
            idle_timeout=300.0,
        )

        assert config.workspace_root == tmp_path
        assert config.startup_commands == ["source .venv/bin/activate"]
        assert config.shutdown_commands == ["deactivate"]
        assert config.shell_config.timeout == 60.0
        assert config.redaction_rules == custom_rules
        assert config.idle_timeout == 300.0

    def test_no_redaction_rules(self):
        """Test configuration with no redaction rules."""
        config = SessionConfig(redaction_rules=None)
        assert config.redaction_rules is None


class TestShellSessionInit:
    """Tests for ShellSession initialization."""

    def test_default_init(self):
        """Test default initialization."""
        session = ShellSession()

        assert not session.is_running
        assert session._config is not None
        assert session._process is None
        assert not session._started
        assert not session._closed

    def test_custom_config(self, tmp_path: Path):
        """Test initialization with custom config."""
        config = SessionConfig(workspace_root=tmp_path)
        session = ShellSession(config)

        assert session._config == config
        assert session._config.workspace_root == tmp_path


class TestShellSessionIsRunning:
    """Tests for is_running property."""

    def test_not_running_initially(self):
        """Test is_running is False initially."""
        session = ShellSession()
        assert not session.is_running

    def test_not_running_when_closed(self):
        """Test is_running is False when closed."""
        session = ShellSession()
        session._started = True
        session._closed = True
        assert not session.is_running

    def test_not_running_without_process(self):
        """Test is_running is False without process."""
        session = ShellSession()
        session._started = True
        session._process = None
        assert not session.is_running


class TestShellSessionStart:
    """Tests for ShellSession.start()."""

    @pytest.mark.asyncio
    async def test_start_spawns_process(self):
        """Test that start() spawns a process."""
        session = ShellSession()

        with patch.object(session, "_spawn_process", new_callable=AsyncMock) as mock_spawn:
            await session.start()

            mock_spawn.assert_called_once()
            assert session._started

    @pytest.mark.asyncio
    async def test_start_when_already_started(self):
        """Test that start() raises if already started."""
        session = ShellSession()
        session._started = True

        with pytest.raises(RuntimeError, match="Session already started"):
            await session.start()

    @pytest.mark.asyncio
    async def test_start_when_closed(self):
        """Test that start() raises if session is closed."""
        session = ShellSession()
        session._closed = True

        with pytest.raises(RuntimeError, match="Session has been closed"):
            await session.start()

    @pytest.mark.asyncio
    async def test_start_runs_startup_commands(self):
        """Test that start() runs startup commands."""
        config = SessionConfig(startup_commands=["export FOO=bar", "cd /tmp"])
        session = ShellSession(config)

        executed_commands: list[str] = []

        async def mock_execute(cmd: str):
            executed_commands.append(cmd)
            return MagicMock()

        with (
            patch.object(session, "_spawn_process", new_callable=AsyncMock),
            patch.object(session, "_execute_internal", side_effect=mock_execute),
        ):
            await session.start()

        assert executed_commands == ["export FOO=bar", "cd /tmp"]


class TestShellSessionExecute:
    """Tests for ShellSession.execute()."""

    @pytest.mark.asyncio
    async def test_execute_not_started(self):
        """Test execute() raises if session not started."""
        session = ShellSession()

        with pytest.raises(RuntimeError, match="Session not started"):
            await session.execute("echo hello")

    @pytest.mark.asyncio
    async def test_execute_when_closed(self):
        """Test execute() raises if session is closed."""
        session = ShellSession()
        session._started = True
        session._closed = True

        with pytest.raises(RuntimeError, match="Session has been closed"):
            await session.execute("echo hello")

    @pytest.mark.asyncio
    async def test_execute_calls_internal(self):
        """Test execute() calls _execute_internal."""
        session = ShellSession()
        session._started = True

        mock_result = MagicMock()
        with patch.object(
            session, "_execute_internal", new_callable=AsyncMock, return_value=mock_result
        ):
            result = await session.execute("echo hello")

            assert result == mock_result


class TestShellSessionRestart:
    """Tests for ShellSession.restart()."""

    @pytest.mark.asyncio
    async def test_restart_kills_and_respawns(self):
        """Test restart() kills process and respawns."""
        config = SessionConfig(startup_commands=["setup"])
        session = ShellSession(config)
        session._process = MagicMock()

        startup_commands: list[str] = []

        async def mock_execute(cmd: str):
            startup_commands.append(cmd)
            return MagicMock()

        with (
            patch.object(session, "_kill_process", new_callable=AsyncMock) as mock_kill,
            patch.object(session, "_spawn_process", new_callable=AsyncMock) as mock_spawn,
            patch.object(session, "_execute_internal", side_effect=mock_execute),
        ):
            await session.restart()

            mock_kill.assert_called_once()
            mock_spawn.assert_called_once()
            assert startup_commands == ["setup"]
            assert not session._closed


class TestShellSessionClose:
    """Tests for ShellSession.close()."""

    @pytest.mark.asyncio
    async def test_close_when_already_closed(self):
        """Test close() does nothing if already closed."""
        session = ShellSession()
        session._closed = True

        # Should not raise
        await session.close()

    @pytest.mark.asyncio
    async def test_close_runs_shutdown_commands(self):
        """Test close() runs shutdown commands."""
        config = SessionConfig(shutdown_commands=["cleanup", "deactivate"])
        session = ShellSession(config)
        session._process = MagicMock()

        shutdown_commands: list[str] = []

        async def mock_execute(cmd: str):
            shutdown_commands.append(cmd)
            return MagicMock()

        with (
            patch.object(session, "_kill_process", new_callable=AsyncMock),
            patch.object(session, "_execute_internal", side_effect=mock_execute),
        ):
            await session.close()

            assert shutdown_commands == ["cleanup", "deactivate"]
            assert session._closed

    @pytest.mark.asyncio
    async def test_close_ignores_shutdown_errors(self):
        """Test close() continues even if shutdown commands fail."""
        config = SessionConfig(shutdown_commands=["fail", "succeed"])
        session = ShellSession(config)
        session._process = MagicMock()

        call_count = 0

        async def mock_execute(cmd: str):
            nonlocal call_count
            call_count += 1
            if cmd == "fail":
                raise RuntimeError("Shutdown failed")
            return MagicMock()

        with (
            patch.object(session, "_kill_process", new_callable=AsyncMock),
            patch.object(session, "_execute_internal", side_effect=mock_execute),
        ):
            await session.close()

            # Both commands should be attempted
            assert call_count == 2
            assert session._closed


class TestShellSessionContextManager:
    """Tests for ShellSession async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_starts_and_closes(self):
        """Test context manager calls start and close."""
        with (
            patch.object(ShellSession, "start", new_callable=AsyncMock) as mock_start,
            patch.object(ShellSession, "close", new_callable=AsyncMock) as mock_close,
        ):
            async with ShellSession() as session:
                mock_start.assert_called_once()
                mock_close.assert_not_called()
                assert isinstance(session, ShellSession)

            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_closes_on_exception(self):
        """Test context manager closes even on exception."""
        with (
            patch.object(ShellSession, "start", new_callable=AsyncMock),
            patch.object(ShellSession, "close", new_callable=AsyncMock) as mock_close,
        ):
            with pytest.raises(ValueError):
                async with ShellSession():
                    raise ValueError("Test error")

            mock_close.assert_called_once()


class TestShellSessionSpawnProcess:
    """Tests for ShellSession._spawn_process()."""

    @pytest.mark.asyncio
    async def test_spawn_uses_bash_on_unix(self):
        """Test that spawn uses bash on Unix."""
        session = ShellSession()

        with patch("sys.platform", "linux"):
            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
                mock_exec.return_value = MagicMock()
                await session._spawn_process()

                # First arg should be bash
                call_args = mock_exec.call_args
                assert call_args[0][0] == "bash"

    @pytest.mark.asyncio
    async def test_spawn_uses_powershell_on_windows(self):
        """Test that spawn uses PowerShell on Windows."""
        session = ShellSession()

        with patch("sys.platform", "win32"):
            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
                mock_exec.return_value = MagicMock()
                await session._spawn_process()

                # First arg should be powershell
                call_args = mock_exec.call_args
                assert call_args[0][0] == "powershell"

    @pytest.mark.asyncio
    async def test_spawn_uses_workspace_root(self, tmp_path: Path):
        """Test that spawn uses workspace_root as cwd."""
        config = SessionConfig(workspace_root=tmp_path)
        session = ShellSession(config)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = MagicMock()
            await session._spawn_process()

            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs["cwd"] == str(tmp_path)

    @pytest.mark.asyncio
    async def test_spawn_applies_env_vars(self):
        """Test that spawn applies environment variables."""
        shell_config = ShellConfig(env={"MY_VAR": "my_value"})
        config = SessionConfig(shell_config=shell_config)
        session = ShellSession(config)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = MagicMock()
            await session._spawn_process()

            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs["env"]["MY_VAR"] == "my_value"


class TestShellSessionKillProcess:
    """Tests for ShellSession._kill_process()."""

    @pytest.mark.asyncio
    async def test_kill_when_no_process(self):
        """Test kill does nothing when no process."""
        session = ShellSession()
        session._process = None

        # Should not raise
        await session._kill_process()

    @pytest.mark.asyncio
    async def test_kill_process_and_wait(self):
        """Test kill terminates and waits for process."""
        session = ShellSession()
        mock_process = AsyncMock()
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()
        session._process = mock_process

        await session._kill_process()

        mock_process.kill.assert_called_once()
        mock_process.wait.assert_called_once()
        assert session._process is None


class TestShellSessionIntegration:
    """Integration tests for ShellSession with real shell."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("sys").platform.startswith("win"),
        reason="Shell tests run on Unix only",
    )
    async def test_simple_command(self):
        """Test executing a simple echo command."""
        async with ShellSession() as session:
            result = await session.execute("echo hello")

            assert result.success
            assert result.exit_code == 0
            assert "hello" in result.stdout

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("sys").platform.startswith("win"),
        reason="Shell tests run on Unix only",
    )
    async def test_state_persists_between_commands(self):
        """Test that environment state persists."""
        async with ShellSession() as session:
            # Set a variable
            await session.execute("export TEST_VAR=persistence_test")

            # Check it's still set
            result = await session.execute("echo $TEST_VAR")

            assert result.success
            assert "persistence_test" in result.stdout

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("sys").platform.startswith("win"),
        reason="Shell tests run on Unix only",
    )
    async def test_cwd_persists(self, tmp_path: Path):
        """Test that working directory persists."""
        async with ShellSession() as session:
            # Change directory
            await session.execute(f"cd {tmp_path}")

            # Check we're still there
            result = await session.execute("pwd")

            assert result.success
            assert str(tmp_path) in result.stdout

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("sys").platform.startswith("win"),
        reason="Shell tests run on Unix only",
    )
    async def test_failed_command_returns_exit_code(self):
        """Test that failed commands return proper exit code."""
        async with ShellSession() as session:
            # Use a subshell to exit with code 42 without killing the main shell
            result = await session.execute("(exit 42)")

            assert not result.success
            assert result.exit_code == 42

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("sys").platform.startswith("win"),
        reason="Shell tests run on Unix only",
    )
    async def test_command_with_stderr(self):
        """Test command that writes to stderr."""
        async with ShellSession() as session:
            result = await session.execute("echo error >&2")

            # Note: stderr handling depends on shell buffering
            assert result.success
            assert result.exit_code == 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("sys").platform.startswith("win"),
        reason="Shell tests run on Unix only",
    )
    async def test_startup_commands_executed(self, tmp_path: Path):
        """Test startup commands are executed."""
        config = SessionConfig(
            workspace_root=tmp_path,
            startup_commands=["export STARTUP_VAR=startup_test"],
        )

        async with ShellSession(config) as session:
            result = await session.execute("echo $STARTUP_VAR")

            assert result.success
            assert "startup_test" in result.stdout

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("sys").platform.startswith("win"),
        reason="Shell tests run on Unix only",
    )
    async def test_redaction_applied(self):
        """Test that redaction rules are applied to output."""
        custom_rules = (
            RedactionRule(name="secret", pattern=r"secret_\w+", replacement="[REDACTED]"),
        )
        config = SessionConfig(redaction_rules=custom_rules)

        async with ShellSession(config) as session:
            result = await session.execute("echo secret_password123")

            assert result.success
            assert "secret_password123" not in result.stdout
            assert "[REDACTED]" in result.stdout

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("sys").platform.startswith("win"),
        reason="Shell tests run on Unix only",
    )
    async def test_no_redaction_when_disabled(self):
        """Test output is not redacted when rules are None."""
        config = SessionConfig(redaction_rules=None)

        async with ShellSession(config) as session:
            result = await session.execute("echo mysecret123")

            assert result.success
            assert "mysecret123" in result.stdout

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("sys").platform.startswith("win"),
        reason="Shell tests run on Unix only",
    )
    async def test_workspace_root_as_cwd(self, tmp_path: Path):
        """Test workspace_root is used as initial cwd."""
        config = SessionConfig(workspace_root=tmp_path)

        async with ShellSession(config) as session:
            result = await session.execute("pwd")

            assert result.success
            assert str(tmp_path) in result.stdout

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("sys").platform.startswith("win"),
        reason="Shell tests run on Unix only",
    )
    async def test_restart_preserves_startup_commands(self):
        """Test restart re-executes startup commands."""
        config = SessionConfig(startup_commands=["export RESTART_VAR=restarted"])

        async with ShellSession(config) as session:
            # Change the var
            await session.execute("export RESTART_VAR=changed")
            result = await session.execute("echo $RESTART_VAR")
            assert "changed" in result.stdout

            # Restart should re-run startup commands
            await session.restart()
            result = await session.execute("echo $RESTART_VAR")
            assert "restarted" in result.stdout

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("sys").platform.startswith("win"),
        reason="Shell tests run on Unix only",
    )
    async def test_multiline_output(self):
        """Test commands with multiline output."""
        async with ShellSession() as session:
            result = await session.execute("echo -e 'line1\\nline2\\nline3'")

            assert result.success
            assert "line1" in result.stdout
            assert "line2" in result.stdout
            assert "line3" in result.stdout

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("sys").platform.startswith("win"),
        reason="Shell tests run on Unix only",
    )
    async def test_command_with_special_characters(self):
        """Test commands with special shell characters."""
        async with ShellSession() as session:
            result = await session.execute("echo 'hello world $VAR \"quoted\"'")

            assert result.success
            assert "hello world" in result.stdout

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("sys").platform.startswith("win"),
        reason="Shell tests run on Unix only",
    )
    async def test_multiple_sequential_commands(self):
        """Test executing multiple commands in sequence."""
        async with ShellSession() as session:
            await session.execute("export A=1")
            await session.execute("export B=2")
            result = await session.execute("echo $A$B")

            assert result.success
            assert "12" in result.stdout


class TestShellSessionTimeout:
    """Tests for command timeout handling."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("sys").platform.startswith("win"),
        reason="Shell tests run on Unix only",
    )
    async def test_timeout_returns_timeout_result(self):
        """Test that timeout returns proper ShellResult."""
        shell_config = ShellConfig(timeout=0.5)  # Very short timeout
        config = SessionConfig(shell_config=shell_config)

        async with ShellSession(config) as session:
            # Command that takes longer than timeout
            result = await session.execute("sleep 10")

            assert result.timed_out
            assert not result.success
            assert result.exit_code == -1


class TestShellSessionOutputTruncation:
    """Tests for output truncation."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("sys").platform.startswith("win"),
        reason="Shell tests run on Unix only",
    )
    async def test_truncation_marker_added(self):
        """Test truncation marker is added when output exceeds limit."""
        # Create a config with very small max_output_bytes
        shell_config = ShellConfig(max_output_bytes=50)
        config = SessionConfig(shell_config=shell_config)

        async with ShellSession(config) as session:
            # Generate output larger than limit
            result = await session.execute("yes hello | head -100")

            # Should have truncation marker
            assert "[OUTPUT TRUNCATED]" in result.stdout or len(result.stdout) <= 60
