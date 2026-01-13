"""Tests for Docker execution policy module.

Phase 4.3 of EXECUTOR_CLI.md - Docker Execution Policy.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.llm.shell.docker import (
    DockerConfig,
    DockerExecutionPolicy,
    DockerSession,
    VolumeMount,
    create_docker_policy,
    create_docker_session,
    is_docker_available,
)
from ai_infra.llm.shell.types import ShellConfig

if TYPE_CHECKING:
    pass


# =============================================================================
# Test VolumeMount
# =============================================================================


class TestVolumeMount:
    """Tests for VolumeMount dataclass."""

    def test_basic_mount(self) -> None:
        """Basic mount should store paths correctly."""
        mount = VolumeMount("/host/path", "/container/path")
        assert mount.host_path == "/host/path"
        assert mount.container_path == "/container/path"
        assert mount.read_only is False

    def test_readonly_mount(self) -> None:
        """Read-only mount should set read_only flag."""
        mount = VolumeMount("/host", "/container", read_only=True)
        assert mount.read_only is True

    def test_to_docker_arg_rw(self) -> None:
        """to_docker_arg should format read-write mount correctly."""
        mount = VolumeMount("/tmp/test", "/workspace", read_only=False)
        result = mount.to_docker_arg()
        # Path is resolved to absolute
        assert ":/workspace:rw" in result
        assert "/tmp/test" in result or "test" in result

    def test_to_docker_arg_ro(self) -> None:
        """to_docker_arg should format read-only mount correctly."""
        mount = VolumeMount("/tmp/test", "/data", read_only=True)
        result = mount.to_docker_arg()
        assert ":ro" in result

    def test_empty_host_path_raises(self) -> None:
        """Empty host path should raise ValueError."""
        with pytest.raises(ValueError, match="host_path cannot be empty"):
            VolumeMount("", "/container")

    def test_empty_container_path_raises(self) -> None:
        """Empty container path should raise ValueError."""
        with pytest.raises(ValueError, match="container_path cannot be empty"):
            VolumeMount("/host", "")

    def test_relative_container_path_raises(self) -> None:
        """Relative container path should raise ValueError."""
        with pytest.raises(ValueError, match="container_path must be absolute"):
            VolumeMount("/host", "relative/path")


# =============================================================================
# Test DockerConfig
# =============================================================================


class TestDockerConfig:
    """Tests for DockerConfig dataclass."""

    def test_default_values(self) -> None:
        """Default DockerConfig should have sensible defaults."""
        config = DockerConfig()
        assert config.image == "python:3.11-slim"
        assert config.memory_limit == "512m"
        assert config.cpu_limit == 1.0
        assert config.network == "none"
        assert config.workdir == "/workspace"
        assert config.mounts == []
        assert config.env_vars == {}
        assert config.remove_on_exit is True

    def test_custom_values(self) -> None:
        """Custom values should be stored correctly."""
        mounts = [VolumeMount("/tmp", "/data")]
        config = DockerConfig(
            image="node:18",
            memory_limit="1g",
            cpu_limit=2.0,
            network="bridge",
            workdir="/app",
            mounts=mounts,
            env_vars={"DEBUG": "1"},
            remove_on_exit=False,
        )
        assert config.image == "node:18"
        assert config.memory_limit == "1g"
        assert config.cpu_limit == 2.0
        assert config.network == "bridge"
        assert config.workdir == "/app"
        assert len(config.mounts) == 1
        assert config.env_vars == {"DEBUG": "1"}
        assert config.remove_on_exit is False

    def test_zero_cpu_raises(self) -> None:
        """Zero CPU limit should raise ValueError."""
        with pytest.raises(ValueError, match="cpu_limit must be > 0"):
            DockerConfig(cpu_limit=0)

    def test_negative_cpu_raises(self) -> None:
        """Negative CPU limit should raise ValueError."""
        with pytest.raises(ValueError, match="cpu_limit must be > 0"):
            DockerConfig(cpu_limit=-1.0)

    def test_invalid_network_raises(self) -> None:
        """Invalid network mode should raise ValueError."""
        with pytest.raises(ValueError, match="network must be"):
            DockerConfig(network="invalid")

    def test_empty_image_raises(self) -> None:
        """Empty image should raise ValueError."""
        with pytest.raises(ValueError, match="image cannot be empty"):
            DockerConfig(image="")

    def test_get_user_default(self) -> None:
        """get_user should return current UID:GID by default."""
        config = DockerConfig()
        user = config.get_user()
        if hasattr(os, "getuid"):
            assert ":" in user
            parts = user.split(":")
            assert len(parts) == 2
            assert parts[0].isdigit()
            assert parts[1].isdigit()
        else:
            assert user == "1000:1000"

    def test_get_user_custom(self) -> None:
        """get_user should return custom user if set."""
        config = DockerConfig(user="nobody")
        assert config.get_user() == "nobody"

    def test_to_docker_args(self) -> None:
        """to_docker_args should produce valid Docker arguments."""
        config = DockerConfig(
            memory_limit="256m",
            cpu_limit=0.5,
            network="bridge",
            workdir="/app",
        )
        args = config.to_docker_args()

        # Check required args are present
        assert "--memory" in args
        assert "256m" in args
        assert "--cpus" in args
        assert "0.5" in args
        assert "--network" in args
        assert "bridge" in args
        assert "--workdir" in args
        assert "/app" in args

    def test_to_docker_args_with_mounts(self) -> None:
        """to_docker_args should include volume mounts."""
        config = DockerConfig(
            mounts=[VolumeMount("/tmp/test", "/data")],
        )
        args = config.to_docker_args()
        assert "-v" in args

    def test_to_docker_args_with_env(self) -> None:
        """to_docker_args should include environment variables."""
        config = DockerConfig(
            env_vars={"FOO": "bar", "DEBUG": "1"},
        )
        args = config.to_docker_args()
        assert "-e" in args
        # Check both env vars are present
        env_count = args.count("-e")
        assert env_count == 2

    def test_with_workspace_factory(self) -> None:
        """with_workspace should create config with workspace mount."""
        config = DockerConfig.with_workspace(
            "/home/user/project",
            image="python:3.11",
            memory_limit="1g",
        )
        assert config.image == "python:3.11"
        assert config.memory_limit == "1g"
        assert len(config.mounts) == 1
        assert config.mounts[0].container_path == "/workspace"
        assert config.mounts[0].read_only is False

    def test_with_workspace_readonly_mounts(self) -> None:
        """with_workspace should add readonly mounts."""
        config = DockerConfig.with_workspace(
            "/home/user/project",
            readonly_mounts=[("/etc/ssl", "/ssl")],
        )
        assert len(config.mounts) == 2
        assert config.mounts[0].read_only is False  # workspace
        assert config.mounts[1].read_only is True  # ssl


# =============================================================================
# Test is_docker_available
# =============================================================================


class TestIsDockerAvailable:
    """Tests for is_docker_available function."""

    def test_returns_bool(self) -> None:
        """is_docker_available should return a boolean."""
        result = is_docker_available()
        assert isinstance(result, bool)

    def test_no_docker_binary(self) -> None:
        """Should return False if docker binary not found."""
        with patch("shutil.which", return_value=None):
            assert is_docker_available() is False

    def test_docker_not_running(self) -> None:
        """Should return False if docker daemon not running."""
        with patch("shutil.which", return_value="/usr/bin/docker"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1)
                assert is_docker_available() is False

    def test_docker_available(self) -> None:
        """Should return True if docker is available and running."""
        with patch("shutil.which", return_value="/usr/bin/docker"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                assert is_docker_available() is True


# =============================================================================
# Test DockerExecutionPolicy
# =============================================================================


class TestDockerExecutionPolicy:
    """Tests for DockerExecutionPolicy."""

    def test_default_config(self) -> None:
        """Default policy should use default config."""
        policy = DockerExecutionPolicy()
        assert policy.config.image == "python:3.11-slim"
        assert policy.config.network == "none"

    def test_custom_config(self) -> None:
        """Policy should accept custom config."""
        config = DockerConfig(image="node:18", network="bridge")
        policy = DockerExecutionPolicy(config=config)
        assert policy.config.image == "node:18"
        assert policy.config.network == "bridge"

    @pytest.mark.asyncio
    async def test_execute_docker_not_available(self) -> None:
        """execute should return error if Docker not available."""
        policy = DockerExecutionPolicy()
        shell_config = ShellConfig()

        with patch(
            "ai_infra.llm.shell.docker._async_is_docker_available",
            return_value=False,
        ):
            result = await policy.execute("echo hello", shell_config)

        assert result.success is False
        assert "Docker is not available" in result.stderr

    @pytest.mark.asyncio
    async def test_execute_success(self) -> None:
        """execute should run command in container."""
        policy = DockerExecutionPolicy()
        shell_config = ShellConfig()

        # Mock subprocess
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"hello\n", b""))
        mock_proc.returncode = 0

        with patch(
            "ai_infra.llm.shell.docker._async_is_docker_available",
            return_value=True,
        ):
            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                result = await policy.execute("echo hello", shell_config)

        assert result.success is True
        assert result.stdout == "hello"
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_execute_failure(self) -> None:
        """execute should handle command failure."""
        policy = DockerExecutionPolicy()
        shell_config = ShellConfig()

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"command not found\n"))
        mock_proc.returncode = 127

        with patch(
            "ai_infra.llm.shell.docker._async_is_docker_available",
            return_value=True,
        ):
            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                result = await policy.execute("nonexistent", shell_config)

        assert result.success is False
        assert result.exit_code == 127

    @pytest.mark.asyncio
    async def test_execute_timeout(self) -> None:
        """execute should handle timeout."""
        policy = DockerExecutionPolicy()
        shell_config = ShellConfig(timeout=1.0)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError())

        # Mock _kill_container to avoid actual docker kill
        with patch(
            "ai_infra.llm.shell.docker._async_is_docker_available",
            return_value=True,
        ):
            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                with patch.object(
                    policy,
                    "_kill_container",
                    return_value=None,
                ):
                    result = await policy.execute("sleep 100", shell_config)

        assert result.timed_out is True
        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_redaction(self) -> None:
        """execute should redact sensitive output."""
        policy = DockerExecutionPolicy()
        shell_config = ShellConfig()

        # Output containing a secret
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(b"Token: sk-abc123def456ghi789jkl012mno345pqrstu\n", b"")
        )
        mock_proc.returncode = 0

        with patch(
            "ai_infra.llm.shell.docker._async_is_docker_available",
            return_value=True,
        ):
            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                result = await policy.execute("echo $SECRET", shell_config)

        assert "sk-abc123" not in result.stdout
        assert "[REDACTED]" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_output_truncation(self) -> None:
        """execute should truncate large output."""
        policy = DockerExecutionPolicy()
        shell_config = ShellConfig(max_output_bytes=20)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"A" * 100, b""))
        mock_proc.returncode = 0

        with patch(
            "ai_infra.llm.shell.docker._async_is_docker_available",
            return_value=True,
        ):
            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                result = await policy.execute("generate_output", shell_config)

        assert "[OUTPUT TRUNCATED]" in result.stdout


# =============================================================================
# Test DockerSession
# =============================================================================


class TestDockerSession:
    """Tests for DockerSession persistent container."""

    def test_initial_state(self) -> None:
        """Session should start in stopped state."""
        session = DockerSession()
        assert session.is_started is False
        assert session.container_id is None
        assert session.container_name.startswith("ai-infra-session-")

    def test_custom_config(self) -> None:
        """Session should accept custom config."""
        config = DockerConfig(image="node:18")
        session = DockerSession(config=config)
        assert session._config.image == "node:18"

    @pytest.mark.asyncio
    async def test_start_docker_not_available(self) -> None:
        """start should raise if Docker not available."""
        session = DockerSession()

        with patch(
            "ai_infra.llm.shell.docker._async_is_docker_available",
            return_value=False,
        ):
            with pytest.raises(RuntimeError, match="Docker is not available"):
                await session.start()

    @pytest.mark.asyncio
    async def test_start_success(self) -> None:
        """start should create container."""
        session = DockerSession()

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"abc123def456\n", b""))
        mock_proc.returncode = 0

        with patch(
            "ai_infra.llm.shell.docker._async_is_docker_available",
            return_value=True,
        ):
            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                await session.start()

        assert session.is_started is True
        assert session.container_id == "abc123def456"

    @pytest.mark.asyncio
    async def test_start_failure(self) -> None:
        """start should raise on container creation failure."""
        session = DockerSession()

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Error: image not found\n"))
        mock_proc.returncode = 1

        with patch(
            "ai_infra.llm.shell.docker._async_is_docker_available",
            return_value=True,
        ):
            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                with pytest.raises(RuntimeError, match="Failed to start container"):
                    await session.start()

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self) -> None:
        """stop should be safe when not started."""
        session = DockerSession()
        await session.stop()  # Should not raise
        assert session.is_started is False

    @pytest.mark.asyncio
    async def test_stop_success(self) -> None:
        """stop should stop and remove container."""
        session = DockerSession()
        session._started = True
        session._container_id = "abc123"

        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            await session.stop()

        assert session.is_started is False
        assert session.container_id is None

    @pytest.mark.asyncio
    async def test_execute_not_started(self) -> None:
        """execute should raise if session not started."""
        session = DockerSession()

        with pytest.raises(RuntimeError, match="Session not started"):
            await session.execute("echo hello")

    @pytest.mark.asyncio
    async def test_execute_success(self) -> None:
        """execute should run command in session container."""
        session = DockerSession()
        session._started = True
        session._container_id = "abc123"

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"hello\n", b""))
        mock_proc.returncode = 0

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            result = await session.execute("echo hello")

        assert result.success is True
        assert result.stdout == "hello"

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Session should work as async context manager."""
        # Mock start and stop
        mock_start_proc = AsyncMock()
        mock_start_proc.communicate = AsyncMock(return_value=(b"container123\n", b""))
        mock_start_proc.returncode = 0

        mock_stop_proc = AsyncMock()
        mock_stop_proc.wait = AsyncMock(return_value=0)

        call_count = 0

        async def mock_subprocess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_start_proc
            return mock_stop_proc

        with patch(
            "ai_infra.llm.shell.docker._async_is_docker_available",
            return_value=True,
        ):
            with patch(
                "asyncio.create_subprocess_exec",
                side_effect=mock_subprocess,
            ):
                async with DockerSession() as session:
                    assert session.is_started is True

                # After context, session should be stopped
                assert session.is_started is False


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestCreateDockerPolicy:
    """Tests for create_docker_policy factory."""

    def test_default_values(self) -> None:
        """Factory should create policy with defaults."""
        policy = create_docker_policy()
        assert policy.config.image == "python:3.11-slim"
        assert policy.config.network == "none"

    def test_custom_image(self) -> None:
        """Factory should accept custom image."""
        policy = create_docker_policy(image="node:18")
        assert policy.config.image == "node:18"

    def test_with_workspace(self) -> None:
        """Factory should add workspace mount."""
        policy = create_docker_policy(workspace="/tmp/project")
        assert len(policy.config.mounts) == 1
        assert policy.config.mounts[0].container_path == "/workspace"
        assert policy.config.mounts[0].read_only is False

    def test_with_readonly_mounts(self) -> None:
        """Factory should add readonly mounts."""
        policy = create_docker_policy(
            readonly_mounts=[("/etc/ssl", "/ssl"), ("/etc/ca", "/ca")],
        )
        assert len(policy.config.mounts) == 2
        assert all(m.read_only for m in policy.config.mounts)

    def test_custom_resources(self) -> None:
        """Factory should accept resource limits."""
        policy = create_docker_policy(
            memory_limit="1g",
            cpu_limit=2.0,
        )
        assert policy.config.memory_limit == "1g"
        assert policy.config.cpu_limit == 2.0


class TestCreateDockerSession:
    """Tests for create_docker_session factory."""

    def test_default_values(self) -> None:
        """Factory should create session with defaults."""
        session = create_docker_session()
        assert session._config.image == "python:3.11-slim"
        assert session._config.network == "none"

    def test_with_workspace(self) -> None:
        """Factory should add workspace mount."""
        session = create_docker_session(workspace="/tmp/project")
        assert len(session._config.mounts) == 1
        assert session._config.mounts[0].container_path == "/workspace"

    def test_custom_network(self) -> None:
        """Factory should accept custom network."""
        session = create_docker_session(network="bridge")
        assert session._config.network == "bridge"


# =============================================================================
# Integration Tests (require Docker)
# =============================================================================


@pytest.mark.skipif(
    not is_docker_available(),
    reason="Docker not available",
)
class TestDockerIntegration:
    """Integration tests that require Docker to be running."""

    @pytest.mark.asyncio
    async def test_real_execution(self) -> None:
        """Test actual Docker execution."""
        policy = DockerExecutionPolicy(
            config=DockerConfig(
                image="python:3.11-slim",
                network="none",
            ),
        )

        result = await policy.execute(
            "python --version",
            ShellConfig(timeout=30.0),
        )

        assert result.success is True
        assert "Python 3.11" in result.stdout or "Python 3.11" in result.stderr

    @pytest.mark.asyncio
    async def test_real_session(self) -> None:
        """Test actual Docker session."""
        async with create_docker_session(
            image="python:3.11-slim",
            network="none",
        ) as session:
            # Note: Each docker exec is a new shell, so env vars don't persist
            # Test multiple commands within same exec instead
            result1 = await session.execute("echo hello && echo world")
            assert result1.success is True
            assert "hello" in result1.stdout
            assert "world" in result1.stdout

            # Second separate command
            result2 = await session.execute("python --version")
            assert result2.success is True
            assert "Python 3.11" in result2.stdout or "Python 3.11" in result2.stderr

    @pytest.mark.asyncio
    async def test_network_isolation(self) -> None:
        """Test that network=none blocks network access."""
        policy = DockerExecutionPolicy(
            config=DockerConfig(
                image="python:3.11-slim",
                network="none",
            ),
        )

        # This should fail because network is isolated
        result = await policy.execute(
            "python -c 'import urllib.request; urllib.request.urlopen(\"http://example.com\")'",
            ShellConfig(timeout=10.0),
        )

        # Network access should fail
        assert result.success is False
