"""Tests for shell resource limits module.

Phase 4.2 of EXECUTOR_CLI.md - Resource Limits.
"""

from __future__ import annotations

import sys

import pytest

from ai_infra.llm.shell.limits import (
    DEFAULT_RESOURCE_LIMITS,
    LimitedExecutionPolicy,
    ResourceLimits,
    create_limit_prelude,
    create_limited_policy,
    is_limits_supported,
)
from ai_infra.llm.shell.types import ShellConfig


class TestResourceLimits:
    """Tests for ResourceLimits dataclass."""

    def test_default_values(self) -> None:
        """Default ResourceLimits should have sensible defaults."""
        limits = ResourceLimits()
        assert limits.memory_mb == 512
        assert limits.cpu_seconds == 60
        assert limits.max_file_size_mb == 100
        assert limits.max_open_files == 256
        assert limits.max_processes == 32
        assert limits.max_output_bytes == 1_000_000
        assert limits.enable_core_dumps is False

    def test_custom_values(self) -> None:
        """Custom values should be stored correctly."""
        limits = ResourceLimits(
            memory_mb=256,
            cpu_seconds=30,
            max_file_size_mb=50,
            max_open_files=128,
            max_processes=16,
            max_output_bytes=500_000,
            enable_core_dumps=True,
        )
        assert limits.memory_mb == 256
        assert limits.cpu_seconds == 30
        assert limits.max_file_size_mb == 50
        assert limits.max_open_files == 128
        assert limits.max_processes == 16
        assert limits.max_output_bytes == 500_000
        assert limits.enable_core_dumps is True

    def test_negative_memory_raises(self) -> None:
        """Negative memory_mb should raise ValueError."""
        with pytest.raises(ValueError, match="memory_mb must be >= 0"):
            ResourceLimits(memory_mb=-1)

    def test_negative_cpu_raises(self) -> None:
        """Negative cpu_seconds should raise ValueError."""
        with pytest.raises(ValueError, match="cpu_seconds must be >= 0"):
            ResourceLimits(cpu_seconds=-1)

    def test_negative_file_size_raises(self) -> None:
        """Negative max_file_size_mb should raise ValueError."""
        with pytest.raises(ValueError, match="max_file_size_mb must be >= 0"):
            ResourceLimits(max_file_size_mb=-1)

    def test_negative_open_files_raises(self) -> None:
        """Negative max_open_files should raise ValueError."""
        with pytest.raises(ValueError, match="max_open_files must be >= 0"):
            ResourceLimits(max_open_files=-1)

    def test_negative_processes_raises(self) -> None:
        """Negative max_processes should raise ValueError."""
        with pytest.raises(ValueError, match="max_processes must be >= 0"):
            ResourceLimits(max_processes=-1)

    def test_negative_output_bytes_raises(self) -> None:
        """Negative max_output_bytes should raise ValueError."""
        with pytest.raises(ValueError, match="max_output_bytes must be >= 0"):
            ResourceLimits(max_output_bytes=-1)

    def test_zero_values_allowed(self) -> None:
        """Zero values should be allowed (means disabled)."""
        limits = ResourceLimits(
            memory_mb=0,
            cpu_seconds=0,
            max_file_size_mb=0,
            max_open_files=0,
            max_processes=0,
            max_output_bytes=0,
        )
        assert limits.memory_mb == 0
        assert limits.cpu_seconds == 0

    def test_to_dict(self) -> None:
        """to_dict should return all fields."""
        limits = ResourceLimits(memory_mb=256, cpu_seconds=30)
        result = limits.to_dict()
        assert result["memory_mb"] == 256
        assert result["cpu_seconds"] == 30
        assert "max_file_size_mb" in result
        assert "enable_core_dumps" in result

    def test_strict_preset(self) -> None:
        """strict() should return conservative limits."""
        limits = ResourceLimits.strict()
        assert limits.memory_mb == 256
        assert limits.cpu_seconds == 30
        assert limits.max_file_size_mb == 10
        assert limits.max_open_files == 64
        assert limits.max_processes == 8
        assert limits.max_output_bytes == 100_000
        assert limits.enable_core_dumps is False

    def test_permissive_preset(self) -> None:
        """permissive() should return generous limits."""
        limits = ResourceLimits.permissive()
        assert limits.memory_mb == 2048
        assert limits.cpu_seconds == 300
        assert limits.max_file_size_mb == 500
        assert limits.max_open_files == 1024
        assert limits.max_processes == 128
        assert limits.max_output_bytes == 10_000_000
        assert limits.enable_core_dumps is False

    def test_unlimited_preset(self) -> None:
        """unlimited() should return all zeros (disabled)."""
        limits = ResourceLimits.unlimited()
        assert limits.memory_mb == 0
        assert limits.cpu_seconds == 0
        assert limits.max_file_size_mb == 0
        assert limits.max_open_files == 0
        assert limits.max_processes == 0
        assert limits.max_output_bytes == 0
        assert limits.enable_core_dumps is True


class TestDefaultResourceLimits:
    """Tests for DEFAULT_RESOURCE_LIMITS constant."""

    def test_default_exists(self) -> None:
        """DEFAULT_RESOURCE_LIMITS should exist."""
        assert DEFAULT_RESOURCE_LIMITS is not None
        assert isinstance(DEFAULT_RESOURCE_LIMITS, ResourceLimits)

    def test_default_matches_resourcelimits_default(self) -> None:
        """DEFAULT_RESOURCE_LIMITS should match ResourceLimits()."""
        default = ResourceLimits()
        assert DEFAULT_RESOURCE_LIMITS.memory_mb == default.memory_mb
        assert DEFAULT_RESOURCE_LIMITS.cpu_seconds == default.cpu_seconds


class TestIsLimitsSupported:
    """Tests for is_limits_supported function."""

    def test_returns_bool(self) -> None:
        """is_limits_supported should return a boolean."""
        result = is_limits_supported()
        assert isinstance(result, bool)

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="Unix only")
    def test_supported_on_unix(self) -> None:
        """Limits should be supported on Unix-like systems."""
        assert is_limits_supported() is True

    @pytest.mark.skipif(not sys.platform.startswith("win"), reason="Windows only")
    def test_not_supported_on_windows(self) -> None:
        """Limits should not be supported on Windows."""
        assert is_limits_supported() is False


class TestCreateLimitPrelude:
    """Tests for create_limit_prelude function."""

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="Unix only")
    def test_generates_ulimit_commands(self) -> None:
        """Should generate ulimit commands for each limit."""
        limits = ResourceLimits(
            memory_mb=256,
            cpu_seconds=30,
            max_file_size_mb=10,
            max_open_files=64,
            max_processes=8,
        )
        prelude = create_limit_prelude(limits)

        # Memory: 256MB = 262144KB
        assert "ulimit -v 262144" in prelude
        # CPU seconds
        assert "ulimit -t 30" in prelude
        # File size: 10MB = 10240KB
        assert "ulimit -f 10240" in prelude
        # Open files
        assert "ulimit -n 64" in prelude
        # Processes
        assert "ulimit -u 8" in prelude
        # Core dumps disabled by default
        assert "ulimit -c 0" in prelude

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="Unix only")
    def test_ends_with_semicolon_space(self) -> None:
        """Prelude should end with '; ' for command chaining."""
        limits = ResourceLimits(memory_mb=256)
        prelude = create_limit_prelude(limits)
        assert prelude.endswith("; ")

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="Unix only")
    def test_zero_values_skipped(self) -> None:
        """Zero values should not generate ulimit commands."""
        limits = ResourceLimits(
            memory_mb=0,
            cpu_seconds=0,
            max_file_size_mb=0,
            max_open_files=0,
            max_processes=0,
            enable_core_dumps=True,  # Don't disable
        )
        prelude = create_limit_prelude(limits)
        assert prelude == ""

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="Unix only")
    def test_core_dumps_enabled(self) -> None:
        """When enable_core_dumps=True, should not set ulimit -c 0."""
        limits = ResourceLimits(memory_mb=256, enable_core_dumps=True)
        prelude = create_limit_prelude(limits)
        assert "ulimit -c 0" not in prelude

    @pytest.mark.skipif(not sys.platform.startswith("win"), reason="Windows only")
    def test_returns_empty_on_windows(self) -> None:
        """On Windows, should return empty string."""
        limits = ResourceLimits(memory_mb=256)
        prelude = create_limit_prelude(limits)
        assert prelude == ""


class TestLimitedExecutionPolicy:
    """Tests for LimitedExecutionPolicy class."""

    def test_init_default_limits(self) -> None:
        """Default init should use DEFAULT_RESOURCE_LIMITS."""
        policy = LimitedExecutionPolicy()
        assert policy.limits == DEFAULT_RESOURCE_LIMITS

    def test_init_custom_limits(self) -> None:
        """Custom limits should be stored."""
        limits = ResourceLimits(memory_mb=256)
        policy = LimitedExecutionPolicy(limits=limits)
        assert policy.limits.memory_mb == 256

    def test_limits_property(self) -> None:
        """limits property should return configured limits."""
        limits = ResourceLimits.strict()
        policy = LimitedExecutionPolicy(limits=limits)
        assert policy.limits is limits

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform.startswith("win"), reason="Unix only")
    async def test_execute_simple_command(self) -> None:
        """Should execute simple commands successfully."""
        # Use permissive limits to allow bash to fork
        limits = ResourceLimits.permissive()
        policy = LimitedExecutionPolicy(limits=limits)
        config = ShellConfig(timeout=10.0)

        result = await policy.execute("echo hello", config)

        assert result.success is True
        assert result.stdout == "hello"
        assert result.exit_code == 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform.startswith("win"), reason="Unix only")
    async def test_execute_with_limits(self) -> None:
        """Should execute with limits applied."""
        # Use permissive limits to allow command to run
        limits = ResourceLimits.permissive()
        policy = LimitedExecutionPolicy(limits=limits)
        config = ShellConfig(timeout=10.0)

        result = await policy.execute("echo 'limits applied'", config)

        assert result.success is True
        assert "limits applied" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_timeout(self) -> None:
        """Should handle command timeout."""
        # Use unlimited limits to avoid resource constraint errors
        limits = ResourceLimits.unlimited()
        policy = LimitedExecutionPolicy(limits=limits)
        config = ShellConfig(timeout=0.1)  # Very short timeout

        result = await policy.execute("sleep 10", config)

        assert result.success is False
        assert result.timed_out is True

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform.startswith("win"), reason="Unix only")
    async def test_output_truncation(self) -> None:
        """Should truncate output exceeding max_output_bytes."""
        limits = ResourceLimits(
            memory_mb=0,  # Disable to avoid issues
            cpu_seconds=0,
            max_file_size_mb=0,
            max_open_files=0,
            max_processes=0,  # Disable to allow bash to fork
            max_output_bytes=20,  # Very small to ensure truncation
        )
        policy = LimitedExecutionPolicy(limits=limits)
        config = ShellConfig(timeout=10.0)

        # Generate output longer than 20 bytes
        result = await policy.execute(
            "echo 'This is a very long string that definitely exceeds twenty bytes'", config
        )

        assert "[OUTPUT TRUNCATED]" in result.stdout

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform.startswith("win"), reason="Unix only")
    async def test_redaction_applied(self) -> None:
        """Should apply redaction rules to output."""
        limits = ResourceLimits.unlimited()
        policy = LimitedExecutionPolicy(limits=limits)
        config = ShellConfig(timeout=10.0)

        # Echo something that looks like an API key
        result = await policy.execute("echo 'key=sk-abc123def456ghi789jkl012mno345pqr'", config)

        assert "sk-abc123" not in result.stdout
        assert "[REDACTED]" in result.stdout

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform.startswith("win"), reason="Unix only")
    async def test_execute_failing_command(self) -> None:
        """Should handle failing commands."""
        limits = ResourceLimits.permissive()
        policy = LimitedExecutionPolicy(limits=limits)
        config = ShellConfig(timeout=10.0)

        result = await policy.execute("exit 42", config)

        assert result.success is False
        assert result.exit_code == 42

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform.startswith("win"), reason="Unix only")
    async def test_annotate_oom_error(self) -> None:
        """Should annotate OOM-style exit codes."""
        policy = LimitedExecutionPolicy()

        stderr, limit_type = policy._annotate_limit_errors("", 137)
        assert "[RESOURCE LIMIT]" in stderr
        assert "memory" in stderr.lower() or "killed" in stderr.lower()
        assert limit_type == "memory"

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform.startswith("win"), reason="Unix only")
    async def test_annotate_cpu_exceeded(self) -> None:
        """Should annotate CPU time exceeded exit codes."""
        limits = ResourceLimits(cpu_seconds=30)
        policy = LimitedExecutionPolicy(limits=limits)

        stderr, limit_type = policy._annotate_limit_errors("", 152)
        assert "[RESOURCE LIMIT]" in stderr
        assert "CPU" in stderr or "cpu" in stderr
        assert limit_type == "cpu"

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform.startswith("win"), reason="Unix only")
    async def test_annotate_file_size_exceeded(self) -> None:
        """Should annotate file size exceeded exit codes."""
        limits = ResourceLimits(max_file_size_mb=10)
        policy = LimitedExecutionPolicy(limits=limits)

        stderr, limit_type = policy._annotate_limit_errors("", 153)
        assert "[RESOURCE LIMIT]" in stderr
        assert "File size" in stderr or "file size" in stderr
        assert limit_type == "file_size"

    @pytest.mark.asyncio
    async def test_annotate_too_many_files(self) -> None:
        """Should annotate 'too many open files' errors."""
        limits = ResourceLimits(max_open_files=64)
        policy = LimitedExecutionPolicy(limits=limits)

        stderr, limit_type = policy._annotate_limit_errors("bash: too many open files", 1)
        assert "[RESOURCE LIMIT]" in stderr
        assert "64" in stderr
        assert limit_type == "open_files"

    @pytest.mark.asyncio
    async def test_annotate_cannot_fork(self) -> None:
        """Should annotate 'cannot fork' errors."""
        limits = ResourceLimits(max_processes=8)
        policy = LimitedExecutionPolicy(limits=limits)

        stderr, limit_type = policy._annotate_limit_errors("bash: cannot fork", 1)
        assert "[RESOURCE LIMIT]" in stderr
        assert "8" in stderr
        assert limit_type == "processes"


class TestCreateLimitedPolicy:
    """Tests for create_limited_policy convenience function."""

    def test_default_values(self) -> None:
        """Should create policy with default values."""
        policy = create_limited_policy()
        assert policy.limits.memory_mb == 512
        assert policy.limits.cpu_seconds == 60
        assert policy.limits.max_processes == 32

    def test_custom_values(self) -> None:
        """Should create policy with custom values."""
        policy = create_limited_policy(
            memory_mb=256,
            cpu_seconds=30,
            max_processes=16,
        )
        assert policy.limits.memory_mb == 256
        assert policy.limits.cpu_seconds == 30
        assert policy.limits.max_processes == 16

    def test_returns_limited_execution_policy(self) -> None:
        """Should return a LimitedExecutionPolicy instance."""
        policy = create_limited_policy()
        assert isinstance(policy, LimitedExecutionPolicy)


class TestNetworkRestrictions:
    """Tests for network restriction integration."""

    def test_validate_command_with_network_import(self) -> None:
        """validate_command_with_network should be importable."""
        from ai_infra.llm.shell.security import validate_command_with_network

        assert validate_command_with_network is not None

    def test_network_command_denied_when_not_allowed(self) -> None:
        """Network commands should be denied when allow_network=False."""
        from ai_infra.llm.shell.security import (
            SecurityPolicy,
            validate_command_with_network,
        )

        policy = SecurityPolicy(allow_network=False)
        result = validate_command_with_network("curl https://example.com", policy)

        assert result.is_denied
        assert "network" in result.reason.lower()

    def test_network_command_allowed_by_default(self) -> None:
        """Network commands should be allowed by default."""
        from ai_infra.llm.shell.security import (
            SecurityPolicy,
            validate_command_with_network,
        )

        policy = SecurityPolicy()  # allow_network=True by default
        result = validate_command_with_network("curl https://example.com", policy)

        assert result.is_allowed

    def test_non_network_command_allowed(self) -> None:
        """Non-network commands should be allowed even with network disabled."""
        from ai_infra.llm.shell.security import (
            SecurityPolicy,
            validate_command_with_network,
        )

        policy = SecurityPolicy(allow_network=False)
        result = validate_command_with_network("echo hello", policy)

        assert result.is_allowed

    @pytest.mark.parametrize(
        "command",
        [
            "curl https://api.example.com",
            "wget https://example.com/file.txt",
            "ssh user@host",
            "scp file.txt user@host:",
            "rsync -av local/ user@host:",
            "ftp ftp.example.com",
            "nc localhost 8080",
            "ping google.com",
        ],
    )
    def test_various_network_commands_denied(self, command: str) -> None:
        """Various network commands should be denied when network disabled."""
        from ai_infra.llm.shell.security import (
            SecurityPolicy,
            validate_command_with_network,
        )

        policy = SecurityPolicy(allow_network=False)
        result = validate_command_with_network(command, policy)

        assert result.is_denied, f"Expected {command!r} to be denied"

    def test_dangerous_command_still_denied(self) -> None:
        """Dangerous commands should still be denied even if network is allowed."""
        from ai_infra.llm.shell.security import (
            SecurityPolicy,
            validate_command_with_network,
        )

        policy = SecurityPolicy(allow_network=True)
        result = validate_command_with_network("rm -rf /", policy)

        assert result.is_denied
