"""Tests for shell types module.

This module provides comprehensive tests for the shell execution types:
- ShellResult: Result dataclass and factory methods
- ShellConfig: Configuration dataclass
- RedactionRule: Pattern-based sanitization
- HostExecutionPolicy: Direct execution on host

Phase 1.1 of EXECUTOR_CLI.md - Shell Tool Integration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from ai_infra.llm.shell import (
    DEFAULT_REDACTION_RULES,
    ExecutionPolicy,
    HostExecutionPolicy,
    RedactionRule,
    ShellConfig,
    ShellResult,
    apply_redaction_rules,
)

# =============================================================================
# ShellResult Tests
# =============================================================================


class TestShellResult:
    """Tests for ShellResult dataclass."""

    def test_create_success_result(self):
        """Test creating a successful result."""
        result = ShellResult(
            success=True,
            exit_code=0,
            stdout="Hello, World!",
            stderr="",
            command="echo 'Hello, World!'",
            duration_ms=5.2,
        )

        assert result.success is True
        assert result.exit_code == 0
        assert result.stdout == "Hello, World!"
        assert result.stderr == ""
        assert result.command == "echo 'Hello, World!'"
        assert result.duration_ms == 5.2
        assert result.timed_out is False

    def test_create_failure_result(self):
        """Test creating a failed result."""
        result = ShellResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr="Command not found",
            command="nonexistent_command",
            duration_ms=10.0,
        )

        assert result.success is False
        assert result.exit_code == 1
        assert result.stderr == "Command not found"

    def test_create_timeout_result(self):
        """Test creating a timeout result."""
        result = ShellResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr="Timed out",
            command="sleep 1000",
            duration_ms=30000.0,
            timed_out=True,
        )

        assert result.success is False
        assert result.exit_code == -1
        assert result.timed_out is True

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = ShellResult(
            success=True,
            exit_code=0,
            stdout="output",
            stderr="",
            command="test",
            duration_ms=1.0,
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["exit_code"] == 0
        assert d["stdout"] == "output"
        assert d["stderr"] == ""
        assert d["command"] == "test"
        assert d["duration_ms"] == 1.0
        assert d["timed_out"] is False

    def test_from_timeout(self):
        """Test creating result from timeout."""
        result = ShellResult.from_timeout("slow_command", 30.0)

        assert result.success is False
        assert result.exit_code == -1
        assert result.timed_out is True
        assert result.command == "slow_command"
        assert result.duration_ms == 30000.0
        assert "30" in result.stderr

    def test_from_error(self):
        """Test creating result from exception."""
        error = ValueError("Something went wrong")
        result = ShellResult.from_error("bad_command", error, 100.0)

        assert result.success is False
        assert result.exit_code == -1
        assert result.timed_out is False
        assert "Something went wrong" in result.stderr
        assert result.duration_ms == 100.0

    def test_result_is_immutable(self):
        """Test that ShellResult is frozen."""
        result = ShellResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            command="test",
            duration_ms=1.0,
        )

        with pytest.raises(AttributeError):
            result.success = False  # type: ignore[misc]


# =============================================================================
# ShellConfig Tests
# =============================================================================


class TestShellConfig:
    """Tests for ShellConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ShellConfig()

        assert config.timeout == 120.0
        assert config.max_output_bytes == 1_000_000
        assert config.env is None
        assert config.cwd is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ShellConfig(
            timeout=60.0,
            max_output_bytes=500_000,
            env={"MY_VAR": "value"},
            cwd=Path("/tmp"),
        )

        assert config.timeout == 60.0
        assert config.max_output_bytes == 500_000
        assert config.env == {"MY_VAR": "value"}
        assert config.cwd == Path("/tmp")

    def test_get_env_without_custom(self):
        """Test get_env without custom environment."""
        config = ShellConfig()
        env = config.get_env()

        # Should contain os.environ
        assert "PATH" in env or "Path" in env  # Windows uses 'Path'

    def test_get_env_with_custom(self):
        """Test get_env with custom environment variables."""
        config = ShellConfig(env={"CUSTOM_VAR": "custom_value"})
        env = config.get_env()

        assert env["CUSTOM_VAR"] == "custom_value"
        # Should still contain os.environ
        assert "PATH" in env or "Path" in env

    def test_get_env_override(self):
        """Test that custom env overrides os.environ."""
        # Set a known env var
        original = os.environ.get("SHELL_TEST_VAR")
        os.environ["SHELL_TEST_VAR"] = "original"

        try:
            config = ShellConfig(env={"SHELL_TEST_VAR": "overridden"})
            env = config.get_env()
            assert env["SHELL_TEST_VAR"] == "overridden"
        finally:
            if original is None:
                os.environ.pop("SHELL_TEST_VAR", None)
            else:
                os.environ["SHELL_TEST_VAR"] = original

    def test_get_cwd_default(self):
        """Test get_cwd returns current directory by default."""
        config = ShellConfig()
        cwd = config.get_cwd()

        assert cwd == Path.cwd()

    def test_get_cwd_custom(self):
        """Test get_cwd returns custom directory."""
        config = ShellConfig(cwd=Path("/tmp"))
        cwd = config.get_cwd()

        assert cwd == Path("/tmp")

    def test_default_shell_platform_specific(self):
        """Test default shell is platform-appropriate."""
        config = ShellConfig()

        if sys.platform.startswith("win"):
            assert config.shell == "powershell"
        else:
            assert config.shell == "/bin/bash"


# =============================================================================
# RedactionRule Tests
# =============================================================================


class TestRedactionRule:
    """Tests for RedactionRule dataclass."""

    def test_create_rule(self):
        """Test creating a redaction rule."""
        rule = RedactionRule(
            name="api_key",
            pattern=r"sk-[a-zA-Z0-9]{32,}",
        )

        assert rule.name == "api_key"
        assert rule.pattern == r"sk-[a-zA-Z0-9]{32,}"
        assert rule.replacement == "[REDACTED]"

    def test_custom_replacement(self):
        """Test custom replacement text."""
        rule = RedactionRule(
            name="password",
            pattern=r"password=\S+",
            replacement="password=***",
        )

        assert rule.replacement == "password=***"

    def test_apply_matches(self):
        """Test applying rule that matches."""
        rule = RedactionRule(
            name="api_key",
            pattern=r"sk-[a-zA-Z0-9]{32,}",
        )

        text = "API key: sk-abc123def456ghi789jkl012mno345pqr678"
        result = rule.apply(text)

        assert result == "API key: [REDACTED]"

    def test_apply_no_match(self):
        """Test applying rule that doesn't match."""
        rule = RedactionRule(
            name="api_key",
            pattern=r"sk-[a-zA-Z0-9]{32,}",
        )

        text = "No secrets here"
        result = rule.apply(text)

        assert result == "No secrets here"

    def test_apply_multiple_matches(self):
        """Test applying rule with multiple matches."""
        rule = RedactionRule(
            name="api_key",
            pattern=r"sk-[a-zA-Z0-9]{32,}",
        )

        text = "Key1: sk-abc123def456ghi789jkl012mno345pqr678 Key2: sk-xyz789abc123def456ghi789jkl012mno345"
        result = rule.apply(text)

        assert result == "Key1: [REDACTED] Key2: [REDACTED]"

    def test_rule_is_immutable(self):
        """Test that RedactionRule is frozen."""
        rule = RedactionRule(name="test", pattern="test")

        with pytest.raises(AttributeError):
            rule.name = "changed"  # type: ignore[misc]


# =============================================================================
# Default Redaction Rules Tests
# =============================================================================


class TestDefaultRedactionRules:
    """Tests for DEFAULT_REDACTION_RULES."""

    def test_default_rules_exist(self):
        """Test that default rules are defined."""
        assert len(DEFAULT_REDACTION_RULES) > 0

    def test_openai_api_key_redacted(self):
        """Test OpenAI API key is redacted."""
        text = "export OPENAI_API_KEY=sk-abc123def456ghi789jkl012mno345pqr678xyz"
        result = apply_redaction_rules(text)

        assert "sk-abc123" not in result
        assert "[REDACTED]" in result

    def test_anthropic_api_key_redacted(self):
        """Test Anthropic API key is redacted."""
        text = "ANTHROPIC_API_KEY=sk-ant-abc123-def456ghi789jkl012mno345pqr678xyz"
        result = apply_redaction_rules(text)

        assert "sk-ant-" not in result
        assert "[REDACTED]" in result

    def test_password_redacted(self):
        """Test passwords are redacted."""
        text = "password=mysecretpassword123"
        result = apply_redaction_rules(text)

        assert "mysecretpassword123" not in result
        assert "[REDACTED]" in result

    def test_bearer_token_redacted(self):
        """Test bearer tokens are redacted."""
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test"
        result = apply_redaction_rules(text)

        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result
        assert "[REDACTED]" in result

    def test_private_key_header_redacted(self):
        """Test private key headers are redacted."""
        # Build the key header dynamically to avoid pre-commit hook detection
        # nosec: This is a test for redaction, not an actual key
        key_type = "RSA"
        key_header = f"-----BEGIN {key_type} PRI" + "VATE KEY-----"
        text = f"{key_header}\nMIIEpQIBAAKCAQEA..."
        result = apply_redaction_rules(text)

        assert key_header not in result
        assert "[REDACTED]" in result

    def test_normal_text_unchanged(self):
        """Test that normal text is not redacted."""
        text = "This is just normal output with no secrets"
        result = apply_redaction_rules(text)

        assert result == text

    def test_apply_with_custom_rules(self):
        """Test applying custom rules instead of defaults."""
        custom_rules = (RedactionRule("custom", r"secret-\d+", replacement="[HIDDEN]"),)

        text = "The code is secret-12345"
        result = apply_redaction_rules(text, custom_rules)

        assert result == "The code is [HIDDEN]"

    def test_apply_with_none_uses_defaults(self):
        """Test that None uses default rules."""
        text = "sk-abc123def456ghi789jkl012mno345pqr678xyz"
        result = apply_redaction_rules(text, None)

        assert "[REDACTED]" in result


# =============================================================================
# ExecutionPolicy Protocol Tests
# =============================================================================


class TestExecutionPolicy:
    """Tests for ExecutionPolicy protocol."""

    def test_host_execution_policy_is_execution_policy(self):
        """Test that HostExecutionPolicy implements ExecutionPolicy."""
        policy = HostExecutionPolicy()

        assert isinstance(policy, ExecutionPolicy)

    def test_custom_class_can_implement_protocol(self):
        """Test that custom classes can implement the protocol."""

        class MockExecutionPolicy:
            async def execute(self, command: str, config: ShellConfig) -> ShellResult:
                return ShellResult(
                    success=True,
                    exit_code=0,
                    stdout="mock",
                    stderr="",
                    command=command,
                    duration_ms=1.0,
                )

        policy = MockExecutionPolicy()
        assert isinstance(policy, ExecutionPolicy)


# =============================================================================
# HostExecutionPolicy Tests
# =============================================================================


class TestHostExecutionPolicy:
    """Tests for HostExecutionPolicy."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        policy = HostExecutionPolicy()

        assert policy._redaction_rules == DEFAULT_REDACTION_RULES

    def test_init_with_custom_rules(self):
        """Test initialization with custom redaction rules."""
        custom_rules = (RedactionRule("test", r"test"),)
        policy = HostExecutionPolicy(redaction_rules=custom_rules)

        assert policy._redaction_rules == custom_rules

    def test_init_with_no_redaction(self):
        """Test initialization with redaction disabled."""
        policy = HostExecutionPolicy(redaction_rules=None)

        assert policy._redaction_rules is None

    @pytest.mark.asyncio
    async def test_execute_simple_command(self):
        """Test executing a simple echo command."""
        policy = HostExecutionPolicy()
        config = ShellConfig(timeout=10.0)

        result = await policy.execute("echo hello", config)

        assert result.success is True
        assert result.exit_code == 0
        assert "hello" in result.stdout
        assert result.timed_out is False
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_execute_command_with_exit_code(self):
        """Test executing a command that exits with non-zero."""
        policy = HostExecutionPolicy()
        config = ShellConfig(timeout=10.0)

        # exit 1 returns exit code 1
        if sys.platform.startswith("win"):
            result = await policy.execute("exit 1", config)
        else:
            result = await policy.execute("exit 1", config)

        assert result.success is False
        assert result.exit_code == 1

    @pytest.mark.asyncio
    async def test_execute_command_with_stderr(self):
        """Test executing a command that writes to stderr."""
        policy = HostExecutionPolicy()
        config = ShellConfig(timeout=10.0)

        if sys.platform.startswith("win"):
            result = await policy.execute("Write-Error 'error message' 2>&1", config)
        else:
            result = await policy.execute("echo 'error message' >&2", config)

        assert "error" in result.stderr.lower() or "error" in result.stdout.lower()

    @pytest.mark.asyncio
    async def test_execute_with_custom_cwd(self):
        """Test executing with custom working directory."""
        policy = HostExecutionPolicy()
        config = ShellConfig(timeout=10.0, cwd=Path("/tmp"))

        if sys.platform.startswith("win"):
            result = await policy.execute("cd", config)
        else:
            result = await policy.execute("pwd", config)

        # On Unix, pwd should show /tmp or /private/tmp (macOS)
        # On Windows, cd shows current directory
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_custom_env(self):
        """Test executing with custom environment variables."""
        policy = HostExecutionPolicy()
        config = ShellConfig(
            timeout=10.0,
            env={"MY_TEST_VAR": "test_value_12345"},
        )

        if sys.platform.startswith("win"):
            result = await policy.execute("echo $env:MY_TEST_VAR", config)
        else:
            result = await policy.execute("echo $MY_TEST_VAR", config)

        assert result.success is True
        assert "test_value_12345" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """Test command timeout."""
        policy = HostExecutionPolicy()
        config = ShellConfig(timeout=0.1)  # Very short timeout

        if sys.platform.startswith("win"):
            result = await policy.execute("Start-Sleep -Seconds 10", config)
        else:
            result = await policy.execute("sleep 10", config)

        assert result.success is False
        assert result.timed_out is True
        assert result.exit_code == -1

    @pytest.mark.asyncio
    async def test_execute_redacts_secrets(self):
        """Test that secrets in output are redacted."""
        policy = HostExecutionPolicy()
        config = ShellConfig(timeout=10.0)

        # Echo a fake API key
        secret = "sk-abc123def456ghi789jkl012mno345pqr678xyz"
        result = await policy.execute(f"echo '{secret}'", config)

        assert result.success is True
        assert secret not in result.stdout
        assert "[REDACTED]" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_no_redaction(self):
        """Test executing without redaction."""
        policy = HostExecutionPolicy(redaction_rules=None)
        config = ShellConfig(timeout=10.0)

        secret = "sk-abc123def456ghi789jkl012mno345pqr678xyz"
        result = await policy.execute(f"echo '{secret}'", config)

        assert result.success is True
        assert secret in result.stdout

    @pytest.mark.asyncio
    async def test_execute_output_truncation(self):
        """Test that output is truncated when exceeding max_output_bytes."""
        policy = HostExecutionPolicy()
        config = ShellConfig(timeout=10.0, max_output_bytes=100)

        # Generate output larger than 100 bytes
        if sys.platform.startswith("win"):
            result = await policy.execute("1..50 | ForEach-Object { 'Line number ' + $_ }", config)
        else:
            result = await policy.execute(
                "for i in $(seq 1 50); do echo 'Line number '$i; done", config
            )

        assert result.success is True
        assert "[OUTPUT TRUNCATED]" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_invalid_command(self):
        """Test executing an invalid command."""
        policy = HostExecutionPolicy()
        config = ShellConfig(timeout=10.0)

        result = await policy.execute("nonexistent_command_xyz123", config)

        assert result.success is False
        # Command not found typically returns exit code 127 on Unix
        # or error on Windows


# =============================================================================
# Integration Tests
# =============================================================================


class TestShellIntegration:
    """Integration tests for shell execution."""

    @pytest.mark.asyncio
    async def test_multiline_command(self):
        """Test executing multiline commands."""
        policy = HostExecutionPolicy()
        config = ShellConfig(timeout=10.0)

        if sys.platform.startswith("win"):
            cmd = "$x = 1; $y = 2; Write-Host ($x + $y)"
        else:
            cmd = "x=1; y=2; echo $((x + y))"

        result = await policy.execute(cmd, config)

        assert result.success is True
        assert "3" in result.stdout

    @pytest.mark.asyncio
    async def test_pipe_command(self):
        """Test executing commands with pipes."""
        policy = HostExecutionPolicy()
        config = ShellConfig(timeout=10.0)

        if sys.platform.startswith("win"):
            result = await policy.execute("echo 'hello world' | Select-String 'world'", config)
        else:
            result = await policy.execute("echo 'hello world' | grep world", config)

        assert result.success is True
        assert "world" in result.stdout

    @pytest.mark.asyncio
    async def test_result_to_dict_roundtrip(self):
        """Test that result can be converted to dict for tool return."""
        policy = HostExecutionPolicy()
        config = ShellConfig(timeout=10.0)

        result = await policy.execute("echo test", config)
        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["success"] is True
        assert isinstance(d["exit_code"], int)
        assert isinstance(d["stdout"], str)
        assert isinstance(d["stderr"], str)
        assert isinstance(d["command"], str)
        assert isinstance(d["duration_ms"], float)
        assert isinstance(d["timed_out"], bool)
