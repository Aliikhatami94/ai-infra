"""Shell execution types for ai-infra.

This module provides the core types for shell command execution:
- ShellResult: Result of a shell command execution
- ShellConfig: Configuration for shell execution
- ExecutionPolicy: Protocol for execution strategies
- HostExecutionPolicy: Direct host execution
- RedactionRule: Pattern-based output sanitization

Phase 1.1 of EXECUTOR_CLI.md - Shell Tool Integration.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass

__all__ = [
    "ShellResult",
    "ShellConfig",
    "ExecutionPolicy",
    "HostExecutionPolicy",
    "RedactionRule",
    "DEFAULT_REDACTION_RULES",
    "apply_redaction_rules",
]


# =============================================================================
# Shell Result
# =============================================================================


@dataclass(frozen=True, slots=True)
class ShellResult:
    """Result of executing a shell command.

    Attributes:
        success: True if exit code was 0.
        exit_code: The command's exit code (-1 for timeout/error).
        stdout: Standard output from the command.
        stderr: Standard error from the command.
        command: The command that was executed.
        duration_ms: Execution time in milliseconds.
        timed_out: True if the command timed out.

    Example:
        >>> result = ShellResult(
        ...     success=True,
        ...     exit_code=0,
        ...     stdout="Hello, World!",
        ...     stderr="",
        ...     command="echo 'Hello, World!'",
        ...     duration_ms=5.2,
        ... )
        >>> result.success
        True
    """

    success: bool
    exit_code: int
    stdout: str
    stderr: str
    command: str
    duration_ms: float
    timed_out: bool = False

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for tool return value."""
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "command": self.command,
            "duration_ms": self.duration_ms,
            "timed_out": self.timed_out,
        }

    @classmethod
    def from_timeout(cls, command: str, timeout: float) -> ShellResult:
        """Create a result for a timed-out command."""
        return cls(
            success=False,
            exit_code=-1,
            stdout="",
            stderr=f"Command timed out after {timeout} seconds",
            command=command,
            duration_ms=timeout * 1000,
            timed_out=True,
        )

    @classmethod
    def from_error(cls, command: str, error: Exception, duration_ms: float = 0.0) -> ShellResult:
        """Create a result for an execution error."""
        return cls(
            success=False,
            exit_code=-1,
            stdout="",
            stderr=str(error),
            command=command,
            duration_ms=duration_ms,
            timed_out=False,
        )


# =============================================================================
# Shell Configuration
# =============================================================================


@dataclass
class ShellConfig:
    """Configuration for shell command execution.

    Attributes:
        timeout: Maximum seconds to wait for command (default: 120).
        max_output_bytes: Maximum bytes to capture from stdout/stderr (default: 1MB).
        shell: Shell executable to use (default: platform-appropriate).
        env: Environment variables to set (merged with os.environ).
        cwd: Working directory for command execution.

    Example:
        >>> config = ShellConfig(timeout=60.0, cwd=Path("/tmp"))
        >>> config.timeout
        60.0
    """

    timeout: float = 120.0
    max_output_bytes: int = 1_000_000  # 1MB
    shell: str = field(
        default_factory=lambda: "powershell" if sys.platform.startswith("win") else "/bin/bash"
    )
    env: dict[str, str] | None = None
    cwd: Path | None = None

    def get_env(self) -> dict[str, str]:
        """Get environment variables, merged with os.environ."""
        base_env = dict(os.environ)
        if self.env:
            base_env.update(self.env)
        return base_env

    def get_cwd(self) -> Path:
        """Get working directory, defaulting to current directory."""
        return self.cwd or Path.cwd()


# =============================================================================
# Redaction Rules
# =============================================================================


@dataclass(frozen=True, slots=True)
class RedactionRule:
    """Rule for redacting sensitive information from shell output.

    Attributes:
        name: Human-readable name for the rule (for logging).
        pattern: Regex pattern to match sensitive content.
        replacement: Text to replace matched content with.

    Example:
        >>> rule = RedactionRule("api_key", r"sk-[a-zA-Z0-9]{32,}")
        >>> rule.apply("Token: sk-abc123def456ghi789jkl012mno345pqr")
        'Token: [REDACTED]'
    """

    name: str
    pattern: str
    replacement: str = "[REDACTED]"

    def apply(self, text: str) -> str:
        """Apply this redaction rule to text."""
        return re.sub(self.pattern, self.replacement, text)


# Default redaction rules for common secrets
DEFAULT_REDACTION_RULES: tuple[RedactionRule, ...] = (
    RedactionRule("openai_api_key", r"sk-[a-zA-Z0-9]{32,}"),
    RedactionRule("anthropic_api_key", r"sk-ant-[a-zA-Z0-9\-]{32,}"),
    RedactionRule(
        "aws_secret_key",
        r"(?i)aws[_-]?secret[_-]?access[_-]?key[=:\s]+['\"]?[A-Za-z0-9/+=]{40}['\"]?",
    ),
    RedactionRule(
        "aws_access_key", r"(?i)aws[_-]?access[_-]?key[_-]?id[=:\s]+['\"]?[A-Z0-9]{20}['\"]?"
    ),
    RedactionRule("generic_api_key", r"(?i)api[_-]?key[=:\s]+['\"]?[a-zA-Z0-9\-_]{20,}['\"]?"),
    RedactionRule("generic_secret", r"(?i)secret[=:\s]+['\"]?[a-zA-Z0-9\-_]{20,}['\"]?"),
    RedactionRule("generic_password", r"(?i)password[=:\s]+['\"]?[^\s'\"]{8,}['\"]?"),
    RedactionRule("bearer_token", r"(?i)bearer\s+[a-zA-Z0-9\-_.]+"),
    RedactionRule("basic_auth", r"(?i)basic\s+[a-zA-Z0-9+/=]+"),
    RedactionRule("private_key_header", r"-----BEGIN[A-Z ]*PRIVATE KEY-----"),
)


def apply_redaction_rules(text: str, rules: tuple[RedactionRule, ...] | None = None) -> str:
    """Apply redaction rules to text.

    Args:
        text: Text to redact.
        rules: Rules to apply. Defaults to DEFAULT_REDACTION_RULES.

    Returns:
        Text with sensitive content redacted.
    """
    if rules is None:
        rules = DEFAULT_REDACTION_RULES
    for rule in rules:
        text = rule.apply(text)
    return text


# =============================================================================
# Execution Policy Protocol
# =============================================================================


@runtime_checkable
class ExecutionPolicy(Protocol):
    """Protocol for shell command execution strategies.

    Implementations define how commands are executed:
    - HostExecutionPolicy: Direct execution on the host
    - DockerExecutionPolicy: Execution in a Docker container (future)
    - SandboxExecutionPolicy: Restricted execution (future)
    """

    async def execute(self, command: str, config: ShellConfig) -> ShellResult:
        """Execute a shell command.

        Args:
            command: The shell command to execute.
            config: Configuration for execution.

        Returns:
            ShellResult with execution outcome.
        """
        ...


# =============================================================================
# Host Execution Policy
# =============================================================================


class HostExecutionPolicy:
    """Execute shell commands directly on the host system.

    This policy provides direct access to the host shell. Use with caution
    in production environments - prefer DockerExecutionPolicy for isolation.

    Example:
        >>> policy = HostExecutionPolicy()
        >>> result = await policy.execute("echo hello", ShellConfig())
        >>> result.stdout
        'hello'
    """

    def __init__(
        self,
        *,
        redaction_rules: tuple[RedactionRule, ...] | None = DEFAULT_REDACTION_RULES,
    ) -> None:
        """Initialize host execution policy.

        Args:
            redaction_rules: Rules for redacting sensitive output.
                Set to None to disable redaction (not recommended).
        """
        self._redaction_rules = redaction_rules

    async def execute(self, command: str, config: ShellConfig) -> ShellResult:
        """Execute a shell command on the host.

        Args:
            command: The shell command to execute.
            config: Configuration for execution.

        Returns:
            ShellResult with execution outcome.
        """
        start_time = time.perf_counter()

        try:
            # Build shell command based on platform
            if sys.platform.startswith("win"):
                shell_args = [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    command,
                ]
            else:
                shell_args = ["bash", "-lc", command]

            # Create subprocess
            proc = await asyncio.create_subprocess_exec(
                *shell_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(config.get_cwd()),
                env=config.get_env(),
            )

            # Wait for completion with timeout
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=config.timeout,
                )
            except TimeoutError:
                # Kill the process on timeout
                proc.kill()
                await proc.wait()
                return ShellResult.from_timeout(command, config.timeout)

            # Decode output
            stdout = (stdout_bytes or b"").decode(errors="replace")
            stderr = (stderr_bytes or b"").decode(errors="replace")

            # Truncate if needed
            if len(stdout) > config.max_output_bytes:
                stdout = stdout[: config.max_output_bytes] + "\n[OUTPUT TRUNCATED]"
            if len(stderr) > config.max_output_bytes:
                stderr = stderr[: config.max_output_bytes] + "\n[OUTPUT TRUNCATED]"

            # Apply redaction
            if self._redaction_rules:
                stdout = apply_redaction_rules(stdout, self._redaction_rules)
                stderr = apply_redaction_rules(stderr, self._redaction_rules)

            duration_ms = (time.perf_counter() - start_time) * 1000
            exit_code = proc.returncode or 0

            return ShellResult(
                success=exit_code == 0,
                exit_code=exit_code,
                stdout=stdout.strip(),
                stderr=stderr.strip(),
                command=command,
                duration_ms=duration_ms,
                timed_out=False,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ShellResult.from_error(command, e, duration_ms)
