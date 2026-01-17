"""Shell session manager for persistent command execution.

This module provides ShellSession for maintaining persistent shell sessions
that preserve state (environment variables, working directory) between commands.

Key features:
- Persistent bash/powershell process
- Command delimiter protocol for reliable output parsing
- Exit code extraction
- Startup/shutdown command support
- Output truncation and redaction
- Async context manager support
- Audit logging for all commands (Phase 11.3)

Phase 1.2 of EXECUTOR_CLI.md - Shell Tool Integration.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ai_infra.llm.shell.types import (
    DEFAULT_REDACTION_RULES,
    RedactionRule,
    ShellConfig,
    ShellResult,
    apply_redaction_rules,
)

if TYPE_CHECKING:
    from ai_infra.llm.shell.audit import ShellAuditLogger

logger = logging.getLogger(__name__)

__all__ = ["ShellSession", "SessionConfig"]


# =============================================================================
# Session Configuration
# =============================================================================


@dataclass
class SessionConfig:
    """Configuration for a shell session.

    Attributes:
        workspace_root: Base directory for the shell session.
        startup_commands: Commands to run after session starts.
        shutdown_commands: Commands to run before session closes.
        shell_config: Configuration for command execution.
        redaction_rules: Rules for sanitizing output.
        idle_timeout: Seconds before idle session is closed (0 = never).
        enable_audit: Enable audit logging for commands (Phase 11.3).
        check_suspicious: Check for suspicious patterns (Phase 11.3.4).

    Example:
        >>> config = SessionConfig(
        ...     workspace_root=Path("/tmp/project"),
        ...     startup_commands=["source .venv/bin/activate"],
        ... )
    """

    workspace_root: Path | None = None
    startup_commands: list[str] = field(default_factory=list)
    shutdown_commands: list[str] = field(default_factory=list)
    shell_config: ShellConfig = field(default_factory=ShellConfig)
    redaction_rules: tuple[RedactionRule, ...] | None = DEFAULT_REDACTION_RULES
    idle_timeout: float = 0.0  # 0 = no timeout
    enable_audit: bool = True  # Phase 11.3: Enable audit logging
    check_suspicious: bool = True  # Phase 11.3.4: Check for suspicious patterns


# =============================================================================
# Shell Session
# =============================================================================


class ShellSession:
    """Persistent shell session that maintains state between commands.

    The session spawns a single shell process and sends commands through it,
    preserving environment variables, working directory, and other state.

    Uses a delimiter protocol to reliably parse command output and exit codes:
    1. Send command followed by delimiter markers
    2. Parse output between markers
    3. Extract exit code from final marker

    Example:
        >>> async with ShellSession() as session:
        ...     result = await session.execute("cd /tmp")
        ...     result = await session.execute("pwd")
        ...     print(result.stdout)  # /tmp

    Example with config:
        >>> config = SessionConfig(
        ...     workspace_root=Path("/project"),
        ...     startup_commands=["export PYTHONPATH=/project"],
        ... )
        >>> async with ShellSession(config) as session:
        ...     result = await session.execute("echo $PYTHONPATH")
    """

    # Delimiter for parsing command output
    _DELIMITER_PREFIX = "___AI_INFRA_SHELL_DELIMITER___"

    def __init__(self, config: SessionConfig | None = None) -> None:
        """Initialize shell session.

        Args:
            config: Session configuration. Defaults to SessionConfig().
        """
        self._config = config or SessionConfig()
        self._process: asyncio.subprocess.Process | None = None
        self._started = False
        self._closed = False
        self._lock = asyncio.Lock()
        self._last_activity = time.monotonic()
        # Phase 2.3.2: Track command history for executor state
        self._command_history: list[ShellResult] = []
        # Phase 11.3: Session ID for audit correlation
        self._session_id = uuid.uuid4().hex[:12]
        # Phase 11.3: Audit logger (lazy initialized)
        self._audit_logger: ShellAuditLogger | None = None

    @property
    def is_running(self) -> bool:
        """Check if the session is running."""
        return self._started and not self._closed and self._process is not None

    @property
    def command_history(self) -> list[ShellResult]:
        """Get the history of commands executed in this session.

        Phase 2.3.2: Provides access to command history for executor state tracking.

        Returns:
            List of ShellResult objects from commands executed in this session.
        """
        return list(self._command_history)

    @property
    def session_id(self) -> str:
        """Get the unique session ID for audit correlation (Phase 11.3)."""
        return self._session_id

    def _get_audit_logger(self) -> ShellAuditLogger | None:
        """Get audit logger if auditing is enabled (Phase 11.3).

        Lazy initializes the logger on first access to avoid import overhead.
        """
        if not self._config.enable_audit:
            return None
        if self._audit_logger is None:
            from ai_infra.llm.shell.audit import get_shell_audit_logger

            self._audit_logger = get_shell_audit_logger()
        return self._audit_logger

    def clear_history(self) -> None:
        """Clear the command history.

        Phase 2.3.2: Useful for resetting history between tasks.
        """
        self._command_history.clear()

    async def start(self) -> None:
        """Start the shell session.

        Spawns the shell process and runs any startup commands.

        Raises:
            RuntimeError: If session is already started or closed.
        """
        if self._closed:
            raise RuntimeError("Session has been closed")
        if self._started:
            raise RuntimeError("Session already started")

        async with self._lock:
            await self._spawn_process()
            self._started = True

            # Phase 11.3: Log session start
            if audit := self._get_audit_logger():
                audit.log_session_started(
                    session_id=self._session_id,
                    config={
                        "workspace_root": str(self._config.workspace_root)
                        if self._config.workspace_root
                        else None,
                        "startup_commands_count": len(self._config.startup_commands),
                    },
                )
            # Run startup commands
            for cmd in self._config.startup_commands:
                await self._execute_internal(cmd)

    async def execute(self, command: str) -> ShellResult:
        """Execute a command in the session.

        Args:
            command: Shell command to execute.

        Returns:
            ShellResult with execution outcome.

        Raises:
            RuntimeError: If session is not started or is closed.
        """
        if not self._started:
            raise RuntimeError("Session not started. Call start() first or use as context manager.")
        if self._closed:
            raise RuntimeError("Session has been closed")

        async with self._lock:
            self._last_activity = time.monotonic()

            # Phase 11.3.4: Check for suspicious patterns before execution
            if self._config.check_suspicious:
                if audit := self._get_audit_logger():
                    audit.check_and_log_suspicious(command, session_id=self._session_id)

            result = await self._execute_internal(command)

            # Phase 11.3: Audit log command execution
            if audit := self._get_audit_logger():
                audit.log_result(result, session_id=self._session_id)

            # Phase 2.3.2: Track command in history for executor state
            self._command_history.append(result)
            return result

    async def restart(self) -> None:
        """Restart the shell session.

        Closes the current process and starts a new one.
        Startup commands are re-executed.
        """
        async with self._lock:
            if self._process is not None:
                await self._kill_process()

            self._closed = False
            await self._spawn_process()

            # Re-run startup commands
            for cmd in self._config.startup_commands:
                await self._execute_internal(cmd)

    async def close(self) -> None:
        """Close the shell session.

        Runs shutdown commands and terminates the shell process.
        """
        if self._closed:
            return

        async with self._lock:
            self._closed = True

            if self._process is not None:
                # Run shutdown commands (ignore errors)
                for cmd in self._config.shutdown_commands:
                    try:
                        await self._execute_internal(cmd)
                    except Exception:
                        pass

                await self._kill_process()

            # Phase 11.3: Log session end
            if audit := self._get_audit_logger():
                audit.log_session_ended(
                    session_id=self._session_id,
                    command_count=len(self._command_history),
                )

    async def __aenter__(self) -> ShellSession:
        """Enter async context manager."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        await self.close()

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _spawn_process(self) -> None:
        """Spawn the shell subprocess."""
        # Determine shell and args based on platform
        if sys.platform.startswith("win"):
            shell_cmd = [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "-",  # Read from stdin
            ]
        else:
            # Use non-interactive bash to avoid prompt output that interferes with parsing
            shell_cmd = ["bash", "--norc", "--noprofile"]

        # Set working directory
        cwd = self._config.workspace_root or self._config.shell_config.cwd or Path.cwd()

        # Build environment
        env = dict(os.environ)
        if self._config.shell_config.env:
            env.update(self._config.shell_config.env)

        # Spawn process
        self._process = await asyncio.create_subprocess_exec(
            *shell_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
            env=env,
        )

    async def _kill_process(self) -> None:
        """Kill the shell subprocess."""
        if self._process is None:
            return

        try:
            self._process.kill()
            await asyncio.wait_for(self._process.wait(), timeout=5.0)
        except (TimeoutError, ProcessLookupError):
            pass
        finally:
            self._process = None

    async def _execute_internal(self, command: str) -> ShellResult:
        """Execute a command and parse the result.

        Uses delimiter protocol:
        1. Generate unique delimiter
        2. Send command with exit code capture and delimiter output
        3. Read until delimiter is found
        4. Parse stdout, stderr, and exit code
        """
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Shell process not available")

        start_time = time.perf_counter()
        delimiter = f"{self._DELIMITER_PREFIX}{uuid.uuid4().hex}"
        timeout = self._config.shell_config.timeout
        max_output = self._config.shell_config.max_output_bytes

        # Build command with delimiter and exit code capture
        if sys.platform.startswith("win"):
            # PowerShell: capture exit code and output delimiter
            wrapped_cmd = f"""
{command}
$LASTEXITCODE_CAPTURED = $LASTEXITCODE
if ($LASTEXITCODE_CAPTURED -eq $null) {{ $LASTEXITCODE_CAPTURED = 0 }}
Write-Host ""
Write-Host "{delimiter}_EXIT_$LASTEXITCODE_CAPTURED"
"""
        else:
            # Bash: capture exit code and output delimiter
            wrapped_cmd = f"""
{command}
__exit_code__=$?
echo ""
echo "{delimiter}_EXIT_${{__exit_code__}}"
"""

        try:
            # Send command
            self._process.stdin.write(wrapped_cmd.encode() + b"\n")
            await self._process.stdin.drain()

            # Read output until delimiter
            stdout_data, stderr_data, exit_code = await asyncio.wait_for(
                self._read_until_delimiter(delimiter, max_output),
                timeout=timeout,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Apply truncation markers if needed
            stdout_truncated = len(stdout_data) >= max_output
            stderr_truncated = len(stderr_data) >= max_output

            if stdout_truncated:
                stdout_data += "\n[OUTPUT TRUNCATED]"
            if stderr_truncated:
                stderr_data += "\n[OUTPUT TRUNCATED]"

            # Apply redaction
            if self._config.redaction_rules:
                stdout_data = apply_redaction_rules(stdout_data, self._config.redaction_rules)
                stderr_data = apply_redaction_rules(stderr_data, self._config.redaction_rules)

            return ShellResult(
                success=exit_code == 0,
                exit_code=exit_code,
                stdout=stdout_data.strip(),
                stderr=stderr_data.strip(),
                command=command,
                duration_ms=duration_ms,
                timed_out=False,
            )

        except TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            # Kill and restart the session on timeout
            await self._kill_process()
            await self._spawn_process()
            self._started = True

            return ShellResult.from_timeout(command, timeout)

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ShellResult.from_error(command, e, duration_ms)

    async def _read_until_delimiter(
        self,
        delimiter: str,
        max_output: int,
    ) -> tuple[str, str, int]:
        """Read output until delimiter is found.

        Returns:
            Tuple of (stdout, stderr, exit_code)
        """
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("Shell process not available")

        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        stdout_size = 0
        stderr_size = 0
        exit_code = 0

        # Pattern to match delimiter with exit code
        delimiter_pattern = re.compile(rf"{re.escape(delimiter)}_EXIT_(-?\d+)")

        # Safety: limit iterations to prevent infinite loops
        # With 0.1s timeout per iteration, 3000 iterations = ~5 min max
        max_iterations = 3000
        iterations = 0

        while iterations < max_iterations:
            iterations += 1
            # Read from stdout
            try:
                chunk = await asyncio.wait_for(
                    self._process.stdout.readline(),
                    timeout=0.1,
                )
                if chunk:
                    line = chunk.decode(errors="replace")

                    # Check for delimiter
                    match = delimiter_pattern.search(line)
                    if match:
                        exit_code = int(match.group(1))
                        # Add any content before the delimiter
                        before_delimiter = line[: match.start()]
                        if before_delimiter.strip() and stdout_size < max_output:
                            stdout_chunks.append(before_delimiter.encode())
                            stdout_size += len(before_delimiter)
                        break

                    # Accumulate output (with truncation)
                    if stdout_size < max_output:
                        stdout_chunks.append(chunk)
                        stdout_size += len(chunk)

            except TimeoutError:
                # Check if process is still alive
                if self._process.returncode is not None:
                    break
                continue

            # Also drain stderr (non-blocking)
            if self._process.stderr is not None:
                try:
                    stderr_chunk = await asyncio.wait_for(
                        self._process.stderr.read(4096),
                        timeout=0.01,
                    )
                    if stderr_chunk and stderr_size < max_output:
                        stderr_chunks.append(stderr_chunk)
                        stderr_size += len(stderr_chunk)
                except TimeoutError:
                    pass

        stdout = b"".join(stdout_chunks).decode(errors="replace")
        stderr = b"".join(stderr_chunks).decode(errors="replace")

        return stdout, stderr, exit_code
