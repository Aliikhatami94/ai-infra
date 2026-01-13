"""Shell execution tool for ai-infra agents.

This module provides the `run_shell` tool for executing shell commands
within an agent context. The tool integrates with ShellSession for
persistent execution and falls back to stateless execution when needed.

Features:
- Session-aware execution (maintains working directory, environment)
- Input validation and dangerous command detection
- Structured error handling (timeout, permission denied, etc.)
- Output redaction for sensitive data

Phase 1.3 of EXECUTOR_CLI.md - Shell Tool Integration.

Example:
    ```python
    from ai_infra import Agent
    from ai_infra.llm.shell.tool import run_shell

    agent = Agent(tools=[run_shell])
    result = agent.run("List the files in the current directory")
    ```
"""

from __future__ import annotations

import contextvars
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.tools import tool

from ai_infra.llm.shell.types import (
    DEFAULT_REDACTION_RULES,
    HostExecutionPolicy,
    ShellConfig,
    ShellResult,
)

if TYPE_CHECKING:
    from ai_infra.llm.shell.session import ShellSession

__all__ = [
    "run_shell",
    "create_shell_tool",
    "get_current_session",
    "set_current_session",
    "DANGEROUS_PATTERNS",
]


# =============================================================================
# Context Variables for Session Management
# =============================================================================

# Context variable to hold the current shell session (set by middleware)
_current_session: contextvars.ContextVar[ShellSession | None] = contextvars.ContextVar(
    "shell_session",
    default=None,
)


def get_current_session() -> ShellSession | None:
    """Get the current shell session from context.

    Returns:
        The current ShellSession if set by middleware, None otherwise.
    """
    return _current_session.get()


def set_current_session(session: ShellSession | None) -> contextvars.Token[ShellSession | None]:
    """Set the current shell session in context.

    Args:
        session: The ShellSession to set, or None to clear.

    Returns:
        Token that can be used to reset the context variable.
    """
    return _current_session.set(session)


# =============================================================================
# Dangerous Command Patterns
# =============================================================================

# Patterns for commands that should be rejected or flagged
DANGEROUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Destructive commands
    re.compile(r"\brm\s+(-[rRf]+\s+)?/\s*$", re.IGNORECASE),  # rm -rf /
    re.compile(r"\brm\s+(-[rRf]+\s+)?/\*", re.IGNORECASE),  # rm -rf /*
    re.compile(r"\brm\s+(-[rRf]+\s+)?~\s*$", re.IGNORECASE),  # rm -rf ~
    re.compile(r":\s*\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.IGNORECASE),  # Fork bomb
    re.compile(r"\bmkfs\b", re.IGNORECASE),  # Format filesystem
    re.compile(r"\bdd\s+.*of=/dev/", re.IGNORECASE),  # dd to device
    # Network exfiltration
    re.compile(r"\bcurl\s+.*\|\s*bash", re.IGNORECASE),  # curl | bash
    re.compile(r"\bwget\s+.*\|\s*bash", re.IGNORECASE),  # wget | bash
    re.compile(r"\bwget\s+.*\|\s*sh\b", re.IGNORECASE),  # wget | sh
    # Privilege escalation (commented out - may be needed for legitimate use)
    # re.compile(r"\bsudo\s+", re.IGNORECASE),  # sudo
    # re.compile(r"\bsu\s+-", re.IGNORECASE),  # su -
    # System modification
    re.compile(r">\s*/etc/passwd\b", re.IGNORECASE),  # Overwrite passwd
    re.compile(r">\s*/etc/shadow\b", re.IGNORECASE),  # Overwrite shadow
    re.compile(r"\bchmod\s+777\s+/", re.IGNORECASE),  # chmod 777 /
)


def is_dangerous_command(command: str) -> tuple[bool, str | None]:
    """Check if a command matches dangerous patterns.

    Args:
        command: Shell command to check.

    Returns:
        Tuple of (is_dangerous, reason). If not dangerous, reason is None.
    """
    for pattern in DANGEROUS_PATTERNS:
        if pattern.search(command):
            return True, f"Command matches dangerous pattern: {pattern.pattern}"
    return False, None


def validate_cwd(cwd: str | None) -> Path | None:
    """Validate and resolve working directory path.

    Args:
        cwd: Working directory path to validate.

    Returns:
        Resolved Path if valid, None if cwd was None.

    Raises:
        ValueError: If path does not exist or is not a directory.
    """
    if cwd is None:
        return None

    path = Path(cwd).expanduser().resolve()

    if not path.exists():
        raise ValueError(f"Working directory does not exist: {cwd}")
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {cwd}")

    return path


# =============================================================================
# Shell Tool
# =============================================================================


@tool
async def run_shell(
    command: str,
    cwd: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """Execute a shell command and return the output.

    Use this tool to run shell commands on the host system. Commands are
    executed in a persistent session when available, maintaining working
    directory and environment variables between calls.

    Args:
        command: Shell command to execute. Supports pipes, redirects, etc.
        cwd: Working directory for the command. If not specified, uses the
            current session's working directory or the current directory.
        timeout: Maximum seconds to wait for the command to complete.
            Default is 120 seconds.

    Returns:
        Dictionary with execution result:
        - success: Whether the command succeeded (exit code 0)
        - exit_code: The command's exit code
        - stdout: Standard output from the command
        - stderr: Standard error from the command
        - command: The command that was executed
        - timed_out: Whether the command timed out

    Example:
        # List files
        run_shell("ls -la")

        # Run in specific directory
        run_shell("npm install", cwd="/path/to/project")

        # Chain commands
        run_shell("cd /tmp && ls -la")

        # With timeout
        run_shell("long_running_script.sh", timeout=300)
    """
    # Validate inputs
    is_dangerous, reason = is_dangerous_command(command)
    if is_dangerous:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Command rejected: {reason}",
            "command": command,
            "timed_out": False,
        }

    try:
        resolved_cwd = validate_cwd(cwd)
    except ValueError as e:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
            "command": command,
            "timed_out": False,
        }

    # Try to use session if available
    session = get_current_session()
    if session is not None and session.is_running:
        result = await _execute_with_session(session, command, timeout)
    else:
        result = await _execute_stateless(command, resolved_cwd, timeout)

    return result.to_dict()


async def _execute_with_session(
    session: ShellSession,
    command: str,
    timeout: float,
) -> ShellResult:
    """Execute command using an existing session.

    Args:
        session: Active ShellSession to use.
        command: Command to execute.
        timeout: Timeout in seconds.

    Returns:
        ShellResult with execution outcome.
    """
    # Note: cwd changes are handled by the session itself
    # If user specifies cwd, they should use 'cd' in the command
    return await session.execute(command)


async def _execute_stateless(
    command: str,
    cwd: Path | None,
    timeout: float,
) -> ShellResult:
    """Execute command without a session (one-shot execution).

    Args:
        command: Command to execute.
        cwd: Working directory, or None for current directory.
        timeout: Timeout in seconds.

    Returns:
        ShellResult with execution outcome.
    """
    policy = HostExecutionPolicy(redaction_rules=DEFAULT_REDACTION_RULES)
    config = ShellConfig(
        timeout=timeout,
        cwd=cwd,
    )

    return await policy.execute(command, config)


# =============================================================================
# Tool Factory
# =============================================================================


def create_shell_tool(
    *,
    session: ShellSession | None = None,
    dangerous_pattern_check: bool = True,
    custom_dangerous_patterns: tuple[re.Pattern[str], ...] | None = None,
    default_timeout: float = 120.0,
    default_cwd: Path | str | None = None,
    allowed_commands: tuple[str, ...] | None = None,
) -> Any:
    """Create a configured shell tool.

    Factory function for creating shell tools with custom configuration.
    Useful when you need multiple tools with different settings.

    Args:
        session: Optional ShellSession to bind to the tool.
        dangerous_pattern_check: Whether to check for dangerous commands.
        custom_dangerous_patterns: Additional dangerous patterns to check.
        default_timeout: Default timeout for commands.
        default_cwd: Default working directory.
        allowed_commands: Optional allowlist of permitted command prefixes (Phase 2.4).
            If set, only commands starting with these prefixes are allowed.
            Example: ("pytest", "npm", "make") allows "pytest -v", "npm install", etc.

    Returns:
        Configured shell tool.

    Example:
        ```python
        # Create tool with custom settings
        my_shell = create_shell_tool(
            default_timeout=300.0,
            default_cwd=Path("/projects/myapp"),
        )

        # Create tool with allowlist (Phase 2.4)
        safe_shell = create_shell_tool(
            allowed_commands=("pytest", "npm", "make"),
        )

        # Use with agent
        agent = Agent(tools=[my_shell])
        ```
    """
    patterns = DANGEROUS_PATTERNS
    if custom_dangerous_patterns:
        patterns = patterns + custom_dangerous_patterns

    @tool
    async def configured_run_shell(
        command: str,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Execute a shell command and return the output.

        Use this tool to run shell commands on the host system.

        Args:
            command: Shell command to execute.
            cwd: Working directory for the command.
            timeout: Maximum seconds to wait.

        Returns:
            Dictionary with success, exit_code, stdout, stderr, command.
        """
        effective_timeout = float(timeout) if timeout is not None else default_timeout

        # Phase 2.4: Check allowlist if configured
        if allowed_commands is not None:
            # Extract the base command (first word)
            cmd_parts = command.strip().split()
            base_cmd = cmd_parts[0] if cmd_parts else ""

            # Check if command starts with any allowed prefix
            is_allowed = any(
                base_cmd == allowed or base_cmd.startswith(f"{allowed} ")
                for allowed in allowed_commands
            )
            if not is_allowed:
                allowed_list = ", ".join(allowed_commands)
                return {
                    "success": False,
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": f"Command rejected: '{base_cmd}' not in allowlist ({allowed_list})",
                    "command": command,
                    "timed_out": False,
                }

        # Validate inputs
        if dangerous_pattern_check:
            for pattern in patterns:
                if pattern.search(command):
                    return {
                        "success": False,
                        "exit_code": -1,
                        "stdout": "",
                        "stderr": "Command rejected: matches dangerous pattern",
                        "command": command,
                        "timed_out": False,
                    }

        # Resolve cwd
        effective_cwd: Path | None = None
        if cwd is not None:
            try:
                effective_cwd = validate_cwd(cwd)
            except ValueError as e:
                return {
                    "success": False,
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": str(e),
                    "command": command,
                    "timed_out": False,
                }
        elif default_cwd is not None:
            effective_cwd = Path(default_cwd).expanduser().resolve()

        # Execute
        if session is not None and session.is_running:
            result = await session.execute(command)
        else:
            result = await _execute_stateless(command, effective_cwd, effective_timeout)

        return result.to_dict()

    return configured_run_shell
