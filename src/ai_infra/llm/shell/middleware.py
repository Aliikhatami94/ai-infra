"""Shell middleware for ai-infra agents.

This module provides `ShellMiddleware` for integrating shell execution
capabilities into agents using the DeepAgents middleware pattern.

The middleware:
- Manages a persistent ShellSession across agent execution
- Provides the `run_shell` tool bound to the session
- Handles startup/shutdown commands
- Applies output redaction for security

Phase 1.4 of EXECUTOR_CLI.md - Shell Tool Integration.

Example:
    ```python
    from ai_infra import Agent
    from ai_infra.llm.shell import ShellMiddleware

    # Create agent with shell capability
    agent = Agent(
        deep=True,
        middleware=[
            ShellMiddleware(
                workspace_root="/path/to/project",
                startup_commands=["source .venv/bin/activate"],
            )
        ],
    )

    result = agent.run("List all Python files and count lines of code")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from langchain_core.tools import BaseTool, tool

from ai_infra.llm.shell.audit import (
    get_shell_audit_logger,
)
from ai_infra.llm.shell.limits import (
    LimitedExecutionPolicy,
    ResourceLimits,
)
from ai_infra.llm.shell.session import SessionConfig, ShellSession
from ai_infra.llm.shell.tool import (
    DANGEROUS_PATTERNS,
    get_current_session,
    set_current_session,
    validate_cwd,
)
from ai_infra.llm.shell.types import (
    DEFAULT_REDACTION_RULES,
    RedactionRule,
    ShellConfig,
    ShellResult,
)

if TYPE_CHECKING:
    import re

    from deepagents.agent import Runtime

__all__ = [
    "ShellMiddleware",
    "ShellMiddlewareConfig",
]


# Type variables for middleware generics
StateT = TypeVar("StateT")
ContextT = TypeVar("ContextT")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ShellMiddlewareConfig:
    """Configuration for ShellMiddleware.

    Attributes:
        workspace_root: Base directory for shell operations.
        startup_commands: Commands to run when session starts.
        shutdown_commands: Commands to run when session ends.
        timeout: Default timeout for commands in seconds.
        max_output_bytes: Maximum bytes to capture from output.
        redaction_rules: Rules for redacting sensitive output.
        dangerous_pattern_check: Whether to reject dangerous commands.
        custom_dangerous_patterns: Additional patterns to check.
        env: Additional environment variables for the session.
        resource_limits: Resource limits for shell execution (Phase 11.1).
        enable_audit: Enable audit logging for commands (Phase 11.3).
        check_suspicious: Check for suspicious patterns (Phase 11.3.4).
    """

    workspace_root: Path | str | None = None
    startup_commands: list[str] = field(default_factory=list)
    shutdown_commands: list[str] = field(default_factory=list)
    timeout: float = 120.0
    max_output_bytes: int = 1_000_000
    redaction_rules: tuple[RedactionRule, ...] | None = DEFAULT_REDACTION_RULES
    dangerous_pattern_check: bool = True
    custom_dangerous_patterns: tuple[re.Pattern[str], ...] | None = None
    env: dict[str, str] | None = None
    resource_limits: ResourceLimits | None = None  # Phase 11.1: Resource limits
    enable_audit: bool = True  # Phase 11.3: Enable audit logging
    check_suspicious: bool = True  # Phase 11.3.4: Check for suspicious patterns


# =============================================================================
# Shell Middleware
# =============================================================================


class ShellMiddleware(Generic[StateT, ContextT]):
    """Middleware that provides shell execution capability to agents.

    This middleware integrates with DeepAgents to provide persistent shell
    sessions for agent execution. The shell session maintains state (working
    directory, environment variables) across tool calls.

    The middleware:
    - Creates a ShellSession when agent starts
    - Provides `run_shell` tool bound to the session
    - Cleans up the session when agent ends
    - Applies output redaction for security

    Example:
        ```python
        from ai_infra import Agent
        from ai_infra.llm.shell import ShellMiddleware

        agent = Agent(
            deep=True,
            middleware=[
                ShellMiddleware(
                    workspace_root="/path/to/project",
                    startup_commands=["cd src", "source .venv/bin/activate"],
                )
            ],
        )

        result = agent.run("Build and test the project")
        ```

    Example with custom config:
        ```python
        from ai_infra.llm.shell import ShellMiddleware, ShellMiddlewareConfig

        config = ShellMiddlewareConfig(
            workspace_root="/project",
            timeout=300.0,
            startup_commands=["export PATH=$PATH:./node_modules/.bin"],
        )

        middleware = ShellMiddleware(config=config)
        ```
    """

    # Middleware interface attributes
    tools: list[BaseTool]

    def __init__(
        self,
        workspace_root: Path | str | None = None,
        startup_commands: list[str] | None = None,
        shutdown_commands: list[str] | None = None,
        timeout: float = 120.0,
        redaction_rules: tuple[RedactionRule, ...] | None = DEFAULT_REDACTION_RULES,
        dangerous_pattern_check: bool = True,
        custom_dangerous_patterns: tuple[re.Pattern[str], ...] | None = None,
        env: dict[str, str] | None = None,
        resource_limits: ResourceLimits | None = None,
        enable_audit: bool = True,
        check_suspicious: bool = True,
        *,
        config: ShellMiddlewareConfig | None = None,
    ) -> None:
        """Initialize ShellMiddleware.

        Args:
            workspace_root: Base directory for shell operations.
            startup_commands: Commands to run when session starts.
            shutdown_commands: Commands to run when session ends.
            timeout: Default timeout for commands in seconds.
            redaction_rules: Rules for redacting sensitive output.
            dangerous_pattern_check: Whether to reject dangerous commands.
            custom_dangerous_patterns: Additional patterns to check.
            env: Additional environment variables for the session.
            resource_limits: Resource limits for shell execution (Phase 11.1).
            enable_audit: Enable audit logging for commands (Phase 11.3).
            check_suspicious: Check for suspicious patterns (Phase 11.3.4).
            config: Full configuration object (overrides other args).
        """
        if config is not None:
            self._config = config
        else:
            self._config = ShellMiddlewareConfig(
                workspace_root=workspace_root,
                startup_commands=startup_commands or [],
                shutdown_commands=shutdown_commands or [],
                timeout=timeout,
                redaction_rules=redaction_rules,
                dangerous_pattern_check=dangerous_pattern_check,
                custom_dangerous_patterns=custom_dangerous_patterns,
                env=env,
                resource_limits=resource_limits,
                enable_audit=enable_audit,
                check_suspicious=check_suspicious,
            )

        # Session will be created in before_agent
        self._session: ShellSession | None = None
        self._context_token: Any = None

        # Create the bound tool
        self.tools = [self._create_shell_tool()]

    @property
    def name(self) -> str:
        """Middleware name for identification."""
        return "ShellMiddleware"

    @property
    def session(self) -> ShellSession | None:
        """Get the current shell session."""
        return self._session

    # =========================================================================
    # Middleware Lifecycle Hooks
    # =========================================================================

    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Synchronous before_agent - not used, see abefore_agent."""
        return None

    async def abefore_agent(
        self,
        state: StateT,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Start shell session before agent execution.

        Creates a new ShellSession and sets it in the context for the
        run_shell tool to use.
        """
        # Build session config
        workspace_root = None
        if self._config.workspace_root:
            workspace_root = Path(self._config.workspace_root).expanduser().resolve()

        session_config = SessionConfig(
            workspace_root=workspace_root,
            startup_commands=self._config.startup_commands,
            shutdown_commands=self._config.shutdown_commands,
            shell_config=ShellConfig(
                timeout=self._config.timeout,
                max_output_bytes=self._config.max_output_bytes,
                env=self._config.env,
            ),
            redaction_rules=self._config.redaction_rules,
            enable_audit=self._config.enable_audit,  # Phase 11.3
            check_suspicious=self._config.check_suspicious,  # Phase 11.3.4
        )

        # Create and start session
        self._session = ShellSession(session_config)
        await self._session.start()

        # Set session in context for tool to use
        self._context_token = set_current_session(self._session)

        return None

    def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Synchronous after_agent - not used, see aafter_agent."""
        return None

    async def aafter_agent(
        self,
        state: StateT,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Clean up shell session after agent execution."""
        if self._session is not None:
            await self._session.close()
            self._session = None

        # Clear context
        if self._context_token is not None:
            set_current_session(None)
            self._context_token = None

        return None

    # =========================================================================
    # Tool Creation
    # =========================================================================

    def _create_shell_tool(self) -> BaseTool:
        """Create the run_shell tool bound to this middleware's session.

        The tool uses the context variable set by the middleware to access
        the session, ensuring it uses the correct session even when multiple
        middlewares are active.
        """
        config = self._config
        patterns = DANGEROUS_PATTERNS
        if config.custom_dangerous_patterns:
            patterns = patterns + config.custom_dangerous_patterns

        @tool
        async def run_shell(
            command: str,
            cwd: str | None = None,
            timeout: int | None = None,
        ) -> dict[str, Any]:
            """Execute a shell command and return the output.

            Use this tool to run shell commands on the host system. Commands are
            executed in a persistent session that maintains working directory and
            environment variables between calls.

            Args:
                command: Shell command to execute. Supports pipes, redirects, etc.
                cwd: Working directory for the command. If not specified, uses the
                    current session's working directory.
                timeout: Maximum seconds to wait for the command to complete.

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
            """
            effective_timeout = float(timeout) if timeout is not None else config.timeout

            # Validate inputs
            if config.dangerous_pattern_check:
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

            # Validate cwd if provided
            if cwd is not None:
                try:
                    validate_cwd(cwd)
                except ValueError as e:
                    return {
                        "success": False,
                        "exit_code": -1,
                        "stdout": "",
                        "stderr": str(e),
                        "command": command,
                        "timed_out": False,
                    }

            # Get session from context
            session = get_current_session()
            if session is not None and session.is_running:
                # Handle cwd by prepending cd command if needed
                if cwd:
                    command = f"cd {cwd} && {command}"
                result = await session.execute(command)
            else:
                # Phase 11.1: Fallback to stateless execution with resource limits
                resolved_cwd = Path(cwd).expanduser().resolve() if cwd else None

                # Phase 11.3.4: Check for suspicious patterns in fallback path
                if config.check_suspicious and config.enable_audit:
                    audit = get_shell_audit_logger()
                    audit.check_and_log_suspicious(command)

                # Use LimitedExecutionPolicy if resource limits are configured
                limits = config.resource_limits
                if limits is not None:
                    policy = LimitedExecutionPolicy(
                        limits=limits,
                        redaction_rules=config.redaction_rules,
                    )
                    shell_config = ShellConfig(
                        timeout=effective_timeout,
                        cwd=resolved_cwd,
                        max_output_bytes=config.max_output_bytes,
                        env=config.env,
                    )
                    result = await policy.execute(command, shell_config)
                else:
                    # No limits configured, use standard stateless execution
                    from ai_infra.llm.shell.tool import _execute_stateless

                    result = await _execute_stateless(command, resolved_cwd, effective_timeout)

                # Phase 11.3: Log command result in fallback path
                if config.enable_audit:
                    audit = get_shell_audit_logger()
                    audit.log_result(result)

            return result.to_dict()

        return run_shell

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def restart_session(self) -> None:
        """Restart the shell session.

        Useful for resetting state when the session becomes corrupted
        or needs a fresh environment.
        """
        if self._session is not None:
            await self._session.restart()

    async def execute(self, command: str) -> ShellResult:
        """Execute a command directly on the session.

        This bypasses the tool interface and executes directly on the
        session. Useful for programmatic access outside the agent loop.

        Args:
            command: Command to execute.

        Returns:
            ShellResult with execution outcome.

        Raises:
            RuntimeError: If session is not started.
        """
        if self._session is None:
            raise RuntimeError("Session not started. Use within agent context.")
        return await self._session.execute(command)
