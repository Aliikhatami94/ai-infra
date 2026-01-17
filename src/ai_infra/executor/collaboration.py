"""Collaboration modes for human-agent interaction.

Phase 4.2 of EXECUTOR_1.md: Collaboration Modes.

This module provides different levels of autonomy for different situations:
- AUTONOMOUS: Agent works independently, only pausing on dangerous operations
- SUPERVISED: Agent pauses on risky operations (file deletes, config changes)
- PAIR: Human reviews each task before proceeding
- GUIDED: Human provides direction, agent executes step by step

Example:
    ```python
    from ai_infra.executor.collaboration import (
        CollaborationMode,
        CollaborationConfig,
        ModeAwareExecutor,
    )

    # Create config for supervised mode
    config = CollaborationConfig.for_mode(CollaborationMode.SUPERVISED)

    # Or customize
    config = CollaborationConfig(
        mode=CollaborationMode.SUPERVISED,
        pause_on_file_delete=True,
        pause_on_config_change=True,
        require_approval_for=["*.py", "config/*"],
    )

    # Check if action should pause
    executor = ModeAwareExecutor(config)
    if await executor.should_pause(action, context):
        # Request human input
        pass
    ```
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_infra.executor.agents.base import ExecutionContext

logger = logging.getLogger(__name__)


# =============================================================================
# Collaboration Modes (Phase 4.2.1)
# =============================================================================


class CollaborationMode(Enum):
    """Collaboration mode defining level of human-agent interaction.

    Phase 4.2.1: Different levels of autonomy for different situations.

    Modes:
        AUTONOMOUS: Agent works independently to completion.
            - Only pauses on dangerous operations (rm -rf, DROP TABLE)
            - Best for: trusted tasks, routine work, experienced users

        SUPERVISED: Agent pauses on risky operations.
            - Pauses on file deletes, config changes, dependency additions
            - Best for: production code, unfamiliar codebases

        PAIR: Human reviews each task.
            - Pauses after every task completion for review
            - Best for: learning new codebases, critical code

        GUIDED: Human provides direction at each step.
            - Explains before every action, waits for approval
            - Best for: teaching, debugging, high-risk operations
    """

    AUTONOMOUS = "autonomous"
    """Run to completion with minimal interruption."""

    SUPERVISED = "supervised"
    """Pause on risky operations for human review."""

    PAIR = "pair"
    """Human reviews each task before proceeding."""

    GUIDED = "guided"
    """Human provides direction, agent executes step by step."""


# =============================================================================
# Collaboration Configuration (Phase 4.2.1)
# =============================================================================


@dataclass
class CollaborationConfig:
    """Configuration for human-agent collaboration behavior.

    Phase 4.2.1: Configurable pause conditions and approval requirements.

    Attributes:
        mode: The collaboration mode (autonomous, supervised, pair, guided).
        pause_on_file_delete: Pause when deleting files.
        pause_on_config_change: Pause when modifying config files.
        pause_on_dependency_add: Pause when adding new dependencies.
        pause_on_shell_command: Pause before running shell commands.
        explain_before_action: Explain reasoning before taking action.
        require_approval_for: Glob patterns for files requiring approval.
        dangerous_patterns: Patterns that always require pause.

    Example:
        ```python
        # Use preset for mode
        config = CollaborationConfig.for_mode(CollaborationMode.SUPERVISED)

        # Customize specific settings
        config = CollaborationConfig(
            mode=CollaborationMode.SUPERVISED,
            pause_on_file_delete=True,
            require_approval_for=["*.py", "config/*", "*.sql"],
        )
        ```
    """

    mode: CollaborationMode
    pause_on_file_delete: bool = True
    pause_on_config_change: bool = True
    pause_on_dependency_add: bool = True
    pause_on_shell_command: bool = False
    explain_before_action: bool = False
    require_approval_for: list[str] = field(default_factory=list)
    dangerous_patterns: list[str] = field(
        default_factory=lambda: [
            "rm -rf",
            "rm -r",
            "rmdir",
            "DROP TABLE",
            "DROP DATABASE",
            "DELETE FROM",
            "TRUNCATE",
            "git push --force",
            "git push -f",
            "git reset --hard",
            "chmod 777",
            "sudo rm",
            "> /dev/",
            "mkfs",
            "dd if=",
        ]
    )

    @classmethod
    def for_mode(cls, mode: CollaborationMode) -> CollaborationConfig:
        """Get default configuration for a collaboration mode.

        Args:
            mode: The collaboration mode to get config for.

        Returns:
            CollaborationConfig with appropriate defaults for the mode.

        Example:
            ```python
            # Get supervised mode defaults
            config = CollaborationConfig.for_mode(CollaborationMode.SUPERVISED)
            assert config.pause_on_config_change is True
            ```
        """
        configs = {
            CollaborationMode.AUTONOMOUS: cls(
                mode=mode,
                pause_on_file_delete=True,  # Still pause on deletes
                pause_on_config_change=False,
                pause_on_dependency_add=False,
                pause_on_shell_command=False,
                explain_before_action=False,
            ),
            CollaborationMode.SUPERVISED: cls(
                mode=mode,
                pause_on_file_delete=True,
                pause_on_config_change=True,
                pause_on_dependency_add=True,
                pause_on_shell_command=True,
                explain_before_action=False,
            ),
            CollaborationMode.PAIR: cls(
                mode=mode,
                pause_on_file_delete=True,
                pause_on_config_change=True,
                pause_on_dependency_add=True,
                pause_on_shell_command=True,
                explain_before_action=True,
            ),
            CollaborationMode.GUIDED: cls(
                mode=mode,
                pause_on_file_delete=True,
                pause_on_config_change=True,
                pause_on_dependency_add=True,
                pause_on_shell_command=True,
                explain_before_action=True,
            ),
        }
        return configs[mode]

    @classmethod
    def autonomous(cls) -> CollaborationConfig:
        """Shortcut for autonomous mode config."""
        return cls.for_mode(CollaborationMode.AUTONOMOUS)

    @classmethod
    def supervised(cls) -> CollaborationConfig:
        """Shortcut for supervised mode config."""
        return cls.for_mode(CollaborationMode.SUPERVISED)

    @classmethod
    def pair(cls) -> CollaborationConfig:
        """Shortcut for pair mode config."""
        return cls.for_mode(CollaborationMode.PAIR)

    @classmethod
    def guided(cls) -> CollaborationConfig:
        """Shortcut for guided mode config."""
        return cls.for_mode(CollaborationMode.GUIDED)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mode": self.mode.value,
            "pause_on_file_delete": self.pause_on_file_delete,
            "pause_on_config_change": self.pause_on_config_change,
            "pause_on_dependency_add": self.pause_on_dependency_add,
            "pause_on_shell_command": self.pause_on_shell_command,
            "explain_before_action": self.explain_before_action,
            "require_approval_for": self.require_approval_for,
            "dangerous_patterns": self.dangerous_patterns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CollaborationConfig:
        """Create from dictionary."""
        return cls(
            mode=CollaborationMode(data["mode"]),
            pause_on_file_delete=data.get("pause_on_file_delete", True),
            pause_on_config_change=data.get("pause_on_config_change", True),
            pause_on_dependency_add=data.get("pause_on_dependency_add", True),
            pause_on_shell_command=data.get("pause_on_shell_command", False),
            explain_before_action=data.get("explain_before_action", False),
            require_approval_for=data.get("require_approval_for", []),
            dangerous_patterns=data.get("dangerous_patterns", []),
        )


# =============================================================================
# Agent Action Types
# =============================================================================


class ActionType(Enum):
    """Types of actions an agent can take.

    Used for determining whether an action should trigger a pause.
    """

    FILE_CREATE = "file_create"
    """Creating a new file."""

    FILE_EDIT = "file_edit"
    """Editing an existing file."""

    FILE_DELETE = "file_delete"
    """Deleting a file."""

    CONFIG_CHANGE = "config_change"
    """Modifying configuration files."""

    DEPENDENCY_ADD = "dependency_add"
    """Adding a new dependency."""

    SHELL_COMMAND = "shell_command"
    """Running a shell command."""

    TASK_COMPLETE = "task_complete"
    """Completing a task (for pair mode)."""

    GIT_OPERATION = "git_operation"
    """Git operations (commit, push, etc.)."""

    OTHER = "other"
    """Other action types."""


@dataclass
class AgentAction:
    """Represents an action the agent wants to take.

    Attributes:
        type: The type of action.
        command: The command or operation string.
        file_path: Path to affected file (if applicable).
        description: Human-readable description of the action.
        metadata: Additional action metadata.

    Example:
        ```python
        action = AgentAction(
            type=ActionType.FILE_DELETE,
            command="rm",
            file_path="src/old_module.py",
            description="Delete deprecated module",
        )
        ```
    """

    type: ActionType
    command: str = ""
    file_path: str | None = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file_operation(
        cls,
        operation: str,
        file_path: str,
        description: str = "",
    ) -> AgentAction:
        """Create action from file operation.

        Args:
            operation: One of 'create', 'edit', 'delete'.
            file_path: Path to the file.
            description: Optional description.

        Returns:
            AgentAction with appropriate type.
        """
        type_map = {
            "create": ActionType.FILE_CREATE,
            "edit": ActionType.FILE_EDIT,
            "delete": ActionType.FILE_DELETE,
        }
        action_type = type_map.get(operation.lower(), ActionType.OTHER)

        # Check if it's a config file
        config_patterns = [
            "*.json",
            "*.yaml",
            "*.yml",
            "*.toml",
            "*.ini",
            "*.cfg",
            "*.env",
            ".env*",
            "config/*",
            "settings/*",
            "pyproject.toml",
            "package.json",
            "Cargo.toml",
        ]
        is_config = any(
            fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(Path(file_path).name, pattern)
            for pattern in config_patterns
        )

        if is_config and operation.lower() in ("edit", "delete"):
            action_type = ActionType.CONFIG_CHANGE

        return cls(
            type=action_type,
            command=operation,
            file_path=file_path,
            description=description or f"{operation} {file_path}",
        )

    @classmethod
    def from_shell_command(
        cls,
        command: str,
        description: str = "",
    ) -> AgentAction:
        """Create action from shell command.

        Args:
            command: The shell command.
            description: Optional description.

        Returns:
            AgentAction with SHELL_COMMAND type.
        """
        return cls(
            type=ActionType.SHELL_COMMAND,
            command=command,
            description=description or f"Run: {command}",
        )


# =============================================================================
# Mode-Aware Executor (Phase 4.2.2)
# =============================================================================


@dataclass
class PauseReason:
    """Reason for pausing execution.

    Attributes:
        should_pause: Whether execution should pause.
        reason: Human-readable reason for pausing.
        severity: Severity level (info, warning, danger).
        action_type: The type of action that triggered the pause.
    """

    should_pause: bool
    reason: str = ""
    severity: str = "info"
    action_type: ActionType | None = None

    @classmethod
    def no_pause(cls) -> PauseReason:
        """Create a no-pause result."""
        return cls(should_pause=False)

    @classmethod
    def pause(
        cls,
        reason: str,
        severity: str = "info",
        action_type: ActionType | None = None,
    ) -> PauseReason:
        """Create a pause result."""
        return cls(
            should_pause=True,
            reason=reason,
            severity=severity,
            action_type=action_type,
        )


class ModeAwareExecutor:
    """Executor component that respects collaboration mode.

    Phase 4.2.2: Determines when to pause for human input based on
    the configured collaboration mode and action characteristics.

    Attributes:
        config: The collaboration configuration.

    Example:
        ```python
        config = CollaborationConfig.for_mode(CollaborationMode.SUPERVISED)
        executor = ModeAwareExecutor(config)

        action = AgentAction.from_shell_command("pip install requests")

        result = await executor.should_pause(action, context)
        if result.should_pause:
            print(f"Pausing: {result.reason}")
            # Request human approval
        ```
    """

    def __init__(self, config: CollaborationConfig) -> None:
        """Initialize the mode-aware executor.

        Args:
            config: The collaboration configuration.
        """
        self.config = config

    async def should_pause(
        self,
        action: AgentAction,
        context: ExecutionContext | None = None,
    ) -> PauseReason:
        """Determine if execution should pause for human input.

        Evaluates the action against the collaboration mode and
        configuration to determine if human approval is needed.

        Args:
            action: The action the agent wants to take.
            context: Optional execution context.

        Returns:
            PauseReason indicating whether to pause and why.

        Example:
            ```python
            result = await executor.should_pause(action, context)
            if result.should_pause:
                if result.severity == "danger":
                    print(f"DANGER: {result.reason}")
                else:
                    print(f"Pause: {result.reason}")
            ```
        """
        # Always check for dangerous operations first
        dangerous_check = self._check_dangerous(action)
        if dangerous_check.should_pause:
            return dangerous_check

        # Mode-specific checks
        if self.config.mode == CollaborationMode.AUTONOMOUS:
            return await self._check_autonomous(action)

        elif self.config.mode == CollaborationMode.SUPERVISED:
            return await self._check_supervised(action)

        elif self.config.mode == CollaborationMode.PAIR:
            return await self._check_pair(action)

        elif self.config.mode == CollaborationMode.GUIDED:
            return await self._check_guided(action)

        return PauseReason.no_pause()

    def _check_dangerous(self, action: AgentAction) -> PauseReason:
        """Check if action contains dangerous patterns.

        Always triggers a pause regardless of mode.

        Args:
            action: The action to check.

        Returns:
            PauseReason if dangerous pattern found.
        """
        command = action.command.lower()

        for pattern in self.config.dangerous_patterns:
            if pattern.lower() in command:
                return PauseReason.pause(
                    reason=f"Dangerous operation detected: {pattern}",
                    severity="danger",
                    action_type=action.type,
                )

        return PauseReason.no_pause()

    async def _check_autonomous(self, action: AgentAction) -> PauseReason:
        """Check for autonomous mode - minimal pauses.

        Only pauses on file deletes (configurable).

        Args:
            action: The action to check.

        Returns:
            PauseReason based on autonomous mode rules.
        """
        if action.type == ActionType.FILE_DELETE and self.config.pause_on_file_delete:
            return PauseReason.pause(
                reason=f"File deletion: {action.file_path}",
                severity="warning",
                action_type=action.type,
            )

        return PauseReason.no_pause()

    async def _check_supervised(self, action: AgentAction) -> PauseReason:
        """Check for supervised mode - pause on risky operations.

        Args:
            action: The action to check.

        Returns:
            PauseReason based on supervised mode rules.
        """
        # Check file-specific approval requirements
        if action.file_path and self.config.require_approval_for:
            for pattern in self.config.require_approval_for:
                if fnmatch.fnmatch(action.file_path, pattern):
                    return PauseReason.pause(
                        reason=f"File matches approval pattern '{pattern}': {action.file_path}",
                        severity="info",
                        action_type=action.type,
                    )

        # Check action type specific pauses
        if action.type == ActionType.FILE_DELETE and self.config.pause_on_file_delete:
            return PauseReason.pause(
                reason=f"File deletion: {action.file_path}",
                severity="warning",
                action_type=action.type,
            )

        if action.type == ActionType.CONFIG_CHANGE and self.config.pause_on_config_change:
            return PauseReason.pause(
                reason=f"Config file change: {action.file_path}",
                severity="info",
                action_type=action.type,
            )

        if action.type == ActionType.DEPENDENCY_ADD and self.config.pause_on_dependency_add:
            return PauseReason.pause(
                reason=f"Adding dependency: {action.description}",
                severity="info",
                action_type=action.type,
            )

        if action.type == ActionType.SHELL_COMMAND and self.config.pause_on_shell_command:
            return PauseReason.pause(
                reason=f"Shell command: {action.command}",
                severity="info",
                action_type=action.type,
            )

        return PauseReason.no_pause()

    async def _check_pair(self, action: AgentAction) -> PauseReason:
        """Check for pair mode - pause after each task.

        Args:
            action: The action to check.

        Returns:
            PauseReason based on pair mode rules.
        """
        # First check supervised rules
        supervised_result = await self._check_supervised(action)
        if supervised_result.should_pause:
            return supervised_result

        # In pair mode, also pause on task completion
        if action.type == ActionType.TASK_COMPLETE:
            return PauseReason.pause(
                reason="Task completed - ready for review",
                severity="info",
                action_type=action.type,
            )

        return PauseReason.no_pause()

    async def _check_guided(self, action: AgentAction) -> PauseReason:
        """Check for guided mode - always pause before execution.

        Args:
            action: The action to check.

        Returns:
            Always returns a pause (guided mode pauses on everything).
        """
        return PauseReason.pause(
            reason=f"Guided mode - approval required: {action.description}",
            severity="info",
            action_type=action.type,
        )

    def requires_explanation(self) -> bool:
        """Check if the mode requires explaining actions before execution.

        Returns:
            True if explain_before_action is enabled.
        """
        return self.config.explain_before_action


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "ActionType",
    "AgentAction",
    "CollaborationConfig",
    "CollaborationMode",
    "ModeAwareExecutor",
    "PauseReason",
]
