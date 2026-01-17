"""Tests for Phase 4.2 Collaboration Modes.

Phase 4.2: Collaboration Modes

Tests cover:
- CollaborationMode enum values
- CollaborationConfig dataclass and factory methods
- AgentAction creation and classification
- ModeAwareExecutor pause logic for each mode
- Dangerous operation detection
- Pattern matching for file approval requirements
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ai_infra.executor.collaboration import (
    ActionType,
    AgentAction,
    CollaborationConfig,
    CollaborationMode,
    ModeAwareExecutor,
    PauseReason,
)

# =============================================================================
# CollaborationMode Tests (4.2.1)
# =============================================================================


class TestCollaborationMode:
    """Tests for CollaborationMode enum."""

    def test_all_modes_exist(self) -> None:
        """Verify all 4 collaboration modes are defined."""
        assert CollaborationMode.AUTONOMOUS.value == "autonomous"
        assert CollaborationMode.SUPERVISED.value == "supervised"
        assert CollaborationMode.PAIR.value == "pair"
        assert CollaborationMode.GUIDED.value == "guided"

    def test_mode_count(self) -> None:
        """Verify we have exactly 4 modes."""
        assert len(CollaborationMode) == 4


# =============================================================================
# CollaborationConfig Tests (4.2.1)
# =============================================================================


class TestCollaborationConfig:
    """Tests for CollaborationConfig dataclass."""

    def test_autonomous_mode_defaults(self) -> None:
        """Test autonomous mode has minimal pauses."""
        config = CollaborationConfig.for_mode(CollaborationMode.AUTONOMOUS)

        assert config.mode == CollaborationMode.AUTONOMOUS
        assert config.pause_on_file_delete is True  # Still pause on deletes
        assert config.pause_on_config_change is False
        assert config.pause_on_dependency_add is False
        assert config.pause_on_shell_command is False
        assert config.explain_before_action is False

    def test_supervised_mode_defaults(self) -> None:
        """Test supervised mode pauses on risky operations."""
        config = CollaborationConfig.for_mode(CollaborationMode.SUPERVISED)

        assert config.mode == CollaborationMode.SUPERVISED
        assert config.pause_on_file_delete is True
        assert config.pause_on_config_change is True
        assert config.pause_on_dependency_add is True
        assert config.pause_on_shell_command is True
        assert config.explain_before_action is False

    def test_pair_mode_defaults(self) -> None:
        """Test pair mode requires explanation."""
        config = CollaborationConfig.for_mode(CollaborationMode.PAIR)

        assert config.mode == CollaborationMode.PAIR
        assert config.pause_on_file_delete is True
        assert config.explain_before_action is True

    def test_guided_mode_defaults(self) -> None:
        """Test guided mode requires explanation."""
        config = CollaborationConfig.for_mode(CollaborationMode.GUIDED)

        assert config.mode == CollaborationMode.GUIDED
        assert config.explain_before_action is True

    def test_shortcut_methods(self) -> None:
        """Test shortcut factory methods."""
        autonomous = CollaborationConfig.autonomous()
        supervised = CollaborationConfig.supervised()
        pair = CollaborationConfig.pair()
        guided = CollaborationConfig.guided()

        assert autonomous.mode == CollaborationMode.AUTONOMOUS
        assert supervised.mode == CollaborationMode.SUPERVISED
        assert pair.mode == CollaborationMode.PAIR
        assert guided.mode == CollaborationMode.GUIDED

    def test_custom_config(self) -> None:
        """Test creating custom config."""
        config = CollaborationConfig(
            mode=CollaborationMode.SUPERVISED,
            pause_on_file_delete=True,
            pause_on_config_change=False,
            require_approval_for=["*.sql", "migrations/*"],
        )

        assert config.mode == CollaborationMode.SUPERVISED
        assert config.pause_on_config_change is False
        assert len(config.require_approval_for) == 2

    def test_serialization(self) -> None:
        """Test config serialization round-trip."""
        original = CollaborationConfig(
            mode=CollaborationMode.SUPERVISED,
            pause_on_file_delete=True,
            require_approval_for=["*.py"],
        )

        data = original.to_dict()
        restored = CollaborationConfig.from_dict(data)

        assert restored.mode == original.mode
        assert restored.pause_on_file_delete == original.pause_on_file_delete
        assert restored.require_approval_for == original.require_approval_for

    def test_dangerous_patterns_default(self) -> None:
        """Test default dangerous patterns are set."""
        config = CollaborationConfig.for_mode(CollaborationMode.AUTONOMOUS)

        assert "rm -rf" in config.dangerous_patterns
        assert "DROP TABLE" in config.dangerous_patterns
        assert "git push --force" in config.dangerous_patterns


# =============================================================================
# AgentAction Tests
# =============================================================================


class TestAgentAction:
    """Tests for AgentAction dataclass."""

    def test_from_file_operation_create(self) -> None:
        """Test creating action from file create operation."""
        action = AgentAction.from_file_operation(
            operation="create",
            file_path="src/new_module.py",
            description="Create new module",
        )

        assert action.type == ActionType.FILE_CREATE
        assert action.file_path == "src/new_module.py"

    def test_from_file_operation_delete(self) -> None:
        """Test creating action from file delete operation."""
        action = AgentAction.from_file_operation(
            operation="delete",
            file_path="src/old_module.py",
        )

        assert action.type == ActionType.FILE_DELETE
        assert action.file_path == "src/old_module.py"

    def test_config_file_detection(self) -> None:
        """Test that config files are detected."""
        # JSON config
        action = AgentAction.from_file_operation(
            operation="edit",
            file_path="config/settings.json",
        )
        assert action.type == ActionType.CONFIG_CHANGE

        # YAML config
        action = AgentAction.from_file_operation(
            operation="edit",
            file_path="docker-compose.yml",
        )
        assert action.type == ActionType.CONFIG_CHANGE

        # pyproject.toml
        action = AgentAction.from_file_operation(
            operation="edit",
            file_path="pyproject.toml",
        )
        assert action.type == ActionType.CONFIG_CHANGE

    def test_from_shell_command(self) -> None:
        """Test creating action from shell command."""
        action = AgentAction.from_shell_command(
            command="pip install requests",
            description="Install requests package",
        )

        assert action.type == ActionType.SHELL_COMMAND
        assert action.command == "pip install requests"


# =============================================================================
# ModeAwareExecutor Tests (4.2.2)
# =============================================================================


class TestModeAwareExecutor:
    """Tests for ModeAwareExecutor."""

    @pytest.fixture
    def sample_context(self) -> Any:
        """Create a sample execution context."""
        from ai_infra.executor.agents.base import ExecutionContext

        return ExecutionContext(workspace=Path("/project"))

    # -------------------------------------------------------------------------
    # Dangerous Operation Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_dangerous_always_pauses(self, sample_context: Any) -> None:
        """Test that dangerous operations always trigger pause."""
        # Even in autonomous mode
        config = CollaborationConfig.autonomous()
        executor = ModeAwareExecutor(config)

        action = AgentAction.from_shell_command("rm -rf /tmp/data")
        result = await executor.should_pause(action, sample_context)

        assert result.should_pause is True
        assert result.severity == "danger"
        assert "rm -rf" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_dangerous_patterns(self, sample_context: Any) -> None:
        """Test various dangerous patterns are detected."""
        config = CollaborationConfig.autonomous()
        executor = ModeAwareExecutor(config)

        dangerous_commands = [
            "DROP TABLE users",
            "git push --force origin main",
            "DELETE FROM orders",
            "sudo rm -rf /var",
        ]

        for cmd in dangerous_commands:
            action = AgentAction.from_shell_command(cmd)
            result = await executor.should_pause(action, sample_context)
            assert result.should_pause is True, f"Should pause for: {cmd}"
            assert result.severity == "danger"

    # -------------------------------------------------------------------------
    # Autonomous Mode Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_autonomous_minimal_pauses(self, sample_context: Any) -> None:
        """Test autonomous mode has minimal pauses."""
        config = CollaborationConfig.autonomous()
        executor = ModeAwareExecutor(config)

        # Normal file edit should not pause
        action = AgentAction.from_file_operation("edit", "src/module.py")
        result = await executor.should_pause(action, sample_context)
        assert result.should_pause is False

        # Shell command should not pause
        action = AgentAction.from_shell_command("pytest tests/")
        result = await executor.should_pause(action, sample_context)
        assert result.should_pause is False

    @pytest.mark.asyncio
    async def test_autonomous_pauses_on_delete(self, sample_context: Any) -> None:
        """Test autonomous mode still pauses on file delete."""
        config = CollaborationConfig.autonomous()
        executor = ModeAwareExecutor(config)

        action = AgentAction.from_file_operation("delete", "src/important.py")
        result = await executor.should_pause(action, sample_context)

        assert result.should_pause is True
        assert "deletion" in result.reason.lower()

    # -------------------------------------------------------------------------
    # Supervised Mode Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_supervised_pauses_on_config(self, sample_context: Any) -> None:
        """Test supervised mode pauses on config changes."""
        config = CollaborationConfig.supervised()
        executor = ModeAwareExecutor(config)

        action = AgentAction.from_file_operation("edit", "pyproject.toml")
        result = await executor.should_pause(action, sample_context)

        assert result.should_pause is True
        assert "config" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_supervised_pauses_on_shell(self, sample_context: Any) -> None:
        """Test supervised mode pauses on shell commands."""
        config = CollaborationConfig.supervised()
        executor = ModeAwareExecutor(config)

        action = AgentAction.from_shell_command("npm install express")
        result = await executor.should_pause(action, sample_context)

        assert result.should_pause is True
        assert "shell" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_supervised_pattern_matching(self, sample_context: Any) -> None:
        """Test supervised mode respects require_approval_for patterns."""
        config = CollaborationConfig(
            mode=CollaborationMode.SUPERVISED,
            require_approval_for=["*.sql", "migrations/*"],
        )
        executor = ModeAwareExecutor(config)

        # SQL file should require approval
        action = AgentAction.from_file_operation("edit", "schema.sql")
        result = await executor.should_pause(action, sample_context)
        assert result.should_pause is True

        # Migration file should require approval
        action = AgentAction.from_file_operation("create", "migrations/001_init.py")
        result = await executor.should_pause(action, sample_context)
        assert result.should_pause is True

        # Regular Python file should not
        action = AgentAction.from_file_operation("edit", "src/main.py")
        result = await executor.should_pause(action, sample_context)
        assert result.should_pause is False

    # -------------------------------------------------------------------------
    # Pair Mode Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_pair_pauses_on_task_complete(self, sample_context: Any) -> None:
        """Test pair mode pauses on task completion."""
        config = CollaborationConfig.pair()
        executor = ModeAwareExecutor(config)

        action = AgentAction(
            type=ActionType.TASK_COMPLETE,
            description="Completed: Add authentication",
        )
        result = await executor.should_pause(action, sample_context)

        assert result.should_pause is True
        assert "review" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_pair_inherits_supervised_rules(self, sample_context: Any) -> None:
        """Test pair mode also applies supervised rules."""
        config = CollaborationConfig.pair()
        executor = ModeAwareExecutor(config)

        # Should pause on config change (supervised rule)
        action = AgentAction.from_file_operation("edit", "package.json")
        result = await executor.should_pause(action, sample_context)
        assert result.should_pause is True

    # -------------------------------------------------------------------------
    # Guided Mode Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_guided_always_pauses(self, sample_context: Any) -> None:
        """Test guided mode pauses on everything."""
        config = CollaborationConfig.guided()
        executor = ModeAwareExecutor(config)

        # Even simple file edit
        action = AgentAction.from_file_operation("edit", "src/main.py")
        result = await executor.should_pause(action, sample_context)

        assert result.should_pause is True
        assert "guided" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_guided_requires_explanation(self) -> None:
        """Test guided mode requires explanation."""
        config = CollaborationConfig.guided()
        executor = ModeAwareExecutor(config)

        assert executor.requires_explanation() is True

    # -------------------------------------------------------------------------
    # PauseReason Tests
    # -------------------------------------------------------------------------

    def test_pause_reason_no_pause(self) -> None:
        """Test creating no-pause result."""
        result = PauseReason.no_pause()

        assert result.should_pause is False
        assert result.reason == ""

    def test_pause_reason_pause(self) -> None:
        """Test creating pause result."""
        result = PauseReason.pause(
            reason="File deletion detected",
            severity="warning",
            action_type=ActionType.FILE_DELETE,
        )

        assert result.should_pause is True
        assert result.reason == "File deletion detected"
        assert result.severity == "warning"
        assert result.action_type == ActionType.FILE_DELETE
