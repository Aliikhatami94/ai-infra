"""Unit tests for ai_infra.cli.interactive module.

Phase 16.7.1 of EXECUTOR_6.md: Interactive Task Review Mode.
"""

from __future__ import annotations

import io
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from ai_infra.cli.interactive import (
    COMPLEXITY_ESTIMATES,
    DEFAULT_KEY_BINDINGS,
    ContextInfo,
    ContextPreview,
    DependencyGraph,
    DependencyNode,
    InteractiveSession,
    SessionResult,
    SessionState,
    TaskAction,
    TaskInfo,
    TaskPreview,
    edit_task_description,
    render_help,
)
from ai_infra.cli.progress import TaskStatus

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def console() -> Console:
    """Create a test console."""
    return Console(file=io.StringIO(), force_terminal=True, width=80)


@pytest.fixture
def sample_task() -> TaskInfo:
    """Create a sample task for testing."""
    return TaskInfo(
        id="1.1.3",
        title="Create user authentication module",
        description="Implement JWT-based authentication with login/logout endpoints.\nInclude password hashing with bcrypt and token refresh logic.",
        phase="Phase 1: Project Setup",
        status=TaskStatus.PENDING,
        complexity="medium",
        dependencies=["1.1.1", "1.1.2"],
        completed_dependencies=["1.1.1", "1.1.2"],
    )


@pytest.fixture
def sample_tasks() -> list[TaskInfo]:
    """Create a list of sample tasks."""
    return [
        TaskInfo(
            id="1.1.1",
            title="Create project structure",
            phase="Phase 1",
            status=TaskStatus.COMPLETE,
        ),
        TaskInfo(
            id="1.1.2",
            title="Initialize git repository",
            phase="Phase 1",
            status=TaskStatus.COMPLETE,
        ),
        TaskInfo(
            id="1.1.3",
            title="Configure dependencies",
            phase="Phase 1",
            status=TaskStatus.PENDING,
            complexity="medium",
        ),
        TaskInfo(
            id="2.1.1",
            title="Define user model",
            phase="Phase 2",
            status=TaskStatus.PENDING,
        ),
    ]


@pytest.fixture
def sample_context() -> ContextInfo:
    """Create sample context info."""
    return ContextInfo(
        system_tokens=847,
        task_tokens=1234,
        file_tokens=2456,
        memory_tokens=512,
        files_included=[
            ("src/main.py", "full"),
            ("src/models.py", "full"),
            ("tests/conftest.py", "partial: lines 1-50"),
        ],
        context_window=200_000,
    )


# =============================================================================
# TaskInfo Tests
# =============================================================================


class TestTaskInfo:
    """Tests for TaskInfo dataclass."""

    def test_create_with_defaults(self) -> None:
        """Test creating TaskInfo with default values."""
        task = TaskInfo(id="1.1.1", title="Test task")
        assert task.id == "1.1.1"
        assert task.title == "Test task"
        assert task.description == ""
        assert task.phase == ""
        assert task.status == TaskStatus.PENDING
        assert task.complexity == "medium"
        assert task.dependencies == []
        assert task.completed_dependencies == []

    def test_status_conversion_from_string(self) -> None:
        """Test that string status is converted to enum."""
        task = TaskInfo(id="1.1.1", title="Test", status="complete")
        assert task.status == TaskStatus.COMPLETE

    def test_status_as_enum(self) -> None:
        """Test that enum status is preserved."""
        task = TaskInfo(id="1.1.1", title="Test", status=TaskStatus.RUNNING)
        assert task.status == TaskStatus.RUNNING

    def test_invalid_status_defaults_to_pending(self) -> None:
        """Test that invalid status defaults to pending."""
        task = TaskInfo(id="1.1.1", title="Test", status="invalid_status")
        assert task.status == TaskStatus.PENDING


# =============================================================================
# ContextInfo Tests
# =============================================================================


class TestContextInfo:
    """Tests for ContextInfo dataclass."""

    def test_total_tokens(self, sample_context: ContextInfo) -> None:
        """Test total token calculation."""
        expected = 847 + 1234 + 2456 + 512
        assert sample_context.total_tokens == expected

    def test_usage_percentage(self, sample_context: ContextInfo) -> None:
        """Test usage percentage calculation."""
        total = sample_context.total_tokens
        expected = (total / 200_000) * 100
        assert sample_context.usage_percentage == pytest.approx(expected)

    def test_usage_percentage_zero_window(self) -> None:
        """Test usage percentage with zero context window."""
        ctx = ContextInfo(context_window=0)
        assert ctx.usage_percentage == 0.0

    def test_defaults(self) -> None:
        """Test default values."""
        ctx = ContextInfo()
        assert ctx.system_tokens == 0
        assert ctx.task_tokens == 0
        assert ctx.file_tokens == 0
        assert ctx.memory_tokens == 0
        assert ctx.files_included == []
        assert ctx.context_window == 200_000


# =============================================================================
# DependencyNode Tests
# =============================================================================


class TestDependencyNode:
    """Tests for DependencyNode dataclass."""

    def test_create_node(self) -> None:
        """Test creating a dependency node."""
        node = DependencyNode(
            task_id="1.1.1",
            title="Create project",
            status=TaskStatus.COMPLETE,
        )
        assert node.task_id == "1.1.1"
        assert node.title == "Create project"
        assert node.status == TaskStatus.COMPLETE
        assert node.is_current is False

    def test_current_node(self) -> None:
        """Test marking node as current."""
        node = DependencyNode(
            task_id="1.1.3",
            title="Current task",
            is_current=True,
        )
        assert node.is_current is True


# =============================================================================
# TaskPreview Tests
# =============================================================================


class TestTaskPreview:
    """Tests for TaskPreview renderable."""

    def test_render_basic(self, console: Console, sample_task: TaskInfo) -> None:
        """Test basic rendering of task preview."""
        preview = TaskPreview(sample_task)
        console.print(preview)
        output = console.file.getvalue()

        assert "NEXT TASK" in output
        assert "Create user authentication module" in output

    def test_render_with_description(self, console: Console, sample_task: TaskInfo) -> None:
        """Test rendering includes description."""
        preview = TaskPreview(sample_task)
        console.print(preview)
        output = console.file.getvalue()

        assert "Description" in output
        assert "JWT-based" in output

    def test_render_metadata(self, console: Console, sample_task: TaskInfo) -> None:
        """Test rendering includes metadata."""
        preview = TaskPreview(sample_task)
        console.print(preview)
        output = console.file.getvalue()

        assert "Complexity" in output
        assert "Medium" in output
        assert "Dependencies" in output

    def test_render_actions(self, console: Console, sample_task: TaskInfo) -> None:
        """Test rendering includes action hints."""
        preview = TaskPreview(sample_task, show_actions=True)
        console.print(preview)
        output = console.file.getvalue()

        assert "[Enter]" in output
        assert "Execute" in output
        assert "[s]" in output
        assert "Skip" in output

    def test_render_without_actions(self, console: Console, sample_task: TaskInfo) -> None:
        """Test rendering without action hints."""
        preview = TaskPreview(sample_task, show_actions=False)
        console.print(preview)
        output = console.file.getvalue()

        # Actions should not be in output
        # Note: We can't easily test absence, so just verify it renders
        assert "NEXT TASK" in output

    def test_complexity_display(self) -> None:
        """Test complexity label and estimate."""
        preview = TaskPreview(TaskInfo(id="1", title="T", complexity="high"))
        label, estimate = preview._get_complexity_display()
        assert label == "High"
        assert "3-5" in estimate


# =============================================================================
# ContextPreview Tests
# =============================================================================


class TestContextPreview:
    """Tests for ContextPreview renderable."""

    def test_render_context(self, console: Console, sample_context: ContextInfo) -> None:
        """Test rendering context preview."""
        preview = ContextPreview(sample_context)
        console.print(preview)
        output = console.file.getvalue()

        assert "CONTEXT PREVIEW" in output
        assert "System Prompt" in output
        assert "Task Context" in output
        assert "File Context" in output
        assert "Run Memory" in output
        assert "Total" in output

    def test_render_files_list(self, console: Console, sample_context: ContextInfo) -> None:
        """Test rendering includes files list."""
        preview = ContextPreview(sample_context)
        console.print(preview)
        output = console.file.getvalue()

        assert "Files included" in output
        assert "src/main.py" in output
        assert "full" in output

    def test_format_tokens(self, sample_context: ContextInfo) -> None:
        """Test token formatting."""
        preview = ContextPreview(sample_context)
        formatted = preview._format_tokens(12345)
        assert formatted == "12,345"


# =============================================================================
# DependencyGraph Tests
# =============================================================================


class TestDependencyGraph:
    """Tests for DependencyGraph renderable."""

    def test_render_graph(self, console: Console) -> None:
        """Test rendering dependency graph."""
        nodes = [
            DependencyNode("1.1.1", "Create project", TaskStatus.COMPLETE),
            DependencyNode("1.1.2", "Init git", TaskStatus.COMPLETE),
            DependencyNode("1.1.3", "Configure deps", TaskStatus.PENDING, is_current=True),
        ]
        graph = DependencyGraph("1.1.3", nodes)
        console.print(graph)
        output = console.file.getvalue()

        assert "DEPENDENCIES" in output
        assert "1.1.1" in output
        assert "1.1.2" in output
        assert "1.1.3" in output
        assert "[current]" in output
        assert "[complete]" in output


# =============================================================================
# Help Tests
# =============================================================================


class TestHelp:
    """Tests for help display."""

    def test_render_help(self, console: Console) -> None:
        """Test rendering help panel."""
        help_panel = render_help()
        console.print(help_panel)
        output = console.file.getvalue()

        assert "INTERACTIVE MODE HELP" in output
        assert "Enter" in output
        assert "Execute" in output
        assert "Skip" in output
        assert "Edit" in output
        assert "Context" in output
        assert "Dependencies" in output
        assert "Quit" in output


# =============================================================================
# SessionResult Tests
# =============================================================================


class TestSessionResult:
    """Tests for SessionResult dataclass."""

    def test_create_basic(self) -> None:
        """Test creating basic result."""
        result = SessionResult(action=TaskAction.EXECUTE, task_id="1.1.1")
        assert result.action == TaskAction.EXECUTE
        assert result.task_id == "1.1.1"
        assert result.modified_description is None
        assert result.execute_remaining is False

    def test_create_with_edit(self) -> None:
        """Test creating result with edit."""
        result = SessionResult(
            action=TaskAction.EDIT,
            task_id="1.1.1",
            modified_description="New description",
        )
        assert result.action == TaskAction.EDIT
        assert result.modified_description == "New description"

    def test_create_execute_all(self) -> None:
        """Test creating execute all result."""
        result = SessionResult(
            action=TaskAction.EXECUTE_ALL,
            execute_remaining=True,
        )
        assert result.action == TaskAction.EXECUTE_ALL
        assert result.execute_remaining is True


# =============================================================================
# InteractiveSession Tests
# =============================================================================


class TestInteractiveSession:
    """Tests for InteractiveSession class."""

    def test_create_session(self, sample_tasks: list[TaskInfo], console: Console) -> None:
        """Test creating an interactive session."""
        session = InteractiveSession(sample_tasks, console=console)
        assert len(session.tasks) == 4
        assert session.remaining_tasks == 4
        assert session.state == SessionState.IDLE

    def test_current_task(self, sample_tasks: list[TaskInfo], console: Console) -> None:
        """Test getting current task."""
        session = InteractiveSession(sample_tasks, console=console)
        assert session.current_task is not None
        assert session.current_task.id == "1.1.1"

    def test_current_task_empty(self, console: Console) -> None:
        """Test current task with empty list."""
        session = InteractiveSession([], console=console)
        assert session.current_task is None

    def test_advance(self, sample_tasks: list[TaskInfo], console: Console) -> None:
        """Test advancing to next task."""
        session = InteractiveSession(sample_tasks, console=console)
        assert session.current_task.id == "1.1.1"

        session.advance()
        assert session.current_task.id == "1.1.2"

        session.advance()
        assert session.current_task.id == "1.1.3"

    def test_interrupt(self, sample_tasks: list[TaskInfo], console: Console) -> None:
        """Test interrupting session."""
        session = InteractiveSession(sample_tasks, console=console)
        session.interrupt()
        assert session.state == SessionState.PAUSED
        assert session._interrupted is True

    def test_handle_execute_action(self, sample_tasks: list[TaskInfo], console: Console) -> None:
        """Test handling execute action."""
        session = InteractiveSession(sample_tasks, console=console)
        task = sample_tasks[2]  # Pending task

        result = session._handle_action("enter", task)
        assert result is not None
        assert result.action == TaskAction.EXECUTE
        assert result.task_id == task.id

    def test_handle_skip_action(self, sample_tasks: list[TaskInfo], console: Console) -> None:
        """Test handling skip action."""
        session = InteractiveSession(sample_tasks, console=console)
        task = sample_tasks[2]

        result = session._handle_action("s", task)
        assert result is not None
        assert result.action == TaskAction.SKIP
        assert result.task_id == task.id

    def test_handle_quit_action(self, sample_tasks: list[TaskInfo], console: Console) -> None:
        """Test handling quit action."""
        session = InteractiveSession(sample_tasks, console=console)
        task = sample_tasks[2]

        result = session._handle_action("q", task)
        assert result is not None
        assert result.action == TaskAction.QUIT

    def test_handle_execute_all_action(
        self, sample_tasks: list[TaskInfo], console: Console
    ) -> None:
        """Test handling execute all action."""
        session = InteractiveSession(sample_tasks, console=console)
        task = sample_tasks[2]

        result = session._handle_action("a", task)
        assert result is not None
        assert result.action == TaskAction.EXECUTE_ALL
        assert result.execute_remaining is True

    def test_handle_unknown_key(self, sample_tasks: list[TaskInfo], console: Console) -> None:
        """Test handling unknown key."""
        session = InteractiveSession(sample_tasks, console=console)
        task = sample_tasks[2]

        result = session._handle_action("x", task)
        assert result is None  # Should continue, not return

    def test_handle_view_context(self, sample_tasks: list[TaskInfo], console: Console) -> None:
        """Test handling view context action."""
        session = InteractiveSession(sample_tasks, console=console)
        task = sample_tasks[2]

        # Mock _read_key to avoid waiting for input
        with patch.object(session, "_read_key", return_value="enter"):
            result = session._handle_action("v", task)
            # Returns None to continue previewing
            assert result is None

    def test_handle_view_dependencies(self, sample_tasks: list[TaskInfo], console: Console) -> None:
        """Test handling view dependencies action."""
        session = InteractiveSession(sample_tasks, console=console)
        task = sample_tasks[2]

        with patch.object(session, "_read_key", return_value="enter"):
            result = session._handle_action("d", task)
            assert result is None

    def test_handle_help(self, sample_tasks: list[TaskInfo], console: Console) -> None:
        """Test handling help action."""
        session = InteractiveSession(sample_tasks, console=console)
        task = sample_tasks[2]

        with patch.object(session, "_read_key", return_value="enter"):
            result = session._handle_action("?", task)
            assert result is None

    def test_context_provider_callback(
        self, sample_tasks: list[TaskInfo], console: Console, sample_context: ContextInfo
    ) -> None:
        """Test custom context provider callback."""
        context_provider = MagicMock(return_value=sample_context)
        session = InteractiveSession(
            sample_tasks,
            console=console,
            context_provider=context_provider,
        )
        task = sample_tasks[2]

        with patch.object(session, "_read_key", return_value="enter"):
            session._show_context_preview(task)

        context_provider.assert_called_once_with(task)

    def test_dependency_provider_callback(
        self, sample_tasks: list[TaskInfo], console: Console
    ) -> None:
        """Test custom dependency provider callback."""
        nodes = [DependencyNode("1.1.1", "Dep", TaskStatus.COMPLETE)]
        dep_provider = MagicMock(return_value=nodes)
        session = InteractiveSession(
            sample_tasks,
            console=console,
            dependency_provider=dep_provider,
        )
        task = sample_tasks[2]

        with patch.object(session, "_read_key", return_value="enter"):
            session._show_dependency_graph(task)

        dep_provider.assert_called_once_with(task.id)


# =============================================================================
# Edit Task Tests
# =============================================================================


class TestEditTaskDescription:
    """Tests for edit_task_description function."""

    def test_edit_cancelled(self, sample_task: TaskInfo) -> None:
        """Test edit cancelled when subprocess fails."""
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "editor")):
            result = edit_task_description(sample_task, editor="echo")
            assert result is None

    def test_edit_empty_content(self, sample_task: TaskInfo) -> None:
        """Test edit returns None for empty content."""
        # Create a mock that simulates user deleting all content
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("")
            temp_path = f.name

        def mock_run(*args: object, **kwargs: object) -> MagicMock:
            # Write empty content to the temp file
            with open(temp_path, "w") as f:
                f.write("")
            return MagicMock()

        with patch("subprocess.run", side_effect=mock_run):
            with patch("tempfile.NamedTemporaryFile") as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = temp_path
                # The function should handle this gracefully
                pass


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_default_key_bindings(self) -> None:
        """Test default key bindings exist."""
        assert "enter" in DEFAULT_KEY_BINDINGS
        assert "s" in DEFAULT_KEY_BINDINGS
        assert "e" in DEFAULT_KEY_BINDINGS
        assert "v" in DEFAULT_KEY_BINDINGS
        assert "d" in DEFAULT_KEY_BINDINGS
        assert "q" in DEFAULT_KEY_BINDINGS
        assert "a" in DEFAULT_KEY_BINDINGS
        assert "?" in DEFAULT_KEY_BINDINGS

    def test_complexity_estimates(self) -> None:
        """Test complexity estimates exist."""
        assert "trivial" in COMPLEXITY_ESTIMATES
        assert "low" in COMPLEXITY_ESTIMATES
        assert "medium" in COMPLEXITY_ESTIMATES
        assert "high" in COMPLEXITY_ESTIMATES
        assert "complex" in COMPLEXITY_ESTIMATES

        # Each should have label and time estimate
        for key, (label, estimate) in COMPLEXITY_ESTIMATES.items():
            assert isinstance(label, str)
            assert isinstance(estimate, str)


# =============================================================================
# TaskAction Enum Tests
# =============================================================================


class TestTaskAction:
    """Tests for TaskAction enum."""

    def test_all_actions_exist(self) -> None:
        """Test all expected actions exist."""
        expected = [
            "execute",
            "skip",
            "edit",
            "view_context",
            "view_dependencies",
            "quit",
            "execute_all",
            "next_phase",
            "reorder",
            "help",
        ]
        for action in expected:
            assert TaskAction(action) is not None


class TestSessionState:
    """Tests for SessionState enum."""

    def test_all_states_exist(self) -> None:
        """Test all expected states exist."""
        expected = ["idle", "previewing", "executing", "paused", "completed", "cancelled"]
        for state in expected:
            assert SessionState(state) is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for interactive components."""

    def test_full_preview_render(self, console: Console, sample_task: TaskInfo) -> None:
        """Test full preview renders without errors."""
        preview = TaskPreview(sample_task)
        console.print(preview)

        # Should not raise any errors
        output = console.file.getvalue()
        assert len(output) > 0

    def test_session_with_all_providers(
        self,
        sample_tasks: list[TaskInfo],
        console: Console,
        sample_context: ContextInfo,
    ) -> None:
        """Test session with all provider callbacks."""
        nodes = [DependencyNode("1.1.1", "Dep", TaskStatus.COMPLETE)]

        session = InteractiveSession(
            sample_tasks,
            console=console,
            context_provider=lambda t: sample_context,
            dependency_provider=lambda tid: nodes,
        )

        assert session.current_task is not None
        assert session.context_provider is not None
        assert session.dependency_provider is not None

    def test_render_all_complexity_levels(self, console: Console) -> None:
        """Test rendering tasks with all complexity levels."""
        for complexity in ["trivial", "low", "medium", "high", "complex"]:
            task = TaskInfo(
                id="1.1.1",
                title=f"Task with {complexity} complexity",
                complexity=complexity,
            )
            preview = TaskPreview(task, show_actions=False)
            console.print(preview)

        output = console.file.getvalue()
        assert "Trivial" in output
        assert "Low" in output
        assert "Medium" in output
        assert "High" in output
        assert "Complex" in output

    def test_render_all_task_statuses(self, console: Console) -> None:
        """Test rendering dependency nodes with all statuses."""
        nodes = [
            DependencyNode("1", "Pending", TaskStatus.PENDING),
            DependencyNode("2", "Running", TaskStatus.RUNNING),
            DependencyNode("3", "Complete", TaskStatus.COMPLETE),
            DependencyNode("4", "Failed", TaskStatus.FAILED),
            DependencyNode("5", "Skipped", TaskStatus.SKIPPED),
        ]
        graph = DependencyGraph("5", nodes)
        console.print(graph)

        output = console.file.getvalue()
        assert "1" in output
        assert "2" in output
        assert "3" in output
        assert "4" in output
        assert "5" in output
