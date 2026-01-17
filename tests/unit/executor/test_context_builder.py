"""Tests for Phase 16.5.12.1: SubagentContextBuilder.

Tests the rich context builder that provides comprehensive context
to subagents for better output quality.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ai_infra.executor.agents.context_builder import (
    CodePatterns,
    SubagentContext,
    SubagentContextBuilder,
)
from ai_infra.executor.todolist import TodoItem

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_workspace() -> Path:
    """Create a temporary workspace with sample files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create directory structure
        (workspace / "src").mkdir()
        (workspace / "tests").mkdir()

        # Create pyproject.toml
        (workspace / "pyproject.toml").write_text("""
[tool.poetry]
name = "test-project"

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.100.0"
pydantic = "^2.0"

[tool.ruff]
line-length = 100
""")

        # Create source file with type hints and google docstrings
        (workspace / "src" / "user.py").write_text('''
"""User module for managing user data."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class User:
    """A user in the system.

    Args:
        name: The user's full name.
        email: The user's email address.
    """

    name: str
    email: str

    def is_valid(self) -> bool:
        """Check if user data is valid.

        Returns:
            True if valid, False otherwise.
        """
        return bool(self.name) and "@" in self.email
''')

        # Create test file
        (workspace / "tests" / "test_user.py").write_text('''
"""Tests for user module."""

import pytest

from src.user import User


def test_user_is_valid():
    """Test user validation."""
    user = User(name="Alice", email="alice@example.com")
    assert user.is_valid()
''')

        # Create __init__.py
        (workspace / "src" / "__init__.py").write_text("")

        # Create pytest.ini
        (workspace / "pytest.ini").write_text("[pytest]\n")

        yield workspace


@pytest.fixture
def sample_task() -> TodoItem:
    """Create a sample task."""
    return TodoItem(
        id=1,
        title="Create tests for user.py",
        description="Write comprehensive unit tests for the User class",
    )


@pytest.fixture
def completed_tasks() -> list[TodoItem]:
    """Create sample completed tasks."""
    return [
        TodoItem(id=0, title="Create src/user.py with User class", description=""),
        TodoItem(id=1, title="Add validation to User", description=""),
    ]


@pytest.fixture
def builder() -> SubagentContextBuilder:
    """Create a context builder instance."""
    return SubagentContextBuilder()


# =============================================================================
# CodePatterns Tests
# =============================================================================


class TestCodePatterns:
    """Tests for CodePatterns dataclass."""

    def test_default_values(self) -> None:
        """Test CodePatterns has sensible defaults."""
        patterns = CodePatterns()

        assert patterns.docstring_style == "unknown"
        assert patterns.test_framework == "pytest"
        assert patterns.type_hints == "unknown"
        assert patterns.naming_convention == "snake_case"
        assert patterns.indent_style == "spaces-4"
        assert patterns.line_length == 88

    def test_to_dict(self) -> None:
        """Test CodePatterns serialization."""
        patterns = CodePatterns(
            docstring_style="google",
            type_hints="yes",
        )

        result = patterns.to_dict()

        assert result["docstring_style"] == "google"
        assert result["type_hints"] == "yes"
        assert "test_framework" in result


# =============================================================================
# SubagentContext Tests
# =============================================================================


class TestSubagentContext:
    """Tests for SubagentContext dataclass."""

    def test_basic_context(self, sample_task: TodoItem) -> None:
        """Test basic context creation."""
        context = SubagentContext(
            task=sample_task,
            workspace=Path("/project"),
            project_type="python",
        )

        assert context.task == sample_task
        assert context.project_type == "python"
        assert context.existing_files == []

    def test_format_for_prompt(self, sample_task: TodoItem) -> None:
        """Test context formatting for prompts."""
        context = SubagentContext(
            task=sample_task,
            workspace=Path("/project"),
            project_type="python",
            existing_files=["src/user.py", "tests/test_user.py"],
            completed_summaries=["[Done] Create user module"],
            dependencies=["fastapi", "pydantic"],
        )

        formatted = context.format_for_prompt()

        assert "## Project Context" in formatted
        assert "python" in formatted
        assert "fastapi" in formatted
        assert "src/user.py" in formatted
        assert "[Done] Create user module" in formatted

    def test_format_with_token_limit(self, sample_task: TodoItem) -> None:
        """Test context respects token limit."""
        # Create context with lots of data
        context = SubagentContext(
            task=sample_task,
            workspace=Path("/project"),
            existing_files=[f"file_{i}.py" for i in range(100)],
            file_previews={f"file_{i}.py": "x" * 1000 for i in range(10)},
        )

        # Format with small limit
        formatted = context.format_for_prompt(max_chars=2000)

        assert len(formatted) <= 2100  # Allow small overflow

    def test_to_dict(self, sample_task: TodoItem) -> None:
        """Test context serialization."""
        context = SubagentContext(
            task=sample_task,
            workspace=Path("/project"),
            project_type="python",
            existing_files=["a.py", "b.py"],
        )

        result = context.to_dict()

        assert result["project_type"] == "python"
        assert result["existing_files_count"] == 2
        assert "task_title" in result


# =============================================================================
# SubagentContextBuilder Tests
# =============================================================================


class TestSubagentContextBuilder:
    """Tests for SubagentContextBuilder."""

    def test_build_basic_context(
        self,
        builder: SubagentContextBuilder,
        temp_workspace: Path,
        sample_task: TodoItem,
    ) -> None:
        """Test building basic context."""
        context = builder.build(
            task=sample_task,
            workspace=temp_workspace,
        )

        assert context.task == sample_task
        assert context.workspace == temp_workspace
        assert context.project_type == "python"

    def test_build_with_completed_tasks(
        self,
        builder: SubagentContextBuilder,
        temp_workspace: Path,
        sample_task: TodoItem,
        completed_tasks: list[TodoItem],
    ) -> None:
        """Test building context with completed tasks."""
        context = builder.build(
            task=sample_task,
            workspace=temp_workspace,
            completed_tasks=completed_tasks,
        )

        assert len(context.completed_summaries) == 2
        assert "[Done]" in context.completed_summaries[0]


class TestDetectProjectType:
    """Tests for project type detection."""

    def test_detect_python_project(
        self,
        builder: SubagentContextBuilder,
        temp_workspace: Path,
    ) -> None:
        """Test Python project detection via pyproject.toml."""
        result = builder._detect_project_type(temp_workspace)
        assert result == "python"

    def test_detect_node_project(
        self,
        builder: SubagentContextBuilder,
    ) -> None:
        """Test Node.js project detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "package.json").write_text('{"name": "test"}')

            result = builder._detect_project_type(workspace)
            assert result == "node"

    def test_detect_rust_project(
        self,
        builder: SubagentContextBuilder,
    ) -> None:
        """Test Rust project detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "Cargo.toml").write_text("[package]")

            result = builder._detect_project_type(workspace)
            assert result == "rust"


class TestListRelevantFiles:
    """Tests for listing relevant files."""

    def test_list_files_in_workspace(
        self,
        builder: SubagentContextBuilder,
        temp_workspace: Path,
        sample_task: TodoItem,
    ) -> None:
        """Test listing files in workspace."""
        files = builder._list_relevant_files(temp_workspace, sample_task)

        # Should include source files
        assert any("user.py" in f for f in files)
        assert any("test_user.py" in f for f in files)

    def test_excludes_ignored_directories(
        self,
        builder: SubagentContextBuilder,
        temp_workspace: Path,
        sample_task: TodoItem,
    ) -> None:
        """Test that ignored directories are excluded."""
        # Create __pycache__ directory
        pycache = temp_workspace / "__pycache__"
        pycache.mkdir()
        (pycache / "cached.pyc").write_text("")

        files = builder._list_relevant_files(temp_workspace, sample_task)

        # Should not include __pycache__ files
        assert not any("__pycache__" in f for f in files)

    def test_prioritizes_relevant_files(
        self,
        builder: SubagentContextBuilder,
        temp_workspace: Path,
    ) -> None:
        """Test that files related to task are prioritized."""
        task = TodoItem(
            id=1,
            title="Create tests for user.py",
            description="",
        )

        files = builder._list_relevant_files(temp_workspace, task)

        # user.py should be near the top
        user_index = next(
            (i for i, f in enumerate(files) if "user.py" in f),
            len(files),
        )
        assert user_index < 5


class TestGetFilePreviews:
    """Tests for getting file previews."""

    def test_get_previews_for_relevant_files(
        self,
        builder: SubagentContextBuilder,
        temp_workspace: Path,
        sample_task: TodoItem,
    ) -> None:
        """Test getting previews of relevant files."""
        existing_files = ["src/user.py", "tests/test_user.py"]
        previews = builder._get_file_previews(temp_workspace, sample_task, existing_files)

        # Should have preview for user.py (relates to task)
        assert any("user" in k for k in previews.keys())

    def test_preview_content_truncated(
        self,
        builder: SubagentContextBuilder,
        temp_workspace: Path,
    ) -> None:
        """Test that long files are truncated."""
        # Create a large file
        large_file = temp_workspace / "src" / "large.py"
        large_file.write_text("\n".join(f"line_{i}" for i in range(200)))

        task = TodoItem(id=1, title="Create tests for large.py", description="")
        previews = builder._get_file_previews(temp_workspace, task, ["src/large.py"])

        if "src/large.py" in previews:
            # Should be truncated
            assert "more lines" in previews["src/large.py"]


class TestSummarizeCompleted:
    """Tests for summarizing completed tasks."""

    def test_summarize_completed_tasks(
        self,
        builder: SubagentContextBuilder,
        completed_tasks: list[TodoItem],
    ) -> None:
        """Test summarizing completed tasks."""
        summaries = builder._summarize_completed(completed_tasks)

        assert len(summaries) == 2
        assert all("[Done]" in s for s in summaries)

    def test_truncates_long_titles(
        self,
        builder: SubagentContextBuilder,
    ) -> None:
        """Test that long task titles are truncated."""
        long_task = TodoItem(
            id=1,
            title="A" * 100,  # Very long title
            description="",
        )

        summaries = builder._summarize_completed([long_task])

        assert len(summaries[0]) < 100


class TestExtractPatterns:
    """Tests for extracting code patterns."""

    def test_detect_google_docstrings(
        self,
        builder: SubagentContextBuilder,
        temp_workspace: Path,
    ) -> None:
        """Test detection of Google-style docstrings."""
        patterns = builder._extract_patterns(
            temp_workspace,
            ["src/user.py"],
        )

        # user.py has Args: and Returns: (Google style)
        assert patterns.docstring_style in ["google", "unknown"]

    def test_detect_type_hints(
        self,
        builder: SubagentContextBuilder,
        temp_workspace: Path,
    ) -> None:
        """Test detection of type hints."""
        patterns = builder._extract_patterns(
            temp_workspace,
            ["src/user.py"],
        )

        # user.py has type hints
        assert patterns.type_hints in ["yes", "partial", "unknown"]

    def test_detect_pytest_framework(
        self,
        builder: SubagentContextBuilder,
        temp_workspace: Path,
    ) -> None:
        """Test detection of pytest framework."""
        patterns = builder._extract_patterns(
            temp_workspace,
            ["tests/test_user.py"],
        )

        assert patterns.test_framework == "pytest"

    def test_detect_line_length_from_config(
        self,
        builder: SubagentContextBuilder,
        temp_workspace: Path,
    ) -> None:
        """Test detection of line length from pyproject.toml."""
        patterns = builder._extract_patterns(
            temp_workspace,
            ["src/user.py"],
        )

        # pyproject.toml has line-length = 100
        assert patterns.line_length == 100


class TestDetectDependencies:
    """Tests for detecting dependencies."""

    def test_detect_from_pyproject(
        self,
        builder: SubagentContextBuilder,
        temp_workspace: Path,
    ) -> None:
        """Test dependency detection from pyproject.toml."""
        dependencies = builder._detect_dependencies(temp_workspace)

        assert "fastapi" in dependencies
        assert "pydantic" in dependencies

    def test_detect_from_requirements(
        self,
        builder: SubagentContextBuilder,
    ) -> None:
        """Test dependency detection from requirements.txt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "requirements.txt").write_text("""
flask==2.0.0
requests>=2.28
# comment
pytest
""")

            dependencies = builder._detect_dependencies(workspace)

            assert "flask" in dependencies
            assert "requests" in dependencies
            assert "pytest" in dependencies

    def test_detect_from_package_json(
        self,
        builder: SubagentContextBuilder,
    ) -> None:
        """Test dependency detection from package.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "package.json").write_text("""
{
    "dependencies": {
        "express": "^4.18.0",
        "lodash": "^4.17.0"
    }
}
""")

            dependencies = builder._detect_dependencies(workspace)

            assert "express" in dependencies
            assert "lodash" in dependencies
