"""Integration tests for Python project execution (Phase 6.2.1).

Tests end-to-end execution of the executor graph with Python projects,
including:
- Simple roadmap execution
- Syntax error handling
- Test failure recovery

These tests require actual LLM API access and are skipped by default.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.executor.graph import ExecutorGraph

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def python_project(tmp_path: Path) -> Path:
    """Create a minimal Python project."""
    # pyproject.toml
    (tmp_path / "pyproject.toml").write_text("""\
[project]
name = "test-project"
version = "0.1.0"
requires-python = ">=3.11"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
""")

    # Source directory
    src = tmp_path / "src"
    src.mkdir()
    (src / "__init__.py").write_text("")

    # Tests directory
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("")

    return tmp_path


@pytest.fixture
def simple_roadmap(tmp_path: Path) -> Path:
    """Create a simple roadmap with two tasks."""
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text("""\
# Test Roadmap

## Overview
Create a simple hello module with tests.

## Tasks

### Phase 1: Implementation

- [ ] **Create hello module**
  - Description: Create src/hello.py with a hello() function that returns "Hello, World!"
  - Files: src/hello.py

- [ ] **Add unit tests**
  - Description: Create tests/test_hello.py with a test for hello()
  - Files: tests/test_hello.py
  - Depends: Create hello module
""")
    return roadmap


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create a mock agent for testing."""
    agent = MagicMock()
    agent.model = "claude-sonnet-4-20250514"
    return agent


# =============================================================================
# Mock Execution Tests (no LLM required)
# =============================================================================


class TestPythonProjectSetup:
    """Tests for Python project fixture setup."""

    def test_project_structure_exists(self, python_project: Path) -> None:
        """Verify project structure is created correctly."""
        assert (python_project / "pyproject.toml").exists()
        assert (python_project / "src").is_dir()
        assert (python_project / "src" / "__init__.py").exists()
        assert (python_project / "tests").is_dir()
        assert (python_project / "tests" / "__init__.py").exists()

    def test_roadmap_structure(self, simple_roadmap: Path) -> None:
        """Verify roadmap is created correctly."""
        content = simple_roadmap.read_text()
        assert "# Test Roadmap" in content
        assert "Create hello module" in content
        assert "Add unit tests" in content


class TestExecutorGraphInitialization:
    """Tests for ExecutorGraph initialization with Python projects."""

    def test_init_with_roadmap_path(
        self, python_project: Path, simple_roadmap: Path, mock_agent: MagicMock
    ) -> None:
        """Can initialize with roadmap path."""
        executor = ExecutorGraph(
            agent=mock_agent,
            roadmap_path=str(simple_roadmap),
        )

        assert executor.roadmap_path == str(simple_roadmap)

    def test_init_with_shell_workspace(
        self, python_project: Path, simple_roadmap: Path, mock_agent: MagicMock
    ) -> None:
        """Can initialize with shell workspace."""
        executor = ExecutorGraph(
            agent=mock_agent,
            roadmap_path=str(simple_roadmap),
            shell_workspace=python_project,
        )

        assert executor.shell_workspace == python_project

    def test_init_with_dry_run(
        self, python_project: Path, simple_roadmap: Path, mock_agent: MagicMock
    ) -> None:
        """Can initialize with dry run mode."""
        executor = ExecutorGraph(
            agent=mock_agent,
            roadmap_path=str(simple_roadmap),
            dry_run=True,
        )

        assert executor.dry_run is True


class TestMockedExecution:
    """Tests using mocked LLM execution."""

    @pytest.fixture
    def mock_graph(self) -> MagicMock:
        """Create a mock LangGraph."""
        graph = MagicMock()
        graph.arun = AsyncMock()
        return graph

    @pytest.mark.asyncio
    async def test_executor_runs_with_mock(
        self, python_project: Path, simple_roadmap: Path, mock_agent: MagicMock
    ) -> None:
        """Executor can be called with mocked graph."""
        executor = ExecutorGraph(
            agent=mock_agent,
            roadmap_path=str(simple_roadmap),
            shell_workspace=python_project,
        )

        # Mock the graph execution
        with patch.object(executor, "graph") as mock_graph:
            mock_result: dict[str, Any] = {
                "status": "completed",
                "tasks_completed_count": 2,
                "tasks_failed_count": 0,
                "files_modified": ["src/hello.py", "tests/test_hello.py"],
            }
            mock_graph.arun = AsyncMock(return_value=mock_result)

            result = await executor.arun()

            assert result["status"] == "completed"
            assert result["tasks_completed_count"] == 2

    @pytest.mark.asyncio
    async def test_executor_handles_errors(
        self, python_project: Path, simple_roadmap: Path, mock_agent: MagicMock
    ) -> None:
        """Executor handles errors gracefully."""
        executor = ExecutorGraph(
            agent=mock_agent,
            roadmap_path=str(simple_roadmap),
            shell_workspace=python_project,
            max_retries=2,
        )

        with patch.object(executor, "graph") as mock_graph:
            mock_result: dict[str, Any] = {
                "status": "completed",
                "tasks_completed_count": 1,
                "tasks_failed_count": 1,
                "error": "Task 2 failed",
            }
            mock_graph.arun = AsyncMock(return_value=mock_result)

            result = await executor.arun()

            assert result["tasks_failed_count"] == 1


# =============================================================================
# Integration Tests (require LLM API)
# =============================================================================


@pytest.mark.skip(reason="Integration test - requires LLM API")
class TestPythonProjectExecution:
    """Integration tests for Python project execution.

    These tests run actual LLM calls and should only be run manually
    or in environments with LLM API access.
    """

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_simple_roadmap_execution(
        self, python_project: Path, simple_roadmap: Path
    ) -> None:
        """Execute a simple roadmap end-to-end."""
        executor = ExecutorGraph(
            roadmap_path=str(simple_roadmap),
            shell_workspace=python_project,
        )

        result = await executor.arun()

        assert result.get("status") == "completed"
        assert (python_project / "src" / "hello.py").exists()
        assert (python_project / "tests" / "test_hello.py").exists()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_handles_syntax_errors(self, python_project: Path) -> None:
        """Should recover from syntax errors."""
        # Setup: Create file with syntax error
        (python_project / "src" / "broken.py").write_text("def broken(")

        roadmap = python_project / "ROADMAP.md"
        roadmap.write_text("""\
# Fix Roadmap

## Tasks

- [ ] **Fix syntax error**
  - Description: Fix the syntax error in src/broken.py
  - Files: src/broken.py
""")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            shell_workspace=python_project,
        )
        result = await executor.arun()

        assert result.get("status") == "completed"

        # Verify file is now valid Python
        content = (python_project / "src" / "broken.py").read_text()
        compile(content, "broken.py", "exec")  # Should not raise

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_test_failure_recovery(self, python_project: Path) -> None:
        """Should fix code when tests fail."""
        # Setup: Create function with bug and failing test
        (python_project / "src" / "math_ops.py").write_text("""\
def add(a: int, b: int) -> int:
    return a - b  # Bug: should be a + b
""")
        (python_project / "tests" / "test_math_ops.py").write_text("""\
from src.math_ops import add

def test_add():
    assert add(2, 3) == 5
""")

        roadmap = python_project / "ROADMAP.md"
        roadmap.write_text("""\
# Fix Tests

## Tasks

- [ ] **Fix failing test**
  - Description: Fix the implementation so test_add passes
  - Files: src/math_ops.py
""")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            shell_workspace=python_project,
        )
        result = await executor.arun()

        assert result.get("status") == "completed"

        # Verify tests pass now
        proc = subprocess.run(
            ["pytest", str(python_project / "tests")],
            capture_output=True,
            cwd=python_project,
        )
        assert proc.returncode == 0


# =============================================================================
# Verification Helper Tests
# =============================================================================


class TestVerificationHelpers:
    """Tests for verification helper functions."""

    def test_python_syntax_valid(self) -> None:
        """Can validate Python syntax."""
        valid_code = "def hello():\n    return 'Hello'\n"
        compile(valid_code, "<test>", "exec")  # Should not raise

    def test_python_syntax_invalid(self) -> None:
        """Can detect Python syntax errors."""
        invalid_code = "def broken("
        with pytest.raises(SyntaxError):
            compile(invalid_code, "<test>", "exec")

    def test_subprocess_pytest(self, python_project: Path) -> None:
        """Can run pytest via subprocess."""
        # Create a simple passing test
        (python_project / "tests" / "test_simple.py").write_text("""\
def test_passes():
    assert True
""")

        proc = subprocess.run(
            ["python", "-m", "pytest", str(python_project / "tests"), "-v"],
            capture_output=True,
            cwd=python_project,
        )

        # Should pass
        assert proc.returncode == 0
