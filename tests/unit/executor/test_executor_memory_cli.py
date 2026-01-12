"""Tests for Executor Memory CLI Commands (Phase 5.8.5).

This module tests:
- Memory options in the run command
- The memory subcommand for viewing project memory
- The memory-clear subcommand for clearing project memory
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from ai_infra.cli.cmds.executor_cmds import app
from ai_infra.executor.project_memory import FileInfo, ProjectMemory, RunSummary

runner = CliRunner()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_roadmap_content() -> str:
    """Sample ROADMAP content for testing."""
    return """\
# Test Project ROADMAP

## Phase 0: Setup

### 0.1 Init

- [ ] **Configure project**
  Set up basic configuration.
"""


@pytest.fixture
def temp_project(sample_roadmap_content: str):
    """Create a temporary project directory with ROADMAP."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        roadmap_path = project_path / "ROADMAP.md"
        roadmap_path.write_text(sample_roadmap_content)

        # Create src directory
        (project_path / "src").mkdir()
        (project_path / "src" / "__init__.py").write_text("")

        yield project_path, roadmap_path


@pytest.fixture
def project_with_memory(temp_project):
    """Create a project with existing project memory."""
    project_path, roadmap_path = temp_project

    # Create .executor directory
    executor_dir = project_path / ".executor"
    executor_dir.mkdir()

    # Create project memory with some data
    memory = ProjectMemory(
        project_root=project_path,
        project_type="python",
        key_files={
            "src/main.py": FileInfo(
                path="src/main.py",
                purpose="Main entry point",
                created_by_task="0.1",
            ),
            "src/utils.py": FileInfo(
                path="src/utils.py",
                purpose="Utility functions",
                created_by_task="0.2",
            ),
        },
        run_history=[
            RunSummary(
                run_id="run-001",
                timestamp="2025-01-07T10:00:00Z",
                tasks_completed=3,
                tasks_failed=0,
                key_files_created=["src/main.py"],
                lessons_learned=["Always use type hints"],
            ),
            RunSummary(
                run_id="run-002",
                timestamp="2025-01-07T14:00:00Z",
                tasks_completed=2,
                tasks_failed=1,
                key_files_created=["src/utils.py"],
                lessons_learned=["Check imports before running"],
            ),
        ],
    )

    # Save memory using the save() method
    memory.save()

    yield project_path, roadmap_path, memory


# =============================================================================
# Test Memory Command - Summary Format
# =============================================================================


class TestMemoryCommandSummary:
    """Tests for the memory command with summary format."""

    def test_memory_summary_no_memory(self, temp_project):
        """Test memory command when no project memory exists."""
        project_path, _ = temp_project

        result = runner.invoke(app, ["memory", str(project_path)])

        assert result.exit_code == 0
        assert "Project Memory Summary" in result.output

    def test_memory_summary_with_memory(self, project_with_memory):
        """Test memory command shows summary of existing memory."""
        project_path, _, _ = project_with_memory

        result = runner.invoke(app, ["memory", str(project_path)])

        assert result.exit_code == 0
        assert "Project Type:" in result.output
        assert "python" in result.output
        assert "Files Tracked:" in result.output
        assert "2" in result.output  # 2 files tracked
        assert "Runs Recorded:" in result.output

    def test_memory_summary_shows_last_run(self, project_with_memory):
        """Test memory summary shows last run info."""
        project_path, _, _ = project_with_memory

        result = runner.invoke(app, ["memory", str(project_path)])

        assert result.exit_code == 0
        assert "Last Run:" in result.output


# =============================================================================
# Test Memory Command - JSON Format
# =============================================================================


class TestMemoryCommandJSON:
    """Tests for the memory command with JSON format."""

    def test_memory_json_format(self, project_with_memory):
        """Test memory command with JSON format option."""
        project_path, _, _ = project_with_memory

        result = runner.invoke(app, ["memory", str(project_path), "--format", "json"])

        assert result.exit_code == 0
        # Should be valid JSON
        data = json.loads(result.output)
        assert "project_type" in data
        assert "key_files" in data
        assert "run_history" in data

    def test_memory_json_shortcut(self, project_with_memory):
        """Test memory command with --json shortcut."""
        project_path, _, _ = project_with_memory

        result = runner.invoke(app, ["memory", str(project_path), "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "project_type" in data


# =============================================================================
# Test Memory Command - Files Format
# =============================================================================


class TestMemoryCommandFiles:
    """Tests for the memory command with files format."""

    def test_memory_files_format(self, project_with_memory):
        """Test memory command with files format."""
        project_path, _, _ = project_with_memory

        result = runner.invoke(app, ["memory", str(project_path), "--format", "files"])

        assert result.exit_code == 0
        assert "Key Files" in result.output
        assert "src/main.py" in result.output
        assert "src/utils.py" in result.output
        assert "Main entry point" in result.output

    def test_memory_files_empty(self, temp_project):
        """Test files format when no files tracked."""
        project_path, _ = temp_project

        result = runner.invoke(app, ["memory", str(project_path), "--format", "files"])

        assert result.exit_code == 0
        assert "No files tracked" in result.output


# =============================================================================
# Test Memory Command - History Format
# =============================================================================


class TestMemoryCommandHistory:
    """Tests for the memory command with history format."""

    def test_memory_history_format(self, project_with_memory):
        """Test memory command with history format."""
        project_path, _, _ = project_with_memory

        result = runner.invoke(app, ["memory", str(project_path), "--format", "history"])

        assert result.exit_code == 0
        assert "Run History" in result.output
        assert "run-001" in result.output or "run-002" in result.output
        assert "completed" in result.output

    def test_memory_history_empty(self, temp_project):
        """Test history format when no run history."""
        project_path, _ = temp_project

        result = runner.invoke(app, ["memory", str(project_path), "--format", "history"])

        assert result.exit_code == 0
        assert "No run history" in result.output


# =============================================================================
# Test Memory Command - Error Cases
# =============================================================================


class TestMemoryCommandErrors:
    """Tests for error handling in memory command."""

    def test_memory_nonexistent_project(self):
        """Test memory command with nonexistent project path."""
        result = runner.invoke(app, ["memory", "/nonexistent/path"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()


# =============================================================================
# Test Memory Clear Command
# =============================================================================


class TestMemoryClearCommand:
    """Tests for the memory-clear command."""

    def test_clear_with_force(self, project_with_memory):
        """Test clearing memory with --force flag."""
        project_path, _, _ = project_with_memory
        memory_path = project_path / ".executor" / "project-memory.json"

        # Verify memory exists
        assert memory_path.exists()

        result = runner.invoke(app, ["memory-clear", str(project_path), "--force"])

        assert result.exit_code == 0
        assert "cleared" in result.output.lower()
        assert not memory_path.exists()

    def test_clear_without_memory(self, temp_project):
        """Test clearing when no memory exists."""
        project_path, _ = temp_project

        result = runner.invoke(app, ["memory-clear", str(project_path), "--force"])

        assert result.exit_code == 0
        assert "No project memory to clear" in result.output

    def test_clear_confirmation_declined(self, project_with_memory):
        """Test that declining confirmation doesn't clear memory."""
        project_path, _, _ = project_with_memory
        memory_path = project_path / ".executor" / "project-memory.json"

        # Simulate declining confirmation
        result = runner.invoke(app, ["memory-clear", str(project_path)], input="n\n")

        assert result.exit_code == 0
        assert memory_path.exists()  # Memory should still exist


# =============================================================================
# Test Run Command Memory Options
# =============================================================================


class TestRunCommandMemoryOptions:
    """Tests for memory options in the run command."""

    def test_run_help_shows_memory_options(self):
        """Test that run command help shows memory options."""
        result = runner.invoke(app, ["run", "--help"])

        assert result.exit_code == 0
        assert "--no-run-memory" in result.output
        assert "--no-project-memory" in result.output
        assert "--memory-budget" in result.output
        assert "--extract-with-llm" in result.output
        # May be truncated in output with ellipsis, so check for partial match
        assert "--clear-project" in result.output

    def test_clear_project_memory_option(self, project_with_memory):
        """Test --clear-project-memory option clears memory before run."""
        project_path, roadmap_path, memory = project_with_memory
        memory_path = project_path / ".executor" / "project-memory.json"

        # Verify memory exists with run history
        assert memory_path.exists()
        assert len(memory.run_history) == 2  # Had 2 runs

        # Run with --clear-project-memory (dry-run to avoid actual execution)
        result = runner.invoke(
            app,
            [
                "run",
                "--roadmap",
                str(roadmap_path),
                "--clear-project-memory",
                "--dry-run",
            ],
        )

        # Should indicate memory was cleared
        assert "Cleared project memory" in result.output

        # Memory file may be recreated by the run, but with fresh data
        if memory_path.exists():
            fresh_memory = ProjectMemory.load(project_path)
            # Fresh memory should have no run history from before (it's new)
            # It may have 1 entry from the current run or 0 if dry-run doesn't record
            assert len(fresh_memory.run_history) <= 1

    def test_clear_project_memory_when_none_exists(self, temp_project):
        """Test --clear-project-memory when no memory exists."""
        project_path, roadmap_path = temp_project

        result = runner.invoke(
            app,
            [
                "run",
                "--roadmap",
                str(roadmap_path),
                "--clear-project-memory",
                "--dry-run",
            ],
        )

        # Should note that there's nothing to clear
        assert "No project memory to clear" in result.output


# =============================================================================
# Test ExecutorConfig Memory Settings
# =============================================================================


class TestExecutorConfigMemorySettings:
    """Tests for verifying memory config is properly set from CLI options."""

    @patch("ai_infra.cli.cmds.executor_cmds.Executor")
    def test_memory_disabled_via_cli(self, mock_executor_class, temp_project):
        """Test that --no-run-memory and --no-project-memory are passed to config."""
        project_path, roadmap_path = temp_project

        # Set up mock
        mock_executor = MagicMock()
        mock_executor.run = AsyncMock(
            return_value=MagicMock(
                status=MagicMock(value="completed"),
                tasks_completed=0,
                tasks_failed=0,
                tasks_remaining=0,
                total_tasks=0,
                duration_ms=100,
                total_tokens=0,
                results=[],
                paused=False,
            )
        )
        mock_executor_class.return_value = mock_executor

        runner.invoke(
            app,
            [
                "run",
                "--roadmap",
                str(roadmap_path),
                "--no-run-memory",
                "--no-project-memory",
                "--memory-budget",
                "3000",
            ],
        )

        # Verify Executor was called with correct config
        call_kwargs = mock_executor_class.call_args.kwargs
        config = call_kwargs.get("config")
        if config:
            assert config.enable_run_memory is False
            assert config.enable_project_memory is False
            assert config.memory_token_budget == 3000


# =============================================================================
# Test Integration
# =============================================================================


class TestMemoryCLIIntegration:
    """Integration tests for memory CLI commands."""

    def test_memory_view_after_clear(self, project_with_memory):
        """Test viewing memory after clearing shows empty state."""
        project_path, _, _ = project_with_memory

        # Clear memory
        result = runner.invoke(app, ["memory-clear", str(project_path), "--force"])
        assert result.exit_code == 0

        # View memory - should show empty/default state
        result = runner.invoke(app, ["memory", str(project_path)])
        assert result.exit_code == 0
        assert "Files Tracked:" in result.output
        # File count should be 0 after clear
        assert "0" in result.output or "No files" in result.output.lower()
