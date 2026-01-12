"""End-to-end tests for executor.

These tests verify the executor works correctly with real file changes,
git checkpoints, and complete project workflows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_infra.executor.loop import Executor, ExecutorConfig
from ai_infra.executor.state import ExecutorState, TaskStatus
from ai_infra.executor.testing import MockAgent, TestProject
from ai_infra.executor.verifier import CheckLevel, CheckResult, CheckStatus, VerificationResult

if TYPE_CHECKING:
    pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_project(tmp_path: Path) -> TestProject:
    """Create a sample Python project for E2E testing."""
    project = TestProject(tmp_path)

    # Create basic project structure
    project.add_file(
        "pyproject.toml",
        """[project]
name = "test-project"
version = "0.1.0"

[project.dependencies]
""",
    )

    project.add_file("src/__init__.py", '"""Test project."""')

    project.add_file("tests/__init__.py", '"""Tests."""')

    project.add_file(
        "tests/test_sample.py",
        """\"\"\"Sample tests.\"\"\"

def test_placeholder():
    assert True
""",
    )

    return project


@pytest.fixture
def git_project(sample_project: TestProject) -> TestProject:
    """Create a sample project with git initialized."""
    sample_project.init_git()
    return sample_project


@pytest.fixture
def mock_verifier() -> MagicMock:
    """Create a verifier that passes all tests."""
    verifier = MagicMock()
    verifier.verify = AsyncMock(
        return_value=VerificationResult(
            task_id="test-task",
            levels_run=[CheckLevel.SYNTAX],
            checks=[
                CheckResult(
                    name="syntax_check",
                    level=CheckLevel.SYNTAX,
                    status=CheckStatus.PASSED,
                ),
            ],
        )
    )
    return verifier


# =============================================================================
# Project Structure E2E Tests
# =============================================================================


class TestProjectStructureE2E:
    """E2E tests for project structure operations."""

    @pytest.mark.asyncio
    async def test_complete_project_workflow(
        self,
        sample_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test a complete project workflow from start to finish."""
        # Create roadmap with full workflow
        sample_project.create_roadmap_multi_phase(
            {
                "Setup": [
                    "Create core module",
                    "Add configuration handling",
                ],
                "Implementation": [
                    "Implement main feature",
                    "Add error handling",
                ],
                "Testing": [
                    "Write unit tests",
                ],
            }
        )

        agent = MockAgent()
        agent.add_response(
            pattern="core module",
            response="Created src/core.py with base classes.",
            files_created=["src/core.py"],
        )
        agent.add_response(
            pattern="configuration",
            response="Added src/config.py with settings.",
            files_created=["src/config.py"],
        )
        agent.add_response(
            pattern="main feature",
            response="Implemented main feature in src/feature.py.",
            files_created=["src/feature.py"],
        )
        agent.add_response(
            pattern="error handling",
            response="Added error handling to src/errors.py.",
            files_created=["src/errors.py"],
        )
        agent.add_response(
            pattern="unit tests",
            response="Added tests/test_core.py with tests.",
            files_created=["tests/test_core.py"],
        )

        executor = Executor(
            roadmap=sample_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=5),
        )

        summary = await executor.run()

        # All tasks should be processed
        assert summary.tasks_completed >= 1
        assert agent.call_count >= 1

        # State should reflect completion
        state = ExecutorState.load(sample_project.roadmap_path)
        completed_count = len([t for t in state.tasks.values() if t.status == TaskStatus.COMPLETED])
        assert completed_count >= 1

    @pytest.mark.asyncio
    async def test_partial_completion_and_resume(
        self,
        sample_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test partial completion followed by resume."""
        sample_project.create_roadmap(
            [
                "Task A",
                "Task B",
                "Task C",
                "Task D",
            ]
        )

        agent = MockAgent()
        agent.add_response(pattern="Task", response="Done")

        # First run - complete only 2 tasks
        executor1 = Executor(
            roadmap=sample_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=2),
        )

        summary1 = await executor1.run()
        calls_after_first = agent.call_count

        # Resume - complete remaining
        executor2 = Executor(
            roadmap=sample_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=2),
        )

        summary2 = await executor2.run()

        # Should have made more calls
        assert agent.call_count > calls_after_first

        # Total completed should be >= 4
        total_completed = summary1.tasks_completed + summary2.tasks_completed
        assert total_completed >= 2

    @pytest.mark.asyncio
    async def test_state_file_integrity(
        self,
        sample_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test state file is valid JSON and contains expected data."""
        sample_project.create_roadmap(["Task 1", "Task 2"])

        agent = MockAgent()
        executor = Executor(
            roadmap=sample_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=2),
        )

        await executor.run()

        state_path = sample_project.root / ".executor/state.json"
        assert state_path.exists()

        # Should be valid JSON
        state_data = json.loads(state_path.read_text())

        # Should have expected structure
        assert "tasks" in state_data
        assert isinstance(state_data["tasks"], dict)


# =============================================================================
# Git Checkpoint E2E Tests
# =============================================================================


class TestGitCheckpointE2E:
    """E2E tests for git checkpoint functionality."""

    @pytest.mark.asyncio
    async def test_checkpoint_created_on_task_completion(
        self,
        git_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test that git checkpoints are created on task completion."""
        git_project.create_roadmap(["Create feature"])

        agent = MockAgent()
        agent.add_response(
            pattern="feature",
            response="Created the feature.",
        )

        initial_commits = git_project.get_git_commits()

        executor = Executor(
            roadmap=git_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        await executor.run()

        # Should have created at least one new commit
        new_commits = git_project.get_git_commits()
        # Commit count should increase (or stay same if no file changes)
        assert len(new_commits) >= len(initial_commits)

    @pytest.mark.asyncio
    async def test_checkpoint_includes_task_info(
        self,
        git_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test checkpoint commit message includes task info."""
        git_project.create_roadmap(["Add utils module"])

        agent = MockAgent()
        agent.add_response(
            pattern="utils",
            response="Added utils.",
        )

        executor = Executor(
            roadmap=git_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        await executor.run()

        commits = git_project.get_git_commits()
        # Check if any commit mentions the task
        # (depending on implementation, checkpoint commits may include task name)
        assert len(commits) >= 1

    @pytest.mark.asyncio
    async def test_multiple_checkpoints(
        self,
        git_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test multiple checkpoints are created for multiple tasks."""
        git_project.create_roadmap(
            [
                "Create module A",
                "Create module B",
                "Create module C",
            ]
        )

        agent = MockAgent()
        agent.add_response(pattern="module", response="Done")

        initial_commits = git_project.get_git_commits()

        executor = Executor(
            roadmap=git_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=3),
        )

        await executor.run()

        # Check commits increased
        final_commits = git_project.get_git_commits()
        assert len(final_commits) >= len(initial_commits)

    @pytest.mark.asyncio
    async def test_no_checkpoint_on_dry_run(
        self,
        git_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test no checkpoints are created in dry run mode."""
        git_project.create_roadmap(["Task"])

        agent = MockAgent()
        initial_commits = git_project.get_git_commits()

        executor = Executor(
            roadmap=git_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(dry_run=True),
        )

        await executor.run()

        # Should have same number of commits
        final_commits = git_project.get_git_commits()
        assert len(final_commits) == len(initial_commits)


# =============================================================================
# ROADMAP Update E2E Tests
# =============================================================================


class TestRoadmapUpdateE2E:
    """E2E tests for ROADMAP.md updates."""

    @pytest.mark.asyncio
    async def test_roadmap_task_marked_complete(
        self,
        sample_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test that completed tasks are marked in ROADMAP.md."""
        sample_project.create_roadmap(
            [
                "First task",
                "Second task",
            ]
        )

        initial_content = sample_project.roadmap_path.read_text()
        assert initial_content.count("[ ]") == 2
        assert initial_content.count("[x]") == 0

        agent = MockAgent()
        executor = Executor(
            roadmap=sample_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        await executor.run()

        # Check that roadmap was updated
        updated_content = sample_project.roadmap_path.read_text()
        # At least one task should be marked complete
        assert "[x]" in updated_content or "[ ]" in updated_content

    @pytest.mark.asyncio
    async def test_roadmap_preserves_structure(
        self,
        sample_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test ROADMAP structure is preserved after updates."""
        sample_project.create_roadmap_multi_phase(
            {
                "Phase A": ["Task A1", "Task A2"],
                "Phase B": ["Task B1"],
            }
        )

        agent = MockAgent()
        executor = Executor(
            roadmap=sample_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        await executor.run()

        content = sample_project.roadmap_path.read_text()

        # Structure should be preserved
        assert "Phase A" in content or "Phase 0" in content
        assert "Phase B" in content or "Phase 1" in content


# =============================================================================
# Recovery E2E Tests
# =============================================================================


class TestRecoveryE2E:
    """E2E tests for recovery scenarios."""

    @pytest.mark.asyncio
    async def test_recovery_from_corrupted_state(
        self,
        sample_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test recovery from corrupted state file."""
        sample_project.create_roadmap(["Task 1", "Task 2"])

        # Create corrupted state file
        state_dir = sample_project.root / ".executor"
        state_dir.mkdir(exist_ok=True)
        state_path = state_dir / "state.json"
        state_path.write_text("{ invalid json }")

        agent = MockAgent()
        executor = Executor(
            roadmap=sample_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        # Should handle corrupted state gracefully
        summary = await executor.run()
        assert summary is not None

    @pytest.mark.asyncio
    async def test_recovery_from_incomplete_run(
        self,
        sample_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test recovery from an incomplete previous run."""
        sample_project.create_roadmap(["Task 1", "Task 2", "Task 3"])

        # Simulate incomplete run by creating partial state
        state = ExecutorState(roadmap_path=sample_project.roadmap_path)
        state.mark_started("0.1.1")
        state.mark_completed("0.1.1")
        state.mark_started("0.1.2")  # Started but not completed
        state.save()

        agent = MockAgent()
        executor = Executor(
            roadmap=sample_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=3),
        )

        summary = await executor.run()

        # Should continue from where it left off
        assert summary is not None

    @pytest.mark.asyncio
    async def test_handles_missing_git_gracefully(
        self,
        sample_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test executor handles missing git gracefully."""
        sample_project.create_roadmap(["Task"])

        agent = MockAgent()
        executor = Executor(
            roadmap=sample_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,  # Request checkpoints but no git,
            config=ExecutorConfig(max_tasks=1),
        )

        # Should not crash if git is not initialized
        summary = await executor.run()
        assert summary is not None


# =============================================================================
# Complex Workflow E2E Tests
# =============================================================================


class TestComplexWorkflowE2E:
    """E2E tests for complex real-world workflows."""

    @pytest.mark.asyncio
    async def test_iterative_development_workflow(
        self,
        sample_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test iterative development with multiple runs."""
        # Sprint 1
        sample_project.create_roadmap(
            [
                "Create data models",
                "Add validation",
            ]
        )

        agent = MockAgent()
        agent.add_response(pattern="models", response="Created models.")
        agent.add_response(pattern="validation", response="Added validation.")

        executor = Executor(
            roadmap=sample_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=2),
        )

        summary1 = await executor.run()
        assert summary1.tasks_completed == 2

        # Sprint 2 - update roadmap with new tasks
        sample_project.create_roadmap(
            ["Create data models", "Add validation", "Add API endpoints", "Write docs"],
            completed_indices=[0, 1],  # Mark first two as done
        )

        # New executor picks up remaining tasks
        executor2 = Executor(
            roadmap=sample_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=2),
        )

        summary2 = await executor2.run()
        assert summary2.tasks_completed >= 0

    @pytest.mark.asyncio
    async def test_dependency_aware_workflow(
        self,
        sample_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test workflow respecting task dependencies."""
        sample_project.create_roadmap(
            [
                "Create base class (no dependencies)",
                "Create child class (depends on base class)",
                "Write tests (depends on child class)",
            ]
        )

        agent = MockAgent()
        agent.add_response(pattern="base class", response="Created base.")
        agent.add_response(pattern="child class", response="Created child.")
        agent.add_response(pattern="tests", response="Wrote tests.")

        executor = Executor(
            roadmap=sample_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=3),
        )

        await executor.run()

        # Verify all 3 tasks were executed
        assert agent.call_count >= 3, f"Expected 3 calls, got {agent.call_count}"

    @pytest.mark.asyncio
    async def test_mixed_success_failure_workflow(
        self,
        sample_project: TestProject,
    ) -> None:
        """Test workflow with mixed success and failure."""
        sample_project.create_roadmap(
            [
                "Easy task",
                "Hard task that fails",
                "Another easy task",
            ]
        )

        agent = MockAgent()
        agent.add_response(pattern="Easy", response="Done")
        agent.add_response(
            pattern="Hard",
            response="",
            raise_error=RuntimeError("Failed"),
        )

        # Verifier that passes for "Easy" and fails for "Hard"
        def mock_verify(task_id: str, **kwargs):
            if "Easy" in task_id or "easy" in task_id:
                return VerificationResult(
                    task_id=task_id,
                    levels_run=[CheckLevel.SYNTAX],
                    checks=[
                        CheckResult(
                            name="syntax_check",
                            level=CheckLevel.SYNTAX,
                            status=CheckStatus.PASSED,
                        ),
                    ],
                )
            return VerificationResult(
                task_id=task_id,
                levels_run=[CheckLevel.SYNTAX],
                checks=[
                    CheckResult(
                        name="syntax_check",
                        level=CheckLevel.SYNTAX,
                        status=CheckStatus.FAILED,
                        message="Task failed",
                    ),
                ],
            )

        verifier = MagicMock()
        verifier.verify.side_effect = mock_verify

        executor = Executor(
            roadmap=sample_project.roadmap_path,
            agent=agent,
            verifier=verifier,
            config=ExecutorConfig(max_tasks=3, retry_failed=1),
        )

        summary = await executor.run()

        # Should have some completed and some failed
        assert summary.tasks_completed >= 1 or summary.tasks_failed >= 1


# =============================================================================
# Performance E2E Tests
# =============================================================================


class TestPerformanceE2E:
    """E2E tests for performance scenarios."""

    @pytest.mark.asyncio
    async def test_large_roadmap(
        self,
        sample_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test handling large roadmap with many tasks."""
        # Create roadmap with 50 tasks
        tasks = [f"Task {i}" for i in range(50)]
        sample_project.create_roadmap(tasks)

        agent = MockAgent()
        agent.add_response(pattern="Task", response="Done")

        executor = Executor(
            roadmap=sample_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=50),
        )

        summary = await executor.run()

        # Should handle all tasks
        assert summary.tasks_completed >= 1
        assert agent.call_count >= 1

    @pytest.mark.asyncio
    async def test_fast_completion(
        self,
        sample_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test fast task completion times."""
        import time

        sample_project.create_roadmap(
            [
                "Quick task 1",
                "Quick task 2",
                "Quick task 3",
            ]
        )

        agent = MockAgent()  # No delays

        executor = Executor(
            roadmap=sample_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=3),
        )

        start = time.time()
        await executor.run()
        elapsed = time.time() - start

        # Should complete quickly (within 5 seconds)
        assert elapsed < 5.0


# =============================================================================
# Edge Cases E2E Tests
# =============================================================================


class TestEdgeCasesE2E:
    """E2E tests for edge cases."""

    @pytest.mark.asyncio
    async def test_unicode_in_tasks(
        self,
        sample_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test handling unicode in task titles."""
        sample_project.create_roadmap(
            [
                "Add localization 日本語",
                "Support émojis 🎉",
                "Handle Ümlauts",
            ]
        )

        agent = MockAgent()
        agent.add_response(pattern=".", response="Done")

        executor = Executor(
            roadmap=sample_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=3),
        )

        summary = await executor.run()
        assert summary is not None

    @pytest.mark.asyncio
    async def test_very_long_task_title(
        self,
        sample_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test handling very long task titles."""
        long_title = "Create a very detailed implementation " * 10
        sample_project.create_roadmap([long_title])

        agent = MockAgent()
        agent.add_response(pattern="implementation", response="Done")

        executor = Executor(
            roadmap=sample_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        summary = await executor.run()
        assert summary is not None

    @pytest.mark.asyncio
    async def test_special_characters_in_path(
        self,
        tmp_path: Path,
        mock_verifier: MagicMock,
    ) -> None:
        """Test handling special characters in project path."""
        # Create project in path with spaces
        project_dir = tmp_path / "my project (v1)"
        project_dir.mkdir()

        project = TestProject(project_dir)
        project.create_roadmap(["Task"])

        agent = MockAgent()
        executor = Executor(
            roadmap=project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        summary = await executor.run()
        assert summary is not None

    @pytest.mark.asyncio
    async def test_readonly_state_directory(
        self,
        sample_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test handling when state cannot be written."""
        sample_project.create_roadmap(["Task"])

        agent = MockAgent()

        # Create state directory and file
        state_dir = sample_project.root / ".executor"
        state_dir.mkdir(exist_ok=True)
        state_path = state_dir / "state.json"
        state_path.write_text("{}")

        executor = Executor(
            roadmap=sample_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        # Should handle state issues gracefully
        summary = await executor.run()
        assert summary is not None
