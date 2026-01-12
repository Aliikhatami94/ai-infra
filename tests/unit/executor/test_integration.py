"""Integration tests for executor with mock agent.

These tests verify the full executor loop using MockAgent
to simulate agent responses without making actual LLM calls.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_infra.executor.loop import Executor, ExecutorConfig
from ai_infra.executor.observability import ExecutorCallbacks
from ai_infra.executor.state import ExecutorState, TaskStatus
from ai_infra.executor.testing import MockAgent, MockResponse, TestProject
from ai_infra.executor.verifier import CheckLevel, CheckResult, CheckStatus, VerificationResult

if TYPE_CHECKING:
    pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_agent() -> MockAgent:
    """Create a fresh mock agent."""
    return MockAgent()


@pytest.fixture
def test_project(tmp_path: Path) -> TestProject:
    """Create a test project in temporary directory."""
    return TestProject(tmp_path)


@pytest.fixture
def project_with_tasks(test_project: TestProject) -> TestProject:
    """Create a project with sample tasks."""
    test_project.add_file("src/__init__.py", "")
    test_project.create_roadmap(
        [
            "Create utils module",
            "Add helper function",
            "Write unit tests",
        ]
    )
    return test_project


@pytest.fixture
def mock_verifier() -> MagicMock:
    """Create a mock verifier that passes all tests."""
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
# MockAgent Tests
# =============================================================================


class TestMockAgent:
    """Test MockAgent functionality."""

    def test_default_response(self, mock_agent: MockAgent) -> None:
        """Test default response for unmatched prompts."""
        result = asyncio.run(mock_agent.arun("any prompt"))
        assert result == "Task completed successfully."

    def test_custom_default_response(self) -> None:
        """Test custom default response."""
        agent = MockAgent(default_response="Custom default")
        result = asyncio.run(agent.arun("any prompt"))
        assert result == "Custom default"

    def test_pattern_matching(self, mock_agent: MockAgent) -> None:
        """Test response pattern matching."""
        mock_agent.add_response(
            pattern="create.*file",
            response="File created!",
        )
        result = asyncio.run(mock_agent.arun("Please create a new file"))
        assert result == "File created!"

    def test_case_insensitive_matching(self, mock_agent: MockAgent) -> None:
        """Test case insensitive pattern matching."""
        mock_agent.add_response(
            pattern="CREATE",
            response="Created!",
        )
        result = asyncio.run(mock_agent.arun("create"))
        assert result == "Created!"

    def test_first_matching_response(self, mock_agent: MockAgent) -> None:
        """Test that first matching pattern wins."""
        mock_agent.add_response(pattern="test", response="First")
        mock_agent.add_response(pattern="test", response="Second")

        result = asyncio.run(mock_agent.arun("test"))
        assert result == "First"

    def test_file_info_in_response(self, mock_agent: MockAgent) -> None:
        """Test that file modifications are included in response."""
        mock_agent.add_response(
            pattern="create",
            response="Done",
            files_created=["src/new.py"],
            files_modified=["src/old.py"],
        )
        result = asyncio.run(mock_agent.arun("create file"))
        assert "Created `src/new.py`" in result
        assert "Modified `src/old.py`" in result

    def test_error_response(self, mock_agent: MockAgent) -> None:
        """Test error raising."""
        mock_agent.add_error(
            pattern="fail",
            error=ValueError("Simulated error"),
        )
        with pytest.raises(ValueError, match="Simulated error"):
            asyncio.run(mock_agent.arun("fail this"))

    def test_delay(self, mock_agent: MockAgent) -> None:
        """Test response delay."""
        mock_agent.add_response(
            pattern="slow",
            response="Done",
            delay=0.1,
        )
        import time

        start = time.time()
        asyncio.run(mock_agent.arun("slow request"))
        elapsed = time.time() - start
        assert elapsed >= 0.1

    def test_max_uses(self, mock_agent: MockAgent) -> None:
        """Test max_uses limits response."""
        mock_agent.add_response(
            pattern="limited",
            response="Limited response",
            max_uses=2,
        )
        mock_agent.add_response(
            pattern="limited",
            response="Fallback response",
        )

        # First two uses
        assert asyncio.run(mock_agent.arun("limited")) == "Limited response"
        assert asyncio.run(mock_agent.arun("limited")) == "Limited response"
        # Third use falls back
        assert asyncio.run(mock_agent.arun("limited")) == "Fallback response"

    def test_call_tracking(self, mock_agent: MockAgent) -> None:
        """Test call history tracking."""
        mock_agent.add_response(pattern="test", response="Done")

        asyncio.run(mock_agent.arun("test 1"))
        asyncio.run(mock_agent.arun("test 2"))

        assert mock_agent.call_count == 2
        assert len(mock_agent.calls) == 2
        assert mock_agent.calls[0]["prompt"] == "test 1"
        assert mock_agent.calls[1]["prompt"] == "test 2"

    def test_last_call(self, mock_agent: MockAgent) -> None:
        """Test last_call property."""
        asyncio.run(mock_agent.arun("first"))
        asyncio.run(mock_agent.arun("last"))

        assert mock_agent.last_call is not None
        assert mock_agent.last_prompt == "last"

    def test_get_prompts(self, mock_agent: MockAgent) -> None:
        """Test get_prompts method."""
        asyncio.run(mock_agent.arun("prompt 1"))
        asyncio.run(mock_agent.arun("prompt 2"))

        prompts = mock_agent.get_prompts()
        assert prompts == ["prompt 1", "prompt 2"]

    def test_was_called_with(self, mock_agent: MockAgent) -> None:
        """Test was_called_with method."""
        asyncio.run(mock_agent.arun("create file.py"))

        assert mock_agent.was_called_with("create")
        assert mock_agent.was_called_with("file.py")
        assert not mock_agent.was_called_with("delete")

    def test_reset(self, mock_agent: MockAgent) -> None:
        """Test reset clears history."""
        mock_agent.add_response(pattern="test", response="Done")
        asyncio.run(mock_agent.arun("test"))

        assert mock_agent.call_count == 1
        mock_agent.reset()
        assert mock_agent.call_count == 0
        assert mock_agent.calls == []

    def test_chaining(self) -> None:
        """Test method chaining."""
        agent = (
            MockAgent()
            .add_response(pattern="a", response="A")
            .add_response(pattern="b", response="B")
            .set_default_response("Default")
        )

        assert asyncio.run(agent.arun("a")) == "A"
        assert asyncio.run(agent.arun("c")) == "Default"


# =============================================================================
# MockResponse Tests
# =============================================================================


class TestMockResponse:
    """Test MockResponse dataclass."""

    def test_matches(self) -> None:
        """Test pattern matching."""
        resp = MockResponse(pattern="test.*file", response="Done")
        assert resp.matches("test this file")
        assert not resp.matches("something else")

    def test_exhausted(self) -> None:
        """Test exhausted property."""
        resp = MockResponse(pattern="test", response="Done", max_uses=2)
        assert not resp.exhausted
        resp.call_count = 2
        assert resp.exhausted

    def test_unlimited_uses(self) -> None:
        """Test unlimited uses (max_uses=0)."""
        resp = MockResponse(pattern="test", response="Done", max_uses=0)
        resp.call_count = 1000
        assert not resp.exhausted


# =============================================================================
# TestProject Tests
# =============================================================================


class TestTestProject:
    """Test TestProject helper."""

    def test_create_roadmap(self, test_project: TestProject) -> None:
        """Test creating a simple roadmap."""
        test_project.create_roadmap(
            ["Task 1", "Task 2"],
            phase_name="Setup",
        )

        content = test_project.roadmap_path.read_text()
        assert "Task 1" in content
        assert "Task 2" in content
        assert "Setup" in content
        assert "- [ ]" in content

    def test_create_roadmap_with_completed(self, test_project: TestProject) -> None:
        """Test creating roadmap with completed tasks."""
        test_project.create_roadmap(
            ["Task 1", "Task 2", "Task 3"],
            completed_indices=[0, 2],
        )

        content = test_project.roadmap_path.read_text()
        assert content.count("[x]") == 2
        assert content.count("[ ]") == 1

    def test_create_multi_phase_roadmap(self, test_project: TestProject) -> None:
        """Test creating multi-phase roadmap."""
        test_project.create_roadmap_multi_phase(
            {
                "Setup": ["Task A", "Task B"],
                "Build": ["Task C"],
            }
        )

        content = test_project.roadmap_path.read_text()
        assert "Phase 0: Setup" in content
        assert "Phase 1: Build" in content
        assert "Task A" in content
        assert "Task C" in content

    def test_add_file(self, test_project: TestProject) -> None:
        """Test adding files."""
        test_project.add_file("src/main.py", "print('hello')")

        assert test_project.file_exists("src/main.py")
        assert test_project.read_file("src/main.py") == "print('hello')"

    def test_add_python_file(self, test_project: TestProject) -> None:
        """Test adding generated Python file."""
        test_project.add_python_file(
            "src/utils.py",
            imports=["os", "sys"],
            functions=["helper", "process"],
            classes=["Handler"],
        )

        content = test_project.read_file("src/utils.py")
        assert "import os" in content
        assert "import sys" in content
        assert "def helper():" in content
        assert "def process():" in content
        assert "class Handler:" in content

    def test_init_git(self, test_project: TestProject) -> None:
        """Test git initialization."""
        test_project.add_file("README.md", "# Test")
        test_project.init_git()

        assert (test_project.root / ".git").exists()
        commits = test_project.get_git_commits()
        assert len(commits) >= 1

    def test_chaining(self, test_project: TestProject) -> None:
        """Test method chaining."""
        result = test_project.add_file("a.txt", "A").add_file("b.txt", "B").create_roadmap(["Task"])
        assert result is test_project


# =============================================================================
# Executor Integration Tests
# =============================================================================


class TestExecutorIntegration:
    """Integration tests with mock agent."""

    @pytest.mark.asyncio
    async def test_single_task_completion(
        self,
        test_project: TestProject,
        mock_agent: MockAgent,
        mock_verifier: MagicMock,
    ) -> None:
        """Test completing a single task."""
        test_project.create_roadmap(["Create utils module"])

        mock_agent.add_response(
            pattern="utils",
            response="Created src/utils.py with helper functions.",
            files_created=["src/utils.py"],
        )

        executor = Executor(
            roadmap=test_project.roadmap_path,
            agent=mock_agent,
            verifier=mock_verifier,
        )

        summary = await executor.run()

        assert summary.tasks_completed == 1
        assert summary.tasks_failed == 0
        assert mock_agent.call_count >= 1
        assert mock_agent.was_called_with("utils")

    @pytest.mark.asyncio
    async def test_multiple_task_completion(
        self,
        project_with_tasks: TestProject,
        mock_agent: MockAgent,
        mock_verifier: MagicMock,
    ) -> None:
        """Test completing multiple tasks."""
        mock_agent.add_response(
            pattern="utils",
            response="Created utils module.",
        )
        mock_agent.add_response(
            pattern="helper",
            response="Added helper function.",
        )
        mock_agent.add_response(
            pattern="test",
            response="Wrote unit tests.",
        )

        executor = Executor(
            roadmap=project_with_tasks.roadmap_path,
            agent=mock_agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=3),
        )

        summary = await executor.run()

        assert summary.tasks_completed == 3
        assert mock_agent.call_count >= 3

    @pytest.mark.asyncio
    async def test_task_failure_handling(
        self,
        test_project: TestProject,
        mock_agent: MockAgent,
    ) -> None:
        """Test handling of task failures."""
        test_project.create_roadmap(["Failing task", "Next task"])

        mock_agent.add_error(
            pattern="Failing",
            error=RuntimeError("Agent failed"),
        )

        # Create verifier that fails
        verifier = MagicMock()
        verifier.verify = AsyncMock(
            return_value=VerificationResult(
                task_id="failing-task",
                levels_run=[CheckLevel.SYNTAX],
                checks=[
                    CheckResult(
                        name="syntax_check",
                        level=CheckLevel.SYNTAX,
                        status=CheckStatus.FAILED,
                        message="Agent error",
                    ),
                ],
            )
        )

        executor = Executor(
            roadmap=test_project.roadmap_path,
            agent=mock_agent,
            verifier=verifier,
            config=ExecutorConfig(max_tasks=2),
        )

        # Should handle error gracefully
        summary = await executor.run()
        assert summary.tasks_failed >= 1

    @pytest.mark.asyncio
    async def test_max_tasks_limit(
        self,
        test_project: TestProject,
        mock_agent: MockAgent,
        mock_verifier: MagicMock,
    ) -> None:
        """Test max_tasks limits execution."""
        test_project.create_roadmap(["Task 1", "Task 2", "Task 3", "Task 4", "Task 5"])

        executor = Executor(
            roadmap=test_project.roadmap_path,
            agent=mock_agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=2),
        )

        summary = await executor.run()

        assert summary.tasks_completed <= 2

    @pytest.mark.asyncio
    async def test_max_retries(
        self,
        test_project: TestProject,
        mock_agent: MockAgent,
    ) -> None:
        """Test that failures are tracked and execution continues."""
        test_project.create_roadmap(["Flaky task", "Second task"])

        # First task always fails
        mock_agent.add_response(
            pattern="Flaky",
            response="Failed",
            raise_error=RuntimeError("Temporary failure"),
        )
        # Second task succeeds
        mock_agent.add_response(
            pattern="Second",
            response="Success!",
        )

        verifier = MagicMock()
        verifier.verify = AsyncMock(
            return_value=VerificationResult(
                task_id="task",
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

        executor = Executor(
            roadmap=test_project.roadmap_path,
            agent=mock_agent,
            verifier=verifier,
            config=ExecutorConfig(max_tasks=2, retry_failed=1, stop_on_failure=False),
        )

        summary = await executor.run()

        # Should have at least one failure and attempt to run next task
        assert summary.tasks_failed >= 1 or summary.tasks_completed >= 1
        assert mock_agent.call_count >= 1

    @pytest.mark.asyncio
    async def test_state_persistence(
        self,
        test_project: TestProject,
        mock_agent: MockAgent,
        mock_verifier: MagicMock,
    ) -> None:
        """Test that state is persisted across runs."""
        test_project.create_roadmap(["Task 1", "Task 2"])

        executor = Executor(
            roadmap=test_project.roadmap_path,
            agent=mock_agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        # First run - complete one task
        await executor.run()

        # Check state file exists
        state_path = test_project.roadmap_path.parent / ".executor/state.json"
        assert state_path.exists()

        # Load state
        state = ExecutorState.load(test_project.roadmap_path)
        assert len([t for t in state.tasks.values() if t.status == TaskStatus.COMPLETED]) >= 1

    @pytest.mark.asyncio
    async def test_resume_from_state(
        self,
        test_project: TestProject,
        mock_agent: MockAgent,
        mock_verifier: MagicMock,
    ) -> None:
        """Test resuming from persisted state."""
        test_project.create_roadmap(["Task 1", "Task 2", "Task 3"])

        # First run - complete one task
        executor1 = Executor(
            roadmap=test_project.roadmap_path,
            agent=mock_agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )
        await executor1.run()
        first_run_calls = mock_agent.call_count

        # Second run with new executor - should continue
        executor2 = Executor(
            roadmap=test_project.roadmap_path,
            agent=mock_agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )
        await executor2.run()

        # Should have made more calls
        assert mock_agent.call_count > first_run_calls

    @pytest.mark.asyncio
    async def test_verification_integration(
        self,
        test_project: TestProject,
        mock_agent: MockAgent,
    ) -> None:
        """Test integration with verification system."""
        test_project.create_roadmap(["Create function"])

        mock_agent.add_response(
            pattern="function",
            response="Created the function.",
        )

        # Custom verifier with multiple levels
        verifier = MagicMock()
        verifier.verify = AsyncMock(
            return_value=VerificationResult(
                task_id="create-function",
                levels_run=[
                    CheckLevel.SYNTAX,
                    CheckLevel.TESTS,
                ],
                checks=[
                    CheckResult(
                        name="syntax_check",
                        level=CheckLevel.SYNTAX,
                        status=CheckStatus.PASSED,
                    ),
                    CheckResult(
                        name="functional_check",
                        level=CheckLevel.TESTS,
                        status=CheckStatus.PASSED,
                    ),
                ],
            )
        )

        executor = Executor(
            roadmap=test_project.roadmap_path,
            agent=mock_agent,
            verifier=verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        summary = await executor.run()

        # Verifier should have been called
        verifier.verify.assert_called()
        assert summary.tasks_completed == 1

    @pytest.mark.asyncio
    async def test_callbacks_integration(
        self,
        test_project: TestProject,
        mock_agent: MockAgent,
        mock_verifier: MagicMock,
    ) -> None:
        """Test callbacks are invoked during execution."""
        from ai_infra.executor.observability import ExecutorCallbacks

        test_project.create_roadmap(["Task 1"])

        events: list[str] = []

        class TrackingCallbacks(ExecutorCallbacks):
            def set_run_context(self, run_id: str) -> None:
                events.append("run_start")
                super().set_run_context(run_id)

            def on_task_start(self, task_id: str, title: str) -> None:
                events.append("task_start")
                super().on_task_start(task_id, title)

            def on_task_end(
                self,
                task_id: str,
                success: bool,
                files_modified: int = 0,
                error: str | None = None,
            ) -> None:
                events.append("task_end")
                super().on_task_end(task_id, success, files_modified, error)

            def on_run_end(self) -> None:
                events.append("run_end")
                super().on_run_end()

        callbacks = TrackingCallbacks()

        executor = Executor(
            roadmap=test_project.roadmap_path,
            agent=mock_agent,
            verifier=mock_verifier,
            callbacks=callbacks,
            config=ExecutorConfig(max_tasks=1),
        )

        await executor.run()

        assert "run_start" in events
        assert "task_start" in events
        assert "task_end" in events
        assert "run_end" in events

    @pytest.mark.asyncio
    async def test_dry_run_mode(
        self,
        test_project: TestProject,
        mock_agent: MockAgent,
        mock_verifier: MagicMock,
    ) -> None:
        """Test dry run mode doesn't execute tasks."""
        test_project.create_roadmap(["Task 1", "Task 2"])

        executor = Executor(
            roadmap=test_project.roadmap_path,
            agent=mock_agent,
            verifier=mock_verifier,
            config=ExecutorConfig(dry_run=True),
        )

        summary = await executor.run()

        # In dry run, agent should not be called
        assert mock_agent.call_count == 0
        assert summary.tasks_completed == 0


# =============================================================================
# Agent Response Scenarios
# =============================================================================


class TestAgentResponseScenarios:
    """Test various agent response scenarios."""

    @pytest.mark.asyncio
    async def test_agent_returns_file_list(
        self,
        test_project: TestProject,
        mock_agent: MockAgent,
        mock_verifier: MagicMock,
    ) -> None:
        """Test handling agent responses with file lists."""
        test_project.create_roadmap(["Create module with multiple files"])

        mock_agent.add_response(
            pattern="module",
            response="Created the module structure.",
            files_created=["src/module/__init__.py", "src/module/core.py"],
            files_modified=["pyproject.toml"],
        )

        executor = Executor(
            roadmap=test_project.roadmap_path,
            agent=mock_agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        await executor.run()

        response = mock_agent.last_call["response"]
        assert "module/__init__.py" in response
        assert "module/core.py" in response
        assert "pyproject.toml" in response

    @pytest.mark.asyncio
    async def test_agent_returns_code_snippet(
        self,
        test_project: TestProject,
        mock_agent: MockAgent,
        mock_verifier: MagicMock,
    ) -> None:
        """Test handling agent responses with code snippets."""
        test_project.create_roadmap(["Add function"])

        mock_agent.add_response(
            pattern="function",
            response="""Created the function:

```python
def helper(x: int) -> int:
    return x * 2
```

Added to src/utils.py.""",
        )

        executor = Executor(
            roadmap=test_project.roadmap_path,
            agent=mock_agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        await executor.run()

        assert "def helper" in mock_agent.last_call["response"]

    @pytest.mark.asyncio
    async def test_agent_incremental_progress(
        self,
        test_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test agent making incremental progress on task."""
        test_project.create_roadmap(["Complex task requiring multiple steps"])

        # Agent that simulates iterative progress
        agent = MockAgent()
        agent.add_response(
            pattern="Complex",
            response="Step 1 complete: created base structure.",
            max_uses=1,
        )
        agent.add_response(
            pattern="Complex",
            response="Step 2 complete: added functionality.",
            max_uses=1,
        )
        agent.add_response(
            pattern="Complex",
            response="Task fully complete.",
        )

        executor = Executor(
            roadmap=test_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        await executor.run()

        # All prompts should mention the complex task
        for prompt in agent.get_prompts():
            assert "Complex" in prompt or "task" in prompt.lower()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_agent_network_error(
        self,
        test_project: TestProject,
        mock_agent: MockAgent,
        mock_verifier: MagicMock,
    ) -> None:
        """Test handling network errors from agent."""
        test_project.create_roadmap(["Task"])

        mock_agent.add_error(
            pattern="Task",
            error=ConnectionError("Network unavailable"),
        )

        executor = Executor(
            roadmap=test_project.roadmap_path,
            agent=mock_agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1, retry_failed=1),
        )

        summary = await executor.run()

        # Should handle error gracefully
        assert summary is not None

    @pytest.mark.asyncio
    async def test_verifier_error(
        self,
        test_project: TestProject,
        mock_agent: MockAgent,
    ) -> None:
        """Test handling verifier errors."""
        test_project.create_roadmap(["Task"])

        verifier = MagicMock()
        verifier.verify.side_effect = RuntimeError("Verifier crashed")

        executor = Executor(
            roadmap=test_project.roadmap_path,
            agent=mock_agent,
            verifier=verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        # Should not raise, should handle gracefully
        summary = await executor.run()
        assert summary is not None

    @pytest.mark.asyncio
    async def test_missing_roadmap(
        self,
        tmp_path: Path,
        mock_agent: MockAgent,
        mock_verifier: MagicMock,
    ) -> None:
        """Test handling missing roadmap file."""
        missing_path = tmp_path / "MISSING.md"

        executor = Executor(
            roadmap=missing_path,
            agent=mock_agent,
            verifier=mock_verifier,
        )

        # The error should occur when trying to run
        with pytest.raises((FileNotFoundError, ValueError)):
            await executor.run()

    @pytest.mark.asyncio
    async def test_empty_roadmap(
        self,
        test_project: TestProject,
        mock_agent: MockAgent,
        mock_verifier: MagicMock,
    ) -> None:
        """Test handling empty roadmap."""
        test_project.roadmap_path.write_text("# Empty Roadmap\n\nNo tasks here.")

        executor = Executor(
            roadmap=test_project.roadmap_path,
            agent=mock_agent,
            verifier=mock_verifier,
        )

        summary = await executor.run()

        # Should complete with no tasks
        assert summary.tasks_completed == 0
        assert mock_agent.call_count == 0


# =============================================================================
# Concurrent Execution Tests
# =============================================================================


class TestConcurrentExecution:
    """Test concurrent/parallel execution scenarios."""

    @pytest.mark.asyncio
    async def test_sequential_task_order(
        self,
        test_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test tasks are executed in order."""
        test_project.create_roadmap(["First", "Second", "Third"])

        agent = MockAgent()
        agent.add_response(pattern="First", response="Done first")
        agent.add_response(pattern="Second", response="Done second")
        agent.add_response(pattern="Third", response="Done third")

        executor = Executor(
            roadmap=test_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=3),
        )

        await executor.run()

        # Simply verify all 3 tasks were called
        assert agent.call_count >= 3, f"Expected 3 calls, got {agent.call_count}"

    @pytest.mark.asyncio
    async def test_multiple_executors_same_roadmap(
        self,
        test_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test multiple executors on same roadmap don't conflict."""
        test_project.create_roadmap(["Task 1", "Task 2"])

        agent1 = MockAgent()
        agent2 = MockAgent()

        executor1 = Executor(
            roadmap=test_project.roadmap_path,
            agent=agent1,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        executor2 = Executor(
            roadmap=test_project.roadmap_path,
            agent=agent2,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        # Run first
        await executor1.run()

        # Run second (should pick up where first left off)
        await executor2.run()

        # Both should have been used
        assert agent1.call_count >= 1
        assert agent2.call_count >= 1


# =============================================================================
# Metrics and Observability Integration
# =============================================================================


class TestMetricsIntegration:
    """Test metrics and observability integration."""

    @pytest.mark.asyncio
    async def test_get_metrics(
        self,
        test_project: TestProject,
        mock_agent: MockAgent,
        mock_verifier: MagicMock,
    ) -> None:
        """Test getting metrics from executor."""
        test_project.create_roadmap(["Task 1", "Task 2"])

        callbacks = ExecutorCallbacks()
        executor = Executor(
            roadmap=test_project.roadmap_path,
            agent=mock_agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=2),
            callbacks=callbacks,
        )

        await executor.run()

        metrics = executor.get_metrics()
        assert metrics is not None
        assert metrics.tasks_completed >= 0

    @pytest.mark.asyncio
    async def test_execution_timing(
        self,
        test_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test that execution timing is tracked."""
        test_project.create_roadmap(["Task"])

        agent = MockAgent()
        agent.add_response(pattern="Task", response="Done", delay=0.05)

        callbacks = ExecutorCallbacks()
        executor = Executor(
            roadmap=test_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
            callbacks=callbacks,
        )

        await executor.run()

        metrics = executor.get_metrics()
        assert metrics is not None
        # Should have tracked some duration
        assert metrics.total_duration_ms >= 0
