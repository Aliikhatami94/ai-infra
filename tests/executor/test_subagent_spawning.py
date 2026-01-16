"""Tests for Phase 3.3: Subagent Spawning.

Tests for the subagent registry, base classes, and specialized agents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from ai_infra.executor.agents.base import (
    ExecutionContext,
    SubAgentResult,
)
from ai_infra.executor.agents.coder import CoderAgent
from ai_infra.executor.agents.registry import SubAgentRegistry, SubAgentType
from ai_infra.executor.agents.reviewer import ReviewerAgent
from ai_infra.executor.agents.spawner import (
    spawn_for_task,
)
from ai_infra.executor.agents.tester import TesterAgent
from ai_infra.executor.todolist import TodoItem

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_context(tmp_path: Path) -> ExecutionContext:
    """Create a sample execution context."""
    return ExecutionContext(
        workspace=tmp_path,
        files_modified=["src/app.py"],
        project_type="python",
        summary="Test project",
    )


@pytest.fixture
def coding_task() -> TodoItem:
    """Create a coding task (no specific keywords - defaults to coder)."""
    return TodoItem(
        id=1,
        title="Add new feature to the app",
        description="Create a new endpoint for the API",
    )


@pytest.fixture
def review_task() -> TodoItem:
    """Create a review task (matches 'review' keyword)."""
    return TodoItem(
        id=2,
        title="Review authentication changes",
        description="Code review for PR #123",
    )


@pytest.fixture
def test_task() -> TodoItem:
    """Create a test task (matches 'run tests' keyword)."""
    return TodoItem(
        id=3,
        title="Run tests for auth module",
        description="Execute pytest for auth module",
    )


# =============================================================================
# SubAgentType Tests
# =============================================================================


class TestSubAgentType:
    """Tests for SubAgentType enum."""

    def test_agent_types_exist(self) -> None:
        """Verify all expected agent types are defined."""
        assert SubAgentType.CODER.value == "coder"
        assert SubAgentType.REVIEWER.value == "reviewer"
        assert SubAgentType.TESTER.value == "tester"
        assert SubAgentType.DEBUGGER.value == "debugger"
        assert SubAgentType.RESEARCHER.value == "researcher"

    def test_all_types_have_values(self) -> None:
        """All types should have string values."""
        for agent_type in SubAgentType:
            assert isinstance(agent_type.value, str)
            assert len(agent_type.value) > 0


# =============================================================================
# SubAgentRegistry Tests
# =============================================================================


class TestSubAgentRegistry:
    """Tests for SubAgentRegistry."""

    def test_coder_registered(self) -> None:
        """CoderAgent should be registered."""
        assert SubAgentType.CODER in SubAgentRegistry._agents
        agent = SubAgentRegistry.get(SubAgentType.CODER)
        assert isinstance(agent, CoderAgent)

    def test_reviewer_registered(self) -> None:
        """ReviewerAgent should be registered."""
        assert SubAgentType.REVIEWER in SubAgentRegistry._agents
        agent = SubAgentRegistry.get(SubAgentType.REVIEWER)
        assert isinstance(agent, ReviewerAgent)

    def test_tester_registered(self) -> None:
        """TesterAgent should be registered."""
        assert SubAgentType.TESTER in SubAgentRegistry._agents
        agent = SubAgentRegistry.get(SubAgentType.TESTER)
        assert isinstance(agent, TesterAgent)

    def test_get_caches_instance(self) -> None:
        """Getting same type should return cached instance."""
        agent1 = SubAgentRegistry.get(SubAgentType.CODER)
        agent2 = SubAgentRegistry.get(SubAgentType.CODER)
        assert agent1 is agent2

    def test_get_different_types(self) -> None:
        """Different types should return different instances."""
        coder = SubAgentRegistry.get(SubAgentType.CODER)
        tester = SubAgentRegistry.get(SubAgentType.TESTER)
        assert coder is not tester
        assert type(coder) is not type(tester)

    def test_get_unknown_agent_raises(self) -> None:
        """Should raise for unknown/invalid agent type."""
        # Passing a string instead of SubAgentType raises AttributeError
        # because the method expects an enum with .value attribute
        with pytest.raises((KeyError, ValueError, AttributeError)):
            SubAgentRegistry.get("unknown_agent_type")  # type: ignore[arg-type]


# =============================================================================
# ExecutionContext Tests
# =============================================================================


class TestExecutionContext:
    """Tests for ExecutionContext."""

    def test_create_basic(self, tmp_path: Path) -> None:
        """Can create with minimal arguments."""
        ctx = ExecutionContext(workspace=tmp_path)
        assert ctx.workspace == tmp_path
        assert ctx.files_modified == []
        assert ctx.project_type == "unknown"

    def test_create_full(self, tmp_path: Path) -> None:
        """Can create with all arguments."""
        ctx = ExecutionContext(
            workspace=tmp_path,
            files_modified=["a.py", "b.py"],
            relevant_files=["c.py"],
            project_type="python",
            summary="Test",
            run_memory={"key": "value"},
            dependencies=["pytest"],
        )
        assert ctx.workspace == tmp_path
        assert ctx.files_modified == ["a.py", "b.py"]
        assert ctx.relevant_files == ["c.py"]
        assert ctx.project_type == "python"
        assert ctx.summary == "Test"
        assert ctx.run_memory == {"key": "value"}
        assert ctx.dependencies == ["pytest"]

    def test_to_dict(self, sample_context: ExecutionContext) -> None:
        """Can convert to dictionary."""
        d = sample_context.to_dict()
        assert "workspace" in d
        assert "files_modified" in d
        assert "project_type" in d

    def test_from_state(self, tmp_path: Path) -> None:
        """Can create from executor state."""
        # Create a sample ROADMAP.md path
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.touch()

        state: dict[str, Any] = {
            "roadmap_path": str(roadmap),
            "files_modified": ["test.py"],
            "context": "Test context",
            "run_memory": {},
        }
        ctx = ExecutionContext.from_state(state)
        assert ctx.workspace == tmp_path
        assert ctx.files_modified == ["test.py"]


# =============================================================================
# SubAgentResult Tests
# =============================================================================


class TestSubAgentResult:
    """Tests for SubAgentResult."""

    def test_create_success(self) -> None:
        """Can create success result."""
        result = SubAgentResult(success=True, output="Done")
        assert result.success is True
        assert result.output == "Done"
        assert result.error is None

    def test_create_failure(self) -> None:
        """Can create failure result."""
        result = SubAgentResult(success=False, error="Something broke")
        assert result.success is False
        assert result.error == "Something broke"

    def test_files_modified(self) -> None:
        """Can track modified files."""
        result = SubAgentResult(
            success=True,
            files_modified=["a.py"],
            files_created=["b.py"],
        )
        assert result.files_modified == ["a.py"]
        assert result.files_created == ["b.py"]

    def test_review_fields(self) -> None:
        """Reviewer fields work correctly."""
        result = SubAgentResult(
            success=True,
            verdict="APPROVE",
            review_comments=["LGTM", "Nice work"],
        )
        assert result.verdict == "APPROVE"
        assert result.review_comments == ["LGTM", "Nice work"]

    def test_test_fields(self) -> None:
        """Tester fields work correctly."""
        result = SubAgentResult(
            success=True,
            tests_run=10,
            tests_passed=8,
            test_output="2 failed",
        )
        assert result.tests_run == 10
        assert result.tests_passed == 8
        assert result.tests_failed == 2

    def test_to_dict(self) -> None:
        """Can convert to dictionary."""
        result = SubAgentResult(
            success=True,
            output="Done",
            tests_run=5,
            tests_passed=5,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["output"] == "Done"
        assert d["tests_run"] == 5


# =============================================================================
# SubAgent Base Class Tests
# =============================================================================


class TestSubAgent:
    """Tests for SubAgent base class."""

    def test_coder_attributes(self) -> None:
        """CoderAgent has correct attributes."""
        agent = CoderAgent()
        assert agent.name == "Coder"
        assert "code" in agent.description.lower()
        assert agent.model == "claude-sonnet-4-20250514"

    def test_reviewer_attributes(self) -> None:
        """ReviewerAgent has correct attributes."""
        agent = ReviewerAgent()
        assert agent.name == "Reviewer"
        assert "review" in agent.description.lower()

    def test_tester_attributes(self) -> None:
        """TesterAgent has correct attributes."""
        agent = TesterAgent()
        assert agent.name == "Tester"
        assert "test" in agent.description.lower()

    def test_model_override(self) -> None:
        """Can override model at init."""
        agent = CoderAgent(model="different-model")
        assert agent._model == "different-model"

    def test_repr(self) -> None:
        """Repr shows useful info."""
        agent = CoderAgent()
        r = repr(agent)
        assert "CoderAgent" in r
        assert "Coder" in r


# =============================================================================
# Spawner Tests
# =============================================================================


class TestSpawner:
    """Tests for spawn_for_task function."""

    @pytest.mark.asyncio
    async def test_spawn_for_task_mock(
        self,
        coding_task: TodoItem,
        sample_context: ExecutionContext,
    ) -> None:
        """spawn_for_task calls the right agent."""
        # Mock the agent execution
        mock_result = SubAgentResult(
            success=True,
            output="Code written",
            files_created=["new_file.py"],
        )

        with patch.object(CoderAgent, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await spawn_for_task(coding_task, sample_context)

            assert result.success is True
            assert result.files_created == ["new_file.py"]
            mock_execute.assert_called_once_with(coding_task, sample_context)


# =============================================================================
# Integration Tests (require real agent - skip in CI)
# =============================================================================


@pytest.mark.skip(reason="Integration test - requires LLM API")
class TestSubagentIntegration:
    """Integration tests that run actual agents."""

    @pytest.mark.asyncio
    async def test_coder_creates_file(
        self,
        coding_task: TodoItem,
        sample_context: ExecutionContext,
    ) -> None:
        """CoderAgent can create files."""
        agent = CoderAgent()
        result = await agent.execute(coding_task, sample_context)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_tester_runs_tests(
        self,
        test_task: TodoItem,
        sample_context: ExecutionContext,
    ) -> None:
        """TesterAgent can run tests."""
        agent = TesterAgent()
        result = await agent.execute(test_task, sample_context)
        assert result.tests_run >= 0
