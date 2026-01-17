"""Scenario tests for subagent delegation (Phase 6.3.5).

Tests scenarios for subagent delegation, including:
- Coder delegating to tester
- Debugger invoked on failure
- Agent selection based on task type
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ai_infra.executor.agents.base import ExecutionContext, SubAgentResult
from ai_infra.executor.agents.coder import CoderAgent
from ai_infra.executor.agents.debugger import DebuggerAgent
from ai_infra.executor.agents.registry import SubAgentRegistry, SubAgentType
from ai_infra.executor.agents.reviewer import ReviewerAgent
from ai_infra.executor.agents.spawner import spawn_for_task
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
        files_modified=[],
        project_type="python",
        summary="Test Python project",
    )


@pytest.fixture
def coding_task() -> TodoItem:
    """Create a coding task."""
    return TodoItem(
        id=1,
        title="Implement user authentication",
        description="Create login/logout endpoints with JWT",
    )


@pytest.fixture
def test_task() -> TodoItem:
    """Create a testing task (run tests)."""
    return TodoItem(
        id=2,
        title="Run tests for auth module",
        description="Execute pytest tests for login/logout",
    )


@pytest.fixture
def review_task() -> TodoItem:
    """Create a review task."""
    return TodoItem(
        id=3,
        title="Review authentication changes",
        description="Code review for PR #42",
    )


@pytest.fixture
def debug_task() -> TodoItem:
    """Create a debugging task."""
    return TodoItem(
        id=4,
        title="Debug login failure issue",
        description="Investigate why JWT validation fails",
    )


@pytest.fixture
def research_task() -> TodoItem:
    """Create a research task."""
    return TodoItem(
        id=5,
        title="Research OAuth2 providers",
        description="Compare Auth0, Okta, and Cognito",
    )


# =============================================================================
# Agent Delegation Tests
# =============================================================================


class TestAgentExecution:
    """Tests for agent execution via spawner."""

    @pytest.mark.asyncio
    async def test_spawn_with_explicit_tester_type(
        self, test_task: TodoItem, sample_context: ExecutionContext
    ) -> None:
        """Spawner should use TesterAgent when agent_type=TESTER is specified."""
        # Mock execution
        mock_result = SubAgentResult(
            success=True,
            output="Tests run",
            tests_run=5,
            tests_passed=5,
        )

        with patch.object(TesterAgent, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            # Explicitly specify TESTER agent type (orchestrator would do this)
            result = await spawn_for_task(test_task, sample_context, agent_type=SubAgentType.TESTER)

            # Tester should have been called
            mock_execute.assert_called_once()
            assert result.success

    @pytest.mark.asyncio
    async def test_spawner_uses_coder_by_default(
        self, coding_task: TodoItem, sample_context: ExecutionContext
    ) -> None:
        """spawn_for_task should default to CODER when no agent_type specified."""
        mock_result = SubAgentResult(
            success=True,
            output="Code written",
            files_modified=["src/auth.py"],
        )

        with patch.object(CoderAgent, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await spawn_for_task(coding_task, sample_context)

            mock_execute.assert_called_once()
            assert result.files_modified == ["src/auth.py"]


class TestDebuggerInvocation:
    """Tests for debugger invocation on failure."""

    def test_get_debugger_for_debug_task(self) -> None:
        """Should get DebuggerAgent from registry."""
        agent = SubAgentRegistry.get(SubAgentType.DEBUGGER)
        assert isinstance(agent, DebuggerAgent)
        assert agent.name == "Debugger"

    def test_debugger_has_correct_attributes(self) -> None:
        """DebuggerAgent should have correct attributes."""
        agent = DebuggerAgent()
        assert agent.name == "Debugger"
        assert "debug" in agent.description.lower() or "fix" in agent.description.lower()

    @pytest.mark.asyncio
    async def test_debugger_invoked_with_explicit_type(
        self, sample_context: ExecutionContext
    ) -> None:
        """Debugger should be invoked when explicitly specified."""
        debug_task = TodoItem(
            id=1,
            title="Debug authentication failure",
            description="Fix JWT validation error",
        )

        mock_result = SubAgentResult(
            success=True,
            output="Bug fixed",
            files_modified=["src/auth.py"],
        )

        with patch.object(DebuggerAgent, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            # Explicitly specify DEBUGGER (orchestrator would do this)
            result = await spawn_for_task(
                debug_task, sample_context, agent_type=SubAgentType.DEBUGGER
            )

            mock_execute.assert_called_once()
            assert result.success


# =============================================================================
# Agent Chaining Tests
# =============================================================================


class TestAgentChaining:
    """Tests for chaining multiple agents."""

    @pytest.mark.asyncio
    async def test_coder_then_testwriter_workflow(self, sample_context: ExecutionContext) -> None:
        """Simulate coder -> testwriter workflow."""
        from ai_infra.executor.agents.testwriter import TestWriterAgent

        # Step 1: Coder implements feature
        code_task = TodoItem(id=1, title="Implement feature", description="")
        code_result = SubAgentResult(
            success=True,
            files_modified=["src/feature.py"],
        )

        with patch.object(CoderAgent, "execute", new_callable=AsyncMock) as mock_coder:
            mock_coder.return_value = code_result
            result1 = await spawn_for_task(code_task, sample_context, agent_type=SubAgentType.CODER)
            assert result1.success

        # Step 2: TestWriter creates tests
        test_task = TodoItem(id=2, title="Write tests for feature", description="")
        test_result = SubAgentResult(
            success=True,
            files_created=["tests/test_feature.py"],
        )

        with patch.object(TestWriterAgent, "execute", new_callable=AsyncMock) as mock_testwriter:
            mock_testwriter.return_value = test_result
            result2 = await spawn_for_task(
                test_task, sample_context, agent_type=SubAgentType.TESTWRITER
            )
            assert result2.success
            assert "tests/test_feature.py" in result2.files_created

    @pytest.mark.asyncio
    async def test_coder_then_reviewer_workflow(self, sample_context: ExecutionContext) -> None:
        """Simulate coder -> reviewer workflow."""
        # Step 1: Coder implements
        code_task = TodoItem(id=1, title="Implement auth", description="")
        code_result = SubAgentResult(
            success=True,
            files_modified=["src/auth.py"],
        )

        with patch.object(CoderAgent, "execute", new_callable=AsyncMock) as mock_coder:
            mock_coder.return_value = code_result
            await spawn_for_task(code_task, sample_context, agent_type=SubAgentType.CODER)

        # Step 2: Reviewer reviews
        review_task = TodoItem(id=2, title="Review auth changes", description="")
        review_result = SubAgentResult(
            success=True,
            verdict="APPROVE",
            review_comments=["LGTM", "Good error handling"],
        )

        with patch.object(ReviewerAgent, "execute", new_callable=AsyncMock) as mock_reviewer:
            mock_reviewer.return_value = review_result
            result = await spawn_for_task(
                review_task, sample_context, agent_type=SubAgentType.REVIEWER
            )
            assert result.verdict == "APPROVE"


# =============================================================================
# Agent Result Tests
# =============================================================================


class TestAgentResults:
    """Tests for agent result handling."""

    def test_coder_result_fields(self) -> None:
        """CoderAgent results should have file fields."""
        result = SubAgentResult(
            success=True,
            output="Code written",
            files_modified=["src/a.py"],
            files_created=["src/b.py"],
        )

        assert result.success
        assert len(result.files_modified) == 1
        assert len(result.files_created) == 1

    def test_tester_result_fields(self) -> None:
        """TesterAgent results should have test fields."""
        result = SubAgentResult(
            success=True,
            tests_run=10,
            tests_passed=8,
            test_output="2 failed",
        )

        assert result.tests_run == 10
        assert result.tests_passed == 8
        assert result.tests_failed == 2

    def test_reviewer_result_fields(self) -> None:
        """ReviewerAgent results should have review fields."""
        result = SubAgentResult(
            success=True,
            verdict="REQUEST_CHANGES",
            review_comments=["Fix typo", "Add docstring"],
        )

        assert result.verdict == "REQUEST_CHANGES"
        assert len(result.review_comments) == 2

    def test_result_to_dict(self) -> None:
        """Results should serialize to dict."""
        result = SubAgentResult(
            success=True,
            output="Done",
            files_modified=["a.py"],
        )

        data = result.to_dict()
        assert data["success"] is True
        assert "files_modified" in data
