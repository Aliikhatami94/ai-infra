"""Tests for planning node.

Phase 2.4.2: Unit tests for plan_task_node and ExecutionPlan.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_infra.executor.nodes.plan import (
    PLANNING_PROMPT,
    ExecutionPlan,
    _parse_planning_response,
    plan_task_node,
)
from ai_infra.executor.state import ExecutorGraphState
from ai_infra.executor.todolist import TodoItem, TodoStatus

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_todo_item() -> TodoItem:
    """Create a mock TodoItem for testing."""
    return TodoItem(
        id=1,
        title="Add authentication module",
        description="Implement JWT-based authentication with refresh tokens",
        status=TodoStatus.NOT_STARTED,
        file_hints=["src/auth.py", "tests/test_auth.py"],
    )


@pytest.fixture
def base_state(mock_todo_item: TodoItem) -> ExecutorGraphState:
    """Create a base ExecutorGraphState for testing."""
    return ExecutorGraphState(
        roadmap_path="ROADMAP.md",
        todos=[mock_todo_item],
        current_task=mock_todo_item,
        context="",
        prompt="Add authentication module",
        agent_result=None,
        files_modified=[],
        verified=False,
        last_checkpoint_sha=None,
        error=None,
        retry_count=0,
        replan_count=0,
        completed_count=0,
        max_tasks=None,
        should_continue=True,
        interrupt_requested=False,
        run_memory={},
        adaptive_mode="auto_fix",
        failure_classification=None,
        failure_reason=None,
        suggested_fix=None,
        execution_plan=None,
        enable_planning=True,
        task_plan=None,
    )


@pytest.fixture
def sample_planning_response() -> str:
    """Sample LLM response in expected format."""
    return """FILES:
- src/auth/jwt_handler.py
- src/auth/refresh_tokens.py
- tests/unit/auth/test_jwt_handler.py
- tests/unit/auth/test_refresh_tokens.py

DEPENDENCIES:
- python-jose
- passlib

RISKS:
- Token expiration edge cases
- Concurrent refresh token usage
- Clock skew between services

APPROACH:
Implement a JWTHandler class that manages token creation and validation.
Use python-jose for JWT operations. Store refresh tokens in Redis with TTL.

COMPLEXITY: MEDIUM
"""


# =============================================================================
# ExecutionPlan Tests
# =============================================================================


class TestExecutionPlan:
    """Tests for ExecutionPlan dataclass."""

    def test_default_values(self):
        """Test default values for ExecutionPlan."""
        plan = ExecutionPlan()
        assert plan.likely_files == []
        assert plan.dependencies == []
        assert plan.risks == []
        assert plan.approach == ""
        assert plan.complexity == "medium"

    def test_to_dict(self):
        """Test converting ExecutionPlan to dict."""
        plan = ExecutionPlan(
            likely_files=["src/auth.py"],
            dependencies=["python-jose"],
            risks=["Token expiration"],
            approach="Use JWT tokens",
            complexity="high",
        )
        result = plan.to_dict()

        assert result == {
            "likely_files": ["src/auth.py"],
            "dependencies": ["python-jose"],
            "risks": ["Token expiration"],
            "approach": "Use JWT tokens",
            "complexity": "high",
        }

    def test_from_dict(self):
        """Test creating ExecutionPlan from dict."""
        data = {
            "likely_files": ["src/auth.py", "tests/test_auth.py"],
            "dependencies": ["python-jose", "passlib"],
            "risks": ["Token expiration", "Clock skew"],
            "approach": "Implement JWT handler",
            "complexity": "medium",
        }
        plan = ExecutionPlan.from_dict(data)

        assert plan.likely_files == ["src/auth.py", "tests/test_auth.py"]
        assert plan.dependencies == ["python-jose", "passlib"]
        assert plan.risks == ["Token expiration", "Clock skew"]
        assert plan.approach == "Implement JWT handler"
        assert plan.complexity == "medium"

    def test_from_dict_with_missing_keys(self):
        """Test from_dict handles missing keys gracefully."""
        data: dict[str, Any] = {}
        plan = ExecutionPlan.from_dict(data)

        assert plan.likely_files == []
        assert plan.dependencies == []
        assert plan.risks == []
        assert plan.approach == ""
        assert plan.complexity == "medium"

    def test_roundtrip(self):
        """Test to_dict and from_dict are inverses."""
        original = ExecutionPlan(
            likely_files=["a.py", "b.py"],
            dependencies=["dep1"],
            risks=["risk1", "risk2"],
            approach="Do the thing",
            complexity="low",
        )
        roundtrip = ExecutionPlan.from_dict(original.to_dict())

        assert roundtrip.likely_files == original.likely_files
        assert roundtrip.dependencies == original.dependencies
        assert roundtrip.risks == original.risks
        assert roundtrip.approach == original.approach
        assert roundtrip.complexity == original.complexity


# =============================================================================
# Response Parsing Tests
# =============================================================================


class TestParsesPlanningResponse:
    """Tests for _parse_planning_response function."""

    def test_parses_complete_response(self, sample_planning_response: str):
        """Test parsing a complete, well-formatted response."""
        plan = _parse_planning_response(sample_planning_response)

        assert plan.likely_files == [
            "src/auth/jwt_handler.py",
            "src/auth/refresh_tokens.py",
            "tests/unit/auth/test_jwt_handler.py",
            "tests/unit/auth/test_refresh_tokens.py",
        ]
        assert plan.dependencies == ["python-jose", "passlib"]
        assert len(plan.risks) == 3
        assert "Token expiration edge cases" in plan.risks
        assert "JWTHandler" in plan.approach
        assert plan.complexity == "medium"

    def test_parses_files_with_asterisks(self):
        """Test parsing files marked with asterisks."""
        response = """FILES:
* src/main.py
* tests/test_main.py

DEPENDENCIES:
* click

RISKS:
* None

APPROACH:
Simple update.

COMPLEXITY: LOW
"""
        plan = _parse_planning_response(response)

        assert plan.likely_files == ["src/main.py", "tests/test_main.py"]
        assert plan.dependencies == ["click"]
        assert plan.complexity == "low"

    def test_parses_uppercase_complexity(self):
        """Test that complexity is normalized to lowercase."""
        response = """FILES:
- src/app.py

DEPENDENCIES:
- None

RISKS:
- None

APPROACH:
Simple.

COMPLEXITY: HIGH
"""
        plan = _parse_planning_response(response)
        assert plan.complexity == "high"

    def test_handles_empty_sections(self):
        """Test parsing response with empty sections."""
        response = """FILES:

DEPENDENCIES:

RISKS:

APPROACH:
Just do it.

COMPLEXITY: LOW
"""
        plan = _parse_planning_response(response)

        assert plan.likely_files == []
        assert plan.dependencies == []
        assert plan.risks == []
        assert "do it" in plan.approach
        assert plan.complexity == "low"

    def test_handles_none_values(self):
        """Test that 'none' entries are filtered out."""
        response = """FILES:
- src/app.py

DEPENDENCIES:
- None
- none

RISKS:
- None

APPROACH:
Update the app.

COMPLEXITY: LOW
"""
        plan = _parse_planning_response(response)

        assert plan.dependencies == []
        assert plan.risks == []

    def test_handles_malformed_response(self):
        """Test parsing malformed response returns defaults."""
        response = "This is not a valid planning response at all."
        plan = _parse_planning_response(response)

        assert plan.likely_files == []
        assert plan.dependencies == []
        assert plan.risks == []
        assert plan.approach == ""
        assert plan.complexity == "medium"  # default

    def test_parses_multiline_approach(self):
        """Test that multi-line approach is captured."""
        response = """FILES:
- src/handler.py

DEPENDENCIES:
- aiohttp

RISKS:
- Rate limiting

APPROACH:
First, create the handler class.
Then, add async methods for API calls.
Finally, implement retry logic.

COMPLEXITY: MEDIUM
"""
        plan = _parse_planning_response(response)

        assert "First, create the handler class" in plan.approach
        assert "Finally, implement retry logic" in plan.approach


# =============================================================================
# plan_task_node Tests
# =============================================================================


class TestPlanTaskNode:
    """Tests for plan_task_node function."""

    @pytest.mark.asyncio
    async def test_generates_plan_with_agent(
        self,
        base_state: ExecutorGraphState,
        sample_planning_response: str,
    ):
        """Test that plan_task_node generates a plan when agent is provided."""
        mock_agent = MagicMock()
        mock_agent.arun = AsyncMock(return_value=sample_planning_response)

        result = await plan_task_node(base_state, planner_agent=mock_agent)

        assert result["task_plan"] is not None
        plan = result["task_plan"]
        assert len(plan["likely_files"]) == 4
        assert "python-jose" in plan["dependencies"]
        assert plan["complexity"] == "medium"

    @pytest.mark.asyncio
    async def test_skips_planning_without_agent(self, base_state: ExecutorGraphState):
        """Test that planning is skipped when no agent is provided."""
        result = await plan_task_node(base_state, planner_agent=None)

        assert result["task_plan"] is None

    @pytest.mark.asyncio
    async def test_skips_planning_without_current_task(
        self,
        base_state: ExecutorGraphState,
    ):
        """Test that planning is skipped when no current_task is set."""
        state = {**base_state, "current_task": None}
        mock_agent = MagicMock()
        mock_agent.arun = AsyncMock(return_value="response")

        result = await plan_task_node(state, planner_agent=mock_agent)

        assert result["task_plan"] is None
        mock_agent.arun.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_agent_error_gracefully(
        self,
        base_state: ExecutorGraphState,
    ):
        """Test that agent errors result in None plan, not exception."""
        mock_agent = MagicMock()
        mock_agent.arun = AsyncMock(side_effect=Exception("LLM error"))

        result = await plan_task_node(base_state, planner_agent=mock_agent)

        assert result["task_plan"] is None
        # Should not raise

    @pytest.mark.asyncio
    async def test_prompt_includes_task_details(
        self,
        base_state: ExecutorGraphState,
    ):
        """Test that the prompt includes task title, description, and file hints."""
        mock_agent = MagicMock()
        mock_agent.arun = AsyncMock(
            return_value="FILES:\n- a.py\n\nDEPENDENCIES:\n\nRISKS:\n\nAPPROACH:\nOk.\n\nCOMPLEXITY: LOW"
        )

        await plan_task_node(base_state, planner_agent=mock_agent)

        # Get the prompt that was passed to the agent
        call_args = mock_agent.arun.call_args[0][0]
        assert "Add authentication module" in call_args
        assert "JWT-based authentication" in call_args
        assert "src/auth.py" in call_args

    @pytest.mark.asyncio
    async def test_handles_task_without_file_hints(
        self,
        mock_todo_item: TodoItem,
    ):
        """Test planning works when task has no file hints."""
        item = TodoItem(
            id=2,
            title="Add logging",
            description="Add structured logging",
            status=TodoStatus.NOT_STARTED,
            file_hints=[],
        )
        state = ExecutorGraphState(
            roadmap_path="ROADMAP.md",
            todos=[item],
            current_task=item,
            context="",
            prompt="",
            agent_result=None,
            files_modified=[],
            verified=False,
            last_checkpoint_sha=None,
            error=None,
            retry_count=0,
            replan_count=0,
            completed_count=0,
            max_tasks=None,
            should_continue=True,
            interrupt_requested=False,
            run_memory={},
            adaptive_mode="auto_fix",
            failure_classification=None,
            failure_reason=None,
            suggested_fix=None,
            execution_plan=None,
            enable_planning=True,
            task_plan=None,
        )

        mock_agent = MagicMock()
        mock_agent.arun = AsyncMock(
            return_value="FILES:\n- x.py\n\nDEPENDENCIES:\n\nRISKS:\n\nAPPROACH:\nYes.\n\nCOMPLEXITY: LOW"
        )

        result = await plan_task_node(state, planner_agent=mock_agent)

        assert result["task_plan"] is not None
        # Check prompt says (none) for file hints
        call_args = mock_agent.arun.call_args[0][0]
        assert "(none)" in call_args


# =============================================================================
# PLANNING_PROMPT Tests
# =============================================================================


class TestPlanningPrompt:
    """Tests for PLANNING_PROMPT template."""

    def test_prompt_has_required_sections(self):
        """Test that prompt template includes required sections."""
        assert "FILES" in PLANNING_PROMPT
        assert "DEPENDENCIES" in PLANNING_PROMPT
        assert "RISKS" in PLANNING_PROMPT
        assert "APPROACH" in PLANNING_PROMPT
        assert "COMPLEXITY" in PLANNING_PROMPT

    def test_prompt_has_placeholders(self):
        """Test that prompt has placeholders for task details."""
        assert "{title}" in PLANNING_PROMPT
        assert "{description}" in PLANNING_PROMPT
        assert "{file_hints}" in PLANNING_PROMPT

    def test_prompt_format_works(self):
        """Test that prompt can be formatted without errors."""
        formatted = PLANNING_PROMPT.format(
            title="Test Task",
            description="Test description",
            file_hints="src/test.py",
        )
        assert "Test Task" in formatted
        assert "Test description" in formatted
        assert "src/test.py" in formatted


# =============================================================================
# Integration Tests - Context Building
# =============================================================================


class TestContextBuilderIntegration:
    """Tests for plan integration with context building."""

    def test_plan_section_helper_exists(self):
        """Test that _build_plan_section helper exists and works."""
        from ai_infra.executor.nodes.context import _build_plan_section

        plan_dict = {
            "likely_files": ["src/app.py"],
            "dependencies": ["click"],
            "risks": ["Breaking change"],
            "approach": "Update the handler",
            "complexity": "low",
        }

        section = _build_plan_section(plan_dict)

        assert "## Execution Plan" in section
        assert "src/app.py" in section
        assert "click" in section
        assert "Breaking change" in section
        assert "Update the handler" in section
        # Complexity is uppercased in the output
        assert "LOW" in section or "low" in section

    def test_plan_section_handles_none(self):
        """Test that _build_plan_section handles None gracefully."""
        from ai_infra.executor.nodes.context import _build_plan_section

        section = _build_plan_section(None)
        assert section == ""

    def test_plan_section_handles_empty_plan(self):
        """Test that _build_plan_section handles empty plan."""
        from ai_infra.executor.nodes.context import _build_plan_section

        # Empty dict returns empty string
        section = _build_plan_section({})
        assert section == ""

        # Plan with only complexity still produces output
        section_with_complexity = _build_plan_section({"complexity": "low"})
        assert "## Execution Plan" in section_with_complexity
        assert "LOW" in section_with_complexity


# =============================================================================
# Edge Routing Tests
# =============================================================================


class TestPlanningRouting:
    """Tests for planning-related routing."""

    def test_route_after_pick_goes_to_plan_when_enabled(
        self,
        base_state: ExecutorGraphState,
    ):
        """Test route_after_pick returns plan_task when planning enabled."""
        from ai_infra.executor.edges.routes import route_after_pick

        state = {**base_state, "enable_planning": True}
        result = route_after_pick(state)
        assert result == "plan_task"

    def test_route_after_pick_goes_to_context_when_disabled(
        self,
        base_state: ExecutorGraphState,
    ):
        """Test route_after_pick returns build_context when planning disabled."""
        from ai_infra.executor.edges.routes import route_after_pick

        state = {**base_state, "enable_planning": False}
        result = route_after_pick(state)
        assert result == "build_context"

    def test_route_after_pick_defaults_to_context(
        self,
        base_state: ExecutorGraphState,
    ):
        """Test route_after_pick defaults to build_context if key missing."""
        from ai_infra.executor.edges.routes import route_after_pick

        # Simulate missing key by using dict without enable_planning
        state = dict(base_state)
        state.pop("enable_planning", None)

        result = route_after_pick(state)
        assert result == "build_context"
