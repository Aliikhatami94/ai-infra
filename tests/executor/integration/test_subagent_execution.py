"""Integration tests for subagent execution (Phase 7.5.2).

Tests that the executor graph correctly routes tasks to specialized
subagents when use_subagents=True.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ai_infra.executor.agents.base import SubAgentResult
from ai_infra.executor.graph import ExecutorGraph
from ai_infra.executor.state import ExecutorGraphState
from ai_infra.executor.todolist import TodoItem

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def test_roadmap(tmp_path: Path) -> Path:
    """Create a roadmap with a test-related task."""
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text("""\
# Test Roadmap

## Phase 1: Testing

- [ ] Write tests for calculator
  Description: Create unit tests for the calculator module.
""")
    return roadmap


@pytest.fixture
def coding_roadmap(tmp_path: Path) -> Path:
    """Create a roadmap with a coding task."""
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text("""\
# Code Roadmap

## Phase 1: Implementation

- [ ] Implement user model
  Description: Create the user data model.
""")
    return roadmap


@pytest.fixture
def debug_roadmap(tmp_path: Path) -> Path:
    """Create a roadmap with a debugging task."""
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text("""\
# Debug Roadmap

## Phase 1: Bug Fixes

- [ ] Fix broken authentication
  Description: Debug the login failure issue.
""")
    return roadmap


@pytest.fixture
def mock_subagent_result() -> SubAgentResult:
    """Create a successful mock subagent result."""
    return SubAgentResult(
        success=True,
        output="Task completed successfully",
        files_modified=["src/test_calc.py"],
        metrics={"duration_ms": 100},
    )


# =============================================================================
# Phase 7.5.2: Integration Tests
# =============================================================================


class TestSubagentRouting:
    """Integration tests for subagent routing."""

    @pytest.mark.asyncio
    async def test_routes_to_tester_for_test_task(
        self, test_roadmap: Path, mock_subagent_result: SubAgentResult
    ) -> None:
        """Test that test-related tasks route to TesterAgent."""
        with patch(
            "ai_infra.executor.nodes.execute.spawn_for_task",
            new_callable=AsyncMock,
            return_value=mock_subagent_result,
        ) as mock_spawn:
            # Create graph with subagents enabled but dry_run to avoid full execution
            graph = ExecutorGraph(
                roadmap_path=str(test_roadmap),
                use_subagents=True,
                max_tasks=1,
            )

            # Get initial state
            initial_state: ExecutorGraphState = {
                "roadmap_path": str(test_roadmap),
                "todos": [],
                "current_task": TodoItem(
                    id=1,
                    title="Write tests for calculator",
                    description="Create unit tests",
                ),
                "prompt": "Write tests for the calculator module",
                "use_subagents": True,
            }

            # Import execute_task_node directly to test routing
            from ai_infra.executor.nodes.execute import execute_task_node

            result = await execute_task_node(
                initial_state,
                agent=None,  # No agent needed, using subagents
                use_subagents=True,
            )

            # Verify spawn_for_task was called
            mock_spawn.assert_called_once()
            # Verify the task was passed correctly
            call_args = mock_spawn.call_args
            task_arg = call_args[0][0]  # First positional arg
            assert "test" in task_arg.title.lower()

    @pytest.mark.asyncio
    async def test_subagent_used_recorded_in_state(
        self, mock_subagent_result: SubAgentResult
    ) -> None:
        """Test that subagent_used is recorded in result state."""
        with patch(
            "ai_infra.executor.nodes.execute.spawn_for_task",
            new_callable=AsyncMock,
            return_value=mock_subagent_result,
        ):
            from ai_infra.executor.nodes.execute import execute_task_node

            initial_state: ExecutorGraphState = {
                "roadmap_path": "/tmp/test.md",
                "todos": [],
                "current_task": TodoItem(
                    id=1,
                    title="Write unit tests",
                    description="Add pytest tests",
                ),
                "prompt": "Write tests",
                "use_subagents": True,
            }

            result = await execute_task_node(
                initial_state,
                agent=None,
                use_subagents=True,
            )

            # Verify subagent_used is set
            # Phase 16.5.11: "Write unit tests" now routes to testwriter via OrchestratorAgent
            assert "subagent_used" in result
            assert result["subagent_used"] == "testwriter"

    @pytest.mark.asyncio
    async def test_subagent_usage_metrics_tracked(
        self, mock_subagent_result: SubAgentResult
    ) -> None:
        """Test that subagent usage metrics are tracked."""
        with patch(
            "ai_infra.executor.nodes.execute.spawn_for_task",
            new_callable=AsyncMock,
            return_value=mock_subagent_result,
        ):
            from ai_infra.executor.nodes.execute import execute_task_node

            initial_state: ExecutorGraphState = {
                "roadmap_path": "/tmp/test.md",
                "todos": [],
                "current_task": TodoItem(
                    id=1,
                    title="Debug the issue",
                    description="Fix the bug",
                ),
                "prompt": "Debug this",
                "use_subagents": True,
                "subagent_usage": {},  # Start with empty usage
            }

            result = await execute_task_node(
                initial_state,
                agent=None,
                use_subagents=True,
            )

            # Verify subagent_usage is updated
            assert "subagent_usage" in result
            assert result["subagent_usage"].get("debugger", 0) == 1

    @pytest.mark.asyncio
    async def test_subagent_config_passed_through(
        self, mock_subagent_result: SubAgentResult
    ) -> None:
        """Test that subagent config is passed to spawn_for_task."""
        from ai_infra.executor.agents.config import SubAgentConfig

        config = SubAgentConfig.with_overrides({"tester": "gpt-4o-mini"})

        with patch(
            "ai_infra.executor.nodes.execute.spawn_for_task",
            new_callable=AsyncMock,
            return_value=mock_subagent_result,
        ) as mock_spawn:
            from ai_infra.executor.nodes.execute import execute_task_node

            initial_state: ExecutorGraphState = {
                "roadmap_path": "/tmp/test.md",
                "todos": [],
                "current_task": TodoItem(
                    id=1,
                    title="Write tests",
                    description="Add tests",
                ),
                "prompt": "Write tests",
                "use_subagents": True,
            }

            await execute_task_node(
                initial_state,
                agent=None,
                use_subagents=True,
                subagent_config=config,
            )

            # Verify config was passed
            mock_spawn.assert_called_once()
            call_args = mock_spawn.call_args
            config_arg = call_args[0][2]  # Third positional arg (config)
            assert config_arg is config


class TestSubagentRoutingWithGraph:
    """Tests for subagent routing through ExecutorGraph."""

    def test_graph_accepts_use_subagents(self, test_roadmap: Path) -> None:
        """Test that ExecutorGraph accepts use_subagents parameter."""
        graph = ExecutorGraph(
            roadmap_path=str(test_roadmap),
            use_subagents=True,
        )
        assert graph.use_subagents is True

    def test_graph_accepts_subagent_config(self, test_roadmap: Path) -> None:
        """Test that ExecutorGraph accepts subagent_config parameter."""
        from ai_infra.executor.agents.config import SubAgentConfig

        config = SubAgentConfig.with_overrides({"coder": "gpt-4o"})
        graph = ExecutorGraph(
            roadmap_path=str(test_roadmap),
            use_subagents=True,
            subagent_config=config,
        )
        assert graph.subagent_config is config

    def test_graph_default_no_subagents(self, test_roadmap: Path) -> None:
        """Test that use_subagents defaults to False."""
        graph = ExecutorGraph(
            roadmap_path=str(test_roadmap),
        )
        assert graph.use_subagents is False


class TestAgentTypeSelection:
    """Tests for correct agent type selection during routing."""

    @pytest.mark.asyncio
    async def test_coder_selected_for_implementation(
        self, mock_subagent_result: SubAgentResult
    ) -> None:
        """Test that implementation tasks use CoderAgent."""
        with patch(
            "ai_infra.executor.nodes.execute.spawn_for_task",
            new_callable=AsyncMock,
            return_value=mock_subagent_result,
        ):
            from ai_infra.executor.nodes.execute import execute_task_node

            state: ExecutorGraphState = {
                "roadmap_path": "/tmp/test.md",
                "todos": [],
                "current_task": TodoItem(
                    id=1,
                    title="Create user service",
                    description="Implement user CRUD",
                ),
                "prompt": "Create service",
                "use_subagents": True,
            }

            result = await execute_task_node(state, agent=None, use_subagents=True)
            assert result["subagent_used"] == "coder"

    @pytest.mark.asyncio
    async def test_debugger_selected_for_fix(self, mock_subagent_result: SubAgentResult) -> None:
        """Test that fix tasks use DebuggerAgent."""
        with patch(
            "ai_infra.executor.nodes.execute.spawn_for_task",
            new_callable=AsyncMock,
            return_value=mock_subagent_result,
        ):
            from ai_infra.executor.nodes.execute import execute_task_node

            state: ExecutorGraphState = {
                "roadmap_path": "/tmp/test.md",
                "todos": [],
                "current_task": TodoItem(
                    id=1,
                    title="Fix authentication bug",
                    description="Debug login failure",
                ),
                "prompt": "Fix bug",
                "use_subagents": True,
            }

            result = await execute_task_node(state, agent=None, use_subagents=True)
            assert result["subagent_used"] == "debugger"

    @pytest.mark.asyncio
    async def test_reviewer_selected_for_review(self, mock_subagent_result: SubAgentResult) -> None:
        """Test that review tasks use ReviewerAgent."""
        with patch(
            "ai_infra.executor.nodes.execute.spawn_for_task",
            new_callable=AsyncMock,
            return_value=mock_subagent_result,
        ):
            from ai_infra.executor.nodes.execute import execute_task_node

            state: ExecutorGraphState = {
                "roadmap_path": "/tmp/test.md",
                "todos": [],
                "current_task": TodoItem(
                    id=1,
                    title="Code review the authentication module",
                    description="Review and refactor the auth code",
                ),
                "prompt": "Review changes",
                "use_subagents": True,
            }

            result = await execute_task_node(state, agent=None, use_subagents=True)
            # Orchestrator may choose reviewer or researcher for review tasks
            assert result["subagent_used"] in ("reviewer", "researcher")
