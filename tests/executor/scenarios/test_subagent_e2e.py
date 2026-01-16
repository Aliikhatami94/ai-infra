"""End-to-end tests for multi-task subagent routing (Phase 7.5.3).

Tests that multiple tasks in a single roadmap route to different
specialized subagents based on task type.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ai_infra.executor.agents.base import SubAgentResult
from ai_infra.executor.agents.config import SubAgentConfig
from ai_infra.executor.agents.registry import SubAgentType
from ai_infra.executor.todolist import TodoItem

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def multi_task_roadmap(tmp_path: Path) -> Path:
    """Create a roadmap with multiple task types."""
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text("""\
# Multi-Agent Test

## Phase 1: Implementation

- [ ] Implement user registration
  Description: Create user signup endpoint with validation.

- [ ] Write tests for registration
  Description: Add unit tests for the registration flow.

- [ ] Fix any failing tests
  Description: Debug and fix test failures.
""")
    return roadmap


@pytest.fixture
def diverse_roadmap(tmp_path: Path) -> Path:
    """Create a roadmap with all agent types."""
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text("""\
# Complete Workflow

## Phase 1: Research
- [ ] Research authentication best practices
  Description: Look up OAuth2 and JWT patterns.

## Phase 2: Implementation
- [ ] Implement auth module
  Description: Create the authentication service.

## Phase 3: Testing
- [ ] Write unit tests for auth
  Description: Add comprehensive test coverage.

## Phase 4: Review
- [ ] Review auth code
  Description: Code review for security.

## Phase 5: Debug
- [ ] Fix authentication bugs
  Description: Debug any issues found.
""")
    return roadmap


@pytest.fixture
def mock_successful_result() -> SubAgentResult:
    """Create a successful subagent result."""
    return SubAgentResult(
        success=True,
        output="Task completed",
        files_modified=["src/module.py"],
        metrics={"duration_ms": 150},
    )


# =============================================================================
# Phase 7.5.3: End-to-End Multi-Task Routing Tests
# =============================================================================


class TestMultiTaskSubagentRouting:
    """E2E tests for multi-task subagent routing."""

    @pytest.mark.asyncio
    async def test_different_tasks_use_different_agents(
        self, mock_successful_result: SubAgentResult
    ) -> None:
        """Test that multiple tasks route to different agents."""
        tasks = [
            TodoItem(
                id=1,
                title="Implement user registration",
                description="Create signup endpoint",
            ),
            TodoItem(
                id=2,
                title="Run tests for registration",
                description="Execute unit tests",
            ),
            TodoItem(
                id=3,
                title="Fix any failing tests",
                description="Debug failures",
            ),
        ]

        agents_used: list[str] = []

        async def mock_spawn(task, context, config=None, agent_type=None):
            if agent_type:
                agents_used.append(agent_type)
            else:
                # Default to CODER when no agent_type specified
                agents_used.append(SubAgentType.CODER.value)
            return mock_successful_result

        with patch(
            "ai_infra.executor.nodes.execute.spawn_for_task",
            new_callable=AsyncMock,
            side_effect=mock_spawn,
        ):
            from ai_infra.executor.nodes.execute import execute_task_node

            for task in tasks:
                state = {
                    "roadmap_path": "/tmp/test.md",
                    "todos": [],
                    "current_task": task,
                    "prompt": task.description,
                    "use_subagents": True,
                    "subagent_usage": {},
                }
                await execute_task_node(state, agent=None, use_subagents=True)

        # Verify different agents were used
        assert "coder" in agents_used
        assert "tester" in agents_used
        assert "debugger" in agents_used

    @pytest.mark.asyncio
    async def test_subagent_usage_accumulates(self, mock_successful_result: SubAgentResult) -> None:
        """Test that subagent_usage metrics accumulate across tasks."""
        tasks = [
            TodoItem(id=1, title="Create module", description="Build the module"),
            TodoItem(id=2, title="Create another feature", description="Build it"),
            TodoItem(id=3, title="Run tests", description="Execute tests"),
        ]

        with patch(
            "ai_infra.executor.nodes.execute.spawn_for_task",
            new_callable=AsyncMock,
            return_value=mock_successful_result,
        ):
            from ai_infra.executor.nodes.execute import execute_task_node

            # Track cumulative usage
            cumulative_usage: dict[str, int] = {}

            for task in tasks:
                state = {
                    "roadmap_path": "/tmp/test.md",
                    "todos": [],
                    "current_task": task,
                    "prompt": task.description or task.title,
                    "use_subagents": True,
                    "subagent_usage": cumulative_usage.copy(),
                }
                result = await execute_task_node(state, agent=None, use_subagents=True)
                cumulative_usage = result.get("subagent_usage", {})

        # Verify usage counts
        assert cumulative_usage.get("coder", 0) >= 2  # Two coding tasks
        assert cumulative_usage.get("tester", 0) >= 1  # One test task


class TestAllAgentTypes:
    """Tests that all agent types can be used in a workflow."""

    @pytest.mark.asyncio
    async def test_all_five_agents_used(self, mock_successful_result: SubAgentResult) -> None:
        """Test workflow using all five agent types."""
        tasks = [
            TodoItem(id=1, title="Research caching options", description="Investigate options"),
            TodoItem(id=2, title="Implement the feature", description="Create the module"),
            TodoItem(id=3, title="Run tests for module", description="Execute test suite"),
            TodoItem(id=4, title="Review the code", description="Code review"),
            TodoItem(id=5, title="Fix any bugs", description="Debug issues"),
        ]

        agents_used: set[str] = set()

        async def track_agent(task, context, config=None, agent_type=None):
            if agent_type:
                agents_used.add(agent_type)
            else:
                # Default to CODER when no agent_type specified
                agents_used.add(SubAgentType.CODER.value)
            return mock_successful_result

        with patch(
            "ai_infra.executor.nodes.execute.spawn_for_task",
            new_callable=AsyncMock,
            side_effect=track_agent,
        ):
            from ai_infra.executor.nodes.execute import execute_task_node

            for task in tasks:
                state = {
                    "roadmap_path": "/tmp/test.md",
                    "todos": [],
                    "current_task": task,
                    "prompt": task.description or task.title,
                    "use_subagents": True,
                }
                await execute_task_node(state, agent=None, use_subagents=True)

        # Verify all agents were used
        expected_agents = {"researcher", "coder", "tester", "reviewer", "debugger"}
        assert agents_used == expected_agents


class TestConfigOverridesInE2E:
    """Tests for config overrides in E2E scenarios."""

    @pytest.mark.asyncio
    async def test_config_overrides_applied(self, mock_successful_result: SubAgentResult) -> None:
        """Test that model overrides are passed through during execution."""
        config = SubAgentConfig.with_overrides(
            {
                "coder": "gpt-4o",
                "tester": "gpt-4o-mini",
            }
        )

        configs_received: list[SubAgentConfig | None] = []

        async def capture_config(task, context, config=None, agent_type=None):
            configs_received.append(config)
            return mock_successful_result

        with patch(
            "ai_infra.executor.nodes.execute.spawn_for_task",
            new_callable=AsyncMock,
            side_effect=capture_config,
        ):
            from ai_infra.executor.nodes.execute import execute_task_node

            tasks = [
                TodoItem(id=1, title="Create feature", description="Build it"),
                TodoItem(id=2, title="Write tests", description="Add tests"),
            ]

            for task in tasks:
                state = {
                    "roadmap_path": "/tmp/test.md",
                    "todos": [],
                    "current_task": task,
                    "prompt": task.description or task.title,
                    "use_subagents": True,
                }
                await execute_task_node(
                    state,
                    agent=None,
                    use_subagents=True,
                    subagent_config=config,
                )

        # Verify config was passed to all calls
        assert all(c is config for c in configs_received)
        assert len(configs_received) == 2


class TestSubagentFailureHandling:
    """Tests for handling subagent failures."""

    @pytest.mark.asyncio
    async def test_failed_subagent_recorded(self) -> None:
        """Test that failed subagent execution is recorded properly."""
        failed_result = SubAgentResult(
            success=False,
            output="",
            error="Task failed due to invalid input",
            metrics={"duration_ms": 50},
        )

        with patch(
            "ai_infra.executor.nodes.execute.spawn_for_task",
            new_callable=AsyncMock,
            return_value=failed_result,
        ):
            from ai_infra.executor.nodes.execute import execute_task_node

            state = {
                "roadmap_path": "/tmp/test.md",
                "todos": [],
                "current_task": TodoItem(
                    id=1,
                    title="Implement feature",
                    description="Create module",
                ),
                "prompt": "Create module",
                "use_subagents": True,
            }

            result = await execute_task_node(state, agent=None, use_subagents=True)

        # Verify failure is recorded
        assert result.get("error") is not None
        assert "subagent_execution" in result["error"]["error_type"]
        # But subagent was still tracked
        assert result["subagent_used"] == "coder"

    @pytest.mark.asyncio
    async def test_spawn_exception_handled(self) -> None:
        """Test that exceptions during spawning are handled gracefully."""
        with patch(
            "ai_infra.executor.nodes.execute.spawn_for_task",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Agent initialization failed"),
        ):
            from ai_infra.executor.nodes.execute import execute_task_node

            state = {
                "roadmap_path": "/tmp/test.md",
                "todos": [],
                "current_task": TodoItem(
                    id=1,
                    title="Create feature",
                    description="Build it",
                ),
                "prompt": "Build it",
                "use_subagents": True,
            }

            result = await execute_task_node(state, agent=None, use_subagents=True)

        # Verify error is captured
        assert result.get("error") is not None
        assert "subagent_routing" in result["error"]["error_type"]
        assert result["error"]["recoverable"] is True


class TestSubagentRoutingDisabled:
    """Tests for when subagent routing is disabled."""

    @pytest.mark.asyncio
    async def test_no_subagent_when_disabled(self) -> None:
        """Test that subagents are not used when use_subagents=False."""
        with patch(
            "ai_infra.executor.nodes.execute.spawn_for_task",
            new_callable=AsyncMock,
        ) as mock_spawn:
            from ai_infra.executor.nodes.execute import execute_task_node

            state = {
                "roadmap_path": "/tmp/test.md",
                "todos": [],
                "current_task": TodoItem(
                    id=1,
                    title="Create feature",
                    description="Build it",
                ),
                "prompt": "Build it",
                "use_subagents": False,  # Disabled
            }

            # Without agent and without subagents, should return error (not raise)
            result = await execute_task_node(
                state,
                agent=None,  # No agent
                use_subagents=False,
            )

            # Verify spawn_for_task was NOT called
            mock_spawn.assert_not_called()
            # Verify error was returned (not exception)
            assert result.get("error") is not None
            assert "No agent or router provided" in result["error"]["message"]
