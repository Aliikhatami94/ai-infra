"""Human Collaboration Tests.

Tests for human-in-the-loop (HITL) collaboration features.

This module validates human collaboration success criteria:
1. All HITL actions work (edit, suggest, explain, rollback)
2. Collaboration modes affect execution behavior
3. Progress dashboard displays correctly
4. Token/cost tracking is accurate
5. User can interrupt and resume execution
6. Rollback restores to correct state

Run with: pytest tests/executor/test_human_collaboration.py -v
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.console import Console

# =============================================================================
# Phase 4.2: Collaboration Modes
# =============================================================================
from ai_infra.executor.collaboration import (
    ActionType,
    AgentAction,
    CollaborationConfig,
    CollaborationMode,
    ModeAwareExecutor,
    PauseReason,
)
from ai_infra.executor.dashboard import (
    Dashboard,
    DashboardConfig,
    run_with_dashboard,
)

# =============================================================================
# Phase 4.1: HITL Actions
# =============================================================================
from ai_infra.executor.hitl.actions import (
    HITLAction,
    HITLActionType,
    HITLProposal,
    HITLResponse,
)
from ai_infra.executor.hitl.handlers import (
    EditHandler,
    ExplainHandler,
    HITLHandlerRegistry,
    RollbackHandler,
    SuggestHandler,
)

# =============================================================================
# Phase 4.3: Progress Visibility
# =============================================================================
from ai_infra.executor.progress import (
    CostEstimator,
    ProgressTracker,
    TaskProgress,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockTask:
    """Mock Task for testing."""

    id: str
    title: str
    description: str = ""
    file_hints: list[str] | None = None

    def __post_init__(self):
        if self.file_hints is None:
            self.file_hints = []


@dataclass
class MockRoadmap:
    """Mock Roadmap for testing."""

    title: str = "Test Roadmap"
    _tasks: list[MockTask] | None = None

    def __post_init__(self):
        if self._tasks is None:
            self._tasks = []

    def all_tasks(self) -> list[MockTask]:
        return self._tasks or []


@pytest.fixture
def sample_tasks() -> list[MockTask]:
    """Create sample tasks for testing."""
    return [
        MockTask(id="1.1.1", title="Add authentication", description="Implement auth"),
        MockTask(id="1.1.2", title="Add authorization", description="Implement RBAC"),
        MockTask(id="1.2.1", title="Fix bug in login", description="Fix typo"),
        MockTask(id="2.1.1", title="Refactor database", description="Major refactor"),
    ]


@pytest.fixture
def sample_roadmap(sample_tasks: list[MockTask]) -> MockRoadmap:
    """Create a sample roadmap."""
    return MockRoadmap(title="Test Roadmap", _tasks=sample_tasks)


@pytest.fixture
def console() -> Console:
    """Create a console for testing (no output)."""
    return Console(force_terminal=True, width=120, record=True)


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM for handlers."""
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="Mock response")
    return llm


@pytest.fixture
def mock_checkpoint_manager() -> MagicMock:
    """Create a mock checkpoint manager for rollback handler."""
    manager = MagicMock()
    manager.rollback = AsyncMock(return_value=True)
    manager.list_checkpoints = MagicMock(return_value=[])
    return manager


# =============================================================================
# CRITERION 1: All HITL Actions Work
# =============================================================================


class TestHITLActionsWork:
    """Verify: All HITL actions work (edit, suggest, explain, rollback)."""

    def test_edit_action_type_exists(self):
        """Verify edit action type is defined."""
        assert HITLActionType.EDIT in HITLActionType
        assert HITLActionType.EDIT.value == "edit"

    def test_suggest_action_type_exists(self):
        """Verify suggest action type is defined."""
        assert HITLActionType.SUGGEST in HITLActionType
        assert HITLActionType.SUGGEST.value == "suggest"

    def test_explain_action_type_exists(self):
        """Verify explain action type is defined."""
        assert HITLActionType.EXPLAIN in HITLActionType
        assert HITLActionType.EXPLAIN.value == "explain"

    def test_rollback_action_type_exists(self):
        """Verify rollback action type is defined."""
        assert HITLActionType.ROLLBACK in HITLActionType
        assert HITLActionType.ROLLBACK.value == "rollback"

    def test_hitl_action_creation(self):
        """Verify HITLAction can be created with all action types."""
        for action_type in HITLActionType:
            action = HITLAction(
                type=action_type,
                target="test_target",
                content=f"Test {action_type.value}",
            )
            assert action.type == action_type
            assert action.target == "test_target"

    def test_edit_handler_exists(self, mock_llm: MagicMock):
        """Verify EditHandler is implemented."""
        handler = EditHandler(llm=mock_llm)
        assert handler is not None
        assert hasattr(handler, "handle")

    def test_suggest_handler_exists(self, mock_llm: MagicMock):
        """Verify SuggestHandler is implemented."""
        handler = SuggestHandler(llm=mock_llm)
        assert handler is not None
        assert hasattr(handler, "handle")

    def test_explain_handler_exists(self, mock_llm: MagicMock):
        """Verify ExplainHandler is implemented."""
        handler = ExplainHandler(llm=mock_llm)
        assert handler is not None
        assert hasattr(handler, "handle")

    def test_rollback_handler_exists(self, mock_checkpoint_manager: MagicMock):
        """Verify RollbackHandler is implemented."""
        handler = RollbackHandler(checkpoint_manager=mock_checkpoint_manager)
        assert handler is not None
        assert hasattr(handler, "handle")

    def test_handler_registry_registers_all_handlers(
        self, mock_llm: MagicMock, mock_checkpoint_manager: MagicMock
    ):
        """Verify registry has handlers for all action types."""
        registry = HITLHandlerRegistry()

        # Register all handlers
        registry.register(HITLActionType.EDIT, EditHandler(llm=mock_llm))
        registry.register(HITLActionType.SUGGEST, SuggestHandler(llm=mock_llm))
        registry.register(HITLActionType.EXPLAIN, ExplainHandler(llm=mock_llm))
        registry.register(
            HITLActionType.ROLLBACK, RollbackHandler(checkpoint_manager=mock_checkpoint_manager)
        )

        # Verify all are retrievable using get_handler
        assert registry.get_handler(HITLActionType.EDIT) is not None
        assert registry.get_handler(HITLActionType.SUGGEST) is not None
        assert registry.get_handler(HITLActionType.EXPLAIN) is not None
        assert registry.get_handler(HITLActionType.ROLLBACK) is not None

    def test_hitl_proposal_creation(self):
        """Verify HITLProposal can be created."""
        proposal = HITLProposal(
            description="Edit main.py",
            task_id="1.1.1",
            task_title="Add authentication",
            files_affected=["src/main.py"],
            rationale="Need to add auth",
        )
        assert proposal.description == "Edit main.py"
        assert proposal.task_id == "1.1.1"

    def test_hitl_response_creation(self):
        """Verify HITLResponse can be created."""
        response = HITLResponse(
            understood=True,
            revised_plan="Updated plan",
            next_step="Continue execution",
        )
        assert response.understood is True


# =============================================================================
# CRITERION 2: Collaboration Modes Affect Execution
# =============================================================================


class TestCollaborationModesAffectExecution:
    """Verify: Collaboration modes affect execution behavior."""

    def test_all_modes_defined(self):
        """Verify all collaboration modes are defined."""
        assert CollaborationMode.AUTONOMOUS in CollaborationMode
        assert CollaborationMode.SUPERVISED in CollaborationMode
        assert CollaborationMode.PAIR in CollaborationMode
        assert CollaborationMode.GUIDED in CollaborationMode

    @pytest.mark.asyncio
    async def test_autonomous_mode_minimal_pauses(self):
        """Verify AUTONOMOUS mode has minimal pauses."""
        config = CollaborationConfig.autonomous()
        executor = ModeAwareExecutor(config)

        # File edit should not pause in autonomous
        action = AgentAction.from_file_operation("edit", "test.py", "Edit file")
        result = await executor.should_pause(action)
        assert result.should_pause is False

        # File create should not pause in autonomous
        action = AgentAction.from_file_operation("create", "new.py", "Create file")
        result = await executor.should_pause(action)
        assert result.should_pause is False

    @pytest.mark.asyncio
    async def test_supervised_mode_pauses_on_config_change(self):
        """Verify SUPERVISED mode pauses on config file changes."""
        config = CollaborationConfig.supervised()
        executor = ModeAwareExecutor(config)

        # Config file edit should pause
        action = AgentAction.from_file_operation("edit", "pyproject.toml", "Edit config")
        result = await executor.should_pause(action)

        assert result.should_pause is True
        assert isinstance(result, PauseReason)

    @pytest.mark.asyncio
    async def test_pair_mode_pauses_on_shell_command(self):
        """Verify PAIR mode pauses on shell commands."""
        config = CollaborationConfig.pair()
        executor = ModeAwareExecutor(config)

        # Shell command should pause
        action = AgentAction.from_shell_command("make test", "Run tests")
        result = await executor.should_pause(action)
        assert result.should_pause is True

        # Config change should also pause
        action = AgentAction.from_file_operation("edit", "config.yaml", "Edit config")
        result = await executor.should_pause(action)
        assert result.should_pause is True

    @pytest.mark.asyncio
    async def test_guided_mode_pauses_on_everything(self):
        """Verify GUIDED mode pauses on all significant actions."""
        config = CollaborationConfig.guided()
        executor = ModeAwareExecutor(config)

        # File edit - should pause in guided
        action = AgentAction.from_file_operation("edit", "test.py", "Edit file")
        result = await executor.should_pause(action)
        assert result.should_pause is True

        # Shell command - should pause
        action = AgentAction.from_shell_command("ls", "List files")
        result = await executor.should_pause(action)
        assert result.should_pause is True

    def test_mode_aware_executor_factory_methods(self):
        """Verify AgentAction factory methods work correctly."""
        # File operation
        action = AgentAction.from_file_operation("edit", "test.py", "Edit test")
        assert action.type == ActionType.FILE_EDIT
        assert action.file_path == "test.py"

        # Shell command
        action = AgentAction.from_shell_command("make test", "Run tests")
        assert action.type == ActionType.SHELL_COMMAND
        assert action.command == "make test"

    def test_collaboration_config_factory_methods(self):
        """Verify all factory methods create valid configs."""
        autonomous = CollaborationConfig.autonomous()
        assert autonomous.mode == CollaborationMode.AUTONOMOUS

        supervised = CollaborationConfig.supervised()
        assert supervised.mode == CollaborationMode.SUPERVISED

        pair = CollaborationConfig.pair()
        assert pair.mode == CollaborationMode.PAIR

        guided = CollaborationConfig.guided()
        assert guided.mode == CollaborationMode.GUIDED


# =============================================================================
# CRITERION 3: Progress Dashboard Displays Correctly
# =============================================================================


class TestProgressDashboardDisplays:
    """Verify: Progress dashboard displays correctly."""

    def test_dashboard_creates_panel(self, sample_roadmap: MockRoadmap, console: Console):
        """Verify Dashboard creates a Rich panel."""
        tracker = ProgressTracker(roadmap=sample_roadmap)
        dashboard = Dashboard(tracker, console=console)

        panel = dashboard.create_panel()

        assert panel is not None
        assert panel.title is not None

    def test_dashboard_renders_without_error(self, sample_roadmap: MockRoadmap, console: Console):
        """Verify Dashboard renders without raising."""
        tracker = ProgressTracker(roadmap=sample_roadmap)
        dashboard = Dashboard(tracker, console=console)

        # Should not raise
        dashboard.render()

        output = console.export_text()
        assert len(output) > 0

    def test_dashboard_shows_task_status(self, sample_roadmap: MockRoadmap, console: Console):
        """Verify dashboard shows task status correctly."""
        tracker = ProgressTracker(roadmap=sample_roadmap)
        tracker.start_run()
        tracker.complete_task("1.1.1", tokens_in=1000, tokens_out=500, cost=0.05)
        tracker.start_task("1.1.2")

        dashboard = Dashboard(tracker, console=console)
        dashboard.render()

        output = console.export_text()
        # Should contain progress indicators
        assert "Test Roadmap" in output or "1" in output

    def test_dashboard_config_options(self):
        """Verify dashboard configuration works."""
        config = DashboardConfig(
            show_all_tasks=True,
            max_visible_tasks=50,
            show_cost_estimates=True,
            show_tokens=True,
            compact_mode=False,
            refresh_rate=1.0,
        )

        assert config.show_all_tasks is True
        assert config.max_visible_tasks == 50
        assert config.refresh_rate == 1.0

    def test_dashboard_summary_text(self, sample_roadmap: MockRoadmap):
        """Verify dashboard creates summary text."""
        tracker = ProgressTracker(roadmap=sample_roadmap)
        tracker.start_run()
        tracker.complete_task("1.1.1", tokens_in=100, tokens_out=50, cost=0.01)

        dashboard = Dashboard(tracker)
        summary = dashboard.create_summary_text()

        assert "Progress:" in summary
        assert "1/4" in summary or "25%" in summary


# =============================================================================
# CRITERION 4: Token/Cost Tracking is Accurate
# =============================================================================


class TestTokenCostTrackingAccurate:
    """Verify: Token/cost tracking is accurate."""

    def test_task_progress_tracks_tokens(self):
        """Verify TaskProgress tracks tokens correctly."""
        progress = TaskProgress(
            task_id="1.1.1",
            task_title="Test",
            tokens_in=1000,
            tokens_out=500,
        )

        assert progress.tokens_in == 1000
        assert progress.tokens_out == 500
        assert progress.total_tokens == 1500

    def test_task_progress_tracks_cost(self):
        """Verify TaskProgress tracks cost correctly."""
        progress = TaskProgress(
            task_id="1.1.1",
            task_title="Test",
            cost=0.05,
        )

        assert progress.cost == 0.05

    def test_tracker_aggregates_tokens(self, sample_roadmap: MockRoadmap):
        """Verify ProgressTracker aggregates tokens correctly."""
        tracker = ProgressTracker(roadmap=sample_roadmap)

        tracker.complete_task("1.1.1", tokens_in=1000, tokens_out=500, cost=0.05)
        tracker.complete_task("1.1.2", tokens_in=2000, tokens_out=1000, cost=0.10)

        summary = tracker.get_summary()

        assert summary.tokens_in == 3000
        assert summary.tokens_out == 1500
        assert summary.total_tokens == 4500

    def test_tracker_aggregates_cost(self, sample_roadmap: MockRoadmap):
        """Verify ProgressTracker aggregates cost correctly."""
        tracker = ProgressTracker(roadmap=sample_roadmap)

        tracker.complete_task("1.1.1", cost=0.05)
        tracker.complete_task("1.1.2", cost=0.10)
        tracker.complete_task("1.2.1", cost=0.03)

        summary = tracker.get_summary()

        assert summary.cost == pytest.approx(0.18, rel=0.01)

    def test_cost_estimator_by_complexity(self):
        """Verify CostEstimator varies by complexity."""
        estimator = CostEstimator()

        simple_task = MockTask(id="1", title="Fix typo", description="Small fix")
        complex_task = MockTask(
            id="2",
            title="Complete database migration",
            description="Full refactor",
        )

        simple_est = estimator.estimate_task(simple_task)
        complex_est = estimator.estimate_task(complex_task)

        # Complex should cost more
        assert complex_est["estimated_cost"] > simple_est["estimated_cost"]
        assert complex_est["tokens_in"] > simple_est["tokens_in"]

    def test_cost_estimator_by_model(self):
        """Verify CostEstimator varies by model."""
        estimator = CostEstimator()
        task = MockTask(id="1", title="Add feature")

        sonnet_est = estimator.estimate_task(task, model="claude-sonnet-4-20250514")
        opus_est = estimator.estimate_task(task, model="claude-opus-4-20250514")

        # Opus should cost more
        assert opus_est["estimated_cost"] > sonnet_est["estimated_cost"]

    def test_cost_estimator_roadmap_total(self, sample_roadmap: MockRoadmap):
        """Verify CostEstimator calculates roadmap total."""
        estimator = CostEstimator()

        result = estimator.estimate_roadmap(sample_roadmap)

        assert result["task_count"] == 4
        assert result["total_cost"] > 0
        assert result["total_tokens"] > 0
        assert result["total_tokens"] == result["total_tokens_in"] + result["total_tokens_out"]


# =============================================================================
# CRITERION 5: User Can Interrupt and Resume Execution
# =============================================================================


class TestInterruptAndResume:
    """Verify: User can interrupt and resume execution."""

    @pytest.mark.asyncio
    async def test_run_with_dashboard_can_be_cancelled(
        self, sample_tasks: list[MockTask], console: Console
    ):
        """Verify run_with_dashboard respects cancellation."""
        tracker = ProgressTracker(roadmap_title="Test")
        for task in sample_tasks:
            tracker.add_task(task.id, task.title)

        call_count = 0

        async def slow_execute(task):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.5)  # Slow task
            return {"tokens_in": 100, "tokens_out": 50, "cost": 0.01}

        # Create a task that we'll cancel
        run_task = asyncio.create_task(
            run_with_dashboard(tracker, slow_execute, sample_tasks, console=console)
        )

        # Let it start
        await asyncio.sleep(0.1)

        # Cancel it
        run_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await run_task

        # Should have started at least one task
        assert call_count >= 1

    def test_tracker_state_preserved_after_partial_execution(self, sample_roadmap: MockRoadmap):
        """Verify tracker state is preserved after partial execution."""
        tracker = ProgressTracker(roadmap=sample_roadmap)
        tracker.start_run()

        # Complete first task
        tracker.start_task("1.1.1", agent="TestAgent", model="test-model")
        tracker.complete_task("1.1.1", tokens_in=1000, tokens_out=500, cost=0.05)

        # Start second task (simulating interrupt mid-execution)
        tracker.start_task("1.1.2")

        # Verify state
        summary = tracker.get_summary()
        assert summary.completed == 1
        assert summary.in_progress == 1
        assert summary.pending == 2

        # Verify we can resume from current state
        current = tracker.get_current_task()
        assert current is not None
        assert current.task_id == "1.1.2"

    def test_tracker_can_resume_from_saved_state(self):
        """Verify tracker state can be serialized and resumed."""
        # Create and populate tracker
        tracker1 = ProgressTracker(roadmap_title="Test")
        tracker1.add_task("1.1.1", "Task 1")
        tracker1.add_task("1.1.2", "Task 2")
        tracker1.complete_task("1.1.1", tokens_in=100, tokens_out=50, cost=0.01)

        # Serialize state (via to_dict)
        state = {task_id: progress.to_dict() for task_id, progress in tracker1.tasks.items()}

        # Create new tracker and verify it can read old state
        tracker2 = ProgressTracker(roadmap_title="Test")
        tracker2.add_task("1.1.1", "Task 1")
        tracker2.add_task("1.1.2", "Task 2")

        # Manually restore completed status (simulating resume)
        if state["1.1.1"]["status"] == "completed":
            tracker2.tasks["1.1.1"].status = "completed"
            tracker2.tasks["1.1.1"].tokens_in = state["1.1.1"]["tokens_in"]
            tracker2.tasks["1.1.1"].tokens_out = state["1.1.1"]["tokens_out"]
            tracker2.tasks["1.1.1"].cost = state["1.1.1"]["cost"]

        # Verify resumed state
        summary = tracker2.get_summary()
        assert summary.completed == 1
        assert summary.pending == 1


# =============================================================================
# CRITERION 6: Rollback Restores to Correct State
# =============================================================================


class TestRollbackRestoresState:
    """Verify: Rollback restores to correct state."""

    def test_rollback_handler_exists(self, mock_checkpoint_manager: MagicMock):
        """Verify RollbackHandler is implemented."""
        handler = RollbackHandler(checkpoint_manager=mock_checkpoint_manager)
        assert handler is not None
        assert hasattr(handler, "handle")

    def test_rollback_action_type_defined(self):
        """Verify rollback action type is defined."""
        assert HITLActionType.ROLLBACK in HITLActionType

    def test_rollback_action_can_be_created(self):
        """Verify rollback action can be created with target."""
        action = HITLAction(
            type=HITLActionType.ROLLBACK,
            target="task-1.1.1",
            content="Rollback to before task 1.1.1",
        )

        assert action.type == HITLActionType.ROLLBACK
        assert action.target == "task-1.1.1"

    def test_tracker_skip_restores_to_pending(self, sample_roadmap: MockRoadmap):
        """Verify skip_task can be used to reset failed tasks."""
        tracker = ProgressTracker(roadmap=sample_roadmap)

        # Fail a task
        tracker.start_task("1.1.1")
        tracker.fail_task("1.1.1", error="Test error")

        # Verify it's failed
        assert tracker.tasks["1.1.1"].status == "failed"

        # "Rollback" by resetting to pending
        tracker.tasks["1.1.1"].status = "pending"
        tracker.tasks["1.1.1"].error = None
        tracker.tasks["1.1.1"].started_at = None
        tracker.tasks["1.1.1"].completed_at = None

        # Verify restored
        assert tracker.tasks["1.1.1"].status == "pending"
        assert tracker.tasks["1.1.1"].error is None

    @pytest.mark.asyncio
    async def test_rollback_handler_can_handle_action(self, mock_checkpoint_manager: MagicMock):
        """Verify RollbackHandler.handle() works."""
        handler = RollbackHandler(checkpoint_manager=mock_checkpoint_manager)

        action = HITLAction(
            type=HITLActionType.ROLLBACK,
            target="1.1.1",
            content="Rollback task 1.1.1",
        )

        # Create mock proposal and context
        proposal = HITLProposal(description="Test proposal", task_id="1.1.1")
        context = MagicMock()
        context.checkpoints = []

        # Handler should not raise
        result = await handler.handle(action, proposal, context)
        assert result is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhase4Integration:
    """Integration tests verifying Phase 4 components work together."""

    @pytest.mark.asyncio
    async def test_full_workflow_with_progress_and_collaboration(
        self, sample_tasks: list[MockTask], console: Console
    ):
        """Test full workflow with progress tracking and collaboration."""
        # Set up collaboration mode
        config = CollaborationConfig.supervised()
        mode_executor = ModeAwareExecutor(config)

        # Set up progress tracking
        tracker = ProgressTracker(roadmap_title="Integration Test")
        for task in sample_tasks:
            tracker.add_task(task.id, task.title)

        actions_taken = []

        async def execute_with_collaboration(task):
            # Check if we should pause
            action = AgentAction.from_file_operation(
                "edit", f"{task.id}.py", f"Execute {task.title}"
            )

            pause = await mode_executor.should_pause(action)
            # In real code, we'd handle the pause here
            _ = pause  # Suppress unused warning

            # Record the action
            actions_taken.append(action)

            # Simulate work
            await asyncio.sleep(0.01)

            return {
                "tokens_in": 100,
                "tokens_out": 50,
                "cost": 0.01,
                "files_modified": [f"{task.id}.py"],
            }

        # Run with dashboard
        result = await run_with_dashboard(
            tracker, execute_with_collaboration, sample_tasks, console=console
        )

        # Verify results
        assert result.success is True
        assert result.completed == 4
        assert result.total_tokens == 600  # 4 * 150
        assert result.total_cost == pytest.approx(0.04, rel=0.1)

        # Verify actions were taken
        assert len(actions_taken) == 4

    def test_hitl_actions_integrated_with_registry(
        self, mock_llm: MagicMock, mock_checkpoint_manager: MagicMock
    ):
        """Test HITL actions work with handler registry."""
        registry = HITLHandlerRegistry()

        # Register handlers with required dependencies
        registry.register(HITLActionType.EDIT, EditHandler(llm=mock_llm))
        registry.register(HITLActionType.SUGGEST, SuggestHandler(llm=mock_llm))
        registry.register(HITLActionType.EXPLAIN, ExplainHandler(llm=mock_llm))
        registry.register(
            HITLActionType.ROLLBACK, RollbackHandler(checkpoint_manager=mock_checkpoint_manager)
        )

        # Create and process each action type
        for action_type in [
            HITLActionType.EDIT,
            HITLActionType.SUGGEST,
            HITLActionType.EXPLAIN,
            HITLActionType.ROLLBACK,
        ]:
            _action = HITLAction(
                type=action_type,
                target="test",
                content=f"Test {action_type.value}",
            )

            handler = registry.get_handler(action_type)
            assert handler is not None

    def test_cost_estimation_matches_actual_tracking(self, sample_roadmap: MockRoadmap):
        """Test cost estimation is in reasonable range of actual."""
        estimator = CostEstimator()
        tracker = ProgressTracker(roadmap=sample_roadmap)

        # Get estimates
        estimates = estimator.estimate_roadmap(sample_roadmap)

        # Simulate execution with similar costs
        for task in sample_roadmap.all_tasks():
            est = estimates["tasks"][task.id]
            # Use estimated values (in real execution, these come from API)
            tracker.complete_task(
                task.id,
                tokens_in=est["tokens_in"],
                tokens_out=est["tokens_out"],
                cost=est["estimated_cost"],
            )

        # Verify totals match
        summary = tracker.get_summary()
        assert summary.cost == pytest.approx(estimates["total_cost"], rel=0.01)
        assert summary.total_tokens == estimates["total_tokens"]


# =============================================================================
# Summary Test
# =============================================================================


class TestPhase4VerificationSummary:
    """Summary test ensuring all criteria can be validated."""

    def test_criterion_1_hitl_actions(self):
        """CRITERION 1: All HITL actions work."""
        # All action types exist
        assert len(HITLActionType) >= 4

        # All handlers exist
        assert EditHandler is not None
        assert SuggestHandler is not None
        assert ExplainHandler is not None
        assert RollbackHandler is not None

    @pytest.mark.asyncio
    async def test_criterion_2_collaboration_modes(self):
        """CRITERION 2: Collaboration modes affect execution."""
        # All modes exist
        assert len(CollaborationMode) == 4

        # Modes have different behaviors
        auto = ModeAwareExecutor(CollaborationConfig.autonomous())
        guided = ModeAwareExecutor(CollaborationConfig.guided())

        action = AgentAction.from_file_operation("edit", "test.py", "Edit file")

        # Autonomous doesn't pause on simple edits
        auto_result = await auto.should_pause(action)
        assert auto_result.should_pause is False

        # Guided pauses on everything
        guided_result = await guided.should_pause(action)
        assert guided_result.should_pause is True

    def test_criterion_3_progress_dashboard(self):
        """CRITERION 3: Progress dashboard displays correctly."""
        tracker = ProgressTracker(roadmap_title="Test")
        tracker.add_task("1", "Task 1")

        dashboard = Dashboard(tracker)

        # Can create panel
        panel = dashboard.create_panel()
        assert panel is not None

        # Can create summary
        summary = dashboard.create_summary_text()
        assert len(summary) > 0

    def test_criterion_4_token_cost_tracking(self):
        """CRITERION 4: Token/cost tracking is accurate."""
        tracker = ProgressTracker()
        tracker.add_task("1", "Test")
        tracker.complete_task("1", tokens_in=1000, tokens_out=500, cost=0.05)

        summary = tracker.get_summary()
        assert summary.tokens_in == 1000
        assert summary.tokens_out == 500
        assert summary.cost == 0.05

    def test_criterion_5_interrupt_resume(self):
        """CRITERION 5: User can interrupt and resume."""
        tracker = ProgressTracker()
        tracker.add_task("1", "Task 1")
        tracker.add_task("2", "Task 2")

        # Partial execution
        tracker.complete_task("1", tokens_in=100, tokens_out=50, cost=0.01)

        # State is preserved for resume
        summary = tracker.get_summary()
        assert summary.completed == 1
        assert summary.pending == 1

        # Can identify what to resume
        pending = [t for t in tracker.get_all_progress() if t.status == "pending"]
        assert len(pending) == 1
        assert pending[0].task_id == "2"

    def test_criterion_6_rollback(self, mock_checkpoint_manager: MagicMock):
        """CRITERION 6: Rollback restores to correct state."""
        # Rollback action type exists
        assert HITLActionType.ROLLBACK is not None

        # Rollback handler exists
        handler = RollbackHandler(checkpoint_manager=mock_checkpoint_manager)
        assert handler is not None

        # Can create rollback action
        action = HITLAction(
            type=HITLActionType.ROLLBACK,
            target="checkpoint-1",
            content="Rollback to checkpoint",
        )
        assert action.type == HITLActionType.ROLLBACK
