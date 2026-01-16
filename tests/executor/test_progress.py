"""Tests for executor progress tracking and dashboard.

Phase 4.3 of EXECUTOR_1.md: Progress Visibility.

Tests cover:
- TaskProgress creation and properties
- ProgressTracker lifecycle (start, complete, fail, skip)
- ProgressSummary calculations and estimates
- CostEstimator complexity assessment and calculations
- Dashboard rendering (unit tests)
- run_with_dashboard integration
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest
from rich.console import Console

from ai_infra.executor.dashboard import (
    STATUS_ICONS,
    STATUS_LABELS,
    Dashboard,
    DashboardConfig,
    run_with_dashboard,
    run_with_simple_progress,
)
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
    """Create a sample roadmap for testing."""
    return MockRoadmap(title="Test Roadmap", _tasks=sample_tasks)


@pytest.fixture
def tracker(sample_roadmap: MockRoadmap) -> ProgressTracker:
    """Create a ProgressTracker with sample roadmap."""
    return ProgressTracker(roadmap=sample_roadmap)


@pytest.fixture
def console() -> Console:
    """Create a console for testing (no output)."""
    return Console(force_terminal=True, width=120, record=True)


# =============================================================================
# TaskProgress Tests
# =============================================================================


class TestTaskProgress:
    """Tests for TaskProgress dataclass."""

    def test_create_task_progress(self):
        """Test creating a TaskProgress instance."""
        progress = TaskProgress(
            task_id="1.1.1",
            task_title="Add authentication",
            status="pending",
        )

        assert progress.task_id == "1.1.1"
        assert progress.task_title == "Add authentication"
        assert progress.status == "pending"
        assert progress.tokens_in == 0
        assert progress.tokens_out == 0
        assert progress.cost == 0.0

    def test_task_progress_total_tokens(self):
        """Test total_tokens property."""
        progress = TaskProgress(
            task_id="1.1.1",
            task_title="Test",
            tokens_in=1000,
            tokens_out=500,
        )

        assert progress.total_tokens == 1500

    def test_task_progress_duration_seconds_with_duration(self):
        """Test duration_seconds with completed task."""
        progress = TaskProgress(
            task_id="1.1.1",
            task_title="Test",
            duration=timedelta(minutes=2, seconds=30),
        )

        assert progress.duration_seconds == 150.0

    def test_task_progress_duration_seconds_in_progress(self):
        """Test duration_seconds for in-progress task."""
        progress = TaskProgress(
            task_id="1.1.1",
            task_title="Test",
            status="in_progress",
            started_at=datetime.now() - timedelta(seconds=10),
        )

        assert progress.duration_seconds >= 10.0

    def test_task_progress_to_dict(self):
        """Test to_dict serialization."""
        now = datetime.now()
        progress = TaskProgress(
            task_id="1.1.1",
            task_title="Test",
            status="completed",
            started_at=now,
            completed_at=now + timedelta(seconds=30),
            tokens_in=1000,
            tokens_out=500,
            cost=0.05,
            agent_used="CoderAgent",
            model_used="claude-sonnet-4-20250514",
            files_modified=["src/main.py"],
        )

        result = progress.to_dict()

        assert result["task_id"] == "1.1.1"
        assert result["tokens_in"] == 1000
        assert result["tokens_out"] == 500
        assert result["cost"] == 0.05
        assert result["agent_used"] == "CoderAgent"
        assert "src/main.py" in result["files_modified"]


# =============================================================================
# ProgressTracker Tests
# =============================================================================


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    def test_create_tracker_with_roadmap(self, sample_roadmap: MockRoadmap):
        """Test creating tracker from roadmap."""
        tracker = ProgressTracker(roadmap=sample_roadmap)

        assert tracker.roadmap_title == "Test Roadmap"
        assert len(tracker.tasks) == 4
        assert "1.1.1" in tracker.tasks

    def test_create_tracker_without_roadmap(self):
        """Test creating tracker without roadmap."""
        tracker = ProgressTracker(roadmap_title="Custom Title")

        assert tracker.roadmap_title == "Custom Title"
        assert len(tracker.tasks) == 0

    def test_add_task(self):
        """Test adding a task to tracker."""
        tracker = ProgressTracker()
        tracker.add_task("1.1.1", "Test Task")

        assert "1.1.1" in tracker.tasks
        assert tracker.tasks["1.1.1"].task_title == "Test Task"
        assert tracker.tasks["1.1.1"].status == "pending"

    def test_start_run(self):
        """Test starting a run."""
        tracker = ProgressTracker()
        tracker.start_run()

        assert tracker.started_at is not None
        assert isinstance(tracker.started_at, datetime)

    def test_start_task(self, tracker: ProgressTracker):
        """Test starting a task."""
        tracker.start_task("1.1.1", agent="CoderAgent", model="claude-sonnet-4-20250514")

        progress = tracker.tasks["1.1.1"]
        assert progress.status == "in_progress"
        assert progress.started_at is not None
        assert progress.agent_used == "CoderAgent"
        assert progress.model_used == "claude-sonnet-4-20250514"

    def test_start_unknown_task_creates_it(self):
        """Test starting an unknown task creates it."""
        tracker = ProgressTracker()
        tracker.start_task("new.task", agent="TestAgent")

        assert "new.task" in tracker.tasks
        assert tracker.tasks["new.task"].status == "in_progress"

    def test_complete_task(self, tracker: ProgressTracker):
        """Test completing a task."""
        tracker.start_task("1.1.1")
        tracker.complete_task(
            "1.1.1",
            tokens_in=1000,
            tokens_out=500,
            cost=0.05,
            files_modified=["src/auth.py"],
            files_created=["src/auth_utils.py"],
        )

        progress = tracker.tasks["1.1.1"]
        assert progress.status == "completed"
        assert progress.completed_at is not None
        assert progress.duration is not None
        assert progress.tokens_in == 1000
        assert progress.tokens_out == 500
        assert progress.cost == 0.05
        assert "src/auth.py" in progress.files_modified
        assert "src/auth_utils.py" in progress.files_created

    def test_fail_task(self, tracker: ProgressTracker):
        """Test failing a task."""
        tracker.start_task("1.1.1")
        tracker.fail_task("1.1.1", error="Test error")

        progress = tracker.tasks["1.1.1"]
        assert progress.status == "failed"
        assert progress.error == "Test error"
        assert progress.completed_at is not None

    def test_skip_task(self, tracker: ProgressTracker):
        """Test skipping a task."""
        tracker.skip_task("1.1.1", reason="Dependencies not met")

        progress = tracker.tasks["1.1.1"]
        assert progress.status == "skipped"
        assert progress.error == "Dependencies not met"

    def test_get_current_task(self, tracker: ProgressTracker):
        """Test getting current task."""
        assert tracker.get_current_task() is None

        tracker.start_task("1.1.1")
        current = tracker.get_current_task()

        assert current is not None
        assert current.task_id == "1.1.1"

    def test_get_elapsed(self, tracker: ProgressTracker):
        """Test getting elapsed time."""
        assert tracker.get_elapsed() is None

        tracker.start_run()
        elapsed = tracker.get_elapsed()

        assert elapsed is not None
        assert elapsed.total_seconds() >= 0

    def test_get_task_progress(self, tracker: ProgressTracker):
        """Test getting specific task progress."""
        progress = tracker.get_task_progress("1.1.1")
        assert progress is not None
        assert progress.task_id == "1.1.1"

        assert tracker.get_task_progress("nonexistent") is None

    def test_get_all_progress(self, tracker: ProgressTracker):
        """Test getting all task progress."""
        all_progress = tracker.get_all_progress()
        assert len(all_progress) == 4
        assert all(isinstance(p, TaskProgress) for p in all_progress)


# =============================================================================
# ProgressSummary Tests
# =============================================================================


class TestProgressSummary:
    """Tests for ProgressSummary calculations."""

    def test_get_summary_initial_state(self, tracker: ProgressTracker):
        """Test summary with no progress."""
        summary = tracker.get_summary()

        assert summary.total == 4
        assert summary.pending == 4
        assert summary.completed == 0
        assert summary.in_progress == 0
        assert summary.failed == 0
        assert summary.skipped == 0
        assert summary.percent == 0.0

    def test_get_summary_with_progress(self, tracker: ProgressTracker):
        """Test summary with mixed progress."""
        tracker.start_run()
        tracker.complete_task("1.1.1", tokens_in=1000, tokens_out=500, cost=0.05)
        tracker.start_task("1.1.2")
        tracker.skip_task("1.2.1")

        summary = tracker.get_summary()

        assert summary.completed == 1
        assert summary.in_progress == 1
        assert summary.skipped == 1
        assert summary.pending == 1
        assert summary.percent == 25.0  # 1/4 * 100

    def test_get_summary_tokens_and_cost(self, tracker: ProgressTracker):
        """Test summary token and cost aggregation."""
        tracker.complete_task("1.1.1", tokens_in=1000, tokens_out=500, cost=0.05)
        tracker.complete_task("1.1.2", tokens_in=2000, tokens_out=1000, cost=0.10)

        summary = tracker.get_summary()

        assert summary.tokens_in == 3000
        assert summary.tokens_out == 1500
        assert summary.total_tokens == 4500
        assert summary.cost == pytest.approx(0.15, rel=0.01)

    def test_get_summary_estimates(self, tracker: ProgressTracker):
        """Test summary remaining estimates."""
        # Complete first task with known duration and cost
        tracker.tasks["1.1.1"].started_at = datetime.now() - timedelta(seconds=60)
        tracker.tasks["1.1.1"].completed_at = datetime.now()
        tracker.tasks["1.1.1"].duration = timedelta(seconds=60)
        tracker.tasks["1.1.1"].status = "completed"
        tracker.tasks["1.1.1"].cost = 0.10

        summary = tracker.get_summary()

        # With 1 task done (60s, $0.10) and 3 remaining,
        # estimate should be 3 * avg
        assert summary.estimated_remaining_cost is not None
        assert summary.estimated_remaining_cost == pytest.approx(0.30, rel=0.1)
        assert summary.estimated_remaining_time is not None

    def test_get_summary_current_task(self, tracker: ProgressTracker):
        """Test summary includes current task info."""
        tracker.start_task("1.1.1")

        summary = tracker.get_summary()

        assert summary.current_task_id == "1.1.1"
        assert summary.current_task_title == "Add authentication"

    def test_summary_to_dict(self, tracker: ProgressTracker):
        """Test ProgressSummary.to_dict()."""
        tracker.start_run()
        tracker.complete_task("1.1.1", tokens_in=100, tokens_out=50, cost=0.01)

        summary = tracker.get_summary()
        result = summary.to_dict()

        assert result["total"] == 4
        assert result["completed"] == 1
        assert result["tokens_in"] == 100
        assert result["cost"] == 0.01
        assert "elapsed_seconds" in result


# =============================================================================
# CostEstimator Tests
# =============================================================================


class TestCostEstimator:
    """Tests for CostEstimator class."""

    def test_estimate_task_default_model(self):
        """Test task estimation with default model."""
        estimator = CostEstimator()
        task = MockTask(id="1.1.1", title="Add feature", description="Implement feature")

        estimate = estimator.estimate_task(task)

        assert "tokens_in" in estimate
        assert "tokens_out" in estimate
        assert "estimated_cost" in estimate
        assert "complexity" in estimate
        assert estimate["complexity"] == "medium"  # "implement" keyword

    def test_complexity_assessment_high(self):
        """Test high complexity detection."""
        estimator = CostEstimator()
        task = MockTask(
            id="1.1.1",
            title="Refactor authentication",
            description="Complete security refactor",
        )

        estimate = estimator.estimate_task(task)

        # "refactor" and "complete" should trigger very_high
        assert estimate["complexity"] in ("high", "very_high")

    def test_complexity_assessment_low(self):
        """Test low complexity detection."""
        estimator = CostEstimator()
        task = MockTask(
            id="1.1.1",
            title="Fix typo in docs",
            description="Fix documentation typo",
        )

        estimate = estimator.estimate_task(task)

        assert estimate["complexity"] == "low"

    def test_file_hints_increase_cost(self):
        """Test that many file hints increase cost estimate."""
        estimator = CostEstimator()

        task_few_files = MockTask(
            id="1.1.1",
            title="Add feature",
            file_hints=["a.py", "b.py"],
        )
        task_many_files = MockTask(
            id="1.1.2",
            title="Add feature",
            file_hints=["a.py", "b.py", "c.py", "d.py", "e.py", "f.py"],
        )

        est_few = estimator.estimate_task(task_few_files)
        est_many = estimator.estimate_task(task_many_files)

        assert est_many["estimated_cost"] > est_few["estimated_cost"]

    def test_estimate_roadmap(self, sample_roadmap: MockRoadmap):
        """Test estimating entire roadmap."""
        estimator = CostEstimator()

        result = estimator.estimate_roadmap(sample_roadmap)

        assert "tasks" in result
        assert len(result["tasks"]) == 4
        assert result["task_count"] == 4
        assert result["total_tokens_in"] > 0
        assert result["total_tokens_out"] > 0
        assert result["total_cost"] > 0

    def test_different_models_different_costs(self):
        """Test that different models have different costs."""
        estimator = CostEstimator()
        task = MockTask(id="1.1.1", title="Add feature")

        est_sonnet = estimator.estimate_task(task, model="claude-sonnet-4-20250514")
        est_opus = estimator.estimate_task(task, model="claude-opus-4-20250514")

        assert est_opus["estimated_cost"] > est_sonnet["estimated_cost"]

    def test_get_model_pricing(self):
        """Test getting model pricing."""
        estimator = CostEstimator()

        pricing = estimator.get_model_pricing("claude-sonnet-4-20250514")
        assert "in" in pricing
        assert "out" in pricing

        # Unknown model should return default
        default_pricing = estimator.get_model_pricing("unknown-model")
        assert default_pricing == estimator.COST_PER_1K["default"]


# =============================================================================
# Dashboard Tests
# =============================================================================


class TestDashboard:
    """Tests for Dashboard class."""

    def test_create_dashboard(self, tracker: ProgressTracker, console: Console):
        """Test creating a dashboard."""
        dashboard = Dashboard(tracker, console=console)

        assert dashboard.tracker is tracker
        assert dashboard.console is console
        assert isinstance(dashboard.config, DashboardConfig)

    def test_create_dashboard_with_config(self, tracker: ProgressTracker):
        """Test creating dashboard with custom config."""
        config = DashboardConfig(
            show_all_tasks=True,
            max_visible_tasks=20,
            compact_mode=True,
        )
        dashboard = Dashboard(tracker, config=config)

        assert dashboard.config.show_all_tasks is True
        assert dashboard.config.max_visible_tasks == 20

    def test_create_panel(self, tracker: ProgressTracker, console: Console):
        """Test creating dashboard panel."""
        dashboard = Dashboard(tracker, console=console)
        panel = dashboard.create_panel()

        assert panel is not None
        assert panel.title is not None

    def test_render_does_not_raise(self, tracker: ProgressTracker, console: Console):
        """Test that render doesn't raise exceptions."""
        dashboard = Dashboard(tracker, console=console)

        # This should not raise
        dashboard.render()

        output = console.export_text()
        assert "Test Roadmap" in output

    def test_render_with_progress(self, tracker: ProgressTracker, console: Console):
        """Test rendering with some progress."""
        tracker.start_run()
        tracker.complete_task("1.1.1", tokens_in=1000, tokens_out=500, cost=0.05)
        tracker.start_task("1.1.2")

        dashboard = Dashboard(tracker, console=console)
        dashboard.render()

        output = console.export_text()
        # Should show progress
        assert "1" in output  # At least 1 task shown

    def test_create_summary_text(self, tracker: ProgressTracker):
        """Test creating plain text summary."""
        tracker.start_run()
        tracker.complete_task("1.1.1", tokens_in=100, tokens_out=50, cost=0.01)

        dashboard = Dashboard(tracker)
        summary = dashboard.create_summary_text()

        assert "Progress:" in summary
        assert "1/4" in summary
        assert "25%" in summary

    def test_format_duration(self, tracker: ProgressTracker):
        """Test duration formatting."""
        dashboard = Dashboard(tracker)

        assert dashboard._format_duration(30) == "30.0s"
        assert dashboard._format_duration(90) == "1m 30s"
        assert dashboard._format_duration(3661) == "1h 1m"


class TestStatusIcons:
    """Tests for status icons and labels."""

    def test_status_icons_defined(self):
        """Test that all status icons are defined."""
        expected = {"pending", "in_progress", "completed", "failed", "skipped"}
        assert set(STATUS_ICONS.keys()) == expected

    def test_status_labels_defined(self):
        """Test that all status labels are defined."""
        expected = {"pending", "in_progress", "completed", "failed", "skipped"}
        assert set(STATUS_LABELS.keys()) == expected


# =============================================================================
# Dashboard Runner Tests
# =============================================================================


class TestRunWithDashboard:
    """Tests for run_with_dashboard function."""

    @pytest.mark.asyncio
    async def test_run_with_dashboard_success(self, sample_tasks: list[MockTask], console: Console):
        """Test successful dashboard run."""
        tracker = ProgressTracker(roadmap_title="Test")
        for task in sample_tasks:
            tracker.add_task(task.id, task.title)

        async def mock_execute(task):
            await asyncio.sleep(0.01)
            return {
                "tokens_in": 100,
                "tokens_out": 50,
                "cost": 0.01,
                "files_modified": [],
                "files_created": [],
            }

        result = await run_with_dashboard(tracker, mock_execute, sample_tasks, console=console)

        assert result.success is True
        assert result.completed == 4
        assert result.failed == 0
        assert result.total_cost == pytest.approx(0.04, rel=0.1)
        assert result.total_tokens == 600  # 4 * 150

    @pytest.mark.asyncio
    async def test_run_with_dashboard_failures(
        self, sample_tasks: list[MockTask], console: Console
    ):
        """Test dashboard run with failures."""
        tracker = ProgressTracker(roadmap_title="Test")
        for task in sample_tasks:
            tracker.add_task(task.id, task.title)

        async def mock_execute(task):
            if task.id == "1.1.2":
                raise ValueError("Test error")
            return {"tokens_in": 100, "tokens_out": 50, "cost": 0.01}

        result = await run_with_dashboard(tracker, mock_execute, sample_tasks, console=console)

        assert result.success is False
        assert result.completed == 3
        assert result.failed == 1
        assert len(result.errors) == 1
        assert "1.1.2" in result.errors[0]

    @pytest.mark.asyncio
    async def test_run_with_simple_progress(self, sample_tasks: list[MockTask], console: Console):
        """Test simple progress bar runner."""
        tracker = ProgressTracker(roadmap_title="Test")
        for task in sample_tasks:
            tracker.add_task(task.id, task.title)

        async def mock_execute(task):
            return {"tokens_in": 100, "tokens_out": 50, "cost": 0.01}

        result = await run_with_simple_progress(
            tracker, mock_execute, sample_tasks, console=console
        )

        assert result.success is True
        assert result.completed == 4


# =============================================================================
# DashboardConfig Tests
# =============================================================================


class TestDashboardConfig:
    """Tests for DashboardConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DashboardConfig()

        assert config.show_all_tasks is False
        assert config.max_visible_tasks == 15
        assert config.show_cost_estimates is True
        assert config.show_file_changes is False
        assert config.show_tokens is True
        assert config.compact_mode is False
        assert config.refresh_rate == 0.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = DashboardConfig(
            show_all_tasks=True,
            max_visible_tasks=50,
            refresh_rate=1.0,
        )

        assert config.show_all_tasks is True
        assert config.max_visible_tasks == 50
        assert config.refresh_rate == 1.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for progress visibility components."""

    def test_full_workflow(self, sample_roadmap: MockRoadmap, console: Console):
        """Test complete tracking and dashboard workflow."""
        # Create tracker
        tracker = ProgressTracker(roadmap=sample_roadmap)
        estimator = CostEstimator()

        # Get estimates
        estimates = estimator.estimate_roadmap(sample_roadmap)
        assert estimates["total_cost"] > 0

        # Start execution
        tracker.start_run()

        # Execute first task
        tracker.start_task("1.1.1", agent="CoderAgent", model="claude-sonnet-4-20250514")
        tracker.complete_task("1.1.1", tokens_in=1000, tokens_out=500, cost=0.05)

        # Fail second task
        tracker.start_task("1.1.2")
        tracker.fail_task("1.1.2", error="Test failure")

        # Skip third task
        tracker.skip_task("1.2.1", reason="Blocked by failure")

        # Get summary
        summary = tracker.get_summary()
        assert summary.completed == 1
        assert summary.failed == 1
        assert summary.skipped == 1
        assert summary.pending == 1

        # Render dashboard
        dashboard = Dashboard(tracker, console=console)
        dashboard.render()

        output = console.export_text()
        assert len(output) > 0

    def test_cost_estimator_accuracy_for_different_tasks(self):
        """Test that estimator produces reasonable cost variations."""
        estimator = CostEstimator()

        simple_task = MockTask(id="1", title="Fix typo", description="Fix small typo")
        complex_task = MockTask(
            id="2",
            title="Complete database migration",
            description="Full refactor of authentication system",
        )

        simple_est = estimator.estimate_task(simple_task)
        complex_est = estimator.estimate_task(complex_task)

        # Complex task should cost more
        assert complex_est["estimated_cost"] > simple_est["estimated_cost"]
        assert complex_est["tokens_in"] > simple_est["tokens_in"]


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_tracker(self):
        """Test tracker with no tasks."""
        tracker = ProgressTracker()
        summary = tracker.get_summary()

        assert summary.total == 0
        assert summary.percent == 0.0

    def test_complete_unknown_task(self):
        """Test completing a task not in tracker."""
        tracker = ProgressTracker()
        # Should not raise
        tracker.complete_task("unknown")

    def test_fail_unknown_task(self):
        """Test failing a task not in tracker."""
        tracker = ProgressTracker()
        # Should not raise
        tracker.fail_task("unknown", "error")

    def test_dashboard_with_empty_tracker(self, console: Console):
        """Test dashboard with no tasks."""
        tracker = ProgressTracker(roadmap_title="Empty")
        dashboard = Dashboard(tracker, console=console)

        # Should not raise
        dashboard.render()

    def test_cost_estimator_empty_roadmap(self):
        """Test estimating empty roadmap."""
        estimator = CostEstimator()
        empty_roadmap = MockRoadmap(title="Empty", _tasks=[])

        result = estimator.estimate_roadmap(empty_roadmap)

        assert result["task_count"] == 0
        assert result["total_cost"] == 0.0
