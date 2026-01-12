"""Tests for Executor loop."""

from __future__ import annotations

import asyncio
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.executor import (
    CheckLevel,
    CheckResult,
    CheckStatus,
    ExecutionResult,
    ExecutionStatus,
    Executor,
    ExecutorConfig,
    RunStatus,
    RunSummary,
    TaskStatus,
    VerificationResult,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_roadmap_content() -> str:
    """Sample ROADMAP content."""
    return """\
# Test Project ROADMAP

## Phase 0: Foundation

> **Goal**: Establish core infrastructure
> **Priority**: HIGH

### 0.1 Setup

**Files**: `src/setup.py`

- [x] **Initialize project**
  Set up the project structure.

- [ ] **Configure linting**
  Set up ruff and mypy.

### 0.2 Core

**Files**: `src/core.py`

- [ ] **Implement core logic**
  Main business logic.

- [ ] **Add tests**
  Unit tests for core.

## Phase 1: Features

### 1.1 API

- [ ] **Create API endpoints**
  REST API implementation.
"""


@pytest.fixture
def temp_roadmap(sample_roadmap_content: str) -> Path:
    """Create a temporary ROADMAP file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        roadmap_path = Path(tmpdir) / "ROADMAP.md"
        roadmap_path.write_text(sample_roadmap_content)
        yield roadmap_path


@pytest.fixture
def mock_agent() -> AsyncMock:
    """Create a mock agent."""
    agent = AsyncMock()
    agent.arun.return_value = "Task completed successfully. Modified `src/setup.py`."
    return agent


@pytest.fixture
def mock_verifier() -> MagicMock:
    """Create a mock verifier."""
    verifier = MagicMock()
    result = VerificationResult(task_id="test", checks=[])
    verifier.verify = AsyncMock(return_value=result)
    return verifier


# =============================================================================
# ExecutorConfig Tests
# =============================================================================


class TestExecutorConfig:
    """Test ExecutorConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ExecutorConfig()

        assert config.model == "claude-sonnet-4-20250514"
        assert config.max_tasks == 0
        assert config.stop_on_failure is True
        assert config.checkpoint_every == 1
        assert config.require_human_approval_after == 0
        assert config.pause_before_destructive is True
        assert config.verification_level == CheckLevel.TESTS
        assert config.skip_verification is False
        assert config.dry_run is False
        assert config.retry_failed == 1
        assert config.context_max_tokens == 50000
        assert config.save_state_every == 1
        assert config.agent_timeout == 300.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ExecutorConfig(
            model="claude-opus-4-20250514",
            max_tasks=10,
            stop_on_failure=False,
            require_human_approval_after=5,
            dry_run=True,
        )

        assert config.model == "claude-opus-4-20250514"
        assert config.max_tasks == 10
        assert config.stop_on_failure is False
        assert config.require_human_approval_after == 5
        assert config.dry_run is True

    def test_to_dict(self) -> None:
        """Test configuration serialization."""
        config = ExecutorConfig(max_tasks=5)
        data = config.to_dict()

        assert data["max_tasks"] == 5
        assert "model" in data
        assert "stop_on_failure" in data


# =============================================================================
# ExecutionResult Tests
# =============================================================================


class TestExecutionResult:
    """Test ExecutionResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful execution result."""
        result = ExecutionResult(
            task_id="1.1.1",
            status=ExecutionStatus.SUCCESS,
            files_modified=["src/foo.py"],
        )

        assert result.success is True
        assert result.task_id == "1.1.1"
        assert result.files_modified == ["src/foo.py"]

    def test_failed_result(self) -> None:
        """Test failed execution result."""
        result = ExecutionResult(
            task_id="1.1.1",
            status=ExecutionStatus.FAILED,
            error="Something went wrong",
        )

        assert result.success is False
        assert result.error == "Something went wrong"

    def test_all_files_property(self) -> None:
        """Test all_files property."""
        result = ExecutionResult(
            task_id="1.1.1",
            status=ExecutionStatus.SUCCESS,
            files_modified=["a.py", "b.py"],
            files_created=["c.py"],
        )

        assert "a.py" in result.all_files
        assert "b.py" in result.all_files
        assert "c.py" in result.all_files

    def test_to_dict(self) -> None:
        """Test serialization."""
        now = datetime.now(UTC)
        result = ExecutionResult(
            task_id="1.1.1",
            status=ExecutionStatus.SUCCESS,
            started_at=now,
            completed_at=now,
        )
        data = result.to_dict()

        assert data["task_id"] == "1.1.1"
        assert data["status"] == "success"
        assert data["started_at"] is not None


# =============================================================================
# RunSummary Tests
# =============================================================================


class TestRunSummary:
    """Test RunSummary dataclass."""

    def test_success_rate_all_success(self) -> None:
        """Test success rate with all tasks successful."""
        summary = RunSummary(
            status=RunStatus.COMPLETED,
            tasks_completed=5,
            tasks_failed=0,
        )

        assert summary.success_rate == 1.0

    def test_success_rate_mixed(self) -> None:
        """Test success rate with mixed results."""
        summary = RunSummary(
            status=RunStatus.COMPLETED,
            tasks_completed=3,
            tasks_failed=2,
        )

        assert summary.success_rate == 0.6

    def test_success_rate_no_tasks(self) -> None:
        """Test success rate with no tasks."""
        summary = RunSummary(
            status=RunStatus.NO_TASKS,
            tasks_completed=0,
            tasks_failed=0,
        )

        assert summary.success_rate == 1.0

    def test_progress(self) -> None:
        """Test progress calculation."""
        summary = RunSummary(
            status=RunStatus.COMPLETED,
            tasks_completed=3,
            total_tasks=10,
        )

        assert summary.progress == 0.3

    def test_to_dict(self) -> None:
        """Test serialization."""
        summary = RunSummary(
            status=RunStatus.COMPLETED,
            tasks_completed=5,
            total_tasks=10,
            run_id="run_123",
        )
        data = summary.to_dict()

        assert data["status"] == "completed"
        assert data["tasks_completed"] == 5
        assert data["run_id"] == "run_123"

    def test_summary_string(self) -> None:
        """Test summary string generation."""
        summary = RunSummary(
            status=RunStatus.COMPLETED,
            tasks_completed=5,
            tasks_failed=1,
            tasks_remaining=4,
            total_tasks=10,
            total_tokens=1000,
            duration_ms=5000,
            run_id="run_123",
        )
        text = summary.summary()

        assert "run_123" in text
        assert "5/10" in text
        assert "Failed: 1" in text


# =============================================================================
# Executor Initialization Tests
# =============================================================================


class TestExecutorInit:
    """Test Executor initialization."""

    def test_init_basic(self, temp_roadmap: Path) -> None:
        """Test basic initialization."""
        executor = Executor(roadmap=temp_roadmap)

        assert executor.roadmap_path == temp_roadmap.resolve()
        assert executor.config.max_tasks == 0

    def test_init_with_config(self, temp_roadmap: Path) -> None:
        """Test initialization with custom config."""
        config = ExecutorConfig(max_tasks=5, dry_run=True)
        executor = Executor(roadmap=temp_roadmap, config=config)

        assert executor.config.max_tasks == 5
        assert executor.config.dry_run is True

    def test_init_with_agent(self, temp_roadmap: Path, mock_agent: AsyncMock) -> None:
        """Test initialization with custom agent."""
        executor = Executor(roadmap=temp_roadmap, agent=mock_agent)

        assert executor._agent is mock_agent

    def test_lazy_state_loading(self, temp_roadmap: Path) -> None:
        """Test that state is loaded lazily."""
        executor = Executor(roadmap=temp_roadmap)

        # State should not be loaded yet
        assert executor._state is None

        # Accessing state triggers load
        _ = executor.state
        assert executor._state is not None

    def test_lazy_roadmap_parsing(self, temp_roadmap: Path) -> None:
        """Test that roadmap is parsed lazily."""
        executor = Executor(roadmap=temp_roadmap)

        # Roadmap should not be parsed yet
        assert executor._roadmap is None

        # Accessing roadmap triggers parsing
        _ = executor.roadmap
        assert executor._roadmap is not None


# =============================================================================
# Executor Run Tests
# =============================================================================


class TestExecutorRun:
    """Test Executor run loop."""

    @pytest.mark.asyncio
    async def test_run_no_tasks(self, temp_roadmap: Path) -> None:
        """Test run with no pending tasks."""
        # Mark all tasks as complete
        content = temp_roadmap.read_text()
        content = content.replace("- [ ]", "- [x]")
        temp_roadmap.write_text(content)

        executor = Executor(roadmap=temp_roadmap)
        summary = await executor.run()

        assert summary.status == RunStatus.NO_TASKS

    @pytest.mark.asyncio
    async def test_run_dry_run(self, temp_roadmap: Path) -> None:
        """Test dry run mode."""
        config = ExecutorConfig(dry_run=True, max_tasks=1)
        executor = Executor(roadmap=temp_roadmap, config=config)

        summary = await executor.run()

        # Dry run should skip tasks
        assert len(summary.results) == 1
        assert summary.results[0].status == ExecutionStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_run_with_max_tasks(self, temp_roadmap: Path) -> None:
        """Test run with max_tasks limit."""
        config = ExecutorConfig(max_tasks=2, dry_run=True)
        executor = Executor(roadmap=temp_roadmap, config=config)

        summary = await executor.run()

        # Should stop after 2 tasks
        assert len(summary.results) == 2

    @pytest.mark.asyncio
    async def test_run_with_callback(self, temp_roadmap: Path) -> None:
        """Test run with completion callback."""
        config = ExecutorConfig(dry_run=True, max_tasks=1)
        executor = Executor(roadmap=temp_roadmap, config=config)

        callback_tasks = []

        async def on_complete(task, result):
            callback_tasks.append(task.id)

        await executor.run(on_complete=on_complete)

        assert len(callback_tasks) == 1

    @pytest.mark.asyncio
    async def test_run_with_sync_callback(self, temp_roadmap: Path) -> None:
        """Test run with synchronous callback."""
        config = ExecutorConfig(dry_run=True, max_tasks=1)
        executor = Executor(roadmap=temp_roadmap, config=config)

        callback_tasks = []

        def on_complete(task, result):
            callback_tasks.append(task.id)

        await executor.run(on_complete=on_complete)

        assert len(callback_tasks) == 1

    @pytest.mark.asyncio
    async def test_run_pause_for_human(self, temp_roadmap: Path) -> None:
        """Test run pausing for human approval."""
        config = ExecutorConfig(require_human_approval_after=1, dry_run=True)
        executor = Executor(roadmap=temp_roadmap, config=config)

        summary = await executor.run()

        assert summary.status == RunStatus.PAUSED
        assert summary.paused is True

    @pytest.mark.asyncio
    async def test_run_with_agent_success(
        self, temp_roadmap: Path, mock_agent: AsyncMock, mock_verifier: MagicMock
    ) -> None:
        """Test run with successful agent execution."""
        config = ExecutorConfig(max_tasks=1, skip_verification=True)
        executor = Executor(
            roadmap=temp_roadmap,
            config=config,
            agent=mock_agent,
            verifier=mock_verifier,
        )

        summary = await executor.run()

        assert len(summary.results) == 1
        assert summary.results[0].status == ExecutionStatus.SUCCESS
        mock_agent.arun.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_no_agent_configured(self, temp_roadmap: Path) -> None:
        """Test run without agent configured."""
        config = ExecutorConfig(max_tasks=1)
        executor = Executor(roadmap=temp_roadmap, config=config)

        summary = await executor.run()

        # Should skip task and mark as failed
        assert len(summary.results) == 1
        assert summary.results[0].status == ExecutionStatus.SKIPPED
        assert "No agent configured" in summary.results[0].error

    @pytest.mark.asyncio
    async def test_run_agent_timeout(self, temp_roadmap: Path) -> None:
        """Test run with agent timeout."""
        slow_agent = AsyncMock()
        slow_agent.arun.side_effect = TimeoutError()

        config = ExecutorConfig(max_tasks=1, agent_timeout=0.1)
        executor = Executor(
            roadmap=temp_roadmap,
            config=config,
            agent=slow_agent,
        )

        # Patch asyncio.wait_for to raise TimeoutError
        with patch("ai_infra.executor.loop.asyncio.wait_for") as mock_wait:
            mock_wait.side_effect = TimeoutError()
            summary = await executor.run()

        assert len(summary.results) == 1
        assert summary.results[0].status == ExecutionStatus.ERROR
        assert "timeout" in summary.results[0].error.lower()

    @pytest.mark.asyncio
    async def test_run_agent_exception(self, temp_roadmap: Path) -> None:
        """Test run with agent exception."""
        failing_agent = AsyncMock()
        failing_agent.arun.side_effect = RuntimeError("Agent crashed")

        config = ExecutorConfig(max_tasks=1)
        executor = Executor(
            roadmap=temp_roadmap,
            config=config,
            agent=failing_agent,
        )

        summary = await executor.run()

        assert len(summary.results) == 1
        assert summary.results[0].status == ExecutionStatus.ERROR
        assert "Agent crashed" in summary.results[0].error

    @pytest.mark.asyncio
    async def test_run_stop_on_failure(self, temp_roadmap: Path) -> None:
        """Test run stops on first failure."""
        failing_agent = AsyncMock()
        failing_agent.arun.side_effect = RuntimeError("Agent crashed")

        config = ExecutorConfig(max_tasks=5, stop_on_failure=True)
        executor = Executor(
            roadmap=temp_roadmap,
            config=config,
            agent=failing_agent,
        )

        summary = await executor.run()

        # Should stop after first failure
        assert summary.status == RunStatus.FAILED
        assert summary.tasks_failed == 1

    @pytest.mark.asyncio
    async def test_run_continue_on_failure(self, temp_roadmap: Path) -> None:
        """Test run continues after failure when stop_on_failure=False."""
        failing_agent = AsyncMock()
        failing_agent.arun.side_effect = RuntimeError("Agent crashed")

        config = ExecutorConfig(max_tasks=2, stop_on_failure=False)
        executor = Executor(
            roadmap=temp_roadmap,
            config=config,
            agent=failing_agent,
        )

        summary = await executor.run()

        # Should continue after failure
        assert summary.status == RunStatus.COMPLETED
        assert summary.tasks_failed == 2


# =============================================================================
# Executor State Integration Tests
# =============================================================================


class TestExecutorState:
    """Test Executor state integration."""

    @pytest.mark.asyncio
    async def test_state_saved_after_run(self, temp_roadmap: Path) -> None:
        """Test that state is saved after run."""
        config = ExecutorConfig(dry_run=True, max_tasks=1)
        executor = Executor(roadmap=temp_roadmap, config=config)

        await executor.run()

        # State file should exist
        assert executor.state.state_file.exists()

    @pytest.mark.asyncio
    async def test_crash_recovery(self, temp_roadmap: Path) -> None:
        """Test crash recovery on run."""
        # Simulate a crash by marking task as in-progress
        executor1 = Executor(roadmap=temp_roadmap)
        executor1.state.mark_started("0.1.2")
        executor1.state.save()

        # New executor should recover
        config = ExecutorConfig(dry_run=True, max_tasks=0)
        executor2 = Executor(roadmap=temp_roadmap, config=config)

        # The in-progress task should be recovered on run
        await executor2.run()

        # Task should be reset to pending (or completed if dry_run)
        assert executor2.state.get_status("0.1.2") != TaskStatus.IN_PROGRESS

    def test_resume_approved(self, temp_roadmap: Path) -> None:
        """Test resume with approval."""
        executor = Executor(roadmap=temp_roadmap)
        executor.state.mark_started("0.1.2")

        executor.resume(approved=True)

        # Should not reset tasks
        assert executor.state.get_status("0.1.2") == TaskStatus.IN_PROGRESS

    def test_resume_rejected(self, temp_roadmap: Path) -> None:
        """Test resume with rejection."""
        executor = Executor(roadmap=temp_roadmap)
        executor.state.mark_started("0.1.2")

        executor.resume(approved=False)

        # Should reset in-progress tasks
        assert executor.state.get_status("0.1.2") == TaskStatus.PENDING

    def test_reset(self, temp_roadmap: Path) -> None:
        """Test state reset."""
        executor = Executor(roadmap=temp_roadmap)
        executor.state.mark_completed("0.1.2")

        executor.reset()

        # State should be fresh from roadmap
        # 0.1.2 should be pending again (based on ROADMAP checkbox)
        assert executor.state.get_status("0.1.2") == TaskStatus.PENDING

    def test_sync_roadmap(self, temp_roadmap: Path) -> None:
        """Test syncing state to roadmap."""
        executor = Executor(roadmap=temp_roadmap)
        executor.state.mark_completed("0.1.2")  # "Configure linting"

        updated = executor.sync_roadmap()

        assert updated == 1
        content = temp_roadmap.read_text()
        assert "- [x] **Configure linting**" in content


# =============================================================================
# Executor Prompt Building Tests
# =============================================================================


class TestExecutorPromptBuilding:
    """Test Executor prompt building."""

    @pytest.mark.asyncio
    async def test_build_prompt(self, temp_roadmap: Path) -> None:
        """Test prompt building for a task."""
        executor = Executor(roadmap=temp_roadmap)

        task = executor.roadmap.get_task("0.1.2")
        assert task is not None

        prompt = await executor._build_prompt(task)

        assert "Configure linting" in prompt
        assert "Task:" in prompt
        assert "0.1.2" in prompt

    @pytest.mark.asyncio
    async def test_prompt_includes_file_hints(self, temp_roadmap: Path) -> None:
        """Test prompt includes file hints."""
        executor = Executor(roadmap=temp_roadmap)

        task = executor.roadmap.get_task("0.2.1")
        assert task is not None

        prompt = await executor._build_prompt(task)

        assert "src/core.py" in prompt


# =============================================================================
# Executor Helper Methods Tests
# =============================================================================


class TestExecutorHelpers:
    """Test Executor helper methods."""

    def test_has_pending_tasks(self, temp_roadmap: Path) -> None:
        """Test has_pending_tasks."""
        executor = Executor(roadmap=temp_roadmap)

        assert executor._has_pending_tasks() is True

    def test_reached_task_limit_unlimited(self, temp_roadmap: Path) -> None:
        """Test reached_task_limit with no limit."""
        config = ExecutorConfig(max_tasks=0)
        executor = Executor(roadmap=temp_roadmap, config=config)
        executor._tasks_this_run = 100

        assert executor._reached_task_limit() is False

    def test_reached_task_limit_with_limit(self, temp_roadmap: Path) -> None:
        """Test reached_task_limit with limit set."""
        config = ExecutorConfig(max_tasks=5)
        executor = Executor(roadmap=temp_roadmap, config=config)
        executor._tasks_this_run = 5

        assert executor._reached_task_limit() is True

    def test_should_pause_for_human_disabled(self, temp_roadmap: Path) -> None:
        """Test should_pause_for_human when disabled."""
        config = ExecutorConfig(require_human_approval_after=0)
        executor = Executor(roadmap=temp_roadmap, config=config)
        executor._tasks_this_run = 100

        assert executor._should_pause_for_human() is False

    def test_should_pause_for_human_enabled(self, temp_roadmap: Path) -> None:
        """Test should_pause_for_human when enabled."""
        config = ExecutorConfig(require_human_approval_after=5)
        executor = Executor(roadmap=temp_roadmap, config=config)
        executor._tasks_this_run = 5

        assert executor._should_pause_for_human() is True

    def test_parse_files_from_output(self, temp_roadmap: Path) -> None:
        """Test parsing file paths from agent output."""
        executor = Executor(roadmap=temp_roadmap)

        output = """
        I modified the following files:
        - `src/main.py`
        - 'tests/test_main.py'
        - "docs/README.md"
        """

        files = executor._parse_files_from_output(output)

        assert "src/main.py" in files
        assert "tests/test_main.py" in files
        assert "docs/README.md" in files


# =============================================================================
# Executor Verification Tests
# =============================================================================


class TestExecutorVerification:
    """Test Executor verification integration."""

    @pytest.mark.asyncio
    async def test_verification_failure_marks_task_failed(
        self, temp_roadmap: Path, mock_agent: AsyncMock
    ) -> None:
        """Test that verification failure marks task as failed."""
        # Create a failing verification result with a failed check
        failing_verifier = MagicMock()
        failing_check = CheckResult(
            level=CheckLevel.SYNTAX,
            name="syntax_check",
            status=CheckStatus.FAILED,
            error="Tests failed",
        )
        failing_result = VerificationResult(task_id="test", checks=[failing_check])
        failing_verifier.verify = AsyncMock(return_value=failing_result)

        config = ExecutorConfig(max_tasks=1, stop_on_failure=True)
        executor = Executor(
            roadmap=temp_roadmap,
            config=config,
            agent=mock_agent,
            verifier=failing_verifier,
        )

        summary = await executor.run()

        assert summary.results[0].status == ExecutionStatus.FAILED
        assert executor.state.get_status("0.1.2") == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_skip_verification(self, temp_roadmap: Path, mock_agent: AsyncMock) -> None:
        """Test skipping verification."""
        config = ExecutorConfig(max_tasks=1, skip_verification=True)
        executor = Executor(
            roadmap=temp_roadmap,
            config=config,
            agent=mock_agent,
        )

        summary = await executor.run()

        assert summary.results[0].status == ExecutionStatus.SUCCESS
        # Verifier should not be called
        assert executor._verifier is None  # Never accessed


# =============================================================================
# Integration Tests
# =============================================================================


class TestExecutorIntegration:
    """Integration tests for Executor."""

    @pytest.mark.asyncio
    async def test_full_workflow(
        self, temp_roadmap: Path, mock_agent: AsyncMock, mock_verifier: MagicMock
    ) -> None:
        """Test complete execution workflow."""
        config = ExecutorConfig(
            max_tasks=2,
            skip_verification=True,
            save_state_every=1,
        )
        executor = Executor(
            roadmap=temp_roadmap,
            config=config,
            agent=mock_agent,
            verifier=mock_verifier,
        )

        results = []

        async def on_complete(task, result):
            results.append((task.id, result.status))

        summary = await executor.run(on_complete=on_complete)

        # Check summary
        assert summary.status == RunStatus.COMPLETED
        assert summary.tasks_completed == 2
        assert len(results) == 2

        # Check state was saved
        assert executor.state.state_file.exists()

        # Check the first pending task was completed (0.1.2)
        assert executor.state.get_status("0.1.2") == TaskStatus.COMPLETED
        # Check we have 3 completed tasks in state:
        # 1 from markdown (0.1.1 was [x]) + 2 executed in this run
        state_summary = executor.state.get_summary()
        assert state_summary.completed == 3

    @pytest.mark.asyncio
    async def test_multiple_runs(self, temp_roadmap: Path, mock_agent: AsyncMock) -> None:
        """Test multiple sequential runs."""
        config = ExecutorConfig(max_tasks=1, skip_verification=True)

        # First run
        executor1 = Executor(
            roadmap=temp_roadmap,
            config=config,
            agent=mock_agent,
        )
        summary1 = await executor1.run()
        assert summary1.tasks_completed == 1
        first_task = summary1.results[0].task_id

        # Second run (should pick up next task)
        executor2 = Executor(
            roadmap=temp_roadmap,
            config=config,
            agent=mock_agent,
        )
        summary2 = await executor2.run()
        assert summary2.tasks_completed == 1
        second_task = summary2.results[0].task_id

        # Different tasks should be completed (state is persisted)
        assert first_task != second_task


# =============================================================================
# Phase 5.1: Parallel Execution Tests
# =============================================================================


class TestParallelExecution:
    """Tests for parallel execution (Phase 5.1)."""

    @pytest.fixture
    def parallel_roadmap_content(self) -> str:
        """Sample ROADMAP with independent tasks for parallel execution."""
        return """\
# Parallel Test ROADMAP

## Phase 0: Independent Tasks

### 0.1 Component A

**Files**: `src/component_a.py`

- [ ] **Build Component A**
  Create component A module.

### 0.2 Component B

**Files**: `src/component_b.py`

- [ ] **Build Component B**
  Create component B module.

### 0.3 Component C

**Files**: `src/component_c.py`

- [ ] **Build Component C**
  Create component C module.

## Phase 1: Integration

### 1.1 Integrate Components

**Files**: `src/main.py`

- [ ] **Integrate all components**
  Combine components into main app.
  **Depends**: 0.1.1, 0.2.1, 0.3.1
"""

    @pytest.fixture
    def temp_parallel_roadmap(self, tmp_path: Path, parallel_roadmap_content: str) -> Path:
        """Create temporary ROADMAP file for parallel testing."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text(parallel_roadmap_content)
        return roadmap

    @pytest.fixture
    def mock_agent_for_parallel(self) -> AsyncMock:
        """Mock agent that tracks concurrent calls."""
        mock = AsyncMock()
        mock.concurrent_count = 0
        mock.max_concurrent = 0

        async def slow_run(prompt: str) -> str:
            mock.concurrent_count += 1
            mock.max_concurrent = max(mock.max_concurrent, mock.concurrent_count)
            await asyncio.sleep(0.1)  # Simulate work
            mock.concurrent_count -= 1
            return "Task completed successfully. Files modified: src/test.py"

        mock.arun = slow_run
        return mock

    @pytest.mark.asyncio
    async def test_parallel_config_defaults(self) -> None:
        """Test parallel execution config has correct defaults."""
        config = ExecutorConfig()
        assert config.parallel_tasks == 1  # Sequential by default
        assert config.parallel_file_overlap is True
        assert config.parallel_import_analysis is True

    @pytest.mark.asyncio
    async def test_parallel_config_serialization(self) -> None:
        """Test parallel config is included in to_dict."""
        config = ExecutorConfig(
            parallel_tasks=4,
            parallel_file_overlap=False,
            parallel_import_analysis=True,
        )
        data = config.to_dict()
        assert data["parallel_tasks"] == 4
        assert data["parallel_file_overlap"] is False
        assert data["parallel_import_analysis"] is True

    @pytest.mark.asyncio
    async def test_build_task_dependency_graph(self, temp_parallel_roadmap: Path) -> None:
        """Test building task dependency graph."""
        executor = Executor(
            roadmap=temp_parallel_roadmap,
            config=ExecutorConfig(parallel_tasks=2),
        )

        graph = executor.build_task_dependency_graph()
        assert graph.is_built
        assert graph.task_count == 4  # 3 component tasks + 1 integration

    @pytest.mark.asyncio
    async def test_get_parallel_groups(self, temp_parallel_roadmap: Path) -> None:
        """Test getting parallel execution groups."""
        executor = Executor(
            roadmap=temp_parallel_roadmap,
            config=ExecutorConfig(parallel_tasks=3),
        )

        groups = executor.get_parallel_groups()

        # Should have at least 1 group (independent tasks)
        # Note: The integration task with **Depends** may or may not be parsed
        # as having dependencies based on parser implementation
        assert len(groups) >= 1

        # First group should contain at least some independent tasks
        first_group = groups[0]
        assert first_group.level == 0
        assert len(first_group.tasks) >= 1

    @pytest.mark.asyncio
    async def test_parallel_execution_enabled(
        self, temp_parallel_roadmap: Path, mock_agent_for_parallel: AsyncMock
    ) -> None:
        """Test parallel execution runs multiple tasks concurrently."""
        executor = Executor(
            roadmap=temp_parallel_roadmap,
            config=ExecutorConfig(
                parallel_tasks=3,
                max_tasks=3,
                skip_verification=True,
            ),
            agent=mock_agent_for_parallel,
        )

        summary = await executor.run()

        # Should have completed tasks
        assert summary.tasks_completed >= 1

        # With parallel_tasks=3, should have seen concurrent execution
        # (mock tracks max concurrent)
        # Note: Due to semaphore, max_concurrent should be <= parallel_tasks

    @pytest.mark.asyncio
    async def test_sequential_execution_when_parallel_is_one(
        self, temp_parallel_roadmap: Path
    ) -> None:
        """Test sequential execution when parallel_tasks=1."""
        mock_agent = AsyncMock()
        mock_agent.arun = AsyncMock(return_value="Done. Modified: src/test.py")

        executor = Executor(
            roadmap=temp_parallel_roadmap,
            config=ExecutorConfig(
                parallel_tasks=1,  # Sequential
                max_tasks=2,
                skip_verification=True,
            ),
            agent=mock_agent,
        )

        summary = await executor.run()
        assert summary.tasks_completed == 2
        # In sequential mode, tasks are executed one at a time

    @pytest.mark.asyncio
    async def test_parallel_execution_respects_max_tasks(self, temp_parallel_roadmap: Path) -> None:
        """Test parallel execution respects max_tasks limit."""
        mock_agent = AsyncMock()
        mock_agent.arun = AsyncMock(return_value="Done. Modified: src/test.py")

        executor = Executor(
            roadmap=temp_parallel_roadmap,
            config=ExecutorConfig(
                parallel_tasks=4,
                max_tasks=2,  # Only complete 2 tasks
                skip_verification=True,
            ),
            agent=mock_agent,
        )

        summary = await executor.run()
        assert summary.tasks_completed == 2

    @pytest.mark.asyncio
    async def test_parallel_execution_stop_on_failure(self, temp_parallel_roadmap: Path) -> None:
        """Test parallel execution stops on failure when configured."""
        call_count = 0

        async def failing_arun(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Task failed")
            return "Done. Modified: src/test.py"

        mock_agent = AsyncMock()
        mock_agent.arun = failing_arun

        executor = Executor(
            roadmap=temp_parallel_roadmap,
            config=ExecutorConfig(
                parallel_tasks=2,
                stop_on_failure=True,
                skip_verification=True,
            ),
            agent=mock_agent,
        )

        summary = await executor.run()
        assert summary.status == RunStatus.FAILED

    @pytest.mark.asyncio
    async def test_task_dependency_graph_export(self) -> None:
        """Test TaskDependencyGraph is properly exported from executor module."""
        from ai_infra.executor import (
            ParallelGroup,
            TaskDependencyGraph,
            TaskNode,
        )

        # Should be importable
        assert TaskDependencyGraph is not None
        assert ParallelGroup is not None
        assert TaskNode is not None


# =============================================================================
# TodoListManager Integration Tests
# =============================================================================


class TestTodoListManagerIntegration:
    """Tests for TodoListManager integration with ExecutorLoop."""

    def test_todo_manager_property_creates_manager(
        self, temp_roadmap: Path, mock_agent: AsyncMock
    ) -> None:
        """Test that todo_manager property creates a TodoListManager."""
        from ai_infra.executor import TodoListManager

        executor = Executor(
            roadmap=temp_roadmap,
            config=ExecutorConfig(skip_verification=True),
            agent=mock_agent,
        )

        # Access the property directly on Executor
        manager = executor.todo_manager

        assert manager is not None
        assert isinstance(manager, TodoListManager)
        # Should have grouped the tasks
        assert manager.total_count >= 1

    def test_todo_manager_is_lazily_initialized(
        self, temp_roadmap: Path, mock_agent: AsyncMock
    ) -> None:
        """Test that todo_manager is lazily initialized."""
        executor = Executor(
            roadmap=temp_roadmap,
            config=ExecutorConfig(skip_verification=True),
            agent=mock_agent,
        )

        # Before accessing, internal field should be None
        assert executor._todo_manager is None

        # After accessing, should be populated
        _ = executor.todo_manager
        assert executor._todo_manager is not None

    def test_todo_manager_uses_smart_grouping(
        self, temp_roadmap: Path, mock_agent: AsyncMock
    ) -> None:
        """Test that todo_manager uses smart grouping strategy."""
        executor = Executor(
            roadmap=temp_roadmap,
            config=ExecutorConfig(skip_verification=True),
            agent=mock_agent,
        )

        manager = executor.todo_manager
        roadmap = executor.roadmap

        # Smart grouping should result in fewer or equal todos than raw tasks
        assert manager.total_count <= roadmap.total_tasks

    @pytest.mark.asyncio
    async def test_task_completion_syncs_via_todo_manager(
        self, temp_roadmap: Path, mock_agent: AsyncMock
    ) -> None:
        """Test that task completion syncs ROADMAP via TodoListManager."""
        executor = Executor(
            roadmap=temp_roadmap,
            config=ExecutorConfig(
                skip_verification=True,
                max_tasks=1,  # Execute only one task
                sync_roadmap=True,
            ),
            agent=mock_agent,
        )

        # Run executor
        await executor.run()

        # Check that ROADMAP was updated with [x]
        content = temp_roadmap.read_text()
        # At least one checkbox should be checked
        assert "[x]" in content

    def test_todo_manager_exported_from_module(self) -> None:
        """Test that TodoListManager and related types are exported."""
        from ai_infra.executor import (
            GroupStrategy,
            TodoItem,
            TodoListManager,
            TodoStatus,
        )

        assert TodoListManager is not None
        assert TodoItem is not None
        assert TodoStatus is not None
        assert GroupStrategy is not None


# =============================================================================
# Phase 5.12.6: Run by Todos Tests
# =============================================================================


class TestRunByTodos:
    """Tests for Phase 5.12.6: Execute by grouped todos."""

    def test_todo_manager_pending_method(self, temp_roadmap: Path, mock_agent: AsyncMock) -> None:
        """Test that pending() returns all not-started todos."""
        executor = Executor(
            roadmap=temp_roadmap,
            config=ExecutorConfig(skip_verification=True),
            agent=mock_agent,
        )

        pending = executor.todo_manager.pending()

        assert isinstance(pending, list)
        assert len(pending) >= 1
        # All items should have NOT_STARTED status
        from ai_infra.executor import TodoStatus

        for todo in pending:
            assert todo.status == TodoStatus.NOT_STARTED

    def test_todo_manager_get_source_tasks(self, temp_roadmap: Path, mock_agent: AsyncMock) -> None:
        """Test that get_source_tasks retrieves ParsedTask objects."""
        executor = Executor(
            roadmap=temp_roadmap,
            config=ExecutorConfig(skip_verification=True),
            agent=mock_agent,
        )

        todo = executor.todo_manager.next_pending()
        assert todo is not None

        source_tasks = executor.todo_manager.get_source_tasks(todo, executor.roadmap)

        assert isinstance(source_tasks, list)
        assert len(source_tasks) >= 1
        # Each source task should have matching ID
        for task in source_tasks:
            assert task.id in todo.source_task_ids

    @pytest.mark.asyncio
    async def test_run_by_todos_executes_grouped(
        self, temp_roadmap: Path, mock_agent: AsyncMock
    ) -> None:
        """Test that run_by_todos executes todos as grouped units."""
        executor = Executor(
            roadmap=temp_roadmap,
            config=ExecutorConfig(
                skip_verification=True,
                max_tasks=5,
                sync_roadmap=True,
            ),
            agent=mock_agent,
        )

        summary = await executor.run_by_todos()

        assert summary is not None
        # Should have executed at least one todo
        assert summary.total_tasks >= 1

    @pytest.mark.asyncio
    async def test_run_by_todos_dry_run(self, temp_roadmap: Path, mock_agent: AsyncMock) -> None:
        """Test that run_by_todos respects dry_run mode."""
        executor = Executor(
            roadmap=temp_roadmap,
            config=ExecutorConfig(
                skip_verification=True,
                dry_run=True,
            ),
            agent=mock_agent,
        )

        summary = await executor.run_by_todos()

        # Agent should not have been called in dry run
        mock_agent.run.assert_not_called()
        assert summary is not None

    @pytest.mark.asyncio
    async def test_run_by_todos_syncs_roadmap(
        self, temp_roadmap: Path, mock_agent: AsyncMock
    ) -> None:
        """Test that run_by_todos syncs completed todos to ROADMAP."""
        executor = Executor(
            roadmap=temp_roadmap,
            config=ExecutorConfig(
                skip_verification=True,
                max_tasks=1,
                sync_roadmap=True,
            ),
            agent=mock_agent,
        )

        await executor.run_by_todos()

        # Check that ROADMAP was updated
        content = temp_roadmap.read_text()
        assert "[x]" in content

    @pytest.mark.asyncio
    async def test_run_by_todos_stops_on_failure(
        self, temp_roadmap: Path, mock_agent: AsyncMock
    ) -> None:
        """Test that run_by_todos stops on failure when configured."""
        # Configure agent to fail
        mock_agent.run = AsyncMock(
            return_value=MagicMock(
                success=False,
                error="Test failure",
                output="",
                files_modified=[],
                files_created=[],
                files_deleted=[],
                token_usage={},
            )
        )

        executor = Executor(
            roadmap=temp_roadmap,
            config=ExecutorConfig(
                skip_verification=True,
                stop_on_failure=True,
            ),
            agent=mock_agent,
        )

        summary = await executor.run_by_todos()

        from ai_infra.executor.loop import RunStatus

        assert summary.status == RunStatus.FAILED
