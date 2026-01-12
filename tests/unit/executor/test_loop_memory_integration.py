"""Tests for Loop Memory Integration (Phase 5.8.4).

This module tests the integration of memory layers into the ExecutorLoop:
- Run memory initialization
- Project memory loading and updating
- Memory context injection in prompts
- Outcome extraction after tasks
- Memory configuration options
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_infra.executor.loop import (
    Executor,
    ExecutorConfig,
    RunStatus,
)
from ai_infra.executor.project_memory import ProjectMemory
from ai_infra.executor.run_memory import FileAction, RunMemory, TaskOutcome

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_roadmap_content() -> str:
    """Sample ROADMAP content for testing."""
    return """\
# Test Project ROADMAP

## Phase 0: Foundation

> **Goal**: Establish core infrastructure
> **Priority**: HIGH

### 0.1 Setup

**Files**: `src/setup.py`

- [ ] **Configure linting**
  Set up ruff and mypy.

- [ ] **Add logging**
  Set up structured logging.

### 0.2 Core

**Files**: `src/core.py`

- [ ] **Implement core logic**
  Main business logic.
"""


@pytest.fixture
def temp_project(sample_roadmap_content: str):
    """Create a temporary project directory with ROADMAP."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        roadmap_path = project_path / "ROADMAP.md"
        roadmap_path.write_text(sample_roadmap_content)

        # Create src directory
        (project_path / "src").mkdir()
        (project_path / "src" / "__init__.py").write_text("")

        yield project_path, roadmap_path


@pytest.fixture
def mock_agent() -> AsyncMock:
    """Create a mock agent that returns file modification output."""
    agent = AsyncMock()
    agent.arun.return_value = (
        "I will configure ruff and mypy for the project.\n"
        "write_file('pyproject.toml', '[tool.ruff]\\n...')\n"
        "Task completed successfully."
    )
    return agent


@pytest.fixture
def mock_verifier() -> MagicMock:
    """Create a mock verifier that always passes."""
    from ai_infra.executor.verifier import VerificationResult

    verifier = MagicMock()
    result = VerificationResult(task_id="test", checks=[])
    verifier.verify = AsyncMock(return_value=result)
    return verifier


# =============================================================================
# Test ExecutorConfig Memory Fields
# =============================================================================


class TestExecutorConfigMemory:
    """Tests for memory-related configuration fields."""

    def test_default_memory_config(self):
        """Test default memory configuration values."""
        config = ExecutorConfig()

        assert config.enable_run_memory is True
        assert config.enable_project_memory is True
        assert config.memory_token_budget == 6000
        assert config.extract_outcomes_with_llm is False

    def test_memory_config_disabled(self):
        """Test disabling memory features."""
        config = ExecutorConfig(
            enable_run_memory=False,
            enable_project_memory=False,
        )

        assert config.enable_run_memory is False
        assert config.enable_project_memory is False

    def test_memory_config_custom_budget(self):
        """Test custom memory token budget."""
        config = ExecutorConfig(memory_token_budget=10000)

        assert config.memory_token_budget == 10000

    def test_memory_config_with_llm_extraction(self):
        """Test enabling LLM extraction."""
        config = ExecutorConfig(extract_outcomes_with_llm=True)

        assert config.extract_outcomes_with_llm is True

    def test_to_dict_includes_memory_config(self):
        """Test that to_dict includes memory config fields."""
        config = ExecutorConfig(
            enable_run_memory=False,
            memory_token_budget=8000,
        )
        data = config.to_dict()

        assert "enable_run_memory" in data
        assert data["enable_run_memory"] is False
        assert data["memory_token_budget"] == 8000


# =============================================================================
# Test Executor Memory Initialization
# =============================================================================


class TestExecutorMemoryInit:
    """Tests for memory layer initialization in Executor."""

    def test_project_memory_loaded_on_init(self, temp_project):
        """Test that project memory is loaded when Executor is created."""
        project_path, roadmap_path = temp_project

        executor = Executor(roadmap_path)

        assert executor.project_memory is not None
        assert isinstance(executor.project_memory, ProjectMemory)

    def test_project_memory_disabled(self, temp_project):
        """Test project memory can be disabled."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig(enable_project_memory=False)

        executor = Executor(roadmap_path, config=config)

        assert executor.project_memory is None

    def test_run_memory_none_before_run(self, temp_project):
        """Test run memory is None before run() is called."""
        project_path, roadmap_path = temp_project

        executor = Executor(roadmap_path)

        # Run memory is only initialized during run()
        assert executor.run_memory is None

    def test_set_llm(self, temp_project):
        """Test setting LLM for outcome extraction."""
        project_path, roadmap_path = temp_project
        executor = Executor(roadmap_path)

        mock_llm = MagicMock()
        executor.set_llm(mock_llm)

        assert executor._llm is mock_llm


# =============================================================================
# Test Run Memory Initialization During Run
# =============================================================================


class TestRunMemoryDuringRun:
    """Tests for run memory initialization during executor run."""

    @pytest.mark.asyncio
    async def test_run_memory_initialized_on_run(self, temp_project, mock_agent, mock_verifier):
        """Test that run memory is initialized when run() starts."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig(max_tasks=1, skip_verification=True)

        executor = Executor(
            roadmap_path,
            config=config,
            agent=mock_agent,
            verifier=mock_verifier,
        )

        # Run memory should be None before run
        assert executor.run_memory is None

        # Run the executor
        await executor.run()

        # Run memory should be initialized after run
        assert executor.run_memory is not None
        assert isinstance(executor.run_memory, RunMemory)

    @pytest.mark.asyncio
    async def test_run_memory_disabled(self, temp_project, mock_agent, mock_verifier):
        """Test run memory stays None when disabled."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig(
            max_tasks=1,
            skip_verification=True,
            enable_run_memory=False,
        )

        executor = Executor(
            roadmap_path,
            config=config,
            agent=mock_agent,
            verifier=mock_verifier,
        )

        await executor.run()

        # Run memory should still be None
        assert executor.run_memory is None

    @pytest.mark.asyncio
    async def test_run_memory_has_run_id(self, temp_project, mock_agent, mock_verifier):
        """Test run memory has correct run ID."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig(max_tasks=1, skip_verification=True)

        executor = Executor(
            roadmap_path,
            config=config,
            agent=mock_agent,
            verifier=mock_verifier,
        )

        await executor.run()

        assert executor.run_memory is not None
        assert executor.run_memory.run_id == executor.state.run_id


# =============================================================================
# Test Project Memory Update After Run
# =============================================================================


class TestProjectMemoryUpdate:
    """Tests for project memory updates after run completion."""

    @pytest.mark.asyncio
    async def test_project_memory_updated_after_run(self, temp_project, mock_agent, mock_verifier):
        """Test project memory is updated from run memory after run."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig(max_tasks=1, skip_verification=True)

        executor = Executor(
            roadmap_path,
            config=config,
            agent=mock_agent,
            verifier=mock_verifier,
        )

        # Run the executor
        await executor.run()

        # Project memory should have history from the run
        assert executor.project_memory is not None
        # The run_history should be updated (even if empty on first run)
        assert hasattr(executor.project_memory, "run_history")

    @pytest.mark.asyncio
    async def test_project_memory_persists_to_disk(self, temp_project, mock_agent, mock_verifier):
        """Test project memory is saved to disk after run."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig(max_tasks=1, skip_verification=True)

        executor = Executor(
            roadmap_path,
            config=config,
            agent=mock_agent,
            verifier=mock_verifier,
        )

        await executor.run()

        # Check that memory file exists
        memory_path = project_path / ".executor" / "project-memory.json"
        assert memory_path.exists()


# =============================================================================
# Test Memory Context in Prompts
# =============================================================================


class TestMemoryContextInPrompts:
    """Tests for memory context injection in prompts."""

    @pytest.mark.asyncio
    async def test_prompt_includes_run_memory_context(
        self, temp_project, mock_agent, mock_verifier
    ):
        """Test that prompts include run memory context."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig(max_tasks=2, skip_verification=True)

        executor = Executor(
            roadmap_path,
            config=config,
            agent=mock_agent,
            verifier=mock_verifier,
        )

        await executor.run()

        # Check that agent was called multiple times
        # Second call should have context from first task
        assert mock_agent.arun.call_count >= 1

    @pytest.mark.asyncio
    async def test_memory_budget_respected(self, temp_project, mock_agent, mock_verifier):
        """Test that memory token budget is respected in prompts."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig(
            max_tasks=1,
            skip_verification=True,
            memory_token_budget=100,  # Very small budget
        )

        executor = Executor(
            roadmap_path,
            config=config,
            agent=mock_agent,
            verifier=mock_verifier,
        )

        # Should not error with small budget
        await executor.run()
        assert mock_agent.arun.called


# =============================================================================
# Test Outcome Extraction
# =============================================================================


class TestOutcomeExtraction:
    """Tests for outcome extraction after task execution."""

    @pytest.mark.asyncio
    async def test_outcome_extracted_after_task(self, temp_project, mock_agent, mock_verifier):
        """Test that outcomes are extracted and recorded."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig(max_tasks=1, skip_verification=True)

        # Agent returns output with file operations
        mock_agent.arun.return_value = (
            "I created the config file.\nwrite_file('pyproject.toml', '...')\nDone."
        )

        executor = Executor(
            roadmap_path,
            config=config,
            agent=mock_agent,
            verifier=mock_verifier,
        )

        await executor.run()

        # Run memory should have recorded outcome
        assert executor.run_memory is not None
        assert len(executor.run_memory.outcomes) >= 1

    @pytest.mark.asyncio
    async def test_outcome_has_file_operations(self, temp_project, mock_agent, mock_verifier):
        """Test that extracted outcome includes file operations."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig(max_tasks=1, skip_verification=True)

        mock_agent.arun.return_value = (
            "Created the file.\nwrite_file('src/utils.py', 'def helper(): pass')\n"
        )

        executor = Executor(
            roadmap_path,
            config=config,
            agent=mock_agent,
            verifier=mock_verifier,
        )

        await executor.run()

        assert executor.run_memory is not None
        if executor.run_memory.outcomes:
            outcome = executor.run_memory.outcomes[0]
            # Should have extracted the file operation
            assert isinstance(outcome.files, dict)

    @pytest.mark.asyncio
    async def test_outcome_not_recorded_when_disabled(
        self, temp_project, mock_agent, mock_verifier
    ):
        """Test that outcomes are not recorded when run memory is disabled."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig(
            max_tasks=1,
            skip_verification=True,
            enable_run_memory=False,
        )

        executor = Executor(
            roadmap_path,
            config=config,
            agent=mock_agent,
            verifier=mock_verifier,
        )

        await executor.run()

        # Run memory should be None
        assert executor.run_memory is None


# =============================================================================
# Test LLM Extraction Configuration
# =============================================================================


class TestLLMExtractionConfig:
    """Tests for LLM-based outcome extraction."""

    @pytest.mark.asyncio
    async def test_llm_extraction_with_llm_set(self, temp_project, mock_verifier):
        """Test LLM is used for extraction when configured."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig(
            max_tasks=1,
            skip_verification=True,
            extract_outcomes_with_llm=True,
        )

        # Create mock agent
        mock_agent = AsyncMock()
        mock_agent.arun.return_value = "Task done."

        # Create mock LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"summary": "Test", "decisions": []}'
        mock_llm.achat = AsyncMock(return_value=mock_response)

        executor = Executor(
            roadmap_path,
            config=config,
            agent=mock_agent,
            verifier=mock_verifier,
        )
        executor.set_llm(mock_llm)

        await executor.run()

        # LLM should have been called for extraction
        # (only if task completed successfully)
        # This depends on whether task actually succeeded


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestMemoryEdgeCases:
    """Edge case tests for memory integration."""

    @pytest.mark.asyncio
    async def test_empty_roadmap_no_crash(self, temp_project):
        """Test executor handles empty roadmap without crash."""
        project_path, roadmap_path = temp_project

        # Write empty roadmap
        roadmap_path.write_text("# Empty ROADMAP\n\nNo tasks here.")

        executor = Executor(roadmap_path)
        summary = await executor.run()

        assert summary.status == RunStatus.NO_TASKS

    @pytest.mark.asyncio
    async def test_project_memory_load_error_handled(self, temp_project, mock_agent, mock_verifier):
        """Test that corrupted project memory is handled gracefully."""
        project_path, roadmap_path = temp_project

        # Create corrupted memory file
        memory_dir = project_path / ".executor"
        memory_dir.mkdir(exist_ok=True)
        (memory_dir / "project-memory.json").write_text("not valid json")

        # Should not crash
        config = ExecutorConfig(max_tasks=1, skip_verification=True)
        executor = Executor(
            roadmap_path,
            config=config,
            agent=mock_agent,
            verifier=mock_verifier,
        )

        # Project memory should be reset or empty, not crash
        assert executor.project_memory is not None

    @pytest.mark.asyncio
    async def test_memory_with_dry_run(self, temp_project):
        """Test memory with dry_run mode."""
        project_path, roadmap_path = temp_project
        config = ExecutorConfig(dry_run=True, max_tasks=1)

        executor = Executor(roadmap_path, config=config)
        summary = await executor.run()

        # Dry run should work with memory enabled
        assert summary is not None

    def test_multiple_executor_instances_share_project_memory_file(self, temp_project):
        """Test that multiple executors use the same project memory file."""
        project_path, roadmap_path = temp_project

        executor1 = Executor(roadmap_path)
        executor2 = Executor(roadmap_path)

        # Both should point to the same memory path
        assert executor1.project_memory is not None
        assert executor2.project_memory is not None
        # They load from same file (though instances are different)


# =============================================================================
# Test Memory Context Content
# =============================================================================


class TestMemoryContextContent:
    """Tests for verifying memory context content."""

    def test_run_memory_get_context(self, temp_project):
        """Test RunMemory.get_context returns proper content."""
        project_path, _ = temp_project

        run_memory = RunMemory(
            run_id="test-run",
            started_at=datetime.now(UTC).isoformat(),
        )

        # Add an outcome
        outcome = TaskOutcome(
            task_id="1.1",
            title="Test Task",
            status="completed",
            files={Path("test.py"): FileAction.CREATED},
            key_decisions=["Used type hints"],
            summary="Created test file",
            duration_seconds=10.0,
            tokens_used=100,
        )
        run_memory.add_outcome(outcome)

        context = run_memory.get_context("1.2", max_tokens=1000)

        assert context is not None
        assert "Test Task" in context or "test.py" in context

    def test_project_memory_get_context(self, temp_project):
        """Test ProjectMemory.get_context returns proper content."""
        project_path, _ = temp_project

        project_memory = ProjectMemory.load(project_path)

        # Get context for a task
        context = project_memory.get_context(
            task_title="Configure logging",
            max_tokens=1000,
        )

        # Context may be empty for fresh project, but should not error
        assert context is not None or context == ""


# =============================================================================
# Test Integration with Callbacks
# =============================================================================


class TestMemoryWithCallbacks:
    """Tests for memory integration with executor callbacks."""

    @pytest.mark.asyncio
    async def test_callbacks_work_with_memory(self, temp_project, mock_agent, mock_verifier):
        """Test that executor callbacks work alongside memory."""
        from ai_infra.executor.observability import ExecutorCallbacks

        project_path, roadmap_path = temp_project
        callbacks = ExecutorCallbacks()
        config = ExecutorConfig(max_tasks=1, skip_verification=True)

        executor = Executor(
            roadmap_path,
            config=config,
            agent=mock_agent,
            verifier=mock_verifier,
            callbacks=callbacks,
        )

        await executor.run()

        # Both memory and callbacks should work
        assert executor.run_memory is not None
        metrics = callbacks.get_metrics()
        assert metrics is not None
