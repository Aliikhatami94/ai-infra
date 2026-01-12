"""Phase 5.6: Retry Loop Integration Tests.

These tests verify the retry loop with adaptive planning integration:
- Retry on verification failure
- Auto-fix applied before retry
- Retry exhausted after max attempts
- Retry with enriched context
- SUGGEST mode behavior (no auto-retry)
- NO_ADAPT mode behavior (immediate failure)
- Parallel execution with retry
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_infra.executor.adaptive import AdaptiveMode
from ai_infra.executor.loop import (
    Executor,
    ExecutorConfig,
)
from ai_infra.executor.observability import ExecutorCallbacks
from ai_infra.executor.testing import MockAgent, TestProject
from ai_infra.executor.verifier import (
    CheckLevel,
    CheckResult,
    CheckStatus,
    VerificationResult,
)

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
def project_with_failing_task(test_project: TestProject) -> TestProject:
    """Create a project with a task that will fail verification."""
    test_project.add_file("src/__init__.py", "")
    test_project.create_roadmap(
        [
            "Create module with syntax error",
        ]
    )
    return test_project


@pytest.fixture
def project_with_multiple_tasks(test_project: TestProject) -> TestProject:
    """Create a project with multiple tasks."""
    test_project.add_file("src/__init__.py", "")
    test_project.create_roadmap(
        [
            "Create utils module",
            "Add helper function",
            "Write unit tests",
        ]
    )
    return test_project


def create_failing_verifier(fail_count: int = 1) -> MagicMock:
    """Create a verifier that fails the first N times, then passes.

    Args:
        fail_count: Number of times to fail before passing.

    Returns:
        Mock verifier with controlled failure behavior.
    """
    call_count = [0]

    async def verify_side_effect(task, levels=None):
        call_count[0] += 1
        if call_count[0] <= fail_count:
            return VerificationResult(
                task_id=task.id,
                levels_run=[CheckLevel.SYNTAX],
                checks=[
                    CheckResult(
                        name="syntax_check",
                        level=CheckLevel.SYNTAX,
                        status=CheckStatus.FAILED,
                        error=f"Syntax error on attempt {call_count[0]}",
                    ),
                ],
            )
        return VerificationResult(
            task_id=task.id,
            levels_run=[CheckLevel.SYNTAX],
            checks=[
                CheckResult(
                    name="syntax_check",
                    level=CheckLevel.SYNTAX,
                    status=CheckStatus.PASSED,
                ),
            ],
        )

    verifier = MagicMock()
    verifier.verify = AsyncMock(side_effect=verify_side_effect)
    return verifier


def create_always_passing_verifier() -> MagicMock:
    """Create a verifier that always passes."""
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


def create_always_failing_verifier() -> MagicMock:
    """Create a verifier that always fails."""
    verifier = MagicMock()
    verifier.verify = AsyncMock(
        return_value=VerificationResult(
            task_id="test-task",
            levels_run=[CheckLevel.SYNTAX],
            checks=[
                CheckResult(
                    name="syntax_check",
                    level=CheckLevel.SYNTAX,
                    status=CheckStatus.FAILED,
                    error="Persistent syntax error",
                ),
            ],
        )
    )
    return verifier


# =============================================================================
# Retry on Verification Failure Tests
# =============================================================================


class TestRetryOnVerificationFailure:
    """Test retry behavior when verification fails."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(
        self, project_with_failing_task: TestProject, mock_agent: MockAgent
    ) -> None:
        """Task succeeds on second attempt after verification fails once."""
        roadmap_path = project_with_failing_task.root / "ROADMAP.md"

        # Verifier fails first time, passes second time
        verifier = create_failing_verifier(fail_count=1)

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1, retry_failed=2),
            agent=mock_agent,
            verifier=verifier,
        )

        # Set to AUTO_FIX mode
        executor.plan_analyzer.mode = AdaptiveMode.AUTO_FIX

        summary = await executor.run()

        # Should succeed after retry
        assert summary.tasks_completed == 1
        assert summary.tasks_failed == 0

        # Verify verifier was called twice (initial + 1 retry)
        assert verifier.verify.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_fails_after_exhausting_attempts(
        self, project_with_failing_task: TestProject, mock_agent: MockAgent
    ) -> None:
        """Task fails after exhausting all retry attempts."""
        roadmap_path = project_with_failing_task.root / "ROADMAP.md"

        # Verifier always fails
        verifier = create_always_failing_verifier()

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1, retry_failed=3),
            agent=mock_agent,
            verifier=verifier,
        )

        # Set to AUTO_FIX mode
        executor.plan_analyzer.mode = AdaptiveMode.AUTO_FIX

        summary = await executor.run()

        # Should fail after all retries exhausted
        assert summary.tasks_completed == 0
        assert summary.tasks_failed == 1

        # Verify verifier was called 3 times (initial + 2 retries)
        assert verifier.verify.call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_with_retry_failed_1(
        self, project_with_failing_task: TestProject, mock_agent: MockAgent
    ) -> None:
        """No retry when retry_failed=1 (single attempt)."""
        roadmap_path = project_with_failing_task.root / "ROADMAP.md"

        # Verifier always fails
        verifier = create_always_failing_verifier()

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1, retry_failed=1),
            agent=mock_agent,
            verifier=verifier,
        )
        executor.plan_analyzer.mode = AdaptiveMode.AUTO_FIX

        summary = await executor.run()

        # Should fail without retry
        assert summary.tasks_failed == 1

        # Verifier called only once
        assert verifier.verify.call_count == 1


# =============================================================================
# Adaptive Mode Behavior Tests
# =============================================================================


class TestAdaptiveModeBehavior:
    """Test behavior differs based on adaptive mode."""

    @pytest.mark.asyncio
    async def test_no_adapt_mode_no_retry(
        self, project_with_failing_task: TestProject, mock_agent: MockAgent
    ) -> None:
        """NO_ADAPT mode: failure is immediate, no retry."""
        roadmap_path = project_with_failing_task.root / "ROADMAP.md"

        # Verifier fails first, passes second - but retry shouldn't happen
        verifier = create_failing_verifier(fail_count=1)

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1, retry_failed=5),
            agent=mock_agent,
            verifier=verifier,
        )

        # Set to NO_ADAPT mode - no retry should occur
        executor.plan_analyzer.mode = AdaptiveMode.NO_ADAPT

        summary = await executor.run()

        # Should fail immediately without retry
        assert summary.tasks_failed == 1

        # Verifier called only once
        assert verifier.verify.call_count == 1

    @pytest.mark.asyncio
    async def test_suggest_mode_no_auto_retry(
        self, project_with_failing_task: TestProject, mock_agent: MockAgent
    ) -> None:
        """SUGGEST mode: generates suggestions but doesn't auto-retry."""
        roadmap_path = project_with_failing_task.root / "ROADMAP.md"

        # Verifier fails first, passes second - but retry shouldn't happen
        verifier = create_failing_verifier(fail_count=1)

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1, retry_failed=5),
            agent=mock_agent,
            verifier=verifier,
        )

        # Set to SUGGEST mode - no auto-retry
        executor.plan_analyzer.mode = AdaptiveMode.SUGGEST

        summary = await executor.run()

        # Should fail without auto-retry
        assert summary.tasks_failed == 1

        # Verifier called only once
        assert verifier.verify.call_count == 1

    @pytest.mark.asyncio
    async def test_auto_fix_mode_retries(
        self, project_with_failing_task: TestProject, mock_agent: MockAgent
    ) -> None:
        """AUTO_FIX mode: retries on failure."""
        roadmap_path = project_with_failing_task.root / "ROADMAP.md"

        # Verifier fails first, passes second
        verifier = create_failing_verifier(fail_count=1)

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1, retry_failed=3),
            agent=mock_agent,
            verifier=verifier,
        )

        # Set to AUTO_FIX mode - retry should occur
        executor.plan_analyzer.mode = AdaptiveMode.AUTO_FIX

        summary = await executor.run()

        # Should succeed on retry
        assert summary.tasks_completed == 1
        assert summary.tasks_failed == 0

        # Verifier called twice
        assert verifier.verify.call_count == 2


# =============================================================================
# Retry Context Tests
# =============================================================================


class TestRetryContext:
    """Test retry context is properly built and passed."""

    @pytest.mark.asyncio
    async def test_retry_context_includes_previous_error(
        self, project_with_failing_task: TestProject
    ) -> None:
        """Retry attempt includes previous error in context."""
        roadmap_path = project_with_failing_task.root / "ROADMAP.md"

        # Track prompts received
        received_prompts: list[str] = []

        class CapturingAgent:
            async def arun(self, prompt: str) -> str:
                received_prompts.append(prompt)
                return "Task completed."

        verifier = create_failing_verifier(fail_count=1)

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1, retry_failed=2),
            agent=CapturingAgent(),
            verifier=verifier,
        )
        executor.plan_analyzer.mode = AdaptiveMode.AUTO_FIX

        await executor.run()

        # Should have received 2 prompts
        assert len(received_prompts) == 2

        # First prompt should not have retry context
        assert "Retry Attempt" not in received_prompts[0]

        # Second prompt should have retry context with error
        # Phase 5.7: Format changed to "## Retry Attempt N"
        assert "Retry Attempt" in received_prompts[1]
        assert "Syntax error on attempt 1" in received_prompts[1]

    @pytest.mark.asyncio
    async def test_retry_context_includes_attempt_number(
        self, project_with_failing_task: TestProject
    ) -> None:
        """Retry context includes attempt number."""
        roadmap_path = project_with_failing_task.root / "ROADMAP.md"

        received_prompts: list[str] = []

        class CapturingAgent:
            async def arun(self, prompt: str) -> str:
                received_prompts.append(prompt)
                return "Task completed."

        # Fail twice to test attempt numbering
        verifier = create_failing_verifier(fail_count=2)

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1, retry_failed=3),
            agent=CapturingAgent(),
            verifier=verifier,
        )
        executor.plan_analyzer.mode = AdaptiveMode.AUTO_FIX

        await executor.run()

        # Should have 3 prompts
        assert len(received_prompts) == 3

        # Second prompt: attempt 2
        # Phase 5.7: Format changed to "## Retry Attempt N"
        assert "Retry Attempt 2" in received_prompts[1]

        # Third prompt: attempt 3
        assert "Retry Attempt 3" in received_prompts[2]


# =============================================================================
# Callback Integration Tests
# =============================================================================


class TestRetryCallbacks:
    """Test retry-related callbacks are properly invoked."""

    @pytest.mark.asyncio
    async def test_on_task_retry_callback(
        self, project_with_failing_task: TestProject, mock_agent: MockAgent
    ) -> None:
        """on_task_retry callback is invoked on retry."""
        roadmap_path = project_with_failing_task.root / "ROADMAP.md"

        callbacks = ExecutorCallbacks()
        callbacks.on_task_retry = MagicMock()

        verifier = create_failing_verifier(fail_count=1)

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1, retry_failed=2),
            agent=mock_agent,
            callbacks=callbacks,
            verifier=verifier,
        )
        executor.plan_analyzer.mode = AdaptiveMode.AUTO_FIX

        await executor.run()

        # Callback should be called once (for the retry)
        callbacks.on_task_retry.assert_called_once()

        # Check callback arguments
        call_args = callbacks.on_task_retry.call_args
        assert call_args.kwargs["attempt"] == 2
        assert call_args.kwargs["max_attempts"] == 2
        assert "Syntax error" in call_args.kwargs["previous_error"]

    @pytest.mark.asyncio
    async def test_on_retry_success_callback(
        self, project_with_failing_task: TestProject, mock_agent: MockAgent
    ) -> None:
        """on_retry_success callback is invoked when retry succeeds."""
        roadmap_path = project_with_failing_task.root / "ROADMAP.md"

        callbacks = ExecutorCallbacks()
        callbacks.on_retry_success = MagicMock()

        verifier = create_failing_verifier(fail_count=1)

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1, retry_failed=2),
            agent=mock_agent,
            callbacks=callbacks,
            verifier=verifier,
        )
        executor.plan_analyzer.mode = AdaptiveMode.AUTO_FIX

        await executor.run()

        # Callback should be called once (success on attempt 2)
        callbacks.on_retry_success.assert_called_once()

        call_args = callbacks.on_retry_success.call_args
        assert call_args.args[1] == 2  # attempt number

    @pytest.mark.asyncio
    async def test_on_retries_exhausted_callback(
        self, project_with_failing_task: TestProject, mock_agent: MockAgent
    ) -> None:
        """on_retries_exhausted callback is invoked when all retries fail."""
        roadmap_path = project_with_failing_task.root / "ROADMAP.md"

        callbacks = ExecutorCallbacks()
        callbacks.on_retries_exhausted = MagicMock()

        verifier = create_always_failing_verifier()

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1, retry_failed=3),
            agent=mock_agent,
            callbacks=callbacks,
            verifier=verifier,
        )
        executor.plan_analyzer.mode = AdaptiveMode.AUTO_FIX

        await executor.run()

        # Callback should be called once
        callbacks.on_retries_exhausted.assert_called_once()

        call_args = callbacks.on_retries_exhausted.call_args
        assert call_args.kwargs["attempts"] == 3


# =============================================================================
# Metrics Integration Tests
# =============================================================================


class TestRetryMetrics:
    """Test retry metrics are properly tracked."""

    @pytest.mark.asyncio
    async def test_retry_metrics_tracked(
        self, project_with_failing_task: TestProject, mock_agent: MockAgent
    ) -> None:
        """Retry metrics are tracked in ExecutorMetrics."""
        roadmap_path = project_with_failing_task.root / "ROADMAP.md"

        callbacks = ExecutorCallbacks()
        verifier = create_failing_verifier(fail_count=2)

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1, retry_failed=3),
            agent=mock_agent,
            callbacks=callbacks,
            verifier=verifier,
        )
        executor.plan_analyzer.mode = AdaptiveMode.AUTO_FIX

        await executor.run()

        # Check metrics
        metrics = callbacks.get_metrics()
        assert metrics.task_retries == 2  # Two retries
        assert metrics.retry_successes == 1  # Succeeded on attempt 3

    @pytest.mark.asyncio
    async def test_retry_metrics_to_dict(
        self, project_with_failing_task: TestProject, mock_agent: MockAgent
    ) -> None:
        """Retry metrics appear in to_dict output."""
        roadmap_path = project_with_failing_task.root / "ROADMAP.md"

        callbacks = ExecutorCallbacks()
        verifier = create_failing_verifier(fail_count=1)

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1, retry_failed=2),
            agent=mock_agent,
            callbacks=callbacks,
            verifier=verifier,
        )
        executor.plan_analyzer.mode = AdaptiveMode.AUTO_FIX

        await executor.run()

        metrics_dict = callbacks.get_metrics().to_dict()
        assert "retry" in metrics_dict
        assert metrics_dict["retry"]["task_retries"] == 1
        assert metrics_dict["retry"]["retry_successes"] == 1


# =============================================================================
# Parallel Execution with Retry Tests
# =============================================================================


class TestParallelExecutionWithRetry:
    """Test retry works correctly in parallel execution mode."""

    @pytest.mark.asyncio
    async def test_parallel_tasks_retry_independently(
        self, project_with_multiple_tasks: TestProject, mock_agent: MockAgent
    ) -> None:
        """Each parallel task retries independently."""
        roadmap_path = project_with_multiple_tasks.root / "ROADMAP.md"

        # Create verifier that fails specific tasks
        call_count = {}

        async def selective_verify(task, levels=None):
            task_id = task.id
            call_count[task_id] = call_count.get(task_id, 0) + 1

            # Fail first task on first attempt only
            if "utils" in task_id.lower() and call_count[task_id] == 1:
                return VerificationResult(
                    task_id=task_id,
                    levels_run=[CheckLevel.SYNTAX],
                    checks=[
                        CheckResult(
                            name="syntax_check",
                            level=CheckLevel.SYNTAX,
                            status=CheckStatus.FAILED,
                            error="Utils syntax error",
                        ),
                    ],
                )
            return VerificationResult(
                task_id=task_id,
                levels_run=[CheckLevel.SYNTAX],
                checks=[
                    CheckResult(
                        name="syntax_check",
                        level=CheckLevel.SYNTAX,
                        status=CheckStatus.PASSED,
                    ),
                ],
            )

        verifier = MagicMock()
        verifier.verify = AsyncMock(side_effect=selective_verify)

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(
                max_tasks=3,
                retry_failed=2,
                parallel_tasks=2,
            ),
            agent=mock_agent,
            verifier=verifier,
        )
        executor.plan_analyzer.mode = AdaptiveMode.AUTO_FIX

        summary = await executor.run()

        # All tasks should succeed (one after retry)
        assert summary.tasks_completed == 3
        assert summary.tasks_failed == 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestRetryEdgeCases:
    """Test edge cases in retry behavior."""

    @pytest.mark.asyncio
    async def test_skip_verification_no_retry(
        self, project_with_failing_task: TestProject, mock_agent: MockAgent
    ) -> None:
        """When verification is skipped, retry is not triggered."""
        roadmap_path = project_with_failing_task.root / "ROADMAP.md"

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(
                max_tasks=1,
                retry_failed=3,
                skip_verification=True,  # Skip verification
            ),
            agent=mock_agent,
        )
        executor.plan_analyzer.mode = AdaptiveMode.AUTO_FIX

        summary = await executor.run()

        # Should succeed without verification
        assert summary.tasks_completed == 1

    @pytest.mark.asyncio
    async def test_dry_run_no_retry(
        self, project_with_failing_task: TestProject, mock_agent: MockAgent
    ) -> None:
        """Dry run doesn't trigger retry logic."""
        roadmap_path = project_with_failing_task.root / "ROADMAP.md"

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(
                max_tasks=1,
                retry_failed=3,
                dry_run=True,
            ),
            agent=mock_agent,
        )

        summary = await executor.run()

        # Task should be skipped (dry run)
        assert summary.tasks_skipped == 1

    @pytest.mark.asyncio
    async def test_timeout_triggers_retry(self, project_with_failing_task: TestProject) -> None:
        """Agent timeout triggers retry in AUTO_FIX mode."""
        roadmap_path = project_with_failing_task.root / "ROADMAP.md"

        timeout_count = [0]

        class TimeoutAgent:
            async def arun(self, prompt: str) -> str:
                timeout_count[0] += 1
                if timeout_count[0] == 1:
                    # First call times out
                    raise TimeoutError("Agent timeout")
                return "Task completed."

        verifier = create_always_passing_verifier()

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(
                max_tasks=1,
                retry_failed=2,
                agent_timeout=1.0,
            ),
            agent=TimeoutAgent(),
            verifier=verifier,
        )
        executor.plan_analyzer.mode = AdaptiveMode.AUTO_FIX

        summary = await executor.run()

        # Should succeed on retry
        assert summary.tasks_completed == 1
        assert timeout_count[0] == 2


# =============================================================================
# Build Retry Context Tests
# =============================================================================


class TestBuildRetryContext:
    """Test _build_retry_context method."""

    def test_context_format(self, test_project: TestProject) -> None:
        """Retry context has expected format."""
        roadmap_path = test_project.root / "ROADMAP.md"
        test_project.create_roadmap(["Test task"])

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1),
        )

        # Access internal method for testing
        from ai_infra.executor.roadmap import ParsedTask

        task = ParsedTask(
            id="test-1",
            title="Test Task",
            phase_id="1",
            section_id="1.1",
        )

        context = executor._build_retry_context(
            task=task,
            previous_error="ModuleNotFoundError: No module named 'foo'",
            attempt=2,
        )

        # Check context content - Phase 5.7 format
        assert "Retry Attempt 2" in context
        assert "ModuleNotFoundError" in context
        assert "root cause" in context.lower()  # Language-agnostic instructions
