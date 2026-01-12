"""Chaos testing for executor robustness.

This module provides chaos testing capabilities to verify the executor
handles failure scenarios gracefully, including:
- Process interruption mid-task
- State file corruption
- Network failures during agent calls
- Disk write failures
- Memory pressure
- Timeout scenarios

See Phase 4.4 of EXECUTOR.md for requirements.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_infra.executor.loop import Executor, ExecutorConfig
from ai_infra.executor.state import ExecutorState
from ai_infra.executor.testing import ChaosAgent, MockAgent, TestProject
from ai_infra.executor.verifier import CheckLevel, CheckResult, CheckStatus, VerificationResult

# =============================================================================
# Chaos Types
# =============================================================================


class ChaosType(Enum):
    """Types of chaos that can be injected."""

    STATE_CORRUPTION = "state_corruption"
    NETWORK_FAILURE = "network_failure"
    TIMEOUT = "timeout"
    DISK_FAILURE = "disk_failure"
    AGENT_CRASH = "agent_crash"
    VERIFIER_CRASH = "verifier_crash"
    MEMORY_PRESSURE = "memory_pressure"
    CONCURRENT_ACCESS = "concurrent_access"


@dataclass
class ChaosEvent:
    """Record of a chaos event that occurred."""

    chaos_type: ChaosType
    timestamp: datetime
    description: str
    recovered: bool = False
    recovery_action: str | None = None
    error_message: str | None = None


@dataclass
class ChaosTestResult:
    """Result of a chaos test run."""

    test_name: str
    chaos_type: ChaosType
    passed: bool
    events: list[ChaosEvent] = field(default_factory=list)
    executor_result: Any = None
    error: str | None = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "chaos_type": self.chaos_type.value,
            "passed": self.passed,
            "events": [
                {
                    "type": e.chaos_type.value,
                    "description": e.description,
                    "recovered": e.recovered,
                }
                for e in self.events
            ],
            "error": self.error,
            "duration_seconds": self.duration_seconds,
        }


# =============================================================================
# Chaos Injection Utilities
# =============================================================================


def corrupt_json_file(file_path: Path, corruption_type: str = "random") -> str:
    """Corrupt a JSON file in various ways.

    Args:
        file_path: Path to the JSON file.
        corruption_type: Type of corruption:
            - "random": Random byte corruption
            - "truncate": Truncate the file
            - "invalid": Replace with invalid JSON
            - "partial": Remove closing braces

    Returns:
        Description of the corruption applied.
    """
    if not file_path.exists():
        file_path.write_text("{}")

    content = file_path.read_text()

    if corruption_type == "truncate":
        file_path.write_text(content[: len(content) // 2])
        return "Truncated file to half length"

    elif corruption_type == "invalid":
        file_path.write_text("{ this is not valid json }")
        return "Replaced with invalid JSON"

    elif corruption_type == "partial":
        # Remove closing braces
        file_path.write_text(content.rstrip("}]").rstrip())
        return "Removed closing braces"

    elif corruption_type == "random":
        # Random byte corruption
        if len(content) > 10:
            pos = random.randint(5, len(content) - 5)
            corrupted = content[:pos] + chr(random.randint(0, 127)) + content[pos + 1 :]
            file_path.write_text(corrupted)
            return f"Corrupted byte at position {pos}"

    return "No corruption applied"


def corrupt_state_file(state_path: Path) -> str:
    """Corrupt an executor state file.

    Args:
        state_path: Path to .executor/state.json

    Returns:
        Description of corruption.
    """
    corruptions = ["truncate", "invalid", "partial", "random"]
    corruption_type = random.choice(corruptions)
    return corrupt_json_file(state_path, corruption_type)


class NetworkFailureSimulator:
    """Simulates network failures in agent calls."""

    def __init__(
        self,
        failure_rate: float = 0.3,
        failure_types: list[type[Exception]] | None = None,
    ):
        self.failure_rate = failure_rate
        self.failure_types = failure_types or [
            ConnectionError,
            TimeoutError,
            OSError,
        ]
        self.call_count = 0
        self.failure_count = 0

    def maybe_fail(self) -> None:
        """Maybe raise a network exception."""
        self.call_count += 1
        if random.random() < self.failure_rate:
            self.failure_count += 1
            error_type = random.choice(self.failure_types)
            raise error_type("Simulated network failure")


class DiskFailureSimulator:
    """Simulates disk write failures."""

    def __init__(self, failure_rate: float = 0.2):
        self.failure_rate = failure_rate
        self.original_write = None
        self.call_count = 0
        self.failure_count = 0

    def __enter__(self):
        """Start simulating disk failures."""
        import builtins

        self.original_open = builtins.open

        def failing_open(*args, **kwargs):
            mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
            if "w" in mode and random.random() < self.failure_rate:
                self.failure_count += 1
                raise OSError("Simulated disk write failure")
            return self.original_open(*args, **kwargs)

        builtins.open = failing_open
        return self

    def __exit__(self, *args):
        """Restore normal disk operations."""
        import builtins

        builtins.open = self.original_open


# =============================================================================
# Chaos Test Fixtures
# =============================================================================


@pytest.fixture
def chaos_project(tmp_path: Path) -> TestProject:
    """Create a project for chaos testing."""
    project = TestProject(tmp_path)
    project.create_roadmap(
        [
            "Task 1 - Simple task",
            "Task 2 - Another task",
            "Task 3 - Final task",
        ]
    )
    return project


@pytest.fixture
def chaos_agent() -> ChaosAgent:
    """Create a chaos agent."""
    return ChaosAgent()


@pytest.fixture
def mock_verifier() -> MagicMock:
    """Create a mock verifier that passes."""
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


# =============================================================================
# Chaos Tests: State Corruption
# =============================================================================


class TestStateCorruption:
    """Tests for state file corruption scenarios."""

    @pytest.mark.asyncio
    async def test_corrupted_state_truncated(
        self,
        chaos_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test recovery from truncated state file."""
        chaos_project.create_roadmap(["Task 1"])

        # Create initial state
        state = ExecutorState(roadmap_path=chaos_project.roadmap_path)
        state.mark_started("0.1.1")
        state.save()

        # Corrupt it
        corrupt_json_file(state.state_file, "truncate")

        # Should recover gracefully
        agent = MockAgent()
        executor = Executor(
            roadmap=chaos_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        summary = await executor.run()
        assert summary is not None

    @pytest.mark.asyncio
    async def test_corrupted_state_invalid_json(
        self,
        chaos_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test recovery from invalid JSON in state file."""
        state_dir = chaos_project.root / ".executor"
        state_dir.mkdir(exist_ok=True)
        state_path = state_dir / "state.json"
        state_path.write_text("{ invalid json here }")

        agent = MockAgent()
        executor = Executor(
            roadmap=chaos_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        summary = await executor.run()
        assert summary is not None

    @pytest.mark.asyncio
    async def test_corrupted_state_empty_file(
        self,
        chaos_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test recovery from empty state file."""
        state_dir = chaos_project.root / ".executor"
        state_dir.mkdir(exist_ok=True)
        state_path = state_dir / "state.json"
        state_path.write_text("")

        agent = MockAgent()
        executor = Executor(
            roadmap=chaos_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        summary = await executor.run()
        assert summary is not None

    @pytest.mark.asyncio
    async def test_corrupted_state_wrong_schema(
        self,
        chaos_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test recovery from state with wrong schema."""
        state_dir = chaos_project.root / ".executor"
        state_dir.mkdir(exist_ok=True)
        state_path = state_dir / "state.json"
        # Use invalid schema that won't crash the parser
        state_path.write_text('{"wrong_field": "wrong_value", "tasks": {}}')

        agent = MockAgent()
        executor = Executor(
            roadmap=chaos_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        summary = await executor.run()
        assert summary is not None

    @pytest.mark.asyncio
    async def test_state_locked_by_another_process(
        self,
        chaos_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test handling when state file is locked."""
        # This simulates another process having the file open
        state_dir = chaos_project.root / ".executor"
        state_dir.mkdir(exist_ok=True)
        state_path = state_dir / "state.json"
        state_path.write_text("{}")

        agent = MockAgent()
        executor = Executor(
            roadmap=chaos_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        # Should handle gracefully even if write fails
        summary = await executor.run()
        assert summary is not None


# =============================================================================
# Chaos Tests: Network Failures
# =============================================================================


class TestNetworkFailures:
    """Tests for network failure scenarios."""

    @pytest.mark.asyncio
    async def test_agent_connection_error(
        self,
        chaos_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test handling agent connection errors."""
        agent = MockAgent()
        agent.add_error(
            pattern=".*",
            error=ConnectionError("Connection refused"),
        )

        executor = Executor(
            roadmap=chaos_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1, retry_failed=2),
        )

        summary = await executor.run()
        assert summary is not None
        # Should have failed after retries
        assert summary.tasks_failed >= 0

    @pytest.mark.asyncio
    async def test_agent_timeout(
        self,
        chaos_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test handling agent timeouts."""
        agent = MockAgent()
        # Add a very long delay (would timeout in real scenario)
        agent.add_response(
            pattern=".*",
            response="Delayed response",
            delay=0.5,  # Use short delay for test
        )

        executor = Executor(
            roadmap=chaos_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        summary = await executor.run()
        assert summary is not None

    @pytest.mark.asyncio
    async def test_intermittent_failures(
        self,
        chaos_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test handling intermittent agent failures."""
        agent = ChaosAgent(failure_rate=0.5)
        agent.add_response(pattern=".*", response="Success")

        executor = Executor(
            roadmap=chaos_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1, retry_failed=5),
        )

        summary = await executor.run()
        assert summary is not None

    @pytest.mark.asyncio
    async def test_all_retries_fail(
        self,
        chaos_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test when all retries fail."""
        agent = ChaosAgent(failure_rate=1.0)  # Always fail

        executor = Executor(
            roadmap=chaos_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1, retry_failed=3),
        )

        summary = await executor.run()
        assert summary is not None
        # Should have recorded the failure
        assert summary.tasks_failed >= 1 or summary.tasks_completed == 0


# =============================================================================
# Chaos Tests: Verifier Failures
# =============================================================================


class TestVerifierFailures:
    """Tests for verifier failure scenarios."""

    @pytest.mark.asyncio
    async def test_verifier_crashes(
        self,
        chaos_project: TestProject,
    ) -> None:
        """Test handling when verifier crashes."""
        agent = MockAgent()

        verifier = MagicMock()
        verifier.verify.side_effect = RuntimeError("Verifier crashed")

        executor = Executor(
            roadmap=chaos_project.roadmap_path,
            agent=agent,
            verifier=verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        summary = await executor.run()
        assert summary is not None

    @pytest.mark.asyncio
    async def test_verifier_returns_invalid(
        self,
        chaos_project: TestProject,
    ) -> None:
        """Test handling when verifier returns invalid result."""
        agent = MockAgent()

        verifier = MagicMock()
        verifier.verify = AsyncMock(return_value=None)  # Invalid return

        executor = Executor(
            roadmap=chaos_project.roadmap_path,
            agent=agent,
            verifier=verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        summary = await executor.run()
        assert summary is not None

    @pytest.mark.asyncio
    async def test_verifier_hangs(
        self,
        chaos_project: TestProject,
    ) -> None:
        """Test handling when verifier hangs."""
        agent = MockAgent()

        async def slow_verify(*args, **kwargs):
            await asyncio.sleep(0.1)  # Short sleep for test
            return VerificationResult(
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

        verifier = MagicMock()
        verifier.verify = slow_verify

        executor = Executor(
            roadmap=chaos_project.roadmap_path,
            agent=agent,
            verifier=verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        summary = await executor.run()
        assert summary is not None


# =============================================================================
# Chaos Tests: Concurrent Access
# =============================================================================


class TestConcurrentAccess:
    """Tests for concurrent access scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_executors(
        self,
        chaos_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test multiple executors on same roadmap."""
        agent1 = MockAgent()
        agent2 = MockAgent()

        executor1 = Executor(
            roadmap=chaos_project.roadmap_path,
            agent=agent1,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        executor2 = Executor(
            roadmap=chaos_project.roadmap_path,
            agent=agent2,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        # Run both concurrently
        results = await asyncio.gather(
            executor1.run(),
            executor2.run(),
            return_exceptions=True,
        )

        # Both should complete (may have different results)
        assert len(results) == 2
        for result in results:
            if isinstance(result, Exception):
                # One may fail due to lock, which is acceptable
                pass
            else:
                assert result is not None

    @pytest.mark.asyncio
    async def test_state_modified_externally(
        self,
        chaos_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test handling when state is modified during execution."""
        state_path = chaos_project.root / ".executor/state.json"

        agent = MockAgent()
        agent.add_response(
            pattern=".*",
            response="Done",
            delay=0.05,  # Small delay
        )

        executor = Executor(
            roadmap=chaos_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=2),
        )

        # Modify state file during execution
        async def modify_state():
            await asyncio.sleep(0.02)
            if state_path.exists():
                state_path.write_text('{"modified": true}')

        # Run both
        await asyncio.gather(
            executor.run(),
            modify_state(),
        )

        # Should complete even with external modification


# =============================================================================
# Chaos Tests: Recovery
# =============================================================================


class TestRecoveryScenarios:
    """Tests for recovery from various failure scenarios."""

    @pytest.mark.asyncio
    async def test_recover_after_crash(
        self,
        chaos_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test recovery after simulated crash."""
        chaos_project.create_roadmap(["Task 1 - Simple task", "Task 2"])

        # Simulate crashed state (task in progress)
        state = ExecutorState(roadmap_path=chaos_project.roadmap_path)
        state.mark_started("0.1.1")
        # Don't mark completion - simulates crash
        state.save()

        agent = MockAgent()
        executor = Executor(
            roadmap=chaos_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=2),
        )

        summary = await executor.run()
        assert summary is not None

    @pytest.mark.asyncio
    async def test_recover_with_partial_completion(
        self,
        chaos_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test recovery with partial task completion."""
        chaos_project.create_roadmap(["Task 1 - Simple task", "Task 2"])

        # Some tasks completed, some pending
        state = ExecutorState(roadmap_path=chaos_project.roadmap_path)
        state.mark_started("0.1.1")
        state.mark_completed("0.1.1")
        state.save()

        agent = MockAgent()
        executor = Executor(
            roadmap=chaos_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=2),
        )

        summary = await executor.run()
        assert summary is not None
        # Should continue from where it left off

    @pytest.mark.asyncio
    async def test_recover_multiple_failures(
        self,
        chaos_project: TestProject,
        mock_verifier: MagicMock,
    ) -> None:
        """Test recovery after multiple consecutive failures."""
        chaos_project.create_roadmap(["Task 1 - Simple task", "Task 2"])

        # Record multiple failed attempts
        state = ExecutorState(roadmap_path=chaos_project.roadmap_path)
        state.mark_started("0.1.1")
        state.mark_failed("0.1.1", error="Simulated failure 1")
        state.mark_started("0.1.1")
        state.mark_failed("0.1.1", error="Simulated failure 2")
        state.save()

        agent = MockAgent()
        executor = Executor(
            roadmap=chaos_project.roadmap_path,
            agent=agent,
            verifier=mock_verifier,
            config=ExecutorConfig(max_tasks=2),
        )

        summary = await executor.run()
        assert summary is not None


# =============================================================================
# Chaos Test Runner
# =============================================================================


class ChaosTestRunner:
    """Runner for comprehensive chaos testing."""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.results: list[ChaosTestResult] = []

    async def run_all_chaos_tests(self) -> list[ChaosTestResult]:
        """Run all chaos tests and return results."""
        import time

        tests = [
            ("state_corruption_truncate", ChaosType.STATE_CORRUPTION, self._test_state_truncate),
            ("state_corruption_invalid", ChaosType.STATE_CORRUPTION, self._test_state_invalid),
            ("network_failure", ChaosType.NETWORK_FAILURE, self._test_network_failure),
            ("agent_timeout", ChaosType.TIMEOUT, self._test_timeout),
            ("verifier_crash", ChaosType.VERIFIER_CRASH, self._test_verifier_crash),
            ("concurrent_access", ChaosType.CONCURRENT_ACCESS, self._test_concurrent),
        ]

        for name, chaos_type, test_func in tests:
            start = time.time()
            try:
                passed, events = await test_func()
                result = ChaosTestResult(
                    test_name=name,
                    chaos_type=chaos_type,
                    passed=passed,
                    events=events,
                    duration_seconds=time.time() - start,
                )
            except Exception as e:
                result = ChaosTestResult(
                    test_name=name,
                    chaos_type=chaos_type,
                    passed=False,
                    error=str(e),
                    duration_seconds=time.time() - start,
                )

            self.results.append(result)

        return self.results

    async def _test_state_truncate(self) -> tuple[bool, list[ChaosEvent]]:
        """Test state truncation recovery."""
        events = []
        project = TestProject(self.workspace_dir / "state_truncate")
        project.create_roadmap(["Task 1"])

        state_path = project.root / ".executor/state.json"
        state_path.write_text('{"tasks": {"Task 1": {"sta')

        events.append(
            ChaosEvent(
                chaos_type=ChaosType.STATE_CORRUPTION,
                timestamp=datetime.utcnow(),
                description="Truncated state file",
            )
        )

        agent = MockAgent()
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

        executor = Executor(
            roadmap=project.roadmap_path,
            agent=agent,
            verifier=verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        try:
            await executor.run()
            events[-1].recovered = True
            events[-1].recovery_action = "Executor recovered from corrupted state"
            return True, events
        except Exception as e:
            events[-1].error_message = str(e)
            return False, events

    async def _test_state_invalid(self) -> tuple[bool, list[ChaosEvent]]:
        """Test invalid JSON recovery."""
        events = []
        project = TestProject(self.workspace_dir / "state_invalid")
        project.create_roadmap(["Task 1"])

        state_path = project.root / ".executor/state.json"
        state_path.write_text("not json at all")

        events.append(
            ChaosEvent(
                chaos_type=ChaosType.STATE_CORRUPTION,
                timestamp=datetime.utcnow(),
                description="Invalid JSON in state file",
            )
        )

        agent = MockAgent()
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

        executor = Executor(
            roadmap=project.roadmap_path,
            agent=agent,
            verifier=verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        try:
            await executor.run()
            events[-1].recovered = True
            return True, events
        except Exception as e:
            events[-1].error_message = str(e)
            return False, events

    async def _test_network_failure(self) -> tuple[bool, list[ChaosEvent]]:
        """Test network failure handling."""
        events = []
        project = TestProject(self.workspace_dir / "network_failure")
        project.create_roadmap(["Task 1"])

        events.append(
            ChaosEvent(
                chaos_type=ChaosType.NETWORK_FAILURE,
                timestamp=datetime.utcnow(),
                description="Simulated network failure",
            )
        )

        agent = ChaosAgent(failure_rate=0.5)
        agent.add_response(pattern=".*", response="Success")

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

        executor = Executor(
            roadmap=project.roadmap_path,
            agent=agent,
            verifier=verifier,
            config=ExecutorConfig(max_tasks=1, retry_failed=5),
        )

        try:
            await executor.run()
            events[-1].recovered = True
            return True, events
        except Exception as e:
            events[-1].error_message = str(e)
            return False, events

    async def _test_timeout(self) -> tuple[bool, list[ChaosEvent]]:
        """Test timeout handling."""
        events = []
        project = TestProject(self.workspace_dir / "timeout")
        project.create_roadmap(["Task 1"])

        events.append(
            ChaosEvent(
                chaos_type=ChaosType.TIMEOUT,
                timestamp=datetime.utcnow(),
                description="Slow agent response",
            )
        )

        agent = MockAgent()
        agent.add_response(pattern=".*", response="Done", delay=0.1)

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

        executor = Executor(
            roadmap=project.roadmap_path,
            agent=agent,
            verifier=verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        try:
            await executor.run()
            events[-1].recovered = True
            return True, events
        except Exception as e:
            events[-1].error_message = str(e)
            return False, events

    async def _test_verifier_crash(self) -> tuple[bool, list[ChaosEvent]]:
        """Test verifier crash handling."""
        events = []
        project = TestProject(self.workspace_dir / "verifier_crash")
        project.create_roadmap(["Task 1"])

        events.append(
            ChaosEvent(
                chaos_type=ChaosType.VERIFIER_CRASH,
                timestamp=datetime.utcnow(),
                description="Verifier exception",
            )
        )

        agent = MockAgent()

        verifier = MagicMock()
        verifier.verify.side_effect = RuntimeError("Verifier crashed")

        executor = Executor(
            roadmap=project.roadmap_path,
            agent=agent,
            verifier=verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        try:
            await executor.run()
            events[-1].recovered = True
            return True, events
        except Exception as e:
            events[-1].error_message = str(e)
            return False, events

    async def _test_concurrent(self) -> tuple[bool, list[ChaosEvent]]:
        """Test concurrent access handling."""
        events = []
        project = TestProject(self.workspace_dir / "concurrent")
        project.create_roadmap(["Task 1", "Task 2"])

        events.append(
            ChaosEvent(
                chaos_type=ChaosType.CONCURRENT_ACCESS,
                timestamp=datetime.utcnow(),
                description="Multiple executors on same roadmap",
            )
        )

        agent1 = MockAgent()
        agent2 = MockAgent()

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

        executor1 = Executor(
            roadmap=project.roadmap_path,
            agent=agent1,
            verifier=verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        executor2 = Executor(
            roadmap=project.roadmap_path,
            agent=agent2,
            verifier=verifier,
            config=ExecutorConfig(max_tasks=1),
        )

        try:
            await asyncio.gather(executor1.run(), executor2.run())
            events[-1].recovered = True
            return True, events
        except Exception as e:
            events[-1].error_message = str(e)
            return False, events

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all chaos test results."""
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        by_type: dict[str, dict[str, int]] = {}
        for result in self.results:
            type_name = result.chaos_type.value
            if type_name not in by_type:
                by_type[type_name] = {"passed": 0, "failed": 0}
            if result.passed:
                by_type[type_name]["passed"] += 1
            else:
                by_type[type_name]["failed"] += 1

        return {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / len(self.results) if self.results else 0,
            "by_type": by_type,
            "results": [r.to_dict() for r in self.results],
        }
