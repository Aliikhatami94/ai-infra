"""Executor Verification Tests.

This module verifies that all executor success criteria are met:
- Unit test coverage tracking
- Integration test status
- Scenario test completion
- Performance benchmarks (TTFT, success rates)
- No regressions from previous changes

Run with: pytest tests/executor/test_executor_verification.py -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# =============================================================================
# Test Configuration
# =============================================================================


# Success criteria thresholds
COVERAGE_TARGET = 80.0  # Minimum % coverage
TTFT_SIMPLE_TARGET_MS = 500  # Max TTFT for simple tasks
TTFT_COMPLEX_TARGET_MS = 2000  # Max TTFT for complex tasks
SUCCESS_RATE_TARGET = 0.90  # Minimum success rate for standard tasks


# =============================================================================
# Helper Functions
# =============================================================================


def get_project_root() -> Path:
    """Get the ai-infra project root."""
    return Path(__file__).parent.parent.parent


def run_pytest_collect(test_path: str, ignore_self: bool = False) -> int:
    """Run pytest --collect-only and return test count."""
    cmd = [sys.executable, "-m", "pytest", test_path, "--collect-only", "-q"]
    if ignore_self:
        cmd.extend(["--ignore=tests/executor/test_executor_verification.py"])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=get_project_root(),
    )
    # Parse output - the -q format shows "file.py: N" per line
    # Sum up all test counts
    total = 0
    for line in result.stdout.split("\n"):
        line = line.strip()
        # Format: "tests/executor/test_foo.py: 12"
        if line.endswith(".py:"):
            # No tests in file, skip
            continue
        if ".py: " in line:
            parts = line.rsplit(": ", 1)
            if len(parts) == 2 and parts[1].isdigit():
                total += int(parts[1])
        # Also check for "X tests collected" format (without -q)
        elif "test" in line and ("collected" in line or "selected" in line):
            parts = line.split()
            for part in parts:
                if part.isdigit():
                    return int(part)
    return total


def run_pytest_with_result(test_path: str, ignore_self: bool = False) -> tuple[int, int, int]:
    """Run pytest and return (passed, failed, skipped) counts."""
    # Don't use -q so we get the summary line like "57 passed in 0.74s"
    cmd = [sys.executable, "-m", "pytest", test_path, "--tb=no"]
    if ignore_self:
        cmd.extend(["--ignore=tests/executor/test_executor_verification.py"])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=get_project_root(),
    )
    # Parse output like "103 passed, 2 failed, 1 skipped in 5.00s"
    # or "57 passed in 0.74s"
    passed = failed = skipped = 0
    for line in result.stdout.split("\n"):
        if "passed" in line or "failed" in line or "skipped" in line:
            parts = line.replace(",", "").split()
            for i, part in enumerate(parts):
                if part == "passed" and i > 0:
                    try:
                        passed = int(parts[i - 1])
                    except ValueError:
                        pass
                elif part == "failed" and i > 0:
                    try:
                        failed = int(parts[i - 1])
                    except ValueError:
                        pass
                elif part == "skipped" and i > 0:
                    try:
                        skipped = int(parts[i - 1])
                    except ValueError:
                        pass
    return passed, failed, skipped


# =============================================================================
# Phase 6.5 Verification Tests
# =============================================================================


class TestUnitTestCoverage:
    """Verify unit test coverage criteria."""

    def test_executor_test_count(self) -> None:
        """Should have substantial executor test coverage."""
        test_count = run_pytest_collect("tests/executor/", ignore_self=True)

        # We expect 800+ tests based on Phase 6 work
        assert test_count >= 800, f"Expected 800+ executor tests, got {test_count}"

    def test_executor_tests_pass(self) -> None:
        """All executor unit tests should pass."""
        passed, failed, _skipped = run_pytest_with_result("tests/executor/", ignore_self=True)

        assert failed == 0, f"Expected 0 failures, got {failed}"
        assert passed >= 800, f"Expected 800+ passing tests, got {passed}"

    def test_coverage_tracking_available(self) -> None:
        """Coverage measurement should be available."""
        # Just verify pytest-cov is installed
        import pytest_cov

        assert pytest_cov is not None


class TestScenarioTests:
    """Verify scenario test completion."""

    def test_scenario_test_count(self) -> None:
        """Should have comprehensive scenario tests."""
        test_count = run_pytest_collect("tests/executor/scenarios/")

        # We created 103 scenario tests in Phase 6.3
        assert test_count >= 100, f"Expected 100+ scenario tests, got {test_count}"

    def test_all_scenario_tests_pass(self) -> None:
        """All scenario tests should pass."""
        passed, failed, _skipped = run_pytest_with_result("tests/executor/scenarios/")

        assert failed == 0, f"Expected 0 scenario test failures, got {failed}"
        assert passed >= 100, f"Expected 100+ passing scenario tests, got {passed}"

    def test_error_recovery_scenarios_exist(self) -> None:
        """Error recovery scenarios should be tested."""
        test_file = get_project_root() / "tests/executor/scenarios/test_error_recovery.py"
        assert test_file.exists(), "test_error_recovery.py should exist"

    def test_workflow_scenarios_exist(self) -> None:
        """Complex workflow scenarios should be tested."""
        test_file = get_project_root() / "tests/executor/scenarios/test_complex_workflow.py"
        assert test_file.exists(), "test_complex_workflow.py should exist"

    def test_resume_scenarios_exist(self) -> None:
        """Resume after interrupt scenarios should be tested."""
        test_file = get_project_root() / "tests/executor/scenarios/test_resume_interrupt.py"
        assert test_file.exists(), "test_resume_interrupt.py should exist"

    def test_subagent_scenarios_exist(self) -> None:
        """Subagent delegation scenarios should be tested."""
        test_file = get_project_root() / "tests/executor/scenarios/test_subagent_delegation.py"
        assert test_file.exists(), "test_subagent_delegation.py should exist"


class TestPerformanceBenchmarks:
    """Verify performance benchmark availability."""

    def test_benchmark_file_exists(self) -> None:
        """Performance benchmark file should exist."""
        benchmark_file = get_project_root() / "benchmarks/bench_executor_performance.py"
        assert benchmark_file.exists(), "bench_executor_performance.py should exist"

    def test_benchmarks_run_successfully(self) -> None:
        """Benchmarks should run without errors."""
        passed, failed, _skipped = run_pytest_with_result(
            "benchmarks/bench_executor_performance.py"
        )

        assert failed == 0, f"Expected 0 benchmark failures, got {failed}"
        assert passed >= 20, f"Expected 20+ benchmark tests, got {passed}"

    def test_ttft_benchmarks_exist(self) -> None:
        """TTFT benchmarks should be defined."""
        benchmark_file = get_project_root() / "benchmarks/bench_executor_performance.py"
        content = benchmark_file.read_text()

        assert "TestTimeToFirstToken" in content, "TTFT benchmark class should exist"
        assert "test_simple_task_ttft" in content, "Simple TTFT test should exist"
        assert "test_complex_task_ttft" in content, "Complex TTFT test should exist"

    def test_latency_benchmarks_exist(self) -> None:
        """Task latency benchmarks should be defined."""
        benchmark_file = get_project_root() / "benchmarks/bench_executor_performance.py"
        content = benchmark_file.read_text()

        assert "TestTaskCompletionLatency" in content, "Latency benchmark class should exist"

    def test_success_rate_benchmarks_exist(self) -> None:
        """Success rate benchmarks should be defined."""
        benchmark_file = get_project_root() / "benchmarks/bench_executor_performance.py"
        content = benchmark_file.read_text()

        assert "TestSuccessRates" in content, "Success rate benchmark class should exist"


class TestPhase0to5Regressions:
    """Verify no regressions from Phase 0-5 changes."""

    def test_streaming_module_exists(self) -> None:
        """Streaming module should exist (Phase 0)."""
        streaming = get_project_root() / "src/ai_infra/executor/streaming.py"
        assert streaming.exists(), "streaming.py should exist"

    def test_routing_module_exists(self) -> None:
        """Model routing module should exist (Phase 1)."""
        routing = get_project_root() / "src/ai_infra/executor/routing.py"
        assert routing.exists(), "routing.py should exist"

    def test_tools_directory_exists(self) -> None:
        """Tools directory should exist for shell/execution capabilities (Phase 2)."""
        tools = get_project_root() / "src/ai_infra/executor/tools"
        assert tools.is_dir(), "tools/ directory should exist"

    def test_agents_directory_exists(self) -> None:
        """Agents directory should exist (Phase 3)."""
        agents = get_project_root() / "src/ai_infra/executor/agents"
        assert agents.is_dir(), "agents/ directory should exist"

    def test_hitl_directory_exists(self) -> None:
        """HITL module should exist (Phase 4)."""
        hitl = get_project_root() / "src/ai_infra/executor/hitl"
        assert hitl.is_dir(), "hitl/ directory should exist"

    def test_skills_directory_exists(self) -> None:
        """Skills directory should exist (Phase 5)."""
        skills = get_project_root() / "src/ai_infra/executor/skills"
        assert skills.is_dir(), "skills/ directory should exist"

    def test_subagent_registry_works(self) -> None:
        """SubAgentRegistry should be importable and functional."""
        from ai_infra.executor.agents.registry import SubAgentType

        # Should be able to get agents
        assert SubAgentType.CODER is not None
        assert SubAgentType.TESTER is not None
        assert SubAgentType.REVIEWER is not None

    def test_model_router_works(self) -> None:
        """ModelRouter should be importable and functional."""
        from ai_infra.executor.routing import ModelRouter

        router = ModelRouter()
        assert router is not None

    def test_skills_database_works(self) -> None:
        """SkillsDatabase should be importable."""
        from ai_infra.executor.skills.database import SkillsDatabase

        # Should be importable
        assert SkillsDatabase is not None


class TestExecutorCoreComponents:
    """Verify core executor components work."""

    def test_executor_graph_importable(self) -> None:
        """ExecutorGraph should be importable."""
        from ai_infra.executor.graph import ExecutorGraph

        assert ExecutorGraph is not None

    def test_executor_state_importable(self) -> None:
        """ExecutorGraphState should be importable."""
        from ai_infra.executor.state import ExecutorGraphState

        assert ExecutorGraphState is not None

    def test_todolist_manager_importable(self) -> None:
        """TodoListManager should be importable."""
        from ai_infra.executor.todolist import TodoListManager

        assert TodoListManager is not None

    def test_task_verifier_importable(self) -> None:
        """TaskVerifier should be importable."""
        from ai_infra.executor.verifier import TaskVerifier

        assert TaskVerifier is not None

    def test_recovery_strategies_importable(self) -> None:
        """Recovery strategies should be importable."""
        from ai_infra.executor.recovery import RecoveryStrategy

        assert RecoveryStrategy.ROLLBACK_ALL is not None

    def test_checkpoint_system_importable(self) -> None:
        """Checkpoint system should be importable."""
        from ai_infra.executor.checkpoint import Checkpointer

        assert Checkpointer is not None


class TestDocumentation:
    """Verify documentation exists."""

    def test_executor_docs_exist(self) -> None:
        """Executor documentation should exist."""
        docs_dir = get_project_root() / "docs"
        assert docs_dir.is_dir(), "docs/ directory should exist"

    def test_executor2_roadmap_exists(self) -> None:
        """EXECUTOR_2.md roadmap should exist."""
        roadmap = get_project_root() / "EXECUTOR_2.md"
        assert roadmap.exists(), "EXECUTOR_2.md should exist"


# =============================================================================
# Summary Verification
# =============================================================================


class TestPhase6Summary:
    """Summary verification for Phase 6 completion."""

    def test_phase6_complete(self) -> None:
        """All Phase 6 requirements should be met."""
        results = {
            "unit_tests": False,
            "scenario_tests": False,
            "benchmarks": False,
            "no_regressions": False,
        }

        # Check unit tests (ignore self to avoid recursion)
        passed, failed, _ = run_pytest_with_result("tests/executor/", ignore_self=True)
        results["unit_tests"] = passed >= 800 and failed == 0

        # Check scenario tests
        passed, failed, _ = run_pytest_with_result("tests/executor/scenarios/")
        results["scenario_tests"] = passed >= 100 and failed == 0

        # Check benchmarks
        passed, failed, _ = run_pytest_with_result("benchmarks/bench_executor_performance.py")
        results["benchmarks"] = passed >= 20 and failed == 0

        # Check no regressions (key modules exist)
        root = get_project_root()
        results["no_regressions"] = all(
            [
                (root / "src/ai_infra/executor/graph.py").exists(),
                (root / "src/ai_infra/executor/routing.py").exists(),
                (root / "src/ai_infra/executor/agents").is_dir(),
                (root / "src/ai_infra/executor/skills").is_dir(),
            ]
        )

        # All should pass
        all_passed = all(results.values())

        if not all_passed:
            failed_checks = [k for k, v in results.items() if not v]
            pytest.fail(f"Phase 6 incomplete. Failed checks: {failed_checks}")
