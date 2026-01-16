"""Tester subagent for running tests.

Phase 3.3.5 of EXECUTOR_1.md: Specialized agent for discovering
and running tests, then interpreting results.

The TesterAgent handles:
- Test discovery
- Running tests
- Parsing test output
- Reporting pass/fail status
- Identifying flaky tests

Example:
    ```python
    from ai_infra.executor.agents import TesterAgent

    agent = TesterAgent()
    result = await agent.execute(task, context)
    print(f"Tests: {result.tests_passed}/{result.tests_run}")
    ```
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any

from ai_infra.callbacks import MetricsCallbacks
from ai_infra.executor.agents.base import (
    TESTER_SYSTEM_PROMPT,
    ExecutionContext,
    SubAgent,
    SubAgentResult,
)
from ai_infra.executor.agents.registry import SubAgentRegistry, SubAgentType
from ai_infra.llm.agent import Agent
from ai_infra.llm.shell.session import SessionConfig, ShellSession
from ai_infra.llm.shell.tool import create_shell_tool, set_current_session
from ai_infra.llm.shell.types import ShellResult
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.todolist import TodoItem

__all__ = ["TesterAgent"]

logger = get_logger("executor.agents.tester")


# =============================================================================
# Tester Prompt Template
# =============================================================================

TESTER_PROMPT = """You are a QA engineer responsible for running tests.

## Task Context
Task: {task_title}
Description: {task_description}

## Project Information
- Workspace: {workspace}
- Project Type: {project_type}
- Files Modified: {files_modified}
{enriched_context}
## Your Process

1. **Discover the project type** by checking for:
   ```bash
   # Python
   ls pyproject.toml setup.py pytest.ini

   # Node.js
   ls package.json

   # Go
   ls go.mod

   # Rust
   ls Cargo.toml
   ```

2. **Find relevant tests** for the modified files:
   ```bash
   # Python
   find tests -name "test_*.py" | head -20

   # Node.js
   find . -name "*.test.js" -o -name "*.spec.js" | head -20
   ```

3. **Run tests** using the appropriate command:
   ```bash
   # Python with pytest
   pytest tests/ -v --tb=short

   # Python focused on specific files
   pytest tests/test_specific.py -v

   # Node.js
   npm test

   # Go
   go test ./... -v

   # Rust
   cargo test
   ```

4. **Interpret results** and report:
   - Total tests run
   - Tests passed
   - Tests failed (with names)
   - Any errors or warnings

## Output Format

After running tests, provide a summary:

### Test Results
- **Total**: X tests
- **Passed**: Y
- **Failed**: Z
- **Skipped**: N

### Failed Tests
- test_name_1: Brief error description
- test_name_2: Brief error description

### Overall Status
PASS or FAIL
"""


# =============================================================================
# TesterAgent
# =============================================================================


@SubAgentRegistry.register(SubAgentType.TESTER)
class TesterAgent(SubAgent):
    """Specialized agent for running tests.

    Discovers and runs tests, parses output, and reports
    pass/fail status with detailed breakdown.
    """

    name = "Tester"
    description = "Runs tests and reports results"
    model = "claude-sonnet-4-20250514"
    system_prompt = TESTER_SYSTEM_PROMPT

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout: float = 600.0,  # Longer for tests
        shell_timeout: float = 300.0,
    ) -> None:
        """Initialize the tester agent.

        Args:
            model: Optional model override.
            timeout: Maximum execution time (10 min default for tests).
            shell_timeout: Timeout for individual test commands.
        """
        super().__init__(model=model)
        self._timeout = timeout
        self._shell_timeout = shell_timeout

    def _get_tools(self) -> list[Any]:
        """Get tools for this agent."""
        return []

    async def execute(
        self,
        task: TodoItem,
        context: ExecutionContext,
    ) -> SubAgentResult:
        """Execute a testing task.

        Args:
            task: The test task.
            context: Execution context.

        Returns:
            SubAgentResult with test metrics and output.
        """
        start_time = time.perf_counter()

        logger.info(f"TesterAgent running tests for: {task.title}")

        # Create shell session
        session_config = SessionConfig(workspace_root=context.workspace)
        session = ShellSession(session_config)

        try:
            await session.start()
            set_current_session(session)

            # Create shell tool
            shell_tool = create_shell_tool(
                session=session,
                default_timeout=self._shell_timeout,
            )

            # Format the prompt with enriched context (Phase 16.5.5)
            files = ", ".join(context.files_modified) if context.files_modified else "All"

            # Phase 16.5.5.10: Include enriched context in prompt
            enriched_context = context.format_for_prompt()
            if enriched_context:
                enriched_context = "\n" + enriched_context + "\n"

            prompt = TESTER_PROMPT.format(
                task_title=task.title,
                task_description=task.description or "Run tests for recent changes.",
                workspace=str(context.workspace),
                project_type=context.project_type,
                files_modified=files,
                enriched_context=enriched_context,
            )

            # Create metrics callback to track token usage
            metrics_cb = MetricsCallbacks()

            # Create agent with callbacks
            agent = Agent(
                tools=[shell_tool],
                model_name=self._model,
                system=prompt,
                callbacks=metrics_cb,
            )

            # Run tests
            files_str = (
                ", ".join(context.files_modified) if context.files_modified else "the project"
            )
            result = await agent.arun(
                f"Run tests to verify: {task.title}\n\nFocus on testing changes in: {files_str}"
            )

            # Extract and parse output
            output = self._extract_output(result)
            tests_run, tests_passed, test_output = self._parse_test_results(
                output, session.command_history
            )

            duration_ms = (time.perf_counter() - start_time) * 1000
            success = tests_run == 0 or tests_passed == tests_run

            # Get token metrics from callback
            llm_metrics = metrics_cb.get_summary().get("llm", {})

            logger.info(
                f"TesterAgent completed in {duration_ms:.0f}ms: "
                f"{tests_passed}/{tests_run} passed, "
                f"{llm_metrics.get('total_tokens', 0)} tokens"
            )

            return SubAgentResult(
                success=success,
                output=output,
                tests_run=tests_run,
                tests_passed=tests_passed,
                test_output=test_output,
                metrics={
                    "duration_ms": duration_ms,
                    "tests_run": tests_run,
                    "tests_passed": tests_passed,
                    "tests_failed": tests_run - tests_passed,
                    "tokens_in": llm_metrics.get("total_tokens", 0),
                    "tokens_out": 0,
                    "total_tokens": llm_metrics.get("total_tokens", 0),
                    "model": self._model,
                    "agent_type": self.name,
                    "llm_calls": llm_metrics.get("calls", 0),
                },
            )

        except TimeoutError as e:
            logger.warning(f"TesterAgent timed out: {e}")
            duration_ms = (time.perf_counter() - start_time) * 1000
            return SubAgentResult(
                success=False,
                error=f"Tests timed out after {self._timeout}s",
                tests_run=0,
                tests_passed=0,
                test_output="Timeout during test execution",
                metrics={"duration_ms": duration_ms},
            )

        except Exception as e:
            logger.exception(f"TesterAgent failed: {e}")
            duration_ms = (time.perf_counter() - start_time) * 1000
            return SubAgentResult(
                success=False,
                error=str(e),
                tests_run=0,
                tests_passed=0,
                test_output=str(e),
                metrics={"duration_ms": duration_ms},
            )

        finally:
            set_current_session(None)
            await session.close()

    def _extract_output(self, result: Any) -> str:
        """Extract text output from agent result."""
        if hasattr(result, "content"):
            return str(result.content)
        elif isinstance(result, str):
            return result
        elif isinstance(result, dict):
            return str(result.get("content", result.get("output", str(result))))
        return str(result)

    def _parse_test_results(
        self,
        output: str,
        command_history: list[str | ShellResult],
    ) -> tuple[int, int, str]:
        """Parse test output to extract metrics.

        Phase 16.5.2: Updated type hint to accept ShellResult objects.

        Args:
            output: Agent output.
            command_history: Shell commands that were run (strings or ShellResult).

        Returns:
            Tuple of (tests_run, tests_passed, raw_test_output).
        """
        tests_run = 0
        tests_passed = 0
        test_output = output

        # Look for pytest-style output
        # "X passed" or "X passed, Y failed"
        pytest_match = re.search(
            r"(\d+)\s+passed(?:,\s*(\d+)\s+failed)?",
            output,
            re.IGNORECASE,
        )
        if pytest_match:
            tests_passed = int(pytest_match.group(1))
            tests_failed = int(pytest_match.group(2)) if pytest_match.group(2) else 0
            tests_run = tests_passed + tests_failed
            return tests_run, tests_passed, test_output

        # Look for "X tests" pattern
        total_match = re.search(r"(\d+)\s+tests?", output, re.IGNORECASE)
        if total_match:
            tests_run = int(total_match.group(1))

        # Look for passed count
        passed_match = re.search(r"passed[:\s]+(\d+)|(\d+)\s+passed", output, re.IGNORECASE)
        if passed_match:
            tests_passed = int(passed_match.group(1) or passed_match.group(2))

        # Look for failed count
        failed_match = re.search(r"failed[:\s]+(\d+)|(\d+)\s+failed", output, re.IGNORECASE)
        if failed_match:
            tests_failed = int(failed_match.group(1) or failed_match.group(2))
            if tests_run == 0:
                tests_run = tests_passed + tests_failed

        # Check for "all tests passed" or similar
        if "all tests pass" in output.lower() or "0 failed" in output.lower():
            if tests_run == 0:
                tests_run = 1  # At least one test must have run
            tests_passed = tests_run

        # Check for clear failure indicators
        if "fail" in output.lower() and tests_run == 0:
            tests_run = 1
            tests_passed = 0

        return tests_run, tests_passed, test_output
