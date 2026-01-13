"""Autonomous verification agent for the Executor.

Phase 3.1 of EXECUTOR_CLI.md: Provides an agent that autonomously
discovers and runs tests to verify code works correctly.

The VerificationAgent uses the shell tool to:
1. Analyze project structure (package.json, pyproject.toml, etc.)
2. Determine test/lint/build commands
3. Run commands and interpret output
4. Report pass/fail with details

Example:
    ```python
    from ai_infra.executor.agents import VerificationAgent
    from ai_infra.executor.models import Task
    from pathlib import Path

    agent = VerificationAgent()
    result = await agent.verify(
        workspace=Path("/my/project"),
        task=task,
        files_modified=["src/app.py", "tests/test_app.py"],
    )

    if result.passed:
        print("Verification passed!")
    else:
        for failure in result.failures:
            print(f"Failed: {failure.command} - {failure.error}")
    ```
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_infra.llm.agent import Agent
from ai_infra.llm.shell.session import ShellSession
from ai_infra.llm.shell.tool import create_shell_tool, set_current_session
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.models import Task

logger = get_logger("executor.agents.verify")


# =============================================================================
# Verification Prompt (Phase 3.1.2)
# =============================================================================

VERIFY_PROMPT = """You are a verification agent. Your job is to verify that code works correctly.

You have access to the `run_shell` tool to execute any CLI command.

## Task Context
Task: {task_title}
Description: {task_description}
Files Modified: {files_modified}

## Verification Strategy

Follow these steps to verify the code:

1. **Analyze Project Structure**
   - Check for package.json, pyproject.toml, Cargo.toml, setup.py, etc.
   - Identify the project type (Python, Node.js, Rust, etc.)

2. **Determine Test Commands**
   - Check README.md for testing instructions
   - Check package.json scripts (npm test, npm run test)
   - Check Makefile targets (make test)
   - Use common conventions when no config exists:
     * Python: pytest, python -m pytest
     * Node.js: npm test
     * Rust: cargo test
     * Go: go test ./...

3. **Run Verification Commands**
   - First run a quick syntax/lint check if available
   - Then run focused tests for modified files if possible
   - Fall back to running the full test suite if needed

4. **Interpret Results**
   - Exit code 0 = success
   - Non-zero exit code = failure (check stderr for details)
   - Parse test output to identify specific failures

## Guidelines

- Focus on the modified files when possible
- Skip tests unrelated to the changes
- If tests are slow, try running only relevant tests first
- Report specific failure messages, not generic errors
- If no tests exist, verify syntax/imports at minimum

## Output Format

After running verification commands, summarize:
1. What commands you ran
2. Whether each passed or failed
3. Specific failure details if any
4. Suggestions for fixing failures
"""


# =============================================================================
# Verification Result Models (Phase 3.1.3)
# =============================================================================


@dataclass
class VerificationFailure:
    """A single verification failure.

    Attributes:
        command: The command that failed.
        exit_code: The exit code of the failed command.
        error: Error message or output from the failure.
        file: Optional file path associated with the failure.
        line: Optional line number of the failure.
        test_name: Optional test name if it was a test failure.
    """

    command: str
    exit_code: int
    error: str
    file: str | None = None
    line: int | None = None
    test_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "command": self.command,
            "exit_code": self.exit_code,
            "error": self.error,
        }
        if self.file:
            result["file"] = self.file
        if self.line is not None:
            result["line"] = self.line
        if self.test_name:
            result["test_name"] = self.test_name
        return result


@dataclass
class VerificationResult:
    """Result of autonomous verification.

    Attributes:
        passed: Whether all verification checks passed.
        checks_run: List of commands that were executed.
        failures: List of VerificationFailure objects for failed checks.
        suggestions: List of fix suggestions from the agent.
        duration_ms: Total verification time in milliseconds.
        agent_output: Raw output from the verification agent.
    """

    passed: bool
    checks_run: list[str] = field(default_factory=list)
    failures: list[VerificationFailure] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    agent_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "checks_run": self.checks_run,
            "failures": [f.to_dict() for f in self.failures],
            "suggestions": self.suggestions,
            "duration_ms": self.duration_ms,
            "agent_output": self.agent_output,
        }

    def summary(self) -> str:
        """Get a human-readable summary of the verification result."""
        if self.passed:
            return f"Verification passed. Ran {len(self.checks_run)} check(s)."
        else:
            failure_summary = "; ".join(f.error[:100] for f in self.failures[:3])
            return f"Verification failed: {failure_summary}"


# =============================================================================
# Verification Heuristics (Phase 3.1.4)
# =============================================================================

# Keywords that indicate a task needs deep verification
DEEP_VERIFICATION_KEYWORDS = frozenset(
    {
        "test",
        "function",
        "class",
        "method",
        "api",
        "endpoint",
        "handler",
        "route",
        "service",
        "model",
        "database",
        "query",
        "mutation",
        "resolver",
        "controller",
        "middleware",
        "hook",
        "component",
        "utility",
        "helper",
        "algorithm",
        "logic",
    }
)

# File patterns that are typically documentation-only
DOCS_ONLY_PATTERNS = frozenset(
    {
        ".md",
        ".rst",
        ".txt",
        ".adoc",
        "readme",
        "changelog",
        "license",
        "contributing",
        "authors",
        "history",
        "news",
    }
)


def task_needs_deep_verification(task: Task) -> bool:
    """Determine if a task needs deep verification.

    Phase 3.1.4: Heuristic to detect if task needs comprehensive testing.

    Args:
        task: The task to check.

    Returns:
        True if the task should undergo deep verification.

    Example:
        >>> task = Task(id="1.1", title="Add user authentication endpoint")
        >>> task_needs_deep_verification(task)
        True
    """
    title_lower = task.title.lower()
    description_lower = task.description.lower() if task.description else ""

    # Check title and description for keywords
    for keyword in DEEP_VERIFICATION_KEYWORDS:
        if keyword in title_lower or keyword in description_lower:
            return True

    return False


def is_docs_only_change(files_modified: list[str]) -> bool:
    """Check if changes are documentation-only.

    Phase 3.1.4: Skip verification for docs-only changes.

    Args:
        files_modified: List of modified file paths.

    Returns:
        True if all changes are documentation files.

    Example:
        >>> is_docs_only_change(["README.md", "docs/guide.md"])
        True
        >>> is_docs_only_change(["src/app.py", "README.md"])
        False
    """
    if not files_modified:
        return False

    for file_path in files_modified:
        file_lower = file_path.lower()
        # Check file extension
        has_docs_extension = any(file_lower.endswith(ext) for ext in DOCS_ONLY_PATTERNS)
        # Check filename patterns
        has_docs_name = any(pattern in file_lower for pattern in DOCS_ONLY_PATTERNS)

        if not (has_docs_extension or has_docs_name):
            return False

    return True


# =============================================================================
# Verification Agent (Phase 3.1.1)
# =============================================================================


class VerificationAgent:
    """Agent that autonomously verifies code works correctly.

    Phase 3.1.1: Uses the shell tool to discover and run tests,
    interpret results, and provide structured verification output.

    Example:
        ```python
        agent = VerificationAgent()
        result = await agent.verify(
            workspace=Path("/my/project"),
            task=task,
            files_modified=["src/app.py"],
        )
        ```
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        model_name: str | None = None,
        timeout: float = 300.0,
        shell_timeout: float = 120.0,
        skip_docs_only: bool = True,
    ) -> None:
        """Initialize the verification agent.

        Args:
            provider: LLM provider (default: uses default provider).
            model_name: LLM model name (default: uses default model).
            timeout: Maximum time for verification in seconds.
            shell_timeout: Timeout for individual shell commands.
            skip_docs_only: Whether to skip verification for docs-only changes.
        """
        self._provider = provider
        self._model_name = model_name
        self._timeout = timeout
        self._shell_timeout = shell_timeout
        self._skip_docs_only = skip_docs_only

    async def verify(
        self,
        workspace: Path,
        task: Task,
        files_modified: list[str],
    ) -> VerificationResult:
        """Verify that code works correctly.

        Uses the shell tool to autonomously discover and run tests.

        Args:
            workspace: Path to the workspace/project root.
            task: The task that was executed.
            files_modified: List of files that were modified.

        Returns:
            VerificationResult with pass/fail status and details.
        """
        start_time = time.perf_counter()

        # Check for docs-only changes
        if self._skip_docs_only and is_docs_only_change(files_modified):
            logger.info(f"Skipping verification for docs-only change: {files_modified}")
            duration_ms = (time.perf_counter() - start_time) * 1000
            return VerificationResult(
                passed=True,
                checks_run=["docs-only-skip"],
                suggestions=[],
                duration_ms=duration_ms,
                agent_output="Skipped verification for documentation-only changes.",
            )

        # Create shell session for verification
        session = ShellSession(cwd=workspace)

        try:
            await session.start()
            set_current_session(session)

            # Create shell tool bound to this session
            shell_tool = create_shell_tool(
                session=session,
                default_timeout=self._shell_timeout,
            )

            # Build the verification prompt
            prompt = VERIFY_PROMPT.format(
                task_title=task.title,
                task_description=task.description or "No description provided.",
                files_modified=", ".join(files_modified) if files_modified else "None specified",
            )

            # Create and run the agent
            agent = Agent(
                tools=[shell_tool],
                provider=self._provider,
                model_name=self._model_name,
                system=prompt,
            )

            # Run verification
            result = await agent.arun(
                "Verify that the code changes work correctly. "
                "Run appropriate tests and report the results.",
            )

            # Parse the result
            duration_ms = (time.perf_counter() - start_time) * 1000

            verification_result = self._parse_agent_result(
                result,
                session.command_history,
                duration_ms,
            )

            return verification_result

        except TimeoutError:
            logger.warning(f"Verification timed out after {self._timeout}s")
            duration_ms = (time.perf_counter() - start_time) * 1000
            return VerificationResult(
                passed=False,
                checks_run=list(session.command_history) if session.command_history else [],
                failures=[
                    VerificationFailure(
                        command="verification",
                        exit_code=-1,
                        error=f"Verification timed out after {self._timeout}s",
                    )
                ],
                suggestions=["Consider increasing the verification timeout."],
                duration_ms=duration_ms,
            )

        except Exception as e:
            logger.exception(f"Verification failed with error: {e}")
            duration_ms = (time.perf_counter() - start_time) * 1000
            return VerificationResult(
                passed=False,
                checks_run=[],
                failures=[
                    VerificationFailure(
                        command="verification",
                        exit_code=-1,
                        error=str(e),
                    )
                ],
                suggestions=[],
                duration_ms=duration_ms,
            )

        finally:
            # Clean up session
            set_current_session(None)
            await session.stop()

    def _parse_agent_result(
        self,
        result: Any,
        command_history: list[str],
        duration_ms: float,
    ) -> VerificationResult:
        """Parse the agent's result into a VerificationResult.

        Args:
            result: The agent's response.
            command_history: Commands executed via shell tool.
            duration_ms: Total duration in milliseconds.

        Returns:
            Parsed VerificationResult.
        """
        # Extract text output from agent result
        agent_output = ""
        if hasattr(result, "content"):
            agent_output = str(result.content)
        elif isinstance(result, str):
            agent_output = result
        elif isinstance(result, dict):
            agent_output = str(result.get("content", result.get("output", str(result))))
        else:
            agent_output = str(result)

        # Analyze the output to determine pass/fail
        output_lower = agent_output.lower()

        # Look for success indicators (check these first)
        # These phrases indicate success and should not trigger failure detection
        success_phrases = [
            "passed",
            "success",
            "all tests pass",
            "verification successful",
            "no errors",
            "no failures",
            "0 errors",
            "0 failures",
            "tests passed",
        ]

        # Look for failure indicators
        # These are specific failure patterns that indicate actual problems
        failure_indicators = [
            "failed",
            "failure",
            "failing",
            "not passing",
            "did not pass",
            "broken",
            "test(s) failed",
            "assertion error",
            "assertionerror",
            "exception occurred",
            "traceback",
            "exit code 1",
            "exit code: 1",
            "error:",  # "Error: ..." pattern (common in error messages)
            "error -",  # "Error - ..." pattern
            "module not found",
            "import error",
            "syntax error",
        ]

        # Check for success first - success phrases override failure detection
        has_success = any(phrase in output_lower for phrase in success_phrases)

        # Check for failures - but exclude patterns that are part of success phrases
        # Only match failure indicators that aren't part of success phrases
        has_failures = False
        for indicator in failure_indicators:
            if indicator in output_lower:
                # Make sure this isn't part of a success phrase like "no errors"
                is_negated = any(
                    f"no {indicator}" in output_lower
                    or f"0 {indicator}" in output_lower
                    or f"zero {indicator}" in output_lower
                    for indicator in [indicator]
                )
                if not is_negated:
                    has_failures = True
                    break

        # Default to passed if success indicators present, or no failure indicators
        passed = has_success or not has_failures

        # Extract failures from output
        failures: list[VerificationFailure] = []
        if has_failures:
            # Try to extract failure details
            failures.append(
                VerificationFailure(
                    command="verification",
                    exit_code=1,
                    error=self._extract_failure_summary(agent_output),
                )
            )

        # Extract suggestions
        suggestions = self._extract_suggestions(agent_output)

        return VerificationResult(
            passed=passed,
            checks_run=list(command_history),
            failures=failures,
            suggestions=suggestions,
            duration_ms=duration_ms,
            agent_output=agent_output,
        )

    def _extract_failure_summary(self, output: str) -> str:
        """Extract a summary of failures from agent output.

        Args:
            output: The agent's output text.

        Returns:
            A summary of the failures.
        """
        # Look for common failure patterns
        lines = output.split("\n")
        failure_lines = []

        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in ["error", "fail", "exception", "assert"]):
                failure_lines.append(line.strip())

        if failure_lines:
            return "; ".join(failure_lines[:5])  # Limit to first 5 failures

        # Fallback to last few lines
        return "; ".join(lines[-3:])

    def _extract_suggestions(self, output: str) -> list[str]:
        """Extract fix suggestions from agent output.

        Args:
            output: The agent's output text.

        Returns:
            List of suggested fixes.
        """
        suggestions: list[str] = []

        # Look for lines with suggestion keywords
        lines = output.split("\n")
        suggestion_keywords = ["suggest", "fix", "try", "should", "could", "recommend"]

        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in suggestion_keywords):
                clean_line = line.strip()
                if clean_line and len(clean_line) > 10:
                    suggestions.append(clean_line)

        return suggestions[:5]  # Limit to 5 suggestions
