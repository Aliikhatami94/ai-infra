"""Mock agent for executor integration testing.

This module provides a mock agent implementation that simulates
agent responses for testing the executor without making actual LLM calls.

Usage:
    from ai_infra.executor.testing import MockAgent, MockResponse

    # Create agent with scripted responses
    agent = MockAgent()
    agent.add_response(
        pattern="Create.*file",
        response="Created src/utils.py with the requested function.",
        files_modified=["src/utils.py"],
    )

    # Use with executor
    executor = Executor(roadmap, agent=agent)
    summary = await executor.run()
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# =============================================================================
# Mock Response
# =============================================================================


@dataclass
class MockResponse:
    """A mock response configuration.

    Attributes:
        pattern: Regex pattern to match against prompts.
        response: The response text to return.
        files_modified: Files to claim as modified.
        files_created: Files to claim as created.
        files_deleted: Files to claim as deleted.
        delay: Delay before returning (simulates latency).
        raise_error: Exception to raise instead of returning.
        call_count: How many times this response has been used.
        max_uses: Maximum times to use this response (0 = unlimited).
    """

    pattern: str
    response: str
    files_modified: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    delay: float = 0.0
    raise_error: Exception | None = None
    call_count: int = 0
    max_uses: int = 0

    def matches(self, prompt: str) -> bool:
        """Check if this response matches the prompt."""
        return bool(re.search(self.pattern, prompt, re.IGNORECASE))

    @property
    def exhausted(self) -> bool:
        """Whether this response has been used up."""
        if self.max_uses == 0:
            return False
        return self.call_count >= self.max_uses


# =============================================================================
# Mock Agent
# =============================================================================


class MockAgent:
    """Mock agent for testing executor without LLM calls.

    Provides scripted responses based on pattern matching against prompts.
    Supports simulating delays, errors, and tracking call history.

    Example:
        agent = MockAgent()

        # Add responses that match patterns
        agent.add_response(
            pattern="Create.*utils",
            response="Created src/utils.py",
            files_created=["src/utils.py"],
        )
        agent.add_response(
            pattern="Add.*test",
            response="Added tests to tests/test_utils.py",
            files_modified=["tests/test_utils.py"],
        )

        # Use with executor
        executor = Executor(roadmap, agent=agent)
        summary = await executor.run()

        # Check history
        assert agent.call_count == 2
        assert "utils" in agent.calls[0]["prompt"]
    """

    def __init__(
        self,
        *,
        default_response: str = "Task completed successfully.",
        default_delay: float = 0.0,
    ) -> None:
        """Initialize mock agent.

        Args:
            default_response: Response when no pattern matches.
            default_delay: Default delay for all responses.
        """
        self._responses: list[MockResponse] = []
        self._default_response = default_response
        self._default_delay = default_delay
        self._calls: list[dict[str, Any]] = []
        self._call_count = 0

    # =========================================================================
    # Configuration
    # =========================================================================

    def add_response(
        self,
        pattern: str,
        response: str,
        *,
        files_modified: list[str] | None = None,
        files_created: list[str] | None = None,
        files_deleted: list[str] | None = None,
        delay: float = 0.0,
        raise_error: Exception | None = None,
        max_uses: int = 0,
    ) -> MockAgent:
        """Add a response configuration.

        Args:
            pattern: Regex pattern to match prompts.
            response: Response text to return.
            files_modified: Files to include in response.
            files_created: Files claimed as created.
            files_deleted: Files claimed as deleted.
            delay: Delay in seconds before responding.
            raise_error: Exception to raise instead of responding.
            max_uses: Max times to use this response (0 = unlimited).

        Returns:
            Self for chaining.
        """
        self._responses.append(
            MockResponse(
                pattern=pattern,
                response=response,
                files_modified=files_modified or [],
                files_created=files_created or [],
                files_deleted=files_deleted or [],
                delay=delay,
                raise_error=raise_error,
                max_uses=max_uses,
            )
        )
        return self

    def add_error(
        self,
        pattern: str,
        error: Exception,
        *,
        delay: float = 0.0,
    ) -> MockAgent:
        """Add an error response.

        Args:
            pattern: Regex pattern to match prompts.
            error: Exception to raise.
            delay: Delay before raising.

        Returns:
            Self for chaining.
        """
        return self.add_response(
            pattern=pattern,
            response="",
            delay=delay,
            raise_error=error,
        )

    def add_timeout(
        self,
        pattern: str,
        timeout_seconds: float = 300.0,
    ) -> MockAgent:
        """Add a timeout simulation.

        Args:
            pattern: Regex pattern to match prompts.
            timeout_seconds: How long to delay (triggers timeout).

        Returns:
            Self for chaining.
        """
        return self.add_response(
            pattern=pattern,
            response="",
            delay=timeout_seconds + 1,  # Ensure it exceeds timeout
        )

    def set_default_response(self, response: str) -> MockAgent:
        """Set the default response for unmatched prompts.

        Args:
            response: Default response text.

        Returns:
            Self for chaining.
        """
        self._default_response = response
        return self

    def clear_responses(self) -> MockAgent:
        """Clear all configured responses.

        Returns:
            Self for chaining.
        """
        self._responses.clear()
        return self

    def reset(self) -> MockAgent:
        """Reset call history and response counters.

        Returns:
            Self for chaining.
        """
        self._calls.clear()
        self._call_count = 0
        for resp in self._responses:
            resp.call_count = 0
        return self

    # =========================================================================
    # Agent Protocol Implementation
    # =========================================================================

    async def arun(self, prompt: str) -> str:
        """Run the mock agent with a prompt.

        Args:
            prompt: The prompt from the executor.

        Returns:
            Mock response text.

        Raises:
            Exception: If configured to raise an error.
        """
        self._call_count += 1

        # Find matching response
        matched_response: MockResponse | None = None
        for resp in self._responses:
            if not resp.exhausted and resp.matches(prompt):
                matched_response = resp
                resp.call_count += 1
                break

        # Record the call
        call_record = {
            "prompt": prompt,
            "call_number": self._call_count,
            "matched_pattern": matched_response.pattern if matched_response else None,
        }
        self._calls.append(call_record)

        # Apply delay
        delay = matched_response.delay if matched_response else self._default_delay
        if delay > 0:
            await asyncio.sleep(delay)

        # Handle error case
        if matched_response and matched_response.raise_error:
            call_record["error"] = str(matched_response.raise_error)
            raise matched_response.raise_error

        # Return response
        if matched_response:
            response = matched_response.response

            # Include file information in response
            files_info = []
            for f in matched_response.files_created:
                files_info.append(f"Created `{f}`")
            for f in matched_response.files_modified:
                files_info.append(f"Modified `{f}`")
            for f in matched_response.files_deleted:
                files_info.append(f"Deleted `{f}`")

            if files_info:
                response = response + "\n\n" + "\n".join(files_info)

            call_record["response"] = response
            return response

        # Default response
        call_record["response"] = self._default_response
        return self._default_response

    # =========================================================================
    # Inspection
    # =========================================================================

    @property
    def call_count(self) -> int:
        """Number of times arun was called."""
        return self._call_count

    @property
    def calls(self) -> list[dict[str, Any]]:
        """History of all calls made."""
        return self._calls.copy()

    @property
    def last_call(self) -> dict[str, Any] | None:
        """Get the last call made."""
        if not self._calls:
            return None
        return self._calls[-1]

    @property
    def last_prompt(self) -> str | None:
        """Get the last prompt sent."""
        if not self._calls:
            return None
        return self._calls[-1]["prompt"]

    def get_prompts(self) -> list[str]:
        """Get all prompts that were sent."""
        return [call["prompt"] for call in self._calls]

    def was_called_with(self, pattern: str) -> bool:
        """Check if any prompt matched the pattern.

        Args:
            pattern: Regex pattern to check.

        Returns:
            True if any prompt matched.
        """
        for call in self._calls:
            if re.search(pattern, call["prompt"], re.IGNORECASE):
                return True
        return False


# =============================================================================
# Test Project Builder
# =============================================================================


@dataclass
class TestProject:
    """A test project for E2E testing.

    Provides helpers for creating project fixtures with specific
    file structures and ROADMAP.md configurations.

    Example:
        project = TestProject(tmp_path)
        project.add_file("src/main.py", "print('hello')")
        project.create_roadmap([
            "Create utils.py with helper functions",
            "Add tests for utils.py",
        ])

        executor = Executor(project.roadmap_path)
        summary = await executor.run()
    """

    root: Path
    roadmap_name: str = "ROADMAP.md"
    _files: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure root directory exists."""
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def roadmap_path(self) -> Path:
        """Get the path to ROADMAP.md."""
        return self.root / self.roadmap_name

    def add_file(self, path: str, content: str) -> TestProject:
        """Add a file to the project.

        Args:
            path: Relative path from project root.
            content: File content.

        Returns:
            Self for chaining.
        """
        full_path = self.root / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        self._files[path] = content
        return self

    def add_python_file(
        self,
        path: str,
        *,
        imports: list[str] | None = None,
        functions: list[str] | None = None,
        classes: list[str] | None = None,
    ) -> TestProject:
        """Add a Python file with generated content.

        Args:
            path: Relative path (should end in .py).
            imports: Import statements to include.
            functions: Function names to define.
            classes: Class names to define.

        Returns:
            Self for chaining.
        """
        lines = ['"""Generated test file."""', ""]

        if imports:
            for imp in imports:
                lines.append(f"import {imp}")
            lines.append("")

        if functions:
            for func in functions:
                lines.extend(
                    [
                        f"def {func}():",
                        f'    """Function {func}."""',
                        "    pass",
                        "",
                    ]
                )

        if classes:
            for cls in classes:
                lines.extend(
                    [
                        f"class {cls}:",
                        f'    """Class {cls}."""',
                        "",
                        "    def __init__(self):",
                        "        pass",
                        "",
                    ]
                )

        return self.add_file(path, "\n".join(lines))

    def create_roadmap(
        self,
        tasks: list[str],
        *,
        phase_name: str = "Test Phase",
        section_name: str = "Tasks",
        completed_indices: list[int] | None = None,
    ) -> TestProject:
        """Create a ROADMAP.md with the given tasks.

        Args:
            tasks: List of task titles.
            phase_name: Name for the phase.
            section_name: Name for the section.
            completed_indices: Indices of tasks to mark complete.

        Returns:
            Self for chaining.
        """
        completed = set(completed_indices or [])

        lines = [
            "# Test Project ROADMAP",
            "",
            f"## Phase 0: {phase_name}",
            "",
            "> **Goal**: Test goal",
            "> **Priority**: HIGH",
            "",
            f"### 0.1 {section_name}",
            "",
        ]

        for i, task in enumerate(tasks):
            marker = "[x]" if i in completed else "[ ]"
            lines.append(f"- {marker} **{task}**")
            lines.append(f"  Description for task {i + 1}.")
            lines.append("")

        content = "\n".join(lines)
        self.roadmap_path.write_text(content)
        return self

    def create_roadmap_multi_phase(
        self,
        phases: dict[str, list[str]],
    ) -> TestProject:
        """Create a ROADMAP.md with multiple phases.

        Args:
            phases: Dict mapping phase names to task lists.

        Returns:
            Self for chaining.
        """
        lines = ["# Test Project ROADMAP", ""]

        for phase_idx, (phase_name, tasks) in enumerate(phases.items()):
            lines.extend(
                [
                    f"## Phase {phase_idx}: {phase_name}",
                    "",
                    "> **Goal**: {phase_name} goal",
                    "> **Priority**: HIGH",
                    "",
                    f"### {phase_idx}.1 Tasks",
                    "",
                ]
            )

            for task in tasks:
                lines.append(f"- [ ] **{task}**")
                lines.append(f"  Description for {task}.")
                lines.append("")

        content = "\n".join(lines)
        self.roadmap_path.write_text(content)
        return self

    def file_exists(self, path: str) -> bool:
        """Check if a file exists in the project.

        Args:
            path: Relative path from project root.

        Returns:
            True if the file exists.
        """
        return (self.root / path).exists()

    def read_file(self, path: str) -> str:
        """Read a file from the project.

        Args:
            path: Relative path from project root.

        Returns:
            File content.
        """
        return (self.root / path).read_text()

    def init_git(self) -> TestProject:
        """Initialize git repository.

        Returns:
            Self for chaining.
        """
        import subprocess

        subprocess.run(
            ["git", "init"],
            cwd=self.root,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=self.root,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=self.root,
            capture_output=True,
            check=True,
        )
        # Initial commit
        subprocess.run(
            ["git", "add", "."],
            cwd=self.root,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit", "--allow-empty"],
            cwd=self.root,
            capture_output=True,
            check=True,
        )
        return self

    def get_git_commits(self) -> list[str]:
        """Get list of git commit messages.

        Returns:
            List of commit messages.
        """
        import subprocess

        result = subprocess.run(
            ["git", "log", "--oneline", "--format=%s"],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split("\n") if result.stdout.strip() else []


# =============================================================================
# Chaos Testing Utilities
# =============================================================================


class ChaosAgent(MockAgent):
    """Agent that introduces chaos for testing robustness.

    Extends MockAgent with capabilities for simulating various
    failure modes like network errors, timeouts, and corrupted responses.

    Example:
        agent = ChaosAgent()
        agent.set_failure_rate(0.3)  # 30% chance of failure
        agent.set_corruption_rate(0.1)  # 10% chance of garbled response
    """

    def __init__(
        self,
        *,
        failure_rate: float = 0.0,
        corruption_rate: float = 0.0,
        timeout_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize chaos agent.

        Args:
            failure_rate: Probability of raising an error (0-1).
            corruption_rate: Probability of returning garbled response.
            timeout_rate: Probability of timing out.
            **kwargs: Arguments passed to MockAgent.
        """
        super().__init__(**kwargs)
        self._failure_rate = failure_rate
        self._corruption_rate = corruption_rate
        self._timeout_rate = timeout_rate
        self._failures_injected = 0
        self._corruptions_injected = 0
        self._timeouts_injected = 0

    def set_failure_rate(self, rate: float) -> ChaosAgent:
        """Set the failure probability.

        Args:
            rate: Probability between 0 and 1.

        Returns:
            Self for chaining.
        """
        self._failure_rate = max(0.0, min(1.0, rate))
        return self

    def set_corruption_rate(self, rate: float) -> ChaosAgent:
        """Set the corruption probability.

        Args:
            rate: Probability between 0 and 1.

        Returns:
            Self for chaining.
        """
        self._corruption_rate = max(0.0, min(1.0, rate))
        return self

    def set_timeout_rate(self, rate: float) -> ChaosAgent:
        """Set the timeout probability.

        Args:
            rate: Probability between 0 and 1.

        Returns:
            Self for chaining.
        """
        self._timeout_rate = max(0.0, min(1.0, rate))
        return self

    async def arun(self, prompt: str) -> str:
        """Run with chaos injection.

        Args:
            prompt: The prompt to process.

        Returns:
            Response (possibly corrupted).

        Raises:
            Exception: Randomly based on failure_rate.
            TimeoutError: Randomly based on timeout_rate.
        """
        import random

        # Check for timeout
        if random.random() < self._timeout_rate:
            self._timeouts_injected += 1
            # Simulate a very long delay that would trigger timeout
            await asyncio.sleep(999999)

        # Check for failure
        if random.random() < self._failure_rate:
            self._failures_injected += 1
            raise ConnectionError("Chaos: Simulated network failure")

        # Get normal response
        response = await super().arun(prompt)

        # Check for corruption
        if random.random() < self._corruption_rate:
            self._corruptions_injected += 1
            # Garble the response
            response = "".join(
                c if random.random() > 0.3 else chr(random.randint(65, 90)) for c in response
            )

        return response

    @property
    def chaos_stats(self) -> dict[str, int]:
        """Get statistics on chaos injected."""
        return {
            "failures_injected": self._failures_injected,
            "corruptions_injected": self._corruptions_injected,
            "timeouts_injected": self._timeouts_injected,
            "total_calls": self.call_count,
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MockAgent",
    "MockResponse",
    "TestProject",
    "ChaosAgent",
]
