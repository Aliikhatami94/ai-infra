"""Tests for Phase 3.1: Autonomous Verification Agent.

This module tests the VerificationAgent functionality:
- VerificationAgent.verify() method
- VERIFY_PROMPT formatting
- VerificationResult and VerificationFailure dataclasses
- Verification heuristics (task_needs_deep_verification, is_docs_only_change)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.executor.agents.verify_agent import (
    DEEP_VERIFICATION_KEYWORDS,
    VERIFY_PROMPT,
    VerificationAgent,
    VerificationFailure,
    VerificationResult,
    is_docs_only_change,
    task_needs_deep_verification,
)
from ai_infra.executor.models import Task

# =============================================================================
# VerificationFailure Tests
# =============================================================================


class TestVerificationFailure:
    """Tests for VerificationFailure dataclass."""

    def test_basic_failure(self) -> None:
        """Test basic failure creation."""
        failure = VerificationFailure(
            command="pytest",
            exit_code=1,
            error="test_app.py::test_main FAILED",
        )

        assert failure.command == "pytest"
        assert failure.exit_code == 1
        assert failure.error == "test_app.py::test_main FAILED"
        assert failure.file is None
        assert failure.line is None
        assert failure.test_name is None

    def test_failure_with_location(self) -> None:
        """Test failure with file and line location."""
        failure = VerificationFailure(
            command="pytest",
            exit_code=1,
            error="AssertionError: assert 1 == 2",
            file="tests/test_app.py",
            line=42,
            test_name="test_main",
        )

        assert failure.file == "tests/test_app.py"
        assert failure.line == 42
        assert failure.test_name == "test_main"

    def test_to_dict_basic(self) -> None:
        """Test to_dict with basic fields."""
        failure = VerificationFailure(
            command="npm test",
            exit_code=1,
            error="FAIL tests/app.test.js",
        )

        result = failure.to_dict()

        assert result == {
            "command": "npm test",
            "exit_code": 1,
            "error": "FAIL tests/app.test.js",
        }

    def test_to_dict_with_optional_fields(self) -> None:
        """Test to_dict includes optional fields when present."""
        failure = VerificationFailure(
            command="cargo test",
            exit_code=101,
            error="thread panicked",
            file="src/lib.rs",
            line=100,
            test_name="test_calculation",
        )

        result = failure.to_dict()

        assert result == {
            "command": "cargo test",
            "exit_code": 101,
            "error": "thread panicked",
            "file": "src/lib.rs",
            "line": 100,
            "test_name": "test_calculation",
        }


# =============================================================================
# VerificationResult Tests
# =============================================================================


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_passed_result(self) -> None:
        """Test a passing verification result."""
        result = VerificationResult(
            passed=True,
            checks_run=["pytest -q", "ruff check"],
            failures=[],
            suggestions=[],
            duration_ms=1234.5,
            agent_output="All tests passed.",
        )

        assert result.passed is True
        assert len(result.checks_run) == 2
        assert len(result.failures) == 0
        assert result.duration_ms == 1234.5

    def test_failed_result(self) -> None:
        """Test a failing verification result."""
        failure = VerificationFailure(
            command="pytest",
            exit_code=1,
            error="1 test failed",
        )

        result = VerificationResult(
            passed=False,
            checks_run=["pytest"],
            failures=[failure],
            suggestions=["Fix the assertion in test_main"],
            duration_ms=5000.0,
        )

        assert result.passed is False
        assert len(result.failures) == 1
        assert result.failures[0].exit_code == 1
        assert len(result.suggestions) == 1

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        failure = VerificationFailure(
            command="pytest",
            exit_code=1,
            error="AssertionError",
        )

        result = VerificationResult(
            passed=False,
            checks_run=["pytest"],
            failures=[failure],
            suggestions=["Check the expected value"],
            duration_ms=2000.0,
            agent_output="Test failed.",
        )

        data = result.to_dict()

        assert data["passed"] is False
        assert data["checks_run"] == ["pytest"]
        assert len(data["failures"]) == 1
        assert data["failures"][0]["command"] == "pytest"
        assert data["suggestions"] == ["Check the expected value"]
        assert data["duration_ms"] == 2000.0
        assert data["agent_output"] == "Test failed."

    def test_summary_passed(self) -> None:
        """Test summary for passed verification."""
        result = VerificationResult(
            passed=True,
            checks_run=["pytest", "ruff", "mypy"],
        )

        assert "passed" in result.summary().lower()
        assert "3" in result.summary()

    def test_summary_failed(self) -> None:
        """Test summary for failed verification."""
        result = VerificationResult(
            passed=False,
            failures=[
                VerificationFailure(
                    command="pytest",
                    exit_code=1,
                    error="AssertionError in test_main",
                ),
            ],
        )

        summary = result.summary()
        assert "failed" in summary.lower()
        assert "AssertionError" in summary


# =============================================================================
# Verification Heuristics Tests
# =============================================================================


class TestTaskNeedsDeepVerification:
    """Tests for task_needs_deep_verification heuristic."""

    @pytest.mark.parametrize(
        "title",
        [
            "Add user authentication endpoint",
            "Create new API handler",
            "Implement test suite",
            "Add database model",
            "Create utility function",
            "Implement algorithm for sorting",
            "Add middleware for logging",
            "Create component for dashboard",
            "Add class for parsing",
            "Implement helper for formatting",
        ],
    )
    def test_tasks_needing_deep_verification(self, title: str) -> None:
        """Test tasks that need deep verification."""
        task = Task(id="1.1", title=title)
        assert task_needs_deep_verification(task) is True

    @pytest.mark.parametrize(
        "title",
        [
            "Update README",
            "Fix typo in documentation",
            "Add license file",
            "Update version number",
            "Bump dependencies",
        ],
    )
    def test_tasks_not_needing_deep_verification(self, title: str) -> None:
        """Test tasks that don't need deep verification."""
        task = Task(id="1.1", title=title)
        assert task_needs_deep_verification(task) is False

    def test_keyword_in_description(self) -> None:
        """Test that keywords in description trigger deep verification."""
        task = Task(
            id="1.1",
            title="Update configuration",
            description="Add a new handler for processing events",
        )
        assert task_needs_deep_verification(task) is True

    def test_keywords_are_comprehensive(self) -> None:
        """Verify that we have a reasonable set of keywords."""
        expected_keywords = {
            "test",
            "function",
            "class",
            "method",
            "api",
            "endpoint",
        }
        assert expected_keywords.issubset(DEEP_VERIFICATION_KEYWORDS)


class TestIsDocsOnlyChange:
    """Tests for is_docs_only_change heuristic."""

    @pytest.mark.parametrize(
        "files",
        [
            ["README.md"],
            ["docs/guide.md"],
            ["CHANGELOG.md"],
            ["LICENSE.txt"],
            ["CONTRIBUTING.md"],
            ["README.md", "docs/api.md", "CHANGELOG.md"],
            ["readme.rst"],
        ],
    )
    def test_docs_only_changes(self, files: list[str]) -> None:
        """Test that documentation-only changes are detected."""
        assert is_docs_only_change(files) is True

    @pytest.mark.parametrize(
        "files",
        [
            ["src/app.py"],
            ["src/main.rs"],
            ["package.json"],
            ["README.md", "src/app.py"],
            ["docs/guide.md", "src/utils.ts"],
        ],
    )
    def test_non_docs_changes(self, files: list[str]) -> None:
        """Test that code changes are not marked as docs-only."""
        assert is_docs_only_change(files) is False

    def test_empty_list(self) -> None:
        """Test that empty list is not docs-only."""
        assert is_docs_only_change([]) is False

    def test_case_insensitive(self) -> None:
        """Test that detection is case-insensitive."""
        assert is_docs_only_change(["README.MD"]) is True
        assert is_docs_only_change(["Readme.md"]) is True
        assert is_docs_only_change(["DOCS/GUIDE.MD"]) is True


# =============================================================================
# VERIFY_PROMPT Tests
# =============================================================================


class TestVerifyPrompt:
    """Tests for the VERIFY_PROMPT template."""

    def test_prompt_contains_placeholders(self) -> None:
        """Test that prompt has all required placeholders."""
        assert "{task_title}" in VERIFY_PROMPT
        assert "{task_description}" in VERIFY_PROMPT
        assert "{files_modified}" in VERIFY_PROMPT

    def test_prompt_can_be_formatted(self) -> None:
        """Test that prompt can be formatted with task info."""
        formatted = VERIFY_PROMPT.format(
            task_title="Add authentication",
            task_description="Implement JWT auth",
            files_modified="src/auth.py, tests/test_auth.py",
        )

        assert "Add authentication" in formatted
        assert "Implement JWT auth" in formatted
        assert "src/auth.py" in formatted

    def test_prompt_has_verification_steps(self) -> None:
        """Test that prompt contains key verification steps."""
        assert "Analyze Project Structure" in VERIFY_PROMPT
        assert "Determine Test Commands" in VERIFY_PROMPT
        assert "Run Verification Commands" in VERIFY_PROMPT
        assert "Interpret Results" in VERIFY_PROMPT

    def test_prompt_mentions_common_tools(self) -> None:
        """Test that prompt mentions common test tools."""
        prompt_lower = VERIFY_PROMPT.lower()
        assert "pytest" in prompt_lower
        assert "npm test" in prompt_lower
        assert "cargo test" in prompt_lower
        assert "go test" in prompt_lower


# =============================================================================
# VerificationAgent Tests
# =============================================================================


class TestVerificationAgentInit:
    """Tests for VerificationAgent initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        agent = VerificationAgent()

        assert agent._timeout == 300.0
        assert agent._shell_timeout == 120.0
        assert agent._skip_docs_only is True
        assert agent._provider is None
        assert agent._model_name is None

    def test_custom_initialization(self) -> None:
        """Test custom initialization."""
        agent = VerificationAgent(
            provider="openai",
            model_name="gpt-4",
            timeout=600.0,
            shell_timeout=60.0,
            skip_docs_only=False,
        )

        assert agent._provider == "openai"
        assert agent._model_name == "gpt-4"
        assert agent._timeout == 600.0
        assert agent._shell_timeout == 60.0
        assert agent._skip_docs_only is False


class TestVerificationAgentDocsOnlySkip:
    """Tests for docs-only change skipping."""

    @pytest.mark.asyncio
    async def test_skips_docs_only_changes(self, tmp_path: Path) -> None:
        """Test that docs-only changes are skipped."""
        agent = VerificationAgent(skip_docs_only=True)
        task = Task(id="1.1", title="Update README")

        result = await agent.verify(
            workspace=tmp_path,
            task=task,
            files_modified=["README.md", "docs/guide.md"],
        )

        assert result.passed is True
        assert result.checks_run == ["docs-only-skip"]
        assert "documentation-only" in result.agent_output.lower()

    @pytest.mark.asyncio
    async def test_does_not_skip_when_disabled(self, tmp_path: Path) -> None:
        """Test that docs-only skipping can be disabled."""
        agent = VerificationAgent(skip_docs_only=False)
        task = Task(id="1.1", title="Update README")

        # Mock the Agent and ShellSession to avoid actual execution
        with (
            patch("ai_infra.executor.agents.verify_agent.ShellSession") as mock_session_class,
            patch("ai_infra.executor.agents.verify_agent.Agent") as mock_agent_class,
        ):
            mock_session = AsyncMock()
            mock_session.command_history = []
            mock_session.is_running = True
            mock_session_class.return_value = mock_session

            mock_agent = MagicMock()
            mock_agent.arun = AsyncMock(return_value="All tests passed.")
            mock_agent_class.return_value = mock_agent

            await agent.verify(
                workspace=tmp_path,
                task=task,
                files_modified=["README.md"],
            )

            # Should have attempted verification
            mock_session.start.assert_called_once()
            mock_agent.arun.assert_called_once()


class TestVerificationAgentVerify:
    """Tests for VerificationAgent.verify() method."""

    @pytest.mark.asyncio
    async def test_verify_passes_with_success_output(self, tmp_path: Path) -> None:
        """Test that verification passes when agent reports success."""
        agent = VerificationAgent()
        task = Task(id="1.1", title="Add feature")

        with (
            patch("ai_infra.executor.agents.verify_agent.ShellSession") as mock_session_class,
            patch("ai_infra.executor.agents.verify_agent.Agent") as mock_agent_class,
        ):
            mock_session = AsyncMock()
            mock_session.command_history = ["pytest -q"]
            mock_session.is_running = True
            mock_session_class.return_value = mock_session

            mock_agent = MagicMock()
            mock_agent.arun = AsyncMock(return_value="All tests passed. Verification successful.")
            mock_agent_class.return_value = mock_agent

            result = await agent.verify(
                workspace=tmp_path,
                task=task,
                files_modified=["src/app.py"],
            )

            assert result.passed is True
            assert "pytest -q" in result.checks_run

    @pytest.mark.asyncio
    async def test_verify_fails_with_failure_output(self, tmp_path: Path) -> None:
        """Test that verification fails when agent reports failures."""
        agent = VerificationAgent()
        task = Task(id="1.1", title="Add feature")

        with (
            patch("ai_infra.executor.agents.verify_agent.ShellSession") as mock_session_class,
            patch("ai_infra.executor.agents.verify_agent.Agent") as mock_agent_class,
        ):
            mock_session = AsyncMock()
            mock_session.command_history = ["pytest -q"]
            mock_session.is_running = True
            mock_session_class.return_value = mock_session

            mock_agent = MagicMock()
            mock_agent.arun = AsyncMock(return_value="Test failed: AssertionError in test_main")
            mock_agent_class.return_value = mock_agent

            result = await agent.verify(
                workspace=tmp_path,
                task=task,
                files_modified=["src/app.py"],
            )

            assert result.passed is False
            assert len(result.failures) > 0

    @pytest.mark.asyncio
    async def test_verify_handles_timeout(self, tmp_path: Path) -> None:
        """Test that verification handles timeout gracefully."""
        agent = VerificationAgent(timeout=1.0)
        task = Task(id="1.1", title="Add feature")

        with (
            patch("ai_infra.executor.agents.verify_agent.ShellSession") as mock_session_class,
            patch("ai_infra.executor.agents.verify_agent.Agent") as mock_agent_class,
        ):
            mock_session = AsyncMock()
            mock_session.command_history = ["pytest"]
            mock_session.is_running = True
            mock_session_class.return_value = mock_session

            mock_agent = MagicMock()
            mock_agent.arun = AsyncMock(side_effect=TimeoutError("Timed out"))
            mock_agent_class.return_value = mock_agent

            result = await agent.verify(
                workspace=tmp_path,
                task=task,
                files_modified=["src/app.py"],
            )

            assert result.passed is False
            assert any("timed out" in f.error.lower() for f in result.failures)

    @pytest.mark.asyncio
    async def test_verify_handles_exception(self, tmp_path: Path) -> None:
        """Test that verification handles exceptions gracefully."""
        agent = VerificationAgent()
        task = Task(id="1.1", title="Add feature")

        with (
            patch("ai_infra.executor.agents.verify_agent.ShellSession") as mock_session_class,
            patch("ai_infra.executor.agents.verify_agent.Agent") as mock_agent_class,
        ):
            mock_session = AsyncMock()
            mock_session.command_history = []
            mock_session.is_running = True
            mock_session_class.return_value = mock_session

            mock_agent = MagicMock()
            mock_agent.arun = AsyncMock(side_effect=RuntimeError("Unexpected error"))
            mock_agent_class.return_value = mock_agent

            result = await agent.verify(
                workspace=tmp_path,
                task=task,
                files_modified=["src/app.py"],
            )

            assert result.passed is False
            assert any("Unexpected error" in f.error for f in result.failures)

    @pytest.mark.asyncio
    async def test_verify_cleans_up_session(self, tmp_path: Path) -> None:
        """Test that shell session is cleaned up after verification."""
        agent = VerificationAgent()
        task = Task(id="1.1", title="Add feature")

        with (
            patch("ai_infra.executor.agents.verify_agent.ShellSession") as mock_session_class,
            patch("ai_infra.executor.agents.verify_agent.Agent") as mock_agent_class,
            patch("ai_infra.executor.agents.verify_agent.set_current_session") as mock_set_session,
        ):
            mock_session = AsyncMock()
            mock_session.command_history = []
            mock_session.is_running = True
            mock_session_class.return_value = mock_session

            mock_agent = MagicMock()
            mock_agent.arun = AsyncMock(return_value="Passed")
            mock_agent_class.return_value = mock_agent

            await agent.verify(
                workspace=tmp_path,
                task=task,
                files_modified=["src/app.py"],
            )

            # Session should be started and stopped
            mock_session.start.assert_called_once()
            mock_session.stop.assert_called_once()

            # set_current_session should be called twice (set and clear)
            assert mock_set_session.call_count == 2


class TestVerificationAgentParsing:
    """Tests for agent output parsing."""

    def test_parse_success_indicators(self) -> None:
        """Test parsing various success indicators."""
        agent = VerificationAgent()

        success_outputs = [
            "All tests passed.",
            "Verification successful.",
            "No errors found.",
            "No failures detected.",
        ]

        for output in success_outputs:
            result = agent._parse_agent_result(output, [], 1000.0)
            assert result.passed is True, f"Expected pass for: {output}"

    def test_parse_failure_indicators(self) -> None:
        """Test parsing various failure indicators."""
        agent = VerificationAgent()

        failure_outputs = [
            "Test failed: AssertionError",
            "Error: Module not found",
            "1 test failing",
            "Verification not passing",
        ]

        for output in failure_outputs:
            result = agent._parse_agent_result(output, [], 1000.0)
            assert result.passed is False, f"Expected fail for: {output}"

    def test_parse_extracts_suggestions(self) -> None:
        """Test that suggestions are extracted from output."""
        agent = VerificationAgent()

        output = """
        Test failed: AssertionError in test_main.

        I suggest fixing the expected value on line 42.
        You could also try checking the input validation.
        """

        result = agent._parse_agent_result(output, [], 1000.0)

        assert len(result.suggestions) > 0
        assert any("suggest" in s.lower() for s in result.suggestions)

    def test_parse_records_command_history(self) -> None:
        """Test that command history is recorded."""
        agent = VerificationAgent()

        commands = ["pytest -q", "ruff check", "mypy src/"]
        result = agent._parse_agent_result("Passed", commands, 1000.0)

        assert result.checks_run == commands


# =============================================================================
# Integration Tests (with mocks)
# =============================================================================


class TestVerificationAgentIntegration:
    """Integration tests for VerificationAgent with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_full_verification_flow(self, tmp_path: Path) -> None:
        """Test complete verification flow."""
        agent = VerificationAgent()
        task = Task(
            id="1.1",
            title="Add user authentication",
            description="Implement JWT authentication for API endpoints",
        )

        with (
            patch("ai_infra.executor.agents.verify_agent.ShellSession") as mock_session_class,
            patch("ai_infra.executor.agents.verify_agent.Agent") as mock_agent_class,
            patch("ai_infra.executor.agents.verify_agent.create_shell_tool") as mock_tool,
        ):
            # Setup mocks
            mock_session = AsyncMock()
            mock_session.command_history = [
                "cat pyproject.toml",
                "pytest -q tests/test_auth.py",
            ]
            mock_session.is_running = True
            mock_session_class.return_value = mock_session

            mock_agent_instance = MagicMock()
            mock_agent_instance.arun = AsyncMock(
                return_value="""
                I analyzed the project structure and found it's a Python project.

                Ran: pytest -q tests/test_auth.py
                Result: All tests passed (3 passed in 1.23s)

                Verification successful.
                """
            )
            mock_agent_class.return_value = mock_agent_instance

            mock_tool.return_value = MagicMock()

            # Run verification
            result = await agent.verify(
                workspace=tmp_path,
                task=task,
                files_modified=["src/auth.py", "tests/test_auth.py"],
            )

            # Verify results
            assert result.passed is True
            assert len(result.checks_run) == 2
            assert "pytest" in result.checks_run[1]
            assert result.duration_ms > 0

            # Verify agent was called with correct system prompt
            call_kwargs = mock_agent_class.call_args[1]
            assert "verification agent" in call_kwargs["system"].lower()
            assert "Add user authentication" in call_kwargs["system"]
