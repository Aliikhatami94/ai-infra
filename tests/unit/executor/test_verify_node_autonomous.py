"""Tests for Phase 3.2: Verify Node Integration with VerificationAgent.

This module tests the updated verify_task_node functionality:
- Level 1 (syntax) and Level 2 (autonomous) verification
- Heuristics for deciding when to run autonomous verification
- Timeout handling with partial results
- State field integration (enable_autonomous_verify, verify_timeout)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.executor.nodes.verify import (
    DEFAULT_VERIFY_TIMEOUT,
    _check_syntax,
    _format_verification_failures,
    _run_autonomous_verification,
    _should_run_autonomous_verify,
    verify_task_node,
)
from ai_infra.executor.state import ExecutorGraphState
from ai_infra.executor.todolist import TodoItem, TodoStatus

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_state() -> ExecutorGraphState:
    """Create a base state for testing."""
    return {
        "roadmap_path": "/tmp/test/ROADMAP.md",
        "todos": [],
        "current_task": None,
        "completed_todos": [],
        "failed_todos": [],
        "files_modified": [],
        "verified": False,
        "error": None,
        "enable_autonomous_verify": False,
        "verify_timeout": 300.0,
    }


@pytest.fixture
def mock_todo_item() -> TodoItem:
    """Create a mock todo item for testing."""
    return TodoItem(
        id="1.1",
        title="Add user authentication endpoint",
        description="Implement JWT authentication for API",
        status=TodoStatus.IN_PROGRESS,
    )


@pytest.fixture
def mock_docs_todo_item() -> TodoItem:
    """Create a mock todo item for docs-only changes."""
    return TodoItem(
        id="1.2",
        title="Update README",
        description="Add installation instructions",
        status=TodoStatus.IN_PROGRESS,
    )


# =============================================================================
# Tests for _should_run_autonomous_verify
# =============================================================================


class TestShouldRunAutonomousVerify:
    """Tests for _should_run_autonomous_verify heuristic."""

    def test_skips_docs_only_changes(self, mock_todo_item: TodoItem) -> None:
        """Should skip verification for docs-only changes."""
        files_modified = ["README.md", "docs/guide.md"]
        assert _should_run_autonomous_verify(mock_todo_item, files_modified) is False

    def test_runs_for_code_changes(self, mock_todo_item: TodoItem) -> None:
        """Should run verification for code changes."""
        files_modified = ["src/app.py"]
        assert _should_run_autonomous_verify(mock_todo_item, files_modified) is True

    def test_runs_for_task_needing_deep_verification(self, mock_todo_item: TodoItem) -> None:
        """Should run for tasks that need deep verification."""
        # mock_todo_item has "endpoint" in title
        files_modified = ["config.json"]  # Non-code file
        assert _should_run_autonomous_verify(mock_todo_item, files_modified) is True

    def test_runs_for_typescript_files(self, mock_docs_todo_item: TodoItem) -> None:
        """Should run verification for TypeScript files."""
        files_modified = ["src/app.ts"]
        assert _should_run_autonomous_verify(mock_docs_todo_item, files_modified) is True

    def test_runs_for_rust_files(self, mock_docs_todo_item: TodoItem) -> None:
        """Should run verification for Rust files."""
        files_modified = ["src/lib.rs"]
        assert _should_run_autonomous_verify(mock_docs_todo_item, files_modified) is True

    def test_skips_for_config_only_with_docs_task(self, mock_docs_todo_item: TodoItem) -> None:
        """Should skip for config-only changes with docs task."""
        files_modified = ["config.yaml", ".gitignore"]
        assert _should_run_autonomous_verify(mock_docs_todo_item, files_modified) is False


# =============================================================================
# Tests for _check_syntax
# =============================================================================


class TestCheckSyntax:
    """Tests for _check_syntax function."""

    @pytest.mark.asyncio
    async def test_passes_without_verifier(self, base_state: ExecutorGraphState) -> None:
        """Should pass when no verifier provided."""
        result = await _check_syntax(base_state, verifier=None, check_level=None)
        assert result["passed"] is True
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_passes_with_successful_verification(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ) -> None:
        """Should pass when verifier returns success."""
        state = {**base_state, "current_task": mock_todo_item}

        mock_verifier = MagicMock()
        mock_result = MagicMock()
        mock_result.overall = True
        mock_verifier.verify = AsyncMock(return_value=mock_result)

        result = await _check_syntax(state, verifier=mock_verifier, check_level=None)

        assert result["passed"] is True
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_fails_with_failed_verification(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ) -> None:
        """Should fail when verifier returns failure."""
        state = {**base_state, "current_task": mock_todo_item}

        mock_verifier = MagicMock()
        mock_result = MagicMock()
        mock_result.overall = False
        mock_failure = MagicMock()
        mock_failure.message = "Syntax error on line 42"
        mock_result.get_failures.return_value = [mock_failure]
        mock_verifier.verify = AsyncMock(return_value=mock_result)

        result = await _check_syntax(state, verifier=mock_verifier, check_level=None)

        assert result["passed"] is False
        assert "Syntax error" in result["error"]

    @pytest.mark.asyncio
    async def test_handles_timeout(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ) -> None:
        """Should handle timeout gracefully."""
        state = {**base_state, "current_task": mock_todo_item}

        mock_verifier = MagicMock()
        mock_verifier.verify = AsyncMock(side_effect=TimeoutError("Timed out"))

        result = await _check_syntax(state, verifier=mock_verifier, check_level=None)

        assert result["passed"] is False
        assert "timed out" in result["error"].lower()


# =============================================================================
# Tests for _run_autonomous_verification
# =============================================================================


class TestRunAutonomousVerification:
    """Tests for _run_autonomous_verification function."""

    @pytest.mark.asyncio
    async def test_runs_verification_agent(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem, tmp_path: Path
    ) -> None:
        """Should run VerificationAgent and return results."""
        state = {**base_state, "roadmap_path": str(tmp_path / "ROADMAP.md")}

        with patch("ai_infra.executor.nodes.verify.VerificationAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_result = MagicMock()
            mock_result.to_dict.return_value = {
                "passed": True,
                "checks_run": ["pytest"],
                "failures": [],
                "suggestions": [],
                "duration_ms": 1234.5,
            }
            mock_agent.verify = AsyncMock(return_value=mock_result)
            mock_agent_class.return_value = mock_agent

            result = await _run_autonomous_verification(
                state=state,
                current_task=mock_todo_item,
                files_modified=["src/app.py"],
                workspace=tmp_path,
            )

            assert result["passed"] is True
            assert "pytest" in result["checks_run"]

    @pytest.mark.asyncio
    async def test_handles_timeout(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem, tmp_path: Path
    ) -> None:
        """Should handle timeout with partial results."""
        state = {**base_state, "verify_timeout": 1.0}

        with patch("ai_infra.executor.nodes.verify.VerificationAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.verify = AsyncMock(side_effect=TimeoutError("Timed out"))
            mock_agent_class.return_value = mock_agent

            result = await _run_autonomous_verification(
                state=state,
                current_task=mock_todo_item,
                files_modified=["src/app.py"],
                workspace=tmp_path,
            )

            assert result["passed"] is False
            assert any("timed out" in f["error"].lower() for f in result["failures"])
            assert any("timeout" in s.lower() for s in result["suggestions"])

    @pytest.mark.asyncio
    async def test_handles_exception(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem, tmp_path: Path
    ) -> None:
        """Should handle exceptions gracefully."""
        state = {**base_state}

        with patch("ai_infra.executor.nodes.verify.VerificationAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.verify = AsyncMock(side_effect=RuntimeError("Test error"))
            mock_agent_class.return_value = mock_agent

            result = await _run_autonomous_verification(
                state=state,
                current_task=mock_todo_item,
                files_modified=["src/app.py"],
                workspace=tmp_path,
            )

            assert result["passed"] is False
            assert any("Test error" in f["error"] for f in result["failures"])

    @pytest.mark.asyncio
    async def test_uses_roadmap_path_for_workspace(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem, tmp_path: Path
    ) -> None:
        """Should derive workspace from roadmap_path when not specified."""
        roadmap_path = tmp_path / "project" / "ROADMAP.md"
        roadmap_path.parent.mkdir(parents=True, exist_ok=True)
        state = {**base_state, "roadmap_path": str(roadmap_path)}

        with patch("ai_infra.executor.nodes.verify.VerificationAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_result = MagicMock()
            mock_result.to_dict.return_value = {"passed": True, "checks_run": []}
            mock_agent.verify = AsyncMock(return_value=mock_result)
            mock_agent_class.return_value = mock_agent

            await _run_autonomous_verification(
                state=state,
                current_task=mock_todo_item,
                files_modified=["src/app.py"],
                workspace=None,
            )

            # Verify agent was called with correct workspace
            call_args = mock_agent.verify.call_args
            assert call_args[1]["workspace"] == roadmap_path.parent


# =============================================================================
# Tests for _format_verification_failures
# =============================================================================


class TestFormatVerificationFailures:
    """Tests for _format_verification_failures function."""

    def test_formats_single_failure(self) -> None:
        """Should format a single failure."""
        result = {"failures": [{"command": "pytest", "error": "test_main FAILED"}]}
        formatted = _format_verification_failures(result)
        assert "[pytest]" in formatted
        assert "test_main FAILED" in formatted

    def test_formats_multiple_failures(self) -> None:
        """Should format multiple failures (max 3)."""
        result = {
            "failures": [
                {"command": "pytest", "error": "Error 1"},
                {"command": "ruff", "error": "Error 2"},
                {"command": "mypy", "error": "Error 3"},
                {"command": "extra", "error": "Error 4"},  # Should be excluded
            ]
        }
        formatted = _format_verification_failures(result)
        assert "Error 1" in formatted
        assert "Error 2" in formatted
        assert "Error 3" in formatted
        assert "Error 4" not in formatted

    def test_handles_empty_failures(self) -> None:
        """Should handle empty failures list."""
        result = {"failures": []}
        formatted = _format_verification_failures(result)
        assert "no details available" in formatted.lower()

    def test_handles_missing_command(self) -> None:
        """Should handle failures without command."""
        result = {"failures": [{"error": "Some error"}]}
        formatted = _format_verification_failures(result)
        assert "Some error" in formatted


# =============================================================================
# Tests for verify_task_node
# =============================================================================


class TestVerifyTaskNodePhase32:
    """Tests for verify_task_node with Phase 3.2 features."""

    @pytest.mark.asyncio
    async def test_no_current_task(self, base_state: ExecutorGraphState) -> None:
        """Should error when no current task."""
        state = {**base_state, "current_task": None}
        result = await verify_task_node(state)

        assert result["verified"] is False
        assert result["error"] is not None
        assert result["error"]["error_type"] == "verification"

    @pytest.mark.asyncio
    async def test_syntax_only_when_autonomous_disabled(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ) -> None:
        """Should only run syntax check when autonomous is disabled."""
        state = {
            **base_state,
            "current_task": mock_todo_item,
            "files_modified": ["src/app.py"],
            "enable_autonomous_verify": False,
        }

        mock_verifier = MagicMock()
        mock_result = MagicMock()
        mock_result.overall = True
        mock_verifier.verify = AsyncMock(return_value=mock_result)

        result = await verify_task_node(state, verifier=mock_verifier)

        assert result["verified"] is True
        assert (
            "autonomous_verify_result" not in result
            or result.get("autonomous_verify_result") is None
        )

    @pytest.mark.asyncio
    async def test_runs_autonomous_when_enabled(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem, tmp_path: Path
    ) -> None:
        """Should run autonomous verification when enabled."""
        state = {
            **base_state,
            "current_task": mock_todo_item,
            "files_modified": ["src/app.py"],
            "enable_autonomous_verify": True,
            "roadmap_path": str(tmp_path / "ROADMAP.md"),
        }

        mock_verifier = MagicMock()
        mock_result = MagicMock()
        mock_result.overall = True
        mock_verifier.verify = AsyncMock(return_value=mock_result)

        with patch("ai_infra.executor.nodes.verify.VerificationAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_verify_result = MagicMock()
            mock_verify_result.to_dict.return_value = {
                "passed": True,
                "checks_run": ["pytest"],
                "failures": [],
                "suggestions": [],
                "duration_ms": 1000.0,
            }
            mock_agent.verify = AsyncMock(return_value=mock_verify_result)
            mock_agent_class.return_value = mock_agent

            result = await verify_task_node(state, verifier=mock_verifier)

            assert result["verified"] is True
            assert result["autonomous_verify_result"]["passed"] is True
            assert "pytest" in result["autonomous_verify_result"]["checks_run"]

    @pytest.mark.asyncio
    async def test_fails_on_syntax_error(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ) -> None:
        """Should fail if syntax check fails."""
        state = {
            **base_state,
            "current_task": mock_todo_item,
            "files_modified": ["src/app.py"],
            "enable_autonomous_verify": True,
        }

        mock_verifier = MagicMock()
        mock_result = MagicMock()
        mock_result.overall = False
        mock_failure = MagicMock()
        mock_failure.message = "Syntax error"
        mock_result.get_failures.return_value = [mock_failure]
        mock_verifier.verify = AsyncMock(return_value=mock_result)

        result = await verify_task_node(state, verifier=mock_verifier)

        assert result["verified"] is False
        assert "Syntax error" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_fails_on_autonomous_failure(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem, tmp_path: Path
    ) -> None:
        """Should fail if autonomous verification fails."""
        state = {
            **base_state,
            "current_task": mock_todo_item,
            "files_modified": ["src/app.py"],
            "enable_autonomous_verify": True,
            "roadmap_path": str(tmp_path / "ROADMAP.md"),
        }

        mock_verifier = MagicMock()
        mock_result = MagicMock()
        mock_result.overall = True
        mock_verifier.verify = AsyncMock(return_value=mock_result)

        with patch("ai_infra.executor.nodes.verify.VerificationAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_verify_result = MagicMock()
            mock_verify_result.to_dict.return_value = {
                "passed": False,
                "checks_run": ["pytest"],
                "failures": [{"command": "pytest", "error": "Test failed"}],
                "suggestions": ["Fix the test"],
                "duration_ms": 1000.0,
            }
            mock_agent.verify = AsyncMock(return_value=mock_verify_result)
            mock_agent_class.return_value = mock_agent

            result = await verify_task_node(state, verifier=mock_verifier)

            assert result["verified"] is False
            assert result["autonomous_verify_result"]["passed"] is False

    @pytest.mark.asyncio
    async def test_skips_autonomous_for_docs_only(
        self, base_state: ExecutorGraphState, mock_docs_todo_item: TodoItem
    ) -> None:
        """Should skip autonomous for docs-only changes even when enabled."""
        state = {
            **base_state,
            "current_task": mock_docs_todo_item,
            "files_modified": ["README.md", "docs/guide.md"],
            "enable_autonomous_verify": True,
        }

        mock_verifier = MagicMock()
        mock_result = MagicMock()
        mock_result.overall = True
        mock_verifier.verify = AsyncMock(return_value=mock_result)

        result = await verify_task_node(state, verifier=mock_verifier)

        assert result["verified"] is True
        assert result["autonomous_verify_result"]["checks_run"] == ["skipped-heuristic"]

    @pytest.mark.asyncio
    async def test_auto_pass_without_verifier(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ) -> None:
        """Should auto-pass syntax when no verifier provided."""
        state = {
            **base_state,
            "current_task": mock_todo_item,
            "files_modified": ["README.md"],
            "enable_autonomous_verify": False,
        }

        result = await verify_task_node(state, verifier=None)

        assert result["verified"] is True


class TestVerifyTaskNodeTimeout:
    """Tests for verify_task_node timeout handling (Phase 3.2.3)."""

    @pytest.mark.asyncio
    async def test_uses_custom_verify_timeout(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem, tmp_path: Path
    ) -> None:
        """Should use verify_timeout from state."""
        custom_timeout = 600.0
        state = {
            **base_state,
            "current_task": mock_todo_item,
            "files_modified": ["src/app.py"],
            "enable_autonomous_verify": True,
            "verify_timeout": custom_timeout,
            "roadmap_path": str(tmp_path / "ROADMAP.md"),
        }

        mock_verifier = MagicMock()
        mock_result = MagicMock()
        mock_result.overall = True
        mock_verifier.verify = AsyncMock(return_value=mock_result)

        with patch("ai_infra.executor.nodes.verify.VerificationAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_verify_result = MagicMock()
            mock_verify_result.to_dict.return_value = {"passed": True, "checks_run": []}
            mock_agent.verify = AsyncMock(return_value=mock_verify_result)
            mock_agent_class.return_value = mock_agent

            await verify_task_node(state, verifier=mock_verifier)

            # Verify agent was created with custom timeout
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs["timeout"] == custom_timeout

    @pytest.mark.asyncio
    async def test_default_verify_timeout(self) -> None:
        """Should have correct default verify timeout."""
        assert DEFAULT_VERIFY_TIMEOUT == 300.0


# =============================================================================
# Tests for State Fields
# =============================================================================


class TestStateFields:
    """Tests for Phase 3.2 state fields."""

    def test_enable_autonomous_verify_default(self, base_state: ExecutorGraphState) -> None:
        """enable_autonomous_verify should default to False in base_state fixture."""
        assert base_state.get("enable_autonomous_verify") is False

    def test_verify_timeout_default(self, base_state: ExecutorGraphState) -> None:
        """verify_timeout should default to 300.0 in base_state fixture."""
        assert base_state.get("verify_timeout") == 300.0
