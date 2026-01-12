"""Phase 5.7: Agent-Driven Error Recovery Tests.

These tests verify that the executor now relies on the agent to fix errors
using its existing tools (write_file, edit_file, terminal) instead of
hardcoded fix types like CREATE_INIT_FILE, CREATE_DIRECTORY, etc.

Test coverage:
- Hardcoded auto-fix is disabled
- Retry context includes language-agnostic instructions
- Agent tools are used for fixes (not hardcoded patterns)
- apply_suggestion logs but doesn't execute
- Works for any language (Python, JS, Rust, Go, etc.)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_infra.executor.adaptive import (
    AdaptiveMode,
    PlanAnalyzer,
    PlanSuggestion,
    SuggestionSafety,
    SuggestionType,
)
from ai_infra.executor.loop import (
    ExecutionResult,
    ExecutionStatus,
    Executor,
    ExecutorConfig,
)
from ai_infra.executor.roadmap import ParsedTask, Phase, Roadmap, Section
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
def sample_task() -> ParsedTask:
    """Create a sample task."""
    return ParsedTask(
        id="1.1.1",
        title="Create module structure",
        description="Create the basic module structure.",
        file_hints=["src/myproject/core.py"],
        line_number=10,
    )


@pytest.fixture
def sample_roadmap(tmp_path: Path) -> Roadmap:
    """Create a sample roadmap."""
    roadmap_path = tmp_path / "ROADMAP.md"
    roadmap_path.write_text(
        """# Test ROADMAP

## Phase 1: Setup

### 1.1 Project Structure

- [ ] **Create module**
  Create the module.
  **Files**: `src/module.py`
"""
    )
    task = ParsedTask(
        id="1.1.1",
        title="Create module",
        description="Create the module.",
        file_hints=["src/module.py"],
        line_number=10,
    )
    section = Section(
        id="1.1",
        title="Project Structure",
        tasks=[task],
    )
    phase = Phase(
        id="1",
        name="Setup",
        goal="Set up the project",
        sections=[section],
    )
    return Roadmap(
        path=str(roadmap_path),
        title="Test ROADMAP",
        phases=[phase],
    )


@pytest.fixture
def failed_result(sample_task: ParsedTask) -> ExecutionResult:
    """Create a failed execution result."""
    return ExecutionResult(
        task_id=sample_task.id,
        status=ExecutionStatus.FAILED,
        error="ModuleNotFoundError: No module named 'src.helpers'",
    )


# =============================================================================
# Hardcoded Auto-Fix Disabled Tests
# =============================================================================


class TestHardcodedAutoFixDisabled:
    """Test that hardcoded auto-fix is disabled in Phase 5.7."""

    @pytest.mark.asyncio
    async def test_apply_pending_fixes_returns_zero(
        self, test_project: TestProject, mock_agent: MockAgent
    ) -> None:
        """_apply_pending_fixes always returns 0 (hardcoded fixes disabled)."""
        test_project.add_file("src/__init__.py", "")
        test_project.create_roadmap(["Create module"])
        roadmap_path = test_project.root / "ROADMAP.md"

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1),
            agent=mock_agent,
        )

        # Create a sample task and result
        task = ParsedTask(
            id="1.1.1",
            title="Create module",
            description="Test",
            file_hints=[],
        )
        result = ExecutionResult(
            task_id="1.1.1",
            status=ExecutionStatus.FAILED,
            error="ImportError: missing module",
        )

        # Call _apply_pending_fixes - should return 0
        fixes_applied = await executor._apply_pending_fixes(task, result)
        assert fixes_applied == 0

    @pytest.mark.asyncio
    async def test_no_hardcoded_fixes_applied_during_retry(
        self, test_project: TestProject, mock_agent: MockAgent
    ) -> None:
        """During retry, no hardcoded fixes are applied - agent handles it."""
        test_project.add_file("src/__init__.py", "")
        test_project.create_roadmap(["Create failing module"])
        roadmap_path = test_project.root / "ROADMAP.md"

        # Track if _apply_suggestion_legacy was called
        legacy_apply_calls = []

        # Verifier fails first, passes second
        call_count = [0]

        async def verify_side_effect(task, levels=None):
            call_count[0] += 1
            if call_count[0] <= 1:
                return VerificationResult(
                    task_id=task.id,
                    levels_run=[CheckLevel.SYNTAX],
                    checks=[
                        CheckResult(
                            name="syntax_check",
                            level=CheckLevel.SYNTAX,
                            status=CheckStatus.FAILED,
                            error="Syntax error",
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

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1, retry_failed=2),
            agent=mock_agent,
            verifier=verifier,
        )
        executor.plan_analyzer.mode = AdaptiveMode.AUTO_FIX

        # Patch _apply_suggestion_legacy to track calls
        original_legacy = executor.plan_analyzer._apply_suggestion_legacy

        def track_legacy(suggestion):
            legacy_apply_calls.append(suggestion)
            return original_legacy(suggestion)

        executor.plan_analyzer._apply_suggestion_legacy = track_legacy

        summary = await executor.run()

        # Should succeed on retry
        assert summary.tasks_completed == 1

        # No legacy (hardcoded) fixes should have been called
        assert len(legacy_apply_calls) == 0


# =============================================================================
# Retry Context Tests
# =============================================================================


class TestRetryContext:
    """Test the enhanced retry context for agent-driven recovery."""

    def test_retry_context_is_language_agnostic(
        self, test_project: TestProject, mock_agent: MockAgent
    ) -> None:
        """Retry context doesn't contain Python-specific instructions."""
        test_project.add_file("src/__init__.py", "")
        test_project.create_roadmap(["Create module"])
        roadmap_path = test_project.root / "ROADMAP.md"

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1),
            agent=mock_agent,
        )

        task = ParsedTask(
            id="1.1.1",
            title="Create module",
            description="Test",
            file_hints=[],
        )
        error = "Error: cannot find module 'helpers'"

        context = executor._build_retry_context(task, error, attempt=2)

        # Should NOT mention Python-specific fixes (like __init__.py for imports)
        assert "__init__.py" not in context
        assert "pip install" not in context

        # Should contain generic instructions
        assert "Analyze" in context
        assert "Investigate" in context
        assert "Fix" in context
        assert "root cause" in context.lower()

    def test_retry_context_includes_error_message(
        self, test_project: TestProject, mock_agent: MockAgent
    ) -> None:
        """Retry context includes the actual error message."""
        test_project.add_file("src/__init__.py", "")
        test_project.create_roadmap(["Create module"])
        roadmap_path = test_project.root / "ROADMAP.md"

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1),
            agent=mock_agent,
        )

        task = ParsedTask(
            id="1.1.1",
            title="Create module",
            description="Test",
            file_hints=[],
        )
        error = "TypeError: cannot read property 'foo' of undefined"

        context = executor._build_retry_context(task, error, attempt=2)

        # Error message should be included
        assert "TypeError" in context
        assert "undefined" in context

    def test_retry_context_allows_any_file_modification(
        self, test_project: TestProject, mock_agent: MockAgent
    ) -> None:
        """Retry context explicitly allows modifying any file."""
        test_project.add_file("src/__init__.py", "")
        test_project.create_roadmap(["Create module"])
        roadmap_path = test_project.root / "ROADMAP.md"

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1),
            agent=mock_agent,
        )

        task = ParsedTask(
            id="1.1.1",
            title="Create module",
            description="Test",
            file_hints=["src/specific_file.py"],
        )
        error = "ImportError: wrong path"

        context = executor._build_retry_context(task, error, attempt=2)

        # Should mention ability to modify any file
        assert "ANY file" in context or "any file" in context.lower()
        assert "root cause" in context.lower()

    def test_retry_context_discourages_placeholders(
        self, test_project: TestProject, mock_agent: MockAgent
    ) -> None:
        """Retry context discourages creating empty placeholder files."""
        test_project.add_file("src/__init__.py", "")
        test_project.create_roadmap(["Create module"])
        roadmap_path = test_project.root / "ROADMAP.md"

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1),
            agent=mock_agent,
        )

        task = ParsedTask(
            id="1.1.1",
            title="Create module",
            description="Test",
            file_hints=[],
        )
        error = "ModuleNotFoundError: No module named 'missing'"

        context = executor._build_retry_context(task, error, attempt=2)

        # Should discourage workarounds
        lower_context = context.lower()
        assert "workaround" in lower_context or "placeholder" in lower_context

    def test_retry_context_includes_attempt_number(
        self, test_project: TestProject, mock_agent: MockAgent
    ) -> None:
        """Retry context includes the current attempt number."""
        test_project.add_file("src/__init__.py", "")
        test_project.create_roadmap(["Create module"])
        roadmap_path = test_project.root / "ROADMAP.md"

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1),
            agent=mock_agent,
        )

        task = ParsedTask(
            id="1.1.1",
            title="Create module",
            description="Test",
            file_hints=[],
        )

        context_2 = executor._build_retry_context(task, "error", attempt=2)
        context_3 = executor._build_retry_context(task, "error", attempt=3)

        assert "2" in context_2 or "Attempt 2" in context_2
        assert "3" in context_3 or "Attempt 3" in context_3


# =============================================================================
# apply_suggestion Behavior Tests
# =============================================================================


class TestApplySuggestionBehavior:
    """Test that apply_suggestion logs but doesn't execute hardcoded fixes."""

    def test_apply_suggestion_logs_only_in_auto_fix_mode(self, sample_roadmap: Roadmap) -> None:
        """In AUTO_FIX mode, apply_suggestion logs but doesn't execute."""
        analyzer = PlanAnalyzer(
            roadmap=sample_roadmap,
            mode=AdaptiveMode.AUTO_FIX,
        )

        suggestion = PlanSuggestion(
            suggestion_type=SuggestionType.CREATE_INIT_FILE,
            description="Create missing __init__.py",
            safety=SuggestionSafety.SAFE,
            target_task_id="1.1.1",
            file_path=Path("/tmp/test/__init__.py"),
            file_content='"""Package."""\n',
        )

        result = analyzer.apply_suggestion(suggestion, force=True)

        # Should succeed (logged) but no file created
        assert result.success is True
        assert "Phase 5.7" in result.message
        assert len(result.changes_made) == 0

        # File should NOT be created
        assert not Path("/tmp/test/__init__.py").exists()

    def test_apply_suggestion_executes_with_legacy_flag(
        self, sample_roadmap: Roadmap, tmp_path: Path
    ) -> None:
        """With legacy_apply=True, apply_suggestion executes the fix."""
        analyzer = PlanAnalyzer(
            roadmap=sample_roadmap,
            mode=AdaptiveMode.AUTO_FIX,
        )

        target_file = tmp_path / "legacy_test" / "__init__.py"

        suggestion = PlanSuggestion(
            suggestion_type=SuggestionType.CREATE_INIT_FILE,
            description="Create missing __init__.py (legacy)",
            safety=SuggestionSafety.SAFE,
            target_task_id="1.1.1",
            file_path=target_file,
            file_content='"""Package."""\n',
            metadata={"legacy_apply": True},  # Enable legacy behavior
        )

        result = analyzer.apply_suggestion(suggestion, force=True)

        # Should succeed with actual file creation
        assert result.success is True
        assert target_file.exists()

    def test_apply_suggestion_still_works_in_suggest_mode(self, sample_roadmap: Roadmap) -> None:
        """SUGGEST mode still returns ready-for-approval result."""
        analyzer = PlanAnalyzer(
            roadmap=sample_roadmap,
            mode=AdaptiveMode.SUGGEST,
        )

        suggestion = PlanSuggestion(
            suggestion_type=SuggestionType.CREATE_INIT_FILE,
            description="Create missing __init__.py",
            safety=SuggestionSafety.SAFE,
            target_task_id="1.1.1",
            file_path=Path("/tmp/test/__init__.py"),
        )

        result = analyzer.apply_suggestion(suggestion)

        # Should indicate ready for approval
        assert result.success is True
        assert "SUGGEST mode" in result.message

    def test_apply_suggestion_rejected_in_no_adapt_mode(self, sample_roadmap: Roadmap) -> None:
        """NO_ADAPT mode rejects suggestion application."""
        analyzer = PlanAnalyzer(
            roadmap=sample_roadmap,
            mode=AdaptiveMode.NO_ADAPT,
        )

        suggestion = PlanSuggestion(
            suggestion_type=SuggestionType.CREATE_INIT_FILE,
            description="Create missing __init__.py",
            safety=SuggestionSafety.SAFE,
            target_task_id="1.1.1",
            file_path=Path("/tmp/test/__init__.py"),
        )

        result = analyzer.apply_suggestion(suggestion)

        # Should fail in NO_ADAPT mode
        assert result.success is False
        assert "NO_ADAPT" in result.message


# =============================================================================
# Language-Agnostic Error Recovery Tests
# =============================================================================


class TestLanguageAgnosticRecovery:
    """Test that error recovery works for any language."""

    def test_retry_context_works_for_javascript_errors(
        self, test_project: TestProject, mock_agent: MockAgent
    ) -> None:
        """Retry context handles JavaScript errors correctly."""
        test_project.add_file("package.json", '{"name": "test"}')
        test_project.create_roadmap(["Create JS module"])
        roadmap_path = test_project.root / "ROADMAP.md"

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1),
            agent=mock_agent,
        )

        task = ParsedTask(
            id="1.1.1",
            title="Create JS module",
            description="Test",
            file_hints=["src/index.js"],
        )
        error = "Error: Cannot find module './helpers' from 'src/index.js'"

        context = executor._build_retry_context(task, error, attempt=2)

        # Error should be included
        assert "Cannot find module" in context
        assert "./helpers" in context

        # Should not suggest Python-specific fixes
        assert "__init__" not in context

    def test_retry_context_works_for_rust_errors(
        self, test_project: TestProject, mock_agent: MockAgent
    ) -> None:
        """Retry context handles Rust errors correctly."""
        test_project.add_file("Cargo.toml", '[package]\nname = "test"')
        test_project.create_roadmap(["Create Rust module"])
        roadmap_path = test_project.root / "ROADMAP.md"

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1),
            agent=mock_agent,
        )

        task = ParsedTask(
            id="1.1.1",
            title="Create Rust module",
            description="Test",
            file_hints=["src/lib.rs"],
        )
        error = "error[E0433]: failed to resolve: could not find `helpers` in `crate`"

        context = executor._build_retry_context(task, error, attempt=2)

        # Error should be included
        assert "E0433" in context
        assert "helpers" in context

    def test_retry_context_works_for_go_errors(
        self, test_project: TestProject, mock_agent: MockAgent
    ) -> None:
        """Retry context handles Go errors correctly."""
        test_project.add_file("go.mod", "module test\ngo 1.21")
        test_project.create_roadmap(["Create Go module"])
        roadmap_path = test_project.root / "ROADMAP.md"

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1),
            agent=mock_agent,
        )

        task = ParsedTask(
            id="1.1.1",
            title="Create Go module",
            description="Test",
            file_hints=["main.go"],
        )
        error = "package helpers is not in GOROOT (/usr/local/go/src/helpers)"

        context = executor._build_retry_context(task, error, attempt=2)

        # Error should be included
        assert "helpers" in context
        assert "GOROOT" in context


# =============================================================================
# Integration Tests
# =============================================================================


class TestAgentRecoveryIntegration:
    """Integration tests for agent-driven recovery."""

    @pytest.mark.asyncio
    async def test_retry_uses_enhanced_context(
        self, test_project: TestProject, mock_agent: MockAgent
    ) -> None:
        """On retry, agent receives enhanced context with error info."""
        test_project.add_file("src/__init__.py", "")
        test_project.create_roadmap(["Create failing module"])
        roadmap_path = test_project.root / "ROADMAP.md"

        # Verifier fails first, passes second
        call_count = [0]

        async def verify_side_effect(task, levels=None):
            call_count[0] += 1
            if call_count[0] <= 1:
                return VerificationResult(
                    task_id=task.id,
                    levels_run=[CheckLevel.SYNTAX],
                    checks=[
                        CheckResult(
                            name="syntax_check",
                            level=CheckLevel.SYNTAX,
                            status=CheckStatus.FAILED,
                            error="NameError: name 'undefined_var' is not defined",
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

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1, retry_failed=2),
            agent=mock_agent,
            verifier=verifier,
        )
        executor.plan_analyzer.mode = AdaptiveMode.AUTO_FIX

        summary = await executor.run()

        # Should have succeeded on retry
        assert summary.tasks_completed == 1

        # Verify the verifier was called twice (initial + retry)
        assert verifier.verify.call_count == 2

    @pytest.mark.asyncio
    async def test_callbacks_notified_on_retry(
        self, test_project: TestProject, mock_agent: MockAgent
    ) -> None:
        """Callbacks are properly notified during retry attempts."""
        test_project.add_file("src/__init__.py", "")
        test_project.create_roadmap(["Create module"])
        roadmap_path = test_project.root / "ROADMAP.md"

        # Verifier fails first, passes second
        call_count = [0]

        async def verify_side_effect(task, levels=None):
            call_count[0] += 1
            if call_count[0] <= 1:
                return VerificationResult(
                    task_id=task.id,
                    levels_run=[CheckLevel.SYNTAX],
                    checks=[
                        CheckResult(
                            name="syntax_check",
                            level=CheckLevel.SYNTAX,
                            status=CheckStatus.FAILED,
                            error="Error",
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

        # Track callback calls
        retry_events = []
        success_events = []

        from ai_infra.executor.observability import ExecutorCallbacks

        callbacks = ExecutorCallbacks()
        callbacks.on_task_retry = lambda **kwargs: retry_events.append(kwargs)
        callbacks.on_retry_success = lambda task_id, attempt: success_events.append(
            (task_id, attempt)
        )

        executor = Executor(
            roadmap=roadmap_path,
            config=ExecutorConfig(max_tasks=1, retry_failed=2),
            agent=mock_agent,
            verifier=verifier,
            callbacks=callbacks,
        )
        executor.plan_analyzer.mode = AdaptiveMode.AUTO_FIX

        await executor.run()

        # Should have one retry event
        assert len(retry_events) == 1
        assert retry_events[0]["attempt"] == 2

        # Should have one success event
        assert len(success_events) == 1
        assert success_events[0][1] == 2  # Succeeded on attempt 2
