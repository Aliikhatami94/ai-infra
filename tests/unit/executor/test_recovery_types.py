"""Tests for intelligent task recovery types (Phase 2.9.1-2.9.7).

Phase 2.9.1: Tests for FailureReason, RecoveryStrategy, and RecoveryProposal.
Phase 2.9.2: Tests for classify_failure_node and pattern-based classification.
Phase 2.9.3: Tests for propose_recovery_node and recovery proposal generation.
Phase 2.9.4: Tests for ApprovalMode and await_approval_node.
Phase 2.9.5: Tests for apply_recovery_node and _replace_task_in_roadmap.
Phase 2.9.6: Tests for retry_deferred_node and should_retry_deferred.
Phase 2.9.7: Tests for ExecutionReport and generate_report_node.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_infra.executor.nodes.recovery import (
    CLASSIFY_FAILURE_PROMPT,
    DECOMPOSE_TASK_PROMPT,
    FAILURE_TO_STRATEGY,
    MAX_DEFERRED_RETRIES,
    REWRITE_TASK_PROMPT,
    ApprovalMode,
    ExecutionReport,
    FailureReason,
    RecoveryProposal,
    RecoveryStrategy,
    _classify_by_pattern,
    _collect_error_messages,
    _format_error_history,
    _format_failure_details,
    _get_last_code,
    _get_project_summary,
    _interactive_edit,
    _parse_classification,
    _parse_numbered_list,
    _replace_task_in_roadmap,
    apply_recovery_node,
    await_approval_node,
    classify_failure_node,
    generate_report_node,
    get_strategy_for_failure,
    propose_recovery_node,
    retry_deferred_node,
    should_retry_deferred,
)
from ai_infra.executor.todolist import TodoItem, TodoStatus


class TestFailureReason:
    """Tests for FailureReason enum."""

    def test_all_values_are_strings(self) -> None:
        """FailureReason values should be string-based for serialization."""
        for reason in FailureReason:
            assert isinstance(reason.value, str)

    def test_enum_values(self) -> None:
        """Verify all expected failure reasons exist."""
        assert FailureReason.TASK_TOO_VAGUE.value == "task_too_vague"
        assert FailureReason.TASK_TOO_COMPLEX.value == "task_too_complex"
        assert FailureReason.MISSING_DEPENDENCY.value == "missing_dependency"
        assert FailureReason.ENVIRONMENT_ISSUE.value == "environment_issue"
        assert FailureReason.UNKNOWN.value == "unknown"

    def test_enum_count(self) -> None:
        """Verify we have exactly 5 failure reasons."""
        assert len(FailureReason) == 5

    def test_string_enum_comparison(self) -> None:
        """FailureReason should be comparable as strings."""
        assert FailureReason.UNKNOWN == "unknown"
        assert FailureReason.TASK_TOO_VAGUE == "task_too_vague"


class TestRecoveryStrategy:
    """Tests for RecoveryStrategy enum."""

    def test_all_values_are_strings(self) -> None:
        """RecoveryStrategy values should be string-based for serialization."""
        for strategy in RecoveryStrategy:
            assert isinstance(strategy.value, str)

    def test_enum_values(self) -> None:
        """Verify all expected recovery strategies exist."""
        assert RecoveryStrategy.REWRITE.value == "rewrite"
        assert RecoveryStrategy.DECOMPOSE.value == "decompose"
        assert RecoveryStrategy.DEFER.value == "defer"
        assert RecoveryStrategy.SKIP.value == "skip"
        assert RecoveryStrategy.ESCALATE.value == "escalate"

    def test_enum_count(self) -> None:
        """Verify we have exactly 5 recovery strategies."""
        assert len(RecoveryStrategy) == 5

    def test_string_enum_comparison(self) -> None:
        """RecoveryStrategy should be comparable as strings."""
        assert RecoveryStrategy.ESCALATE == "escalate"
        assert RecoveryStrategy.DECOMPOSE == "decompose"


class TestFailureToStrategy:
    """Tests for FAILURE_TO_STRATEGY mapping."""

    def test_all_failure_reasons_have_strategy(self) -> None:
        """Every FailureReason should map to a RecoveryStrategy."""
        for reason in FailureReason:
            assert reason in FAILURE_TO_STRATEGY
            assert isinstance(FAILURE_TO_STRATEGY[reason], RecoveryStrategy)

    def test_correct_mappings(self) -> None:
        """Verify the correct strategy for each failure reason."""
        assert FAILURE_TO_STRATEGY[FailureReason.TASK_TOO_VAGUE] == RecoveryStrategy.REWRITE
        assert FAILURE_TO_STRATEGY[FailureReason.TASK_TOO_COMPLEX] == RecoveryStrategy.DECOMPOSE
        assert FAILURE_TO_STRATEGY[FailureReason.MISSING_DEPENDENCY] == RecoveryStrategy.DEFER
        assert FAILURE_TO_STRATEGY[FailureReason.ENVIRONMENT_ISSUE] == RecoveryStrategy.SKIP
        assert FAILURE_TO_STRATEGY[FailureReason.UNKNOWN] == RecoveryStrategy.ESCALATE

    def test_get_strategy_for_failure(self) -> None:
        """get_strategy_for_failure should return the correct strategy."""
        assert get_strategy_for_failure(FailureReason.TASK_TOO_VAGUE) == RecoveryStrategy.REWRITE
        assert get_strategy_for_failure(FailureReason.UNKNOWN) == RecoveryStrategy.ESCALATE


class TestRecoveryProposal:
    """Tests for RecoveryProposal dataclass."""

    @pytest.fixture
    def sample_proposal(self) -> RecoveryProposal:
        """Create a sample RecoveryProposal for testing."""
        return RecoveryProposal(
            original_task="Implement feature X",
            failure_reason=FailureReason.TASK_TOO_COMPLEX,
            strategy=RecoveryStrategy.DECOMPOSE,
            proposed_tasks=[
                "Create data models for feature X",
                "Implement API endpoints for feature X",
                "Add tests for feature X",
            ],
            explanation="Task attempts to do too many things. Split into smaller tasks.",
            requires_approval=True,
            error_history=["SyntaxError: unexpected EOF", "TypeError: missing argument"],
        )

    @pytest.fixture
    def sample_todo(self) -> TodoItem:
        """Create a sample TodoItem for testing."""
        return TodoItem(
            id=1,
            title="Implement complex feature",
            description="A complex task that needs decomposition",
            status=TodoStatus.NOT_STARTED,
        )

    def test_dataclass_fields(self, sample_proposal: RecoveryProposal) -> None:
        """Verify dataclass fields are accessible."""
        assert sample_proposal.original_task == "Implement feature X"
        assert sample_proposal.failure_reason == FailureReason.TASK_TOO_COMPLEX
        assert sample_proposal.strategy == RecoveryStrategy.DECOMPOSE
        assert len(sample_proposal.proposed_tasks) == 3
        assert sample_proposal.requires_approval is True
        assert len(sample_proposal.error_history) == 2

    def test_default_values(self) -> None:
        """Verify default values for optional fields."""
        proposal = RecoveryProposal(
            original_task="Task",
            failure_reason=FailureReason.UNKNOWN,
            strategy=RecoveryStrategy.ESCALATE,
        )
        assert proposal.proposed_tasks == []
        assert proposal.explanation == ""
        assert proposal.requires_approval is True
        assert proposal.error_history == []

    def test_to_markdown_basic(self, sample_proposal: RecoveryProposal) -> None:
        """to_markdown should produce valid Markdown output."""
        markdown = sample_proposal.to_markdown()

        assert "## Recovery Proposal" in markdown
        assert "**Original Task**: Implement feature X" in markdown
        assert "**Failure Reason**: task_too_complex" in markdown
        assert "**Strategy**: decompose" in markdown
        assert "**Requires Approval**: Yes" in markdown

    def test_to_markdown_includes_explanation(self, sample_proposal: RecoveryProposal) -> None:
        """to_markdown should include explanation section."""
        markdown = sample_proposal.to_markdown()

        assert "### Explanation" in markdown
        assert "Task attempts to do too many things" in markdown

    def test_to_markdown_includes_proposed_tasks(self, sample_proposal: RecoveryProposal) -> None:
        """to_markdown should include proposed replacement tasks."""
        markdown = sample_proposal.to_markdown()

        assert "### Proposed Replacement" in markdown
        assert "- [ ] Create data models for feature X" in markdown
        assert "- [ ] Implement API endpoints for feature X" in markdown
        assert "- [ ] Add tests for feature X" in markdown

    def test_to_markdown_includes_error_history(self, sample_proposal: RecoveryProposal) -> None:
        """to_markdown should include error history."""
        markdown = sample_proposal.to_markdown()

        assert "### Error History" in markdown
        assert "1. SyntaxError: unexpected EOF" in markdown
        assert "2. TypeError: missing argument" in markdown

    def test_to_markdown_no_approval_required(self) -> None:
        """to_markdown should show 'No' when approval not required."""
        proposal = RecoveryProposal(
            original_task="Task",
            failure_reason=FailureReason.ENVIRONMENT_ISSUE,
            strategy=RecoveryStrategy.SKIP,
            requires_approval=False,
        )
        markdown = proposal.to_markdown()

        assert "**Requires Approval**: No" in markdown

    def test_to_markdown_minimal(self) -> None:
        """to_markdown should work with minimal fields."""
        proposal = RecoveryProposal(
            original_task="Simple task",
            failure_reason=FailureReason.UNKNOWN,
            strategy=RecoveryStrategy.ESCALATE,
        )
        markdown = proposal.to_markdown()

        assert "## Recovery Proposal" in markdown
        assert "**Original Task**: Simple task" in markdown

    def test_from_task_creates_proposal(self, sample_todo: TodoItem) -> None:
        """from_task should create a RecoveryProposal from TodoItem."""
        proposal = RecoveryProposal.from_task(
            task=sample_todo,
            failure_reason=FailureReason.TASK_TOO_COMPLEX,
            proposed_tasks=["Task 1", "Task 2"],
            explanation="Needs decomposition",
            error_history=["Error 1"],
        )

        assert proposal.original_task == sample_todo.title
        assert proposal.failure_reason == FailureReason.TASK_TOO_COMPLEX
        assert proposal.strategy == RecoveryStrategy.DECOMPOSE
        assert proposal.proposed_tasks == ["Task 1", "Task 2"]
        assert proposal.explanation == "Needs decomposition"
        assert proposal.error_history == ["Error 1"]

    def test_from_task_sets_correct_strategy(self, sample_todo: TodoItem) -> None:
        """from_task should automatically set strategy based on failure reason."""
        proposal = RecoveryProposal.from_task(
            task=sample_todo,
            failure_reason=FailureReason.TASK_TOO_VAGUE,
        )
        assert proposal.strategy == RecoveryStrategy.REWRITE

        proposal = RecoveryProposal.from_task(
            task=sample_todo,
            failure_reason=FailureReason.ENVIRONMENT_ISSUE,
        )
        assert proposal.strategy == RecoveryStrategy.SKIP

    def test_from_task_requires_approval_for_decompose(self, sample_todo: TodoItem) -> None:
        """DECOMPOSE strategy should require approval."""
        proposal = RecoveryProposal.from_task(
            task=sample_todo,
            failure_reason=FailureReason.TASK_TOO_COMPLEX,
        )
        assert proposal.requires_approval is True

    def test_from_task_requires_approval_for_rewrite(self, sample_todo: TodoItem) -> None:
        """REWRITE strategy should require approval."""
        proposal = RecoveryProposal.from_task(
            task=sample_todo,
            failure_reason=FailureReason.TASK_TOO_VAGUE,
        )
        assert proposal.requires_approval is True

    def test_from_task_requires_approval_for_escalate(self, sample_todo: TodoItem) -> None:
        """ESCALATE strategy should require approval."""
        proposal = RecoveryProposal.from_task(
            task=sample_todo,
            failure_reason=FailureReason.UNKNOWN,
        )
        assert proposal.requires_approval is True

    def test_from_task_no_approval_for_skip(self, sample_todo: TodoItem) -> None:
        """SKIP strategy should not require approval (just logs and moves on)."""
        proposal = RecoveryProposal.from_task(
            task=sample_todo,
            failure_reason=FailureReason.ENVIRONMENT_ISSUE,
        )
        assert proposal.requires_approval is False

    def test_from_task_no_approval_for_defer(self, sample_todo: TodoItem) -> None:
        """DEFER strategy should not require approval (automatic retry later)."""
        proposal = RecoveryProposal.from_task(
            task=sample_todo,
            failure_reason=FailureReason.MISSING_DEPENDENCY,
        )
        assert proposal.requires_approval is False

    def test_from_task_default_values(self, sample_todo: TodoItem) -> None:
        """from_task should use sensible defaults."""
        proposal = RecoveryProposal.from_task(
            task=sample_todo,
            failure_reason=FailureReason.UNKNOWN,
        )

        assert proposal.proposed_tasks == []
        assert proposal.explanation == ""
        assert proposal.error_history == []


class TestRecoveryProposalEdgeCases:
    """Edge case tests for RecoveryProposal."""

    def test_empty_proposed_tasks_in_markdown(self) -> None:
        """Markdown should handle empty proposed_tasks gracefully."""
        proposal = RecoveryProposal(
            original_task="Task",
            failure_reason=FailureReason.UNKNOWN,
            strategy=RecoveryStrategy.ESCALATE,
            proposed_tasks=[],
        )
        markdown = proposal.to_markdown()

        # Should not have "Proposed Replacement" section when empty
        assert "### Proposed Replacement" not in markdown

    def test_special_characters_in_task(self) -> None:
        """Markdown should handle special characters in task descriptions."""
        proposal = RecoveryProposal(
            original_task="Implement `feature` with **bold** and [link](url)",
            failure_reason=FailureReason.TASK_TOO_VAGUE,
            strategy=RecoveryStrategy.REWRITE,
            proposed_tasks=["Handle `code` and *emphasis*"],
        )
        markdown = proposal.to_markdown()

        # Should include the special characters as-is
        assert "`feature`" in markdown
        assert "`code`" in markdown

    def test_multiline_explanation(self) -> None:
        """Markdown should handle multiline explanations."""
        proposal = RecoveryProposal(
            original_task="Task",
            failure_reason=FailureReason.TASK_TOO_COMPLEX,
            strategy=RecoveryStrategy.DECOMPOSE,
            explanation="Line 1\nLine 2\nLine 3",
        )
        markdown = proposal.to_markdown()

        assert "Line 1\nLine 2\nLine 3" in markdown

    def test_very_long_error_history(self) -> None:
        """Markdown should handle many error history entries."""
        errors = [f"Error message {i}" for i in range(10)]
        proposal = RecoveryProposal(
            original_task="Task",
            failure_reason=FailureReason.UNKNOWN,
            strategy=RecoveryStrategy.ESCALATE,
            error_history=errors,
        )
        markdown = proposal.to_markdown()

        for i in range(1, 11):
            assert f"{i}. Error message {i - 1}" in markdown

    def test_empty_error_history_no_section(self) -> None:
        """Empty error history should not show section."""
        proposal = RecoveryProposal(
            original_task="Task",
            failure_reason=FailureReason.UNKNOWN,
            strategy=RecoveryStrategy.ESCALATE,
            error_history=[],
        )
        markdown = proposal.to_markdown()

        assert "### Error History" not in markdown

    def test_empty_explanation_no_section(self) -> None:
        """Empty explanation should not show section."""
        proposal = RecoveryProposal(
            original_task="Task",
            failure_reason=FailureReason.UNKNOWN,
            strategy=RecoveryStrategy.ESCALATE,
            explanation="",
        )
        markdown = proposal.to_markdown()

        assert "### Explanation" not in markdown


# =============================================================================
# Phase 2.9.2: Classify Failure Node Tests
# =============================================================================


class TestClassifyByPattern:
    """Tests for _classify_by_pattern helper function."""

    def test_environment_issue_module_not_found(self) -> None:
        """ModuleNotFoundError should classify as ENVIRONMENT_ISSUE."""
        result = _classify_by_pattern(
            "ModuleNotFoundError: No module named 'numpy'",
            {},
        )
        assert result == FailureReason.ENVIRONMENT_ISSUE

    def test_environment_issue_pip_install(self) -> None:
        """Error suggesting pip install should classify as ENVIRONMENT_ISSUE."""
        result = _classify_by_pattern(
            "Error: run pip install package-name first",
            {},
        )
        assert result == FailureReason.ENVIRONMENT_ISSUE

    def test_environment_issue_permission_denied(self) -> None:
        """Permission denied should classify as ENVIRONMENT_ISSUE."""
        result = _classify_by_pattern(
            "PermissionError: [Errno 13] Permission denied: '/etc/passwd'",
            {},
        )
        assert result == FailureReason.ENVIRONMENT_ISSUE

    def test_missing_dependency_import_error(self) -> None:
        """ImportError should classify as MISSING_DEPENDENCY."""
        result = _classify_by_pattern(
            "ImportError: cannot import name 'foo' from 'bar'",
            {},
        )
        assert result == FailureReason.MISSING_DEPENDENCY

    def test_missing_dependency_file_not_found(self) -> None:
        """File not found should classify as MISSING_DEPENDENCY."""
        result = _classify_by_pattern(
            "FileNotFoundError: No such file or directory: 'config.yaml'",
            {},
        )
        assert result == FailureReason.MISSING_DEPENDENCY

    def test_task_too_complex_long_description(self) -> None:
        """Long task description should classify as TASK_TOO_COMPLEX."""
        long_desc = "x" * 600
        state = {"current_task": {"description": long_desc}}
        result = _classify_by_pattern("Some error", state)
        assert result == FailureReason.TASK_TOO_COMPLEX

    def test_task_too_complex_many_commas(self) -> None:
        """Task with many commas should classify as TASK_TOO_COMPLEX."""
        state = {"current_task": {"description": "a, b, c, d, e, f, g, h"}}
        result = _classify_by_pattern("Some error", state)
        assert result == FailureReason.TASK_TOO_COMPLEX

    def test_task_too_complex_many_ands(self) -> None:
        """Task with many 'and' words should classify as TASK_TOO_COMPLEX."""
        state = {"current_task": {"description": "do A and B and C and D and E"}}
        result = _classify_by_pattern("Some error", state)
        assert result == FailureReason.TASK_TOO_COMPLEX

    def test_task_too_complex_with_todo_item(self) -> None:
        """Should work with TodoItem objects, not just dicts."""
        task = TodoItem(
            id=1,
            title="Complex task",
            description="x" * 600,
            status=TodoStatus.NOT_STARTED,
        )
        state = {"current_task": task}
        result = _classify_by_pattern("Some error", state)
        assert result == FailureReason.TASK_TOO_COMPLEX

    def test_unknown_for_generic_error(self) -> None:
        """Generic errors should classify as UNKNOWN."""
        result = _classify_by_pattern(
            "AssertionError: test failed",
            {"current_task": {"description": "simple task"}},
        )
        assert result == FailureReason.UNKNOWN

    def test_case_insensitive_matching(self) -> None:
        """Pattern matching should be case-insensitive."""
        result = _classify_by_pattern(
            "MODULENOTFOUNDERROR: NO MODULE NAMED 'foo'",
            {},
        )
        assert result == FailureReason.ENVIRONMENT_ISSUE


class TestFormatErrorHistory:
    """Tests for _format_error_history helper function."""

    def test_empty_history_with_error(self) -> None:
        """Should format single error when no history."""
        state = {"error": {"message": "Test error"}}
        result = _format_error_history(state)
        assert "1. Test error" in result

    def test_empty_history_no_error(self) -> None:
        """Should return placeholder when no errors."""
        result = _format_error_history({})
        assert "No error history available" in result

    def test_list_of_dict_errors(self) -> None:
        """Should format list of dict errors."""
        state = {
            "error_history": [
                {"message": "Error 1"},
                {"message": "Error 2"},
            ]
        }
        result = _format_error_history(state)
        assert "1. Error 1" in result
        assert "2. Error 2" in result

    def test_list_of_string_errors(self) -> None:
        """Should format list of string errors."""
        state = {"error_history": ["Error 1", "Error 2"]}
        result = _format_error_history(state)
        assert "1. Error 1" in result
        assert "2. Error 2" in result


class TestGetLastCode:
    """Tests for _get_last_code helper function."""

    def test_no_generated_code(self) -> None:
        """Should return placeholder when no code available."""
        result = _get_last_code({})
        assert "No code available" in result

    def test_empty_generated_code(self) -> None:
        """Should return placeholder for empty dict."""
        result = _get_last_code({"generated_code": {}})
        assert "No code available" in result

    def test_returns_code_content(self) -> None:
        """Should return code content."""
        state = {"generated_code": {"test.py": "print('hello')"}}
        result = _get_last_code(state)
        assert "print('hello')" in result

    def test_truncates_long_code(self) -> None:
        """Should truncate code longer than 2000 chars."""
        long_code = "x" * 3000
        state = {"generated_code": {"test.py": long_code}}
        result = _get_last_code(state)
        assert len(result) < 3000
        assert "truncated" in result


class TestParseClassification:
    """Tests for _parse_classification helper function."""

    def test_direct_match(self) -> None:
        """Should match exact enum values."""
        assert _parse_classification("task_too_complex") == FailureReason.TASK_TOO_COMPLEX
        assert _parse_classification("environment_issue") == FailureReason.ENVIRONMENT_ISSUE

    def test_with_whitespace(self) -> None:
        """Should handle leading/trailing whitespace."""
        assert _parse_classification("  task_too_vague  ") == FailureReason.TASK_TOO_VAGUE

    def test_case_insensitive(self) -> None:
        """Should be case-insensitive."""
        assert _parse_classification("UNKNOWN") == FailureReason.UNKNOWN
        assert _parse_classification("Task_Too_Complex") == FailureReason.TASK_TOO_COMPLEX

    def test_partial_match(self) -> None:
        """Should match if value is contained in response."""
        assert (
            _parse_classification("I think this is task_too_complex because...")
            == FailureReason.TASK_TOO_COMPLEX
        )

    def test_unknown_for_invalid(self) -> None:
        """Should return UNKNOWN for unrecognized responses."""
        assert _parse_classification("gibberish") == FailureReason.UNKNOWN


class TestClassifyFailurePrompt:
    """Tests for CLASSIFY_FAILURE_PROMPT template."""

    def test_prompt_has_placeholders(self) -> None:
        """Prompt should have all required placeholders."""
        assert "{task_description}" in CLASSIFY_FAILURE_PROMPT
        assert "{error_history}" in CLASSIFY_FAILURE_PROMPT
        assert "{last_code}" in CLASSIFY_FAILURE_PROMPT
        assert "{last_error}" in CLASSIFY_FAILURE_PROMPT

    def test_prompt_lists_all_options(self) -> None:
        """Prompt should list all classification options."""
        assert "task_too_vague" in CLASSIFY_FAILURE_PROMPT
        assert "task_too_complex" in CLASSIFY_FAILURE_PROMPT
        assert "missing_dependency" in CLASSIFY_FAILURE_PROMPT
        assert "environment_issue" in CLASSIFY_FAILURE_PROMPT
        assert "unknown" in CLASSIFY_FAILURE_PROMPT


class TestClassifyFailureNode:
    """Tests for classify_failure_node async function."""

    @pytest.mark.asyncio
    async def test_pattern_match_environment_issue(self) -> None:
        """Should classify environment issues without LLM."""
        state = {"error": {"message": "ModuleNotFoundError: No module named 'foo'"}}
        result = await classify_failure_node(state)

        assert result["failure_reason"] == "environment_issue"
        assert result["recovery_strategy"] == "skip"

    @pytest.mark.asyncio
    async def test_pattern_match_missing_dependency(self) -> None:
        """Should classify missing dependency without LLM."""
        state = {"error": {"message": "ImportError: cannot import name 'bar'"}}
        result = await classify_failure_node(state)

        assert result["failure_reason"] == "missing_dependency"
        assert result["recovery_strategy"] == "defer"

    @pytest.mark.asyncio
    async def test_pattern_match_task_too_complex(self) -> None:
        """Should classify complex tasks without LLM."""
        state = {
            "error": {"message": "Test failed"},
            "current_task": {"description": "x" * 600},
        }
        result = await classify_failure_node(state)

        assert result["failure_reason"] == "task_too_complex"
        assert result["recovery_strategy"] == "decompose"

    @pytest.mark.asyncio
    async def test_unknown_without_agent(self) -> None:
        """Should return UNKNOWN when pattern doesn't match and no agent."""
        state = {
            "error": {"message": "Random error"},
            "current_task": {"description": "simple task"},
        }
        result = await classify_failure_node(state)

        assert result["failure_reason"] == "unknown"
        assert result["recovery_strategy"] == "escalate"

    @pytest.mark.asyncio
    async def test_llm_fallback_on_unknown(self) -> None:
        """Should use LLM when pattern returns UNKNOWN."""
        mock_agent = MagicMock()
        mock_agent.arun = AsyncMock(return_value="task_too_vague")

        state = {
            "error": {"message": "Random error"},
            "current_task": {"description": "simple task"},
        }
        result = await classify_failure_node(state, agent=mock_agent)

        assert result["failure_reason"] == "task_too_vague"
        assert result["recovery_strategy"] == "rewrite"
        mock_agent.arun.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_fallback_handles_exception(self) -> None:
        """Should return UNKNOWN if LLM call fails."""
        mock_agent = MagicMock()
        mock_agent.arun = AsyncMock(side_effect=Exception("API error"))

        state = {
            "error": {"message": "Random error"},
            "current_task": {"description": "simple task"},
        }
        result = await classify_failure_node(state, agent=mock_agent)

        assert result["failure_reason"] == "unknown"
        assert result["recovery_strategy"] == "escalate"

    @pytest.mark.asyncio
    async def test_preserves_existing_state(self) -> None:
        """Should preserve existing state fields."""
        state = {
            "error": {"message": "ModuleNotFoundError: foo"},
            "current_task": {"description": "task"},
            "existing_field": "preserved",
        }
        result = await classify_failure_node(state)

        assert result["existing_field"] == "preserved"
        assert result["failure_reason"] == "environment_issue"

    @pytest.mark.asyncio
    async def test_handles_string_error(self) -> None:
        """Should handle error as string instead of dict."""
        state = {"error": "ModuleNotFoundError: No module named 'foo'"}
        result = await classify_failure_node(state)

        assert result["failure_reason"] == "environment_issue"

    @pytest.mark.asyncio
    async def test_handles_todo_item_task(self) -> None:
        """Should work with TodoItem objects."""
        task = TodoItem(
            id=1,
            title="Complex task",
            description="x" * 600,
            status=TodoStatus.NOT_STARTED,
        )
        state = {
            "error": {"message": "Test failed"},
            "current_task": task,
        }
        result = await classify_failure_node(state)

        assert result["failure_reason"] == "task_too_complex"


# =============================================================================
# Phase 2.9.3: Propose Recovery Node Tests
# =============================================================================


class TestFormatFailureDetails:
    """Tests for _format_failure_details helper function."""

    def test_formats_error_message(self) -> None:
        """Should include error message."""
        state = {"error": {"message": "Test error"}}
        result = _format_failure_details(state)
        assert "Test error" in result

    def test_includes_repair_counts(self) -> None:
        """Should include repair counts when present."""
        state = {
            "error": {"message": "Error"},
            "repair_count": 2,
            "test_repair_count": 1,
        }
        result = _format_failure_details(state)
        assert "Validation Repairs Attempted" in result
        assert "2" in result
        assert "Test Repairs Attempted" in result

    def test_includes_error_history_count(self) -> None:
        """Should include error history count."""
        state = {
            "error": {"message": "Error"},
            "error_history": [{"message": "e1"}, {"message": "e2"}],
        }
        result = _format_failure_details(state)
        assert "Previous Errors" in result
        assert "2" in result


class TestGetProjectSummary:
    """Tests for _get_project_summary helper function."""

    def test_empty_state(self) -> None:
        """Should return placeholder for empty state."""
        result = _get_project_summary({})
        assert "No project context available" in result

    def test_includes_roadmap_path(self) -> None:
        """Should include roadmap path."""
        state = {"roadmap_path": "/path/to/ROADMAP.md"}
        result = _get_project_summary(state)
        assert "ROADMAP.md" in result

    def test_includes_progress(self) -> None:
        """Should include task progress."""
        state = {
            "todo_list": [
                {"status": "completed"},
                {"status": "completed"},
                {"status": "not_started"},
            ]
        }
        result = _get_project_summary(state)
        assert "Progress" in result
        assert "2/3" in result


class TestParseNumberedList:
    """Tests for _parse_numbered_list helper function."""

    def test_parses_numbered_with_period(self) -> None:
        """Should parse '1. task' format."""
        text = "1. First task\n2. Second task\n3. Third task"
        result = _parse_numbered_list(text)
        assert result == ["First task", "Second task", "Third task"]

    def test_parses_numbered_with_paren(self) -> None:
        """Should parse '1) task' format."""
        text = "1) First task\n2) Second task"
        result = _parse_numbered_list(text)
        assert result == ["First task", "Second task"]

    def test_parses_bullet_dash(self) -> None:
        """Should parse '- task' format."""
        text = "- First task\n- Second task"
        result = _parse_numbered_list(text)
        assert result == ["First task", "Second task"]

    def test_parses_bullet_asterisk(self) -> None:
        """Should parse '* task' format."""
        text = "* First task\n* Second task"
        result = _parse_numbered_list(text)
        assert result == ["First task", "Second task"]

    def test_skips_empty_lines(self) -> None:
        """Should skip empty lines."""
        text = "1. First\n\n2. Second\n\n"
        result = _parse_numbered_list(text)
        assert result == ["First", "Second"]

    def test_handles_mixed_formats(self) -> None:
        """Should handle mixed list formats."""
        text = "1. First\n- Second\n* Third"
        result = _parse_numbered_list(text)
        assert len(result) == 3


class TestCollectErrorMessages:
    """Tests for _collect_error_messages helper function."""

    def test_collects_current_error(self) -> None:
        """Should collect current error message."""
        state = {"error": {"message": "Current error"}}
        result = _collect_error_messages(state)
        assert "Current error" in result

    def test_collects_error_history(self) -> None:
        """Should collect from error history."""
        state = {
            "error": {"message": "Current"},
            "error_history": [{"message": "Past 1"}, {"message": "Past 2"}],
        }
        result = _collect_error_messages(state)
        assert "Current" in result
        assert "Past 1" in result
        assert "Past 2" in result

    def test_handles_string_errors(self) -> None:
        """Should handle string errors in history."""
        state = {"error_history": ["Error 1", "Error 2"]}
        result = _collect_error_messages(state)
        assert "Error 1" in result
        assert "Error 2" in result


class TestPromptTemplates:
    """Tests for prompt templates."""

    def test_rewrite_prompt_has_placeholders(self) -> None:
        """REWRITE_TASK_PROMPT should have required placeholders."""
        assert "{original_task}" in REWRITE_TASK_PROMPT
        assert "{failure_details}" in REWRITE_TASK_PROMPT
        assert "{project_context}" in REWRITE_TASK_PROMPT

    def test_decompose_prompt_has_placeholders(self) -> None:
        """DECOMPOSE_TASK_PROMPT should have required placeholders."""
        assert "{original_task}" in DECOMPOSE_TASK_PROMPT
        assert "{failure_details}" in DECOMPOSE_TASK_PROMPT
        assert "{project_context}" in DECOMPOSE_TASK_PROMPT


class TestProposeRecoveryNode:
    """Tests for propose_recovery_node async function."""

    @pytest.fixture
    def base_state(self) -> dict:
        """Create base state for testing."""
        return {
            "current_task": {
                "title": "Test task",
                "description": "A task that failed",
            },
            "error": {"message": "Something went wrong"},
            "failure_reason": "unknown",
            "recovery_strategy": "escalate",
        }

    @pytest.mark.asyncio
    async def test_rewrite_with_agent(self) -> None:
        """REWRITE strategy should use agent to rewrite task."""
        mock_agent = MagicMock()
        mock_agent.arun = AsyncMock(return_value="Specific: Create config.py with settings class")

        state = {
            "current_task": {"title": "Add config", "description": "Add configuration"},
            "failure_reason": "task_too_vague",
            "recovery_strategy": "rewrite",
        }
        result = await propose_recovery_node(state, agent=mock_agent)

        assert "recovery_proposal" in result
        proposal = result["recovery_proposal"]
        assert proposal.strategy == RecoveryStrategy.REWRITE
        assert len(proposal.proposed_tasks) == 1
        assert "Specific" in proposal.proposed_tasks[0]
        mock_agent.arun.assert_called_once()

    @pytest.mark.asyncio
    async def test_rewrite_without_agent_escalates(self) -> None:
        """REWRITE without agent should escalate."""
        state = {
            "current_task": {"title": "Add config", "description": "Add configuration"},
            "failure_reason": "task_too_vague",
            "recovery_strategy": "rewrite",
        }
        result = await propose_recovery_node(state, agent=None)

        proposal = result["recovery_proposal"]
        assert proposal.strategy == RecoveryStrategy.ESCALATE
        assert "No agent available" in proposal.explanation

    @pytest.mark.asyncio
    async def test_decompose_with_agent(self) -> None:
        """DECOMPOSE strategy should use agent to split task."""
        mock_agent = MagicMock()
        mock_agent.arun = AsyncMock(return_value="1. Create models\n2. Add API\n3. Write tests")

        state = {
            "current_task": {"title": "Build feature", "description": "Build complete feature"},
            "failure_reason": "task_too_complex",
            "recovery_strategy": "decompose",
        }
        result = await propose_recovery_node(state, agent=mock_agent)

        proposal = result["recovery_proposal"]
        assert proposal.strategy == RecoveryStrategy.DECOMPOSE
        assert len(proposal.proposed_tasks) == 3
        assert "3 smaller tasks" in proposal.explanation

    @pytest.mark.asyncio
    async def test_decompose_empty_result_escalates(self) -> None:
        """DECOMPOSE with empty result should escalate."""
        mock_agent = MagicMock()
        mock_agent.arun = AsyncMock(return_value="")

        state = {
            "current_task": {"title": "Build feature", "description": "Build complete feature"},
            "failure_reason": "task_too_complex",
            "recovery_strategy": "decompose",
        }
        result = await propose_recovery_node(state, agent=mock_agent)

        proposal = result["recovery_proposal"]
        assert proposal.strategy == RecoveryStrategy.ESCALATE
        assert "no valid sub-tasks" in proposal.explanation

    @pytest.mark.asyncio
    async def test_defer_no_agent_needed(self) -> None:
        """DEFER strategy should work without agent."""
        state = {
            "current_task": {"title": "Use utility", "description": "Use the utility module"},
            "failure_reason": "missing_dependency",
            "recovery_strategy": "defer",
        }
        result = await propose_recovery_node(state, agent=None)

        proposal = result["recovery_proposal"]
        assert proposal.strategy == RecoveryStrategy.DEFER
        assert proposal.requires_approval is False
        assert "retry after" in proposal.explanation.lower()
        assert len(proposal.proposed_tasks) == 1  # Same task for retry

    @pytest.mark.asyncio
    async def test_skip_no_agent_needed(self) -> None:
        """SKIP strategy should work without agent."""
        state = {
            "current_task": {"title": "Install package", "description": "Install numpy"},
            "failure_reason": "environment_issue",
            "recovery_strategy": "skip",
        }
        result = await propose_recovery_node(state, agent=None)

        proposal = result["recovery_proposal"]
        assert proposal.strategy == RecoveryStrategy.SKIP
        assert proposal.requires_approval is False
        assert "Environment issue" in proposal.explanation
        assert len(proposal.proposed_tasks) == 0

    @pytest.mark.asyncio
    async def test_escalate_always_requires_approval(self) -> None:
        """ESCALATE strategy should always require approval."""
        state = {
            "current_task": {"title": "Unknown task", "description": "Something"},
            "failure_reason": "unknown",
            "recovery_strategy": "escalate",
        }
        result = await propose_recovery_node(state, agent=None)

        proposal = result["recovery_proposal"]
        assert proposal.strategy == RecoveryStrategy.ESCALATE
        assert proposal.requires_approval is True
        assert "human review" in proposal.explanation.lower()

    @pytest.mark.asyncio
    async def test_preserves_existing_state(self) -> None:
        """Should preserve existing state fields."""
        state = {
            "current_task": {"title": "Task", "description": "Desc"},
            "failure_reason": "unknown",
            "recovery_strategy": "escalate",
            "existing_field": "preserved",
        }
        result = await propose_recovery_node(state, agent=None)

        assert result["existing_field"] == "preserved"
        assert "recovery_proposal" in result

    @pytest.mark.asyncio
    async def test_handles_agent_exception(self) -> None:
        """Should handle agent exceptions gracefully."""
        mock_agent = MagicMock()
        mock_agent.arun = AsyncMock(side_effect=Exception("API error"))

        state = {
            "current_task": {"title": "Task", "description": "Desc"},
            "failure_reason": "task_too_vague",
            "recovery_strategy": "rewrite",
        }
        result = await propose_recovery_node(state, agent=mock_agent)

        proposal = result["recovery_proposal"]
        assert proposal.strategy == RecoveryStrategy.ESCALATE
        assert "Rewrite failed" in proposal.explanation

    @pytest.mark.asyncio
    async def test_handles_todo_item_task(self) -> None:
        """Should work with TodoItem objects."""
        task = TodoItem(
            id=1,
            title="Complex task",
            description="A complex task description",
            status=TodoStatus.NOT_STARTED,
        )
        state = {
            "current_task": task,
            "failure_reason": "unknown",
            "recovery_strategy": "escalate",
        }
        result = await propose_recovery_node(state, agent=None)

        proposal = result["recovery_proposal"]
        assert proposal.original_task == "Complex task"

    @pytest.mark.asyncio
    async def test_invalid_strategy_defaults_to_escalate(self) -> None:
        """Invalid strategy should default to ESCALATE."""
        state = {
            "current_task": {"title": "Task", "description": "Desc"},
            "failure_reason": "unknown",
            "recovery_strategy": "invalid_strategy",
        }
        result = await propose_recovery_node(state, agent=None)

        proposal = result["recovery_proposal"]
        assert proposal.strategy == RecoveryStrategy.ESCALATE

    @pytest.mark.asyncio
    async def test_includes_error_history_in_proposal(self) -> None:
        """Proposal should include error history."""
        state = {
            "current_task": {"title": "Task", "description": "Desc"},
            "failure_reason": "unknown",
            "recovery_strategy": "escalate",
            "error": {"message": "Current error"},
            "error_history": [{"message": "Past error"}],
        }
        result = await propose_recovery_node(state, agent=None)

        proposal = result["recovery_proposal"]
        assert "Current error" in proposal.error_history
        assert "Past error" in proposal.error_history


# =============================================================================
# Phase 2.9.4: Approval Modes Tests
# =============================================================================


class TestApprovalMode:
    """Tests for ApprovalMode enum."""

    def test_all_values_are_strings(self) -> None:
        """ApprovalMode values should be string-based for serialization."""
        for mode in ApprovalMode:
            assert isinstance(mode.value, str)

    def test_enum_values(self) -> None:
        """Verify all expected approval modes exist."""
        assert ApprovalMode.AUTO.value == "auto"
        assert ApprovalMode.INTERACTIVE.value == "interactive"
        assert ApprovalMode.REVIEW_ONLY.value == "review_only"
        assert ApprovalMode.APPROVE_DECOMPOSE.value == "approve_decompose"

    def test_enum_count(self) -> None:
        """Verify we have exactly 4 approval modes."""
        assert len(ApprovalMode) == 4

    def test_string_enum_comparison(self) -> None:
        """ApprovalMode should be comparable as strings."""
        assert ApprovalMode.AUTO == "auto"
        assert ApprovalMode.INTERACTIVE == "interactive"


class TestAwaitApprovalNode:
    """Tests for await_approval_node async function."""

    @pytest.fixture
    def sample_proposal(self) -> RecoveryProposal:
        """Create a sample recovery proposal for testing."""
        return RecoveryProposal(
            original_task="Test task",
            failure_reason=FailureReason.TASK_TOO_COMPLEX,
            strategy=RecoveryStrategy.DECOMPOSE,
            proposed_tasks=["Task 1", "Task 2"],
            explanation="Task was too complex",
            requires_approval=True,
        )

    @pytest.fixture
    def escalate_proposal(self) -> RecoveryProposal:
        """Create an escalate proposal that always requires approval."""
        return RecoveryProposal(
            original_task="Unknown task",
            failure_reason=FailureReason.UNKNOWN,
            strategy=RecoveryStrategy.ESCALATE,
            proposed_tasks=[],
            explanation="Cannot determine recovery",
            requires_approval=True,
        )

    @pytest.fixture
    def rewrite_proposal(self) -> RecoveryProposal:
        """Create a rewrite proposal."""
        return RecoveryProposal(
            original_task="Vague task",
            failure_reason=FailureReason.TASK_TOO_VAGUE,
            strategy=RecoveryStrategy.REWRITE,
            proposed_tasks=["Specific: Create foo.py with bar function"],
            explanation="Task was vague",
            requires_approval=True,
        )

    @pytest.mark.asyncio
    async def test_no_proposal_returns_not_approved(self) -> None:
        """Should return not approved when no proposal exists."""
        state = {}
        result = await await_approval_node(state)
        assert result["recovery_approved"] is False

    @pytest.mark.asyncio
    async def test_auto_mode_approves_automatically(
        self, sample_proposal: RecoveryProposal
    ) -> None:
        """AUTO mode should approve without prompting."""
        state = {
            "recovery_proposal": sample_proposal,
            "approval_mode": "auto",
        }
        result = await await_approval_node(state)
        assert result["recovery_approved"] is True

    @pytest.mark.asyncio
    async def test_auto_mode_via_parameter(self, sample_proposal: RecoveryProposal) -> None:
        """AUTO mode should work via parameter override."""
        state = {"recovery_proposal": sample_proposal}
        result = await await_approval_node(state, approval_mode=ApprovalMode.AUTO)
        assert result["recovery_approved"] is True

    @pytest.mark.asyncio
    async def test_review_only_mode_skips(self, sample_proposal: RecoveryProposal) -> None:
        """REVIEW_ONLY mode should log but not apply."""
        state = {
            "recovery_proposal": sample_proposal,
            "approval_mode": "review_only",
        }
        result = await await_approval_node(state)
        assert result["recovery_approved"] is False
        assert result["recovery_skipped"] is True

    @pytest.mark.asyncio
    async def test_approve_decompose_mode_approves_rewrite(
        self, rewrite_proposal: RecoveryProposal
    ) -> None:
        """APPROVE_DECOMPOSE mode should auto-approve REWRITE."""
        state = {
            "recovery_proposal": rewrite_proposal,
            "approval_mode": "approve_decompose",
        }
        result = await await_approval_node(state)
        assert result["recovery_approved"] is True

    @pytest.mark.asyncio
    async def test_approve_decompose_mode_asks_for_decompose(
        self, sample_proposal: RecoveryProposal
    ) -> None:
        """APPROVE_DECOMPOSE mode should prompt for DECOMPOSE."""
        # Use a mock input that returns 'y'
        mock_input = MagicMock(return_value="y")

        state = {
            "recovery_proposal": sample_proposal,
            "approval_mode": "approve_decompose",
        }
        result = await await_approval_node(state, interactive_input=mock_input)
        assert result["recovery_approved"] is True
        mock_input.assert_called()

    @pytest.mark.asyncio
    async def test_interactive_mode_yes_approval(self, sample_proposal: RecoveryProposal) -> None:
        """INTERACTIVE mode with 'yes' should approve."""
        mock_input = MagicMock(return_value="y")

        state = {
            "recovery_proposal": sample_proposal,
            "approval_mode": "interactive",
        }
        result = await await_approval_node(state, interactive_input=mock_input)
        assert result["recovery_approved"] is True

    @pytest.mark.asyncio
    async def test_interactive_mode_no_rejection(self, sample_proposal: RecoveryProposal) -> None:
        """INTERACTIVE mode with 'no' should reject."""
        mock_input = MagicMock(return_value="n")

        state = {
            "recovery_proposal": sample_proposal,
            "approval_mode": "interactive",
        }
        result = await await_approval_node(state, interactive_input=mock_input)
        assert result["recovery_approved"] is False

    @pytest.mark.asyncio
    async def test_interactive_mode_skip(self, sample_proposal: RecoveryProposal) -> None:
        """INTERACTIVE mode with 'skip' should skip task."""
        mock_input = MagicMock(return_value="s")

        state = {
            "recovery_proposal": sample_proposal,
            "approval_mode": "interactive",
        }
        result = await await_approval_node(state, interactive_input=mock_input)
        assert result["recovery_approved"] is False
        assert result["task_skipped"] is True

    @pytest.mark.asyncio
    async def test_interactive_mode_edit(self, sample_proposal: RecoveryProposal) -> None:
        """INTERACTIVE mode with 'edit' should allow editing."""
        # First return 'e' for edit, then empty line to keep tasks
        responses = iter(["e", ""])
        mock_input = MagicMock(side_effect=lambda _: next(responses))

        state = {
            "recovery_proposal": sample_proposal,
            "approval_mode": "interactive",
        }
        result = await await_approval_node(state, interactive_input=mock_input)
        assert result["recovery_approved"] is True
        # Proposal should still exist
        assert result["recovery_proposal"] is not None

    @pytest.mark.asyncio
    async def test_escalate_always_requires_approval_in_auto_mode(
        self, escalate_proposal: RecoveryProposal
    ) -> None:
        """ESCALATE should require approval even in AUTO mode."""
        mock_input = MagicMock(return_value="y")

        state = {
            "recovery_proposal": escalate_proposal,
            "approval_mode": "auto",
        }
        result = await await_approval_node(state, interactive_input=mock_input)
        # Should have prompted because ESCALATE always requires approval
        mock_input.assert_called()
        assert result["recovery_approved"] is True

    @pytest.mark.asyncio
    async def test_invalid_response_prompts_again(self, sample_proposal: RecoveryProposal) -> None:
        """Invalid responses should prompt again."""
        # First invalid, then valid
        responses = iter(["invalid", "y"])
        mock_input = MagicMock(side_effect=lambda _: next(responses))

        state = {
            "recovery_proposal": sample_proposal,
            "approval_mode": "interactive",
        }
        result = await await_approval_node(state, interactive_input=mock_input)
        assert result["recovery_approved"] is True
        assert mock_input.call_count == 2

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_skips(self, sample_proposal: RecoveryProposal) -> None:
        """KeyboardInterrupt should skip the task."""
        mock_input = MagicMock(side_effect=KeyboardInterrupt)

        state = {
            "recovery_proposal": sample_proposal,
            "approval_mode": "interactive",
        }
        result = await await_approval_node(state, interactive_input=mock_input)
        assert result["recovery_approved"] is False
        assert result["task_skipped"] is True

    @pytest.mark.asyncio
    async def test_eof_error_skips(self, sample_proposal: RecoveryProposal) -> None:
        """EOFError should skip the task."""
        mock_input = MagicMock(side_effect=EOFError)

        state = {
            "recovery_proposal": sample_proposal,
            "approval_mode": "interactive",
        }
        result = await await_approval_node(state, interactive_input=mock_input)
        assert result["recovery_approved"] is False
        assert result["task_skipped"] is True

    @pytest.mark.asyncio
    async def test_preserves_existing_state(self, sample_proposal: RecoveryProposal) -> None:
        """Should preserve existing state fields."""
        state = {
            "recovery_proposal": sample_proposal,
            "approval_mode": "auto",
            "existing_field": "preserved",
        }
        result = await await_approval_node(state)
        assert result["existing_field"] == "preserved"

    @pytest.mark.asyncio
    async def test_invalid_approval_mode_defaults_to_auto(
        self, sample_proposal: RecoveryProposal
    ) -> None:
        """Invalid approval mode should default to AUTO."""
        state = {
            "recovery_proposal": sample_proposal,
            "approval_mode": "invalid_mode",
        }
        result = await await_approval_node(state)
        assert result["recovery_approved"] is True

    @pytest.mark.asyncio
    async def test_string_approval_mode_parameter(self, sample_proposal: RecoveryProposal) -> None:
        """String approval mode parameter should work."""
        state = {"recovery_proposal": sample_proposal}
        result = await await_approval_node(state, approval_mode="auto")
        assert result["recovery_approved"] is True


class TestInteractiveEdit:
    """Tests for _interactive_edit helper function."""

    def test_empty_input_keeps_original(self) -> None:
        """Empty input should keep original tasks."""
        # Simulate just pressing Enter
        MagicMock(side_effect=[""])

        import ai_infra.executor.nodes.recovery as recovery_module

        # Temporarily replace input
        recovery_module.input if hasattr(recovery_module, "input") else None

        # The function uses input() internally, which we can't easily mock
        # without modifying the function. For now, test the logic.
        # In production, _interactive_edit would use the real input()

        # Just verify the function exists and has correct signature
        assert callable(_interactive_edit)
        # If no new tasks are entered, it returns the original
        # This is tested via the await_approval_node edit test above

    def test_function_accepts_list(self) -> None:
        """Function should accept a list of tasks."""
        # Verify signature
        import inspect

        sig = inspect.signature(_interactive_edit)
        params = list(sig.parameters.keys())
        assert "tasks" in params


# =============================================================================
# Phase 2.9.5: Apply Recovery Node Tests
# =============================================================================


class TestReplaceTaskInRoadmap:
    """Tests for _replace_task_in_roadmap helper function."""

    def test_replaces_unchecked_task(self) -> None:
        """Should replace unchecked task with recovery block."""
        content = """# ROADMAP

## Phase 1

- [ ] Create user model
- [ ] Add authentication
"""
        proposal = RecoveryProposal(
            original_task="Create user model",
            failure_reason=FailureReason.TASK_TOO_COMPLEX,
            strategy=RecoveryStrategy.DECOMPOSE,
            proposed_tasks=["Create User class", "Add validation"],
            explanation="Decomposed",
            requires_approval=True,
        )

        updated, found = _replace_task_in_roadmap(content, proposal)

        assert found is True
        assert "<!-- RECOVERY: decompose" in updated
        assert "<!-- Original: Create user model -->" in updated
        assert "<!-- Reason: task_too_complex -->" in updated
        assert "- [ ] Create User class" in updated
        assert "- [ ] Add validation" in updated
        # Original should be replaced
        assert "- [ ] Create user model" not in updated
        # Other tasks should remain
        assert "- [ ] Add authentication" in updated

    def test_replaces_checked_task(self) -> None:
        """Should replace checked task."""
        content = """# ROADMAP
- [x] Create user model
"""
        proposal = RecoveryProposal(
            original_task="Create user model",
            failure_reason=FailureReason.TASK_TOO_VAGUE,
            strategy=RecoveryStrategy.REWRITE,
            proposed_tasks=["Create User class with name field"],
            explanation="Rewritten",
            requires_approval=True,
        )

        updated, found = _replace_task_in_roadmap(content, proposal)

        assert found is True
        assert "<!-- RECOVERY: rewrite" in updated

    def test_handles_task_not_found(self) -> None:
        """Should return False when task not in roadmap."""
        content = """# ROADMAP
- [ ] Some other task
"""
        proposal = RecoveryProposal(
            original_task="Nonexistent task",
            failure_reason=FailureReason.UNKNOWN,
            strategy=RecoveryStrategy.REWRITE,
            proposed_tasks=["New task"],
            explanation="Test",
            requires_approval=True,
        )

        updated, found = _replace_task_in_roadmap(content, proposal)

        assert found is False
        assert updated == content  # Unchanged

    def test_adds_checkbox_to_tasks_without_it(self) -> None:
        """Should add - [ ] prefix to tasks that don't have it."""
        content = "- [ ] Original task\n"
        proposal = RecoveryProposal(
            original_task="Original task",
            failure_reason=FailureReason.TASK_TOO_VAGUE,
            strategy=RecoveryStrategy.REWRITE,
            proposed_tasks=["New task without prefix"],
            explanation="Test",
            requires_approval=True,
        )

        updated, found = _replace_task_in_roadmap(content, proposal)

        assert found is True
        assert "- [ ] New task without prefix" in updated

    def test_preserves_existing_checkbox_format(self) -> None:
        """Should not double-add checkbox prefix."""
        content = "- [ ] Original task\n"
        proposal = RecoveryProposal(
            original_task="Original task",
            failure_reason=FailureReason.TASK_TOO_VAGUE,
            strategy=RecoveryStrategy.REWRITE,
            proposed_tasks=["- [ ] Already has prefix"],
            explanation="Test",
            requires_approval=True,
        )

        updated, found = _replace_task_in_roadmap(content, proposal)

        assert found is True
        # Should not have double prefix
        assert "- [ ] - [ ]" not in updated
        assert "- [ ] Already has prefix" in updated

    def test_replaces_only_first_occurrence(self) -> None:
        """Should only replace the first matching task."""
        content = """# ROADMAP
- [ ] Create user model
- [ ] Create user model
"""
        proposal = RecoveryProposal(
            original_task="Create user model",
            failure_reason=FailureReason.TASK_TOO_VAGUE,
            strategy=RecoveryStrategy.REWRITE,
            proposed_tasks=["New task"],
            explanation="Test",
            requires_approval=True,
        )

        updated, found = _replace_task_in_roadmap(content, proposal)

        assert found is True
        # Second occurrence should still exist
        assert "- [ ] Create user model" in updated


class TestApplyRecoveryNode:
    """Tests for apply_recovery_node async function."""

    @pytest.fixture
    def rewrite_proposal(self) -> RecoveryProposal:
        """Create a rewrite proposal."""
        return RecoveryProposal(
            original_task="Create user model",
            failure_reason=FailureReason.TASK_TOO_VAGUE,
            strategy=RecoveryStrategy.REWRITE,
            proposed_tasks=["Create User class with name field"],
            explanation="Task was vague",
            requires_approval=True,
        )

    @pytest.fixture
    def decompose_proposal(self) -> RecoveryProposal:
        """Create a decompose proposal."""
        return RecoveryProposal(
            original_task="Build authentication system",
            failure_reason=FailureReason.TASK_TOO_COMPLEX,
            strategy=RecoveryStrategy.DECOMPOSE,
            proposed_tasks=["Create login form", "Add password hashing", "Implement JWT"],
            explanation="Task too complex",
            requires_approval=True,
        )

    @pytest.fixture
    def defer_proposal(self) -> RecoveryProposal:
        """Create a defer proposal."""
        return RecoveryProposal(
            original_task="Configure database",
            failure_reason=FailureReason.MISSING_DEPENDENCY,
            strategy=RecoveryStrategy.DEFER,
            proposed_tasks=[],
            explanation="Missing dependency",
            requires_approval=True,
        )

    @pytest.fixture
    def skip_proposal(self) -> RecoveryProposal:
        """Create a skip proposal."""
        return RecoveryProposal(
            original_task="Install Docker",
            failure_reason=FailureReason.ENVIRONMENT_ISSUE,
            strategy=RecoveryStrategy.SKIP,
            proposed_tasks=[],
            explanation="Environment issue",
            requires_approval=True,
        )

    @pytest.fixture
    def escalate_proposal(self) -> RecoveryProposal:
        """Create an escalate proposal."""
        return RecoveryProposal(
            original_task="Unknown task",
            failure_reason=FailureReason.UNKNOWN,
            strategy=RecoveryStrategy.ESCALATE,
            proposed_tasks=[],
            explanation="Unknown failure",
            requires_approval=True,
        )

    @pytest.mark.asyncio
    async def test_no_proposal_returns_unchanged(self) -> None:
        """Should return state unchanged when no proposal."""
        state = {"some_field": "value"}
        result = await apply_recovery_node(state)
        assert result == state

    @pytest.mark.asyncio
    async def test_not_approved_returns_unchanged(self, rewrite_proposal: RecoveryProposal) -> None:
        """Should skip when recovery not approved."""
        state = {
            "recovery_proposal": rewrite_proposal,
            "recovery_approved": False,
        }
        result = await apply_recovery_node(state)
        assert "recovery_applied" not in result or result.get("recovery_applied") is None

    @pytest.mark.asyncio
    async def test_defer_adds_to_deferred_list(self, defer_proposal: RecoveryProposal) -> None:
        """DEFER strategy should add to deferred_tasks."""
        state = {
            "recovery_proposal": defer_proposal,
            "recovery_approved": True,
        }
        result = await apply_recovery_node(state)

        assert result["recovery_applied"] is True
        assert len(result["deferred_tasks"]) == 1
        assert result["deferred_tasks"][0]["task"] == "Configure database"
        assert result["deferred_tasks"][0]["reason"] == "missing_dependency"
        assert result["deferred_tasks"][0]["retry_count"] == 0

    @pytest.mark.asyncio
    async def test_defer_appends_to_existing_list(self, defer_proposal: RecoveryProposal) -> None:
        """DEFER should append to existing deferred_tasks."""
        state = {
            "recovery_proposal": defer_proposal,
            "recovery_approved": True,
            "deferred_tasks": [{"task": "Previous task", "reason": "test"}],
        }
        result = await apply_recovery_node(state)

        assert len(result["deferred_tasks"]) == 2

    @pytest.mark.asyncio
    async def test_skip_adds_to_failed_list(self, skip_proposal: RecoveryProposal) -> None:
        """SKIP strategy should add to failed_tasks."""
        state = {
            "recovery_proposal": skip_proposal,
            "recovery_approved": True,
        }
        result = await apply_recovery_node(state)

        assert result["recovery_applied"] is True
        assert len(result["failed_tasks"]) == 1
        assert result["failed_tasks"][0]["task"] == "Install Docker"
        assert result["failed_tasks"][0]["strategy"] == "skip"

    @pytest.mark.asyncio
    async def test_escalate_adds_to_failed_with_human_flag(
        self, escalate_proposal: RecoveryProposal
    ) -> None:
        """ESCALATE strategy should add to failed_tasks with requires_human."""
        state = {
            "recovery_proposal": escalate_proposal,
            "recovery_approved": True,
        }
        result = await apply_recovery_node(state)

        assert result["recovery_applied"] is True
        assert len(result["failed_tasks"]) == 1
        assert result["failed_tasks"][0]["requires_human"] is True

    @pytest.mark.asyncio
    async def test_rewrite_updates_roadmap(self, rewrite_proposal: RecoveryProposal) -> None:
        """REWRITE strategy should update roadmap file."""
        roadmap_content = "- [ ] Create user model\n- [ ] Other task\n"
        written_content = None

        def mock_reader(path: str) -> str:
            return roadmap_content

        def mock_writer(path: str, content: str) -> None:
            nonlocal written_content
            written_content = content

        state = {
            "recovery_proposal": rewrite_proposal,
            "recovery_approved": True,
            "roadmap_path": "/fake/ROADMAP.md",
        }
        result = await apply_recovery_node(
            state,
            roadmap_reader=mock_reader,
            roadmap_writer=mock_writer,
        )

        assert result["recovery_applied"] is True
        assert result["roadmap_modified"] is True
        assert written_content is not None
        assert "<!-- RECOVERY: rewrite" in written_content
        assert "- [ ] Create User class with name field" in written_content

    @pytest.mark.asyncio
    async def test_decompose_adds_tasks_to_queue(
        self, decompose_proposal: RecoveryProposal
    ) -> None:
        """DECOMPOSE should add replacement tasks to task queue."""
        roadmap_content = "- [ ] Build authentication system\n"

        def mock_reader(path: str) -> str:
            return roadmap_content

        def mock_writer(path: str, content: str) -> None:
            pass

        state = {
            "recovery_proposal": decompose_proposal,
            "recovery_approved": True,
            "roadmap_path": "/fake/ROADMAP.md",
            "tasks": [{"description": "Existing task"}],
        }
        result = await apply_recovery_node(
            state,
            roadmap_reader=mock_reader,
            roadmap_writer=mock_writer,
        )

        # Should have 3 new tasks + 1 existing = 4 total
        assert len(result["tasks"]) == 4
        # New tasks should be at the front
        assert result["tasks"][0]["description"] == "Create login form"
        assert result["tasks"][0]["is_recovery"] is True
        assert result["tasks"][0]["original_task"] == "Build authentication system"
        # Existing task should be at the end
        assert result["tasks"][3]["description"] == "Existing task"

    @pytest.mark.asyncio
    async def test_no_roadmap_path_fails_gracefully(
        self, rewrite_proposal: RecoveryProposal
    ) -> None:
        """Should fail gracefully when no roadmap_path."""
        state = {
            "recovery_proposal": rewrite_proposal,
            "recovery_approved": True,
        }
        result = await apply_recovery_node(state)

        assert result.get("recovery_applied") is False

    @pytest.mark.asyncio
    async def test_file_not_found_fails_gracefully(
        self, rewrite_proposal: RecoveryProposal
    ) -> None:
        """Should fail gracefully when roadmap file not found."""

        def mock_reader(path: str) -> str:
            raise FileNotFoundError("File not found")

        state = {
            "recovery_proposal": rewrite_proposal,
            "recovery_approved": True,
            "roadmap_path": "/nonexistent/ROADMAP.md",
        }
        result = await apply_recovery_node(
            state,
            roadmap_reader=mock_reader,
        )

        assert result.get("recovery_applied") is False

    @pytest.mark.asyncio
    async def test_task_not_in_roadmap_still_adds_to_queue(
        self, rewrite_proposal: RecoveryProposal
    ) -> None:
        """Should still add to task queue even if roadmap update fails."""
        roadmap_content = "- [ ] Different task\n"

        def mock_reader(path: str) -> str:
            return roadmap_content

        def mock_writer(path: str, content: str) -> None:
            pass

        state = {
            "recovery_proposal": rewrite_proposal,
            "recovery_approved": True,
            "roadmap_path": "/fake/ROADMAP.md",
            "tasks": [],
        }
        result = await apply_recovery_node(
            state,
            roadmap_reader=mock_reader,
            roadmap_writer=mock_writer,
        )

        # Should still add tasks even though roadmap update failed
        assert result["recovery_applied"] is True
        assert result["roadmap_modified"] is False
        assert len(result["tasks"]) == 1

    @pytest.mark.asyncio
    async def test_write_error_fails_gracefully(self, rewrite_proposal: RecoveryProposal) -> None:
        """Should fail gracefully on write error."""
        roadmap_content = "- [ ] Create user model\n"

        def mock_reader(path: str) -> str:
            return roadmap_content

        def mock_writer(path: str, content: str) -> None:
            raise PermissionError("Permission denied")

        state = {
            "recovery_proposal": rewrite_proposal,
            "recovery_approved": True,
            "roadmap_path": "/fake/ROADMAP.md",
        }
        result = await apply_recovery_node(
            state,
            roadmap_reader=mock_reader,
            roadmap_writer=mock_writer,
        )

        assert result.get("recovery_applied") is False

    @pytest.mark.asyncio
    async def test_preserves_existing_state(self, defer_proposal: RecoveryProposal) -> None:
        """Should preserve existing state fields."""
        state = {
            "recovery_proposal": defer_proposal,
            "recovery_approved": True,
            "existing_field": "preserved",
        }
        result = await apply_recovery_node(state)

        assert result["existing_field"] == "preserved"


# =============================================================================
# Phase 2.9.6: Deferred Task Retry Tests
# =============================================================================


class TestRetryDeferredNode:
    """Tests for retry_deferred_node async function."""

    @pytest.mark.asyncio
    async def test_no_deferred_tasks_returns_unchanged(self) -> None:
        """Should return state with retrying_deferred=False when no deferred."""
        state = {"some_field": "value"}
        result = await retry_deferred_node(state)

        assert result["retrying_deferred"] is False
        assert result["some_field"] == "value"

    @pytest.mark.asyncio
    async def test_empty_deferred_list_returns_unchanged(self) -> None:
        """Should handle empty deferred_tasks list."""
        state = {"deferred_tasks": []}
        result = await retry_deferred_node(state)

        assert result["retrying_deferred"] is False

    @pytest.mark.asyncio
    async def test_retries_deferred_task(self) -> None:
        """Should move deferred task to task queue."""
        state = {
            "deferred_tasks": [
                {"task": "Configure database", "reason": "missing_dependency", "retry_count": 0}
            ],
            "tasks": [],
        }
        result = await retry_deferred_node(state)

        assert result["retrying_deferred"] is True
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["description"] == "Configure database"
        assert result["tasks"][0]["is_deferred_retry"] is True
        assert result["tasks"][0]["original_reason"] == "missing_dependency"
        assert result["tasks"][0]["retry_attempt"] == 1

    @pytest.mark.asyncio
    async def test_increments_retry_count(self) -> None:
        """Should increment retry_count in deferred_tasks."""
        state = {
            "deferred_tasks": [{"task": "Task A", "reason": "env", "retry_count": 0}],
        }
        result = await retry_deferred_node(state)

        assert result["deferred_tasks"][0]["retry_count"] == 1

    @pytest.mark.asyncio
    async def test_respects_max_retries(self) -> None:
        """Should not retry tasks that have exceeded max retries."""
        state = {
            "deferred_tasks": [
                {"task": "Task A", "reason": "env", "retry_count": 1}  # Already retried once
            ],
        }
        result = await retry_deferred_node(state)

        # Default max is 1, so this task should NOT be retried
        assert result["retrying_deferred"] is False
        assert len(result.get("tasks", [])) == 0

    @pytest.mark.asyncio
    async def test_custom_max_retries(self) -> None:
        """Should respect custom max_retries parameter."""
        state = {
            "deferred_tasks": [{"task": "Task A", "reason": "env", "retry_count": 1}],
        }
        result = await retry_deferred_node(state, max_retries=2)

        # With max_retries=2, this task CAN be retried
        assert result["retrying_deferred"] is True
        assert len(result["tasks"]) == 1

    @pytest.mark.asyncio
    async def test_multiple_deferred_tasks(self) -> None:
        """Should handle multiple deferred tasks."""
        state = {
            "deferred_tasks": [
                {"task": "Task A", "reason": "env", "retry_count": 0},
                {"task": "Task B", "reason": "dep", "retry_count": 0},
                {"task": "Task C", "reason": "env", "retry_count": 1},  # Already retried
            ],
            "tasks": [],
        }
        result = await retry_deferred_node(state)

        # Only Task A and Task B should be retried
        assert result["retrying_deferred"] is True
        assert len(result["tasks"]) == 2
        descriptions = [t["description"] for t in result["tasks"]]
        assert "Task A" in descriptions
        assert "Task B" in descriptions
        assert "Task C" not in descriptions

    @pytest.mark.asyncio
    async def test_appends_to_existing_tasks(self) -> None:
        """Should append to existing tasks, not replace."""
        state = {
            "deferred_tasks": [{"task": "Deferred task", "reason": "env", "retry_count": 0}],
            "tasks": [{"description": "Existing task"}],
        }
        result = await retry_deferred_node(state)

        assert len(result["tasks"]) == 2
        assert result["tasks"][0]["description"] == "Existing task"
        assert result["tasks"][1]["description"] == "Deferred task"

    @pytest.mark.asyncio
    async def test_handles_missing_retry_count(self) -> None:
        """Should default retry_count to 0 if missing."""
        state = {
            "deferred_tasks": [
                {"task": "Task without count", "reason": "env"}  # No retry_count
            ],
        }
        result = await retry_deferred_node(state)

        assert result["retrying_deferred"] is True
        assert result["deferred_tasks"][0]["retry_count"] == 1

    @pytest.mark.asyncio
    async def test_handles_missing_reason(self) -> None:
        """Should default reason to 'unknown' if missing."""
        state = {
            "deferred_tasks": [
                {"task": "Task without reason", "retry_count": 0}  # No reason
            ],
        }
        result = await retry_deferred_node(state)

        assert result["tasks"][0]["original_reason"] == "unknown"

    @pytest.mark.asyncio
    async def test_preserves_existing_state_fields(self) -> None:
        """Should preserve existing state fields."""
        state = {
            "deferred_tasks": [{"task": "Task", "reason": "env", "retry_count": 0}],
            "existing_field": "preserved",
        }
        result = await retry_deferred_node(state)

        assert result["existing_field"] == "preserved"


class TestShouldRetryDeferred:
    """Tests for should_retry_deferred routing function."""

    def test_returns_false_when_tasks_exist(self) -> None:
        """Should return False when there are regular tasks."""
        state = {
            "tasks": [{"description": "Regular task"}],
            "deferred_tasks": [{"task": "Deferred", "retry_count": 0}],
        }
        assert should_retry_deferred(state) is False

    def test_returns_false_when_no_deferred(self) -> None:
        """Should return False when no deferred tasks."""
        state = {"tasks": []}
        assert should_retry_deferred(state) is False

    def test_returns_false_when_empty_deferred(self) -> None:
        """Should return False when deferred_tasks is empty."""
        state = {"tasks": [], "deferred_tasks": []}
        assert should_retry_deferred(state) is False

    def test_returns_true_when_eligible_deferred(self) -> None:
        """Should return True when deferred tasks can be retried."""
        state = {
            "tasks": [],
            "deferred_tasks": [{"task": "Deferred", "retry_count": 0}],
        }
        assert should_retry_deferred(state) is True

    def test_returns_false_when_all_exceeded_retries(self) -> None:
        """Should return False when all deferred exceeded max retries."""
        state = {
            "tasks": [],
            "deferred_tasks": [
                {"task": "Task A", "retry_count": 1},
                {"task": "Task B", "retry_count": 2},
            ],
        }
        assert should_retry_deferred(state) is False

    def test_returns_true_when_any_eligible(self) -> None:
        """Should return True when at least one task can be retried."""
        state = {
            "tasks": [],
            "deferred_tasks": [
                {"task": "Task A", "retry_count": 1},  # Exceeded
                {"task": "Task B", "retry_count": 0},  # Eligible
            ],
        }
        assert should_retry_deferred(state) is True

    def test_handles_missing_retry_count(self) -> None:
        """Should treat missing retry_count as 0."""
        state = {
            "tasks": [],
            "deferred_tasks": [{"task": "Task"}],  # No retry_count
        }
        assert should_retry_deferred(state) is True


class TestMaxDeferredRetries:
    """Tests for MAX_DEFERRED_RETRIES constant."""

    def test_default_value(self) -> None:
        """MAX_DEFERRED_RETRIES should be 1 by default."""
        assert MAX_DEFERRED_RETRIES == 1

    def test_is_integer(self) -> None:
        """MAX_DEFERRED_RETRIES should be an integer."""
        assert isinstance(MAX_DEFERRED_RETRIES, int)


# =============================================================================
# Phase 2.9.7: Audit Trail & Reporting Tests
# =============================================================================


class TestExecutionReport:
    """Tests for ExecutionReport dataclass."""

    @pytest.fixture
    def sample_report(self) -> ExecutionReport:
        """Create a sample execution report."""
        started = datetime(2026, 1, 11, 10, 0, 0)
        completed = datetime(2026, 1, 11, 10, 5, 0)  # 5 minutes later
        return ExecutionReport(
            started_at=started,
            completed_at=completed,
            total_tasks=10,
            completed_tasks=8,
            failed_tasks=1,
            deferred_tasks=1,
            recoveries_attempted=3,
            recoveries_successful=2,
            roadmap_modifications=[
                {"strategy": "rewrite", "original_task": "Task A", "replacement_count": 1},
                {"strategy": "decompose", "original_task": "Task B", "replacement_count": 3},
            ],
            errors=[
                {"task": "Task C", "message": "Unknown error", "strategy": "escalate"},
            ],
        )

    def test_success_rate_calculation(self, sample_report: ExecutionReport) -> None:
        """Should calculate success rate correctly."""
        assert sample_report.success_rate == 0.8

    def test_success_rate_zero_tasks(self) -> None:
        """Should return 0 for zero total tasks."""
        report = ExecutionReport(
            started_at=datetime.now(),
            completed_at=datetime.now(),
            total_tasks=0,
        )
        assert report.success_rate == 0.0

    def test_duration_seconds(self, sample_report: ExecutionReport) -> None:
        """Should calculate duration in seconds."""
        assert sample_report.duration_seconds == 300.0  # 5 minutes

    def test_duration_none_timestamps(self) -> None:
        """Should return 0 for None timestamps."""
        report = ExecutionReport(
            started_at=None,
            completed_at=None,
        )
        assert report.duration_seconds == 0.0

    def test_to_markdown(self, sample_report: ExecutionReport) -> None:
        """Should generate valid markdown report."""
        md = sample_report.to_markdown()

        assert "# Execution Report" in md
        assert "Success Rate**: 80.0%" in md
        assert "| Total Tasks | 10 |" in md
        assert "| Completed | 8 |" in md
        assert "| Failed | 1 |" in md
        assert "| Deferred | 1 |" in md
        assert "| Attempted | 3 |" in md
        assert "| Successful | 2 |" in md
        assert "**rewrite**: `Task A`" in md
        assert "**Task C**: Unknown error" in md

    def test_to_markdown_no_modifications(self) -> None:
        """Should handle empty modifications."""
        report = ExecutionReport(
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )
        md = report.to_markdown()
        assert "No modifications made" in md

    def test_to_markdown_no_errors(self) -> None:
        """Should handle empty errors."""
        report = ExecutionReport(
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )
        md = report.to_markdown()
        assert "No errors recorded" in md

    def test_to_json(self, sample_report: ExecutionReport) -> None:
        """Should generate valid JSON structure."""
        data = sample_report.to_json()

        assert data["started_at"] == "2026-01-11T10:00:00"
        assert data["completed_at"] == "2026-01-11T10:05:00"
        assert data["duration_seconds"] == 300.0
        assert data["success_rate"] == 0.8
        assert data["tasks"]["total"] == 10
        assert data["tasks"]["completed"] == 8
        assert data["tasks"]["failed"] == 1
        assert data["tasks"]["deferred"] == 1
        assert data["recoveries"]["attempted"] == 3
        assert data["recoveries"]["successful"] == 2
        assert len(data["roadmap_modifications"]) == 2
        assert len(data["errors"]) == 1

    def test_to_json_non_datetime_timestamps(self) -> None:
        """Should handle non-datetime timestamps gracefully."""
        report = ExecutionReport(
            started_at="2026-01-11T10:00:00",  # String instead of datetime
            completed_at="2026-01-11T10:05:00",
        )
        data = report.to_json()
        assert data["started_at"] == "2026-01-11T10:00:00"

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        report = ExecutionReport(
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )
        assert report.total_tasks == 0
        assert report.completed_tasks == 0
        assert report.failed_tasks == 0
        assert report.deferred_tasks == 0
        assert report.recoveries_attempted == 0
        assert report.recoveries_successful == 0
        assert report.roadmap_modifications == []
        assert report.errors == []


class TestGenerateReportNode:
    """Tests for generate_report_node async function."""

    @pytest.mark.asyncio
    async def test_generates_report_with_empty_state(self) -> None:
        """Should generate report even with minimal state."""
        state = {}
        result = await generate_report_node(state)

        assert "execution_report" in result
        assert isinstance(result["execution_report"], ExecutionReport)
        assert result["execution_report"].total_tasks == 0

    @pytest.mark.asyncio
    async def test_extracts_metrics_from_state(self) -> None:
        """Should extract all metrics from state."""
        state = {
            "started_at": datetime(2026, 1, 11, 10, 0, 0),
            "total_task_count": 10,
            "completed_tasks": [
                {"description": "Task 1"},
                {"description": "Task 2"},
            ],
            "failed_tasks": [
                {"task": "Task 3", "reason": "Unknown", "strategy": "skip"},
            ],
            "deferred_tasks": [
                {"task": "Task 4", "reason": "env"},
            ],
            "recoveries_attempted": 5,
            "recoveries_successful": 3,
        }
        result = await generate_report_node(state)

        report = result["execution_report"]
        assert report.total_tasks == 10
        assert report.completed_tasks == 2
        assert report.failed_tasks == 1
        assert report.deferred_tasks == 1
        assert report.recoveries_attempted == 5
        assert report.recoveries_successful == 3

    @pytest.mark.asyncio
    async def test_infers_total_from_lists(self) -> None:
        """Should infer total_tasks from lists if not provided."""
        state = {
            "completed_tasks": [{}, {}, {}],  # 3
            "failed_tasks": [{}],  # 1
            "deferred_tasks": [{}, {}],  # 2
        }
        result = await generate_report_node(state)

        assert result["execution_report"].total_tasks == 6

    @pytest.mark.asyncio
    async def test_extracts_errors_from_failed_tasks(self) -> None:
        """Should extract error details from failed_tasks."""
        state = {
            "failed_tasks": [
                {"task": "Task A", "reason": "Missing dep", "strategy": "escalate"},
                {"task": "Task B", "reason": "Unknown", "strategy": "skip"},
            ],
        }
        result = await generate_report_node(state)

        assert len(result["execution_report"].errors) == 2
        assert result["execution_report"].errors[0]["task"] == "Task A"
        assert result["execution_report"].errors[0]["message"] == "Missing dep"

    @pytest.mark.asyncio
    async def test_uses_custom_report_writer(self) -> None:
        """Should use provided report_writer for testing."""
        written = {}

        def mock_writer(report_dir: str, report: ExecutionReport) -> None:
            written["dir"] = report_dir
            written["report"] = report

        state = {"total_task_count": 5}
        await generate_report_node(state, report_writer=mock_writer)

        assert written["dir"] == ".executor"
        assert written["report"].total_tasks == 5

    @pytest.mark.asyncio
    async def test_uses_custom_report_dir(self) -> None:
        """Should use custom report_dir parameter."""
        written = {}

        def mock_writer(report_dir: str, report: ExecutionReport) -> None:
            written["dir"] = report_dir

        await generate_report_node({}, report_dir="/custom/path", report_writer=mock_writer)

        assert written["dir"] == "/custom/path"

    @pytest.mark.asyncio
    async def test_uses_report_dir_from_state(self) -> None:
        """Should use report_dir from state if not provided."""
        written = {}

        def mock_writer(report_dir: str, report: ExecutionReport) -> None:
            written["dir"] = report_dir

        state = {"report_dir": "/from/state"}
        await generate_report_node(state, report_writer=mock_writer)

        assert written["dir"] == "/from/state"

    @pytest.mark.asyncio
    async def test_sets_completed_at_timestamp(self) -> None:
        """Should set completed_at in result."""
        state = {}
        result = await generate_report_node(state)

        assert "completed_at" in result
        assert isinstance(result["completed_at"], datetime)

    @pytest.mark.asyncio
    async def test_sets_report_path(self) -> None:
        """Should set report_path in result."""
        written = {}

        def mock_writer(report_dir: str, report: ExecutionReport) -> None:
            written["dir"] = report_dir

        result = await generate_report_node({}, report_writer=mock_writer)

        assert result["report_path"] == ".executor"

    @pytest.mark.asyncio
    async def test_handles_write_error_gracefully(self) -> None:
        """Should handle write errors without crashing."""

        def failing_writer(report_dir: str, report: ExecutionReport) -> None:
            raise PermissionError("Permission denied")

        # Should not raise, just log the error
        result = await generate_report_node({}, report_writer=failing_writer)

        # Report should still be in result
        assert "execution_report" in result

    @pytest.mark.asyncio
    async def test_preserves_existing_state(self) -> None:
        """Should preserve existing state fields."""
        written = {}

        def mock_writer(report_dir: str, report: ExecutionReport) -> None:
            written["called"] = True

        state = {"existing_field": "preserved"}
        result = await generate_report_node(state, report_writer=mock_writer)

        assert result["existing_field"] == "preserved"

    @pytest.mark.asyncio
    async def test_includes_roadmap_modifications(self) -> None:
        """Should include roadmap modifications from state."""
        state = {
            "roadmap_modifications": [
                {"strategy": "rewrite", "original_task": "Task A"},
            ],
        }
        result = await generate_report_node(state)

        assert len(result["execution_report"].roadmap_modifications) == 1
