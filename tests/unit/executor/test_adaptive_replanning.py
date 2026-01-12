"""Tests for adaptive replanning nodes.

Phase 2.3.1: Unit tests for analyze_failure_node and replan_task_node.
Phase 2.4.1: Unit tests for detailed failure_category detection.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ai_infra.executor.edges.routes import (
    route_after_analyze_failure,
    route_after_replan,
    route_after_verify,
)
from ai_infra.executor.failure import FailureCategory
from ai_infra.executor.nodes.failure import (
    FailureClassification,
    analyze_failure_node,
    classify_error_by_pattern,
    detect_failure_category,
)
from ai_infra.executor.nodes.replan import (
    MAX_REPLANS,
    replan_task_node,
    should_replan,
)
from ai_infra.executor.state import ExecutorGraphState
from ai_infra.executor.todolist import TodoItem, TodoStatus

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_todo_item() -> TodoItem:
    """Create a mock TodoItem for testing."""
    return TodoItem(
        id=1,
        title="Install and use structlog",
        description="Add structured logging to the auth module",
        status=TodoStatus.NOT_STARTED,
        file_hints=["src/auth.py"],
    )


@pytest.fixture
def base_state(mock_todo_item: TodoItem) -> ExecutorGraphState:
    """Create a base ExecutorGraphState for testing."""
    return ExecutorGraphState(
        roadmap_path="ROADMAP.md",
        todos=[mock_todo_item],
        current_task=mock_todo_item,
        context="",
        prompt="Add logging to auth module",
        agent_result=None,
        files_modified=[],
        verified=False,
        last_checkpoint_sha=None,
        error=None,
        retry_count=0,
        replan_count=0,
        completed_count=0,
        max_tasks=None,
        should_continue=True,
        interrupt_requested=False,
        run_memory={},
        adaptive_mode="auto_fix",
        failure_classification=None,
        failure_reason=None,
        suggested_fix=None,
        execution_plan=None,
    )


# =============================================================================
# FailureClassification Pattern Tests
# =============================================================================


class TestFailureClassificationPatterns:
    """Tests for pattern-based failure classification."""

    def test_transient_rate_limit(self):
        """Test rate limit errors are classified as TRANSIENT."""
        error = "Rate limit exceeded. Please retry after 60 seconds"
        result = classify_error_by_pattern(error)
        assert result == FailureClassification.TRANSIENT

    def test_transient_timeout(self):
        """Test timeout errors are classified as TRANSIENT."""
        error = "Request timed out after 30 seconds"
        result = classify_error_by_pattern(error)
        assert result == FailureClassification.TRANSIENT

    def test_transient_connection_refused(self):
        """Test connection refused is classified as TRANSIENT."""
        error = "Connection refused to api.example.com:443"
        result = classify_error_by_pattern(error)
        assert result == FailureClassification.TRANSIENT

    def test_wrong_approach_module_not_found(self):
        """Test ModuleNotFoundError is classified as WRONG_APPROACH."""
        error = "ModuleNotFoundError: No module named 'structlog'"
        result = classify_error_by_pattern(error)
        assert result == FailureClassification.WRONG_APPROACH

    def test_wrong_approach_import_error(self):
        """Test ImportError is classified as WRONG_APPROACH."""
        error = "ImportError: cannot import name 'foo' from 'bar'"
        result = classify_error_by_pattern(error)
        assert result == FailureClassification.WRONG_APPROACH

    def test_wrong_approach_file_not_found(self):
        """Test FileNotFoundError is classified as WRONG_APPROACH."""
        error = "FileNotFoundError: [Errno 2] No such file or directory: 'config.yaml'"
        result = classify_error_by_pattern(error)
        assert result == FailureClassification.WRONG_APPROACH

    def test_wrong_approach_syntax_error(self):
        """Test SyntaxError is classified as WRONG_APPROACH."""
        error = "SyntaxError: invalid syntax at line 42"
        result = classify_error_by_pattern(error)
        assert result == FailureClassification.WRONG_APPROACH

    def test_wrong_approach_type_error(self):
        """Test TypeError is classified as WRONG_APPROACH."""
        error = "TypeError: expected str, got int"
        result = classify_error_by_pattern(error)
        assert result == FailureClassification.WRONG_APPROACH

    def test_fatal_permission_denied(self):
        """Test permission denied is classified as FATAL."""
        error = "PermissionError: Permission denied: '/etc/passwd'"
        result = classify_error_by_pattern(error)
        assert result == FailureClassification.FATAL

    def test_fatal_authentication_failed(self):
        """Test authentication failed is classified as FATAL."""
        error = "AuthenticationError: Authentication failed for user 'admin'"
        result = classify_error_by_pattern(error)
        assert result == FailureClassification.FATAL

    def test_fatal_api_key(self):
        """Test invalid API key is classified as FATAL."""
        error = "Invalid API key provided"
        result = classify_error_by_pattern(error)
        assert result == FailureClassification.FATAL

    def test_unknown_error_returns_none(self):
        """Test unknown errors return None for LLM classification."""
        error = "Some weird error that doesn't match any pattern"
        result = classify_error_by_pattern(error)
        assert result is None


# =============================================================================
# analyze_failure_node Tests
# =============================================================================


class TestAnalyzeFailureNode:
    """Tests for analyze_failure_node."""

    @pytest.mark.asyncio
    async def test_no_error_returns_transient(self, base_state: ExecutorGraphState):
        """Test returns TRANSIENT when no error in state."""
        state = {**base_state, "error": None}

        result = await analyze_failure_node(state)

        assert result["failure_classification"] == FailureClassification.TRANSIENT
        assert "No error provided" in result["failure_reason"]

    @pytest.mark.asyncio
    async def test_pattern_classifies_module_not_found(self, base_state: ExecutorGraphState):
        """Test pattern classification for ModuleNotFoundError."""
        state = {
            **base_state,
            "error": {
                "error_type": "ModuleNotFoundError",
                "message": "No module named 'structlog'",
                "node": "execute_task",
                "task_id": "1",
                "recoverable": True,
                "stack_trace": None,
            },
        }

        result = await analyze_failure_node(state)

        assert result["failure_classification"] == FailureClassification.WRONG_APPROACH
        assert "structlog" in result["failure_reason"]
        assert "install" in result["suggested_fix"].lower()

    @pytest.mark.asyncio
    async def test_pattern_classifies_rate_limit(self, base_state: ExecutorGraphState):
        """Test pattern classification for rate limit error."""
        state = {
            **base_state,
            "error": {
                "error_type": "RateLimitError",
                "message": "Rate limit exceeded",
                "node": "execute_task",
                "task_id": "1",
                "recoverable": True,
                "stack_trace": None,
            },
        }

        result = await analyze_failure_node(state)

        assert result["failure_classification"] == FailureClassification.TRANSIENT

    @pytest.mark.asyncio
    async def test_pattern_classifies_permission_denied(self, base_state: ExecutorGraphState):
        """Test pattern classification for permission denied."""
        state = {
            **base_state,
            "error": {
                "error_type": "PermissionError",
                "message": "Permission denied: /etc/passwd",
                "node": "execute_task",
                "task_id": "1",
                "recoverable": True,
                "stack_trace": None,
            },
        }

        result = await analyze_failure_node(state)

        assert result["failure_classification"] == FailureClassification.FATAL

    @pytest.mark.asyncio
    async def test_llm_fallback_when_pattern_fails(self, base_state: ExecutorGraphState):
        """Test LLM fallback when pattern doesn't match."""
        state = {
            **base_state,
            "error": {
                "error_type": "CustomError",
                "message": "Some obscure error that needs LLM analysis",
                "node": "execute_task",
                "task_id": "1",
                "recoverable": True,
                "stack_trace": None,
            },
        }

        # Mock agent that classifies as WRONG_APPROACH
        mock_agent = AsyncMock()
        mock_agent.arun.return_value = """
CLASSIFICATION: WRONG_APPROACH
REASON: The error indicates a configuration issue
FIX: Update the configuration file
"""

        result = await analyze_failure_node(state, analyzer_agent=mock_agent)

        assert result["failure_classification"] == FailureClassification.WRONG_APPROACH
        assert "configuration" in result["failure_reason"].lower()
        assert mock_agent.arun.called


# =============================================================================
# Phase 2.4.1: Detailed Failure Category Tests
# =============================================================================


class TestDetectFailureCategory:
    """Phase 2.4.1: Tests for detect_failure_category function."""

    def test_detects_syntax_error(self):
        """Test detects SYNTAX_ERROR from SyntaxError message."""
        result = detect_failure_category("SyntaxError: invalid syntax at line 10")
        assert result == FailureCategory.SYNTAX_ERROR

    def test_detects_indentation_error(self):
        """Test detects SYNTAX_ERROR from IndentationError message."""
        result = detect_failure_category("IndentationError: unexpected indent")
        assert result == FailureCategory.SYNTAX_ERROR

    def test_detects_import_error(self):
        """Test detects IMPORT_ERROR from ImportError message."""
        result = detect_failure_category("ImportError: cannot import name 'foo'")
        assert result == FailureCategory.IMPORT_ERROR

    def test_detects_module_not_found(self):
        """Test detects IMPORT_ERROR from ModuleNotFoundError message."""
        result = detect_failure_category("ModuleNotFoundError: No module named 'structlog'")
        assert result == FailureCategory.IMPORT_ERROR

    def test_detects_type_error(self):
        """Test detects TYPE_ERROR from TypeError message."""
        result = detect_failure_category("TypeError: expected str, got int")
        assert result == FailureCategory.TYPE_ERROR

    def test_detects_test_failure(self):
        """Test detects TEST_FAILURE from test failure message."""
        result = detect_failure_category("AssertionError: expected 5, got 10")
        assert result == FailureCategory.TEST_FAILURE

    def test_detects_test_failure_pytest(self):
        """Test detects TEST_FAILURE from pytest message."""
        result = detect_failure_category("FAILED tests/test_foo.py::test_bar - pytest")
        assert result == FailureCategory.TEST_FAILURE

    def test_detects_file_not_found(self):
        """Test detects FILE_NOT_FOUND from FileNotFoundError."""
        result = detect_failure_category(
            "FileNotFoundError: No such file or directory: 'config.yaml'"
        )
        assert result == FailureCategory.FILE_NOT_FOUND

    def test_detects_timeout(self):
        """Test detects TIMEOUT from timeout message."""
        result = detect_failure_category("TimeoutError: Operation timed out after 30s")
        assert result == FailureCategory.TIMEOUT

    def test_detects_api_error(self):
        """Test detects API_ERROR from rate limit message."""
        result = detect_failure_category("RateLimitError: Rate limit exceeded, retry after 60s")
        assert result == FailureCategory.API_ERROR

    def test_detects_wrong_approach(self):
        """Test detects WRONG_APPROACH from NameError."""
        result = detect_failure_category("NameError: name 'undefined_var' is not defined")
        assert result == FailureCategory.WRONG_APPROACH

    def test_returns_unknown_for_unrecognized(self):
        """Test returns UNKNOWN for unrecognized errors."""
        result = detect_failure_category("Some completely random error message")
        assert result == FailureCategory.UNKNOWN


class TestAnalyzeFailureNodeCategory:
    """Phase 2.4.1: Tests for failure_category in analyze_failure_node."""

    @pytest.mark.asyncio
    async def test_returns_failure_category_for_syntax_error(self, base_state: ExecutorGraphState):
        """Test failure_category is set for syntax error."""
        state = {
            **base_state,
            "error": {
                "error_type": "SyntaxError",
                "message": "SyntaxError: invalid syntax at line 42",
                "node": "execute_task",
                "task_id": "1",
                "recoverable": True,
                "stack_trace": None,
            },
        }

        result = await analyze_failure_node(state)

        assert result["failure_category"] == FailureCategory.SYNTAX_ERROR.value
        assert result["failure_classification"] == FailureClassification.WRONG_APPROACH

    @pytest.mark.asyncio
    async def test_returns_failure_category_for_import_error(self, base_state: ExecutorGraphState):
        """Test failure_category is set for import error."""
        state = {
            **base_state,
            "error": {
                "error_type": "ModuleNotFoundError",
                "message": "No module named 'pandas'",
                "node": "execute_task",
                "task_id": "1",
                "recoverable": True,
                "stack_trace": None,
            },
        }

        result = await analyze_failure_node(state)

        assert result["failure_category"] == FailureCategory.IMPORT_ERROR.value

    @pytest.mark.asyncio
    async def test_returns_failure_category_for_rate_limit(self, base_state: ExecutorGraphState):
        """Test failure_category is set for rate limit (API_ERROR)."""
        state = {
            **base_state,
            "error": {
                "error_type": "RateLimitError",
                "message": "Rate limit exceeded",
                "node": "execute_task",
                "task_id": "1",
                "recoverable": True,
                "stack_trace": None,
            },
        }

        result = await analyze_failure_node(state)

        assert result["failure_category"] == FailureCategory.API_ERROR.value
        assert result["failure_classification"] == FailureClassification.TRANSIENT

    @pytest.mark.asyncio
    async def test_returns_unknown_category_for_no_error(self, base_state: ExecutorGraphState):
        """Test failure_category is UNKNOWN when no error."""
        state = {**base_state, "error": None}

        result = await analyze_failure_node(state)

        assert result["failure_category"] == FailureCategory.UNKNOWN.value


# =============================================================================
# replan_task_node Tests
# =============================================================================


class TestReplanTaskNode:
    """Tests for replan_task_node."""

    @pytest.mark.asyncio
    async def test_increments_replan_count(self, base_state: ExecutorGraphState):
        """Test replan count is incremented."""
        state = {
            **base_state,
            "replan_count": 0,
            "failure_reason": "Module not found",
            "suggested_fix": "pip install structlog",
        }

        result = await replan_task_node(state)

        assert result["replan_count"] == 1
        assert result["error"] is None
        assert result["retry_count"] == 0

    @pytest.mark.asyncio
    async def test_generates_plan_with_agent(self, base_state: ExecutorGraphState):
        """Test generates execution plan when agent provided."""
        state = {
            **base_state,
            "replan_count": 0,
            "failure_reason": "structlog not installed",
            "suggested_fix": "pip install structlog",
            "prompt": "Add structured logging to auth module",
        }

        mock_agent = AsyncMock()
        mock_agent.arun.return_value = (
            "1. Install structlog\n2. Import structlog\n3. Configure logger"
        )

        result = await replan_task_node(state, planner_agent=mock_agent)

        assert result["execution_plan"] is not None
        assert "structlog" in result["execution_plan"].lower()
        assert mock_agent.arun.called

    @pytest.mark.asyncio
    async def test_uses_suggested_fix_without_agent(self, base_state: ExecutorGraphState):
        """Test uses suggested fix when no agent."""
        state = {
            **base_state,
            "replan_count": 0,
            "failure_reason": "Module not found",
            "suggested_fix": "pip install structlog",
        }

        result = await replan_task_node(state)

        assert "pip install structlog" in result["execution_plan"]

    @pytest.mark.asyncio
    async def test_clears_error_state(self, base_state: ExecutorGraphState):
        """Test clears error and verified states for retry."""
        state = {
            **base_state,
            "replan_count": 0,
            "error": {"message": "test error"},
            "verified": True,
            "agent_result": {"some": "result"},
            "retry_count": 2,
        }

        result = await replan_task_node(state)

        assert result["error"] is None
        assert result["verified"] is False
        assert result["agent_result"] is None
        assert result["retry_count"] == 0


# =============================================================================
# should_replan Tests
# =============================================================================


class TestShouldReplan:
    """Tests for should_replan helper."""

    def test_replans_on_wrong_approach(self, base_state: ExecutorGraphState):
        """Test returns True for WRONG_APPROACH within limit."""
        state = {
            **base_state,
            "failure_classification": FailureClassification.WRONG_APPROACH,
            "replan_count": 0,
        }

        assert should_replan(state) is True

    def test_no_replan_at_max(self, base_state: ExecutorGraphState):
        """Test returns False when replan limit reached."""
        state = {
            **base_state,
            "failure_classification": FailureClassification.WRONG_APPROACH,
            "replan_count": MAX_REPLANS,
        }

        assert should_replan(state) is False

    def test_no_replan_for_transient(self, base_state: ExecutorGraphState):
        """Test returns False for TRANSIENT errors."""
        state = {
            **base_state,
            "failure_classification": FailureClassification.TRANSIENT,
            "replan_count": 0,
        }

        assert should_replan(state) is False

    def test_no_replan_for_fatal(self, base_state: ExecutorGraphState):
        """Test returns False for FATAL errors."""
        state = {
            **base_state,
            "failure_classification": FailureClassification.FATAL,
            "replan_count": 0,
        }

        assert should_replan(state) is False


# =============================================================================
# Routing Tests
# =============================================================================


class TestAdaptiveRouting:
    """Tests for adaptive replanning routing functions."""

    def test_verify_routes_to_analyze_failure_in_adaptive_mode(
        self, base_state: ExecutorGraphState
    ):
        """Test verify_task routes to analyze_failure when adaptive mode enabled."""
        state = {
            **base_state,
            "verified": False,
            "adaptive_mode": "auto_fix",
        }

        result = route_after_verify(state)

        assert result == "analyze_failure"

    def test_verify_routes_to_handle_failure_in_no_adapt(self, base_state: ExecutorGraphState):
        """Test verify_task routes to handle_failure when adaptive mode disabled."""
        state = {
            **base_state,
            "verified": False,
            "adaptive_mode": "no_adapt",
        }

        result = route_after_verify(state)

        assert result == "handle_failure"

    def test_verify_routes_to_checkpoint_on_success(self, base_state: ExecutorGraphState):
        """Test verify_task routes to checkpoint on success."""
        state = {
            **base_state,
            "verified": True,
            "adaptive_mode": "auto_fix",
        }

        result = route_after_verify(state)

        assert result == "checkpoint"

    def test_analyze_failure_routes_to_replan_on_wrong_approach(
        self, base_state: ExecutorGraphState
    ):
        """Test analyze_failure routes to replan_task for WRONG_APPROACH."""
        state = {
            **base_state,
            "failure_classification": FailureClassification.WRONG_APPROACH,
            "replan_count": 0,
        }

        result = route_after_analyze_failure(state)

        assert result == "replan_task"

    def test_analyze_failure_routes_to_handle_failure_on_transient(
        self, base_state: ExecutorGraphState
    ):
        """Test analyze_failure routes to handle_failure for TRANSIENT."""
        state = {
            **base_state,
            "failure_classification": FailureClassification.TRANSIENT,
            "replan_count": 0,
        }

        result = route_after_analyze_failure(state)

        assert result == "handle_failure"

    def test_analyze_failure_routes_to_decide_next_on_fatal(self, base_state: ExecutorGraphState):
        """Test analyze_failure routes to decide_next for FATAL."""
        state = {
            **base_state,
            "failure_classification": FailureClassification.FATAL,
            "replan_count": 0,
        }

        result = route_after_analyze_failure(state)

        assert result == "decide_next"

    def test_analyze_failure_routes_to_handle_failure_at_max_replans(
        self, base_state: ExecutorGraphState
    ):
        """Test routes to handle_failure when replan limit exceeded."""
        state = {
            **base_state,
            "failure_classification": FailureClassification.WRONG_APPROACH,
            "replan_count": MAX_REPLANS,
        }

        result = route_after_analyze_failure(state)

        assert result == "handle_failure"

    def test_replan_routes_to_build_context(self, base_state: ExecutorGraphState):
        """Test replan_task always routes to build_context."""
        result = route_after_replan(base_state)

        assert result == "build_context"
