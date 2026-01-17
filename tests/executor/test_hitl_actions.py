"""Tests for Phase 4.1 Enhanced HITL Actions.

Phase 4.1: Enhanced HITL Actions

Tests cover:
- HITLActionType enum values
- HITLAction dataclass serialization
- HITLProposal dataclass functionality
- HITLResponse dataclass functionality
- EditHandler processing
- SuggestHandler processing
- ExplainHandler processing
- RollbackHandler processing
- HITLHandlerRegistry functionality
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_infra.executor.hitl import (
    EditHandler,
    ExplainHandler,
    HITLAction,
    HITLActionType,
    HITLHandlerRegistry,
    HITLProposal,
    HITLResponse,
    RollbackHandler,
    SuggestHandler,
)

# =============================================================================
# HITLActionType Tests (4.1.1)
# =============================================================================


class TestHITLActionType:
    """Tests for HITLActionType enum."""

    def test_all_action_types_exist(self) -> None:
        """Verify all 8 action types are defined."""
        assert HITLActionType.APPROVE.value == "approve"
        assert HITLActionType.REJECT.value == "reject"
        assert HITLActionType.EDIT.value == "edit"
        assert HITLActionType.SUGGEST.value == "suggest"
        assert HITLActionType.EXPLAIN.value == "explain"
        assert HITLActionType.ROLLBACK.value == "rollback"
        assert HITLActionType.SKIP.value == "skip"
        assert HITLActionType.DELEGATE.value == "delegate"

    def test_action_type_count(self) -> None:
        """Verify we have exactly 8 action types."""
        assert len(HITLActionType) == 8


# =============================================================================
# HITLAction Tests
# =============================================================================


class TestHITLAction:
    """Tests for HITLAction dataclass."""

    def test_simple_action(self) -> None:
        """Test creating a simple action."""
        action = HITLAction(type=HITLActionType.APPROVE)
        assert action.type == HITLActionType.APPROVE
        assert action.content is None
        assert action.target is None
        assert isinstance(action.timestamp, datetime)

    def test_action_with_content(self) -> None:
        """Test creating an action with content."""
        action = HITLAction(
            type=HITLActionType.EDIT,
            content="Use async/await",
        )
        assert action.type == HITLActionType.EDIT
        assert action.content == "Use async/await"

    def test_action_with_target(self) -> None:
        """Test creating an action with target."""
        action = HITLAction(
            type=HITLActionType.ROLLBACK,
            target="task_1.2.1",
        )
        assert action.type == HITLActionType.ROLLBACK
        assert action.target == "task_1.2.1"

    def test_action_to_dict(self) -> None:
        """Test serialization to dict."""
        action = HITLAction(
            type=HITLActionType.SUGGEST,
            content="Consider caching",
            metadata={"source": "user"},
        )
        data = action.to_dict()

        assert data["type"] == "suggest"
        assert data["content"] == "Consider caching"
        assert data["metadata"] == {"source": "user"}
        assert "timestamp" in data

    def test_action_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "type": "explain",
            "content": "Why SQLite?",
            "timestamp": "2024-01-01T12:00:00",
        }
        action = HITLAction.from_dict(data)

        assert action.type == HITLActionType.EXPLAIN
        assert action.content == "Why SQLite?"


# =============================================================================
# HITLProposal Tests
# =============================================================================


class TestHITLProposal:
    """Tests for HITLProposal dataclass."""

    def test_basic_proposal(self) -> None:
        """Test creating a basic proposal."""
        proposal = HITLProposal(description="Add logging")
        assert proposal.description == "Add logging"
        assert proposal.task_id == ""
        assert proposal.risk_level == "low"

    def test_full_proposal(self) -> None:
        """Test creating a full proposal."""
        proposal = HITLProposal(
            description="Add JWT authentication",
            task_id="2.1.3",
            task_title="Implement JWT",
            planned_actions=["Create token module", "Add middleware"],
            files_affected=["src/auth.py", "src/middleware.py"],
            rationale="JWT is standard for REST APIs",
            alternatives_considered=["Session cookies", "API keys"],
            risk_level="medium",
            estimated_tokens=5000,
        )

        assert proposal.description == "Add JWT authentication"
        assert proposal.task_id == "2.1.3"
        assert len(proposal.planned_actions) == 2
        assert len(proposal.files_affected) == 2
        assert proposal.risk_level == "medium"

    def test_proposal_serialization(self) -> None:
        """Test proposal serialization round-trip."""
        original = HITLProposal(
            description="Test proposal",
            task_id="1.1.1",
            planned_actions=["Action 1"],
            risk_level="high",
        )

        data = original.to_dict()
        restored = HITLProposal.from_dict(data)

        assert restored.description == original.description
        assert restored.task_id == original.task_id
        assert restored.risk_level == original.risk_level


# =============================================================================
# HITLResponse Tests
# =============================================================================


class TestHITLResponse:
    """Tests for HITLResponse dataclass."""

    def test_success_response(self) -> None:
        """Test creating a success response."""
        response = HITLResponse(
            understood=True,
            revised_plan="Updated approach using async",
            next_step="Proceeding with implementation",
        )

        assert response.understood is True
        assert response.revised_plan is not None
        assert response.error is None

    def test_failure_response(self) -> None:
        """Test creating a failure response."""
        response = HITLResponse(
            understood=False,
            error="No checkpoint found",
            next_step="Continuing from current state",
        )

        assert response.understood is False
        assert response.error is not None
        assert response.revised_plan is None

    def test_explanation_response(self) -> None:
        """Test response with explanation."""
        response = HITLResponse(
            understood=True,
            explanation="SQLite was chosen for simplicity...",
            next_step="Waiting for user decision",
        )

        assert response.explanation is not None
        assert response.revised_plan is None

    def test_response_serialization(self) -> None:
        """Test response serialization round-trip."""
        original = HITLResponse(
            understood=True,
            revised_plan="New plan",
            next_step="Continue",
            metadata={"key": "value"},
        )

        data = original.to_dict()
        restored = HITLResponse.from_dict(data)

        assert restored.understood == original.understood
        assert restored.revised_plan == original.revised_plan
        assert restored.metadata == original.metadata


# =============================================================================
# EditHandler Tests (4.1.2)
# =============================================================================


class TestEditHandler:
    """Tests for EditHandler."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        llm = MagicMock()
        llm.acomplete = AsyncMock(return_value=MagicMock(content="Revised plan: Use async"))
        return llm

    @pytest.fixture
    def handler(self, mock_llm: MagicMock) -> EditHandler:
        """Create an EditHandler with mock LLM."""
        return EditHandler(llm=mock_llm)

    @pytest.fixture
    def sample_proposal(self) -> HITLProposal:
        """Create a sample proposal."""
        return HITLProposal(
            description="Add logging to auth module",
            task_id="2.1.3",
            task_title="Add structured logging",
            planned_actions=["Create logger", "Add log calls"],
            files_affected=["src/auth.py"],
            rationale="Logging enables debugging",
        )

    @pytest.fixture
    def sample_context(self) -> Any:
        """Create a sample execution context."""
        from ai_infra.executor.agents.base import ExecutionContext

        return ExecutionContext(
            workspace=Path("/project"),
            summary="Python project with FastAPI",
        )

    @pytest.mark.asyncio
    async def test_edit_success(
        self,
        handler: EditHandler,
        sample_proposal: HITLProposal,
        sample_context: Any,
    ) -> None:
        """Test successful edit processing."""
        action = HITLAction(
            type=HITLActionType.EDIT,
            content="Use structlog instead of logging module",
        )

        response = await handler.handle(action, sample_proposal, sample_context)

        assert response.understood is True
        assert response.revised_plan is not None
        assert response.error is None

    @pytest.mark.asyncio
    async def test_edit_without_content(
        self,
        handler: EditHandler,
        sample_proposal: HITLProposal,
        sample_context: Any,
    ) -> None:
        """Test edit without content fails gracefully."""
        action = HITLAction(type=HITLActionType.EDIT, content=None)

        response = await handler.handle(action, sample_proposal, sample_context)

        assert response.understood is False
        assert "requires content" in response.error.lower()

    @pytest.mark.asyncio
    async def test_edit_llm_failure(
        self,
        handler: EditHandler,
        sample_proposal: HITLProposal,
        sample_context: Any,
    ) -> None:
        """Test handling of LLM failure."""
        handler.llm.acomplete = AsyncMock(side_effect=Exception("LLM error"))
        action = HITLAction(
            type=HITLActionType.EDIT,
            content="Change approach",
        )

        response = await handler.handle(action, sample_proposal, sample_context)

        assert response.understood is False
        assert "failed" in response.error.lower()


# =============================================================================
# SuggestHandler Tests (4.1.3)
# =============================================================================


class TestSuggestHandler:
    """Tests for SuggestHandler."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        llm = MagicMock()
        llm.acomplete = AsyncMock(return_value=MagicMock(content="Enhanced plan with caching"))
        return llm

    @pytest.fixture
    def handler(self, mock_llm: MagicMock) -> SuggestHandler:
        """Create a SuggestHandler with mock LLM."""
        return SuggestHandler(llm=mock_llm)

    @pytest.fixture
    def sample_proposal(self) -> HITLProposal:
        """Create a sample proposal."""
        return HITLProposal(
            description="Add API endpoint",
            task_id="1.1.1",
            planned_actions=["Create route", "Add handler"],
        )

    @pytest.fixture
    def sample_context(self) -> Any:
        """Create a sample execution context."""
        from ai_infra.executor.agents.base import ExecutionContext

        return ExecutionContext(
            workspace=Path("/project"),
            summary="FastAPI project",
        )

    @pytest.mark.asyncio
    async def test_suggest_success(
        self,
        handler: SuggestHandler,
        sample_proposal: HITLProposal,
        sample_context: Any,
    ) -> None:
        """Test successful suggestion processing."""
        action = HITLAction(
            type=HITLActionType.SUGGEST,
            content="Consider adding rate limiting",
        )

        response = await handler.handle(action, sample_proposal, sample_context)

        assert response.understood is True
        assert response.revised_plan is not None
        assert "suggestion" in response.metadata

    @pytest.mark.asyncio
    async def test_suggest_without_content(
        self,
        handler: SuggestHandler,
        sample_proposal: HITLProposal,
        sample_context: Any,
    ) -> None:
        """Test suggestion without content fails."""
        action = HITLAction(type=HITLActionType.SUGGEST, content=None)

        response = await handler.handle(action, sample_proposal, sample_context)

        assert response.understood is False


# =============================================================================
# ExplainHandler Tests (4.1.4)
# =============================================================================


class TestExplainHandler:
    """Tests for ExplainHandler."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        llm = MagicMock()
        llm.acomplete = AsyncMock(
            return_value=MagicMock(content="SQLite was chosen because it requires no setup...")
        )
        return llm

    @pytest.fixture
    def handler(self, mock_llm: MagicMock) -> ExplainHandler:
        """Create an ExplainHandler with mock LLM."""
        return ExplainHandler(llm=mock_llm)

    @pytest.fixture
    def sample_proposal(self) -> HITLProposal:
        """Create a sample proposal."""
        return HITLProposal(
            description="Add SQLite database",
            task_id="3.1.1",
            rationale="Simple local storage needed",
            alternatives_considered=["PostgreSQL", "MongoDB"],
        )

    @pytest.fixture
    def sample_context(self) -> Any:
        """Create a sample execution context."""
        from ai_infra.executor.agents.base import ExecutionContext

        return ExecutionContext(
            workspace=Path("/project"),
            summary="CLI tool",
            relevant_files=["src/db.py", "src/models.py"],
        )

    @pytest.mark.asyncio
    async def test_explain_with_question(
        self,
        handler: ExplainHandler,
        sample_proposal: HITLProposal,
        sample_context: Any,
    ) -> None:
        """Test explanation with specific question."""
        action = HITLAction(
            type=HITLActionType.EXPLAIN,
            content="Why not PostgreSQL?",
        )

        response = await handler.handle(action, sample_proposal, sample_context)

        assert response.understood is True
        assert response.explanation is not None
        assert response.revised_plan is None  # Explain doesn't change plan
        assert "question" in response.metadata

    @pytest.mark.asyncio
    async def test_explain_default_question(
        self,
        handler: ExplainHandler,
        sample_proposal: HITLProposal,
        sample_context: Any,
    ) -> None:
        """Test explanation with no specific question."""
        action = HITLAction(type=HITLActionType.EXPLAIN, content=None)

        response = await handler.handle(action, sample_proposal, sample_context)

        assert response.understood is True
        assert response.explanation is not None


# =============================================================================
# RollbackHandler Tests (4.1.5)
# =============================================================================


class TestRollbackHandler:
    """Tests for RollbackHandler."""

    @pytest.fixture
    def mock_checkpoint_manager(self) -> MagicMock:
        """Create a mock checkpoint manager."""
        manager = MagicMock()
        manager.get_executor_commits = MagicMock(
            return_value=[
                MagicMock(sha="abc123", short_sha="abc123", message="Task 1.1.1"),
            ]
        )
        manager.get_commits_for_task = MagicMock(
            return_value=[
                MagicMock(sha="def456", short_sha="def456", message="Task 1.2.1"),
            ]
        )
        manager.rollback_to = MagicMock()
        return manager

    @pytest.fixture
    def handler(self, mock_checkpoint_manager: MagicMock) -> RollbackHandler:
        """Create a RollbackHandler with mock manager."""
        return RollbackHandler(checkpoint_manager=mock_checkpoint_manager)

    @pytest.fixture
    def sample_proposal(self) -> HITLProposal:
        """Create a sample proposal."""
        return HITLProposal(description="Current task", task_id="1.2.2")

    @pytest.fixture
    def sample_context(self) -> Any:
        """Create a sample execution context."""
        from ai_infra.executor.agents.base import ExecutionContext

        return ExecutionContext(workspace=Path("/project"))

    @pytest.mark.asyncio
    async def test_rollback_to_last(
        self,
        handler: RollbackHandler,
        sample_proposal: HITLProposal,
        sample_context: Any,
    ) -> None:
        """Test rollback to last checkpoint."""
        action = HITLAction(type=HITLActionType.ROLLBACK, target=None)

        response = await handler.handle(action, sample_proposal, sample_context)

        assert response.understood is True
        assert "rolled back" in response.revised_plan.lower()

    @pytest.mark.asyncio
    async def test_rollback_to_specific_task(
        self,
        handler: RollbackHandler,
        sample_proposal: HITLProposal,
        sample_context: Any,
    ) -> None:
        """Test rollback to specific task checkpoint."""
        action = HITLAction(
            type=HITLActionType.ROLLBACK,
            target="task_1.2.1",
        )

        response = await handler.handle(action, sample_proposal, sample_context)

        assert response.understood is True

    @pytest.mark.asyncio
    async def test_rollback_no_checkpoint_found(
        self,
        handler: RollbackHandler,
        sample_proposal: HITLProposal,
        sample_context: Any,
    ) -> None:
        """Test rollback when no checkpoint exists."""
        handler.checkpoint_manager.get_commits_for_task = MagicMock(return_value=[])
        action = HITLAction(
            type=HITLActionType.ROLLBACK,
            target="nonexistent_task",
        )

        response = await handler.handle(action, sample_proposal, sample_context)

        assert response.understood is False
        assert "no checkpoint found" in response.error.lower()


# =============================================================================
# HITLHandlerRegistry Tests
# =============================================================================


class TestHITLHandlerRegistry:
    """Tests for HITLHandlerRegistry."""

    def test_register_and_get(self) -> None:
        """Test registering and retrieving handlers."""
        registry = HITLHandlerRegistry()
        mock_llm = MagicMock()
        handler = EditHandler(llm=mock_llm)

        registry.register(HITLActionType.EDIT, handler)

        retrieved = registry.get_handler(HITLActionType.EDIT)
        assert retrieved is handler

    def test_get_unregistered(self) -> None:
        """Test getting unregistered handler returns None."""
        registry = HITLHandlerRegistry()

        result = registry.get_handler(HITLActionType.EXPLAIN)
        assert result is None

    def test_has_handler(self) -> None:
        """Test checking handler registration."""
        registry = HITLHandlerRegistry()
        mock_llm = MagicMock()

        registry.register(HITLActionType.EDIT, EditHandler(llm=mock_llm))

        assert registry.has_handler(HITLActionType.EDIT) is True
        assert registry.has_handler(HITLActionType.SUGGEST) is False

    def test_create_default(self) -> None:
        """Test creating default registry."""
        mock_llm = MagicMock()
        mock_checkpoint_manager = MagicMock()

        registry = HITLHandlerRegistry.create_default(
            llm=mock_llm,
            checkpoint_manager=mock_checkpoint_manager,
        )

        # Should have all 4 handlers
        assert registry.has_handler(HITLActionType.EDIT)
        assert registry.has_handler(HITLActionType.SUGGEST)
        assert registry.has_handler(HITLActionType.EXPLAIN)
        assert registry.has_handler(HITLActionType.ROLLBACK)

    def test_create_default_without_checkpoint_manager(self) -> None:
        """Test default registry without checkpoint manager."""
        mock_llm = MagicMock()

        registry = HITLHandlerRegistry.create_default(llm=mock_llm)

        # Should have 3 handlers (no rollback)
        assert registry.has_handler(HITLActionType.EDIT)
        assert registry.has_handler(HITLActionType.SUGGEST)
        assert registry.has_handler(HITLActionType.EXPLAIN)
        assert registry.has_handler(HITLActionType.ROLLBACK) is False
