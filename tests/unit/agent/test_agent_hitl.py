"""Tests for Agent Human-in-the-Loop (HITL) approval flow.

Tests cover:
- ApprovalConfig configuration
- ApprovalRequest/ApprovalResponse models
- Built-in approval handlers (auto_approve, auto_reject)
- Selective approval (by tool name)
- Callable approval conditions
- Async approval handlers
- Session-based pause/resume flow

Phase 1.1.3 of production readiness test plan.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from ai_infra import Agent
from ai_infra.llm.session import memory
from ai_infra.llm.tools.approval import (
    ApprovalRequest,
    ApprovalResponse,
    auto_approve_handler,
    auto_reject_handler,
    create_selective_handler,
)
from ai_infra.llm.tools.hitl import ApprovalConfig

# =============================================================================
# ApprovalRequest Tests
# =============================================================================


class TestApprovalRequest:
    """Tests for ApprovalRequest model."""

    def test_basic_request(self):
        """Basic approval request creation."""
        req = ApprovalRequest(tool_name="delete_file", args={"path": "/tmp/test"})

        assert req.tool_name == "delete_file"
        assert req.args == {"path": "/tmp/test"}
        assert req.id is not None

    def test_default_timeout(self):
        """Default timeout is 300 seconds (5 minutes)."""
        req = ApprovalRequest(tool_name="test", args={})
        assert req.timeout == 300

    def test_custom_timeout(self):
        """Custom timeout is stored."""
        req = ApprovalRequest(tool_name="test", args={}, timeout=60)
        assert req.timeout == 60

    def test_context_metadata(self):
        """Context and metadata are stored."""
        req = ApprovalRequest(
            tool_name="test",
            args={},
            context={"conversation": "hello"},
            metadata={"user_id": "123"},
        )
        assert req.context == {"conversation": "hello"}
        assert req.metadata == {"user_id": "123"}

    def test_auto_generated_id(self):
        """ID is auto-generated if not provided."""
        req = ApprovalRequest(tool_name="test", args={})
        assert req.id is not None
        assert len(req.id) > 0

    def test_timestamp_is_datetime(self):
        """Timestamp is a datetime object."""
        req = ApprovalRequest(tool_name="test", args={})
        assert isinstance(req.timestamp, datetime)

    def test_console_prompt_format(self):
        """to_console_prompt includes tool name and args."""
        req = ApprovalRequest(tool_name="delete_file", args={"path": "/tmp/test.txt"})
        prompt = req.to_console_prompt()

        assert "delete_file" in prompt
        assert "/tmp/test.txt" in prompt


# =============================================================================
# ApprovalResponse Tests
# =============================================================================


class TestApprovalResponse:
    """Tests for ApprovalResponse model."""

    def test_approve_factory(self):
        """approve() factory creates approved response."""
        resp = ApprovalResponse.approve(reason="Looks good")

        assert resp.approved is True
        assert resp.reason == "Looks good"

    def test_reject_factory(self):
        """reject() factory creates rejected response."""
        resp = ApprovalResponse.reject(reason="Too dangerous")

        assert resp.approved is False
        assert resp.reason == "Too dangerous"

    def test_approve_with_modified_args(self):
        """Approval can include modified arguments."""
        resp = ApprovalResponse.approve(
            modified_args={"amount": 100},
            reason="Reduced amount for safety",
        )

        assert resp.approved is True
        assert resp.modified_args == {"amount": 100}


# =============================================================================
# Built-in Handler Tests
# =============================================================================


class TestBuiltinHandlers:
    """Tests for built-in approval handlers."""

    def test_auto_approve_handler(self):
        """auto_approve_handler approves all requests."""
        req = ApprovalRequest(tool_name="any_tool", args={"x": 1})
        resp = auto_approve_handler(req)

        assert resp.approved is True
        assert resp.approver == "auto"

    def test_auto_reject_handler(self):
        """auto_reject_handler rejects all requests."""
        req = ApprovalRequest(tool_name="any_tool", args={"x": 1})
        resp = auto_reject_handler(req)

        assert resp.approved is False
        assert resp.approver == "auto"


# =============================================================================
# Selective Handler Tests
# =============================================================================


class TestSelectiveHandler:
    """Tests for selective approval handlers."""

    def test_selective_handler_approves_safe_tools(self):
        """Selective handler auto-approves tools not in list."""
        handler = create_selective_handler(
            tools_requiring_approval=["dangerous_tool"],
            handler=auto_reject_handler,
        )

        req = ApprovalRequest(tool_name="safe_tool", args={})
        resp = handler(req)

        assert resp.approved is True

    def test_selective_handler_uses_handler_for_listed_tools(self):
        """Selective handler uses provided handler for listed tools."""
        handler = create_selective_handler(
            tools_requiring_approval=["dangerous_tool"],
            handler=auto_reject_handler,
        )

        req = ApprovalRequest(tool_name="dangerous_tool", args={})
        resp = handler(req)

        assert resp.approved is False


# =============================================================================
# ApprovalConfig Tests
# =============================================================================


class TestApprovalConfig:
    """Tests for ApprovalConfig."""

    def test_default_no_approval(self):
        """Default config requires no approval."""
        config = ApprovalConfig()
        assert config.require_approval is False
        assert config.needs_approval("any_tool") is False

    def test_require_all_approval(self):
        """require_approval=True requires all tools."""
        config = ApprovalConfig(require_approval=True)
        assert config.needs_approval("any_tool") is True

    def test_require_specific_tools(self):
        """require_approval=[list] requires only listed tools."""
        config = ApprovalConfig(require_approval=["tool_a", "tool_b"])

        assert config.needs_approval("tool_a") is True
        assert config.needs_approval("tool_b") is True
        assert config.needs_approval("tool_c") is False

    def test_auto_approve_overrides(self):
        """auto_approve=True overrides require_approval."""
        config = ApprovalConfig(require_approval=True, auto_approve=True)
        assert config.needs_approval("any_tool") is False

    def test_callable_approval_condition(self):
        """require_approval can be a callable."""

        def condition(tool_name: str, args: dict) -> bool:
            if tool_name == "transfer":
                return args.get("amount", 0) > 1000
            return False

        config = ApprovalConfig(require_approval=condition)

        assert config.needs_approval("transfer", {"amount": 500}) is False
        assert config.needs_approval("transfer", {"amount": 2000}) is True
        assert config.needs_approval("other", {}) is False

    @pytest.mark.asyncio
    async def test_request_approval_auto_approve(self):
        """request_approval with auto_approve returns approved."""
        config = ApprovalConfig(auto_approve=True)
        req = ApprovalRequest(tool_name="test", args={})

        resp = await config.request_approval(req)

        assert resp.approved is True
        assert resp.approver == "auto"

    @pytest.mark.asyncio
    async def test_request_approval_with_sync_handler(self):
        """request_approval works with sync handler."""
        config = ApprovalConfig(
            require_approval=True,
            approval_handler=auto_reject_handler,
        )
        req = ApprovalRequest(tool_name="test", args={})

        resp = await config.request_approval(req)

        assert resp.approved is False

    @pytest.mark.asyncio
    async def test_request_approval_with_async_handler(self):
        """request_approval works with async handler."""

        async def async_handler(req: ApprovalRequest) -> ApprovalResponse:
            return ApprovalResponse.approve(reason="Async approved")

        config = ApprovalConfig(
            require_approval=True,
            approval_handler_async=async_handler,
        )
        req = ApprovalRequest(tool_name="test", args={})

        resp = await config.request_approval(req)

        assert resp.approved is True
        assert resp.reason == "Async approved"


# =============================================================================
# Agent Approval Integration Tests
# =============================================================================


class TestAgentApprovalIntegration:
    """Tests for Agent with approval configuration."""

    def test_agent_require_approval_true(self):
        """Agent accepts require_approval=True."""
        agent = Agent(require_approval=True)
        assert agent._approval_config is not None
        assert agent._approval_config.require_approval is True

    def test_agent_require_approval_list(self):
        """Agent accepts require_approval as list of tools."""
        agent = Agent(require_approval=["delete_file", "execute"])
        assert agent._approval_config is not None
        assert agent._approval_config.require_approval == ["delete_file", "execute"]

    def test_agent_with_approval_handler(self):
        """Agent accepts custom approval handler."""
        agent = Agent(
            require_approval=True,
            approval_handler=auto_reject_handler,
        )
        assert agent._approval_config is not None

    def test_agent_no_approval_by_default(self):
        """Agent has no approval config by default."""
        agent = Agent()
        assert agent._approval_config is None


# =============================================================================
# Session Pause/Resume Tests
# =============================================================================


class TestSessionPauseResume:
    """Tests for session-based pause/resume HITL."""

    def test_agent_with_session_and_pause_before(self):
        """Agent accepts session with pause_before tools."""
        session = memory()
        agent = Agent(
            session=session,
            pause_before=["dangerous_tool"],
        )

        assert agent._session_config is not None
        assert "dangerous_tool" in agent._session_config.pause_before

    def test_agent_with_session_and_pause_after(self):
        """Agent accepts session with pause_after tools."""
        session = memory()
        agent = Agent(
            session=session,
            pause_after=["review_tool"],
        )

        assert agent._session_config is not None
        assert "review_tool" in agent._session_config.pause_after

    def test_agent_with_both_pause_before_and_after(self):
        """Agent accepts both pause_before and pause_after."""
        session = memory()
        agent = Agent(
            session=session,
            pause_before=["dangerous"],
            pause_after=["review"],
        )

        assert agent._session_config is not None
        assert "dangerous" in agent._session_config.pause_before
        assert "review" in agent._session_config.pause_after

    def test_session_config_requires_session(self):
        """pause_before/after without session doesn't create config."""
        agent = Agent(pause_before=["dangerous"])
        # Without session, no session config is created
        assert agent._session_config is None


# =============================================================================
# Callable Approval Condition Tests
# =============================================================================


class TestCallableApprovalCondition:
    """Tests for callable approval conditions."""

    def test_callable_condition_function(self):
        """Callable condition can inspect tool name and args."""

        def condition(tool_name: str, args: dict) -> bool:
            if tool_name == "transfer":
                return args.get("amount", 0) > 1000
            return False

        config = ApprovalConfig(require_approval=condition)

        assert config.needs_approval("transfer", {"amount": 500}) is False
        assert config.needs_approval("transfer", {"amount": 2000}) is True
        assert config.needs_approval("other", {}) is False
