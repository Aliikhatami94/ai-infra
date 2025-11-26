"""Tests for session management (persistence, pause/resume)."""

import pytest

from ai_infra.llm.session import (
    MemoryStorage,
    PendingAction,
    ResumeDecision,
    SessionConfig,
    SessionResult,
    generate_session_id,
    get_pending_action,
    is_paused,
    memory,
)


class TestSessionResult:
    """Test SessionResult model."""

    def test_not_paused(self):
        result = SessionResult(
            content="Hello!",
            paused=False,
            session_id="test-123",
        )
        assert result.content == "Hello!"
        assert result.paused is False
        assert result.pending_action is None
        assert result.session_id == "test-123"

    def test_paused_with_pending_action(self):
        pending = PendingAction(
            id="action-1",
            action_type="tool_call",
            tool_name="delete_file",
            args={"path": "/tmp/test.txt"},
            message="Delete file /tmp/test.txt?",
        )
        result = SessionResult(
            content="",
            paused=True,
            pending_action=pending,
            session_id="test-123",
        )
        assert result.paused is True
        assert result.pending_action is not None
        assert result.pending_action.tool_name == "delete_file"


class TestPendingAction:
    """Test PendingAction model."""

    def test_tool_call_action(self):
        action = PendingAction(
            id="action-1",
            action_type="tool_call",
            tool_name="transfer_money",
            args={"amount": 1000, "to": "bob"},
        )
        assert action.action_type == "tool_call"
        assert action.tool_name == "transfer_money"
        assert action.args["amount"] == 1000

    def test_output_review_action(self):
        action = PendingAction(
            id="action-2",
            action_type="output_review",
            message="Review model output before sending",
        )
        assert action.action_type == "output_review"
        assert action.tool_name is None


class TestResumeDecision:
    """Test ResumeDecision model."""

    def test_approve(self):
        decision = ResumeDecision(approved=True, reason="Looks good")
        assert decision.approved is True
        assert decision.modified_args is None

    def test_reject(self):
        decision = ResumeDecision(approved=False, reason="Too risky")
        assert decision.approved is False

    def test_approve_with_modified_args(self):
        decision = ResumeDecision(
            approved=True,
            modified_args={"amount": 100},
            reason="Reduced amount",
        )
        assert decision.approved is True
        assert decision.modified_args == {"amount": 100}


class TestMemoryStorage:
    """Test MemoryStorage backend."""

    def test_creation(self):
        storage = memory()
        assert isinstance(storage, MemoryStorage)

    def test_get_checkpointer(self):
        storage = memory()
        checkpointer = storage.get_checkpointer()
        assert checkpointer is not None
        # Should be MemorySaver from langgraph
        assert hasattr(checkpointer, "get")
        assert hasattr(checkpointer, "put")

    def test_get_store(self):
        storage = memory()
        store = storage.get_store()
        # MemoryStorage doesn't provide a store by default
        assert store is None


class TestSessionConfig:
    """Test SessionConfig dataclass."""

    def test_get_config(self):
        storage = memory()
        config = SessionConfig(
            storage=storage,
            pause_before=["delete_file"],
            pause_after=["send_email"],
        )

        langgraph_config = config.get_config("user-123")
        assert langgraph_config == {"configurable": {"thread_id": "user-123"}}

    def test_pause_before_after(self):
        storage = memory()
        config = SessionConfig(
            storage=storage,
            pause_before=["dangerous_tool"],
            pause_after=["notify_tool"],
        )
        assert "dangerous_tool" in config.pause_before
        assert "notify_tool" in config.pause_after


class TestSessionHelpers:
    """Test session helper functions."""

    def test_generate_session_id(self):
        id1 = generate_session_id()
        id2 = generate_session_id()
        assert id1 != id2
        assert len(id1) == 36  # UUID format

    def test_is_paused_with_session_result(self):
        result = SessionResult(content="", paused=True, session_id="test")
        assert is_paused(result) is True

        result2 = SessionResult(content="Hello", paused=False, session_id="test")
        assert is_paused(result2) is False

    def test_is_paused_with_langgraph_interrupt(self):
        # Simulate LangGraph interrupt format
        result = {"__interrupt__": [{"value": {"tool_name": "delete"}}]}
        assert is_paused(result) is True

        result2 = {"messages": []}
        assert is_paused(result2) is False

    def test_get_pending_action_from_session_result(self):
        pending = PendingAction(
            id="action-1",
            action_type="tool_call",
            tool_name="delete_file",
        )
        result = SessionResult(
            content="",
            paused=True,
            pending_action=pending,
            session_id="test",
        )
        action = get_pending_action(result)
        assert action is not None
        assert action.tool_name == "delete_file"

    def test_get_pending_action_not_paused(self):
        result = SessionResult(content="Hello", paused=False, session_id="test")
        action = get_pending_action(result)
        assert action is None


class TestAgentWithSession:
    """Test Agent integration with sessions."""

    def test_agent_without_session(self):
        """Agent without session returns plain string."""
        from ai_infra.llm import Agent

        agent = Agent(tools=[])
        # Without session, run() returns str (legacy behavior)
        # We can't easily test without API keys, so just check initialization
        assert agent._session_config is None

    def test_agent_with_session(self):
        """Agent with session is properly configured."""
        from ai_infra.llm import Agent

        agent = Agent(
            tools=[],
            session=memory(),
            pause_before=["dangerous_tool"],
        )
        assert agent._session_config is not None
        assert "dangerous_tool" in agent._session_config.pause_before

    def test_resume_requires_session(self):
        """resume() raises error without session."""
        from ai_infra.llm import Agent

        agent = Agent(tools=[])
        with pytest.raises(ValueError, match="requires session"):
            agent.resume(session_id="test-123", approved=True)

    @pytest.mark.asyncio
    async def test_aresume_requires_session(self):
        """aresume() raises error without session."""
        from ai_infra.llm import Agent

        agent = Agent(tools=[])
        with pytest.raises(ValueError, match="requires session"):
            await agent.aresume(session_id="test-123", approved=True)
