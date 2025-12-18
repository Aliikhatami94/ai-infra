"""Tests for HITL event hooks.

These tests verify that approval events are properly created,
emitted, and handled during approval workflows.
"""

import pytest

from ai_infra.llm.tools.events import ApprovalEvent, ApprovalEvents

# =============================================================================
# ApprovalEvent Factory Tests
# =============================================================================


class TestApprovalEventFactories:
    """Tests for ApprovalEvent factory methods."""

    def test_requested_creates_event(self):
        """Test creating an approval requested event."""
        event = ApprovalEvent.requested(
            tool_name="delete_file",
            args={"path": "/tmp/test.txt"},
        )

        assert event.event_type == "approval_requested"
        assert event.tool_name == "delete_file"
        assert event.args == {"path": "/tmp/test.txt"}
        assert event.approved is None
        assert event.id  # Has an ID

    def test_granted_creates_event(self):
        """Test creating an approval granted event."""
        event = ApprovalEvent.granted(
            tool_name="delete_file",
            args={"path": "/tmp/test.txt"},
            approver="admin",
            reason="Looks safe",
        )

        assert event.event_type == "approval_granted"
        assert event.tool_name == "delete_file"
        assert event.approved is True
        assert event.approver == "admin"
        assert event.reason == "Looks safe"

    def test_denied_creates_event(self):
        """Test creating an approval denied event."""
        event = ApprovalEvent.denied(
            tool_name="delete_file",
            args={"path": "/etc/passwd"},
            approver="security",
            reason="Dangerous file",
        )

        assert event.event_type == "approval_denied"
        assert event.approved is False
        assert event.approver == "security"
        assert event.reason == "Dangerous file"

    def test_modified_creates_event(self):
        """Test creating an approval modified event."""
        event = ApprovalEvent.modified(
            tool_name="transfer_money",
            args={"amount": 10000, "to": "account123"},
            modified_args={"amount": 5000, "to": "account123"},
            approver="manager",
            reason="Reduced amount",
        )

        assert event.event_type == "approval_modified"
        assert event.approved is True
        assert event.modified_args == {"amount": 5000, "to": "account123"}
        assert event.reason == "Reduced amount"

    def test_error_creates_event(self):
        """Test creating an approval error event."""
        event = ApprovalEvent.error(
            tool_name="risky_tool",
            args={"x": 1},
            reason="Handler timed out",
        )

        assert event.event_type == "approval_error"
        assert event.reason == "Handler timed out"

    def test_event_has_timestamp(self):
        """Test that events have timestamps."""
        event = ApprovalEvent.requested("test_tool", {})
        assert event.timestamp is not None

    def test_event_summary_requested(self):
        """Test summary for requested event."""
        event = ApprovalEvent.requested("delete_file", {})
        assert "delete_file" in event.summary
        assert "Requesting" in event.summary

    def test_event_summary_granted(self):
        """Test summary for granted event."""
        event = ApprovalEvent.granted("delete_file", {}, approver="admin")
        assert "Approved" in event.summary
        assert "admin" in event.summary

    def test_event_summary_denied(self):
        """Test summary for denied event."""
        event = ApprovalEvent.denied("delete_file", {}, reason="Too dangerous")
        assert "Denied" in event.summary
        assert "Too dangerous" in event.summary


# =============================================================================
# ApprovalEvents Hook Tests
# =============================================================================


class TestApprovalEventsSync:
    """Tests for sync event handlers."""

    def test_on_event_callback(self):
        """Test that on_event callback is called."""
        received_events: list[ApprovalEvent] = []

        events = ApprovalEvents(
            on_event=lambda e: received_events.append(e),
        )

        event = ApprovalEvent.requested("test", {})
        events.emit(event)

        assert len(received_events) == 1
        assert received_events[0].tool_name == "test"

    def test_on_requested_callback(self):
        """Test that on_requested callback is called."""
        received_events: list[ApprovalEvent] = []

        events = ApprovalEvents(
            on_requested=lambda e: received_events.append(e),
        )

        event = ApprovalEvent.requested("test", {})
        events.emit(event)

        assert len(received_events) == 1

    def test_on_granted_callback(self):
        """Test that on_granted callback is called."""
        received_events: list[ApprovalEvent] = []

        events = ApprovalEvents(
            on_granted=lambda e: received_events.append(e),
        )

        # Emit requested event - should not trigger on_granted
        event = ApprovalEvent.requested("test", {})
        events.emit(event)
        assert len(received_events) == 0

        # Emit granted event - should trigger
        event = ApprovalEvent.granted("test", {})
        events.emit(event)
        assert len(received_events) == 1

    def test_on_denied_callback(self):
        """Test that on_denied callback is called."""
        received_events: list[ApprovalEvent] = []

        events = ApprovalEvents(
            on_denied=lambda e: received_events.append(e),
        )

        event = ApprovalEvent.denied("test", {})
        events.emit(event)

        assert len(received_events) == 1

    def test_handler_errors_dont_break_flow(self):
        """Test that handler errors are caught and don't break the flow."""

        def bad_handler(e: ApprovalEvent):
            raise RuntimeError("Handler error")

        events = ApprovalEvents(on_event=bad_handler)

        # Should not raise
        event = ApprovalEvent.requested("test", {})
        events.emit(event)

    def test_include_args_false_strips_args(self):
        """Test that include_args=False strips args from events."""
        received_events: list[ApprovalEvent] = []

        events = ApprovalEvents(
            on_event=lambda e: received_events.append(e),
            include_args=False,
        )

        event = ApprovalEvent.requested("test", {"secret": "password"})
        events.emit(event)

        assert received_events[0].args == {}

    def test_include_metadata_false_strips_metadata(self):
        """Test that include_metadata=False strips metadata from events."""
        received_events: list[ApprovalEvent] = []

        events = ApprovalEvents(
            on_event=lambda e: received_events.append(e),
            include_metadata=False,
        )

        event = ApprovalEvent.requested("test", {}, metadata={"ip": "127.0.0.1"})
        events.emit(event)

        assert received_events[0].metadata == {}


class TestApprovalEventsAsync:
    """Tests for async event handlers."""

    @pytest.mark.asyncio
    async def test_on_event_async_callback(self):
        """Test that on_event_async callback is called."""
        received_events: list[ApprovalEvent] = []

        async def handler(e: ApprovalEvent):
            received_events.append(e)

        events = ApprovalEvents(on_event_async=handler)

        event = ApprovalEvent.requested("test", {})
        await events.emit_async(event)

        assert len(received_events) == 1

    @pytest.mark.asyncio
    async def test_sync_handler_called_in_thread(self):
        """Test that sync handlers are called when async not available."""
        received_events: list[ApprovalEvent] = []

        def sync_handler(e: ApprovalEvent):
            received_events.append(e)

        events = ApprovalEvents(on_event=sync_handler)

        event = ApprovalEvent.requested("test", {})
        await events.emit_async(event)

        assert len(received_events) == 1

    @pytest.mark.asyncio
    async def test_async_handler_errors_dont_break_flow(self):
        """Test that async handler errors don't break the flow."""

        async def bad_handler(e: ApprovalEvent):
            raise RuntimeError("Async handler error")

        events = ApprovalEvents(on_event_async=bad_handler)

        # Should not raise
        event = ApprovalEvent.requested("test", {})
        await events.emit_async(event)

    @pytest.mark.asyncio
    async def test_on_granted_async(self):
        """Test async granted handler."""
        received_events: list[ApprovalEvent] = []

        async def handler(e: ApprovalEvent):
            received_events.append(e)

        events = ApprovalEvents(on_granted_async=handler)

        event = ApprovalEvent.granted("test", {})
        await events.emit_async(event)

        assert len(received_events) == 1

    @pytest.mark.asyncio
    async def test_on_denied_async(self):
        """Test async denied handler."""
        received_events: list[ApprovalEvent] = []

        async def handler(e: ApprovalEvent):
            received_events.append(e)

        events = ApprovalEvents(on_denied_async=handler)

        event = ApprovalEvent.denied("test", {})
        await events.emit_async(event)

        assert len(received_events) == 1


# =============================================================================
# Integration with ApprovalConfig Tests
# =============================================================================


class TestEventsWithApprovalConfig:
    """Tests for events integration with ApprovalConfig."""

    @pytest.mark.asyncio
    async def test_events_fired_during_approval(self):
        """Test that events are fired during approval workflow."""
        from langchain_core.tools import tool

        from ai_infra.llm.tools import ApprovalConfig, ApprovalResponse
        from ai_infra.llm.tools.hitl import wrap_tool_for_approval

        received_events: list[ApprovalEvent] = []

        async def event_handler(e: ApprovalEvent):
            received_events.append(e)

        async def auto_approve(req):
            return ApprovalResponse.approve(approver="test")

        @tool
        def my_tool(x: int) -> int:
            """Test tool."""
            return x * 2

        config = ApprovalConfig(
            approval_handler_async=auto_approve,
            require_approval=True,
            events=ApprovalEvents(on_event_async=event_handler),
        )

        wrapped = wrap_tool_for_approval(my_tool, config)
        result = await wrapped.ainvoke({"x": 5})

        assert result == 10
        # Should have both requested and granted events
        assert len(received_events) == 2
        assert received_events[0].event_type == "approval_requested"
        assert received_events[1].event_type == "approval_granted"

    @pytest.mark.asyncio
    async def test_denied_event_fired(self):
        """Test that denied event is fired when approval is rejected."""
        from langchain_core.tools import tool

        from ai_infra.llm.tools import ApprovalConfig, ApprovalResponse
        from ai_infra.llm.tools.hitl import wrap_tool_for_approval

        received_events: list[ApprovalEvent] = []

        async def auto_deny(req):
            return ApprovalResponse.reject(reason="Test rejection")

        @tool
        def my_tool(x: int) -> int:
            """Test tool."""
            return x * 2

        config = ApprovalConfig(
            approval_handler_async=auto_deny,
            require_approval=True,
            events=ApprovalEvents(
                on_event_async=lambda e: received_events.append(e),
            ),
        )

        wrapped = wrap_tool_for_approval(my_tool, config)
        result = await wrapped.ainvoke({"x": 5})

        assert "rejected" in result.lower()
        # Should have both requested and denied events
        assert len(received_events) == 2
        assert received_events[0].event_type == "approval_requested"
        assert received_events[1].event_type == "approval_denied"

    @pytest.mark.asyncio
    async def test_modified_event_fired(self):
        """Test that modified event is fired when args are modified."""
        from langchain_core.tools import tool

        from ai_infra.llm.tools import ApprovalConfig, ApprovalResponse
        from ai_infra.llm.tools.hitl import wrap_tool_for_approval

        received_events: list[ApprovalEvent] = []

        async def modify_handler(req):
            return ApprovalResponse.approve(
                modified_args={"x": 100},
                reason="Modified amount",
            )

        @tool
        def my_tool(x: int) -> int:
            """Test tool."""
            return x * 2

        config = ApprovalConfig(
            approval_handler_async=modify_handler,
            require_approval=True,
            events=ApprovalEvents(
                on_event_async=lambda e: received_events.append(e),
            ),
        )

        wrapped = wrap_tool_for_approval(my_tool, config)
        result = await wrapped.ainvoke({"x": 5})

        # Should use modified args (100 * 2 = 200)
        assert result == 200
        # Should have modified event
        assert received_events[1].event_type == "approval_modified"
        assert received_events[1].modified_args == {"x": 100}

    @pytest.mark.asyncio
    async def test_no_events_when_approval_not_needed(self):
        """Test that no events fire when approval is not needed."""
        from langchain_core.tools import tool

        from ai_infra.llm.tools import ApprovalConfig
        from ai_infra.llm.tools.hitl import wrap_tool_for_approval

        received_events: list[ApprovalEvent] = []

        @tool
        def my_tool(x: int) -> int:
            """Test tool."""
            return x * 2

        config = ApprovalConfig(
            require_approval=False,  # No approval needed
            events=ApprovalEvents(
                on_event_async=lambda e: received_events.append(e),
            ),
        )

        wrapped = wrap_tool_for_approval(my_tool, config)
        result = await wrapped.ainvoke({"x": 5})

        assert result == 10
        # No events since approval not needed
        assert len(received_events) == 0

    @pytest.mark.asyncio
    async def test_selective_event_handlers(self):
        """Test using selective event handlers."""
        from langchain_core.tools import tool

        from ai_infra.llm.tools import ApprovalConfig, ApprovalResponse
        from ai_infra.llm.tools.hitl import wrap_tool_for_approval

        granted_events: list[ApprovalEvent] = []
        denied_events: list[ApprovalEvent] = []

        async def auto_approve(req):
            return ApprovalResponse.approve()

        @tool
        def my_tool(x: int) -> int:
            """Test tool."""
            return x * 2

        config = ApprovalConfig(
            approval_handler_async=auto_approve,
            require_approval=True,
            events=ApprovalEvents(
                on_granted_async=lambda e: granted_events.append(e),
                on_denied_async=lambda e: denied_events.append(e),
            ),
        )

        wrapped = wrap_tool_for_approval(my_tool, config)
        await wrapped.ainvoke({"x": 5})

        # Only granted handler should be called
        assert len(granted_events) == 1
        assert len(denied_events) == 0
