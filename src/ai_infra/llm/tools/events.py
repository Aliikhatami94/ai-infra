"""Event hooks for HITL observability.

This module provides event hooks that fire during approval workflows,
allowing you to log, audit, or monitor HITL interactions.

Example - Simple logging:
    ```python
    from ai_infra.llm.tools import ApprovalEvents, ApprovalConfig

    def log_event(event: ApprovalEvent):
        print(f"[{event.event_type}] {event.tool_name}: {event.summary}")

    config = ApprovalConfig(
        require_approval=True,
        events=ApprovalEvents(on_event=log_event),
    )
    ```

Example - Audit logging:
    ```python
    import json
    from datetime import datetime

    async def audit_log(event: ApprovalEvent):
        await db.insert("approval_audit", {
            "id": event.id,
            "event_type": event.event_type,
            "tool_name": event.tool_name,
            "args": json.dumps(event.args),
            "user": event.approver,
            "approved": event.approved,
            "timestamp": datetime.utcnow(),
        })

    config = ApprovalConfig(
        require_approval=True,
        events=ApprovalEvents(on_event_async=audit_log),
    )
    ```
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

# Event type literals
EventType = Literal[
    "approval_requested",  # Approval request sent to handler
    "approval_granted",  # Request approved
    "approval_denied",  # Request denied
    "approval_modified",  # Approved with modified args
    "approval_timeout",  # Request timed out
    "approval_error",  # Error during approval
    "output_reviewed",  # Output review completed
    "output_modified",  # Output modified by reviewer
    "output_blocked",  # Output blocked by reviewer
]


@dataclass
class ApprovalEvent:
    """Event fired during approval workflow.

    Attributes:
        id: Unique event ID
        event_type: Type of event
        tool_name: Name of the tool being approved
        args: Tool arguments
        approved: Whether approved (for approval events)
        approver: Who approved/denied
        reason: Reason for decision
        modified_args: Modified arguments if any
        duration_ms: Time taken for approval
        timestamp: When event occurred
        metadata: Additional event data
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = "approval_requested"
    tool_name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    approved: bool | None = None
    approver: str | None = None
    reason: str | None = None
    modified_args: dict[str, Any] | None = None
    duration_ms: float | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        """Human-readable summary of the event."""
        if self.event_type == "approval_requested":
            return f"Requesting approval for {self.tool_name}"
        elif self.event_type == "approval_granted":
            by = f" by {self.approver}" if self.approver else ""
            return f"Approved{by}"
        elif self.event_type == "approval_denied":
            by = f" by {self.approver}" if self.approver else ""
            reason = f": {self.reason}" if self.reason else ""
            return f"Denied{by}{reason}"
        elif self.event_type == "approval_modified":
            return f"Approved with modifications by {self.approver}"
        elif self.event_type == "approval_timeout":
            return f"Approval timed out for {self.tool_name}"
        elif self.event_type == "approval_error":
            return f"Error during approval: {self.reason}"
        elif self.event_type == "output_reviewed":
            return "Output passed review"
        elif self.event_type == "output_modified":
            return f"Output modified: {self.reason}"
        elif self.event_type == "output_blocked":
            return f"Output blocked: {self.reason}"
        return f"{self.event_type}: {self.tool_name}"

    # Factory methods for creating events
    @classmethod
    def requested(
        cls,
        tool_name: str,
        args: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> ApprovalEvent:
        """Create an approval requested event."""
        return cls(
            event_type="approval_requested",
            tool_name=tool_name,
            args=args,
            metadata=metadata or {},
        )

    @classmethod
    def granted(
        cls,
        tool_name: str,
        args: dict[str, Any],
        approver: str | None = None,
        reason: str | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ApprovalEvent:
        """Create an approval granted event."""
        return cls(
            event_type="approval_granted",
            tool_name=tool_name,
            args=args,
            approved=True,
            approver=approver,
            reason=reason,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

    @classmethod
    def denied(
        cls,
        tool_name: str,
        args: dict[str, Any],
        approver: str | None = None,
        reason: str | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ApprovalEvent:
        """Create an approval denied event."""
        return cls(
            event_type="approval_denied",
            tool_name=tool_name,
            args=args,
            approved=False,
            approver=approver,
            reason=reason,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

    @classmethod
    def modified(
        cls,
        tool_name: str,
        args: dict[str, Any],
        modified_args: dict[str, Any],
        approver: str | None = None,
        reason: str | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ApprovalEvent:
        """Create an approval modified event."""
        return cls(
            event_type="approval_modified",
            tool_name=tool_name,
            args=args,
            approved=True,
            approver=approver,
            reason=reason,
            modified_args=modified_args,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

    @classmethod
    def error(
        cls,
        tool_name: str,
        args: dict[str, Any],
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> ApprovalEvent:
        """Create an approval error event."""
        return cls(
            event_type="approval_error",
            tool_name=tool_name,
            args=args,
            reason=reason,
            metadata=metadata or {},
        )


# Type aliases for event handlers
EventHandler = Callable[[ApprovalEvent], None]
AsyncEventHandler = Callable[[ApprovalEvent], Any]  # Returns awaitable


@dataclass
class ApprovalEvents:
    """Event hooks for approval workflow observability.

    Configure callbacks that fire during approval workflows.
    Useful for logging, auditing, monitoring, and analytics.

    Attributes:
        on_event: Sync callback for all events
        on_event_async: Async callback for all events
        on_requested: Callback when approval is requested
        on_granted: Callback when approval is granted
        on_denied: Callback when approval is denied
        on_error: Callback on approval errors
        include_args: Whether to include tool args in events (default: True)
        include_metadata: Whether to include metadata in events

    Example - Log all events:
        ```python
        events = ApprovalEvents(
            on_event=lambda e: print(f"[{e.event_type}] {e.summary}")
        )
        ```

    Example - Selective logging:
        ```python
        events = ApprovalEvents(
            on_denied=lambda e: logger.warning(f"Denied: {e.tool_name}"),
            on_error=lambda e: logger.error(f"Error: {e.reason}"),
        )
        ```

    Example - Audit without sensitive args:
        ```python
        events = ApprovalEvents(
            on_event=audit_callback,
            include_args=False,  # Don't log args
        )
        ```
    """

    # General event handlers
    on_event: EventHandler | None = None
    on_event_async: AsyncEventHandler | None = None

    # Specific event handlers
    on_requested: EventHandler | None = None
    on_requested_async: AsyncEventHandler | None = None
    on_granted: EventHandler | None = None
    on_granted_async: AsyncEventHandler | None = None
    on_denied: EventHandler | None = None
    on_denied_async: AsyncEventHandler | None = None
    on_error: EventHandler | None = None
    on_error_async: AsyncEventHandler | None = None

    # Options
    include_args: bool = True
    include_metadata: bool = True

    def _prepare_event(self, event: ApprovalEvent) -> ApprovalEvent:
        """Prepare event for emission (optionally strip sensitive data)."""
        if not self.include_args:
            event = ApprovalEvent(
                id=event.id,
                event_type=event.event_type,
                tool_name=event.tool_name,
                args={},  # Stripped
                approved=event.approved,
                approver=event.approver,
                reason=event.reason,
                modified_args=None,  # Stripped
                duration_ms=event.duration_ms,
                timestamp=event.timestamp,
                metadata=event.metadata if self.include_metadata else {},
            )
        elif not self.include_metadata:
            event = ApprovalEvent(
                id=event.id,
                event_type=event.event_type,
                tool_name=event.tool_name,
                args=event.args,
                approved=event.approved,
                approver=event.approver,
                reason=event.reason,
                modified_args=event.modified_args,
                duration_ms=event.duration_ms,
                timestamp=event.timestamp,
                metadata={},  # Stripped
            )
        return event

    def emit(self, event: ApprovalEvent) -> None:
        """Emit an event synchronously."""
        event = self._prepare_event(event)

        # Call general handler
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                pass  # Don't let event handler errors break the flow

        # Call specific handlers
        handler = self._get_sync_handler(event.event_type)
        if handler:
            try:
                handler(event)
            except Exception:
                pass

    async def emit_async(self, event: ApprovalEvent) -> None:
        """Emit an event asynchronously."""
        import asyncio
        import inspect

        event = self._prepare_event(event)

        # Call general handler
        if self.on_event_async:
            try:
                result = self.on_event_async(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass
        elif self.on_event:
            try:
                await asyncio.to_thread(self.on_event, event)
            except Exception:
                pass

        # Call specific handlers
        handler = self._get_async_handler(event.event_type)
        sync_handler = self._get_sync_handler(event.event_type)

        if handler:
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass
        elif sync_handler:
            try:
                await asyncio.to_thread(sync_handler, event)
            except Exception:
                pass

    def _get_sync_handler(self, event_type: EventType) -> EventHandler | None:
        """Get sync handler for event type."""
        handlers = {
            "approval_requested": self.on_requested,
            "approval_granted": self.on_granted,
            "approval_denied": self.on_denied,
            "approval_error": self.on_error,
        }
        return handlers.get(event_type)

    def _get_async_handler(self, event_type: EventType) -> AsyncEventHandler | None:
        """Get async handler for event type."""
        handlers = {
            "approval_requested": self.on_requested_async,
            "approval_granted": self.on_granted_async,
            "approval_denied": self.on_denied_async,
            "approval_error": self.on_error_async,
        }
        return handlers.get(event_type)


# =============================================================================
# Pre-built Event Handlers
# =============================================================================


def create_logging_handler(
    logger: Any = None,
    level: str = "info",
) -> EventHandler:
    """Create a logging event handler.

    Args:
        logger: Logger instance (uses print if None)
        level: Log level (info, warning, error)

    Returns:
        Event handler that logs events

    Example:
        ```python
        import logging
        logger = logging.getLogger("hitl")

        events = ApprovalEvents(
            on_event=create_logging_handler(logger, level="info"),
        )
        ```
    """

    def handler(event: ApprovalEvent) -> None:
        message = f"[HITL] {event.event_type}: {event.summary}"

        if logger:
            log_fn = getattr(logger, level, logger.info)
            log_fn(message)
        else:
            print(message)

    return handler


def create_json_logger(
    output: Any = None,
    pretty: bool = False,
) -> EventHandler:
    """Create a JSON logging handler.

    Args:
        output: File-like object to write to (uses print if None)
        pretty: Whether to pretty-print JSON

    Returns:
        Event handler that logs JSON events

    Example:
        ```python
        with open("audit.jsonl", "a") as f:
            events = ApprovalEvents(
                on_event=create_json_logger(f),
            )
        ```
    """
    import json

    def handler(event: ApprovalEvent) -> None:
        data = {
            "id": event.id,
            "type": event.event_type,
            "tool": event.tool_name,
            "approved": event.approved,
            "approver": event.approver,
            "reason": event.reason,
            "duration_ms": event.duration_ms,
            "timestamp": event.timestamp.isoformat(),
        }

        indent = 2 if pretty else None
        line = json.dumps(data, indent=indent, default=str)

        if output:
            output.write(line + "\n")
            output.flush()
        else:
            print(line)

    return handler


def create_metrics_counter(
    counters: dict[str, int] | None = None,
) -> EventHandler:
    """Create a simple metrics counter handler.

    Args:
        counters: Dict to store counts (created if None)

    Returns:
        Event handler that counts events

    Example:
        ```python
        metrics = {}
        events = ApprovalEvents(
            on_event=create_metrics_counter(metrics),
        )

        # Later:
        print(metrics)
        # {"approval_granted": 10, "approval_denied": 2, ...}
        ```
    """
    if counters is None:
        counters = {}

    def handler(event: ApprovalEvent) -> None:
        key = event.event_type
        counters[key] = counters.get(key, 0) + 1

        # Also track per-tool
        tool_key = f"{event.tool_name}:{event.event_type}"
        counters[tool_key] = counters.get(tool_key, 0) + 1

    return handler
