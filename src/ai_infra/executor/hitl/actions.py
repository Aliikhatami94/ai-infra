"""HITL action types and data models.

Phase 4.1.1 of EXECUTOR_1.md: Define HITL action types.

This module defines:
- HITLActionType: Enum of all supported human actions
- HITLAction: Dataclass representing a human action
- HITLProposal: What the agent proposes to do
- HITLResponse: Agent's response to a human action
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class HITLActionType(Enum):
    """Types of human actions in the collaboration loop.

    Phase 4.1.1: Rich set of actions beyond binary approve/reject.

    Actions:
        APPROVE: Accept the proposal and continue execution.
        REJECT: Stop execution or request a different approach.
        EDIT: Modify the proposal with specific changes.
        SUGGEST: Provide a hint without full specification.
        EXPLAIN: Ask for reasoning behind the proposal.
        ROLLBACK: Return to a previous checkpoint state.
        SKIP: Skip the current task and continue to next.
        DELEGATE: Hand off to a different agent type.
    """

    APPROVE = "approve"
    """Accept the proposal and continue execution."""

    REJECT = "reject"
    """Stop execution or request a different approach."""

    EDIT = "edit"
    """Modify the proposal with specific changes."""

    SUGGEST = "suggest"
    """Provide a hint without full specification."""

    EXPLAIN = "explain"
    """Ask for reasoning behind the proposal."""

    ROLLBACK = "rollback"
    """Return to a previous checkpoint state."""

    SKIP = "skip"
    """Skip the current task and continue to next."""

    DELEGATE = "delegate"
    """Hand off to a different agent type."""


@dataclass
class HITLAction:
    """A human action in the collaboration loop.

    Represents what the user wants to do with the current proposal.

    Attributes:
        type: The type of action (approve, edit, suggest, etc.).
        content: Optional content for the action (edit text, question, etc.).
        target: Optional target for the action (checkpoint name, agent type).
        timestamp: When the action was taken.
        metadata: Additional metadata about the action.

    Example:
        ```python
        # Simple approval
        action = HITLAction(type=HITLActionType.APPROVE)

        # Edit with specific changes
        action = HITLAction(
            type=HITLActionType.EDIT,
            content="Use async/await instead of threads for concurrency",
        )

        # Ask for explanation
        action = HITLAction(
            type=HITLActionType.EXPLAIN,
            content="Why did you choose SQLite over PostgreSQL?",
        )

        # Rollback to checkpoint
        action = HITLAction(
            type=HITLActionType.ROLLBACK,
            target="checkpoint_task_1.2.1",
        )

        # Delegate to different agent
        action = HITLAction(
            type=HITLActionType.DELEGATE,
            target="reviewer",
        )
        ```
    """

    type: HITLActionType
    content: str | None = None
    target: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "content": self.content,
            "target": self.target,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HITLAction:
        """Create from dictionary."""
        return cls(
            type=HITLActionType(data["type"]),
            content=data.get("content"),
            target=data.get("target"),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.utcnow(),
            metadata=data.get("metadata", {}),
        )


@dataclass
class HITLProposal:
    """A proposal from the agent awaiting human decision.

    Represents what the agent intends to do before getting approval.

    Attributes:
        description: Human-readable description of the proposal.
        task_id: ID of the task this proposal relates to.
        task_title: Title of the task.
        planned_actions: List of planned actions in sequence.
        files_affected: Files that will be modified/created.
        rationale: Why this approach was chosen.
        alternatives_considered: Other approaches that were considered.
        risk_level: Estimated risk (low, medium, high).
        estimated_tokens: Estimated token usage.
        context_summary: Summary of context used for decision.

    Example:
        ```python
        proposal = HITLProposal(
            description="Add logging to the authentication module",
            task_id="2.1.3",
            task_title="Add structured logging",
            planned_actions=[
                "Create logger configuration",
                "Add log calls to auth.py",
                "Add log calls to middleware.py",
            ],
            files_affected=["src/auth.py", "src/middleware.py"],
            rationale="Logging enables debugging and monitoring",
            risk_level="low",
        )
        ```
    """

    description: str
    task_id: str = ""
    task_title: str = ""
    planned_actions: list[str] = field(default_factory=list)
    files_affected: list[str] = field(default_factory=list)
    rationale: str = ""
    alternatives_considered: list[str] = field(default_factory=list)
    risk_level: str = "low"
    estimated_tokens: int = 0
    context_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "description": self.description,
            "task_id": self.task_id,
            "task_title": self.task_title,
            "planned_actions": self.planned_actions,
            "files_affected": self.files_affected,
            "rationale": self.rationale,
            "alternatives_considered": self.alternatives_considered,
            "risk_level": self.risk_level,
            "estimated_tokens": self.estimated_tokens,
            "context_summary": self.context_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HITLProposal:
        """Create from dictionary."""
        return cls(
            description=data["description"],
            task_id=data.get("task_id", ""),
            task_title=data.get("task_title", ""),
            planned_actions=data.get("planned_actions", []),
            files_affected=data.get("files_affected", []),
            rationale=data.get("rationale", ""),
            alternatives_considered=data.get("alternatives_considered", []),
            risk_level=data.get("risk_level", "low"),
            estimated_tokens=data.get("estimated_tokens", 0),
            context_summary=data.get("context_summary", ""),
        )


@dataclass
class HITLResponse:
    """Response from the agent after processing a human action.

    Represents how the agent understood and will respond to the action.

    Attributes:
        understood: Whether the action was understood and can be processed.
        revised_plan: Updated plan if the action changed the approach.
        explanation: Explanation text (for EXPLAIN actions).
        next_step: Description of what happens next.
        error: Error message if the action could not be processed.
        metadata: Additional response metadata.

    Example:
        ```python
        # Response to edit action
        response = HITLResponse(
            understood=True,
            revised_plan="Will use async/await for concurrency as requested",
            next_step="Proceeding with async implementation",
        )

        # Response to explain action
        response = HITLResponse(
            understood=True,
            explanation="SQLite was chosen because this is a local CLI tool...",
            next_step="Waiting for user decision after explanation",
        )

        # Response when action cannot be processed
        response = HITLResponse(
            understood=False,
            error="No checkpoint found for rollback target",
            next_step="Continuing from current state",
        )
        ```
    """

    understood: bool
    revised_plan: str | None = None
    explanation: str | None = None
    next_step: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "understood": self.understood,
            "revised_plan": self.revised_plan,
            "explanation": self.explanation,
            "next_step": self.next_step,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HITLResponse:
        """Create from dictionary."""
        return cls(
            understood=data["understood"],
            revised_plan=data.get("revised_plan"),
            explanation=data.get("explanation"),
            next_step=data.get("next_step", ""),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )
