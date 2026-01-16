"""Human-in-the-Loop (HITL) implementation for executor graph.

Phase 1.5: Human-in-the-Loop Implementation.

This module provides:
- InterruptConfig: Configuration for interrupt points
- HITLManager: Manages interrupt/resume state and decisions
- Resume logic: Load state, apply decision, continue execution
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.state import ExecutorGraphState

logger = get_logger("executor.hitl")


# =============================================================================
# Constants
# =============================================================================

EXECUTOR_DIR = ".executor"
HITL_STATE_FILE = "hitl_state.json"


# =============================================================================
# Interrupt Configuration (Phase 1.5.1)
# =============================================================================


class InterruptPoint(str, Enum):
    """Predefined interrupt points in the executor graph.

    Phase 1.5.1: Standard interrupt points for HITL.
    """

    BEFORE_EXECUTE = "execute_task"
    """Pause before executing a task (approve before execution)."""

    AFTER_VERIFY = "verify_task"
    """Pause after verification (review results)."""

    BEFORE_CHECKPOINT = "checkpoint"
    """Pause before git checkpoint (review before commit)."""

    AFTER_FAILURE = "handle_failure"
    """Pause after failure handling (review error, decide retry)."""


@dataclass
class InterruptConfig:
    """Configuration for Human-in-the-Loop interrupt points.

    Phase 1.5.1: Configurable interrupt points.

    Example:
        ```python
        # Approve before every task execution
        config = InterruptConfig(
            interrupt_before=["execute_task"],
        )

        # Review after verification
        config = InterruptConfig(
            interrupt_after=["verify_task"],
        )

        # Both
        config = InterruptConfig(
            interrupt_before=["execute_task"],
            interrupt_after=["verify_task"],
        )
        ```
    """

    interrupt_before: list[str] = field(default_factory=list)
    """Nodes to pause BEFORE execution (approval required)."""

    interrupt_after: list[str] = field(default_factory=list)
    """Nodes to pause AFTER execution (review required)."""

    require_approval: bool = True
    """Whether to require explicit approval to continue."""

    timeout_seconds: float | None = None
    """Optional timeout for human response. None = wait forever."""

    @classmethod
    def approval_mode(cls) -> InterruptConfig:
        """Create config that requires approval before execution.

        Returns:
            InterruptConfig with interrupt_before=["execute_task"].
        """
        return cls(
            interrupt_before=[InterruptPoint.BEFORE_EXECUTE.value],
            require_approval=True,
        )

    @classmethod
    def review_mode(cls) -> InterruptConfig:
        """Create config that pauses after verification for review.

        Returns:
            InterruptConfig with interrupt_after=["verify_task"].
        """
        return cls(
            interrupt_after=[InterruptPoint.AFTER_VERIFY.value],
            require_approval=True,
        )

    @classmethod
    def full_control_mode(cls) -> InterruptConfig:
        """Create config with maximum human control.

        Returns:
            InterruptConfig with both before and after interrupts.
        """
        return cls(
            interrupt_before=[InterruptPoint.BEFORE_EXECUTE.value],
            interrupt_after=[InterruptPoint.AFTER_VERIFY.value],
            require_approval=True,
        )

    @classmethod
    def no_interrupt(cls) -> InterruptConfig:
        """Create config with no interrupts (autonomous mode).

        Returns:
            InterruptConfig with empty interrupt lists.
        """
        return cls(
            interrupt_before=[],
            interrupt_after=[],
            require_approval=False,
        )


# =============================================================================
# HITL Decision (Phase 1.5.2)
# =============================================================================


class HITLDecision(str, Enum):
    """Possible human decisions at an interrupt point.

    Phase 1.5.2: Decision types for resume logic.
    """

    APPROVE = "approve"
    """Continue with the action (approve task execution)."""

    REJECT = "reject"
    """Skip the current task and move to next."""

    ABORT = "abort"
    """Abort the entire execution."""

    RETRY = "retry"
    """Retry the current action (after failure)."""

    MODIFY = "modify"
    """Modify the context/prompt before continuing."""


@dataclass
class HITLState:
    """Persisted state for HITL interrupt/resume.

    Phase 1.5.2: State persisted to `.executor/hitl_state.json`.
    """

    thread_id: str
    """Graph thread ID for resuming."""

    interrupt_node: str
    """Node where execution was interrupted."""

    interrupt_type: str
    """'before' or 'after' the node."""

    task_id: str | None
    """Current task ID (if any)."""

    task_title: str | None
    """Current task title for display."""

    interrupted_at: str
    """ISO timestamp of interrupt."""

    context_summary: str
    """Brief summary of current context for human review."""

    decision: str | None = None
    """Human decision (once made)."""

    decision_at: str | None = None
    """ISO timestamp of decision."""

    notes: str | None = None
    """Optional human notes about the decision."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "thread_id": self.thread_id,
            "interrupt_node": self.interrupt_node,
            "interrupt_type": self.interrupt_type,
            "task_id": self.task_id,
            "task_title": self.task_title,
            "interrupted_at": self.interrupted_at,
            "context_summary": self.context_summary,
            "decision": self.decision,
            "decision_at": self.decision_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HITLState:
        """Create from dict."""
        return cls(
            thread_id=data["thread_id"],
            interrupt_node=data["interrupt_node"],
            interrupt_type=data["interrupt_type"],
            task_id=data.get("task_id"),
            task_title=data.get("task_title"),
            interrupted_at=data["interrupted_at"],
            context_summary=data.get("context_summary", ""),
            decision=data.get("decision"),
            decision_at=data.get("decision_at"),
            notes=data.get("notes"),
        )


# =============================================================================
# HITL Manager (Phase 1.5.2)
# =============================================================================


class HITLManager:
    """Manages Human-in-the-Loop state and decisions.

    Phase 1.5.2: Implements resume logic for interrupted executions.

    Example:
        ```python
        manager = HITLManager(Path("/project"))

        # Check for pending interrupt
        if manager.has_pending_interrupt():
            state = manager.get_pending_state()
            print(f"Waiting for decision on: {state.task_title}")

        # Apply decision and resume
        manager.apply_decision(HITLDecision.APPROVE, notes="Looks good")
        ```
    """

    def __init__(self, project_root: Path):
        """Initialize the HITL manager.

        Args:
            project_root: Root directory of the project.
        """
        self.project_root = project_root
        self._executor_dir = project_root / EXECUTOR_DIR
        self._hitl_file = self._executor_dir / HITL_STATE_FILE

    def save_interrupt_state(
        self,
        thread_id: str,
        interrupt_node: str,
        interrupt_type: str,
        graph_state: ExecutorGraphState,
    ) -> HITLState:
        """Save state when execution is interrupted.

        Args:
            thread_id: Graph thread ID.
            interrupt_node: Node where interrupted.
            interrupt_type: 'before' or 'after'.
            graph_state: Current graph state.

        Returns:
            The saved HITLState.
        """
        self._executor_dir.mkdir(parents=True, exist_ok=True)

        current_task = graph_state.get("current_task")
        task_id = str(current_task.id) if current_task else None
        task_title = current_task.title if current_task else None

        # Build context summary
        context_summary = self._build_context_summary(graph_state, interrupt_node)

        hitl_state = HITLState(
            thread_id=thread_id,
            interrupt_node=interrupt_node,
            interrupt_type=interrupt_type,
            task_id=task_id,
            task_title=task_title,
            interrupted_at=datetime.now(UTC).isoformat(),
            context_summary=context_summary,
        )

        self._hitl_file.write_text(
            json.dumps(hitl_state.to_dict(), indent=2),
            encoding="utf-8",
        )

        logger.info(f"Saved interrupt state: {interrupt_type} {interrupt_node} (task: {task_id})")

        return hitl_state

    def has_pending_interrupt(self) -> bool:
        """Check if there's a pending interrupt waiting for decision.

        Returns:
            True if waiting for human decision.
        """
        if not self._hitl_file.exists():
            return False

        state = self.get_pending_state()
        return state is not None and state.decision is None

    def get_pending_state(self) -> HITLState | None:
        """Get the pending HITL state.

        Returns:
            HITLState or None if no pending interrupt.
        """
        if not self._hitl_file.exists():
            return None

        try:
            data = json.loads(self._hitl_file.read_text(encoding="utf-8"))
            return HITLState.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load HITL state: {e}")
            return None

    def apply_decision(
        self,
        decision: HITLDecision | str,
        notes: str | None = None,
    ) -> HITLState:
        """Apply a human decision to the pending interrupt.

        Args:
            decision: The decision (approve, reject, abort, etc.).
            notes: Optional notes about the decision.

        Returns:
            Updated HITLState with decision.

        Raises:
            ValueError: If no pending interrupt exists.
        """
        state = self.get_pending_state()
        if state is None:
            raise ValueError("No pending interrupt to apply decision to")

        decision_str = decision.value if isinstance(decision, HITLDecision) else decision
        state.decision = decision_str
        state.decision_at = datetime.now(UTC).isoformat()
        state.notes = notes

        self._hitl_file.write_text(
            json.dumps(state.to_dict(), indent=2),
            encoding="utf-8",
        )

        logger.info(f"Applied decision: {decision_str} (notes: {notes})")

        return state

    def clear_interrupt_state(self) -> None:
        """Clear the interrupt state after successful resume."""
        if self._hitl_file.exists():
            self._hitl_file.unlink()
            logger.info("Cleared interrupt state")

    def get_resume_config(self) -> dict[str, Any]:
        """Get the LangGraph config for resuming execution.

        Returns:
            Config dict with thread_id for resuming.

        Raises:
            ValueError: If no pending interrupt exists.
        """
        state = self.get_pending_state()
        if state is None:
            raise ValueError("No pending interrupt to resume from")

        return {
            "configurable": {
                "thread_id": state.thread_id,
            }
        }

    def should_continue_after_decision(self) -> bool:
        """Check if execution should continue based on decision.

        Returns:
            True if decision allows continuation.
        """
        state = self.get_pending_state()
        if state is None or state.decision is None:
            return False

        # Abort means stop
        if state.decision == HITLDecision.ABORT.value:
            return False

        # All other decisions allow some form of continuation
        return True

    def get_decision_action(self) -> str | None:
        """Get the action to take based on decision.

        Returns:
            Action string: 'continue', 'skip', 'abort', 'retry', 'modify'.
        """
        state = self.get_pending_state()
        if state is None or state.decision is None:
            return None

        decision = state.decision

        if decision == HITLDecision.APPROVE.value:
            return "continue"
        elif decision == HITLDecision.REJECT.value:
            return "skip"
        elif decision == HITLDecision.ABORT.value:
            return "abort"
        elif decision == HITLDecision.RETRY.value:
            return "retry"
        elif decision == HITLDecision.MODIFY.value:
            return "modify"
        else:
            return "continue"  # Default to continue

    def _build_context_summary(
        self,
        state: ExecutorGraphState,
        interrupt_node: str,
    ) -> str:
        """Build a human-readable summary of current context.

        Args:
            state: Current graph state.
            interrupt_node: Where execution was interrupted.

        Returns:
            Summary string for human review.
        """
        parts = []

        # Task info
        current_task = state.get("current_task")
        if current_task:
            parts.append(f"Task: [{current_task.id}] {current_task.title}")
            if hasattr(current_task, "description") and current_task.description:
                desc = current_task.description[:200]
                if len(current_task.description) > 200:
                    desc += "..."
                parts.append(f"Description: {desc}")

        # Files modified (for after-verify)
        files = state.get("files_modified", [])
        if files:
            parts.append(f"Files modified: {len(files)}")
            for f in files[:5]:
                parts.append(f"  - {f}")
            if len(files) > 5:
                parts.append(f"  ... and {len(files) - 5} more")

        # Error info (for after-failure)
        error = state.get("error")
        if error:
            parts.append(f"Error: {error.get('message', 'Unknown error')}")

        # Verification result
        if interrupt_node == "verify_task":
            verified = state.get("verified", False)
            parts.append(f"Verification: {'PASSED' if verified else 'FAILED'}")

        return "\n".join(parts) if parts else "No context available"


# =============================================================================
# Helper Functions
# =============================================================================


def create_hitl_config_from_executor_config(
    executor_config: Any,
) -> InterruptConfig:
    """Create InterruptConfig from ExecutorConfig.

    Phase 1.5.1: Via ExecutorConfig option.

    Args:
        executor_config: ExecutorConfig instance.

    Returns:
        InterruptConfig based on executor_config settings.
    """
    # Check for HITL mode flags
    hitl_mode = getattr(executor_config, "hitl_mode", None)
    require_approval = getattr(executor_config, "require_approval", False)

    if hitl_mode == "approval":
        return InterruptConfig.approval_mode()
    elif hitl_mode == "review":
        return InterruptConfig.review_mode()
    elif hitl_mode == "full":
        return InterruptConfig.full_control_mode()
    elif require_approval:
        return InterruptConfig.approval_mode()
    else:
        return InterruptConfig.no_interrupt()


def get_interrupt_lists(
    config: InterruptConfig,
) -> tuple[list[str], list[str]]:
    """Get interrupt_before and interrupt_after lists for Graph.

    Phase 1.5.1: Returns lists for Graph constructor.

    Args:
        config: InterruptConfig instance.

    Returns:
        Tuple of (interrupt_before, interrupt_after) lists.
    """
    return (config.interrupt_before, config.interrupt_after)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "InterruptPoint",
    "InterruptConfig",
    "HITLDecision",
    "HITLState",
    "HITLManager",
    "create_hitl_config_from_executor_config",
    "get_interrupt_lists",
    "HITL_STATE_FILE",
]
