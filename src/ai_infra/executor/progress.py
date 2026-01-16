"""Progress tracking for executor roadmap execution.

Phase 4.3.1 of EXECUTOR_1.md: Progress Visibility.

This module provides:
- TaskProgress: Individual task progress with metrics
- ProgressTracker: Tracks progress through roadmap execution
- CostEstimator: Estimates tokens and costs by task complexity

Example:
    ```python
    from ai_infra.executor.progress import ProgressTracker, CostEstimator

    # Create tracker for a roadmap
    tracker = ProgressTracker(roadmap)

    # Track task execution
    tracker.start_task("1.1.1", agent="CoderAgent", model="claude-sonnet-4-20250514")
    # ... task executes ...
    tracker.complete_task("1.1.1", tokens_in=1000, tokens_out=500, cost=0.02)

    # Get summary
    summary = tracker.get_summary()
    print(f"Progress: {summary['percent']:.0f}%")
    print(f"Cost: ${summary['cost']:.2f}")
    ```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_infra.executor.models import Task
    from ai_infra.executor.parser import Roadmap

logger = logging.getLogger(__name__)


# =============================================================================
# Task Progress (Phase 4.3.1)
# =============================================================================


@dataclass
class TaskProgress:
    """Progress tracking for an individual task.

    Attributes:
        task_id: Unique task identifier.
        task_title: Human-readable task title.
        status: Current status (pending, in_progress, completed, failed, skipped).
        started_at: When the task started execution.
        completed_at: When the task completed.
        duration: Time taken to complete (computed from started/completed).
        tokens_in: Input tokens used.
        tokens_out: Output tokens generated.
        cost: Total cost for this task.
        agent_used: Name of the agent that executed the task.
        model_used: Model name used for execution.
        error: Error message if task failed.
        files_modified: List of files modified during task.
        files_created: List of files created during task.
        subagent_tokens: Tokens used by subagent execution (Phase 16.5.1).

    Example:
        ```python
        progress = TaskProgress(
            task_id="1.1.1",
            task_title="Add authentication",
            status="in_progress",
            started_at=datetime.now(),
            agent_used="CoderAgent",
            model_used="claude-sonnet-4-20250514",
        )
        ```
    """

    task_id: str
    task_title: str
    status: str = "pending"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration: timedelta | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    agent_used: str | None = None
    model_used: str | None = None
    error: str | None = None
    files_modified: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    subagent_tokens: int = 0  # Phase 16.5.1: Token tracking for subagents

    @property
    def total_tokens(self) -> int:
        """Get total tokens (in + out + subagent)."""
        return self.tokens_in + self.tokens_out + self.subagent_tokens

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        if self.duration:
            return self.duration.total_seconds()
        elif self.started_at and self.status == "in_progress":
            return (datetime.now() - self.started_at).total_seconds()
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "task_title": self.task_title,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "subagent_tokens": self.subagent_tokens,  # Phase 16.5.1
            "cost": self.cost,
            "agent_used": self.agent_used,
            "model_used": self.model_used,
            "error": self.error,
            "files_modified": self.files_modified,
            "files_created": self.files_created,
        }


# =============================================================================
# Progress Tracker (Phase 4.3.1)
# =============================================================================


@dataclass
class ProgressSummary:
    """Summary of progress through roadmap execution.

    Attributes:
        total: Total number of tasks.
        completed: Number of completed tasks.
        in_progress: Number of tasks currently in progress.
        pending: Number of pending tasks.
        failed: Number of failed tasks.
        skipped: Number of skipped tasks.
        percent: Percentage complete.
        tokens_in: Total input tokens used.
        tokens_out: Total output tokens generated.
        subagent_tokens: Total tokens used by subagents (Phase 16.5.1).
        cost: Total cost so far.
        elapsed: Time elapsed since start.
        estimated_remaining_time: Estimated time remaining.
        estimated_remaining_cost: Estimated cost remaining.
        current_task_id: ID of current task (if any).
        current_task_title: Title of current task.
    """

    total: int
    completed: int
    in_progress: int
    pending: int
    failed: int
    skipped: int
    percent: float
    tokens_in: int
    tokens_out: int
    subagent_tokens: int = 0  # Phase 16.5.1
    cost: float = 0.0
    elapsed: timedelta | None = None
    estimated_remaining_time: timedelta | None = None
    estimated_remaining_cost: float | None = None
    current_task_id: str | None = None
    current_task_title: str | None = None

    @property
    def total_tokens(self) -> int:
        """Get total tokens (in + out + subagent)."""
        return self.tokens_in + self.tokens_out + self.subagent_tokens

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total": self.total,
            "completed": self.completed,
            "in_progress": self.in_progress,
            "pending": self.pending,
            "failed": self.failed,
            "skipped": self.skipped,
            "percent": self.percent,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "subagent_tokens": self.subagent_tokens,  # Phase 16.5.1
            "total_tokens": self.total_tokens,  # Phase 16.5.1: convenience field
            "cost": self.cost,
            "elapsed_seconds": self.elapsed.total_seconds() if self.elapsed else None,
            "estimated_remaining_seconds": (
                self.estimated_remaining_time.total_seconds()
                if self.estimated_remaining_time
                else None
            ),
            "estimated_remaining_cost": self.estimated_remaining_cost,
            "current_task_id": self.current_task_id,
            "current_task_title": self.current_task_title,
        }


class ProgressTracker:
    """Track progress through roadmap execution.

    Phase 4.3.1: Provides real-time progress tracking for executor runs.

    Attributes:
        roadmap_title: Title of the roadmap being executed.
        tasks: Dictionary mapping task IDs to TaskProgress.
        started_at: When execution started.

    Example:
        ```python
        tracker = ProgressTracker(roadmap)

        # Start tracking
        tracker.start_run()

        # Track individual tasks
        tracker.start_task("1.1.1", agent="CoderAgent", model="claude-sonnet-4-20250514")
        tracker.complete_task("1.1.1", tokens_in=1000, tokens_out=500, cost=0.02)

        # Get summary
        summary = tracker.get_summary()
        print(f"Completed: {summary.completed}/{summary.total}")
        ```
    """

    def __init__(
        self,
        roadmap: Roadmap | None = None,
        roadmap_title: str = "Roadmap",
    ) -> None:
        """Initialize the progress tracker.

        Args:
            roadmap: Optional Roadmap instance to track.
            roadmap_title: Title for display (used if no roadmap).
        """
        self.roadmap_title = roadmap_title
        self.tasks: dict[str, TaskProgress] = {}
        self.started_at: datetime | None = None

        if roadmap:
            self.roadmap_title = getattr(roadmap, "title", roadmap_title)
            self._initialize_from_roadmap(roadmap)

    def _initialize_from_roadmap(self, roadmap: Roadmap) -> None:
        """Initialize task tracking from roadmap.

        Args:
            roadmap: Roadmap instance with tasks.
        """
        for task in roadmap.all_tasks():
            self.tasks[task.id] = TaskProgress(
                task_id=task.id,
                task_title=task.title,
                status="pending",
            )

    def add_task(self, task_id: str, task_title: str) -> None:
        """Add a task to track.

        Args:
            task_id: Unique task identifier.
            task_title: Human-readable title.
        """
        if task_id not in self.tasks:
            self.tasks[task_id] = TaskProgress(
                task_id=task_id,
                task_title=task_title,
                status="pending",
            )

    def start_run(self) -> None:
        """Mark the start of the execution run."""
        self.started_at = datetime.now()
        logger.info("Progress tracking started", extra={"roadmap": self.roadmap_title})

    def start_task(
        self,
        task_id: str,
        agent: str | None = None,
        model: str | None = None,
    ) -> None:
        """Mark a task as started.

        Args:
            task_id: ID of the task to start.
            agent: Name of the agent executing the task.
            model: Model being used.
        """
        if task_id not in self.tasks:
            self.tasks[task_id] = TaskProgress(
                task_id=task_id,
                task_title=f"Task {task_id}",
            )

        progress = self.tasks[task_id]
        progress.status = "in_progress"
        progress.started_at = datetime.now()
        progress.agent_used = agent
        progress.model_used = model

        logger.debug(
            "Task started",
            extra={"task_id": task_id, "agent": agent, "model": model},
        )

    def complete_task(
        self,
        task_id: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost: float = 0.0,
        files_modified: list[str] | None = None,
        files_created: list[str] | None = None,
    ) -> None:
        """Mark a task as completed.

        Args:
            task_id: ID of the task to complete.
            tokens_in: Input tokens used.
            tokens_out: Output tokens generated.
            cost: Cost of the task execution.
            files_modified: Files that were modified.
            files_created: Files that were created.
        """
        if task_id not in self.tasks:
            logger.warning(f"Task {task_id} not found in tracker")
            return

        progress = self.tasks[task_id]
        progress.status = "completed"
        progress.completed_at = datetime.now()

        if progress.started_at:
            progress.duration = progress.completed_at - progress.started_at

        progress.tokens_in = tokens_in
        progress.tokens_out = tokens_out
        progress.cost = cost
        progress.files_modified = files_modified or []
        progress.files_created = files_created or []

        logger.debug(
            "Task completed",
            extra={
                "task_id": task_id,
                "duration_seconds": progress.duration_seconds,
                "tokens": tokens_in + tokens_out,
                "cost": cost,
            },
        )

    def fail_task(self, task_id: str, error: str) -> None:
        """Mark a task as failed.

        Args:
            task_id: ID of the task that failed.
            error: Error message.
        """
        if task_id not in self.tasks:
            return

        progress = self.tasks[task_id]
        progress.status = "failed"
        progress.completed_at = datetime.now()
        progress.error = error

        if progress.started_at:
            progress.duration = progress.completed_at - progress.started_at

        logger.debug("Task failed", extra={"task_id": task_id, "error": error})

    def skip_task(self, task_id: str, reason: str = "") -> None:
        """Mark a task as skipped.

        Args:
            task_id: ID of the task to skip.
            reason: Reason for skipping.
        """
        if task_id not in self.tasks:
            return

        progress = self.tasks[task_id]
        progress.status = "skipped"
        progress.error = reason if reason else None

        logger.debug("Task skipped", extra={"task_id": task_id, "reason": reason})

    def get_current_task(self) -> TaskProgress | None:
        """Get the currently executing task.

        Returns:
            TaskProgress for the in-progress task, or None.
        """
        for progress in self.tasks.values():
            if progress.status == "in_progress":
                return progress
        return None

    def get_elapsed(self) -> timedelta | None:
        """Get elapsed time since start.

        Returns:
            Timedelta since start, or None if not started.
        """
        if self.started_at:
            return datetime.now() - self.started_at
        return None

    def get_summary(self) -> ProgressSummary:
        """Get progress summary with statistics and estimates.

        Returns:
            ProgressSummary with current state and estimates.
        """
        total = len(self.tasks)

        # Count by status
        completed = sum(1 for t in self.tasks.values() if t.status == "completed")
        in_progress = sum(1 for t in self.tasks.values() if t.status == "in_progress")
        failed = sum(1 for t in self.tasks.values() if t.status == "failed")
        skipped = sum(1 for t in self.tasks.values() if t.status == "skipped")
        pending = total - completed - in_progress - failed - skipped

        # Token and cost totals
        total_tokens_in = sum(t.tokens_in for t in self.tasks.values())
        total_tokens_out = sum(t.tokens_out for t in self.tasks.values())
        total_subagent_tokens = sum(t.subagent_tokens for t in self.tasks.values())
        total_cost = sum(t.cost for t in self.tasks.values())

        # Calculate percent
        percent = (completed / total) * 100 if total > 0 else 0

        # Elapsed time
        elapsed = self.get_elapsed()

        # Estimate remaining time and cost
        estimated_remaining_time = None
        estimated_remaining_cost = None

        if completed > 0:
            # Calculate average duration
            completed_durations = [
                t.duration.total_seconds()
                for t in self.tasks.values()
                if t.duration and t.status == "completed"
            ]
            if completed_durations:
                avg_duration = sum(completed_durations) / len(completed_durations)
                remaining_tasks = pending + in_progress
                estimated_remaining_time = timedelta(seconds=remaining_tasks * avg_duration)

            # Calculate average cost
            avg_cost = total_cost / completed
            remaining_tasks = pending + in_progress
            estimated_remaining_cost = remaining_tasks * avg_cost

        # Get current task info
        current = self.get_current_task()

        return ProgressSummary(
            total=total,
            completed=completed,
            in_progress=in_progress,
            pending=pending,
            failed=failed,
            skipped=skipped,
            percent=percent,
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
            subagent_tokens=total_subagent_tokens,  # Phase 16.5.1
            cost=total_cost,
            elapsed=elapsed,
            estimated_remaining_time=estimated_remaining_time,
            estimated_remaining_cost=estimated_remaining_cost,
            current_task_id=current.task_id if current else None,
            current_task_title=current.task_title if current else None,
        )

    def get_task_progress(self, task_id: str) -> TaskProgress | None:
        """Get progress for a specific task.

        Args:
            task_id: ID of the task.

        Returns:
            TaskProgress or None if not found.
        """
        return self.tasks.get(task_id)

    def get_all_progress(self) -> list[TaskProgress]:
        """Get progress for all tasks in order.

        Returns:
            List of TaskProgress for all tasks.
        """
        return list(self.tasks.values())


# =============================================================================
# Cost Estimator (Phase 4.3.4)
# =============================================================================


class CostEstimator:
    """Estimate tokens and costs based on task complexity and model.

    Phase 4.3.4: Provides cost estimation for roadmap planning.

    Example:
        ```python
        estimator = CostEstimator()

        # Estimate for a task
        estimate = estimator.estimate_task(task, model="claude-sonnet-4-20250514")
        print(f"Estimated cost: ${estimate['estimated_cost']:.2f}")
        print(f"Complexity: {estimate['complexity']}")

        # Estimate for entire roadmap
        total = estimator.estimate_roadmap(roadmap, model="claude-sonnet-4-20250514")
        print(f"Total estimated: ${total['total_cost']:.2f}")
        ```
    """

    # Tokens per task by complexity
    TOKENS_BY_COMPLEXITY: dict[str, dict[str, int]] = {
        "low": {"in": 500, "out": 200},
        "medium": {"in": 2000, "out": 800},
        "high": {"in": 5000, "out": 2000},
        "very_high": {"in": 10000, "out": 4000},
    }

    # Cost per 1K tokens by model (input, output)
    COST_PER_1K: dict[str, dict[str, float]] = {
        # OpenAI models
        "gpt-4.1-nano": {"in": 0.0001, "out": 0.0004},
        "gpt-4.1-mini": {"in": 0.0004, "out": 0.0016},
        "gpt-4.1": {"in": 0.002, "out": 0.008},
        "gpt-4o": {"in": 0.005, "out": 0.015},
        "gpt-4o-mini": {"in": 0.00015, "out": 0.0006},
        # Anthropic models
        "claude-3-5-haiku-20241022": {"in": 0.001, "out": 0.005},
        "claude-sonnet-4-20250514": {"in": 0.003, "out": 0.015},
        "claude-opus-4-20250514": {"in": 0.015, "out": 0.075},
        # Default fallback
        "default": {"in": 0.003, "out": 0.015},
    }

    # Complexity indicators
    COMPLEXITY_KEYWORDS: dict[str, list[str]] = {
        "high": [
            "refactor",
            "architecture",
            "migration",
            "integration",
            "security",
            "authentication",
            "authorization",
            "database",
            "performance",
            "optimization",
        ],
        "medium": [
            "implement",
            "create",
            "add",
            "feature",
            "endpoint",
            "handler",
            "service",
            "test",
            "tests",
        ],
        "low": [
            "fix",
            "update",
            "rename",
            "document",
            "comment",
            "format",
            "lint",
            "style",
            "typo",
        ],
    }

    def estimate_task(
        self,
        task: Task,
        model: str = "claude-sonnet-4-20250514",
    ) -> dict[str, Any]:
        """Estimate tokens and cost for a single task.

        Args:
            task: The task to estimate.
            model: Model to use for cost calculation.

        Returns:
            Dictionary with tokens_in, tokens_out, estimated_cost, complexity.
        """
        complexity = self._assess_complexity(task)
        tokens = self.TOKENS_BY_COMPLEXITY[complexity]
        costs = self.COST_PER_1K.get(model, self.COST_PER_1K["default"])

        cost = (tokens["in"] * costs["in"] + tokens["out"] * costs["out"]) / 1000

        # Adjust for file hints (more files = more tokens)
        file_count = len(getattr(task, "file_hints", []))
        if file_count > 3:
            multiplier = 1 + (file_count - 3) * 0.1
            cost *= multiplier
            tokens = {
                "in": int(tokens["in"] * multiplier),
                "out": int(tokens["out"] * multiplier),
            }

        return {
            "tokens_in": tokens["in"],
            "tokens_out": tokens["out"],
            "estimated_cost": cost,
            "complexity": complexity,
        }

    def estimate_roadmap(
        self,
        roadmap: Roadmap,
        model: str = "claude-sonnet-4-20250514",
    ) -> dict[str, Any]:
        """Estimate total tokens and cost for a roadmap.

        Args:
            roadmap: The roadmap to estimate.
            model: Model to use for cost calculation.

        Returns:
            Dictionary with per-task estimates and totals.
        """
        estimates = {}
        total_tokens_in = 0
        total_tokens_out = 0
        total_cost = 0.0

        for task in roadmap.all_tasks():
            estimate = self.estimate_task(task, model)
            estimates[task.id] = estimate
            total_tokens_in += estimate["tokens_in"]
            total_tokens_out += estimate["tokens_out"]
            total_cost += estimate["estimated_cost"]

        return {
            "tasks": estimates,
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "total_tokens": total_tokens_in + total_tokens_out,
            "total_cost": total_cost,
            "task_count": len(estimates),
        }

    def _assess_complexity(self, task: Task) -> str:
        """Assess task complexity from title and description.

        Args:
            task: The task to assess.

        Returns:
            Complexity level: low, medium, high, or very_high.
        """
        text = f"{task.title} {task.description}".lower()

        # Check for high complexity keywords
        for keyword in self.COMPLEXITY_KEYWORDS["high"]:
            if keyword in text:
                # Check if it's a very complex task
                if any(word in text for word in ["full", "complete", "entire", "all"]):
                    return "very_high"
                return "high"

        # Check for medium complexity keywords
        for keyword in self.COMPLEXITY_KEYWORDS["medium"]:
            if keyword in text:
                return "medium"

        # Check for low complexity keywords
        for keyword in self.COMPLEXITY_KEYWORDS["low"]:
            if keyword in text:
                return "low"

        # Default to medium
        return "medium"

    def get_model_pricing(self, model: str) -> dict[str, float]:
        """Get pricing for a specific model.

        Args:
            model: Model name.

        Returns:
            Dictionary with 'in' and 'out' costs per 1K tokens.
        """
        return self.COST_PER_1K.get(model, self.COST_PER_1K["default"])


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "CostEstimator",
    "ProgressSummary",
    "ProgressTracker",
    "TaskProgress",
]
