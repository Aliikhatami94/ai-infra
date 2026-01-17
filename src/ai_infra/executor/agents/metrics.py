"""Orchestrator metrics and observability.

Phase 16.5.13 of EXECUTOR_5.md: Provides full visibility into orchestrator
routing decisions and accuracy tracking.

This module includes:
- OrchestratorMetrics: Aggregate statistics for routing decisions
- RoutingRecord: Individual routing decision record for analysis
- MetricsCollector: Collects and aggregates routing metrics
- Accuracy tracking with misrouting heuristics

Example:
    ```python
    from ai_infra.executor.agents.metrics import (
        MetricsCollector,
        OrchestratorMetrics,
    )

    # Create collector
    collector = MetricsCollector()

    # Record routing decisions
    collector.record_routing(
        task_id=1,
        task_title="Create tests for user.py",
        agent_type=SubAgentType.TESTWRITER,
        confidence=0.92,
        latency_ms=1250.5,
        tokens_used=380,
        used_fallback=False,
    )

    # Get aggregated metrics
    metrics = collector.get_metrics()
    print(collector.format_summary())
    ```
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_infra.executor.agents.registry import SubAgentType
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    pass

__all__ = [
    "OrchestratorMetrics",
    "RoutingRecord",
    "RoutingOutcome",
    "MetricsCollector",
    "format_metrics_summary",
    "check_routing_mismatch",
]

logger = get_logger("executor.agents.metrics")


# =============================================================================
# Constants
# =============================================================================

# Keywords that strongly suggest specific agent types
_TESTWRITER_KEYWORDS = frozenset(
    {
        "test",
        "tests",
        "testing",
        "unittest",
        "pytest",
        "spec",
        "test_",
        "_test",
        "coverage",
        "mock",
        "fixture",
    }
)

_TESTER_KEYWORDS = frozenset(
    {
        "run test",
        "execute test",
        "verify test",
        "check test",
        "run pytest",
        "run tests",
        "execute tests",
    }
)

_DEBUGGER_KEYWORDS = frozenset(
    {
        "fix",
        "debug",
        "bug",
        "error",
        "fail",
        "broken",
        "issue",
        "crash",
        "exception",
        "traceback",
    }
)

_REVIEWER_KEYWORDS = frozenset(
    {
        "review",
        "refactor",
        "optimize",
        "improve",
        "clean",
        "lint",
        "format",
        "style",
    }
)


# =============================================================================
# Data Classes (16.5.13.1.1)
# =============================================================================


@dataclass
class OrchestratorMetrics:
    """Aggregate metrics for orchestrator routing decisions.

    Tracks statistics about routing performance including latency,
    token usage, confidence scores, and agent distribution.

    Attributes:
        total_routings: Total number of routing decisions made.
        routing_latency_ms: List of latencies in milliseconds.
        routing_tokens: List of token counts per routing.
        confidence_scores: List of confidence scores (0.0-1.0).
        fallback_count: Number of times keyword fallback was used.
        agent_distribution: Count of tasks routed to each agent type.
        misroute_count: Number of potential misrouting detected.
        successful_tasks: Number of tasks that completed successfully.
        failed_tasks: Number of tasks that failed after routing.
    """

    total_routings: int = 0
    routing_latency_ms: list[float] = field(default_factory=list)
    routing_tokens: list[int] = field(default_factory=list)
    confidence_scores: list[float] = field(default_factory=list)
    fallback_count: int = 0
    agent_distribution: dict[str, int] = field(default_factory=dict)
    misroute_count: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0

    @property
    def avg_latency_ms(self) -> float:
        """Average routing latency in milliseconds."""
        if not self.routing_latency_ms:
            return 0.0
        return sum(self.routing_latency_ms) / len(self.routing_latency_ms)

    @property
    def min_latency_ms(self) -> float:
        """Minimum routing latency in milliseconds."""
        if not self.routing_latency_ms:
            return 0.0
        return min(self.routing_latency_ms)

    @property
    def max_latency_ms(self) -> float:
        """Maximum routing latency in milliseconds."""
        if not self.routing_latency_ms:
            return 0.0
        return max(self.routing_latency_ms)

    @property
    def avg_tokens(self) -> float:
        """Average tokens used per routing."""
        if not self.routing_tokens:
            return 0.0
        return sum(self.routing_tokens) / len(self.routing_tokens)

    @property
    def avg_confidence(self) -> float:
        """Average confidence score."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)

    @property
    def fallback_rate(self) -> float:
        """Rate of fallback usage (0.0-1.0)."""
        if self.total_routings == 0:
            return 0.0
        return self.fallback_count / self.total_routings

    @property
    def success_rate(self) -> float:
        """Task success rate after routing (0.0-1.0)."""
        total = self.successful_tasks + self.failed_tasks
        if total == 0:
            return 0.0
        return self.successful_tasks / total

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_routings": self.total_routings,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "avg_tokens": round(self.avg_tokens, 1),
            "avg_confidence": round(self.avg_confidence, 3),
            "fallback_count": self.fallback_count,
            "fallback_rate": round(self.fallback_rate, 3),
            "agent_distribution": self.agent_distribution.copy(),
            "misroute_count": self.misroute_count,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": round(self.success_rate, 3),
        }


@dataclass
class RoutingOutcome:
    """Outcome of a routed task for accuracy tracking.

    Attributes:
        success: Whether the task completed successfully.
        error_message: Error message if task failed.
        might_be_misrouted: Heuristic flag for potential misrouting.
        actual_work_done: Description of what the agent actually did.
    """

    success: bool
    error_message: str | None = None
    might_be_misrouted: bool = False
    actual_work_done: str | None = None


@dataclass
class RoutingRecord:
    """Individual routing decision record for analysis.

    Stores complete information about a single routing decision
    for later analysis and accuracy tracking.

    Attributes:
        task_id: Unique identifier for the task.
        task_title: Title of the routed task.
        task_description: Description of the task (if available).
        agent_type: The agent type selected for routing.
        confidence: Confidence score from the orchestrator.
        latency_ms: Time taken for routing decision.
        tokens_used: Number of tokens used for routing.
        used_fallback: Whether keyword fallback was used.
        reasoning: Reasoning provided by the orchestrator.
        timestamp: When the routing decision was made.
        outcome: Outcome of the task (set after completion).
    """

    task_id: int
    task_title: str
    task_description: str | None
    agent_type: SubAgentType
    confidence: float
    latency_ms: float
    tokens_used: int
    used_fallback: bool
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    outcome: RoutingOutcome | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "task_id": self.task_id,
            "task_title": self.task_title,
            "task_description": self.task_description,
            "agent_type": self.agent_type.value,
            "confidence": self.confidence,
            "latency_ms": round(self.latency_ms, 2),
            "tokens_used": self.tokens_used,
            "used_fallback": self.used_fallback,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.outcome:
            result["outcome"] = {
                "success": self.outcome.success,
                "error_message": self.outcome.error_message,
                "might_be_misrouted": self.outcome.might_be_misrouted,
                "actual_work_done": self.outcome.actual_work_done,
            }
        return result


# =============================================================================
# Misrouting Detection (16.5.13.2.1)
# =============================================================================


def check_routing_mismatch(
    task_title: str,
    task_description: str | None,
    agent_used: SubAgentType,
) -> bool:
    """Check if a routing decision might be incorrect.

    Uses heuristics to detect potential misrouting based on
    task content and the agent that was selected.

    Args:
        task_title: Title of the task.
        task_description: Description of the task.
        agent_used: The agent type that was selected.

    Returns:
        True if the routing might be incorrect, False otherwise.

    Examples:
        >>> check_routing_mismatch("Create tests for user.py", None, SubAgentType.CODER)
        True  # Task mentions tests but went to Coder
        >>> check_routing_mismatch("Implement user login", None, SubAgentType.CODER)
        False  # Correctly routed to Coder
    """
    text = f"{task_title} {task_description or ''}".lower()

    # Check for TestWriter keywords routed elsewhere
    if agent_used != SubAgentType.TESTWRITER:
        for keyword in _TESTWRITER_KEYWORDS:
            if keyword in text:
                # Avoid false positives for "run tests" which should go to Tester
                if not any(tk in text for tk in _TESTER_KEYWORDS):
                    return True

    # Check for Tester keywords routed elsewhere
    if agent_used != SubAgentType.TESTER:
        for keyword in _TESTER_KEYWORDS:
            if keyword in text:
                return True

    # Check for Debugger keywords routed elsewhere
    if agent_used != SubAgentType.DEBUGGER:
        for keyword in _DEBUGGER_KEYWORDS:
            if keyword in text:
                return True

    # Check for Reviewer keywords routed elsewhere
    if agent_used != SubAgentType.REVIEWER:
        for keyword in _REVIEWER_KEYWORDS:
            if keyword in text:
                return True

    return False


# =============================================================================
# Metrics Collector (16.5.13.1.2)
# =============================================================================


class MetricsCollector:
    """Collects and aggregates orchestrator routing metrics.

    Provides methods to record individual routing decisions and
    aggregate them into summary statistics.

    Example:
        ```python
        collector = MetricsCollector()

        # Record routing with timing
        with collector.timing_context() as timer:
            decision = await orchestrator.route(task, context)

        collector.record_routing(
            task_id=task.id,
            task_title=task.title,
            agent_type=decision.agent_type,
            confidence=decision.confidence,
            latency_ms=timer.elapsed_ms,
            tokens_used=orchestrator.last_token_usage.get("total_tokens", 0),
            used_fallback=decision.used_fallback,
            reasoning=decision.reasoning,
        )
        ```
    """

    def __init__(self, *, record_routing_feedback: bool = False) -> None:
        """Initialize the metrics collector.

        Args:
            record_routing_feedback: If True, store detailed records
                                     for manual labeling and analysis.
        """
        self._metrics = OrchestratorMetrics()
        self._records: list[RoutingRecord] = []
        self._record_feedback = record_routing_feedback

    @property
    def metrics(self) -> OrchestratorMetrics:
        """Get current aggregated metrics."""
        return self._metrics

    @property
    def records(self) -> list[RoutingRecord]:
        """Get all routing records (if recording enabled)."""
        return self._records.copy()

    def record_routing(
        self,
        *,
        task_id: int,
        task_title: str,
        agent_type: SubAgentType,
        confidence: float,
        latency_ms: float,
        tokens_used: int = 0,
        used_fallback: bool = False,
        reasoning: str = "",
        task_description: str | None = None,
    ) -> None:
        """Record a routing decision.

        Updates aggregate metrics and optionally stores detailed record.

        Args:
            task_id: Unique identifier for the task.
            task_title: Title of the routed task.
            agent_type: The agent type selected.
            confidence: Confidence score (0.0-1.0).
            latency_ms: Routing latency in milliseconds.
            tokens_used: Number of tokens used (optional).
            used_fallback: Whether keyword fallback was used.
            reasoning: Reasoning from the orchestrator.
            task_description: Description of the task (optional).
        """
        # Update aggregate metrics
        self._metrics.total_routings += 1
        self._metrics.routing_latency_ms.append(latency_ms)
        self._metrics.confidence_scores.append(confidence)

        if tokens_used > 0:
            self._metrics.routing_tokens.append(tokens_used)

        if used_fallback:
            self._metrics.fallback_count += 1

        # Update agent distribution
        agent_key = agent_type.value
        self._metrics.agent_distribution[agent_key] = (
            self._metrics.agent_distribution.get(agent_key, 0) + 1
        )

        # Log structured decision (16.5.13.1.4)
        logger.info(
            "Routing decision recorded",
            extra={
                "routing_task_id": task_id,
                "routing_task_title": task_title[:50],
                "routing_agent": agent_type.value,
                "routing_confidence": round(confidence, 3),
                "routing_latency_ms": round(latency_ms, 2),
                "routing_tokens": tokens_used,
                "routing_fallback": used_fallback,
            },
        )

        # Store detailed record if feedback recording enabled
        if self._record_feedback:
            record = RoutingRecord(
                task_id=task_id,
                task_title=task_title,
                task_description=task_description,
                agent_type=agent_type,
                confidence=confidence,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                used_fallback=used_fallback,
                reasoning=reasoning,
            )
            self._records.append(record)

    def record_outcome(
        self,
        task_id: int,
        *,
        success: bool,
        error_message: str | None = None,
        actual_work_done: str | None = None,
    ) -> None:
        """Record the outcome of a routed task.

        Updates success/failure counts and checks for potential misrouting.

        Args:
            task_id: ID of the task that completed.
            success: Whether the task completed successfully.
            error_message: Error message if task failed.
            actual_work_done: Description of what the agent did.
        """
        if success:
            self._metrics.successful_tasks += 1
        else:
            self._metrics.failed_tasks += 1

        # Update detailed record if available
        for record in self._records:
            if record.task_id == task_id:
                # Check for misrouting heuristic
                might_be_misrouted = False
                if not success:
                    might_be_misrouted = check_routing_mismatch(
                        record.task_title,
                        record.task_description,
                        record.agent_type,
                    )
                    if might_be_misrouted:
                        self._metrics.misroute_count += 1
                        logger.warning(
                            f"Possible misroute detected: '{record.task_title[:40]}' "
                            f"-> {record.agent_type.value}",
                            extra={
                                "routing_task_id": task_id,
                                "routing_agent": record.agent_type.value,
                                "routing_possible_misroute": True,
                            },
                        )

                record.outcome = RoutingOutcome(
                    success=success,
                    error_message=error_message,
                    might_be_misrouted=might_be_misrouted,
                    actual_work_done=actual_work_done,
                )
                break

    def get_metrics(self) -> OrchestratorMetrics:
        """Get current aggregated metrics.

        Returns:
            Copy of current OrchestratorMetrics.
        """
        return OrchestratorMetrics(
            total_routings=self._metrics.total_routings,
            routing_latency_ms=self._metrics.routing_latency_ms.copy(),
            routing_tokens=self._metrics.routing_tokens.copy(),
            confidence_scores=self._metrics.confidence_scores.copy(),
            fallback_count=self._metrics.fallback_count,
            agent_distribution=self._metrics.agent_distribution.copy(),
            misroute_count=self._metrics.misroute_count,
            successful_tasks=self._metrics.successful_tasks,
            failed_tasks=self._metrics.failed_tasks,
        )

    def format_summary(self) -> str:
        """Format metrics as a visual summary.

        Returns:
            Formatted string suitable for terminal display.
        """
        return format_metrics_summary(self._metrics)

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        self._metrics = OrchestratorMetrics()
        self._records.clear()

    def export_records(self, path: Path) -> None:
        """Export routing records to a JSON file for analysis.

        Args:
            path: Path to write the JSON file.
        """
        import json

        data = {
            "metrics": self._metrics.to_dict(),
            "records": [r.to_dict() for r in self._records],
            "exported_at": datetime.now().isoformat(),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

        logger.info(f"Exported {len(self._records)} routing records to {path}")


# =============================================================================
# Timing Context Manager
# =============================================================================


class RoutingTimer:
    """Context manager for timing routing operations.

    Example:
        ```python
        timer = RoutingTimer()
        with timer:
            decision = await orchestrator.route(task, context)
        print(f"Routing took {timer.elapsed_ms}ms")
        ```
    """

    def __init__(self) -> None:
        """Initialize the timer."""
        self._start: float = 0.0
        self._end: float = 0.0

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return (self._end - self._start) * 1000

    def __enter__(self) -> RoutingTimer:
        """Start timing."""
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop timing."""
        self._end = time.perf_counter()


# =============================================================================
# Summary Formatting (16.5.13.1.3)
# =============================================================================


def format_metrics_summary(metrics: OrchestratorMetrics) -> str:
    """Format orchestrator metrics as a visual summary.

    Creates a box-formatted summary suitable for terminal display.

    Args:
        metrics: The metrics to format.

    Returns:
        Formatted string with metrics summary.

    Example output:
        ```
        ╭─────────── Orchestrator Routing Summary ───────────╮
        │  Routings:    8 total (0 fallbacks)               │
        │  Latency:     1.2s avg (0.8s - 1.8s)              │
        │  Tokens:      380 avg per routing                 │
        │  Confidence:  0.92 avg                            │
        │  Distribution:                                     │
        │    - Coder: 5 tasks                               │
        │    - TestWriter: 2 tasks                          │
        │    - Tester: 1 task                               │
        ╰────────────────────────────────────────────────────╯
        ```
    """
    if metrics.total_routings == 0:
        return "No routing decisions recorded."

    # Build distribution lines
    distribution_lines = []
    for agent, count in sorted(
        metrics.agent_distribution.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        task_word = "task" if count == 1 else "tasks"
        distribution_lines.append(f"    - {agent.title()}: {count} {task_word}")

    # Format latency
    avg_latency_s = metrics.avg_latency_ms / 1000
    min_latency_s = metrics.min_latency_ms / 1000
    max_latency_s = metrics.max_latency_ms / 1000

    # Build content lines
    lines = [
        f"  Routings:    {metrics.total_routings} total ({metrics.fallback_count} fallbacks)",
        f"  Latency:     {avg_latency_s:.1f}s avg ({min_latency_s:.1f}s - {max_latency_s:.1f}s)",
    ]

    if metrics.routing_tokens:
        lines.append(f"  Tokens:      {metrics.avg_tokens:.0f} avg per routing")

    lines.append(f"  Confidence:  {metrics.avg_confidence:.2f} avg")

    # Add success rate if outcomes recorded
    if metrics.successful_tasks + metrics.failed_tasks > 0:
        lines.append(
            f"  Success:     {metrics.success_rate * 100:.0f}% "
            f"({metrics.successful_tasks}/{metrics.successful_tasks + metrics.failed_tasks})"
        )

    if metrics.misroute_count > 0:
        lines.append(f"  Misroutes:   {metrics.misroute_count} detected")

    lines.append("  Distribution:")
    lines.extend(distribution_lines)

    # Calculate box width
    content_width = max(len(line) for line in lines) + 2
    box_width = max(content_width, 50)

    # Build box
    title = " Orchestrator Routing Summary "
    title_padding = (box_width - len(title)) // 2

    result = []
    result.append(f"╭{'─' * title_padding}{title}{'─' * (box_width - title_padding - len(title))}╮")

    for line in lines:
        padding = box_width - len(line)
        result.append(f"│{line}{' ' * padding}│")

    result.append(f"╰{'─' * box_width}╯")

    return "\n".join(result)
