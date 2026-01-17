"""Per-node metrics tracking for executor graph.

Phase 2.4.3: Track tokens, duration, and LLM calls per graph node.
Phase 10.1.2: Add TokenMetrics for accurate cost calculation.

This module provides:
- NodeMetrics dataclass for per-node statistics
- TokenMetrics dataclass for input/output/cached token tracking with cost calculation
- track_node_metrics decorator for instrumented nodes
- Helper functions for aggregating and formatting node metrics

Note: TokenTracker has been moved to tracing.py and is re-exported here
for backwards compatibility.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

# Re-export TokenTracker from tracing for backwards compatibility
from ai_infra.executor.tracing import TokenTracker
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.state import ExecutorGraphState

logger = get_logger("executor.metrics")


# =============================================================================
# TokenMetrics for Cost Calculation (Phase 10.1.2)
# =============================================================================


@dataclass
class TokenMetrics:
    """Token usage metrics with cost calculation.

    Phase 10.1.2: Tracks input, output, and cached tokens separately
    to enable accurate cost calculation based on model-specific pricing.

    Attributes:
        input_tokens: Number of input tokens (non-cached).
        output_tokens: Number of output tokens.
        cached_tokens: Number of cached input tokens (prompt caching).
        model: Model name used for cost calculation.

    Example:
        ```python
        from ai_infra.executor.metrics import TokenMetrics

        # Track token usage
        metrics = TokenMetrics(
            input_tokens=1500,
            output_tokens=500,
            cached_tokens=1000,
            model="claude-sonnet-4-20250514",
        )

        # Calculate cost
        cost = metrics.calculate_cost()
        print(f"Total cost: ${cost:.6f}")

        # Or specify a different model
        cost_opus = metrics.calculate_cost(model="claude-opus-4-20250514")
        ```
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    model: str = ""

    @property
    def total(self) -> int:
        """Total tokens (input + output + cached)."""
        return self.input_tokens + self.output_tokens + self.cached_tokens

    @property
    def total_input(self) -> int:
        """Total input tokens (regular + cached)."""
        return self.input_tokens + self.cached_tokens

    def calculate_cost(self, model: str | None = None) -> Decimal:
        """Calculate cost based on model pricing.

        Uses the model-specific pricing from the pricing module to
        calculate the actual cost of the token usage.

        Args:
            model: Model name to use for pricing. If None, uses self.model.
                Falls back to conservative default if model is unknown.

        Returns:
            Total cost in USD as Decimal.

        Example:
            ```python
            metrics = TokenMetrics(input_tokens=1000, output_tokens=500)
            cost = metrics.calculate_cost(model="gpt-4o-mini")
            # Returns Decimal for precise financial calculations
            ```
        """
        from ai_infra.executor.pricing import get_pricing

        target_model = model if model is not None else self.model
        if not target_model:
            target_model = "claude-sonnet-4-20250514"  # Default model

        pricing = get_pricing(target_model)

        return pricing.calculate_cost(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_tokens=self.cached_tokens,
        )

    def calculate_cost_float(self, model: str | None = None) -> float:
        """Calculate cost as a float for display.

        Convenience method that returns float instead of Decimal.
        Use calculate_cost() for precise financial calculations.

        Args:
            model: Model name to use for pricing.

        Returns:
            Total cost in USD as float.
        """
        return float(self.calculate_cost(model))

    def merge(self, other: TokenMetrics) -> TokenMetrics:
        """Merge with another TokenMetrics (additive).

        Args:
            other: Another TokenMetrics instance to merge with.

        Returns:
            New TokenMetrics with summed values.
        """
        return TokenMetrics(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            model=self.model or other.model,  # Keep first non-empty model
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "total": self.total,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenMetrics:
        """Create from dictionary."""
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cached_tokens=data.get("cached_tokens", 0),
            model=data.get("model", ""),
        )

    def format_summary(self, include_cost: bool = True) -> str:
        """Format token usage for display.

        Args:
            include_cost: Whether to include cost calculation.

        Returns:
            Formatted string like "1,500 in / 500 out / 1,000 cached = $0.0123"
        """
        parts = [
            f"{self.input_tokens:,} in",
            f"{self.output_tokens:,} out",
        ]
        if self.cached_tokens > 0:
            parts.append(f"{self.cached_tokens:,} cached")

        summary = " / ".join(parts)

        if include_cost and self.model:
            cost = self.calculate_cost_float()
            summary += f" = ${cost:.4f}"

        return summary


# =============================================================================
# NodeMetrics Data Model
# =============================================================================


@dataclass
class NodeMetrics:
    """Metrics for a single graph node execution.

    Phase 2.4.3: Tracks tokens, duration, and LLM calls per node.

    Attributes:
        tokens_in: Input tokens consumed by this node.
        tokens_out: Output tokens generated by this node.
        duration_ms: Execution duration in milliseconds.
        llm_calls: Number of LLM calls made by this node.
        invocations: Number of times this node was invoked.
    """

    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: int = 0
    llm_calls: int = 0
    invocations: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.tokens_in + self.tokens_out

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for state storage."""
        return {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "duration_ms": self.duration_ms,
            "llm_calls": self.llm_calls,
            "invocations": self.invocations,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeMetrics:
        """Create from dictionary."""
        return cls(
            tokens_in=data.get("tokens_in", 0),
            tokens_out=data.get("tokens_out", 0),
            duration_ms=data.get("duration_ms", 0),
            llm_calls=data.get("llm_calls", 0),
            invocations=data.get("invocations", 0),
        )

    def merge(self, other: NodeMetrics) -> NodeMetrics:
        """Merge with another NodeMetrics (additive)."""
        return NodeMetrics(
            tokens_in=self.tokens_in + other.tokens_in,
            tokens_out=self.tokens_out + other.tokens_out,
            duration_ms=self.duration_ms + other.duration_ms,
            llm_calls=self.llm_calls + other.llm_calls,
            invocations=self.invocations + other.invocations,
        )


# =============================================================================
# Metrics Aggregation
# =============================================================================


@dataclass
class AggregatedNodeMetrics:
    """Aggregated metrics across all nodes.

    Provides summary statistics and per-node breakdown.
    """

    node_metrics: dict[str, NodeMetrics] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens across all nodes."""
        return sum(m.total_tokens for m in self.node_metrics.values())

    @property
    def total_tokens_in(self) -> int:
        """Total input tokens across all nodes."""
        return sum(m.tokens_in for m in self.node_metrics.values())

    @property
    def total_tokens_out(self) -> int:
        """Total output tokens across all nodes."""
        return sum(m.tokens_out for m in self.node_metrics.values())

    @property
    def total_duration_ms(self) -> int:
        """Total duration across all nodes."""
        return sum(m.duration_ms for m in self.node_metrics.values())

    @property
    def total_llm_calls(self) -> int:
        """Total LLM calls across all nodes."""
        return sum(m.llm_calls for m in self.node_metrics.values())

    def get_node_percentage(self, node_name: str) -> float:
        """Get percentage of total tokens used by a node."""
        if self.total_tokens == 0:
            return 0.0
        node = self.node_metrics.get(node_name)
        if not node:
            return 0.0
        return (node.total_tokens / self.total_tokens) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "nodes": {name: m.to_dict() for name, m in self.node_metrics.items()},
            "totals": {
                "tokens_in": self.total_tokens_in,
                "tokens_out": self.total_tokens_out,
                "total_tokens": self.total_tokens,
                "duration_ms": self.total_duration_ms,
                "llm_calls": self.total_llm_calls,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AggregatedNodeMetrics:
        """Create from dictionary."""
        nodes = data.get("nodes", {})
        return cls(node_metrics={name: NodeMetrics.from_dict(m) for name, m in nodes.items()})

    def format_breakdown(self) -> str:
        """Format per-node breakdown for display.

        Returns human-readable breakdown like:
        ```
        Per-Node Breakdown:
            build_context: 12,340 tokens (10%), 2,100ms
            execute_task:  98,200 tokens (78%), 38,000ms  <- Optimization target
            verify_task:   14,890 tokens (12%), 5,130ms
            checkpoint:    0 tokens, 1,000ms
        ```
        """
        if not self.node_metrics:
            return "No node metrics collected."

        lines = ["Per-Node Breakdown:"]

        # Sort by total tokens descending
        sorted_nodes = sorted(
            self.node_metrics.items(),
            key=lambda x: x[1].total_tokens,
            reverse=True,
        )

        # Find max node name length for alignment
        max_name_len = max(len(name) for name, _ in sorted_nodes)

        # Find highest token consumer for marker
        highest_tokens = max(m.total_tokens for _, m in sorted_nodes) if sorted_nodes else 0

        for name, metrics in sorted_nodes:
            pct = self.get_node_percentage(name)
            tokens_str = f"{metrics.total_tokens:,}"
            duration_str = f"{metrics.duration_ms:,}ms"

            # Add marker for highest consumer
            marker = ""
            if metrics.total_tokens == highest_tokens and metrics.total_tokens > 0:
                marker = "  <- Optimization target"

            line = (
                f"    {name:<{max_name_len}}: "
                f"{tokens_str:>10} tokens ({pct:>5.1f}%), "
                f"{duration_str:>10}"
                f"{marker}"
            )
            lines.append(line)

        return "\n".join(lines)


# =============================================================================
# Decorator for Node Metrics
# =============================================================================


F = TypeVar("F", bound=Callable[..., Any])


def track_node_metrics(node_name: str) -> Callable[[F], F]:
    """Decorator to track metrics for a graph node.

    Phase 2.4.3: Wraps node functions to track execution time and token usage.

    The decorator:
    1. Records start time before node execution
    2. Resets token tracker
    3. Executes the node
    4. Calculates duration and retrieves token usage
    5. Updates node_metrics in state

    Usage:
        @track_node_metrics("execute_task")
        async def execute_task_node(state, *, agent):
            ...

    Args:
        node_name: Name of the node for metrics collection.

    Returns:
        Decorated function that tracks metrics.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(
            state: ExecutorGraphState,
            **kwargs: Any,
        ) -> ExecutorGraphState:
            # Skip tracking if node_metrics not enabled
            if not state.get("enable_node_metrics", True):
                return await func(state, **kwargs)

            # Record start time
            start_time = time.perf_counter()

            # Reset token tracker for this node
            TokenTracker.reset()

            # Execute the node
            result = await func(state, **kwargs)

            # Calculate duration
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            # Get token usage from TokenTracker (direct LLM calls)
            tokens_in, tokens_out, llm_calls = TokenTracker.get_usage()

            # Phase 16.5.1: Also include subagent tokens if present (execute_task only)
            # Subagent tokens are tracked separately and returned in state.
            # Only count them for the execute_task node to avoid double-counting.
            if node_name == "execute_task":
                subagent_tokens = result.get("subagent_tokens_task", 0)
                if subagent_tokens > 0:
                    # Add subagent tokens to tokens_out (they're total from subagent)
                    tokens_out += subagent_tokens
                    # Count subagent execution as 1 LLM call for metrics purposes
                    llm_calls += 1

            # Create metrics for this invocation
            new_metrics = NodeMetrics(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                duration_ms=duration_ms,
                llm_calls=llm_calls,
                invocations=1,
            )

            # Get existing node_metrics from result or state
            existing_metrics: dict[str, Any] = result.get(
                "node_metrics", state.get("node_metrics", {})
            )

            # Merge with existing metrics for this node
            if node_name in existing_metrics:
                existing = NodeMetrics.from_dict(existing_metrics[node_name])
                merged = existing.merge(new_metrics)
                existing_metrics[node_name] = merged.to_dict()
            else:
                existing_metrics[node_name] = new_metrics.to_dict()

            logger.debug(
                f"Node [{node_name}] completed: {duration_ms}ms, {tokens_in + tokens_out} tokens"
            )

            # Return result with updated metrics
            return {
                **result,
                "node_metrics": existing_metrics,
            }

        return wrapper  # type: ignore[return-value]

    return decorator


# =============================================================================
# Helper Functions
# =============================================================================


def aggregate_node_metrics(
    node_metrics: dict[str, Any] | None,
) -> AggregatedNodeMetrics:
    """Aggregate node metrics from state.

    Args:
        node_metrics: Raw node_metrics dict from state.

    Returns:
        Aggregated metrics with summary statistics.
    """
    if not node_metrics:
        return AggregatedNodeMetrics()

    return AggregatedNodeMetrics(
        node_metrics={name: NodeMetrics.from_dict(data) for name, data in node_metrics.items()}
    )


def format_run_summary_with_nodes(
    total_duration_ms: int,
    total_tokens: int,
    node_metrics: dict[str, Any] | None,
    token_metrics: TokenMetrics | None = None,
    model: str | None = None,
) -> str:
    """Format run summary with per-node breakdown.

    Phase 2.4.3: Enhanced summary output with node-level details.
    Phase 10.1.3: Added cost display with model-specific pricing.

    Args:
        total_duration_ms: Total run duration in milliseconds.
        total_tokens: Total tokens used.
        node_metrics: Raw node_metrics dict from state.
        token_metrics: Token metrics for cost calculation (Phase 10.1.3).
        model: Model name for cost calculation (Phase 10.1.3).

    Returns:
        Formatted summary string.
    """
    lines = [
        "Run Summary:",
        f"    Total Duration: {total_duration_ms:,}ms",
        f"    Total Tokens: {total_tokens:,}",
    ]

    # Phase 10.1.3: Add cost display with model-specific pricing
    if token_metrics is not None:
        target_model = model or token_metrics.model or "unknown"
        cost = token_metrics.calculate_cost_float(target_model)
        lines.append(
            f"    Tokens: {token_metrics.total:,} "
            f"(input: {token_metrics.input_tokens:,}, output: {token_metrics.output_tokens:,})"
        )
        if token_metrics.cached_tokens > 0:
            lines[-1] = lines[-1].rstrip(")") + f", cached: {token_metrics.cached_tokens:,})"
        lines.append(f"    Cost: ${cost:.4f} ({target_model})")

    if node_metrics:
        aggregated = aggregate_node_metrics(node_metrics)
        lines.append("")
        lines.append(aggregated.format_breakdown())

    return "\n".join(lines)
