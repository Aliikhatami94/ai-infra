"""Edge routing module for executor graph.

Phase 1.1.3: Conditional edge routing functions.
Phase 1.2.3: Added validation and repair routing.
Phase 1.3: Added write files routing.
Phase 2.9.8: Added recovery flow routing.

This module provides all routing logic for the executor graph edges.
Each function takes the current state and returns the next node name.
"""

from ai_infra.executor.edges.routes import (
    END,
    route_after_apply_recovery,
    route_after_await_approval,
    route_after_classify_failure,
    route_after_decide,
    route_after_decide_with_recovery,
    route_after_execute,
    route_after_failure,
    route_after_pick,
    route_after_pick_with_recovery,
    route_after_propose_recovery,
    route_after_repair,
    route_after_retry_deferred,
    route_after_rollback,
    route_after_validate,
    route_after_verify,
    route_after_write,
    route_to_recovery_or_failure,
)

__all__ = [
    "route_after_pick",
    "route_after_execute",
    "route_after_verify",
    "route_after_failure",
    "route_after_rollback",
    "route_after_decide",
    # Phase 1.2.3: Validation and repair routes
    "route_after_validate",
    "route_after_repair",
    # Phase 1.3: Write files route
    "route_after_write",
    # Phase 2.9.8: Recovery flow routes
    "route_after_classify_failure",
    "route_after_propose_recovery",
    "route_after_await_approval",
    "route_after_apply_recovery",
    "route_after_retry_deferred",
    "route_after_pick_with_recovery",
    "route_after_decide_with_recovery",
    "route_to_recovery_or_failure",
    "END",
]
