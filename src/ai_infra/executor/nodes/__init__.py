"""Graph nodes for the Executor.

Phase 1.2.2: All node implementations for the executor graph.
Phase 2.3.1: Added analyze_failure_node and replan_task_node for adaptive replanning.
Phase 2.4.2: Added plan_task_node for pre-execution planning.
Phase 1.1: Added validate_code_node for pre-write validation.

Each node is a function that:
- Takes ExecutorGraphState as input
- Returns ExecutorGraphState with updates
- Uses immutable update pattern: {**state, "key": value}
"""

from ai_infra.executor.nodes.checkpoint import checkpoint_node
from ai_infra.executor.nodes.context import build_context_node
from ai_infra.executor.nodes.decide import decide_next_node
from ai_infra.executor.nodes.execute import execute_task_node
from ai_infra.executor.nodes.failure import (
    FailureClassification,
    analyze_failure_node,
    handle_failure_node,
)
from ai_infra.executor.nodes.parse import parse_roadmap_node
from ai_infra.executor.nodes.pick import pick_task_node
from ai_infra.executor.nodes.plan import plan_task_node
from ai_infra.executor.nodes.recovery import (
    CLASSIFY_FAILURE_PROMPT,
    DECOMPOSE_TASK_PROMPT,
    FAILURE_TO_STRATEGY,
    MAX_DEFERRED_RETRIES,
    REWRITE_TASK_PROMPT,
    ApprovalMode,
    ExecutionReport,
    FailureReason,
    RecoveryProposal,
    RecoveryStrategy,
    apply_recovery_node,
    await_approval_node,
    classify_failure_node,
    generate_report_node,
    get_strategy_for_failure,
    propose_recovery_node,
    retry_deferred_node,
    should_retry_deferred,
)
from ai_infra.executor.nodes.repair import (
    MAX_REPAIRS,
    repair_code_node,
    should_repair,
)
from ai_infra.executor.nodes.repair_test import (
    TestFailure,
    repair_test_node,
)
from ai_infra.executor.nodes.replan import replan_task_node
from ai_infra.executor.nodes.rollback import rollback_node  # Deprecated in Phase 2.1
from ai_infra.executor.nodes.validate import (
    ValidationResult,
    validate_code_node,
    validate_json,
    validate_python_code,
    validate_yaml,
)
from ai_infra.executor.nodes.verify import verify_task_node
from ai_infra.executor.nodes.write import write_files_node

__all__ = [
    # Core flow nodes
    "parse_roadmap_node",
    "pick_task_node",
    "plan_task_node",  # Phase 2.4.2
    "build_context_node",
    "execute_task_node",
    "validate_code_node",  # Phase 1.1: Pre-write validation
    "repair_code_node",  # Phase 1.2: Surgical repair
    "write_files_node",  # Phase 1.3: Separated file writing
    "verify_task_node",
    "checkpoint_node",
    # Error handling nodes (Phase 2.1: rollback_node is deprecated)
    "rollback_node",  # Deprecated - kept for backward compatibility
    "handle_failure_node",
    # Phase 2.3.1: Adaptive replanning nodes
    "analyze_failure_node",
    "replan_task_node",
    "FailureClassification",
    # Phase 2.9.1: Intelligent task recovery
    "FailureReason",
    "RecoveryStrategy",
    "RecoveryProposal",
    "FAILURE_TO_STRATEGY",
    "get_strategy_for_failure",
    # Phase 2.9.2: Classify failure node
    "classify_failure_node",
    "CLASSIFY_FAILURE_PROMPT",
    # Phase 2.9.3: Propose recovery node
    "propose_recovery_node",
    "REWRITE_TASK_PROMPT",
    "DECOMPOSE_TASK_PROMPT",
    # Phase 2.9.4: Approval modes
    "ApprovalMode",
    "await_approval_node",
    # Phase 2.9.5: Apply recovery
    "apply_recovery_node",
    # Phase 2.9.6: Deferred task retry
    "retry_deferred_node",
    "should_retry_deferred",
    "MAX_DEFERRED_RETRIES",
    # Phase 2.9.7: Audit trail & reporting
    "ExecutionReport",
    "generate_report_node",
    # Phase 1.1: Validation helpers
    "ValidationResult",
    "validate_python_code",
    "validate_json",
    "validate_yaml",
    # Phase 1.2: Repair helpers
    "MAX_REPAIRS",
    "should_repair",
    # Decision nodes
    "decide_next_node",
    # Test nodes
    "repair_test_node",
    "TestFailure",
]
