"""Enhanced Human-in-the-Loop (HITL) actions for rich human-agent collaboration.

Phase 4.1 of EXECUTOR_1.md: Enhanced HITL Actions.

This module transforms binary approve/reject into rich collaboration:
- Edit: Modify the proposal with specific changes
- Suggest: Provide hints without full specification
- Explain: Ask for reasoning behind decisions
- Rollback: Return to previous checkpoint state
- Skip: Skip current task and continue
- Delegate: Hand off to another agent type

Example:
    ```python
    from ai_infra.executor.hitl import (
        HITLAction,
        HITLActionType,
        HITLResponse,
        EditHandler,
        SuggestHandler,
        ExplainHandler,
    )

    # User edits a proposal
    action = HITLAction(
        type=HITLActionType.EDIT,
        content="Use async/await instead of threads",
    )

    handler = EditHandler(llm=my_llm)
    response = await handler.handle(action, proposal, context)

    if response.understood:
        print(f"Revised plan: {response.revised_plan}")
    ```
"""

# Re-export legacy HITL components for backward compatibility (Phase 1.5)
# Phase 4.1 Enhanced HITL Actions
from ai_infra.executor.hitl.actions import (
    HITLAction,
    HITLActionType,
    HITLProposal,
    HITLResponse,
)
from ai_infra.executor.hitl.handlers import (
    EditHandler,
    ExplainHandler,
    HITLActionHandler,
    HITLHandlerRegistry,
    RollbackHandler,
    SuggestHandler,
)
from ai_infra.executor.hitl_legacy import (
    HITL_STATE_FILE,
    HITLDecision,
    HITLManager,
    HITLState,
    InterruptConfig,
    InterruptPoint,
    create_hitl_config_from_executor_config,
    get_interrupt_lists,
)

__all__ = [
    # Legacy exports (Phase 1.5)
    "HITL_STATE_FILE",
    "HITLDecision",
    "HITLManager",
    "HITLState",
    "InterruptConfig",
    "InterruptPoint",
    "create_hitl_config_from_executor_config",
    "get_interrupt_lists",
    # Phase 4.1 Action types
    "HITLAction",
    "HITLActionType",
    "HITLProposal",
    "HITLResponse",
    # Phase 4.1 Handlers
    "EditHandler",
    "ExplainHandler",
    "HITLActionHandler",
    "HITLHandlerRegistry",
    "RollbackHandler",
    "SuggestHandler",
]
