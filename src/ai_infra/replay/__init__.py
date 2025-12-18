"""
Workflow Replay: Record and replay agent workflows for debugging and testing.

This module provides the ability to record agent workflow executions
and replay them with optional modifications (inject fake tool results,
start from a specific step, etc.).

Example:
    ```python
    from ai_infra import Agent
    from ai_infra.replay import replay, WorkflowRecorder

    # Enable recording on agent
    agent = Agent(tools=[...], record=True)

    # Run with recording
    result = agent.run(
        "Analyze Q4 sales and send report",
        record_id="workflow_123",
    )

    # Later: replay with modifications
    new_result = replay(
        "workflow_123",
        from_step=2,
        inject={"fetch_sales": {"total": 1000000}},
    )

    # View workflow timeline
    timeline = replay("workflow_123").timeline()
    ```
"""

from ai_infra.replay.recorder import WorkflowRecorder, WorkflowStep
from ai_infra.replay.replay import ReplayResult, replay
from ai_infra.replay.storage import MemoryStorage, SQLiteStorage, Storage

__all__ = [
    "MemoryStorage",
    "ReplayResult",
    "SQLiteStorage",
    "Storage",
    "WorkflowRecorder",
    "WorkflowStep",
    "replay",
]
