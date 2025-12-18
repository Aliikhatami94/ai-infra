"""
Replay function: Replay recorded workflows with modifications.

The replay() function allows you to:
- Replay a recorded workflow from the beginning or a specific step
- Inject fake tool results for testing/debugging
- Modify workflow behavior without re-running the full agent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_infra.replay.recorder import WorkflowStep
from ai_infra.replay.storage import Storage, get_default_storage


@dataclass
class ReplayResult:
    """
    Result of a workflow replay.

    Contains the replayed steps and provides utilities for
    inspecting the workflow.

    Attributes:
        record_id: ID of the original recording
        steps: List of replayed workflow steps
        injected_steps: Steps where results were injected
        from_step: Step number replay started from
    """

    record_id: str
    steps: list[WorkflowStep] = field(default_factory=list)
    injected_steps: list[int] = field(default_factory=list)
    from_step: int = 0

    def timeline(self) -> list[str]:
        """
        Get a human-readable timeline of the replayed workflow.

        Returns:
            List of step descriptions with injection markers

        Example:
            ```python
            timeline = result.timeline()
            # ['[Step 0: llm_call]', '[Step 1: tool_call] get_weather',
            #  '[Step 2: tool_result] get_weather [INJECTED]', ...]
            ```
        """
        result = []
        for step in self.steps:
            name = step.data.get("name", step.data.get("content", "")[:30])
            marker = " [INJECTED]" if step.step_id in self.injected_steps else ""
            result.append(f"[Step {step.step_id}: {step.step_type}] {name}{marker}")
        return result

    @property
    def output(self) -> Any:
        """
        Get the final output of the workflow.

        Returns the content of the last agent_response step,
        or the last tool_result if no agent_response exists.
        """
        for step in reversed(self.steps):
            if step.step_type == "agent_response":
                return step.data.get("content")
            if step.step_type == "tool_result":
                return step.data.get("result")
        return None

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        """
        Get all tool calls from the workflow.

        Returns:
            List of tool call data (name, args)
        """
        return [step.data for step in self.steps if step.step_type == "tool_call"]

    @property
    def tool_results(self) -> list[dict[str, Any]]:
        """
        Get all tool results from the workflow.

        Returns:
            List of tool result data (name, result, error)
        """
        return [step.data for step in self.steps if step.step_type == "tool_result"]


def replay(
    record_id: str,
    *,
    from_step: int = 0,
    inject: dict[str, Any] | None = None,
    storage: Storage | None = None,
) -> ReplayResult:
    """
    Replay a recorded workflow with optional modifications.

    This function loads a previously recorded workflow and creates
    a ReplayResult. When `inject` is provided, tool results matching
    the specified tool names will be replaced with the injected values.

    Args:
        record_id: ID of the workflow recording to replay
        from_step: Step number to start replay from (0 = beginning)
        inject: Dict mapping tool names to fake results.
            When a tool_result step matches a tool name in inject,
            the result is replaced with the injected value.
        storage: Storage backend to load from. Uses default if None.

    Returns:
        ReplayResult containing the (potentially modified) workflow

    Raises:
        ValueError: If recording not found

    Example - Basic replay:
        ```python
        from ai_infra.replay import replay

        # Replay from beginning
        result = replay("workflow_123")
        print(result.timeline())
        print(result.output)
        ```

    Example - Replay from specific step:
        ```python
        # Skip first 2 steps
        result = replay("workflow_123", from_step=2)
        ```

    Example - Inject fake tool results:
        ```python
        # Replace fetch_sales result with fake data
        result = replay(
            "workflow_123",
            inject={"fetch_sales": {"total": 1000000, "q4": 500000}},
        )

        # The replayed workflow uses the fake data
        for step in result.steps:
            if step.step_type == "tool_result":
                print(step.data)  # Shows injected data
        ```

    Example - Combine from_step and inject:
        ```python
        # Start from step 2 and inject fake data
        result = replay(
            "workflow_123",
            from_step=2,
            inject={"send_email": {"status": "sent"}},
        )
        ```
    """
    # Get storage backend
    if storage is None:
        storage = get_default_storage()

    # Load recording
    steps = storage.load(record_id)
    if steps is None:
        raise ValueError(f"Recording not found: {record_id}")

    # Filter to steps from from_step onwards
    filtered_steps = [s for s in steps if s.step_id >= from_step]

    # Track which steps were injected
    injected_step_ids = []

    # Apply injections
    if inject:
        result_steps = []
        for step in filtered_steps:
            if step.step_type == "tool_result":
                tool_name = step.data.get("name")
                if tool_name and tool_name in inject:
                    # Replace result with injected value
                    new_data = step.data.copy()
                    new_data["result"] = inject[tool_name]
                    new_data["injected"] = True
                    step = WorkflowStep(
                        step_id=step.step_id,
                        step_type=step.step_type,
                        timestamp=step.timestamp,
                        data=new_data,
                    )
                    injected_step_ids.append(step.step_id)
            result_steps.append(step)
        filtered_steps = result_steps

    return ReplayResult(
        record_id=record_id,
        steps=filtered_steps,
        injected_steps=injected_step_ids,
        from_step=from_step,
    )


def get_recording(
    record_id: str,
    storage: Storage | None = None,
) -> list[WorkflowStep] | None:
    """
    Load a recording without replaying.

    Utility function to inspect raw recording data.

    Args:
        record_id: ID of the recording
        storage: Storage backend. Uses default if None.

    Returns:
        List of workflow steps, or None if not found
    """
    if storage is None:
        storage = get_default_storage()
    return storage.load(record_id)


def list_recordings(storage: Storage | None = None) -> list[str]:
    """
    List all available recording IDs.

    Args:
        storage: Storage backend. Uses default if None.

    Returns:
        List of recording IDs
    """
    if storage is None:
        storage = get_default_storage()
    return storage.list_recordings()


def delete_recording(
    record_id: str,
    storage: Storage | None = None,
) -> bool:
    """
    Delete a recording.

    Args:
        record_id: ID of the recording to delete
        storage: Storage backend. Uses default if None.

    Returns:
        True if deleted, False if not found
    """
    if storage is None:
        storage = get_default_storage()
    return storage.delete(record_id)
