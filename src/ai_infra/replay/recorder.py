"""
WorkflowRecorder: Records agent workflow steps for replay.

Records LLM calls, tool invocations, and tool results to enable
replaying workflows with modifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ai_infra.replay.storage import Storage


StepType = Literal["llm_call", "tool_call", "tool_result", "agent_response"]


@dataclass
class WorkflowStep:
    """
    A single step in a recorded workflow.

    Represents one discrete action: an LLM call, tool invocation,
    tool result, or final agent response.

    Attributes:
        step_id: Sequential step number (0-indexed)
        step_type: Type of step (llm_call, tool_call, tool_result, agent_response)
        timestamp: When the step occurred
        data: Step-specific data (messages, tool name, arguments, results, etc.)
    """

    step_id: int
    step_type: StepType
    timestamp: datetime
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize step to dictionary."""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowStep:
        """Deserialize step from dictionary."""
        return cls(
            step_id=data["step_id"],
            step_type=data["step_type"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data.get("data", {}),
        )


class WorkflowRecorder:
    """
    Records agent workflow steps for later replay.

    The recorder captures:
    - LLM calls (messages sent to model)
    - Tool calls (tool name and arguments)
    - Tool results (return values from tools)
    - Final agent responses

    Example:
        ```python
        from ai_infra.replay import WorkflowRecorder, MemoryStorage

        storage = MemoryStorage()
        recorder = WorkflowRecorder("workflow_123", storage)

        # Record steps as agent runs
        recorder.record_llm_call([{"role": "user", "content": "Hello"}])
        recorder.record_tool_call("get_weather", {"city": "NYC"})
        recorder.record_tool_result("get_weather", {"temp": 72})
        recorder.record_agent_response("The weather in NYC is 72°F")

        # Save to storage
        recorder.save()
        ```
    """

    def __init__(
        self,
        record_id: str,
        storage: Storage | None = None,
    ):
        """
        Initialize a workflow recorder.

        Args:
            record_id: Unique identifier for this recording
            storage: Storage backend. If None, uses default storage.
        """
        self.record_id = record_id
        self._storage = storage
        self._steps: list[WorkflowStep] = []
        self._step_counter = 0

    @property
    def storage(self) -> Storage:
        """Get storage backend, using default if not set."""
        if self._storage is None:
            from ai_infra.replay.storage import get_default_storage

            return get_default_storage()
        return self._storage

    @property
    def steps(self) -> list[WorkflowStep]:
        """Get recorded steps."""
        return self._steps.copy()

    def record_llm_call(
        self,
        messages: list[dict[str, Any]],
        response: str | None = None,
        model: str | None = None,
        **metadata: Any,
    ) -> WorkflowStep:
        """
        Record an LLM call.

        Args:
            messages: Messages sent to the model
            response: Optional response from the model
            model: Model name/identifier
            **metadata: Additional metadata to record

        Returns:
            The recorded step
        """
        step = WorkflowStep(
            step_id=self._step_counter,
            step_type="llm_call",
            timestamp=datetime.now(),
            data={
                "messages": messages,
                "response": response,
                "model": model,
                **metadata,
            },
        )
        self._steps.append(step)
        self._step_counter += 1
        return step

    def record_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        **metadata: Any,
    ) -> WorkflowStep:
        """
        Record a tool invocation.

        Args:
            tool_name: Name of the tool being called
            args: Arguments passed to the tool
            **metadata: Additional metadata to record

        Returns:
            The recorded step
        """
        step = WorkflowStep(
            step_id=self._step_counter,
            step_type="tool_call",
            timestamp=datetime.now(),
            data={
                "name": tool_name,
                "args": args,
                **metadata,
            },
        )
        self._steps.append(step)
        self._step_counter += 1
        return step

    def record_tool_result(
        self,
        tool_name: str,
        result: Any,
        duration_ms: float | None = None,
        error: str | None = None,
        **metadata: Any,
    ) -> WorkflowStep:
        """
        Record a tool result.

        Args:
            tool_name: Name of the tool
            result: Return value from the tool
            duration_ms: Execution time in milliseconds
            error: Error message if tool failed
            **metadata: Additional metadata to record

        Returns:
            The recorded step
        """
        step = WorkflowStep(
            step_id=self._step_counter,
            step_type="tool_result",
            timestamp=datetime.now(),
            data={
                "name": tool_name,
                "result": result,
                "duration_ms": duration_ms,
                "error": error,
                **metadata,
            },
        )
        self._steps.append(step)
        self._step_counter += 1
        return step

    def record_agent_response(
        self,
        content: str,
        **metadata: Any,
    ) -> WorkflowStep:
        """
        Record the final agent response.

        Args:
            content: The agent's final response text
            **metadata: Additional metadata to record

        Returns:
            The recorded step
        """
        step = WorkflowStep(
            step_id=self._step_counter,
            step_type="agent_response",
            timestamp=datetime.now(),
            data={
                "content": content,
                **metadata,
            },
        )
        self._steps.append(step)
        self._step_counter += 1
        return step

    def save(self) -> None:
        """Persist recording to storage."""
        self.storage.save(self.record_id, self._steps)

    def clear(self) -> None:
        """Clear all recorded steps."""
        self._steps = []
        self._step_counter = 0

    def timeline(self) -> list[str]:
        """
        Get a human-readable timeline of the workflow.

        Returns:
            List of step descriptions

        Example:
            ```python
            timeline = recorder.timeline()
            # ['[Step 0: llm_call] ', '[Step 1: tool_call] get_weather', ...]
            ```
        """
        result = []
        for step in self._steps:
            name = step.data.get("name", step.data.get("content", "")[:30])
            result.append(f"[Step {step.step_id}: {step.step_type}] {name}")
        return result
