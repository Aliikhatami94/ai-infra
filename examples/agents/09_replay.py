#!/usr/bin/env python
"""Workflow Replay and Debugging Example.

This example demonstrates:
- Recording agent workflow executions
- Replaying workflows with modifications
- Injecting fake tool results for testing
- Starting replay from specific steps
- Debugging agent behavior

Replay is essential for debugging complex agent workflows
and testing edge cases without re-running the full agent.

Required API Keys:
- OPENAI_API_KEY or ANTHROPIC_API_KEY
"""

from ai_infra.replay import (
    MemoryStorage,
    WorkflowRecorder,
    replay,
)
from ai_infra.replay.storage import get_default_storage, set_default_storage

# =============================================================================
# Basic Recording and Replay
# =============================================================================


def basic_recording():
    """Record a workflow manually for demonstration."""
    print("=" * 60)
    print("Basic Workflow Recording")
    print("=" * 60)

    # Create in-memory storage for demo
    storage = MemoryStorage()
    set_default_storage(storage)

    # Create a recorder
    recorder = WorkflowRecorder("demo_workflow_001", storage)

    # Simulate recording an agent workflow
    print("\nRecording workflow steps...")

    # Step 1: LLM call (user prompt)
    recorder.record_llm_call(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in NYC?"},
        ]
    )
    print("  Recorded: LLM call with user prompt")

    # Step 2: Tool call
    recorder.record_tool_call("get_weather", {"city": "NYC"})
    print("  Recorded: Tool call - get_weather(city='NYC')")

    # Step 3: Tool result
    recorder.record_tool_result("get_weather", {"temp": 72, "condition": "sunny"})
    print("  Recorded: Tool result - {'temp': 72, 'condition': 'sunny'}")

    # Step 4: Agent response
    recorder.record_agent_response("The weather in NYC is 72°F and sunny.")
    print("  Recorded: Agent response")

    # Save to storage
    recorder.save()
    print("\n✓ Workflow saved with ID: demo_workflow_001")


def basic_replay():
    """Replay a recorded workflow."""
    print("\n" + "=" * 60)
    print("Basic Workflow Replay")
    print("=" * 60)

    # Replay the workflow we just recorded
    result = replay("demo_workflow_001")

    print("\nReplayed workflow timeline:")
    for line in result.timeline():
        print(f"  {line}")

    print(f"\nFinal output: {result.output}")


# =============================================================================
# Replay with Injection
# =============================================================================


def replay_with_injection():
    """Replay workflow with injected tool results."""
    print("\n" + "=" * 60)
    print("Replay with Fake Tool Results")
    print("=" * 60)

    # Inject a different weather result
    result = replay(
        "demo_workflow_001",
        inject={"get_weather": {"temp": 32, "condition": "snowing"}},
    )

    print("\nReplayed with injected data:")
    for line in result.timeline():
        print(f"  {line}")

    print("\nTool results after injection:")
    for tool_result in result.tool_results:
        name = tool_result.get("name")
        data = tool_result.get("result")
        injected = tool_result.get("injected", False)
        marker = " [INJECTED]" if injected else ""
        print(f"  {name}: {data}{marker}")


# =============================================================================
# Replay from Specific Step
# =============================================================================


def replay_from_step():
    """Replay workflow from a specific step."""
    print("\n" + "=" * 60)
    print("Replay from Specific Step")
    print("=" * 60)

    # Skip the LLM call and tool call, start from tool result
    result = replay("demo_workflow_001", from_step=2)

    print("\nReplayed from step 2 (skipping initial steps):")
    for line in result.timeline():
        print(f"  {line}")

    print(f"\nSteps included: {len(result.steps)}")


# =============================================================================
# Multi-Step Workflow
# =============================================================================


def multi_step_workflow():
    """Record and replay a multi-step workflow."""
    print("\n" + "=" * 60)
    print("Multi-Step Workflow Recording")
    print("=" * 60)

    storage = get_default_storage()
    recorder = WorkflowRecorder("multi_step_001", storage)

    print("\nRecording multi-step workflow...")

    # Initial user request
    recorder.record_llm_call(
        [
            {"role": "user", "content": "Analyze sales data and send a report"},
        ]
    )

    # First tool: fetch sales data
    recorder.record_tool_call("fetch_sales", {"quarter": "Q4", "year": 2024})
    recorder.record_tool_result(
        "fetch_sales",
        {
            "total": 500000,
            "by_region": {"east": 200000, "west": 150000, "central": 150000},
        },
    )

    # Second tool: analyze data
    recorder.record_tool_call("analyze_data", {"data_type": "sales"})
    recorder.record_tool_result(
        "analyze_data",
        {
            "trend": "up",
            "growth": "15%",
            "top_product": "Widget Pro",
        },
    )

    # Third tool: create chart
    recorder.record_tool_call("create_chart", {"chart_type": "bar"})
    recorder.record_tool_result(
        "create_chart",
        {
            "chart_id": "chart_123",
            "url": "https://example.com/charts/123",
        },
    )

    # Fourth tool: send email
    recorder.record_tool_call(
        "send_email",
        {
            "to": "team@example.com",
            "subject": "Q4 Sales Report",
        },
    )
    recorder.record_tool_result("send_email", {"status": "sent", "id": "email_456"})

    # Final response
    recorder.record_agent_response(
        "I've analyzed the Q4 sales data and sent a report to the team. "
        "Key findings: 15% growth, top product was Widget Pro."
    )

    recorder.save()
    print("✓ Multi-step workflow saved\n")

    # Replay and show timeline
    result = replay("multi_step_001")
    print("Workflow timeline:")
    for line in result.timeline():
        print(f"  {line}")


def test_with_injected_data():
    """Test workflow with different data scenarios."""
    print("\n" + "=" * 60)
    print("Testing with Injected Data")
    print("=" * 60)

    # Scenario 1: What if sales were lower?
    print("\nScenario 1: Lower sales numbers")
    result = replay(
        "multi_step_001",
        inject={
            "fetch_sales": {"total": 100000, "by_region": {"east": 50000, "west": 30000}},
        },
    )
    sales_result = next(r for r in result.tool_results if r.get("name") == "fetch_sales")
    print(f"  Injected sales: {sales_result['result']}")

    # Scenario 2: What if email failed?
    print("\nScenario 2: Email sending failed")
    result = replay(
        "multi_step_001",
        inject={
            "send_email": {"status": "failed", "error": "SMTP timeout"},
        },
    )
    email_result = next(r for r in result.tool_results if r.get("name") == "send_email")
    print(f"  Injected email result: {email_result['result']}")


# =============================================================================
# ReplayResult Utilities
# =============================================================================


def replay_result_utilities():
    """Explore ReplayResult utility methods."""
    print("\n" + "=" * 60)
    print("ReplayResult Utilities")
    print("=" * 60)

    result = replay("multi_step_001")

    # Get all tool calls
    print("\nTool calls in workflow:")
    for call in result.tool_calls:
        print(f"  {call.get('name')}: {call.get('args')}")

    # Get all tool results
    print("\nTool results in workflow:")
    for res in result.tool_results:
        print(f"  {res.get('name')}: {res.get('result')}")

    # Get final output
    print(f"\nFinal output: {result.output[:100]}...")


# =============================================================================
# WorkflowStep Details
# =============================================================================


def workflow_step_details():
    """Examine individual workflow steps."""
    print("\n" + "=" * 60)
    print("WorkflowStep Details")
    print("=" * 60)

    result = replay("multi_step_001")

    print("\nDetailed step inspection:")
    for step in result.steps[:3]:  # First 3 steps
        print(f"\n  Step {step.step_id}:")
        print(f"    Type: {step.step_type}")
        print(f"    Timestamp: {step.timestamp}")
        print(f"    Data keys: {list(step.data.keys())}")

        # Serialize to dict (for storage)
        serialized = step.to_dict()
        print(f"    Serialized: {serialized['step_type']}")


# =============================================================================
# Storage Backends
# =============================================================================


def storage_backends():
    """Demonstrate different storage backends."""
    print("\n" + "=" * 60)
    print("Storage Backends")
    print("=" * 60)

    print("\n1. MemoryStorage (default for testing):")
    print("   - In-memory, no persistence")
    print("   - Fast, good for unit tests")
    memory_storage = MemoryStorage()
    print(f"   Created: {memory_storage}")

    print("\n2. SQLiteStorage (persistent):")
    print("   - File-based persistence")
    print("   - Good for development/debugging")
    from ai_infra.replay import SQLiteStorage

    sqlite_storage = SQLiteStorage("/tmp/workflows.db")
    print(f"   Created: {sqlite_storage}")

    # Record to SQLite
    recorder = WorkflowRecorder("sqlite_demo", sqlite_storage)
    recorder.record_llm_call([{"role": "user", "content": "Test"}])
    recorder.record_agent_response("Test response")
    recorder.save()
    print("\n   Recorded workflow to SQLite")

    # Replay from SQLite
    result = replay("sqlite_demo", storage=sqlite_storage)
    print(f"   Replayed: {len(result.steps)} steps")


# =============================================================================
# Debugging Patterns
# =============================================================================


def debugging_patterns():
    """Common debugging patterns with replay."""
    print("\n" + "=" * 60)
    print("Debugging Patterns")
    print("=" * 60)

    print("""
Common debugging patterns with replay:

1. REPRODUCE BUGS
   - Record the failing workflow
   - Replay to see exact sequence of events
   - Inject different inputs to narrow down cause

2. TEST EDGE CASES
   - Record a successful workflow
   - Inject edge case data (empty results, errors, etc.)
   - See how agent would handle different scenarios

3. COMPARE BEFORE/AFTER
   - Record workflow with old logic
   - Make code changes
   - Replay with same inputs
   - Compare outputs

4. SKIP SLOW STEPS
   - Record full workflow including slow API calls
   - Replay from step N to skip slow initialization
   - Inject cached results for expensive operations

5. TEST ERROR HANDLING
   - Record successful workflow
   - Inject error results
   - Verify agent handles errors gracefully

Example - Testing error handling:
    result = replay(
        "workflow_id",
        inject={"api_call": {"error": "Connection timeout"}},
    )
    # Check if agent handled error appropriately
""")


# =============================================================================
# Integration with Agent (Conceptual)
# =============================================================================


def agent_integration_concept():
    """Show how replay integrates with Agent (conceptual)."""
    print("\n" + "=" * 60)
    print("Agent Integration (Conceptual)")
    print("=" * 60)

    print("""
Replay integrates with Agent through callbacks and session:

1. RECORDING (during agent.run):
    ```python
    from ai_infra import Agent
    from ai_infra.replay import WorkflowRecorder

    agent = Agent(tools=[...])

    # Enable recording via callback or wrapper
    # (Implementation details may vary)
    recorder = WorkflowRecorder("session_123")

    result = agent.run(
        "Analyze data",
        callbacks=[recorder.as_callback()],
    )
    ```

2. REPLAYING (for debugging):
    ```python
    from ai_infra.replay import replay

    # Replay the recorded session
    result = replay("session_123")
    print(result.timeline())

    # Inject fake data for testing
    result = replay(
        "session_123",
        inject={"fetch_data": {"test": "data"}},
    )
    ```

3. SESSION INTEGRATION:
    - Session storage can include replay data
    - Pause/resume works with replay
    - Full workflow history available
""")


if __name__ == "__main__":
    # Set up in-memory storage for examples
    storage = MemoryStorage()
    set_default_storage(storage)

    # Run examples
    basic_recording()
    basic_replay()
    replay_with_injection()
    replay_from_step()
    multi_step_workflow()
    test_with_injected_data()
    replay_result_utilities()
    workflow_step_details()
    storage_backends()
    debugging_patterns()
    agent_integration_concept()
