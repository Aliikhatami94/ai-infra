#!/usr/bin/env python
"""Basic Graph Workflow: Building State Machine Workflows.

This example demonstrates:
- Creating workflows with Graph
- Defining nodes as simple functions
- Connecting nodes with edges
- Running workflows synchronously and asynchronously
- Streaming workflow execution
- Analyzing graph structure

The Graph class wraps LangGraph's StateGraph with a simpler API.
No API keys required for these basic examples.

Key concepts:
- nodes: Dict mapping names to functions
- edges: List of (from_node, to_node) tuples
- entry: Name of the first node to execute
- state_schema: TypedDict defining the state shape
- run/arun: Execute workflow and return final state
- stream/astream: Execute and yield intermediate states
"""

import asyncio
from typing import TypedDict

from ai_infra import Graph

# =============================================================================
# Example 1: Minimal Graph
# =============================================================================


def example_minimal_graph() -> None:
    """Create the simplest possible workflow."""
    print("\n" + "=" * 60)
    print("Example 1: Minimal Graph")
    print("=" * 60)

    # Nodes are just functions that take state and return updates
    def step_a(state: dict) -> dict:
        return {"value": state.get("value", 0) + 1}

    def step_b(state: dict) -> dict:
        return {"value": state["value"] * 2}

    def step_c(state: dict) -> dict:
        return {"result": f"Final value: {state['value']}"}

    # Create graph with simple dict-based API
    graph = Graph(
        nodes={
            "step_a": step_a,
            "step_b": step_b,
            "step_c": step_c,
        },
        edges=[
            ("step_a", "step_b"),
            ("step_b", "step_c"),
        ],
        entry="step_a",
    )

    # Run the workflow
    result = graph.run({"value": 5})

    print("\nInitial state: {'value': 5}")
    print(f"Final state: {result}")
    print("\nFlow: step_a(+1) -> step_b(*2) -> step_c(format)")
    print("Expected: 5 + 1 = 6, 6 * 2 = 12, result = 'Final value: 12'")


# =============================================================================
# Example 2: Typed State Schema
# =============================================================================


class ProcessingState(TypedDict, total=False):
    """Typed state for a data processing workflow."""

    input_text: str
    tokens: list[str]
    word_count: int
    processed: bool


def example_typed_state() -> None:
    """Use TypedDict for type-safe state."""
    print("\n" + "=" * 60)
    print("Example 2: Typed State Schema")
    print("=" * 60)

    def tokenize(state: ProcessingState) -> dict:
        """Split input into tokens."""
        tokens = state["input_text"].split()
        return {"tokens": tokens}

    def count_words(state: ProcessingState) -> dict:
        """Count the number of tokens."""
        return {"word_count": len(state["tokens"])}

    def finalize(state: ProcessingState) -> dict:
        """Mark processing as complete."""
        return {"processed": True}

    graph = Graph(
        nodes={
            "tokenize": tokenize,
            "count": count_words,
            "finalize": finalize,
        },
        edges=[
            ("tokenize", "count"),
            ("count", "finalize"),
        ],
        entry="tokenize",
        state_schema=ProcessingState,
    )

    result = graph.run({"input_text": "Hello world this is a test"})

    print("\nInput: 'Hello world this is a test'")
    print(f"Tokens: {result.get('tokens')}")
    print(f"Word count: {result.get('word_count')}")
    print(f"Processed: {result.get('processed')}")


# =============================================================================
# Example 3: Streaming Execution
# =============================================================================


def example_streaming() -> None:
    """Stream intermediate states during execution."""
    print("\n" + "=" * 60)
    print("Example 3: Streaming Execution")
    print("=" * 60)

    def step1(state: dict) -> dict:
        return {"steps": ["step1"], "counter": 1}

    def step2(state: dict) -> dict:
        return {"steps": state["steps"] + ["step2"], "counter": state["counter"] + 1}

    def step3(state: dict) -> dict:
        return {"steps": state["steps"] + ["step3"], "counter": state["counter"] + 1}

    def step4(state: dict) -> dict:
        return {"steps": state["steps"] + ["step4"], "counter": state["counter"] + 1}

    graph = Graph(
        nodes={
            "step1": step1,
            "step2": step2,
            "step3": step3,
            "step4": step4,
        },
        edges=[
            ("step1", "step2"),
            ("step2", "step3"),
            ("step3", "step4"),
        ],
        entry="step1",
    )

    print("\n--- Streaming Updates ---\n")

    # Stream mode "updates" shows what each node returned
    for mode, chunk in graph.stream({}):
        if mode == "updates":
            print(f"Update: {chunk}")

    print("\n--- Streaming Values ---\n")

    # stream_values shows cumulative state after each node
    for state in graph.stream_values({}):
        print(f"State: {state}")


# =============================================================================
# Example 4: Async Workflow
# =============================================================================


async def example_async_workflow() -> None:
    """Run workflows asynchronously."""
    print("\n" + "=" * 60)
    print("Example 4: Async Workflow")
    print("=" * 60)

    async def fetch_data(state: dict) -> dict:
        """Simulate async data fetching."""
        await asyncio.sleep(0.1)  # Simulate network call
        return {"data": {"id": state["user_id"], "name": "Alice"}}

    async def process_data(state: dict) -> dict:
        """Process the fetched data."""
        await asyncio.sleep(0.05)  # Simulate processing
        return {"processed": {"greeting": f"Hello, {state['data']['name']}!"}}

    async def save_result(state: dict) -> dict:
        """Save the processed result."""
        await asyncio.sleep(0.05)  # Simulate save
        return {"saved": True}

    graph = Graph(
        nodes={
            "fetch": fetch_data,
            "process": process_data,
            "save": save_result,
        },
        edges=[
            ("fetch", "process"),
            ("process", "save"),
        ],
        entry="fetch",
    )

    print("\n--- Running async workflow ---\n")

    result = await graph.arun({"user_id": "user123"})

    print("Input: {'user_id': 'user123'}")
    print(f"Fetched: {result.get('data')}")
    print(f"Processed: {result.get('processed')}")
    print(f"Saved: {result.get('saved')}")


# =============================================================================
# Example 5: Graph Analysis
# =============================================================================


def example_graph_analysis() -> None:
    """Analyze graph structure and generate diagrams."""
    print("\n" + "=" * 60)
    print("Example 5: Graph Analysis")
    print("=" * 60)

    def node_a(state: dict) -> dict:
        return {"visited": state.get("visited", []) + ["A"]}

    def node_b(state: dict) -> dict:
        return {"visited": state["visited"] + ["B"]}

    def node_c(state: dict) -> dict:
        return {"visited": state["visited"] + ["C"]}

    def node_d(state: dict) -> dict:
        return {"visited": state["visited"] + ["D"]}

    graph = Graph(
        nodes={
            "node_a": node_a,
            "node_b": node_b,
            "node_c": node_c,
            "node_d": node_d,
        },
        edges=[
            ("node_a", "node_b"),
            ("node_b", "node_c"),
            ("node_c", "node_d"),
        ],
        entry="node_a",
    )

    # Get graph structure
    structure = graph.analyze()

    print("\n--- Graph Structure ---\n")
    print(f"State type: {structure.state_type_name}")
    print(f"Node count: {structure.node_count}")
    print(f"Nodes: {structure.nodes}")
    print(f"Edge count: {structure.edge_count}")
    print(f"Edges: {structure.edges}")
    print(f"Entry points: {structure.entry_points}")
    print(f"Exit points: {structure.exit_points}")
    print(f"Has memory: {structure.has_memory}")

    # Get Mermaid diagram
    print("\n--- Mermaid Diagram ---\n")
    diagram = graph.get_arch_diagram()
    print(diagram)


# =============================================================================
# Example 6: Lambda and Method Nodes
# =============================================================================


class DataProcessor:
    """Example class with methods that can be graph nodes."""

    def __init__(self, multiplier: int = 2):
        self.multiplier = multiplier

    def multiply(self, state: dict) -> dict:
        """Multiply value by configured multiplier."""
        return {"value": state["value"] * self.multiplier}


def example_callable_nodes() -> None:
    """Use lambdas and methods as graph nodes."""
    print("\n" + "=" * 60)
    print("Example 6: Lambda and Method Nodes")
    print("=" * 60)

    processor = DataProcessor(multiplier=3)

    graph = Graph(
        nodes={
            # Lambda node
            "init": lambda state: {"value": 10},
            # Instance method node
            "multiply": processor.multiply,
            # Lambda with computation
            "double": lambda state: {"value": state["value"] * 2},
            # Lambda for formatting
            "format": lambda state: {"result": f"Value is {state['value']}"},
        },
        edges=[
            ("init", "multiply"),
            ("multiply", "double"),
            ("double", "format"),
        ],
        entry="init",
    )

    result = graph.run({})

    print("\nFlow: init(10) -> multiply(*3) -> double(*2) -> format")
    print("Computation: 10 -> 30 -> 60 -> 'Value is 60'")
    print(f"Result: {result.get('result')}")


# =============================================================================
# Example 7: Hooks and Tracing
# =============================================================================


def example_hooks_and_tracing() -> None:
    """Add hooks for debugging and tracing."""
    print("\n" + "=" * 60)
    print("Example 7: Hooks and Tracing")
    print("=" * 60)

    def step_a(state: dict) -> dict:
        return {"a": True}

    def step_b(state: dict) -> dict:
        return {"b": True}

    def step_c(state: dict) -> dict:
        return {"c": True}

    graph = Graph(
        nodes={
            "step_a": step_a,
            "step_b": step_b,
            "step_c": step_c,
        },
        edges=[
            ("step_a", "step_b"),
            ("step_b", "step_c"),
        ],
        entry="step_a",
    )

    # Track execution with hooks
    execution_log = []

    def on_enter(node_name: str, state: dict) -> None:
        execution_log.append(f"ENTER: {node_name}")

    def on_exit(node_name: str, state: dict) -> None:
        execution_log.append(f"EXIT: {node_name}")

    def trace_fn(node_name: str, state: dict, result: dict) -> None:
        execution_log.append(f"TRACE: {node_name} returned {result}")

    print("\n--- Running with hooks ---\n")

    result = graph.run({}, on_enter=on_enter, on_exit=on_exit, trace=trace_fn)

    for log_entry in execution_log:
        print(log_entry)

    print(f"\nFinal state: {result}")


# =============================================================================
# Example 8: State Validation
# =============================================================================


class ValidatedState(TypedDict, total=False):
    """State with validation enabled."""

    input: str
    output: str
    step: int


def example_state_validation() -> None:
    """Enable state validation for debugging."""
    print("\n" + "=" * 60)
    print("Example 8: State Validation")
    print("=" * 60)

    def process(state: ValidatedState) -> dict:
        return {"output": state["input"].upper(), "step": 1}

    graph = Graph(
        nodes={"process": process},
        edges=[],
        entry="process",
        state_schema=ValidatedState,
        validate_state=True,
    )

    # Valid state
    print("\n--- Valid State ---")
    result = graph.run({"input": "hello"})
    print(f"Input: 'hello' -> Output: {result.get('output')}")

    # With validation enabled, invalid states would raise
    print("\n--- State Validation Enabled ---")
    print("If state doesn't match schema, validation will catch it.")
    print("This helps catch bugs early in development.")


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run all examples."""
    print("=" * 60)
    print("Basic Graph Workflow Examples")
    print("Building State Machine Workflows")
    print("=" * 60)

    # Sync examples
    example_minimal_graph()
    example_typed_state()
    example_streaming()
    example_graph_analysis()
    example_callable_nodes()
    example_hooks_and_tracing()
    example_state_validation()

    # Async examples
    await example_async_workflow()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
