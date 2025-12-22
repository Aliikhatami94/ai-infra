#!/usr/bin/env python
"""Conditional Graph Workflows: Dynamic Branching and Routing.

This example demonstrates:
- Conditional edges that branch based on state
- Router functions for dynamic path selection
- path_map for named branches
- Multiple conditional paths from one node
- Combining conditional and regular edges

Conditional edges let your workflow make decisions at runtime,
choosing different paths based on the current state.

Key concepts:
- Conditional edge tuple: (source, router_fn, path_map)
- Router function: Takes state, returns path key
- path_map: Dict mapping keys to target node names
- Combined with regular edges: Full flexibility
"""

import asyncio
import random
from typing import TypedDict

from ai_infra import Graph

# =============================================================================
# Example 1: Basic Conditional Branch
# =============================================================================


def example_basic_conditional() -> None:
    """Create a simple if/else branch in the workflow."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Conditional Branch")
    print("=" * 60)

    def evaluate(state: dict) -> dict:
        """Evaluate the input and set a flag."""
        score = state.get("score", 0)
        return {"is_passing": score >= 70}

    def on_pass(state: dict) -> dict:
        """Handle passing scores."""
        return {"result": "Congratulations! You passed!"}

    def on_fail(state: dict) -> dict:
        """Handle failing scores."""
        return {"result": "Sorry, you need to try again."}

    # Router function determines which path to take
    def route_by_result(state: dict) -> str:
        return "pass" if state.get("is_passing") else "fail"

    graph = Graph(
        nodes={
            "evaluate": evaluate,
            "on_pass": on_pass,
            "on_fail": on_fail,
        },
        edges=[
            # Conditional edge: (source, router_fn, path_map)
            ("evaluate", route_by_result, {"pass": "on_pass", "fail": "on_fail"}),
        ],
        entry="evaluate",
    )

    # Test with passing score
    result_pass = graph.run({"score": 85})
    print(f"\nScore: 85 -> {result_pass.get('result')}")

    # Test with failing score
    result_fail = graph.run({"score": 55})
    print(f"Score: 55 -> {result_fail.get('result')}")


# =============================================================================
# Example 2: Multi-Way Branch
# =============================================================================


class GradeState(TypedDict, total=False):
    """State for grade calculation workflow."""

    score: int
    grade: str
    feedback: str


def example_multi_way_branch() -> None:
    """Branch into multiple paths based on grade."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Way Branch")
    print("=" * 60)

    def calculate_grade(state: GradeState) -> dict:
        """Calculate letter grade from score."""
        score = state.get("score", 0)
        if score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B"
        elif score >= 70:
            grade = "C"
        elif score >= 60:
            grade = "D"
        else:
            grade = "F"
        return {"grade": grade}

    def grade_a_feedback(state: GradeState) -> dict:
        return {"feedback": "Excellent work! Keep it up!"}

    def grade_b_feedback(state: GradeState) -> dict:
        return {"feedback": "Good job! Room for improvement."}

    def grade_c_feedback(state: GradeState) -> dict:
        return {"feedback": "Satisfactory. Consider extra study."}

    def grade_d_feedback(state: GradeState) -> dict:
        return {"feedback": "Needs improvement. Seek help if needed."}

    def grade_f_feedback(state: GradeState) -> dict:
        return {"feedback": "Please schedule a meeting with instructor."}

    def route_by_grade(state: GradeState) -> str:
        return state.get("grade", "F")

    graph = Graph(
        nodes={
            "calculate": calculate_grade,
            "A": grade_a_feedback,
            "B": grade_b_feedback,
            "C": grade_c_feedback,
            "D": grade_d_feedback,
            "F": grade_f_feedback,
        },
        edges=[
            ("calculate", route_by_grade, {"A": "A", "B": "B", "C": "C", "D": "D", "F": "F"}),
        ],
        entry="calculate",
        state_schema=GradeState,
    )

    test_scores = [95, 82, 73, 65, 45]
    print()
    for score in test_scores:
        result = graph.run({"score": score})
        print(f"Score {score}: Grade {result['grade']} - {result['feedback']}")


# =============================================================================
# Example 3: Conditional with Continuation
# =============================================================================


def example_conditional_continuation() -> None:
    """Combine conditional branches with converging paths."""
    print("\n" + "=" * 60)
    print("Example 3: Conditional with Continuation")
    print("=" * 60)

    def classify(state: dict) -> dict:
        """Classify the input type."""
        value = state.get("value", 0)
        return {"type": "positive" if value > 0 else "negative" if value < 0 else "zero"}

    def handle_positive(state: dict) -> dict:
        return {"processed": state["value"] * 2, "handler": "positive"}

    def handle_negative(state: dict) -> dict:
        return {"processed": abs(state["value"]), "handler": "negative"}

    def handle_zero(state: dict) -> dict:
        return {"processed": 1, "handler": "zero"}

    def finalize(state: dict) -> dict:
        """Common finalization for all paths."""
        return {"result": f"Handler: {state['handler']}, Output: {state['processed']}"}

    def route_by_type(state: dict) -> str:
        return state.get("type", "zero")

    graph = Graph(
        nodes={
            "classify": classify,
            "positive": handle_positive,
            "negative": handle_negative,
            "zero": handle_zero,
            "finalize": finalize,
        },
        edges=[
            # Conditional branching
            (
                "classify",
                route_by_type,
                {"positive": "positive", "negative": "negative", "zero": "zero"},
            ),
            # All paths converge to finalize
            ("positive", "finalize"),
            ("negative", "finalize"),
            ("zero", "finalize"),
        ],
        entry="classify",
    )

    print()
    for value in [10, -5, 0]:
        result = graph.run({"value": value})
        print(f"Value {value:3d}: {result['result']}")


# =============================================================================
# Example 4: Chained Conditions
# =============================================================================


def example_chained_conditions() -> None:
    """Multiple conditional checks in sequence."""
    print("\n" + "=" * 60)
    print("Example 4: Chained Conditions")
    print("=" * 60)

    def check_auth(state: dict) -> dict:
        """Check if user is authenticated."""
        return {"authenticated": state.get("token") is not None}

    def check_permissions(state: dict) -> dict:
        """Check if user has required permissions."""
        return {"authorized": state.get("role") in ("admin", "editor")}

    def access_granted(state: dict) -> dict:
        return {"status": "access_granted", "message": "Welcome!"}

    def auth_failed(state: dict) -> dict:
        return {"status": "auth_failed", "message": "Please log in."}

    def permission_denied(state: dict) -> dict:
        return {"status": "permission_denied", "message": "Insufficient permissions."}

    def route_auth(state: dict) -> str:
        return "authenticated" if state.get("authenticated") else "not_authenticated"

    def route_permissions(state: dict) -> str:
        return "authorized" if state.get("authorized") else "not_authorized"

    graph = Graph(
        nodes={
            "check_auth": check_auth,
            "check_permissions": check_permissions,
            "access_granted": access_granted,
            "auth_failed": auth_failed,
            "permission_denied": permission_denied,
        },
        edges=[
            # First check: authentication
            (
                "check_auth",
                route_auth,
                {
                    "authenticated": "check_permissions",
                    "not_authenticated": "auth_failed",
                },
            ),
            # Second check: permissions (only if authenticated)
            (
                "check_permissions",
                route_permissions,
                {
                    "authorized": "access_granted",
                    "not_authorized": "permission_denied",
                },
            ),
        ],
        entry="check_auth",
    )

    print()
    test_cases = [
        {"token": None, "role": "user"},  # Not authenticated
        {"token": "abc123", "role": "user"},  # Authenticated but not authorized
        {"token": "abc123", "role": "admin"},  # Full access
    ]

    for case in test_cases:
        result = graph.run(case)
        print(f"Token: {case['token']}, Role: {case['role']}")
        print(f"  -> {result['status']}: {result['message']}\n")


# =============================================================================
# Example 5: Loop with Condition
# =============================================================================


def example_loop_with_condition() -> None:
    """Create a loop that continues until a condition is met."""
    print("\n" + "=" * 60)
    print("Example 5: Loop with Condition")
    print("=" * 60)

    def process_step(state: dict) -> dict:
        """Process one iteration."""
        current = state.get("counter", 0)
        iterations = state.get("iterations", [])
        return {
            "counter": current + 1,
            "iterations": iterations + [current + 1],
        }

    def check_done(state: dict) -> dict:
        """Check if we should stop."""
        return {"done": state.get("counter", 0) >= state.get("max_iterations", 5)}

    def finalize(state: dict) -> dict:
        """Finalize after loop completes."""
        return {"result": f"Completed {state['counter']} iterations"}

    def route_loop(state: dict) -> str:
        return "done" if state.get("done") else "continue"

    graph = Graph(
        nodes={
            "process": process_step,
            "check": check_done,
            "finalize": finalize,
        },
        edges=[
            ("process", "check"),
            (
                "check",
                route_loop,
                {
                    "continue": "process",  # Loop back
                    "done": "finalize",  # Exit loop
                },
            ),
        ],
        entry="process",
    )

    result = graph.run({"max_iterations": 3})
    print("\nMax iterations: 3")
    print(f"Iterations: {result['iterations']}")
    print(f"Result: {result['result']}")


# =============================================================================
# Example 6: Random Routing (Stochastic Workflow)
# =============================================================================


def example_random_routing() -> None:
    """Use randomness in routing decisions."""
    print("\n" + "=" * 60)
    print("Example 6: Random Routing (Stochastic)")
    print("=" * 60)

    def start(state: dict) -> dict:
        return {"path": None}

    def path_a(state: dict) -> dict:
        return {"path": "A", "message": "You went down path A!"}

    def path_b(state: dict) -> dict:
        return {"path": "B", "message": "You went down path B!"}

    def path_c(state: dict) -> dict:
        return {"path": "C", "message": "You went down path C!"}

    def random_route(state: dict) -> str:
        """Randomly choose a path."""
        return random.choice(["a", "b", "c"])

    graph = Graph(
        nodes={
            "start": start,
            "path_a": path_a,
            "path_b": path_b,
            "path_c": path_c,
        },
        edges=[
            ("start", random_route, {"a": "path_a", "b": "path_b", "c": "path_c"}),
        ],
        entry="start",
    )

    print("\nRunning 5 times with random routing:\n")
    for i in range(5):
        result = graph.run({})
        print(f"  Run {i + 1}: {result['message']}")


# =============================================================================
# Example 7: Conditional Edge Analysis
# =============================================================================


def example_conditional_analysis() -> None:
    """Analyze graphs with conditional edges."""
    print("\n" + "=" * 60)
    print("Example 7: Conditional Edge Analysis")
    print("=" * 60)

    def start(state: dict) -> dict:
        return {}

    def option_a(state: dict) -> dict:
        return {"chose": "A"}

    def option_b(state: dict) -> dict:
        return {"chose": "B"}

    def option_c(state: dict) -> dict:
        return {"chose": "C"}

    def router(state: dict) -> str:
        return state.get("choice", "a")

    graph = Graph(
        nodes={
            "start": start,
            "option_a": option_a,
            "option_b": option_b,
            "option_c": option_c,
        },
        edges=[
            ("start", router, {"a": "option_a", "b": "option_b", "c": "option_c"}),
        ],
        entry="start",
    )

    # Analyze structure
    structure = graph.analyze()

    print("\n--- Graph Analysis ---\n")
    print(f"Nodes: {structure.nodes}")
    print(f"Regular edges: {structure.edges}")
    print(f"Conditional edge count: {structure.conditional_edge_count}")

    if structure.conditional_edges:
        print("\nConditional edges:")
        for edge in structure.conditional_edges:
            print(f"  From: {edge['start']}")
            print(f"  Router: {edge['router_function']}")
            print(f"  Options: {edge['path_options']}")

    # Mermaid diagram
    print("\n--- Mermaid Diagram ---\n")
    print(graph.get_arch_diagram())


# =============================================================================
# Example 8: Async Conditional Workflow
# =============================================================================


async def example_async_conditional() -> None:
    """Async workflow with conditional routing."""
    print("\n" + "=" * 60)
    print("Example 8: Async Conditional Workflow")
    print("=" * 60)

    async def fetch_user(state: dict) -> dict:
        """Fetch user data."""
        await asyncio.sleep(0.1)
        # Simulate finding/not finding user
        user_id = state.get("user_id", "")
        found = user_id.startswith("valid_")
        return {"user_found": found, "user_data": {"id": user_id} if found else None}

    async def process_user(state: dict) -> dict:
        """Process existing user."""
        await asyncio.sleep(0.05)
        return {"action": "updated", "user": state["user_data"]}

    async def create_user(state: dict) -> dict:
        """Create new user."""
        await asyncio.sleep(0.05)
        return {"action": "created", "user": {"id": state.get("user_id")}}

    def route_user(state: dict) -> str:
        return "exists" if state.get("user_found") else "new"

    graph = Graph(
        nodes={
            "fetch": fetch_user,
            "process": process_user,
            "create": create_user,
        },
        edges=[
            ("fetch", route_user, {"exists": "process", "new": "create"}),
        ],
        entry="fetch",
    )

    print()
    for user_id in ["valid_user123", "unknown_user"]:
        result = await graph.arun({"user_id": user_id})
        print(f"User '{user_id}': {result['action']} -> {result['user']}")


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run all examples."""
    print("=" * 60)
    print("Conditional Graph Workflow Examples")
    print("Dynamic Branching and Routing")
    print("=" * 60)

    # Sync examples
    example_basic_conditional()
    example_multi_way_branch()
    example_conditional_continuation()
    example_chained_conditions()
    example_loop_with_condition()
    example_random_routing()
    example_conditional_analysis()

    # Async example
    await example_async_conditional()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
