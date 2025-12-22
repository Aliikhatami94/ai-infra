#!/usr/bin/env python
"""Graph with Agent: Combining Workflows and AI Agents.

This example demonstrates:
- Using Agent as a node in a Graph workflow
- Coordinating multiple agents in a workflow
- Agent tools within graph nodes
- Human-in-the-loop with agents
- Hybrid workflows (deterministic + AI)

This pattern lets you combine the predictability of state machines
with the flexibility of AI agents for complex workflows.

Required API Keys (at least one):
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GOOGLE_API_KEY

Key concepts:
- Agent as node: Wrap agent.run/arun in a graph node
- Tool results: Pass tool results through state
- Agent handoff: One agent's output feeds another
- Hybrid logic: Mix deterministic code with AI
"""

import asyncio
from typing import TypedDict

from ai_infra import Agent, Graph

# =============================================================================
# Example 1: Simple Agent Node
# =============================================================================


class SimpleState(TypedDict, total=False):
    """State for simple agent workflow."""

    input: str
    analysis: str
    final_output: str


async def example_simple_agent_node() -> None:
    """Use a single agent as a graph node."""
    print("\n" + "=" * 60)
    print("Example 1: Simple Agent Node")
    print("=" * 60)

    # Create an agent
    analyzer = Agent(
        system_prompt="You are a text analyzer. Analyze the input briefly.",
    )

    # Wrap agent in a node function
    async def analyze_node(state: SimpleState) -> dict:
        """Run agent analysis on input."""
        response = await analyzer.arun(f"Analyze this: {state['input']}")
        return {"analysis": response}

    def format_output(state: SimpleState) -> dict:
        """Format the final output."""
        return {"final_output": f"Analysis Complete:\n{state['analysis']}"}

    graph = Graph(
        nodes={
            "analyze": analyze_node,
            "format": format_output,
        },
        edges=[
            ("analyze", "format"),
        ],
        entry="analyze",
        state_schema=SimpleState,
    )

    print("\n--- Running Agent Node ---\n")

    try:
        result = await graph.arun({"input": "The quick brown fox jumps."})
        print("Input: 'The quick brown fox jumps.'")
        print(f"\n{result.get('final_output', 'No output')}")
    except Exception as e:
        print(f"Error (likely missing API key): {e}")


# =============================================================================
# Example 2: Multi-Agent Pipeline
# =============================================================================


class PipelineState(TypedDict, total=False):
    """State for multi-agent pipeline."""

    topic: str
    research: str
    draft: str
    review: str
    final: str


async def example_multi_agent_pipeline() -> None:
    """Chain multiple agents in a workflow."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Agent Pipeline")
    print("=" * 60)

    # Create specialized agents
    researcher = Agent(
        system_prompt="""You are a research assistant.
Given a topic, provide 2-3 key facts in bullet points.
Be brief and factual.""",
    )

    writer = Agent(
        system_prompt="""You are a technical writer.
Given research notes, write a brief paragraph.
Be clear and concise.""",
    )

    reviewer = Agent(
        system_prompt="""You are an editor.
Review the draft and suggest one improvement.
Be constructive and brief.""",
    )

    # Node functions
    async def research_node(state: PipelineState) -> dict:
        response = await researcher.arun(f"Research topic: {state['topic']}")
        return {"research": response}

    async def write_node(state: PipelineState) -> dict:
        response = await writer.arun(f"Write about:\n{state['research']}")
        return {"draft": response}

    async def review_node(state: PipelineState) -> dict:
        response = await reviewer.arun(f"Review this:\n{state['draft']}")
        return {"review": response}

    def finalize_node(state: PipelineState) -> dict:
        return {
            "final": f"## Research\n{state['research']}\n\n"
            f"## Draft\n{state['draft']}\n\n"
            f"## Review\n{state['review']}"
        }

    graph = Graph(
        nodes={
            "research": research_node,
            "write": write_node,
            "review": review_node,
            "finalize": finalize_node,
        },
        edges=[
            ("research", "write"),
            ("write", "review"),
            ("review", "finalize"),
        ],
        entry="research",
        state_schema=PipelineState,
    )

    print("\n--- Multi-Agent Pipeline ---\n")

    try:
        result = await graph.arun({"topic": "Python async programming"})
        print(result.get("final", "No output"))
    except Exception as e:
        print(f"Error (likely missing API key): {e}")


# =============================================================================
# Example 3: Agent with Conditional Routing
# =============================================================================


class ClassifyState(TypedDict, total=False):
    """State for classification workflow."""

    text: str
    category: str
    sentiment: str
    response: str


async def example_agent_conditional() -> None:
    """Use agent output to drive conditional routing."""
    print("\n" + "=" * 60)
    print("Example 3: Agent with Conditional Routing")
    print("=" * 60)

    classifier = Agent(
        system_prompt="""You are a text classifier.
Classify the input into one of: QUESTION, COMPLAINT, PRAISE
Respond with just the category name.""",
    )

    # Classify input
    async def classify_node(state: ClassifyState) -> dict:
        response = await classifier.arun(f"Classify: {state['text']}")
        # Normalize the response
        category = response.strip().upper()
        if "QUESTION" in category:
            category = "question"
        elif "COMPLAINT" in category:
            category = "complaint"
        elif "PRAISE" in category:
            category = "praise"
        else:
            category = "other"
        return {"category": category}

    def handle_question(state: ClassifyState) -> dict:
        return {"response": "I'll look up the answer for you!"}

    def handle_complaint(state: ClassifyState) -> dict:
        return {"response": "I'm sorry to hear that. Let me help resolve this."}

    def handle_praise(state: ClassifyState) -> dict:
        return {"response": "Thank you for the kind words!"}

    def handle_other(state: ClassifyState) -> dict:
        return {"response": "Thank you for your message."}

    def route_by_category(state: ClassifyState) -> str:
        return state.get("category", "other")

    graph = Graph(
        nodes={
            "classify": classify_node,
            "question": handle_question,
            "complaint": handle_complaint,
            "praise": handle_praise,
            "other": handle_other,
        },
        edges=[
            (
                "classify",
                route_by_category,
                {
                    "question": "question",
                    "complaint": "complaint",
                    "praise": "praise",
                    "other": "other",
                },
            ),
        ],
        entry="classify",
        state_schema=ClassifyState,
    )

    test_inputs = [
        "How do I reset my password?",
        "This product is terrible!",
        "Great service, thank you!",
    ]

    print("\n--- Classification Results ---\n")

    for text in test_inputs:
        try:
            result = await graph.arun({"text": text})
            print(f"Input: {text}")
            print(f"Category: {result.get('category')}")
            print(f"Response: {result.get('response')}\n")
        except Exception as e:
            print(f"Error: {e}\n")


# =============================================================================
# Example 4: Agent with Tools in Graph
# =============================================================================


def get_weather(city: str) -> str:
    """Get weather for a city (simulated).

    Args:
        city: Name of the city.

    Returns:
        Weather information.
    """
    # Simulated weather data
    weather_data = {
        "tokyo": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "new york": "Rainy, 18°C",
    }
    return weather_data.get(city.lower(), f"Weather data for {city} not available")


def get_time(city: str) -> str:
    """Get current time in a city (simulated).

    Args:
        city: Name of the city.

    Returns:
        Current time.
    """
    # Simulated time data
    time_data = {
        "tokyo": "10:00 AM JST",
        "london": "1:00 AM GMT",
        "new york": "8:00 PM EST",
    }
    return time_data.get(city.lower(), f"Time for {city} not available")


class ToolState(TypedDict, total=False):
    """State for tool-using agent workflow."""

    query: str
    tool_results: str
    summary: str


async def example_agent_with_tools() -> None:
    """Use agent with tools as a graph node."""
    print("\n" + "=" * 60)
    print("Example 4: Agent with Tools in Graph")
    print("=" * 60)

    # Agent with tools
    info_agent = Agent(
        system_prompt="""You are a travel assistant.
Use the available tools to answer questions about cities.
Always use tools when asked about weather or time.""",
        tools=[get_weather, get_time],
    )

    async def gather_info(state: ToolState) -> dict:
        """Use agent with tools to gather information."""
        response = await info_agent.arun(state["query"])
        return {"tool_results": response}

    def create_summary(state: ToolState) -> dict:
        """Create a formatted summary."""
        return {"summary": f"Travel Info:\n{state['tool_results']}"}

    graph = Graph(
        nodes={
            "gather": gather_info,
            "summarize": create_summary,
        },
        edges=[
            ("gather", "summarize"),
        ],
        entry="gather",
        state_schema=ToolState,
    )

    print("\n--- Agent with Tools ---\n")

    try:
        result = await graph.arun({"query": "What's the weather and time in Tokyo?"})
        print(result.get("summary", "No output"))
    except Exception as e:
        print(f"Error (likely missing API key): {e}")


# =============================================================================
# Example 5: Hybrid Deterministic + AI Workflow
# =============================================================================


class HybridState(TypedDict, total=False):
    """State for hybrid workflow."""

    raw_data: str
    parsed: dict
    enriched: dict
    ai_analysis: str
    report: str


async def example_hybrid_workflow() -> None:
    """Combine deterministic processing with AI analysis."""
    print("\n" + "=" * 60)
    print("Example 5: Hybrid Deterministic + AI Workflow")
    print("=" * 60)

    analyst = Agent(
        system_prompt="""You are a data analyst.
Given structured data, provide a brief insight.
Be concise (1-2 sentences).""",
    )

    # Deterministic node
    def parse_data(state: HybridState) -> dict:
        """Parse raw data (deterministic)."""
        parts = state["raw_data"].split(",")
        parsed = {
            "name": parts[0] if len(parts) > 0 else "Unknown",
            "value": int(parts[1]) if len(parts) > 1 else 0,
            "status": parts[2] if len(parts) > 2 else "unknown",
        }
        return {"parsed": parsed}

    # Deterministic node
    def enrich_data(state: HybridState) -> dict:
        """Enrich parsed data with computed fields."""
        parsed = state["parsed"]
        enriched = {
            **parsed,
            "is_high_value": parsed["value"] > 100,
            "is_active": parsed["status"] == "active",
        }
        return {"enriched": enriched}

    # AI node
    async def analyze_with_ai(state: HybridState) -> dict:
        """Use AI to analyze the enriched data."""
        data_str = str(state["enriched"])
        response = await analyst.arun(f"Analyze this data: {data_str}")
        return {"ai_analysis": response}

    # Deterministic node
    def generate_report(state: HybridState) -> dict:
        """Generate final report."""
        report = (
            f"## Data Report\n"
            f"Name: {state['enriched']['name']}\n"
            f"Value: {state['enriched']['value']}\n"
            f"High Value: {state['enriched']['is_high_value']}\n"
            f"Active: {state['enriched']['is_active']}\n\n"
            f"## AI Analysis\n{state['ai_analysis']}"
        )
        return {"report": report}

    graph = Graph(
        nodes={
            "parse": parse_data,
            "enrich": enrich_data,
            "analyze": analyze_with_ai,
            "report": generate_report,
        },
        edges=[
            ("parse", "enrich"),
            ("enrich", "analyze"),
            ("analyze", "report"),
        ],
        entry="parse",
        state_schema=HybridState,
    )

    print("\n--- Hybrid Workflow ---\n")

    try:
        result = await graph.arun({"raw_data": "ProductX,150,active"})
        print(result.get("report", "No output"))
    except Exception as e:
        print(f"Error (likely missing API key): {e}")


# =============================================================================
# Example 6: Agent Loop with Exit Condition
# =============================================================================


class LoopState(TypedDict, total=False):
    """State for agent loop workflow."""

    goal: str
    attempts: int
    max_attempts: int
    history: list[str]
    success: bool
    final_answer: str


async def example_agent_loop() -> None:
    """Agent that retries until success or max attempts."""
    print("\n" + "=" * 60)
    print("Example 6: Agent Loop with Exit Condition")
    print("=" * 60)

    solver = Agent(
        system_prompt="""You are a problem solver.
Given a goal, try to solve it.
If you can solve it, respond with "SOLVED: [answer]"
If you can't, respond with "RETRY: [why]"
Be brief.""",
    )

    async def attempt_solve(state: LoopState) -> dict:
        """Make one attempt to solve."""
        attempts = state.get("attempts", 0) + 1
        history = state.get("history", [])

        history_context = "\n".join(history) if history else "No previous attempts"

        response = await solver.arun(
            f"Goal: {state['goal']}\nAttempt: {attempts}\nPrevious: {history_context}"
        )

        history.append(f"Attempt {attempts}: {response}")

        success = "SOLVED:" in response.upper()
        final = response if success else None

        return {
            "attempts": attempts,
            "history": history,
            "success": success,
            "final_answer": final,
        }

    def check_complete(state: LoopState) -> dict:
        """Check if we should stop."""
        return {}  # Just pass through to routing

    def finalize_success(state: LoopState) -> dict:
        return {"final_answer": f"Success! {state['final_answer']}"}

    def finalize_failure(state: LoopState) -> dict:
        return {"final_answer": f"Failed after {state['attempts']} attempts"}

    def route_result(state: LoopState) -> str:
        if state.get("success"):
            return "success"
        if state.get("attempts", 0) >= state.get("max_attempts", 3):
            return "max_reached"
        return "retry"

    graph = Graph(
        nodes={
            "attempt": attempt_solve,
            "check": check_complete,
            "success": finalize_success,
            "failure": finalize_failure,
        },
        edges=[
            ("attempt", "check"),
            (
                "check",
                route_result,
                {
                    "success": "success",
                    "retry": "attempt",
                    "max_reached": "failure",
                },
            ),
        ],
        entry="attempt",
        state_schema=LoopState,
    )

    print("\n--- Agent Loop ---\n")

    try:
        result = await graph.arun(
            {
                "goal": "Calculate 2 + 2",
                "max_attempts": 3,
            }
        )
        print("Goal: Calculate 2 + 2")
        print(f"Attempts: {result.get('attempts')}")
        print(f"Result: {result.get('final_answer')}")

        if result.get("history"):
            print("\nHistory:")
            for h in result["history"]:
                print(f"  {h}")
    except Exception as e:
        print(f"Error (likely missing API key): {e}")


# =============================================================================
# Example 7: Mock Agent for Testing
# =============================================================================


class MockAgent:
    """Mock agent for testing graph integration."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0

    async def arun(self, message: str) -> str:
        """Return next mock response."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


async def example_mock_agent() -> None:
    """Test graph with mock agent (no API key needed)."""
    print("\n" + "=" * 60)
    print("Example 7: Mock Agent for Testing")
    print("=" * 60)

    # Create mock agent with predetermined responses
    mock = MockAgent(
        [
            "Analysis: The text is positive.",
            "Summary: All good!",
        ]
    )

    async def analyze(state: dict) -> dict:
        response = await mock.arun(state["input"])
        return {"analysis": response}

    async def summarize(state: dict) -> dict:
        response = await mock.arun(state["analysis"])
        return {"summary": response}

    graph = Graph(
        nodes={
            "analyze": analyze,
            "summarize": summarize,
        },
        edges=[
            ("analyze", "summarize"),
        ],
        entry="analyze",
    )

    print("\n--- Mock Agent Test ---\n")

    result = await graph.arun({"input": "Test input"})

    print("Input: Test input")
    print(f"Analysis: {result.get('analysis')}")
    print(f"Summary: {result.get('summary')}")
    print(f"\nMock agent called {mock.call_count} times")


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run all examples."""
    print("=" * 60)
    print("Graph with Agent Examples")
    print("Combining Workflows and AI Agents")
    print("=" * 60)

    # Mock example (always works)
    await example_mock_agent()

    # Real agent examples (require API key)
    print("\n" + "=" * 60)
    print("Running examples with real agents...")
    print("(Requires API key - will skip on error)")
    print("=" * 60)

    try:
        await example_simple_agent_node()
    except Exception as e:
        print(f"Skipped (error: {e})")

    try:
        await example_agent_conditional()
    except Exception as e:
        print(f"Skipped (error: {e})")

    try:
        await example_agent_with_tools()
    except Exception as e:
        print(f"Skipped (error: {e})")

    try:
        await example_hybrid_workflow()
    except Exception as e:
        print(f"Skipped (error: {e})")

    try:
        await example_multi_agent_pipeline()
    except Exception as e:
        print(f"Skipped (error: {e})")

    try:
        await example_agent_loop()
    except Exception as e:
        print(f"Skipped (error: {e})")

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
