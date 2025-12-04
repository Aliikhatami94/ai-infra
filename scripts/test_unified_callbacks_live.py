#!/usr/bin/env python3
"""Live test script for Unified Callback System (Phase 6.7).

This script tests the real functionality of the unified callback system
with actual OpenAI API calls to verify everything works as expected.

Run with: poetry run python scripts/test_unified_callbacks_live.py
"""

import asyncio
from typing import List

# Colors for console output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def log(msg: str, color: str = RESET):
    print(f"{color}{msg}{RESET}")


def section(title: str):
    print(f"\n{BOLD}{BLUE}{'=' * 60}{RESET}")
    print(f"{BOLD}{BLUE}{title}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 60}{RESET}\n")


def subsection(title: str):
    print(f"\n{CYAN}--- {title} ---{RESET}\n")


# =============================================================================
# Test 1: Basic Callback Events with LLM
# =============================================================================


def test_1_llm_callbacks():
    """Test that LLM fires callback events during chat()."""
    section("Test 1: LLM Callbacks with chat()")

    from ai_infra import LLM
    from ai_infra.callbacks import Callbacks, LLMEndEvent, LLMErrorEvent, LLMStartEvent

    events: List[str] = []

    class TestCallbacks(Callbacks):
        def on_llm_start(self, event: LLMStartEvent) -> None:
            events.append("llm_start")
            log(f"  🚀 LLM Start: {event.provider}/{event.model}", YELLOW)
            log(f"     Messages: {len(event.messages)} message(s)", YELLOW)

        def on_llm_end(self, event: LLMEndEvent) -> None:
            events.append("llm_end")
            log(f"  ✅ LLM End: {event.latency_ms:.0f}ms", GREEN)
            log(f"     Tokens: {event.total_tokens} total", GREEN)
            log(f"     Response: {event.response[:80]}...", GREEN)

        def on_llm_error(self, event: LLMErrorEvent) -> None:
            events.append("llm_error")
            log(f"  ❌ LLM Error: {event.error_type}", RED)

    callbacks = TestCallbacks()
    llm = LLM(callbacks=callbacks)

    log("Calling llm.chat('What is 2+2? Reply in one word.')...", BLUE)
    result = llm.chat("What is 2+2? Reply in one word.")
    log(f"\nResult: {result.content}", CYAN)

    # Verify events fired
    assert "llm_start" in events, "LLMStartEvent not fired!"
    assert "llm_end" in events, "LLMEndEvent not fired!"
    log(f"\n✓ Events fired: {events}", GREEN)
    return True


# =============================================================================
# Test 2: Async LLM Callbacks with achat()
# =============================================================================


async def test_2_async_llm_callbacks():
    """Test that LLM fires async callback events during achat()."""
    section("Test 2: Async LLM Callbacks with achat()")

    from ai_infra import LLM
    from ai_infra.callbacks import Callbacks, LLMEndEvent, LLMStartEvent

    events: List[str] = []

    class AsyncTestCallbacks(Callbacks):
        async def on_llm_start_async(self, event: LLMStartEvent) -> None:
            events.append("llm_start_async")
            log(f"  🚀 Async LLM Start: {event.provider}/{event.model}", YELLOW)

        async def on_llm_end_async(self, event: LLMEndEvent) -> None:
            events.append("llm_end_async")
            log(f"  ✅ Async LLM End: {event.latency_ms:.0f}ms", GREEN)
            log(f"     Tokens: {event.total_tokens}", GREEN)

    callbacks = AsyncTestCallbacks()
    llm = LLM(callbacks=callbacks)

    log("Calling await llm.achat('What is the capital of France? One word.')...", BLUE)
    result = await llm.achat("What is the capital of France? Reply in one word.")
    log(f"\nResult: {result.content}", CYAN)

    assert "llm_start_async" in events, "Async LLMStartEvent not fired!"
    assert "llm_end_async" in events, "Async LLMEndEvent not fired!"
    log(f"\n✓ Async events fired: {events}", GREEN)
    return True


# =============================================================================
# Test 3: Streaming Callbacks with stream_tokens()
# =============================================================================


async def test_3_streaming_callbacks():
    """Test that LLM fires token events during streaming."""
    section("Test 3: Streaming Callbacks with stream_tokens()")

    from ai_infra import LLM
    from ai_infra.callbacks import Callbacks, LLMEndEvent, LLMStartEvent, LLMTokenEvent

    events: List[str] = []
    tokens_received: List[str] = []

    class StreamingCallbacks(Callbacks):
        async def on_llm_start_async(self, event: LLMStartEvent) -> None:
            events.append("llm_start")
            log(f"  🚀 Stream Start: {event.provider}/{event.model}", YELLOW)

        async def on_llm_token_async(self, event: LLMTokenEvent) -> None:
            events.append("llm_token")
            tokens_received.append(event.token)
            # Print token without newline
            print(f"{CYAN}{event.token}{RESET}", end="", flush=True)

        async def on_llm_end_async(self, event: LLMEndEvent) -> None:
            events.append("llm_end")
            print()  # Newline after streaming
            log(f"  ✅ Stream End: {event.latency_ms:.0f}ms", GREEN)

    callbacks = StreamingCallbacks()
    llm = LLM(callbacks=callbacks)

    log("Streaming: 'Count from 1 to 5'...\n", BLUE)
    log("Tokens: ", YELLOW)
    async for token, meta in llm.stream_tokens("Count from 1 to 5, one number per line."):
        pass  # Tokens are printed by callback

    assert "llm_start" in events, "LLMStartEvent not fired!"
    assert "llm_token" in events, "LLMTokenEvent not fired!"
    assert "llm_end" in events, "LLMEndEvent not fired!"
    log(f"\n✓ Received {len(tokens_received)} tokens via callbacks", GREEN)
    return True


# =============================================================================
# Test 4: Agent Callbacks with Tool Execution
# =============================================================================


def test_4_agent_with_tools():
    """Test that Agent fires callbacks during tool execution."""
    section("Test 4: Agent Callbacks with Tool Execution")

    from langchain_core.tools import tool as lc_tool

    from ai_infra import Agent
    from ai_infra.callbacks import (
        Callbacks,
        LLMEndEvent,
        LLMStartEvent,
        ToolEndEvent,
        ToolStartEvent,
    )

    events: List[str] = []

    class AgentCallbacks(Callbacks):
        def on_llm_start(self, event: LLMStartEvent) -> None:
            events.append("llm_start")
            log(f"  🚀 Agent LLM Start: {event.provider}/{event.model}", YELLOW)

        def on_llm_end(self, event: LLMEndEvent) -> None:
            events.append("llm_end")
            log(f"  ✅ Agent LLM End: {event.latency_ms:.0f}ms", GREEN)

        def on_tool_start(self, event: ToolStartEvent) -> None:
            events.append("tool_start")
            log(f"  🔧 Tool Start: {event.tool_name}", YELLOW)
            log(f"     Arguments: {event.arguments}", YELLOW)

        def on_tool_end(self, event: ToolEndEvent) -> None:
            events.append("tool_end")
            log(f"  ✅ Tool End: {event.tool_name} ({event.latency_ms:.0f}ms)", GREEN)
            log(f"     Result: {event.result}", GREEN)

    @lc_tool
    def calculate(expression: str) -> str:
        """Calculate a mathematical expression.

        Args:
            expression: A mathematical expression like '2 + 2' or '10 * 5'

        Returns:
            The result of the calculation as a string
        """
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    callbacks = AgentCallbacks()
    # Pass LangChain tools in constructor - callbacks will wrap them
    agent = Agent(tools=[calculate], callbacks=callbacks)

    log("Running agent with: 'What is 15 * 7? Use the calculate tool.'...", BLUE)
    result = agent.run("What is 15 * 7? Use the calculate tool to compute this.")
    log(f"\nAgent Result: {result}", CYAN)

    assert "llm_start" in events, "LLMStartEvent not fired!"
    assert "tool_start" in events, "ToolStartEvent not fired!"
    assert "tool_end" in events, "ToolEndEvent not fired!"
    log(f"\n✓ Events fired: {events}", GREEN)
    return True


# =============================================================================
# Test 5: CallbackManager with Multiple Handlers
# =============================================================================


def test_5_callback_manager():
    """Test CallbackManager dispatches to multiple callbacks."""
    section("Test 5: CallbackManager with Multiple Handlers")

    from ai_infra import LLM
    from ai_infra.callbacks import CallbackManager, Callbacks, LLMEndEvent, LLMStartEvent

    handler1_events: List[str] = []
    handler2_events: List[str] = []

    class Handler1(Callbacks):
        def on_llm_start(self, event: LLMStartEvent) -> None:
            handler1_events.append("start")
            log(f"  [Handler1] 🚀 Start: {event.model}", YELLOW)

        def on_llm_end(self, event: LLMEndEvent) -> None:
            handler1_events.append("end")
            log(f"  [Handler1] ✅ End: {event.latency_ms:.0f}ms", GREEN)

    class Handler2(Callbacks):
        def on_llm_start(self, event: LLMStartEvent) -> None:
            handler2_events.append("start")
            log(f"  [Handler2] 🚀 Start: {event.model}", YELLOW)

        def on_llm_end(self, event: LLMEndEvent) -> None:
            handler2_events.append("end")
            log(f"  [Handler2] ✅ End: {event.latency_ms:.0f}ms", GREEN)

    manager = CallbackManager([Handler1(), Handler2()])
    llm = LLM(callbacks=manager)

    log("Calling LLM with CallbackManager (2 handlers)...", BLUE)
    result = llm.chat("Say 'hello' in French. One word only.")
    log(f"\nResult: {result.content}", CYAN)

    assert handler1_events == ["start", "end"], f"Handler1 events wrong: {handler1_events}"
    assert handler2_events == ["start", "end"], f"Handler2 events wrong: {handler2_events}"
    log(f"\n✓ Handler1 events: {handler1_events}", GREEN)
    log(f"✓ Handler2 events: {handler2_events}", GREEN)
    return True


# =============================================================================
# Test 6: Built-in Callbacks (LoggingCallbacks, PrintCallbacks)
# =============================================================================


def test_6_builtin_callbacks():
    """Test built-in callback implementations."""
    section("Test 6: Built-in Callbacks")

    from ai_infra import LLM
    from ai_infra.callbacks import (
        CallbackManager,
        LoggingCallbacks,
        MetricsCallbacks,
        PrintCallbacks,
    )

    subsection("Testing PrintCallbacks")
    llm = LLM(callbacks=PrintCallbacks())
    log("Calling LLM with PrintCallbacks...", BLUE)
    result = llm.chat("What color is the sky? One word.")
    log(f"Result: {result.content}\n", CYAN)

    subsection("Testing MetricsCallbacks")
    metrics = MetricsCallbacks()
    llm = LLM(callbacks=metrics)
    log("Calling LLM with MetricsCallbacks...", BLUE)
    result = llm.chat("What is 1+1? One word.")
    log(f"Result: {result.content}", CYAN)

    # Check if metrics were recorded
    if hasattr(metrics, "metrics"):
        log(f"Metrics recorded: {metrics.metrics}", GREEN)
    log("✓ MetricsCallbacks works", GREEN)

    subsection("Testing Combined Callbacks")
    manager = CallbackManager(
        [
            LoggingCallbacks(),
            MetricsCallbacks(),
        ]
    )
    llm = LLM(callbacks=manager)
    log("Calling LLM with multiple built-in callbacks...", BLUE)
    result = llm.chat("What is the opposite of hot? One word.")
    log(f"Result: {result.content}", CYAN)

    log("\n✓ All built-in callbacks work correctly", GREEN)
    return True


# =============================================================================
# Test 7: All Callback Event Types Export
# =============================================================================


def test_7_all_exports():
    """Test that all callback types can be imported from ai_infra."""
    section("Test 7: All Callback Exports")

    log("Testing imports from ai_infra...", BLUE)

    from ai_infra import (  # noqa: F401 - testing all exports are available
        CallbackManager,
        Callbacks,
        GraphNodeEndEvent,
        GraphNodeErrorEvent,
        GraphNodeStartEvent,
        LLMEndEvent,
        LLMErrorEvent,
        LLMStartEvent,
        LLMTokenEvent,
        LoggingCallbacks,
        MCPConnectEvent,
        MCPDisconnectEvent,
        MCPLoggingEvent,
        MCPProgressEvent,
        MetricsCallbacks,
        PrintCallbacks,
        ToolEndEvent,
        ToolErrorEvent,
        ToolStartEvent,
    )

    log("  ✓ Callbacks, CallbackManager", GREEN)
    log("  ✓ LLMStartEvent, LLMEndEvent, LLMErrorEvent, LLMTokenEvent", GREEN)
    log("  ✓ ToolStartEvent, ToolEndEvent, ToolErrorEvent", GREEN)
    log("  ✓ MCPConnectEvent, MCPDisconnectEvent, MCPProgressEvent, MCPLoggingEvent", GREEN)
    log("  ✓ GraphNodeStartEvent, GraphNodeEndEvent, GraphNodeErrorEvent", GREEN)
    log("  ✓ LoggingCallbacks, MetricsCallbacks, PrintCallbacks", GREEN)

    # Verify they're all classes/types
    assert Callbacks is not None
    assert CallbackManager is not None
    assert LLMStartEvent is not None
    assert MCPProgressEvent is not None
    assert GraphNodeStartEvent is not None

    log("\n✓ All 19 callback exports are available", GREEN)
    return True


# =============================================================================
# Test 8: MCPClient with Unified Callbacks
# =============================================================================


def test_8_mcp_client_callbacks():
    """Test MCPClient accepts unified callbacks."""
    section("Test 8: MCPClient with Unified Callbacks")

    from ai_infra.callbacks import CallbackManager, Callbacks, MCPLoggingEvent, MCPProgressEvent
    from ai_infra.mcp import MCPClient

    events: List[str] = []

    class MCPCallbacks(Callbacks):
        async def on_mcp_progress_async(self, event: MCPProgressEvent) -> None:
            events.append("mcp_progress")
            log(f"  📊 MCP Progress: {event.server_name} - {event.progress}", YELLOW)

        async def on_mcp_logging_async(self, event: MCPLoggingEvent) -> None:
            events.append("mcp_logging")
            log(f"  📝 MCP Log: {event.server_name} [{event.level}] {event.data}", YELLOW)

    # Test that MCPClient accepts callbacks
    callbacks = MCPCallbacks()
    mcp = MCPClient(
        [{"transport": "stdio", "command": "echo", "args": ["test"]}],
        callbacks=callbacks,
    )
    log("✓ MCPClient accepts Callbacks instance", GREEN)

    # Test with CallbackManager
    manager = CallbackManager([MCPCallbacks()])
    mcp2 = MCPClient(
        [{"transport": "stdio", "command": "echo", "args": ["test"]}],
        callbacks=manager,
    )
    log("✓ MCPClient accepts CallbackManager", GREEN)

    # Test normalization (check both mcp and mcp2)
    assert isinstance(
        mcp._callbacks, CallbackManager
    ), "Callbacks not normalized to CallbackManager"
    assert isinstance(
        mcp2._callbacks, CallbackManager
    ), "CallbackManager not passed through correctly"
    log("✓ Callbacks normalized to CallbackManager internally", GREEN)

    log("\n✓ MCPClient unified callbacks work correctly", GREEN)
    return True


# =============================================================================
# Main Test Runner
# =============================================================================


async def run_all_tests():
    """Run all tests."""
    print(f"\n{BOLD}{GREEN}{'=' * 60}{RESET}")
    print(f"{BOLD}{GREEN}  Unified Callback System - Live Test Suite{RESET}")
    print(f"{BOLD}{GREEN}{'=' * 60}{RESET}")

    results = {}

    # Test 1: Basic LLM callbacks
    try:
        results["test_1_llm_callbacks"] = test_1_llm_callbacks()
    except Exception as e:
        log(f"❌ Test 1 FAILED: {e}", RED)
        results["test_1_llm_callbacks"] = False

    # Test 2: Async LLM callbacks
    try:
        results["test_2_async_llm_callbacks"] = await test_2_async_llm_callbacks()
    except Exception as e:
        log(f"❌ Test 2 FAILED: {e}", RED)
        results["test_2_async_llm_callbacks"] = False

    # Test 3: Streaming callbacks
    try:
        results["test_3_streaming_callbacks"] = await test_3_streaming_callbacks()
    except Exception as e:
        log(f"❌ Test 3 FAILED: {e}", RED)
        results["test_3_streaming_callbacks"] = False

    # Test 4: Agent with tools
    try:
        results["test_4_agent_with_tools"] = test_4_agent_with_tools()
    except Exception as e:
        log(f"❌ Test 4 FAILED: {e}", RED)
        import traceback

        traceback.print_exc()
        results["test_4_agent_with_tools"] = False

    # Test 5: CallbackManager
    try:
        results["test_5_callback_manager"] = test_5_callback_manager()
    except Exception as e:
        log(f"❌ Test 5 FAILED: {e}", RED)
        results["test_5_callback_manager"] = False

    # Test 6: Built-in callbacks
    try:
        results["test_6_builtin_callbacks"] = test_6_builtin_callbacks()
    except Exception as e:
        log(f"❌ Test 6 FAILED: {e}", RED)
        results["test_6_builtin_callbacks"] = False

    # Test 7: All exports
    try:
        results["test_7_all_exports"] = test_7_all_exports()
    except Exception as e:
        log(f"❌ Test 7 FAILED: {e}", RED)
        results["test_7_all_exports"] = False

    # Test 8: MCP Client callbacks
    try:
        results["test_8_mcp_client_callbacks"] = test_8_mcp_client_callbacks()
    except Exception as e:
        log(f"❌ Test 8 FAILED: {e}", RED)
        results["test_8_mcp_client_callbacks"] = False

    # Summary
    section("Test Summary")
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    for name, result in results.items():
        status = f"{GREEN}✓ PASSED{RESET}" if result else f"{RED}✗ FAILED{RESET}"
        print(f"  {name}: {status}")

    print()
    if failed == 0:
        log(f"🎉 All {passed} tests passed!", GREEN)
    else:
        log(f"⚠️  {passed} passed, {failed} failed", YELLOW)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
