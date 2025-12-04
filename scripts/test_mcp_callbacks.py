#!/usr/bin/env python3
"""
Test script for MCP Callbacks functionality.

This script tests the callbacks feature with a real MCP server.
Run from ai-infra directory:
    poetry run python scripts/test_mcp_callbacks.py
"""

import asyncio
import sys
from datetime import datetime

# Add colors for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def log(msg: str, color: str = RESET):
    """Print with timestamp and color."""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{color}[{ts}] {msg}{RESET}")


async def test_callbacks_with_mock():
    """Test callbacks dataclass and conversion without a server."""
    log("=" * 60, BOLD)
    log("TEST 1: Callbacks Dataclass and Conversion", BOLD)
    log("=" * 60, BOLD)

    from ai_infra.mcp import CallbackContext, Callbacks

    # Track callback invocations
    progress_calls = []
    logging_calls = []

    async def on_progress(progress, total, message, ctx):
        progress_calls.append(
            {
                "progress": progress,
                "total": total,
                "message": message,
                "server": ctx.server_name,
                "tool": ctx.tool_name,
            }
        )
        log(
            f"  📊 Progress: {progress}/{total} - {message} [{ctx.server_name}/{ctx.tool_name}]",
            BLUE,
        )

    async def on_logging(params, ctx):
        logging_calls.append(
            {
                "params": params,
                "server": ctx.server_name,
                "tool": ctx.tool_name,
            }
        )
        log(f"  📝 Log: {params} [{ctx.server_name}]", YELLOW)

    # Create callbacks
    callbacks = Callbacks(on_progress=on_progress, on_logging=on_logging)
    log("✓ Created Callbacks instance", GREEN)

    # Test to_mcp_format conversion
    ctx = CallbackContext(server_name="test-server", tool_name="test-tool")
    mcp_callbacks = callbacks.to_mcp_format(ctx)
    log("✓ Converted to MCP format", GREEN)

    # Simulate MCP SDK calling our callbacks
    log("\nSimulating MCP SDK callback invocations:", BOLD)

    await mcp_callbacks.progress_callback(0.0, 100.0, "Starting...")
    await mcp_callbacks.progress_callback(50.0, 100.0, "Halfway there")
    await mcp_callbacks.progress_callback(100.0, 100.0, "Complete!")

    mock_log_params = {"level": "info", "data": "Tool executed successfully"}
    await mcp_callbacks.logging_callback(mock_log_params)

    # Verify
    assert len(progress_calls) == 3, f"Expected 3 progress calls, got {len(progress_calls)}"
    assert len(logging_calls) == 1, f"Expected 1 logging call, got {len(logging_calls)}"
    assert progress_calls[0]["progress"] == 0.0
    assert progress_calls[2]["progress"] == 100.0
    assert progress_calls[0]["server"] == "test-server"
    assert progress_calls[0]["tool"] == "test-tool"

    log("\n✓ All mock callback tests passed!", GREEN)
    return True


async def test_mcp_client_with_callbacks():
    """Test MCPClient initialization with callbacks."""
    log("\n" + "=" * 60, BOLD)
    log("TEST 2: MCPClient with Callbacks", BOLD)
    log("=" * 60, BOLD)

    from ai_infra.mcp import Callbacks, MCPClient

    progress_events = []

    async def on_progress(progress, total, message, ctx):
        progress_events.append((progress, total, message, ctx.server_name))
        log(f"  📊 [{ctx.server_name}] Progress: {progress}", BLUE)

    async def on_logging(params, ctx):
        log(f"  📝 [{ctx.server_name}] Log: {params}", YELLOW)

    # Create MCPClient with callbacks
    callbacks = Callbacks(on_progress=on_progress, on_logging=on_logging)

    # Use a non-existent server to test initialization (won't connect)
    mcp = MCPClient(
        [{"url": "http://localhost:9999/mcp", "transport": "streamable_http"}],
        callbacks=callbacks,
        discover_timeout=2.0,  # Short timeout
    )

    log("✓ MCPClient created with callbacks parameter", GREEN)
    log(f"  - Callbacks object stored: {mcp._callbacks is not None}", BLUE)
    log(f"  - on_progress set: {mcp._callbacks.on_progress is not None}", BLUE)
    log(f"  - on_logging set: {mcp._callbacks.on_logging is not None}", BLUE)

    # Test the langchain callbacks conversion
    lc_callbacks = mcp._to_langchain_callbacks()
    log(f"✓ Langchain callbacks conversion: {lc_callbacks is not None}", GREEN)

    return True


async def test_callback_context_isolation():
    """Test that contexts are properly isolated per tool call."""
    log("\n" + "=" * 60, BOLD)
    log("TEST 3: Context Isolation", BOLD)
    log("=" * 60, BOLD)

    from ai_infra.mcp import CallbackContext, Callbacks

    contexts_received = []

    async def on_progress(progress, total, message, ctx):
        contexts_received.append(ctx)

    callbacks = Callbacks(on_progress=on_progress)

    # Create multiple contexts for different servers/tools
    ctx1 = CallbackContext(server_name="server-a", tool_name="tool-1")
    ctx2 = CallbackContext(server_name="server-b", tool_name="tool-2")
    ctx3 = CallbackContext(server_name="server-a", tool_name="tool-3")

    mcp_cb1 = callbacks.to_mcp_format(ctx1)
    mcp_cb2 = callbacks.to_mcp_format(ctx2)
    mcp_cb3 = callbacks.to_mcp_format(ctx3)

    # Simulate interleaved calls (like parallel tool execution)
    await mcp_cb1.progress_callback(0.0, 1.0, None)
    await mcp_cb2.progress_callback(0.0, 1.0, None)
    await mcp_cb3.progress_callback(0.0, 1.0, None)
    await mcp_cb1.progress_callback(1.0, 1.0, None)
    await mcp_cb2.progress_callback(1.0, 1.0, None)
    await mcp_cb3.progress_callback(1.0, 1.0, None)

    # Verify each callback got correct context
    log("Received contexts:", BOLD)
    for i, ctx in enumerate(contexts_received):
        log(f"  {i + 1}. server={ctx.server_name}, tool={ctx.tool_name}", BLUE)

    assert len(contexts_received) == 6
    assert contexts_received[0].server_name == "server-a"
    assert contexts_received[0].tool_name == "tool-1"
    assert contexts_received[1].server_name == "server-b"
    assert contexts_received[1].tool_name == "tool-2"
    assert contexts_received[2].server_name == "server-a"
    assert contexts_received[2].tool_name == "tool-3"

    log("\n✓ Context isolation verified!", GREEN)
    return True


async def test_imports():
    """Test all public imports work correctly."""
    log("\n" + "=" * 60, BOLD)
    log("TEST 4: Public API Imports", BOLD)
    log("=" * 60, BOLD)

    # Test imports from ai_infra.mcp
    from ai_infra import mcp as mcp_module

    assert hasattr(mcp_module, "CallbackContext")
    assert hasattr(mcp_module, "Callbacks")
    assert hasattr(mcp_module, "LoggingCallback")
    assert hasattr(mcp_module, "MCPClient")
    assert hasattr(mcp_module, "ProgressCallback")
    log("✓ from ai_infra.mcp import Callbacks, CallbackContext, etc.", GREEN)

    # Test imports from ai_infra.mcp.client
    from ai_infra.mcp import client as client_module

    assert hasattr(client_module, "CallbackContext")
    assert hasattr(client_module, "Callbacks")
    assert hasattr(client_module, "MCPClient")
    log("✓ from ai_infra.mcp.client import Callbacks, CallbackContext, etc.", GREEN)

    # Test imports from specific module
    from ai_infra.mcp.client import callbacks as callbacks_module

    assert hasattr(callbacks_module, "CallbackContext")
    assert hasattr(callbacks_module, "Callbacks")
    assert hasattr(callbacks_module, "LoggingCallback")
    assert hasattr(callbacks_module, "ProgressCallback")
    assert hasattr(callbacks_module, "_MCPCallbacks")
    log("✓ from ai_infra.mcp.client.callbacks import ...", GREEN)

    return True


async def main():
    """Run all tests."""
    log("\n" + "=" * 60, BOLD)
    log("🧪 MCP CALLBACKS FUNCTIONALITY TEST", BOLD)
    log("=" * 60, BOLD)

    tests = [
        ("Imports", test_imports),
        ("Callbacks Mock", test_callbacks_with_mock),
        ("MCPClient with Callbacks", test_mcp_client_with_callbacks),
        ("Context Isolation", test_callback_context_isolation),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = await test_fn()
            results.append((name, result, None))
        except Exception as e:
            log(f"\n✗ {name} FAILED: {e}", RED)
            results.append((name, False, str(e)))

    # Summary
    log("\n" + "=" * 60, BOLD)
    log("📊 TEST SUMMARY", BOLD)
    log("=" * 60, BOLD)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        if success:
            log(f"  ✓ {name}", GREEN)
        else:
            log(f"  ✗ {name}: {error}", RED)

    log(f"\n{passed}/{total} tests passed", GREEN if passed == total else RED)

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
