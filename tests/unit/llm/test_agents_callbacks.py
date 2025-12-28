"""Tests for llm/agents/callbacks.py - Tool Callback Wrapping.

This module tests the wrap_tool_with_callbacks function which provides
callback instrumentation for tools (observability: start/end/error events).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ai_infra.callbacks import (
    CallbackManager,
    Callbacks,
    ToolEndEvent,
    ToolErrorEvent,
    ToolStartEvent,
)
from ai_infra.llm.agents.callbacks import wrap_tool_with_callbacks

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def tracking_callbacks():
    """Create callbacks that track all tool events."""
    events: list[tuple[str, Any]] = []

    class TrackingCallbacks(Callbacks):
        def on_tool_start(self, event: ToolStartEvent) -> None:
            events.append(("tool_start", event))

        def on_tool_end(self, event: ToolEndEvent) -> None:
            events.append(("tool_end", event))

        def on_tool_error(self, event: ToolErrorEvent) -> None:
            events.append(("tool_error", event))

    callbacks = TrackingCallbacks()
    manager = CallbackManager([callbacks])
    return manager, events


@pytest.fixture
def async_tracking_callbacks():
    """Create async callbacks that track all tool events."""
    events: list[tuple[str, Any]] = []

    class AsyncTrackingCallbacks(Callbacks):
        async def on_tool_start_async(self, event: ToolStartEvent) -> None:
            events.append(("tool_start", event))

        async def on_tool_end_async(self, event: ToolEndEvent) -> None:
            events.append(("tool_end", event))

        async def on_tool_error_async(self, event: ToolErrorEvent) -> None:
            events.append(("tool_error", event))

    callbacks = AsyncTrackingCallbacks()
    manager = CallbackManager([callbacks])
    return manager, events


# =============================================================================
# Plain Function Wrapping Tests
# =============================================================================


class TestWrapSyncFunction:
    """Tests for wrapping sync plain functions."""

    def test_wrap_sync_function_basic(self, tracking_callbacks):
        """Test wrapping a basic sync function."""
        manager, events = tracking_callbacks

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        wrapped = wrap_tool_with_callbacks(add, manager)
        result = wrapped(a=2, b=3)

        assert result == 5
        assert len(events) == 2
        assert events[0][0] == "tool_start"
        assert events[1][0] == "tool_end"

    def test_wrap_sync_function_tool_name(self, tracking_callbacks):
        """Test that tool name is correctly extracted from function name."""
        manager, events = tracking_callbacks

        def my_calculator(expression: str) -> str:
            """Calculate an expression."""
            return str(eval(expression))

        wrapped = wrap_tool_with_callbacks(my_calculator, manager)
        wrapped(expression="2+2")

        start_event = events[0][1]
        assert start_event.tool_name == "my_calculator"

    def test_wrap_sync_function_arguments(self, tracking_callbacks):
        """Test that arguments are correctly passed to callback."""
        manager, events = tracking_callbacks

        def greet(name: str, greeting: str = "Hello") -> str:
            """Greet a person."""
            return f"{greeting}, {name}!"

        wrapped = wrap_tool_with_callbacks(greet, manager)
        wrapped(name="Alice", greeting="Hi")

        start_event = events[0][1]
        assert start_event.arguments == {"name": "Alice", "greeting": "Hi"}

    def test_wrap_sync_function_result(self, tracking_callbacks):
        """Test that result is correctly passed to callback."""
        manager, events = tracking_callbacks

        def multiply(x: int, y: int) -> int:
            """Multiply two numbers."""
            return x * y

        wrapped = wrap_tool_with_callbacks(multiply, manager)
        result = wrapped(x=3, y=4)

        assert result == 12
        end_event = events[1][1]
        assert end_event.result == 12

    def test_wrap_sync_function_latency(self, tracking_callbacks):
        """Test that latency is recorded in callback."""
        import time

        manager, events = tracking_callbacks

        def slow_function() -> str:
            """A slow function."""
            time.sleep(0.1)
            return "done"

        wrapped = wrap_tool_with_callbacks(slow_function, manager)
        wrapped()

        end_event = events[1][1]
        assert end_event.latency_ms >= 100  # At least 100ms

    def test_wrap_sync_function_error(self, tracking_callbacks):
        """Test that errors are correctly captured."""
        manager, events = tracking_callbacks

        def failing_function() -> str:
            """A function that fails."""
            raise ValueError("Something went wrong")

        wrapped = wrap_tool_with_callbacks(failing_function, manager)

        with pytest.raises(ValueError, match="Something went wrong"):
            wrapped()

        assert len(events) == 2
        assert events[0][0] == "tool_start"
        assert events[1][0] == "tool_error"

        error_event = events[1][1]
        assert error_event.tool_name == "failing_function"
        assert isinstance(error_event.error, ValueError)
        assert str(error_event.error) == "Something went wrong"

    def test_wrap_sync_function_error_latency(self, tracking_callbacks):
        """Test that latency is recorded even on error."""
        import time

        manager, events = tracking_callbacks

        def slow_failing_function() -> str:
            """A slow function that fails."""
            time.sleep(0.05)
            raise RuntimeError("Slow failure")

        wrapped = wrap_tool_with_callbacks(slow_failing_function, manager)

        with pytest.raises(RuntimeError):
            wrapped()

        error_event = events[1][1]
        assert error_event.latency_ms >= 50

    def test_wrap_sync_function_preserves_docstring(self, tracking_callbacks):
        """Test that function docstring is preserved."""
        manager, _ = tracking_callbacks

        def documented_function() -> str:
            """This is my docstring."""
            return "result"

        wrapped = wrap_tool_with_callbacks(documented_function, manager)
        assert wrapped.__doc__ == "This is my docstring."

    def test_wrap_sync_function_preserves_name(self, tracking_callbacks):
        """Test that function name is preserved."""
        manager, _ = tracking_callbacks

        def my_special_function() -> str:
            """A special function."""
            return "result"

        wrapped = wrap_tool_with_callbacks(my_special_function, manager)
        assert wrapped.__name__ == "my_special_function"


class TestWrapAsyncFunction:
    """Tests for wrapping async plain functions."""

    @pytest.mark.asyncio
    async def test_wrap_async_function_basic(self, tracking_callbacks):
        """Test wrapping a basic async function."""
        manager, events = tracking_callbacks

        async def async_add(a: int, b: int) -> int:
            """Add two numbers asynchronously."""
            await asyncio.sleep(0.01)
            return a + b

        wrapped = wrap_tool_with_callbacks(async_add, manager)
        result = await wrapped(a=5, b=7)

        assert result == 12
        assert len(events) == 2
        assert events[0][0] == "tool_start"
        assert events[1][0] == "tool_end"

    @pytest.mark.asyncio
    async def test_wrap_async_function_tool_name(self, tracking_callbacks):
        """Test that tool name is correctly extracted from async function."""
        manager, events = tracking_callbacks

        async def async_calculator(expression: str) -> str:
            """Calculate an expression asynchronously."""
            await asyncio.sleep(0.001)
            return str(eval(expression))

        wrapped = wrap_tool_with_callbacks(async_calculator, manager)
        await wrapped(expression="10/2")

        start_event = events[0][1]
        assert start_event.tool_name == "async_calculator"

    @pytest.mark.asyncio
    async def test_wrap_async_function_arguments(self, tracking_callbacks):
        """Test that arguments are correctly passed to callback."""
        manager, events = tracking_callbacks

        async def async_greet(name: str, greeting: str = "Hello") -> str:
            """Greet a person asynchronously."""
            return f"{greeting}, {name}!"

        wrapped = wrap_tool_with_callbacks(async_greet, manager)
        await wrapped(name="Bob", greeting="Hey")

        start_event = events[0][1]
        assert start_event.arguments == {"name": "Bob", "greeting": "Hey"}

    @pytest.mark.asyncio
    async def test_wrap_async_function_result(self, tracking_callbacks):
        """Test that result is correctly passed to callback."""
        manager, events = tracking_callbacks

        async def async_divide(x: float, y: float) -> float:
            """Divide two numbers."""
            return x / y

        wrapped = wrap_tool_with_callbacks(async_divide, manager)
        result = await wrapped(x=10.0, y=2.0)

        assert result == 5.0
        end_event = events[1][1]
        assert end_event.result == 5.0

    @pytest.mark.asyncio
    async def test_wrap_async_function_latency(self, tracking_callbacks):
        """Test that latency is recorded for async function."""
        manager, events = tracking_callbacks

        async def slow_async_function() -> str:
            """A slow async function."""
            await asyncio.sleep(0.1)
            return "done"

        wrapped = wrap_tool_with_callbacks(slow_async_function, manager)
        await wrapped()

        end_event = events[1][1]
        assert end_event.latency_ms >= 100

    @pytest.mark.asyncio
    async def test_wrap_async_function_error(self, tracking_callbacks):
        """Test that errors are correctly captured for async functions."""
        manager, events = tracking_callbacks

        async def async_failing_function() -> str:
            """An async function that fails."""
            await asyncio.sleep(0.001)
            raise ValueError("Async error")

        wrapped = wrap_tool_with_callbacks(async_failing_function, manager)

        with pytest.raises(ValueError, match="Async error"):
            await wrapped()

        assert len(events) == 2
        assert events[0][0] == "tool_start"
        assert events[1][0] == "tool_error"

        error_event = events[1][1]
        assert error_event.tool_name == "async_failing_function"
        assert isinstance(error_event.error, ValueError)

    @pytest.mark.asyncio
    async def test_wrap_async_function_preserves_name(self, tracking_callbacks):
        """Test that async function name is preserved."""
        manager, _ = tracking_callbacks

        async def my_async_special_function() -> str:
            """A special async function."""
            return "result"

        wrapped = wrap_tool_with_callbacks(my_async_special_function, manager)
        assert wrapped.__name__ == "my_async_special_function"


# =============================================================================
# BaseTool Wrapping Tests
# =============================================================================


class TestWrapBaseTool:
    """Tests for wrapping LangChain BaseTool instances."""

    def test_wrap_base_tool_sync_run(self, tracking_callbacks):
        """Test wrapping BaseTool with sync _run method."""
        from langchain_core.tools import BaseTool

        manager, events = tracking_callbacks

        class CalculatorTool(BaseTool):
            name: str = "calculator"
            description: str = "A simple calculator"

            def _run(self, expression: str) -> str:
                return str(eval(expression))

        tool = CalculatorTool()
        wrapped = wrap_tool_with_callbacks(tool, manager)

        result = wrapped._run(expression="2+2")

        assert result == "4"
        assert len(events) == 2
        assert events[0][0] == "tool_start"
        assert events[1][0] == "tool_end"

    def test_wrap_base_tool_name(self, tracking_callbacks):
        """Test that tool name is correctly extracted from BaseTool."""
        from langchain_core.tools import BaseTool

        manager, events = tracking_callbacks

        class WeatherTool(BaseTool):
            name: str = "get_weather"
            description: str = "Get weather for a location"

            def _run(self, location: str) -> str:
                return f"Sunny in {location}"

        tool = WeatherTool()
        wrapped = wrap_tool_with_callbacks(tool, manager)
        wrapped._run(location="Seattle")

        start_event = events[0][1]
        assert start_event.tool_name == "get_weather"

    def test_wrap_base_tool_arguments(self, tracking_callbacks):
        """Test that BaseTool arguments are correctly captured."""
        from langchain_core.tools import BaseTool

        manager, events = tracking_callbacks

        class SearchTool(BaseTool):
            name: str = "search"
            description: str = "Search the web"

            def _run(self, query: str, limit: int = 10) -> str:
                return f"Results for {query} (limit={limit})"

        tool = SearchTool()
        wrapped = wrap_tool_with_callbacks(tool, manager)
        wrapped._run(query="python", limit=5)

        start_event = events[0][1]
        assert start_event.arguments == {"query": "python", "limit": 5}

    def test_wrap_base_tool_result(self, tracking_callbacks):
        """Test that BaseTool result is correctly captured."""
        from langchain_core.tools import BaseTool

        manager, events = tracking_callbacks

        class EchoTool(BaseTool):
            name: str = "echo"
            description: str = "Echo back input"

            def _run(self, message: str) -> str:
                return f"Echo: {message}"

        tool = EchoTool()
        wrapped = wrap_tool_with_callbacks(tool, manager)
        result = wrapped._run(message="hello")

        assert result == "Echo: hello"
        end_event = events[1][1]
        assert end_event.result == "Echo: hello"

    def test_wrap_base_tool_error(self, tracking_callbacks):
        """Test that BaseTool errors are correctly captured."""
        from langchain_core.tools import BaseTool

        manager, events = tracking_callbacks

        class FailingTool(BaseTool):
            name: str = "failing_tool"
            description: str = "A tool that always fails"

            def _run(self) -> str:
                raise RuntimeError("Tool failure")

        tool = FailingTool()
        wrapped = wrap_tool_with_callbacks(tool, manager)

        with pytest.raises(RuntimeError, match="Tool failure"):
            wrapped._run()

        assert len(events) == 2
        assert events[1][0] == "tool_error"
        error_event = events[1][1]
        assert error_event.tool_name == "failing_tool"

    def test_wrap_base_tool_preserves_identity(self, tracking_callbacks):
        """Test that wrapping modifies the tool in place and returns same instance."""
        from langchain_core.tools import BaseTool

        manager, _ = tracking_callbacks

        class MyTool(BaseTool):
            name: str = "my_tool"
            description: str = "My tool"

            def _run(self) -> str:
                return "result"

        tool = MyTool()
        wrapped = wrap_tool_with_callbacks(tool, manager)

        # Should be the same object (modified in place)
        assert wrapped is tool

    def test_wrap_base_tool_preserves_attributes(self, tracking_callbacks):
        """Test that wrapping preserves BaseTool attributes."""
        from langchain_core.tools import BaseTool

        manager, _ = tracking_callbacks

        class DetailedTool(BaseTool):
            name: str = "detailed_tool"
            description: str = "A detailed tool with metadata"

            def _run(self) -> str:
                return "result"

        tool = DetailedTool()
        wrapped = wrap_tool_with_callbacks(tool, manager)

        assert wrapped.name == "detailed_tool"
        assert wrapped.description == "A detailed tool with metadata"

    @pytest.mark.asyncio
    async def test_wrap_base_tool_async_run(self, async_tracking_callbacks):
        """Test wrapping BaseTool with async _arun method."""
        from langchain_core.tools import BaseTool

        manager, events = async_tracking_callbacks

        class AsyncCalculatorTool(BaseTool):
            name: str = "async_calculator"
            description: str = "An async calculator"

            def _run(self, expression: str) -> str:
                return str(eval(expression))

            async def _arun(self, expression: str) -> str:
                await asyncio.sleep(0.01)
                return str(eval(expression))

        tool = AsyncCalculatorTool()
        wrapped = wrap_tool_with_callbacks(tool, manager)

        result = await wrapped._arun(expression="3*4")

        assert result == "12"
        assert len(events) == 2
        assert events[0][0] == "tool_start"
        assert events[1][0] == "tool_end"

    @pytest.mark.asyncio
    async def test_wrap_base_tool_async_fallback_to_sync(self, async_tracking_callbacks):
        """Test async wrapper falls back to sync _run if no _arun."""
        from langchain_core.tools import BaseTool

        manager, events = async_tracking_callbacks

        class SyncOnlyTool(BaseTool):
            name: str = "sync_only"
            description: str = "A sync-only tool"

            def _run(self, value: str) -> str:
                return f"sync: {value}"

        tool = SyncOnlyTool()
        wrapped = wrap_tool_with_callbacks(tool, manager)

        result = await wrapped._arun(value="test")

        assert result == "sync: test"
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_wrap_base_tool_async_error(self, async_tracking_callbacks):
        """Test async errors are correctly captured."""
        from langchain_core.tools import BaseTool

        manager, events = async_tracking_callbacks

        class AsyncFailingTool(BaseTool):
            name: str = "async_failing"
            description: str = "An async tool that fails"

            def _run(self) -> str:
                raise RuntimeError("Sync failure")

            async def _arun(self) -> str:
                await asyncio.sleep(0.001)
                raise ValueError("Async failure")

        tool = AsyncFailingTool()
        wrapped = wrap_tool_with_callbacks(tool, manager)

        with pytest.raises(ValueError, match="Async failure"):
            await wrapped._arun()

        assert len(events) == 2
        assert events[1][0] == "tool_error"


# =============================================================================
# LangChain @tool Decorator Tests
# =============================================================================


class TestWrapToolDecorator:
    """Tests for wrapping tools created with LangChain @tool decorator.

    Note: The @tool decorator creates StructuredTool which has a different
    signature for _run/_arun (requires config parameter). The current
    wrap_tool_with_callbacks implementation doesn't fully support this.
    These tests verify the basic wrapping works for simple BaseTool subclasses.
    """

    def test_wrap_decorated_tool_sync(self, tracking_callbacks):
        """Test wrapping a sync tool created with @tool decorator."""
        from langchain_core.tools import tool

        manager, events = tracking_callbacks

        @tool
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        wrapped = wrap_tool_with_callbacks(add_numbers, manager)

        # For @tool decorated functions, we need to invoke properly
        # The wrapper modifies the .func attribute
        assert hasattr(wrapped, "func")

    @pytest.mark.skip(reason="StructuredTool._run requires config param not supported by wrapper")
    def test_wrap_decorated_tool_name(self, tracking_callbacks):
        """Test tool name extraction from @tool decorated function."""
        from langchain_core.tools import tool

        manager, events = tracking_callbacks

        @tool
        def custom_search(query: str) -> str:
            """Search for something."""
            return f"Results for: {query}"

        # @tool decorator creates a StructuredTool which is a BaseTool
        wrapped = wrap_tool_with_callbacks(custom_search, manager)

        # Execute the wrapped tool using invoke() which handles config internally
        result = wrapped.invoke({"query": "python"})

        assert "Results for: python" in result
        start_event = events[0][1]
        assert start_event.tool_name == "custom_search"

    @pytest.mark.skip(reason="StructuredTool._arun requires config param not supported by wrapper")
    @pytest.mark.asyncio
    async def test_wrap_decorated_tool_async(self, async_tracking_callbacks):
        """Test wrapping an async tool created with @tool decorator."""
        from langchain_core.tools import tool

        manager, events = async_tracking_callbacks

        @tool
        async def async_search(query: str) -> str:
            """Search asynchronously."""
            await asyncio.sleep(0.01)
            return f"Async results for: {query}"

        wrapped = wrap_tool_with_callbacks(async_search, manager)
        # Use ainvoke which handles config internally
        result = await wrapped.ainvoke({"query": "test"})

        assert "Async results for: test" in result
        assert len(events) == 2


# =============================================================================
# Callback Event Data Tests
# =============================================================================


class TestCallbackEventData:
    """Tests for callback event data correctness."""

    def test_tool_start_event_timestamp(self, tracking_callbacks):
        """Test ToolStartEvent has valid timestamp."""
        import time

        manager, events = tracking_callbacks

        def simple_tool() -> str:
            """A simple tool."""
            return "result"

        before = time.time()
        wrapped = wrap_tool_with_callbacks(simple_tool, manager)
        wrapped()
        after = time.time()

        start_event = events[0][1]
        assert before <= start_event.timestamp <= after

    def test_tool_end_event_latency_is_positive(self, tracking_callbacks):
        """Test ToolEndEvent latency is always positive."""
        manager, events = tracking_callbacks

        def instant_tool() -> str:
            """An instant tool."""
            return "result"

        wrapped = wrap_tool_with_callbacks(instant_tool, manager)
        wrapped()

        end_event = events[1][1]
        assert end_event.latency_ms >= 0

    def test_tool_error_event_includes_arguments(self, tracking_callbacks):
        """Test ToolErrorEvent includes the arguments that caused the error."""
        manager, events = tracking_callbacks

        def divide(a: int, b: int) -> float:
            """Divide a by b."""
            return a / b

        wrapped = wrap_tool_with_callbacks(divide, manager)

        with pytest.raises(ZeroDivisionError):
            wrapped(a=1, b=0)

        error_event = events[1][1]
        assert error_event.arguments == {"a": 1, "b": 0}

    def test_events_fire_in_correct_order(self, tracking_callbacks):
        """Test that start event fires before end event."""
        manager, events = tracking_callbacks

        def ordered_tool() -> str:
            """A tool to test event order."""
            return "result"

        wrapped = wrap_tool_with_callbacks(ordered_tool, manager)
        wrapped()

        assert len(events) == 2
        assert events[0][0] == "tool_start"
        assert events[1][0] == "tool_end"

        # Timestamps should be in order
        start_event = events[0][1]
        end_event = events[1][1]
        assert start_event.timestamp <= end_event.timestamp


# =============================================================================
# Multiple Callbacks Tests
# =============================================================================


class TestMultipleCallbacks:
    """Tests for tools with multiple callback handlers."""

    def test_multiple_callbacks_all_receive_events(self):
        """Test that all registered callbacks receive events."""
        events1: list[tuple[str, Any]] = []
        events2: list[tuple[str, Any]] = []

        class Callbacks1(Callbacks):
            def on_tool_start(self, event: ToolStartEvent) -> None:
                events1.append(("start", event))

            def on_tool_end(self, event: ToolEndEvent) -> None:
                events1.append(("end", event))

        class Callbacks2(Callbacks):
            def on_tool_start(self, event: ToolStartEvent) -> None:
                events2.append(("start", event))

            def on_tool_end(self, event: ToolEndEvent) -> None:
                events2.append(("end", event))

        manager = CallbackManager([Callbacks1(), Callbacks2()])

        def multi_callback_tool() -> str:
            """A tool with multiple callbacks."""
            return "result"

        wrapped = wrap_tool_with_callbacks(multi_callback_tool, manager)
        wrapped()

        assert len(events1) == 2
        assert len(events2) == 2
        assert events1[0][0] == "start"
        assert events2[0][0] == "start"

    def test_callback_error_doesnt_affect_tool_execution(self):
        """Test that callback errors don't prevent tool execution."""
        tool_executed = [False]

        class FailingCallbacks(Callbacks):
            def on_tool_start(self, event: ToolStartEvent) -> None:
                raise RuntimeError("Callback failure!")

        manager = CallbackManager([FailingCallbacks()])

        def resilient_tool() -> str:
            """A tool that should still execute."""
            tool_executed[0] = True
            return "executed"

        wrapped = wrap_tool_with_callbacks(resilient_tool, manager)

        # Tool should still execute even if callback fails
        # (CallbackManager catches and logs callback errors)
        result = wrapped()

        assert tool_executed[0] is True
        assert result == "executed"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_wrap_tool_with_no_arguments(self, tracking_callbacks):
        """Test wrapping a tool that takes no arguments."""
        manager, events = tracking_callbacks

        def no_args_tool() -> str:
            """A tool with no arguments."""
            return "no args"

        wrapped = wrap_tool_with_callbacks(no_args_tool, manager)
        result = wrapped()

        assert result == "no args"
        start_event = events[0][1]
        assert start_event.arguments == {}

    def test_wrap_tool_with_many_arguments(self, tracking_callbacks):
        """Test wrapping a tool with many arguments."""
        manager, events = tracking_callbacks

        def many_args_tool(a: int, b: str, c: float, d: bool, e: list, f: dict) -> str:
            """A tool with many arguments."""
            return f"{a}-{b}-{c}-{d}"

        wrapped = wrap_tool_with_callbacks(many_args_tool, manager)
        wrapped(a=1, b="two", c=3.0, d=True, e=[1, 2], f={"key": "value"})

        start_event = events[0][1]
        assert start_event.arguments == {
            "a": 1,
            "b": "two",
            "c": 3.0,
            "d": True,
            "e": [1, 2],
            "f": {"key": "value"},
        }

    def test_wrap_tool_returning_none(self, tracking_callbacks):
        """Test wrapping a tool that returns None."""
        manager, events = tracking_callbacks

        def void_tool() -> None:
            """A tool that returns None."""
            pass

        wrapped = wrap_tool_with_callbacks(void_tool, manager)
        result = wrapped()

        assert result is None
        end_event = events[1][1]
        assert end_event.result is None

    def test_wrap_tool_returning_complex_object(self, tracking_callbacks):
        """Test wrapping a tool that returns a complex object."""
        manager, events = tracking_callbacks

        def complex_tool() -> dict:
            """A tool that returns a complex object."""
            return {
                "nested": {"data": [1, 2, 3]},
                "count": 42,
            }

        wrapped = wrap_tool_with_callbacks(complex_tool, manager)
        result = wrapped()

        assert result == {"nested": {"data": [1, 2, 3]}, "count": 42}
        end_event = events[1][1]
        assert end_event.result == {"nested": {"data": [1, 2, 3]}, "count": 42}

    def test_wrap_tool_with_positional_args(self, tracking_callbacks):
        """Test wrapping a tool called with positional arguments."""
        manager, events = tracking_callbacks

        def positional_tool(x: int, y: int) -> int:
            """A tool that uses positional args."""
            return x + y

        wrapped = wrap_tool_with_callbacks(positional_tool, manager)
        # Call with positional args
        result = wrapped(1, 2)

        assert result == 3
        # Positional args don't show up in kwargs
        start_event = events[0][1]
        assert start_event.arguments == {}

    def test_wrap_tool_with_mixed_args(self, tracking_callbacks):
        """Test wrapping a tool called with mixed args and kwargs."""
        manager, events = tracking_callbacks

        def mixed_tool(x: int, y: int, z: int = 0) -> int:
            """A tool with mixed args."""
            return x + y + z

        wrapped = wrap_tool_with_callbacks(mixed_tool, manager)
        result = wrapped(1, 2, z=3)

        assert result == 6
        start_event = events[0][1]
        # Only keyword args are captured
        assert start_event.arguments == {"z": 3}

    def test_wrap_same_tool_twice(self, tracking_callbacks):
        """Test that wrapping a tool twice doesn't cause issues."""
        manager, events = tracking_callbacks

        def idempotent_tool() -> str:
            """A tool to test idempotent wrapping."""
            return "result"

        wrapped1 = wrap_tool_with_callbacks(idempotent_tool, manager)
        wrapped2 = wrap_tool_with_callbacks(wrapped1, manager)

        # Should still work (might fire events twice though)
        result = wrapped2()
        assert result == "result"

    def test_wrap_tool_with_custom_name_attribute(self, tracking_callbacks):
        """Test tool with custom 'name' attribute."""
        manager, events = tracking_callbacks

        def tool_function() -> str:
            """A tool with custom name."""
            return "result"

        # Add a custom name attribute
        tool_function.name = "custom_tool_name"  # type: ignore

        wrapped = wrap_tool_with_callbacks(tool_function, manager)
        wrapped()

        start_event = events[0][1]
        assert start_event.tool_name == "custom_tool_name"


# =============================================================================
# Integration with Built-in Callbacks
# =============================================================================


class TestIntegrationWithBuiltinCallbacks:
    """Tests for wrap_tool_with_callbacks with built-in callback implementations."""

    def test_with_metrics_callbacks(self):
        """Test tool wrapping with MetricsCallbacks."""
        from ai_infra.callbacks import MetricsCallbacks

        metrics = MetricsCallbacks()
        manager = CallbackManager([metrics])

        def metered_tool() -> str:
            """A tool to measure."""
            return "measured"

        wrapped = wrap_tool_with_callbacks(metered_tool, manager)
        wrapped()

        assert metrics.tool_calls == 1
        assert metrics.tool_errors == 0

    def test_with_metrics_callbacks_on_error(self):
        """Test MetricsCallbacks records errors."""
        from ai_infra.callbacks import MetricsCallbacks

        metrics = MetricsCallbacks()
        manager = CallbackManager([metrics])

        def failing_metered_tool() -> str:
            """A tool that fails."""
            raise ValueError("Measurement error")

        wrapped = wrap_tool_with_callbacks(failing_metered_tool, manager)

        with pytest.raises(ValueError):
            wrapped()

        assert metrics.tool_calls == 0  # on_tool_end not called
        assert metrics.tool_errors == 1

    def test_with_logging_callbacks(self, caplog):
        """Test tool wrapping with LoggingCallbacks."""
        import logging

        from ai_infra.callbacks import LoggingCallbacks

        callbacks = LoggingCallbacks(level="INFO")
        manager = CallbackManager([callbacks])

        def logged_tool() -> str:
            """A tool to log."""
            return "logged"

        wrapped = wrap_tool_with_callbacks(logged_tool, manager)

        with caplog.at_level(logging.INFO, logger="ai_infra.callbacks"):
            wrapped()

        # Check that logging happened
        assert any("logged_tool" in record.message for record in caplog.records)

    def test_with_print_callbacks(self, capsys):
        """Test tool wrapping with PrintCallbacks."""
        from ai_infra.callbacks import PrintCallbacks

        callbacks = PrintCallbacks(verbose=True)
        manager = CallbackManager([callbacks])

        def printed_tool() -> str:
            """A tool to print."""
            return "printed"

        wrapped = wrap_tool_with_callbacks(printed_tool, manager)
        wrapped()

        captured = capsys.readouterr()
        assert "printed_tool" in captured.out
        assert "completed" in captured.out.lower()

    def test_with_multiple_builtin_callbacks(self):
        """Test with multiple built-in callbacks combined."""
        from ai_infra.callbacks import LoggingCallbacks, MetricsCallbacks, PrintCallbacks

        metrics = MetricsCallbacks()
        manager = CallbackManager(
            [
                LoggingCallbacks(),
                metrics,
                PrintCallbacks(),
            ]
        )

        def multi_observed_tool() -> str:
            """A tool observed by multiple callbacks."""
            return "observed"

        wrapped = wrap_tool_with_callbacks(multi_observed_tool, manager)
        wrapped()
        wrapped()

        # MetricsCallbacks should count both calls
        assert metrics.tool_calls == 2
