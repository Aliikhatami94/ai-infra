"""Tests for progress streaming - @progress decorator and ProgressStream."""

import asyncio

import pytest

from ai_infra.tools import ProgressEvent, ProgressStream, is_progress_enabled, progress


class TestProgressEvent:
    """Tests for ProgressEvent dataclass."""

    def test_create_basic_event(self):
        event = ProgressEvent(
            tool="my_tool",
            message="Processing...",
        )

        assert event.type == "progress"
        assert event.tool == "my_tool"
        assert event.message == "Processing..."
        assert event.percent is None
        assert event.data is None

    def test_create_full_event(self):
        event = ProgressEvent(
            tool="analyze",
            message="50% complete",
            percent=50,
            data={"files_processed": 5},
        )

        assert event.percent == 50
        assert event.data["files_processed"] == 5

    def test_to_dict(self):
        event = ProgressEvent(
            tool="my_tool",
            message="Working...",
            percent=75,
            data={"current": "file.txt"},
        )

        d = event.to_dict()

        assert d["type"] == "progress"
        assert d["tool"] == "my_tool"
        assert d["message"] == "Working..."
        assert d["percent"] == 75
        assert d["data"]["current"] == "file.txt"

    def test_to_dict_minimal(self):
        event = ProgressEvent(
            tool="tool",
            message="Status",
        )

        d = event.to_dict()

        # Should not include None values
        assert "percent" not in d
        assert "data" not in d


class TestProgressStream:
    """Tests for ProgressStream class."""

    @pytest.mark.asyncio
    async def test_update_records_events(self):
        stream = ProgressStream("my_tool")

        await stream.update("Starting...")
        await stream.update("Halfway", percent=50)
        await stream.update("Done!", percent=100, data={"result": "ok"})

        events = stream.events

        assert len(events) == 3
        assert events[0].message == "Starting..."
        assert events[1].percent == 50
        assert events[2].data["result"] == "ok"

    @pytest.mark.asyncio
    async def test_update_with_callback(self):
        received_events = []

        def callback(event):
            received_events.append(event)

        stream = ProgressStream("my_tool", callback)

        await stream.update("Update 1")
        await stream.update("Update 2", percent=50)

        assert len(received_events) == 2
        assert received_events[0].message == "Update 1"
        assert received_events[1].percent == 50

    @pytest.mark.asyncio
    async def test_update_with_async_callback(self):
        received_events = []

        async def async_callback(event):
            await asyncio.sleep(0)  # Simulate async work
            received_events.append(event)

        stream = ProgressStream("my_tool", async_callback)

        await stream.update("Async update")

        assert len(received_events) == 1
        assert received_events[0].message == "Async update"

    @pytest.mark.asyncio
    async def test_tool_name_in_events(self):
        stream = ProgressStream("specific_tool")

        await stream.update("Test")

        assert stream.events[0].tool == "specific_tool"


class TestProgressDecorator:
    """Tests for @progress decorator."""

    def test_decorator_requires_async(self):
        with pytest.raises(TypeError, match="async function"):

            @progress
            def sync_function(x, stream):
                pass

    @pytest.mark.asyncio
    async def test_decorated_function_receives_stream(self):
        received_stream = None

        @progress
        async def my_tool(value: str, stream) -> str:
            nonlocal received_stream
            received_stream = stream
            await stream.update("Processing...")
            return f"Result: {value}"

        # Now returns StructuredTool, use _async_wrapper
        result = await my_tool._async_wrapper(value="test")

        assert result == "Result: test"
        assert received_stream is not None
        assert isinstance(received_stream, ProgressStream)
        assert len(received_stream.events) == 1

    @pytest.mark.asyncio
    async def test_progress_enabled_flag(self):
        @progress
        async def my_tool(x: str, stream):
            return x

        assert my_tool._progress_enabled is True
        assert is_progress_enabled(my_tool) is True

    @pytest.mark.asyncio
    async def test_progress_callback_injection(self):
        received_events = []

        async def callback(event):
            received_events.append(event)

        @progress
        async def my_tool(value: str, stream) -> str:
            await stream.update("Step 1", percent=25)
            await stream.update("Step 2", percent=50)
            await stream.update("Step 3", percent=75)
            await stream.update("Done", percent=100)
            return value

        result = await my_tool._async_wrapper(value="test", _progress_callback=callback)

        assert result == "test"
        assert len(received_events) == 4
        assert received_events[0].percent == 25
        assert received_events[3].percent == 100

    @pytest.mark.asyncio
    async def test_preserves_function_metadata(self):
        @progress
        async def documented_tool(x: str, stream) -> str:
            """This is a documented tool."""
            return x

        # StructuredTool has name and description attributes
        assert documented_tool.name == "documented_tool"
        assert "documented tool" in documented_tool.description

    @pytest.mark.asyncio
    async def test_progress_with_additional_args(self):
        @progress
        async def complex_tool(
            a: int, b: str, c: bool = False, stream: ProgressStream = None
        ) -> dict:
            await stream.update(f"Processing {a}, {b}, {c}")
            return {"a": a, "b": b, "c": c}

        result = await complex_tool._async_wrapper(a=1, b="test", c=True)

        assert result == {"a": 1, "b": "test", "c": True}

    def test_is_progress_enabled_false_for_regular_function(self):
        def regular_function():
            pass

        assert is_progress_enabled(regular_function) is False

    @pytest.mark.asyncio
    async def test_original_function_preserved(self):
        @progress
        async def my_tool(x, stream):
            return x * 2

        assert hasattr(my_tool, "_original_fn")
        # Original function should have the same name
        assert my_tool._original_fn.__name__ == "my_tool"


class TestProgressIntegration:
    """Integration tests for progress streaming."""

    @pytest.mark.asyncio
    async def test_realistic_progress_scenario(self):
        """Test a realistic scenario with chunked processing."""
        events = []

        async def collect_events(event):
            events.append(event)

        @progress
        async def process_data(items: list, stream: ProgressStream = None) -> dict:
            """Process items with progress updates."""
            results = []
            total = len(items)

            await stream.update("Starting processing...", percent=0)

            for i, item in enumerate(items):
                # Simulate processing
                await asyncio.sleep(0)
                results.append(item.upper())

                percent = int((i + 1) / total * 100)
                await stream.update(
                    f"Processed {i + 1}/{total}",
                    percent=percent,
                    data={"current_item": item},
                )

            await stream.update("Complete!", percent=100)

            return {"results": results, "count": len(results)}

        items = ["a", "b", "c", "d", "e"]
        result = await process_data._async_wrapper(items=items, _progress_callback=collect_events)

        assert result["count"] == 5
        assert result["results"] == ["A", "B", "C", "D", "E"]

        # Should have 7 events: 1 start + 5 per-item + 1 complete
        assert len(events) == 7

        # First event
        assert events[0].message == "Starting processing..."
        assert events[0].percent == 0

        # Last event
        assert events[-1].message == "Complete!"
        assert events[-1].percent == 100

        # Progress should increment
        percents = [e.percent for e in events if e.percent is not None]
        assert percents == [0, 20, 40, 60, 80, 100, 100]

    @pytest.mark.asyncio
    async def test_progress_without_callback(self):
        """Tool should work even without progress callback."""

        @progress
        async def simple_tool(x: int, stream: ProgressStream = None) -> int:
            await stream.update("Working...")
            return x * 2

        # Call without _progress_callback
        result = await simple_tool._async_wrapper(x=5)

        assert result == 10

    @pytest.mark.asyncio
    async def test_progress_with_error_in_tool(self):
        """Progress should work even if tool raises an error."""
        events = []

        async def collect(event):
            events.append(event)

        @progress
        async def failing_tool(x: int, stream: ProgressStream = None):
            await stream.update("Starting...")
            await stream.update("About to fail...")
            raise ValueError("Intentional error")

        with pytest.raises(ValueError, match="Intentional error"):
            await failing_tool._async_wrapper(x=1, _progress_callback=collect)

        # Events before error should be recorded
        assert len(events) == 2
        assert events[0].message == "Starting..."
        assert events[1].message == "About to fail..."
