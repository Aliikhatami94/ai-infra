#!/usr/bin/env python
"""Progress Streaming: Real-Time Updates from Long-Running Tools.

This example demonstrates:
- Using the @progress decorator to enable progress streaming
- Sending progress updates from async tools
- Consuming progress events during agent execution
- Building progress UIs with percentage and status messages
- Cancellation and error handling with progress tools

Required API Keys (at least one):
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GOOGLE_API_KEY

The @progress decorator transforms an async function into a tool that can
send real-time progress updates while executing. The agent's streaming
interface exposes these updates as "progress" events.

Key concepts:
- @progress decorator: Marks a tool as progress-enabled
- stream parameter: Injected into the tool for sending updates
- stream.update(): Send progress message with optional percentage
- agent.astream(): Consume progress events in real-time
"""

import asyncio
import random

from ai_infra import Agent
from ai_infra.tools import is_progress_enabled, progress

# =============================================================================
# Example 1: Basic Progress Tool
# =============================================================================


@progress
async def analyze_data(file_path: str, stream) -> dict:
    """Analyze a data file with progress updates.

    Args:
        file_path: Path to the data file to analyze.

    Returns:
        Analysis results including row count and summary statistics.
    """
    await stream.update("Starting analysis...", percent=0)

    # Simulate loading data
    await asyncio.sleep(0.5)
    await stream.update(f"Loading {file_path}...", percent=10)

    # Simulate processing
    total_chunks = 5
    for i in range(total_chunks):
        await asyncio.sleep(0.3)
        percent = 10 + int((i + 1) / total_chunks * 80)
        await stream.update(f"Processing chunk {i + 1}/{total_chunks}", percent=percent)

    # Simulate final analysis
    await asyncio.sleep(0.3)
    await stream.update("Generating summary...", percent=95)

    await asyncio.sleep(0.2)
    await stream.update("Analysis complete!", percent=100)

    return {
        "file": file_path,
        "rows": random.randint(10000, 100000),
        "columns": random.randint(10, 50),
        "status": "success",
    }


def example_basic_progress() -> None:
    """Demonstrate basic progress streaming."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Progress Tool")
    print("=" * 60)

    # Check that the tool is progress-enabled
    print(f"\nTool name: {analyze_data.name}")
    print(f"Progress enabled: {is_progress_enabled(analyze_data)}")
    print(f"Description: {analyze_data.description}")


# =============================================================================
# Example 2: Progress Tool with Data Payload
# =============================================================================


@progress
async def download_files(urls: list[str], stream) -> dict:
    """Download multiple files with detailed progress.

    Args:
        urls: List of URLs to download.

    Returns:
        Download results for each file.
    """
    results = []
    total = len(urls)

    await stream.update(f"Starting download of {total} files...", percent=0)

    for i, url in enumerate(urls):
        # Extract filename from URL
        filename = url.split("/")[-1] if "/" in url else f"file_{i}"

        # Send progress with structured data
        await stream.update(
            f"Downloading {filename}...",
            percent=int((i / total) * 100),
            data={
                "current_file": filename,
                "files_completed": i,
                "files_total": total,
            },
        )

        # Simulate download time
        await asyncio.sleep(random.uniform(0.2, 0.5))

        results.append(
            {
                "url": url,
                "filename": filename,
                "size_kb": random.randint(100, 5000),
                "status": "downloaded",
            }
        )

    await stream.update(
        "All downloads complete!",
        percent=100,
        data={"total_files": total, "total_size_kb": sum(r["size_kb"] for r in results)},
    )

    return {"downloads": results, "success": True}


async def example_progress_with_data() -> None:
    """Show progress updates with structured data payloads."""
    print("\n" + "=" * 60)
    print("Example 2: Progress with Data Payload")
    print("=" * 60)

    # Call tool directly (no agent) to see progress events
    # Note: When called directly, progress is recorded but not streamed
    result = await download_files.ainvoke(
        {
            "urls": [
                "https://example.com/file1.pdf",
                "https://example.com/file2.csv",
                "https://example.com/file3.json",
            ]
        }
    )

    print(f"\nDownload result: {result}")


# =============================================================================
# Example 3: Multiple Progress Tools
# =============================================================================


@progress
async def extract_text(document: str, stream) -> str:
    """Extract text from a document.

    Args:
        document: Document path or content.

    Returns:
        Extracted text content.
    """
    await stream.update("Opening document...", percent=0)
    await asyncio.sleep(0.2)

    await stream.update("Extracting text...", percent=50)
    await asyncio.sleep(0.3)

    await stream.update("Text extraction complete", percent=100)
    return f"Extracted text from {document}: Lorem ipsum dolor sit amet..."


@progress
async def translate_text(text: str, target_language: str, stream) -> str:
    """Translate text to target language.

    Args:
        text: Text to translate.
        target_language: Target language code (e.g., 'es', 'fr', 'de').

    Returns:
        Translated text.
    """
    await stream.update(f"Preparing translation to {target_language}...", percent=0)
    await asyncio.sleep(0.2)

    # Simulate translation in chunks
    words = text.split()
    for i in range(0, len(words), 5):
        percent = int((i / len(words)) * 80) + 10
        await stream.update(f"Translating... {percent}%", percent=percent)
        await asyncio.sleep(0.1)

    await stream.update("Translation complete!", percent=100)
    return f"[Translated to {target_language}]: {text[:50]}... (simulated)"


@progress
async def summarize_text(text: str, max_sentences: int, stream) -> str:
    """Summarize text to a specified length.

    Args:
        text: Text to summarize.
        max_sentences: Maximum sentences in summary.

    Returns:
        Summarized text.
    """
    await stream.update("Analyzing text structure...", percent=10)
    await asyncio.sleep(0.2)

    await stream.update("Identifying key points...", percent=40)
    await asyncio.sleep(0.2)

    await stream.update("Generating summary...", percent=70)
    await asyncio.sleep(0.2)

    await stream.update("Summary complete!", percent=100)
    return f"Summary ({max_sentences} sentences): This is a simulated summary of the input text."


async def example_multiple_tools() -> None:
    """Demonstrate using multiple progress tools together."""
    print("\n" + "=" * 60)
    print("Example 3: Multiple Progress Tools")
    print("=" * 60)

    tools = [extract_text, translate_text, summarize_text]

    print("\nProgress-enabled tools:")
    for tool in tools:
        print(f"  - {tool.name}: {is_progress_enabled(tool)}")


# =============================================================================
# Example 4: Agent with Progress Streaming
# =============================================================================


async def example_agent_streaming() -> None:
    """Use progress tools with an Agent and consume progress events."""
    print("\n" + "=" * 60)
    print("Example 4: Agent with Progress Streaming")
    print("=" * 60)

    # Create agent with progress-enabled tools
    agent = Agent(
        tools=[analyze_data, download_files, extract_text],
        system_prompt="""You are a file processing assistant.
You can analyze data files, download files from URLs, and extract text.
Use the appropriate tool based on the user's request.""",
    )

    print("\n--- Streaming Progress Events ---\n")
    print("Query: 'Analyze the sales_2024.csv file'\n")

    try:
        # Stream and show progress events
        async for event in agent.astream("Analyze the sales_2024.csv file"):
            if hasattr(event, "type"):
                if event.type == "progress":
                    # Progress event from tool
                    bar = "█" * (event.percent // 5) if event.percent else ""
                    bar = bar.ljust(20, "░")
                    print(f"  [{bar}] {event.percent or 0:3d}% - {event.message}")
                elif event.type == "token":
                    # Streaming token from LLM
                    print(event.content, end="", flush=True)
                elif event.type == "tool_call":
                    print(f"\n Calling tool: {event.tool}")
                elif event.type == "tool_result":
                    print(f"\n[OK] Tool completed: {event.tool}")
            else:
                # Handle dict-style events
                if isinstance(event, dict):
                    if event.get("type") == "progress":
                        pct = event.get("percent", 0)
                        msg = event.get("message", "")
                        bar = "█" * (pct // 5) if pct else ""
                        bar = bar.ljust(20, "░")
                        print(f"  [{bar}] {pct:3d}% - {msg}")

        print("\n\n--- Streaming Complete ---")

    except Exception as e:
        print(f"\nError during streaming: {e}")
        print("(This is expected if no API key is configured)")


# =============================================================================
# Example 5: Progress Event Recording
# =============================================================================


@progress
async def batch_process(items: list[str], stream) -> dict:
    """Process a batch of items with recorded progress.

    Args:
        items: List of items to process.

    Returns:
        Processing results.
    """
    processed = []

    for i, item in enumerate(items):
        percent = int((i + 1) / len(items) * 100)
        await stream.update(f"Processing: {item}", percent=percent, data={"item": item})
        await asyncio.sleep(0.1)
        processed.append({"item": item, "status": "done"})

    return {"processed": processed, "count": len(items)}


async def example_event_recording() -> None:
    """Demonstrate progress event recording without streaming."""
    print("\n" + "=" * 60)
    print("Example 5: Progress Event Recording")
    print("=" * 60)

    # When tools are called directly (not through streaming agent),
    # progress events are recorded internally
    result = await batch_process.ainvoke(
        {"items": ["item_a", "item_b", "item_c", "item_d", "item_e"]}
    )

    print(f"\nProcessing result: {result}")
    print("\nNote: Progress events were recorded but not displayed")
    print("(Use agent.astream() to see real-time progress updates)")


# =============================================================================
# Example 6: Custom Progress Handler
# =============================================================================


async def example_custom_handler() -> None:
    """Show how progress callbacks work internally."""
    print("\n" + "=" * 60)
    print("Example 6: Understanding Progress Flow")
    print("=" * 60)

    print("""
The @progress decorator works as follows:

1. Decorator wraps async function:
   @progress
   async def my_tool(input: str, stream) -> str:
       await stream.update("Working...", percent=50)
       return "done"

2. 'stream' parameter is injected when tool is called:
   - ProgressStream instance with callback
   - Callback routes updates to agent streaming

3. Progress events flow:
   Tool → stream.update() → callback → agent.astream()

4. Event format:
   ProgressEvent(
       type="progress",
       tool="my_tool",
       message="Working...",
       percent=50,
       data=None
   )

5. In agent streaming:
   async for event in agent.astream("Do something"):
       if event.type == "progress":
           print(f"[{event.tool}] {event.message} ({event.percent}%)")
""")


# =============================================================================
# Example 7: Error Handling in Progress Tools
# =============================================================================


@progress
async def risky_operation(operation: str, stream) -> dict:
    """Perform a risky operation that might fail.

    Args:
        operation: Type of operation to perform.

    Returns:
        Operation result.
    """
    await stream.update("Starting operation...", percent=0)
    await asyncio.sleep(0.2)

    await stream.update("Validating inputs...", percent=20)
    await asyncio.sleep(0.2)

    # Simulate potential failure
    if operation == "fail":
        await stream.update("Error detected!", percent=50)
        raise ValueError("Operation failed: simulated error")

    await stream.update("Processing...", percent=60)
    await asyncio.sleep(0.2)

    await stream.update("Finalizing...", percent=90)
    await asyncio.sleep(0.1)

    await stream.update("Complete!", percent=100)
    return {"operation": operation, "success": True}


async def example_error_handling() -> None:
    """Demonstrate error handling with progress tools."""
    print("\n" + "=" * 60)
    print("Example 7: Error Handling")
    print("=" * 60)

    # Success case
    print("\n--- Success Case ---")
    try:
        result = await risky_operation.ainvoke({"operation": "safe"})
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

    # Failure case
    print("\n--- Failure Case ---")
    try:
        result = await risky_operation.ainvoke({"operation": "fail"})
        print(f"Result: {result}")
    except ValueError as e:
        print(f"Caught expected error: {e}")
        print("(Progress was sent before error occurred)")


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run all examples."""
    print("=" * 60)
    print("Progress Streaming Examples")
    print("Real-Time Updates from Long-Running Tools")
    print("=" * 60)

    # Sync examples
    example_basic_progress()

    # Async examples
    await example_progress_with_data()
    await example_multiple_tools()
    await example_event_recording()
    await example_custom_handler()
    await example_error_handling()

    # Agent streaming (requires API key)
    print("\n" + "=" * 60)
    print("Running Agent streaming example...")
    print("(Requires API key - will skip if not available)")
    print("=" * 60)

    try:
        await example_agent_streaming()
    except Exception as e:
        print(f"Agent example skipped: {e}")

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
