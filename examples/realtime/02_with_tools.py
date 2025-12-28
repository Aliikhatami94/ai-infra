#!/usr/bin/env python
"""Realtime Voice with Tool Calling Example.

This example demonstrates:
- Defining tools for voice conversations
- Tool execution during voice sessions
- Returning tool results to the AI
- Multi-tool conversations
- Async tool execution

Tools enable your AI assistant to take actions and
fetch information during voice conversations.
"""

import asyncio
from datetime import datetime

from ai_infra.llm.realtime import (
    RealtimeConfig,
    RealtimeVoice,
    ToolCallRequest,
    ToolDefinition,
)

# =============================================================================
# Example 1: Defining Tools
# =============================================================================


async def defining_tools():
    """Define tools for voice conversations."""
    print("=" * 60)
    print("1. Defining Tools")
    print("=" * 60)

    # Define tools using ToolDefinition
    weather_tool = ToolDefinition(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    )

    calendar_tool = ToolDefinition(
        name="check_calendar",
        description="Check calendar events for a date",
        parameters={
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format",
                },
            },
            "required": ["date"],
        },
    )

    print("\n  Defined tools:")
    print(f"    - {weather_tool.name}: {weather_tool.description}")
    print(f"    - {calendar_tool.name}: {calendar_tool.description}")

    print("\n  Tool parameters use JSON Schema format")


# =============================================================================
# Example 2: Registering Tools with Config
# =============================================================================


async def registering_tools():
    """Register tools in realtime config."""
    print("\n" + "=" * 60)
    print("2. Registering Tools")
    print("=" * 60)

    # Define tools
    tools = [
        ToolDefinition(
            name="get_time",
            description="Get the current time",
            parameters={"type": "object", "properties": {}},
        ),
        ToolDefinition(
            name="set_reminder",
            description="Set a reminder",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "minutes": {"type": "integer"},
                },
                "required": ["message", "minutes"],
            },
        ),
    ]

    # Create config with tools
    config = RealtimeConfig(
        tools=tools,
        model="gpt-4o-realtime-preview",
    )

    print(f"\n  Registered {len(config.tools)} tools:")
    for tool in config.tools:
        print(f"    - {tool.name}")

    # Create voice with config
    _voice = RealtimeVoice(config=config)
    print("\n  Voice configured with tools")


# =============================================================================
# Example 3: Handling Tool Calls
# =============================================================================


async def handling_tool_calls():
    """Handle tool calls from the AI."""
    print("\n" + "=" * 60)
    print("3. Handling Tool Calls")
    print("=" * 60)

    # Tool implementation functions
    def get_weather(location: str, unit: str = "celsius") -> dict:
        """Simulate weather API."""
        return {
            "location": location,
            "temperature": 22,
            "unit": unit,
            "condition": "Sunny",
        }

    def get_time() -> dict:
        """Get current time."""
        return {
            "time": datetime.now().strftime("%H:%M:%S"),
            "date": datetime.now().strftime("%Y-%m-%d"),
        }

    # Tool registry
    tool_handlers = {
        "get_weather": get_weather,
        "get_time": get_time,
    }

    # Configure voice with tools
    tools = [
        ToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        ),
        ToolDefinition(
            name="get_time",
            description="Get current time",
            parameters={"type": "object", "properties": {}},
        ),
    ]

    config = RealtimeConfig(tools=tools)
    voice = RealtimeVoice(config=config)

    # Handle tool calls
    @voice.on_tool_call
    async def handle_tool_call(request: ToolCallRequest):
        """Execute tool and return result."""
        print(f"\n  Tool call: {request.name}")
        print(f"  Arguments: {request.arguments}")

        # Look up and execute tool
        handler = tool_handlers.get(request.name)
        if handler:
            result = handler(**request.arguments)
            print(f"  Result: {result}")

            # Return result to AI (it will continue speaking)
            return result
        else:
            return {"error": f"Unknown tool: {request.name}"}

    print("\n  Tool call flow:")
    print("    1. User asks 'What's the weather in Tokyo?'")
    print("    2. AI decides to call get_weather tool")
    print("    3. on_tool_call handler executes")
    print("    4. Result returned to AI")
    print("    5. AI responds with weather information")


# =============================================================================
# Example 4: Complete Tool Conversation
# =============================================================================


async def complete_conversation():
    """Run a complete conversation with tools."""
    print("\n" + "=" * 60)
    print("4. Complete Tool Conversation")
    print("=" * 60)

    print("\n  Full conversation flow:")
    print("""
    # Define tools
    tools = [
        ToolDefinition(
            name="search_products",
            description="Search for products",
            parameters={...}
        ),
        ToolDefinition(
            name="add_to_cart",
            description="Add item to shopping cart",
            parameters={...}
        ),
    ]

    # Create voice with tools
    config = RealtimeConfig(tools=tools)
    voice = RealtimeVoice(config=config)

    # Handle transcripts
    @voice.on_transcript
    async def on_transcript(text: str, is_final: bool):
        if is_final:
            print(f"AI: {text}")

    # Handle tool calls
    @voice.on_tool_call
    async def on_tool_call(request: ToolCallRequest):
        if request.name == "search_products":
            return search_database(request.arguments["query"])
        elif request.name == "add_to_cart":
            return add_to_cart(request.arguments["product_id"])

    # Run conversation
    async with voice.connect() as session:
        # User says: "Find me a blue jacket"
        await session.send_audio(audio_data)

        # AI calls search_products("blue jacket")
        # Handler returns results
        # AI says: "I found 3 blue jackets..."

        # User says: "Add the first one to my cart"
        # AI calls add_to_cart(product_id)
        # AI says: "Added to your cart!"
""")


# =============================================================================
# Example 5: Async Tool Execution
# =============================================================================


async def async_tools():
    """Execute tools asynchronously."""
    print("\n" + "=" * 60)
    print("5. Async Tool Execution")
    print("=" * 60)

    print("\n  Async tool handlers for I/O operations:")
    print("""
    import httpx

    async def fetch_weather(location: str) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.weather.com/{location}"
            )
            return response.json()

    @voice.on_tool_call
    async def handle_tool(request: ToolCallRequest):
        if request.name == "get_weather":
            # Async API call
            result = await fetch_weather(request.arguments["location"])
            return result
""")

    print("\n  Benefits of async tools:")
    print("    - Non-blocking during API calls")
    print("    - Can run multiple tools concurrently")
    print("    - Better responsiveness")


# =============================================================================
# Example 6: Tool Result Types
# =============================================================================


async def tool_result_types():
    """Different types of tool results."""
    print("\n" + "=" * 60)
    print("6. Tool Result Types")
    print("=" * 60)

    print("\n  Tool results can be:")

    # Dictionary (most common)
    print("\n  1. Dictionary (JSON-serializable):")
    print("""
    return {
        "temperature": 22,
        "condition": "Sunny",
        "humidity": 65,
    }
""")

    # String
    print("  2. String:")
    print("""
    return "The current time is 2:30 PM"
""")

    # List
    print("  3. List:")
    print("""
    return [
        {"name": "Product A", "price": 29.99},
        {"name": "Product B", "price": 39.99},
    ]
""")

    # Error handling
    print("  4. Error response:")
    print("""
    try:
        result = fetch_data()
        return result
    except Exception as e:
        return {"error": str(e), "status": "failed"}
""")


# =============================================================================
# Example 7: Tool with Confirmation
# =============================================================================


async def tool_with_confirmation():
    """Tools that require user confirmation."""
    print("\n" + "=" * 60)
    print("7. Tools with User Confirmation")
    print("=" * 60)

    print("\n  Pattern for sensitive actions:")
    print("""
    pending_action = None

    @voice.on_tool_call
    async def handle_tool(request: ToolCallRequest):
        global pending_action

        if request.name == "make_purchase":
            # Store action for confirmation
            pending_action = request.arguments
            return {
                "status": "pending_confirmation",
                "message": f"Confirm purchase of {request.arguments['item']}?"
            }

        elif request.name == "confirm_action":
            if pending_action:
                # Execute the pending action
                result = execute_purchase(pending_action)
                pending_action = None
                return result
            return {"error": "No pending action"}
""")

    print("\n  The AI will ask user to confirm before executing")


# =============================================================================
# Example 8: Multiple Tool Calls
# =============================================================================


async def multiple_tools():
    """Handle multiple sequential tool calls."""
    print("\n" + "=" * 60)
    print("8. Multiple Tool Calls")
    print("=" * 60)

    print("\n  AI can call multiple tools in sequence:")
    print("""
    # User: "What's the weather and do I have any meetings today?"

    # AI calls get_weather first
    @voice.on_tool_call
    async def handle(request: ToolCallRequest):
        if request.name == "get_weather":
            return {"temp": 22, "condition": "Sunny"}

        elif request.name == "get_calendar":
            return {
                "meetings": [
                    {"time": "10:00", "title": "Team standup"},
                    {"time": "14:00", "title": "Product review"},
                ]
            }

    # AI responds: "It's 22 degrees and sunny. You have
    # two meetings today: Team standup at 10 AM and
    # Product review at 2 PM."
""")


# =============================================================================
# Example 9: Tool Error Handling
# =============================================================================


async def tool_error_handling():
    """Handle errors in tool execution."""
    print("\n" + "=" * 60)
    print("9. Tool Error Handling")
    print("=" * 60)

    print("\n  Graceful error handling:")
    print("""
    @voice.on_tool_call
    async def handle_tool(request: ToolCallRequest):
        try:
            if request.name == "get_stock_price":
                price = await fetch_stock(request.arguments["symbol"])
                return {"price": price}

        except ValueError as e:
            # User-friendly error
            return {
                "error": "invalid_symbol",
                "message": f"I couldn't find that stock symbol."
            }

        except ConnectionError:
            return {
                "error": "connection_failed",
                "message": "I'm having trouble connecting. Try again."
            }

        except Exception as e:
            # Log for debugging, return generic error
            logger.error(f"Tool error: {e}")
            return {
                "error": "unknown",
                "message": "Something went wrong. Please try again."
            }
""")

    print("\n  The AI will convey errors naturally in speech")


# =============================================================================
# Example 10: Real-World Tool Example
# =============================================================================


async def real_world_example():
    """Complete real-world voice assistant example."""
    print("\n" + "=" * 60)
    print("10. Real-World Voice Assistant")
    print("=" * 60)

    print("\n  Complete voice assistant with tools:")
    print("""
    from ai_infra.llm.realtime import RealtimeVoice, RealtimeConfig, ToolDefinition

    # Define assistant tools
    tools = [
        ToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        ),
        ToolDefinition(
            name="set_alarm",
            description="Set an alarm",
            parameters={
                "type": "object",
                "properties": {
                    "time": {"type": "string", "description": "Time like '7:30 AM'"},
                    "label": {"type": "string"},
                },
                "required": ["time"],
            },
        ),
        ToolDefinition(
            name="play_music",
            description="Play music by artist or genre",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        ),
    ]

    # Create assistant
    config = RealtimeConfig(
        tools=tools,
        system_prompt="You are a helpful voice assistant.",
    )
    voice = RealtimeVoice(config=config)

    # Tool handlers
    @voice.on_tool_call
    async def handle(request: ToolCallRequest):
        handlers = {
            "get_weather": lambda args: weather_api.get(args["location"]),
            "set_alarm": lambda args: alarm_service.set(args["time"], args.get("label")),
            "play_music": lambda args: music_player.search_and_play(args["query"]),
        }
        return await handlers[request.name](request.arguments)

    # Run assistant
    async with voice.connect() as session:
        await stream_microphone_to_session(session)
""")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Realtime Voice with Tool Calling")
    print("=" * 60)
    print("\nTools enable AI to take actions during voice conversations.\n")

    await defining_tools()
    await registering_tools()
    await handling_tool_calls()
    await complete_conversation()
    await async_tools()
    await tool_result_types()
    await tool_with_confirmation()
    await multiple_tools()
    await tool_error_handling()
    await real_world_example()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Define tools with ToolDefinition")
    print("  2. Register tools in RealtimeConfig")
    print("  3. Use @voice.on_tool_call to handle calls")
    print("  4. Return dict/string results to continue conversation")
    print("  5. Handle errors gracefully for good UX")


if __name__ == "__main__":
    asyncio.run(main())
