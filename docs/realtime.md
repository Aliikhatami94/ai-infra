# Realtime Voice API

The Realtime Voice API provides streaming voice conversations with LLMs, supporting both OpenAI and Google Gemini providers. It offers real-time bidirectional audio streaming, voice activity detection (VAD), and tool integration.

## Quick Start

```python
from ai_infra.llm.realtime import RealtimeVoice, RealtimeConfig

# Create a voice instance (auto-selects provider based on environment)
voice = RealtimeVoice()

# Register callbacks
@voice.on_audio
async def handle_audio(audio: bytes):
    play_audio(audio)  # Your audio playback function

@voice.on_transcript
async def handle_transcript(text: str, is_final: bool):
    print(f"{'Final' if is_final else 'Partial'}: {text}")

# Start a conversation
async with voice.connect() as session:
    await session.send_audio(microphone_data)
```

## Provider Selection

The RealtimeVoice facade automatically selects a provider based on environment variables:

| Priority | Provider | Environment Variable |
|----------|----------|---------------------|
| 1 | OpenAI | `OPENAI_API_KEY` |
| 2 | Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |

### Explicit Provider Selection

```python
# Force OpenAI
voice = RealtimeVoice(provider="openai")

# Force Gemini
voice = RealtimeVoice(provider="gemini")
```

### Check Available Providers

```python
from ai_infra.llm.realtime import RealtimeVoice

# List all registered providers
print(RealtimeVoice.available_providers())  # ['openai', 'gemini']

# List configured providers (with API keys)
print(RealtimeVoice.configured_providers())  # ['openai'] if only OpenAI key set
```

## Configuration

Use `RealtimeConfig` to customize the voice session:

```python
from ai_infra.llm.realtime import RealtimeConfig, VADMode

config = RealtimeConfig(
    # Model settings
    model="gpt-4o-realtime-preview",  # or "gemini-2.0-flash-exp"
    voice="alloy",  # OpenAI: alloy, echo, shimmer; Gemini: Puck, Charon, etc.

    # System prompt
    instructions="You are a helpful voice assistant.",

    # Voice Activity Detection
    vad_mode=VADMode.SERVER,  # SERVER (auto) or MANUAL
    vad_threshold=0.5,
    vad_prefix_padding_ms=300,
    vad_silence_duration_ms=500,

    # Generation settings
    temperature=0.8,
    max_tokens=4096,

    # Tool functions
    tools=[my_function],
)

voice = RealtimeVoice(config=config)
```

### VAD Modes

- **`VADMode.SERVER`** (default): Provider automatically detects speech segments
- **`VADMode.MANUAL`**: You control when speech input begins/ends

### Available Voices

#### OpenAI
- `alloy` - Neutral and balanced
- `echo` - Natural and conversational
- `shimmer` - Warm and expressive

#### Gemini
- `Puck` - Upbeat and friendly
- `Charon` - Informative and clear
- `Kore` - Firm and direct
- `Fenrir` - Excitable and energetic
- `Aoede` - Breezy and personable

## Callbacks

Register callbacks to handle events from the voice session:

### Audio Output

```python
@voice.on_audio
async def handle_audio(audio: bytes):
    """Receive audio chunks from the assistant."""
    play_audio(audio)
```

### Transcripts

```python
@voice.on_transcript
async def handle_transcript(text: str, is_final: bool):
    """Receive text transcriptions.

    Args:
        text: The transcribed text
        is_final: True if this is the final version (no more updates)
    """
    if is_final:
        print(f"Assistant: {text}")
    else:
        print(f"(typing...): {text}")
```

### Tool Calls

```python
from ai_infra.llm.realtime import ToolCallRequest

@voice.on_tool_call
async def handle_tool(request: ToolCallRequest):
    """Handle tool/function calls from the assistant.

    Args:
        request: Contains function_name, call_id, and arguments

    Returns:
        The result to send back to the assistant
    """
    if request.function_name == "get_weather":
        city = request.arguments.get("city", "Unknown")
        return f"Weather in {city}: Sunny, 72°F"
```

### Errors

```python
from ai_infra.llm.realtime import RealtimeError

@voice.on_error
async def handle_error(error: RealtimeError):
    """Handle errors during the session."""
    print(f"Error: {error.message}")
```

### Interruptions

```python
@voice.on_interrupted
async def handle_interrupted():
    """Called when assistant speech is interrupted."""
    stop_audio_playback()
```

## Tool Integration

The Realtime API supports the same tool pattern as the Agent class:

```python
from ai_infra.llm.realtime import RealtimeVoice, RealtimeConfig

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

config = RealtimeConfig(
    instructions="Use tools when asked for weather or search.",
    tools=[get_weather, search_web],
)

voice = RealtimeVoice(config=config)

# Tools are automatically executed when the assistant calls them
async with voice.connect() as session:
    await session.send_text("What's the weather in Tokyo?")
```

### Custom Tool Handling

For more control, use the `on_tool_call` callback:

```python
tool_results = []

@voice.on_tool_call
async def custom_tool_handler(request: ToolCallRequest):
    # Log the call
    tool_results.append(request)

    # Execute and return result
    if request.function_name == "get_weather":
        return get_weather(**request.arguments)

    return "Tool not found"
```

## Streaming Audio

### Sending Audio Input

```python
async with voice.connect() as session:
    # Send audio chunk
    await session.send_audio(audio_bytes)

    # Stream from microphone
    async for chunk in microphone_stream():
        await session.send_audio(chunk)
```

### Using the run() Method

For simpler streaming, use the `run()` method:

```python
from ai_infra.llm.realtime import AudioChunk, TranscriptDelta

async for event in voice.run(microphone_stream()):
    if isinstance(event, AudioChunk):
        play_audio(event.data)
    elif isinstance(event, TranscriptDelta):
        print(event.text, end="", flush=True)
```

## Direct Provider Access

For advanced use cases, access providers directly:

### OpenAI Provider

```python
from ai_infra.llm.realtime import OpenAIRealtimeProvider, RealtimeConfig

config = RealtimeConfig(
    model="gpt-4o-realtime-preview",
    voice="alloy",
)

provider = OpenAIRealtimeProvider(config=config)

async def on_transcript(text: str, is_final: bool):
    print(text)

provider.on_transcript(on_transcript)

session = await provider.connect()
try:
    await session.send_text("Hello!")
    await asyncio.sleep(5)
finally:
    await provider.disconnect()
```

### Gemini Provider

```python
from ai_infra.llm.realtime import GeminiRealtimeProvider, RealtimeConfig

config = RealtimeConfig(
    model="gemini-2.0-flash-exp",
    voice="Puck",
)

provider = GeminiRealtimeProvider(config=config)

async def on_audio(audio: bytes):
    play_audio(audio)

provider.on_audio(on_audio)

session = await provider.connect()
try:
    await session.send_audio(audio_data)
    await asyncio.sleep(5)
finally:
    await provider.disconnect()
```

## Error Handling

```python
from ai_infra.llm.realtime import (
    RealtimeError,
    RealtimeConnectionError,
)

try:
    async with voice.connect() as session:
        await session.send_audio(audio_data)
except RealtimeConnectionError as e:
    print(f"Connection failed: {e}")
except RealtimeError as e:
    print(f"Realtime error: {e}")
```

## Data Models

### AudioChunk

```python
from ai_infra.llm.realtime import AudioChunk

chunk = AudioChunk(
    data=b"...",  # Raw audio bytes
    duration_ms=100,
    is_final=False,
)
```

### TranscriptDelta

```python
from ai_infra.llm.realtime import TranscriptDelta

delta = TranscriptDelta(
    text="Hello there",
    is_final=True,
    speaker="assistant",  # or "user" for input transcription
)
```

### ToolCallRequest

```python
from ai_infra.llm.realtime import ToolCallRequest

request = ToolCallRequest(
    call_id="call_123",
    function_name="get_weather",
    arguments={"city": "Tokyo"},
)
```

### ToolDefinition

```python
from ai_infra.llm.realtime import ToolDefinition

tool = ToolDefinition(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"],
    },
)
```

## Provider Comparison

| Feature | OpenAI | Gemini |
|---------|--------|--------|
| Model | `gpt-4o-realtime-preview` | `gemini-2.0-flash-exp` |
| WebSocket Protocol | Custom | Custom |
| Audio Format | PCM16 @ 24kHz | PCM16 @ 16kHz |
| Server VAD | ✅ | ✅ |
| Tool Calling | ✅ | ✅ |
| Interruption | ✅ | ✅ |
| Input Transcription | ✅ | Limited |
| Pricing | Pay per minute | Pay per minute |

### When to Use OpenAI

- Need highest quality voice output
- Require reliable input transcription
- Using other OpenAI services (embeddings, vision)

### When to Use Gemini

- Cost optimization
- Google Cloud ecosystem
- Experimental features

## Best Practices

1. **Always clean up**: Use async context managers or try/finally
2. **Handle errors**: Register error callbacks
3. **Buffer audio**: Don't send very small chunks (aim for 100-200ms)
4. **Test with text**: Use `send_text()` during development before audio
5. **Monitor latency**: Add timestamps to track round-trip time

## Examples

### Voice Assistant

```python
import asyncio
from ai_infra.llm.realtime import RealtimeVoice, RealtimeConfig

async def voice_assistant():
    config = RealtimeConfig(
        instructions="You are a helpful voice assistant. Be concise.",
        voice="alloy",
    )

    voice = RealtimeVoice(config=config)

    @voice.on_audio
    async def play(audio: bytes):
        # Your audio playback implementation
        pass

    @voice.on_transcript
    async def show(text: str, is_final: bool):
        if is_final:
            print(f"Assistant: {text}")

    async with voice.connect() as session:
        # Simple text-based testing
        await session.send_text("What's 2 + 2?")
        await asyncio.sleep(10)

asyncio.run(voice_assistant())
```

### With Tools

```python
from ai_infra.llm.realtime import RealtimeVoice, RealtimeConfig

def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

async def math_assistant():
    config = RealtimeConfig(
        instructions="You are a math assistant. Use the calculator tool.",
        tools=[calculator],
    )

    voice = RealtimeVoice(config=config)

    @voice.on_transcript
    async def show(text: str, is_final: bool):
        if is_final:
            print(f"Assistant: {text}")

    async with voice.connect() as session:
        await session.send_text("What's 15 * 23?")
        await asyncio.sleep(10)
```

## Related

- [Agent](agent.md) - Non-realtime LLM agent
- [Callbacks](callbacks.md) - Callback patterns
- [Errors](errors.md) - Error handling
