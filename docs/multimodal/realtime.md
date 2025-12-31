# Realtime Voice API

> Speech-to-speech conversations with real-time streaming.

## Quick Start

### Simplest Example

```python
from ai_infra import RealtimeVoice

voice = RealtimeVoice()  # Auto-detects provider

async with voice.connect() as session:
    # Send audio, get audio back
    response = await session.send_audio(audio_bytes)
    play_audio(response.audio)
```

### With Event Handlers (Recommended)

For real-time applications, use event handlers:

```python
from ai_infra import RealtimeVoice

voice = RealtimeVoice()

@voice.on_transcript
async def handle_transcript(text: str, is_final: bool):
    print(f"{'>' if is_final else '...'} {text}")

@voice.on_audio
async def handle_audio(audio: bytes):
    play_audio(audio)

async with voice.connect() as session:
    await session.send_audio(microphone_data)
```

---

## Supported Providers

| Provider | Models | VAD | Tools |
|----------|--------|-----|-------|
| OpenAI | gpt-4o-realtime-preview | [OK] | [OK] |
| Google | gemini-2.0-flash-exp | [OK] | [OK] |

---

## Basic Usage

### Auto-Detect Provider

```python
from ai_infra import RealtimeVoice

# Auto-detects from env vars (OpenAI -> Google)
voice = RealtimeVoice()
```

### Explicit Provider

```python
# OpenAI Realtime
voice = RealtimeVoice(provider="openai")

# Google Gemini Live
voice = RealtimeVoice(provider="gemini")
```

---

## Configuration

```python
from ai_infra import RealtimeVoice, RealtimeConfig, VADMode

config = RealtimeConfig(
    model="gpt-4o-realtime-preview",
    voice="alloy",
    vad_mode=VADMode.SERVER,  # Server-side voice detection
    system_prompt="You are a helpful assistant.",
)

voice = RealtimeVoice(config=config)
```

### VAD Modes

```python
from ai_infra import VADMode

# Server handles voice activity detection
VADMode.SERVER

# Client handles voice activity detection
VADMode.CLIENT

# No automatic detection (manual control)
VADMode.NONE
```

---

## Event Handlers

### Transcript Events

```python
@voice.on_transcript
async def handle_transcript(text: str, is_final: bool):
    """Called when transcript text is received."""
    if is_final:
        print(f"Final: {text}")
    else:
        print(f"Partial: {text}")
```

### Audio Events

```python
@voice.on_audio
async def handle_audio(audio: bytes):
    """Called when audio response is received."""
    # Play audio or buffer for later
    audio_queue.put(audio)
```

### Tool Call Events

```python
@voice.on_tool_call
async def handle_tool_call(tool_name: str, args: dict) -> str:
    """Called when model wants to use a tool."""
    if tool_name == "get_weather":
        return get_weather(args["city"])
    return "Unknown tool"
```

### Connection Events

```python
@voice.on_connect
async def handle_connect():
    print("Connected!")

@voice.on_disconnect
async def handle_disconnect():
    print("Disconnected")

@voice.on_error
async def handle_error(error: Exception):
    print(f"Error: {error}")
```

---

## Voice Session

### Connect and Send Audio

```python
async with voice.connect() as session:
    # Send audio chunks
    while recording:
        chunk = get_microphone_chunk()
        await session.send_audio(chunk)

    # Signal end of speech (optional with VAD)
    await session.end_turn()
```

### Send Text

```python
async with voice.connect() as session:
    # Send text instead of audio
    await session.send_text("Hello, how are you?")
```

### Interrupt

```python
async with voice.connect() as session:
    # Stop current response
    await session.interrupt()
```

---

## Tool Integration

Use existing agent tools with realtime:

```python
from ai_infra import RealtimeVoice

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: 72°F"

def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

voice = RealtimeVoice(tools=[get_weather, search])

@voice.on_tool_call
async def handle_tool(name: str, args: dict) -> str:
    if name == "get_weather":
        return get_weather(args["city"])
    elif name == "search":
        return search(args["query"])
```

---

## Provider-Specific

### OpenAI Realtime

```python
from ai_infra import RealtimeVoice

voice = RealtimeVoice(provider="openai")

# Available voices: alloy, ash, ballad, coral, echo, sage, shimmer, verse
# Models: gpt-4o-realtime-preview, gpt-4o-realtime-preview-2024-10-01
```

### Google Gemini Live

```python
voice = RealtimeVoice(provider="gemini")

# Available voices: Puck, Charon, Kore, Fenrir, Aoede
# Models: gemini-2.0-flash-exp, gemini-2.0-flash-thinking-exp
```

---

## Discovery

```python
from ai_infra import RealtimeVoice

# List available providers
providers = RealtimeVoice.list_providers()
# ['openai', 'google_genai']

# List configured providers (have API keys)
configured = RealtimeVoice.list_configured_providers()
# ['openai']

# List models for a provider
models = RealtimeVoice.list_models("openai")
# ['gpt-4o-realtime-preview', ...]

# List voices for a provider
voices = RealtimeVoice.list_voices("openai")
# ['alloy', 'ash', 'ballad', ...]
```

---

## FastAPI Integration

```python
from fastapi import FastAPI, WebSocket
from ai_infra import RealtimeVoice

app = FastAPI()

@app.websocket("/voice")
async def voice_endpoint(websocket: WebSocket):
    await websocket.accept()

    voice = RealtimeVoice()

    @voice.on_audio
    async def send_audio(audio: bytes):
        await websocket.send_bytes(audio)

    @voice.on_transcript
    async def send_transcript(text: str, is_final: bool):
        await websocket.send_json({"text": text, "final": is_final})

    async with voice.connect() as session:
        while True:
            data = await websocket.receive_bytes()
            await session.send_audio(data)
```

---

## Error Handling

```python
from ai_infra import RealtimeVoice
from ai_infra.llm.realtime import RealtimeError, RealtimeConnectionError

try:
    voice = RealtimeVoice()
    async with voice.connect() as session:
        await session.send_audio(audio)
except RealtimeConnectionError as e:
    print(f"Connection failed: {e}")
except RealtimeError as e:
    print(f"Realtime error: {e}")
```

---

## Audio Utilities

```python
from ai_infra.llm.realtime.utils import (
    resample_pcm16,
    chunk_audio,
    pcm16_to_float32,
    float32_to_pcm16,
)

# Resample audio to required rate
resampled = resample_pcm16(audio_bytes, from_rate=44100, to_rate=24000)

# Split audio into chunks
chunks = chunk_audio(audio_bytes, chunk_size=4096)

# Convert between formats
float_audio = pcm16_to_float32(pcm16_bytes)
pcm16_audio = float32_to_pcm16(float_array)
```

---

## See Also

- [TTS](tts.md) - Text-to-speech (non-realtime)
- [STT](stt.md) - Speech-to-text (non-realtime)
- [Agent](../core/agents.md) - Tool-calling agents
- [Providers](../core/providers.md) - Provider configuration
