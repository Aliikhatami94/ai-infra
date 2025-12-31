# Speech-to-Text (STT)

> Transcribe audio to text with multiple providers.

## Quick Start

```python
from ai_infra import STT

stt = STT()
result = stt.transcribe("audio.mp3")
print(result.text)  # "Hello, world!"
```

---

## Supported Providers

| Provider | Models | Languages | Streaming |
|----------|--------|-----------|-----------|
| OpenAI | whisper-1 | 50+ | [X] |
| Deepgram | nova-2, nova | 30+ | [OK] |
| Google | Multiple | 100+ | [OK] |

---

## Basic Usage

### Auto-Detect Provider

```python
from ai_infra import STT

# Auto-detects from env vars (OpenAI -> Deepgram -> Google)
stt = STT()
result = stt.transcribe("recording.mp3")
print(result.text)
```

### Explicit Provider

```python
stt = STT(provider="deepgram")
result = stt.transcribe("recording.mp3")
```

---

## Input Formats

### From File

```python
stt = STT()

# MP3, WAV, M4A, WEBM, and more
result = stt.transcribe("audio.mp3")
result = stt.transcribe("audio.wav")
result = stt.transcribe("audio.m4a")
```

### From Bytes

```python
with open("audio.mp3", "rb") as f:
    audio_bytes = f.read()

result = stt.transcribe(audio_bytes)
```

### From URL

```python
result = stt.transcribe("https://example.com/audio.mp3")
```

---

## OpenAI Whisper

```python
from ai_infra import STT

stt = STT(provider="openai")
result = stt.transcribe("audio.mp3")

# Whisper-1 is the only model
print(result.text)
print(result.language)  # Detected language
```

### With Language Hint

```python
result = stt.transcribe("audio.mp3", language="es")
```

---

## Deepgram

```python
from ai_infra import STT

stt = STT(provider="deepgram")

# High accuracy
result = stt.transcribe("audio.mp3", model="nova-2")

# With features
result = stt.transcribe(
    "audio.mp3",
    punctuate=True,
    diarize=True,  # Speaker detection
    smart_format=True,
)
```

### Streaming with Deepgram

```python
async def transcribe_stream(audio_stream):
    stt = STT(provider="deepgram")

    async for result in stt.stream(audio_stream):
        print(result.text, end="", flush=True)
```

---

## Google STT

```python
from ai_infra import STT

stt = STT(provider="google_genai")
result = stt.transcribe("audio.mp3")
```

---

## Transcription Result

```python
result = stt.transcribe("audio.mp3")

# Basic result
print(result.text)      # Full transcription
print(result.language)  # Detected language code

# Detailed result (if available)
print(result.confidence)  # Confidence score
print(result.duration)    # Audio duration in seconds
print(result.words)       # Word-level timestamps (if supported)
```

### Word Timestamps

```python
result = stt.transcribe("audio.mp3", timestamps=True)

for word in result.words:
    print(f"{word.text}: {word.start}s - {word.end}s")
```

---

## Async Usage

```python
import asyncio
from ai_infra import STT

async def main():
    stt = STT()
    result = await stt.atranscribe("audio.mp3")
    print(result.text)

asyncio.run(main())
```

---

## Configuration

```python
stt = STT(
    provider="deepgram",
    model="nova-2",
    language="en",
)
```

---

## Error Handling

```python
from ai_infra import STT
from ai_infra.errors import AIInfraError, ProviderError

try:
    stt = STT()
    result = stt.transcribe("audio.mp3")
except ProviderError as e:
    print(f"Provider error: {e}")
except AIInfraError as e:
    print(f"STT error: {e}")
```

---

## See Also

- [TTS](tts.md) - Text-to-speech
- [Realtime](realtime.md) - Real-time voice conversations
- [Providers](../core/providers.md) - Provider configuration
