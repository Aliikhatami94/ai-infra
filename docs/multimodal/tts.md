# Text-to-Speech (TTS)

> Convert text to natural-sounding speech with multiple providers.

## Quick Start

```python
from ai_infra import TTS

tts = TTS()
audio = tts.speak("Hello, world!")
audio.save("output.mp3")
```

---

## Supported Providers

| Provider | Voices | Models | Streaming |
|----------|--------|--------|-----------|
| OpenAI | 6 voices | tts-1, tts-1-hd | [OK] |
| ElevenLabs | 10+ voices | eleven_multilingual_v2 | [OK] |
| Google | 5 voices | Multiple | [OK] |

---

## Basic Usage

### Auto-Detect Provider

```python
from ai_infra import TTS

# Auto-detects from env vars (OpenAI -> ElevenLabs -> Google)
tts = TTS()
audio = tts.speak("Hello!")
```

### Explicit Provider

```python
tts = TTS(provider="elevenlabs")
audio = tts.speak("Hello from ElevenLabs!")
```

---

## Voice Selection

### List Available Voices

```python
from ai_infra import TTS

# Get voices for current provider
voices = TTS.list_voices()
# ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']

# Get voices for specific provider
voices = TTS.list_voices(provider="elevenlabs")
# ['Rachel', 'Domi', 'Bella', 'Antoni', ...]
```

### Use Specific Voice

```python
tts = TTS(provider="openai", voice="nova")
audio = tts.speak("Using the Nova voice")

# Or per-call
audio = tts.speak("Different voice", voice="echo")
```

---

## OpenAI TTS

```python
from ai_infra import TTS

tts = TTS(provider="openai")

# Available voices: alloy, echo, fable, onyx, nova, shimmer
audio = tts.speak("Hello!", voice="alloy")

# HD quality model
audio = tts.speak("HD quality", model="tts-1-hd")

# Standard quality (faster)
audio = tts.speak("Standard quality", model="tts-1")
```

---

## ElevenLabs TTS

```python
from ai_infra import TTS

tts = TTS(provider="elevenlabs")

# High-quality multilingual
audio = tts.speak("Hello!", voice="Rachel")

# Specify model
audio = tts.speak(
    "Multilingual support",
    model="eleven_multilingual_v2"
)
```

---

## Google TTS

```python
from ai_infra import TTS

tts = TTS(provider="google_genai")

audio = tts.speak("Hello from Google!", voice="Puck")
```

---

## Audio Output

### Save to File

```python
audio = tts.speak("Hello!")

# Save as MP3
audio.save("output.mp3")

# Save as WAV
audio.save("output.wav", format="wav")
```

### Get Raw Bytes

```python
audio = tts.speak("Hello!")

# Get audio bytes
audio_bytes = audio.data

# Get format info
print(audio.format)  # 'mp3'
print(audio.sample_rate)  # 24000
```

### Stream Audio

```python
# Stream chunks as they're generated
for chunk in tts.stream("Long text to speak..."):
    play_audio_chunk(chunk)
```

---

## Async Usage

```python
import asyncio
from ai_infra import TTS

async def main():
    tts = TTS()
    audio = await tts.aspeak("Async speech!")
    audio.save("output.mp3")

asyncio.run(main())
```

---

## Configuration

```python
tts = TTS(
    provider="openai",
    voice="alloy",
    model="tts-1-hd",
    speed=1.0,  # 0.25 to 4.0
)
```

---

## Error Handling

```python
from ai_infra import TTS
from ai_infra.errors import AIInfraError, ProviderError

try:
    tts = TTS()
    audio = tts.speak("Hello!")
except ProviderError as e:
    print(f"Provider error: {e}")
except AIInfraError as e:
    print(f"TTS error: {e}")
```

---

## See Also

- [STT](stt.md) - Speech-to-text
- [Realtime](realtime.md) - Real-time voice conversations
- [Providers](../core/providers.md) - Provider configuration
