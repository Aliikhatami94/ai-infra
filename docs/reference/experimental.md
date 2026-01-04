# Experimental APIs

This page lists APIs that are considered experimental. Experimental APIs may change, be renamed, or be removed in minor versions without following the normal deprecation policy.

## What Does "Experimental" Mean?

Experimental APIs are:
- **Functional** and can be used in production at your own risk
- **Subject to change** without the standard 2-version deprecation period
- **Not yet battle-tested** in large-scale production environments
- **Seeking feedback** from early adopters

## Current Experimental APIs

### Realtime Voice API

**Status**: Experimental (since v0.1.150)

The Realtime Voice API provides speech-to-speech capabilities with streaming audio.

```python
from ai_infra import RealtimeVoice, RealtimeConfig, VADMode

voice = RealtimeVoice()
async with voice.connect() as session:
    await session.send_audio(audio_bytes)
```

**Why experimental**:
- Provider APIs (OpenAI, Gemini) are themselves in preview
- Audio streaming edge cases still being discovered
- VAD (Voice Activity Detection) modes need more tuning

**Exports**:
- `RealtimeVoice`
- `realtime_voice` (context manager)
- `RealtimeConfig`
- `VADMode`

### Image Generation API

**Status**: Experimental (since v0.1.160)

The ImageGen API provides image generation across multiple providers.

```python
from ai_infra import ImageGen

gen = ImageGen()
image = await gen.generate("A sunset over mountains")
```

**Why experimental**:
- Multi-provider support is new
- Model capabilities vary significantly between providers
- Prompt optimization strategies still evolving

**Exports**:
- `ImageGen`
- `GeneratedImage`
- `ImageGenProvider`

### DeepAgent (Nested Agents)

**Status**: Experimental (since v0.1.155)

DeepAgent enables hierarchical agent orchestration where agents can spawn sub-agents.

**Why experimental**:
- Recursion depth limits need tuning
- Token budget management across agent tree is complex
- Error propagation patterns still being refined

**Usage**: Available via `Agent` class with `deep=True` configuration.

## Stability Tiers

| Tier | Meaning | Deprecation Policy |
|------|---------|-------------------|
| **Stable** | Production-ready, fully tested | 2+ minor versions notice |
| **Experimental** | Functional but may change | May change in any release |
| **Internal** | Not exported, implementation detail | No guarantees |

## Providing Feedback

If you're using experimental APIs, we want to hear from you:

- GitHub Issues: Report bugs or suggest improvements
- Discussions: Share use cases and patterns that work well

## Graduating to Stable

Experimental APIs graduate to stable when:

1. No breaking changes for 2+ minor versions
2. Comprehensive test coverage (>80%)
3. Production usage validates the design
4. Documentation is complete with examples
