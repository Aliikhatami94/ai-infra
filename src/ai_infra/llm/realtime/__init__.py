"""
Realtime Voice API for ai-infra.

This module provides a unified interface for real-time voice conversations
with support for multiple providers (OpenAI, Gemini).

Quick Start:
    >>> from ai_infra.llm.realtime import RealtimeVoice, RealtimeConfig
    >>>
    >>> voice = RealtimeVoice()  # Auto-selects provider
    >>>
    >>> @voice.on_transcript
    >>> async def handle_transcript(text: str, is_final: bool):
    ...     print(f"{'Final' if is_final else 'Interim'}: {text}")
    >>>
    >>> @voice.on_audio
    >>> async def handle_audio(audio: bytes):
    ...     play_audio(audio)
    >>>
    >>> async with voice.connect() as session:
    ...     await session.send_audio(microphone_data)

Provider Selection:
    By default, RealtimeVoice auto-selects the first configured provider.
    You can also explicitly specify a provider:

    >>> voice = RealtimeVoice(provider="openai")
    >>> voice = RealtimeVoice(provider="gemini")

Configuration:
    >>> config = RealtimeConfig(
    ...     model="gpt-4o-realtime-preview",
    ...     voice="alloy",
    ...     vad_mode=VADMode.SERVER,
    ... )
    >>> voice = RealtimeVoice(config=config)
"""

from .base import BaseRealtimeProvider
from .models import (
    AudioChunk,
    AudioFormat,
    RealtimeConfig,
    RealtimeConnectionError,
    RealtimeError,
    ToolCallRequest,
    ToolDefinition,
    TranscriptDelta,
    VADMode,
    VoiceSession,
)
from .utils import (
    calculate_duration_ms,
    chunk_audio,
    float32_to_pcm16,
    pcm16_to_float32,
    resample_pcm16,
    silence_pcm16,
)
from .voice import RealtimeVoice, realtime_voice

__all__ = [
    # Main facade
    "RealtimeVoice",
    "realtime_voice",
    # Base class for providers
    "BaseRealtimeProvider",
    # Providers
    "OpenAIRealtimeProvider",
    "GeminiRealtimeProvider",
    # Configuration
    "VADMode",
    "AudioFormat",
    "RealtimeConfig",
    # Events/Messages
    "AudioChunk",
    "TranscriptDelta",
    "ToolCallRequest",
    "ToolDefinition",
    "VoiceSession",
    # Errors
    "RealtimeError",
    "RealtimeConnectionError",
    # Utilities
    "resample_pcm16",
    "chunk_audio",
    "pcm16_to_float32",
    "float32_to_pcm16",
    "calculate_duration_ms",
    "silence_pcm16",
]


def __getattr__(name: str):
    """
    Lazy import providers to avoid import errors when dependencies aren't installed.

    This allows importing from the realtime module even if specific provider
    dependencies (like websockets) aren't installed.
    """
    if name == "OpenAIRealtimeProvider":
        from .openai import OpenAIRealtimeProvider

        return OpenAIRealtimeProvider
    elif name == "GeminiRealtimeProvider":
        from .gemini import GeminiRealtimeProvider

        return GeminiRealtimeProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _register_providers() -> None:
    """Auto-register all available providers."""
    try:
        from . import openai as _openai  # noqa: F401
    except ImportError:
        pass
    try:
        from . import gemini as _gemini  # noqa: F401
    except ImportError:
        pass


# Auto-register providers on import
_register_providers()
