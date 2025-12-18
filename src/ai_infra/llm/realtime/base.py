"""Abstract base class for Realtime Voice providers.

This module defines the interface that all realtime voice providers
must implement. New providers (e.g., Anthropic when available) should
subclass BaseRealtimeProvider.

The design follows ai-infra patterns:
- Auto-provider detection based on environment (like TTS/STT)
- Callback-based event handling
- Async context manager for connection lifecycle
- svc-infra WebSocket integration where available
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_infra.llm.realtime.models import (
        AudioChunk,
        RealtimeConfig,
        RealtimeError,
        ToolCallRequest,
        TranscriptDelta,
        VoiceSession,
    )

logger = logging.getLogger(__name__)

# Type aliases for callbacks
AudioCallback = Callable[[bytes], Awaitable[None]]
TranscriptCallback = Callable[[str, bool], Awaitable[None]]
ToolCallCallback = Callable[["ToolCallRequest"], Awaitable[Any]]
ErrorCallback = Callable[["RealtimeError"], Awaitable[None]]
InterruptedCallback = Callable[[], Awaitable[None]]


class BaseRealtimeProvider(ABC):
    """Abstract base class for realtime voice providers.

    Implement this to add a new provider (OpenAI, Gemini, etc.).

    All providers must implement:
    - connect(): Establish WebSocket connection, return VoiceSession
    - disconnect(): Close connection
    - send_audio(): Send audio input
    - run(): Streaming conversation loop
    - Static methods for provider discovery

    Example implementation:
        ```python
        class MyProvider(BaseRealtimeProvider):
            @property
            def provider_name(self) -> str:
                return "myprovider"

            @staticmethod
            def is_configured() -> bool:
                return bool(os.environ.get("MYPROVIDER_API_KEY"))

            async def connect(self) -> VoiceSession:
                self._ws = await connect_websocket(...)
                return MyVoiceSession(self._ws, self.config, self)

            async def send_audio(self, audio: bytes) -> None:
                await self._session.send_audio(audio)
        ```
    """

    def __init__(self, config: RealtimeConfig | None = None):
        """Initialize the provider.

        Args:
            config: Session configuration. If None, uses defaults.
        """
        from ai_infra.llm.realtime.models import RealtimeConfig

        self.config = config or RealtimeConfig()

        # Callback lists (multiple callbacks allowed per event type)
        self._audio_callbacks: list[AudioCallback] = []
        self._transcript_callbacks: list[TranscriptCallback] = []
        self._tool_call_callbacks: list[ToolCallCallback] = []
        self._error_callbacks: list[ErrorCallback] = []
        self._interrupted_callbacks: list[InterruptedCallback] = []

    # =========================================================================
    # Abstract Properties & Methods - Must implement per provider
    # =========================================================================

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'openai', 'gemini').

        This is used for provider discovery and selection.
        """
        ...

    @staticmethod
    @abstractmethod
    def is_configured() -> bool:
        """Check if this provider has required API keys set.

        Returns True if the provider can be used (API key available).
        """
        ...

    @staticmethod
    @abstractmethod
    def list_models() -> list[str]:
        """List available realtime models for this provider.

        Returns:
            List of model names (e.g., ["gpt-4o-realtime-preview", ...])
        """
        ...

    @staticmethod
    @abstractmethod
    def get_default_model() -> str:
        """Return the default model for this provider.

        This is used when no model is explicitly specified.
        """
        ...

    @abstractmethod
    async def connect(self) -> VoiceSession:
        """Connect to the provider's WebSocket endpoint.

        Establishes the WebSocket connection and returns a VoiceSession
        that can be used to interact with the realtime API.

        Returns:
            VoiceSession instance for the active connection.

        Raises:
            RealtimeConnectionError: If connection fails.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the provider.

        Should clean up resources and close the WebSocket.
        """
        ...

    @abstractmethod
    async def send_audio(self, audio: bytes) -> None:
        """Send audio input to the provider.

        Args:
            audio: Raw audio bytes (PCM16 format).
                   Sample rate depends on provider (16kHz or 24kHz).
        """
        ...

    @abstractmethod
    def run(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[AudioChunk | TranscriptDelta]:
        """Run a streaming conversation.

        This method handles the full lifecycle:
        1. Connect to the provider
        2. Stream audio input from the iterator
        3. Yield audio/transcript events
        4. Disconnect when done

        Args:
            audio_stream: Async iterator of audio bytes from microphone.

        Yields:
            AudioChunk and TranscriptDelta events from the conversation.

        Example:
            ```python
            async for event in provider.run(microphone_stream()):
                if isinstance(event, AudioChunk):
                    play_audio(event.data)
                elif isinstance(event, TranscriptDelta):
                    print(event.text)
            ```
        """
        ...

    # =========================================================================
    # Callback Registration - Shared across providers
    # =========================================================================

    def on_audio(self, callback: AudioCallback) -> AudioCallback:
        """Register a callback for audio output.

        Args:
            callback: Async function receiving audio bytes.

        Returns:
            The callback (for decorator usage).
        """
        self._audio_callbacks.append(callback)
        return callback

    def on_transcript(self, callback: TranscriptCallback) -> TranscriptCallback:
        """Register a callback for transcript updates.

        Args:
            callback: Async function receiving (text, is_final).

        Returns:
            The callback (for decorator usage).
        """
        self._transcript_callbacks.append(callback)
        return callback

    def on_tool_call(self, callback: ToolCallCallback) -> ToolCallCallback:
        """Register a callback for tool calls.

        Args:
            callback: Async function receiving ToolCallRequest, returning result.

        Returns:
            The callback (for decorator usage).
        """
        self._tool_call_callbacks.append(callback)
        return callback

    def on_error(self, callback: ErrorCallback) -> ErrorCallback:
        """Register a callback for errors.

        Args:
            callback: Async function receiving RealtimeError.

        Returns:
            The callback (for decorator usage).
        """
        self._error_callbacks.append(callback)
        return callback

    def on_interrupted(self, callback: InterruptedCallback) -> InterruptedCallback:
        """Register a callback for interruptions.

        Args:
            callback: Async function called when user interrupts.

        Returns:
            The callback (for decorator usage).
        """
        self._interrupted_callbacks.append(callback)
        return callback

    # =========================================================================
    # Dispatch Helpers - Call registered callbacks
    # =========================================================================

    async def _dispatch_audio(self, audio: bytes) -> None:
        """Dispatch audio to all registered callbacks."""
        for callback in self._audio_callbacks:
            try:
                await callback(audio)
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")

    async def _dispatch_transcript(self, text: str, is_final: bool) -> None:
        """Dispatch transcript to all registered callbacks."""
        for callback in self._transcript_callbacks:
            try:
                await callback(text, is_final)
            except Exception as e:
                logger.error(f"Error in transcript callback: {e}")

    async def _dispatch_tool_call(self, request: ToolCallRequest) -> Any:
        """Dispatch tool call to first registered callback."""
        for callback in self._tool_call_callbacks:
            try:
                return await callback(request)
            except Exception as e:
                logger.error(f"Error in tool call callback: {e}")
                raise
        return None

    async def _dispatch_error(self, error: RealtimeError) -> None:
        """Dispatch error to all registered callbacks."""
        for callback in self._error_callbacks:
            try:
                await callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    async def _dispatch_interrupted(self) -> None:
        """Dispatch interruption to all registered callbacks."""
        for callback in self._interrupted_callbacks:
            try:
                await callback()
            except Exception as e:
                logger.error(f"Error in interrupted callback: {e}")

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    @asynccontextmanager
    async def session(self):
        """Context manager for a voice session.

        Example:
            ```python
            async with provider.session() as session:
                await session.send_audio(audio_data)
            ```
        """
        session = await self.connect()
        try:
            yield session
        finally:
            await self.disconnect()
