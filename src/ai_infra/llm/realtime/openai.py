"""
OpenAI Realtime Voice API provider.

This module implements the OpenAI Realtime API for real-time voice conversations
using WebSocket connections. It integrates with svc-infra's WebSocket client.

The OpenAI Realtime API provides:
- Low-latency voice-to-voice conversations
- Server-side Voice Activity Detection (VAD)
- Function/tool calling during conversations
- Multiple voice options (alloy, echo, fable, onyx, nova, shimmer)

Example:
    >>> from ai_infra.llm.realtime import RealtimeVoice, RealtimeConfig
    >>>
    >>> voice = RealtimeVoice(provider="openai")
    >>>
    >>> @voice.on_transcript
    >>> async def on_transcript(text: str, is_final: bool):
    ...     print(text)
    >>>
    >>> async with voice.connect() as session:
    ...     await session.send_audio(audio_bytes)

References:
    - https://platform.openai.com/docs/guides/realtime
    - https://platform.openai.com/docs/api-reference/realtime
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from ai_infra.llm.realtime.base import BaseRealtimeProvider
from ai_infra.llm.realtime.models import (
    AudioChunk,
    RealtimeConfig,
    RealtimeConnectionError,
    RealtimeError,
    ToolCallRequest,
    TranscriptDelta,
    VADMode,
    VoiceSession,
)
from ai_infra.llm.realtime.voice import RealtimeVoice
from ai_infra.providers import ProviderCapability, ProviderRegistry

logger = logging.getLogger(__name__)

# OpenAI Realtime API configuration
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"


def _get_openai_realtime_config():
    """Get OpenAI realtime config from registry."""
    config = ProviderRegistry.get("openai")
    if config:
        return config.get_capability(ProviderCapability.REALTIME)
    return None


# Get models and voices from registry (with fallbacks)
_realtime_cap = _get_openai_realtime_config()
OPENAI_REALTIME_MODELS = (
    _realtime_cap.models
    if _realtime_cap and _realtime_cap.models
    else [
        "gpt-4o-realtime-preview",
        "gpt-4o-realtime-preview-2024-10-01",
        "gpt-4o-realtime-preview-2024-12-17",
    ]
)
OPENAI_REALTIME_VOICES = (
    _realtime_cap.voices
    if _realtime_cap and _realtime_cap.voices
    else ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]
)
DEFAULT_MODEL = (
    _realtime_cap.default_model
    if _realtime_cap and _realtime_cap.default_model
    else "gpt-4o-realtime-preview"
)


class OpenAIVoiceSession(VoiceSession):
    """
    VoiceSession implementation for OpenAI Realtime API.

    Wraps the WebSocket connection and provides the VoiceSession interface.
    """

    def __init__(
        self,
        ws: Any,  # WebSocketClient from svc-infra
        config: RealtimeConfig,
        provider: OpenAIRealtimeProvider,
    ):
        self._ws = ws
        self._config = config
        self._provider = provider
        self._session_id: str | None = None
        self._closed = False

    @property
    def session_id(self) -> str:
        """Get the session ID."""
        return self._session_id or ""

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return not self._closed and self._ws is not None

    async def send_audio(self, audio: bytes) -> None:
        """
        Send audio data to the model.

        Audio should be PCM16 format at 24kHz mono, matching the
        RealtimeConfig.audio_format settings.

        Args:
            audio: Raw PCM audio bytes.
        """
        if not self.is_active:
            raise RealtimeError("Session is not active")

        # Encode as base64 for JSON transport
        audio_b64 = base64.b64encode(audio).decode("utf-8")

        await self._ws.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": audio_b64,
            }
        )

    async def send_text(self, text: str) -> None:
        """
        Send a text message (creates a conversation item).

        Args:
            text: Text message to send.
        """
        if not self.is_active:
            raise RealtimeError("Session is not active")

        await self._ws.send_json(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
        # Trigger response
        await self._ws.send_json({"type": "response.create"})

    async def interrupt(self) -> None:
        """Interrupt the current response."""
        if not self.is_active:
            return

        await self._ws.send_json({"type": "response.cancel"})

    async def commit_audio(self) -> None:
        """
        Commit the audio buffer and trigger response.

        Call this when using VAD disabled mode to signal
        end of speech.
        """
        if not self.is_active:
            return

        await self._ws.send_json({"type": "input_audio_buffer.commit"})
        await self._ws.send_json({"type": "response.create"})

    async def close(self) -> None:
        """Close the session."""
        self._closed = True


class OpenAIRealtimeProvider(BaseRealtimeProvider):
    """
    OpenAI Realtime API provider implementation.

    Uses svc-infra's WebSocketClient for robust WebSocket connections
    with automatic reconnection and proper cleanup.

    Environment Variables:
        OPENAI_API_KEY: Required. Your OpenAI API key.

    Example:
        >>> provider = OpenAIRealtimeProvider()
        >>> async with provider.connect() as session:
        ...     await session.send_audio(audio_data)
    """

    def __init__(self, config: RealtimeConfig | None = None):
        """
        Initialize the OpenAI Realtime provider.

        Args:
            config: Realtime configuration. If None, uses defaults.
        """
        super().__init__(config)
        self._api_key = os.environ.get("OPENAI_API_KEY", "")
        self._ws: Any = None
        self._session: OpenAIVoiceSession | None = None
        self._receive_task: asyncio.Task | None = None

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "openai"

    @staticmethod
    def is_configured() -> bool:
        """Check if OpenAI API key is configured."""
        return bool(os.environ.get("OPENAI_API_KEY"))

    @staticmethod
    def list_models() -> list[str]:
        """List available OpenAI Realtime models."""
        return OPENAI_REALTIME_MODELS.copy()

    @staticmethod
    def get_default_model() -> str:
        """Get the default model."""
        return DEFAULT_MODEL

    @staticmethod
    def list_voices() -> list[str]:
        """List available voices."""
        return OPENAI_REALTIME_VOICES.copy()

    def _get_ws_url(self) -> str:
        """Build the WebSocket URL with model parameter."""
        model = self.config.model or DEFAULT_MODEL
        return f"{OPENAI_REALTIME_URL}?model={model}"

    def _get_ws_headers(self) -> dict[str, str]:
        """Build WebSocket headers with authentication."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

    def _build_session_config(self) -> dict[str, Any]:
        """Build session.update payload from config."""
        session_config: dict[str, Any] = {
            "modalities": ["text", "audio"],
            "voice": self.config.voice or "alloy",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
        }

        # Instructions/system prompt
        if self.config.instructions:
            session_config["instructions"] = self.config.instructions

        # Temperature
        if self.config.temperature is not None:
            session_config["temperature"] = self.config.temperature

        # Max tokens
        if self.config.max_tokens:
            session_config["max_response_output_tokens"] = self.config.max_tokens

        # VAD configuration
        if self.config.vad_mode == VADMode.SERVER:
            session_config["turn_detection"] = {
                "type": "server_vad",
                "threshold": self.config.vad_threshold,
                "prefix_padding_ms": self.config.vad_prefix_padding_ms,
                "silence_duration_ms": self.config.vad_silence_duration_ms,
            }
        elif self.config.vad_mode == VADMode.DISABLED:
            session_config["turn_detection"] = None

        # Tools
        if self.config.tools:
            session_config["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.parameters or {"type": "object", "properties": {}},
                }
                for tool in self.config.tools
            ]
            session_config["tool_choice"] = "auto"

        return session_config

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Handle incoming WebSocket message."""
        msg_type = message.get("type", "")

        if msg_type == "session.created":
            if self._session:
                self._session._session_id = message.get("session", {}).get("id")
            logger.debug("Session created: %s", self._session.session_id if self._session else "")

        elif msg_type == "session.updated":
            logger.debug("Session updated")

        elif msg_type == "response.audio.delta":
            # Audio output from the model
            audio_b64 = message.get("delta", "")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                await self._dispatch_audio(audio_bytes)

        elif msg_type == "response.audio_transcript.delta":
            # Transcript of model's speech (assistant)
            text = message.get("delta", "")
            if text:
                await self._dispatch_transcript(text, is_final=False)

        elif msg_type == "response.audio_transcript.done":
            # Final transcript of model's speech
            text = message.get("transcript", "")
            if text:
                await self._dispatch_transcript(text, is_final=True)

        elif msg_type == "conversation.item.input_audio_transcription.completed":
            # Transcript of user's speech (if enabled)
            text = message.get("transcript", "")
            if text:
                logger.debug("User said: %s", text)

        elif msg_type == "response.function_call_arguments.done":
            # Function call completed
            call_id = message.get("call_id", "")
            name = message.get("name", "")
            arguments_str = message.get("arguments", "{}")

            try:
                arguments = json.loads(arguments_str)
            except json.JSONDecodeError:
                arguments = {}

            request = ToolCallRequest(
                call_id=call_id,
                name=name,
                arguments=arguments,
            )

            # Dispatch and get result
            result = await self._dispatch_tool_call(request)

            # Send function result back
            if self._ws:
                await self._ws.send_json(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps(result) if result is not None else "null",
                        },
                    }
                )
                # Trigger response to continue
                await self._ws.send_json({"type": "response.create"})

        elif msg_type == "input_audio_buffer.speech_started":
            # User started speaking - interrupt if model is speaking
            await self._dispatch_interrupted()

        elif msg_type == "error":
            error_data = message.get("error", {})
            error = RealtimeError(
                message=error_data.get("message", "Unknown error"),
                code=error_data.get("code"),
            )
            await self._dispatch_error(error)

        elif msg_type == "response.done":
            # Response completed
            logger.debug("Response completed")

        else:
            # Log unknown message types for debugging
            logger.debug("Unhandled message type: %s", msg_type)

    async def _receive_loop(self) -> None:
        """Background task to receive and handle messages."""
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                data = json.loads(message)
                await self._handle_message(data)
        except Exception as e:
            if not self._session or not self._session._closed:
                error = RealtimeError(f"WebSocket error: {e}")
                await self._dispatch_error(error)

    async def connect(self) -> OpenAIVoiceSession:
        """
        Connect to OpenAI Realtime API.

        Returns:
            VoiceSession for the connected session.

        Raises:
            RealtimeConnectionError: If connection fails.
        """
        if not self._api_key:
            raise RealtimeConnectionError(
                "OPENAI_API_KEY not set. Set the environment variable to use OpenAI Realtime."
            )

        try:
            # Import svc-infra WebSocket client
            from svc_infra.websocket import WebSocketClient, WebSocketConfig

            ws_config = WebSocketConfig(
                ping_interval=20.0,
                ping_timeout=20.0,
                max_message_size=16 * 1024 * 1024,  # 16MB for audio
            )

            self._ws = WebSocketClient(
                self._get_ws_url(),
                config=ws_config,
                headers=self._get_ws_headers(),
            )
            await self._ws.connect()

            # Create session object
            self._session = OpenAIVoiceSession(self._ws, self.config, self)

            # Start receive loop
            self._receive_task = asyncio.create_task(self._receive_loop())

            # Wait for session.created
            await asyncio.sleep(0.1)  # Brief wait for initial message

            # Send session.update with configuration
            await self._ws.send_json(
                {
                    "type": "session.update",
                    "session": self._build_session_config(),
                }
            )

            return self._session

        except ImportError:
            # Fallback to websockets library directly if svc-infra not available
            return await self._connect_fallback()
        except Exception as e:
            raise RealtimeConnectionError(f"Failed to connect: {e}") from e

    async def _connect_fallback(self) -> OpenAIVoiceSession:
        """Fallback connection using websockets library directly."""
        try:
            import websockets

            self._ws = await websockets.connect(
                self._get_ws_url(),
                additional_headers=self._get_ws_headers(),
                ping_interval=20,
                ping_timeout=20,
                max_size=16 * 1024 * 1024,
            )

            # Wrap in a simple adapter
            self._session = OpenAIVoiceSession(
                _WebSocketAdapter(self._ws),
                self.config,
                self,
            )

            # Start receive loop
            self._receive_task = asyncio.create_task(self._receive_loop_fallback())

            await asyncio.sleep(0.1)

            await self._ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": self._build_session_config(),
                    }
                )
            )

            return self._session

        except Exception as e:
            raise RealtimeConnectionError(f"Failed to connect: {e}") from e

    async def _receive_loop_fallback(self) -> None:
        """Receive loop for fallback websockets connection."""
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                data = json.loads(message)
                await self._handle_message(data)
        except Exception as e:
            if not self._session or not self._session._closed:
                error = RealtimeError(f"WebSocket error: {e}")
                await self._dispatch_error(error)

    async def disconnect(self) -> None:
        """Disconnect from the Realtime API."""
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._session:
            await self._session.close()
            self._session = None

        if self._ws:
            await self._ws.close()
            self._ws = None

    async def send_audio(self, audio: bytes) -> None:
        """Send audio to the current session."""
        if self._session:
            await self._session.send_audio(audio)

    async def run(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[AudioChunk | TranscriptDelta]:
        """
        Run a conversation with streaming input/output.

        Args:
            audio_stream: Async iterator of audio bytes.

        Yields:
            AudioChunk and TranscriptDelta events.
        """
        # Queue to collect events
        event_queue: asyncio.Queue[AudioChunk | TranscriptDelta | None] = asyncio.Queue()

        # Override callbacks to queue events
        original_audio_callbacks = self._audio_callbacks.copy()
        original_transcript_callbacks = self._transcript_callbacks.copy()

        async def queue_audio(audio: bytes) -> None:
            await event_queue.put(AudioChunk(data=audio))
            for cb in original_audio_callbacks:
                await cb(audio)

        async def queue_transcript(text: str, is_final: bool) -> None:
            await event_queue.put(TranscriptDelta(text=text, is_final=is_final))
            for cb in original_transcript_callbacks:
                await cb(text, is_final)

        self._audio_callbacks = [queue_audio]
        self._transcript_callbacks = [queue_transcript]

        try:
            session = await self.connect()

            async def send_audio_task():
                async for chunk in audio_stream:
                    await session.send_audio(chunk)
                # Signal end of audio if VAD disabled
                if self.config.vad_mode == VADMode.DISABLED:
                    await session.commit_audio()

            # Start sending audio
            send_task = asyncio.create_task(send_audio_task())

            try:
                while True:
                    try:
                        event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                        if event is None:
                            break
                        yield event
                    except TimeoutError:
                        if send_task.done():
                            # Give some time for final responses
                            await asyncio.sleep(1.0)
                            if event_queue.empty():
                                break
            finally:
                send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass
        finally:
            self._audio_callbacks = original_audio_callbacks
            self._transcript_callbacks = original_transcript_callbacks
            await self.disconnect()


class _WebSocketAdapter:
    """Simple adapter to make websockets library compatible with our interface."""

    def __init__(self, ws: Any):
        self._ws = ws

    async def send_json(self, data: Any) -> None:
        await self._ws.send(json.dumps(data))

    async def close(self) -> None:
        await self._ws.close()

    def __aiter__(self):
        return self._ws.__aiter__()


# Register provider with RealtimeVoice facade
RealtimeVoice.register_provider("openai", OpenAIRealtimeProvider)
