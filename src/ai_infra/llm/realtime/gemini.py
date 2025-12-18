"""
Google Gemini Realtime Voice API provider.

This module implements the Gemini Live API for real-time voice conversations.
Gemini's approach differs from OpenAI - it uses a multimodal streaming API
that can handle audio, video, and text inputs simultaneously.

The Gemini Live API provides:
- Low-latency multimodal conversations
- Native audio understanding (not just transcription)
- Support for live video input
- Server-side voice activity detection

Example:
    >>> from ai_infra.llm.realtime import RealtimeVoice, RealtimeConfig
    >>>
    >>> voice = RealtimeVoice(provider="gemini")
    >>>
    >>> @voice.on_transcript
    >>> async def on_transcript(text: str, is_final: bool):
    ...     print(text)
    >>>
    >>> async with voice.connect() as session:
    ...     await session.send_audio(audio_bytes)

References:
    - https://ai.google.dev/gemini-api/docs/live
    - https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/realtime-api
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
    VoiceSession,
)
from ai_infra.llm.realtime.voice import RealtimeVoice
from ai_infra.providers import ProviderCapability, ProviderRegistry

logger = logging.getLogger(__name__)

# Gemini Live API configuration
GEMINI_LIVE_URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent"


def _get_gemini_realtime_config():
    """Get Gemini realtime config from registry."""
    config = ProviderRegistry.get("google_genai")
    if config:
        return config.get_capability(ProviderCapability.REALTIME)
    return None


# Get models and voices from registry (with fallbacks)
_realtime_cap = _get_gemini_realtime_config()
GEMINI_LIVE_MODELS = (
    _realtime_cap.models
    if _realtime_cap and _realtime_cap.models
    else [
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash-thinking-exp",
    ]
)
GEMINI_VOICES = (
    _realtime_cap.voices
    if _realtime_cap and _realtime_cap.voices
    else ["Puck", "Charon", "Kore", "Fenrir", "Aoede"]
)
DEFAULT_MODEL = (
    _realtime_cap.default_model
    if _realtime_cap and _realtime_cap.default_model
    else "gemini-2.0-flash-exp"
)


class GeminiVoiceSession(VoiceSession):
    """
    VoiceSession implementation for Gemini Live API.

    Wraps the WebSocket connection and provides the VoiceSession interface.
    """

    def __init__(
        self,
        ws: Any,
        config: RealtimeConfig,
        provider: GeminiRealtimeProvider,
    ):
        self._ws = ws
        self._config = config
        self._provider = provider
        self._session_id: str = ""
        self._closed = False
        self._setup_complete = False

    @property
    def session_id(self) -> str:
        """Get the session ID."""
        return self._session_id

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return not self._closed and self._ws is not None and self._setup_complete

    async def send_audio(self, audio: bytes) -> None:
        """
        Send audio data to the model.

        Audio should be PCM16 format at 16kHz mono for Gemini.

        Args:
            audio: Raw PCM audio bytes.
        """
        if not self.is_active:
            raise RealtimeError("Session is not active")

        # Gemini expects audio in realtime_input message
        audio_b64 = base64.b64encode(audio).decode("utf-8")

        await self._ws.send_json(
            {
                "realtime_input": {
                    "media_chunks": [
                        {
                            "mime_type": "audio/pcm",
                            "data": audio_b64,
                        }
                    ]
                }
            }
        )

    async def send_text(self, text: str) -> None:
        """
        Send a text message.

        Args:
            text: Text message to send.
        """
        if not self.is_active:
            raise RealtimeError("Session is not active")

        await self._ws.send_json(
            {
                "client_content": {
                    "turns": [{"role": "user", "parts": [{"text": text}]}],
                    "turn_complete": True,
                }
            }
        )

    async def interrupt(self) -> None:
        """Interrupt the current response."""
        if not self.is_active:
            return

        # Gemini doesn't have explicit interrupt - just send new input
        # The model will naturally stop speaking when it detects new input
        logger.debug("Interrupt requested (Gemini handles this automatically)")

    async def commit_audio(self) -> None:
        """
        Signal end of audio input.

        For Gemini, we send an end_of_turn signal.
        """
        if not self.is_active:
            return

        await self._ws.send_json(
            {
                "client_content": {
                    "turn_complete": True,
                }
            }
        )

    async def close(self) -> None:
        """Close the session."""
        self._closed = True


class GeminiRealtimeProvider(BaseRealtimeProvider):
    """
    Gemini Live API provider implementation.

    Uses WebSocket connections for real-time multimodal conversations.

    Environment Variables:
        GEMINI_API_KEY or GOOGLE_API_KEY: Required. Your Google API key.

    Example:
        >>> provider = GeminiRealtimeProvider()
        >>> async with provider.connect() as session:
        ...     await session.send_audio(audio_data)
    """

    def __init__(self, config: RealtimeConfig | None = None):
        """
        Initialize the Gemini Realtime provider.

        Args:
            config: Realtime configuration. If None, uses defaults.
        """
        super().__init__(config)
        self._api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GOOGLE_GENAI_API_KEY")
            or ""
        )
        self._ws: Any = None
        self._session: GeminiVoiceSession | None = None
        self._receive_task: asyncio.Task | None = None

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "gemini"

    @staticmethod
    def is_configured() -> bool:
        """Check if Gemini API key is configured."""
        return bool(
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GOOGLE_GENAI_API_KEY")
        )

    @staticmethod
    def list_models() -> list[str]:
        """List available Gemini Live models."""
        return GEMINI_LIVE_MODELS.copy()

    @staticmethod
    def get_default_model() -> str:
        """Get the default model."""
        return DEFAULT_MODEL

    @staticmethod
    def list_voices() -> list[str]:
        """List available voices."""
        return GEMINI_VOICES.copy()

    def _get_ws_url(self) -> str:
        """Build the WebSocket URL with API key."""
        return f"{GEMINI_LIVE_URL}?key={self._api_key}"

    def _build_setup_message(self) -> dict[str, Any]:
        """Build the setup message for the session."""
        model = self.config.model or DEFAULT_MODEL

        setup: dict[str, Any] = {
            "setup": {
                "model": f"models/{model}",
                "generation_config": {
                    "response_modalities": ["AUDIO"],
                    "speech_config": {
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": self.config.voice or "Puck",
                            }
                        }
                    },
                },
            }
        }

        # Add system instruction
        if self.config.instructions:
            setup["setup"]["system_instruction"] = {"parts": [{"text": self.config.instructions}]}

        # Add tools if configured
        if self.config.tools:
            setup["setup"]["tools"] = [
                {
                    "function_declarations": [
                        {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.parameters or {"type": "object", "properties": {}},
                        }
                        for tool in self.config.tools
                    ]
                }
            ]

        return setup

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Handle incoming WebSocket message."""

        # Setup complete
        if "setupComplete" in message:
            if self._session:
                self._session._setup_complete = True
                self._session._session_id = message.get("setupComplete", {}).get("sessionId", "")
            logger.debug("Gemini session setup complete")
            return

        # Server content (model response)
        server_content = message.get("serverContent")
        if server_content:
            # Check for model turn
            model_turn = server_content.get("modelTurn")
            if model_turn:
                parts = model_turn.get("parts", [])
                for part in parts:
                    # Audio output
                    if "inlineData" in part:
                        inline_data = part["inlineData"]
                        if inline_data.get("mimeType", "").startswith("audio/"):
                            audio_b64 = inline_data.get("data", "")
                            if audio_b64:
                                audio_bytes = base64.b64decode(audio_b64)
                                await self._dispatch_audio(audio_bytes)

                    # Text output (transcript of what model is saying)
                    if "text" in part:
                        text = part["text"]
                        is_final = server_content.get("turnComplete", False)
                        await self._dispatch_transcript(text, is_final)

            # Turn complete
            if server_content.get("turnComplete"):
                logger.debug("Model turn complete")

            # Interrupted
            if server_content.get("interrupted"):
                await self._dispatch_interrupted()

        # Tool call
        tool_call = message.get("toolCall")
        if tool_call:
            function_calls = tool_call.get("functionCalls", [])
            for fc in function_calls:
                request = ToolCallRequest(
                    call_id=fc.get("id", ""),
                    name=fc.get("name", ""),
                    arguments=fc.get("args", {}),
                )

                result = await self._dispatch_tool_call(request)

                # Send function response
                if self._ws:
                    await self._ws.send_json(
                        {
                            "tool_response": {
                                "function_responses": [
                                    {
                                        "id": request.call_id,
                                        "name": request.name,
                                        "response": {"result": result},
                                    }
                                ]
                            }
                        }
                    )

        # Error
        if "error" in message:
            error_data = message["error"]
            error = RealtimeError(
                message=error_data.get("message", "Unknown error"),
                code=error_data.get("code"),
            )
            await self._dispatch_error(error)

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

    async def connect(self) -> GeminiVoiceSession:
        """
        Connect to Gemini Live API.

        Returns:
            VoiceSession for the connected session.

        Raises:
            RealtimeConnectionError: If connection fails.
        """
        if not self._api_key:
            raise RealtimeConnectionError(
                "GEMINI_API_KEY or GOOGLE_API_KEY not set. "
                "Set the environment variable to use Gemini Live."
            )

        try:
            # Try svc-infra WebSocket client first
            from svc_infra.websocket import WebSocketClient, WebSocketConfig

            ws_config = WebSocketConfig(
                ping_interval=20.0,
                ping_timeout=20.0,
                max_message_size=16 * 1024 * 1024,
            )

            self._ws = WebSocketClient(
                self._get_ws_url(),
                config=ws_config,
            )
            await self._ws.connect()

            # Create session
            self._session = GeminiVoiceSession(self._ws, self.config, self)

            # Start receive loop
            self._receive_task = asyncio.create_task(self._receive_loop())

            # Send setup message
            await self._ws.send_json(self._build_setup_message())

            # Wait for setup complete
            for _ in range(50):  # 5 second timeout
                await asyncio.sleep(0.1)
                if self._session._setup_complete:
                    break

            if not self._session._setup_complete:
                raise RealtimeConnectionError("Setup timeout - no setupComplete received")

            return self._session

        except ImportError:
            return await self._connect_fallback()
        except Exception as e:
            raise RealtimeConnectionError(f"Failed to connect: {e}") from e

    async def _connect_fallback(self) -> GeminiVoiceSession:
        """Fallback connection using websockets library directly."""
        try:
            import websockets

            self._ws = await websockets.connect(
                self._get_ws_url(),
                ping_interval=20,
                ping_timeout=20,
                max_size=16 * 1024 * 1024,
            )

            # Wrap in adapter
            self._session = GeminiVoiceSession(
                _WebSocketAdapter(self._ws),
                self.config,
                self,
            )

            # Start receive loop
            self._receive_task = asyncio.create_task(self._receive_loop_fallback())

            # Send setup
            await self._ws.send(json.dumps(self._build_setup_message()))

            # Wait for setup complete
            for _ in range(50):
                await asyncio.sleep(0.1)
                if self._session._setup_complete:
                    break

            if not self._session._setup_complete:
                raise RealtimeConnectionError("Setup timeout")

            return self._session

        except Exception as e:
            raise RealtimeConnectionError(f"Failed to connect: {e}") from e

    async def _receive_loop_fallback(self) -> None:
        """Receive loop for fallback connection."""
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
        """Disconnect from the Live API."""
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
        event_queue: asyncio.Queue[AudioChunk | TranscriptDelta | None] = asyncio.Queue()

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
                await session.commit_audio()

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
    """Simple adapter for websockets library."""

    def __init__(self, ws: Any):
        self._ws = ws

    async def send_json(self, data: Any) -> None:
        await self._ws.send(json.dumps(data))

    async def close(self) -> None:
        await self._ws.close()

    def __aiter__(self):
        return self._ws.__aiter__()


# Register provider with RealtimeVoice facade
RealtimeVoice.register_provider("gemini", GeminiRealtimeProvider)
