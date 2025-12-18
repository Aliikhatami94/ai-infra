"""Provider-agnostic data models for Realtime Voice API.

This module defines the core data structures used across all realtime
voice providers (OpenAI, Gemini, etc.). These models are designed to
abstract away provider-specific differences.
"""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class VADMode(str, Enum):
    """Voice Activity Detection mode.

    Controls how turn-taking is handled in the conversation.

    - SERVER: Provider handles VAD automatically (recommended for most use cases)
      - OpenAI: Uses "server_vad" mode
      - Gemini: Uses automatic turn detection
    - MANUAL: Client controls turn-taking explicitly
      - Call commit_audio() to signal end of user speech
      - Useful for push-to-talk interfaces
    - DISABLED: VAD is disabled, no automatic turn detection
      - User must call commit_audio() after each input
    """

    SERVER = "server"
    MANUAL = "manual"
    DISABLED = "disabled"


@runtime_checkable
class VoiceSession(Protocol):
    """Protocol for active voice sessions.

    Returned by provider.connect() and used to interact with
    the active realtime session.
    """

    @property
    def session_id(self) -> str:
        """Get the unique session identifier."""
        ...

    @property
    def is_active(self) -> bool:
        """Check if the session is still active."""
        ...

    @abstractmethod
    async def send_audio(self, audio: bytes) -> None:
        """Send audio data to the model.

        Args:
            audio: Raw PCM16 audio bytes.
        """
        ...

    @abstractmethod
    async def send_text(self, text: str) -> None:
        """Send a text message to the model.

        Args:
            text: Text message.
        """
        ...

    @abstractmethod
    async def interrupt(self) -> None:
        """Interrupt the current model response."""
        ...

    @abstractmethod
    async def commit_audio(self) -> None:
        """Signal end of audio input (for manual VAD mode)."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the session."""
        ...


class ToolDefinition(BaseModel):
    """Tool definition for realtime sessions.

    This is the normalized format used internally. Users can pass
    plain functions which are converted to this format.
    """

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class AudioFormat(BaseModel):
    """Audio format specification.

    All providers use PCM16 format, but may differ in sample rate.
    The RealtimeVoice class handles conversion internally.

    Attributes:
        encoding: Audio encoding format (always "pcm16" for realtime APIs)
        sample_rate: Sample rate in Hz (24000 for output, may vary for input)
        channels: Number of audio channels (always 1 for mono)
    """

    encoding: Literal["pcm16"] = "pcm16"
    sample_rate: int = 24000
    channels: int = 1


class RealtimeConfig(BaseModel):
    """Provider-agnostic configuration for Realtime Voice session.

    This configuration is translated to provider-specific settings
    internally by each provider implementation.

    Example:
        ```python
        config = RealtimeConfig(
            voice="alloy",
            instructions="You are a helpful assistant.",
            vad_mode=VADMode.SERVER,
            temperature=0.8,
        )
        voice = RealtimeVoice(config=config)
        ```

    Example with tools:
        ```python
        def get_weather(city: str) -> str:
            '''Get weather for a city.'''
            return f"Weather in {city}: Sunny, 72°F"

        config = RealtimeConfig(
            instructions="You are a helpful assistant with weather access.",
            tools=[get_weather],
        )
        voice = RealtimeVoice(config=config)
        ```
    """

    # Model selection (optional - uses provider default if not specified)
    model: str | None = Field(
        default=None,
        description="Model name (e.g., 'gpt-4o-realtime-preview'). Uses provider default if None.",
    )

    # Voice selection
    voice: str = Field(
        default="alloy",
        description="Voice name. Provider-specific voices are mapped internally.",
    )

    # System behavior
    instructions: str = Field(
        default="You are a helpful assistant.",
        description="System instructions for the model.",
    )

    # Turn detection
    vad_mode: VADMode = Field(
        default=VADMode.SERVER,
        description="Voice activity detection mode.",
    )

    # VAD configuration (for server mode)
    vad_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="VAD activation threshold (0.0-1.0).",
    )
    vad_prefix_padding_ms: int = Field(
        default=300,
        ge=0,
        description="Audio to include before speech starts (ms).",
    )
    vad_silence_duration_ms: int = Field(
        default=500,
        ge=0,
        description="Silence duration to end turn (ms).",
    )

    # Transcription settings
    transcribe_input: bool = Field(
        default=True,
        description="Whether to transcribe user speech.",
    )
    transcribe_output: bool = Field(
        default=True,
        description="Whether to transcribe model speech.",
    )

    # Generation settings
    temperature: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for response generation.",
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens in model response.",
    )

    # Tools (same format as Agent)
    tools: list[Any] = Field(
        default_factory=list,
        description="List of tool functions the model can call. Same format as Agent tools.",
    )


class AudioChunk(BaseModel):
    """Audio data chunk from the model.

    Represents a piece of audio output from the model during streaming.
    Audio is always PCM16 format at 24kHz.

    Attributes:
        data: Raw audio bytes (PCM16, 24kHz, mono)
        sample_rate: Sample rate (always 24000)
        is_final: Whether this is the final chunk in the response
    """

    data: bytes
    sample_rate: int = 24000
    is_final: bool = False

    model_config = {"arbitrary_types_allowed": True}


class TranscriptDelta(BaseModel):
    """Transcript update during streaming.

    Represents incremental text from transcription (input or output).

    Attributes:
        text: The transcript text (may be partial)
        is_final: Whether this is the final transcript for this turn
        role: Who is speaking ("user" or "assistant")
    """

    text: str
    is_final: bool = False
    role: Literal["user", "assistant"] = "assistant"


class ToolCallRequest(BaseModel):
    """Tool call request from the model.

    When the model decides to call a tool, this object contains
    the details. Same format as Agent tool calls.

    Attributes:
        call_id: Unique identifier for this tool call
        name: Name of the tool to call
        arguments: Arguments to pass to the tool
    """

    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallResult(BaseModel):
    """Result of a tool call to send back to the model.

    Attributes:
        call_id: The call_id from the ToolCallRequest
        output: The result of the tool call (will be JSON-serialized)
        error: Optional error message if the tool failed
    """

    call_id: str
    output: Any = None
    error: str | None = None


class RealtimeEvent(BaseModel):
    """Base class for realtime events.

    Used internally for event routing.
    """

    type: str


class SessionCreatedEvent(RealtimeEvent):
    """Emitted when a session is successfully created."""

    type: Literal["session.created"] = "session.created"
    session_id: str


class SessionErrorEvent(RealtimeEvent):
    """Emitted when a session error occurs."""

    type: Literal["session.error"] = "session.error"
    error: str
    code: str | None = None


class ResponseDoneEvent(RealtimeEvent):
    """Emitted when the model finishes responding."""

    type: Literal["response.done"] = "response.done"


class InterruptedEvent(RealtimeEvent):
    """Emitted when the user interrupts the model."""

    type: Literal["interrupted"] = "interrupted"


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────


class RealtimeError(Exception):
    """Base exception for realtime voice errors.

    Attributes:
        message: Error description.
        code: Optional error code from the provider.
    """

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self) -> str:
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message


class RealtimeConnectionError(RealtimeError):
    """Error connecting to the realtime API.

    Raised when the WebSocket connection fails or cannot be established.
    """

    pass
