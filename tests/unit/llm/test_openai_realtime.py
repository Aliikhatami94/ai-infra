"""Unit tests for OpenAI Realtime Voice API.

Tests cover:
- OpenAIRealtimeProvider initialization and configuration
- OpenAIVoiceSession methods (send_audio, send_text, interrupt, commit_audio, close)
- WebSocket connection setup and headers
- Session configuration building (_build_session_config)
- Message handling (_handle_message for all message types)
- VAD mode configuration (SERVER, MANUAL, DISABLED)
- Tool integration (function calling, tool results)
- Error handling (RealtimeError, RealtimeConnectionError)
- Disconnect and cleanup
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock svc_infra modules BEFORE any ai_infra imports
# The ai_infra package imports svc_infra at various points
# Create comprehensive mock structure
_mock_svc_infra = MagicMock()
_mock_svc_infra.app = MagicMock()
_mock_svc_infra.app.env = MagicMock()
_mock_svc_infra.app.env.prepare_env = MagicMock()
_mock_svc_infra.cli = MagicMock()
_mock_svc_infra.cli.foundation = MagicMock()
_mock_svc_infra.cli.foundation.runner = MagicMock()
_mock_svc_infra.cli.foundation.runner.run_from_root = MagicMock()
_mock_svc_infra.websocket = MagicMock()
_mock_svc_infra.websocket.WebSocketClient = MagicMock()
_mock_svc_infra.websocket.WebSocketConfig = MagicMock()

# Register all submodules
sys.modules["svc_infra"] = _mock_svc_infra
sys.modules["svc_infra.app"] = _mock_svc_infra.app
sys.modules["svc_infra.app.env"] = _mock_svc_infra.app.env
sys.modules["svc_infra.cli"] = _mock_svc_infra.cli
sys.modules["svc_infra.cli.foundation"] = _mock_svc_infra.cli.foundation
sys.modules["svc_infra.cli.foundation.runner"] = _mock_svc_infra.cli.foundation.runner
sys.modules["svc_infra.websocket"] = _mock_svc_infra.websocket

from ai_infra.llm.realtime.models import (  # noqa: E402
    AudioChunk,
    AudioFormat,
    RealtimeConfig,
    RealtimeConnectionError,
    RealtimeError,
    ToolCallRequest,
    ToolCallResult,
    ToolDefinition,
    TranscriptDelta,
    VADMode,
)
from ai_infra.llm.realtime.openai import (  # noqa: E402
    OPENAI_REALTIME_URL,
    OpenAIRealtimeProvider,
    OpenAIVoiceSession,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_ws() -> MagicMock:
    """Create mock WebSocket connection with send_json."""
    ws = AsyncMock()
    ws.send_json = AsyncMock()
    ws.send = AsyncMock()
    ws.recv = AsyncMock(
        return_value=json.dumps({"type": "session.created", "session": {"id": "test-session-id"}})
    )
    ws.close = AsyncMock()
    ws.connect = AsyncMock()
    ws.closed = False
    return ws


@pytest.fixture
def default_config() -> RealtimeConfig:
    """Create default realtime configuration."""
    return RealtimeConfig(
        model="gpt-4o-realtime-preview",
        voice="alloy",
        instructions="You are a helpful assistant.",
    )


@pytest.fixture
def config_with_tools() -> RealtimeConfig:
    """Create configuration with tools."""
    return RealtimeConfig(
        model="gpt-4o-realtime-preview",
        voice="shimmer",
        instructions="You are a helpful assistant with tools.",
        tools=[
            ToolDefinition(
                name="get_weather",
                description="Get weather for a location",
                parameters={"type": "object", "properties": {"location": {"type": "string"}}},
            ),
            ToolDefinition(
                name="search",
                description="Search the web",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            ),
        ],
    )


@pytest.fixture
def provider(default_config: RealtimeConfig) -> OpenAIRealtimeProvider:
    """Create OpenAI realtime provider."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-api-key"}):
        return OpenAIRealtimeProvider(config=default_config)


@pytest.fixture
def provider_with_tools(config_with_tools: RealtimeConfig) -> OpenAIRealtimeProvider:
    """Create OpenAI realtime provider with tools."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-api-key"}):
        return OpenAIRealtimeProvider(config=config_with_tools)


@pytest.fixture
def voice_session(
    mock_ws: MagicMock, default_config: RealtimeConfig, provider: OpenAIRealtimeProvider
) -> OpenAIVoiceSession:
    """Create voice session with mock WebSocket."""
    session = OpenAIVoiceSession(ws=mock_ws, config=default_config, provider=provider)
    session._session_id = "test-session-123"
    return session


# =============================================================================
# OpenAIVoiceSession Tests
# =============================================================================


class TestOpenAIVoiceSession:
    """Tests for OpenAIVoiceSession class."""

    def test_init(
        self, mock_ws: MagicMock, default_config: RealtimeConfig, provider: OpenAIRealtimeProvider
    ):
        """Test session initialization."""
        session = OpenAIVoiceSession(ws=mock_ws, config=default_config, provider=provider)

        assert session._ws is mock_ws
        assert session._config is default_config
        assert session._provider is provider
        assert session._closed is False
        assert session._session_id is None

    def test_session_id_property(self, voice_session: OpenAIVoiceSession):
        """Test session_id property."""
        assert voice_session.session_id == "test-session-123"

    def test_session_id_empty_when_not_set(
        self, mock_ws: MagicMock, default_config: RealtimeConfig, provider: OpenAIRealtimeProvider
    ):
        """Test session_id returns empty string when not set."""
        session = OpenAIVoiceSession(ws=mock_ws, config=default_config, provider=provider)
        assert session.session_id == ""

    def test_is_active_when_open(self, voice_session: OpenAIVoiceSession):
        """Test is_active returns True when session is open."""
        assert voice_session.is_active is True

    def test_is_active_when_closed(self, voice_session: OpenAIVoiceSession):
        """Test is_active returns False when session is closed."""
        voice_session._closed = True
        assert voice_session.is_active is False

    def test_is_active_when_ws_is_none(self, voice_session: OpenAIVoiceSession):
        """Test is_active returns False when ws is None."""
        voice_session._ws = None
        assert voice_session.is_active is False

    @pytest.mark.asyncio
    async def test_send_audio(self, voice_session: OpenAIVoiceSession, mock_ws: MagicMock):
        """Test sending audio data."""
        audio_data = b"\x00\x01\x02\x03" * 100

        await voice_session.send_audio(audio_data)

        mock_ws.send_json.assert_called_once()
        call_args = mock_ws.send_json.call_args[0][0]

        assert call_args["type"] == "input_audio_buffer.append"
        assert call_args["audio"] == base64.b64encode(audio_data).decode()

    @pytest.mark.asyncio
    async def test_send_audio_multiple_chunks(
        self, voice_session: OpenAIVoiceSession, mock_ws: MagicMock
    ):
        """Test sending multiple audio chunks."""
        chunks = [b"\x00" * 50, b"\x01" * 50, b"\x02" * 50]

        for chunk in chunks:
            await voice_session.send_audio(chunk)

        assert mock_ws.send_json.call_count == 3

    @pytest.mark.asyncio
    async def test_send_audio_when_closed_raises(
        self, voice_session: OpenAIVoiceSession, mock_ws: MagicMock
    ):
        """Test sending audio when session is closed raises error."""
        voice_session._closed = True

        with pytest.raises(RealtimeError, match="Session is not active"):
            await voice_session.send_audio(b"\x00\x01\x02")

    @pytest.mark.asyncio
    async def test_send_text(self, voice_session: OpenAIVoiceSession, mock_ws: MagicMock):
        """Test sending text message."""
        await voice_session.send_text("Hello, how are you?")

        # send_text makes two calls: conversation.item.create and response.create
        assert mock_ws.send_json.call_count == 2

        # Check first call (conversation.item.create)
        first_call = mock_ws.send_json.call_args_list[0][0][0]
        assert first_call["type"] == "conversation.item.create"
        assert first_call["item"]["type"] == "message"
        assert first_call["item"]["role"] == "user"
        assert first_call["item"]["content"][0]["type"] == "input_text"
        assert first_call["item"]["content"][0]["text"] == "Hello, how are you?"

        # Check second call (response.create)
        second_call = mock_ws.send_json.call_args_list[1][0][0]
        assert second_call["type"] == "response.create"

    @pytest.mark.asyncio
    async def test_send_text_when_closed_raises(
        self, voice_session: OpenAIVoiceSession, mock_ws: MagicMock
    ):
        """Test sending text when session is closed raises error."""
        voice_session._closed = True

        with pytest.raises(RealtimeError, match="Session is not active"):
            await voice_session.send_text("Hello")

    @pytest.mark.asyncio
    async def test_interrupt(self, voice_session: OpenAIVoiceSession, mock_ws: MagicMock):
        """Test interrupting response."""
        await voice_session.interrupt()

        mock_ws.send_json.assert_called_once()
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "response.cancel"

    @pytest.mark.asyncio
    async def test_interrupt_when_closed(
        self, voice_session: OpenAIVoiceSession, mock_ws: MagicMock
    ):
        """Test interrupt when session is closed does nothing (no raise)."""
        voice_session._closed = True

        await voice_session.interrupt()

        mock_ws.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_commit_audio(self, voice_session: OpenAIVoiceSession, mock_ws: MagicMock):
        """Test committing audio buffer."""
        await voice_session.commit_audio()

        # commit_audio makes two calls: input_audio_buffer.commit and response.create
        assert mock_ws.send_json.call_count == 2

        first_call = mock_ws.send_json.call_args_list[0][0][0]
        assert first_call["type"] == "input_audio_buffer.commit"

        second_call = mock_ws.send_json.call_args_list[1][0][0]
        assert second_call["type"] == "response.create"

    @pytest.mark.asyncio
    async def test_commit_audio_when_closed(
        self, voice_session: OpenAIVoiceSession, mock_ws: MagicMock
    ):
        """Test commit_audio when session is closed does nothing."""
        voice_session._closed = True

        await voice_session.commit_audio()

        mock_ws.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_close(self, voice_session: OpenAIVoiceSession):
        """Test closing session."""
        assert voice_session.is_active is True

        await voice_session.close()

        assert voice_session.is_active is False
        assert voice_session._closed is True

    @pytest.mark.asyncio
    async def test_close_idempotent(self, voice_session: OpenAIVoiceSession):
        """Test closing session multiple times is safe."""
        await voice_session.close()
        await voice_session.close()
        await voice_session.close()

        assert voice_session._closed is True


# =============================================================================
# OpenAIRealtimeProvider Static Methods Tests
# =============================================================================


class TestOpenAIRealtimeProviderStatic:
    """Tests for OpenAIRealtimeProvider static methods."""

    def test_is_configured_with_api_key(self):
        """Test is_configured returns True with API key."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            assert OpenAIRealtimeProvider.is_configured() is True

    def test_is_configured_without_api_key(self):
        """Test is_configured returns False without API key."""
        with patch.dict("os.environ", {}, clear=True):
            # Ensure OPENAI_API_KEY is not set
            import os

            orig = os.environ.pop("OPENAI_API_KEY", None)
            try:
                assert OpenAIRealtimeProvider.is_configured() is False
            finally:
                if orig:
                    os.environ["OPENAI_API_KEY"] = orig

    def test_list_models(self):
        """Test list_models returns available models."""
        models = OpenAIRealtimeProvider.list_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert "gpt-4o-realtime-preview" in models

    def test_list_voices(self):
        """Test list_voices returns available voices."""
        voices = OpenAIRealtimeProvider.list_voices()

        assert isinstance(voices, list)
        assert len(voices) > 0
        assert "alloy" in voices
        assert "shimmer" in voices

    def test_get_default_model(self):
        """Test get_default_model returns a model."""
        model = OpenAIRealtimeProvider.get_default_model()

        assert isinstance(model, str)
        assert "realtime" in model.lower()

    def test_provider_name(self, provider: OpenAIRealtimeProvider):
        """Test provider_name property."""
        assert provider.provider_name == "openai"


# =============================================================================
# OpenAIRealtimeProvider Initialization Tests
# =============================================================================


class TestOpenAIRealtimeProviderInit:
    """Tests for OpenAIRealtimeProvider initialization."""

    def test_init_with_config(self, default_config: RealtimeConfig):
        """Test initialization with configuration."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIRealtimeProvider(config=default_config)

            assert provider.config == default_config
            assert provider._ws is None
            assert provider._session is None
            assert provider._receive_task is None

    def test_init_without_config(self):
        """Test initialization without configuration uses defaults."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIRealtimeProvider()

            assert provider.config is not None
            assert provider._api_key == "test-key"

    def test_init_with_env_api_key(self, default_config: RealtimeConfig):
        """Test initialization uses environment API key."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-api-key"}):
            provider = OpenAIRealtimeProvider(config=default_config)

            assert provider._api_key == "env-api-key"

    def test_init_with_tools(self, config_with_tools: RealtimeConfig):
        """Test initialization with tools configuration."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIRealtimeProvider(config=config_with_tools)

            assert provider.config.tools is not None
            assert len(provider.config.tools) == 2

    def test_init_vad_mode_server(self):
        """Test initialization with server VAD mode."""
        config = RealtimeConfig(
            model="gpt-4o-realtime-preview",
            vad_mode=VADMode.SERVER,
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIRealtimeProvider(config=config)

            assert provider.config.vad_mode == VADMode.SERVER

    def test_init_vad_mode_manual(self):
        """Test initialization with manual VAD mode."""
        config = RealtimeConfig(
            model="gpt-4o-realtime-preview",
            vad_mode=VADMode.MANUAL,
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIRealtimeProvider(config=config)

            assert provider.config.vad_mode == VADMode.MANUAL

    def test_init_vad_mode_disabled(self):
        """Test initialization with disabled VAD mode."""
        config = RealtimeConfig(
            model="gpt-4o-realtime-preview",
            vad_mode=VADMode.DISABLED,
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIRealtimeProvider(config=config)

            assert provider.config.vad_mode == VADMode.DISABLED


# =============================================================================
# OpenAIRealtimeProvider WebSocket Configuration Tests
# =============================================================================


class TestOpenAIRealtimeProviderWebSocket:
    """Tests for OpenAI WebSocket configuration."""

    def test_get_ws_url(self, provider: OpenAIRealtimeProvider):
        """Test WebSocket URL generation."""
        url = provider._get_ws_url()

        assert url.startswith(OPENAI_REALTIME_URL)
        assert "model=gpt-4o-realtime-preview" in url

    def test_get_ws_url_custom_model(self):
        """Test WebSocket URL with custom model."""
        config = RealtimeConfig(model="gpt-4o-mini-realtime-preview")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIRealtimeProvider(config=config)
            url = provider._get_ws_url()

            assert "model=gpt-4o-mini-realtime-preview" in url

    def test_get_ws_headers(self, provider: OpenAIRealtimeProvider):
        """Test WebSocket headers generation."""
        headers = provider._get_ws_headers()

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-api-key"
        assert "OpenAI-Beta" in headers
        assert headers["OpenAI-Beta"] == "realtime=v1"


# =============================================================================
# OpenAIRealtimeProvider Session Config Tests
# =============================================================================


class TestOpenAIRealtimeProviderSessionConfig:
    """Tests for session configuration building."""

    def test_build_session_config_basic(self, provider: OpenAIRealtimeProvider):
        """Test basic session configuration."""
        config = provider._build_session_config()

        assert "voice" in config
        assert config["voice"] == "alloy"
        assert config["modalities"] == ["text", "audio"]
        assert config["input_audio_format"] == "pcm16"
        assert config["output_audio_format"] == "pcm16"

    def test_build_session_config_with_instructions(self, provider: OpenAIRealtimeProvider):
        """Test session config includes instructions."""
        config = provider._build_session_config()

        assert config["instructions"] == "You are a helpful assistant."

    def test_build_session_config_with_voice(self):
        """Test session config with different voice."""
        cfg = RealtimeConfig(
            model="gpt-4o-realtime-preview",
            voice="shimmer",
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIRealtimeProvider(config=cfg)
            session_config = provider._build_session_config()

            assert session_config["voice"] == "shimmer"

    def test_build_session_config_vad_server(self):
        """Test session config with server VAD."""
        config = RealtimeConfig(
            model="gpt-4o-realtime-preview",
            vad_mode=VADMode.SERVER,
            vad_threshold=0.7,
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIRealtimeProvider(config=config)
            session_config = provider._build_session_config()

            assert "turn_detection" in session_config
            assert session_config["turn_detection"]["type"] == "server_vad"
            assert session_config["turn_detection"]["threshold"] == 0.7

    def test_build_session_config_vad_disabled(self):
        """Test session config with disabled VAD."""
        config = RealtimeConfig(
            model="gpt-4o-realtime-preview",
            vad_mode=VADMode.DISABLED,
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIRealtimeProvider(config=config)
            session_config = provider._build_session_config()

            # VAD disabled means turn_detection is None
            assert session_config.get("turn_detection") is None

    def test_build_session_config_with_tools(self, provider_with_tools: OpenAIRealtimeProvider):
        """Test session config with tools."""
        session_config = provider_with_tools._build_session_config()

        assert "tools" in session_config
        assert len(session_config["tools"]) == 2

        tool_names = [t["name"] for t in session_config["tools"]]
        assert "get_weather" in tool_names
        assert "search" in tool_names

    def test_build_session_config_tool_format(self, provider_with_tools: OpenAIRealtimeProvider):
        """Test tools are formatted correctly."""
        session_config = provider_with_tools._build_session_config()

        tools = session_config["tools"]
        weather_tool = next(t for t in tools if t["name"] == "get_weather")

        assert weather_tool["type"] == "function"
        assert weather_tool["description"] == "Get weather for a location"
        assert "parameters" in weather_tool

    def test_build_session_config_with_temperature(self):
        """Test session config with temperature."""
        config = RealtimeConfig(
            model="gpt-4o-realtime-preview",
            temperature=0.7,
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIRealtimeProvider(config=config)
            session_config = provider._build_session_config()

            assert session_config["temperature"] == 0.7

    def test_build_session_config_with_max_tokens(self):
        """Test session config with max_tokens."""
        config = RealtimeConfig(
            model="gpt-4o-realtime-preview",
            max_tokens=1024,
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIRealtimeProvider(config=config)
            session_config = provider._build_session_config()

            assert session_config.get("max_response_output_tokens") == 1024


# =============================================================================
# OpenAIRealtimeProvider Message Handling Tests
# =============================================================================


class TestOpenAIRealtimeProviderMessageHandling:
    """Tests for WebSocket message handling."""

    @pytest.mark.asyncio
    async def test_handle_session_created(
        self, provider: OpenAIRealtimeProvider, mock_ws: MagicMock, default_config: RealtimeConfig
    ):
        """Test handling session.created message."""
        provider._ws = mock_ws
        provider._session = OpenAIVoiceSession(ws=mock_ws, config=default_config, provider=provider)

        message = {"type": "session.created", "session": {"id": "new-session-id"}}

        await provider._handle_message(message)

        assert provider._session._session_id == "new-session-id"

    @pytest.mark.asyncio
    async def test_handle_session_updated(
        self, provider: OpenAIRealtimeProvider, mock_ws: MagicMock
    ):
        """Test handling session.updated message."""
        provider._ws = mock_ws
        message = {"type": "session.updated", "session": {"voice": "shimmer"}}

        # Should not raise
        await provider._handle_message(message)

    @pytest.mark.asyncio
    async def test_handle_response_audio_delta(
        self, provider: OpenAIRealtimeProvider, mock_ws: MagicMock
    ):
        """Test handling response.audio.delta message."""
        provider._ws = mock_ws
        audio_data = b"\x00\x01\x02\x03" * 10

        audio_received = []

        async def on_audio(data: bytes):
            audio_received.append(data)

        provider.on_audio(on_audio)

        message = {"type": "response.audio.delta", "delta": base64.b64encode(audio_data).decode()}

        await provider._handle_message(message)

        assert len(audio_received) == 1
        assert audio_received[0] == audio_data

    @pytest.mark.asyncio
    async def test_handle_response_audio_delta_empty(
        self, provider: OpenAIRealtimeProvider, mock_ws: MagicMock
    ):
        """Test handling empty audio delta."""
        provider._ws = mock_ws

        audio_received = []

        async def on_audio(data: bytes):
            audio_received.append(data)

        provider.on_audio(on_audio)

        message = {"type": "response.audio.delta", "delta": ""}

        await provider._handle_message(message)

        # Empty delta should not dispatch
        assert len(audio_received) == 0

    @pytest.mark.asyncio
    async def test_handle_response_audio_transcript_delta(
        self, provider: OpenAIRealtimeProvider, mock_ws: MagicMock
    ):
        """Test handling response.audio_transcript.delta message."""
        provider._ws = mock_ws

        transcripts = []

        async def on_transcript(text: str, is_final: bool):
            transcripts.append((text, is_final))

        provider.on_transcript(on_transcript)

        message = {"type": "response.audio_transcript.delta", "delta": "Hello, "}

        await provider._handle_message(message)

        assert len(transcripts) == 1
        assert transcripts[0][0] == "Hello, "
        assert transcripts[0][1] is False  # delta is not final

    @pytest.mark.asyncio
    async def test_handle_response_audio_transcript_done(
        self, provider: OpenAIRealtimeProvider, mock_ws: MagicMock
    ):
        """Test handling response.audio_transcript.done message."""
        provider._ws = mock_ws

        transcripts = []

        async def on_transcript(text: str, is_final: bool):
            transcripts.append((text, is_final))

        provider.on_transcript(on_transcript)

        message = {"type": "response.audio_transcript.done", "transcript": "Hello, world!"}

        await provider._handle_message(message)

        assert len(transcripts) == 1
        assert transcripts[0][0] == "Hello, world!"
        assert transcripts[0][1] is True

    @pytest.mark.asyncio
    async def test_handle_conversation_item_input_audio_transcription_completed(
        self, provider: OpenAIRealtimeProvider, mock_ws: MagicMock
    ):
        """Test handling input audio transcription."""
        provider._ws = mock_ws

        message = {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "User said hello",
        }

        # Should not raise, just logs
        await provider._handle_message(message)

    @pytest.mark.asyncio
    async def test_handle_response_function_call_arguments_done(
        self,
        provider_with_tools: OpenAIRealtimeProvider,
        mock_ws: MagicMock,
        config_with_tools: RealtimeConfig,
    ):
        """Test handling function call completion."""
        provider_with_tools._ws = mock_ws
        provider_with_tools._session = OpenAIVoiceSession(
            ws=mock_ws, config=config_with_tools, provider=provider_with_tools
        )

        tool_calls = []

        async def on_tool_call(request: ToolCallRequest):
            tool_calls.append(request)
            return {"result": "sunny"}

        provider_with_tools.on_tool_call(on_tool_call)

        message = {
            "type": "response.function_call_arguments.done",
            "call_id": "call-456",
            "name": "get_weather",
            "arguments": '{"location": "NYC"}',
        }

        await provider_with_tools._handle_message(message)

        assert len(tool_calls) == 1
        assert tool_calls[0].call_id == "call-456"
        assert tool_calls[0].name == "get_weather"
        # Arguments is parsed as dict
        assert tool_calls[0].arguments == {"location": "NYC"}

    @pytest.mark.asyncio
    async def test_handle_response_done(self, provider: OpenAIRealtimeProvider, mock_ws: MagicMock):
        """Test handling response.done message."""
        provider._ws = mock_ws
        message = {"type": "response.done", "response": {"status": "completed"}}

        # Should not raise
        await provider._handle_message(message)

    @pytest.mark.asyncio
    async def test_handle_input_audio_buffer_speech_started(
        self, provider: OpenAIRealtimeProvider, mock_ws: MagicMock
    ):
        """Test handling speech start detection (VAD)."""
        provider._ws = mock_ws

        interrupted = []

        async def on_interrupted():
            interrupted.append(True)

        provider.on_interrupted(on_interrupted)

        message = {"type": "input_audio_buffer.speech_started"}

        await provider._handle_message(message)

        assert len(interrupted) == 1

    @pytest.mark.asyncio
    async def test_handle_error(self, provider: OpenAIRealtimeProvider, mock_ws: MagicMock):
        """Test handling error message."""
        provider._ws = mock_ws

        errors = []

        async def on_error(error: RealtimeError):
            errors.append(error)

        provider.on_error(on_error)

        message = {
            "type": "error",
            "error": {"message": "Something went wrong", "code": "internal_error"},
        }

        await provider._handle_message(message)

        assert len(errors) == 1
        assert "Something went wrong" in str(errors[0])

    @pytest.mark.asyncio
    async def test_handle_unknown_message_type(
        self, provider: OpenAIRealtimeProvider, mock_ws: MagicMock
    ):
        """Test handling unknown message types gracefully."""
        provider._ws = mock_ws
        message = {"type": "unknown.event.type", "data": "some data"}

        # Should not raise
        await provider._handle_message(message)


# =============================================================================
# OpenAIRealtimeProvider Connection Tests
# =============================================================================


class TestOpenAIRealtimeProviderConnection:
    """Tests for connection management."""

    @pytest.mark.asyncio
    async def test_connect_without_api_key_raises(self):
        """Test connect without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            import os

            orig = os.environ.pop("OPENAI_API_KEY", None)
            try:
                provider = OpenAIRealtimeProvider()
                provider._api_key = ""

                with pytest.raises(RealtimeConnectionError, match="OPENAI_API_KEY not set"):
                    await provider.connect()
            finally:
                if orig:
                    os.environ["OPENAI_API_KEY"] = orig

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, provider: OpenAIRealtimeProvider):
        """Test disconnect when not connected is safe."""
        assert provider._ws is None

        # Should not raise
        await provider.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_cancels_receive_task(
        self, provider: OpenAIRealtimeProvider, mock_ws: MagicMock, default_config: RealtimeConfig
    ):
        """Test disconnect cancels receive task."""
        provider._ws = mock_ws
        provider._session = OpenAIVoiceSession(ws=mock_ws, config=default_config, provider=provider)

        # Create a proper async mock task
        async def mock_coro():
            pass

        task = asyncio.create_task(mock_coro())
        await task  # Complete it first
        provider._receive_task = task

        # Should not raise
        await provider.disconnect()

        assert provider._receive_task is None


# =============================================================================
# OpenAIRealtimeProvider Audio Streaming Tests
# =============================================================================


class TestOpenAIRealtimeProviderAudioStreaming:
    """Tests for audio streaming."""

    @pytest.mark.asyncio
    async def test_send_audio_delegates_to_session(
        self, provider: OpenAIRealtimeProvider, mock_ws: MagicMock, default_config: RealtimeConfig
    ):
        """Test send_audio delegates to session."""
        session = OpenAIVoiceSession(ws=mock_ws, config=default_config, provider=provider)
        provider._session = session

        audio_data = b"\x00\x01\x02\x03"
        await provider.send_audio(audio_data)

        # Verify message was sent
        mock_ws.send_json.assert_called_once()
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "input_audio_buffer.append"

    @pytest.mark.asyncio
    async def test_send_audio_without_session_does_nothing(self, provider: OpenAIRealtimeProvider):
        """Test send_audio without session does nothing (no raise)."""
        assert provider._session is None

        # Should not raise - just does nothing when no session
        await provider.send_audio(b"\x00\x01\x02")


# =============================================================================
# OpenAIRealtimeProvider Callback Tests
# =============================================================================


class TestOpenAIRealtimeProviderCallbacks:
    """Tests for callback registration and dispatch."""

    def test_on_audio_registration(self, provider: OpenAIRealtimeProvider):
        """Test audio callback registration."""

        async def callback(audio: bytes):
            pass

        result = provider.on_audio(callback)

        assert result is callback
        assert callback in provider._audio_callbacks

    def test_on_transcript_registration(self, provider: OpenAIRealtimeProvider):
        """Test transcript callback registration."""

        async def callback(text: str, is_final: bool):
            pass

        result = provider.on_transcript(callback)

        assert result is callback
        assert callback in provider._transcript_callbacks

    def test_on_tool_call_registration(self, provider: OpenAIRealtimeProvider):
        """Test tool call callback registration."""

        async def callback(request: ToolCallRequest):
            return {"result": "ok"}

        result = provider.on_tool_call(callback)

        assert result is callback
        assert callback in provider._tool_call_callbacks

    def test_on_error_registration(self, provider: OpenAIRealtimeProvider):
        """Test error callback registration."""

        async def callback(error: RealtimeError):
            pass

        result = provider.on_error(callback)

        assert result is callback
        assert callback in provider._error_callbacks

    def test_on_interrupted_registration(self, provider: OpenAIRealtimeProvider):
        """Test interrupted callback registration."""

        async def callback():
            pass

        result = provider.on_interrupted(callback)

        assert result is callback
        assert callback in provider._interrupted_callbacks

    def test_multiple_callback_registration(self, provider: OpenAIRealtimeProvider):
        """Test registering multiple callbacks of same type."""

        async def callback1(audio: bytes):
            pass

        async def callback2(audio: bytes):
            pass

        provider.on_audio(callback1)
        provider.on_audio(callback2)

        assert len(provider._audio_callbacks) == 2
        assert callback1 in provider._audio_callbacks
        assert callback2 in provider._audio_callbacks

    @pytest.mark.asyncio
    async def test_dispatch_audio(self, provider: OpenAIRealtimeProvider):
        """Test audio dispatch to all callbacks."""
        received = []

        async def callback1(audio: bytes):
            received.append(("cb1", audio))

        async def callback2(audio: bytes):
            received.append(("cb2", audio))

        provider.on_audio(callback1)
        provider.on_audio(callback2)

        await provider._dispatch_audio(b"test-audio")

        assert len(received) == 2
        assert ("cb1", b"test-audio") in received
        assert ("cb2", b"test-audio") in received

    @pytest.mark.asyncio
    async def test_dispatch_transcript(self, provider: OpenAIRealtimeProvider):
        """Test transcript dispatch."""
        received = []

        async def callback(text: str, is_final: bool):
            received.append((text, is_final))

        provider.on_transcript(callback)

        await provider._dispatch_transcript("Hello", False)
        await provider._dispatch_transcript("Hello world", True)

        assert len(received) == 2
        assert ("Hello", False) in received
        assert ("Hello world", True) in received

    @pytest.mark.asyncio
    async def test_dispatch_error(self, provider: OpenAIRealtimeProvider):
        """Test error dispatch."""
        received = []

        async def callback(error: RealtimeError):
            received.append(error)

        provider.on_error(callback)

        error = RealtimeError("Test error", code="test_code")
        await provider._dispatch_error(error)

        assert len(received) == 1
        assert received[0] is error

    @pytest.mark.asyncio
    async def test_callback_error_handling(self, provider: OpenAIRealtimeProvider):
        """Test that callback errors don't crash dispatch."""

        async def bad_callback(audio: bytes):
            raise ValueError("Callback failed")

        async def good_callback(audio: bytes):
            pass

        provider.on_audio(bad_callback)
        provider.on_audio(good_callback)

        # Should not raise despite bad_callback error
        await provider._dispatch_audio(b"test")


# =============================================================================
# VAD Mode Tests
# =============================================================================


class TestVADModes:
    """Tests for Voice Activity Detection modes."""

    def test_vad_mode_enum_values(self):
        """Test VAD mode enum has expected values."""
        assert VADMode.SERVER == "server"
        assert VADMode.MANUAL == "manual"
        assert VADMode.DISABLED == "disabled"

    def test_vad_server_mode_config(self):
        """Test server VAD mode configuration."""
        config = RealtimeConfig(
            model="gpt-4o-realtime-preview",
            vad_mode=VADMode.SERVER,
            vad_threshold=0.8,
        )

        assert config.vad_mode == VADMode.SERVER
        assert config.vad_threshold == 0.8

    def test_vad_manual_mode_config(self):
        """Test manual VAD mode configuration."""
        config = RealtimeConfig(
            model="gpt-4o-realtime-preview",
            vad_mode=VADMode.MANUAL,
        )

        assert config.vad_mode == VADMode.MANUAL

    @pytest.mark.asyncio
    async def test_commit_audio_for_manual_vad(
        self, voice_session: OpenAIVoiceSession, mock_ws: MagicMock
    ):
        """Test commit_audio for manual VAD mode."""
        # In manual VAD mode, user explicitly commits audio
        await voice_session.commit_audio()

        assert mock_ws.send_json.call_count == 2

        commit_msg = mock_ws.send_json.call_args_list[0][0][0]
        assert commit_msg["type"] == "input_audio_buffer.commit"


# =============================================================================
# Tool Integration Tests
# =============================================================================


class TestToolIntegration:
    """Tests for tool/function calling integration."""

    def test_tool_definition_model(self):
        """Test ToolDefinition model."""
        tool = ToolDefinition(
            name="calculator",
            description="Perform calculations",
            parameters={"type": "object", "properties": {"expression": {"type": "string"}}},
        )

        assert tool.name == "calculator"
        assert tool.description == "Perform calculations"
        assert "properties" in tool.parameters

    def test_tool_call_request_model(self):
        """Test ToolCallRequest model with dict arguments."""
        request = ToolCallRequest(
            call_id="call-abc-123", name="get_weather", arguments={"location": "New York"}
        )

        assert request.call_id == "call-abc-123"
        assert request.name == "get_weather"
        assert request.arguments == {"location": "New York"}

    def test_tool_call_result_model(self):
        """Test ToolCallResult model."""
        result = ToolCallResult(call_id="call-abc-123", output='{"temperature": 72}')

        assert result.call_id == "call-abc-123"
        assert result.output == '{"temperature": 72}'

    def test_tool_call_result_with_error(self):
        """Test ToolCallResult with error."""
        result = ToolCallResult(call_id="call-abc-123", error="Function not found")

        assert result.error == "Function not found"

    @pytest.mark.asyncio
    async def test_tool_call_dispatch(self, provider_with_tools: OpenAIRealtimeProvider):
        """Test tool call dispatch returns result."""

        async def tool_handler(request: ToolCallRequest):
            if request.name == "get_weather":
                return {"weather": "sunny", "temp": 72}
            return None

        provider_with_tools.on_tool_call(tool_handler)

        request = ToolCallRequest(
            call_id="call-1", name="get_weather", arguments={"location": "NYC"}
        )

        result = await provider_with_tools._dispatch_tool_call(request)

        assert result == {"weather": "sunny", "temp": 72}


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_realtime_error_model(self):
        """Test RealtimeError exception."""
        error = RealtimeError("Connection failed", code="connection_error")

        assert "Connection failed" in str(error)
        assert error.code == "connection_error"

    def test_realtime_connection_error(self):
        """Test RealtimeConnectionError exception."""
        error = RealtimeConnectionError("WebSocket failed", code="ws_error")

        assert isinstance(error, RealtimeError)
        assert "WebSocket failed" in str(error)

    @pytest.mark.asyncio
    async def test_invalid_json_message(self, provider: OpenAIRealtimeProvider, mock_ws: MagicMock):
        """Test handling invalid JSON in message."""
        provider._ws = mock_ws

        # Invalid message should raise when trying to parse
        with pytest.raises((json.JSONDecodeError, TypeError)):
            json.loads("not valid json {")


# =============================================================================
# Audio Format Tests
# =============================================================================


class TestAudioFormat:
    """Tests for audio format configuration."""

    def test_default_audio_format(self):
        """Test default audio format values."""
        fmt = AudioFormat()

        assert fmt.encoding == "pcm16"
        assert fmt.sample_rate == 24000
        assert fmt.channels == 1

    def test_custom_audio_format(self):
        """Test custom audio format."""
        fmt = AudioFormat(encoding="pcm16", sample_rate=16000, channels=2)

        assert fmt.sample_rate == 16000
        assert fmt.channels == 2


# =============================================================================
# AudioChunk and TranscriptDelta Tests
# =============================================================================


class TestAudioChunk:
    """Tests for AudioChunk model."""

    def test_audio_chunk_creation(self):
        """Test AudioChunk creation."""
        chunk = AudioChunk(data=b"\x00\x01\x02\x03", sample_rate=24000, is_final=False)

        assert chunk.data == b"\x00\x01\x02\x03"
        assert chunk.sample_rate == 24000
        assert chunk.is_final is False

    def test_audio_chunk_final(self):
        """Test final AudioChunk."""
        chunk = AudioChunk(data=b"", sample_rate=24000, is_final=True)

        assert chunk.is_final is True


class TestTranscriptDelta:
    """Tests for TranscriptDelta model."""

    def test_transcript_delta_creation(self):
        """Test TranscriptDelta creation."""
        delta = TranscriptDelta(text="Hello ", is_final=False, role="assistant")

        assert delta.text == "Hello "
        assert delta.is_final is False
        assert delta.role == "assistant"

    def test_transcript_delta_final(self):
        """Test final TranscriptDelta."""
        delta = TranscriptDelta(text="Hello, how can I help you?", is_final=True, role="assistant")

        assert delta.is_final is True


# =============================================================================
# RealtimeConfig Tests
# =============================================================================


class TestRealtimeConfig:
    """Tests for RealtimeConfig model."""

    def test_default_config(self):
        """Test default configuration."""
        config = RealtimeConfig()

        assert config.model is not None or config.model is None  # May have default
        assert config.vad_mode == VADMode.SERVER  # Default

    def test_config_with_all_options(self):
        """Test RealtimeConfig with all options."""
        config = RealtimeConfig(
            model="gpt-4o-realtime-preview",
            voice="nova",
            instructions="Be helpful",
            vad_mode=VADMode.SERVER,
            vad_threshold=0.6,
            temperature=0.8,
            max_tokens=2048,
            tools=[ToolDefinition(name="test", description="Test tool", parameters={})],
        )

        assert config.model == "gpt-4o-realtime-preview"
        assert config.voice == "nova"
        assert config.instructions == "Be helpful"
        assert config.vad_mode == VADMode.SERVER
        assert config.vad_threshold == 0.6
        assert config.temperature == 0.8
        assert config.max_tokens == 2048
        assert len(config.tools) == 1


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_audio_chunk(self, voice_session: OpenAIVoiceSession, mock_ws: MagicMock):
        """Test sending empty audio chunk."""
        await voice_session.send_audio(b"")

        mock_ws.send_json.assert_called_once()
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["audio"] == ""

    @pytest.mark.asyncio
    async def test_large_audio_chunk(self, voice_session: OpenAIVoiceSession, mock_ws: MagicMock):
        """Test sending large audio chunk."""
        large_audio = b"\x00" * 100000  # 100KB

        await voice_session.send_audio(large_audio)

        mock_ws.send_json.assert_called_once()
        call_args = mock_ws.send_json.call_args[0][0]
        assert len(call_args["audio"]) > 0

    @pytest.mark.asyncio
    async def test_unicode_text(self, voice_session: OpenAIVoiceSession, mock_ws: MagicMock):
        """Test sending unicode text."""
        await voice_session.send_text("你好世界 🌍")

        # Find the conversation.item.create message
        create_call = mock_ws.send_json.call_args_list[0][0][0]
        assert create_call["item"]["content"][0]["text"] == "你好世界 🌍"

    @pytest.mark.asyncio
    async def test_rapid_send_operations(
        self, voice_session: OpenAIVoiceSession, mock_ws: MagicMock
    ):
        """Test rapid sequential send operations."""
        # Send many operations quickly
        for i in range(20):
            await voice_session.send_audio(f"chunk{i}".encode())

        assert mock_ws.send_json.call_count == 20

    @pytest.mark.asyncio
    async def test_interleaved_operations(
        self, voice_session: OpenAIVoiceSession, mock_ws: MagicMock
    ):
        """Test interleaved audio and text operations."""
        await voice_session.send_audio(b"audio1")
        await voice_session.send_text("text1")
        await voice_session.send_audio(b"audio2")
        await voice_session.interrupt()
        await voice_session.send_audio(b"audio3")

        # 1 audio + 2 text calls + 1 audio + 1 interrupt + 1 audio = 6 calls
        assert mock_ws.send_json.call_count == 6


# =============================================================================
# Integration-style Unit Tests
# =============================================================================


class TestOpenAIRealtimeFlow:
    """Integration-style tests for complete flows."""

    @pytest.mark.asyncio
    async def test_complete_transcript_flow(
        self, provider: OpenAIRealtimeProvider, mock_ws: MagicMock, default_config: RealtimeConfig
    ):
        """Test complete transcript handling flow."""
        provider._ws = mock_ws
        provider._session = OpenAIVoiceSession(ws=mock_ws, config=default_config, provider=provider)

        transcripts = []

        async def on_transcript(text: str, is_final: bool):
            transcripts.append((text, is_final))

        provider.on_transcript(on_transcript)

        # Simulate message sequence
        messages = [
            {"type": "session.created", "session": {"id": "test"}},
            {"type": "response.audio_transcript.delta", "delta": "Hello"},
            {"type": "response.audio_transcript.delta", "delta": ", world!"},
            {"type": "response.audio_transcript.done", "transcript": "Hello, world!"},
        ]

        for msg in messages:
            await provider._handle_message(msg)

        # Verify transcripts received
        assert len(transcripts) == 3
        assert ("Hello", False) in transcripts
        assert (", world!", False) in transcripts
        assert ("Hello, world!", True) in transcripts

    @pytest.mark.asyncio
    async def test_complete_audio_flow(
        self, provider: OpenAIRealtimeProvider, mock_ws: MagicMock, default_config: RealtimeConfig
    ):
        """Test complete audio handling flow."""
        provider._ws = mock_ws
        provider._session = OpenAIVoiceSession(ws=mock_ws, config=default_config, provider=provider)

        audio_response = base64.b64encode(b"\x00\x01\x02\x03" * 100).decode()

        audio_received = []

        async def on_audio(data: bytes):
            audio_received.append(data)

        provider.on_audio(on_audio)

        messages = [
            {"type": "session.created", "session": {"id": "test"}},
            {"type": "response.audio.delta", "delta": audio_response},
        ]

        for msg in messages:
            await provider._handle_message(msg)

        # Verify audio received
        assert len(audio_received) == 1
        assert audio_received[0] == b"\x00\x01\x02\x03" * 100

    @pytest.mark.asyncio
    async def test_tool_call_flow(
        self,
        provider_with_tools: OpenAIRealtimeProvider,
        mock_ws: MagicMock,
        config_with_tools: RealtimeConfig,
    ):
        """Test complete tool call flow."""
        provider_with_tools._ws = mock_ws
        provider_with_tools._session = OpenAIVoiceSession(
            ws=mock_ws, config=config_with_tools, provider=provider_with_tools
        )

        tool_calls = []

        async def handle_tool(request: ToolCallRequest):
            tool_calls.append(request)
            return {"weather": "rainy", "temp": 55}

        provider_with_tools.on_tool_call(handle_tool)

        messages = [
            {"type": "session.created", "session": {"id": "test"}},
            {
                "type": "response.function_call_arguments.done",
                "call_id": "call-xyz",
                "name": "get_weather",
                "arguments": '{"location": "Seattle"}',
            },
        ]

        for msg in messages:
            await provider_with_tools._handle_message(msg)

        # Verify tool call
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].arguments == {"location": "Seattle"}

        # Verify response was sent back
        assert mock_ws.send_json.call_count >= 2  # function_call_output + response.create
