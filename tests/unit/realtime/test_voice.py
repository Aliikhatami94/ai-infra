"""Unit tests for RealtimeVoice and related classes.

Tests cover:
- Provider discovery and selection
- Configuration handling
- Tool conversion
- Callback registration
- Model data structures
"""

import os

import pytest

from ai_infra.llm.realtime import (
    AudioChunk,
    AudioFormat,
    RealtimeConfig,
    RealtimeConnectionError,
    RealtimeError,
    RealtimeVoice,
    ToolCallRequest,
    ToolDefinition,
    TranscriptDelta,
    VADMode,
)


@pytest.fixture
def mock_openai_key(monkeypatch):
    """Set a fake OPENAI_API_KEY for tests requiring provider initialization."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-unit-tests")
    yield
    # Cleanup happens automatically via monkeypatch


# ─────────────────────────────────────────────────────────────────────────────
# Model Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestVADMode:
    """Test VADMode enum."""

    def test_server_mode(self):
        assert VADMode.SERVER == "server"
        assert VADMode.SERVER.value == "server"

    def test_manual_mode(self):
        assert VADMode.MANUAL == "manual"
        assert VADMode.MANUAL.value == "manual"


class TestAudioFormat:
    """Test AudioFormat model."""

    def test_defaults(self):
        fmt = AudioFormat()
        assert fmt.encoding == "pcm16"
        assert fmt.sample_rate == 24000
        assert fmt.channels == 1

    def test_custom_sample_rate(self):
        fmt = AudioFormat(sample_rate=16000)
        assert fmt.sample_rate == 16000


class TestRealtimeConfig:
    """Test RealtimeConfig model."""

    def test_defaults(self):
        config = RealtimeConfig()
        assert config.model is None
        assert config.voice == "alloy"
        assert config.vad_mode == VADMode.SERVER
        assert config.temperature == 0.8
        assert config.max_tokens == 4096
        assert config.tools == []

    def test_custom_config(self):
        config = RealtimeConfig(
            model="gpt-4o-realtime-preview",
            voice="nova",
            instructions="Be helpful.",
            vad_mode=VADMode.MANUAL,
            temperature=0.5,
        )
        assert config.model == "gpt-4o-realtime-preview"
        assert config.voice == "nova"
        assert config.instructions == "Be helpful."
        assert config.vad_mode == VADMode.MANUAL
        assert config.temperature == 0.5

    def test_vad_settings(self):
        config = RealtimeConfig(
            vad_threshold=0.7,
            vad_prefix_padding_ms=200,
            vad_silence_duration_ms=800,
        )
        assert config.vad_threshold == 0.7
        assert config.vad_prefix_padding_ms == 200
        assert config.vad_silence_duration_ms == 800

    def test_temperature_bounds(self):
        with pytest.raises(ValueError):
            RealtimeConfig(temperature=-0.1)
        with pytest.raises(ValueError):
            RealtimeConfig(temperature=2.1)


class TestAudioChunk:
    """Test AudioChunk model."""

    def test_basic(self):
        chunk = AudioChunk(data=b"\x00\x01\x02")
        assert chunk.data == b"\x00\x01\x02"
        assert chunk.sample_rate == 24000
        assert chunk.is_final is False

    def test_final_chunk(self):
        chunk = AudioChunk(data=b"audio", is_final=True)
        assert chunk.is_final is True


class TestTranscriptDelta:
    """Test TranscriptDelta model."""

    def test_defaults(self):
        delta = TranscriptDelta(text="Hello")
        assert delta.text == "Hello"
        assert delta.is_final is False
        assert delta.role == "assistant"

    def test_user_transcript(self):
        delta = TranscriptDelta(text="Hi there", role="user", is_final=True)
        assert delta.role == "user"
        assert delta.is_final is True


class TestToolCallRequest:
    """Test ToolCallRequest model."""

    def test_basic(self):
        request = ToolCallRequest(
            call_id="call_123",
            name="get_weather",
            arguments={"city": "NYC"},
        )
        assert request.call_id == "call_123"
        assert request.name == "get_weather"
        assert request.arguments == {"city": "NYC"}

    def test_empty_arguments(self):
        request = ToolCallRequest(call_id="call_456", name="get_time")
        assert request.arguments == {}


class TestToolDefinition:
    """Test ToolDefinition model."""

    def test_basic(self):
        tool = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        assert tool.name == "search"
        assert tool.description == "Search the web"
        assert "query" in tool.parameters["properties"]

    def test_minimal(self):
        tool = ToolDefinition(name="simple_tool")
        assert tool.name == "simple_tool"
        assert tool.description is None
        assert tool.parameters is None


class TestRealtimeError:
    """Test error classes."""

    def test_realtime_error(self):
        error = RealtimeError("Something went wrong", code="ERR_001")
        assert str(error) == "[ERR_001] Something went wrong"
        assert error.message == "Something went wrong"
        assert error.code == "ERR_001"

    def test_realtime_error_no_code(self):
        error = RealtimeError("Just a message")
        assert str(error) == "Just a message"

    def test_connection_error(self):
        error = RealtimeConnectionError("Failed to connect")
        assert isinstance(error, RealtimeError)


# ─────────────────────────────────────────────────────────────────────────────
# Provider Discovery Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestProviderDiscovery:
    """Test provider discovery and registration."""

    def test_available_providers(self):
        providers = RealtimeVoice.available_providers()
        assert "openai" in providers
        assert "gemini" in providers

    def test_configured_providers_with_openai(self):
        # OpenAI key is set in test environment
        configured = RealtimeVoice.configured_providers()
        assert isinstance(configured, list)
        # If OPENAI_API_KEY is set, it should be in the list
        if os.environ.get("OPENAI_API_KEY"):
            assert "openai" in configured


class TestRealtimeVoice:
    """Test RealtimeVoice facade class."""

    def test_auto_select_provider(self, mock_openai_key):
        """Test that provider is auto-selected based on env."""
        voice = RealtimeVoice()
        assert voice.provider_name in ["openai", "gemini"]

    def test_explicit_provider(self, mock_openai_key):
        """Test explicit provider selection."""
        voice = RealtimeVoice(provider="openai")
        assert voice.provider_name == "openai"

    def test_with_config(self, mock_openai_key):
        """Test creating voice with config."""
        config = RealtimeConfig(voice="nova", temperature=0.5)
        voice = RealtimeVoice(config=config)
        assert voice.config.voice == "nova"
        assert voice.config.temperature == 0.5

    def test_tool_conversion(self, mock_openai_key):
        """Test that tool functions are converted to ToolDefinition."""

        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}"

        config = RealtimeConfig(tools=[get_weather])
        voice = RealtimeVoice(config=config)

        assert len(voice.config.tools) == 1
        assert isinstance(voice.config.tools[0], ToolDefinition)
        assert voice.config.tools[0].name == "get_weather"
        assert "get_weather" in voice._tool_functions


class TestCallbackRegistration:
    """Test callback registration on RealtimeVoice."""

    def test_on_audio_decorator(self, mock_openai_key):
        voice = RealtimeVoice()

        @voice.on_audio
        async def handle_audio(audio: bytes):
            pass

        assert len(voice._audio_callbacks) == 1

    def test_on_transcript_decorator(self, mock_openai_key):
        voice = RealtimeVoice()

        @voice.on_transcript
        async def handle_transcript(text: str, is_final: bool):
            pass

        assert len(voice._transcript_callbacks) == 1

    def test_on_error_decorator(self, mock_openai_key):
        voice = RealtimeVoice()

        @voice.on_error
        async def handle_error(error: RealtimeError):
            pass

        assert len(voice._error_callbacks) == 1

    def test_multiple_callbacks(self, mock_openai_key):
        voice = RealtimeVoice()

        @voice.on_audio
        async def handler1(audio: bytes):
            pass

        @voice.on_audio
        async def handler2(audio: bytes):
            pass

        assert len(voice._audio_callbacks) == 2


class TestToolExecution:
    """Test tool execution in RealtimeVoice."""

    @pytest.mark.asyncio
    async def test_sync_tool_execution(self, mock_openai_key):
        """Test executing a sync tool function."""

        def get_time() -> str:
            """Get current time."""
            return "12:00 PM"

        config = RealtimeConfig(tools=[get_time])
        voice = RealtimeVoice(config=config)

        request = ToolCallRequest(call_id="call_1", name="get_time", arguments={})
        result = await voice._dispatch_tool_call(request)

        assert result == "12:00 PM"

    @pytest.mark.asyncio
    async def test_async_tool_execution(self, mock_openai_key):
        """Test executing an async tool function."""

        async def async_search(query: str) -> str:
            """Search asynchronously."""
            return f"Results for: {query}"

        config = RealtimeConfig(tools=[async_search])
        voice = RealtimeVoice(config=config)

        request = ToolCallRequest(
            call_id="call_2",
            name="async_search",
            arguments={"query": "test"},
        )
        result = await voice._dispatch_tool_call(request)

        assert result == "Results for: test"

    @pytest.mark.asyncio
    async def test_tool_with_callback_fallback(self, mock_openai_key):
        """Test that callback is used when tool not found."""
        voice = RealtimeVoice()

        callback_called = False

        @voice.on_tool_call
        async def handle_tool(request: ToolCallRequest):
            nonlocal callback_called
            callback_called = True
            return "callback result"

        request = ToolCallRequest(
            call_id="call_3",
            name="unknown_tool",
            arguments={},
        )
        result = await voice._dispatch_tool_call(request)

        assert callback_called
        assert result == "callback result"


# ─────────────────────────────────────────────────────────────────────────────
# Provider Base Class Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBaseRealtimeProvider:
    """Test BaseRealtimeProvider abstract class."""

    def test_callback_registration(self):
        """Test callback registration on base provider."""
        from ai_infra.llm.realtime.openai import OpenAIRealtimeProvider

        provider = OpenAIRealtimeProvider()

        async def audio_handler(audio: bytes):
            pass

        provider.on_audio(audio_handler)
        assert len(provider._audio_callbacks) == 1

    def test_provider_info(self):
        """Test provider info methods."""
        from ai_infra.llm.realtime.openai import OpenAIRealtimeProvider

        assert OpenAIRealtimeProvider.is_configured() == bool(
            os.environ.get("OPENAI_API_KEY")
        )

        models = OpenAIRealtimeProvider.list_models()
        assert isinstance(models, list)
        assert len(models) > 0

        default = OpenAIRealtimeProvider.get_default_model()
        assert isinstance(default, str)
        assert "realtime" in default
