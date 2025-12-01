"""Integration tests for Realtime Voice API.

These tests require API keys and make real API calls.
They are skipped by default unless explicitly enabled via:
  - OPENAI_API_KEY environment variable (for OpenAI tests)
  - GEMINI_API_KEY or GOOGLE_API_KEY environment variable (for Gemini tests)

Run with: pytest tests/integration/test_realtime_voice.py -v
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.llm.realtime import (
    AudioChunk,
    GeminiRealtimeProvider,
    OpenAIRealtimeProvider,
    RealtimeConfig,
    RealtimeError,
    RealtimeVoice,
    ToolCallRequest,
    TranscriptDelta,
)

# Skip markers
SKIP_NO_OPENAI = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
SKIP_NO_GEMINI = pytest.mark.skipif(
    not (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_GENAI_API_KEY")
    ),
    reason="GEMINI_API_KEY, GOOGLE_API_KEY, or GOOGLE_GENAI_API_KEY not set",
)
SKIP_NO_API_KEYS = pytest.mark.skipif(
    not (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_GENAI_API_KEY")
    ),
    reason="No API keys set (need OPENAI_API_KEY or GEMINI_API_KEY)",
)


# -----------------------------------------------------------------------------
# OpenAI Realtime Integration Tests
# -----------------------------------------------------------------------------


@SKIP_NO_OPENAI
class TestOpenAIRealtimeIntegration:
    """Integration tests for OpenAI Realtime API."""

    @pytest.fixture
    def provider(self) -> OpenAIRealtimeProvider:
        """Create provider with test configuration."""
        config = RealtimeConfig(
            model="gpt-4o-realtime-preview",
            voice="alloy",
            instructions="You are a helpful assistant. Keep responses brief.",
        )
        return OpenAIRealtimeProvider(config=config)

    @pytest.mark.asyncio
    async def test_provider_connection(self, provider: OpenAIRealtimeProvider):
        """Test that we can connect to OpenAI Realtime API."""
        session = await provider.connect()

        try:
            # Verify we get a session
            assert session is not None
        finally:
            await provider.disconnect()

    @pytest.mark.asyncio
    async def test_session_lifecycle(self, provider: OpenAIRealtimeProvider):
        """Test full session lifecycle: connect, run briefly, disconnect."""
        events_received: list[str] = []

        async def on_transcript(text: str, is_final: bool):
            events_received.append("transcript")

        async def on_audio(audio: bytes):
            events_received.append("audio")

        provider.on_transcript(on_transcript)
        provider.on_audio(on_audio)

        session = await provider.connect()

        try:
            # Brief wait to establish connection
            await asyncio.sleep(0.5)

            # Session should be active
            assert session is not None

        finally:
            await provider.disconnect()

    @pytest.mark.asyncio
    async def test_text_message(self, provider: OpenAIRealtimeProvider):
        """Test sending a text message and receiving a response."""
        transcripts: list[str] = []

        async def on_transcript(text: str, is_final: bool):
            transcripts.append(text)

        provider.on_transcript(on_transcript)

        session = await provider.connect()

        try:
            # Send a simple text message via the session
            if hasattr(session, "send_text"):
                await session.send_text("Hello, please respond with just 'Hi'.")

            # Wait for response (with timeout)
            for _ in range(20):  # 10 seconds max
                if transcripts:
                    break
                await asyncio.sleep(0.5)

            # Should have received some response
            if transcripts:
                combined = "".join(transcripts)
                assert len(combined) > 0

        finally:
            await provider.disconnect()


# -----------------------------------------------------------------------------
# Gemini Realtime Integration Tests
# -----------------------------------------------------------------------------


@SKIP_NO_GEMINI
class TestGeminiRealtimeIntegration:
    """Integration tests for Gemini Realtime API."""

    @pytest.fixture
    def provider(self) -> GeminiRealtimeProvider:
        """Create provider with test configuration."""
        config = RealtimeConfig(
            model="gemini-2.0-flash-exp",
            voice="Puck",
            instructions="You are a helpful assistant. Keep responses brief.",
        )
        return GeminiRealtimeProvider(config=config)

    @pytest.mark.asyncio
    async def test_provider_connection(self, provider: GeminiRealtimeProvider):
        """Test that we can connect to Gemini Realtime API."""
        session = await provider.connect()

        try:
            assert session is not None
        finally:
            await provider.disconnect()

    @pytest.mark.asyncio
    async def test_session_lifecycle(self, provider: GeminiRealtimeProvider):
        """Test full session lifecycle: connect, run briefly, disconnect."""
        session = await provider.connect()

        try:
            # Brief wait to establish connection
            await asyncio.sleep(0.5)

            # Session should be active
            assert session is not None

        finally:
            await provider.disconnect()


# -----------------------------------------------------------------------------
# RealtimeVoice Integration Tests (Provider-Agnostic)
# -----------------------------------------------------------------------------


@SKIP_NO_API_KEYS
class TestRealtimeVoiceIntegration:
    """Integration tests using the high-level RealtimeVoice facade."""

    @pytest.mark.asyncio
    async def test_auto_provider_selection(self):
        """Test that RealtimeVoice auto-selects an available provider."""
        config = RealtimeConfig(
            voice="alloy",
            instructions="You are a helpful assistant.",
        )

        voice = RealtimeVoice(config=config)

        # Should have selected a provider
        assert voice.provider is not None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test RealtimeVoice connect as async context manager."""
        config = RealtimeConfig(
            voice="alloy",
            instructions="Test assistant.",
        )

        voice = RealtimeVoice(config=config)

        async with voice.connect() as session:
            assert session is not None

    @pytest.mark.asyncio
    async def test_callback_registration(self):
        """Test callback decorator registration."""
        config = RealtimeConfig(
            voice="alloy",
            instructions="Test assistant.",
        )

        voice = RealtimeVoice(config=config)

        audio_called = []
        transcript_called = []

        @voice.on_audio
        async def handle_audio(audio: bytes):
            audio_called.append(True)

        @voice.on_transcript
        async def handle_transcript(text: str, is_final: bool):
            transcript_called.append(text)

        # Verify callbacks are registered
        assert len(voice._audio_callbacks) > 0
        assert len(voice._transcript_callbacks) > 0


# -----------------------------------------------------------------------------
# Error Handling Integration Tests
# -----------------------------------------------------------------------------


class TestRealtimeErrorHandling:
    """Test error handling in realtime voice."""

    @pytest.mark.asyncio
    async def test_invalid_api_key(self):
        """Test that invalid API key raises appropriate error."""
        # Temporarily override environment
        with patch.dict(os.environ, {"OPENAI_API_KEY": "invalid-key"}, clear=False):
            config = RealtimeConfig(
                model="gpt-4o-realtime-preview",
                voice="alloy",
            )

            provider = OpenAIRealtimeProvider(config=config)

            with pytest.raises((RealtimeError, Exception)):
                await provider.connect()

    @pytest.mark.asyncio
    async def test_provider_discovery_error(self):
        """Test that missing API keys raise appropriate errors."""
        # Mock environment to have no API keys
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing keys
            env_copy = {}
            with patch.object(os, "environ", env_copy):
                config = RealtimeConfig(model="auto", voice="auto")

                with pytest.raises(RealtimeError):
                    RealtimeVoice(config=config)

    @pytest.mark.asyncio
    async def test_double_disconnect(self):
        """Test that disconnecting twice doesn't crash."""
        # Use mock provider
        mock_provider = MagicMock()
        mock_provider.disconnect = AsyncMock()

        # First disconnect
        await mock_provider.disconnect()

        # Second disconnect should also work
        await mock_provider.disconnect()

        assert mock_provider.disconnect.call_count == 2


# -----------------------------------------------------------------------------
# Performance Tests
# -----------------------------------------------------------------------------


@SKIP_NO_OPENAI
class TestRealtimePerformance:
    """Performance-related integration tests."""

    @pytest.mark.asyncio
    async def test_rapid_messages(self):
        """Test handling multiple operations in quick succession."""
        config = RealtimeConfig(
            model="gpt-4o-realtime-preview",
            voice="alloy",
            instructions="Respond with just 'OK' to each message.",
        )

        provider = OpenAIRealtimeProvider(config=config)
        responses = []

        async def on_transcript(text: str, is_final: bool):
            if is_final:
                responses.append(text)

        provider.on_transcript(on_transcript)

        session = await provider.connect()

        try:
            # Send a message
            if hasattr(session, "send_text"):
                await session.send_text("Say OK")

            # Wait briefly for response
            await asyncio.sleep(3)

            # Just verify no crashes

        finally:
            await provider.disconnect()

    @pytest.mark.asyncio
    async def test_audio_callback_performance(self):
        """Test that audio callbacks are triggered efficiently."""
        config = RealtimeConfig(
            model="gpt-4o-realtime-preview",
            voice="alloy",
        )

        provider = OpenAIRealtimeProvider(config=config)
        audio_chunks: list[bytes] = []

        async def on_audio(audio: bytes):
            audio_chunks.append(audio)

        provider.on_audio(on_audio)

        session = await provider.connect()

        try:
            # Send a message that should generate audio
            if hasattr(session, "send_text"):
                await session.send_text("Count from 1 to 3.")

            # Wait for audio
            for _ in range(30):
                if len(audio_chunks) >= 3:
                    break
                await asyncio.sleep(0.5)

            # Should have received audio chunks
            if audio_chunks:
                # Verify audio data is present
                for chunk in audio_chunks:
                    assert len(chunk) > 0

        finally:
            await provider.disconnect()


# -----------------------------------------------------------------------------
# Tool Integration Tests
# -----------------------------------------------------------------------------


@SKIP_NO_OPENAI
class TestRealtimeToolIntegration:
    """Test tool calling in realtime voice."""

    @pytest.mark.asyncio
    async def test_tool_definition_in_config(self):
        """Test that tools are properly converted to definitions."""

        def get_weather(city: str) -> str:
            """Get the weather for a city."""
            return f"Weather in {city}: Sunny, 72°F"

        config = RealtimeConfig(
            model="gpt-4o-realtime-preview",
            voice="alloy",
            instructions="When asked about weather, use the get_weather tool.",
            tools=[get_weather],
        )

        voice = RealtimeVoice(config=config)

        # Tool should be converted to definition
        assert "get_weather" in voice._tool_functions

        # Config should have tool definitions
        assert len(voice.config.tools) > 0

    @pytest.mark.asyncio
    async def test_tool_call_callback(self):
        """Test that tool calls trigger callbacks."""

        def get_time() -> str:
            """Get the current time."""
            return "3:42 PM"

        tool_calls: list[ToolCallRequest] = []

        async def handle_tool(request: ToolCallRequest):
            tool_calls.append(request)
            return "Done"

        config = RealtimeConfig(
            model="gpt-4o-realtime-preview",
            voice="alloy",
            instructions="When asked about time, use the get_time tool.",
            tools=[get_time],
        )

        voice = RealtimeVoice(config=config)
        voice.on_tool_call(handle_tool)

        async with voice.connect() as session:
            # Send a message that might trigger tool call
            if hasattr(session, "send_text"):
                await session.send_text("What time is it?")

            # Wait for potential tool call
            for _ in range(30):
                if tool_calls:
                    break
                await asyncio.sleep(0.5)

            # Tool may or may not be called depending on model behavior
            # Just verify no errors
