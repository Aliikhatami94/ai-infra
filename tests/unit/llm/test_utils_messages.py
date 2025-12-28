"""Tests for LLM message utilities (llm/utils/messages.py).

This module provides comprehensive tests for message handling:
- make_messages()
- is_valid_response()
- Multimodal message handling (images, audio)

Phase 0.2 of the ai-infra v1.0.0 release plan.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

from ai_infra.llm.utils.messages import is_valid_response, make_messages

# =============================================================================
# TEST: make_messages()
# =============================================================================


class TestMakeMessages:
    """Test make_messages function."""

    def test_simple_user_message(self):
        """Test simple user message creation."""
        messages = make_messages("Hello, world!")

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, world!"

    def test_with_system_message(self):
        """Test user message with system message."""
        messages = make_messages("Hello", system="You are a helpful assistant.")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    def test_with_extras(self):
        """Test user message with extra messages."""
        extras = [
            {"role": "assistant", "content": "I can help with that."},
            {"role": "user", "content": "Thanks!"},
        ]
        messages = make_messages("Hello", extras=extras)

        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "Thanks!"

    def test_with_system_and_extras(self):
        """Test with both system message and extras."""
        extras = [{"role": "assistant", "content": "Response"}]
        messages = make_messages(
            "Hello",
            system="System prompt",
            extras=extras,
        )

        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    def test_empty_user_message(self):
        """Test with empty user message."""
        messages = make_messages("")

        assert len(messages) == 1
        assert messages[0]["content"] == ""

    def test_provider_parameter_ignored(self):
        """Test provider parameter is ignored (backwards compat)."""
        messages = make_messages("Hello", provider="openai")

        assert len(messages) == 1
        assert messages[0]["content"] == "Hello"


# =============================================================================
# TEST: make_messages() with images
# =============================================================================


class TestMakeMessagesWithImages:
    """Test make_messages with image input."""

    def test_with_image_url(self):
        """Test message with image URL."""
        with patch("ai_infra.llm.multimodal.vision.build_vision_content") as mock_build:
            # Mock returns list with text and image blocks
            mock_build.return_value = [
                {"type": "text", "text": ""},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ]

            messages = make_messages(
                "Describe this image",
                images=["https://example.com/img.png"],
            )

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        # Content should be a list with text and image
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Describe this image"

    def test_with_multiple_images(self):
        """Test message with multiple images."""
        with patch("ai_infra.llm.multimodal.vision.build_vision_content") as mock_build:
            mock_build.return_value = [
                {"type": "text", "text": ""},
                {"type": "image_url", "image_url": {"url": "https://example.com/1.png"}},
                {"type": "image_url", "image_url": {"url": "https://example.com/2.png"}},
            ]

            messages = make_messages(
                "Compare these images",
                images=["https://example.com/1.png", "https://example.com/2.png"],
            )

        content = messages[0]["content"]
        assert isinstance(content, list)
        # Text + 2 images
        assert len(content) == 3

    def test_with_image_and_system(self):
        """Test message with image and system message."""
        with patch("ai_infra.llm.multimodal.vision.build_vision_content") as mock_build:
            mock_build.return_value = [
                {"type": "text", "text": ""},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ]

            messages = make_messages(
                "What's in this image?",
                system="You are an image analyzer.",
                images=["https://example.com/img.png"],
            )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert isinstance(messages[1]["content"], list)


# =============================================================================
# TEST: make_messages() with audio
# =============================================================================


class TestMakeMessagesWithAudio:
    """Test make_messages with audio input."""

    def test_with_audio_url(self):
        """Test message with audio URL."""
        with patch("ai_infra.llm.multimodal.audio.encode_audio") as mock_encode:
            mock_encode.return_value = {
                "type": "input_audio",
                "input_audio": {"data": "base64data", "format": "wav"},
            }

            messages = make_messages(
                "Transcribe this audio",
                audio="https://example.com/audio.wav",
            )

        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "input_audio"

    def test_with_audio_and_images(self):
        """Test message with both audio and images."""
        with (
            patch("ai_infra.llm.multimodal.vision.build_vision_content") as mock_vision,
            patch("ai_infra.llm.multimodal.audio.encode_audio") as mock_audio,
        ):
            mock_vision.return_value = [
                {"type": "text", "text": ""},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ]
            mock_audio.return_value = {
                "type": "input_audio",
                "input_audio": {"data": "base64data", "format": "wav"},
            }

            messages = make_messages(
                "Analyze this",
                images=["https://example.com/img.png"],
                audio="https://example.com/audio.wav",
            )

        content = messages[0]["content"]
        assert isinstance(content, list)
        # Text + image + audio
        assert len(content) == 3


# =============================================================================
# TEST: is_valid_response()
# =============================================================================


class TestIsValidResponse:
    """Test is_valid_response function."""

    def test_response_with_content_attribute(self):
        """Test response object with content attribute."""
        response = Mock()
        response.content = "Hello, world!"

        assert is_valid_response(response) is True

    def test_response_with_empty_content(self):
        """Test response with empty content is invalid."""
        response = Mock()
        response.content = ""

        assert is_valid_response(response) is False

    def test_response_with_whitespace_content(self):
        """Test response with whitespace-only content is invalid."""
        response = Mock()
        response.content = "   \n\t  "

        assert is_valid_response(response) is False

    def test_response_none_content(self):
        """Test response with None content checks dict format."""
        response = Mock(spec=[])  # No content attribute

        # Should return True since response is not None
        assert is_valid_response(response) is True

    def test_dict_response_with_messages(self):
        """Test dict response with messages list."""
        response = {
            "messages": [
                Mock(content="First message"),
                Mock(content="Last message"),
            ]
        }

        assert is_valid_response(response) is True

    def test_dict_response_with_empty_messages(self):
        """Test dict response with empty messages list."""
        response = {"messages": []}

        # Empty messages list, but response itself is not None
        assert is_valid_response(response) is True

    def test_dict_response_with_empty_last_content(self):
        """Test dict response where last message has empty content."""
        response = {
            "messages": [
                Mock(content="First"),
                Mock(content=""),
            ]
        }

        assert is_valid_response(response) is False

    def test_dict_message_format(self):
        """Test dict response with dict messages."""
        response = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }

        assert is_valid_response(response) is True

    def test_dict_message_empty_content(self):
        """Test dict response with dict message having empty content."""
        response = {
            "messages": [
                {"role": "assistant", "content": ""},
            ]
        }

        assert is_valid_response(response) is False

    def test_none_response(self):
        """Test None response is invalid."""
        assert is_valid_response(None) is False

    def test_string_response(self):
        """Test string response (non-empty) is valid."""
        # String doesn't have content attribute, but is not None
        assert is_valid_response("hello") is True

    def test_empty_string_response(self):
        """Test empty string response."""
        # Empty string is not None, so it's valid
        assert is_valid_response("") is True


# =============================================================================
# EDGE CASES
# =============================================================================


class TestMessagesEdgeCases:
    """Test edge cases for message utilities."""

    def test_unicode_in_messages(self):
        """Test unicode characters in messages."""
        messages = make_messages(
            "こんにちは",
            system="你好",
        )

        assert messages[0]["content"] == "你好"
        assert messages[1]["content"] == "こんにちは"

    def test_very_long_message(self):
        """Test very long message content."""
        long_message = "x" * 100000
        messages = make_messages(long_message)

        assert len(messages[0]["content"]) == 100000

    def test_special_characters(self):
        """Test special characters in messages."""
        messages = make_messages("Test with <script>alert('xss')</script> & newlines\n\t\r")

        assert "<script>" in messages[0]["content"]
        assert "\n" in messages[0]["content"]

    def test_multimodal_preserves_text(self):
        """Test multimodal message preserves user text."""
        with patch("ai_infra.llm.multimodal.vision.build_vision_content") as mock:
            mock.return_value = [
                {"type": "text", "text": ""},
                {"type": "image_url", "image_url": {"url": "test.png"}},
            ]

            messages = make_messages("Analyze this image", images=["test.png"])

        content = messages[0]["content"]
        text_block = content[0]
        assert text_block["text"] == "Analyze this image"

    def test_extras_order_preserved(self):
        """Test extras maintain order."""
        extras = [
            {"role": "assistant", "content": "First"},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "Third"},
        ]
        messages = make_messages("Start", extras=extras)

        assert messages[0]["content"] == "Start"
        assert messages[1]["content"] == "First"
        assert messages[2]["content"] == "Second"
        assert messages[3]["content"] == "Third"
