"""Tests for LLM vision (image input) functionality."""

from __future__ import annotations

from ai_infra.llm.multimodal.vision import build_vision_content, create_vision_message, encode_image


class TestEncodeImage:
    """Tests for encode_image function."""

    def test_encode_bytes(self):
        """Test encoding raw bytes."""
        img_bytes = b"fake image data"
        result = encode_image(img_bytes)

        # Should return dict with image_url type
        assert result["type"] == "image_url"
        assert "image_url" in result
        assert result["image_url"]["url"].startswith("data:image/")

    def test_encode_url(self):
        """Test encoding URL."""
        url = "https://example.com/image.jpg"
        result = encode_image(url)

        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == url

    def test_encode_data_url(self):
        """Test data URL encoding."""
        data_url = "data:image/png;base64,iVBORw0KGgo="
        result = encode_image(data_url)

        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == data_url


class TestBuildVisionContent:
    """Tests for build_vision_content function."""

    def test_single_image(self):
        """Test building content with single image."""
        content = build_vision_content("Describe this", [b"fake image"])

        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Describe this"
        assert content[1]["type"] == "image_url"

    def test_multiple_images(self):
        """Test building content with multiple images."""
        content = build_vision_content(
            "Compare these",
            [b"image1", b"image2", b"image3"],
        )

        assert len(content) == 4  # 1 text + 3 images
        assert content[0]["type"] == "text"
        for i in range(1, 4):
            assert content[i]["type"] == "image_url"


class TestCreateVisionMessage:
    """Tests for create_vision_message function."""

    def test_creates_human_message(self):
        """Test that it creates a HumanMessage."""
        msg = create_vision_message("What's this?", [b"image"])

        assert msg.__class__.__name__ == "HumanMessage"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
