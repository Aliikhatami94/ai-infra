"""Tests for multimodal agent tools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestTranscribeAudioTool:
    """Tests for transcribe_audio tool."""

    def test_tool_has_correct_name(self):
        """Test that tool has correct name."""
        from ai_infra.llm.tools.custom.multimodal import transcribe_audio

        assert transcribe_audio.name == "transcribe_audio"

    def test_tool_has_description(self):
        """Test that tool has a description."""
        from ai_infra.llm.tools.custom.multimodal import transcribe_audio

        assert transcribe_audio.description
        assert "transcribe" in transcribe_audio.description.lower()


class TestAnalyzeImageTool:
    """Tests for analyze_image tool."""

    def test_tool_has_correct_name(self):
        """Test that tool has correct name."""
        from ai_infra.llm.tools.custom.multimodal import analyze_image

        assert analyze_image.name == "analyze_image"

    def test_tool_has_description(self):
        """Test that tool has a description."""
        from ai_infra.llm.tools.custom.multimodal import analyze_image

        assert analyze_image.description
        assert "image" in analyze_image.description.lower()


class TestGenerateImageTool:
    """Tests for generate_image tool."""

    def test_tool_has_correct_name(self):
        """Test that tool has correct name."""
        from ai_infra.llm.tools.custom.multimodal import generate_image

        assert generate_image.name == "generate_image"

    def test_tool_has_description(self):
        """Test that tool has a description."""
        from ai_infra.llm.tools.custom.multimodal import generate_image

        assert generate_image.description
        assert "generate" in generate_image.description.lower()


class TestToolImports:
    """Tests for tool imports."""

    def test_all_tools_importable(self):
        """Test that all tools can be imported."""
        from ai_infra.llm.tools.custom.multimodal import (
            analyze_image,
            generate_image,
            transcribe_audio,
        )

        assert transcribe_audio is not None
        assert analyze_image is not None
        assert generate_image is not None

    def test_tools_exported_from_main(self):
        """Test that tools are exported from main tools module."""
        from ai_infra.llm.tools import analyze_image, generate_image, transcribe_audio

        assert transcribe_audio is not None
        assert analyze_image is not None
        assert generate_image is not None
