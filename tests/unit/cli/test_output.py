"""Unit tests for ai_infra.cli.output module.

Tests for the beautified CLI output utilities.
"""

from __future__ import annotations

from ai_infra.cli.output import (
    print_all_models,
    print_banner,
    print_cli_error,
    print_cli_info,
    print_cli_success,
    print_cli_warning,
    print_image_providers,
    print_models,
    print_providers,
    print_quick_start,
    print_sessions,
    print_tts_providers,
    print_voices,
)

# =============================================================================
# Provider Output Tests
# =============================================================================


class TestPrintProviders:
    """Tests for print_providers function."""

    def test_prints_all_providers(self, capsys) -> None:
        """Should print all providers."""
        providers = ["openai", "anthropic", "google"]
        configured = ["openai"]

        print_providers(providers, configured=configured)

        # Just verify it runs without error
        # Rich output goes to console, not capsys

    def test_handles_empty_providers(self, capsys) -> None:
        """Should handle empty provider list."""
        print_providers([])

    def test_handles_none_configured(self, capsys) -> None:
        """Should handle None configured list."""
        print_providers(["openai"], configured=None)


class TestPrintModels:
    """Tests for print_models function."""

    def test_prints_models_for_provider(self, capsys) -> None:
        """Should print models for a provider."""
        print_models("openai", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])

    def test_truncates_long_list(self, capsys) -> None:
        """Should truncate long model lists."""
        models = [f"model-{i}" for i in range(30)]
        print_models("test", models, max_display=10)

    def test_shows_all_when_requested(self, capsys) -> None:
        """Should show all models when show_all=True."""
        models = [f"model-{i}" for i in range(30)]
        print_models("test", models, show_all=True)

    def test_handles_empty_models(self, capsys) -> None:
        """Should handle empty model list."""
        print_models("test", [])


class TestPrintAllModels:
    """Tests for print_all_models function."""

    def test_prints_multiple_providers(self, capsys) -> None:
        """Should print models for multiple providers."""
        provider_models = {
            "openai": ["gpt-4o", "gpt-4o-mini"],
            "anthropic": ["claude-sonnet-4", "claude-opus-4"],
        }
        print_all_models(provider_models)

    def test_handles_empty_provider(self, capsys) -> None:
        """Should handle provider with no models."""
        provider_models = {
            "openai": ["gpt-4o"],
            "empty": [],
        }
        print_all_models(provider_models)


# =============================================================================
# Session Output Tests
# =============================================================================


class TestPrintSessions:
    """Tests for print_sessions function."""

    def test_prints_sessions(self, capsys) -> None:
        """Should print session list."""
        sessions = [
            {
                "session_id": "test-session",
                "message_count": 10,
                "model": "gpt-4o",
                "updated_at": "2026-01-15T10:00:00",
            },
        ]
        print_sessions(sessions)

    def test_handles_empty_sessions(self, capsys) -> None:
        """Should handle empty session list."""
        print_sessions([])

    def test_handles_missing_fields(self, capsys) -> None:
        """Should handle sessions with missing fields."""
        sessions = [{"session_id": "minimal"}]
        print_sessions(sessions)


# =============================================================================
# Error / Success Output Tests
# =============================================================================


class TestCLIMessages:
    """Tests for CLI message functions."""

    def test_print_cli_error(self, capsys) -> None:
        """Should print error message."""
        print_cli_error("Something went wrong")

    def test_print_cli_error_with_hint(self, capsys) -> None:
        """Should print error with hint."""
        print_cli_error("API key not set", hint="Set OPENAI_API_KEY")

    def test_print_cli_warning(self, capsys) -> None:
        """Should print warning message."""
        print_cli_warning("Proceeding with defaults")

    def test_print_cli_success(self, capsys) -> None:
        """Should print success message."""
        print_cli_success("Operation completed")

    def test_print_cli_success_with_details(self, capsys) -> None:
        """Should print success with details."""
        print_cli_success("Created file", details="src/main.py")

    def test_print_cli_info(self, capsys) -> None:
        """Should print info message."""
        print_cli_info("Processing request...")


# =============================================================================
# Banner / Help Tests
# =============================================================================


class TestBannerAndHelp:
    """Tests for banner and help functions."""

    def test_print_banner(self, capsys) -> None:
        """Should print banner."""
        print_banner()

    def test_print_quick_start(self, capsys) -> None:
        """Should print quick start guide."""
        print_quick_start()


# =============================================================================
# Multimodal Output Tests
# =============================================================================


class TestMultimodalOutput:
    """Tests for image/TTS/STT output functions."""

    def test_print_image_providers(self, capsys) -> None:
        """Should print image providers."""
        print_image_providers(["openai", "stability"], configured=["openai"])

    def test_print_tts_providers(self, capsys) -> None:
        """Should print TTS providers."""
        print_tts_providers(["openai", "elevenlabs"], configured=["openai"])

    def test_print_voices(self, capsys) -> None:
        """Should print voices."""
        voices = [
            {"id": "alloy", "name": "Alloy", "language": "en"},
            {"id": "echo", "name": "Echo", "language": "en"},
        ]
        print_voices("openai", voices)

    def test_print_voices_truncates(self, capsys) -> None:
        """Should truncate long voice list."""
        voices = [{"id": f"voice-{i}", "name": f"Voice {i}"} for i in range(30)]
        print_voices("test", voices)
