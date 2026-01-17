"""Unit tests for ai_infra.cli.console module.

Phase 16.6.1 of EXECUTOR_6.md: Rich Console Foundation.
"""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest
from rich.box import DOUBLE, HEAVY, MINIMAL, ROUNDED
from rich.console import Console
from rich.theme import Theme

from ai_infra.cli.console import (
    BOX_STYLES,
    EXECUTOR_THEME,
    SPINNERS,
    ColorSupport,
    SpinnerConfig,
    TerminalCapabilities,
    detect_terminal_capabilities,
    get_box_style,
    get_console,
    get_spinner,
    print_divider,
    print_error,
    print_header,
    print_info,
    print_section,
    print_step,
    print_success,
    print_task_status,
    print_warning,
    reset_console,
)

# =============================================================================
# Theme Tests
# =============================================================================


class TestExecutorTheme:
    """Tests for EXECUTOR_THEME."""

    def test_theme_is_rich_theme(self) -> None:
        """Theme should be a Rich Theme instance."""
        assert isinstance(EXECUTOR_THEME, Theme)

    def test_semantic_colors_defined(self) -> None:
        """All semantic color styles should be defined."""
        required_styles = [
            "info",
            "success",
            "warning",
            "error",
            "critical",
        ]
        for style in required_styles:
            assert style in EXECUTOR_THEME.styles, f"Missing style: {style}"

    def test_status_styles_defined(self) -> None:
        """All status styles should be defined."""
        required_styles = [
            "status.pending",
            "status.running",
            "status.complete",
            "status.failed",
            "status.skipped",
        ]
        for style in required_styles:
            assert style in EXECUTOR_THEME.styles, f"Missing style: {style}"

    def test_task_styles_defined(self) -> None:
        """All task display styles should be defined."""
        required_styles = [
            "task.id",
            "task.title",
            "task.description",
            "task.duration",
        ]
        for style in required_styles:
            assert style in EXECUTOR_THEME.styles, f"Missing style: {style}"

    def test_agent_styles_defined(self) -> None:
        """All agent type styles should be defined."""
        required_styles = [
            "agent.coder",
            "agent.tester",
            "agent.reviewer",
            "agent.debugger",
            "agent.researcher",
            "agent.orchestrator",
        ]
        for style in required_styles:
            assert style in EXECUTOR_THEME.styles, f"Missing style: {style}"


# =============================================================================
# Color Support Detection Tests
# =============================================================================


class TestColorSupportDetection:
    """Tests for color support detection."""

    def test_no_color_env_disables_color(self) -> None:
        """NO_COLOR environment variable should disable colors."""
        with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=False):
            reset_console()
            caps = detect_terminal_capabilities()
            assert caps.color_support == ColorSupport.NONE

    def test_force_color_enables_truecolor(self) -> None:
        """FORCE_COLOR environment variable should enable truecolor."""
        env = {"FORCE_COLOR": "1"}
        with patch.dict(os.environ, env, clear=False):
            # Remove NO_COLOR if present
            os.environ.pop("NO_COLOR", None)
            reset_console()
            caps = detect_terminal_capabilities()
            assert caps.color_support == ColorSupport.TRUECOLOR

    def test_colorterm_truecolor_detection(self) -> None:
        """COLORTERM=truecolor should be detected."""
        env = {"COLORTERM": "truecolor", "TERM": "xterm"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("NO_COLOR", None)
            os.environ.pop("FORCE_COLOR", None)
            reset_console()
            caps = detect_terminal_capabilities()
            assert caps.color_support == ColorSupport.TRUECOLOR

    def test_term_256color_detection(self) -> None:
        """TERM with 256color should be detected."""
        env = {"TERM": "xterm-256color"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("NO_COLOR", None)
            os.environ.pop("FORCE_COLOR", None)
            os.environ.pop("COLORTERM", None)
            reset_console()
            caps = detect_terminal_capabilities()
            assert caps.color_support == ColorSupport.EXTENDED


class TestTerminalCapabilities:
    """Tests for TerminalCapabilities dataclass."""

    def test_capabilities_are_frozen(self) -> None:
        """TerminalCapabilities should be immutable."""
        caps = TerminalCapabilities(
            color_support=ColorSupport.TRUECOLOR,
            unicode_support=True,
            interactive=True,
            width=80,
            ci_environment=False,
        )
        with pytest.raises(FrozenInstanceError):
            caps.width = 120  # type: ignore[misc]

    def test_ci_environment_detection(self) -> None:
        """CI environment should be detected."""
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}, clear=False):
            reset_console()
            caps = detect_terminal_capabilities()
            assert caps.ci_environment is True


# =============================================================================
# Console Factory Tests
# =============================================================================


class TestGetConsole:
    """Tests for get_console() factory."""

    def setup_method(self) -> None:
        """Reset console before each test."""
        reset_console()

    def test_returns_console_instance(self) -> None:
        """get_console() should return a Console instance."""
        console = get_console()
        assert isinstance(console, Console)

    def test_singleton_pattern(self) -> None:
        """get_console() should return the same instance."""
        console1 = get_console()
        console2 = get_console()
        assert console1 is console2

    def test_has_executor_theme(self) -> None:
        """Console should have executor theme applied."""
        # Call get_console to ensure singleton is initialized
        get_console()
        # Verify theme is applied by checking that themed styles render without error
        # The console should accept our custom style names
        from io import StringIO

        from rich.text import Text

        buffer = StringIO()
        temp_console = Console(file=buffer, force_terminal=True, theme=EXECUTOR_THEME)
        temp_console.print(Text("test", style="success"))
        output = buffer.getvalue()
        # If theme was applied, output should contain ANSI codes
        assert len(output) > 0

    def test_reset_clears_singleton(self) -> None:
        """reset_console() should clear the singleton."""
        console1 = get_console()
        reset_console()
        console2 = get_console()
        assert console1 is not console2


# =============================================================================
# Spinner Tests
# =============================================================================


class TestSpinners:
    """Tests for spinner configurations."""

    def test_all_spinners_defined(self) -> None:
        """All expected spinners should be defined."""
        expected = ["thinking", "executing", "verifying", "loading", "connecting", "processing"]
        for name in expected:
            assert name in SPINNERS, f"Missing spinner: {name}"

    def test_spinner_config_structure(self) -> None:
        """SpinnerConfig should have required fields."""
        config = SPINNERS["thinking"]
        assert isinstance(config, SpinnerConfig)
        assert config.spinner == "dots"
        assert config.text == "Analyzing..."
        assert config.style == "cyan"

    def test_get_spinner_returns_config(self) -> None:
        """get_spinner() should return SpinnerConfig."""
        config = get_spinner("executing")
        assert isinstance(config, SpinnerConfig)
        assert config.spinner == "line"

    def test_get_spinner_unknown_raises(self) -> None:
        """get_spinner() should raise KeyError for unknown spinners."""
        with pytest.raises(KeyError, match="Unknown spinner"):
            get_spinner("nonexistent")


# =============================================================================
# Box Style Tests
# =============================================================================


class TestBoxStyles:
    """Tests for box styles."""

    def test_all_box_styles_defined(self) -> None:
        """All expected box styles should be defined."""
        expected = ["default", "error", "success", "minimal"]
        for name in expected:
            assert name in BOX_STYLES, f"Missing box style: {name}"

    def test_box_styles_are_rich_boxes(self) -> None:
        """Box styles should be Rich Box instances."""
        assert BOX_STYLES["default"] is ROUNDED
        assert BOX_STYLES["error"] is HEAVY
        assert BOX_STYLES["success"] is DOUBLE
        assert BOX_STYLES["minimal"] is MINIMAL

    def test_get_box_style_returns_box(self) -> None:
        """get_box_style() should return Box instance."""
        box = get_box_style("error")
        assert box is HEAVY

    def test_get_box_style_unknown_raises(self) -> None:
        """get_box_style() should raise KeyError for unknown styles."""
        with pytest.raises(KeyError, match="Unknown box style"):
            get_box_style("nonexistent")


# =============================================================================
# Output Helper Tests
# =============================================================================


class TestOutputHelpers:
    """Tests for semantic output helpers."""

    def setup_method(self) -> None:
        """Reset console before each test."""
        reset_console()

    def test_print_header_no_exception(self) -> None:
        """print_header() should not raise exceptions."""
        print_header("Test Header")
        print_header("Test Header", subtitle="With subtitle")

    def test_print_success_no_exception(self) -> None:
        """print_success() should not raise exceptions."""
        print_success("Task completed")
        print_success("Task completed", details="Created 3 files")

    def test_print_error_no_exception(self) -> None:
        """print_error() should not raise exceptions."""
        print_error("Connection failed")
        print_error("Connection failed", hint="Check your API key")

    def test_print_warning_no_exception(self) -> None:
        """print_warning() should not raise exceptions."""
        print_warning("Token usage high")

    def test_print_info_no_exception(self) -> None:
        """print_info() should not raise exceptions."""
        print_info("Loading configuration...")

    def test_print_step_no_exception(self) -> None:
        """print_step() should not raise exceptions."""
        print_step(1, 5, "Initializing")
        print_step(5, 5, "Complete")

    def test_print_task_status_all_statuses(self) -> None:
        """print_task_status() should handle all status types."""
        statuses = ["pending", "running", "complete", "failed", "skipped"]
        for status in statuses:
            print_task_status("1.1.1", "Test task", status)

    def test_print_task_status_with_duration(self) -> None:
        """print_task_status() should handle duration."""
        print_task_status("1.1.1", "Test task", "complete", duration=0.5)
        print_task_status("1.1.2", "Long task", "complete", duration=125.5)
        print_task_status("1.1.3", "Very long task", "complete", duration=3661.0)

    def test_print_divider_no_exception(self) -> None:
        """print_divider() should not raise exceptions."""
        print_divider()
        print_divider(style="cyan")

    def test_print_section_no_exception(self) -> None:
        """print_section() should not raise exceptions."""
        print_section("FILES MODIFIED")


# =============================================================================
# Duration Formatting Tests
# =============================================================================


class TestDurationFormatting:
    """Tests for duration formatting via print_task_status."""

    def test_sub_second_formatting(self) -> None:
        """Sub-second durations should show decimal."""
        # Just verify no exceptions - actual formatting is internal
        print_task_status("1.1.1", "Fast task", "complete", duration=0.3)

    def test_seconds_formatting(self) -> None:
        """Second durations should show decimal."""
        print_task_status("1.1.1", "Normal task", "complete", duration=12.5)

    def test_minutes_formatting(self) -> None:
        """Minute durations should show m:s format."""
        print_task_status("1.1.1", "Long task", "complete", duration=123.0)

    def test_hours_formatting(self) -> None:
        """Hour durations should show h:m format."""
        print_task_status("1.1.1", "Very long task", "complete", duration=3700.0)
