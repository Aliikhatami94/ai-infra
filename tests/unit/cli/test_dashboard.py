"""Unit tests for ai_infra.cli.dashboard module.

Phase 16.6.3 of EXECUTOR_6.md: Live Status Dashboard.
"""

from __future__ import annotations

import time
from io import StringIO

from rich.console import Console

from ai_infra.cli.dashboard import (
    DEFAULT_SHORTCUTS,
    KeyboardShortcut,
    LiveStatusBar,
    ModelIndicator,
    ModelProvider,
    StatusBar,
    StatusBarState,
    SubagentActivity,
    TokenBudget,
    TokenBudgetLevel,
    get_model_provider,
    render_keyboard_hints,
)

# =============================================================================
# Model Provider Tests
# =============================================================================


class TestModelProvider:
    """Tests for model provider detection."""

    def test_anthropic_models(self) -> None:
        """Claude models should map to Anthropic."""
        assert get_model_provider("claude-sonnet-4") == ModelProvider.ANTHROPIC
        assert get_model_provider("claude-3-opus") == ModelProvider.ANTHROPIC
        assert get_model_provider("claude-instant") == ModelProvider.ANTHROPIC

    def test_openai_models(self) -> None:
        """GPT and o-series models should map to OpenAI."""
        assert get_model_provider("gpt-4o") == ModelProvider.OPENAI
        assert get_model_provider("gpt-4-turbo") == ModelProvider.OPENAI
        assert get_model_provider("o1-preview") == ModelProvider.OPENAI
        assert get_model_provider("o3-mini") == ModelProvider.OPENAI

    def test_google_models(self) -> None:
        """Gemini models should map to Google."""
        assert get_model_provider("gemini-2.0-flash") == ModelProvider.GOOGLE
        assert get_model_provider("gemini-pro") == ModelProvider.GOOGLE

    def test_mistral_models(self) -> None:
        """Mistral models should map to Mistral."""
        assert get_model_provider("mistral-large") == ModelProvider.MISTRAL
        assert get_model_provider("mixtral-8x7b") == ModelProvider.MISTRAL
        assert get_model_provider("codestral-latest") == ModelProvider.MISTRAL

    def test_other_providers(self) -> None:
        """Other known providers should be detected."""
        assert get_model_provider("command-r") == ModelProvider.COHERE
        assert get_model_provider("llama-3") == ModelProvider.META
        assert get_model_provider("titan-express") == ModelProvider.AMAZON
        assert get_model_provider("nova-pro") == ModelProvider.AMAZON

    def test_unknown_model(self) -> None:
        """Unknown models should return UNKNOWN provider."""
        assert get_model_provider("some-unknown-model") == ModelProvider.UNKNOWN
        assert get_model_provider("custom-fine-tuned") == ModelProvider.UNKNOWN

    def test_case_insensitive(self) -> None:
        """Detection should be case-insensitive."""
        assert get_model_provider("CLAUDE-SONNET-4") == ModelProvider.ANTHROPIC
        assert get_model_provider("GPT-4o") == ModelProvider.OPENAI


# =============================================================================
# Model Indicator Tests
# =============================================================================


class TestModelIndicator:
    """Tests for model indicator display."""

    def test_render_with_provider(self) -> None:
        """Should render model name with provider abbreviation."""
        indicator = ModelIndicator("claude-sonnet-4")
        text = indicator.render()
        plain = text.plain
        assert "Model:" in plain
        assert "claude-sonnet-4" in plain
        assert "ANT" in plain

    def test_render_without_provider(self) -> None:
        """Should render without provider when disabled."""
        indicator = ModelIndicator("claude-sonnet-4", show_provider=False)
        text = indicator.render()
        plain = text.plain
        assert "claude-sonnet-4" in plain
        assert "ANT" not in plain

    def test_render_compact(self) -> None:
        """Compact render should abbreviate long names."""
        indicator = ModelIndicator("claude-sonnet-4-20250514-extended")
        text = indicator.render_compact()
        plain = text.plain
        # Should be truncated
        assert len(plain) <= 20

    def test_render_compact_short_name(self) -> None:
        """Compact render should not truncate short names."""
        indicator = ModelIndicator("gpt-4o")
        text = indicator.render_compact()
        assert text.plain == "gpt-4o"


# =============================================================================
# Token Budget Tests
# =============================================================================


class TestTokenBudget:
    """Tests for token budget visualization."""

    def test_percentage_calculation(self) -> None:
        """Should calculate percentage correctly."""
        budget = TokenBudget(used=5000, limit=10000)
        assert budget.percentage == 50.0

    def test_percentage_zero_limit(self) -> None:
        """Should handle zero limit gracefully."""
        budget = TokenBudget(used=100, limit=0)
        assert budget.percentage == 0.0

    def test_level_low(self) -> None:
        """0-50% should be low level."""
        budget = TokenBudget(used=4000, limit=10000)
        assert budget.level == TokenBudgetLevel.LOW

    def test_level_medium(self) -> None:
        """50-80% should be medium level."""
        budget = TokenBudget(used=6000, limit=10000)
        assert budget.level == TokenBudgetLevel.MEDIUM

    def test_level_high(self) -> None:
        """80-100% should be high level."""
        budget = TokenBudget(used=9000, limit=10000)
        assert budget.level == TokenBudgetLevel.HIGH

    def test_style_low(self) -> None:
        """Low level should be green."""
        budget = TokenBudget(used=1000, limit=10000)
        assert "green" in budget.get_style()

    def test_style_medium(self) -> None:
        """Medium level should be yellow."""
        budget = TokenBudget(used=6000, limit=10000)
        assert "yellow" in budget.get_style()

    def test_style_high(self) -> None:
        """High level should be red."""
        budget = TokenBudget(used=9000, limit=10000)
        assert "red" in budget.get_style()

    def test_format_count_small(self) -> None:
        """Small counts should be plain numbers."""
        budget = TokenBudget(used=500, limit=1000)
        assert budget._format_count(500) == "500"

    def test_format_count_thousands(self) -> None:
        """Large counts should use k suffix."""
        budget = TokenBudget(used=12400, limit=100000)
        assert budget._format_count(12400) == "12.4k"

    def test_format_count_millions(self) -> None:
        """Very large counts should use M suffix."""
        budget = TokenBudget(used=1500000, limit=2000000)
        assert budget._format_count(1500000) == "1.5M"

    def test_render_contains_values(self) -> None:
        """Render should include token values."""
        budget = TokenBudget(used=12400, limit=100000)
        text = budget.render()
        plain = text.plain
        assert "Tokens:" in plain
        assert "12.4k" in plain
        assert "100.0k" in plain

    def test_render_compact(self) -> None:
        """Compact render should omit label."""
        budget = TokenBudget(used=12400, limit=100000)
        text = budget.render_compact()
        plain = text.plain
        assert "Tokens:" not in plain
        assert "12.4k" in plain


# =============================================================================
# Subagent Activity Tests
# =============================================================================


class TestSubagentActivity:
    """Tests for subagent activity indicator."""

    def test_render_coder(self) -> None:
        """Should render coder activity."""
        activity = SubagentActivity("coder", "implementing auth module...")
        text = activity.render()
        plain = text.plain
        assert "coder" in plain
        assert "implementing auth module..." in plain

    def test_render_tester(self) -> None:
        """Should render tester activity."""
        activity = SubagentActivity("tester", "running unit tests...")
        text = activity.render()
        plain = text.plain
        assert "tester" in plain
        assert "running unit tests..." in plain

    def test_render_reviewer(self) -> None:
        """Should render reviewer activity."""
        activity = SubagentActivity("reviewer", "analyzing code quality...")
        text = activity.render()
        plain = text.plain
        assert "reviewer" in plain

    def test_agent_style(self) -> None:
        """Should return correct agent style."""
        coder = SubagentActivity("coder", "test")
        assert coder.get_agent_style() == "agent.coder"

        tester = SubagentActivity("tester", "test")
        assert tester.get_agent_style() == "agent.tester"

    def test_unknown_agent_style(self) -> None:
        """Unknown agent should get default style."""
        activity = SubagentActivity("unknown", "test")
        assert activity.get_agent_style() == "bold white"

    def test_render_compact(self) -> None:
        """Compact render should truncate long status."""
        activity = SubagentActivity("coder", "implementing a very long description...")
        text = activity.render_compact()
        plain = text.plain
        assert len(plain) < len(activity.status) + 10  # Should be truncated


# =============================================================================
# Keyboard Hints Tests
# =============================================================================


class TestKeyboardHints:
    """Tests for keyboard shortcut hints."""

    def test_default_shortcuts_exist(self) -> None:
        """DEFAULT_SHORTCUTS should have standard shortcuts."""
        keys = [s.key for s in DEFAULT_SHORTCUTS]
        assert "q" in keys
        assert "p" in keys
        assert "?" in keys

    def test_render_all_shortcuts(self) -> None:
        """Should render all shortcuts."""
        text = render_keyboard_hints()
        plain = text.plain
        assert "[q]" in plain
        assert "quit" in plain
        assert "[?]" in plain
        assert "help" in plain

    def test_render_compact(self) -> None:
        """Compact mode should show fewer shortcuts."""
        full = render_keyboard_hints(compact=False)
        compact = render_keyboard_hints(compact=True)
        assert len(compact.plain) < len(full.plain)

    def test_custom_shortcuts(self) -> None:
        """Should render custom shortcuts."""
        shortcuts = [
            KeyboardShortcut("x", "execute"),
            KeyboardShortcut("r", "retry"),
        ]
        text = render_keyboard_hints(shortcuts)
        plain = text.plain
        assert "[x]" in plain
        assert "execute" in plain
        assert "[r]" in plain
        assert "retry" in plain


# =============================================================================
# Status Bar State Tests
# =============================================================================


class TestStatusBarState:
    """Tests for StatusBarState dataclass."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        state = StatusBarState()
        assert state.model_name == "unknown"
        assert state.tokens_used == 0
        assert state.tokens_limit == 100_000
        assert state.current_phase == 1
        assert state.total_phases == 1
        assert state.show_shortcuts is True

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        state = StatusBarState(
            model_name="claude-sonnet-4",
            tokens_used=12400,
            tokens_limit=100000,
            current_phase=2,
            total_phases=4,
        )
        assert state.model_name == "claude-sonnet-4"
        assert state.tokens_used == 12400
        assert state.current_phase == 2


# =============================================================================
# Status Bar Tests
# =============================================================================


class TestStatusBar:
    """Tests for StatusBar renderable."""

    def test_creation(self) -> None:
        """StatusBar should be creatable."""
        bar = StatusBar()
        assert bar.title == "ai-infra executor"

    def test_creation_with_state(self) -> None:
        """StatusBar should accept initial state."""
        state = StatusBarState(model_name="gpt-4o")
        bar = StatusBar(state)
        assert bar.state.model_name == "gpt-4o"

    def test_renders_without_error(self) -> None:
        """StatusBar should render to console."""
        state = StatusBarState(
            model_name="claude-sonnet-4",
            tokens_used=12400,
            tokens_limit=100000,
            elapsed_seconds=154.0,
            current_phase=2,
            total_phases=4,
            completed_tasks=5,
            total_tasks=12,
        )
        bar = StatusBar(state)

        console = Console(file=StringIO(), force_terminal=True, width=100)
        console.print(bar)
        output = console.file.getvalue()

        assert "ai-infra executor" in output
        assert "claude-sonnet-4" in output
        assert "2/4" in output  # Phase
        assert "5/12" in output  # Tasks

    def test_renders_with_agent_activity(self) -> None:
        """StatusBar should show agent activity."""
        state = StatusBarState(
            model_name="claude-sonnet-4",
            active_agent="coder",
            agent_status="writing tests...",
        )
        bar = StatusBar(state)

        console = Console(file=StringIO(), force_terminal=True, width=100)
        console.print(bar)
        output = console.file.getvalue()

        assert "coder" in output
        assert "writing tests" in output

    def test_narrow_terminal_layout(self) -> None:
        """StatusBar should adapt to narrow terminals."""
        state = StatusBarState(model_name="claude-sonnet-4")
        bar = StatusBar(state)

        # Narrow terminal (70 cols)
        console = Console(file=StringIO(), force_terminal=True, width=70)
        console.print(bar)
        output = console.file.getvalue()

        # Should still render without error
        assert "ai-infra" in output

    def test_is_narrow_detection(self) -> None:
        """Should detect narrow terminals correctly."""
        bar = StatusBar()
        assert bar._is_narrow(79) is True
        assert bar._is_narrow(80) is False
        assert bar._is_narrow(100) is False


# =============================================================================
# Live Status Bar Tests
# =============================================================================


class TestLiveStatusBar:
    """Tests for LiveStatusBar."""

    def test_creation(self) -> None:
        """LiveStatusBar should be creatable."""
        bar = LiveStatusBar()
        assert bar.state is not None
        assert bar.title == "ai-infra executor"

    def test_update_state(self) -> None:
        """update() should modify state."""
        bar = LiveStatusBar()
        bar.update(model_name="gpt-4o", tokens_used=5000)
        assert bar.state.model_name == "gpt-4o"
        assert bar.state.tokens_used == 5000

    def test_set_agent_activity(self) -> None:
        """set_agent_activity() should update agent state."""
        bar = LiveStatusBar()
        bar.set_agent_activity("tester", "running tests...")
        assert bar.state.active_agent == "tester"
        assert bar.state.agent_status == "running tests..."

    def test_clear_agent_activity(self) -> None:
        """clear_agent_activity() should clear agent state."""
        bar = LiveStatusBar()
        bar.set_agent_activity("coder", "working...")
        bar.clear_agent_activity()
        assert bar.state.active_agent is None
        assert bar.state.agent_status is None

    def test_increment_tasks(self) -> None:
        """increment_tasks() should increase count."""
        bar = LiveStatusBar()
        bar.state.completed_tasks = 5
        bar.increment_tasks()
        assert bar.state.completed_tasks == 6

    def test_set_phase(self) -> None:
        """set_phase() should update phase state."""
        bar = LiveStatusBar()
        bar.set_phase(3, 5)
        assert bar.state.current_phase == 3
        assert bar.state.total_phases == 5

    def test_set_phase_current_only(self) -> None:
        """set_phase() should work with just current."""
        bar = LiveStatusBar()
        bar.state.total_phases = 10
        bar.set_phase(7)
        assert bar.state.current_phase == 7
        assert bar.state.total_phases == 10

    def test_set_tokens(self) -> None:
        """set_tokens() should update token state."""
        bar = LiveStatusBar()
        bar.set_tokens(25000, 50000)
        assert bar.state.tokens_used == 25000
        assert bar.state.tokens_limit == 50000

    def test_set_tokens_used_only(self) -> None:
        """set_tokens() should work with just used."""
        bar = LiveStatusBar()
        bar.state.tokens_limit = 100000
        bar.set_tokens(30000)
        assert bar.state.tokens_used == 30000
        assert bar.state.tokens_limit == 100000

    def test_stop_returns_elapsed(self) -> None:
        """stop() should return elapsed time."""
        bar = LiveStatusBar()
        bar._start_time = time.time() - 5.0
        elapsed = bar.stop()
        assert 4.9 < elapsed < 5.5
