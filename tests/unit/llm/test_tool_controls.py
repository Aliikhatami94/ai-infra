"""Tests for llm/tools/tool_controls.py - Tool Control Configuration.

This module tests the tool control system which provides:
- ToolCallControls dataclass for configuring tool behavior
- Helper functions (no_tools, force_tool)
- Provider-specific normalization of tool controls
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_infra.llm.providers import Providers
from ai_infra.llm.tools.tool_controls import (
    ToolCallControls,
    _ensure_dict,
    _extract_name,
    force_tool,
    no_tools,
    normalize_tool_controls,
)

# =============================================================================
# ToolCallControls Dataclass Tests
# =============================================================================


class TestToolCallControls:
    """Tests for ToolCallControls dataclass."""

    def test_default_values(self):
        """Test ToolCallControls has correct default values."""
        controls = ToolCallControls()

        assert controls.tool_choice is None
        assert controls.parallel_tool_calls is True
        assert controls.force_once is False

    def test_with_tool_choice_dict(self):
        """Test ToolCallControls with tool_choice as dict."""
        controls = ToolCallControls(tool_choice={"name": "calculator"})

        assert controls.tool_choice == {"name": "calculator"}

    def test_with_tool_choice_string(self):
        """Test ToolCallControls with tool_choice as string."""
        controls = ToolCallControls(tool_choice="none")
        assert controls.tool_choice == "none"

        controls = ToolCallControls(tool_choice="auto")
        assert controls.tool_choice == "auto"

        controls = ToolCallControls(tool_choice="any")
        assert controls.tool_choice == "any"

    def test_parallel_tool_calls_disabled(self):
        """Test ToolCallControls with parallel_tool_calls disabled."""
        controls = ToolCallControls(parallel_tool_calls=False)

        assert controls.parallel_tool_calls is False

    def test_force_once_enabled(self):
        """Test ToolCallControls with force_once enabled."""
        controls = ToolCallControls(force_once=True)

        assert controls.force_once is True

    def test_all_options_combined(self):
        """Test ToolCallControls with all options set."""
        controls = ToolCallControls(
            tool_choice={"name": "search"},
            parallel_tool_calls=False,
            force_once=True,
        )

        assert controls.tool_choice == {"name": "search"}
        assert controls.parallel_tool_calls is False
        assert controls.force_once is True


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestNoTools:
    """Tests for no_tools helper function."""

    def test_no_tools_returns_dict(self):
        """Test no_tools returns correct structure."""
        result = no_tools()

        assert isinstance(result, dict)
        assert "tool_controls" in result

    def test_no_tools_sets_tool_choice_none(self):
        """Test no_tools sets tool_choice to 'none'."""
        result = no_tools()

        assert result["tool_controls"]["tool_choice"] == "none"

    def test_no_tools_is_mergeable(self):
        """Test no_tools result can be used as kwargs."""
        result = no_tools()

        # Should be usable in function calls like: llm.chat(..., **no_tools())
        assert isinstance(result, dict)
        assert len(result) == 1


class TestForceTool:
    """Tests for force_tool helper function."""

    def test_force_tool_basic(self):
        """Test force_tool with just a name."""
        result = force_tool("calculator")

        assert isinstance(result, dict)
        assert "tool_controls" in result
        assert result["tool_controls"]["tool_choice"] == {"name": "calculator"}

    def test_force_tool_sets_defaults(self):
        """Test force_tool sets correct defaults."""
        result = force_tool("my_tool")

        controls = result["tool_controls"]
        assert controls["force_once"] is False
        assert controls["parallel_tool_calls"] is False

    def test_force_tool_with_once(self):
        """Test force_tool with once=True."""
        result = force_tool("my_tool", once=True)

        assert result["tool_controls"]["force_once"] is True

    def test_force_tool_with_parallel(self):
        """Test force_tool with parallel=True."""
        result = force_tool("my_tool", parallel=True)

        assert result["tool_controls"]["parallel_tool_calls"] is True

    def test_force_tool_all_options(self):
        """Test force_tool with all options."""
        result = force_tool("search", once=True, parallel=True)

        controls = result["tool_controls"]
        assert controls["tool_choice"] == {"name": "search"}
        assert controls["force_once"] is True
        assert controls["parallel_tool_calls"] is True

    def test_force_tool_is_mergeable(self):
        """Test force_tool result can be used as kwargs."""
        result = force_tool("calculator")

        # Should be usable in function calls
        assert isinstance(result, dict)
        assert len(result) == 1


# =============================================================================
# Internal Helper Tests
# =============================================================================


class TestEnsureDict:
    """Tests for _ensure_dict internal helper."""

    def test_ensure_dict_with_none(self):
        """Test _ensure_dict returns None for None input."""
        result = _ensure_dict(None)
        assert result is None

    def test_ensure_dict_with_empty_dict(self):
        """Test _ensure_dict returns None for empty dict."""
        result = _ensure_dict({})
        assert result is None

    def test_ensure_dict_with_dict(self):
        """Test _ensure_dict returns dict as-is."""
        input_dict = {"tool_choice": "auto"}
        result = _ensure_dict(input_dict)

        assert result is input_dict

    def test_ensure_dict_with_dataclass(self):
        """Test _ensure_dict converts dataclass to dict."""
        controls = ToolCallControls(tool_choice="none")
        result = _ensure_dict(controls)

        assert isinstance(result, dict)
        assert result["tool_choice"] == "none"

    def test_ensure_dict_with_custom_dataclass(self):
        """Test _ensure_dict with custom dataclass."""

        @dataclass
        class CustomControls:
            option: str = "value"

        custom = CustomControls()
        result = _ensure_dict(custom)

        assert isinstance(result, dict)
        assert result["option"] == "value"

    def test_ensure_dict_with_dataclass_class(self):
        """Test _ensure_dict returns None for dataclass class (not instance)."""
        # Passing the class itself, not an instance
        result = _ensure_dict(ToolCallControls)
        assert result is None

    def test_ensure_dict_with_non_convertible(self):
        """Test _ensure_dict returns None for non-convertible types."""
        assert _ensure_dict("string") is None
        assert _ensure_dict(123) is None
        assert _ensure_dict([1, 2, 3]) is None


class TestExtractName:
    """Tests for _extract_name internal helper."""

    def test_extract_name_from_name_key(self):
        """Test _extract_name extracts from 'name' key."""
        result = _extract_name({"name": "calculator"})
        assert result == "calculator"

    def test_extract_name_from_function_key(self):
        """Test _extract_name extracts from nested 'function.name' key."""
        result = _extract_name({"function": {"name": "search"}})
        assert result == "search"

    def test_extract_name_prefers_name_over_function(self):
        """Test _extract_name prefers direct 'name' over nested."""
        result = _extract_name({"name": "direct", "function": {"name": "nested"}})
        assert result == "direct"

    def test_extract_name_with_non_dict(self):
        """Test _extract_name returns None for non-dict."""
        assert _extract_name("string") is None
        assert _extract_name(None) is None
        assert _extract_name(123) is None

    def test_extract_name_with_empty_dict(self):
        """Test _extract_name returns None for empty dict."""
        assert _extract_name({}) is None

    def test_extract_name_with_no_name(self):
        """Test _extract_name returns None when no name present."""
        assert _extract_name({"other": "value"}) is None


# =============================================================================
# Provider Normalization Tests - OpenAI
# =============================================================================


class TestNormalizeToolControlsOpenAI:
    """Tests for normalize_tool_controls with OpenAI provider."""

    def test_normalize_none_controls(self):
        """Test normalizing None controls."""
        tool_choice, parallel, force_once = normalize_tool_controls(Providers.openai, None)

        assert tool_choice is None
        assert parallel is True
        assert force_once is False

    def test_normalize_empty_controls(self):
        """Test normalizing empty dict controls."""
        tool_choice, parallel, force_once = normalize_tool_controls(Providers.openai, {})

        assert tool_choice is None
        assert parallel is True
        assert force_once is False

    def test_normalize_tool_choice_none_string(self):
        """Test normalizing tool_choice='none'."""
        tool_choice, parallel, force_once = normalize_tool_controls(
            Providers.openai, {"tool_choice": "none"}
        )

        assert tool_choice == "none"

    def test_normalize_tool_choice_auto_string(self):
        """Test normalizing tool_choice='auto'."""
        tool_choice, parallel, force_once = normalize_tool_controls(
            Providers.openai, {"tool_choice": "auto"}
        )

        assert tool_choice == "auto"

    def test_normalize_tool_choice_any_string(self):
        """Test normalizing tool_choice='any'."""
        tool_choice, parallel, force_once = normalize_tool_controls(
            Providers.openai, {"tool_choice": "any"}
        )

        assert tool_choice == "any"

    def test_normalize_tool_choice_with_name(self):
        """Test normalizing tool_choice with name dict."""
        tool_choice, parallel, force_once = normalize_tool_controls(
            Providers.openai, {"tool_choice": {"name": "calculator"}}
        )

        # OpenAI format: {"type": "function", "function": {"name": "..."}}
        assert tool_choice == {"type": "function", "function": {"name": "calculator"}}

    def test_normalize_tool_choice_with_function_format(self):
        """Test normalizing tool_choice already in function format."""
        tool_choice, parallel, force_once = normalize_tool_controls(
            Providers.openai, {"tool_choice": {"function": {"name": "search"}}}
        )

        assert tool_choice == {"type": "function", "function": {"name": "search"}}

    def test_normalize_parallel_tool_calls_true(self):
        """Test normalizing parallel_tool_calls=True."""
        tool_choice, parallel, force_once = normalize_tool_controls(
            Providers.openai, {"parallel_tool_calls": True}
        )

        assert parallel is True

    def test_normalize_parallel_tool_calls_false(self):
        """Test normalizing parallel_tool_calls=False."""
        tool_choice, parallel, force_once = normalize_tool_controls(
            Providers.openai, {"parallel_tool_calls": False}
        )

        assert parallel is False

    def test_normalize_force_once_true(self):
        """Test normalizing force_once=True."""
        tool_choice, parallel, force_once = normalize_tool_controls(
            Providers.openai, {"force_once": True}
        )

        assert force_once is True

    def test_normalize_force_once_false(self):
        """Test normalizing force_once=False."""
        tool_choice, parallel, force_once = normalize_tool_controls(
            Providers.openai, {"force_once": False}
        )

        assert force_once is False

    def test_normalize_all_options(self):
        """Test normalizing with all options set."""
        tool_choice, parallel, force_once = normalize_tool_controls(
            Providers.openai,
            {
                "tool_choice": {"name": "search"},
                "parallel_tool_calls": False,
                "force_once": True,
            },
        )

        assert tool_choice == {"type": "function", "function": {"name": "search"}}
        assert parallel is False
        assert force_once is True

    def test_normalize_with_dataclass(self):
        """Test normalizing ToolCallControls dataclass."""
        controls = ToolCallControls(
            tool_choice={"name": "calculator"},
            parallel_tool_calls=False,
            force_once=True,
        )

        tool_choice, parallel, force_once = normalize_tool_controls(Providers.openai, controls)

        assert tool_choice == {"type": "function", "function": {"name": "calculator"}}
        assert parallel is False
        assert force_once is True


# =============================================================================
# Provider Normalization Tests - XAI (Same as OpenAI)
# =============================================================================


class TestNormalizeToolControlsXAI:
    """Tests for normalize_tool_controls with XAI provider."""

    def test_xai_uses_openai_format(self):
        """Test XAI uses same format as OpenAI."""
        tool_choice, parallel, force_once = normalize_tool_controls(
            Providers.xai, {"tool_choice": {"name": "my_tool"}}
        )

        # XAI should use OpenAI format
        assert tool_choice == {"type": "function", "function": {"name": "my_tool"}}

    def test_xai_string_passthrough(self):
        """Test XAI passes through string tool_choice."""
        tool_choice, _, _ = normalize_tool_controls(Providers.xai, {"tool_choice": "auto"})

        assert tool_choice == "auto"


# =============================================================================
# Provider Normalization Tests - Anthropic
# =============================================================================


class TestNormalizeToolControlsAnthropic:
    """Tests for normalize_tool_controls with Anthropic provider."""

    def test_normalize_tool_choice_none_string(self):
        """Test normalizing tool_choice='none' for Anthropic."""
        tool_choice, _, _ = normalize_tool_controls(Providers.anthropic, {"tool_choice": "none"})

        assert tool_choice == "none"

    def test_normalize_tool_choice_auto_string(self):
        """Test normalizing tool_choice='auto' for Anthropic."""
        tool_choice, _, _ = normalize_tool_controls(Providers.anthropic, {"tool_choice": "auto"})

        assert tool_choice == "auto"

    def test_normalize_tool_choice_any_string(self):
        """Test normalizing tool_choice='any' for Anthropic."""
        tool_choice, _, _ = normalize_tool_controls(Providers.anthropic, {"tool_choice": "any"})

        assert tool_choice == "any"

    def test_normalize_tool_choice_with_name(self):
        """Test normalizing tool_choice with name for Anthropic."""
        tool_choice, _, _ = normalize_tool_controls(
            Providers.anthropic, {"tool_choice": {"name": "search"}}
        )

        # Anthropic format: {"type": "tool", "name": "..."}
        assert tool_choice == {"type": "tool", "name": "search"}

    def test_normalize_tool_choice_with_function_format(self):
        """Test normalizing nested function format for Anthropic."""
        tool_choice, _, _ = normalize_tool_controls(
            Providers.anthropic, {"tool_choice": {"function": {"name": "calc"}}}
        )

        assert tool_choice == {"type": "tool", "name": "calc"}

    def test_normalize_preserves_other_options(self):
        """Test Anthropic normalization preserves parallel and force_once."""
        tool_choice, parallel, force_once = normalize_tool_controls(
            Providers.anthropic,
            {
                "tool_choice": {"name": "tool"},
                "parallel_tool_calls": False,
                "force_once": True,
            },
        )

        assert parallel is False
        assert force_once is True


# =============================================================================
# Provider Normalization Tests - Google GenAI (Gemini)
# =============================================================================


class TestNormalizeToolControlsGoogleGenAI:
    """Tests for normalize_tool_controls with Google GenAI (Gemini) provider."""

    def test_normalize_tool_choice_none_string(self):
        """Test normalizing tool_choice='none' for Gemini."""
        tool_choice, _, _ = normalize_tool_controls(Providers.google_genai, {"tool_choice": "none"})

        # Gemini format: {"function_calling_config": {"mode": "NONE"}}
        assert tool_choice == {"function_calling_config": {"mode": "NONE"}}

    def test_normalize_tool_choice_auto_string(self):
        """Test normalizing tool_choice='auto' for Gemini."""
        tool_choice, _, _ = normalize_tool_controls(Providers.google_genai, {"tool_choice": "auto"})

        assert tool_choice == {"function_calling_config": {"mode": "AUTO"}}

    def test_normalize_tool_choice_any_string(self):
        """Test normalizing tool_choice='any' for Gemini."""
        tool_choice, _, _ = normalize_tool_controls(Providers.google_genai, {"tool_choice": "any"})

        assert tool_choice == {"function_calling_config": {"mode": "ANY"}}

    def test_normalize_tool_choice_unknown_string(self):
        """Test normalizing unknown string defaults to AUTO for Gemini."""
        tool_choice, _, _ = normalize_tool_controls(
            Providers.google_genai, {"tool_choice": "required"}
        )

        # Unknown strings default to AUTO
        assert tool_choice == {"function_calling_config": {"mode": "AUTO"}}

    def test_normalize_tool_choice_with_name(self):
        """Test normalizing tool_choice with name for Gemini."""
        tool_choice, _, _ = normalize_tool_controls(
            Providers.google_genai, {"tool_choice": {"name": "search_tool"}}
        )

        # Gemini with specific tool: {"function_calling_config": {"mode": "ANY", "allowed_function_names": [...]}}
        expected = {
            "function_calling_config": {"mode": "ANY", "allowed_function_names": ["search_tool"]}
        }
        assert tool_choice == expected

    def test_normalize_tool_choice_case_insensitive(self):
        """Test Gemini normalization is case insensitive for strings."""
        tool_choice, _, _ = normalize_tool_controls(Providers.google_genai, {"tool_choice": "NONE"})
        assert tool_choice == {"function_calling_config": {"mode": "NONE"}}

        tool_choice, _, _ = normalize_tool_controls(Providers.google_genai, {"tool_choice": "Auto"})
        assert tool_choice == {"function_calling_config": {"mode": "AUTO"}}

    def test_normalize_preserves_other_options(self):
        """Test Gemini normalization preserves parallel and force_once."""
        tool_choice, parallel, force_once = normalize_tool_controls(
            Providers.google_genai,
            {
                "tool_choice": "auto",
                "parallel_tool_calls": False,
                "force_once": True,
            },
        )

        assert parallel is False
        assert force_once is True


# =============================================================================
# Provider Normalization Tests - Other Providers
# =============================================================================


class TestNormalizeToolControlsOtherProviders:
    """Tests for normalize_tool_controls with other/unknown providers."""

    def test_unknown_provider_passthrough(self):
        """Test unknown provider passes through tool_choice."""
        tool_choice, parallel, force_once = normalize_tool_controls(
            "unknown_provider", {"tool_choice": {"custom": "format"}}
        )

        # Should pass through unchanged
        assert tool_choice == {"custom": "format"}
        assert parallel is True
        assert force_once is False

    def test_unknown_provider_string_passthrough(self):
        """Test unknown provider passes through string tool_choice."""
        tool_choice, _, _ = normalize_tool_controls(
            "some_provider", {"tool_choice": "custom_value"}
        )

        assert tool_choice == "custom_value"


# =============================================================================
# Integration Tests
# =============================================================================


class TestToolControlsIntegration:
    """Integration tests for tool controls."""

    def test_no_tools_normalizes_correctly(self):
        """Test no_tools() output normalizes correctly."""
        controls = no_tools()["tool_controls"]

        tool_choice, parallel, force_once = normalize_tool_controls(Providers.openai, controls)

        assert tool_choice == "none"

    def test_force_tool_normalizes_correctly_openai(self):
        """Test force_tool() output normalizes correctly for OpenAI."""
        controls = force_tool("calculator")["tool_controls"]

        tool_choice, parallel, force_once = normalize_tool_controls(Providers.openai, controls)

        assert tool_choice == {"type": "function", "function": {"name": "calculator"}}
        assert parallel is False  # force_tool defaults parallel=False

    def test_force_tool_normalizes_correctly_anthropic(self):
        """Test force_tool() output normalizes correctly for Anthropic."""
        controls = force_tool("search")["tool_controls"]

        tool_choice, parallel, force_once = normalize_tool_controls(Providers.anthropic, controls)

        assert tool_choice == {"type": "tool", "name": "search"}

    def test_force_tool_normalizes_correctly_gemini(self):
        """Test force_tool() output normalizes correctly for Gemini."""
        controls = force_tool("my_tool")["tool_controls"]

        tool_choice, parallel, force_once = normalize_tool_controls(
            Providers.google_genai, controls
        )

        expected = {
            "function_calling_config": {"mode": "ANY", "allowed_function_names": ["my_tool"]}
        }
        assert tool_choice == expected

    def test_dataclass_and_dict_produce_same_result(self):
        """Test that dataclass and dict produce the same normalization."""
        dict_controls = {
            "tool_choice": {"name": "tool"},
            "parallel_tool_calls": False,
            "force_once": True,
        }

        dataclass_controls = ToolCallControls(
            tool_choice={"name": "tool"},
            parallel_tool_calls=False,
            force_once=True,
        )

        dict_result = normalize_tool_controls(Providers.openai, dict_controls)
        dataclass_result = normalize_tool_controls(Providers.openai, dataclass_controls)

        assert dict_result == dataclass_result


# =============================================================================
# Edge Cases
# =============================================================================


class TestToolControlsEdgeCases:
    """Tests for edge cases in tool controls."""

    def test_empty_name_dict(self):
        """Test tool_choice with empty name dict passes through unchanged."""
        tool_choice, _, _ = normalize_tool_controls(Providers.openai, {"tool_choice": {"name": ""}})

        # Empty name is falsy, so _extract_name returns None
        # and the dict passes through unchanged
        assert tool_choice == {"name": ""}

    def test_none_in_tool_choice_dict(self):
        """Test tool_choice with None name."""
        tool_choice, _, _ = normalize_tool_controls(
            Providers.openai, {"tool_choice": {"name": None}}
        )

        # None name means no specific tool, pass through
        assert tool_choice == {"name": None}

    def test_extra_keys_in_controls(self):
        """Test controls with extra keys are handled."""
        tool_choice, parallel, force_once = normalize_tool_controls(
            Providers.openai,
            {
                "tool_choice": "auto",
                "unknown_key": "value",
                "another_key": 123,
            },
        )

        # Extra keys should be ignored
        assert tool_choice == "auto"
        assert parallel is True
        assert force_once is False

    def test_truthy_force_once_values(self):
        """Test force_once with various truthy values."""
        _, _, force_once = normalize_tool_controls(Providers.openai, {"force_once": 1})
        assert force_once is True

        _, _, force_once = normalize_tool_controls(Providers.openai, {"force_once": "yes"})
        assert force_once is True

    def test_falsy_force_once_values(self):
        """Test force_once with various falsy values."""
        _, _, force_once = normalize_tool_controls(Providers.openai, {"force_once": 0})
        assert force_once is False

        _, _, force_once = normalize_tool_controls(Providers.openai, {"force_once": ""})
        assert force_once is False

    def test_parallel_tool_calls_default_when_missing(self):
        """Test parallel_tool_calls defaults to True when not specified."""
        _, parallel, _ = normalize_tool_controls(Providers.openai, {"tool_choice": "auto"})

        assert parallel is True

    def test_deeply_nested_function_format(self):
        """Test handling of deeply nested function format."""
        tool_choice, _, _ = normalize_tool_controls(
            Providers.openai, {"tool_choice": {"function": {"name": "nested_tool"}}}
        )

        assert tool_choice == {"type": "function", "function": {"name": "nested_tool"}}

    def test_gemini_with_no_tool_choice(self):
        """Test Gemini with no tool_choice returns None."""
        tool_choice, _, _ = normalize_tool_controls(
            Providers.google_genai, {"parallel_tool_calls": False}
        )

        assert tool_choice is None
