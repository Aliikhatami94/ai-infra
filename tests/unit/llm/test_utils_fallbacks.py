"""Tests for LLM fallback utilities (llm/utils/fallbacks.py).

This module provides comprehensive tests for fallback execution:
- run_with_fallbacks()
- arun_with_fallbacks()
- FallbackError
- Candidate resolution
- Override merging

Phase 0.2 of the ai-infra v1.0.0 release plan.
"""

from __future__ import annotations

import pytest

from ai_infra.llm.utils.fallbacks import (
    FallbackError,
    _resolve_candidate,
    arun_with_fallbacks,
    merge_overrides,
    run_with_fallbacks,
)

# =============================================================================
# TEST: _resolve_candidate()
# =============================================================================


class TestResolveCandidate:
    """Test _resolve_candidate function."""

    def test_tuple_candidate(self):
        """Test tuple candidate resolution."""
        candidate = ("openai", "gpt-4o")
        provider, model, overrides = _resolve_candidate(candidate)

        assert provider == "openai"
        assert model == "gpt-4o"
        assert overrides == {}

    def test_dict_candidate_with_model_name(self):
        """Test dict candidate with model_name key."""
        candidate = {
            "provider": "anthropic",
            "model_name": "claude-3-5-sonnet-latest",
            "temperature": 0.5,
        }
        provider, model, overrides = _resolve_candidate(candidate)

        assert provider == "anthropic"
        assert model == "claude-3-5-sonnet-latest"
        assert overrides == {"temperature": 0.5}

    def test_dict_candidate_with_model_key(self):
        """Test dict candidate with 'model' key instead of 'model_name'."""
        candidate = {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "max_tokens": 100,
        }
        provider, model, overrides = _resolve_candidate(candidate)

        assert provider == "openai"
        assert model == "gpt-4o-mini"
        assert overrides == {"max_tokens": 100}

    def test_dict_candidate_missing_provider_raises(self):
        """Test dict candidate without provider raises ValueError."""
        candidate = {"model_name": "gpt-4o"}

        with pytest.raises(ValueError, match="provider"):
            _resolve_candidate(candidate)

    def test_dict_candidate_missing_model_raises(self):
        """Test dict candidate without model raises ValueError."""
        candidate = {"provider": "openai"}

        with pytest.raises(ValueError, match="model_name"):
            _resolve_candidate(candidate)


# =============================================================================
# TEST: merge_overrides()
# =============================================================================


class TestMergeOverrides:
    """Test merge_overrides function."""

    def test_basic_merge(self):
        """Test basic override merging."""
        base_extra = {"key1": "value1"}
        base_model_kwargs = {"temperature": 0.5}
        base_tools = [lambda x: x]
        base_tool_controls = {"tool_choice": "auto"}
        overrides = {
            "extra": {"key2": "value2"},
            "model_kwargs": {"max_tokens": 100},
        }

        extra, model_kwargs, tools, tool_controls = merge_overrides(
            base_extra, base_model_kwargs, base_tools, base_tool_controls, overrides
        )

        assert extra == {"key1": "value1", "key2": "value2"}
        assert model_kwargs == {"temperature": 0.5, "max_tokens": 100}
        assert tools is base_tools
        assert tool_controls is base_tool_controls

    def test_override_tools(self):
        """Test tools can be overridden."""
        new_tools = [lambda y: y * 2]
        overrides = {"tools": new_tools}

        _, _, tools, _ = merge_overrides(None, None, [lambda x: x], None, overrides)

        assert tools is new_tools

    def test_override_tool_controls(self):
        """Test tool_controls can be overridden."""
        new_controls = {"tool_choice": "required"}
        overrides = {"tool_controls": new_controls}

        _, _, _, tool_controls = merge_overrides(
            None, None, None, {"tool_choice": "auto"}, overrides
        )

        assert tool_controls == new_controls

    def test_none_base_values(self):
        """Test handling of None base values."""
        overrides = {"extra": {"key": "value"}}

        extra, model_kwargs, tools, tool_controls = merge_overrides(
            None, None, None, None, overrides
        )

        assert extra == {"key": "value"}
        assert model_kwargs == {}
        assert tools is None
        assert tool_controls is None


# =============================================================================
# TEST: run_with_fallbacks()
# =============================================================================


class TestRunWithFallbacks:
    """Test run_with_fallbacks function."""

    def test_primary_success_no_fallback(self):
        """Test primary candidate succeeds, no fallback needed."""
        call_log = []

        def run_single(provider, model, overrides):
            call_log.append((provider, model))
            return f"result from {provider}"

        candidates = [("openai", "gpt-4o"), ("anthropic", "claude-3-5-sonnet-latest")]
        result = run_with_fallbacks(candidates, run_single)

        assert result == "result from openai"
        assert len(call_log) == 1
        assert call_log[0] == ("openai", "gpt-4o")

    def test_primary_fails_fallback_succeeds(self):
        """Test primary fails, fallback succeeds."""
        call_log = []

        def run_single(provider, model, overrides):
            call_log.append((provider, model))
            if provider == "openai":
                raise RuntimeError("OpenAI is down")
            return f"result from {provider}"

        candidates = [("openai", "gpt-4o"), ("anthropic", "claude-3-5-sonnet-latest")]
        result = run_with_fallbacks(candidates, run_single)

        assert result == "result from anthropic"
        assert len(call_log) == 2

    def test_all_candidates_fail_raises_fallback_error(self):
        """Test all candidates fail raises FallbackError."""

        def run_single(provider, model, overrides):
            raise RuntimeError(f"{provider} failed")

        candidates = [("openai", "gpt-4o"), ("anthropic", "claude-3-5-sonnet-latest")]

        with pytest.raises(FallbackError) as exc_info:
            run_with_fallbacks(candidates, run_single)

        assert "All fallback candidates failed" in str(exc_info.value)
        assert len(exc_info.value.errors) == 2

    def test_custom_validate_function(self):
        """Test custom validate function."""
        call_log = []

        def run_single(provider, model, overrides):
            call_log.append(provider)
            if provider == "openai":
                return "bad"  # Invalid result
            return "good"  # Valid result

        def validate(result):
            return result == "good"

        candidates = [("openai", "gpt-4o"), ("anthropic", "claude-3-5-sonnet-latest")]
        result = run_with_fallbacks(candidates, run_single, validate=validate)

        assert result == "good"
        assert len(call_log) == 2

    def test_custom_should_retry_function(self):
        """Test custom should_retry function."""

        def run_single(provider, model, overrides):
            if provider == "openai":
                raise ValueError("Invalid")
            return "result"

        def should_retry(exc, result, idx, provider, model):
            # Don't retry ValueError
            if isinstance(exc, ValueError):
                return False
            return True

        candidates = [("openai", "gpt-4o"), ("anthropic", "claude-3-5-sonnet-latest")]

        # Should re-raise ValueError without trying fallback
        with pytest.raises(ValueError, match="Invalid"):
            run_with_fallbacks(candidates, run_single, should_retry=should_retry)

    def test_on_attempt_callback(self):
        """Test on_attempt callback is called."""
        attempts = []

        def run_single(provider, model, overrides):
            return "result"

        def on_attempt(idx, provider, model):
            attempts.append((idx, provider, model))

        candidates = [("openai", "gpt-4o")]
        run_with_fallbacks(candidates, run_single, on_attempt=on_attempt)

        assert attempts == [(0, "openai", "gpt-4o")]

    def test_dict_candidates_with_overrides(self):
        """Test dict candidates pass overrides to run_single."""
        received_overrides = []

        def run_single(provider, model, overrides):
            received_overrides.append(overrides)
            return "result"

        candidates = [{"provider": "openai", "model_name": "gpt-4o", "temperature": 0.7}]
        run_with_fallbacks(candidates, run_single)

        assert received_overrides[0] == {"temperature": 0.7}

    def test_none_result_triggers_retry(self):
        """Test None result triggers retry by default."""
        call_log = []

        def run_single(provider, model, overrides):
            call_log.append(provider)
            if provider == "openai":
                return None
            return "valid result"

        candidates = [("openai", "gpt-4o"), ("anthropic", "claude-3-5-sonnet-latest")]
        result = run_with_fallbacks(candidates, run_single)

        assert result == "valid result"
        assert len(call_log) == 2


# =============================================================================
# TEST: arun_with_fallbacks()
# =============================================================================


class TestArunWithFallbacks:
    """Test arun_with_fallbacks async function."""

    @pytest.mark.asyncio
    async def test_async_primary_success(self):
        """Test async primary candidate succeeds."""
        call_log = []

        async def run_single_async(provider, model, overrides):
            call_log.append(provider)
            return f"async result from {provider}"

        candidates = [("openai", "gpt-4o"), ("anthropic", "claude-3-5-sonnet-latest")]
        result = await arun_with_fallbacks(candidates, run_single_async)

        assert result == "async result from openai"
        assert len(call_log) == 1

    @pytest.mark.asyncio
    async def test_async_primary_fails_fallback_succeeds(self):
        """Test async primary fails, fallback succeeds."""
        call_log = []

        async def run_single_async(provider, model, overrides):
            call_log.append(provider)
            if provider == "openai":
                raise RuntimeError("OpenAI async error")
            return f"async result from {provider}"

        candidates = [("openai", "gpt-4o"), ("anthropic", "claude-3-5-sonnet-latest")]
        result = await arun_with_fallbacks(candidates, run_single_async)

        assert result == "async result from anthropic"
        assert len(call_log) == 2

    @pytest.mark.asyncio
    async def test_async_all_fail_raises(self):
        """Test async all candidates fail raises FallbackError."""

        async def run_single_async(provider, model, overrides):
            raise RuntimeError(f"{provider} async failed")

        candidates = [("openai", "gpt-4o"), ("anthropic", "claude-3-5-sonnet-latest")]

        with pytest.raises(FallbackError) as exc_info:
            await arun_with_fallbacks(candidates, run_single_async)

        assert "async fallback candidates failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_custom_validate(self):
        """Test async with custom validate function."""
        call_log = []

        async def run_single_async(provider, model, overrides):
            call_log.append(provider)
            if provider == "openai":
                return {"status": "error"}
            return {"status": "success"}

        def validate(result):
            return result.get("status") == "success"

        candidates = [("openai", "gpt-4o"), ("anthropic", "claude-3-5-sonnet-latest")]
        result = await arun_with_fallbacks(candidates, run_single_async, validate=validate)

        assert result["status"] == "success"
        assert len(call_log) == 2

    @pytest.mark.asyncio
    async def test_async_on_attempt_callback(self):
        """Test async on_attempt callback."""
        attempts = []

        async def run_single_async(provider, model, overrides):
            if provider == "openai":
                raise RuntimeError("fail")
            return "result"

        def on_attempt(idx, provider, model):
            attempts.append((idx, provider, model))

        candidates = [("openai", "gpt-4o"), ("anthropic", "claude-3-5-sonnet-latest")]

        await arun_with_fallbacks(candidates, run_single_async, on_attempt=on_attempt)

        assert len(attempts) == 2
        assert attempts[0] == (0, "openai", "gpt-4o")
        assert attempts[1] == (1, "anthropic", "claude-3-5-sonnet-latest")


# =============================================================================
# TEST: FallbackError
# =============================================================================


class TestFallbackError:
    """Test FallbackError class."""

    def test_error_message(self):
        """Test FallbackError message."""
        errors = [RuntimeError("Error 1"), ValueError("Error 2")]
        error = FallbackError("All failed", errors)

        assert str(error) == "All failed"
        assert error.errors == errors

    def test_error_is_runtime_error(self):
        """Test FallbackError is RuntimeError subclass."""
        error = FallbackError("Test", [])

        assert isinstance(error, RuntimeError)

    def test_error_with_empty_errors_list(self):
        """Test FallbackError with empty errors list."""
        error = FallbackError("No errors recorded", [])

        assert error.errors == []


# =============================================================================
# EDGE CASES
# =============================================================================


class TestFallbackEdgeCases:
    """Test edge cases for fallback utilities."""

    def test_single_candidate(self):
        """Test with single candidate."""

        def run_single(provider, model, overrides):
            return "only result"

        candidates = [("openai", "gpt-4o")]
        result = run_with_fallbacks(candidates, run_single)

        assert result == "only result"

    def test_empty_candidates_raises(self):
        """Test empty candidates raises error."""

        def run_single(provider, model, overrides):
            return "result"

        candidates: list = []

        with pytest.raises(RuntimeError, match="invalid results"):
            run_with_fallbacks(candidates, run_single)

    def test_mixed_candidate_types(self):
        """Test mixed tuple and dict candidates."""
        call_log = []

        def run_single(provider, model, overrides):
            call_log.append((provider, model, overrides))
            if provider == "openai":
                raise RuntimeError("fail")
            return "result"

        candidates = [
            ("openai", "gpt-4o"),
            {"provider": "anthropic", "model_name": "claude-3-5-sonnet-latest", "temp": 0.5},
        ]
        result = run_with_fallbacks(candidates, run_single)

        assert result == "result"
        assert call_log[0] == ("openai", "gpt-4o", {})
        assert call_log[1] == ("anthropic", "claude-3-5-sonnet-latest", {"temp": 0.5})

    @pytest.mark.asyncio
    async def test_async_single_candidate(self):
        """Test async with single candidate."""

        async def run_single_async(provider, model, overrides):
            return "async only"

        candidates = [("openai", "gpt-4o")]
        result = await arun_with_fallbacks(candidates, run_single_async)

        assert result == "async only"
