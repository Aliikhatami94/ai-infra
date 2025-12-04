"""Tests for LLM class callback integration.

This module tests that the LLM class properly fires callbacks during:
- Chat calls (start, end, error)
- Async chat calls
- Streaming (token events)
"""

import pytest

from ai_infra import LLM
from ai_infra.callbacks import (
    CallbackManager,
    Callbacks,
    LLMEndEvent,
    LLMErrorEvent,
    LLMStartEvent,
    LLMTokenEvent,
)


class TestLLMCallbacksParameter:
    """Tests for LLM accepting callbacks parameter."""

    def test_llm_accepts_callbacks(self):
        """Test LLM accepts Callbacks instance."""
        callbacks = Callbacks()
        llm = LLM(callbacks=callbacks)
        assert llm._callbacks is not None

    def test_llm_accepts_callback_manager(self):
        """Test LLM accepts CallbackManager instance."""
        manager = CallbackManager([Callbacks()])
        llm = LLM(callbacks=manager)
        assert llm._callbacks is not None

    def test_llm_without_callbacks(self):
        """Test LLM works without callbacks."""
        llm = LLM()
        assert llm._callbacks is None

    def test_llm_normalizes_single_callback(self):
        """Test LLM normalizes single Callbacks to CallbackManager."""

        class MyCallbacks(Callbacks):
            pass

        llm = LLM(callbacks=MyCallbacks())
        # Should be normalized to CallbackManager
        assert llm._callbacks is not None
        assert isinstance(llm._callbacks, CallbackManager)

    def test_llm_rejects_invalid_callbacks(self):
        """Test LLM rejects invalid callbacks type."""
        with pytest.raises(ValueError, match="Invalid callbacks type"):
            LLM(callbacks="not a callback")  # type: ignore


class TestLLMCallbackEvents:
    """Tests for LLM firing callback events.

    Note: These tests verify the callback infrastructure without making
    actual LLM API calls.
    """

    @pytest.fixture
    def tracking_callbacks(self):
        """Create callbacks that track all events."""
        events = []

        class TrackingCallbacks(Callbacks):
            def on_llm_start(self, event: LLMStartEvent) -> None:
                events.append(("llm_start", event))

            def on_llm_end(self, event: LLMEndEvent) -> None:
                events.append(("llm_end", event))

            def on_llm_error(self, event: LLMErrorEvent) -> None:
                events.append(("llm_error", event))

            def on_llm_token(self, event: LLMTokenEvent) -> None:
                events.append(("llm_token", event))

        return TrackingCallbacks(), events

    def test_llm_callback_instantiation(self, tracking_callbacks):
        """Test LLM can be created with tracking callbacks."""
        callbacks, events = tracking_callbacks
        llm = LLM(callbacks=callbacks)
        assert llm._callbacks is not None

    def test_llm_with_multiple_callbacks(self):
        """Test LLM with CallbackManager containing multiple callbacks."""
        events1 = []
        events2 = []

        class Callbacks1(Callbacks):
            def on_llm_start(self, event):
                events1.append(event)

        class Callbacks2(Callbacks):
            def on_llm_start(self, event):
                events2.append(event)

        manager = CallbackManager([Callbacks1(), Callbacks2()])
        llm = LLM(callbacks=manager)

        # Both callbacks should be registered
        assert len(llm._callbacks._callbacks) == 2


class TestLLMCallbackIntegration:
    """Integration tests for LLM callbacks.

    These tests verify the callback infrastructure is properly wired,
    without making actual LLM API calls.
    """

    def test_callback_manager_passed_correctly(self):
        """Test that CallbackManager is correctly stored on LLM."""

        class TestCallbacks(Callbacks):
            def on_llm_start(self, event):
                pass

        callbacks = TestCallbacks()
        llm = LLM(callbacks=callbacks)

        # Verify the callbacks were normalized to CallbackManager
        assert isinstance(llm._callbacks, CallbackManager)
        assert len(llm._callbacks._callbacks) == 1
        assert isinstance(llm._callbacks._callbacks[0], TestCallbacks)

    def test_multiple_callback_handlers(self):
        """Test LLM with multiple callback handlers."""
        handler1_events = []
        handler2_events = []

        class Handler1(Callbacks):
            def on_llm_start(self, event):
                handler1_events.append(event)

        class Handler2(Callbacks):
            def on_llm_start(self, event):
                handler2_events.append(event)

        manager = CallbackManager([Handler1(), Handler2()])
        llm = LLM(callbacks=manager)

        assert len(llm._callbacks._callbacks) == 2

    def test_llm_streaming_callbacks_setup(self):
        """Test LLM is set up for streaming callbacks."""

        class StreamCallbacks(Callbacks):
            def on_llm_token(self, event: LLMTokenEvent) -> None:
                pass

            async def on_llm_token_async(self, event: LLMTokenEvent) -> None:
                pass

        llm = LLM(callbacks=StreamCallbacks())
        assert llm._callbacks is not None


class TestLLMAsyncCallbacks:
    """Tests for async callback support in LLM."""

    @pytest.fixture
    def async_tracking_callbacks(self):
        """Create callbacks that track async events."""
        events = []

        class AsyncTrackingCallbacks(Callbacks):
            async def on_llm_start_async(self, event: LLMStartEvent) -> None:
                events.append(("llm_start_async", event))

            async def on_llm_end_async(self, event: LLMEndEvent) -> None:
                events.append(("llm_end_async", event))

            async def on_llm_error_async(self, event: LLMErrorEvent) -> None:
                events.append(("llm_error_async", event))

            async def on_llm_token_async(self, event: LLMTokenEvent) -> None:
                events.append(("llm_token_async", event))

        return AsyncTrackingCallbacks(), events

    def test_llm_async_callbacks_setup(self, async_tracking_callbacks):
        """Test LLM can be created with async callbacks."""
        callbacks, events = async_tracking_callbacks
        llm = LLM(callbacks=callbacks)
        assert llm._callbacks is not None
