"""
RealtimeVoice facade for automatic provider selection and unified voice API.

This module provides a high-level interface for real-time voice conversations
with automatic provider selection based on environment configuration.

Example:
    >>> from ai_infra.llm.realtime import RealtimeVoice, RealtimeConfig
    >>>
    >>> voice = RealtimeVoice()  # Auto-selects provider based on env vars
    >>>
    >>> @voice.on_transcript
    >>> async def handle_transcript(text: str, is_final: bool):
    ...     print(f"{'Final' if is_final else 'Interim'}: {text}")
    >>>
    >>> @voice.on_audio
    >>> async def handle_audio(audio: bytes):
    ...     play_audio(audio)
    >>>
    >>> async with voice.connect() as session:
    ...     await session.send_audio(microphone_stream())
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from .base import BaseRealtimeProvider
from .models import (
    AudioChunk,
    RealtimeConfig,
    RealtimeError,
    ToolCallRequest,
    ToolDefinition,
    TranscriptDelta,
)

# Import provider registry for configuration checks
try:
    from ai_infra.providers import ProviderCapability, ProviderRegistry

    _HAS_PROVIDER_REGISTRY = True
except ImportError:
    _HAS_PROVIDER_REGISTRY = False


logger = logging.getLogger(__name__)

# Type aliases for callbacks
AudioCallback = Callable[[bytes], Awaitable[None]]
TranscriptCallback = Callable[[str, bool], Awaitable[None]]
ToolCallCallback = Callable[[ToolCallRequest], Awaitable[Any]]
ErrorCallback = Callable[[RealtimeError], Awaitable[None]]
InterruptedCallback = Callable[[], Awaitable[None]]


def _convert_tools_to_definitions(tools: list[Any]) -> list[ToolDefinition]:
    """Convert tool functions to ToolDefinition objects.

    Reuses ai-infra's existing tool conversion infrastructure.
    """
    if not tools:
        return []

    try:
        from ai_infra.llm.tools import tools_from_functions

        # Convert functions to LangChain tools
        lc_tools = tools_from_functions(tools)

        definitions = []
        for tool in lc_tools:
            # Extract JSON schema from the tool
            schema: dict[str, Any] = {}
            args_schema = getattr(tool, "args_schema", None)
            if args_schema is not None:
                from pydantic import BaseModel

                if isinstance(args_schema, type) and issubclass(args_schema, BaseModel):
                    schema = args_schema.model_json_schema()

            definitions.append(
                ToolDefinition(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=schema,
                )
            )

        return definitions
    except ImportError:
        # Fallback: Extract info from function signatures
        import inspect

        definitions = []
        for func in tools:
            if callable(func):
                sig = inspect.signature(func)
                params: dict[str, Any] = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }
                for name, param in sig.parameters.items():
                    param_type = "string"
                    if param.annotation != inspect.Parameter.empty:
                        if param.annotation is int:
                            param_type = "integer"
                        elif param.annotation is float:
                            param_type = "number"
                        elif param.annotation is bool:
                            param_type = "boolean"
                    params["properties"][name] = {"type": param_type}
                    if param.default == inspect.Parameter.empty:
                        params["required"].append(name)

                definitions.append(
                    ToolDefinition(
                        name=func.__name__,
                        description=func.__doc__ or "",
                        parameters=params,
                    )
                )

        return definitions


class RealtimeVoice:
    """
    High-level facade for real-time voice conversations.

    Automatically selects the best available provider based on environment
    configuration, or allows explicit provider selection.

    Provider Selection Order:
        1. Explicit provider passed to constructor
        2. REALTIME_VOICE_PROVIDER environment variable
        3. First configured provider (OpenAI → Gemini)

    Attributes:
        config: The realtime configuration for voice sessions.
        provider: The underlying realtime provider instance.

    Example:
        >>> # Auto-select provider
        >>> voice = RealtimeVoice()
        >>>
        >>> # Explicit provider
        >>> voice = RealtimeVoice(provider="openai")
        >>>
        >>> # Custom config
        >>> config = RealtimeConfig(model="gpt-4o-realtime-preview")
        >>> voice = RealtimeVoice(config=config)
    """

    # Registry of available providers (populated by provider modules)
    _providers: dict[str, type[BaseRealtimeProvider]] = {}

    def __init__(
        self,
        provider: str | BaseRealtimeProvider | None = None,
        config: RealtimeConfig | None = None,
    ) -> None:
        """
        Initialize RealtimeVoice with optional provider and config.

        Args:
            provider: Provider name ("openai", "gemini") or provider instance.
                     If None, auto-selects based on environment.
            config: Configuration for the realtime session.
                   If None, uses default configuration.

        Raises:
            RealtimeError: If no provider is configured or available.
        """
        self.config = config or RealtimeConfig()
        self._provider: BaseRealtimeProvider | None = None
        self._audio_callbacks: list[AudioCallback] = []
        self._transcript_callbacks: list[TranscriptCallback] = []
        self._tool_call_callbacks: list[ToolCallCallback] = []
        self._error_callbacks: list[ErrorCallback] = []
        self._interrupted_callbacks: list[InterruptedCallback] = []
        self._tool_functions: dict[str, Callable] = {}

        # Convert tool functions to ToolDefinition objects
        if self.config.tools:
            tool_defs = _convert_tools_to_definitions(self.config.tools)
            # Store original functions for execution
            for func in self.config.tools:
                if callable(func):
                    self._tool_functions[func.__name__] = func
            # Replace raw functions with definitions in config
            self.config = self.config.model_copy(update={"tools": tool_defs})

        # Resolve provider
        if isinstance(provider, BaseRealtimeProvider):
            self._provider = provider
        elif isinstance(provider, str):
            self._provider = self._get_provider_by_name(provider)
        else:
            self._provider = self._auto_select_provider()

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseRealtimeProvider]) -> None:
        """
        Register a provider class for auto-selection.

        This is called by provider modules when they are imported.

        Args:
            name: Provider name (e.g., "openai", "gemini").
            provider_class: The provider class to register.
        """
        cls._providers[name.lower()] = provider_class
        logger.debug(f"Registered realtime provider: {name}")

    @classmethod
    def available_providers(cls) -> list[str]:
        """
        List all registered provider names.

        Returns:
            List of provider names that have been registered.
        """
        return list(cls._providers.keys())

    @classmethod
    def configured_providers(cls) -> list[str]:
        """
        List providers that are configured (have API keys).

        Returns:
            List of provider names that have valid configuration.
        """
        configured = []

        # Use central registry if available
        if _HAS_PROVIDER_REGISTRY:
            realtime_providers = ProviderRegistry.list_for_capability(ProviderCapability.REALTIME)
            for name in realtime_providers:
                if ProviderRegistry.is_configured(name):
                    # Map registry names to local provider names
                    local_name = "gemini" if name == "google_genai" else name
                    if local_name in cls._providers:
                        configured.append(local_name)
            return configured

        # Fallback to checking each provider class
        for name, provider_class in cls._providers.items():
            try:
                if provider_class.is_configured():
                    configured.append(name)
            except Exception:
                pass
        return configured

    def _get_provider_by_name(self, name: str) -> BaseRealtimeProvider:
        """Get a provider instance by name."""
        name_lower = name.lower()
        if name_lower not in self._providers:
            available = ", ".join(self._providers.keys()) or "none"
            raise RealtimeError(f"Unknown provider '{name}'. Available: {available}")

        provider_class = self._providers[name_lower]
        return provider_class(config=self.config)

    def _auto_select_provider(self) -> BaseRealtimeProvider:
        """Auto-select the best available provider."""
        # Check environment variable first
        env_provider = os.environ.get("REALTIME_VOICE_PROVIDER", "").lower()
        if env_provider and env_provider in self._providers:
            provider_class = self._providers[env_provider]
            if provider_class.is_configured():
                logger.info(f"Using provider from env: {env_provider}")
                return provider_class(config=self.config)

        # Try providers in priority order
        priority_order = ["openai", "gemini"]

        for name in priority_order:
            if name in self._providers:
                provider_class = self._providers[name]
                try:
                    if provider_class.is_configured():
                        logger.info(f"Auto-selected provider: {name}")
                        return provider_class(config=self.config)
                except Exception as e:
                    logger.debug(f"Provider {name} not configured: {e}")

        # Try any remaining providers
        for name, provider_class in self._providers.items():
            if name not in priority_order:
                try:
                    if provider_class.is_configured():
                        logger.info(f"Auto-selected provider: {name}")
                        return provider_class(config=self.config)
                except Exception:
                    pass

        raise RealtimeError(
            "No realtime voice provider configured. "
            "Set OPENAI_API_KEY or GOOGLE_API_KEY environment variable."
        )

    @property
    def provider(self) -> BaseRealtimeProvider:
        """Get the underlying provider instance."""
        if self._provider is None:
            raise RealtimeError("No provider configured")
        return self._provider

    @property
    def provider_name(self) -> str:
        """Get the name of the current provider."""
        return self.provider.provider_name

    # Callback decorators
    def on_audio(self, callback: AudioCallback) -> AudioCallback:
        """
        Register a callback for audio output.

        Args:
            callback: Async function receiving audio bytes.

        Returns:
            The callback (for use as decorator).

        Example:
            >>> @voice.on_audio
            >>> async def handle_audio(audio: bytes):
            ...     play_audio(audio)
        """
        self._audio_callbacks.append(callback)
        return callback

    def on_transcript(self, callback: TranscriptCallback) -> TranscriptCallback:
        """
        Register a callback for transcript updates.

        Args:
            callback: Async function receiving (text, is_final).

        Returns:
            The callback (for use as decorator).

        Example:
            >>> @voice.on_transcript
            >>> async def handle_transcript(text: str, is_final: bool):
            ...     print(text)
        """
        self._transcript_callbacks.append(callback)
        return callback

    def on_tool_call(self, callback: ToolCallCallback) -> ToolCallCallback:
        """
        Register a callback for tool/function calls.

        Args:
            callback: Async function receiving ToolCallRequest, returning result.

        Returns:
            The callback (for use as decorator).

        Example:
            >>> @voice.on_tool_call
            >>> async def handle_tool(request: ToolCallRequest) -> Any:
            ...     return execute_tool(request.name, request.arguments)
        """
        self._tool_call_callbacks.append(callback)
        return callback

    def on_error(self, callback: ErrorCallback) -> ErrorCallback:
        """
        Register a callback for errors.

        Args:
            callback: Async function receiving RealtimeError.

        Returns:
            The callback (for use as decorator).
        """
        self._error_callbacks.append(callback)
        return callback

    def on_interrupted(self, callback: InterruptedCallback) -> InterruptedCallback:
        """
        Register a callback for interruption events.

        Args:
            callback: Async function called when user interrupts.

        Returns:
            The callback (for use as decorator).
        """
        self._interrupted_callbacks.append(callback)
        return callback

    async def _dispatch_audio(self, audio: bytes) -> None:
        """Dispatch audio to all registered callbacks."""
        for callback in self._audio_callbacks:
            try:
                await callback(audio)
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")

    async def _dispatch_transcript(self, text: str, is_final: bool) -> None:
        """Dispatch transcript to all registered callbacks."""
        for callback in self._transcript_callbacks:
            try:
                await callback(text, is_final)
            except Exception as e:
                logger.error(f"Error in transcript callback: {e}")

    async def _dispatch_tool_call(self, request: ToolCallRequest) -> Any:
        """Dispatch tool call - execute tool function or delegate to callback."""
        # First, try to execute the tool function directly if we have it
        tool_func = self._tool_functions.get(request.name)
        if tool_func:
            try:
                import asyncio

                if asyncio.iscoroutinefunction(tool_func):
                    result = await tool_func(**request.arguments)
                else:
                    result = tool_func(**request.arguments)
                return result
            except Exception as e:
                logger.error(f"Error executing tool {request.name}: {e}")
                raise

        # Fall back to registered callbacks
        for callback in self._tool_call_callbacks:
            try:
                return await callback(request)
            except Exception as e:
                logger.error(f"Error in tool call callback: {e}")
                raise

        logger.warning(f"No handler found for tool: {request.name}")
        return None

    async def _dispatch_error(self, error: RealtimeError) -> None:
        """Dispatch error to all registered callbacks."""
        for callback in self._error_callbacks:
            try:
                await callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    async def _dispatch_interrupted(self) -> None:
        """Dispatch interruption to all registered callbacks."""
        for callback in self._interrupted_callbacks:
            try:
                await callback()
            except Exception as e:
                logger.error(f"Error in interrupted callback: {e}")

    @asynccontextmanager
    async def connect(self):
        """
        Connect to the realtime voice service.

        Returns an async context manager that yields a VoiceSession.

        Example:
            >>> async with voice.connect() as session:
            ...     await session.send_audio(audio_data)
        """
        # Set up provider callbacks
        self.provider.on_audio(self._dispatch_audio)
        self.provider.on_transcript(self._dispatch_transcript)
        self.provider.on_tool_call(self._dispatch_tool_call)
        self.provider.on_error(self._dispatch_error)
        self.provider.on_interrupted(self._dispatch_interrupted)

        async with self.provider.session() as session:
            yield session

    async def run(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[AudioChunk | TranscriptDelta]:
        """
        Run a voice conversation with streaming input/output.

        This is a convenience method that handles the connection
        lifecycle and streams results.

        Args:
            audio_stream: Async iterator of audio bytes from microphone.

        Yields:
            AudioChunk or TranscriptDelta events from the conversation.

        Example:
            >>> async for event in voice.run(microphone_stream()):
            ...     if isinstance(event, AudioChunk):
            ...         play_audio(event.data)
            ...     elif isinstance(event, TranscriptDelta):
            ...         print(event.text)
        """
        # Set up provider callbacks
        self.provider.on_audio(self._dispatch_audio)
        self.provider.on_transcript(self._dispatch_transcript)
        self.provider.on_tool_call(self._dispatch_tool_call)
        self.provider.on_error(self._dispatch_error)
        self.provider.on_interrupted(self._dispatch_interrupted)

        async for event in self.provider.run(audio_stream):
            yield event


def realtime_voice(
    provider: str | None = None,
    config: RealtimeConfig | None = None,
) -> RealtimeVoice:
    """
    Create a RealtimeVoice instance.

    This is a convenience function for creating RealtimeVoice instances.

    Args:
        provider: Optional provider name ("openai", "gemini").
        config: Optional configuration.

    Returns:
        Configured RealtimeVoice instance.

    Example:
        >>> voice = realtime_voice()  # Auto-select
        >>> voice = realtime_voice("openai")  # Explicit
    """
    return RealtimeVoice(provider=provider, config=config)
