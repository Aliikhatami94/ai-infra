"""LLM class for direct model interaction.

This module provides the LLM class for simple chat-based interactions
without agent/tool capabilities.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel

from ai_infra.llm.base import BaseLLM
from ai_infra.llm.tools import apply_output_gate, apply_output_gate_async
from ai_infra.llm.utils.error_handler import translate_provider_error
from ai_infra.llm.utils.logging_hooks import ErrorContext, RequestContext, ResponseContext
from ai_infra.llm.utils.structured import coerce_structured_result, is_pydantic_schema

from .utils import make_messages as _make_messages
from .utils import sanitize_model_kwargs
from .utils import with_retry as _with_retry_util


class LLM(BaseLLM):
    """Direct model convenience interface (no agent graph).

    The LLM class provides a simple API for chat-based interactions
    with language models. Use this when you don't need tool calling.

    Example - Basic usage:
        ```python
        llm = LLM()
        response = llm.chat("What is the capital of France?")
        print(response.content)  # "Paris is the capital of France."
        ```

    Example - With structured output:
        ```python
        from pydantic import BaseModel

        class Answer(BaseModel):
            city: str
            country: str

        llm = LLM()
        result = llm.chat(
            "What is the capital of France?",
            output_schema=Answer,
        )
        print(result.city)  # "Paris"
        ```

    Example - Streaming tokens:
        ```python
        llm = LLM()
        async for token, meta in llm.stream_tokens("Tell me a story"):
            print(token, end="", flush=True)
        ```
    """

    # =========================================================================
    # Discovery API - Static methods for provider/model discovery
    # =========================================================================

    @staticmethod
    def list_providers() -> List[str]:
        """
        List all supported provider names.

        Returns:
            List of provider names: ["openai", "anthropic", "google_genai", "xai"]

        Example:
            >>> LLM.list_providers()
            ['openai', 'anthropic', 'google_genai', 'xai']
        """
        from ai_infra.llm.providers.discovery import list_providers

        return list_providers()

    @staticmethod
    def list_configured_providers() -> List[str]:
        """
        List providers that have API keys configured.

        Returns:
            List of provider names with configured API keys.

        Example:
            >>> LLM.list_configured_providers()
            ['openai', 'anthropic']  # Only if these have API keys set
        """
        from ai_infra.llm.providers.discovery import list_configured_providers

        return list_configured_providers()

    @staticmethod
    def list_models(provider: str, *, refresh: bool = False) -> List[str]:
        """
        List available models for a specific provider.

        Fetches models dynamically from the provider's API.
        Results are cached for 1 hour by default.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            refresh: Force refresh from API, bypassing cache

        Returns:
            List of model IDs available from the provider.

        Raises:
            ValueError: If provider is not supported.
            RuntimeError: If provider is not configured (no API key).

        Example:
            >>> LLM.list_models("openai")
            ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', ...]
        """
        from ai_infra.llm.providers.discovery import list_models

        return list_models(provider, refresh=refresh)

    @staticmethod
    def list_all_models(*, refresh: bool = False) -> Dict[str, List[str]]:
        """
        List models for all configured providers.

        Args:
            refresh: Force refresh from API, bypassing cache

        Returns:
            Dict mapping provider name to list of model IDs.

        Example:
            >>> LLM.list_all_models()
            {
                'openai': ['gpt-4o', 'gpt-4o-mini', ...],
                'anthropic': ['claude-sonnet-4-20250514', ...],
            }
        """
        from ai_infra.llm.providers.discovery import list_all_models

        return list_all_models(refresh=refresh)

    @staticmethod
    def is_provider_configured(provider: str) -> bool:
        """
        Check if a provider has its API key configured.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")

        Returns:
            True if the provider's API key environment variable is set.

        Example:
            >>> LLM.is_provider_configured("openai")
            True
        """
        from ai_infra.llm.providers.discovery import is_provider_configured

        return is_provider_configured(provider)

    # =========================================================================
    # Chat methods
    # =========================================================================

    def chat(
        self,
        user_msg: str,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        system: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        output_schema: Union[type[BaseModel], Dict[str, Any], None] = None,
        output_method: (
            Literal["json_schema", "json_mode", "function_calling", "prompt"] | None
        ) = "prompt",
        **model_kwargs,
    ):
        """Send a chat message and get a response.

        Args:
            user_msg: The user's message
            provider: LLM provider (auto-detected if None)
            model_name: Model name (uses provider default if None)
            system: Optional system message
            extra: Extra options (e.g., {"retry": {"max_attempts": 3}})
            output_schema: Pydantic model for structured output
            output_method: How to extract structured output
            **model_kwargs: Additional model kwargs

        Returns:
            Response message or structured output if output_schema provided
        """
        sanitize_model_kwargs(model_kwargs)

        # Resolve provider and model (auto-detect if not specified)
        provider, model_name = self._resolve_provider_and_model(provider, model_name)

        # Create request context for logging hooks
        request_ctx = RequestContext(
            user_msg=user_msg,
            system=system,
            provider=provider,
            model_name=model_name,
            model_kwargs=model_kwargs,
        )
        self._logging_hooks.call_request_sync(request_ctx)
        start_time = time.time()

        try:
            # PROMPT method uses shared helper
            if output_schema is not None and output_method == "prompt":
                res = self._prompt_structured_sync(
                    user_msg=user_msg,
                    system=system,
                    provider=provider,
                    model_name=model_name,
                    schema=output_schema,
                    extra=extra,
                    model_kwargs=model_kwargs,
                )
            else:
                # otherwise: existing structured (json_mode/function_calling/json_schema) or plain
                if output_schema is not None:
                    model = self.with_structured_output(
                        provider, model_name, output_schema, method=output_method, **model_kwargs
                    )
                else:
                    model = self.set_model(provider, model_name, **model_kwargs)

                messages = _make_messages(user_msg, system)

                def _call():
                    return model.invoke(messages)

                retry_cfg = (extra or {}).get("retry") if extra else None
                res = _call() if not retry_cfg else self._run_with_retry_sync(_call, retry_cfg)

            # Call response hook
            duration_ms = (time.time() - start_time) * 1000
            response_ctx = ResponseContext(
                request=request_ctx,
                response=res,
                duration_ms=duration_ms,
                token_usage=getattr(res, "usage_metadata", None),
            )
            self._logging_hooks.call_response_sync(response_ctx)

            if output_schema is not None and is_pydantic_schema(output_schema):
                return coerce_structured_result(output_schema, res)

            try:
                return apply_output_gate(res, self._hitl)
            except Exception:
                return res

        except Exception as e:
            # Call error hook
            duration_ms = (time.time() - start_time) * 1000
            error_ctx = ErrorContext(
                request=request_ctx,
                error=e,
                duration_ms=duration_ms,
            )
            self._logging_hooks.call_error_sync(error_ctx)
            # Translate provider error to ai-infra error
            raise translate_provider_error(e, provider=provider, model=model_name) from e

    async def achat(
        self,
        user_msg: str,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        system: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        output_schema: Union[type[BaseModel], Dict[str, Any], None] = None,
        output_method: (
            Literal["json_schema", "json_mode", "function_calling", "prompt"] | None
        ) = "prompt",
        **model_kwargs,
    ):
        """Async version of chat().

        Args:
            user_msg: The user's message
            provider: LLM provider (auto-detected if None)
            model_name: Model name (uses provider default if None)
            system: Optional system message
            extra: Extra options (e.g., {"retry": {"max_attempts": 3}})
            output_schema: Pydantic model for structured output
            output_method: How to extract structured output
            **model_kwargs: Additional model kwargs

        Returns:
            Response message or structured output if output_schema provided
        """
        sanitize_model_kwargs(model_kwargs)

        # Resolve provider and model (auto-detect if not specified)
        provider, model_name = self._resolve_provider_and_model(provider, model_name)

        # Create request context for logging hooks
        request_ctx = RequestContext(
            user_msg=user_msg,
            system=system,
            provider=provider,
            model_name=model_name,
            model_kwargs=model_kwargs,
        )
        await self._logging_hooks.call_request_async(request_ctx)
        start_time = time.time()

        try:
            if output_schema is not None and output_method == "prompt":
                res = await self._prompt_structured_async(
                    user_msg=user_msg,
                    system=system,
                    provider=provider,
                    model_name=model_name,
                    schema=output_schema,
                    extra=extra,
                    model_kwargs=model_kwargs,
                )
            else:
                if output_schema is not None:
                    model = self.with_structured_output(
                        provider, model_name, output_schema, method=output_method, **model_kwargs
                    )
                else:
                    model = self.set_model(provider, model_name, **model_kwargs)

                messages = _make_messages(user_msg, system)

                async def _call():
                    return await model.ainvoke(messages)

                retry_cfg = (extra or {}).get("retry") if extra else None
                res = await (_with_retry_util(_call, **retry_cfg) if retry_cfg else _call())

            # Call response hook
            duration_ms = (time.time() - start_time) * 1000
            response_ctx = ResponseContext(
                request=request_ctx,
                response=res,
                duration_ms=duration_ms,
                token_usage=getattr(res, "usage_metadata", None),
            )
            await self._logging_hooks.call_response_async(response_ctx)

            if output_schema is not None and is_pydantic_schema(output_schema):
                return coerce_structured_result(output_schema, res)

            try:
                return await apply_output_gate_async(res, self._hitl)
            except Exception:
                return res

        except Exception as e:
            # Call error hook
            duration_ms = (time.time() - start_time) * 1000
            error_ctx = ErrorContext(
                request=request_ctx,
                error=e,
                duration_ms=duration_ms,
            )
            await self._logging_hooks.call_error_async(error_ctx)
            # Translate provider error to ai-infra error
            raise translate_provider_error(e, provider=provider, model=model_name) from e

    async def stream_tokens(
        self,
        user_msg: str,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        system: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **model_kwargs,
    ):
        """Stream tokens from the model.

        Args:
            user_msg: The user's message
            provider: LLM provider (auto-detected if None)
            model_name: Model name (uses provider default if None)
            system: Optional system message
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Maximum tokens to generate
            **model_kwargs: Additional model kwargs

        Yields:
            Tuple of (token, metadata) for each token
        """
        sanitize_model_kwargs(model_kwargs)

        # Resolve provider and model (auto-detect if not specified)
        provider, model_name = self._resolve_provider_and_model(provider, model_name)

        if temperature is not None:
            model_kwargs["temperature"] = temperature
        if top_p is not None:
            model_kwargs["top_p"] = top_p
        if max_tokens is not None:
            model_kwargs["max_tokens"] = max_tokens
        model = self.set_model(provider, model_name, **model_kwargs)
        messages = _make_messages(user_msg, system)
        async for event in model.astream(messages):
            text = getattr(event, "content", None)
            if text is None:
                text = getattr(event, "delta", None) or getattr(event, "text", None)
            if text is None:
                text = str(event)
            meta = {"raw": event}
            yield text, meta
