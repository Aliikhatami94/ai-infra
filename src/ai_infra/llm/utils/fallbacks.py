from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

ProviderModel = tuple[str, str]
Candidate = ProviderModel | dict


class FallbackError(RuntimeError):
    def __init__(self, message: str, errors: list[BaseException]):
        super().__init__(message)
        self.errors = errors


# Py3.11+ has typing.assert_never
try:
    from typing import assert_never
except ImportError:  # Py3.10 compatibility
    from typing import assert_never


def _resolve_candidate(c: Candidate) -> tuple[str, str, dict]:
    """Normalize a candidate to (provider, model_name, overrides)."""
    if isinstance(c, tuple):
        provider, model_name = c
        return provider, model_name, {}
    if isinstance(c, dict):
        provider_val: str | None = c.get("provider")
        model_name_val: str | None = c.get("model_name") or c.get("model")
        if not provider_val or not model_name_val:
            raise ValueError(
                "Candidate dict must include 'provider' and 'model_name' (or 'model')."
            )
        overrides = {k: v for k, v in c.items() if k not in ("provider", "model_name", "model")}
        return provider_val, model_name_val, overrides
    # statically “unreachable”, but keep the guard for runtime safety
    assert_never(c)  # will raise TypeError if ever reached


def merge_overrides(
    base_extra: dict[str, Any] | None,
    base_model_kwargs: dict[str, Any] | None,
    base_tools: list[Any] | None,
    base_tool_controls: Any | None,  # ToolCallControls | dict[str, Any]
    overrides: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[Any] | None, Any | None]:
    eff_extra = {**(base_extra or {}), **overrides.get("extra", {})}
    eff_model_kwargs = {
        **(base_model_kwargs or {}),
        **overrides.get("model_kwargs", {}),
    }
    eff_tools = overrides.get("tools", base_tools)
    eff_tool_controls = overrides.get("tool_controls", base_tool_controls)
    return eff_extra, eff_model_kwargs, eff_tools, eff_tool_controls


def run_with_fallbacks(
    candidates: Sequence[Candidate],
    run_single: Callable[[str, str, dict], Any],
    *,
    validate: Callable[[Any], bool] | None = None,
    should_retry: Callable[[BaseException | None, Any, int, str, str], bool] | None = None,
    on_attempt: Callable[[int, str, str], None] | None = None,
) -> Any:
    """
    Try each candidate until one returns a valid result.

    - run_single(provider, model_name, overrides) -> result
    - validate(result) -> True means accept; default accepts any non-None result.
    - should_retry(exc, result, attempt_idx, provider, model_name) -> True to continue.
      If exc is not None, result will be None.
    - candidates may be (provider, model_name) or dicts with overrides.
    """
    errs: list[BaseException] = []
    if validate is None:

        def _default_validate(result: Any) -> bool:
            return result is not None

        validate = _default_validate
    if should_retry is None:

        def _default_should_retry(
            exc: BaseException | None,
            res: Any,
            attempt_idx: int,
            provider: str,
            model_name: str,
        ) -> bool:
            return (exc is not None) or (not validate(res))

        should_retry = _default_should_retry

    for i, cand in enumerate(candidates):
        provider, model_name, overrides = _resolve_candidate(cand)
        if on_attempt:
            on_attempt(i, provider, model_name)
        try:
            result = run_single(provider, model_name, overrides)
            if not should_retry(None, result, i, provider, model_name):
                return result
        except BaseException as e:
            errs.append(e)
            if not should_retry(e, None, i, provider, model_name):
                raise

    if errs:
        raise FallbackError("All fallback candidates failed.", errs)
    raise RuntimeError("All fallback candidates produced invalid results.")


async def arun_with_fallbacks(
    candidates: Sequence[Candidate],
    run_single_async: Callable[[str, str, dict], Any],
    *,
    validate: Callable[[Any], bool] | None = None,
    should_retry: Callable[[BaseException | None, Any, int, str, str], bool] | None = None,
    on_attempt: Callable[[int, str, str], None] | None = None,
) -> Any:
    errs: list[BaseException] = []
    if validate is None:

        def _default_validate(result: Any) -> bool:
            return result is not None

        validate = _default_validate
    if should_retry is None:

        def _default_should_retry(
            exc: BaseException | None,
            res: Any,
            attempt_idx: int,
            provider: str,
            model_name: str,
        ) -> bool:
            return (exc is not None) or (not validate(res))

        should_retry = _default_should_retry

    for i, cand in enumerate(candidates):
        provider, model_name, overrides = _resolve_candidate(cand)
        if on_attempt:
            on_attempt(i, provider, model_name)
        try:
            result = await run_single_async(provider, model_name, overrides)
            if not should_retry(None, result, i, provider, model_name):
                return result
        except BaseException as e:
            errs.append(e)
            if not should_retry(e, None, i, provider, model_name):
                raise

    if errs:
        raise FallbackError("All async fallback candidates failed.", errs)
    raise RuntimeError("All async fallback candidates produced invalid results.")
