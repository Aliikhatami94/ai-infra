from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

ProviderModel = Tuple[str, str]
Candidate = Union[ProviderModel, dict]

class FallbackError(RuntimeError):
    def __init__(self, message: str, errors: List[BaseException]):
        super().__init__(message)
        self.errors = errors

# Py3.11+ has typing.assert_never
try:
    from typing import assert_never
except Exception:  # Py3.10 compatibility
    from typing_extensions import assert_never  # type: ignore

def _resolve_candidate(c: Candidate) -> Tuple[str, str, dict]:
    """Normalize a candidate to (provider, model_name, overrides)."""
    if isinstance(c, tuple):
        prov, model = c
        return prov, model, {}
    if isinstance(c, dict):
        prov = c.get("provider")
        model = c.get("model_name") or c.get("model")
        if not prov or not model:
            raise ValueError("Candidate dict must include 'provider' and 'model_name' (or 'model').")
        overrides = {k: v for k, v in c.items() if k not in ("provider", "model_name", "model")}
        return prov, model, overrides
    # statically “unreachable”, but keep the guard for runtime safety
    assert_never(c)  # will raise TypeError if ever reached

def merge_overrides(
        base_extra: Optional[Dict[str, Any]],
        base_model_kwargs: Optional[Dict[str, Any]],
        base_tools: Optional[List[Any]],
        base_tool_controls: Optional[Any],  # ToolCallControls | Dict[str, Any]
        overrides: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[List[Any]], Optional[Any]]:
    eff_extra = {**(base_extra or {}), **overrides.get("extra", {})}
    eff_model_kwargs = {**(base_model_kwargs or {}), **overrides.get("model_kwargs", {})}
    eff_tools = overrides.get("tools", base_tools)
    eff_tool_controls = overrides.get("tool_controls", base_tool_controls)
    return eff_extra, eff_model_kwargs, eff_tools, eff_tool_controls

def run_with_fallbacks(
        candidates: Sequence[Candidate],
        run_single: Callable[[str, str, dict], Any],
        *,
        validate: Optional[Callable[[Any], bool]] = None,
        should_retry: Optional[Callable[[Optional[BaseException], Any, int, str, str], bool]] = None,
        on_attempt: Optional[Callable[[int, str, str], None]] = None,
) -> Any:
    """
    Try each candidate until one returns a valid result.

    - run_single(provider, model_name, overrides) -> result
    - validate(result) -> True means accept; default accepts any non-None result.
    - should_retry(exc, result, attempt_idx, provider, model_name) -> True to continue.
      If exc is not None, result will be None.
    - candidates may be (provider, model_name) or dicts with overrides.
    """
    errs: List[BaseException] = []
    if validate is None:
        validate = lambda r: r is not None
    if should_retry is None:
        should_retry = lambda exc, res, i, p, m: (exc is not None) or (not validate(res))

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
        validate: Optional[Callable[[Any], bool]] = None,
        should_retry: Optional[Callable[[Optional[BaseException], Any, int, str, str], bool]] = None,
        on_attempt: Optional[Callable[[int, str, str], None]] = None,
) -> Any:
    errs: List[BaseException] = []
    if validate is None:
        validate = lambda r: r is not None
    if should_retry is None:
        should_retry = lambda exc, res, i, p, m: (exc is not None) or (not validate(res))

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