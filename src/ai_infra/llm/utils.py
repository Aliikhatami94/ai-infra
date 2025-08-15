import os
from langchain.chat_models import init_chat_model
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Dict
import asyncio, random

from ai_infra.llm.providers import Providers
from ai_infra.llm.models import Models
from ai_infra.llm.tool_controls import ToolCallControls

ProviderModel = Tuple[str, str]
Candidate = Union[ProviderModel, dict]

class FallbackError(RuntimeError):
    def __init__(self, message: str, errors: List[BaseException]):
        super().__init__(message)
        self.errors = errors

# Validation Functions
def validate_provider(provider: str) -> None:
    """Validate that the provider is supported."""
    provider_names = [v for k, v in Providers.__dict__.items() if not k.startswith('__') and not callable(v)]
    if provider not in provider_names:
        raise ValueError(f"Unknown provider: {provider}")

def validate_model(provider: str, model_name: str) -> None:
    """Validate that the model is supported for the given provider."""
    valid_models = getattr(Models, provider)
    if model_name not in [m.value for m in valid_models]:
        raise ValueError(f"Invalid model_name '{model_name}' for provider '{provider}'.")

def validate_provider_and_model(provider: str, model_name: str) -> None:
    """Validate both provider and model in a single call."""
    validate_provider(provider)
    validate_model(provider, model_name)

# Model Utility Functions
def build_model_key(provider: str, model_name: str) -> str:
    """Build a unique key for caching models."""
    return f"{provider}:{model_name}"

def initialize_model(key: str, provider: str, **kwargs):
    """Initialize a chat model with the given parameters."""
    return init_chat_model(
        key,
        api_key=os.environ.get(f"{provider.upper()}_API_KEY"),
        **kwargs
    )

def sanitize_model_kwargs(model_kwargs: Dict[str, Any], banned: Optional[List[str]] = None) -> Dict[str, Any]:
    """Remove agent/tool-only kwargs from a model kwargs dict (in place).

    Returns the mutated dict for chaining.
    """
    if not model_kwargs:
        return model_kwargs
    banned = banned or ["tools", "tool_choice", "parallel_tool_calls", "force_once"]
    for b in banned:
        model_kwargs.pop(b, None)
    return model_kwargs

async def with_retry(afn: Callable[[], Any], *, max_tries: int = 3, base: float = 0.5, jitter: float = 0.2):
    """Generic exponential backoff retry around an awaited call factory.

    Parameters:
        afn: zero-arg async callable to execute.
        max_tries: maximum attempts (>=1)
        base: base backoff seconds (grows exponentially)
        jitter: max random jitter seconds added each attempt
    """
    last = None
    for i in range(max_tries):
        try:
            return await afn()
        except Exception as e:  # pragma: no cover - defensive
            last = e
            if i == max_tries - 1:
                break
            await asyncio.sleep(base * (2 ** i) + random.random() * jitter)
    raise last if last else RuntimeError("Retry failed with unknown error")

def make_messages(user: str, system: Optional[str] = None, extras: Optional[List[Dict[str, Any]]] = None):
    msgs: List[Dict[str, Any]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    if extras:
        msgs.extend(extras)
    return msgs

def _resolve_candidate(c: Candidate) -> Tuple[str, str, dict]:
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
    raise TypeError(f"Unsupported candidate type: {type(c)}")

def is_valid_response(res: Any) -> bool:
    """Generic 'did we get something usable?' check."""
    content = getattr(res, "content", None)
    if content is not None:
        return str(content).strip() != ""
    if isinstance(res, dict) and isinstance(res.get("messages"), list) and res["messages"]:
        last = res["messages"][-1]
        if hasattr(last, "content"):
            return str(getattr(last, "content", "")).strip() != ""
        if isinstance(last, dict):
            return str(last.get("content", "")).strip() != ""
    return res is not None

def merge_overrides(
        base_extra: Optional[Dict[str, Any]],
        base_model_kwargs: Optional[Dict[str, Any]],
        base_tools: Optional[List[Any]],
        base_tool_controls: Optional[ToolCallControls | Dict[str, Any]],
        overrides: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[List[Any]], Optional[ToolCallControls | Dict[str, Any]]]:
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