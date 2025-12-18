"""Runtime validation for ai-infra.

This module provides input and output validation for LLM operations:
- Input validation (provider, model, temperature, max_tokens)
- Output validation (structured output against Pydantic schemas)
- Validation decorators for automatic validation

Usage:
    from ai_infra.validation import validate_llm_params, validate_output

    # Validate LLM parameters
    validate_llm_params(
        provider="openai",
        model="gpt-4o",
        temperature=0.7,
        max_tokens=1000,
    )

    # Validate output against schema
    result = validate_output(response, MyModel)
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar, get_type_hints

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from ai_infra.errors import ConfigurationError, ValidationError

# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T", bound=BaseModel)
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Constants
# =============================================================================

SUPPORTED_PROVIDERS = [
    "openai",
    "anthropic",
    "google_genai",
    "xai",
    "ollama",
    "azure_openai",
    "bedrock",
    "together",
    "groq",
    "deepseek",
]

# Temperature ranges by provider (most use 0-2, some 0-1)
TEMPERATURE_RANGES: dict[str, tuple[float, float]] = {
    "openai": (0.0, 2.0),
    "anthropic": (0.0, 1.0),
    "google_genai": (0.0, 2.0),
    "xai": (0.0, 2.0),
    "ollama": (0.0, 2.0),
    "azure_openai": (0.0, 2.0),
    "bedrock": (0.0, 1.0),
    "together": (0.0, 2.0),
    "groq": (0.0, 2.0),
    "deepseek": (0.0, 2.0),
}

# Default range if provider not specified
DEFAULT_TEMPERATURE_RANGE = (0.0, 2.0)


# =============================================================================
# Input Validators
# =============================================================================


def validate_provider(provider: str) -> None:
    """Validate that provider is supported.

    Args:
        provider: Provider name to validate

    Raises:
        ValidationError: If provider is not supported

    Example:
        validate_provider("openai")  # OK
        validate_provider("invalid")  # Raises ValidationError
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValidationError(
            f"Unknown provider: {provider}",
            field="provider",
            expected=f"one of {SUPPORTED_PROVIDERS}",
            details={"provider": provider, "supported": SUPPORTED_PROVIDERS},
        )


def validate_temperature(
    temperature: float,
    provider: str | None = None,
) -> None:
    """Validate temperature is within acceptable range.

    Args:
        temperature: Temperature value to validate
        provider: Optional provider for provider-specific ranges

    Raises:
        ValidationError: If temperature is out of range

    Example:
        validate_temperature(0.7)  # OK
        validate_temperature(5.0)  # Raises ValidationError
        validate_temperature(1.5, provider="anthropic")  # Raises (max is 1.0)
    """
    if provider:
        min_temp, max_temp = TEMPERATURE_RANGES.get(provider, DEFAULT_TEMPERATURE_RANGE)
    else:
        min_temp, max_temp = DEFAULT_TEMPERATURE_RANGE

    if not (min_temp <= temperature <= max_temp):
        raise ValidationError(
            f"Temperature {temperature} out of range [{min_temp}, {max_temp}]",
            field="temperature",
            expected=f"[{min_temp}, {max_temp}]",
            details={
                "temperature": temperature,
                "min": min_temp,
                "max": max_temp,
                "provider": provider,
            },
        )


def validate_max_tokens(max_tokens: int) -> None:
    """Validate max_tokens is a positive integer.

    Args:
        max_tokens: Max tokens value to validate

    Raises:
        ValidationError: If max_tokens is not positive
    """
    if max_tokens <= 0:
        raise ValidationError(
            f"max_tokens must be positive, got {max_tokens}",
            field="max_tokens",
            expected="positive integer",
            details={"max_tokens": max_tokens},
        )


def validate_messages(messages: list[dict[str, Any]]) -> None:
    """Validate message format for chat APIs.

    Args:
        messages: List of message dictionaries

    Raises:
        ValidationError: If messages are invalid
    """
    if not messages:
        raise ValidationError(
            "Messages cannot be empty",
            field="messages",
            expected="non-empty list of messages",
        )

    valid_roles = {"system", "user", "assistant", "function", "tool"}

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValidationError(
                f"Message at index {i} must be a dict, got {type(msg).__name__}",
                field=f"messages[{i}]",
                expected="dict",
            )

        if "role" not in msg:
            raise ValidationError(
                f"Message at index {i} missing 'role' field",
                field=f"messages[{i}].role",
                expected="role field required",
            )

        if msg["role"] not in valid_roles:
            raise ValidationError(
                f"Invalid role '{msg['role']}' at index {i}",
                field=f"messages[{i}].role",
                expected=f"one of {list(valid_roles)}",
                details={"role": msg["role"], "valid_roles": list(valid_roles)},
            )

        if "content" not in msg and "tool_calls" not in msg:
            raise ValidationError(
                f"Message at index {i} missing 'content' or 'tool_calls'",
                field=f"messages[{i}].content",
                expected="content or tool_calls field required",
            )


def validate_llm_params(
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> None:
    """Validate all LLM parameters at once.

    Args:
        provider: LLM provider name
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        messages: Chat messages

    Raises:
        ValidationError: If any parameter is invalid

    Example:
        validate_llm_params(
            provider="openai",
            temperature=0.7,
            max_tokens=1000,
        )
    """
    errors: list[str] = []

    if provider is not None:
        try:
            validate_provider(provider)
        except ValidationError as e:
            errors.append(str(e.message))

    if temperature is not None:
        try:
            validate_temperature(temperature, provider)
        except ValidationError as e:
            errors.append(str(e.message))

    if max_tokens is not None:
        try:
            validate_max_tokens(max_tokens)
        except ValidationError as e:
            errors.append(str(e.message))

    if messages is not None:
        try:
            validate_messages(messages)
        except ValidationError as e:
            errors.append(str(e.message))

    if errors:
        raise ValidationError(
            f"Validation failed: {'; '.join(errors)}",
            details={"errors": errors},
        )


# =============================================================================
# Output Validators
# =============================================================================


def validate_output(
    output: Any,
    schema: type[T],
    *,
    strict: bool = True,
) -> T:
    """Validate output against a Pydantic schema.

    Args:
        output: Raw output to validate (dict, str, or model instance)
        schema: Pydantic model class to validate against
        strict: If True, raise on validation failure. If False, return None.

    Returns:
        Validated Pydantic model instance

    Raises:
        ValidationError: If output doesn't match schema (when strict=True)

    Example:
        class Person(BaseModel):
            name: str
            age: int

        result = validate_output({"name": "Alice", "age": 30}, Person)
        print(result.name)  # "Alice"
    """
    # Already the right type
    if isinstance(output, schema):
        return output

    try:
        # Dict input
        if isinstance(output, dict):
            return schema.model_validate(output)

        # String input (JSON)
        if isinstance(output, str):
            return schema.model_validate_json(output)

        # Try to extract from response-like object
        if hasattr(output, "content"):
            content = output.content
            if isinstance(content, str):
                return schema.model_validate_json(content)
            if isinstance(content, dict):
                return schema.model_validate(content)

        # Last resort: try direct validation
        return schema.model_validate(output)

    except PydanticValidationError as e:
        if not strict:
            return None  # type: ignore

        # Convert Pydantic errors to our ValidationError
        errors = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            errors.append(f"{loc}: {err['msg']}")

        raise ValidationError(
            f"Output validation failed against {schema.__name__}",
            details={
                "schema": schema.__name__,
                "errors": errors,
                "raw_errors": e.errors(),
            },
        ) from e
    except Exception as e:
        if not strict:
            return None  # type: ignore

        raise ValidationError(
            f"Failed to validate output: {e}",
            details={"schema": schema.__name__, "error": str(e)},
        ) from e


def validate_json_output(
    json_str: str,
    schema: type[T],
) -> T:
    """Validate JSON string against a Pydantic schema.

    Args:
        json_str: JSON string to validate
        schema: Pydantic model class

    Returns:
        Validated model instance

    Raises:
        ValidationError: If JSON is invalid or doesn't match schema
    """
    import json

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValidationError(
            f"Invalid JSON: {e}",
            details={"json": json_str[:100], "error": str(e)},
        ) from e

    return validate_output(data, schema)


# =============================================================================
# Validation Decorators
# =============================================================================


def validate_inputs(func: F) -> F:
    """Decorator to validate function inputs against type hints.

    Uses Pydantic to validate arguments that have Pydantic model type hints.

    Example:
        @validate_inputs
        def process(data: MyModel) -> str:
            return data.name

        process({"name": "test"})  # Validates and converts to MyModel
    """
    hints = get_type_hints(func)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Get parameter names
        import inspect

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Validate positional args
        args_list = list(args)
        for i, (param_name, arg) in enumerate(zip(params, args_list)):
            if param_name in hints:
                hint = hints[param_name]
                if isinstance(hint, type) and issubclass(hint, BaseModel):
                    if not isinstance(arg, hint):
                        try:
                            args_list[i] = hint.model_validate(arg)
                        except PydanticValidationError as e:
                            raise ValidationError(
                                f"Invalid argument '{param_name}'",
                                field=param_name,
                                details={"errors": e.errors()},
                            ) from e
        args = tuple(args_list)

        # Validate keyword args
        for param_name, arg in kwargs.items():
            if param_name in hints:
                hint = hints[param_name]
                if isinstance(hint, type) and issubclass(hint, BaseModel):
                    if not isinstance(arg, hint):
                        try:
                            kwargs[param_name] = hint.model_validate(arg)
                        except PydanticValidationError as e:
                            raise ValidationError(
                                f"Invalid argument '{param_name}'",
                                field=param_name,
                                details={"errors": e.errors()},
                            ) from e

        return func(*args, **kwargs)

    return wrapper  # type: ignore


def validate_return(schema: type[T]) -> Callable[[F], F]:
    """Decorator to validate function return value against schema.

    Example:
        @validate_return(MyModel)
        def get_data() -> MyModel:
            return {"name": "test"}  # Will be validated and converted
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            result = func(*args, **kwargs)
            return validate_output(result, schema)

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# Configuration Validators
# =============================================================================


def validate_config(config: dict[str, Any], required: list[str]) -> None:
    """Validate configuration dictionary has required keys.

    Args:
        config: Configuration dictionary
        required: List of required key names

    Raises:
        ConfigurationError: If required keys are missing
    """
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigurationError(
            f"Missing required configuration: {missing}",
            config_key=missing[0],
            details={"missing": missing, "provided": list(config.keys())},
        )


def validate_env_var(name: str, required: bool = True) -> str | None:
    """Validate and return environment variable.

    Args:
        name: Environment variable name
        required: If True, raise if not set

    Returns:
        Environment variable value or None

    Raises:
        ConfigurationError: If required and not set
    """
    import os

    value = os.getenv(name)
    if required and not value:
        raise ConfigurationError(
            f"Required environment variable not set: {name}",
            config_key=name,
            details={"env_var": name},
        )
    return value


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    "SUPPORTED_PROVIDERS",
    "TEMPERATURE_RANGES",
    # Input validators
    "validate_provider",
    "validate_temperature",
    "validate_max_tokens",
    "validate_messages",
    "validate_llm_params",
    # Output validators
    "validate_output",
    "validate_json_output",
    # Decorators
    "validate_inputs",
    "validate_return",
    # Config validators
    "validate_config",
    "validate_env_var",
]
