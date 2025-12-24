"""Convert Python objects to AI-compatible function tools.

This module provides utilities to automatically convert any Python object's
methods into function tools that can be used with ai-infra Agent.

Example:
    >>> from ai_infra.tools import tools_from_object
    >>>
    >>> class Calculator:
    ...     def add(self, a: float, b: float) -> float:
    ...         '''Add two numbers.'''
    ...         return a + b
    ...
    >>> calc = Calculator()
    >>> tools = tools_from_object(calc)
    >>>
    >>> from ai_infra import Agent
    >>> agent = Agent(tools=tools)
    >>> result = agent.run("What is 5 + 3?")

Note:
    We intentionally do NOT use `from __future__ import annotations` here
    because ai-infra Agent needs actual type objects (not string annotations)
    for Pydantic model parameter resolution.
"""

import asyncio
import functools
import inspect
import logging
import re
from collections.abc import Callable
from typing import Any, TypeVar, get_type_hints

logger = logging.getLogger(__name__)


# Marker attribute for @tool decorator
_TOOL_CONFIG_ATTR = "_ai_infra_tool_config"
# Marker attribute for @tool_exclude decorator
_TOOL_EXCLUDE_ATTR = "_ai_infra_tool_exclude"


__all__ = [
    "tool",
    "tool_exclude",
    "tools_from_object",
]


# =============================================================================
# Decorators
# =============================================================================


class ToolConfig:
    """Configuration for a method marked with @tool decorator."""

    def __init__(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        include: bool = True,
    ):
        self.name = name
        self.description = description
        self.include = include


F = TypeVar("F", bound=Callable[..., Any])


def tool(
    name: str | None = None,
    description: str | None = None,
) -> Callable[[F], F]:
    """Decorator to mark a method as an AI tool with custom configuration.

    Use this to override the default name or description for a method,
    or to explicitly mark a method for inclusion.

    Args:
        name: Custom tool name (default: method name with prefix).
        description: Custom description (default: method docstring).

    Returns:
        Decorator function.

    Example:
        >>> class Service:
        ...     @tool(name="fetch_user", description="Get a user by ID")
        ...     def get_user(self, user_id: str) -> User:
        ...         return self.db.get(user_id)
    """

    def decorator(func: F) -> F:
        config = ToolConfig(name=name, description=description, include=True)
        setattr(func, _TOOL_CONFIG_ATTR, config)
        return func

    return decorator


def tool_exclude(func: F) -> F:
    """Decorator to exclude a method from tool generation.

    Use this to prevent a public method from being converted to a tool.

    Example:
        >>> class Service:
        ...     def public_action(self) -> str:
        ...         '''This becomes a tool.'''
        ...         return "done"
        ...
        ...     @tool_exclude
        ...     def helper_method(self) -> str:
        ...         '''This is NOT a tool.'''
        ...         return "helper"
    """
    setattr(func, _TOOL_EXCLUDE_ATTR, True)
    return func


# =============================================================================
# Core Implementation
# =============================================================================


def _to_snake_case(name: str) -> str:
    """Convert CamelCase or PascalCase to snake_case.

    Args:
        name: The name to convert.

    Returns:
        snake_case version of the name.

    Examples:
        >>> _to_snake_case("MyClass")
        'my_class'
        >>> _to_snake_case("RobotArm")
        'robot_arm'
        >>> _to_snake_case("HTTPClient")
        'http_client'
    """
    # Insert underscore before uppercase letters (except at start)
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    # Handle consecutive uppercase (e.g., HTTPClient -> http_client)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


def _get_method_candidates(obj: Any) -> list[tuple[str, Callable[..., Any]]]:
    """Get all callable methods from an object.

    Args:
        obj: The object to inspect.

    Returns:
        List of (name, method) tuples.
    """
    candidates: list[tuple[str, Callable[..., Any]]] = []

    for name, method in inspect.getmembers(obj):
        # Skip non-callables
        if not callable(method):
            continue

        # Skip dunder methods
        if name.startswith("__") and name.endswith("__"):
            continue

        # Check if it's a bound method or function
        if inspect.ismethod(method) or inspect.isfunction(method):
            candidates.append((name, method))
        # Also include callable objects (like functools.partial)
        elif hasattr(method, "__call__") and not isinstance(method, type):
            # Skip classes and types
            candidates.append((name, method))

    return candidates


def _filter_methods(
    candidates: list[tuple[str, Callable[..., Any]]],
    *,
    methods: list[str] | None = None,
    exclude: list[str] | None = None,
    include_private: bool = False,
) -> list[tuple[str, Callable[..., Any]]]:
    """Filter method candidates based on rules.

    Filtering order:
    1. Exclude private methods (unless include_private=True)
    2. Apply methods include list (if specified)
    3. Apply exclude list
    4. Check @tool_exclude decorator

    Args:
        candidates: List of (name, method) tuples.
        methods: Specific methods to include (None = all).
        exclude: Methods to exclude.
        include_private: Whether to include _underscore methods.

    Returns:
        Filtered list of (name, method) tuples.
    """
    exclude = exclude or []
    filtered: list[tuple[str, Callable[..., Any]]] = []

    for name, method in candidates:
        # 1. Check private methods
        if name.startswith("_") and not include_private:
            continue

        # 2. Check @tool_exclude decorator
        if getattr(method, _TOOL_EXCLUDE_ATTR, False):
            continue

        # 3. Apply methods include list
        if methods is not None and name not in methods:
            # Exception: if method has @tool decorator, still include it
            if not hasattr(method, _TOOL_CONFIG_ATTR):
                continue

        # 4. Apply exclude list
        if name in exclude:
            continue

        filtered.append((name, method))

    return filtered


def _generate_docstring(
    method: Callable[..., Any],
    method_name: str,
    class_name: str,
    type_hints: dict[str, Any],
) -> str:
    """Generate a docstring for a method.

    Uses the original docstring if available, otherwise generates one
    from the method signature.

    Args:
        method: The method.
        method_name: Name of the method.
        class_name: Name of the class.
        type_hints: Type hints from the method.

    Returns:
        Docstring for the tool.
    """
    original_doc = inspect.getdoc(method)

    if original_doc:
        return original_doc

    # Generate fallback docstring from signature
    params = []
    sig = inspect.signature(method)
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        param_type = type_hints.get(param_name, Any)
        type_name = getattr(param_type, "__name__", str(param_type))
        params.append(f"{param_name} ({type_name})")

    param_str = ", ".join(params) if params else "none"
    return f"Call {method_name} on {class_name}. Parameters: {param_str}."


def _create_tool_function(
    obj: Any,
    method: Callable[..., Any],
    method_name: str,
    tool_name: str,
    docstring: str,
    is_async: bool,
    async_wrapper: bool,
) -> Callable[..., Any]:
    """Create a tool function that wraps a method.

    Args:
        obj: The object instance.
        method: The bound method.
        method_name: Original method name.
        tool_name: Name for the tool.
        docstring: Tool docstring.
        is_async: Whether the method is async.
        async_wrapper: Whether to wrap sync methods for async.

    Returns:
        Callable tool function.
    """
    sig = inspect.signature(method)

    # Get parameters excluding 'self'
    params = [p for name, p in sig.parameters.items() if name != "self"]
    new_sig = sig.replace(parameters=params)

    if is_async:
        # Async method - create async wrapper
        @functools.wraps(method)
        async def async_tool(*args: Any, **kwargs: Any) -> Any:
            return await method(*args, **kwargs)

        async_tool.__name__ = tool_name
        async_tool.__doc__ = docstring
        async_tool.__signature__ = new_sig  # type: ignore[attr-defined]
        return async_tool

    elif async_wrapper:
        # Sync method with async wrapper
        @functools.wraps(method)
        async def async_wrapped_tool(*args: Any, **kwargs: Any) -> Any:
            # Run sync method in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, functools.partial(method, *args, **kwargs))

        async_wrapped_tool.__name__ = tool_name
        async_wrapped_tool.__doc__ = docstring
        async_wrapped_tool.__signature__ = new_sig  # type: ignore[attr-defined]
        return async_wrapped_tool

    else:
        # Sync method without wrapper
        @functools.wraps(method)
        def sync_tool(*args: Any, **kwargs: Any) -> Any:
            return method(*args, **kwargs)

        sync_tool.__name__ = tool_name
        sync_tool.__doc__ = docstring
        sync_tool.__signature__ = new_sig  # type: ignore[attr-defined]
        return sync_tool


def tools_from_object(
    obj: Any,
    *,
    methods: list[str] | None = None,
    exclude: list[str] | None = None,
    prefix: str | None = None,
    include_private: bool = False,
    async_wrapper: bool = True,
) -> list[Callable[..., Any]]:
    """Convert an object's methods into AI function tools.

    Automatically extracts callable methods from an object and wraps them
    as function tools compatible with ai-infra Agent.

    Args:
        obj: The object instance to convert.
        methods: Specific methods to include (None = all public methods).
        exclude: Methods to exclude from conversion.
        prefix: Tool name prefix (default: class name in snake_case).
        include_private: Include _underscore methods (default: False).
        async_wrapper: Wrap sync methods for async compatibility (default: True).

    Returns:
        List of callable functions compatible with ai-infra Agent.

    Example:
        >>> class Calculator:
        ...     def add(self, a: float, b: float) -> float:
        ...         '''Add two numbers.'''
        ...         return a + b
        ...
        >>> tools = tools_from_object(Calculator())
        >>> from ai_infra import Agent
        >>> agent = Agent(tools=tools)
        >>> agent.run("What is 5 + 3?")
    """
    tools: list[Callable[..., Any]] = []

    # Get class info
    cls = type(obj)
    class_name = cls.__name__

    # Determine prefix
    if prefix is None:
        prefix = _to_snake_case(class_name)
    elif prefix == "":
        prefix = ""  # No prefix

    # Get method candidates
    candidates = _get_method_candidates(obj)

    # Filter methods
    filtered = _filter_methods(
        candidates,
        methods=methods,
        exclude=exclude,
        include_private=include_private,
    )

    for method_name, method in filtered:
        # Get type hints (may fail for some methods)
        try:
            type_hints = get_type_hints(method)
        except Exception:
            type_hints = {}

        # Check for @tool decorator config
        tool_config: ToolConfig | None = getattr(method, _TOOL_CONFIG_ATTR, None)

        # Determine tool name
        if tool_config and tool_config.name:
            tool_name = tool_config.name
        elif prefix:
            tool_name = f"{prefix}_{method_name}"
        else:
            tool_name = method_name

        # Determine docstring
        if tool_config and tool_config.description:
            docstring = tool_config.description
        else:
            docstring = _generate_docstring(method, method_name, class_name, type_hints)

        # Check if method is async
        is_async = asyncio.iscoroutinefunction(method)

        # Create tool function
        tool_func = _create_tool_function(
            obj=obj,
            method=method,
            method_name=method_name,
            tool_name=tool_name,
            docstring=docstring,
            is_async=is_async,
            async_wrapper=async_wrapper,
        )

        tools.append(tool_func)

    logger.debug(
        "Created %d tools from %s (prefix=%r)",
        len(tools),
        class_name,
        prefix,
    )

    return tools


# =============================================================================
# Property Support (Advanced)
# =============================================================================


def tools_from_object_with_properties(
    obj: Any,
    *,
    methods: list[str] | None = None,
    exclude: list[str] | None = None,
    prefix: str | None = None,
    include_private: bool = False,
    async_wrapper: bool = True,
    include_properties: bool = True,
) -> list[Callable[..., Any]]:
    """Convert an object's methods and properties into AI function tools.

    Like tools_from_object, but also converts readable properties into
    getter tools.

    Args:
        obj: The object instance to convert.
        methods: Specific methods to include (None = all public methods).
        exclude: Methods to exclude from conversion.
        prefix: Tool name prefix (default: class name in snake_case).
        include_private: Include _underscore methods (default: False).
        async_wrapper: Wrap sync methods for async compatibility (default: True).
        include_properties: Include property getters as tools (default: True).

    Returns:
        List of callable functions compatible with ai-infra Agent.

    Example:
        >>> class Robot:
        ...     @property
        ...     def position(self) -> dict:
        ...         '''Current robot position.'''
        ...         return {"x": 0, "y": 0}
        ...
        >>> tools = tools_from_object_with_properties(Robot())
        >>> # Creates: robot_get_position() tool
    """
    # Get base tools
    tools = tools_from_object(
        obj,
        methods=methods,
        exclude=exclude,
        prefix=prefix,
        include_private=include_private,
        async_wrapper=async_wrapper,
    )

    if not include_properties:
        return tools

    # Get class info
    cls = type(obj)
    class_name = cls.__name__

    # Determine prefix
    if prefix is None:
        prefix = _to_snake_case(class_name)
    elif prefix == "":
        prefix = ""

    exclude = exclude or []

    # Find properties
    for name in dir(cls):
        # Skip private/dunder
        if name.startswith("_"):
            continue

        # Skip if in exclude list
        if name in exclude:
            continue

        # Check if it's a property
        attr = getattr(cls, name, None)
        if not isinstance(attr, property):
            continue

        # Get the getter
        getter = attr.fget
        if getter is None:
            continue

        # Get docstring
        docstring = inspect.getdoc(getter) or f"Get {name} from {class_name}."

        # Create tool name
        if prefix:
            tool_name = f"{prefix}_get_{name}"
        else:
            tool_name = f"get_{name}"

        # Create getter tool
        def make_getter(prop_name: str) -> Callable[[], Any]:
            def getter_tool() -> Any:
                return getattr(obj, prop_name)

            return getter_tool

        getter_func = make_getter(name)
        getter_func.__name__ = tool_name
        getter_func.__doc__ = docstring

        tools.append(getter_func)

    return tools
