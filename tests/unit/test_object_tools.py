"""Tests for object_tools - tools_from_object() function."""

import asyncio
import inspect

import pytest

from ai_infra.tools.object_tools import (
    _filter_methods,
    _generate_docstring,
    _get_method_candidates,
    _to_snake_case,
    tool,
    tool_exclude,
    tools_from_object,
    tools_from_object_with_properties,
)

# =============================================================================
# Test Classes (Fixtures)
# =============================================================================


class SimpleCalculator:
    """A simple calculator for testing."""

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

    def _private_helper(self) -> str:
        """A private method that should be excluded by default."""
        return "private"

    def __dunder_method__(self) -> str:
        """A dunder method that should always be excluded."""
        return "dunder"


class AsyncService:
    """A service with async methods."""

    async def fetch_data(self, url: str) -> dict:
        """Fetch data from a URL."""
        return {"url": url, "data": "mocked"}

    async def process(self, data: dict) -> str:
        """Process data asynchronously."""
        return f"processed: {data}"

    def sync_method(self) -> str:
        """A sync method in an async service."""
        return "sync"


class DecoratedService:
    """A service with decorator-configured methods."""

    @tool(name="custom_action", description="A custom action with overrides")
    def my_action(self, value: int) -> str:
        """Original docstring that will be overridden."""
        return f"action: {value}"

    @tool_exclude
    def excluded_method(self) -> str:
        """This method should NOT become a tool."""
        return "excluded"

    def normal_method(self) -> str:
        """A normal method without decorators."""
        return "normal"


class NoDocstrings:
    """A class with methods that have no docstrings."""

    def action_one(self, x: int) -> str:
        return f"one: {x}"

    def action_two(self, a: str, b: int) -> dict:
        return {"a": a, "b": b}


class PropertyClass:
    """A class with properties."""

    def __init__(self):
        self._position = {"x": 0, "y": 0}
        self._status = "idle"

    @property
    def position(self) -> dict:
        """Current position coordinates."""
        return self._position

    @property
    def status(self) -> str:
        """Current status of the object."""
        return self._status

    def move(self, x: float, y: float) -> str:
        """Move to a new position."""
        self._position = {"x": x, "y": y}
        return f"Moved to ({x}, {y})"


class InheritedClass(SimpleCalculator):
    """A class that inherits from SimpleCalculator."""

    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a."""
        return a - b


class EmptyClass:
    """A class with no methods (except dunders)."""

    pass


class OnlyPrivate:
    """A class with only private methods."""

    def _private_one(self) -> str:
        return "one"

    def _private_two(self) -> str:
        return "two"


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestToSnakeCase:
    """Tests for _to_snake_case helper."""

    def test_pascal_case(self):
        assert _to_snake_case("MyClass") == "my_class"

    def test_camel_case(self):
        assert _to_snake_case("myClass") == "my_class"

    def test_already_snake_case(self):
        assert _to_snake_case("my_class") == "my_class"

    def test_uppercase_acronym(self):
        assert _to_snake_case("HTTPClient") == "http_client"

    def test_multiple_words(self):
        assert _to_snake_case("RobotArmController") == "robot_arm_controller"

    def test_single_word(self):
        assert _to_snake_case("Calculator") == "calculator"

    def test_all_lowercase(self):
        assert _to_snake_case("calculator") == "calculator"


class TestGetMethodCandidates:
    """Tests for _get_method_candidates helper."""

    def test_finds_public_methods(self):
        calc = SimpleCalculator()
        candidates = _get_method_candidates(calc)
        names = [name for name, _ in candidates]

        assert "add" in names
        assert "multiply" in names

    def test_finds_private_methods(self):
        calc = SimpleCalculator()
        candidates = _get_method_candidates(calc)
        names = [name for name, _ in candidates]

        assert "_private_helper" in names

    def test_excludes_dunder_methods(self):
        calc = SimpleCalculator()
        candidates = _get_method_candidates(calc)
        names = [name for name, _ in candidates]

        # Should not find __init__, __str__, etc.
        dunder_names = [n for n in names if n.startswith("__") and n.endswith("__")]
        assert len(dunder_names) == 0


class TestFilterMethods:
    """Tests for _filter_methods helper."""

    def test_filters_private_by_default(self):
        calc = SimpleCalculator()
        candidates = _get_method_candidates(calc)
        filtered = _filter_methods(candidates)
        names = [name for name, _ in filtered]

        assert "_private_helper" not in names
        assert "add" in names
        assert "multiply" in names

    def test_include_private(self):
        calc = SimpleCalculator()
        candidates = _get_method_candidates(calc)
        filtered = _filter_methods(candidates, include_private=True)
        names = [name for name, _ in filtered]

        assert "_private_helper" in names

    def test_methods_include_filter(self):
        calc = SimpleCalculator()
        candidates = _get_method_candidates(calc)
        filtered = _filter_methods(candidates, methods=["add"])
        names = [name for name, _ in filtered]

        assert "add" in names
        assert "multiply" not in names

    def test_exclude_filter(self):
        calc = SimpleCalculator()
        candidates = _get_method_candidates(calc)
        filtered = _filter_methods(candidates, exclude=["add"])
        names = [name for name, _ in filtered]

        assert "add" not in names
        assert "multiply" in names

    def test_tool_exclude_decorator(self):
        service = DecoratedService()
        candidates = _get_method_candidates(service)
        filtered = _filter_methods(candidates)
        names = [name for name, _ in filtered]

        assert "excluded_method" not in names
        assert "my_action" in names
        assert "normal_method" in names


class TestGenerateDocstring:
    """Tests for _generate_docstring helper."""

    def test_uses_existing_docstring(self):
        calc = SimpleCalculator()
        doc = _generate_docstring(calc.add, "add", "SimpleCalculator", {})
        assert doc == "Add two numbers."

    def test_generates_fallback(self):
        service = NoDocstrings()
        doc = _generate_docstring(service.action_one, "action_one", "NoDocstrings", {"x": int})
        assert "action_one" in doc
        assert "NoDocstrings" in doc
        assert "x" in doc


# =============================================================================
# tools_from_object Tests
# =============================================================================


class TestToolsFromObjectBasic:
    """Basic tests for tools_from_object."""

    def test_creates_tools_from_simple_class(self):
        calc = SimpleCalculator()
        tools = tools_from_object(calc)

        assert len(tools) >= 2  # At least add and multiply
        tool_names = [t.__name__ for t in tools]
        assert "simple_calculator_add" in tool_names
        assert "simple_calculator_multiply" in tool_names

    def test_tools_are_callable(self):
        calc = SimpleCalculator()
        tools = tools_from_object(calc)

        for t in tools:
            assert callable(t)

    def test_tools_have_docstrings(self):
        calc = SimpleCalculator()
        tools = tools_from_object(calc)

        for t in tools:
            assert t.__doc__ is not None
            assert len(t.__doc__) > 0

    def test_tools_have_signatures(self):
        calc = SimpleCalculator()
        tools = tools_from_object(calc)

        for t in tools:
            sig = inspect.signature(t)
            # Should not have 'self' parameter
            assert "self" not in sig.parameters


class TestToolsFromObjectPrefixNaming:
    """Tests for prefix and naming behavior."""

    def test_default_prefix_is_snake_case_class_name(self):
        calc = SimpleCalculator()
        tools = tools_from_object(calc)
        tool_names = [t.__name__ for t in tools]

        assert all(name.startswith("simple_calculator_") for name in tool_names)

    def test_custom_prefix(self):
        calc = SimpleCalculator()
        tools = tools_from_object(calc, prefix="calc")
        tool_names = [t.__name__ for t in tools]

        assert "calc_add" in tool_names
        assert "calc_multiply" in tool_names

    def test_empty_prefix(self):
        calc = SimpleCalculator()
        tools = tools_from_object(calc, prefix="")
        tool_names = [t.__name__ for t in tools]

        assert "add" in tool_names
        assert "multiply" in tool_names


class TestToolsFromObjectFiltering:
    """Tests for method filtering."""

    def test_methods_include_filter(self):
        calc = SimpleCalculator()
        tools = tools_from_object(calc, methods=["add"])
        tool_names = [t.__name__ for t in tools]

        assert len(tools) == 1
        assert "simple_calculator_add" in tool_names

    def test_exclude_filter(self):
        calc = SimpleCalculator()
        tools = tools_from_object(calc, exclude=["multiply"])
        tool_names = [t.__name__ for t in tools]

        assert "simple_calculator_add" in tool_names
        assert "simple_calculator_multiply" not in tool_names

    def test_private_methods_excluded_by_default(self):
        calc = SimpleCalculator()
        tools = tools_from_object(calc)
        tool_names = [t.__name__ for t in tools]

        assert not any("_private" in name for name in tool_names)

    def test_include_private(self):
        calc = SimpleCalculator()
        tools = tools_from_object(calc, include_private=True)
        tool_names = [t.__name__ for t in tools]

        assert "simple_calculator__private_helper" in tool_names


class TestToolsFromObjectAsync:
    """Tests for async method handling."""

    def test_async_methods_become_async_tools(self):
        service = AsyncService()
        tools = tools_from_object(service)

        fetch_tool = next(t for t in tools if "fetch_data" in t.__name__)
        assert asyncio.iscoroutinefunction(fetch_tool)

    def test_sync_methods_wrapped_as_async(self):
        service = AsyncService()
        tools = tools_from_object(service, async_wrapper=True)

        sync_tool = next(t for t in tools if "sync_method" in t.__name__)
        # With async_wrapper=True, sync methods become async
        assert asyncio.iscoroutinefunction(sync_tool)

    def test_sync_methods_stay_sync_when_wrapper_disabled(self):
        service = AsyncService()
        tools = tools_from_object(service, async_wrapper=False)

        sync_tool = next(t for t in tools if "sync_method" in t.__name__)
        # With async_wrapper=False, sync methods stay sync
        assert not asyncio.iscoroutinefunction(sync_tool)

    @pytest.mark.asyncio
    async def test_async_tool_execution(self):
        service = AsyncService()
        tools = tools_from_object(service)

        fetch_tool = next(t for t in tools if "fetch_data" in t.__name__)
        result = await fetch_tool(url="https://example.com")

        assert result["url"] == "https://example.com"


class TestToolsFromObjectDecorators:
    """Tests for @tool and @tool_exclude decorators."""

    def test_tool_decorator_custom_name(self):
        service = DecoratedService()
        tools = tools_from_object(service)
        tool_names = [t.__name__ for t in tools]

        # Should use custom name from @tool decorator
        assert "custom_action" in tool_names
        assert "decorated_service_my_action" not in tool_names

    def test_tool_decorator_custom_description(self):
        service = DecoratedService()
        tools = tools_from_object(service)

        custom_tool = next(t for t in tools if t.__name__ == "custom_action")
        assert custom_tool.__doc__ == "A custom action with overrides"

    def test_tool_exclude_decorator(self):
        service = DecoratedService()
        tools = tools_from_object(service)
        tool_names = [t.__name__ for t in tools]

        # excluded_method should not be in tools
        assert not any("excluded_method" in name for name in tool_names)

    def test_normal_method_included(self):
        service = DecoratedService()
        tools = tools_from_object(service)
        tool_names = [t.__name__ for t in tools]

        assert "decorated_service_normal_method" in tool_names


class TestToolsFromObjectEdgeCases:
    """Tests for edge cases."""

    def test_empty_class(self):
        obj = EmptyClass()
        tools = tools_from_object(obj)

        # Should return empty list, not error
        assert tools == []

    def test_only_private_methods(self):
        obj = OnlyPrivate()
        tools = tools_from_object(obj)

        # Should return empty list since private excluded by default
        assert tools == []

    def test_only_private_with_include_private(self):
        obj = OnlyPrivate()
        tools = tools_from_object(obj, include_private=True)

        assert len(tools) == 2

    def test_all_methods_excluded(self):
        calc = SimpleCalculator()
        tools = tools_from_object(calc, exclude=["add", "multiply"])

        # Should return empty list
        assert tools == []

    def test_inherited_methods(self):
        obj = InheritedClass()
        tools = tools_from_object(obj)
        tool_names = [t.__name__ for t in tools]

        # Should include inherited methods
        assert "inherited_class_add" in tool_names
        assert "inherited_class_multiply" in tool_names
        assert "inherited_class_subtract" in tool_names

    def test_no_docstrings_fallback(self):
        obj = NoDocstrings()
        tools = tools_from_object(obj)

        for t in tools:
            # Should have generated docstrings
            assert t.__doc__ is not None
            assert "NoDocstrings" in t.__doc__


class TestToolsFromObjectToolExecution:
    """Tests that tools actually execute correctly."""

    def test_sync_tool_execution(self):
        calc = SimpleCalculator()
        tools = tools_from_object(calc, async_wrapper=False)

        add_tool = next(t for t in tools if "add" in t.__name__)
        result = add_tool(a=5, b=3)

        assert result == 8.0

    def test_sync_tool_positional_args(self):
        calc = SimpleCalculator()
        tools = tools_from_object(calc, async_wrapper=False)

        multiply_tool = next(t for t in tools if "multiply" in t.__name__)
        result = multiply_tool(4, 7)

        assert result == 28.0

    @pytest.mark.asyncio
    async def test_wrapped_sync_tool_execution(self):
        calc = SimpleCalculator()
        tools = tools_from_object(calc, async_wrapper=True)

        add_tool = next(t for t in tools if "add" in t.__name__)
        result = await add_tool(a=10, b=20)

        assert result == 30.0


# =============================================================================
# tools_from_object_with_properties Tests
# =============================================================================


class TestToolsFromObjectWithProperties:
    """Tests for tools_from_object_with_properties."""

    def test_includes_property_getters(self):
        obj = PropertyClass()
        tools = tools_from_object_with_properties(obj)
        tool_names = [t.__name__ for t in tools]

        assert "property_class_get_position" in tool_names
        assert "property_class_get_status" in tool_names

    def test_includes_regular_methods(self):
        obj = PropertyClass()
        tools = tools_from_object_with_properties(obj)
        tool_names = [t.__name__ for t in tools]

        assert "property_class_move" in tool_names

    def test_property_getter_execution(self):
        obj = PropertyClass()
        tools = tools_from_object_with_properties(obj)

        position_tool = next(t for t in tools if "get_position" in t.__name__)
        result = position_tool()

        assert result == {"x": 0, "y": 0}

    def test_property_getter_has_docstring(self):
        obj = PropertyClass()
        tools = tools_from_object_with_properties(obj)

        position_tool = next(t for t in tools if "get_position" in t.__name__)
        assert position_tool.__doc__ is not None
        assert "position" in position_tool.__doc__.lower()

    def test_can_disable_properties(self):
        obj = PropertyClass()
        tools = tools_from_object_with_properties(obj, include_properties=False)
        tool_names = [t.__name__ for t in tools]

        assert not any("get_position" in name for name in tool_names)
        assert not any("get_status" in name for name in tool_names)
        assert "property_class_move" in tool_names


# =============================================================================
# AI-Infra Agent Compatibility Tests
# =============================================================================


class TestAIInfraCompatibility:
    """Tests for compatibility with ai-infra Agent."""

    def test_tools_have_name_attribute(self):
        calc = SimpleCalculator()
        tools = tools_from_object(calc)

        for t in tools:
            assert hasattr(t, "__name__")
            assert isinstance(t.__name__, str)
            assert len(t.__name__) > 0

    def test_tools_have_doc_attribute(self):
        calc = SimpleCalculator()
        tools = tools_from_object(calc)

        for t in tools:
            assert hasattr(t, "__doc__")
            assert t.__doc__ is not None

    def test_tools_have_signature(self):
        calc = SimpleCalculator()
        tools = tools_from_object(calc)

        for t in tools:
            sig = inspect.signature(t)
            assert sig is not None

    def test_tool_signature_no_self(self):
        calc = SimpleCalculator()
        tools = tools_from_object(calc)

        for t in tools:
            sig = inspect.signature(t)
            params = list(sig.parameters.keys())
            assert "self" not in params

    def test_import_from_ai_infra(self):
        """Verify we can import from ai_infra root."""
        from ai_infra import (
            tool,
            tool_exclude,
            tools_from_object,
            tools_from_object_with_properties,
        )

        assert callable(tools_from_object)
        assert callable(tools_from_object_with_properties)
        assert callable(tool)
        assert callable(tool_exclude)
