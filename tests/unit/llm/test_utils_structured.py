"""Tests for LLM structured output utilities (llm/utils/structured.py).

This module provides comprehensive tests for structured output parsing:
- build_structured_messages()
- validate_or_raise()
- coerce_from_text_or_fragment()
- coerce_structured_result()
- _extract_json_candidate()
- structured_mode_call_sync/async

Phase 0.2 of the ai-infra v1.0.0 release plan.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from pydantic import BaseModel

from ai_infra.llm.utils.structured import (
    _extract_json_candidate,
    build_structured_messages,
    coerce_from_text_or_fragment,
    coerce_structured_result,
    is_pydantic_schema,
    structured_mode_call_async,
    structured_mode_call_sync,
    validate_or_raise,
)

# =============================================================================
# Test Schemas
# =============================================================================


class SimpleSchema(BaseModel):
    """Simple schema for testing."""

    name: str
    value: int


class NestedSchema(BaseModel):
    """Schema with nested object."""

    title: str
    details: SimpleSchema


class OptionalSchema(BaseModel):
    """Schema with optional fields."""

    required_field: str
    optional_field: str | None = None
    default_field: int = 42


class ListSchema(BaseModel):
    """Schema with list field."""

    items: list[str]
    count: int


# =============================================================================
# TEST: build_structured_messages()
# =============================================================================


class TestBuildStructuredMessages:
    """Test build_structured_messages function."""

    def test_basic_pydantic_schema(self):
        """Test message building with Pydantic model."""
        messages = build_structured_messages(
            schema=SimpleSchema,
            user_msg="Give me a name and value",
        )

        assert len(messages) == 2
        # First is system message
        assert messages[0].type == "system"
        assert "JSON" in messages[0].content
        # Second is user message
        assert messages[1].type == "human"
        assert messages[1].content == "Give me a name and value"

    def test_with_system_preamble(self):
        """Test message building with custom system preamble."""
        messages = build_structured_messages(
            schema=SimpleSchema,
            user_msg="Test prompt",
            system_preamble="You are a helpful assistant.",
        )

        assert len(messages) == 2
        system_content = messages[0].content
        assert "You are a helpful assistant." in system_content
        assert "JSON" in system_content

    def test_with_dict_schema(self):
        """Test message building with dict schema instead of Pydantic."""
        dict_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }

        messages = build_structured_messages(
            schema=dict_schema,
            user_msg="Get user info",
        )

        assert len(messages) == 2
        system_content = messages[0].content
        assert "JSON" in system_content
        assert "Schema:" in system_content

    def test_forbid_prose_true(self):
        """Test forbid_prose=True adds instruction."""
        messages = build_structured_messages(
            schema=SimpleSchema,
            user_msg="Test",
            forbid_prose=True,
        )

        system_content = messages[0].content
        assert "Do NOT include any prose" in system_content

    def test_forbid_prose_false(self):
        """Test forbid_prose=False omits instruction."""
        messages = build_structured_messages(
            schema=SimpleSchema,
            user_msg="Test",
            forbid_prose=False,
        )

        system_content = messages[0].content
        assert "Do NOT include any prose" not in system_content


# =============================================================================
# TEST: validate_or_raise()
# =============================================================================


class TestValidateOrRaise:
    """Test validate_or_raise function."""

    def test_valid_json_pydantic_schema(self):
        """Test validation with valid JSON against Pydantic schema."""
        raw_json = '{"name": "test", "value": 42}'
        result = validate_or_raise(SimpleSchema, raw_json)

        assert isinstance(result, SimpleSchema)
        assert result.name == "test"
        assert result.value == 42

    def test_valid_json_dict_schema(self):
        """Test validation with valid JSON against dict schema."""
        dict_schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        raw_json = '{"name": "test"}'
        result = validate_or_raise(dict_schema, raw_json)

        assert isinstance(result, dict)
        assert result["name"] == "test"

    def test_invalid_json_raises_error(self):
        """Test validation raises on invalid JSON."""
        with pytest.raises(json.JSONDecodeError):
            validate_or_raise(SimpleSchema, "not valid json")

    def test_missing_required_field_raises_error(self):
        """Test validation raises when required field is missing."""
        from pydantic import ValidationError

        # Missing 'value' field
        raw_json = '{"name": "test"}'
        with pytest.raises(ValidationError):
            validate_or_raise(SimpleSchema, raw_json)

    def test_wrong_type_raises_error(self):
        """Test validation raises on wrong field type."""
        from pydantic import ValidationError

        # 'value' should be int, not string
        raw_json = '{"name": "test", "value": "not an int"}'
        with pytest.raises(ValidationError):
            validate_or_raise(SimpleSchema, raw_json)

    def test_nested_schema_validation(self):
        """Test validation with nested schema."""
        raw_json = '{"title": "Main", "details": {"name": "nested", "value": 100}}'
        result = validate_or_raise(NestedSchema, raw_json)

        assert isinstance(result, NestedSchema)
        assert result.title == "Main"
        assert result.details.name == "nested"
        assert result.details.value == 100

    def test_optional_fields(self):
        """Test validation with optional fields."""
        raw_json = '{"required_field": "test"}'
        result = validate_or_raise(OptionalSchema, raw_json)

        assert result.required_field == "test"
        assert result.optional_field is None
        assert result.default_field == 42

    def test_list_schema_validation(self):
        """Test validation with list fields."""
        raw_json = '{"items": ["a", "b", "c"], "count": 3}'
        result = validate_or_raise(ListSchema, raw_json)

        assert result.items == ["a", "b", "c"]
        assert result.count == 3


# =============================================================================
# TEST: _extract_json_candidate()
# =============================================================================


class TestExtractJsonCandidate:
    """Test _extract_json_candidate function."""

    def test_pure_json_object(self):
        """Test extraction from pure JSON object."""
        text = '{"name": "test", "value": 42}'
        result = _extract_json_candidate(text)

        assert result == {"name": "test", "value": 42}

    def test_pure_json_array(self):
        """Test extraction from pure JSON array."""
        text = "[1, 2, 3]"
        result = _extract_json_candidate(text)

        assert result == [1, 2, 3]

    def test_json_in_markdown_code_fence(self):
        """Test extraction from markdown code fence."""
        text = """Here's the result:
```json
{"name": "test", "value": 42}
```
Hope that helps!"""
        result = _extract_json_candidate(text)

        assert result == {"name": "test", "value": 42}

    def test_json_in_plain_code_fence(self):
        """Test extraction from plain code fence (no language)."""
        text = """Result:
```
{"name": "test", "value": 42}
```"""
        result = _extract_json_candidate(text)

        assert result == {"name": "test", "value": 42}

    def test_json_with_surrounding_prose(self):
        """Test extraction from text with surrounding prose."""
        text = 'The answer is {"name": "test", "value": 42} as requested.'
        result = _extract_json_candidate(text)

        assert result == {"name": "test", "value": 42}

    def test_json_with_trailing_comma(self):
        """Test extraction handles trailing comma."""
        text = '{"name": "test", "value": 42,}'
        result = _extract_json_candidate(text)

        assert result == {"name": "test", "value": 42}

    def test_nested_json_objects(self):
        """Test extraction of nested JSON."""
        text = '{"outer": {"inner": "value"}, "list": [1, 2, 3]}'
        result = _extract_json_candidate(text)

        assert result == {"outer": {"inner": "value"}, "list": [1, 2, 3]}

    def test_empty_string_returns_none(self):
        """Test empty string returns None."""
        assert _extract_json_candidate("") is None
        assert _extract_json_candidate("   ") is None

    def test_no_json_returns_none(self):
        """Test text without JSON returns None."""
        text = "This is just plain text without any JSON."
        assert _extract_json_candidate(text) is None

    def test_malformed_json_returns_none(self):
        """Test malformed JSON returns None."""
        text = '{"name": "test", "value": }'  # Invalid JSON
        assert _extract_json_candidate(text) is None

    def test_json_with_escaped_strings(self):
        """Test JSON with escaped characters in strings."""
        text = '{"message": "Hello \\"world\\"", "path": "C:\\\\Users"}'
        result = _extract_json_candidate(text)

        assert result["message"] == 'Hello "world"'
        assert result["path"] == "C:\\Users"

    def test_json_array_in_markdown(self):
        """Test array extraction from markdown."""
        text = """```json
["item1", "item2", "item3"]
```"""
        result = _extract_json_candidate(text)

        assert result == ["item1", "item2", "item3"]


# =============================================================================
# TEST: coerce_from_text_or_fragment()
# =============================================================================


class TestCoerceFromTextOrFragment:
    """Test coerce_from_text_or_fragment function."""

    def test_strict_valid_json(self):
        """Test strict validation path with valid JSON."""
        text = '{"name": "test", "value": 42}'
        result = coerce_from_text_or_fragment(SimpleSchema, text)

        assert isinstance(result, SimpleSchema)
        assert result.name == "test"

    def test_fragment_extraction_fallback(self):
        """Test fallback to fragment extraction."""
        text = 'Here is the answer: {"name": "test", "value": 42} for you.'
        result = coerce_from_text_or_fragment(SimpleSchema, text)

        assert isinstance(result, SimpleSchema)
        assert result.name == "test"

    def test_markdown_json_extraction(self):
        """Test extraction from markdown code block."""
        text = """```json
{"name": "test", "value": 42}
```"""
        result = coerce_from_text_or_fragment(SimpleSchema, text)

        assert isinstance(result, SimpleSchema)

    def test_invalid_json_returns_none(self):
        """Test invalid JSON returns None."""
        text = "This has no valid JSON at all."
        result = coerce_from_text_or_fragment(SimpleSchema, text)

        assert result is None

    def test_json_not_matching_schema_returns_none(self):
        """Test JSON that doesn't match schema returns None."""
        # Missing required 'value' field
        text = '{"name": "test"}'
        result = coerce_from_text_or_fragment(SimpleSchema, text)

        assert result is None

    def test_dict_schema(self):
        """Test with dict schema instead of Pydantic."""
        dict_schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        text = '{"x": 10}'
        result = coerce_from_text_or_fragment(dict_schema, text)

        assert result == {"x": 10}


# =============================================================================
# TEST: coerce_structured_result()
# =============================================================================


class TestCoerceStructuredResult:
    """Test coerce_structured_result function."""

    def test_already_correct_type(self):
        """Test input already of correct type returns as-is."""
        obj = SimpleSchema(name="test", value=42)
        result = coerce_structured_result(SimpleSchema, obj)

        assert result is obj

    def test_dict_input(self):
        """Test dict input is validated to schema."""
        data = {"name": "test", "value": 42}
        result = coerce_structured_result(SimpleSchema, data)

        assert isinstance(result, SimpleSchema)
        assert result.name == "test"

    def test_aimessage_like_with_content(self):
        """Test AIMessage-like object with content attribute."""
        mock_response = Mock()
        mock_response.content = '{"name": "test", "value": 42}'

        result = coerce_structured_result(SimpleSchema, mock_response)

        assert isinstance(result, SimpleSchema)
        assert result.name == "test"

    def test_string_input(self):
        """Test string input is parsed."""
        text = '{"name": "test", "value": 42}'
        result = coerce_structured_result(SimpleSchema, text)

        assert isinstance(result, SimpleSchema)

    def test_string_with_prose(self):
        """Test string with surrounding prose."""
        text = 'Here is the answer: {"name": "test", "value": 42}'
        result = coerce_structured_result(SimpleSchema, text)

        assert isinstance(result, SimpleSchema)

    def test_invalid_input_raises_error(self):
        """Test invalid input raises ValueError."""
        mock_response = Mock()
        mock_response.content = "No JSON here at all!"

        with pytest.raises(ValueError, match="Could not coerce"):
            coerce_structured_result(SimpleSchema, mock_response)

    def test_empty_content_raises_error(self):
        """Test empty content raises ValueError."""
        mock_response = Mock()
        mock_response.content = ""

        with pytest.raises(ValueError, match="Could not coerce"):
            coerce_structured_result(SimpleSchema, mock_response)

    def test_markdown_wrapped_json(self):
        """Test markdown-wrapped JSON in response."""
        mock_response = Mock()
        mock_response.content = """```json
{"name": "test", "value": 42}
```"""

        result = coerce_structured_result(SimpleSchema, mock_response)

        assert isinstance(result, SimpleSchema)
        assert result.name == "test"


# =============================================================================
# TEST: is_pydantic_schema()
# =============================================================================


class TestIsPydanticSchema:
    """Test is_pydantic_schema function."""

    def test_pydantic_model_class(self):
        """Test returns True for Pydantic model class."""
        assert is_pydantic_schema(SimpleSchema) is True
        assert is_pydantic_schema(NestedSchema) is True

    def test_pydantic_instance(self):
        """Test returns False for Pydantic instance (not class)."""
        obj = SimpleSchema(name="test", value=42)
        assert is_pydantic_schema(obj) is False

    def test_dict_schema(self):
        """Test returns False for dict schema."""
        dict_schema = {"type": "object"}
        assert is_pydantic_schema(dict_schema) is False

    def test_other_types(self):
        """Test returns False for other types."""
        assert is_pydantic_schema(str) is False
        assert is_pydantic_schema(int) is False
        assert is_pydantic_schema(list) is False


# =============================================================================
# TEST: structured_mode_call_sync/async
# =============================================================================


class TestStructuredModeCallSync:
    """Test structured_mode_call_sync function."""

    def test_sync_structured_mode(self):
        """Test sync structured mode call."""
        mock_model = MagicMock()
        mock_model.invoke.return_value = Mock(content='{"name": "test", "value": 42}')

        def with_structured_output(provider, model, schema, method, **kwargs):
            return mock_model

        result = structured_mode_call_sync(
            with_structured_output,
            provider="openai",
            model_name="gpt-4o",
            schema=SimpleSchema,
            messages=[],
            model_kwargs={},
        )

        assert isinstance(result, SimpleSchema)
        assert result.name == "test"

    def test_sync_invalid_json_raises(self):
        """Test sync call raises on invalid JSON."""
        mock_model = MagicMock()
        mock_model.invoke.return_value = Mock(content="not valid json")

        def with_structured_output(provider, model, schema, method, **kwargs):
            return mock_model

        with pytest.raises(json.JSONDecodeError):
            structured_mode_call_sync(
                with_structured_output,
                provider="openai",
                model_name="gpt-4o",
                schema=SimpleSchema,
                messages=[],
                model_kwargs={},
            )


class TestStructuredModeCallAsync:
    """Test structured_mode_call_async function."""

    @pytest.mark.asyncio
    async def test_async_structured_mode(self):
        """Test async structured mode call."""
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=Mock(content='{"name": "test", "value": 42}'))

        def with_structured_output(provider, model, schema, method, **kwargs):
            return mock_model

        result = await structured_mode_call_async(
            with_structured_output,
            provider="openai",
            model_name="gpt-4o",
            schema=SimpleSchema,
            messages=[],
            model_kwargs={},
        )

        assert isinstance(result, SimpleSchema)
        assert result.name == "test"

    @pytest.mark.asyncio
    async def test_async_invalid_json_raises(self):
        """Test async call raises on invalid JSON."""
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=Mock(content="not valid json"))

        def with_structured_output(provider, model, schema, method, **kwargs):
            return mock_model

        with pytest.raises(json.JSONDecodeError):
            await structured_mode_call_async(
                with_structured_output,
                provider="openai",
                model_name="gpt-4o",
                schema=SimpleSchema,
                messages=[],
                model_kwargs={},
            )


# =============================================================================
# EDGE CASES
# =============================================================================


class TestStructuredEdgeCases:
    """Test edge cases for structured output utilities."""

    def test_unicode_in_json(self):
        """Test handling of unicode characters in JSON."""
        text = '{"name": "日本語テスト", "value": 42}'
        result = coerce_from_text_or_fragment(SimpleSchema, text)

        assert result.name == "日本語テスト"

    def test_very_long_json(self):
        """Test handling of very long JSON strings."""
        long_name = "x" * 10000
        text = f'{{"name": "{long_name}", "value": 42}}'
        result = coerce_from_text_or_fragment(SimpleSchema, text)

        assert result.name == long_name

    def test_json_with_special_chars(self):
        """Test JSON with special characters."""
        text = '{"name": "test\\nwith\\nnewlines", "value": 42}'
        result = _extract_json_candidate(text)

        assert result["name"] == "test\nwith\nnewlines"

    def test_deeply_nested_json(self):
        """Test deeply nested JSON extraction."""

        class DeepSchema(BaseModel):
            level1: dict[str, Any]

        text = '{"level1": {"level2": {"level3": {"level4": "deep"}}}}'
        result = coerce_from_text_or_fragment(DeepSchema, text)

        assert result.level1["level2"]["level3"]["level4"] == "deep"

    def test_multiple_json_objects_takes_first(self):
        """Test multiple JSON objects in text takes first one."""
        text = '{"name": "first", "value": 1} and {"name": "second", "value": 2}'
        result = _extract_json_candidate(text)

        assert result["name"] == "first"
        assert result["value"] == 1
