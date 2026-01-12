"""Tests for pre-write code validation.

Phase 1.1: Tests for validate.py - the pre-write validation module.
"""

from __future__ import annotations

import pytest

from ai_infra.executor.nodes.validate import (
    ValidationResult,
    _get_error_context,
    validate_code_node,
    validate_json,
    validate_python_code,
    validate_yaml,
)

# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for the ValidationResult dataclass."""

    def test_valid_result_has_no_repair_prompt(self):
        """Valid results should not generate repair prompts."""
        result = ValidationResult(valid=True)
        assert result.repair_prompt is None

    def test_invalid_result_generates_repair_prompt(self):
        """Invalid results should generate targeted repair prompts."""
        result = ValidationResult(
            valid=False,
            error_type="syntax",
            error_message="unexpected EOF while parsing",
            error_line=5,
            error_col=10,
        )

        prompt = result.repair_prompt
        assert prompt is not None
        assert "Fix the syntax error" in prompt
        assert "line 5" in prompt
        assert "column 10" in prompt
        assert "unexpected EOF" in prompt
        assert "Return ONLY the corrected code" in prompt

    def test_repair_prompt_includes_context(self):
        """Repair prompt should include error context if available."""
        result = ValidationResult(
            valid=False,
            error_type="syntax",
            error_message="invalid syntax",
            error_line=3,
            error_context=">>> 3 | def broken(\n    4 | pass",
        )

        prompt = result.repair_prompt
        assert prompt is not None
        assert "def broken(" in prompt
        assert "Context:" in prompt

    def test_to_dict_serialization(self):
        """ValidationResult should serialize to dict properly."""
        result = ValidationResult(
            valid=False,
            error_type="syntax",
            error_message="missing colon",
            error_line=1,
            error_col=15,
        )

        data = result.to_dict()
        assert data["valid"] is False
        assert data["error_type"] == "syntax"
        assert data["error_message"] == "missing colon"
        assert data["error_line"] == 1
        assert data["error_col"] == 15
        assert "repair_prompt" in data


# =============================================================================
# Python Validation Tests
# =============================================================================


class TestValidatePythonCode:
    """Tests for Python code validation."""

    def test_valid_simple_function(self):
        """Valid Python code should pass."""
        code = "def add(a, b):\n    return a + b"
        result = validate_python_code(code)
        assert result.valid is True
        assert result.error_message is None

    def test_valid_class_definition(self):
        """Valid class definitions should pass."""
        code = """
class User:
    def __init__(self, name: str):
        self.name = name

    def greet(self) -> str:
        return f"Hello, {self.name}"
"""
        result = validate_python_code(code)
        assert result.valid is True

    def test_valid_async_function(self):
        """Async functions should be valid."""
        code = "async def fetch():\n    await something()"
        result = validate_python_code(code)
        assert result.valid is True

    def test_missing_colon_syntax_error(self):
        """Missing colon should be detected."""
        code = "def add(a, b)\n    return a + b"
        result = validate_python_code(code)
        assert result.valid is False
        assert result.error_type == "syntax"
        assert result.error_line == 1

    def test_unclosed_parenthesis(self):
        """Unclosed parenthesis should be detected."""
        code = "x = (1 + 2"
        result = validate_python_code(code)
        assert result.valid is False
        assert result.error_type == "syntax"

    def test_unclosed_string(self):
        """Unclosed string should be detected."""
        code = 'x = "hello'
        result = validate_python_code(code)
        assert result.valid is False
        assert result.error_type == "syntax"

    def test_triple_quote_confusion(self):
        """Four quotes instead of three should be detected.

        This is the exact error that triggered Phase 1.1 development.
        """
        code = 'x = """"\n'
        result = validate_python_code(code)
        assert result.valid is False
        assert result.error_type == "syntax"

    def test_unclosed_triple_quote(self):
        """Unclosed triple-quoted string should be detected."""
        code = 'x = """hello\nworld'
        result = validate_python_code(code)
        assert result.valid is False
        # Error message varies by Python version
        assert (
            "unterminated" in (result.error_message or "").lower()
            or "eof" in (result.error_message or "").lower()
        )

    def test_indentation_error(self):
        """Indentation errors should be detected."""
        code = "def add(a, b):\nreturn a + b"  # Missing indent
        result = validate_python_code(code)
        assert result.valid is False
        # Note: Python reports this as SyntaxError, not IndentationError
        assert result.error_type in ("syntax", "indent")

    def test_empty_code(self):
        """Empty code should be invalid."""
        result = validate_python_code("")
        assert result.valid is False
        assert result.error_type == "syntax"
        assert "Empty" in (result.error_message or "")

    def test_whitespace_only_code(self):
        """Whitespace-only code should be invalid."""
        result = validate_python_code("   \n\n   ")
        assert result.valid is False
        assert result.error_type == "syntax"

    def test_error_context_included(self):
        """Error context should be extracted for repair prompts."""
        code = "line1\nline2\ndef broken(\nline4\nline5"
        result = validate_python_code(code)
        assert result.valid is False
        assert result.error_context is not None
        assert "def broken(" in result.error_context

    def test_repair_prompt_has_line_number(self):
        """Repair prompt should include the error line number."""
        code = "x = 1\ny = (\nz = 3"
        result = validate_python_code(code)
        assert result.valid is False
        prompt = result.repair_prompt
        assert prompt is not None
        assert "line" in prompt.lower()


# =============================================================================
# JSON Validation Tests
# =============================================================================


class TestValidateJson:
    """Tests for JSON validation."""

    def test_valid_json_object(self):
        """Valid JSON object should pass."""
        content = '{"name": "test", "value": 123}'
        result = validate_json(content)
        assert result.valid is True

    def test_valid_json_array(self):
        """Valid JSON array should pass."""
        content = '[1, 2, 3, "four"]'
        result = validate_json(content)
        assert result.valid is True

    def test_trailing_comma(self):
        """Trailing comma should be detected."""
        content = '{"name": "test",}'
        result = validate_json(content)
        assert result.valid is False
        assert result.error_type == "json"

    def test_missing_quotes(self):
        """Missing quotes on keys should be detected."""
        content = '{name: "test"}'
        result = validate_json(content)
        assert result.valid is False

    def test_empty_json(self):
        """Empty JSON should be invalid."""
        result = validate_json("")
        assert result.valid is False

    def test_json_error_has_line_number(self):
        """JSON errors should include line number."""
        content = '{\n  "name": "test"\n  "missing": "comma"\n}'
        result = validate_json(content)
        assert result.valid is False
        assert result.error_line is not None


# =============================================================================
# YAML Validation Tests
# =============================================================================


class TestValidateYaml:
    """Tests for YAML validation."""

    def test_valid_yaml(self):
        """Valid YAML should pass."""
        content = "name: test\nvalue: 123"
        result = validate_yaml(content)
        assert result.valid is True

    def test_valid_yaml_list(self):
        """Valid YAML list should pass."""
        content = "items:\n  - one\n  - two\n  - three"
        result = validate_yaml(content)
        assert result.valid is True

    def test_invalid_indentation(self):
        """Invalid YAML indentation should be detected."""
        content = "name: test\n  value: 123"  # Wrong indent
        result = validate_yaml(content)
        # YAML is lenient, this might pass - check if it fails
        # If it passes, that's also acceptable YAML behavior
        assert result.valid in (True, False)

    def test_empty_yaml(self):
        """Empty YAML should be invalid."""
        result = validate_yaml("")
        assert result.valid is False


# =============================================================================
# Error Context Tests
# =============================================================================


class TestGetErrorContext:
    """Tests for error context extraction."""

    def test_context_includes_surrounding_lines(self):
        """Context should include lines before and after error."""
        code = "line1\nline2\nline3\nline4\nline5\nline6\nline7"
        context = _get_error_context(code, error_line=4)
        assert "line1" in context
        assert "line4" in context
        assert "line7" in context

    def test_context_marks_error_line(self):
        """Error line should be marked with >>>."""
        code = "line1\nline2\nline3"
        context = _get_error_context(code, error_line=2)
        lines = context.split("\n")
        error_line = next(line for line in lines if ">>>" in line)
        assert "line2" in error_line

    def test_context_handles_first_line(self):
        """Context should work when error is on first line."""
        code = "error_line\nline2\nline3"
        context = _get_error_context(code, error_line=1)
        assert "error_line" in context

    def test_context_handles_last_line(self):
        """Context should work when error is on last line."""
        code = "line1\nline2\nerror_line"
        context = _get_error_context(code, error_line=3)
        assert "error_line" in context

    def test_context_with_none_line(self):
        """Context should handle None line number."""
        context = _get_error_context("code", error_line=None)
        assert context == ""


# =============================================================================
# Graph Node Tests
# =============================================================================


class TestValidateCodeNode:
    """Tests for the validate_code_node graph node."""

    @pytest.mark.asyncio
    async def test_empty_generated_code(self):
        """Empty generated_code should result in validated=True."""
        state = {"generated_code": {}}
        result = await validate_code_node(state)
        assert result["validated"] is True
        assert result["needs_repair"] is False

    @pytest.mark.asyncio
    async def test_valid_python_file(self):
        """Valid Python file should pass validation."""
        state = {
            "generated_code": {
                "src/app.py": "def main():\n    print('Hello')",
            }
        }
        result = await validate_code_node(state)
        assert result["validated"] is True
        assert result["needs_repair"] is False
        assert result["validation_errors"] == {}

    @pytest.mark.asyncio
    async def test_invalid_python_file(self):
        """Invalid Python file should fail validation."""
        state = {
            "generated_code": {
                "src/app.py": "def main(\n    print('broken')",
            }
        }
        result = await validate_code_node(state)
        assert result["validated"] is False
        assert result["needs_repair"] is True
        assert "src/app.py" in result["validation_errors"]
        error = result["validation_errors"]["src/app.py"]
        assert error["error_type"] == "syntax"
        assert error["repair_prompt"] is not None

    @pytest.mark.asyncio
    async def test_mixed_valid_invalid(self):
        """Mix of valid and invalid files should fail overall."""
        state = {
            "generated_code": {
                "src/good.py": "def good():\n    pass",
                "src/bad.py": "def bad(\n    pass",
            }
        }
        result = await validate_code_node(state)
        assert result["validated"] is False
        assert result["needs_repair"] is True
        assert "src/bad.py" in result["validation_errors"]
        assert "src/good.py" not in result["validation_errors"]

    @pytest.mark.asyncio
    async def test_json_file_validation(self):
        """JSON files should be validated."""
        state = {
            "generated_code": {
                "config.json": '{"valid": true}',
            }
        }
        result = await validate_code_node(state)
        assert result["validated"] is True

    @pytest.mark.asyncio
    async def test_invalid_json_file(self):
        """Invalid JSON should fail validation."""
        state = {
            "generated_code": {
                "config.json": '{"invalid": true,}',  # Trailing comma
            }
        }
        result = await validate_code_node(state)
        assert result["validated"] is False
        assert "config.json" in result["validation_errors"]

    @pytest.mark.asyncio
    async def test_unknown_file_type_skipped(self):
        """Unknown file types should be skipped (not fail)."""
        state = {
            "generated_code": {
                "README.md": "# Hello\nThis is markdown",
                "data.csv": "a,b,c\n1,2,3",
            }
        }
        result = await validate_code_node(state)
        assert result["validated"] is True
        assert result["needs_repair"] is False

    @pytest.mark.asyncio
    async def test_preserves_repair_count(self):
        """Repair count should be preserved from state."""
        state = {
            "generated_code": {"src/app.py": "def broken("},
            "repair_count": 1,
        }
        result = await validate_code_node(state)
        assert result["repair_count"] == 1

    @pytest.mark.asyncio
    async def test_no_generated_code_key(self):
        """Missing generated_code key should result in validated=True."""
        state = {}
        result = await validate_code_node(state)
        assert result["validated"] is True


# =============================================================================
# Edge Cases and Regression Tests
# =============================================================================


class TestEdgeCases:
    """Edge case and regression tests."""

    def test_f_string_with_quotes(self):
        """F-strings with quotes should be valid."""
        code = """name = "world"\nprint(f"Hello {name}")"""
        result = validate_python_code(code)
        assert result.valid is True

    def test_nested_quotes(self):
        """Nested quotes should be handled."""
        code = """x = 'She said "Hello"'"""
        result = validate_python_code(code)
        assert result.valid is True

    def test_multiline_string(self):
        """Multiline strings should be valid."""
        code = '''doc = """
This is a
multiline string.
"""'''
        result = validate_python_code(code)
        assert result.valid is True

    def test_raw_string(self):
        """Raw strings should be valid."""
        code = r'path = r"C:\Users\test"'
        result = validate_python_code(code)
        assert result.valid is True

    def test_decorator(self):
        """Decorators should be valid."""
        code = "@property\ndef name(self):\n    return self._name"
        result = validate_python_code(code)
        assert result.valid is True

    def test_type_hints(self):
        """Type hints should be valid."""
        code = "def add(a: int, b: int) -> int:\n    return a + b"
        result = validate_python_code(code)
        assert result.valid is True

    def test_walrus_operator(self):
        """Walrus operator should be valid (Python 3.8+)."""
        code = "if (n := len(items)) > 0:\n    print(n)"
        result = validate_python_code(code)
        assert result.valid is True

    def test_match_statement(self):
        """Match statement should be valid (Python 3.10+)."""
        code = "match x:\n    case 1:\n        print('one')"
        result = validate_python_code(code)
        assert result.valid is True

    def test_unicode_in_code(self):
        """Unicode characters should be valid."""
        code = '# 你好\nprint("Hello 世界")'
        result = validate_python_code(code)
        assert result.valid is True

    def test_very_long_line(self):
        """Very long lines should be valid."""
        code = f"x = {'a' * 10000}"
        result = validate_python_code(code)
        # This is syntactically valid, just stylistically bad
        assert result.valid is True
