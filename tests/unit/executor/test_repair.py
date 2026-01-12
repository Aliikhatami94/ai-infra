"""Tests for repair code node.

Phase 1.2: Tests for repair.py - the surgical repair module.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ai_infra.executor.nodes.repair import (
    MAX_REPAIRS,
    _clean_repaired_code,
    _get_repair_prompt_template,
    repair_code_node,
    should_repair,
)

# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for repair module constants."""

    def test_max_repairs_is_reasonable(self):
        """MAX_REPAIRS should be a small positive number."""
        assert MAX_REPAIRS > 0
        assert MAX_REPAIRS <= 5  # Prevent runaway repairs


# =============================================================================
# Prompt Template Tests
# =============================================================================


class TestGetRepairPromptTemplate:
    """Tests for prompt template selection."""

    def test_python_syntax_error_uses_python_template(self):
        """Python syntax errors should use Python repair template."""
        template = _get_repair_prompt_template("syntax")
        assert "Python code" in template
        assert "```python" in template

    def test_indent_error_uses_python_template(self):
        """Indentation errors should use Python repair template."""
        template = _get_repair_prompt_template("indent")
        assert "Python code" in template

    def test_encoding_error_uses_python_template(self):
        """Encoding errors should use Python repair template."""
        template = _get_repair_prompt_template("encoding")
        assert "Python code" in template

    def test_json_error_uses_json_template(self):
        """JSON errors should use JSON repair template."""
        template = _get_repair_prompt_template("json")
        assert "JSON" in template
        assert "```json" in template
        assert "Python" not in template

    def test_yaml_error_uses_yaml_template(self):
        """YAML errors should use YAML repair template."""
        template = _get_repair_prompt_template("yaml")
        assert "YAML" in template
        assert "```yaml" in template
        assert "Python" not in template

    def test_unknown_error_type_defaults_to_python(self):
        """Unknown error types should default to Python template."""
        template = _get_repair_prompt_template("unknown_error")
        assert "Python code" in template


# =============================================================================
# Code Cleaning Tests
# =============================================================================


class TestCleanRepairedCode:
    """Tests for cleaning LLM responses."""

    def test_clean_code_without_fences(self):
        """Code without fences should be returned as-is (stripped)."""
        code = "def add(a, b):\n    return a + b"
        result = _clean_repaired_code(code)
        assert result == code

    def test_removes_python_code_fence(self):
        """Python code fences should be removed."""
        code = "```python\ndef add(a, b):\n    return a + b\n```"
        result = _clean_repaired_code(code)
        assert result == "def add(a, b):\n    return a + b"
        assert "```" not in result

    def test_removes_json_code_fence(self):
        """JSON code fences should be removed."""
        code = '```json\n{"key": "value"}\n```'
        result = _clean_repaired_code(code)
        assert result == '{"key": "value"}'
        assert "```" not in result

    def test_removes_yaml_code_fence(self):
        """YAML code fences should be removed."""
        code = "```yaml\nkey: value\n```"
        result = _clean_repaired_code(code)
        assert result == "key: value"

    def test_removes_generic_code_fence(self):
        """Generic code fences should be removed."""
        code = "```\nsome code\n```"
        result = _clean_repaired_code(code)
        assert result == "some code"

    def test_strips_whitespace(self):
        """Leading/trailing whitespace should be stripped."""
        code = "\n\n  def add(a, b):\n    return a + b  \n\n"
        result = _clean_repaired_code(code)
        assert result == "def add(a, b):\n    return a + b"

    def test_handles_empty_string(self):
        """Empty strings should return empty."""
        result = _clean_repaired_code("")
        assert result == ""

    def test_handles_whitespace_only(self):
        """Whitespace-only strings should return empty."""
        result = _clean_repaired_code("   \n  \t  ")
        assert result == ""

    def test_preserves_internal_code_fences(self):
        """Code fences inside the code should be preserved."""
        # This is edge case - if LLM returns code that includes a markdown block
        code = '```python\ncode = """```test```"""\n```'
        result = _clean_repaired_code(code)
        # The outer fence is removed, inner content preserved
        assert '"""```test```"""' in result


# =============================================================================
# should_repair Function Tests
# =============================================================================


class TestShouldRepair:
    """Tests for should_repair helper function."""

    def test_should_repair_when_needs_repair_and_under_limit(self):
        """Should return True when repair is needed and under limit."""
        state = {"needs_repair": True, "repair_count": 0}
        assert should_repair(state) is True

    def test_should_repair_at_limit_minus_one(self):
        """Should return True at MAX_REPAIRS - 1."""
        state = {"needs_repair": True, "repair_count": MAX_REPAIRS - 1}
        assert should_repair(state) is True

    def test_should_not_repair_at_limit(self):
        """Should return False when at MAX_REPAIRS."""
        state = {"needs_repair": True, "repair_count": MAX_REPAIRS}
        assert should_repair(state) is False

    def test_should_not_repair_over_limit(self):
        """Should return False when over MAX_REPAIRS."""
        state = {"needs_repair": True, "repair_count": MAX_REPAIRS + 5}
        assert should_repair(state) is False

    def test_should_not_repair_when_not_needed(self):
        """Should return False when needs_repair is False."""
        state = {"needs_repair": False, "repair_count": 0}
        assert should_repair(state) is False

    def test_should_not_repair_with_missing_needs_repair(self):
        """Should return False when needs_repair is missing."""
        state = {"repair_count": 0}
        assert should_repair(state) is False

    def test_should_repair_with_missing_repair_count(self):
        """Should return True when repair_count is missing but needs_repair is True."""
        state = {"needs_repair": True}
        assert should_repair(state) is True


# =============================================================================
# Repair Node Tests
# =============================================================================


class TestRepairCodeNode:
    """Tests for repair_code_node function."""

    @pytest.mark.asyncio
    async def test_max_repairs_exceeded_sets_error(self):
        """When max repairs exceeded, should set error and not repair."""
        state = {
            "validation_errors": {"test.py": {"error_type": "syntax"}},
            "generated_code": {"test.py": "broken code"},
            "repair_count": MAX_REPAIRS,
        }

        result = await repair_code_node(state)

        assert result["needs_repair"] is False
        assert result["validated"] is False
        assert result["error"] is not None
        assert result["error"]["error_type"] == "validation"
        assert "Failed to repair" in result["error"]["message"]
        assert result["error"]["recoverable"] is True

    @pytest.mark.asyncio
    async def test_no_errors_to_repair(self):
        """When no validation errors, should just clear needs_repair."""
        state = {
            "validation_errors": {},
            "generated_code": {},
            "repair_count": 0,
        }

        result = await repair_code_node(state)

        assert result["needs_repair"] is False
        assert "error" not in result or result.get("error") is None

    @pytest.mark.asyncio
    async def test_no_agent_available_sets_error(self):
        """When no agent is available, should set unrecoverable error."""
        state = {
            "validation_errors": {"test.py": {"error_type": "syntax"}},
            "generated_code": {"test.py": "broken code"},
            "repair_count": 0,
        }

        result = await repair_code_node(state, agent=None)

        assert result["needs_repair"] is False
        assert result["validated"] is False
        assert result["error"] is not None
        assert result["error"]["recoverable"] is False

    @pytest.mark.asyncio
    async def test_successful_repair_updates_code(self):
        """Successful repair should update generated_code."""
        mock_agent = AsyncMock()
        mock_agent.arun.return_value = "def fixed():\n    pass"

        state = {
            "validation_errors": {
                "test.py": {
                    "error_type": "syntax",
                    "error_message": "missing colon",
                    "error_line": 1,
                }
            },
            "generated_code": {"test.py": "def broken()\n    pass"},
            "repair_count": 0,
        }

        result = await repair_code_node(state, agent=mock_agent)

        assert result["generated_code"]["test.py"] == "def fixed():\n    pass"
        assert result["repair_count"] == 1
        assert result["needs_repair"] is False
        assert result["validation_errors"] == {}
        mock_agent.arun.assert_called_once()

    @pytest.mark.asyncio
    async def test_repair_cleans_code_fences(self):
        """Repair should clean markdown code fences from LLM response."""
        mock_agent = AsyncMock()
        mock_agent.arun.return_value = "```python\ndef fixed():\n    pass\n```"

        state = {
            "validation_errors": {
                "test.py": {
                    "error_type": "syntax",
                    "error_message": "missing colon",
                    "error_line": 1,
                }
            },
            "generated_code": {"test.py": "def broken()\n    pass"},
            "repair_count": 0,
        }

        result = await repair_code_node(state, agent=mock_agent)

        # Code fences should be removed
        assert "```" not in result["generated_code"]["test.py"]
        assert result["generated_code"]["test.py"] == "def fixed():\n    pass"

    @pytest.mark.asyncio
    async def test_repair_increments_count(self):
        """Each repair attempt should increment repair_count."""
        mock_agent = AsyncMock()
        mock_agent.arun.return_value = "fixed code"

        state = {
            "validation_errors": {"test.py": {"error_type": "syntax"}},
            "generated_code": {"test.py": "broken"},
            "repair_count": 1,
        }

        result = await repair_code_node(state, agent=mock_agent)

        assert result["repair_count"] == 2

    @pytest.mark.asyncio
    async def test_repair_multiple_files(self):
        """Should repair multiple files in one call."""
        mock_agent = AsyncMock()
        mock_agent.arun.side_effect = ["fixed code 1", "fixed code 2"]

        state = {
            "validation_errors": {
                "file1.py": {"error_type": "syntax", "error_message": "error 1"},
                "file2.py": {"error_type": "indent", "error_message": "error 2"},
            },
            "generated_code": {
                "file1.py": "broken 1",
                "file2.py": "broken 2",
            },
            "repair_count": 0,
        }

        result = await repair_code_node(state, agent=mock_agent)

        assert result["generated_code"]["file1.py"] == "fixed code 1"
        assert result["generated_code"]["file2.py"] == "fixed code 2"
        assert mock_agent.arun.call_count == 2

    @pytest.mark.asyncio
    async def test_repair_preserves_unaffected_files(self):
        """Files without errors should not be modified."""
        mock_agent = AsyncMock()
        mock_agent.arun.return_value = "fixed code"

        state = {
            "validation_errors": {"broken.py": {"error_type": "syntax"}},
            "generated_code": {
                "broken.py": "broken code",
                "good.py": "good code",  # Should not be changed
            },
            "repair_count": 0,
        }

        result = await repair_code_node(state, agent=mock_agent)

        assert result["generated_code"]["good.py"] == "good code"
        assert result["generated_code"]["broken.py"] == "fixed code"

    @pytest.mark.asyncio
    async def test_repair_skips_missing_original_code(self):
        """Should skip files with no original code."""
        mock_agent = AsyncMock()

        state = {
            "validation_errors": {"missing.py": {"error_type": "syntax"}},
            "generated_code": {},  # No original code
            "repair_count": 0,
        }

        result = await repair_code_node(state, agent=mock_agent)

        # Should not call agent for missing file
        mock_agent.arun.assert_not_called()
        assert result["repair_count"] == 1

    @pytest.mark.asyncio
    async def test_repair_handles_agent_error(self):
        """Should handle agent errors gracefully."""
        mock_agent = AsyncMock()
        mock_agent.arun.side_effect = Exception("LLM error")

        state = {
            "validation_errors": {
                "test.py": {"error_type": "syntax", "error_message": "missing colon"}
            },
            "generated_code": {"test.py": "original code"},
            "repair_count": 0,
        }

        result = await repair_code_node(state, agent=mock_agent)

        # Should keep original code on failure
        assert result["generated_code"]["test.py"] == "original code"
        assert result["repair_results"]["test.py"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_repair_tracks_results(self):
        """Should track repair results for each file."""
        mock_agent = AsyncMock()
        mock_agent.arun.return_value = "fixed code"

        state = {
            "validation_errors": {
                "test.py": {"error_type": "syntax", "error_message": "missing colon"}
            },
            "generated_code": {"test.py": "broken code"},
            "repair_count": 0,
        }

        result = await repair_code_node(state, agent=mock_agent)

        assert "repair_results" in result
        assert result["repair_results"]["test.py"]["status"] == "repaired"
        assert "missing colon" in result["repair_results"]["test.py"]["original_error"]

    @pytest.mark.asyncio
    async def test_repair_uses_correct_prompt_for_json(self):
        """Should use JSON-specific prompt for JSON errors."""
        mock_agent = AsyncMock()
        mock_agent.arun.return_value = '{"fixed": true}'

        state = {
            "validation_errors": {
                "config.json": {
                    "error_type": "json",
                    "error_message": "Expecting property name",
                }
            },
            "generated_code": {"config.json": '{"broken": }'},
            "repair_count": 0,
        }

        await repair_code_node(state, agent=mock_agent)

        # Check the prompt passed to agent
        call_args = mock_agent.arun.call_args[0][0]
        assert "JSON" in call_args

    @pytest.mark.asyncio
    async def test_repair_uses_correct_prompt_for_yaml(self):
        """Should use YAML-specific prompt for YAML errors."""
        mock_agent = AsyncMock()
        mock_agent.arun.return_value = "key: value"

        state = {
            "validation_errors": {
                "config.yaml": {
                    "error_type": "yaml",
                    "error_message": "mapping values not allowed here",
                }
            },
            "generated_code": {"config.yaml": "key:bad:syntax"},
            "repair_count": 0,
        }

        await repair_code_node(state, agent=mock_agent)

        # Check the prompt passed to agent
        call_args = mock_agent.arun.call_args[0][0]
        assert "YAML" in call_args


# =============================================================================
# Route Function Tests
# =============================================================================


class TestRouteAfterValidate:
    """Tests for route_after_validate function."""

    def test_validated_routes_to_write_files(self):
        """When validated is True, should route to write_files."""
        from ai_infra.executor.edges.routes import route_after_validate

        state = {"validated": True}
        assert route_after_validate(state) == "write_files"

    def test_needs_repair_under_limit_routes_to_repair(self):
        """When needs_repair and under limit, should route to repair_code."""
        from ai_infra.executor.edges.routes import route_after_validate

        state = {"validated": False, "needs_repair": True, "repair_count": 0}
        assert route_after_validate(state) == "repair_code"

    def test_needs_repair_at_limit_routes_to_failure(self):
        """When needs_repair and at limit, should route to handle_failure."""
        from ai_infra.executor.edges.routes import route_after_validate

        state = {
            "validated": False,
            "needs_repair": True,
            "repair_count": MAX_REPAIRS,
        }
        assert route_after_validate(state) == "handle_failure"

    def test_default_routes_to_write_files(self):
        """When no specific flags, should default to write_files."""
        from ai_infra.executor.edges.routes import route_after_validate

        state = {}
        assert route_after_validate(state) == "write_files"


class TestRouteAfterRepair:
    """Tests for route_after_repair function."""

    def test_no_error_routes_to_validate(self):
        """When no error, should route back to validate_code."""
        from ai_infra.executor.edges.routes import route_after_repair

        state = {}
        assert route_after_repair(state) == "validate_code"

    def test_with_error_routes_to_failure(self):
        """When error is present, should route to handle_failure."""
        from ai_infra.executor.edges.routes import route_after_repair

        state = {"error": {"message": "repair failed"}}
        assert route_after_repair(state) == "handle_failure"
