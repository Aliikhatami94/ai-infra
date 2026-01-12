"""Integration tests for validate → repair → validate flow.

Phase 1.7: Tests for the complete validation and repair cycle.
These tests verify that validation errors are detected and repaired
before files are written to disk.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_infra.executor.edges.routes import (
    route_after_validate,
)
from ai_infra.executor.nodes.repair import (
    MAX_REPAIRS,
    repair_code_node,
    should_repair,
)
from ai_infra.executor.nodes.validate import (
    validate_code_node,
    validate_python_code,
)
from ai_infra.executor.todolist import TodoItem, TodoStatus

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_todo() -> TodoItem:
    """Create a sample todo item for testing."""
    return TodoItem(
        id=1,
        title="Implement feature",
        description="Add a new feature",
        status=TodoStatus.IN_PROGRESS,
    )


@pytest.fixture
def base_state(sample_todo: TodoItem) -> dict[str, Any]:
    """Create a base state for testing."""
    return {
        "roadmap_path": "ROADMAP.md",
        "todos": [sample_todo],
        "current_task": sample_todo,
        "context": "Test context",
        "prompt": "Test prompt",
        "agent_result": None,
        "files_modified": [],
        "verified": False,
        "last_checkpoint_sha": None,
        "error": None,
        "retry_count": 0,
        "completed_count": 0,
        "max_tasks": None,
        "should_continue": True,
        "interrupt_requested": False,
        "run_memory": {},
        "generated_code": {},
        "validation_errors": {},
        "needs_repair": False,
        "repair_count": 0,
    }


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create a mock agent for repair tests."""
    agent = MagicMock()
    agent.arun = AsyncMock()
    return agent


# =============================================================================
# Test: Full Validate → Repair → Validate Cycle
# =============================================================================


class TestValidateRepairCycle:
    """Tests for the complete validation and repair cycle."""

    @pytest.mark.asyncio
    async def test_valid_code_skips_repair(self, base_state: dict[str, Any]) -> None:
        """Valid code should pass validation and skip repair."""
        valid_code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        state = {
            **base_state,
            "generated_code": {"src/math.py": valid_code},
        }

        result = await validate_code_node(state)

        assert result["needs_repair"] is False
        assert result["validation_errors"] == {}
        assert route_after_validate(result) == "write_files"

    @pytest.mark.asyncio
    async def test_invalid_code_triggers_repair(self, base_state: dict[str, Any]) -> None:
        """Invalid code should trigger repair route."""
        invalid_code = "def broken(\n    pass"  # Missing closing paren
        state = {
            **base_state,
            "generated_code": {"src/broken.py": invalid_code},
        }

        result = await validate_code_node(state)

        assert result["needs_repair"] is True
        assert "src/broken.py" in result["validation_errors"]
        assert route_after_validate(result) == "repair_code"

    @pytest.mark.asyncio
    async def test_repair_then_revalidate_success(
        self, base_state: dict[str, Any], mock_agent: MagicMock
    ) -> None:
        """After successful repair, code should be revalidated."""
        # Start with invalid code
        invalid_code = "def broken("
        state = {
            **base_state,
            "generated_code": {"src/app.py": invalid_code},
            "validation_errors": {
                "src/app.py": {
                    "valid": False,
                    "error_type": "syntax",
                    "error_message": "unexpected EOF",
                    "error_line": 1,
                    "error_col": None,
                    "error_context": None,
                    "repair_prompt": "Fix the syntax error on line 1",
                },
            },
            "needs_repair": True,
            "repair_count": 0,
        }

        # Mock agent returns fixed code
        fixed_code = "def fixed():\n    pass\n"
        mock_agent.arun.return_value = fixed_code

        repair_result = await repair_code_node(state, agent=mock_agent)

        # Verify repair updated the code (cleaned code may strip trailing newline)
        repaired_code = repair_result["generated_code"]["src/app.py"]
        assert repaired_code.strip() == fixed_code.strip()
        assert repair_result["repair_count"] == 1

        # Now validate the repaired code
        validate_result = await validate_code_node(repair_result)

        assert validate_result["needs_repair"] is False
        assert validate_result["validation_errors"] == {}
        assert route_after_validate(validate_result) == "write_files"

    @pytest.mark.asyncio
    async def test_max_repairs_triggers_failure(
        self, base_state: dict[str, Any], mock_agent: MagicMock
    ) -> None:
        """Exceeding max repairs should trigger handle_failure."""
        state = {
            **base_state,
            "generated_code": {"src/broken.py": "def broken("},
            "validation_errors": {
                "src/broken.py": {
                    "valid": False,
                    "error_type": "syntax",
                    "error_message": "unexpected EOF",
                    "error_line": 1,
                    "error_col": None,
                    "error_context": None,
                    "repair_prompt": "Fix the syntax error",
                },
            },
            "needs_repair": True,
            "repair_count": MAX_REPAIRS,  # Already at max
        }

        result = await repair_code_node(state, agent=mock_agent)

        # Should not call agent when at max repairs
        mock_agent.arun.assert_not_called()

        # Should set error for escalation
        assert result["error"] is not None
        assert "repair" in result["error"]["message"].lower()
        assert result["error"]["recoverable"] is True

    @pytest.mark.asyncio
    async def test_repair_route_on_still_invalid(
        self, base_state: dict[str, Any], mock_agent: MagicMock
    ) -> None:
        """Repair that still produces invalid code should route back to repair."""
        # Start with invalid code
        state = {
            **base_state,
            "generated_code": {"src/app.py": "def broken("},
            "validation_errors": {
                "src/app.py": {
                    "valid": False,
                    "error_type": "syntax",
                    "error_message": "unexpected EOF",
                    "error_line": 1,
                    "error_col": None,
                    "error_context": None,
                    "repair_prompt": "Fix the syntax error",
                },
            },
            "needs_repair": True,
            "repair_count": 0,
        }

        # Mock agent returns still-invalid code
        still_broken = "def also_broken("
        mock_agent.arun.return_value = still_broken

        repair_result = await repair_code_node(state, agent=mock_agent)

        # Validate the "repaired" code
        validate_result = await validate_code_node(repair_result)

        # Should need another repair
        assert validate_result["needs_repair"] is True
        assert should_repair(validate_result)
        assert route_after_validate(validate_result) == "repair_code"


# =============================================================================
# Test: Multiple Files Validation
# =============================================================================


class TestMultipleFilesValidation:
    """Tests for validating multiple files at once."""

    @pytest.mark.asyncio
    async def test_all_valid_files_pass(self, base_state: dict[str, Any]) -> None:
        """All valid files should pass validation."""
        state = {
            **base_state,
            "generated_code": {
                "src/utils.py": "def helper():\n    return 42\n",
                "src/main.py": "from utils import helper\n\nprint(helper())\n",
                "tests/test_main.py": "def test_helper():\n    assert True\n",
            },
        }

        result = await validate_code_node(state)

        assert result["needs_repair"] is False
        assert result["validation_errors"] == {}

    @pytest.mark.asyncio
    async def test_one_invalid_file_triggers_repair(self, base_state: dict[str, Any]) -> None:
        """One invalid file among valid ones should trigger repair."""
        state = {
            **base_state,
            "generated_code": {
                "src/utils.py": "def helper():\n    return 42\n",
                "src/broken.py": "def broken(",  # Invalid
                "src/main.py": "print('hello')\n",
            },
        }

        result = await validate_code_node(state)

        assert result["needs_repair"] is True
        assert "src/broken.py" in result["validation_errors"]
        assert "src/utils.py" not in result["validation_errors"]
        assert "src/main.py" not in result["validation_errors"]

    @pytest.mark.asyncio
    async def test_multiple_invalid_files_all_reported(self, base_state: dict[str, Any]) -> None:
        """Multiple invalid files should all be reported."""
        state = {
            **base_state,
            "generated_code": {
                "src/broken1.py": "def broken1(",
                "src/broken2.py": "class Broken:\n    def method(self)",  # Missing colon
                "src/valid.py": "x = 1\n",
            },
        }

        result = await validate_code_node(state)

        assert result["needs_repair"] is True
        assert len(result["validation_errors"]) == 2
        assert "src/broken1.py" in result["validation_errors"]
        assert "src/broken2.py" in result["validation_errors"]
        assert "src/valid.py" not in result["validation_errors"]


# =============================================================================
# Test: File Type Detection
# =============================================================================


class TestFileTypeValidation:
    """Tests for validating different file types."""

    @pytest.mark.asyncio
    async def test_json_file_validated_as_json(self, base_state: dict[str, Any]) -> None:
        """JSON files should be validated with JSON parser."""
        state = {
            **base_state,
            "generated_code": {
                "config.json": '{"key": "value", "number": 42}',
            },
        }

        result = await validate_code_node(state)

        assert result["needs_repair"] is False

    @pytest.mark.asyncio
    async def test_invalid_json_detected(self, base_state: dict[str, Any]) -> None:
        """Invalid JSON should be detected."""
        state = {
            **base_state,
            "generated_code": {
                "config.json": '{"key": "value",}',  # Trailing comma
            },
        }

        result = await validate_code_node(state)

        assert result["needs_repair"] is True
        assert "config.json" in result["validation_errors"]
        error = result["validation_errors"]["config.json"]
        assert error["error_type"] == "json"

    @pytest.mark.asyncio
    async def test_yaml_file_validated_as_yaml(self, base_state: dict[str, Any]) -> None:
        """YAML files should be validated with YAML parser."""
        state = {
            **base_state,
            "generated_code": {
                "config.yaml": "key: value\nnumber: 42\n",
            },
        }

        result = await validate_code_node(state)

        assert result["needs_repair"] is False

    @pytest.mark.asyncio
    async def test_yml_extension_handled(self, base_state: dict[str, Any]) -> None:
        """YML extension should also be validated as YAML."""
        state = {
            **base_state,
            "generated_code": {
                "config.yml": "items:\n  - one\n  - two\n",
            },
        }

        result = await validate_code_node(state)

        assert result["needs_repair"] is False

    @pytest.mark.asyncio
    async def test_markdown_files_skipped(self, base_state: dict[str, Any]) -> None:
        """Markdown files should not be validated (always pass)."""
        state = {
            **base_state,
            "generated_code": {
                "README.md": "# Header\n\nSome content",
            },
        }

        result = await validate_code_node(state)

        # Markdown is not validated for syntax
        assert result["needs_repair"] is False


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in the validate/repair flow."""

    @pytest.mark.asyncio
    async def test_empty_generated_code(self, base_state: dict[str, Any]) -> None:
        """Empty generated_code should pass (nothing to validate)."""
        state = {
            **base_state,
            "generated_code": {},
        }

        result = await validate_code_node(state)

        assert result["needs_repair"] is False
        assert result["validation_errors"] == {}

    @pytest.mark.asyncio
    async def test_whitespace_only_code_invalid(self, base_state: dict[str, Any]) -> None:
        """Whitespace-only Python code should be invalid."""
        state = {
            **base_state,
            "generated_code": {
                "src/empty.py": "   \n\n   \n",
            },
        }

        result = await validate_code_node(state)

        assert result["needs_repair"] is True
        assert "src/empty.py" in result["validation_errors"]

    @pytest.mark.asyncio
    async def test_repair_count_increments(
        self, base_state: dict[str, Any], mock_agent: MagicMock
    ) -> None:
        """Repair count should increment after each repair."""
        state = {
            **base_state,
            "generated_code": {"src/app.py": "def broken("},
            "validation_errors": {
                "src/app.py": {
                    "valid": False,
                    "error_type": "syntax",
                    "error_message": "unexpected EOF",
                    "error_line": 1,
                    "error_col": None,
                    "error_context": None,
                    "repair_prompt": "Fix the error",
                },
            },
            "needs_repair": True,
            "repair_count": 0,
        }

        mock_agent.arun.return_value = "def fixed():\n    pass\n"

        result = await repair_code_node(state, agent=mock_agent)

        assert result["repair_count"] == 1

        # Second repair
        result["needs_repair"] = True
        result["validation_errors"] = {
            "src/app.py": {
                "valid": False,
                "error_type": "syntax",
                "error_message": "still broken",
                "error_line": 1,
                "error_col": None,
                "error_context": None,
                "repair_prompt": "Fix the error",
            },
        }

        result = await repair_code_node(result, agent=mock_agent)

        assert result["repair_count"] == 2

    @pytest.mark.asyncio
    async def test_should_repair_helper(self, base_state: dict[str, Any]) -> None:
        """should_repair() should correctly check state."""
        # Should repair: needs_repair=True, under limit
        state = {**base_state, "needs_repair": True, "repair_count": 0}
        assert should_repair(state) is True

        # Should repair: at limit - 1
        state = {**base_state, "needs_repair": True, "repair_count": MAX_REPAIRS - 1}
        assert should_repair(state) is True

        # Should NOT repair: at limit
        state = {**base_state, "needs_repair": True, "repair_count": MAX_REPAIRS}
        assert should_repair(state) is False

        # Should NOT repair: needs_repair=False
        state = {**base_state, "needs_repair": False, "repair_count": 0}
        assert should_repair(state) is False


# =============================================================================
# Test: Repair Prompt Quality
# =============================================================================


class TestRepairPromptQuality:
    """Tests for repair prompt generation quality."""

    def test_syntax_error_prompt_includes_line(self) -> None:
        """Syntax error prompts should include line number."""
        code = "x = 1\ny = (\nz = 3"  # Line 2 has unclosed paren
        result = validate_python_code(code)

        assert result.valid is False
        prompt = result.repair_prompt
        assert prompt is not None
        assert "line" in prompt.lower()
        assert result.error_line is not None

    def test_indentation_error_prompt(self) -> None:
        """Indentation error prompts should be clear."""
        code = "def foo():\nreturn 1"  # Missing indent
        result = validate_python_code(code)

        assert result.valid is False
        assert (
            "indent" in (result.error_type or "").lower()
            or "syntax" in (result.error_type or "").lower()
        )

    def test_repair_prompt_includes_error_message(self) -> None:
        """Repair prompts should include the error message."""
        code = "def broken("
        result = validate_python_code(code)

        prompt = result.repair_prompt
        assert prompt is not None
        assert result.error_message is not None
        # Error message should be in prompt - check for actual message content
        assert "never closed" in prompt.lower() or "eof" in prompt.lower()


# =============================================================================
# Test: Validation Does Not Write Files
# =============================================================================


class TestValidationNoSideEffects:
    """Tests to ensure validation has no side effects."""

    @pytest.mark.asyncio
    async def test_validation_does_not_modify_filesystem(
        self, base_state: dict[str, Any], tmp_path
    ) -> None:
        """Validation should not create any files."""
        import os

        # Track files before validation
        files_before = set(os.listdir(tmp_path))

        state = {
            **base_state,
            "generated_code": {
                str(tmp_path / "should_not_exist.py"): "def foo():\n    pass\n",
            },
        }

        await validate_code_node(state)

        # Track files after validation
        files_after = set(os.listdir(tmp_path))

        # No new files should be created
        assert files_before == files_after
        assert not (tmp_path / "should_not_exist.py").exists()

    @pytest.mark.asyncio
    async def test_validation_is_idempotent(self, base_state: dict[str, Any]) -> None:
        """Multiple validations should produce same result."""
        state = {
            **base_state,
            "generated_code": {
                "src/app.py": "def valid():\n    return 42\n",
            },
        }

        result1 = await validate_code_node(state)
        result2 = await validate_code_node(state)
        result3 = await validate_code_node(state)

        assert result1["needs_repair"] == result2["needs_repair"] == result3["needs_repair"]
        assert (
            result1["validation_errors"]
            == result2["validation_errors"]
            == result3["validation_errors"]
        )
