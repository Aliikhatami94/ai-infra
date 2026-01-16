"""Tests for Phase 16.5.10: File validation and newline repair.

This module tests the CoderAgent's ability to:
- Detect files with literal \\n instead of real newlines
- Repair malformed files automatically
- Validate Python file syntax after creation

These issues occur when LLMs incorrectly use `echo` with escaped newlines
instead of heredoc or other proper file writing methods.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_infra.executor.agents.coder import CoderAgent

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def coder_agent() -> CoderAgent:
    """Create a CoderAgent instance for testing."""
    return CoderAgent()


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace directory."""
    return tmp_path


# =============================================================================
# Tests: _has_malformed_newlines()
# =============================================================================


class TestHasMalformedNewlines:
    """Tests for detecting literal \\n in file content."""

    def test_detects_literal_newlines(self, coder_agent: CoderAgent) -> None:
        """Test: Detects content with literal \\n instead of real newlines."""
        # This is what happens when echo writes literal \n
        malformed = "import os\\n\\ndef main():\\n    pass"
        assert coder_agent._has_malformed_newlines(malformed) is True

    def test_accepts_real_newlines(self, coder_agent: CoderAgent) -> None:
        """Test: Accepts content with real newlines."""
        valid = "import os\n\ndef main():\n    pass"
        assert coder_agent._has_malformed_newlines(valid) is False

    def test_accepts_empty_content(self, coder_agent: CoderAgent) -> None:
        """Test: Accepts empty content."""
        assert coder_agent._has_malformed_newlines("") is False

    def test_accepts_single_line(self, coder_agent: CoderAgent) -> None:
        """Test: Accepts single-line content without newlines."""
        assert coder_agent._has_malformed_newlines("print('hello')") is False

    def test_detects_more_literal_than_real(self, coder_agent: CoderAgent) -> None:
        """Test: Detects when literal \\n count exceeds real newlines."""
        # Content with 3 literal \n but only 1 real newline
        malformed = "line1\\nline2\\nline3\n"
        assert coder_agent._has_malformed_newlines(malformed) is True

    def test_accepts_mixed_content_with_many_real(self, coder_agent: CoderAgent) -> None:
        """Test: Accepts content with \\n in strings but many real newlines."""
        # Valid Python with a string containing \n
        valid = """import os

def main():
    text = "line1\\nline2"
    print(text)

if __name__ == "__main__":
    main()
"""
        # Has one literal \n but many real newlines - should be OK
        assert coder_agent._has_malformed_newlines(valid) is False


# =============================================================================
# Tests: _validate_created_files()
# =============================================================================


class TestValidateCreatedFiles:
    """Tests for validating created files."""

    def test_detects_malformed_python_file(
        self,
        coder_agent: CoderAgent,
        temp_workspace: Path,
    ) -> None:
        """Test: Detects Python file with literal \\n."""
        # Create malformed file
        bad_file = temp_workspace / "bad.py"
        bad_file.write_text("import os\\n\\ndef main():\\n    pass")

        errors = coder_agent._validate_created_files(temp_workspace, ["bad.py"])

        assert len(errors) >= 1
        assert "bad.py" in errors[0]
        assert "literal \\n" in errors[0].lower() or "syntax error" in errors[0].lower()

    def test_accepts_valid_python_file(
        self,
        coder_agent: CoderAgent,
        temp_workspace: Path,
    ) -> None:
        """Test: Accepts valid Python file with real newlines."""
        good_file = temp_workspace / "good.py"
        good_file.write_text("import os\n\ndef main():\n    pass\n")

        errors = coder_agent._validate_created_files(temp_workspace, ["good.py"])

        assert len(errors) == 0

    def test_detects_syntax_error(
        self,
        coder_agent: CoderAgent,
        temp_workspace: Path,
    ) -> None:
        """Test: Detects Python syntax errors."""
        syntax_error_file = temp_workspace / "syntax_error.py"
        syntax_error_file.write_text("def main(\n    pass\n")  # Missing closing paren

        errors = coder_agent._validate_created_files(temp_workspace, ["syntax_error.py"])

        assert len(errors) == 1
        assert "syntax_error.py" in errors[0]
        assert "syntax error" in errors[0].lower()

    def test_handles_nonexistent_file(
        self,
        coder_agent: CoderAgent,
        temp_workspace: Path,
    ) -> None:
        """Test: Gracefully handles nonexistent files."""
        errors = coder_agent._validate_created_files(temp_workspace, ["nonexistent.py"])

        # Should not error, just skip
        assert len(errors) == 0

    def test_validates_multiple_files(
        self,
        coder_agent: CoderAgent,
        temp_workspace: Path,
    ) -> None:
        """Test: Validates multiple files at once."""
        # Create one good and one bad file
        good = temp_workspace / "good.py"
        good.write_text('print("hello")\n')

        bad = temp_workspace / "bad.py"
        bad.write_text('import sys\\n\\nprint("bad")')

        errors = coder_agent._validate_created_files(temp_workspace, ["good.py", "bad.py"])

        # bad.py should have errors (malformed newlines + resulting syntax error)
        assert len(errors) >= 1
        assert any("bad.py" in err for err in errors)
        assert not any("good.py" in err for err in errors)

    def test_skips_non_python_files(
        self,
        coder_agent: CoderAgent,
        temp_workspace: Path,
    ) -> None:
        """Test: Non-Python files skip syntax validation (only check newlines)."""
        # Create text file with literal \n
        text_file = temp_workspace / "notes.txt"
        text_file.write_text("line1\\nline2\\nline3")

        errors = coder_agent._validate_created_files(temp_workspace, ["notes.txt"])

        # Should detect malformed newlines
        assert len(errors) == 1
        assert "notes.txt" in errors[0]


# =============================================================================
# Tests: _repair_newlines()
# =============================================================================


class TestRepairNewlines:
    """Tests for repairing malformed newlines in files."""

    def test_repairs_literal_newlines(
        self,
        coder_agent: CoderAgent,
        temp_workspace: Path,
    ) -> None:
        """Test: Repairs file with literal \\n."""
        bad_file = temp_workspace / "repair_me.py"
        bad_file.write_text("import os\\n\\ndef main():\\n    pass")

        result = coder_agent._repair_newlines(bad_file)

        assert result is True
        content = bad_file.read_text()
        assert "\\n" not in content
        assert "\n" in content

    def test_repairs_literal_tabs(
        self,
        coder_agent: CoderAgent,
        temp_workspace: Path,
    ) -> None:
        """Test: Repairs file with literal \\t."""
        bad_file = temp_workspace / "tabs.py"
        bad_file.write_text("def main():\\n\\tpass")

        result = coder_agent._repair_newlines(bad_file)

        assert result is True
        content = bad_file.read_text()
        assert "\\t" not in content
        assert "\t" in content

    def test_no_repair_needed(
        self,
        coder_agent: CoderAgent,
        temp_workspace: Path,
    ) -> None:
        """Test: Returns False when no repair needed."""
        good_file = temp_workspace / "good.py"
        good_file.write_text("import os\n\ndef main():\n    pass\n")

        result = coder_agent._repair_newlines(good_file)

        assert result is False

    def test_repaired_file_has_valid_syntax(
        self,
        coder_agent: CoderAgent,
        temp_workspace: Path,
    ) -> None:
        """Test: Repaired Python file has valid syntax."""
        bad_file = temp_workspace / "fixable.py"
        bad_file.write_text('import os\\n\\ndef main():\\n    print("hello")\\n')

        coder_agent._repair_newlines(bad_file)

        content = bad_file.read_text()
        # Should compile without error
        compile(content, "fixable.py", "exec")

    def test_handles_windows_newlines(
        self,
        coder_agent: CoderAgent,
        temp_workspace: Path,
    ) -> None:
        """Test: Repairs literal \\r\\n to real newlines."""
        bad_file = temp_workspace / "windows.py"
        bad_file.write_text("import os\\r\\n\\r\\ndef main():\\r\\n    pass")

        result = coder_agent._repair_newlines(bad_file)

        assert result is True
        content = bad_file.read_text()
        assert "\\r\\n" not in content
        assert "\\r" not in content


# =============================================================================
# Tests: Integration scenarios
# =============================================================================


class TestFileValidationIntegration:
    """Integration tests for the full validation and repair flow."""

    def test_validate_and_repair_flow(
        self,
        coder_agent: CoderAgent,
        temp_workspace: Path,
    ) -> None:
        """Test: Full flow of validation, repair, and re-validation."""
        # Create malformed file
        bad_file = temp_workspace / "integration.py"
        bad_file.write_text("import sys\\n\\ndef main():\\n    print(sys.version)")

        # Initial validation should find errors
        initial_errors = coder_agent._validate_created_files(temp_workspace, ["integration.py"])
        assert len(initial_errors) >= 1

        # Repair should succeed
        repaired = coder_agent._repair_newlines(bad_file)
        assert repaired is True

        # Re-validation should pass
        final_errors = coder_agent._validate_created_files(temp_workspace, ["integration.py"])
        assert len(final_errors) == 0

    def test_complex_python_file(
        self,
        coder_agent: CoderAgent,
        temp_workspace: Path,
    ) -> None:
        """Test: Repair and validate complex Python file."""
        # More realistic file content (what an LLM might generate with echo)
        malformed_content = (
            "import argparse\\n"
            "\\n"
            "def main():\\n"
            '    parser = argparse.ArgumentParser(description="Test")\\n'
            '    parser.add_argument("--name", help="Name")\\n'
            "    args = parser.parse_args()\\n"
            '    print(f"Hello, {args.name}")\\n'
            "\\n"
            'if __name__ == "__main__":\\n'
            "    main()\\n"
        )

        cli_file = temp_workspace / "cli.py"
        cli_file.write_text(malformed_content)

        # Repair
        coder_agent._repair_newlines(cli_file)

        # Should now be valid Python
        content = cli_file.read_text()
        compile(content, "cli.py", "exec")

        # Validate
        errors = coder_agent._validate_created_files(temp_workspace, ["cli.py"])
        assert len(errors) == 0


# =============================================================================
# Tests: Edge cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for file validation."""

    def test_binary_file_handling(
        self,
        coder_agent: CoderAgent,
        temp_workspace: Path,
    ) -> None:
        """Test: Gracefully handles binary files."""
        binary_file = temp_workspace / "data.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03")

        # Should not raise, just return error about reading
        errors = coder_agent._validate_created_files(temp_workspace, ["data.bin"])
        # Either empty (skipped) or error reading
        assert len(errors) <= 1

    def test_empty_file(
        self,
        coder_agent: CoderAgent,
        temp_workspace: Path,
    ) -> None:
        """Test: Handles empty files."""
        empty_file = temp_workspace / "empty.py"
        empty_file.write_text("")

        errors = coder_agent._validate_created_files(temp_workspace, ["empty.py"])
        assert len(errors) == 0

    def test_file_with_only_whitespace(
        self,
        coder_agent: CoderAgent,
        temp_workspace: Path,
    ) -> None:
        """Test: Handles files with only whitespace."""
        ws_file = temp_workspace / "whitespace.py"
        ws_file.write_text("   \n\n\t\n")

        errors = coder_agent._validate_created_files(temp_workspace, ["whitespace.py"])
        assert len(errors) == 0

    def test_deeply_nested_file(
        self,
        coder_agent: CoderAgent,
        temp_workspace: Path,
    ) -> None:
        """Test: Handles files in nested directories."""
        nested_dir = temp_workspace / "src" / "pkg" / "subpkg"
        nested_dir.mkdir(parents=True)
        nested_file = nested_dir / "module.py"
        nested_file.write_text("x = 1\\ny = 2")

        errors = coder_agent._validate_created_files(temp_workspace, ["src/pkg/subpkg/module.py"])
        # Expect malformed newlines error (may also report syntax error)
        assert len(errors) >= 1
        assert any("module.py" in err for err in errors)
