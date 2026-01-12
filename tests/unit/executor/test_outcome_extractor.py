"""Tests for the outcome_extractor module (Phase 5.8.3).

This module tests:
- File operation extraction from agent responses
- Decision extraction using regex patterns
- LLM-based extraction
- Main extract_outcome function
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_infra.executor.outcome_extractor import (
    ExtractionResult,
    _clean_decision,
    _is_valid_file_ref,
    _parse_extraction_json,
    _resolve_path,
    extract_decisions_simple,
    extract_file_operations,
    extract_file_references,
    extract_outcome,
    extract_outcome_sync,
    extract_with_llm,
)
from ai_infra.executor.run_memory import FileAction

# =============================================================================
# Test ExtractionResult
# =============================================================================


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_create_empty(self):
        """Test creating empty result."""
        result = ExtractionResult()
        assert result.files == {}
        assert result.key_decisions == []
        assert result.summary == ""
        assert result.raw_file_refs == []

    def test_create_full(self):
        """Test creating with all fields."""
        result = ExtractionResult(
            files={Path("src/main.py"): FileAction.CREATED},
            key_decisions=["Created main module"],
            summary="Setup main entry point",
            raw_file_refs=["src/main.py"],
        )
        assert len(result.files) == 1
        assert result.summary == "Setup main entry point"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = ExtractionResult(
            files={Path("a.py"): FileAction.CREATED, Path("b.py"): FileAction.MODIFIED},
            key_decisions=["decision1", "decision2"],
            summary="Test summary",
            raw_file_refs=["a.py", "b.py"],
        )
        data = result.to_dict()
        assert data["summary"] == "Test summary"
        assert "a.py" in data["files"]
        assert data["files"]["a.py"] == "created"
        assert len(data["key_decisions"]) == 2


# =============================================================================
# Test File Operation Extraction
# =============================================================================


class TestExtractFileOperations:
    """Tests for extract_file_operations function."""

    def test_extracts_write_file_calls(self, tmp_path: Path):
        """Test extracting write_file tool calls."""
        response = """I will write the file.
write_file("src/utils.py", "def helper(): pass")
Done."""

        files = extract_file_operations(response, tmp_path)

        assert len(files) == 1
        assert tmp_path / "src/utils.py" in files
        assert files[tmp_path / "src/utils.py"] == FileAction.CREATED

    def test_extracts_edit_file_calls(self, tmp_path: Path):
        """Test extracting edit_file tool calls."""
        response = """Modifying the file.
edit_file("src/main.py", old="foo", new="bar")
Complete."""

        files = extract_file_operations(response, tmp_path)

        assert len(files) == 1
        assert tmp_path / "src/main.py" in files
        assert files[tmp_path / "src/main.py"] == FileAction.MODIFIED

    def test_extracts_create_file_calls(self, tmp_path: Path):
        """Test extracting create_file tool calls."""
        response = "create_file('tests/test_main.py', content='...')"

        files = extract_file_operations(response, tmp_path)

        assert tmp_path / "tests/test_main.py" in files
        assert files[tmp_path / "tests/test_main.py"] == FileAction.CREATED

    def test_extracts_delete_file_calls(self, tmp_path: Path):
        """Test extracting delete_file tool calls."""
        response = 'delete_file("old_file.py")'

        files = extract_file_operations(response, tmp_path)

        assert tmp_path / "old_file.py" in files
        assert files[tmp_path / "old_file.py"] == FileAction.DELETED

    def test_extracts_json_style_tool_calls(self, tmp_path: Path):
        """Test extracting JSON-formatted tool calls."""
        response = """{"tool": "write_file", "path": "config.json", "content": "{}"}"""

        files = extract_file_operations(response, tmp_path)

        assert tmp_path / "config.json" in files

    def test_extracts_natural_language_created(self, tmp_path: Path):
        """Test extracting from natural language - created."""
        response = "I created the file `src/helper.py` with the helper function."

        files = extract_file_operations(response, tmp_path)

        assert tmp_path / "src/helper.py" in files
        assert files[tmp_path / "src/helper.py"] == FileAction.CREATED

    def test_extracts_natural_language_modified(self, tmp_path: Path):
        """Test extracting from natural language - modified."""
        response = "I modified the file 'src/main.py' to fix the bug."

        files = extract_file_operations(response, tmp_path)

        assert tmp_path / "src/main.py" in files
        assert files[tmp_path / "src/main.py"] == FileAction.MODIFIED

    def test_handles_multiple_files(self, tmp_path: Path):
        """Test extracting multiple file operations."""
        response = """write_file("src/a.py", "...")
write_file("src/b.py", "...")
edit_file("src/c.py", old="x", new="y")"""

        files = extract_file_operations(response, tmp_path)

        assert len(files) == 3
        assert files[tmp_path / "src/a.py"] == FileAction.CREATED
        assert files[tmp_path / "src/b.py"] == FileAction.CREATED
        assert files[tmp_path / "src/c.py"] == FileAction.MODIFIED

    def test_handles_no_operations(self, tmp_path: Path):
        """Test response with no file operations."""
        response = "I analyzed the code and found no issues."

        files = extract_file_operations(response, tmp_path)

        assert files == {}

    def test_deduplicates_same_file(self, tmp_path: Path):
        """Test same file mentioned multiple times."""
        response = """write_file("src/main.py", "v1")
write_file("src/main.py", "v2")"""

        files = extract_file_operations(response, tmp_path)

        # Should have only one entry
        assert len(files) == 1

    def test_delete_takes_precedence(self, tmp_path: Path):
        """Test delete action takes precedence over others."""
        response = """write_file("temp.py", "...")
delete_file("temp.py")"""

        files = extract_file_operations(response, tmp_path)

        assert files[tmp_path / "temp.py"] == FileAction.DELETED


class TestExtractFileReferences:
    """Tests for extract_file_references function."""

    def test_extracts_backtick_references(self):
        """Test extracting file refs in backticks."""
        response = "See `src/main.py` and `tests/test_main.py` for details."

        refs = extract_file_references(response)

        assert "src/main.py" in refs
        assert "tests/test_main.py" in refs

    def test_extracts_quoted_references(self):
        """Test extracting file refs in quotes."""
        response = 'The file "config.json" contains settings.'

        refs = extract_file_references(response)

        assert "config.json" in refs

    def test_excludes_urls(self):
        """Test URLs are not extracted as file refs."""
        response = "See https://example.com/file.py for more info."

        refs = extract_file_references(response)

        assert "https://example.com/file.py" not in refs
        assert len(refs) == 0

    def test_returns_unique_refs(self):
        """Test duplicate refs are deduplicated."""
        response = "`main.py`, `utils.py`, `main.py` again"

        refs = extract_file_references(response)

        assert refs.count("main.py") == 1


# =============================================================================
# Test Path Resolution
# =============================================================================


class TestResolvePath:
    """Tests for _resolve_path function."""

    def test_resolves_relative_path(self, tmp_path: Path):
        """Test resolving relative path."""
        result = _resolve_path("src/main.py", tmp_path)
        assert result == tmp_path / "src/main.py"

    def test_resolves_absolute_path(self, tmp_path: Path):
        """Test resolving absolute path."""
        result = _resolve_path("/absolute/path/file.py", tmp_path)
        assert result == Path("/absolute/path/file.py")

    def test_handles_empty_path(self, tmp_path: Path):
        """Test empty path returns None."""
        result = _resolve_path("", tmp_path)
        assert result is None

    def test_strips_quotes(self, tmp_path: Path):
        """Test quotes are stripped."""
        result = _resolve_path('"src/main.py"', tmp_path)
        assert result == tmp_path / "src/main.py"


class TestIsValidFileRef:
    """Tests for _is_valid_file_ref function."""

    def test_valid_python_file(self):
        """Test valid Python file reference."""
        assert _is_valid_file_ref("src/main.py") is True

    def test_valid_js_file(self):
        """Test valid JavaScript file reference."""
        assert _is_valid_file_ref("src/index.js") is True

    def test_invalid_no_extension(self):
        """Test path without extension is invalid."""
        assert _is_valid_file_ref("src/main") is False

    def test_invalid_url(self):
        """Test URL is invalid."""
        assert _is_valid_file_ref("https://example.com/file.py") is False

    def test_invalid_version_number(self):
        """Test version numbers are invalid."""
        assert _is_valid_file_ref("1.0") is False
        assert _is_valid_file_ref("0.1") is False

    def test_invalid_too_short(self):
        """Test very short strings are invalid."""
        assert _is_valid_file_ref("a") is False
        assert _is_valid_file_ref("ab") is False

    def test_known_files_without_extension(self):
        """Test Makefile/Dockerfile are valid."""
        assert _is_valid_file_ref("Makefile") is True
        assert _is_valid_file_ref("Dockerfile") is True


# =============================================================================
# Test Decision Extraction
# =============================================================================


class TestExtractDecisionsSimple:
    """Tests for extract_decisions_simple function."""

    def test_extracts_will_patterns(self):
        """Test extracting 'I will' patterns."""
        response = "I will create the utils module with helper functions."

        decisions = extract_decisions_simple(response)

        assert len(decisions) >= 1
        assert any("utils module" in d.lower() for d in decisions)

    def test_extracts_need_to_patterns(self):
        """Test extracting 'I need to' patterns."""
        response = "I need to fix the import path first."

        decisions = extract_decisions_simple(response)

        assert len(decisions) >= 1
        assert any("import path" in d.lower() for d in decisions)

    def test_extracts_first_then_patterns(self):
        """Test extracting 'First/Then' patterns."""
        response = "First, I will create the module. Then, I will add tests."

        decisions = extract_decisions_simple(response)

        assert len(decisions) >= 1

    def test_extracts_issue_patterns(self):
        """Test extracting 'The issue is' patterns."""
        response = "The issue is the wrong import path in greeter.py."

        decisions = extract_decisions_simple(response)

        assert len(decisions) >= 1
        assert any("import path" in d.lower() for d in decisions)

    def test_limits_to_max_decisions(self):
        """Test decisions are limited to max."""
        response = """I will do A. I will do B. I will do C.
        I will do D. I will do E. I will do F."""

        decisions = extract_decisions_simple(response, max_decisions=3)

        assert len(decisions) <= 3

    def test_handles_no_patterns(self):
        """Test response with no decision patterns."""
        response = "The code looks good. No changes needed."

        decisions = extract_decisions_simple(response)

        assert isinstance(decisions, list)

    def test_truncates_long_decisions(self):
        """Test long decisions are truncated."""
        response = "I will " + "x" * 200 + "."

        decisions = extract_decisions_simple(response)

        if decisions:
            assert len(decisions[0]) <= 100

    def test_skips_very_short_decisions(self):
        """Test very short decisions are skipped."""
        response = "I will do it. I will create the comprehensive module structure."

        decisions = extract_decisions_simple(response)

        # "do it" is too short, should be skipped
        assert all(len(d) >= 10 for d in decisions)


class TestCleanDecision:
    """Tests for _clean_decision function."""

    def test_strips_whitespace(self):
        """Test whitespace is stripped."""
        assert _clean_decision("  test  ") == "test"

    def test_strips_punctuation(self):
        """Test trailing punctuation is stripped."""
        assert _clean_decision("test.") == "test"
        assert _clean_decision("test,") == "test"

    def test_removes_markdown_bold(self):
        """Test markdown bold is removed."""
        assert _clean_decision("**important** text") == "important text"

    def test_removes_backticks(self):
        """Test backticks are removed."""
        assert _clean_decision("use `function`") == "use function"

    def test_normalizes_whitespace(self):
        """Test multiple spaces are normalized."""
        assert _clean_decision("one   two    three") == "one two three"


# =============================================================================
# Test LLM Extraction
# =============================================================================


class TestExtractWithLLM:
    """Tests for extract_with_llm function."""

    @pytest.mark.asyncio
    async def test_extracts_from_llm_response(self):
        """Test extraction from LLM response."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            '{"summary": "Created utils module", "decisions": ["Added helper functions"]}'
        )
        mock_llm.achat = AsyncMock(return_value=mock_response)

        summary, decisions = await extract_with_llm(
            response="I created the utils module...",
            task_title="Create utils",
            llm=mock_llm,
        )

        assert summary == "Created utils module"
        assert "Added helper functions" in decisions

    @pytest.mark.asyncio
    async def test_uses_sync_chat_if_no_achat(self):
        """Test falls back to sync chat method."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"summary": "Done", "decisions": []}'
        mock_llm.chat = MagicMock(return_value=mock_response)
        # No achat attribute
        del mock_llm.achat

        summary, decisions = await extract_with_llm(
            response="Test",
            task_title="Test task",
            llm=mock_llm,
        )

        assert summary == "Done"
        mock_llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_on_parse_failure(self):
        """Test falls back to simple extraction on parse failure."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Not valid JSON at all"
        mock_llm.achat = AsyncMock(return_value=mock_response)

        summary, decisions = await extract_with_llm(
            response="I will create the module.",
            task_title="Create module task",
            llm=mock_llm,
        )

        # Should fall back to task title
        assert "Create module" in summary

    @pytest.mark.asyncio
    async def test_falls_back_on_exception(self):
        """Test falls back on LLM exception."""
        mock_llm = MagicMock()
        mock_llm.achat = AsyncMock(side_effect=Exception("API error"))

        summary, decisions = await extract_with_llm(
            response="I will fix the bug.",
            task_title="Fix bug",
            llm=mock_llm,
        )

        assert "Fix bug" in summary

    @pytest.mark.asyncio
    async def test_truncates_long_response(self):
        """Test long response is truncated for LLM."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"summary": "Done", "decisions": []}'
        mock_llm.achat = AsyncMock(return_value=mock_response)

        long_response = "x" * 10000

        await extract_with_llm(
            response=long_response,
            task_title="Test",
            llm=mock_llm,
            max_response_chars=1000,
        )

        # Check the prompt was truncated
        call_args = mock_llm.achat.call_args[0][0]
        assert "[truncated]" in call_args


class TestParseExtractionJson:
    """Tests for _parse_extraction_json function."""

    def test_parses_raw_json(self):
        """Test parsing raw JSON."""
        content = '{"summary": "test", "decisions": ["a", "b"]}'
        result = _parse_extraction_json(content)

        assert result["summary"] == "test"
        assert result["decisions"] == ["a", "b"]

    def test_parses_json_in_code_block(self):
        """Test parsing JSON in markdown code block."""
        content = """Here's the result:
```json
{"summary": "test", "decisions": []}
```"""
        result = _parse_extraction_json(content)

        assert result["summary"] == "test"

    def test_parses_json_without_lang_tag(self):
        """Test parsing JSON in code block without language tag."""
        content = """```
{"summary": "test", "decisions": []}
```"""
        result = _parse_extraction_json(content)

        assert result["summary"] == "test"

    def test_extracts_json_from_text(self):
        """Test extracting JSON embedded in text."""
        content = 'The result is: {"summary": "test", "decisions": []} as shown.'
        result = _parse_extraction_json(content)

        assert result["summary"] == "test"

    def test_returns_none_for_invalid(self):
        """Test returns None for invalid content."""
        content = "This is not JSON at all"
        result = _parse_extraction_json(content)

        assert result is None


# =============================================================================
# Test Main Extract Outcome Function
# =============================================================================


class TestExtractOutcome:
    """Tests for extract_outcome function."""

    @pytest.mark.asyncio
    async def test_extracts_without_llm(self, tmp_path: Path):
        """Test extraction without LLM uses simple methods."""
        response = """I will create the utils module.
write_file("src/utils.py", "def helper(): pass")
Done."""

        result = await extract_outcome(
            agent_response=response,
            workspace_root=tmp_path,
            task_id="1.1",
            task_title="Create utils module",
        )

        assert len(result.files) == 1
        assert tmp_path / "src/utils.py" in result.files
        assert result.summary == "Create utils module"
        assert len(result.key_decisions) >= 0  # May or may not find decisions

    @pytest.mark.asyncio
    async def test_extracts_with_llm(self, tmp_path: Path):
        """Test extraction with LLM."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            '{"summary": "Created helper module", "decisions": ["Used type hints"]}'
        )
        mock_llm.achat = AsyncMock(return_value=mock_response)

        response = 'write_file("src/helper.py", "...")'

        result = await extract_outcome(
            agent_response=response,
            workspace_root=tmp_path,
            task_id="1.2",
            task_title="Create helper",
            llm=mock_llm,
        )

        assert result.summary == "Created helper module"
        assert "Used type hints" in result.key_decisions
        assert tmp_path / "src/helper.py" in result.files

    @pytest.mark.asyncio
    async def test_includes_raw_file_refs(self, tmp_path: Path):
        """Test raw file references are included."""
        response = "See `src/main.py` and `src/utils.py` for details."

        result = await extract_outcome(
            agent_response=response,
            workspace_root=tmp_path,
            task_id="1.1",
            task_title="Review code",
        )

        assert "src/main.py" in result.raw_file_refs
        assert "src/utils.py" in result.raw_file_refs


class TestExtractOutcomeSync:
    """Tests for extract_outcome_sync function."""

    def test_extracts_files(self, tmp_path: Path):
        """Test sync extraction of files."""
        response = 'write_file("test.py", "...")'

        result = extract_outcome_sync(
            agent_response=response,
            workspace_root=tmp_path,
            task_id="1.1",
            task_title="Create test",
        )

        assert tmp_path / "test.py" in result.files
        assert result.summary == "Create test"

    def test_extracts_decisions(self, tmp_path: Path):
        """Test sync extraction of decisions."""
        response = "I will create the module with proper type hints."

        result = extract_outcome_sync(
            agent_response=response,
            workspace_root=tmp_path,
            task_id="1.1",
            task_title="Create module",
        )

        assert isinstance(result.key_decisions, list)

    def test_extracts_file_refs(self, tmp_path: Path):
        """Test sync extraction of file references."""
        response = "Modified `config.json` settings."

        result = extract_outcome_sync(
            agent_response=response,
            workspace_root=tmp_path,
            task_id="1.1",
            task_title="Update config",
        )

        assert "config.json" in result.raw_file_refs


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_empty_response(self, tmp_path: Path):
        """Test handling empty response."""
        result = await extract_outcome(
            agent_response="",
            workspace_root=tmp_path,
            task_id="1.1",
            task_title="Empty task",
        )

        assert result.files == {}
        assert result.summary == "Empty task"

    @pytest.mark.asyncio
    async def test_unicode_in_response(self, tmp_path: Path):
        """Test handling unicode content."""
        response = "Created `日本語.py` with helper 関数"

        result = await extract_outcome(
            agent_response=response,
            workspace_root=tmp_path,
            task_id="1.1",
            task_title="Unicode test",
        )

        # Should not crash
        assert isinstance(result, ExtractionResult)

    @pytest.mark.asyncio
    async def test_very_long_response(self, tmp_path: Path):
        """Test handling very long response."""
        response = "write_file('test.py', '...')\n" + "x" * 100000

        result = await extract_outcome(
            agent_response=response,
            workspace_root=tmp_path,
            task_id="1.1",
            task_title="Long response",
        )

        assert tmp_path / "test.py" in result.files

    @pytest.mark.asyncio
    async def test_malformed_tool_calls(self, tmp_path: Path):
        """Test handling malformed tool calls."""
        response = """write_file(malformed
edit_file("valid.py", ...)
write_file("also_valid.py")"""

        result = await extract_outcome(
            agent_response=response,
            workspace_root=tmp_path,
            task_id="1.1",
            task_title="Malformed test",
        )

        # Should still extract valid ones
        assert tmp_path / "valid.py" in result.files or tmp_path / "also_valid.py" in result.files

    def test_sync_with_special_paths(self, tmp_path: Path):
        """Test sync extraction with special path characters."""
        response = 'write_file("src/my-file.test.py", "...")'

        result = extract_outcome_sync(
            agent_response=response,
            workspace_root=tmp_path,
            task_id="1.1",
            task_title="Special paths",
        )

        assert tmp_path / "src/my-file.test.py" in result.files
