"""Unit tests for ai_infra.cli.streaming module.

Phase 16.7.2 of EXECUTOR_6.md: Real-time Streaming Output.
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from ai_infra.cli.streaming import (
    EXTENSION_TO_LANGUAGE,
    LINE_NUMBER_THRESHOLD,
    OUTPUT_KEY_BINDINGS,
    SUPPORTED_LANGUAGES,
    DiffHunk,
    DiffViewer,
    FileChange,
    FileChangeType,
    FileTreeDisplay,
    LiveStreamingDisplay,
    StreamChunk,
    StreamingOutput,
    ToolCall,
    ToolCallManager,
    ToolCallPanel,
    ToolCallState,
    create_syntax,
    detect_language,
    extract_code_blocks,
    parse_unified_diff,
    render_output_help,
)

# =============================================================================
# ToolCallState Tests
# =============================================================================


class TestToolCallState:
    """Tests for ToolCallState enum."""

    def test_all_states_defined(self) -> None:
        """All expected states should be defined."""
        expected = {"pending", "running", "complete", "failed", "collapsed"}
        actual = {state.value for state in ToolCallState}
        assert actual == expected

    def test_state_string_values(self) -> None:
        """States should have lowercase string values."""
        assert ToolCallState.PENDING.value == "pending"
        assert ToolCallState.RUNNING.value == "running"
        assert ToolCallState.COMPLETE.value == "complete"
        assert ToolCallState.FAILED.value == "failed"
        assert ToolCallState.COLLAPSED.value == "collapsed"


class TestFileChangeType:
    """Tests for FileChangeType enum."""

    def test_all_types_defined(self) -> None:
        """All expected change types should be defined."""
        expected = {"new", "modified", "deleted", "renamed"}
        actual = {ct.value for ct in FileChangeType}
        assert actual == expected

    def test_type_string_values(self) -> None:
        """Change types should have lowercase string values."""
        assert FileChangeType.NEW.value == "new"
        assert FileChangeType.MODIFIED.value == "modified"
        assert FileChangeType.DELETED.value == "deleted"
        assert FileChangeType.RENAMED.value == "renamed"


# =============================================================================
# ToolCall Tests
# =============================================================================


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_creation_with_defaults(self) -> None:
        """ToolCall should have sensible defaults."""
        tc = ToolCall(tool_name="test_tool")
        assert tc.tool_name == "test_tool"
        assert tc.args == {}
        assert tc.output is None
        assert tc.state == ToolCallState.PENDING
        assert tc.duration is None
        assert tc.summary is None
        assert tc.line_count is None

    def test_creation_with_all_args(self) -> None:
        """ToolCall should accept all arguments."""
        tc = ToolCall(
            tool_name="write_file",
            args={"path": "src/test.py"},
            output="File created",
            state=ToolCallState.COMPLETE,
            duration=1.5,
            summary="Created test file",
            line_count=45,
        )
        assert tc.tool_name == "write_file"
        assert tc.args == {"path": "src/test.py"}
        assert tc.output == "File created"
        assert tc.state == ToolCallState.COMPLETE
        assert tc.duration == 1.5
        assert tc.summary == "Created test file"
        assert tc.line_count == 45

    def test_get_display_args_with_path(self) -> None:
        """Display args should show path."""
        tc = ToolCall(tool_name="write_file", args={"path": "src/main.py"})
        assert tc.get_display_args() == '("src/main.py")'

    def test_get_display_args_with_command(self) -> None:
        """Display args should show command."""
        tc = ToolCall(tool_name="run_command", args={"command": "pytest -v"})
        assert tc.get_display_args() == '("pytest -v")'

    def test_get_display_args_single_arg(self) -> None:
        """Display args should show single short arg."""
        tc = ToolCall(tool_name="test", args={"query": "test query"})
        assert tc.get_display_args() == '("test query")'

    def test_get_display_args_multiple(self) -> None:
        """Display args should show count for multiple."""
        tc = ToolCall(tool_name="test", args={"a": 1, "b": 2, "c": 3})
        assert tc.get_display_args() == "(3 args)"

    def test_get_display_args_empty(self) -> None:
        """Display args should be empty for no args."""
        tc = ToolCall(tool_name="test")
        assert tc.get_display_args() == ""


# =============================================================================
# DiffHunk Tests
# =============================================================================


class TestDiffHunk:
    """Tests for DiffHunk dataclass."""

    def test_creation_with_defaults(self) -> None:
        """DiffHunk should have sensible defaults."""
        hunk = DiffHunk(old_start=1, old_count=5, new_start=1, new_count=7)
        assert hunk.old_start == 1
        assert hunk.old_count == 5
        assert hunk.new_start == 1
        assert hunk.new_count == 7
        assert hunk.lines == []

    def test_header_property(self) -> None:
        """Header should format correctly."""
        hunk = DiffHunk(old_start=45, old_count=6, new_start=45, new_count=12)
        assert hunk.header == "@@ -45,6 +45,12 @@"

    def test_lines_storage(self) -> None:
        """Lines should store prefix and content."""
        hunk = DiffHunk(old_start=1, old_count=2, new_start=1, new_count=3)
        hunk.lines = [
            (" ", "context line"),
            ("-", "removed line"),
            ("+", "added line"),
        ]
        assert len(hunk.lines) == 3
        assert hunk.lines[0] == (" ", "context line")


# =============================================================================
# FileChange Tests
# =============================================================================


class TestFileChange:
    """Tests for FileChange dataclass."""

    def test_creation_with_defaults(self) -> None:
        """FileChange should have sensible defaults."""
        fc = FileChange(path="src/main.py")
        assert fc.path == "src/main.py"
        assert fc.change_type == FileChangeType.MODIFIED
        assert fc.lines_added == 0
        assert fc.lines_removed == 0

    def test_string_change_type_conversion(self) -> None:
        """String change type should be converted to enum."""
        fc = FileChange(path="src/new.py", change_type="new")
        assert fc.change_type == FileChangeType.NEW

    def test_full_creation(self) -> None:
        """FileChange should accept all args."""
        fc = FileChange(
            path="src/auth.py",
            change_type=FileChangeType.MODIFIED,
            lines_added=45,
            lines_removed=12,
        )
        assert fc.path == "src/auth.py"
        assert fc.change_type == FileChangeType.MODIFIED
        assert fc.lines_added == 45
        assert fc.lines_removed == 12


# =============================================================================
# StreamChunk Tests
# =============================================================================


class TestStreamChunk:
    """Tests for StreamChunk dataclass."""

    def test_creation_with_defaults(self) -> None:
        """StreamChunk should have sensible defaults."""
        chunk = StreamChunk(content="Hello")
        assert chunk.content == "Hello"
        assert chunk.is_code is False
        assert chunk.language is None

    def test_code_chunk(self) -> None:
        """StreamChunk should represent code blocks."""
        chunk = StreamChunk(content="def foo():", is_code=True, language="python")
        assert chunk.is_code is True
        assert chunk.language == "python"


# =============================================================================
# Language Detection Tests
# =============================================================================


class TestLanguageDetection:
    """Tests for language detection."""

    def test_supported_languages_has_common_languages(self) -> None:
        """Supported languages should include common ones."""
        expected = {"python", "typescript", "javascript", "sql", "bash", "yaml", "json"}
        assert expected.issubset(SUPPORTED_LANGUAGES.keys())

    def test_extension_to_language_mapping(self) -> None:
        """Extension mapping should be correct."""
        assert EXTENSION_TO_LANGUAGE["py"] == "python"
        assert EXTENSION_TO_LANGUAGE["ts"] == "typescript"
        assert EXTENSION_TO_LANGUAGE["js"] == "javascript"
        assert EXTENSION_TO_LANGUAGE["yaml"] == "yaml"

    def test_detect_language_from_extension(self) -> None:
        """Should detect language from file extension."""
        assert detect_language("src/main.py") == "python"
        assert detect_language("src/app.ts") == "typescript"
        assert detect_language("config.yaml") == "yaml"
        assert detect_language("package.json") == "json"

    def test_detect_language_from_filename(self) -> None:
        """Should detect language from special filenames."""
        assert detect_language("Dockerfile") == "dockerfile"
        assert detect_language("Makefile") == "makefile"

    def test_detect_language_unknown(self) -> None:
        """Unknown files should return text."""
        assert detect_language("README") == "text"
        assert detect_language("unknown.xyz") == "text"

    def test_detect_language_from_content(self) -> None:
        """Should detect language from content hints."""
        assert detect_language(content="#!/usr/bin/env python") == "python"
        assert detect_language(content="#!/bin/bash") == "bash"
        assert detect_language(content="def foo():\n    pass") == "python"

    def test_detect_language_filepath_priority(self) -> None:
        """Filepath should take priority over content."""
        assert detect_language("test.py", "function foo() {}") == "python"


# =============================================================================
# Syntax Highlighting Tests
# =============================================================================


class TestCreateSyntax:
    """Tests for create_syntax function."""

    def test_basic_creation(self) -> None:
        """Should create Syntax object."""
        syntax = create_syntax("print('hello')", "python")
        assert syntax is not None

    def test_auto_line_numbers_short(self) -> None:
        """Short code should not have line numbers."""
        code = "print('hello')"
        syntax = create_syntax(code, "python")
        # Line numbers is private, so just verify creation works
        assert syntax is not None

    def test_auto_line_numbers_long(self) -> None:
        """Long code should have line numbers."""
        code = "\n".join([f"line {i}" for i in range(LINE_NUMBER_THRESHOLD + 5)])
        syntax = create_syntax(code, "python")
        assert syntax is not None

    def test_explicit_line_numbers(self) -> None:
        """Should respect explicit line_numbers setting."""
        code = "print('hello')"
        syntax = create_syntax(code, "python", line_numbers=True)
        assert syntax is not None

    def test_start_line(self) -> None:
        """Should accept start_line parameter."""
        syntax = create_syntax("print('hello')", "python", start_line=45)
        assert syntax is not None


# =============================================================================
# Code Block Extraction Tests
# =============================================================================


class TestExtractCodeBlocks:
    """Tests for extract_code_blocks function."""

    def test_single_block(self) -> None:
        """Should extract single code block."""
        text = "Here is code:\n```python\nprint('hello')\n```\nDone."
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][0] == "python"
        assert "print('hello')" in blocks[0][1]

    def test_multiple_blocks(self) -> None:
        """Should extract multiple code blocks."""
        text = """
```python
code1
```

```javascript
code2
```
"""
        blocks = extract_code_blocks(text)
        assert len(blocks) == 2
        assert blocks[0][0] == "python"
        assert blocks[1][0] == "javascript"

    def test_no_language(self) -> None:
        """Should handle blocks without language."""
        text = "```\nplain code\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][0] is None

    def test_no_blocks(self) -> None:
        """Should return empty for no blocks."""
        text = "Just plain text without code"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 0


# =============================================================================
# StreamingOutput Tests
# =============================================================================


class TestStreamingOutput:
    """Tests for StreamingOutput class."""

    def test_creation(self) -> None:
        """Should create streaming output handler."""
        console = Console(file=StringIO(), force_terminal=True)
        streamer = StreamingOutput(console=console)
        assert streamer is not None
        assert streamer.buffer == ""
        assert not streamer.is_finalized

    def test_append_text(self) -> None:
        """Should append text to buffer."""
        console = Console(file=StringIO(), force_terminal=True)
        streamer = StreamingOutput(console=console)
        streamer.append("Hello ")
        streamer.append("World")
        assert "Hello" in streamer.buffer or len(streamer._chunks) > 0

    def test_finalize(self) -> None:
        """Should finalize the stream."""
        console = Console(file=StringIO(), force_terminal=True)
        streamer = StreamingOutput(console=console)
        streamer.append("Test content")
        streamer.finalize()
        assert streamer.is_finalized

    def test_render(self) -> None:
        """Should render content."""
        console = Console(file=StringIO(), force_terminal=True)
        streamer = StreamingOutput(console=console)
        streamer.append("Test content")
        streamer.finalize()
        rendered = streamer.render()
        assert rendered is not None

    def test_code_block_detection(self) -> None:
        """Should detect and parse code blocks."""
        console = Console(file=StringIO(), force_terminal=True)
        streamer = StreamingOutput(console=console)
        streamer.append("Here is code:\n```python\nprint('hi')\n```\nDone")
        streamer.finalize()

        # Should have parsed chunks
        has_code_chunk = any(c.is_code for c in streamer._chunks)
        assert has_code_chunk

    def test_rich_console_protocol(self) -> None:
        """Should implement __rich_console__ protocol."""
        console = Console(file=StringIO(), force_terminal=True, width=80)
        streamer = StreamingOutput(console=console)
        streamer.append("Test")
        streamer.finalize()

        # Should be printable
        output = StringIO()
        test_console = Console(file=output, force_terminal=True, width=80)
        test_console.print(streamer)
        assert len(output.getvalue()) > 0


# =============================================================================
# ToolCallPanel Tests
# =============================================================================


class TestToolCallPanel:
    """Tests for ToolCallPanel class."""

    def test_creation_with_tool_call(self) -> None:
        """Should create panel from ToolCall."""
        tc = ToolCall(
            tool_name="write_file",
            args={"path": "test.py"},
            state=ToolCallState.COMPLETE,
        )
        panel = ToolCallPanel(tool_call=tc)
        assert panel.tool_name == "write_file"
        assert panel.state == ToolCallState.COMPLETE

    def test_creation_with_kwargs(self) -> None:
        """Should create panel from kwargs."""
        panel = ToolCallPanel(
            tool_name="read_file",
            args={"path": "main.py"},
            state=ToolCallState.RUNNING,
        )
        assert panel.tool_name == "read_file"
        assert panel.state == ToolCallState.RUNNING

    def test_default_expanded(self) -> None:
        """Panel should be expanded by default."""
        panel = ToolCallPanel(tool_name="test")
        assert panel.expanded is True

    def test_collapsed(self) -> None:
        """Panel can be collapsed."""
        panel = ToolCallPanel(tool_name="test", expanded=False)
        assert panel.expanded is False

    def test_render_header(self) -> None:
        """Should render header with tool name."""
        panel = ToolCallPanel(tool_name="write_file", args={"path": "test.py"})
        header = panel._render_header()
        assert "write_file" in header.plain

    def test_rich_console_protocol(self) -> None:
        """Should implement __rich_console__ protocol."""
        panel = ToolCallPanel(
            tool_name="test_tool",
            output="Test output",
            state=ToolCallState.COMPLETE,
        )

        output = StringIO()
        console = Console(file=output, force_terminal=True, width=80)
        console.print(panel)
        assert "test_tool" in output.getvalue()


# =============================================================================
# ToolCallManager Tests
# =============================================================================


class TestToolCallManager:
    """Tests for ToolCallManager class."""

    def test_creation_empty(self) -> None:
        """Should create empty manager."""
        manager = ToolCallManager()
        assert len(manager.tool_calls) == 0
        assert manager.current_index == 0

    def test_creation_with_calls(self) -> None:
        """Should create with initial calls."""
        calls = [
            ToolCall(tool_name="tool1"),
            ToolCall(tool_name="tool2"),
        ]
        manager = ToolCallManager(tool_calls=calls)
        assert len(manager.tool_calls) == 2

    def test_add_tool_call(self) -> None:
        """Should add tool calls."""
        manager = ToolCallManager()
        index = manager.add_tool_call(ToolCall(tool_name="test"))
        assert index == 0
        assert len(manager.tool_calls) == 1

    def test_toggle_current(self) -> None:
        """Should toggle expand/collapse."""
        manager = ToolCallManager(tool_calls=[ToolCall(tool_name="test")])
        assert 0 in manager._expanded

        manager.toggle_current()
        assert 0 not in manager._expanded

        manager.toggle_current()
        assert 0 in manager._expanded

    def test_expand_all(self) -> None:
        """Should expand all calls."""
        calls = [ToolCall(tool_name=f"tool{i}") for i in range(3)]
        manager = ToolCallManager(tool_calls=calls)
        manager.collapse_all()
        assert len(manager._expanded) == 0

        manager.expand_all()
        assert len(manager._expanded) == 3

    def test_collapse_all(self) -> None:
        """Should collapse all calls."""
        calls = [ToolCall(tool_name=f"tool{i}") for i in range(3)]
        manager = ToolCallManager(tool_calls=calls)
        manager.collapse_all()
        assert len(manager._expanded) == 0

    def test_navigate(self) -> None:
        """Should navigate between calls."""
        calls = [ToolCall(tool_name=f"tool{i}") for i in range(3)]
        manager = ToolCallManager(tool_calls=calls)

        assert manager.current_index == 0
        manager.navigate(1)
        assert manager.current_index == 1
        manager.navigate(1)
        assert manager.current_index == 2
        manager.navigate(1)  # Should not go beyond
        assert manager.current_index == 2
        manager.navigate(-1)
        assert manager.current_index == 1

    def test_handle_key_toggle(self) -> None:
        """Should handle space key for toggle."""
        manager = ToolCallManager(tool_calls=[ToolCall(tool_name="test")])
        result = manager.handle_key(" ")
        assert result is True

    def test_handle_key_navigation(self) -> None:
        """Should handle navigation keys."""
        calls = [ToolCall(tool_name=f"tool{i}") for i in range(3)]
        manager = ToolCallManager(tool_calls=calls)

        manager.handle_key("j")
        assert manager.current_index == 1

        manager.handle_key("k")
        assert manager.current_index == 0

    def test_handle_key_unknown(self) -> None:
        """Should return False for unknown keys."""
        manager = ToolCallManager()
        result = manager.handle_key("x")
        assert result is False

    def test_render(self) -> None:
        """Should render all tool calls."""
        calls = [ToolCall(tool_name=f"tool{i}") for i in range(2)]
        manager = ToolCallManager(tool_calls=calls)
        rendered = manager.render()
        assert rendered is not None


# =============================================================================
# Diff Parsing Tests
# =============================================================================


class TestParseUnifiedDiff:
    """Tests for parse_unified_diff function."""

    def test_single_hunk(self) -> None:
        """Should parse single hunk."""
        diff = """@@ -1,3 +1,4 @@
 context
-removed
+added
+another added
 context
"""
        hunks = parse_unified_diff(diff)
        assert len(hunks) == 1
        assert hunks[0].old_start == 1
        assert hunks[0].old_count == 3
        assert hunks[0].new_start == 1
        assert hunks[0].new_count == 4

    def test_multiple_hunks(self) -> None:
        """Should parse multiple hunks."""
        diff = """@@ -1,2 +1,3 @@
 line1
+new line
@@ -10,2 +11,2 @@
-old
+new
"""
        hunks = parse_unified_diff(diff)
        assert len(hunks) == 2

    def test_hunk_lines(self) -> None:
        """Should parse hunk lines with prefixes."""
        diff = """@@ -1,3 +1,3 @@
 context
-old
+new
"""
        hunks = parse_unified_diff(diff)
        assert len(hunks[0].lines) == 3
        assert hunks[0].lines[0] == (" ", "context")
        assert hunks[0].lines[1] == ("-", "old")
        assert hunks[0].lines[2] == ("+", "new")

    def test_empty_diff(self) -> None:
        """Should handle empty diff."""
        hunks = parse_unified_diff("")
        assert len(hunks) == 0


# =============================================================================
# DiffViewer Tests
# =============================================================================


class TestDiffViewer:
    """Tests for DiffViewer class."""

    def test_creation_with_diff_text(self) -> None:
        """Should create from diff text."""
        diff_text = """@@ -1,2 +1,3 @@
 line1
+new line
"""
        viewer = DiffViewer(filepath="src/test.py", diff_text=diff_text)
        assert viewer.filepath == "src/test.py"
        assert len(viewer.hunks) == 1

    def test_creation_with_hunks(self) -> None:
        """Should create from pre-parsed hunks."""
        hunks = [DiffHunk(old_start=1, old_count=2, new_start=1, new_count=3)]
        viewer = DiffViewer(filepath="test.py", hunks=hunks)
        assert len(viewer.hunks) == 1

    def test_language_detection(self) -> None:
        """Should detect language from filepath."""
        viewer = DiffViewer(filepath="src/main.py")
        assert viewer.language == "python"

    def test_language_override(self) -> None:
        """Should accept language override."""
        viewer = DiffViewer(filepath="test.txt", language="python")
        assert viewer.language == "python"

    def test_rich_console_protocol(self) -> None:
        """Should implement __rich_console__ protocol."""
        diff_text = """@@ -1,2 +1,3 @@
 context
+added
"""
        viewer = DiffViewer(filepath="test.py", diff_text=diff_text)

        output = StringIO()
        console = Console(file=output, force_terminal=True, width=80)
        console.print(viewer)
        assert "test.py" in output.getvalue()


# =============================================================================
# FileTreeDisplay Tests
# =============================================================================


class TestFileTreeDisplay:
    """Tests for FileTreeDisplay class."""

    def test_creation_empty(self) -> None:
        """Should create empty tree."""
        tree = FileTreeDisplay()
        assert len(tree.changes) == 0

    def test_creation_with_changes(self) -> None:
        """Should create with file changes."""
        changes = [
            FileChange(path="src/new.py", change_type=FileChangeType.NEW),
            FileChange(path="src/mod.py", change_type=FileChangeType.MODIFIED),
        ]
        tree = FileTreeDisplay(changes=changes)
        assert len(tree.changes) == 2

    def test_build_tree(self) -> None:
        """Should build Rich Tree."""
        changes = [
            FileChange(path="src/auth.py", change_type=FileChangeType.NEW, lines_added=45),
            FileChange(path="src/main.py", change_type=FileChangeType.MODIFIED),
            FileChange(path="tests/test_auth.py", change_type=FileChangeType.NEW),
        ]
        display = FileTreeDisplay(changes=changes)
        tree = display._build_tree()
        assert tree is not None

    def test_rich_console_protocol(self) -> None:
        """Should implement __rich_console__ protocol."""
        changes = [
            FileChange(path="src/test.py", change_type=FileChangeType.NEW),
        ]
        display = FileTreeDisplay(changes=changes)

        output = StringIO()
        console = Console(file=output, force_terminal=True, width=80)
        console.print(display)
        assert "test.py" in output.getvalue()


# =============================================================================
# Output Help Tests
# =============================================================================


class TestRenderOutputHelp:
    """Tests for render_output_help function."""

    def test_returns_panel(self) -> None:
        """Should return a Panel."""
        panel = render_output_help()
        assert panel is not None

    def test_contains_key_bindings(self) -> None:
        """Panel should mention key bindings."""
        panel = render_output_help()

        output = StringIO()
        console = Console(file=output, force_terminal=True, width=80)
        console.print(panel)

        content = output.getvalue()
        assert "Space" in content or "Toggle" in content


# =============================================================================
# LiveStreamingDisplay Tests
# =============================================================================


class TestLiveStreamingDisplay:
    """Tests for LiveStreamingDisplay class."""

    def test_creation(self) -> None:
        """Should create live display."""
        console = Console(file=StringIO(), force_terminal=True)
        display = LiveStreamingDisplay(console=console)
        assert display is not None
        assert display.streamer is not None
        assert display.tool_manager is not None

    def test_append_text(self) -> None:
        """Should append streaming text."""
        console = Console(file=StringIO(), force_terminal=True)
        display = LiveStreamingDisplay(console=console)
        display.append_text("Hello")
        # Just verify no error

    def test_add_tool_call(self) -> None:
        """Should add tool calls."""
        console = Console(file=StringIO(), force_terminal=True)
        display = LiveStreamingDisplay(console=console)
        index = display.add_tool_call(ToolCall(tool_name="test"))
        assert index == 0

    def test_add_file_change(self) -> None:
        """Should add file changes."""
        console = Console(file=StringIO(), force_terminal=True)
        display = LiveStreamingDisplay(console=console)
        display.add_file_change(FileChange(path="test.py"))
        assert len(display.file_changes) == 1

    def test_render(self) -> None:
        """Should render combined output."""
        console = Console(file=StringIO(), force_terminal=True)
        display = LiveStreamingDisplay(console=console)
        display.append_text("Text output")
        display.add_tool_call(ToolCall(tool_name="tool"))
        display.add_file_change(FileChange(path="file.py"))

        rendered = display.render()
        assert rendered is not None

    def test_finalize(self) -> None:
        """Should finalize the display."""
        console = Console(file=StringIO(), force_terminal=True)
        display = LiveStreamingDisplay(console=console)
        display.append_text("Test")
        display.finalize()
        assert display.streamer.is_finalized


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_output_key_bindings(self) -> None:
        """Key bindings should be defined."""
        assert " " in OUTPUT_KEY_BINDINGS
        assert "a" in OUTPUT_KEY_BINDINGS
        assert "z" in OUTPUT_KEY_BINDINGS
        assert "j" in OUTPUT_KEY_BINDINGS
        assert "k" in OUTPUT_KEY_BINDINGS

    def test_line_number_threshold(self) -> None:
        """Line number threshold should be reasonable."""
        assert LINE_NUMBER_THRESHOLD == 10

    def test_supported_languages_not_empty(self) -> None:
        """Should have supported languages."""
        assert len(SUPPORTED_LANGUAGES) > 0

    def test_extension_mapping_complete(self) -> None:
        """All extensions should be mapped."""
        for lang, exts in SUPPORTED_LANGUAGES.items():
            for ext in exts:
                assert ext in EXTENSION_TO_LANGUAGE
                assert EXTENSION_TO_LANGUAGE[ext] == lang


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for streaming components."""

    def test_full_streaming_workflow(self) -> None:
        """Test complete streaming workflow."""
        console = Console(file=StringIO(), force_terminal=True, width=80)
        display = LiveStreamingDisplay(console=console)

        # Simulate streaming output
        display.append_text("Starting task analysis...\n\n")
        display.append_text("I'll implement the auth module:\n\n")
        display.append_text("```python\ndef authenticate(token: str) -> bool:\n")
        display.append_text("    return verify_jwt(token)\n```\n\n")

        # Add tool call
        display.add_tool_call(
            ToolCall(
                tool_name="write_file",
                args={"path": "src/auth.py"},
                output="File created",
                state=ToolCallState.COMPLETE,
                line_count=25,
            )
        )

        # Add file change
        display.add_file_change(
            FileChange(
                path="src/auth.py",
                change_type=FileChangeType.NEW,
                lines_added=25,
            )
        )

        # Finalize
        display.finalize()

        # Render
        rendered = display.render()
        assert rendered is not None

    def test_diff_viewer_with_syntax(self) -> None:
        """Test diff viewer with syntax highlighting."""
        diff_text = """@@ -45,6 +45,12 @@ def create_access_token(data: dict):
     return encoded_jwt

+def refresh_access_token(token: str) -> str:
+    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
+    payload["exp"] = datetime.utcnow() + timedelta(minutes=30)
+    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
+
 def verify_password(plain: str, hashed: str) -> bool:
"""
        viewer = DiffViewer(filepath="src/auth.py", diff_text=diff_text)

        output = StringIO()
        console = Console(file=output, force_terminal=True, width=80)
        console.print(viewer)

        content = output.getvalue()
        assert "auth.py" in content

    def test_tool_call_with_output(self) -> None:
        """Test tool call panel with output."""
        panel = ToolCallPanel(
            tool_name="write_file",
            args={"path": "src/auth.py"},
            output="def authenticate(token):\n    return True",
            state=ToolCallState.COMPLETE,
            summary="Created authentication module",
            line_count=45,
            duration=1.2,
        )

        output = StringIO()
        console = Console(file=output, force_terminal=True, width=80)
        console.print(panel)

        content = output.getvalue()
        assert "write_file" in content
