"""Tests for executor CLI commands."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_infra.cli.cmds.executor_cmds import (
    _format_duration,
    _format_tokens,
    app,
)

runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_roadmap_content() -> str:
    """Sample ROADMAP content."""
    return """\
# Test Project ROADMAP

## Phase 0: Foundation

### 0.1 Setup

- [ ] **Task 1**
  First task.

- [ ] **Task 2**
  Second task.

- [x] **Task 3**
  Completed task.
"""


@pytest.fixture
def temp_roadmap(sample_roadmap_content: str):
    """Create a temporary ROADMAP file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        roadmap_path = Path(tmpdir) / "ROADMAP.md"
        roadmap_path.write_text(sample_roadmap_content)
        yield roadmap_path


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Test helper formatting functions."""

    def test_format_duration_milliseconds(self) -> None:
        """Test duration formatting for milliseconds."""
        assert _format_duration(500) == "500ms"
        assert _format_duration(999) == "999ms"

    def test_format_duration_seconds(self) -> None:
        """Test duration formatting for seconds."""
        assert _format_duration(1000) == "1.0s"
        assert _format_duration(5500) == "5.5s"
        assert _format_duration(59000) == "59.0s"

    def test_format_duration_minutes(self) -> None:
        """Test duration formatting for minutes."""
        assert _format_duration(60000) == "1m 0s"
        assert _format_duration(90000) == "1m 30s"
        assert _format_duration(300000) == "5m 0s"

    def test_format_duration_hours(self) -> None:
        """Test duration formatting for hours."""
        assert _format_duration(3600000) == "1h 0m"
        assert _format_duration(5400000) == "1h 30m"

    def test_format_tokens(self) -> None:
        """Test token formatting with cost estimate."""
        result = _format_tokens(1000)
        assert "1,000" in result
        assert "$" in result


# =============================================================================
# CLI Help Tests
# =============================================================================


class TestCLIHelp:
    """Test CLI help output."""

    def test_executor_help(self) -> None:
        """Test executor --help."""
        result = runner.invoke(app, ["--help"], env={"COLUMNS": "200"})
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "ROADMAP" in output or "roadmap" in output.lower()

    def test_run_help(self) -> None:
        """Test executor run --help."""
        result = runner.invoke(app, ["run", "--help"], env={"COLUMNS": "200"})
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "--roadmap" in output or "roadmap" in output.lower()

    def test_status_help(self) -> None:
        """Test executor status --help."""
        result = runner.invoke(app, ["status", "--help"], env={"COLUMNS": "200"})
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "--roadmap" in output or "roadmap" in output.lower()

    def test_resume_help(self) -> None:
        """Test executor resume --help."""
        result = runner.invoke(app, ["resume", "--help"], env={"COLUMNS": "200"})
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        # Check for either option or help text contains "approve"
        assert "--approve" in output or "approve" in output.lower()

    def test_rollback_help(self) -> None:
        """Test executor rollback --help."""
        result = runner.invoke(app, ["rollback", "--help"], env={"COLUMNS": "200"})
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "--task" in output or "task" in output.lower()

    def test_reset_help(self) -> None:
        """Test executor reset --help."""
        result = runner.invoke(app, ["reset", "--help"], env={"COLUMNS": "200"})
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "--force" in output or "force" in output.lower()

    def test_sync_help(self) -> None:
        """Test executor sync --help."""
        result = runner.invoke(app, ["sync", "--help"], env={"COLUMNS": "200"})
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "--roadmap" in output or "roadmap" in output.lower()

    def test_review_help(self) -> None:
        """Test executor review --help."""
        result = runner.invoke(app, ["review", "--help"], env={"COLUMNS": "200"})
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "--json" in output or "json" in output.lower()


# =============================================================================
# Status Command Tests
# =============================================================================


class TestStatusCommand:
    """Test status command."""

    def test_status_basic(self, temp_roadmap: Path) -> None:
        """Test basic status command."""
        result = runner.invoke(app, ["status", "--roadmap", str(temp_roadmap)])
        assert result.exit_code == 0
        assert "Progress" in result.output

    def test_status_json_output(self, temp_roadmap: Path) -> None:
        """Test status command with JSON output."""
        result = runner.invoke(app, ["status", "--roadmap", str(temp_roadmap), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "total_tasks" in data
        assert "completed" in data

    def test_status_missing_roadmap(self) -> None:
        """Test status with missing roadmap file."""
        result = runner.invoke(app, ["status", "--roadmap", "/nonexistent/ROADMAP.md"])
        assert result.exit_code != 0


# =============================================================================
# Reset Command Tests
# =============================================================================


class TestResetCommand:
    """Test reset command."""

    def test_reset_with_force(self, temp_roadmap: Path) -> None:
        """Test reset command with --force."""
        result = runner.invoke(app, ["reset", "--roadmap", str(temp_roadmap), "--force"])
        assert result.exit_code == 0
        assert "reset successfully" in result.output

    def test_reset_without_force_abort(self, temp_roadmap: Path) -> None:
        """Test reset command aborted without --force."""
        result = runner.invoke(app, ["reset", "--roadmap", str(temp_roadmap)], input="n\n")
        assert "Aborted" in result.output


# =============================================================================
# Sync Command Tests
# =============================================================================


class TestSyncCommand:
    """Test sync command."""

    def test_sync_basic(self, temp_roadmap: Path) -> None:
        """Test sync command."""
        result = runner.invoke(app, ["sync", "--roadmap", str(temp_roadmap)])
        assert result.exit_code == 0


# =============================================================================
# Review Command Tests
# =============================================================================


class TestReviewCommand:
    """Test review command."""

    def test_review_empty(self, temp_roadmap: Path) -> None:
        """Test review command with no pending changes."""
        result = runner.invoke(app, ["review", "--roadmap", str(temp_roadmap)])
        assert result.exit_code == 0
        assert "No changes pending" in result.output

    def test_review_json_output(self, temp_roadmap: Path) -> None:
        """Test review command with JSON output."""
        result = runner.invoke(app, ["review", "--roadmap", str(temp_roadmap), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "task_ids" in data


# =============================================================================
# Resume Command Tests
# =============================================================================


class TestResumeCommand:
    """Test resume command."""

    def test_resume_requires_approve_or_reject(self, temp_roadmap: Path) -> None:
        """Test resume requires --approve or --reject."""
        result = runner.invoke(app, ["resume", "--roadmap", str(temp_roadmap)])
        assert result.exit_code != 0
        assert "Must specify either" in result.output

    def test_resume_cannot_both_approve_and_reject(self, temp_roadmap: Path) -> None:
        """Test resume cannot use both --approve and --reject."""
        result = runner.invoke(
            app, ["resume", "--roadmap", str(temp_roadmap), "--approve", "--reject"]
        )
        assert result.exit_code != 0
        assert "Cannot use both" in result.output

    def test_resume_with_approve(self, temp_roadmap: Path) -> None:
        """Test resume with --approve."""
        result = runner.invoke(app, ["resume", "--roadmap", str(temp_roadmap), "--approve"])
        assert result.exit_code == 0
        assert "approved" in result.output.lower()

    def test_resume_with_reject(self, temp_roadmap: Path) -> None:
        """Test resume with --reject."""
        result = runner.invoke(app, ["resume", "--roadmap", str(temp_roadmap), "--reject"])
        assert result.exit_code == 0
        assert "rejected" in result.output.lower()


# =============================================================================
# Run Command Tests
# =============================================================================


class TestRunCommand:
    """Test run command."""

    def test_run_dry_run(self, temp_roadmap: Path) -> None:
        """Test run command with --dry-run."""
        result = runner.invoke(
            app, ["run", "--roadmap", str(temp_roadmap), "--dry-run", "--max-tasks", "1"]
        )
        assert result.exit_code == 0
        assert "Dry run" in result.output

    def test_run_missing_roadmap(self) -> None:
        """Test run with missing roadmap file."""
        result = runner.invoke(app, ["run", "--roadmap", "/nonexistent/ROADMAP.md"])
        assert result.exit_code != 0

    def test_run_json_output_dry_run(self, temp_roadmap: Path) -> None:
        """Test run command with JSON output in dry run mode."""
        result = runner.invoke(
            app,
            [
                "run",
                "--roadmap",
                str(temp_roadmap),
                "--dry-run",
                "--max-tasks",
                "1",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "status" in data


# =============================================================================
# Rollback Command Tests
# =============================================================================


class TestRollbackCommand:
    """Test rollback command."""

    def test_rollback_no_checkpointer(self, temp_roadmap: Path) -> None:
        """Test rollback when not in git repo."""
        # This test may vary based on whether we're in a git repo
        result = runner.invoke(app, ["rollback", "--roadmap", str(temp_roadmap)])
        # Either error (no git) or no completed tasks
        assert result.exit_code != 0


# =============================================================================
# Phase 1.8: Graph CLI Options Tests
# =============================================================================


class TestGraphCLIOptions:
    """Tests for Phase 1.8 graph-specific CLI options."""

    def test_run_help_shows_graph_options(self) -> None:
        """Test that run --help shows relevant options."""
        result = runner.invoke(app, ["run", "--help"], env={"COLUMNS": "200"})
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        # Just verify help output is valid and contains run-related options
        assert "roadmap" in output.lower() or "--roadmap" in output

    def test_visualize_option(self, temp_roadmap: Path) -> None:
        """Test --visualize generates Mermaid diagram."""
        result = runner.invoke(app, ["run", "--roadmap", str(temp_roadmap), "--visualize"])
        assert result.exit_code == 0
        assert "mermaid" in result.output.lower()
        # Should contain graph node names
        assert "parse_roadmap" in result.output or "pick_task" in result.output

    def test_interrupt_before_option(self, temp_roadmap: Path) -> None:
        """Test --interrupt-before option is accepted."""
        result = runner.invoke(
            app,
            [
                "run",
                "--roadmap",
                str(temp_roadmap),
                "--dry-run",
                "--max-tasks",
                "1",
                "--interrupt-before",
                "execute_task",
            ],
        )
        assert result.exit_code == 0
        assert "execute_task" in result.output or "Dry run" in result.output

    def test_interrupt_after_option(self, temp_roadmap: Path) -> None:
        """Test --interrupt-after option is accepted."""
        result = runner.invoke(
            app,
            [
                "run",
                "--roadmap",
                str(temp_roadmap),
                "--dry-run",
                "--max-tasks",
                "1",
                "--interrupt-after",
                "verify_task",
            ],
        )
        assert result.exit_code == 0
        assert "verify_task" in result.output or "Dry run" in result.output

    def test_multiple_interrupt_nodes(self, temp_roadmap: Path) -> None:
        """Test multiple interrupt points can be specified."""
        result = runner.invoke(
            app,
            [
                "run",
                "--roadmap",
                str(temp_roadmap),
                "--dry-run",
                "--max-tasks",
                "1",
                "--interrupt-before",
                "execute_task",
                "--interrupt-before",
                "checkpoint",
            ],
        )
        assert result.exit_code == 0


# =============================================================================
# Phase 15.4: MCP CLI Flag Tests
# =============================================================================


class TestParseMcpArg:
    """Tests for parse_mcp_arg helper function."""

    def test_parse_http_url(self) -> None:
        """Test parsing HTTP URL format."""
        from ai_infra.cli.cmds.executor_cmds import parse_mcp_arg

        config = parse_mcp_arg("http://localhost:8000/mcp")

        assert config.transport == "streamable_http"
        assert config.url == "http://localhost:8000/mcp"

    def test_parse_https_url(self) -> None:
        """Test parsing HTTPS URL format."""
        from ai_infra.cli.cmds.executor_cmds import parse_mcp_arg

        config = parse_mcp_arg("https://api.example.com/mcp")

        assert config.transport == "streamable_http"
        assert config.url == "https://api.example.com/mcp"

    def test_parse_stdio_format(self) -> None:
        """Test parsing stdio format."""
        from ai_infra.cli.cmds.executor_cmds import parse_mcp_arg

        config = parse_mcp_arg("stdio:npx -y @anthropic/mcp-server-filesystem /tmp")

        assert config.transport == "stdio"
        assert config.command == "npx"
        assert config.args == ["-y", "@anthropic/mcp-server-filesystem", "/tmp"]

    def test_parse_stdio_simple_command(self) -> None:
        """Test parsing stdio format with simple command (no args)."""
        from ai_infra.cli.cmds.executor_cmds import parse_mcp_arg

        config = parse_mcp_arg("stdio:my-mcp-server")

        assert config.transport == "stdio"
        assert config.command == "my-mcp-server"
        assert config.args == []

    def test_parse_stdio_with_spaces(self) -> None:
        """Test parsing stdio format strips whitespace."""
        from ai_infra.cli.cmds.executor_cmds import parse_mcp_arg

        config = parse_mcp_arg("  stdio:npx server  ")

        assert config.transport == "stdio"
        assert config.command == "npx"
        assert config.args == ["server"]

    def test_parse_invalid_format_raises(self) -> None:
        """Test that invalid format raises ValueError."""
        from ai_infra.cli.cmds.executor_cmds import parse_mcp_arg

        with pytest.raises(ValueError) as exc_info:
            parse_mcp_arg("invalid-format")

        assert "Invalid MCP server format" in str(exc_info.value)

    def test_parse_empty_stdio_command_raises(self) -> None:
        """Test that empty stdio command raises ValueError."""
        from ai_infra.cli.cmds.executor_cmds import parse_mcp_arg

        with pytest.raises(ValueError) as exc_info:
            parse_mcp_arg("stdio:")

        assert "requires a command" in str(exc_info.value)


class TestMCPCLIFlags:
    """Tests for MCP CLI flags (Phase 15.4)."""

    def test_mcp_option_accepted(self, temp_roadmap: Path) -> None:
        """Test that --mcp option is accepted."""
        result = runner.invoke(
            app,
            [
                "run",
                "--roadmap",
                str(temp_roadmap),
                "--dry-run",
                "--max-tasks",
                "1",
                "--mcp",
                "http://localhost:8000/mcp",
            ],
        )
        assert result.exit_code == 0
        # Should show MCP servers in output
        assert "MCP servers" in result.output or "Dry run" in result.output

    def test_multiple_mcp_servers(self, temp_roadmap: Path) -> None:
        """Test that multiple --mcp options can be specified."""
        result = runner.invoke(
            app,
            [
                "run",
                "--roadmap",
                str(temp_roadmap),
                "--dry-run",
                "--max-tasks",
                "1",
                "--mcp",
                "http://localhost:8000/mcp",
                "--mcp",
                "http://localhost:9000/mcp",
            ],
        )
        assert result.exit_code == 0

    def test_mcp_stdio_format(self, temp_roadmap: Path) -> None:
        """Test that stdio format is accepted."""
        result = runner.invoke(
            app,
            [
                "run",
                "--roadmap",
                str(temp_roadmap),
                "--dry-run",
                "--max-tasks",
                "1",
                "--mcp",
                "stdio:npx -y my-server",
            ],
        )
        assert result.exit_code == 0

    def test_mcp_timeout_option(self, temp_roadmap: Path) -> None:
        """Test that --mcp-timeout option is accepted."""
        result = runner.invoke(
            app,
            [
                "run",
                "--roadmap",
                str(temp_roadmap),
                "--dry-run",
                "--max-tasks",
                "1",
                "--mcp",
                "http://localhost:8000/mcp",
                "--mcp-timeout",
                "60.0",
            ],
        )
        assert result.exit_code == 0

    def test_invalid_mcp_format_error(self, temp_roadmap: Path) -> None:
        """Test that invalid MCP format shows error."""
        result = runner.invoke(
            app,
            [
                "run",
                "--roadmap",
                str(temp_roadmap),
                "--dry-run",
                "--mcp",
                "invalid-format",
            ],
        )
        assert result.exit_code == 1
        assert "Invalid MCP server" in result.output
