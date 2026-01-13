"""Tests for Phase 2.1 & 2.2: Shell Tool Integration in ExecutorGraph.

This module tests the shell tool integration with the executor:
- Shell configuration parameters (enable_shell, shell_timeout, shell_workspace)
- Shell session lifecycle (start/close in arun/astream)
- Shell tool addition to agent
- State fields for shell results
- ShellError type (Phase 2.2)
- ExecutorErrorType.SHELL (Phase 2.2)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_infra.executor.graph import ExecutorGraph
from ai_infra.executor.state import (
    ExecutorErrorType,
    ExecutorGraphState,
    ShellError,
)
from ai_infra.llm.shell.tool import get_current_session, set_current_session


class TestShellConfiguration:
    """Tests for shell configuration in ExecutorGraph."""

    def test_shell_enabled_by_default(self, tmp_path: Path) -> None:
        """Shell should be enabled by default."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(roadmap_path=str(roadmap))

        assert executor.enable_shell is True

    def test_shell_can_be_disabled(self, tmp_path: Path) -> None:
        """Shell can be disabled via enable_shell=False."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            enable_shell=False,
        )

        assert executor.enable_shell is False

    def test_default_shell_timeout(self, tmp_path: Path) -> None:
        """Default shell timeout should be 120 seconds."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(roadmap_path=str(roadmap))

        assert executor.shell_timeout == 120.0

    def test_custom_shell_timeout(self, tmp_path: Path) -> None:
        """Custom shell timeout can be set."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            shell_timeout=60.0,
        )

        assert executor.shell_timeout == 60.0

    def test_default_shell_workspace(self, tmp_path: Path) -> None:
        """Default shell workspace should be roadmap directory."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(roadmap_path=str(roadmap))

        assert executor.shell_workspace == tmp_path

    def test_custom_shell_workspace(self, tmp_path: Path) -> None:
        """Custom shell workspace can be set."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")
        custom_workspace = tmp_path / "workspace"
        custom_workspace.mkdir()

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            shell_workspace=str(custom_workspace),
        )

        assert executor.shell_workspace == custom_workspace

    def test_shell_session_initially_none(self, tmp_path: Path) -> None:
        """Shell session should be None before arun/astream."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(roadmap_path=str(roadmap))

        assert executor._shell_session is None


class TestInitialState:
    """Tests for shell fields in initial state."""

    def test_initial_state_has_enable_shell(self, tmp_path: Path) -> None:
        """Initial state should include enable_shell field."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(roadmap_path=str(roadmap))
        state = executor.get_initial_state()

        assert "enable_shell" in state
        assert state["enable_shell"] is True

    def test_initial_state_shell_disabled(self, tmp_path: Path) -> None:
        """Initial state should reflect enable_shell=False."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            enable_shell=False,
        )
        state = executor.get_initial_state()

        assert state["enable_shell"] is False

    def test_initial_state_has_empty_shell_results(self, tmp_path: Path) -> None:
        """Initial state should have empty shell_results list."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(roadmap_path=str(roadmap))
        state = executor.get_initial_state()

        assert "shell_results" in state
        assert state["shell_results"] == []


class TestShellSessionLifecycle:
    """Tests for shell session lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_shell_session_creates_session(self, tmp_path: Path) -> None:
        """_start_shell_session should create a ShellSession."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(roadmap_path=str(roadmap))
        assert executor._shell_session is None

        await executor._start_shell_session()

        assert executor._shell_session is not None

        # Cleanup
        await executor._close_shell_session()

    @pytest.mark.asyncio
    async def test_close_shell_session_cleans_up(self, tmp_path: Path) -> None:
        """_close_shell_session should close and clear the session."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(roadmap_path=str(roadmap))
        await executor._start_shell_session()
        assert executor._shell_session is not None

        await executor._close_shell_session()

        assert executor._shell_session is None

    @pytest.mark.asyncio
    async def test_shell_session_adds_tool_to_agent(self, tmp_path: Path) -> None:
        """Starting shell session should add run_shell tool to agent."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        # Create mock agent with empty tools
        mock_agent = MagicMock()
        mock_agent.tools = []

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            agent=mock_agent,
        )

        await executor._start_shell_session()

        # Tool should be added to agent
        assert len(mock_agent.tools) == 1
        tool = mock_agent.tools[0]
        assert hasattr(tool, "name")
        # The tool is named "configured_run_shell" when created via create_shell_tool
        assert tool.name == "configured_run_shell"

        # Cleanup
        await executor._close_shell_session()

    @pytest.mark.asyncio
    async def test_shell_tool_not_added_when_disabled(self, tmp_path: Path) -> None:
        """Shell tool should not be added when enable_shell=False."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        mock_agent = MagicMock()
        mock_agent.tools = []

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            agent=mock_agent,
            enable_shell=False,
        )

        # Shell session should not start when disabled
        assert executor._shell_session is None
        assert len(mock_agent.tools) == 0

    @pytest.mark.asyncio
    async def test_shell_tool_not_duplicated(self, tmp_path: Path) -> None:
        """Shell tool should not be added if already present."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        # Create mock agent with existing run_shell tool (either name variant)
        existing_tool = MagicMock()
        existing_tool.name = "configured_run_shell"
        mock_agent = MagicMock()
        mock_agent.tools = [existing_tool]

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            agent=mock_agent,
        )

        await executor._start_shell_session()

        # Should still have only one tool
        assert len(mock_agent.tools) == 1

        # Cleanup
        await executor._close_shell_session()


class TestArunShellIntegration:
    """Tests for shell integration in arun method."""

    @pytest.mark.asyncio
    async def test_arun_starts_and_closes_shell_session(self, tmp_path: Path) -> None:
        """arun should start shell session before running and close after."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(roadmap_path=str(roadmap))

        # Mock the graph.arun to avoid actual execution
        mock_result = {"tasks_completed_count": 0}
        executor.graph.arun = AsyncMock(return_value=mock_result)

        # Track session state
        session_during_run = None

        async def capture_session(state, config=None):
            nonlocal session_during_run
            session_during_run = executor._shell_session
            return mock_result

        executor.graph.arun = capture_session

        # Ensure session is None before
        assert executor._shell_session is None

        await executor.arun()

        # Session should have been active during run
        assert session_during_run is not None
        # Session should be None after run (cleaned up)
        assert executor._shell_session is None

    @pytest.mark.asyncio
    async def test_arun_sets_context_variable(self, tmp_path: Path) -> None:
        """arun should set the shell session in context variable."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(roadmap_path=str(roadmap))

        # Track context variable
        context_session = None

        async def capture_context(state, config=None):
            nonlocal context_session
            context_session = get_current_session()
            return {"tasks_completed_count": 0}

        executor.graph.arun = capture_context

        await executor.arun()

        # Context variable should have been set during run
        assert context_session is not None
        # Context should be cleared after run
        assert get_current_session() is None

    @pytest.mark.asyncio
    async def test_arun_cleans_up_on_error(self, tmp_path: Path) -> None:
        """arun should clean up shell session even on error."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(roadmap_path=str(roadmap))

        # Mock graph.arun to raise an exception
        async def raise_error(state, config=None):
            raise RuntimeError("Test error")

        executor.graph.arun = raise_error

        with pytest.raises(RuntimeError, match="Test error"):
            await executor.arun()

        # Session should be cleaned up even on error
        assert executor._shell_session is None
        assert get_current_session() is None

    @pytest.mark.asyncio
    async def test_arun_skips_shell_when_disabled(self, tmp_path: Path) -> None:
        """arun should not start shell session when disabled."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            enable_shell=False,
        )

        session_checked = False

        async def check_no_session(state, config=None):
            nonlocal session_checked
            session_checked = True
            assert executor._shell_session is None
            assert get_current_session() is None
            return {"tasks_completed_count": 0}

        executor.graph.arun = check_no_session

        await executor.arun()

        assert session_checked


class TestAstreamShellIntegration:
    """Tests for shell integration in astream method."""

    @pytest.mark.asyncio
    async def test_astream_manages_shell_session(self, tmp_path: Path) -> None:
        """astream should manage shell session lifecycle."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(roadmap_path=str(roadmap))

        # Mock the graph.astream to yield a single event
        async def mock_astream(state, config=None):
            # Session should be active during streaming
            assert executor._shell_session is not None
            assert get_current_session() is not None
            yield {"test_node": {"key": "value"}}

        executor.graph.astream = mock_astream

        # Consume the stream
        events = []
        async for node_name, node_state in executor.astream():
            events.append((node_name, node_state))

        # Session should be cleaned up after streaming
        assert executor._shell_session is None
        assert get_current_session() is None
        assert len(events) == 1


class TestExecutorGraphStateShellFields:
    """Tests for shell-related fields in ExecutorGraphState."""

    def test_state_accepts_enable_shell(self) -> None:
        """ExecutorGraphState should accept enable_shell field."""
        state = ExecutorGraphState(
            roadmap_path="/test/ROADMAP.md",
            enable_shell=True,
        )
        assert state["enable_shell"] is True

    def test_state_accepts_shell_results(self) -> None:
        """ExecutorGraphState should accept shell_results field."""
        shell_result = {
            "command": "echo hello",
            "exit_code": 0,
            "stdout": "hello\n",
            "stderr": "",
            "duration_ms": 50,
        }
        state = ExecutorGraphState(
            roadmap_path="/test/ROADMAP.md",
            shell_results=[shell_result],
        )
        assert len(state["shell_results"]) == 1
        assert state["shell_results"][0]["command"] == "echo hello"


# =============================================================================
# Phase 2.2 Tests: Executor State Updates
# =============================================================================


class TestShellErrorType:
    """Tests for ShellError TypedDict (Phase 2.2.2)."""

    def test_shell_error_basic_creation(self) -> None:
        """ShellError should accept all expected fields."""
        error: ShellError = {
            "command": "npm install",
            "exit_code": 1,
            "stderr": "npm ERR! code ERESOLVE",
            "stdout": "",
            "cwd": "/project",
            "timed_out": False,
        }

        assert error["command"] == "npm install"
        assert error["exit_code"] == 1
        assert error["stderr"] == "npm ERR! code ERESOLVE"
        assert error["stdout"] == ""
        assert error["cwd"] == "/project"
        assert error["timed_out"] is False

    def test_shell_error_timeout(self) -> None:
        """ShellError should capture timeout scenarios."""
        error: ShellError = {
            "command": "long_running_process",
            "exit_code": -1,
            "stderr": "",
            "stdout": "partial output...",
            "cwd": None,
            "timed_out": True,
        }

        assert error["timed_out"] is True
        assert error["exit_code"] == -1

    def test_shell_error_partial(self) -> None:
        """ShellError should work with partial fields (total=False)."""
        error: ShellError = {
            "command": "echo test",
            "exit_code": 0,
        }

        assert error["command"] == "echo test"
        assert error["exit_code"] == 0
        # Other fields not required


class TestExecutorErrorTypeShell:
    """Tests for ExecutorErrorType.SHELL (Phase 2.2.2)."""

    def test_shell_error_type_exists(self) -> None:
        """ExecutorErrorType should have SHELL constant."""
        assert hasattr(ExecutorErrorType, "SHELL")
        assert ExecutorErrorType.SHELL == "shell"

    def test_all_error_types_present(self) -> None:
        """All expected error types should be present."""
        expected = [
            "EXECUTION",
            "VERIFICATION",
            "VALIDATION",
            "TIMEOUT",
            "ROLLBACK",
            "PARSE",
            "CONTEXT",
            "SHELL",
        ]
        for error_type in expected:
            assert hasattr(ExecutorErrorType, error_type)


class TestExecutorGraphStatePhase22:
    """Tests for Phase 2.2 state fields."""

    def test_state_accepts_shell_session_active(self) -> None:
        """ExecutorGraphState should accept shell_session_active field."""
        state = ExecutorGraphState(
            roadmap_path="/test/ROADMAP.md",
            shell_session_active=True,
        )
        assert state["shell_session_active"] is True

    def test_state_accepts_shell_error(self) -> None:
        """ExecutorGraphState should accept shell_error field."""
        error: ShellError = {
            "command": "pytest",
            "exit_code": 1,
            "stderr": "tests failed",
            "stdout": "",
            "cwd": "/project",
            "timed_out": False,
        }
        state = ExecutorGraphState(
            roadmap_path="/test/ROADMAP.md",
            shell_error=error,
        )
        assert state["shell_error"] is not None
        assert state["shell_error"]["command"] == "pytest"
        assert state["shell_error"]["exit_code"] == 1

    def test_state_shell_error_none(self) -> None:
        """ExecutorGraphState should accept shell_error=None."""
        state = ExecutorGraphState(
            roadmap_path="/test/ROADMAP.md",
            shell_error=None,
        )
        assert state["shell_error"] is None

    def test_initial_state_has_shell_session_active(self, tmp_path: Path) -> None:
        """Initial state should have shell_session_active=False."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(roadmap_path=str(roadmap))
        state = executor.get_initial_state()

        assert "shell_session_active" in state
        assert state["shell_session_active"] is False

    def test_initial_state_has_shell_error_none(self, tmp_path: Path) -> None:
        """Initial state should have shell_error=None."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(roadmap_path=str(roadmap))
        state = executor.get_initial_state()

        assert "shell_error" in state
        assert state["shell_error"] is None


# =============================================================================
# Phase 2.3 Tests: Execute Node Shell Integration
# =============================================================================


class TestShellSessionHistory:
    """Tests for ShellSession command history (Phase 2.3.2)."""

    @pytest.mark.asyncio
    async def test_session_tracks_command_history(self) -> None:
        """ShellSession should track executed commands in history."""
        from ai_infra.llm.shell.session import SessionConfig, ShellSession

        session = ShellSession(SessionConfig())
        await session.start()

        try:
            # Execute a command
            await session.execute("echo hello")

            # Verify history is tracked
            assert len(session.command_history) == 1
            assert session.command_history[0].command == "echo hello"
            assert session.command_history[0].success is True
        finally:
            await session.close()

    @pytest.mark.asyncio
    async def test_session_history_multiple_commands(self) -> None:
        """ShellSession should track multiple commands in order."""
        from ai_infra.llm.shell.session import SessionConfig, ShellSession

        session = ShellSession(SessionConfig())
        await session.start()

        try:
            await session.execute("echo one")
            await session.execute("echo two")
            await session.execute("echo three")

            assert len(session.command_history) == 3
            assert session.command_history[0].command == "echo one"
            assert session.command_history[1].command == "echo two"
            assert session.command_history[2].command == "echo three"
        finally:
            await session.close()

    @pytest.mark.asyncio
    async def test_session_clear_history(self) -> None:
        """ShellSession.clear_history() should clear command history."""
        from ai_infra.llm.shell.session import SessionConfig, ShellSession

        session = ShellSession(SessionConfig())
        await session.start()

        try:
            await session.execute("echo test")
            assert len(session.command_history) == 1

            session.clear_history()
            assert len(session.command_history) == 0
        finally:
            await session.close()

    @pytest.mark.asyncio
    async def test_session_history_captures_failures(self) -> None:
        """ShellSession should capture failed commands in history."""
        from ai_infra.llm.shell.session import SessionConfig, ShellSession

        session = ShellSession(SessionConfig())
        await session.start()

        try:
            # Execute a failing command (false always returns exit 1)
            await session.execute("false")

            assert len(session.command_history) == 1
            assert session.command_history[0].success is False
            assert session.command_history[0].exit_code == 1
        finally:
            await session.close()


class TestExtractShellResults:
    """Tests for _extract_shell_results helper (Phase 2.3.2)."""

    def test_extract_shell_results_no_session(self) -> None:
        """_extract_shell_results returns empty when no session."""
        from ai_infra.executor.nodes.execute import _extract_shell_results

        # Ensure no session is set
        set_current_session(None)

        results, error = _extract_shell_results()

        assert results == []
        assert error is None

    @pytest.mark.asyncio
    async def test_extract_shell_results_with_session(self) -> None:
        """_extract_shell_results extracts results from session history."""
        from ai_infra.executor.nodes.execute import _extract_shell_results
        from ai_infra.llm.shell.session import SessionConfig, ShellSession

        session = ShellSession(SessionConfig())
        await session.start()

        try:
            set_current_session(session)

            await session.execute("echo hello")

            results, error = _extract_shell_results()

            assert len(results) == 1
            assert results[0]["command"] == "echo hello"
            assert results[0]["exit_code"] == 0
            assert error is None
        finally:
            set_current_session(None)
            await session.close()

    @pytest.mark.asyncio
    async def test_extract_shell_results_captures_error(self) -> None:
        """_extract_shell_results captures shell_error on failure."""
        from ai_infra.executor.nodes.execute import _extract_shell_results
        from ai_infra.llm.shell.session import SessionConfig, ShellSession

        session = ShellSession(SessionConfig())
        await session.start()

        try:
            set_current_session(session)

            await session.execute("false")  # Always returns exit 1

            results, error = _extract_shell_results()

            assert len(results) == 1
            assert results[0]["exit_code"] == 1
            assert error is not None
            assert error["exit_code"] == 1
            assert error["timed_out"] is False
        finally:
            set_current_session(None)
            await session.close()


class TestExecuteNodeShellIntegration:
    """Tests for execute_task_node shell integration (Phase 2.3)."""

    @pytest.mark.asyncio
    async def test_execute_node_sets_shell_session_active(self) -> None:
        """execute_task_node should set shell_session_active in state."""
        from ai_infra.executor.nodes.execute import execute_task_node
        from ai_infra.llm.shell.session import SessionConfig, ShellSession

        # Create a mock agent
        mock_agent = MagicMock()
        mock_agent.arun = AsyncMock(return_value="Task completed")

        # Create initial state with prompt
        state = ExecutorGraphState(
            roadmap_path="/test/ROADMAP.md",
            prompt="Do something",
            current_task=MagicMock(id="1.1", title="Test task"),
            shell_results=[],
        )

        # Set up shell session
        session = ShellSession(SessionConfig())
        await session.start()

        try:
            set_current_session(session)

            result = await execute_task_node(state, agent=mock_agent)

            assert result["shell_session_active"] is True
        finally:
            set_current_session(None)
            await session.close()

    @pytest.mark.asyncio
    async def test_execute_node_accumulates_shell_results(self) -> None:
        """execute_task_node should accumulate shell results in state."""
        from ai_infra.executor.nodes.execute import execute_task_node
        from ai_infra.llm.shell.session import SessionConfig, ShellSession

        mock_agent = MagicMock()

        async def mock_arun(prompt):
            # Simulate agent using shell
            session = get_current_session()
            await session.execute("echo from agent")
            return "Done"

        mock_agent.arun = mock_arun

        state = ExecutorGraphState(
            roadmap_path="/test/ROADMAP.md",
            prompt="Run shell command",
            current_task=MagicMock(id="1.1", title="Test task"),
            shell_results=[{"command": "previous", "exit_code": 0}],
        )

        session = ShellSession(SessionConfig())
        await session.start()

        try:
            set_current_session(session)

            result = await execute_task_node(state, agent=mock_agent)

            # Should have both previous and new results
            assert len(result["shell_results"]) == 2
            assert result["shell_results"][0]["command"] == "previous"
            assert result["shell_results"][1]["command"] == "echo from agent"
        finally:
            set_current_session(None)
            await session.close()

    @pytest.mark.asyncio
    async def test_execute_node_handles_shell_failure_gracefully(self) -> None:
        """execute_task_node should not fail task on shell command failure."""
        from ai_infra.executor.nodes.execute import execute_task_node
        from ai_infra.llm.shell.session import SessionConfig, ShellSession

        mock_agent = MagicMock()

        async def mock_arun(prompt):
            # Simulate agent using shell with a failing command
            session = get_current_session()
            await session.execute("false")  # Always returns exit 1
            return "Agent handled the failure"

        mock_agent.arun = mock_arun

        state = ExecutorGraphState(
            roadmap_path="/test/ROADMAP.md",
            prompt="Run failing shell command",
            current_task=MagicMock(id="1.1", title="Test task"),
            shell_results=[],
        )

        session = ShellSession(SessionConfig())
        await session.start()

        try:
            set_current_session(session)

            result = await execute_task_node(state, agent=mock_agent)

            # Task should succeed even though shell command failed
            assert result["error"] is None
            assert result["agent_result"] == "Agent handled the failure"

            # Shell error should be captured
            assert result["shell_error"] is not None
            assert result["shell_error"]["exit_code"] == 1
        finally:
            set_current_session(None)
            await session.close()

    @pytest.mark.asyncio
    async def test_execute_node_no_shell_error_on_success(self) -> None:
        """execute_task_node should have shell_error=None on success."""
        from ai_infra.executor.nodes.execute import execute_task_node
        from ai_infra.llm.shell.session import SessionConfig, ShellSession

        mock_agent = MagicMock()

        async def mock_arun(prompt):
            session = get_current_session()
            await session.execute("echo success")
            return "Done"

        mock_agent.arun = mock_arun

        state = ExecutorGraphState(
            roadmap_path="/test/ROADMAP.md",
            prompt="Run successful command",
            current_task=MagicMock(id="1.1", title="Test task"),
            shell_results=[],
        )

        session = ShellSession(SessionConfig())
        await session.start()

        try:
            set_current_session(session)

            result = await execute_task_node(state, agent=mock_agent)

            assert result["shell_error"] is None
        finally:
            set_current_session(None)
            await session.close()


# =============================================================================
# Phase 2.4 Tests: CLI Flag Integration
# =============================================================================


class TestShellAllowedCommandsConfig:
    """Tests for shell_allowed_commands configuration (Phase 2.4.3)."""

    def test_executor_graph_accepts_shell_allowed_commands(self, tmp_path: Path) -> None:
        """ExecutorGraph should accept shell_allowed_commands parameter."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            shell_allowed_commands=("pytest", "npm", "make"),
        )

        assert executor.shell_allowed_commands == ("pytest", "npm", "make")

    def test_executor_graph_shell_allowed_commands_default_none(self, tmp_path: Path) -> None:
        """ExecutorGraph should have shell_allowed_commands=None by default."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(roadmap_path=str(roadmap))

        assert executor.shell_allowed_commands is None


class TestShellToolAllowlist:
    """Tests for shell tool allowlist functionality (Phase 2.4.3)."""

    @pytest.mark.asyncio
    async def test_allowed_command_succeeds(self) -> None:
        """Allowed command should execute successfully."""
        from ai_infra.llm.shell.tool import create_shell_tool

        shell_tool = create_shell_tool(
            allowed_commands=("echo", "ls", "pwd"),
        )

        # Get the async function
        tool_fn = shell_tool.coroutine if hasattr(shell_tool, "coroutine") else shell_tool

        result = await tool_fn(command="echo hello")

        assert result["success"] is True
        assert result["exit_code"] == 0
        assert "hello" in result["stdout"]

    @pytest.mark.asyncio
    async def test_disallowed_command_rejected(self) -> None:
        """Command not in allowlist should be rejected."""
        from ai_infra.llm.shell.tool import create_shell_tool

        shell_tool = create_shell_tool(
            allowed_commands=("pytest", "npm"),
        )

        tool_fn = shell_tool.coroutine if hasattr(shell_tool, "coroutine") else shell_tool

        result = await tool_fn(command="rm -rf /")

        assert result["success"] is False
        assert result["exit_code"] == -1
        assert "not in allowlist" in result["stderr"]
        assert "pytest, npm" in result["stderr"]

    @pytest.mark.asyncio
    async def test_allowed_command_with_args(self) -> None:
        """Allowed command with arguments should work."""
        from ai_infra.llm.shell.tool import create_shell_tool

        shell_tool = create_shell_tool(
            allowed_commands=("echo",),
        )

        tool_fn = shell_tool.coroutine if hasattr(shell_tool, "coroutine") else shell_tool

        result = await tool_fn(command="echo -n test")

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_no_allowlist_allows_all(self) -> None:
        """Without allowlist, all commands should be allowed."""
        from ai_infra.llm.shell.tool import create_shell_tool

        shell_tool = create_shell_tool(
            allowed_commands=None,  # No allowlist
        )

        tool_fn = shell_tool.coroutine if hasattr(shell_tool, "coroutine") else shell_tool

        result = await tool_fn(command="echo test")

        assert result["success"] is True


class TestShellCLIOptions:
    """Tests for shell CLI option handling (Phase 2.4)."""

    def test_cli_shell_options_defaults(self, tmp_path: Path) -> None:
        """CLI options should have correct defaults."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        # Default values
        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
        )

        assert executor.enable_shell is True
        assert executor.shell_timeout == 120.0
        assert executor.shell_allowed_commands is None

    def test_cli_shell_options_custom(self, tmp_path: Path) -> None:
        """CLI options should accept custom values."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            enable_shell=False,
            shell_timeout=60.0,
            shell_allowed_commands=("pytest", "npm", "make"),
        )

        assert executor.enable_shell is False
        assert executor.shell_timeout == 60.0
        assert executor.shell_allowed_commands == ("pytest", "npm", "make")


class TestAutonomousVerificationConfiguration:
    """Tests for autonomous verification configuration in ExecutorGraph."""

    def test_autonomous_verify_disabled_by_default(self, tmp_path: Path) -> None:
        """Autonomous verification should be disabled by default."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(roadmap_path=str(roadmap))

        assert executor.enable_autonomous_verify is False

    def test_autonomous_verify_enabled(self, tmp_path: Path) -> None:
        """Autonomous verification can be explicitly enabled."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            enable_autonomous_verify=True,
        )

        assert executor.enable_autonomous_verify is True

    def test_default_verify_timeout(self, tmp_path: Path) -> None:
        """Default verify timeout should be 300 seconds (5 minutes)."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(roadmap_path=str(roadmap))

        assert executor.verify_timeout == 300.0

    def test_custom_verify_timeout(self, tmp_path: Path) -> None:
        """Custom verify timeout can be set."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            verify_timeout=600.0,
        )

        assert executor.verify_timeout == 600.0

    def test_autonomous_verify_with_custom_timeout(self, tmp_path: Path) -> None:
        """Autonomous verification can be enabled with custom timeout."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            enable_autonomous_verify=True,
            verify_timeout=120.0,
        )

        assert executor.enable_autonomous_verify is True
        assert executor.verify_timeout == 120.0

    def test_autonomous_verify_works_with_shell_enabled(self, tmp_path: Path) -> None:
        """Autonomous verification works alongside shell tool."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            enable_shell=True,
            enable_autonomous_verify=True,
            shell_timeout=60.0,
            verify_timeout=180.0,
        )

        assert executor.enable_shell is True
        assert executor.enable_autonomous_verify is True
        assert executor.shell_timeout == 60.0
        assert executor.verify_timeout == 180.0
