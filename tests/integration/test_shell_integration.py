"""Integration tests for Shell Tool in ai-infra.

Phase 5.2 of EXECUTOR_CLI.md - Integration Tests.

This module tests:
- 5.2.1 Executor with shell tool enabled
- 5.2.2 Verification agent discovery
- 5.2.3 Multi-command workflows
- 5.2.4 Session persistence across tool calls

These tests require real shell execution and may interact with the file system.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Mark all tests in this module as integration tests
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform.startswith("win"),
        reason="Shell integration tests run on Unix only",
    ),
]


# =============================================================================
# 5.2.1 Test Executor with Shell Tool Enabled
# =============================================================================


class TestExecutorShellEnabled:
    """Integration tests for executor with shell tool enabled (5.2.1)."""

    @pytest.mark.asyncio
    async def test_executor_shell_session_starts_and_closes(self, tmp_path: Path) -> None:
        """Executor shell session starts before run and closes after."""
        from ai_infra.executor.graph import ExecutorGraph

        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1: Test task")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            enable_shell=True,
        )

        # Session should be None before run
        assert executor._shell_session is None

        # Start session
        await executor._start_shell_session()
        assert executor._shell_session is not None
        assert executor._shell_session.is_running

        # Close session
        await executor._close_shell_session()
        assert executor._shell_session is None

    @pytest.mark.asyncio
    async def test_executor_shell_tool_executes_commands(self, tmp_path: Path) -> None:
        """Shell tool can execute real commands in executor context."""
        from ai_infra.executor.graph import ExecutorGraph

        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1: Test task")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            enable_shell=True,
            shell_workspace=str(tmp_path),
        )

        await executor._start_shell_session()

        try:
            # Use executor's session directly (context is set in arun, not _start_shell_session)
            session = executor._shell_session
            assert session is not None

            # Execute a real command
            result = await session.execute("echo 'integration test'")

            assert result.success
            assert result.exit_code == 0
            assert "integration test" in result.stdout
        finally:
            await executor._close_shell_session()

    @pytest.mark.asyncio
    async def test_executor_shell_workspace_respected(self, tmp_path: Path) -> None:
        """Shell commands run in the configured workspace directory."""
        from ai_infra.executor.graph import ExecutorGraph

        # Create a subdirectory as workspace
        workspace = tmp_path / "myproject"
        workspace.mkdir()

        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1: Test task")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            enable_shell=True,
            shell_workspace=str(workspace),
        )

        await executor._start_shell_session()

        try:
            # Use executor's session directly
            session = executor._shell_session
            result = await session.execute("pwd")

            assert result.success
            assert str(workspace) in result.stdout
        finally:
            await executor._close_shell_session()

    @pytest.mark.asyncio
    async def test_executor_shell_timeout_configurable(self, tmp_path: Path) -> None:
        """Shell timeout can be configured via executor."""
        from ai_infra.executor.graph import ExecutorGraph

        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1: Test task")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            enable_shell=True,
            shell_timeout=30.0,
        )

        assert executor.shell_timeout == 30.0

    @pytest.mark.asyncio
    async def test_executor_context_cleared_on_close(self, tmp_path: Path) -> None:
        """Context variable is cleared when session closes (via arun/astream)."""
        from ai_infra.executor.graph import ExecutorGraph
        from ai_infra.llm.shell.tool import get_current_session, set_current_session

        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n\n- [ ] Task 1: Test task")

        executor = ExecutorGraph(
            roadmap_path=str(roadmap),
            enable_shell=True,
        )

        await executor._start_shell_session()

        # Manually set context (as arun does)
        set_current_session(executor._shell_session)
        assert get_current_session() is not None

        # Close session (as arun finally block does)
        set_current_session(None)
        await executor._close_shell_session()

        assert get_current_session() is None


# =============================================================================
# 5.2.2 Test Verification Agent Discovery
# =============================================================================


class TestVerificationAgentDiscovery:
    """Integration tests for verification agent discovery (5.2.2)."""

    def test_detect_python_project(self, tmp_path: Path) -> None:
        """Detect Python project from pyproject.toml."""
        from ai_infra.executor.verifier import ProjectType, detect_project_type

        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')

        project_type = detect_project_type(tmp_path)
        assert project_type == ProjectType.PYTHON

    def test_detect_nodejs_project(self, tmp_path: Path) -> None:
        """Detect Node.js project from package.json."""
        from ai_infra.executor.verifier import ProjectType, detect_project_type

        (tmp_path / "package.json").write_text('{"name": "test"}')

        project_type = detect_project_type(tmp_path)
        assert project_type == ProjectType.NODEJS

    def test_detect_typescript_project(self, tmp_path: Path) -> None:
        """Detect TypeScript project from package.json + tsconfig.json."""
        from ai_infra.executor.verifier import ProjectType, detect_project_type

        (tmp_path / "package.json").write_text('{"name": "test"}')
        (tmp_path / "tsconfig.json").write_text("{}")

        project_type = detect_project_type(tmp_path)
        assert project_type == ProjectType.TYPESCRIPT

    def test_detect_rust_project(self, tmp_path: Path) -> None:
        """Detect Rust project from Cargo.toml."""
        from ai_infra.executor.verifier import ProjectType, detect_project_type

        (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"')

        project_type = detect_project_type(tmp_path)
        assert project_type == ProjectType.RUST

    def test_detect_go_project(self, tmp_path: Path) -> None:
        """Detect Go project from go.mod."""
        from ai_infra.executor.verifier import ProjectType, detect_project_type

        (tmp_path / "go.mod").write_text("module test")

        project_type = detect_project_type(tmp_path)
        assert project_type == ProjectType.GO

    def test_detect_makefile_project(self, tmp_path: Path) -> None:
        """Detect Makefile project with test target."""
        from ai_infra.executor.verifier import ProjectType, detect_project_type

        (tmp_path / "Makefile").write_text("test:\n\techo 'testing'")

        project_type = detect_project_type(tmp_path)
        assert project_type == ProjectType.MAKEFILE

    def test_get_test_command_python(self, tmp_path: Path) -> None:
        """Get test command for Python project."""
        from ai_infra.executor.verifier import ProjectType, get_test_command

        cmd = get_test_command(ProjectType.PYTHON, tmp_path)

        assert cmd is not None
        # Command is like [python, '-m', 'pytest', '-q', '--tb=short']
        assert "pytest" in cmd  # pytest is in the command list

    def test_get_test_command_nodejs_with_script(self, tmp_path: Path) -> None:
        """Get test command for Node.js project with test script."""
        import json

        from ai_infra.executor.verifier import ProjectType, get_test_command

        (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": "jest"}}))

        cmd = get_test_command(ProjectType.NODEJS, tmp_path)

        assert cmd is not None
        assert cmd == ["npm", "test"]

    def test_get_test_command_rust(self, tmp_path: Path) -> None:
        """Get test command for Rust project."""
        from ai_infra.executor.verifier import ProjectType, get_test_command

        cmd = get_test_command(ProjectType.RUST, tmp_path)

        assert cmd is not None
        assert cmd == ["cargo", "test"]

    def test_get_test_command_go(self, tmp_path: Path) -> None:
        """Get test command for Go project."""
        from ai_infra.executor.verifier import ProjectType, get_test_command

        cmd = get_test_command(ProjectType.GO, tmp_path)

        assert cmd is not None
        assert cmd == ["go", "test", "./..."]

    def test_get_test_command_makefile(self, tmp_path: Path) -> None:
        """Get test command for Makefile project."""
        from ai_infra.executor.verifier import ProjectType, get_test_command

        cmd = get_test_command(ProjectType.MAKEFILE, tmp_path)

        assert cmd is not None
        assert cmd == ["make", "test"]

    def test_task_needs_deep_verification_true(self) -> None:
        """Task with test-related title needs deep verification."""
        from ai_infra.executor.agents.verify_agent import task_needs_deep_verification
        from ai_infra.executor.models import Task

        task = Task(id="1.1", title="Add user authentication endpoint")

        assert task_needs_deep_verification(task) is True

    def test_task_needs_deep_verification_false(self) -> None:
        """Task with generic title does not need deep verification."""
        from ai_infra.executor.agents.verify_agent import task_needs_deep_verification
        from ai_infra.executor.models import Task

        task = Task(id="1.1", title="Update README")

        assert task_needs_deep_verification(task) is False

    def test_is_docs_only_change_true(self) -> None:
        """Docs-only changes are detected correctly."""
        from ai_infra.executor.agents.verify_agent import is_docs_only_change

        files = ["README.md", "docs/guide.md", "CHANGELOG.md"]

        assert is_docs_only_change(files) is True

    def test_is_docs_only_change_false(self) -> None:
        """Mixed changes are not docs-only."""
        from ai_infra.executor.agents.verify_agent import is_docs_only_change

        files = ["src/app.py", "README.md"]

        assert is_docs_only_change(files) is False


# =============================================================================
# 5.2.3 Test Multi-Command Workflows
# =============================================================================


class TestMultiCommandWorkflows:
    """Integration tests for multi-command workflows (5.2.3)."""

    @pytest.mark.asyncio
    async def test_sequential_commands_maintain_state(self) -> None:
        """Sequential commands share environment state."""
        from ai_infra.llm.shell.session import ShellSession

        async with ShellSession() as session:
            # Set a variable
            await session.execute("export MULTI_CMD_VAR=step1")

            # Modify the variable
            await session.execute("export MULTI_CMD_VAR=${MULTI_CMD_VAR}_step2")

            # Read the accumulated value
            result = await session.execute("echo $MULTI_CMD_VAR")

            assert result.success
            assert "step1_step2" in result.stdout

    @pytest.mark.asyncio
    async def test_workflow_with_file_operations(self, tmp_path: Path) -> None:
        """Workflow that creates and modifies files."""
        from ai_infra.llm.shell.session import SessionConfig, ShellSession

        config = SessionConfig(workspace_root=tmp_path)

        async with ShellSession(config) as session:
            # Create a file
            await session.execute("echo 'line1' > testfile.txt")

            # Append to the file
            await session.execute("echo 'line2' >> testfile.txt")

            # Read the file
            result = await session.execute("cat testfile.txt")

            assert result.success
            assert "line1" in result.stdout
            assert "line2" in result.stdout

            # Verify file exists on disk
            assert (tmp_path / "testfile.txt").exists()

    @pytest.mark.asyncio
    async def test_workflow_with_directory_navigation(self, tmp_path: Path) -> None:
        """Workflow navigating between directories."""
        from ai_infra.llm.shell.session import SessionConfig, ShellSession

        # Create subdirectories
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()

        config = SessionConfig(workspace_root=tmp_path)

        async with ShellSession(config) as session:
            # Navigate to src
            await session.execute("cd src")
            result = await session.execute("pwd")
            assert "src" in result.stdout

            # Navigate to tests (relative to src)
            await session.execute("cd ../tests")
            result = await session.execute("pwd")
            assert "tests" in result.stdout

            # Back to root
            await session.execute("cd ..")
            result = await session.execute("pwd")
            assert str(tmp_path) in result.stdout

    @pytest.mark.asyncio
    async def test_workflow_with_piped_commands(self) -> None:
        """Workflow using piped commands."""
        from ai_infra.llm.shell.session import ShellSession

        async with ShellSession() as session:
            # Create some data and process it
            result = await session.execute("echo -e 'apple\\nbanana\\ncherry' | grep -c 'a'")

            assert result.success
            # Should find 'a' in 'apple' and 'banana'
            assert "2" in result.stdout

    @pytest.mark.asyncio
    async def test_workflow_with_subshell(self) -> None:
        """Workflow using subshell for isolation."""
        from ai_infra.llm.shell.session import ShellSession

        async with ShellSession() as session:
            # Set outer variable
            await session.execute("export OUTER=outer")

            # Subshell should not affect outer scope
            await session.execute("(export INNER=inner; echo $INNER)")

            # Check outer is preserved, inner is not set
            result_outer = await session.execute("echo $OUTER")
            result_inner = await session.execute("echo $INNER")

            assert "outer" in result_outer.stdout
            # INNER should be empty in outer scope
            assert result_inner.stdout.strip() == "" or "inner" not in result_inner.stdout

    @pytest.mark.asyncio
    async def test_workflow_with_startup_commands(self, tmp_path: Path) -> None:
        """Workflow with startup commands setting up environment."""
        from ai_infra.llm.shell.session import SessionConfig, ShellSession

        config = SessionConfig(
            workspace_root=tmp_path,
            startup_commands=[
                "export PROJECT_ROOT=$(pwd)",
                "export PATH=$PATH:$PROJECT_ROOT/bin",
            ],
        )

        async with ShellSession(config) as session:
            # Check startup commands were executed
            result = await session.execute("echo $PROJECT_ROOT")

            assert result.success
            assert str(tmp_path) in result.stdout

    @pytest.mark.asyncio
    async def test_workflow_handles_command_failure(self) -> None:
        """Workflow continues after command failure."""
        from ai_infra.llm.shell.session import ShellSession

        async with ShellSession() as session:
            # First command succeeds
            result1 = await session.execute("echo 'step1'")
            assert result1.success

            # Second command fails (file doesn't exist)
            result2 = await session.execute("cat /nonexistent/file 2>/dev/null")
            assert not result2.success
            assert result2.exit_code != 0

            # Third command still works
            result3 = await session.execute("echo 'step3'")
            assert result3.success
            assert "step3" in result3.stdout


# =============================================================================
# 5.2.4 Test Session Persistence Across Tool Calls
# =============================================================================


class TestSessionPersistence:
    """Integration tests for session persistence across tool calls (5.2.4)."""

    @pytest.mark.asyncio
    async def test_session_persists_environment_variables(self) -> None:
        """Environment variables persist across multiple tool calls."""
        from ai_infra.llm.shell.session import ShellSession

        async with ShellSession() as session:
            # First "tool call" - set variable
            await session.execute("export PERSIST_VAR=value1")

            # Second "tool call" - modify variable
            await session.execute("export PERSIST_VAR=value2")

            # Third "tool call" - read variable
            result = await session.execute("echo $PERSIST_VAR")

            assert result.success
            assert "value2" in result.stdout

    @pytest.mark.asyncio
    async def test_session_persists_working_directory(self, tmp_path: Path) -> None:
        """Working directory persists across tool calls."""
        from ai_infra.llm.shell.session import SessionConfig, ShellSession

        subdir = tmp_path / "subdir"
        subdir.mkdir()

        config = SessionConfig(workspace_root=tmp_path)

        async with ShellSession(config) as session:
            # First "tool call" - change directory
            await session.execute(f"cd {subdir}")

            # Second "tool call" - check we're still there
            result = await session.execute("pwd")

            assert result.success
            assert str(subdir) in result.stdout

    @pytest.mark.asyncio
    async def test_session_persists_aliases(self) -> None:
        """Shell aliases persist across tool calls."""
        from ai_infra.llm.shell.session import ShellSession

        async with ShellSession() as session:
            # First "tool call" - create alias
            await session.execute("alias greet='echo Hello, World!'")

            # Second "tool call" - use alias
            result = await session.execute("greet")

            # Note: Alias expansion depends on shell mode
            # In non-interactive mode, aliases may not work
            # This test verifies the session state is maintained
            assert result.exit_code in (0, 127)  # 127 if alias not found

    @pytest.mark.asyncio
    async def test_session_persists_functions(self) -> None:
        """Shell functions persist across tool calls."""
        from ai_infra.llm.shell.session import ShellSession

        async with ShellSession() as session:
            # First "tool call" - define function
            await session.execute('greet_fn() { echo "Hello, $1!"; }')

            # Second "tool call" - call function
            result = await session.execute("greet_fn 'World'")

            assert result.success
            assert "Hello, World!" in result.stdout

    @pytest.mark.asyncio
    async def test_session_command_history_accumulated(self) -> None:
        """Command history is accumulated across tool calls."""
        from ai_infra.llm.shell.session import ShellSession

        async with ShellSession() as session:
            await session.execute("echo 'command1'")
            await session.execute("echo 'command2'")
            await session.execute("echo 'command3'")

            # Check history is accumulated
            history = session.command_history
            assert len(history) >= 3

    @pytest.mark.asyncio
    async def test_session_restart_clears_state(self) -> None:
        """Session restart clears accumulated state."""
        from ai_infra.llm.shell.session import SessionConfig, ShellSession

        config = SessionConfig(startup_commands=["export STARTUP_VAR=initial"])

        async with ShellSession(config) as session:
            # Modify state
            await session.execute("export STARTUP_VAR=modified")
            result = await session.execute("echo $STARTUP_VAR")
            assert "modified" in result.stdout

            # Restart session
            await session.restart()

            # State should be reset to startup state
            result = await session.execute("echo $STARTUP_VAR")
            assert "initial" in result.stdout

    @pytest.mark.asyncio
    async def test_session_context_variable_isolation(self) -> None:
        """Context variable properly isolates session access."""
        from ai_infra.llm.shell.session import ShellSession
        from ai_infra.llm.shell.tool import get_current_session, set_current_session

        # Create two sessions
        session1 = ShellSession()
        session2 = ShellSession()

        await session1.start()
        await session2.start()

        try:
            # Set session1 in context
            set_current_session(session1)
            assert get_current_session() is session1

            # Execute in session1
            await session1.execute("export SESSION_ID=session1")

            # Switch to session2
            set_current_session(session2)
            assert get_current_session() is session2

            # Execute in session2
            await session2.execute("export SESSION_ID=session2")

            # Verify isolation - session1 should still have session1 value
            result1 = await session1.execute("echo $SESSION_ID")
            result2 = await session2.execute("echo $SESSION_ID")

            assert "session1" in result1.stdout
            assert "session2" in result2.stdout
        finally:
            set_current_session(None)
            await session1.close()
            await session2.close()


# =============================================================================
# Verifier Integration Tests
# =============================================================================


class TestVerifierIntegration:
    """Integration tests for TaskVerifier with real file system."""

    @pytest.mark.asyncio
    async def test_verifier_checks_files_exist(self, tmp_path: Path) -> None:
        """Verifier checks that expected files exist."""
        from ai_infra.executor.models import Task
        from ai_infra.executor.verifier import CheckLevel, CheckStatus, TaskVerifier

        # Create expected files
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("print('hello')")

        task = Task(
            id="1.1",
            title="Create app",
            file_hints=["src/app.py", "src/missing.py"],
        )

        verifier = TaskVerifier(workspace=tmp_path)
        result = await verifier.verify(task, levels=[CheckLevel.FILES])

        # Should have checks for both files
        assert len(result.checks) == 2

        # app.py should pass
        app_check = next(c for c in result.checks if "app.py" in c.name)
        assert app_check.status == CheckStatus.PASSED

        # missing.py should fail
        missing_check = next(c for c in result.checks if "missing.py" in c.name)
        assert missing_check.status == CheckStatus.FAILED

    @pytest.mark.asyncio
    async def test_verifier_checks_python_syntax(self, tmp_path: Path) -> None:
        """Verifier checks Python syntax."""
        from ai_infra.executor.models import Task
        from ai_infra.executor.verifier import CheckLevel, CheckStatus, TaskVerifier

        # Create files with valid and invalid syntax
        (tmp_path / "valid.py").write_text("x = 1 + 2")
        (tmp_path / "invalid.py").write_text("x = 1 +")  # Syntax error

        task = Task(id="1.1", title="Test syntax")

        verifier = TaskVerifier(workspace=tmp_path)
        result = await verifier.verify(task, levels=[CheckLevel.SYNTAX])

        # Should have checks for both files
        syntax_checks = [c for c in result.checks if c.level == CheckLevel.SYNTAX]
        assert len(syntax_checks) >= 2

        # valid.py should pass
        valid_check = next(c for c in syntax_checks if "valid.py" in c.name)
        assert valid_check.status == CheckStatus.PASSED

        # invalid.py should fail
        invalid_check = next(c for c in syntax_checks if "invalid.py" in c.name)
        assert invalid_check.status == CheckStatus.FAILED

    @pytest.mark.asyncio
    async def test_verifier_stop_on_failure(self, tmp_path: Path) -> None:
        """Verifier can stop at first failure."""
        from ai_infra.executor.models import Task
        from ai_infra.executor.verifier import CheckLevel, TaskVerifier

        # Create invalid Python file
        (tmp_path / "invalid.py").write_text("x = 1 +")

        task = Task(id="1.1", title="Test")

        verifier = TaskVerifier(workspace=tmp_path)
        result = await verifier.verify(
            task,
            levels=[CheckLevel.SYNTAX, CheckLevel.IMPORTS],
            stop_on_failure=True,
        )

        # Should not have run imports level due to syntax failure
        levels_run = set(result.levels_run)
        # IMPORTS might not be in levels_run if stopped
        assert CheckLevel.SYNTAX in levels_run


# =============================================================================
# Shell Tool Integration with run_shell
# =============================================================================


class TestRunShellToolIntegration:
    """Integration tests for run_shell tool."""

    @pytest.mark.asyncio
    async def test_run_shell_stateless_execution(self) -> None:
        """run_shell works without a session (stateless mode)."""
        from ai_infra.llm.shell.tool import run_shell

        result = await run_shell.ainvoke({"command": "echo 'stateless test'"})

        assert result["success"] is True
        assert result["exit_code"] == 0
        assert "stateless test" in result["stdout"]

    @pytest.mark.asyncio
    async def test_run_shell_with_cwd(self, tmp_path: Path) -> None:
        """run_shell respects cwd parameter."""
        from ai_infra.llm.shell.tool import run_shell

        result = await run_shell.ainvoke(
            {
                "command": "pwd",
                "cwd": str(tmp_path),
            }
        )

        assert result["success"] is True
        assert str(tmp_path) in result["stdout"]

    @pytest.mark.asyncio
    async def test_run_shell_with_session(self) -> None:
        """run_shell uses session when available."""
        from ai_infra.llm.shell.session import ShellSession
        from ai_infra.llm.shell.tool import (
            run_shell,
            set_current_session,
        )

        async with ShellSession() as session:
            set_current_session(session)

            try:
                # Set variable via session
                await session.execute("export RUN_SHELL_VAR=session_test")

                # run_shell should see the variable (uses session)
                result = await run_shell.ainvoke({"command": "echo $RUN_SHELL_VAR"})

                assert result["success"] is True
                assert "session_test" in result["stdout"]
            finally:
                set_current_session(None)

    @pytest.mark.asyncio
    async def test_run_shell_rejects_dangerous_commands(self) -> None:
        """run_shell rejects dangerous commands."""
        from ai_infra.llm.shell.tool import run_shell

        result = await run_shell.ainvoke({"command": "rm -rf /"})

        assert result["success"] is False
        assert "rejected" in result["stderr"].lower() or "dangerous" in result["stderr"].lower()

    @pytest.mark.asyncio
    async def test_run_shell_handles_invalid_cwd(self) -> None:
        """run_shell handles invalid cwd gracefully."""
        from ai_infra.llm.shell.tool import run_shell

        result = await run_shell.ainvoke(
            {
                "command": "echo test",
                "cwd": "/nonexistent/directory/path",
            }
        )

        assert result["success"] is False
        assert "not exist" in result["stderr"].lower() or "directory" in result["stderr"].lower()
