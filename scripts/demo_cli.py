#!/usr/bin/env python3
"""Demo script to showcase the new CLI UI/UX components.

Run with: poetry run python scripts/demo_cli.py
"""

from __future__ import annotations

import time

from rich.panel import Panel
from rich.text import Text

from ai_infra.cli.console import (
    ColorSupport,
    detect_terminal_capabilities,
    get_console,
    print_error,
    print_info,
    print_success,
    print_warning,
)
from ai_infra.cli.dashboard import (
    ModelIndicator,
    StatusBar,
    StatusBarState,
    TokenBudget,
)
from ai_infra.cli.interactive import (
    ContextInfo,
    ContextPreview,
    DependencyGraph,
    DependencyNode,
    TaskInfo,
    TaskPreview,
    render_help,
)
from ai_infra.cli.progress import (
    ETACalculator,
    PhaseSection,
    PhaseState,
    TaskItem,
    TaskProgressPanel,
    TaskSpinner,
    TaskStatus,
    TokenCounter,
)
from ai_infra.cli.streaming import (
    DiffViewer,
    FileTreeDisplay,
    StreamingOutput,
    ToolCallPanel,
    ToolCallState,
    create_syntax,
    render_output_help,
)
from ai_infra.cli.streaming import (
    FileChange as StreamingFileChange,
)
from ai_infra.cli.streaming import (
    FileChangeType as StreamingFileChangeType,
)
from ai_infra.cli.summary import (
    CostEstimator,
    ExecutionResult,
    ExecutionStatus,
    ExecutionSummary,
    FileChange,
    FileChangeSummary,
    FileChangeType,
    GitCheckpoint,
    GitCheckpointSummary,
    TestResults,
    generate_next_steps,
)


def demo_console() -> None:
    """Demonstrate console theming and semantic print functions."""
    console = get_console()

    console.print()
    console.print(Panel("[bold]Phase 16.6.1: Rich Console Foundation[/]", border_style="cyan"))
    console.print()

    print_success("Task completed successfully!")
    print_error("Something went wrong!")
    print_warning("Proceeding with caution...")
    print_info("Processing your request...")
    console.print()

    caps = detect_terminal_capabilities()
    support_name = {
        ColorSupport.TRUECOLOR: "True Color (16M colors)",
        ColorSupport.EXTENDED: "Extended (256 colors)",
        ColorSupport.BASIC: "Basic (16 colors)",
        ColorSupport.NONE: "No color support",
    }
    console.print(
        f"[dim]Terminal color support:[/] {support_name.get(caps.color_support, 'Unknown')}"
    )
    console.print()


def demo_progress() -> None:
    """Demonstrate task progress display."""
    console = get_console()

    console.print(Panel("[bold]Phase 16.6.2: Task Progress Display[/]", border_style="cyan"))
    console.print()

    # Create tasks with different statuses
    tasks = [
        TaskItem(id="16.6.1", title="Parse ROADMAP.md", status=TaskStatus.COMPLETE, duration=2.3),
        TaskItem(
            id="16.6.2", title="Load project memory", status=TaskStatus.COMPLETE, duration=0.5
        ),
        TaskItem(
            id="16.6.3", title="Execute task 16.6.1", status=TaskStatus.COMPLETE, duration=45.2
        ),
        TaskItem(
            id="16.6.4",
            title="Execute task 16.6.2",
            status=TaskStatus.RUNNING,
            start_time=time.time() - 12.8,
        ),
        TaskItem(id="16.6.5", title="Execute task 16.6.3", status=TaskStatus.PENDING),
        TaskItem(id="16.6.6", title="Execute task 16.6.4", status=TaskStatus.PENDING),
    ]

    panel = TaskProgressPanel(
        tasks=tasks,
        phase_title="Phase 16.6: CLI UI/UX Improvements",
    )
    console.print(panel)
    console.print()

    # Phase sections
    console.print("[bold]Phase Sections:[/]")
    expanded = PhaseSection(title="Phase 16.6", completed=4, total=4, state=PhaseState.EXPANDED)
    collapsed = PhaseSection(title="Phase 16.7", completed=0, total=8, state=PhaseState.COLLAPSED)
    console.print(expanded.render())
    console.print(collapsed.render())
    console.print()

    # Token counter
    console.print("[bold]Token Counters:[/]")
    low = TokenCounter(used=50_000, limit=200_000)
    medium = TokenCounter(used=120_000, limit=200_000)
    high = TokenCounter(used=180_000, limit=200_000)
    console.print(f"  Low usage:    {low.render()}")
    console.print(f"  Medium usage: {medium.render()}")
    console.print(f"  High usage:   {high.render()}")
    console.print()

    # ETA Calculator
    eta = ETACalculator()
    eta.add_completed(15.0)
    eta.add_completed(12.5)
    eta.add_completed(18.2)
    estimated = eta.estimate_remaining(remaining_count=3)
    console.print(f"[bold]ETA Estimate:[/] {estimated:.1f}s for 3 remaining tasks")
    console.print()


def demo_dashboard() -> None:
    """Demonstrate live status dashboard."""
    console = get_console()

    console.print(Panel("[bold]Phase 16.6.3: Live Status Dashboard[/]", border_style="cyan"))
    console.print()

    # Model indicator
    console.print("[bold]Model Indicators:[/]")
    models = [
        ModelIndicator(model_name="claude-sonnet-4-20250514"),
        ModelIndicator(model_name="gpt-4o"),
        ModelIndicator(model_name="gemini-2.0-flash"),
    ]
    for m in models:
        console.print(f"  {m.render()}")
    console.print()

    # Token budget
    console.print("[bold]Token Budgets:[/]")
    budgets = [
        TokenBudget(used=25_000, limit=200_000),
        TokenBudget(used=100_000, limit=200_000),
        TokenBudget(used=175_000, limit=200_000),
    ]
    for b in budgets:
        console.print(f"  {b.render()}")
    console.print()

    # Full status bar
    console.print("[bold]Status Bar:[/]")
    state = StatusBarState(
        model_name="claude-sonnet-4-20250514",
        tokens_used=75_000,
        tokens_limit=200_000,
        elapsed_seconds=125.5,
        completed_tasks=3,
        total_tasks=8,
        current_phase=16,
        total_phases=20,
    )
    status_bar = StatusBar(state=state)
    console.print(status_bar)
    console.print()


def demo_summary() -> None:
    """Demonstrate execution summary."""
    console = get_console()

    console.print(Panel("[bold]Phase 16.6.4: Execution Summary[/]", border_style="cyan"))
    console.print()

    # Cost estimation
    console.print("[bold]Cost Estimation:[/]")
    estimator = CostEstimator(
        model_name="claude-sonnet-4-20250514",
        tokens_input=50_000,
        tokens_output=15_000,
    )
    console.print(f"  {estimator.format_cost()}")
    console.print(f"  {estimator.format_tokens()}")
    console.print()

    # File changes
    console.print("[bold]File Changes:[/]")
    changes = [
        FileChange(path="src/cli/console.py", change_type=FileChangeType.NEW, lines_added=150),
        FileChange(path="src/cli/progress.py", change_type=FileChangeType.NEW, lines_added=280),
        FileChange(path="src/cli/dashboard.py", change_type=FileChangeType.NEW, lines_added=320),
        FileChange(
            path="tests/unit/cli/test_console.py",
            change_type=FileChangeType.MODIFIED,
            lines_added=200,
            lines_removed=10,
        ),
    ]
    file_summary = FileChangeSummary(changes=changes)
    console.print(file_summary)
    console.print()

    # Git checkpoints
    console.print("[bold]Git Checkpoints:[/]")
    checkpoints = [
        GitCheckpoint(
            task_id="16.6.1",
            commit_hash="abc1234",
            message="feat(cli): add rich console foundation",
        ),
        GitCheckpoint(
            task_id="16.6.2",
            commit_hash="def5678",
            message="feat(cli): add task progress display",
        ),
    ]
    git_summary = GitCheckpointSummary(checkpoints=checkpoints)
    console.print(git_summary.render())
    console.print()

    # Next steps
    console.print("[bold]Suggested Next Steps:[/]")
    next_steps = generate_next_steps(
        status=ExecutionStatus.COMPLETED,
        has_git_changes=True,
    )
    for step in next_steps:
        cmd = f"  ({step.command})" if step.command else ""
        console.print(f"  [cyan]>[/] {step.description}{cmd}")
    console.print()


def demo_live_progress() -> None:
    """Demonstrate live progress with spinner."""
    console = get_console()

    console.print(Panel("[bold]Live Progress Demo[/]", border_style="green"))
    console.print()
    console.print("[dim]Watch the spinner animate in real-time...[/]")
    console.print()

    # Create a live spinner demo
    spinner = TaskSpinner(task_id="16.7.1", title="Processing files")
    _ = spinner.start(console)

    try:
        for _ in range(15):
            time.sleep(0.2)
            spinner.update()
    finally:
        spinner.stop()

    print_success("Live demo complete!")
    console.print()


def demo_full_summary() -> None:
    """Demonstrate full execution summary panel."""
    console = get_console()

    console.print(Panel("[bold]Full Execution Summary[/]", border_style="magenta"))
    console.print()

    # Create file changes summary
    file_changes = FileChangeSummary(
        changes=[
            FileChange(
                path="src/ai_infra/cli/console.py", change_type=FileChangeType.NEW, lines_added=150
            ),
            FileChange(
                path="src/ai_infra/cli/progress.py", change_type=FileChangeType.NEW, lines_added=350
            ),
            FileChange(
                path="src/ai_infra/cli/dashboard.py",
                change_type=FileChangeType.NEW,
                lines_added=380,
            ),
            FileChange(
                path="src/ai_infra/cli/summary.py", change_type=FileChangeType.NEW, lines_added=420
            ),
        ]
    )

    # Create git checkpoints summary
    git_checkpoints = GitCheckpointSummary(
        checkpoints=[
            GitCheckpoint(
                commit_hash="a1b2c3d",
                message="feat(cli): rich console foundation",
                task_id="16.6.1",
            ),
            GitCheckpoint(
                commit_hash="e4f5g6h", message="feat(cli): task progress display", task_id="16.6.2"
            ),
            GitCheckpoint(
                commit_hash="i7j8k9l", message="feat(cli): live status dashboard", task_id="16.6.3"
            ),
            GitCheckpoint(
                commit_hash="m0n1o2p", message="feat(cli): execution summary", task_id="16.6.4"
            ),
        ]
    )

    result = ExecutionResult(
        status=ExecutionStatus.COMPLETED,
        total_tasks=4,
        completed_tasks=4,
        duration_seconds=245.8,
        tokens_input=85_000,
        tokens_output=28_000,
        model_name="claude-sonnet-4-20250514",
        file_changes=file_changes,
        git_checkpoints=git_checkpoints,
        test_results=TestResults(
            passed=190,
            failed=0,
            skipped=0,
            coverage=85.5,
        ),
    )

    summary = ExecutionSummary(result=result)
    console.print(summary)
    console.print()


def demo_interactive() -> None:
    """Demonstrate interactive mode components."""
    console = get_console()

    console.print(Panel("[bold]Phase 16.7.1: Interactive Task Review Mode[/]", border_style="cyan"))
    console.print()

    # Task Preview
    console.print("[bold]Task Preview Panel:[/]")
    task = TaskInfo(
        id="1.1.3",
        title="Create user authentication module",
        description="Implement JWT-based authentication with login/logout endpoints.\nInclude password hashing with bcrypt and token refresh logic.",
        phase="Phase 1: Project Setup",
        complexity="medium",
        dependencies=["1.1.1", "1.1.2"],
        completed_dependencies=["1.1.1", "1.1.2"],
    )
    preview = TaskPreview(task, show_actions=True)
    console.print(preview)
    console.print()

    # Context Preview
    console.print("[bold]Context Preview:[/]")
    context = ContextInfo(
        system_tokens=847,
        task_tokens=1234,
        file_tokens=2456,
        memory_tokens=512,
        files_included=[
            ("src/main.py", "full"),
            ("src/models.py", "full"),
            ("tests/conftest.py", "partial: lines 1-50"),
        ],
    )
    ctx_preview = ContextPreview(context)
    console.print(ctx_preview)
    console.print()

    # Dependency Graph
    console.print("[bold]Dependency Graph:[/]")
    nodes = [
        DependencyNode("1.1.1", "Create project structure", TaskStatus.COMPLETE),
        DependencyNode("1.1.2", "Initialize database", TaskStatus.COMPLETE),
        DependencyNode("1.1.3", "Create auth module", TaskStatus.PENDING, is_current=True),
    ]
    graph = DependencyGraph("1.1.3", nodes)
    console.print(graph)
    console.print()

    # Help
    console.print("[bold]Interactive Mode Help:[/]")
    console.print(render_help())
    console.print()


def demo_streaming() -> None:
    """Demonstrate streaming output components."""
    console = get_console()

    console.print(Panel("[bold]Phase 16.7.2: Real-time Streaming Output[/]", border_style="cyan"))
    console.print()

    # Streaming output with code blocks
    console.print("[bold]Streaming LLM Output with Syntax Highlighting:[/]")
    streamer = StreamingOutput(console=console)
    streamer.append("Analyzing task requirements...\n\n")
    streamer.append("I'll implement the authentication module with the following approach:\n\n")
    streamer.append("1. Create JWT token generation using python-jose\n")
    streamer.append("2. Implement password hashing with bcrypt\n")
    streamer.append("3. Add login/logout endpoints with FastAPI\n\n")
    streamer.append("```python\ndef create_access_token(data: dict) -> str:\n")
    streamer.append("    to_encode = data.copy()\n")
    streamer.append("    expire = datetime.utcnow() + timedelta(minutes=30)\n")
    streamer.append("    to_encode.update({'exp': expire})\n")
    streamer.append("    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)\n```\n")
    streamer.finalize()
    console.print(streamer)
    console.print()

    # Tool call panels
    console.print("[bold]Collapsible Tool Call Panels:[/]")
    # Expanded tool call
    tc1 = ToolCallPanel(
        tool_name="write_file",
        args={"path": "src/auth.py"},
        output="Creating authentication module with JWT support...\n- Added create_access_token() function\n- Added verify_password() with bcrypt\n- Added get_current_user() dependency",
        state=ToolCallState.COMPLETE,
        summary="Created authentication module",
        line_count=45,
        duration=1.2,
        expanded=True,
    )
    console.print(tc1)
    console.print()

    # Collapsed tool call
    tc2 = ToolCallPanel(
        tool_name="run_command",
        args={"command": "pytest tests/test_auth.py -v"},
        state=ToolCallState.COMPLETE,
        line_count=12,
        duration=3.5,
        expanded=False,
    )
    console.print(tc2)
    console.print()

    # Syntax highlighting demo
    console.print("[bold]Syntax Highlighting:[/]")
    code = '''def authenticate(token: str) -> User:
    """Authenticate user from JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401)
        return get_user(user_id)
    except JWTError:
        raise HTTPException(status_code=401)'''
    syntax = create_syntax(code, "python", line_numbers=True)
    console.print(syntax)
    console.print()

    # Diff viewer
    console.print("[bold]Diff Visualization:[/]")
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
    console.print(viewer)
    console.print()

    # File tree display
    console.print("[bold]File Tree Updates:[/]")
    changes = [
        StreamingFileChange(
            path="src/auth.py", change_type=StreamingFileChangeType.NEW, lines_added=45
        ),
        StreamingFileChange(
            path="src/main.py",
            change_type=StreamingFileChangeType.MODIFIED,
            lines_added=5,
            lines_removed=2,
        ),
        StreamingFileChange(
            path="tests/test_auth.py", change_type=StreamingFileChangeType.NEW, lines_added=80
        ),
    ]
    tree = FileTreeDisplay(changes=changes)
    console.print(tree)
    console.print()

    # Output controls help
    console.print("[bold]Output Control Keys:[/]")
    console.print(render_output_help())
    console.print()


def main() -> None:
    """Run all demos."""
    console = get_console()
    console.clear()

    title = Text()
    title.append(
        "╔════════════════════════════════════════════════════════════╗\n", style="bold cyan"
    )
    title.append("║", style="bold cyan")
    title.append("  ai-infra CLI UI/UX Demo — Phase 16.6 & 16.7.1-16.7.2    ", style="bold white")
    title.append("║\n", style="bold cyan")
    title.append(
        "╚════════════════════════════════════════════════════════════╝", style="bold cyan"
    )
    console.print(title)
    console.print()

    demo_console()
    demo_progress()
    demo_dashboard()
    demo_summary()
    demo_live_progress()
    demo_full_summary()
    demo_interactive()
    demo_streaming()

    console.print()
    print_success("All Phase 16.6 & 16.7.1-16.7.2 CLI components demonstrated!")
    console.print("[dim]Streaming output is ready - next: Phase 16.7.3 (Keyboard Shortcuts)[/]")
    console.print()


if __name__ == "__main__":
    main()
