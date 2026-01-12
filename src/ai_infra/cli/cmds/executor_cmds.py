"""
CLI commands for the Executor module.

Usage:
    ai-infra executor run --roadmap ./ROADMAP.md --max-tasks 5
    ai-infra executor status --roadmap ./ROADMAP.md
    ai-infra executor resume --roadmap ./ROADMAP.md --approve
    ai-infra executor rollback --roadmap ./ROADMAP.md
    ai-infra executor reset --roadmap ./ROADMAP.md
    ai-infra executor run --dry-run --roadmap ./ROADMAP.md

The executor reads tasks from a ROADMAP.md file and executes them
autonomously using an AI agent.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ai_infra import Agent
from ai_infra.executor import (
    ExecutionStatus,
    Executor,
    ExecutorCallbacks,
    ExecutorConfig,
    ReviewInfo,
    RunStatus,
    RunSummary,
    TaskStatus,
    VerifyMode,
)
from ai_infra.executor.adaptive import AdaptiveMode
from ai_infra.llm.workspace import Workspace

app = typer.Typer(help="Autonomous task execution from ROADMAP.md")
console = Console()


# =============================================================================
# Constants
# =============================================================================

# Status icons for task display
STATUS_ICONS = {
    TaskStatus.COMPLETED: "[green]✓[/green]",
    TaskStatus.FAILED: "[red]✗[/red]",
    TaskStatus.IN_PROGRESS: "[yellow]→[/yellow]",
    TaskStatus.PENDING: "[dim]○[/dim]",
    TaskStatus.SKIPPED: "[dim]⊘[/dim]",
}

# Status colors for run status
RUN_STATUS_COLORS = {
    RunStatus.COMPLETED: "green",
    RunStatus.PAUSED: "yellow",
    RunStatus.FAILED: "red",
    RunStatus.STOPPED: "yellow",
    RunStatus.NO_TASKS: "dim",
}


# =============================================================================
# Output Formatting
# =============================================================================


def _format_duration(ms: float) -> str:
    """Format duration in milliseconds to human-readable string."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m"


def _format_tokens(tokens: int) -> str:
    """Format token count with cost estimate."""
    # Rough estimate: $0.003 per 1K tokens (average of input/output)
    cost = (tokens / 1000) * 0.003
    return f"{tokens:,} (≈${cost:.2f})"


def _render_node_metrics(node_metrics: dict[str, Any] | None) -> Panel | None:
    """Render per-node metrics as a Rich panel.

    Phase 2.4.3: Display per-node cost breakdown.

    Args:
        node_metrics: Dict of node_name -> NodeMetrics.to_dict()

    Returns:
        Rich Panel with node breakdown, or None if no metrics.
    """
    if not node_metrics:
        return None

    from ai_infra.executor.metrics import aggregate_node_metrics

    aggregated = aggregate_node_metrics(node_metrics)

    if not aggregated.node_metrics:
        return None

    # Build table
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("Node", style="cyan")
    table.add_column("Tokens", justify="right")
    table.add_column("%", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Calls", justify="right")

    # Sort by total tokens descending
    sorted_nodes = sorted(
        aggregated.node_metrics.items(),
        key=lambda x: x[1].total_tokens,
        reverse=True,
    )

    # Find highest for highlighting
    highest_tokens = max((m.total_tokens for _, m in sorted_nodes), default=0)

    for name, metrics in sorted_nodes:
        pct = aggregated.get_node_percentage(name)
        tokens_str = f"{metrics.total_tokens:,}"

        # Highlight highest consumer
        if metrics.total_tokens == highest_tokens and metrics.total_tokens > 0:
            name = f"[bold yellow]{name}[/bold yellow]"
            tokens_str = f"[bold yellow]{tokens_str}[/bold yellow]"

        table.add_row(
            name,
            tokens_str,
            f"{pct:.1f}%",
            _format_duration(metrics.duration_ms),
            str(metrics.llm_calls) if metrics.llm_calls > 0 else "-",
        )

    # Add totals row
    table.add_row("", "", "", "", "")
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{aggregated.total_tokens:,}[/bold]",
        "[bold]100%[/bold]",
        f"[bold]{_format_duration(aggregated.total_duration_ms)}[/bold]",
        f"[bold]{aggregated.total_llm_calls}[/bold]"
        if aggregated.total_llm_calls > 0
        else "[bold]-[/bold]",
    )

    return Panel(
        table,
        title="Per-Node Cost Breakdown",
        subtitle="[dim]Highest consumer highlighted[/dim]",
        border_style="blue",
    )


def _render_run_summary(summary: RunSummary) -> Panel:
    """Render a run summary as a Rich panel."""
    status_color = RUN_STATUS_COLORS.get(summary.status, "white")

    # Build summary lines
    lines = []
    lines.append(
        f"[bold]Status:[/bold]     [{status_color}]{summary.status.value}[/{status_color}]"
    )

    # Task counts
    total = summary.total_tasks
    completed = summary.tasks_completed
    failed = summary.tasks_failed
    remaining = summary.tasks_remaining
    skipped = summary.tasks_skipped

    tasks_line = f"[bold]Tasks:[/bold]      {completed}/{total} completed"
    if remaining > 0:
        tasks_line += f" ({remaining} remaining"
        if skipped > 0:
            tasks_line += f", {skipped} skipped"
        tasks_line += ")"
    elif failed > 0:
        tasks_line += f" ({failed} failed)"
    lines.append(tasks_line)

    # Duration
    lines.append(f"[bold]Duration:[/bold]   {_format_duration(summary.duration_ms)}")

    # Tokens
    if summary.total_tokens > 0:
        lines.append(f"[bold]Tokens:[/bold]     {_format_tokens(summary.total_tokens)}")

    # Files modified
    total_files = sum(len(r.files_modified) + len(r.files_created) for r in summary.results)
    if total_files > 0:
        files_modified = sum(len(r.files_modified) for r in summary.results)
        files_created = sum(len(r.files_created) for r in summary.results)
        files_line = f"[bold]Files:[/bold]      {files_modified} modified"
        if files_created > 0:
            files_line += f", {files_created} created"
        lines.append(files_line)

    # Paused reason
    if summary.paused and summary.pause_reason:
        lines.append(f"[bold]Paused:[/bold]     {summary.pause_reason}")

    content = "\n".join(lines)

    return Panel(
        content,
        title="Executor Run Summary",
        border_style=status_color,
    )


def _render_results_table(summary: RunSummary) -> Table:
    """Render execution results as a Rich table."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("", width=3)  # Status icon
    table.add_column("Task ID", style="cyan")
    table.add_column("Title", no_wrap=False)
    table.add_column("Duration", justify="right")
    table.add_column("Tokens", justify="right")

    for result in summary.results:
        if result.status == ExecutionStatus.SUCCESS:
            icon = "[green]✓[/green]"
            duration = _format_duration(result.duration_ms) if result.duration_ms > 0 else "-"
            tokens = f"{sum(result.token_usage.values()):,}" if result.token_usage else "-"
        elif result.status == ExecutionStatus.FAILED:
            icon = "[red]✗[/red]"
            duration = "-"
            tokens = "(failed)"
        elif result.status == ExecutionStatus.SKIPPED:
            icon = "[dim]⊘[/dim]"
            duration = "-"
            tokens = "(skipped)"
        else:
            icon = "[yellow]?[/yellow]"
            duration = "-"
            tokens = "-"

        # Truncate title if too long (max 50 chars)
        title = result.title[:47] + "..." if len(result.title) > 50 else result.title
        table.add_row(icon, result.task_id, title, duration, tokens)

    return table


def _render_status_table(executor: Executor) -> Table:
    """Render task status as a Rich table."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("", width=3)  # Status icon
    table.add_column("Task ID", style="cyan")
    table.add_column("Title", no_wrap=False)
    table.add_column("Status")

    for task in executor.roadmap.all_tasks():
        status = executor.state.get_status(task.id)
        icon = STATUS_ICONS.get(status, "?")
        status_text = status.value

        if status == TaskStatus.COMPLETED:
            status_style = "green"
        elif status == TaskStatus.FAILED:
            status_style = "red"
        elif status == TaskStatus.IN_PROGRESS:
            status_style = "yellow"
        else:
            status_style = "dim"

        table.add_row(icon, task.id, task.title, f"[{status_style}]{status_text}[/{status_style}]")

    return table


def _render_review_info(review: ReviewInfo) -> Panel:
    """Render review info as a Rich panel."""
    lines = []
    lines.append(f"[bold]Tasks Executed:[/bold] {len(review.task_ids)}")

    if review.task_ids:
        lines.append(f"  {', '.join(review.task_ids)}")

    lines.append(f"\n[bold]Files Modified:[/bold] {len(review.files_modified)}")
    for f in review.files_modified[:10]:  # Limit to first 10
        lines.append(f"  • {f}")
    if len(review.files_modified) > 10:
        lines.append(f"  ... and {len(review.files_modified) - 10} more")

    if review.files_created:
        lines.append(f"\n[bold]Files Created:[/bold] {len(review.files_created)}")
        for f in review.files_created[:5]:
            lines.append(f"  • {f}")

    if review.files_deleted:
        lines.append(f"\n[bold red]Files Deleted:[/bold red] {len(review.files_deleted)}")
        for f in review.files_deleted:
            lines.append(f"  • [red]{f}[/red]")

    if review.has_destructive:
        lines.append("\n[bold red]WARNING: Destructive operations detected[/bold red]")

    if review.commits:
        lines.append(f"\n[bold]Git Commits:[/bold] {len(review.commits)}")
        for commit in review.commits[:5]:
            lines.append(f"  • {commit.short_sha}: {commit.message[:50]}")

    content = "\n".join(lines)

    border_color = "red" if review.has_destructive else "yellow"
    return Panel(
        content,
        title="Changes for Review",
        border_style=border_color,
    )


# =============================================================================
# executor run - Run the executor
# =============================================================================


@app.command("run")
def run_cmd(
    roadmap: Annotated[
        Path,
        typer.Option(
            "--roadmap",
            "-r",
            help="Path to ROADMAP.md file",
            exists=True,
            dir_okay=False,
        ),
    ] = Path("./ROADMAP.md"),
    max_tasks: Annotated[
        int,
        typer.Option(
            "--max-tasks",
            "-n",
            help="Maximum tasks to execute (0 = unlimited)",
        ),
    ] = 0,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Model to use for execution",
        ),
    ] = "claude-sonnet-4-20250514",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show what would be done without executing",
        ),
    ] = False,
    pause_destructive: Annotated[
        bool,
        typer.Option(
            "--pause-destructive/--no-pause-destructive",
            help="Pause and confirm before destructive operations (rm -rf, DROP TABLE, etc.)",
        ),
    ] = True,
    enable_planning: Annotated[
        bool,
        typer.Option(
            "--enable-planning/--no-planning",
            help="Enable pre-execution planning to identify files, dependencies, and risks (Phase 2.4.2)",
        ),
    ] = False,
    skip_verification: Annotated[
        bool,
        typer.Option(
            "--skip-verification",
            help="Skip task verification after execution",
        ),
    ] = False,
    stop_on_failure: Annotated[
        bool,
        typer.Option(
            "--stop-on-failure/--continue-on-failure",
            help="Stop execution on first failure",
        ),
    ] = True,
    require_approval: Annotated[
        int,
        typer.Option(
            "--require-approval",
            help="Pause for human approval after N tasks (0 = disabled)",
        ),
    ] = 0,
    checkpoint: Annotated[
        int,
        typer.Option(
            "--checkpoint",
            help="Create git checkpoint every N tasks (0 = disabled)",
        ),
    ] = 1,
    sync_roadmap: Annotated[
        bool,
        typer.Option(
            "--sync/--no-sync",
            help="Sync completed tasks to ROADMAP.md checkboxes after each task",
        ),
    ] = True,
    retry_failed: Annotated[
        int,
        typer.Option(
            "--retry-failed",
            help="Number of retry attempts for failed tasks (1 = no retry)",
        ),
    ] = 1,
    adaptive_mode: Annotated[
        str,
        typer.Option(
            "--adaptive-mode",
            help="Adaptive planning mode: no-adapt, suggest, or auto-fix",
        ),
    ] = "auto-fix",
    verify_mode: Annotated[
        str,
        typer.Option(
            "--verify-mode",
            help="Verification mode: auto (detect runner), agent (agent verifies), skip, pytest",
        ),
    ] = "auto",
    # Phase 5.8.5: Memory options
    no_run_memory: Annotated[
        bool,
        typer.Option(
            "--no-run-memory",
            help="Disable run memory (task-to-task context within a run)",
        ),
    ] = False,
    no_project_memory: Annotated[
        bool,
        typer.Option(
            "--no-project-memory",
            help="Disable project memory (cross-run persistence)",
        ),
    ] = False,
    memory_budget: Annotated[
        int,
        typer.Option(
            "--memory-budget",
            help="Token budget for memory context in prompts",
        ),
    ] = 6000,
    extract_with_llm: Annotated[
        bool,
        typer.Option(
            "--extract-with-llm",
            help="Use LLM for outcome extraction (more accurate, slower)",
        ),
    ] = False,
    clear_project_memory: Annotated[
        bool,
        typer.Option(
            "--clear-project-memory",
            help="Clear project memory before run (fresh start)",
        ),
    ] = False,
    execution_mode: Annotated[
        str,
        typer.Option(
            "--execution-mode",
            help="Execution mode: by-tasks (default) or by-todos (grouped execution)",
        ),
    ] = "by-tasks",
    normalize_with_llm: Annotated[
        bool,
        typer.Option(
            "--normalize-with-llm",
            help="Use LLM to normalize non-checkbox ROADMAP formats (emojis, prose, etc.)",
        ),
    ] = False,
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output as JSON",
        ),
    ] = False,
    # Phase 1.8.1: Graph-specific CLI options
    graph_mode: Annotated[
        bool,
        typer.Option(
            "--graph-mode/--legacy-mode",
            help="Use graph-based executor (default) or legacy imperative loop",
        ),
    ] = True,
    visualize: Annotated[
        bool,
        typer.Option(
            "--visualize",
            help="Generate and display Mermaid diagram of the executor graph",
        ),
    ] = False,
    interrupt_before: Annotated[
        list[str] | None,
        typer.Option(
            "--interrupt-before",
            help="Nodes to pause before (graph mode only). Valid: execute_task, verify_task, checkpoint",
        ),
    ] = None,
    interrupt_after: Annotated[
        list[str] | None,
        typer.Option(
            "--interrupt-after",
            help="Nodes to pause after (graph mode only). Valid: execute_task, verify_task, checkpoint",
        ),
    ] = None,
    max_iterations: Annotated[
        int,
        typer.Option(
            "--max-iterations",
            help="Maximum graph transitions before abort (default: 100, LangGraph default is 25)",
        ),
    ] = 100,
):
    """Run the executor on a ROADMAP.md file."""
    # Validate roadmap exists
    if not roadmap.exists():
        console.print(f"[red]Error: ROADMAP file not found: {roadmap}[/red]")
        raise typer.Exit(1)

    # Parse adaptive mode
    mode_map = {
        "no-adapt": AdaptiveMode.NO_ADAPT,
        "suggest": AdaptiveMode.SUGGEST,
        "auto-fix": AdaptiveMode.AUTO_FIX,
    }
    parsed_mode = mode_map.get(adaptive_mode.lower())
    if parsed_mode is None:
        console.print(f"[red]Invalid adaptive mode: {adaptive_mode}[/red]")
        console.print("Valid options: no-adapt, suggest, auto-fix")
        raise typer.Exit(1)

    # Parse verify mode (Phase 5.9.2)
    verify_mode_map = {
        "auto": VerifyMode.AUTO,
        "agent": VerifyMode.AGENT,
        "skip": VerifyMode.SKIP,
        "pytest": VerifyMode.PYTEST,
    }
    parsed_verify_mode = verify_mode_map.get(verify_mode.lower())
    if parsed_verify_mode is None:
        console.print(f"[red]Invalid verify mode: {verify_mode}[/red]")
        console.print("Valid options: auto, agent, skip, pytest")
        raise typer.Exit(1)

    # Phase 5.8.5: Clear project memory if requested
    if clear_project_memory:
        memory_path = roadmap.parent / ".executor" / "project-memory.json"
        if memory_path.exists():
            memory_path.unlink()
            if not output_json:
                console.print("[yellow]Cleared project memory[/yellow]")
        elif not output_json:
            console.print("[dim]No project memory to clear[/dim]")

    # Create config
    config = ExecutorConfig(
        model=model,
        max_tasks=max_tasks,
        dry_run=dry_run,
        skip_verification=skip_verification,
        stop_on_failure=stop_on_failure,
        require_human_approval_after=require_approval,
        pause_before_destructive=pause_destructive,
        checkpoint_every=checkpoint,
        sync_roadmap=sync_roadmap,
        retry_failed=retry_failed,
        adaptive_mode=parsed_mode,
        verify_mode=parsed_verify_mode,
        # Phase 5.8.5: Memory configuration
        enable_run_memory=not no_run_memory,
        enable_project_memory=not no_project_memory,
        memory_token_budget=memory_budget,
        extract_outcomes_with_llm=extract_with_llm,
        # Phase 5.13: LLM normalization
        normalize_with_llm=normalize_with_llm,
    )

    # Show what we're doing
    if not output_json:
        if dry_run:
            console.print("[yellow]Dry run mode - no changes will be made[/yellow]\n")
        console.print(f"Running executor on [cyan]{roadmap}[/cyan]")
        if max_tasks > 0:
            console.print(f"  Max tasks: {max_tasks}")
        console.print(f"  Model: {model}")
        if retry_failed > 1:
            console.print(f"  Retry attempts: {retry_failed}")
            console.print(f"  Adaptive mode: {adaptive_mode}")
        if normalize_with_llm:
            console.print("  [cyan]LLM normalization: enabled[/cyan]")
        # Phase 1.8.1: Show graph mode
        if graph_mode:
            console.print("  [cyan]Mode: graph[/cyan]")
            if interrupt_before:
                console.print(f"    Interrupt before: {', '.join(interrupt_before)}")
            if interrupt_after:
                console.print(f"    Interrupt after: {', '.join(interrupt_after)}")
        else:
            console.print("  [dim]Mode: legacy[/dim]")
        console.print()

    # Phase 1.8.1: Handle --visualize (just show diagram and exit)
    if visualize:
        from ai_infra.executor.graph import ExecutorGraph

        try:
            graph_executor = ExecutorGraph(
                agent=None,
                roadmap_path=str(roadmap),
            )
            mermaid = graph_executor.get_mermaid()
            console.print("[bold]Executor Graph Diagram (Mermaid)[/bold]\n")
            console.print("```mermaid")
            console.print(mermaid)
            console.print("```")
            console.print("\n[dim]Copy the above diagram to a Mermaid-compatible viewer.[/dim]")
            return
        except Exception as e:
            console.print(f"[red]Error generating graph diagram: {e}[/red]")
            raise typer.Exit(1)

    # Create callbacks for observability (token tracking, metrics)
    callbacks = ExecutorCallbacks()

    # Create agent for task execution (unless dry-run, but still needed for LLM normalization)
    agent = None
    needs_agent = not dry_run or normalize_with_llm
    if needs_agent:
        try:
            # Use workspace in sandboxed mode to confine agent to project directory
            # "sandboxed" prevents filesystem access outside roadmap.parent
            from typing import Literal, cast

            workspace_mode = cast(
                Literal["virtual", "sandboxed", "full"], "sandboxed" if not dry_run else "virtual"
            )
            workspace = Workspace(roadmap.parent, mode=workspace_mode)
            agent = Agent(
                deep=True,
                model_name=model,
                workspace=workspace,
                callbacks=callbacks,  # Track LLM tokens
            )
        except Exception as e:
            console.print(f"[red]Error creating agent: {e}[/red]")
            raise typer.Exit(1)

    # Create executor
    try:
        from typing import Any, cast

        executor = Executor(
            roadmap=roadmap, config=config, agent=cast(Any, agent), callbacks=callbacks
        )
    except Exception as e:
        console.print(f"[red]Error creating executor: {e}[/red]")
        raise typer.Exit(1)

    # Validate execution mode
    use_by_todos = execution_mode.lower() == "by-todos"
    if execution_mode.lower() not in ("by-tasks", "by-todos"):
        console.print(f"[red]Invalid execution mode: {execution_mode}[/red]")
        console.print("Valid options: by-tasks, by-todos")
        raise typer.Exit(1)

    # Phase 1.8.2: Run using graph or legacy executor
    async def _run():
        if graph_mode:
            # Use graph-based executor (default)
            from ai_infra.executor.graph import ExecutorGraph

            try:
                # Ensure todo_manager is initialized for ROADMAP sync
                await executor.ensure_todo_manager()

                graph_executor = ExecutorGraph(
                    agent=agent,
                    roadmap_path=str(roadmap),
                    checkpointer=executor.checkpointer,
                    verifier=executor.verifier if hasattr(executor, "verifier") else None,
                    todo_manager=executor.todo_manager
                    if hasattr(executor, "todo_manager")
                    else None,
                    callbacks=callbacks,  # Phase 2.2.1: Pass callbacks for token tracking
                    use_llm_normalization=normalize_with_llm,
                    sync_roadmap=sync_roadmap,
                    max_tasks=max_tasks if max_tasks > 0 else None,
                    max_retries=retry_failed,  # Phase 2.2.2: Configurable retry count
                    dry_run=dry_run,  # Phase 2.3.2: Dry run mode
                    pause_destructive=pause_destructive,  # Phase 2.3.3: Pause destructive
                    enable_planning=enable_planning,  # Phase 2.4.2: Pre-execution planning
                    adaptive_mode=parsed_mode,  # Phase 2.3.1: Adaptive replanning mode
                    recursion_limit=max_iterations,  # Phase 1.6: Max graph transitions
                    interrupt_before=interrupt_before,
                    interrupt_after=interrupt_after,
                )
                result = await graph_executor.arun()

                # Convert graph result to RunSummary for consistent output
                completed = result.get("tasks_completed_count", 0)
                failed = len(result.get("failed_todos", []))
                total = len(result.get("todos", []))

                # Phase 2.4.3: Capture node metrics for display
                node_metrics_data = result.get("node_metrics")

                summary = RunSummary(
                    status=(
                        RunStatus.COMPLETED
                        if failed == 0 and not result.get("interrupt_requested")
                        else RunStatus.PAUSED
                        if result.get("interrupt_requested")
                        else RunStatus.FAILED
                    ),
                    total_tasks=total,
                    tasks_completed=completed,
                    tasks_failed=failed,
                    tasks_remaining=total - completed - failed,
                    tasks_skipped=0,
                    duration_ms=float(result.get("duration_ms") or 0),  # type: ignore[arg-type]
                    total_tokens=int(result.get("tokens_used") or 0),  # type: ignore[call-overload]
                    results=[],
                    paused=result.get("interrupt_requested", False),
                    pause_reason="HITL interrupt" if result.get("interrupt_requested") else "",
                )
                # Return tuple with summary and node_metrics for graph mode
                return (summary, node_metrics_data)
            except ImportError as e:
                if not output_json:
                    console.print(
                        f"[yellow]Graph executor unavailable ({e}), falling back to legacy mode[/yellow]"
                    )
                # Fall through to legacy mode
            except Exception as e:
                if not output_json:
                    console.print(
                        f"[yellow]Graph executor failed ({e}), falling back to legacy mode[/yellow]"
                    )
                # Fall through to legacy mode

        # Legacy mode (fallback)
        if use_by_todos:
            await executor.ensure_todo_manager()
            if not output_json:
                todo_manager = executor.todo_manager
                console.print("[cyan]Execution mode: by-todos[/cyan]")
                if normalize_with_llm:
                    console.print(f"  LLM-normalized todos: {todo_manager.total_count}")
                    console.print(f"  JSON-only mode: {todo_manager.uses_json_only}")
                else:
                    console.print(f"  Raw tasks: {executor.roadmap.total_tasks}")
                    console.print(f"  Grouped todos: {todo_manager.total_count}")
                console.print()
            return (await executor.run_by_todos(), None)
        return (await executor.run(), None)

    try:
        run_result = asyncio.run(_run())
        # Handle both graph mode (tuple) and legacy mode (just summary)
        if isinstance(run_result, tuple):
            summary, node_metrics = run_result
        else:
            summary = run_result
            node_metrics = None
    except KeyboardInterrupt:
        console.print("\n[yellow]Execution interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"[red]Execution error: {e}[/red]")
        raise typer.Exit(1)

    # Output results
    if output_json:
        import json

        output_data = summary.to_dict()
        # Phase 2.4.3: Include node_metrics in JSON output
        if node_metrics:
            output_data["node_metrics"] = node_metrics
        console.print(json.dumps(output_data, indent=2))
    else:
        # Render summary panel
        console.print(_render_run_summary(summary))

        # Phase 2.4.3: Render per-node cost breakdown
        if node_metrics:
            node_panel = _render_node_metrics(node_metrics)
            if node_panel:
                console.print()
                console.print(node_panel)

        # Render results table if we have results
        if summary.results:
            console.print()
            console.print(_render_results_table(summary))

        # Show next steps if paused
        if summary.paused:
            console.print()
            console.print("[yellow]Execution paused. Review changes and run:[/yellow]")
            console.print(f"  [cyan]ai-infra executor resume --roadmap {roadmap} --approve[/cyan]")
            console.print("Or to reject and rollback:")
            console.print(
                f"  [cyan]ai-infra executor resume --roadmap {roadmap} --reject --rollback[/cyan]"
            )

    # Exit with appropriate code
    if summary.status == RunStatus.FAILED:
        raise typer.Exit(1)


# =============================================================================
# executor status - Show current status
# =============================================================================


@app.command("status")
def status_cmd(
    roadmap: Annotated[
        Path,
        typer.Option(
            "--roadmap",
            "-r",
            help="Path to ROADMAP.md file",
            exists=True,
            dir_okay=False,
        ),
    ] = Path("./ROADMAP.md"),
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output as JSON",
        ),
    ] = False,
):
    """Show current executor status for a ROADMAP.md file."""
    if not roadmap.exists():
        console.print(f"[red]Error: ROADMAP file not found: {roadmap}[/red]")
        raise typer.Exit(1)

    try:
        executor = Executor(roadmap=roadmap)
    except Exception as e:
        console.print(f"[red]Error loading executor: {e}[/red]")
        raise typer.Exit(1)

    # Get state summary
    state_summary = executor.state.get_summary()

    if output_json:
        import json

        data = {
            "roadmap": str(roadmap),
            "run_id": executor.state.run_id,
            "total_tasks": executor.roadmap.total_tasks,
            "completed": state_summary.completed,
            "failed": state_summary.failed,
            "in_progress": state_summary.in_progress,
            "pending": state_summary.pending,
            "progress": state_summary.completed / executor.roadmap.total_tasks
            if executor.roadmap.total_tasks > 0
            else 0,
        }
        console.print(json.dumps(data, indent=2))
    else:
        # Summary panel
        total = executor.roadmap.total_tasks
        completed = state_summary.completed
        progress = completed / total if total > 0 else 0

        lines = [
            f"[bold]ROADMAP:[/bold]     {roadmap}",
            f"[bold]Run ID:[/bold]      {executor.state.run_id}",
            f"[bold]Progress:[/bold]    {completed}/{total} ({progress:.0%})",
            f"  • Completed:   [green]{state_summary.completed}[/green]",
            f"  • Failed:      [red]{state_summary.failed}[/red]",
            f"  • In Progress: [yellow]{state_summary.in_progress}[/yellow]",
            f"  • Pending:     [dim]{state_summary.pending}[/dim]",
        ]

        console.print(Panel("\n".join(lines), title="Executor Status", border_style="blue"))

        # Task table
        console.print()
        console.print(_render_status_table(executor))


# =============================================================================
# executor resume - Resume after pause
# =============================================================================


@app.command("resume")
def resume_cmd(
    roadmap: Annotated[
        Path,
        typer.Option(
            "--roadmap",
            "-r",
            help="Path to ROADMAP.md file",
            exists=True,
            dir_okay=False,
        ),
    ] = Path("./ROADMAP.md"),
    approve: Annotated[
        bool,
        typer.Option(
            "--approve",
            help="Approve pending changes and continue",
        ),
    ] = False,
    reject: Annotated[
        bool,
        typer.Option(
            "--reject",
            help="Reject pending changes",
        ),
    ] = False,
    rollback: Annotated[
        bool,
        typer.Option(
            "--rollback",
            help="Rollback git changes (only with --reject)",
        ),
    ] = False,
):
    """Resume executor after a pause."""
    if not roadmap.exists():
        console.print(f"[red]Error: ROADMAP file not found: {roadmap}[/red]")
        raise typer.Exit(1)

    # Validate options
    if approve and reject:
        console.print("[red]Error: Cannot use both --approve and --reject[/red]")
        raise typer.Exit(1)

    if not approve and not reject:
        console.print("[red]Error: Must specify either --approve or --reject[/red]")
        raise typer.Exit(1)

    if rollback and not reject:
        console.print("[yellow]Warning: --rollback only has effect with --reject[/yellow]")

    try:
        executor = Executor(roadmap=roadmap)
    except Exception as e:
        console.print(f"[red]Error loading executor: {e}[/red]")
        raise typer.Exit(1)

    # Show what's being reviewed
    review = executor.get_changes_for_review()
    if review.task_ids:
        console.print(_render_review_info(review))
        console.print()

    # Resume with approval or rejection
    if approve:
        executor.resume(approved=True)
        console.print("[green]Changes approved. Ready to continue.[/green]")
        console.print(f"Run [cyan]ai-infra executor run --roadmap {roadmap}[/cyan] to continue.")
    else:
        result = executor.resume(approved=False, rollback=rollback)
        if rollback and result:
            if result.success:
                console.print(
                    f"[green]Rolled back {result.commits_reverted} commit(s) "
                    f"to {result.target_sha}[/green]"
                )
            else:
                console.print(f"[red]Rollback failed: {result.error}[/red]")
        console.print("[yellow]Changes rejected. In-progress tasks reset.[/yellow]")


# =============================================================================
# executor rollback - Rollback last task
# =============================================================================


@app.command("rollback")
def rollback_cmd(
    roadmap: Annotated[
        Path,
        typer.Option(
            "--roadmap",
            "-r",
            help="Path to ROADMAP.md file",
            exists=True,
            dir_okay=False,
        ),
    ] = Path("./ROADMAP.md"),
    task_id: Annotated[
        str | None,
        typer.Option(
            "--task",
            "-t",
            help="Task ID to rollback to (default: last completed task)",
        ),
    ] = None,
    hard: Annotated[
        bool,
        typer.Option(
            "--hard",
            help="Hard reset (discard all changes)",
        ),
    ] = False,
):
    """Rollback to the state before a task was executed."""
    if not roadmap.exists():
        console.print(f"[red]Error: ROADMAP file not found: {roadmap}[/red]")
        raise typer.Exit(1)

    try:
        executor = Executor(roadmap=roadmap)
    except Exception as e:
        console.print(f"[red]Error loading executor: {e}[/red]")
        raise typer.Exit(1)

    checkpointer = executor.checkpointer
    if checkpointer is None:
        console.print("[red]Error: Checkpointing is not enabled or not in a git repository[/red]")
        raise typer.Exit(1)

    # Get task to rollback
    if task_id is None:
        # Get last completed task
        completed_tasks = [
            tid
            for tid in executor.state._tasks
            if executor.state.get_status(tid) == TaskStatus.COMPLETED
        ]
        if not completed_tasks:
            console.print("[red]Error: No completed tasks to rollback[/red]")
            raise typer.Exit(1)
        task_id = completed_tasks[-1]

    console.print(f"Rolling back task [cyan]{task_id}[/cyan]...")

    result = checkpointer.rollback(task_id, hard=hard)

    if result.success:
        console.print(
            f"[green]Successfully rolled back {result.commits_reverted} commit(s)[/green]"
        )
        console.print(f"  Target: {result.target_sha}")
        console.print(f"  Message: {result.message}")

        # Reset state for the rolled back task
        executor.state.reset_task(task_id)
        executor.state.save()
        console.print(f"  Task [cyan]{task_id}[/cyan] reset to pending")
    else:
        console.print(f"[red]Rollback failed: {result.error}[/red]")
        raise typer.Exit(1)


# =============================================================================
# executor reset - Reset executor state
# =============================================================================


@app.command("reset")
def reset_cmd(
    roadmap: Annotated[
        Path,
        typer.Option(
            "--roadmap",
            "-r",
            help="Path to ROADMAP.md file",
            exists=True,
            dir_okay=False,
        ),
    ] = Path("./ROADMAP.md"),
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Skip confirmation prompt",
        ),
    ] = False,
):
    """Reset executor state completely (re-parse ROADMAP)."""
    if not roadmap.exists():
        console.print(f"[red]Error: ROADMAP file not found: {roadmap}[/red]")
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(
            "This will reset all executor state. Tasks will be re-read from ROADMAP.md. Continue?"
        )
        if not confirm:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(0)

    try:
        executor = Executor(roadmap=roadmap)
        executor.reset()
        console.print("[green]Executor state reset successfully.[/green]")
        console.print(f"  Tasks: {executor.roadmap.total_tasks}")
        console.print(f"  Run ID: {executor.state.run_id}")
    except Exception as e:
        console.print(f"[red]Error resetting executor: {e}[/red]")
        raise typer.Exit(1)


# =============================================================================
# executor sync-roadmap - Sync from todos.json to ROADMAP (Phase 5.13.5)
# =============================================================================


@app.command("sync-roadmap")
def sync_roadmap_cmd(
    roadmap: Annotated[
        Path,
        typer.Option(
            "--roadmap",
            "-r",
            help="Path to ROADMAP.md file",
            exists=True,
            dir_okay=False,
        ),
    ] = Path("./ROADMAP.md"),
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show what would be updated without making changes",
        ),
    ] = False,
):
    """Sync completed todos from .executor/todos.json back to ROADMAP.md.

    This command reads the normalized todos from .executor/todos.json
    (created during executor run with LLM normalization) and updates
    the checkboxes in the original ROADMAP.md file.

    Use this after execution to update the ROADMAP with completion status.

    Example:
        ai-infra executor sync-roadmap --roadmap ./ROADMAP.md
        ai-infra executor sync-roadmap --dry-run
    """
    from ai_infra.executor.todolist import NormalizedTodoFile, TodoListManager

    todos_json = roadmap.parent / ".executor" / "todos.json"

    if not todos_json.exists():
        console.print(
            f"[red]Error: No todos.json found at {todos_json}[/red]\n"
            "[dim]Run 'executor run' first to create normalized todos.[/dim]"
        )
        raise typer.Exit(1)

    try:
        # Load and display what will be synced
        todo_file = NormalizedTodoFile.load(todos_json)
        completed = [t for t in todo_file.todos if t.status == "completed"]
        pending = [t for t in todo_file.todos if t.status == "pending"]
        skipped = [t for t in todo_file.todos if t.status == "skipped"]

        # Display summary
        table = Table(title="Todos Status Summary")
        table.add_column("Status", style="bold")
        table.add_column("Count", justify="right")
        table.add_row("[green]Completed[/green]", str(len(completed)))
        table.add_row("[yellow]Pending[/yellow]", str(len(pending)))
        table.add_row("[dim]Skipped[/dim]", str(len(skipped)))
        console.print(table)

        if not completed:
            console.print("\n[dim]No completed todos to sync.[/dim]")
            return

        if dry_run:
            console.print("\n[bold]Completed todos to sync:[/bold]")
            for todo in completed:
                console.print(f"  [green]✓[/green] {todo.title}")
            console.print("\n[dim]Dry run - no changes made.[/dim]")
            return

        # Perform the sync
        updated = TodoListManager.sync_json_to_roadmap(roadmap)

        if updated > 0:
            console.print(
                f"\n[green]Successfully updated {updated} checkbox(es) in ROADMAP.md[/green]"
            )
        else:
            console.print(
                "\n[yellow]Warning: No checkboxes were updated.[/yellow]\n"
                "[dim]The completed todos may not have matching checkboxes in ROADMAP.[/dim]"
            )

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error syncing roadmap: {e}[/red]")
        raise typer.Exit(1)


# =============================================================================
# executor sync - Sync state to ROADMAP
# =============================================================================


@app.command("sync")
def sync_cmd(
    roadmap: Annotated[
        Path,
        typer.Option(
            "--roadmap",
            "-r",
            help="Path to ROADMAP.md file",
            exists=True,
            dir_okay=False,
        ),
    ] = Path("./ROADMAP.md"),
):
    """Sync completed tasks back to ROADMAP.md checkboxes."""
    if not roadmap.exists():
        console.print(f"[red]Error: ROADMAP file not found: {roadmap}[/red]")
        raise typer.Exit(1)

    try:
        executor = Executor(roadmap=roadmap)
        updated = executor.sync_roadmap()

        if updated > 0:
            console.print(f"[green]Updated {updated} checkbox(es) in ROADMAP.md[/green]")
        else:
            console.print("[dim]No checkboxes to update.[/dim]")
    except Exception as e:
        console.print(f"[red]Error syncing: {e}[/red]")
        raise typer.Exit(1)


# =============================================================================
# executor review - Show changes for review
# =============================================================================


@app.command("review")
def review_cmd(
    roadmap: Annotated[
        Path,
        typer.Option(
            "--roadmap",
            "-r",
            help="Path to ROADMAP.md file",
            exists=True,
            dir_okay=False,
        ),
    ] = Path("./ROADMAP.md"),
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output as JSON",
        ),
    ] = False,
):
    """Show changes pending human review."""
    if not roadmap.exists():
        console.print(f"[red]Error: ROADMAP file not found: {roadmap}[/red]")
        raise typer.Exit(1)

    try:
        executor = Executor(roadmap=roadmap)
    except Exception as e:
        console.print(f"[red]Error loading executor: {e}[/red]")
        raise typer.Exit(1)

    review = executor.get_changes_for_review()

    if output_json:
        import json

        console.print(json.dumps(review.to_dict(), indent=2))
    else:
        if not review.task_ids:
            console.print("[dim]No changes pending review.[/dim]")
        else:
            console.print(_render_review_info(review))


# =============================================================================
# executor memory - Show project memory (Phase 5.8.5)
# =============================================================================


@app.command("memory")
def memory_cmd(
    project: Annotated[
        Path,
        typer.Argument(
            help="Project root directory (containing .executor/)",
        ),
    ] = Path("."),
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: summary, json, files, history",
        ),
    ] = "summary",
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output as JSON (shortcut for --format json)",
        ),
    ] = False,
):
    """Show project memory for a project.

    Project memory tracks:
    - Key files created/modified during execution
    - Run history (completed tasks, failures)
    - Learned patterns from previous runs

    Examples:
        ai-infra executor memory .
        ai-infra executor memory --format files
        ai-infra executor memory --format history
        ai-infra executor memory --json
    """
    from ai_infra.executor.project_memory import ProjectMemory

    # Resolve project path
    project_path = project.resolve()
    if not project_path.exists():
        console.print(f"[red]Error: Project directory not found: {project}[/red]")
        raise typer.Exit(1)

    # Load project memory
    try:
        memory = ProjectMemory.load(project_path)
    except Exception as e:
        console.print(f"[red]Error loading project memory: {e}[/red]")
        raise typer.Exit(1)

    # Override format if --json flag is used
    if output_json:
        format = "json"

    # Output based on format
    if format == "json":
        import json

        console.print(json.dumps(memory._to_dict(), indent=2))

    elif format == "files":
        if not memory.key_files:
            console.print("[dim]No files tracked in project memory.[/dim]")
        else:
            console.print(f"[bold]Key Files ({len(memory.key_files)}):[/bold]\n")
            for path, info in memory.key_files.items():
                purpose = info.purpose or "(no description)"
                created_by = (
                    f" [dim](created by {info.created_by_task})[/dim]"
                    if info.created_by_task
                    else ""
                )
                console.print(f"  [cyan]{path}[/cyan]: {purpose}{created_by}")

    elif format == "history":
        if not memory.run_history:
            console.print("[dim]No run history in project memory.[/dim]")
        else:
            console.print(f"[bold]Run History ({len(memory.run_history)} runs):[/bold]\n")
            # Show most recent runs first
            for run in reversed(memory.run_history[-10:]):
                date = run.timestamp[:10] if run.timestamp else "unknown"
                status_color = "green" if run.tasks_failed == 0 else "red"
                console.print(
                    f"  [{status_color}]{date}[/{status_color}] "
                    f"[bold]{run.run_id[:8]}...[/bold] - "
                    f"{run.tasks_completed} completed, {run.tasks_failed} failed"
                )
                if run.lessons_learned:
                    for lesson in run.lessons_learned[:2]:
                        console.print(
                            f"    └─ [dim]{lesson[:60]}...[/dim]"
                            if len(lesson) > 60
                            else f"    └─ [dim]{lesson}[/dim]"
                        )

    else:  # summary (default)
        console.print("[bold]Project Memory Summary[/bold]\n")

        # Project type
        project_type = memory.project_type or "unknown"
        console.print(f"  [bold]Project Type:[/bold] {project_type}")

        # Files tracked
        console.print(f"  [bold]Files Tracked:[/bold] {len(memory.key_files)}")

        # Run history
        console.print(f"  [bold]Runs Recorded:[/bold] {len(memory.run_history)}")

        # Last run info
        if memory.run_history:
            last = memory.run_history[-1]
            date = last.timestamp[:10] if last.timestamp else "unknown"
            status = "success" if last.tasks_failed == 0 else "had failures"
            console.print(
                f"  [bold]Last Run:[/bold] {date} - "
                f"{last.tasks_completed} completed, {last.tasks_failed} failed ({status})"
            )

        # Lessons learned count
        total_lessons = sum(len(r.lessons_learned) for r in memory.run_history)
        if total_lessons > 0:
            console.print(f"  [bold]Lessons Learned:[/bold] {total_lessons}")

        # Memory file location
        memory_file = project_path / ".executor" / "project-memory.json"
        if memory_file.exists():
            size_kb = memory_file.stat().st_size / 1024
            console.print(f"\n  [dim]Memory file: {memory_file} ({size_kb:.1f} KB)[/dim]")


@app.command("memory-clear")
def memory_clear_cmd(
    project: Annotated[
        Path,
        typer.Argument(
            help="Project root directory (containing .executor/)",
        ),
    ] = Path("."),
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Skip confirmation prompt",
        ),
    ] = False,
):
    """Clear project memory for a fresh start.

    This removes:
    - Key files map
    - Run history
    - Learned patterns

    Use this when you want the executor to start fresh without
    any context from previous runs.

    Examples:
        ai-infra executor memory-clear .
        ai-infra executor memory-clear . --force
    """
    project_path = project.resolve()
    memory_path = project_path / ".executor" / "project-memory.json"

    if not memory_path.exists():
        console.print("[dim]No project memory to clear.[/dim]")
        return

    # Confirm unless --force
    if not force:
        confirm = typer.confirm(
            f"Clear project memory at {project_path}? This cannot be undone.",
            default=False,
        )
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(0)

    # Remove the file
    try:
        memory_path.unlink()
        console.print("[green]Project memory cleared.[/green]")
    except Exception as e:
        console.print(f"[red]Error clearing memory: {e}[/red]")
        raise typer.Exit(1)


# =============================================================================
# Registration
# =============================================================================


def register(parent: typer.Typer):
    """Register executor commands with the parent CLI app."""
    parent.add_typer(app, name="executor", help="Autonomous task execution from ROADMAP.md")
