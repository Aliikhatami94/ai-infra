"""Dashboard rendering for executor progress visibility.

Phase 4.3.2-4.3.3 of EXECUTOR_1.md: Dashboard and Real-time Updates.

This module provides:
- Dashboard: Rich terminal rendering of progress
- run_with_dashboard: Real-time progress display using Rich Live

Example:
    ```python
    from ai_infra.executor.dashboard import Dashboard, run_with_dashboard
    from ai_infra.executor.progress import ProgressTracker

    # Create dashboard
    tracker = ProgressTracker(roadmap)
    dashboard = Dashboard(tracker)

    # Render once
    dashboard.render()

    # Or run with real-time updates
    async def execute_task(task):
        # ... do work
        pass

    await run_with_dashboard(tracker, execute_task, tasks)
    ```
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from ai_infra.executor.progress import ProgressTracker, TaskProgress

if TYPE_CHECKING:
    from ai_infra.executor.models import Task

logger = logging.getLogger(__name__)


# =============================================================================
# Status Icons
# =============================================================================

STATUS_ICONS: dict[str, str] = {
    "pending": "[dim]○[/dim]",
    "in_progress": "[cyan]◉[/cyan]",
    "completed": "[green]●[/green]",
    "failed": "[red]✗[/red]",
    "skipped": "[yellow]◌[/yellow]",
}

STATUS_LABELS: dict[str, str] = {
    "pending": "[dim]Pending[/dim]",
    "in_progress": "[cyan]Running[/cyan]",
    "completed": "[green]Done[/green]",
    "failed": "[red]Failed[/red]",
    "skipped": "[yellow]Skipped[/yellow]",
}


# =============================================================================
# Dashboard Configuration
# =============================================================================


@dataclass
class DashboardConfig:
    """Configuration for dashboard rendering.

    Attributes:
        show_all_tasks: Whether to show all tasks or just recent.
        max_visible_tasks: Maximum number of tasks to display at once.
        show_cost_estimates: Whether to show cost estimates.
        show_file_changes: Whether to show file modifications.
        show_tokens: Whether to show token usage.
        compact_mode: Use compact layout.
        refresh_rate: Refresh rate for live updates (seconds).
    """

    show_all_tasks: bool = False
    max_visible_tasks: int = 15
    show_cost_estimates: bool = True
    show_file_changes: bool = False
    show_tokens: bool = True
    compact_mode: bool = False
    refresh_rate: float = 0.5


# =============================================================================
# Dashboard (Phase 4.3.2)
# =============================================================================


class Dashboard:
    """Rich terminal dashboard for progress visualization.

    Phase 4.3.2: Provides beautiful terminal UI for progress tracking.

    Attributes:
        tracker: The ProgressTracker to visualize.
        config: Dashboard configuration options.
        console: Rich Console for rendering.

    Example:
        ```python
        tracker = ProgressTracker(roadmap)
        dashboard = Dashboard(tracker)

        # Render to console
        dashboard.render()

        # Or get as renderable
        panel = dashboard.create_panel()
        console.print(panel)
        ```
    """

    def __init__(
        self,
        tracker: ProgressTracker,
        config: DashboardConfig | None = None,
        console: Console | None = None,
    ) -> None:
        """Initialize the dashboard.

        Args:
            tracker: ProgressTracker to visualize.
            config: Optional configuration.
            console: Optional Rich Console (creates one if not provided).
        """
        self.tracker = tracker
        self.config = config or DashboardConfig()
        self.console = console or Console()

    def render(self) -> None:
        """Render the dashboard to the console."""
        panel = self.create_panel()
        self.console.print(panel)

    def create_panel(self) -> Panel:
        """Create the dashboard panel.

        Returns:
            Rich Panel containing the dashboard.
        """
        # Get summary
        summary = self.tracker.get_summary()

        # Build content
        content = Table.grid(padding=(0, 1))
        content.add_column(justify="left")

        # Add header with progress bar
        header = self._create_header(summary)
        content.add_row(header)
        content.add_row("")

        # Add task list
        task_table = self._create_task_table()
        content.add_row(task_table)
        content.add_row("")

        # Add footer with stats
        footer = self._create_footer(summary)
        content.add_row(footer)

        # Create panel
        title = f"[bold]{self.tracker.roadmap_title}[/bold]"
        subtitle = f"[dim]{summary.completed}/{summary.total} tasks[/dim]"

        return Panel(
            content,
            title=title,
            subtitle=subtitle,
            border_style="blue",
        )

    def _create_header(self, summary: Any) -> Table:
        """Create the header section with progress bar.

        Args:
            summary: ProgressSummary from tracker.

        Returns:
            Rich Table with header content.
        """
        table = Table.grid(expand=True)
        table.add_column(justify="left", ratio=2)
        table.add_column(justify="right", ratio=1)

        # Progress info
        left = Text()
        left.append(f"{summary.percent:.0f}% Complete", style="bold")

        # Current task
        if summary.current_task_title:
            left.append(f"\n[cyan]Current: {summary.current_task_title}[/cyan]")

        # Time info
        right = Text()
        if summary.elapsed:
            elapsed = self._format_duration(summary.elapsed.total_seconds())
            right.append(f"Elapsed: {elapsed}")

            if summary.estimated_remaining_time:
                remaining = self._format_duration(summary.estimated_remaining_time.total_seconds())
                right.append(f"\nRemaining: ~{remaining}")

        table.add_row(left, right)

        # Add progress bar
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            expand=True,
        )
        task_id = progress.add_task("Progress", total=100, completed=summary.percent)
        _ = task_id  # Suppress unused warning

        outer = Table.grid(expand=True)
        outer.add_column()
        outer.add_row(table)
        outer.add_row("")
        outer.add_row(progress)

        return outer

    def _create_task_table(self) -> Table:
        """Create the task list table.

        Returns:
            Rich Table with task list.
        """
        table = Table(show_header=True, header_style="bold", expand=True)
        table.add_column("", width=3, justify="center")  # Status icon
        table.add_column("Task", ratio=3)
        table.add_column("Status", width=10)

        if self.config.show_tokens:
            table.add_column("Tokens", justify="right", width=10)

        if self.config.show_cost_estimates:
            table.add_column("Cost", justify="right", width=10)

        table.add_column("Duration", justify="right", width=10)

        # Get tasks to display
        all_progress = self.tracker.get_all_progress()

        # Determine which tasks to show
        if self.config.show_all_tasks:
            display_tasks = all_progress
        else:
            # Show in-progress, recently completed, and upcoming
            display_tasks = self._get_relevant_tasks(all_progress)

        for progress in display_tasks[: self.config.max_visible_tasks]:
            self._add_task_row(table, progress)

        # Show truncation indicator
        total_hidden = len(display_tasks) - min(len(display_tasks), self.config.max_visible_tasks)
        if total_hidden > 0:
            table.add_row(
                "",
                f"[dim]... and {total_hidden} more tasks[/dim]",
                "",
                *[""] * (table.column_count - 3),
            )

        return table

    def _add_task_row(self, table: Table, progress: TaskProgress) -> None:
        """Add a task row to the table.

        Args:
            table: Rich Table to add to.
            progress: TaskProgress for the task.
        """
        icon = STATUS_ICONS.get(progress.status, "○")
        status = STATUS_LABELS.get(progress.status, progress.status)

        # Task title (truncate if too long)
        title = progress.task_title
        if len(title) > 50:
            title = title[:47] + "..."

        # Format task ID and title
        task_cell = f"[dim]{progress.task_id}[/dim] {title}"

        # Token count
        tokens = ""
        if self.config.show_tokens and progress.total_tokens > 0:
            tokens = f"{progress.total_tokens:,}"

        # Cost
        cost = ""
        if self.config.show_cost_estimates and progress.cost > 0:
            cost = f"${progress.cost:.3f}"

        # Duration
        duration = ""
        if progress.duration_seconds > 0:
            duration = self._format_duration(progress.duration_seconds)

        row = [icon, task_cell, status]
        if self.config.show_tokens:
            row.append(tokens)
        if self.config.show_cost_estimates:
            row.append(cost)
        row.append(duration)

        table.add_row(*row)

    def _get_relevant_tasks(self, all_progress: list[TaskProgress]) -> list[TaskProgress]:
        """Get relevant tasks to display.

        Prioritizes in-progress and recently completed tasks.

        Args:
            all_progress: All task progress items.

        Returns:
            Filtered list of tasks to display.
        """
        in_progress = [p for p in all_progress if p.status == "in_progress"]
        completed = [p for p in all_progress if p.status == "completed"]
        failed = [p for p in all_progress if p.status == "failed"]
        pending = [p for p in all_progress if p.status == "pending"]

        # Show: in_progress first, then recent completed, then upcoming pending
        result = []
        result.extend(in_progress)
        result.extend(completed[-3:])  # Last 3 completed
        result.extend(failed)  # All failed
        result.extend(pending[:5])  # Next 5 pending

        return result

    def _create_footer(self, summary: Any) -> Table:
        """Create the footer section with stats.

        Args:
            summary: ProgressSummary from tracker.

        Returns:
            Rich Table with footer content.
        """
        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="center")
        table.add_column(justify="right")

        # Left: Status counts
        left = Text()
        left.append(f"[green]{summary.completed}[/green] done  ")
        left.append(f"[cyan]{summary.in_progress}[/cyan] running  ")
        left.append(f"[dim]{summary.pending}[/dim] pending")
        if summary.failed > 0:
            left.append(f"  [red]{summary.failed}[/red] failed")

        # Center: Token count
        center = Text()
        if self.config.show_tokens:
            center.append(f"[dim]Tokens:[/dim] {summary.total_tokens:,}", style="dim")

        # Right: Cost
        right = Text()
        if self.config.show_cost_estimates:
            right.append(f"[dim]Cost:[/dim] ${summary.cost:.2f}")
            if summary.estimated_remaining_cost:
                right.append(f" (+${summary.estimated_remaining_cost:.2f} est)")

        table.add_row(left, center, right)

        return table

    def _format_duration(self, seconds: float) -> str:
        """Format a duration in seconds to human-readable string.

        Args:
            seconds: Duration in seconds.

        Returns:
            Formatted string like "1m 23s" or "2h 15m".
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def create_summary_text(self) -> str:
        """Create a plain text summary.

        Returns:
            Plain text summary string.
        """
        summary = self.tracker.get_summary()

        lines = [
            f"Progress: {summary.completed}/{summary.total} ({summary.percent:.0f}%)",
            f"Status: {summary.completed} done, {summary.in_progress} running, "
            f"{summary.pending} pending",
        ]

        if summary.failed > 0:
            lines.append(f"Failed: {summary.failed}")

        if summary.cost > 0:
            lines.append(f"Cost: ${summary.cost:.2f}")

        if summary.elapsed:
            lines.append(f"Elapsed: {self._format_duration(summary.elapsed.total_seconds())}")

        return "\n".join(lines)


# =============================================================================
# Real-time Dashboard Runner (Phase 4.3.3)
# =============================================================================


@dataclass
class DashboardRunResult:
    """Result from a dashboard run.

    Attributes:
        success: Whether all tasks completed successfully.
        completed: Number of completed tasks.
        failed: Number of failed tasks.
        total_cost: Total cost incurred.
        total_tokens: Total tokens used.
        errors: List of error messages from failed tasks.
    """

    success: bool
    completed: int
    failed: int
    total_cost: float
    total_tokens: int
    errors: list[str]


async def run_with_dashboard(
    tracker: ProgressTracker,
    execute_fn: Callable[[Task], Coroutine[Any, Any, dict[str, Any]]],
    tasks: list[Task],
    config: DashboardConfig | None = None,
    console: Console | None = None,
) -> DashboardRunResult:
    """Run tasks with a live-updating dashboard.

    Phase 4.3.3: Real-time progress visualization using Rich Live.

    This function executes tasks sequentially while displaying a live-updating
    dashboard showing progress, token usage, and cost estimates.

    Args:
        tracker: ProgressTracker instance.
        execute_fn: Async function that executes a task and returns result dict
                   with keys: tokens_in, tokens_out, cost, files_modified, files_created
        tasks: List of tasks to execute.
        config: Optional dashboard configuration.
        console: Optional Rich Console.

    Returns:
        DashboardRunResult with execution summary.

    Example:
        ```python
        async def execute_task(task):
            # Execute task with agent
            result = await agent.execute(task)
            return {
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "cost": result.cost,
                "files_modified": result.files_modified,
                "files_created": result.files_created,
            }

        result = await run_with_dashboard(tracker, execute_task, tasks)
        if not result.success:
            print(f"Failed tasks: {result.errors}")
        ```
    """
    console = console or Console()
    config = config or DashboardConfig()
    dashboard = Dashboard(tracker, config, console)

    errors: list[str] = []
    tracker.start_run()

    with Live(
        dashboard.create_panel(),
        console=console,
        refresh_per_second=1 / config.refresh_rate,
        transient=False,
    ) as live:
        for task in tasks:
            # Start task tracking
            tracker.start_task(task.id)

            # Update dashboard
            live.update(dashboard.create_panel())

            try:
                # Execute the task
                result = await execute_fn(task)

                # Complete the task
                tracker.complete_task(
                    task.id,
                    tokens_in=result.get("tokens_in", 0),
                    tokens_out=result.get("tokens_out", 0),
                    cost=result.get("cost", 0.0),
                    files_modified=result.get("files_modified", []),
                    files_created=result.get("files_created", []),
                )

            except Exception as e:
                error_msg = str(e)
                tracker.fail_task(task.id, error_msg)
                errors.append(f"{task.id}: {error_msg}")
                logger.error(
                    "Task execution failed",
                    extra={"task_id": task.id, "error": error_msg},
                )

            # Update dashboard
            live.update(dashboard.create_panel())

    # Get final summary
    summary = tracker.get_summary()

    return DashboardRunResult(
        success=summary.failed == 0,
        completed=summary.completed,
        failed=summary.failed,
        total_cost=summary.cost,
        total_tokens=summary.total_tokens,
        errors=errors,
    )


async def run_with_simple_progress(
    tracker: ProgressTracker,
    execute_fn: Callable[[Task], Coroutine[Any, Any, dict[str, Any]]],
    tasks: list[Task],
    console: Console | None = None,
) -> DashboardRunResult:
    """Run tasks with a simple progress bar (no full dashboard).

    Alternative to run_with_dashboard for simpler output.

    Args:
        tracker: ProgressTracker instance.
        execute_fn: Async function that executes a task.
        tasks: List of tasks to execute.
        console: Optional Rich Console.

    Returns:
        DashboardRunResult with execution summary.
    """
    console = console or Console()
    errors: list[str] = []
    tracker.start_run()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        progress_task = progress.add_task(
            f"[cyan]Executing {len(tasks)} tasks...", total=len(tasks)
        )

        for task in tasks:
            progress.update(progress_task, description=f"[cyan]{task.title[:40]}...")

            tracker.start_task(task.id)

            try:
                result = await execute_fn(task)
                tracker.complete_task(
                    task.id,
                    tokens_in=result.get("tokens_in", 0),
                    tokens_out=result.get("tokens_out", 0),
                    cost=result.get("cost", 0.0),
                    files_modified=result.get("files_modified", []),
                    files_created=result.get("files_created", []),
                )
            except Exception as e:
                error_msg = str(e)
                tracker.fail_task(task.id, error_msg)
                errors.append(f"{task.id}: {error_msg}")

            progress.advance(progress_task)

    summary = tracker.get_summary()

    return DashboardRunResult(
        success=summary.failed == 0,
        completed=summary.completed,
        failed=summary.failed,
        total_cost=summary.cost,
        total_tokens=summary.total_tokens,
        errors=errors,
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "Dashboard",
    "DashboardConfig",
    "DashboardRunResult",
    "STATUS_ICONS",
    "STATUS_LABELS",
    "run_with_dashboard",
    "run_with_simple_progress",
]
