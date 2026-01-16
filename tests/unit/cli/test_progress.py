"""Unit tests for ai_infra.cli.progress module.

Phase 16.6.2 of EXECUTOR_6.md: Task Progress Display.
"""

from __future__ import annotations

import time
from io import StringIO

from rich.console import Console

from ai_infra.cli.progress import (
    STATUS_ICONS,
    ETACalculator,
    LiveProgressDisplay,
    PhaseSection,
    PhaseState,
    StatusIcon,
    TaskItem,
    TaskProgressPanel,
    TaskSpinner,
    TaskStatus,
    TokenCounter,
    format_duration,
    get_status_icon,
    render_phase_list,
)

# =============================================================================
# Status Icon Tests
# =============================================================================


class TestStatusIcons:
    """Tests for status icons."""

    def test_all_statuses_have_icons(self) -> None:
        """All task statuses should have defined icons."""
        for status in TaskStatus:
            assert status in STATUS_ICONS, f"Missing icon for {status}"

    def test_status_icon_is_dataclass(self) -> None:
        """StatusIcon should be a frozen dataclass."""
        icon = STATUS_ICONS[TaskStatus.PENDING]
        assert isinstance(icon, StatusIcon)
        assert hasattr(icon, "ascii")
        assert hasattr(icon, "unicode")

    def test_ascii_icons_are_bracketed(self) -> None:
        """ASCII icons should use bracket notation."""
        for status, icon in STATUS_ICONS.items():
            assert icon.ascii.startswith("["), f"{status} ASCII icon should start with ["
            assert icon.ascii.endswith("]"), f"{status} ASCII icon should end with ]"

    def test_get_status_icon_unicode(self) -> None:
        """get_status_icon should return unicode by default."""
        icon = get_status_icon(TaskStatus.COMPLETE)
        assert icon == "●"

    def test_get_status_icon_ascii(self) -> None:
        """get_status_icon should return ASCII when requested."""
        icon = get_status_icon(TaskStatus.COMPLETE, use_unicode=False)
        assert icon == "[x]"

    def test_get_status_icon_from_string(self) -> None:
        """get_status_icon should accept string status."""
        icon = get_status_icon("complete")
        assert icon == "●"

    def test_get_status_icon_running(self) -> None:
        """Running status should have appropriate icons."""
        assert get_status_icon(TaskStatus.RUNNING, use_unicode=True) == "◐"
        assert get_status_icon(TaskStatus.RUNNING, use_unicode=False) == "[~]"

    def test_get_status_icon_failed(self) -> None:
        """Failed status should have appropriate icons."""
        assert get_status_icon(TaskStatus.FAILED, use_unicode=True) == "✗"
        assert get_status_icon(TaskStatus.FAILED, use_unicode=False) == "[!]"


# =============================================================================
# Duration Formatting Tests
# =============================================================================


class TestFormatDuration:
    """Tests for duration formatting."""

    def test_sub_second(self) -> None:
        """Durations under 1 second should show one decimal."""
        assert format_duration(0.3) == "0.3s"
        assert format_duration(0.15) == "0.1s"  # Rounds to 0.1 with %.1f
        assert format_duration(0.0) == "0.0s"

    def test_seconds(self) -> None:
        """Durations under 60 seconds should show seconds."""
        assert format_duration(12.5) == "12.5s"
        assert format_duration(59.9) == "59.9s"

    def test_minutes(self) -> None:
        """Durations under 1 hour should show minutes and seconds."""
        assert format_duration(60) == "1m 0s"
        assert format_duration(154) == "2m 34s"
        assert format_duration(3599) == "59m 59s"

    def test_hours(self) -> None:
        """Durations over 1 hour should show hours and minutes."""
        assert format_duration(3600) == "1h 0m"
        assert format_duration(4980) == "1h 23m"
        assert format_duration(7200) == "2h 0m"


# =============================================================================
# TaskItem Tests
# =============================================================================


class TestTaskItem:
    """Tests for TaskItem dataclass."""

    def test_creation_with_defaults(self) -> None:
        """TaskItem should have sensible defaults."""
        task = TaskItem(id="1.1.1", title="Test task")
        assert task.id == "1.1.1"
        assert task.title == "Test task"
        assert task.status == TaskStatus.PENDING
        assert task.duration is None
        assert task.start_time is None

    def test_string_status_conversion(self) -> None:
        """String status should be converted to enum."""
        task = TaskItem(id="1.1.1", title="Test", status="complete")
        assert task.status == TaskStatus.COMPLETE

    def test_get_elapsed_for_running_task(self) -> None:
        """get_elapsed should return time since start for running tasks."""
        start = time.time() - 5.0
        task = TaskItem(
            id="1.1.1",
            title="Test",
            status=TaskStatus.RUNNING,
            start_time=start,
        )
        elapsed = task.get_elapsed()
        assert elapsed is not None
        assert 4.9 < elapsed < 5.5  # Allow some margin

    def test_get_elapsed_for_completed_task(self) -> None:
        """get_elapsed should return duration for completed tasks."""
        task = TaskItem(
            id="1.1.1",
            title="Test",
            status=TaskStatus.COMPLETE,
            duration=2.5,
        )
        assert task.get_elapsed() == 2.5

    def test_get_elapsed_for_pending_task(self) -> None:
        """get_elapsed should return None for pending tasks."""
        task = TaskItem(id="1.1.1", title="Test", status=TaskStatus.PENDING)
        assert task.get_elapsed() is None


# =============================================================================
# ETACalculator Tests
# =============================================================================


class TestETACalculator:
    """Tests for ETA calculation."""

    def test_empty_calculator(self) -> None:
        """Empty calculator should return 0 average."""
        calc = ETACalculator()
        assert calc.get_average_duration() == 0.0

    def test_add_completed(self) -> None:
        """Adding completed tasks should update average."""
        calc = ETACalculator()
        calc.add_completed(1.0)
        calc.add_completed(2.0)
        calc.add_completed(3.0)
        assert calc.get_average_duration() == 2.0

    def test_estimate_remaining(self) -> None:
        """Should estimate remaining time based on average."""
        calc = ETACalculator()
        calc.add_completed(2.0)
        calc.add_completed(4.0)
        # Average is 3.0
        assert calc.estimate_remaining(5) == 15.0

    def test_estimate_remaining_with_weights(self) -> None:
        """Should use weights when available."""
        calc = ETACalculator(task_weights={"a": 2.0, "b": 0.5})
        calc.add_completed(4.0)
        # Average is 4.0, total weight is 2.5
        estimate = calc.estimate_remaining(2, remaining_task_ids=["a", "b"])
        assert estimate == 10.0  # 4.0 * 2.5

    def test_format_eta_empty(self) -> None:
        """Empty calculator should return empty ETA string."""
        calc = ETACalculator()
        assert calc.format_eta(5) == ""

    def test_format_eta_formatted(self) -> None:
        """ETA should be formatted with tilde prefix."""
        calc = ETACalculator()
        calc.add_completed(60.0)
        eta = calc.format_eta(3)
        assert eta.startswith("~")
        assert "3m" in eta


# =============================================================================
# TaskProgressPanel Tests
# =============================================================================


class TestTaskProgressPanel:
    """Tests for TaskProgressPanel renderable."""

    def test_panel_creation(self) -> None:
        """Panel should be creatable with tasks."""
        tasks = [
            TaskItem(id="1.1.1", title="Task 1", status="complete", duration=0.8),
            TaskItem(id="1.1.2", title="Task 2", status="running"),
            TaskItem(id="1.1.3", title="Task 3", status="pending"),
        ]
        panel = TaskProgressPanel(phase_title="Test Phase", tasks=tasks)
        assert panel.phase_title == "Test Phase"
        assert len(panel.tasks) == 3

    def test_panel_renders(self) -> None:
        """Panel should render to console without errors."""
        tasks = [
            TaskItem(id="1.1.1", title="Task 1", status="complete", duration=0.8),
            TaskItem(id="1.1.2", title="Task 2", status="pending"),
        ]
        panel = TaskProgressPanel(phase_title="Test Phase", tasks=tasks)

        console = Console(file=StringIO(), force_terminal=True, width=80)
        console.print(panel)
        output = console.file.getvalue()
        assert "Test Phase" in output
        assert "Task 1" in output
        assert "Task 2" in output

    def test_panel_shows_progress(self) -> None:
        """Panel should show progress count."""
        tasks = [
            TaskItem(id="1.1.1", title="Task 1", status="complete", duration=0.5),
            TaskItem(id="1.1.2", title="Task 2", status="complete", duration=0.5),
            TaskItem(id="1.1.3", title="Task 3", status="pending"),
        ]
        panel = TaskProgressPanel(phase_title="Test", tasks=tasks, show_progress_bar=True)

        console = Console(file=StringIO(), force_terminal=True, width=80)
        console.print(panel)
        output = console.file.getvalue()
        assert "2/3" in output

    def test_panel_without_progress_bar(self) -> None:
        """Panel should hide progress bar when requested."""
        tasks = [TaskItem(id="1.1.1", title="Task 1", status="pending")]
        panel = TaskProgressPanel(
            phase_title="Test",
            tasks=tasks,
            show_progress_bar=False,
        )

        console = Console(file=StringIO(), force_terminal=True, width=80)
        console.print(panel)
        output = console.file.getvalue()
        assert "Progress:" not in output

    def test_panel_shows_elapsed(self) -> None:
        """Panel should show elapsed time."""
        tasks = [TaskItem(id="1.1.1", title="Task 1", status="running")]
        panel = TaskProgressPanel(
            phase_title="Test",
            tasks=tasks,
            elapsed=65.0,
        )

        console = Console(file=StringIO(), force_terminal=True, width=80)
        console.print(panel)
        output = console.file.getvalue()
        assert "1m 5s" in output

    def test_panel_shows_eta(self) -> None:
        """Panel should show ETA when calculator has data."""
        tasks = [
            TaskItem(id="1.1.1", title="Task 1", status="complete", duration=30.0),
            TaskItem(id="1.1.2", title="Task 2", status="pending"),
        ]
        calc = ETACalculator()
        calc.add_completed(30.0)

        panel = TaskProgressPanel(
            phase_title="Test",
            tasks=tasks,
            eta_calculator=calc,
        )

        console = Console(file=StringIO(), force_terminal=True, width=80)
        console.print(panel)
        output = console.file.getvalue()
        assert "ETA:" in output


# =============================================================================
# PhaseSection Tests
# =============================================================================


class TestPhaseSection:
    """Tests for collapsible phase sections."""

    def test_phase_creation(self) -> None:
        """PhaseSection should be creatable."""
        phase = PhaseSection(title="Phase 1: Setup", completed=3, total=5)
        assert phase.title == "Phase 1: Setup"
        assert phase.completed == 3
        assert phase.total == 5
        assert phase.state == PhaseState.COLLAPSED

    def test_get_status_complete(self) -> None:
        """Complete phase should show COMPLETE status."""
        phase = PhaseSection(title="Test", completed=5, total=5)
        assert phase.get_status_text() == "COMPLETE"

    def test_get_status_in_progress(self) -> None:
        """Partially complete phase should show IN PROGRESS."""
        phase = PhaseSection(title="Test", completed=2, total=5)
        assert phase.get_status_text() == "IN PROGRESS"

    def test_get_status_pending(self) -> None:
        """Not started phase should show PENDING."""
        phase = PhaseSection(title="Test", completed=0, total=5)
        assert phase.get_status_text() == "PENDING"

    def test_get_status_override(self) -> None:
        """Custom status should override calculated status."""
        phase = PhaseSection(title="Test", completed=5, total=5, status="BLOCKED")
        assert phase.get_status_text() == "BLOCKED"

    def test_render_collapsed_unicode(self) -> None:
        """Collapsed phase should show right arrow."""
        phase = PhaseSection(
            title="Phase 1",
            completed=0,
            total=5,
            state=PhaseState.COLLAPSED,
        )
        text = phase.render(use_unicode=True)
        plain = text.plain
        assert "▶" in plain
        assert "Phase 1" in plain
        assert "0/5" in plain

    def test_render_expanded_unicode(self) -> None:
        """Expanded phase should show down arrow."""
        phase = PhaseSection(
            title="Phase 1",
            completed=5,
            total=5,
            state=PhaseState.EXPANDED,
        )
        text = phase.render(use_unicode=True)
        plain = text.plain
        assert "▼" in plain
        assert "COMPLETE" in plain

    def test_render_collapsed_ascii(self) -> None:
        """ASCII mode should use > and v."""
        phase = PhaseSection(
            title="Phase 1",
            completed=0,
            total=5,
            state=PhaseState.COLLAPSED,
        )
        text = phase.render(use_unicode=False)
        plain = text.plain
        assert ">" in plain

    def test_render_phase_list(self) -> None:
        """render_phase_list should return list of Text objects."""
        phases = [
            PhaseSection("Phase 1", completed=5, total=5),
            PhaseSection("Phase 2", completed=2, total=8),
            PhaseSection("Phase 3", completed=0, total=4),
        ]
        texts = render_phase_list(phases, use_unicode=True)
        assert len(texts) == 3
        assert "COMPLETE" in texts[0].plain
        assert "IN PROGRESS" in texts[1].plain
        assert "PENDING" in texts[2].plain


# =============================================================================
# TokenCounter Tests
# =============================================================================


class TestTokenCounter:
    """Tests for token counter visualization."""

    def test_percentage_calculation(self) -> None:
        """Should calculate percentage correctly."""
        counter = TokenCounter(used=5000, limit=10000)
        assert counter.percentage == 50.0

    def test_percentage_zero_limit(self) -> None:
        """Should handle zero limit gracefully."""
        counter = TokenCounter(used=100, limit=0)
        assert counter.percentage == 0.0

    def test_percentage_over_limit(self) -> None:
        """Should cap percentage at 100%."""
        counter = TokenCounter(used=15000, limit=10000)
        assert counter.percentage == 100.0

    def test_level_low(self) -> None:
        """0-50% should be low level."""
        counter = TokenCounter(used=4000, limit=10000)
        assert counter.level == "low"

    def test_level_medium(self) -> None:
        """50-80% should be medium level."""
        counter = TokenCounter(used=6000, limit=10000)
        assert counter.level == "medium"

    def test_level_high(self) -> None:
        """80-100% should be high level."""
        counter = TokenCounter(used=9000, limit=10000)
        assert counter.level == "high"

    def test_format_count_small(self) -> None:
        """Small counts should use commas."""
        counter = TokenCounter(used=1234, limit=10000)
        assert counter._format_count(1234) == "1,234"

    def test_format_count_thousands(self) -> None:
        """Large counts should use k suffix."""
        counter = TokenCounter(used=12431, limit=100000)
        assert counter._format_count(12431) == "12.4k"

    def test_format_count_millions(self) -> None:
        """Very large counts should use M suffix."""
        counter = TokenCounter(used=1200000, limit=2000000)
        assert counter._format_count(1200000) == "1.2M"

    def test_render_contains_counts(self) -> None:
        """Render should include token counts."""
        counter = TokenCounter(used=12431, limit=100000)
        text = counter.render()
        plain = text.plain
        assert "Tokens:" in plain
        assert "12.4k" in plain
        assert "100.0k" in plain
        assert "12%" in plain

    def test_render_progress_bar(self) -> None:
        """Render should include progress bar characters."""
        counter = TokenCounter(used=5000, limit=10000, width=20)
        text = counter.render()
        plain = text.plain
        assert "=" in plain
        assert "." in plain


# =============================================================================
# TaskSpinner Tests
# =============================================================================


class TestTaskSpinner:
    """Tests for animated task spinner."""

    def test_spinner_creation(self) -> None:
        """TaskSpinner should be creatable."""
        spinner = TaskSpinner(task_id="1.1.1", title="Test task")
        assert spinner.task_id == "1.1.1"
        assert spinner.title == "Test task"

    def test_spinner_render(self) -> None:
        """Spinner should render task info."""
        spinner = TaskSpinner(task_id="1.1.1", title="Test task")
        spinner._start_time = time.time() - 2.0
        text = spinner._render()
        plain = text.plain
        assert "1.1.1" in plain
        assert "Test task" in plain

    def test_spinner_stop_returns_elapsed(self) -> None:
        """stop() should return elapsed time."""
        spinner = TaskSpinner(task_id="1.1.1", title="Test")
        spinner._start_time = time.time() - 5.0
        elapsed = spinner.stop()
        assert 4.9 < elapsed < 5.5


# =============================================================================
# LiveProgressDisplay Tests
# =============================================================================


class TestLiveProgressDisplay:
    """Tests for live progress display."""

    def test_display_creation(self) -> None:
        """LiveProgressDisplay should be creatable."""
        tasks = [TaskItem(id="1.1.1", title="Test", status="pending")]
        display = LiveProgressDisplay(phase_title="Test", tasks=tasks)
        assert display.phase_title == "Test"
        assert len(display.tasks) == 1

    def test_mark_running(self) -> None:
        """mark_running should update task status."""
        tasks = [TaskItem(id="1.1.1", title="Test", status="pending")]
        display = LiveProgressDisplay(phase_title="Test", tasks=tasks)
        display.mark_running("1.1.1")
        assert display.tasks[0].status == TaskStatus.RUNNING
        assert display.tasks[0].start_time is not None

    def test_mark_complete(self) -> None:
        """mark_complete should update task status and duration."""
        tasks = [TaskItem(id="1.1.1", title="Test", status="running", start_time=time.time() - 2.0)]
        display = LiveProgressDisplay(phase_title="Test", tasks=tasks)
        display.mark_complete("1.1.1")
        assert display.tasks[0].status == TaskStatus.COMPLETE
        assert display.tasks[0].duration is not None

    def test_mark_complete_with_duration(self) -> None:
        """mark_complete should accept override duration."""
        tasks = [TaskItem(id="1.1.1", title="Test", status="running")]
        display = LiveProgressDisplay(phase_title="Test", tasks=tasks)
        display.mark_complete("1.1.1", duration=5.0)
        assert display.tasks[0].duration == 5.0

    def test_mark_failed(self) -> None:
        """mark_failed should update task status."""
        tasks = [TaskItem(id="1.1.1", title="Test", status="running")]
        display = LiveProgressDisplay(phase_title="Test", tasks=tasks)
        display.mark_failed("1.1.1")
        assert display.tasks[0].status == TaskStatus.FAILED

    def test_eta_calculator_updated_on_complete(self) -> None:
        """Completing a task should update ETA calculator."""
        tasks = [TaskItem(id="1.1.1", title="Test", status="running")]
        display = LiveProgressDisplay(phase_title="Test", tasks=tasks)
        display.mark_complete("1.1.1", duration=3.0)
        assert 3.0 in display.eta_calculator.completed_durations

    def test_stop_returns_elapsed(self) -> None:
        """stop() should return elapsed time."""
        tasks = [TaskItem(id="1.1.1", title="Test", status="pending")]
        display = LiveProgressDisplay(phase_title="Test", tasks=tasks)
        display._start_time = time.time() - 10.0
        elapsed = display.stop()
        assert 9.9 < elapsed < 10.5
