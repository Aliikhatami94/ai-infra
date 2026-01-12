"""Failure analysis for the Executor module.

Provides infrastructure to track, categorize, and analyze task failures:
- FailureCategory enum for classifying failure types
- FailureRecord for capturing failure details
- FailureAnalyzer for building and querying failure datasets
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ai_infra.executor.models import Task
from ai_infra.executor.verifier import VerificationResult
from ai_infra.logging import get_logger

logger = get_logger("executor.failure")


class FailureCategory(str, Enum):
    """Categories of task failures for analysis.

    These categories help identify patterns in why tasks fail,
    informing improvements to context building and agent prompting.
    """

    # Context-related failures
    CONTEXT_MISSING = "context_missing"
    """Agent didn't have enough information to complete the task."""

    CONTEXT_STALE = "context_stale"
    """Context was outdated or didn't reflect current codebase state."""

    CONTEXT_OVERWHELMING = "context_overwhelming"
    """Too much context caused the agent to lose focus."""

    # File-related failures
    WRONG_FILE = "wrong_file"
    """Agent edited the wrong file(s)."""

    FILE_NOT_FOUND = "file_not_found"
    """Agent tried to edit a file that doesn't exist."""

    FILE_CONFLICT = "file_conflict"
    """Concurrent edit conflict or file locked."""

    # Change-related failures
    PARTIAL_CHANGE = "partial_change"
    """Agent made some changes but missed others."""

    INCOMPLETE_IMPLEMENTATION = "incomplete_implementation"
    """Implementation started but not finished (TODOs, pass statements)."""

    REGRESSION = "regression"
    """Change broke existing functionality."""

    # Code quality failures
    SYNTAX_ERROR = "syntax_error"
    """Agent produced code that doesn't parse."""

    TYPE_ERROR = "type_error"
    """Type checking failed."""

    IMPORT_ERROR = "import_error"
    """Broken imports or missing dependencies."""

    TEST_FAILURE = "test_failure"
    """Tests failed after the change."""

    # Understanding failures
    WRONG_APPROACH = "wrong_approach"
    """Agent misunderstood the task requirements."""

    SCOPE_CREEP = "scope_creep"
    """Agent did more than asked, causing issues."""

    # System failures
    TOOL_FAILURE = "tool_failure"
    """A tool the agent used errored."""

    TIMEOUT = "timeout"
    """Task took too long to complete."""

    API_ERROR = "api_error"
    """LLM API error (rate limit, network, etc.)."""

    # Unknown
    UNKNOWN = "unknown"
    """Failure category could not be determined."""


class FailureSeverity(str, Enum):
    """Severity levels for failures."""

    CRITICAL = "critical"
    """Task completely failed, no usable output."""

    HIGH = "high"
    """Major issues that need manual intervention."""

    MEDIUM = "medium"
    """Partial success, some manual fixes needed."""

    LOW = "low"
    """Minor issues, mostly successful."""


@dataclass
class FailureRecord:
    """A record of a task failure for analysis.

    Attributes:
        task_id: The task that failed
        task_title: Human-readable task title
        category: Primary failure category
        secondary_categories: Additional contributing factors
        severity: How severe the failure was
        error_message: The error or failure description
        verification_result: Result from TaskVerifier if available
        agent_output: What the agent produced (truncated)
        context_summary: Summary of context provided to agent
        duration_seconds: How long the task ran
        timestamp: When the failure occurred
        metadata: Additional failure-specific data
    """

    task_id: str
    task_title: str
    category: FailureCategory
    severity: FailureSeverity = FailureSeverity.MEDIUM
    secondary_categories: list[FailureCategory] = field(default_factory=list)
    error_message: str = ""
    verification_result: dict[str, Any] | None = None
    agent_output: str = ""
    context_summary: str = ""
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "task_title": self.task_title,
            "category": self.category.value,
            "severity": self.severity.value,
            "secondary_categories": [c.value for c in self.secondary_categories],
            "error_message": self.error_message,
            "verification_result": self.verification_result,
            "agent_output": self.agent_output[:2000] if self.agent_output else "",
            "context_summary": self.context_summary[:500] if self.context_summary else "",
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailureRecord:
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            task_title=data["task_title"],
            category=FailureCategory(data["category"]),
            severity=FailureSeverity(data.get("severity", "medium")),
            secondary_categories=[FailureCategory(c) for c in data.get("secondary_categories", [])],
            error_message=data.get("error_message", ""),
            verification_result=data.get("verification_result"),
            agent_output=data.get("agent_output", ""),
            context_summary=data.get("context_summary", ""),
            duration_seconds=data.get("duration_seconds", 0.0),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.now(UTC),
            metadata=data.get("metadata", {}),
        )


@dataclass
class FailureStats:
    """Statistics about failures in the dataset.

    Attributes:
        total_failures: Total number of failures
        by_category: Count by failure category
        by_severity: Count by severity level
        avg_duration: Average task duration before failure
        success_rate: Success rate if total_tasks provided
        top_patterns: Most common failure patterns
    """

    total_failures: int
    by_category: dict[str, int]
    by_severity: dict[str, int]
    avg_duration: float
    success_rate: float | None = None
    top_patterns: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_failures": self.total_failures,
            "by_category": self.by_category,
            "by_severity": self.by_severity,
            "avg_duration": self.avg_duration,
            "success_rate": self.success_rate,
            "top_patterns": self.top_patterns,
        }

    def summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            f"Total Failures: {self.total_failures}",
            f"Avg Duration: {self.avg_duration:.1f}s",
        ]

        if self.success_rate is not None:
            lines.append(f"Success Rate: {self.success_rate:.1%}")

        lines.append("\nBy Category:")
        for cat, count in sorted(self.by_category.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {cat}: {count}")

        lines.append("\nBy Severity:")
        for sev, count in sorted(self.by_severity.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {sev}: {count}")

        if self.top_patterns:
            lines.append("\nTop Patterns:")
            for pattern, count in self.top_patterns[:5]:
                lines.append(f"  {pattern}: {count}")

        return "\n".join(lines)


class FailureAnalyzer:
    """Analyze and track task failures to identify patterns.

    The FailureAnalyzer maintains a dataset of failures and provides
    methods to query, categorize, and identify patterns.

    Example:
        analyzer = FailureAnalyzer(data_dir=Path("./failure_data"))

        # Record a failure
        record = analyzer.record_failure(
            task=task,
            category=FailureCategory.SYNTAX_ERROR,
            error_message="SyntaxError: unexpected EOF",
            verification_result=verifier_result,
        )

        # Get statistics
        stats = analyzer.get_stats()
        print(stats.summary())

        # Find patterns
        patterns = analyzer.find_patterns()
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        *,
        auto_save: bool = True,
        max_records: int = 10000,
    ):
        """Initialize the failure analyzer.

        Args:
            data_dir: Directory to store failure data (None for in-memory only)
            auto_save: Automatically save after each record
            max_records: Maximum records to keep in memory
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.auto_save = auto_save
        self.max_records = max_records
        self._records: list[FailureRecord] = []
        self._total_tasks: int = 0
        self._successful_tasks: int = 0

        if self.data_dir:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._load_existing()

    def _load_existing(self) -> None:
        """Load existing failure records from disk."""
        if not self.data_dir:
            return

        data_file = self.data_dir / "failures.json"
        if data_file.exists():
            try:
                with open(data_file) as f:
                    data = json.load(f)

                self._records = [FailureRecord.from_dict(r) for r in data.get("records", [])]
                self._total_tasks = data.get("total_tasks", 0)
                self._successful_tasks = data.get("successful_tasks", 0)

                logger.info(f"Loaded {len(self._records)} failure records")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load failure data: {e}")

    def save(self) -> None:
        """Save failure records to disk."""
        if not self.data_dir:
            return

        data_file = self.data_dir / "failures.json"
        data = {
            "records": [r.to_dict() for r in self._records],
            "total_tasks": self._total_tasks,
            "successful_tasks": self._successful_tasks,
            "last_updated": datetime.now(UTC).isoformat(),
        }

        with open(data_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved {len(self._records)} failure records")

    def record_failure(
        self,
        task: Task,
        category: FailureCategory,
        *,
        severity: FailureSeverity = FailureSeverity.MEDIUM,
        secondary_categories: list[FailureCategory] | None = None,
        error_message: str = "",
        verification_result: VerificationResult | None = None,
        agent_output: str = "",
        context_summary: str = "",
        duration_seconds: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> FailureRecord:
        """Record a task failure.

        Args:
            task: The task that failed
            category: Primary failure category
            severity: How severe the failure was
            secondary_categories: Additional contributing factors
            error_message: Description of the failure
            verification_result: Result from TaskVerifier
            agent_output: What the agent produced
            context_summary: Summary of context provided
            duration_seconds: How long the task ran
            metadata: Additional data

        Returns:
            The created FailureRecord
        """
        record = FailureRecord(
            task_id=task.id,
            task_title=task.title,
            category=category,
            severity=severity,
            secondary_categories=secondary_categories or [],
            error_message=error_message,
            verification_result=verification_result.to_dict() if verification_result else None,
            agent_output=agent_output,
            context_summary=context_summary,
            duration_seconds=duration_seconds,
            metadata=metadata or {},
        )

        self._records.append(record)
        self._total_tasks += 1

        # Trim old records if needed
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records :]

        if self.auto_save and self.data_dir:
            self.save()

        logger.info(f"Recorded failure for task {task.id}: {category.value} ({severity.value})")
        return record

    def record_success(self, task: Task) -> None:
        """Record a successful task (for success rate calculation).

        Args:
            task: The task that succeeded
        """
        self._total_tasks += 1
        self._successful_tasks += 1

        if self.auto_save and self.data_dir:
            self.save()

        logger.debug(f"Recorded success for task {task.id}")

    def get_records(
        self,
        *,
        category: FailureCategory | None = None,
        severity: FailureSeverity | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[FailureRecord]:
        """Get failure records with optional filtering.

        Args:
            category: Filter by category
            severity: Filter by severity
            since: Only records after this timestamp
            limit: Maximum records to return

        Returns:
            List of matching FailureRecords
        """
        records = self._records

        if category:
            records = [r for r in records if r.category == category]

        if severity:
            records = [r for r in records if r.severity == severity]

        if since:
            records = [r for r in records if r.timestamp >= since]

        if limit:
            records = records[-limit:]

        return records

    def get_stats(
        self,
        *,
        since: datetime | None = None,
    ) -> FailureStats:
        """Get statistics about failures.

        Args:
            since: Only include records after this timestamp

        Returns:
            FailureStats with aggregated data
        """
        records = self.get_records(since=since)

        if not records:
            return FailureStats(
                total_failures=0,
                by_category={},
                by_severity={},
                avg_duration=0.0,
                success_rate=self._calculate_success_rate(),
            )

        # Count by category
        category_counts = Counter(r.category.value for r in records)

        # Count by severity
        severity_counts = Counter(r.severity.value for r in records)

        # Average duration
        durations = [r.duration_seconds for r in records if r.duration_seconds > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        # Find patterns (category + secondary combinations)
        pattern_counts: Counter[str] = Counter()
        for r in records:
            if r.secondary_categories:
                pattern = f"{r.category.value}+{'+'.join(c.value for c in r.secondary_categories)}"
            else:
                pattern = r.category.value
            pattern_counts[pattern] += 1

        return FailureStats(
            total_failures=len(records),
            by_category=dict(category_counts),
            by_severity=dict(severity_counts),
            avg_duration=avg_duration,
            success_rate=self._calculate_success_rate(),
            top_patterns=pattern_counts.most_common(10),
        )

    def _calculate_success_rate(self) -> float | None:
        """Calculate the success rate."""
        if self._total_tasks == 0:
            return None
        return self._successful_tasks / self._total_tasks

    def find_patterns(
        self,
        min_occurrences: int = 2,
    ) -> list[dict[str, Any]]:
        """Find common failure patterns.

        Args:
            min_occurrences: Minimum occurrences to include

        Returns:
            List of patterns with counts and examples
        """
        patterns: dict[str, list[FailureRecord]] = {}

        for record in self._records:
            # Pattern key: category + severity
            key = f"{record.category.value}:{record.severity.value}"
            if key not in patterns:
                patterns[key] = []
            patterns[key].append(record)

        result = []
        for key, records in patterns.items():
            if len(records) >= min_occurrences:
                category, severity = key.split(":")
                result.append(
                    {
                        "pattern": key,
                        "category": category,
                        "severity": severity,
                        "count": len(records),
                        "example_tasks": [r.task_id for r in records[:3]],
                        "avg_duration": sum(r.duration_seconds for r in records) / len(records),
                    }
                )

        # Sort by count descending
        result.sort(key=lambda x: x["count"], reverse=True)
        return result

    def categorize_from_verification(
        self,
        result: VerificationResult,
    ) -> tuple[FailureCategory, list[FailureCategory]]:
        """Determine failure categories from a VerificationResult.

        Args:
            result: The verification result to analyze

        Returns:
            Tuple of (primary_category, secondary_categories)
        """
        from ai_infra.executor.verifier import CheckLevel, CheckStatus

        primary: FailureCategory = FailureCategory.UNKNOWN
        secondary: list[FailureCategory] = []

        for check in result.checks:
            if check.status != CheckStatus.FAILED:
                continue

            if check.level == CheckLevel.FILES:
                if primary == FailureCategory.UNKNOWN:
                    primary = FailureCategory.FILE_NOT_FOUND
                else:
                    secondary.append(FailureCategory.FILE_NOT_FOUND)

            elif check.level == CheckLevel.SYNTAX:
                if primary == FailureCategory.UNKNOWN:
                    primary = FailureCategory.SYNTAX_ERROR
                else:
                    secondary.append(FailureCategory.SYNTAX_ERROR)

            elif check.level == CheckLevel.IMPORTS:
                if primary == FailureCategory.UNKNOWN:
                    primary = FailureCategory.IMPORT_ERROR
                else:
                    secondary.append(FailureCategory.IMPORT_ERROR)

            elif check.level == CheckLevel.TESTS:
                if primary == FailureCategory.UNKNOWN:
                    primary = FailureCategory.TEST_FAILURE
                else:
                    secondary.append(FailureCategory.TEST_FAILURE)

            elif check.level == CheckLevel.TYPES:
                if primary == FailureCategory.UNKNOWN:
                    primary = FailureCategory.TYPE_ERROR
                else:
                    secondary.append(FailureCategory.TYPE_ERROR)

        return primary, secondary

    def suggest_improvements(
        self,
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Suggest improvements based on failure patterns.

        Args:
            top_n: Number of top suggestions to return

        Returns:
            List of improvement suggestions
        """
        stats = self.get_stats()
        suggestions: list[dict[str, Any]] = []

        # Analyze top failure categories
        for category, count in sorted(stats.by_category.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]:
            suggestion = self._get_suggestion_for_category(FailureCategory(category), count)
            if suggestion:
                suggestions.append(suggestion)

        return suggestions

    def _get_suggestion_for_category(
        self,
        category: FailureCategory,
        count: int,
    ) -> dict[str, Any] | None:
        """Get improvement suggestion for a failure category."""
        suggestions_map: dict[FailureCategory, dict[str, str]] = {
            FailureCategory.CONTEXT_MISSING: {
                "issue": "Agent lacks sufficient context",
                "suggestion": "Increase semantic search results, add more file hints",
                "action": "Expand ProjectContext.get_task_context() to include more relevant files",
            },
            FailureCategory.SYNTAX_ERROR: {
                "issue": "Agent produces invalid Python syntax",
                "suggestion": "Add syntax validation before applying changes",
                "action": "Implement pre-commit style syntax check in agent loop",
            },
            FailureCategory.WRONG_FILE: {
                "issue": "Agent edits wrong files",
                "suggestion": "Improve file hints in ROADMAP, add file path validation",
                "action": "Enhance ROADMAP parser to extract more precise file hints",
            },
            FailureCategory.PARTIAL_CHANGE: {
                "issue": "Agent makes incomplete changes",
                "suggestion": "Break down tasks into smaller units",
                "action": "Add sub-task detection in ROADMAP parser",
            },
            FailureCategory.TEST_FAILURE: {
                "issue": "Changes break existing tests",
                "suggestion": "Include test file context, run tests before completion",
                "action": "Add test file content to task context automatically",
            },
            FailureCategory.TIMEOUT: {
                "issue": "Tasks take too long",
                "suggestion": "Set stricter timeouts, break down complex tasks",
                "action": "Add task complexity estimation and timeout scaling",
            },
            FailureCategory.IMPORT_ERROR: {
                "issue": "Agent creates broken imports",
                "suggestion": "Include import structure in context",
                "action": "Add dependency graph to ProjectContext",
            },
            FailureCategory.WRONG_APPROACH: {
                "issue": "Agent misunderstands requirements",
                "suggestion": "Improve task descriptions, add examples",
                "action": "Enhance ROADMAP format with explicit examples",
            },
        }

        if category not in suggestions_map:
            return None

        suggestion = suggestions_map[category]
        return {
            "category": category.value,
            "occurrences": count,
            **suggestion,
        }

    def export_dataset(
        self,
        output_path: Path,
        *,
        format: str = "json",
    ) -> None:
        """Export failure dataset for analysis.

        Args:
            output_path: Path to write the export
            format: Export format ('json' or 'csv')
        """
        if format == "json":
            data = {
                "records": [r.to_dict() for r in self._records],
                "stats": self.get_stats().to_dict(),
                "patterns": self.find_patterns(),
                "exported_at": datetime.now(UTC).isoformat(),
            }
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

        elif format == "csv":
            import csv

            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "task_id",
                        "task_title",
                        "category",
                        "severity",
                        "error_message",
                        "duration_seconds",
                        "timestamp",
                    ],
                )
                writer.writeheader()
                for record in self._records:
                    writer.writerow(
                        {
                            "task_id": record.task_id,
                            "task_title": record.task_title,
                            "category": record.category.value,
                            "severity": record.severity.value,
                            "error_message": record.error_message[:200],
                            "duration_seconds": record.duration_seconds,
                            "timestamp": record.timestamp.isoformat(),
                        }
                    )

        logger.info(f"Exported {len(self._records)} records to {output_path}")

    def clear(self) -> None:
        """Clear all failure records."""
        self._records = []
        self._total_tasks = 0
        self._successful_tasks = 0

        if self.auto_save and self.data_dir:
            self.save()

        logger.info("Cleared all failure records")

    @property
    def record_count(self) -> int:
        """Get the number of failure records."""
        return len(self._records)

    @property
    def total_tasks(self) -> int:
        """Get total tasks (success + failure)."""
        return self._total_tasks

    @property
    def success_rate(self) -> float | None:
        """Get the success rate."""
        return self._calculate_success_rate()
