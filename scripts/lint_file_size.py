#!/usr/bin/env python3
"""File size linter for enforcing code quality standards.

Phase 13.1 of EXECUTOR_4.md - File Size Enforcement.

This script checks Python files to ensure they don't exceed a configurable
line limit. Large files are harder to maintain, test, and understand.

Target: All files under 500 lines except TypedDict state files.

Usage:
    python scripts/lint_file_size.py [directory] [--max-lines N] [--json]

Examples:
    # Check executor directory
    python scripts/lint_file_size.py src/ai_infra/executor

    # Check with custom limit
    python scripts/lint_file_size.py src/ --max-lines 400

    # Output as JSON
    python scripts/lint_file_size.py src/ --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MAX_LINES = 500

# Files that are allowed to exceed the limit
DEFAULT_EXCEPTIONS = {
    "state.py",  # TypedDict definitions can be verbose
    "types.py",  # Type definitions can be verbose
    "__init__.py",  # Re-export files can be long
}

# Directories to skip
SKIP_DIRECTORIES = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    ".tox",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "htmlcov",
    "dist",
    "build",
    ".eggs",
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FileInfo:
    """Information about a Python file."""

    path: Path
    line_count: int
    is_exception: bool = False

    @property
    def relative_path(self) -> str:
        """Get path relative to current directory."""
        try:
            return str(self.path.relative_to(Path.cwd()))
        except ValueError:
            return str(self.path)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.relative_path,
            "line_count": self.line_count,
            "is_exception": self.is_exception,
        }


@dataclass
class LintResult:
    """Result of file size linting."""

    violations: list[FileInfo] = field(default_factory=list)
    warnings: list[FileInfo] = field(default_factory=list)
    passed: list[FileInfo] = field(default_factory=list)
    max_lines: int = DEFAULT_MAX_LINES

    @property
    def has_violations(self) -> bool:
        """Check if there are any violations."""
        return len(self.violations) > 0

    @property
    def total_files(self) -> int:
        """Get total number of files checked."""
        return len(self.violations) + len(self.warnings) + len(self.passed)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_lines": self.max_lines,
            "total_files": self.total_files,
            "violations_count": len(self.violations),
            "warnings_count": len(self.warnings),
            "passed_count": len(self.passed),
            "violations": [v.to_dict() for v in self.violations],
            "warnings": [w.to_dict() for w in self.warnings],
        }


# =============================================================================
# Core Functions
# =============================================================================


def count_lines(file_path: Path) -> int:
    """Count the number of lines in a file.

    Args:
        file_path: Path to the file.

    Returns:
        Number of lines in the file.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        return len(content.splitlines())
    except (OSError, UnicodeDecodeError):
        return 0


def should_skip_directory(dir_path: Path) -> bool:
    """Check if a directory should be skipped.

    Args:
        dir_path: Path to check.

    Returns:
        True if directory should be skipped.
    """
    return dir_path.name in SKIP_DIRECTORIES


def check_file_sizes(
    directory: Path,
    max_lines: int = DEFAULT_MAX_LINES,
    exceptions: set[str] | None = None,
    warning_threshold: float = 0.8,
) -> LintResult:
    """Check all Python files in a directory for size violations.

    Args:
        directory: Directory to check.
        max_lines: Maximum allowed lines per file.
        exceptions: Set of filenames that are allowed to exceed the limit.
        warning_threshold: Fraction of max_lines to trigger warning (0.8 = 80%).

    Returns:
        LintResult with violations, warnings, and passed files.
    """
    if exceptions is None:
        exceptions = DEFAULT_EXCEPTIONS

    result = LintResult(max_lines=max_lines)
    warning_lines = int(max_lines * warning_threshold)

    for py_file in directory.rglob("*.py"):
        # Skip directories we don't want to check
        if any(should_skip_directory(p) for p in py_file.parents):
            continue

        line_count = count_lines(py_file)
        is_exception = py_file.name in exceptions

        file_info = FileInfo(
            path=py_file,
            line_count=line_count,
            is_exception=is_exception,
        )

        if line_count > max_lines:
            if is_exception:
                result.warnings.append(file_info)
            else:
                result.violations.append(file_info)
        elif line_count > warning_lines:
            result.warnings.append(file_info)
        else:
            result.passed.append(file_info)

    # Sort by line count descending
    result.violations.sort(key=lambda f: f.line_count, reverse=True)
    result.warnings.sort(key=lambda f: f.line_count, reverse=True)

    return result


def format_result(result: LintResult, verbose: bool = False) -> str:
    """Format lint result as human-readable string.

    Args:
        result: LintResult to format.
        verbose: Whether to include all files.

    Returns:
        Formatted string.
    """
    lines = []

    if result.has_violations:
        lines.append(f"Files exceeding {result.max_lines} line limit:")
        lines.append("=" * 50)
        for v in result.violations:
            lines.append(f"  {v.relative_path}: {v.line_count} lines")
        lines.append("")

    if result.warnings:
        lines.append("Warnings (approaching limit or exceptions):")
        lines.append("-" * 50)
        for w in result.warnings:
            status = " (exception)" if w.is_exception else ""
            lines.append(f"  {w.relative_path}: {w.line_count} lines{status}")
        lines.append("")

    if verbose and result.passed:
        lines.append("Passed files:")
        lines.append("-" * 50)
        for p in result.passed:
            lines.append(f"  {p.relative_path}: {p.line_count} lines")
        lines.append("")

    # Summary
    lines.append("Summary:")
    lines.append(f"  Total files: {result.total_files}")
    lines.append(f"  Violations: {len(result.violations)}")
    lines.append(f"  Warnings: {len(result.warnings)}")
    lines.append(f"  Passed: {len(result.passed)}")

    if result.has_violations:
        lines.append("\nStatus: FAILED")
    else:
        lines.append("\nStatus: PASSED")

    return "\n".join(lines)


# =============================================================================
# Refactoring Recommendations
# =============================================================================


def get_refactoring_recommendations(result: LintResult) -> list[dict[str, Any]]:
    """Generate refactoring recommendations for large files.

    Args:
        result: LintResult with violations.

    Returns:
        List of recommendation dictionaries.
    """
    recommendations = []

    for v in result.violations:
        rec: dict[str, Any] = {
            "file": v.relative_path,
            "current_lines": v.line_count,
            "target_lines": result.max_lines,
            "reduction_needed": v.line_count - result.max_lines,
        }

        # Add specific recommendations based on file name
        if "loop" in v.path.name.lower():
            rec["suggestion"] = "Split into loop/, nodes/, edges/ submodules"
            rec["priority"] = "high"
        elif "dependencies" in v.path.name.lower():
            rec["suggestion"] = "Review for redundant code, extract utilities"
            rec["priority"] = "medium"
        elif "todolist" in v.path.name.lower():
            rec["suggestion"] = "Split list management vs item logic"
            rec["priority"] = "medium"
        elif "graph" in v.path.name.lower():
            rec["suggestion"] = "Extract node implementations to separate files"
            rec["priority"] = "high"
        else:
            rec["suggestion"] = "Review for extraction opportunities"
            rec["priority"] = "low"

        recommendations.append(rec)

    return recommendations


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    """Main entry point for CLI.

    Returns:
        Exit code (0 = success, 1 = violations found).
    """
    parser = argparse.ArgumentParser(
        description="Check Python files for size violations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path("src"),
        help="Directory to check (default: src)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=DEFAULT_MAX_LINES,
        help=f"Maximum lines per file (default: {DEFAULT_MAX_LINES})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Include passed files in output",
    )
    parser.add_argument(
        "--recommendations",
        action="store_true",
        help="Include refactoring recommendations",
    )
    parser.add_argument(
        "--exceptions",
        nargs="*",
        help="Additional exception filenames",
    )

    args = parser.parse_args()

    if not args.directory.exists():
        print(f"Error: Directory not found: {args.directory}", file=sys.stderr)
        return 1

    # Build exceptions set
    exceptions = DEFAULT_EXCEPTIONS.copy()
    if args.exceptions:
        exceptions.update(args.exceptions)

    # Run check
    result = check_file_sizes(
        directory=args.directory,
        max_lines=args.max_lines,
        exceptions=exceptions,
    )

    # Output results
    if args.json:
        output = result.to_dict()
        if args.recommendations:
            output["recommendations"] = get_refactoring_recommendations(result)
        print(json.dumps(output, indent=2))
    else:
        print(format_result(result, verbose=args.verbose))
        if args.recommendations and result.violations:
            print("\nRefactoring Recommendations:")
            print("=" * 50)
            for rec in get_refactoring_recommendations(result):
                print(f"\n{rec['file']}:")
                print(f"  Current: {rec['current_lines']} lines")
                print(f"  Reduction needed: {rec['reduction_needed']} lines")
                print(f"  Suggestion: {rec['suggestion']}")
                print(f"  Priority: {rec['priority']}")

    return 1 if result.has_violations else 0


if __name__ == "__main__":
    sys.exit(main())
