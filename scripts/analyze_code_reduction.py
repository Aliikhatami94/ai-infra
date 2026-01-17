#!/usr/bin/env python3
"""Analyze executor code for reduction opportunities.

Phase 13.4 of EXECUTOR_4.md - Code Reduction Target.
Identifies files that could be consolidated or reduced.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileAnalysis:
    """Analysis of a single file."""

    path: Path
    line_count: int
    category: str = "keep"
    notes: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "line_count": self.line_count,
            "category": self.category,
            "notes": self.notes,
        }


@dataclass
class ReductionAnalysis:
    """Complete code reduction analysis."""

    total_lines: int = 0
    target_lines: int = 20000
    files: list[FileAnalysis] = field(default_factory=list)
    candidates: list[FileAnalysis] = field(default_factory=list)
    potential_reduction: int = 0

    @property
    def reduction_needed(self) -> int:
        """Lines that need to be reduced to meet target."""
        return max(0, self.total_lines - self.target_lines)

    @property
    def reduction_percentage(self) -> float:
        """Percentage reduction needed."""
        if self.total_lines == 0:
            return 0.0
        return (self.reduction_needed / self.total_lines) * 100

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_lines": self.total_lines,
            "target_lines": self.target_lines,
            "reduction_needed": self.reduction_needed,
            "reduction_percentage": round(self.reduction_percentage, 1),
            "potential_reduction": self.potential_reduction,
            "file_count": len(self.files),
            "candidate_count": len(self.candidates),
            "candidates": [c.to_dict() for c in self.candidates],
        }


# Categories for reduction analysis
REDUCTION_CATEGORIES = {
    # Files that duplicate ai_infra core functionality
    "duplicate": {
        "tracing.py": "Consider using ai_infra.tracing instead",
        "streaming.py": "Consider using ai_infra.llm.streaming instead",
    },
    # Files that may have redundant code
    "consolidate": {
        "recovery.py": "May overlap with nodes/recovery.py (1018 + 1901 = 2919 lines)",
        "context.py": "May overlap with context_carryover.py (1005 + 1226 = 2231 lines)",
    },
    # Files that are unusually large and need splitting
    "split": {
        "loop.py": "3759 lines - should be split into smaller modules",
        "todolist.py": "1935 lines - split list vs item logic",
        "graph.py": "1472 lines - split into graph components",
    },
    # Files to review for dead code
    "review": {
        "dependencies.py": "1442 lines - review for removal or consolidation",
        "phase1_metrics.py": "865 lines - may be legacy code",
    },
}


def count_lines(file_path: Path) -> int:
    """Count lines in a file."""
    try:
        return len(file_path.read_text().splitlines())
    except Exception:
        return 0


def analyze_directory(directory: Path) -> ReductionAnalysis:
    """Analyze a directory for code reduction opportunities."""
    analysis = ReductionAnalysis()

    # Get all Python files
    py_files = sorted(directory.rglob("*.py"))

    for py_file in py_files:
        if "__pycache__" in str(py_file):
            continue

        line_count = count_lines(py_file)
        analysis.total_lines += line_count

        file_name = py_file.name
        relative_path = py_file.relative_to(directory)

        # Check each category
        category = "keep"
        notes = ""

        for cat_name, cat_files in REDUCTION_CATEGORIES.items():
            if file_name in cat_files:
                category = cat_name
                notes = cat_files[file_name]
                break

        file_analysis = FileAnalysis(
            path=relative_path,
            line_count=line_count,
            category=category,
            notes=notes,
        )

        analysis.files.append(file_analysis)

        if category != "keep":
            analysis.candidates.append(file_analysis)
            # Estimate potential reduction (50% for consolidate, 30% for split)
            if category == "duplicate":
                analysis.potential_reduction += int(line_count * 0.8)
            elif category == "consolidate":
                analysis.potential_reduction += int(line_count * 0.3)
            elif category == "split":
                analysis.potential_reduction += int(line_count * 0.1)
            elif category == "review":
                analysis.potential_reduction += int(line_count * 0.5)

    # Sort candidates by line count
    analysis.candidates.sort(key=lambda x: x.line_count, reverse=True)

    return analysis


def format_analysis(analysis: ReductionAnalysis, verbose: bool = False) -> str:
    """Format analysis as human-readable text."""
    lines = [
        "=" * 60,
        "CODE REDUCTION ANALYSIS",
        "=" * 60,
        "",
        f"Current total:       {analysis.total_lines:,} lines",
        f"Target:              {analysis.target_lines:,} lines",
        f"Reduction needed:    {analysis.reduction_needed:,} lines ({analysis.reduction_percentage:.1f}%)",
        f"Potential reduction: {analysis.potential_reduction:,} lines",
        "",
    ]

    if analysis.candidates:
        lines.extend(
            [
                "-" * 60,
                "REDUCTION CANDIDATES",
                "-" * 60,
                "",
            ]
        )

        for candidate in analysis.candidates:
            lines.append(f"  [{candidate.category.upper()}] {candidate.path}")
            lines.append(f"    Lines: {candidate.line_count:,}")
            if candidate.notes:
                lines.append(f"    Notes: {candidate.notes}")
            lines.append("")

    # Summary by category
    lines.extend(
        [
            "-" * 60,
            "SUMMARY BY CATEGORY",
            "-" * 60,
            "",
        ]
    )

    category_totals: dict[str, int] = {}
    for candidate in analysis.candidates:
        category_totals[candidate.category] = (
            category_totals.get(candidate.category, 0) + candidate.line_count
        )

    for category, total in sorted(category_totals.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {category}: {total:,} lines")

    lines.extend(
        [
            "",
            "-" * 60,
            "RECOMMENDATIONS",
            "-" * 60,
            "",
            "1. PRIORITY: Split loop.py (3,759 lines) into smaller modules",
            "2. Consolidate recovery.py with nodes/recovery.py",
            "3. Consolidate context.py with context_carryover.py",
            "4. Review dependencies.py for dead code removal",
            "5. Evaluate tracing.py vs ai_infra.tracing for consolidation",
            "",
        ]
    )

    if analysis.reduction_needed > analysis.potential_reduction:
        lines.extend(
            [
                "WARNING: Identified candidates may not achieve target.",
                f"         Need {analysis.reduction_needed:,} lines, identified {analysis.potential_reduction:,}",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> int:
    """Run code reduction analysis."""
    parser = argparse.ArgumentParser(description="Analyze code for reduction opportunities")
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path("src/ai_infra/executor"),
        help="Directory to analyze (default: src/ai_infra/executor)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=20000,
        help="Target line count (default: 20000)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if not args.directory.exists():
        print(f"Error: Directory not found: {args.directory}", file=sys.stderr)
        return 1

    analysis = analyze_directory(args.directory)
    analysis.target_lines = args.target

    if args.json:
        print(json.dumps(analysis.to_dict(), indent=2))
    else:
        print(format_analysis(analysis, verbose=args.verbose))

    return 0


if __name__ == "__main__":
    sys.exit(main())
