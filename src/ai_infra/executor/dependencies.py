"""Multi-file dependency tracking for the Executor module.

Provides import graph building and dependency analysis to:
- Track which files import a given file
- Warn when changes might break dependent files
- Identify files that need updates when interfaces change
- Support automatic propagation of changes to dependents
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ai_infra.logging import get_logger

logger = get_logger("executor.dependencies")


# =============================================================================
# Data Models
# =============================================================================


class DependencyType(Enum):
    """Type of dependency between files."""

    IMPORT = "import"  # import module
    FROM_IMPORT = "from_import"  # from module import X
    DYNAMIC = "dynamic"  # __import__, importlib
    TYPE_ONLY = "type_only"  # TYPE_CHECKING block
    RELATIVE = "relative"  # from . import X


class ImpactLevel(Enum):
    """Level of impact when a file changes."""

    NONE = "none"  # No impact on other files
    LOW = "low"  # Internal changes, unlikely to break dependents
    MEDIUM = "medium"  # Interface changes that may affect some dependents
    HIGH = "high"  # Breaking changes that likely affect all dependents
    CRITICAL = "critical"  # Core file, changes affect entire codebase


@dataclass
class ImportInfo:
    """Information about a single import.

    Attributes:
        module: The imported module path (e.g., "ai_infra.executor.loop")
        names: Specific names imported (empty for `import module`)
        alias: Import alias if any
        dependency_type: Type of import statement
        line_number: Line number in source file
        is_relative: Whether this is a relative import
        level: Relative import level (0 for absolute)
    """

    module: str
    names: list[str] = field(default_factory=list)
    alias: str | None = None
    dependency_type: DependencyType = DependencyType.IMPORT
    line_number: int = 0
    is_relative: bool = False
    level: int = 0

    @property
    def is_from_import(self) -> bool:
        """Whether this is a 'from X import Y' statement."""
        return self.dependency_type in (
            DependencyType.FROM_IMPORT,
            DependencyType.RELATIVE,
            DependencyType.TYPE_ONLY,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "module": self.module,
            "names": self.names,
            "alias": self.alias,
            "dependency_type": self.dependency_type.value,
            "line_number": self.line_number,
            "is_relative": self.is_relative,
            "level": self.level,
        }


@dataclass
class FileDependency:
    """Dependency relationship between two files.

    Attributes:
        source_file: File that has the import
        target_file: File being imported
        imports: List of imports from target to source
        is_direct: Whether this is a direct import (not transitive)
    """

    source_file: Path
    target_file: Path
    imports: list[ImportInfo] = field(default_factory=list)
    is_direct: bool = True

    @property
    def import_count(self) -> int:
        """Number of imports from target file."""
        return len(self.imports)

    @property
    def imported_names(self) -> set[str]:
        """All names imported from target file."""
        names: set[str] = set()
        for imp in self.imports:
            names.update(imp.names)
        return names

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_file": str(self.source_file),
            "target_file": str(self.target_file),
            "imports": [i.to_dict() for i in self.imports],
            "is_direct": self.is_direct,
        }


@dataclass
class DependencyWarning:
    """Warning about potential breaking change.

    Attributes:
        changed_file: File that was changed
        dependent_file: File that might be affected
        reason: Why this might break
        imported_names: Names imported that might be affected
        impact_level: Estimated impact level
        suggested_action: What to do about it
    """

    changed_file: Path
    dependent_file: Path
    reason: str
    imported_names: list[str] = field(default_factory=list)
    impact_level: ImpactLevel = ImpactLevel.MEDIUM
    suggested_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "changed_file": str(self.changed_file),
            "dependent_file": str(self.dependent_file),
            "reason": self.reason,
            "imported_names": self.imported_names,
            "impact_level": self.impact_level.value,
            "suggested_action": self.suggested_action,
        }


@dataclass
class ChangeAnalysis:
    """Analysis of changes and their impact.

    Attributes:
        changed_files: Files that were modified
        affected_files: Files that might be affected by changes
        warnings: Warnings about potential issues
        impact_level: Overall impact level
        safe_to_proceed: Whether it's safe to proceed without review
    """

    changed_files: list[Path] = field(default_factory=list)
    affected_files: list[Path] = field(default_factory=list)
    warnings: list[DependencyWarning] = field(default_factory=list)
    impact_level: ImpactLevel = ImpactLevel.NONE
    safe_to_proceed: bool = True

    @property
    def has_warnings(self) -> bool:
        """Whether there are any warnings."""
        return len(self.warnings) > 0

    @property
    def warning_count(self) -> int:
        """Number of warnings."""
        return len(self.warnings)

    @property
    def high_impact_count(self) -> int:
        """Number of high/critical impact warnings."""
        return sum(
            1 for w in self.warnings if w.impact_level in (ImpactLevel.HIGH, ImpactLevel.CRITICAL)
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "changed_files": [str(f) for f in self.changed_files],
            "affected_files": [str(f) for f in self.affected_files],
            "warnings": [w.to_dict() for w in self.warnings],
            "impact_level": self.impact_level.value,
            "safe_to_proceed": self.safe_to_proceed,
        }


# =============================================================================
# Import Parser
# =============================================================================


class ImportParser:
    """Parse Python files to extract import information."""

    def __init__(self) -> None:
        """Initialize the parser."""
        self._dynamic_import_pattern = re.compile(
            r"(?:__import__|importlib\.import_module)\s*\(\s*['\"]([^'\"]+)['\"]"
        )

    def parse_file(self, file_path: Path) -> list[ImportInfo]:
        """Parse a Python file and extract all imports.

        Args:
            file_path: Path to the Python file.

        Returns:
            List of ImportInfo objects for each import.

        Raises:
            FileNotFoundError: If file doesn't exist.
            SyntaxError: If file has syntax errors.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = file_path.read_text(encoding="utf-8")
        return self.parse_source(content)

    def parse_source(self, source: str) -> list[ImportInfo]:
        """Parse Python source code and extract imports.

        Args:
            source: Python source code.

        Returns:
            List of ImportInfo objects.
        """
        imports: list[ImportInfo] = []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Fall back to regex for files with syntax errors
            return self._parse_with_regex(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check for TYPE_CHECKING guard
                if self._is_type_checking_guard(node):
                    type_imports = self._extract_imports_from_block(node.body)
                    for imp in type_imports:
                        imp.dependency_type = DependencyType.TYPE_ONLY
                    imports.extend(type_imports)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        ImportInfo(
                            module=alias.name,
                            alias=alias.asname,
                            dependency_type=DependencyType.IMPORT,
                            line_number=node.lineno,
                        )
                    )

            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    module = ""
                else:
                    module = node.module

                names = [alias.name for alias in node.names]
                dep_type = DependencyType.RELATIVE if node.level > 0 else DependencyType.FROM_IMPORT

                imports.append(
                    ImportInfo(
                        module=module,
                        names=names,
                        dependency_type=dep_type,
                        line_number=node.lineno,
                        is_relative=node.level > 0,
                        level=node.level,
                    )
                )

        # Also check for dynamic imports
        dynamic_imports = self._find_dynamic_imports(source)
        imports.extend(dynamic_imports)

        return imports

    def _is_type_checking_guard(self, node: ast.If) -> bool:
        """Check if an If node is a TYPE_CHECKING guard."""
        if isinstance(node.test, ast.Name):
            return node.test.id == "TYPE_CHECKING"
        if isinstance(node.test, ast.Attribute):
            return node.test.attr == "TYPE_CHECKING"
        return False

    def _extract_imports_from_block(self, body: list[ast.stmt]) -> list[ImportInfo]:
        """Extract imports from a block of statements."""
        imports: list[ImportInfo] = []
        for stmt in body:
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    imports.append(
                        ImportInfo(
                            module=alias.name,
                            alias=alias.asname,
                            dependency_type=DependencyType.IMPORT,
                            line_number=stmt.lineno,
                        )
                    )
            elif isinstance(stmt, ast.ImportFrom):
                module = stmt.module or ""
                names = [alias.name for alias in stmt.names]
                imports.append(
                    ImportInfo(
                        module=module,
                        names=names,
                        dependency_type=DependencyType.FROM_IMPORT,
                        line_number=stmt.lineno,
                        is_relative=stmt.level > 0,
                        level=stmt.level,
                    )
                )
        return imports

    def _find_dynamic_imports(self, source: str) -> list[ImportInfo]:
        """Find dynamic imports using regex."""
        imports: list[ImportInfo] = []
        for match in self._dynamic_import_pattern.finditer(source):
            imports.append(
                ImportInfo(
                    module=match.group(1),
                    dependency_type=DependencyType.DYNAMIC,
                    line_number=source[: match.start()].count("\n") + 1,
                )
            )
        return imports

    def _parse_with_regex(self, source: str) -> list[ImportInfo]:
        """Fallback regex parsing for files with syntax errors."""
        imports: list[ImportInfo] = []

        # Match 'import X' and 'import X as Y'
        import_pattern = re.compile(r"^import\s+([\w.]+)(?:\s+as\s+(\w+))?", re.MULTILINE)
        for match in import_pattern.finditer(source):
            imports.append(
                ImportInfo(
                    module=match.group(1),
                    alias=match.group(2),
                    dependency_type=DependencyType.IMPORT,
                    line_number=source[: match.start()].count("\n") + 1,
                )
            )

        # Match 'from X import Y'
        from_pattern = re.compile(
            r"^from\s+(\.*)?([\w.]*)\s+import\s+(.+?)(?:\s*#|$)", re.MULTILINE
        )
        for match in from_pattern.finditer(source):
            level = len(match.group(1) or "")
            module = match.group(2) or ""
            names_str = match.group(3)
            names = [n.strip().split(" as ")[0] for n in names_str.split(",")]

            imports.append(
                ImportInfo(
                    module=module,
                    names=names,
                    dependency_type=DependencyType.RELATIVE
                    if level > 0
                    else DependencyType.FROM_IMPORT,
                    line_number=source[: match.start()].count("\n") + 1,
                    is_relative=level > 0,
                    level=level,
                )
            )

        return imports


# =============================================================================
# Dependency Tracker
# =============================================================================


class DependencyTracker:
    """Track file dependencies across a Python project.

    This class builds an import graph for a project and provides
    utilities to analyze dependencies and potential breaking changes.

    Example:
        >>> tracker = DependencyTracker(Path("./src"))
        >>> await tracker.build_graph()
        >>> dependents = tracker.get_dependents(Path("./src/core.py"))
        >>> print(f"{len(dependents)} files depend on core.py")
    """

    def __init__(
        self,
        root: Path,
        *,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        """Initialize the dependency tracker.

        Args:
            root: Root directory to scan.
            include_patterns: Glob patterns to include (default: ["**/*.py"]).
            exclude_patterns: Glob patterns to exclude.
        """
        self.root = Path(root).resolve()
        self.include_patterns = include_patterns or ["**/*.py"]
        self.exclude_patterns = exclude_patterns or [
            "**/node_modules/**",
            "**/.venv/**",
            "**/venv/**",
            "**/__pycache__/**",
            "**/dist/**",
            "**/build/**",
            "**/.git/**",
            "**/htmlcov/**",
            "**/.pytest_cache/**",
            "**/.mypy_cache/**",
            "**/.ruff_cache/**",
        ]

        self._parser = ImportParser()
        self._graph: dict[Path, list[FileDependency]] = {}  # file -> dependencies
        self._reverse_graph: dict[Path, list[FileDependency]] = {}  # file -> dependents
        self._file_imports: dict[Path, list[ImportInfo]] = {}
        self._built = False

    @property
    def is_built(self) -> bool:
        """Whether the dependency graph has been built."""
        return self._built

    @property
    def file_count(self) -> int:
        """Number of files in the graph."""
        return len(self._file_imports)

    @property
    def dependency_count(self) -> int:
        """Total number of dependencies."""
        return sum(len(deps) for deps in self._graph.values())

    async def build_graph(self, *, force: bool = False) -> None:
        """Build the dependency graph for the project.

        Args:
            force: Rebuild even if already built.
        """
        if self._built and not force:
            return

        logger.info(f"Building dependency graph for {self.root}")
        self._graph.clear()
        self._reverse_graph.clear()
        self._file_imports.clear()

        # Find all Python files
        files = self._find_files()
        logger.debug(f"Found {len(files)} Python files")

        # Parse imports from each file
        for file_path in files:
            try:
                imports = self._parser.parse_file(file_path)
                self._file_imports[file_path] = imports
            except (SyntaxError, FileNotFoundError) as e:
                logger.warning(f"Failed to parse {file_path}: {e}")
                self._file_imports[file_path] = []

        # Build the dependency graph
        for file_path, imports in self._file_imports.items():
            self._graph[file_path] = []
            for imp in imports:
                target = self._resolve_import(file_path, imp)
                if target and target in self._file_imports:
                    dep = FileDependency(
                        source_file=file_path,
                        target_file=target,
                        imports=[imp],
                    )
                    self._graph[file_path].append(dep)

                    # Update reverse graph
                    if target not in self._reverse_graph:
                        self._reverse_graph[target] = []
                    self._reverse_graph[target].append(dep)

        self._built = True
        logger.info(
            f"Dependency graph built: {self.file_count} files, {self.dependency_count} dependencies"
        )

    def _find_files(self) -> list[Path]:
        """Find all files matching include patterns, excluding excluded ones."""
        files: set[Path] = set()

        for pattern in self.include_patterns:
            for file_path in self.root.glob(pattern):
                if file_path.is_file():
                    files.add(file_path.resolve())

        # Filter out excluded files by checking if their relative path matches any exclude pattern
        def is_excluded(file_path: Path) -> bool:
            try:
                rel_path = file_path.relative_to(self.root)
                rel_str = str(rel_path)
                # Check each exclude pattern
                for pattern in self.exclude_patterns:
                    # Convert glob pattern to check
                    # Remove leading **/ for simpler matching
                    clean_pattern = pattern.lstrip("*").lstrip("/")
                    # Check if any excluded directory is in the path
                    if clean_pattern.endswith("/**"):
                        dir_name = clean_pattern[:-3]
                        if f"/{dir_name}/" in f"/{rel_str}" or rel_str.startswith(f"{dir_name}/"):
                            return True
                    elif clean_pattern in rel_str:
                        return True
            except ValueError:
                pass
            return False

        return sorted(f for f in files if not is_excluded(f))

    def _resolve_import(self, source_file: Path, imp: ImportInfo) -> Path | None:
        """Resolve an import to a file path.

        Args:
            source_file: File containing the import.
            imp: Import information.

        Returns:
            Path to the imported file, or None if not resolvable.
        """
        if imp.is_relative:
            return self._resolve_relative_import(source_file, imp)
        return self._resolve_absolute_import(imp)

    def _resolve_relative_import(self, source_file: Path, imp: ImportInfo) -> Path | None:
        """Resolve a relative import."""
        # Start from the source file's directory
        base_dir = source_file.parent

        # Go up directories based on level
        for _ in range(imp.level - 1):
            base_dir = base_dir.parent

        # Resolve the module path
        if imp.module:
            parts = imp.module.split(".")
            for part in parts:
                base_dir = base_dir / part

        # Try as a package (__init__.py) or module (.py)
        init_file = base_dir / "__init__.py"
        module_file = base_dir.with_suffix(".py")

        if init_file.exists():
            return init_file.resolve()
        if module_file.exists():
            return module_file.resolve()

        return None

    def _resolve_absolute_import(self, imp: ImportInfo) -> Path | None:
        """Resolve an absolute import."""
        parts = imp.module.split(".")

        # Try to find the module in the project
        for start_dir in [self.root, self.root / "src"]:
            if not start_dir.exists():
                continue

            # Try as nested package
            module_dir = start_dir
            for part in parts:
                module_dir = module_dir / part

            init_file = module_dir / "__init__.py"
            module_file = module_dir.with_suffix(".py")

            if init_file.exists():
                return init_file.resolve()
            if module_file.exists():
                return module_file.resolve()

        return None

    def get_dependencies(self, file_path: Path) -> list[FileDependency]:
        """Get files that a given file depends on.

        Args:
            file_path: Path to the file.

        Returns:
            List of files this file imports from.
        """
        file_path = file_path.resolve()
        return self._graph.get(file_path, [])

    def get_dependents(self, file_path: Path) -> list[FileDependency]:
        """Get files that depend on a given file.

        Args:
            file_path: Path to the file.

        Returns:
            List of files that import from this file.
        """
        file_path = file_path.resolve()
        return self._reverse_graph.get(file_path, [])

    def get_dependent_files(self, file_path: Path) -> list[Path]:
        """Get just the file paths of dependents.

        Args:
            file_path: Path to the file.

        Returns:
            List of file paths that depend on this file.
        """
        deps = self.get_dependents(file_path)
        return [d.source_file for d in deps]

    def get_transitive_dependents(self, file_path: Path, *, max_depth: int = 10) -> list[Path]:
        """Get all files that transitively depend on a file.

        Args:
            file_path: Path to the file.
            max_depth: Maximum traversal depth.

        Returns:
            All files that directly or indirectly depend on this file.
        """
        file_path = file_path.resolve()
        visited: set[Path] = set()
        to_visit = [file_path]
        depth = 0

        while to_visit and depth < max_depth:
            current = to_visit.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for dep in self.get_dependents(current):
                if dep.source_file not in visited:
                    to_visit.append(dep.source_file)

            depth += 1

        # Remove the original file
        visited.discard(file_path)
        return sorted(visited)

    def get_imports(self, file_path: Path) -> list[ImportInfo]:
        """Get all imports from a file.

        Args:
            file_path: Path to the file.

        Returns:
            List of imports in the file.
        """
        file_path = file_path.resolve()
        return self._file_imports.get(file_path, [])

    def analyze_changes(
        self,
        changed_files: list[Path],
        *,
        check_transitive: bool = False,
    ) -> ChangeAnalysis:
        """Analyze the impact of file changes.

        Args:
            changed_files: List of files that were changed.
            check_transitive: Whether to check transitive dependencies.

        Returns:
            ChangeAnalysis with warnings and affected files.
        """
        analysis = ChangeAnalysis(changed_files=[f.resolve() for f in changed_files])

        affected: set[Path] = set()
        max_impact = ImpactLevel.NONE

        for file_path in changed_files:
            file_path = file_path.resolve()

            # Get direct dependents
            dependents = self.get_dependents(file_path)

            # Optionally get transitive dependents
            if check_transitive:
                transitive = self.get_transitive_dependents(file_path)
                for t in transitive:
                    if t not in affected:
                        affected.add(t)
            else:
                for dep in dependents:
                    affected.add(dep.source_file)

            # Generate warnings for each dependent
            for dep in dependents:
                warning = self._create_warning(file_path, dep)
                analysis.warnings.append(warning)

                if warning.impact_level.value > max_impact.value:
                    max_impact = warning.impact_level

        analysis.affected_files = sorted(affected)
        analysis.impact_level = max_impact
        analysis.safe_to_proceed = max_impact in (ImpactLevel.NONE, ImpactLevel.LOW)

        return analysis

    def _create_warning(self, changed_file: Path, dependency: FileDependency) -> DependencyWarning:
        """Create a warning for a potentially affected file."""
        imported_names = list(dependency.imported_names)

        # Determine impact level based on what's imported
        if not imported_names or imported_names == ["*"]:
            impact = ImpactLevel.HIGH
            reason = "Imports everything or uses wildcard"
            action = "Review dependent file for compatibility"
        elif len(imported_names) > 5:
            impact = ImpactLevel.MEDIUM
            reason = f"Imports {len(imported_names)} names from changed file"
            action = "Verify imported names still exist"
        else:
            impact = ImpactLevel.LOW
            reason = f"Imports: {', '.join(imported_names[:3])}"
            action = "Quick check that imports still work"

        return DependencyWarning(
            changed_file=changed_file,
            dependent_file=dependency.source_file,
            reason=reason,
            imported_names=imported_names,
            impact_level=impact,
            suggested_action=action,
        )

    def find_circular_dependencies(self) -> list[list[Path]]:
        """Find circular dependency chains.

        Returns:
            List of circular dependency chains.
        """
        cycles: list[list[Path]] = []
        visited: set[Path] = set()
        rec_stack: set[Path] = set()
        path: list[Path] = []

        def dfs(node: Path) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for dep in self._graph.get(node, []):
                target = dep.target_file
                if target not in visited:
                    if dfs(target):
                        return True
                elif target in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(target)
                    cycle = path[cycle_start:] + [target]
                    cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)
            return False

        for file_path in self._file_imports:
            if file_path not in visited:
                dfs(file_path)

        return cycles

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the dependency graph.

        Returns:
            Dictionary with graph statistics.
        """
        if not self._built:
            return {"error": "Graph not built"}

        # Calculate in-degree and out-degree
        in_degrees = {f: len(self._reverse_graph.get(f, [])) for f in self._file_imports}
        out_degrees = {f: len(self._graph.get(f, [])) for f in self._file_imports}

        # Find most connected files
        most_dependents = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        most_dependencies = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_files": self.file_count,
            "total_dependencies": self.dependency_count,
            "files_with_dependents": sum(1 for d in in_degrees.values() if d > 0),
            "files_with_dependencies": sum(1 for d in out_degrees.values() if d > 0),
            "avg_dependencies_per_file": (
                self.dependency_count / self.file_count if self.file_count > 0 else 0
            ),
            "most_dependents": [
                {"file": str(f.relative_to(self.root)), "count": c}
                for f, c in most_dependents
                if c > 0
            ],
            "most_dependencies": [
                {"file": str(f.relative_to(self.root)), "count": c}
                for f, c in most_dependencies
                if c > 0
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        """Export the dependency graph as a dictionary.

        Returns:
            Dictionary representation of the graph.
        """
        return {
            "root": str(self.root),
            "file_count": self.file_count,
            "dependency_count": self.dependency_count,
            "files": {
                str(f.relative_to(self.root)): {
                    "imports": [i.to_dict() for i in imports],
                    "dependents": [
                        str(d.source_file.relative_to(self.root))
                        for d in self._reverse_graph.get(f, [])
                    ],
                }
                for f, imports in self._file_imports.items()
            },
        }


# =============================================================================
# Change Detector
# =============================================================================


class ChangeDetector:
    """Detect what changed in a file and assess impact.

    This class compares file contents before and after changes
    to determine what symbols were modified and their impact.
    """

    def __init__(self) -> None:
        """Initialize the change detector."""
        pass

    def detect_changes(self, file_path: Path, old_content: str, new_content: str) -> dict[str, Any]:
        """Detect changes between old and new file content.

        Args:
            file_path: Path to the file.
            old_content: Previous file content.
            new_content: New file content.

        Returns:
            Dictionary describing the changes.
        """
        changes: dict[str, Any] = {
            "file": str(file_path),
            "added_symbols": [],
            "removed_symbols": [],
            "modified_symbols": [],
            "added_lines": 0,
            "removed_lines": 0,
        }

        # Parse both versions
        try:
            old_tree = ast.parse(old_content)
            new_tree = ast.parse(new_content)
        except SyntaxError:
            # Can't do detailed analysis with syntax errors
            old_lines = old_content.count("\n")
            new_lines = new_content.count("\n")
            changes["added_lines"] = max(0, new_lines - old_lines)
            changes["removed_lines"] = max(0, old_lines - new_lines)
            return changes

        # Extract symbols from both versions
        old_symbols = self._extract_symbols(old_tree)
        new_symbols = self._extract_symbols(new_tree)

        # Find added, removed, modified
        old_names = set(old_symbols.keys())
        new_names = set(new_symbols.keys())

        changes["added_symbols"] = list(new_names - old_names)
        changes["removed_symbols"] = list(old_names - new_names)

        # Check for modifications (same name, different signature)
        for name in old_names & new_names:
            if old_symbols[name] != new_symbols[name]:
                changes["modified_symbols"].append(name)

        # Line changes
        old_lines = old_content.count("\n")
        new_lines = new_content.count("\n")
        changes["added_lines"] = max(0, new_lines - old_lines)
        changes["removed_lines"] = max(0, old_lines - new_lines)

        return changes

    def _extract_symbols(self, tree: ast.AST) -> dict[str, str]:
        """Extract function and class signatures from AST.

        Returns a dict of symbol_name -> signature_hash.
        """
        symbols: dict[str, str] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                sig = self._function_signature(node)
                symbols[node.name] = sig
            elif isinstance(node, ast.ClassDef):
                sig = self._class_signature(node)
                symbols[node.name] = sig

        return symbols

    def _function_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Get a string representation of function signature."""
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f":{ast.dump(arg.annotation)}"
            args.append(arg_str)

        return_annotation = ""
        if node.returns:
            return_annotation = f"->{ast.dump(node.returns)}"

        return f"{node.name}({','.join(args)}){return_annotation}"

    def _class_signature(self, node: ast.ClassDef) -> str:
        """Get a string representation of class signature."""
        bases = [ast.dump(b) for b in node.bases]
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                methods.append(item.name)
        return f"{node.name}({','.join(bases)})[{','.join(sorted(methods))}]"

    def assess_impact(self, changes: dict[str, Any], tracker: DependencyTracker) -> ImpactLevel:
        """Assess the impact level of changes.

        Args:
            changes: Output from detect_changes.
            tracker: Dependency tracker with built graph.

        Returns:
            Impact level assessment.
        """
        if not changes["removed_symbols"] and not changes["modified_symbols"]:
            # Only additions, very safe
            return ImpactLevel.LOW

        # Check if removed/modified symbols are used by dependents
        file_path = Path(changes["file"])
        dependents = tracker.get_dependents(file_path)

        if not dependents:
            # No dependents, internal only
            return ImpactLevel.LOW

        # Check what's imported by dependents
        affected_symbols = set(changes["removed_symbols"] + changes["modified_symbols"])
        for dep in dependents:
            imported = dep.imported_names
            if affected_symbols & imported:
                # Breaking change - dependents import affected symbols
                return ImpactLevel.HIGH

        # Dependents exist but don't import affected symbols
        return ImpactLevel.MEDIUM


# =============================================================================
# Task Dependency Graph (Phase 5.1: Parallel Execution)
# =============================================================================


@dataclass
class TaskNode:
    """A node in the task dependency graph.

    Attributes:
        task_id: Unique identifier for the task.
        file_hints: Files this task will modify.
        explicit_deps: Explicitly declared task dependencies.
        inferred_deps: Dependencies inferred from file overlap.
    """

    task_id: str
    file_hints: set[str] = field(default_factory=set)
    explicit_deps: set[str] = field(default_factory=set)
    inferred_deps: set[str] = field(default_factory=set)

    @property
    def all_dependencies(self) -> set[str]:
        """Get all dependencies (explicit + inferred)."""
        return self.explicit_deps | self.inferred_deps

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "file_hints": sorted(self.file_hints),
            "explicit_deps": sorted(self.explicit_deps),
            "inferred_deps": sorted(self.inferred_deps),
        }


@dataclass
class ParallelGroup:
    """A group of tasks that can execute in parallel.

    Attributes:
        tasks: Task IDs in this group.
        level: Execution level (0 = can start immediately, 1 = after level 0, etc.)
        estimated_duration: Estimated total duration in seconds.
    """

    tasks: list[str] = field(default_factory=list)
    level: int = 0
    estimated_duration: float = 0.0

    @property
    def size(self) -> int:
        """Number of tasks in this group."""
        return len(self.tasks)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tasks": self.tasks,
            "level": self.level,
            "estimated_duration": self.estimated_duration,
        }


class TaskDependencyGraph:
    """Analyze task dependencies and identify parallelizable groups.

    This class builds a dependency graph from task file_hints and explicit
    dependencies to determine which tasks can safely run in parallel.

    Two tasks can run in parallel if:
    1. They have no overlapping file_hints (no shared files to modify)
    2. Neither explicitly depends on the other
    3. They don't transitively depend on each other through file overlap

    Example:
        >>> from ai_infra.executor.dependencies import TaskDependencyGraph
        >>> graph = TaskDependencyGraph()
        >>> graph.add_task("task-1", file_hints=["src/a.py"])
        >>> graph.add_task("task-2", file_hints=["src/b.py"])
        >>> graph.add_task("task-3", file_hints=["src/a.py"], dependencies=["task-1"])
        >>> groups = graph.get_parallel_groups()
        >>> print(groups[0].tasks)  # ["task-1", "task-2"] can run together
        >>> print(groups[1].tasks)  # ["task-3"] must wait for task-1
    """

    def __init__(
        self,
        *,
        dependency_tracker: DependencyTracker | None = None,
        use_file_overlap: bool = True,
        use_import_analysis: bool = True,
    ) -> None:
        """Initialize the task dependency graph.

        Args:
            dependency_tracker: Optional DependencyTracker for import analysis.
            use_file_overlap: Consider tasks with overlapping files as dependent.
            use_import_analysis: Use import graph to find transitive dependencies.
        """
        self._tracker = dependency_tracker
        self._use_file_overlap = use_file_overlap
        self._use_import_analysis = use_import_analysis

        self._nodes: dict[str, TaskNode] = {}
        self._file_to_tasks: dict[str, set[str]] = {}  # file -> tasks that modify it
        self._built = False

    @property
    def is_built(self) -> bool:
        """Whether the graph has been built."""
        return self._built

    @property
    def task_count(self) -> int:
        """Number of tasks in the graph."""
        return len(self._nodes)

    def add_task(
        self,
        task_id: str,
        *,
        file_hints: list[str] | None = None,
        dependencies: list[str] | None = None,
    ) -> None:
        """Add a task to the graph.

        Args:
            task_id: Unique task identifier.
            file_hints: Files this task will modify.
            dependencies: Explicit task dependencies.
        """
        file_set = set(file_hints or [])
        dep_set = set(dependencies or [])

        node = TaskNode(
            task_id=task_id,
            file_hints=file_set,
            explicit_deps=dep_set,
        )
        self._nodes[task_id] = node

        # Index files for overlap detection
        for file_hint in file_set:
            if file_hint not in self._file_to_tasks:
                self._file_to_tasks[file_hint] = set()
            self._file_to_tasks[file_hint].add(task_id)

        self._built = False  # Need to rebuild

    def add_tasks(
        self,
        tasks: list[tuple[str, list[str], list[str]]],
    ) -> None:
        """Add multiple tasks at once.

        Args:
            tasks: List of (task_id, file_hints, dependencies) tuples.
        """
        for task_id, file_hints, dependencies in tasks:
            self.add_task(task_id, file_hints=file_hints, dependencies=dependencies)

    def build(self) -> None:
        """Build the dependency graph by analyzing file overlaps.

        This populates inferred_deps for each task based on:
        1. File hint overlap (two tasks modify the same file)
        2. Import graph analysis (if dependency_tracker is provided)
        """
        if self._built:
            return

        logger.debug(f"Building task dependency graph for {len(self._nodes)} tasks")

        # Reset inferred dependencies
        for node in self._nodes.values():
            node.inferred_deps.clear()

        # Find dependencies from file overlap
        if self._use_file_overlap:
            self._infer_from_file_overlap()

        # Find dependencies from import analysis
        if self._use_import_analysis and self._tracker and self._tracker.is_built:
            self._infer_from_imports()

        self._built = True
        logger.debug(
            f"Task dependency graph built: "
            f"{sum(len(n.all_dependencies) for n in self._nodes.values())} dependencies"
        )

    def _infer_from_file_overlap(self) -> None:
        """Infer dependencies based on overlapping file_hints.

        If two tasks both modify the same file, the later task depends on
        the earlier one. This uses task_id ordering as a proxy for order.
        """
        for file_hint, task_ids in self._file_to_tasks.items():
            if len(task_ids) <= 1:
                continue

            # Sort tasks by ID to establish order
            sorted_tasks = sorted(task_ids)

            # Each task depends on all earlier tasks that share this file
            for i, task_id in enumerate(sorted_tasks[1:], start=1):
                earlier_tasks = set(sorted_tasks[:i])
                self._nodes[task_id].inferred_deps.update(earlier_tasks)

    def _infer_from_imports(self) -> None:
        """Infer dependencies based on import graph analysis.

        If task A modifies file X and task B modifies file Y that imports X,
        then task B depends on task A.
        """
        if not self._tracker:
            return

        # For each task, check if any of its files are imported by files
        # that other tasks will modify
        for task_id, node in self._nodes.items():
            for file_hint in node.file_hints:
                file_path = Path(file_hint)
                if not file_path.is_absolute():
                    continue

                dependents = self._tracker.get_dependent_files(file_path)
                for dependent_file in dependents:
                    dep_str = str(dependent_file)
                    # Find tasks that modify the dependent file
                    if dep_str in self._file_to_tasks:
                        for other_task_id in self._file_to_tasks[dep_str]:
                            if other_task_id != task_id:
                                # The other task depends on this task
                                self._nodes[other_task_id].inferred_deps.add(task_id)

    def get_dependencies(self, task_id: str) -> set[str]:
        """Get all dependencies for a task.

        Args:
            task_id: Task identifier.

        Returns:
            Set of task IDs this task depends on.
        """
        if not self._built:
            self.build()

        if task_id not in self._nodes:
            return set()

        return self._nodes[task_id].all_dependencies

    def get_dependents(self, task_id: str) -> set[str]:
        """Get all tasks that depend on a given task.

        Args:
            task_id: Task identifier.

        Returns:
            Set of task IDs that depend on this task.
        """
        if not self._built:
            self.build()

        dependents: set[str] = set()
        for other_id, node in self._nodes.items():
            if task_id in node.all_dependencies:
                dependents.add(other_id)
        return dependents

    def can_run_parallel(self, task_a: str, task_b: str) -> bool:
        """Check if two tasks can run in parallel.

        Args:
            task_a: First task ID.
            task_b: Second task ID.

        Returns:
            True if the tasks can safely run in parallel.
        """
        if not self._built:
            self.build()

        if task_a not in self._nodes or task_b not in self._nodes:
            return False

        # Check if either task depends on the other
        if task_a in self._nodes[task_b].all_dependencies:
            return False
        if task_b in self._nodes[task_a].all_dependencies:
            return False

        return True

    def get_parallel_groups(
        self,
        *,
        pending_only: list[str] | None = None,
    ) -> list[ParallelGroup]:
        """Get groups of tasks that can execute in parallel.

        Returns tasks organized into levels. All tasks in a level can
        run in parallel. Level N+1 can only start after level N completes.

        Args:
            pending_only: If provided, only consider these task IDs.

        Returns:
            List of ParallelGroup objects, ordered by execution level.
        """
        if not self._built:
            self.build()

        # Filter to pending tasks if specified
        if pending_only is not None:
            task_ids = set(pending_only) & set(self._nodes.keys())
        else:
            task_ids = set(self._nodes.keys())

        if not task_ids:
            return []

        # Topological sort with level tracking
        groups: list[ParallelGroup] = []
        remaining = set(task_ids)
        completed: set[str] = set()
        level = 0

        while remaining:
            # Find tasks with all dependencies satisfied
            ready: list[str] = []
            for task_id in remaining:
                deps = self.get_dependencies(task_id)
                # Only consider dependencies that are in our task set
                relevant_deps = deps & task_ids
                if relevant_deps <= completed:
                    ready.append(task_id)

            if not ready:
                # Circular dependency detected - break by taking first remaining
                logger.warning(
                    f"Circular dependency detected in tasks: {remaining}. "
                    "Breaking cycle by processing remaining tasks sequentially."
                )
                ready = [sorted(remaining)[0]]

            # Create group from ready tasks
            group = ParallelGroup(tasks=sorted(ready), level=level)
            groups.append(group)

            # Mark as completed
            for task_id in ready:
                completed.add(task_id)
                remaining.discard(task_id)

            level += 1

        return groups

    def get_execution_order(
        self,
        *,
        pending_only: list[str] | None = None,
    ) -> list[str]:
        """Get a valid sequential execution order respecting dependencies.

        Args:
            pending_only: If provided, only consider these task IDs.

        Returns:
            List of task IDs in valid execution order.
        """
        groups = self.get_parallel_groups(pending_only=pending_only)
        order: list[str] = []
        for group in groups:
            order.extend(group.tasks)
        return order

    def get_independent_tasks(
        self,
        *,
        pending_only: list[str] | None = None,
    ) -> list[str]:
        """Get tasks that have no dependencies and can start immediately.

        Args:
            pending_only: If provided, only consider these task IDs.

        Returns:
            List of task IDs that can start immediately.
        """
        groups = self.get_parallel_groups(pending_only=pending_only)
        if groups:
            return groups[0].tasks
        return []

    def to_dict(self) -> dict[str, Any]:
        """Convert graph to dictionary for debugging."""
        return {
            "task_count": self.task_count,
            "is_built": self._built,
            "nodes": {tid: node.to_dict() for tid, node in self._nodes.items()},
            "file_to_tasks": {f: sorted(tasks) for f, tasks in self._file_to_tasks.items()},
        }

    def visualize(self) -> str:
        """Generate a text visualization of the dependency graph.

        Returns:
            ASCII art representation of the graph.
        """
        if not self._built:
            self.build()

        lines = ["Task Dependency Graph", "=" * 40]

        groups = self.get_parallel_groups()
        for group in groups:
            lines.append(f"\nLevel {group.level} (can run in parallel):")
            for task_id in group.tasks:
                node = self._nodes[task_id]
                deps = node.all_dependencies
                if deps:
                    lines.append(f"  - {task_id} (depends on: {', '.join(sorted(deps))})")
                else:
                    lines.append(f"  - {task_id}")

        return "\n".join(lines)
