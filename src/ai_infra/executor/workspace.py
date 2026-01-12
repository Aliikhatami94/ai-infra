"""Multi-project workspace support for the Executor (Phase 5.4).

Provides infrastructure for:
- Monorepo structure detection and handling
- Cross-project dependency tracking
- Coordinated checkpoints across multiple projects
- Workspace-aware task execution

A workspace can contain multiple projects, each with its own:
- ROADMAP.md
- Package configuration (pyproject.toml, package.json, etc.)
- Dependencies on other projects in the workspace
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ai_infra.executor.checkpoint import (
    Checkpointer,
    CheckpointError,
    CheckpointResult,
    GitOperationError,
    NotAGitRepoError,
    RollbackResult,
)
from ai_infra.logging import get_logger

logger = get_logger("executor.workspace")


# =============================================================================
# Enums
# =============================================================================


class ProjectType(Enum):
    """Type of project in the workspace."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"
    UNKNOWN = "unknown"


class DependencyScope(Enum):
    """Scope of a cross-project dependency."""

    BUILD = "build"  # Required at build time
    RUNTIME = "runtime"  # Required at runtime
    DEV = "dev"  # Development dependency only
    OPTIONAL = "optional"  # Optional dependency


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ProjectInfo:
    """Information about a project in the workspace.

    Attributes:
        name: Project name.
        path: Absolute path to the project root.
        project_type: Type of project (Python, JS, etc.).
        has_roadmap: Whether the project has a ROADMAP.md.
        roadmap_path: Path to the ROADMAP.md if it exists.
        config_file: Path to the main config file (pyproject.toml, etc.).
        version: Project version if detectable.
        dependencies: List of dependencies on other workspace projects.
    """

    name: str
    path: Path
    project_type: ProjectType = ProjectType.UNKNOWN
    has_roadmap: bool = False
    roadmap_path: Path | None = None
    config_file: Path | None = None
    version: str = ""
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "path": str(self.path),
            "project_type": self.project_type.value,
            "has_roadmap": self.has_roadmap,
            "roadmap_path": str(self.roadmap_path) if self.roadmap_path else None,
            "config_file": str(self.config_file) if self.config_file else None,
            "version": self.version,
            "dependencies": self.dependencies,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectInfo:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            path=Path(data["path"]),
            project_type=ProjectType(data.get("project_type", "unknown")),
            has_roadmap=data.get("has_roadmap", False),
            roadmap_path=Path(data["roadmap_path"]) if data.get("roadmap_path") else None,
            config_file=Path(data["config_file"]) if data.get("config_file") else None,
            version=data.get("version", ""),
            dependencies=data.get("dependencies", []),
        )


@dataclass
class CrossProjectDependency:
    """A dependency between two projects in the workspace.

    Attributes:
        source_project: Name of the project that has the dependency.
        target_project: Name of the project being depended on.
        scope: Scope of the dependency.
        version_constraint: Version constraint if specified.
        import_path: Import path used to reference the dependency.
    """

    source_project: str
    target_project: str
    scope: DependencyScope = DependencyScope.RUNTIME
    version_constraint: str = ""
    import_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_project": self.source_project,
            "target_project": self.target_project,
            "scope": self.scope.value,
            "version_constraint": self.version_constraint,
            "import_path": self.import_path,
        }


@dataclass
class WorkspaceCheckpointResult:
    """Result of a coordinated checkpoint across multiple projects.

    Attributes:
        success: Whether all project checkpoints succeeded.
        project_results: Results per project.
        commit_sha: The common commit SHA (if single repo).
        failed_projects: Projects that failed to checkpoint.
        message: Overall status message.
    """

    success: bool
    project_results: dict[str, CheckpointResult] = field(default_factory=dict)
    commit_sha: str | None = None
    failed_projects: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "project_results": {k: v.to_dict() for k, v in self.project_results.items()},
            "commit_sha": self.commit_sha,
            "failed_projects": self.failed_projects,
            "message": self.message,
        }


@dataclass
class WorkspaceRollbackResult:
    """Result of a coordinated rollback across multiple projects.

    Attributes:
        success: Whether all project rollbacks succeeded.
        project_results: Results per project.
        failed_projects: Projects that failed to rollback.
        message: Overall status message.
    """

    success: bool
    project_results: dict[str, RollbackResult] = field(default_factory=dict)
    failed_projects: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "project_results": {k: v.to_dict() for k, v in self.project_results.items()},
            "failed_projects": self.failed_projects,
            "message": self.message,
        }


# =============================================================================
# Workspace
# =============================================================================


class Workspace:
    """Multi-project workspace manager.

    Handles monorepo structures with multiple projects, tracking
    cross-project dependencies and coordinating checkpoints.

    Example:
        >>> workspace = Workspace(Path("/path/to/monorepo"))
        >>> workspace.discover_projects()
        >>>
        >>> # Get all projects with roadmaps
        >>> for project in workspace.projects_with_roadmaps():
        ...     print(f"{project.name}: {project.roadmap_path}")
        >>>
        >>> # Check cross-project dependencies
        >>> deps = workspace.get_dependencies("my-service")
        >>> for dep in deps:
        ...     print(f"Depends on: {dep.target_project}")
        >>>
        >>> # Coordinated checkpoint
        >>> result = workspace.checkpoint_all(
        ...     task_id="1.1.1",
        ...     task_title="Add feature",
        ...     affected_projects=["my-service", "shared-lib"],
        ... )
    """

    def __init__(
        self,
        root: Path | str,
        *,
        auto_discover: bool = True,
        include_hidden: bool = False,
        max_depth: int = 3,
    ) -> None:
        """Initialize the workspace.

        Args:
            root: Root path of the workspace.
            auto_discover: Automatically discover projects on init.
            include_hidden: Include hidden directories in discovery.
            max_depth: Maximum depth for project discovery.
        """
        self._root = Path(root).resolve()
        self._include_hidden = include_hidden
        self._max_depth = max_depth

        # Project tracking
        self._projects: dict[str, ProjectInfo] = {}
        self._dependencies: list[CrossProjectDependency] = []

        # Git integration (lazy init)
        self._checkpointer: Checkpointer | None = None
        self._is_git_repo: bool | None = None

        if auto_discover:
            self.discover_projects()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def root(self) -> Path:
        """The workspace root path."""
        return self._root

    @property
    def projects(self) -> dict[str, ProjectInfo]:
        """Dictionary of discovered projects by name."""
        return self._projects.copy()

    @property
    def project_count(self) -> int:
        """Number of projects in the workspace."""
        return len(self._projects)

    @property
    def is_git_repo(self) -> bool:
        """Whether the workspace is a git repository."""
        if self._is_git_repo is None:
            self._is_git_repo = (self._root / ".git").is_dir()
        return self._is_git_repo

    @property
    def is_monorepo(self) -> bool:
        """Whether this appears to be a monorepo."""
        # Monorepo indicators: multiple projects or workspace config files
        if self.project_count > 1:
            return True
        # Check for workspace config files
        workspace_indicators = [
            self._root / "pnpm-workspace.yaml",
            self._root / "lerna.json",
            self._root / "nx.json",
            self._root / "rush.json",
        ]
        return any(f.exists() for f in workspace_indicators)

    # =========================================================================
    # Project Discovery
    # =========================================================================

    def discover_projects(self) -> list[ProjectInfo]:
        """Discover all projects in the workspace.

        Scans the workspace for project indicators like:
        - pyproject.toml (Python)
        - package.json (JavaScript/TypeScript)
        - Cargo.toml (Rust)
        - go.mod (Go)
        - ROADMAP.md (any project with executor tasks)

        Returns:
            List of discovered projects.
        """
        self._projects.clear()
        self._dependencies.clear()

        discovered = self._scan_for_projects(self._root, depth=0)

        # Build dependency graph
        self._analyze_dependencies()

        logger.info(f"Discovered {len(discovered)} projects in workspace")
        return discovered

    def _scan_for_projects(self, path: Path, depth: int) -> list[ProjectInfo]:
        """Recursively scan for projects."""
        if depth > self._max_depth:
            return []

        projects: list[ProjectInfo] = []

        # Check if current directory is a project
        project = self._detect_project(path)
        if project:
            self._projects[project.name] = project
            projects.append(project)
            # Don't recurse into recognized projects (they're self-contained)
            # But do check immediate subdirectories for nested projects
            if depth == 0:
                # Only recurse at root level
                for subdir in self._get_subdirectories(path):
                    projects.extend(self._scan_for_projects(subdir, depth + 1))
        else:
            # Not a project, check subdirectories
            for subdir in self._get_subdirectories(path):
                projects.extend(self._scan_for_projects(subdir, depth + 1))

        return projects

    def _get_subdirectories(self, path: Path) -> list[Path]:
        """Get subdirectories to scan."""
        subdirs = []
        try:
            for item in path.iterdir():
                if not item.is_dir():
                    continue
                name = item.name
                # Skip hidden directories unless configured
                if name.startswith(".") and not self._include_hidden:
                    continue
                # Skip common non-project directories
                if name in {
                    "node_modules",
                    "__pycache__",
                    ".git",
                    ".venv",
                    "venv",
                    "dist",
                    "build",
                    "target",
                    "htmlcov",
                    ".pytest_cache",
                    ".mypy_cache",
                    ".ruff_cache",
                    "site-packages",
                }:
                    continue
                subdirs.append(item)
        except PermissionError:
            pass
        return subdirs

    def _detect_project(self, path: Path) -> ProjectInfo | None:
        """Detect if a path is a project and gather info."""
        # Check for project indicators
        pyproject = path / "pyproject.toml"
        package_json = path / "package.json"
        cargo_toml = path / "Cargo.toml"
        go_mod = path / "go.mod"
        roadmap = path / "ROADMAP.md"

        project_type = ProjectType.UNKNOWN
        config_file = None
        version = ""
        name = path.name

        if pyproject.exists():
            project_type = ProjectType.PYTHON
            config_file = pyproject
            name, version = self._parse_pyproject(pyproject)
        elif package_json.exists():
            project_type = ProjectType.JAVASCRIPT
            config_file = package_json
            name, version, is_ts = self._parse_package_json(package_json, path)
            if is_ts:
                project_type = ProjectType.TYPESCRIPT
        elif cargo_toml.exists():
            project_type = ProjectType.RUST
            config_file = cargo_toml
            name, version = self._parse_cargo_toml(cargo_toml)
        elif go_mod.exists():
            project_type = ProjectType.GO
            config_file = go_mod
            name = self._parse_go_mod(go_mod)
        elif roadmap.exists():
            # Has a roadmap but no recognized config
            project_type = ProjectType.UNKNOWN
        else:
            # Not a project
            return None

        return ProjectInfo(
            name=name or path.name,
            path=path,
            project_type=project_type,
            has_roadmap=roadmap.exists(),
            roadmap_path=roadmap if roadmap.exists() else None,
            config_file=config_file,
            version=version,
        )

    def _parse_pyproject(self, path: Path) -> tuple[str, str]:
        """Parse pyproject.toml for name and version."""
        try:
            import tomllib

            with open(path, "rb") as f:
                data = tomllib.load(f)
            project = data.get("project", {})
            poetry = data.get("tool", {}).get("poetry", {})
            name = project.get("name") or poetry.get("name") or ""
            version = project.get("version") or poetry.get("version") or ""
            return name, version
        except Exception:
            return "", ""

    def _parse_package_json(self, path: Path, project_path: Path) -> tuple[str, str, bool]:
        """Parse package.json for name, version, and TypeScript detection."""
        try:
            with open(path) as f:
                data = json.load(f)
            name = data.get("name", "")
            version = data.get("version", "")
            # Check for TypeScript
            is_ts = (
                "typescript" in data.get("devDependencies", {})
                or "typescript" in data.get("dependencies", {})
                or (project_path / "tsconfig.json").exists()
            )
            return name, version, is_ts
        except Exception:
            return "", "", False

    def _parse_cargo_toml(self, path: Path) -> tuple[str, str]:
        """Parse Cargo.toml for name and version."""
        try:
            import tomllib

            with open(path, "rb") as f:
                data = tomllib.load(f)
            package = data.get("package", {})
            return package.get("name", ""), package.get("version", "")
        except Exception:
            return "", ""

    def _parse_go_mod(self, path: Path) -> str:
        """Parse go.mod for module name."""
        try:
            content = path.read_text()
            match = re.search(r"^module\s+(\S+)", content, re.MULTILINE)
            if match:
                # Get last part of module path
                module_path = match.group(1)
                return module_path.split("/")[-1]
        except Exception:
            pass
        return ""

    # =========================================================================
    # Dependency Analysis
    # =========================================================================

    def _analyze_dependencies(self) -> None:
        """Analyze cross-project dependencies."""
        project_names = set(self._projects.keys())

        for project in self._projects.values():
            if project.project_type == ProjectType.PYTHON:
                self._analyze_python_dependencies(project, project_names)
            elif project.project_type in (ProjectType.JAVASCRIPT, ProjectType.TYPESCRIPT):
                self._analyze_js_dependencies(project, project_names)

    def _analyze_python_dependencies(
        self,
        project: ProjectInfo,
        project_names: set[str],
    ) -> None:
        """Analyze Python project dependencies."""
        if not project.config_file or not project.config_file.exists():
            return

        try:
            import tomllib

            with open(project.config_file, "rb") as f:
                data = tomllib.load(f)

            # Check pyproject.toml dependencies
            deps: list[str] = []

            # PEP 621 style
            deps.extend(data.get("project", {}).get("dependencies", []))
            deps.extend(data.get("project", {}).get("optional-dependencies", {}).get("dev", []))

            # Poetry style
            poetry = data.get("tool", {}).get("poetry", {})
            deps.extend(poetry.get("dependencies", {}).keys())
            deps.extend(poetry.get("dev-dependencies", {}).keys())
            deps.extend(poetry.get("group", {}).get("dev", {}).get("dependencies", {}).keys())

            # Check for workspace project references
            for dep in deps:
                # Normalize dependency name
                dep_name = re.split(r"[<>=\[\]]", dep)[0].strip().replace("-", "_")
                dep_normalized = dep_name.lower()

                # Check if this matches a workspace project
                for project_name in project_names:
                    if project_name == project.name:
                        continue
                    # Normalize project name for comparison
                    proj_normalized = project_name.lower().replace("-", "_")
                    if dep_normalized == proj_normalized or dep_normalized.endswith(
                        f"_{proj_normalized}"
                    ):
                        self._dependencies.append(
                            CrossProjectDependency(
                                source_project=project.name,
                                target_project=project_name,
                                scope=DependencyScope.RUNTIME,
                                import_path=dep_name,
                            )
                        )
                        if project_name not in project.dependencies:
                            project.dependencies.append(project_name)

        except Exception as e:
            logger.debug(f"Failed to parse Python dependencies for {project.name}: {e}")

    def _analyze_js_dependencies(
        self,
        project: ProjectInfo,
        project_names: set[str],
    ) -> None:
        """Analyze JavaScript/TypeScript project dependencies."""
        if not project.config_file or not project.config_file.exists():
            return

        try:
            with open(project.config_file) as f:
                data = json.load(f)

            # Collect all dependencies
            all_deps: dict[str, str] = {}
            all_deps.update(data.get("dependencies", {}))
            all_deps.update(data.get("devDependencies", {}))
            all_deps.update(data.get("peerDependencies", {}))

            for dep_name, version in all_deps.items():
                # Check if this matches a workspace project
                for project_name in project_names:
                    if project_name == project.name:
                        continue
                    if dep_name == project_name or dep_name.endswith(f"/{project_name}"):
                        scope = (
                            DependencyScope.DEV
                            if dep_name in data.get("devDependencies", {})
                            else DependencyScope.RUNTIME
                        )
                        self._dependencies.append(
                            CrossProjectDependency(
                                source_project=project.name,
                                target_project=project_name,
                                scope=scope,
                                version_constraint=version,
                                import_path=dep_name,
                            )
                        )
                        if project_name not in project.dependencies:
                            project.dependencies.append(project_name)

        except Exception as e:
            logger.debug(f"Failed to parse JS dependencies for {project.name}: {e}")

    def get_dependencies(self, project_name: str) -> list[CrossProjectDependency]:
        """Get cross-project dependencies for a project.

        Args:
            project_name: Name of the project.

        Returns:
            List of dependencies.
        """
        return [d for d in self._dependencies if d.source_project == project_name]

    def get_dependents(self, project_name: str) -> list[CrossProjectDependency]:
        """Get projects that depend on the given project.

        Args:
            project_name: Name of the project.

        Returns:
            List of dependencies where this project is the target.
        """
        return [d for d in self._dependencies if d.target_project == project_name]

    def get_dependency_order(self) -> list[str]:
        """Get project names in dependency order (dependencies first).

        Returns:
            List of project names in topological order.
        """
        # Build adjacency list
        graph: dict[str, set[str]] = {name: set() for name in self._projects}
        for dep in self._dependencies:
            if dep.source_project in graph and dep.target_project in graph:
                graph[dep.source_project].add(dep.target_project)

        # Topological sort (Kahn's algorithm)
        in_degree: dict[str, int] = dict.fromkeys(self._projects, 0)
        for deps in graph.values():
            for dep in deps:
                in_degree[dep] = in_degree.get(dep, 0) + 1

        # Start with nodes that have no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in graph.get(node, set()):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Handle cycles (add remaining in any order)
        for name in self._projects:
            if name not in result:
                result.append(name)

        # Reverse so dependencies come first
        return list(reversed(result))

    # =========================================================================
    # Project Queries
    # =========================================================================

    def get_project(self, name: str) -> ProjectInfo | None:
        """Get a project by name."""
        return self._projects.get(name)

    def get_project_by_path(self, path: Path | str) -> ProjectInfo | None:
        """Get a project by its path."""
        path = Path(path).resolve()
        for project in self._projects.values():
            if project.path == path:
                return project
        return None

    def projects_with_roadmaps(self) -> list[ProjectInfo]:
        """Get all projects that have ROADMAP.md files."""
        return [p for p in self._projects.values() if p.has_roadmap]

    def projects_of_type(self, project_type: ProjectType) -> list[ProjectInfo]:
        """Get all projects of a specific type."""
        return [p for p in self._projects.values() if p.project_type == project_type]

    def find_project_for_file(self, file_path: Path | str) -> ProjectInfo | None:
        """Find which project a file belongs to.

        Args:
            file_path: Path to the file.

        Returns:
            The project containing the file, or None.
        """
        file_path = Path(file_path).resolve()

        # Find the project with the longest matching path
        best_match: ProjectInfo | None = None
        best_length = 0

        for project in self._projects.values():
            try:
                file_path.relative_to(project.path)
                if len(str(project.path)) > best_length:
                    best_match = project
                    best_length = len(str(project.path))
            except ValueError:
                continue

        return best_match

    # =========================================================================
    # Coordinated Checkpointing
    # =========================================================================

    def _get_checkpointer(self) -> Checkpointer | None:
        """Get or create the checkpointer."""
        if self._checkpointer is None and self.is_git_repo:
            try:
                self._checkpointer = Checkpointer(self._root)
            except (GitOperationError, NotAGitRepoError):
                self._checkpointer = None
        return self._checkpointer

    def checkpoint_all(
        self,
        task_id: str,
        task_title: str,
        *,
        affected_projects: list[str] | None = None,
        files_modified: list[str] | None = None,
        message: str | None = None,
    ) -> WorkspaceCheckpointResult:
        """Create a coordinated checkpoint across affected projects.

        For monorepos with a single git root, this creates a single commit.
        For multi-repo workspaces, it attempts to commit in each affected repo.

        Args:
            task_id: The task ID.
            task_title: The task title.
            affected_projects: Projects affected by the task (auto-detected if None).
            files_modified: Files that were modified.
            message: Custom commit message.

        Returns:
            WorkspaceCheckpointResult with per-project results.
        """
        files_modified = files_modified or []

        # Auto-detect affected projects from modified files
        if affected_projects is None:
            affected_projects = []
            for file_path in files_modified:
                project = self.find_project_for_file(file_path)
                if project and project.name not in affected_projects:
                    affected_projects.append(project.name)

        if not affected_projects:
            return WorkspaceCheckpointResult(
                success=True,
                message="No affected projects to checkpoint",
            )

        # Single git repo at workspace root - single commit
        if self.is_git_repo:
            checkpointer = self._get_checkpointer()
            if checkpointer:
                # Build commit message with affected projects
                affected_str = ", ".join(affected_projects)
                if message is None:
                    title_truncated = task_title[:40]
                    if len(task_title) > 40:
                        title_truncated = title_truncated[:37] + "..."
                    message = f"executor({task_id}): {title_truncated}\n\nAffected: {affected_str}"

                result = checkpointer.checkpoint(
                    task_id=task_id,
                    task_title=task_title,
                    files_modified=files_modified,
                    message=message,
                )

                project_results = dict.fromkeys(affected_projects, result)

                return WorkspaceCheckpointResult(
                    success=result.success,
                    project_results=project_results,
                    commit_sha=result.commit_sha,
                    failed_projects=[] if result.success else affected_projects,
                    message=result.message,
                )

        # Multi-repo workspace - checkpoint each project's repo
        project_results: dict[str, CheckpointResult] = {}
        failed_projects: list[str] = []

        for project_name in affected_projects:
            project = self._projects.get(project_name)
            if not project:
                continue

            # Check if project has its own git repo
            if (project.path / ".git").is_dir():
                try:
                    checkpointer = Checkpointer(project.path)
                    project_files = [
                        f for f in files_modified if self.find_project_for_file(f) == project
                    ]
                    result = checkpointer.checkpoint(
                        task_id=task_id,
                        task_title=task_title,
                        files_modified=project_files,
                        message=message,
                    )
                    project_results[project_name] = result
                    if not result.success:
                        failed_projects.append(project_name)
                except CheckpointError as e:
                    project_results[project_name] = CheckpointResult(
                        success=False,
                        error=str(e),
                    )
                    failed_projects.append(project_name)

        success = len(failed_projects) == 0
        return WorkspaceCheckpointResult(
            success=success,
            project_results=project_results,
            failed_projects=failed_projects,
            message=f"Checkpointed {len(project_results) - len(failed_projects)}/{len(project_results)} projects",
        )

    def rollback_all(
        self,
        task_id: str,
        *,
        affected_projects: list[str] | None = None,
        hard: bool = False,
    ) -> WorkspaceRollbackResult:
        """Rollback a task across all affected projects.

        Args:
            task_id: The task ID to rollback.
            affected_projects: Projects to rollback (all if None).
            hard: Whether to discard changes.

        Returns:
            WorkspaceRollbackResult with per-project results.
        """
        if affected_projects is None:
            affected_projects = list(self._projects.keys())

        # Single git repo
        if self.is_git_repo:
            checkpointer = self._get_checkpointer()
            if checkpointer:
                result = checkpointer.rollback(task_id, hard=hard)
                project_results = dict.fromkeys(affected_projects, result)

                return WorkspaceRollbackResult(
                    success=result.success,
                    project_results=project_results,
                    failed_projects=[] if result.success else affected_projects,
                    message=result.message,
                )

        # Multi-repo
        project_results: dict[str, RollbackResult] = {}
        failed_projects: list[str] = []

        for project_name in affected_projects:
            project = self._projects.get(project_name)
            if not project:
                continue

            if (project.path / ".git").is_dir():
                try:
                    checkpointer = Checkpointer(project.path)
                    result = checkpointer.rollback(task_id, hard=hard)
                    project_results[project_name] = result
                    if not result.success:
                        failed_projects.append(project_name)
                except CheckpointError as e:
                    project_results[project_name] = RollbackResult(
                        success=False,
                        error=str(e),
                    )
                    failed_projects.append(project_name)

        success = len(failed_projects) == 0
        return WorkspaceRollbackResult(
            success=success,
            project_results=project_results,
            failed_projects=failed_projects,
            message=f"Rolled back {len(project_results) - len(failed_projects)}/{len(project_results)} projects",
        )

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Convert workspace info to dictionary."""
        return {
            "root": str(self._root),
            "is_monorepo": self.is_monorepo,
            "is_git_repo": self.is_git_repo,
            "project_count": self.project_count,
            "projects": {name: p.to_dict() for name, p in self._projects.items()},
            "dependencies": [d.to_dict() for d in self._dependencies],
            "dependency_order": self.get_dependency_order(),
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Workspace: {self._root.name}",
            f"Path: {self._root}",
            f"Is Monorepo: {self.is_monorepo}",
            f"Is Git Repo: {self.is_git_repo}",
            f"Projects: {self.project_count}",
        ]

        if self._projects:
            lines.append("\nProjects:")
            for name, project in sorted(self._projects.items()):
                roadmap_indicator = " (has ROADMAP)" if project.has_roadmap else ""
                deps_indicator = (
                    f" -> [{', '.join(project.dependencies)}]" if project.dependencies else ""
                )
                lines.append(
                    f"  - {name} ({project.project_type.value}){roadmap_indicator}{deps_indicator}"
                )

        if self._dependencies:
            lines.append(f"\nCross-Project Dependencies: {len(self._dependencies)}")

        return "\n".join(lines)
