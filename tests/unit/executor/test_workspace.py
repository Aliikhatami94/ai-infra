"""Tests for the workspace module (Phase 5.4)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ai_infra.executor.workspace import (
    CrossProjectDependency,
    DependencyScope,
    ProjectInfo,
    ProjectType,
    Workspace,
    WorkspaceCheckpointResult,
    WorkspaceRollbackResult,
)

# =============================================================================
# Test Data Models
# =============================================================================


class TestProjectType:
    """Tests for ProjectType enum."""

    def test_project_type_values(self):
        """Test all project types have correct values."""
        assert ProjectType.PYTHON.value == "python"
        assert ProjectType.JAVASCRIPT.value == "javascript"
        assert ProjectType.TYPESCRIPT.value == "typescript"
        assert ProjectType.RUST.value == "rust"
        assert ProjectType.GO.value == "go"
        assert ProjectType.UNKNOWN.value == "unknown"


class TestDependencyScope:
    """Tests for DependencyScope enum."""

    def test_dependency_scope_values(self):
        """Test all dependency scopes have correct values."""
        assert DependencyScope.BUILD.value == "build"
        assert DependencyScope.RUNTIME.value == "runtime"
        assert DependencyScope.DEV.value == "dev"
        assert DependencyScope.OPTIONAL.value == "optional"


class TestProjectInfo:
    """Tests for ProjectInfo dataclass."""

    def test_create_minimal(self):
        """Test creating with minimal args."""
        project = ProjectInfo(
            name="my-project",
            path=Path("/path/to/project"),
        )
        assert project.name == "my-project"
        assert project.project_type == ProjectType.UNKNOWN
        assert project.has_roadmap is False
        assert project.dependencies == []

    def test_create_full(self):
        """Test creating with all args."""
        project = ProjectInfo(
            name="my-project",
            path=Path("/path/to/project"),
            project_type=ProjectType.PYTHON,
            has_roadmap=True,
            roadmap_path=Path("/path/to/project/ROADMAP.md"),
            config_file=Path("/path/to/project/pyproject.toml"),
            version="1.0.0",
            dependencies=["other-project"],
        )
        assert project.project_type == ProjectType.PYTHON
        assert project.has_roadmap is True
        assert project.version == "1.0.0"
        assert "other-project" in project.dependencies

    def test_to_dict(self):
        """Test serialization to dict."""
        project = ProjectInfo(
            name="my-project",
            path=Path("/path/to/project"),
            project_type=ProjectType.PYTHON,
            version="1.0.0",
        )
        data = project.to_dict()

        assert data["name"] == "my-project"
        assert data["project_type"] == "python"
        assert data["version"] == "1.0.0"

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "name": "my-project",
            "path": "/path/to/project",
            "project_type": "javascript",
            "has_roadmap": True,
            "version": "2.0.0",
        }
        project = ProjectInfo.from_dict(data)

        assert project.name == "my-project"
        assert project.project_type == ProjectType.JAVASCRIPT
        assert project.has_roadmap is True

    def test_round_trip(self):
        """Test serialization round-trip."""
        original = ProjectInfo(
            name="test-project",
            path=Path("/test"),
            project_type=ProjectType.TYPESCRIPT,
            has_roadmap=True,
            dependencies=["dep1", "dep2"],
        )
        data = original.to_dict()
        restored = ProjectInfo.from_dict(data)

        assert restored.name == original.name
        assert restored.project_type == original.project_type
        assert restored.dependencies == original.dependencies


class TestCrossProjectDependency:
    """Tests for CrossProjectDependency dataclass."""

    def test_create_minimal(self):
        """Test creating with minimal args."""
        dep = CrossProjectDependency(
            source_project="project-a",
            target_project="project-b",
        )
        assert dep.source_project == "project-a"
        assert dep.target_project == "project-b"
        assert dep.scope == DependencyScope.RUNTIME

    def test_create_full(self):
        """Test creating with all args."""
        dep = CrossProjectDependency(
            source_project="project-a",
            target_project="project-b",
            scope=DependencyScope.DEV,
            version_constraint="^1.0.0",
            import_path="@org/project-b",
        )
        assert dep.scope == DependencyScope.DEV
        assert dep.version_constraint == "^1.0.0"

    def test_to_dict(self):
        """Test serialization to dict."""
        dep = CrossProjectDependency(
            source_project="project-a",
            target_project="project-b",
            scope=DependencyScope.BUILD,
        )
        data = dep.to_dict()

        assert data["source_project"] == "project-a"
        assert data["target_project"] == "project-b"
        assert data["scope"] == "build"


class TestWorkspaceCheckpointResult:
    """Tests for WorkspaceCheckpointResult dataclass."""

    def test_success_result(self):
        """Test creating a success result."""
        from ai_infra.executor.checkpoint import CheckpointResult

        result = WorkspaceCheckpointResult(
            success=True,
            project_results={"proj1": CheckpointResult(success=True, commit_sha="abc123")},
            commit_sha="abc123",
            message="Checkpointed successfully",
        )
        assert result.success is True
        assert result.commit_sha == "abc123"
        assert "proj1" in result.project_results

    def test_to_dict(self):
        """Test serialization to dict."""
        result = WorkspaceCheckpointResult(
            success=True,
            message="OK",
        )
        data = result.to_dict()

        assert data["success"] is True
        assert data["message"] == "OK"


class TestWorkspaceRollbackResult:
    """Tests for WorkspaceRollbackResult dataclass."""

    def test_success_result(self):
        """Test creating a success result."""
        result = WorkspaceRollbackResult(
            success=True,
            message="Rolled back successfully",
        )
        assert result.success is True

    def test_to_dict(self):
        """Test serialization to dict."""
        result = WorkspaceRollbackResult(
            success=False,
            failed_projects=["proj1"],
            message="Failed",
        )
        data = result.to_dict()

        assert data["success"] is False
        assert "proj1" in data["failed_projects"]


# =============================================================================
# Test Workspace - Project Discovery
# =============================================================================


class TestWorkspaceDiscovery:
    """Tests for workspace project discovery."""

    def test_discover_python_project(self):
        """Test discovering a Python project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create a Python project
            pyproject = root / "pyproject.toml"
            pyproject.write_text("""
[project]
name = "my-python-project"
version = "1.0.0"
""")

            workspace = Workspace(root)

            assert workspace.project_count == 1
            project = workspace.get_project("my-python-project")
            assert project is not None
            assert project.project_type == ProjectType.PYTHON
            assert project.version == "1.0.0"

    def test_discover_javascript_project(self):
        """Test discovering a JavaScript project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create a JS project
            package_json = root / "package.json"
            package_json.write_text(
                json.dumps(
                    {
                        "name": "my-js-project",
                        "version": "2.0.0",
                    }
                )
            )

            workspace = Workspace(root)

            assert workspace.project_count == 1
            project = workspace.get_project("my-js-project")
            assert project is not None
            assert project.project_type == ProjectType.JAVASCRIPT
            assert project.version == "2.0.0"

    def test_discover_typescript_project(self):
        """Test discovering a TypeScript project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create a TS project
            package_json = root / "package.json"
            package_json.write_text(
                json.dumps(
                    {
                        "name": "my-ts-project",
                        "version": "1.0.0",
                        "devDependencies": {
                            "typescript": "^5.0.0",
                        },
                    }
                )
            )

            workspace = Workspace(root)

            project = workspace.get_project("my-ts-project")
            assert project is not None
            assert project.project_type == ProjectType.TYPESCRIPT

    def test_discover_project_with_roadmap(self):
        """Test discovering project with ROADMAP.md."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()

            # Create project with roadmap
            pyproject = root / "pyproject.toml"
            pyproject.write_text('[project]\nname = "project-with-roadmap"')

            roadmap = root / "ROADMAP.md"
            roadmap.write_text("# Project Roadmap\n\n- [ ] Task 1")

            workspace = Workspace(root)

            project = workspace.get_project("project-with-roadmap")
            assert project is not None
            assert project.has_roadmap is True
            # Compare resolved paths to handle symlinks
            assert project.roadmap_path is not None
            assert project.roadmap_path.resolve() == roadmap.resolve()

    def test_discover_monorepo(self):
        """Test discovering multiple projects in monorepo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create multiple projects
            (root / "frontend").mkdir()
            (root / "frontend" / "package.json").write_text(
                json.dumps(
                    {
                        "name": "frontend",
                        "version": "1.0.0",
                    }
                )
            )

            (root / "backend").mkdir()
            (root / "backend" / "pyproject.toml").write_text(
                '[project]\nname = "backend"\nversion = "1.0.0"'
            )

            workspace = Workspace(root)

            assert workspace.project_count == 2
            assert workspace.is_monorepo is True
            assert workspace.get_project("frontend") is not None
            assert workspace.get_project("backend") is not None

    def test_skip_hidden_directories(self):
        """Test that hidden directories are skipped by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create hidden project (should be skipped)
            (root / ".hidden").mkdir()
            (root / ".hidden" / "pyproject.toml").write_text('[project]\nname = "hidden"')

            # Create visible project
            (root / "visible").mkdir()
            (root / "visible" / "pyproject.toml").write_text('[project]\nname = "visible"')

            workspace = Workspace(root, include_hidden=False)

            assert workspace.project_count == 1
            assert workspace.get_project("hidden") is None
            assert workspace.get_project("visible") is not None

    def test_skip_node_modules(self):
        """Test that node_modules is skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create main project
            (root / "package.json").write_text(json.dumps({"name": "main"}))

            # Create nested package in node_modules (should be skipped)
            (root / "node_modules" / "dep").mkdir(parents=True)
            (root / "node_modules" / "dep" / "package.json").write_text(json.dumps({"name": "dep"}))

            workspace = Workspace(root)

            assert workspace.project_count == 1
            assert workspace.get_project("dep") is None


class TestWorkspaceDependencies:
    """Tests for cross-project dependency analysis."""

    def test_python_workspace_dependencies(self):
        """Test detecting Python workspace dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create shared library
            (root / "shared-lib").mkdir()
            (root / "shared-lib" / "pyproject.toml").write_text(
                '[project]\nname = "shared_lib"\nversion = "1.0.0"'
            )

            # Create project that depends on shared-lib
            (root / "my-service").mkdir()
            (root / "my-service" / "pyproject.toml").write_text("""
[project]
name = "my-service"
version = "1.0.0"
dependencies = ["shared_lib>=1.0.0"]
""")

            workspace = Workspace(root)

            deps = workspace.get_dependencies("my-service")
            assert len(deps) >= 1
            assert any(d.target_project == "shared_lib" for d in deps)

    def test_js_workspace_dependencies(self):
        """Test detecting JavaScript workspace dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create shared package
            (root / "shared").mkdir()
            (root / "shared" / "package.json").write_text(
                json.dumps(
                    {
                        "name": "shared",
                        "version": "1.0.0",
                    }
                )
            )

            # Create app that depends on shared
            (root / "app").mkdir()
            (root / "app" / "package.json").write_text(
                json.dumps(
                    {
                        "name": "app",
                        "version": "1.0.0",
                        "dependencies": {
                            "shared": "workspace:*",
                        },
                    }
                )
            )

            workspace = Workspace(root)

            deps = workspace.get_dependencies("app")
            assert len(deps) == 1
            assert deps[0].target_project == "shared"

    def test_get_dependents(self):
        """Test getting projects that depend on a project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create core package
            (root / "core").mkdir()
            (root / "core" / "package.json").write_text(
                json.dumps(
                    {
                        "name": "core",
                    }
                )
            )

            # Create multiple apps depending on core
            for app_name in ["app1", "app2"]:
                (root / app_name).mkdir()
                (root / app_name / "package.json").write_text(
                    json.dumps(
                        {
                            "name": app_name,
                            "dependencies": {"core": "1.0.0"},
                        }
                    )
                )

            workspace = Workspace(root)

            dependents = workspace.get_dependents("core")
            assert len(dependents) == 2

    def test_dependency_order(self):
        """Test getting projects in dependency order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create dependency chain: app -> service -> core
            (root / "core").mkdir()
            (root / "core" / "package.json").write_text(json.dumps({"name": "core"}))

            (root / "service").mkdir()
            (root / "service" / "package.json").write_text(
                json.dumps(
                    {
                        "name": "service",
                        "dependencies": {"core": "1.0.0"},
                    }
                )
            )

            (root / "app").mkdir()
            (root / "app" / "package.json").write_text(
                json.dumps(
                    {
                        "name": "app",
                        "dependencies": {"service": "1.0.0"},
                    }
                )
            )

            workspace = Workspace(root)
            order = workspace.get_dependency_order()

            # Core should come before service, service before app
            core_idx = order.index("core")
            service_idx = order.index("service")
            app_idx = order.index("app")

            assert core_idx < service_idx
            assert service_idx < app_idx


class TestWorkspaceQueries:
    """Tests for workspace query methods."""

    def test_get_project_by_path(self):
        """Test getting a project by its path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            (root / "project").mkdir()
            (root / "project" / "pyproject.toml").write_text('[project]\nname = "project"')

            workspace = Workspace(root)
            project = workspace.get_project_by_path(root / "project")

            assert project is not None
            assert project.name == "project"

    def test_projects_with_roadmaps(self):
        """Test filtering projects with roadmaps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Project with roadmap
            (root / "with-roadmap").mkdir()
            (root / "with-roadmap" / "pyproject.toml").write_text(
                '[project]\nname = "with-roadmap"'
            )
            (root / "with-roadmap" / "ROADMAP.md").write_text("# Roadmap")

            # Project without roadmap
            (root / "without-roadmap").mkdir()
            (root / "without-roadmap" / "pyproject.toml").write_text(
                '[project]\nname = "without-roadmap"'
            )

            workspace = Workspace(root)
            with_roadmaps = workspace.projects_with_roadmaps()

            assert len(with_roadmaps) == 1
            assert with_roadmaps[0].name == "with-roadmap"

    def test_projects_of_type(self):
        """Test filtering projects by type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            (root / "py-project").mkdir()
            (root / "py-project" / "pyproject.toml").write_text('[project]\nname = "py-project"')

            (root / "js-project").mkdir()
            (root / "js-project" / "package.json").write_text(json.dumps({"name": "js-project"}))

            workspace = Workspace(root)

            python_projects = workspace.projects_of_type(ProjectType.PYTHON)
            js_projects = workspace.projects_of_type(ProjectType.JAVASCRIPT)

            assert len(python_projects) == 1
            assert len(js_projects) == 1

    def test_find_project_for_file(self):
        """Test finding which project a file belongs to."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            (root / "project-a").mkdir()
            (root / "project-a" / "pyproject.toml").write_text('[project]\nname = "project-a"')
            (root / "project-a" / "src").mkdir()
            (root / "project-a" / "src" / "main.py").write_text("# main")

            workspace = Workspace(root)

            project = workspace.find_project_for_file(root / "project-a" / "src" / "main.py")
            assert project is not None
            assert project.name == "project-a"


class TestWorkspaceProperties:
    """Tests for workspace properties."""

    def test_is_monorepo_multiple_projects(self):
        """Test monorepo detection with multiple projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            for name in ["project1", "project2"]:
                (root / name).mkdir()
                (root / name / "pyproject.toml").write_text(f'[project]\nname = "{name}"')

            workspace = Workspace(root)
            assert workspace.is_monorepo is True

    def test_is_monorepo_workspace_file(self):
        """Test monorepo detection with workspace config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Single project with pnpm workspace file
            (root / "package.json").write_text(json.dumps({"name": "mono"}))
            (root / "pnpm-workspace.yaml").write_text("packages:\n  - packages/*")

            workspace = Workspace(root)
            assert workspace.is_monorepo is True

    def test_is_git_repo(self):
        """Test git repo detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Not a git repo
            workspace = Workspace(root, auto_discover=False)
            assert workspace.is_git_repo is False

            # Create .git directory
            (root / ".git").mkdir()
            workspace2 = Workspace(root, auto_discover=False)
            assert workspace2.is_git_repo is True


class TestWorkspaceSerialization:
    """Tests for workspace serialization."""

    def test_to_dict(self):
        """Test serializing workspace to dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            (root / "project").mkdir()
            (root / "project" / "pyproject.toml").write_text('[project]\nname = "project"')

            workspace = Workspace(root)
            data = workspace.to_dict()

            assert "root" in data
            assert "projects" in data
            assert "project" in data["projects"]

    def test_summary(self):
        """Test human-readable summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            (root / "my-project").mkdir()
            (root / "my-project" / "pyproject.toml").write_text('[project]\nname = "my-project"')
            (root / "my-project" / "ROADMAP.md").write_text("# Roadmap")

            workspace = Workspace(root)
            summary = workspace.summary()

            assert "Workspace:" in summary
            assert "my-project" in summary
            assert "python" in summary
            assert "ROADMAP" in summary


# =============================================================================
# Test Coordinated Checkpointing
# =============================================================================


class TestWorkspaceCheckpointing:
    """Tests for coordinated checkpointing."""

    def test_checkpoint_all_no_git(self):
        """Test checkpoint when not a git repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            workspace = Workspace(root, auto_discover=False)
            result = workspace.checkpoint_all(
                task_id="1.1.1",
                task_title="Test task",
                affected_projects=["project"],
            )

            # Should succeed but do nothing (no git repo)
            assert result.success is True

    def test_checkpoint_all_no_affected(self):
        """Test checkpoint with no affected projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            workspace = Workspace(root, auto_discover=False)
            result = workspace.checkpoint_all(
                task_id="1.1.1",
                task_title="Test task",
                affected_projects=[],
            )

            assert result.success is True
            assert "No affected projects" in result.message

    def test_checkpoint_auto_detect_affected(self):
        """Test auto-detecting affected projects from files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            (root / "project").mkdir()
            (root / "project" / "pyproject.toml").write_text('[project]\nname = "project"')
            (root / "project" / "main.py").write_text("# code")

            workspace = Workspace(root)

            # Auto-detect affected project
            result = workspace.checkpoint_all(
                task_id="1.1.1",
                task_title="Test task",
                files_modified=[str(root / "project" / "main.py")],
            )

            # Should identify "project" as affected
            assert "project" in result.project_results or result.success

    def test_rollback_all_no_git(self):
        """Test rollback when not a git repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            workspace = Workspace(root, auto_discover=False)
            result = workspace.rollback_all(
                task_id="1.1.1",
                affected_projects=["project"],
            )

            # No git repo, so no actual rollback
            assert result.success is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestWorkspaceIntegration:
    """Integration tests for workspace functionality."""

    def test_full_monorepo_workflow(self):
        """Test complete workflow with a monorepo structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create monorepo structure
            # - shared (no deps)
            # - api (depends on shared)
            # - web (depends on shared)

            (root / "shared").mkdir()
            (root / "shared" / "pyproject.toml").write_text(
                '[project]\nname = "shared"\nversion = "1.0.0"'
            )
            (root / "shared" / "ROADMAP.md").write_text("# Shared Roadmap")

            (root / "api").mkdir()
            (root / "api" / "pyproject.toml").write_text("""
[project]
name = "api"
version = "1.0.0"
dependencies = ["shared"]
""")
            (root / "api" / "ROADMAP.md").write_text("# API Roadmap")

            (root / "web").mkdir()
            (root / "web" / "package.json").write_text(
                json.dumps(
                    {
                        "name": "web",
                        "version": "1.0.0",
                    }
                )
            )

            workspace = Workspace(root)

            # Verify discovery
            assert workspace.project_count == 3
            assert workspace.is_monorepo is True

            # Verify roadmaps
            roadmap_projects = workspace.projects_with_roadmaps()
            assert len(roadmap_projects) == 2

            # Verify dependencies
            api_deps = workspace.get_dependencies("api")
            assert len(api_deps) == 1
            assert api_deps[0].target_project == "shared"

            # Verify dependency order
            order = workspace.get_dependency_order()
            shared_idx = order.index("shared")
            api_idx = order.index("api")
            assert shared_idx < api_idx

            # Verify summary
            summary = workspace.summary()
            assert "3" in summary or "Projects: 3" in summary

    def test_mixed_language_monorepo(self):
        """Test monorepo with multiple language types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Python backend
            (root / "backend").mkdir()
            (root / "backend" / "pyproject.toml").write_text('[project]\nname = "backend"')

            # TypeScript frontend
            (root / "frontend").mkdir()
            (root / "frontend" / "package.json").write_text(
                json.dumps(
                    {
                        "name": "frontend",
                        "devDependencies": {"typescript": "^5.0"},
                    }
                )
            )
            (root / "frontend" / "tsconfig.json").write_text("{}")

            # Rust service
            (root / "service").mkdir()
            (root / "service" / "Cargo.toml").write_text(
                '[package]\nname = "service"\nversion = "0.1.0"'
            )

            workspace = Workspace(root)

            assert workspace.project_count == 3

            python_projects = workspace.projects_of_type(ProjectType.PYTHON)
            ts_projects = workspace.projects_of_type(ProjectType.TYPESCRIPT)
            rust_projects = workspace.projects_of_type(ProjectType.RUST)

            assert len(python_projects) == 1
            assert len(ts_projects) == 1
            assert len(rust_projects) == 1
