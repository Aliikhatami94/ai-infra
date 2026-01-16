"""Project analyzer for roadmap generation.

This module provides project analysis capabilities to understand the context
before generating a ROADMAP.md from natural language prompts.

Phase 3.1.2 of EXECUTOR_1.md - Intelligent Task Handling.

Usage:
    from ai_infra.executor.project_analyzer import ProjectAnalyzer, ProjectInfo

    analyzer = ProjectAnalyzer()
    info = await analyzer.analyze(Path("/path/to/project"))
    print(f"Language: {info.language}, Framework: {info.framework}")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ProjectAnalyzer",
    "ProjectInfo",
]


# =============================================================================
# Project Info Dataclass
# =============================================================================


@dataclass
class ProjectInfo:
    """Information about a project's structure and configuration.

    Attributes:
        language: Primary programming language (python, javascript, etc.).
        framework: Detected framework (fastapi, react, etc.) or None.
        build_system: Build system in use (poetry, npm, cargo, make).
        structure_summary: Brief description of project structure.
        file_list: List of key files in the project.
        dependencies: List of key dependencies.
        readme_summary: Summary of README.md content.
        config_files: List of configuration files found.
        test_framework: Detected test framework if any.
        entry_points: Detected entry points (main.py, index.ts, etc.).
    """

    language: str = "unknown"
    framework: str | None = None
    build_system: str = "unknown"
    structure_summary: str = ""
    file_list: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    readme_summary: str = ""
    config_files: list[str] = field(default_factory=list)
    test_framework: str | None = None
    entry_points: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "language": self.language,
            "framework": self.framework,
            "build_system": self.build_system,
            "structure_summary": self.structure_summary,
            "file_list": self.file_list,
            "dependencies": self.dependencies,
            "readme_summary": self.readme_summary,
            "config_files": self.config_files,
            "test_framework": self.test_framework,
            "entry_points": self.entry_points,
        }

    def to_context_string(self) -> str:
        """Convert to a string suitable for LLM context."""
        lines = [
            f"Language: {self.language}",
            f"Framework: {self.framework or 'None detected'}",
            f"Build System: {self.build_system}",
            f"Structure: {self.structure_summary}",
        ]
        if self.entry_points:
            lines.append(f"Entry Points: {', '.join(self.entry_points[:5])}")
        if self.dependencies:
            lines.append(f"Key Dependencies: {', '.join(self.dependencies[:10])}")
        if self.test_framework:
            lines.append(f"Test Framework: {self.test_framework}")
        if self.readme_summary:
            lines.append(f"README Summary: {self.readme_summary[:200]}")
        return "\n".join(lines)


# =============================================================================
# Language and Framework Detection
# =============================================================================


# Language indicators: language -> list of indicator files
LANGUAGE_INDICATORS: dict[str, list[str]] = {
    "python": ["pyproject.toml", "setup.py", "requirements.txt", "Pipfile"],
    "javascript": ["package.json"],
    "typescript": ["tsconfig.json"],
    "rust": ["Cargo.toml"],
    "go": ["go.mod", "go.sum"],
    "java": ["pom.xml", "build.gradle", "build.gradle.kts"],
    "csharp": ["*.csproj", "*.sln"],
    "ruby": ["Gemfile", "*.gemspec"],
    "php": ["composer.json"],
    "swift": ["Package.swift"],
    "kotlin": ["build.gradle.kts"],
}

# Framework indicators: framework -> (language, indicator_patterns)
FRAMEWORK_INDICATORS: dict[str, tuple[str, list[str]]] = {
    # Python frameworks
    "fastapi": ("python", ["fastapi"]),
    "django": ("python", ["django"]),
    "flask": ("python", ["flask"]),
    "streamlit": ("python", ["streamlit"]),
    "pytest": ("python", ["pytest"]),
    # JavaScript/TypeScript frameworks
    "react": ("javascript", ["react", "react-dom"]),
    "next": ("javascript", ["next"]),
    "vue": ("javascript", ["vue"]),
    "angular": ("javascript", ["@angular/core"]),
    "express": ("javascript", ["express"]),
    "nest": ("javascript", ["@nestjs/core"]),
    # Rust frameworks
    "actix": ("rust", ["actix-web"]),
    "axum": ("rust", ["axum"]),
    "rocket": ("rust", ["rocket"]),
    # Go frameworks
    "gin": ("go", ["github.com/gin-gonic/gin"]),
    "fiber": ("go", ["github.com/gofiber/fiber"]),
    "echo": ("go", ["github.com/labstack/echo"]),
}

# Build system indicators
BUILD_SYSTEM_INDICATORS: dict[str, str] = {
    "pyproject.toml": "poetry",
    "setup.py": "setuptools",
    "requirements.txt": "pip",
    "Pipfile": "pipenv",
    "package.json": "npm",
    "pnpm-lock.yaml": "pnpm",
    "yarn.lock": "yarn",
    "Cargo.toml": "cargo",
    "go.mod": "go",
    "pom.xml": "maven",
    "build.gradle": "gradle",
    "Makefile": "make",
    "CMakeLists.txt": "cmake",
}

# Test framework indicators
TEST_FRAMEWORK_INDICATORS: dict[str, tuple[str, list[str]]] = {
    "pytest": ("python", ["pytest", "pytest-asyncio"]),
    "unittest": ("python", ["unittest"]),
    "jest": ("javascript", ["jest"]),
    "vitest": ("javascript", ["vitest"]),
    "mocha": ("javascript", ["mocha"]),
    "rspec": ("ruby", ["rspec"]),
    "junit": ("java", ["junit"]),
    "cargo-test": ("rust", []),  # Built-in
}


# =============================================================================
# Project Analyzer Class
# =============================================================================


class ProjectAnalyzer:
    """Analyze project structure to understand context.

    Provides methods to detect language, framework, dependencies,
    and project structure for use in roadmap generation.

    Example:
        >>> analyzer = ProjectAnalyzer()
        >>> info = await analyzer.analyze(Path("/path/to/project"))
        >>> print(info.language)
        python
        >>> print(info.framework)
        fastapi
    """

    def __init__(
        self,
        max_files: int = 100,
        max_depth: int = 4,
        max_readme_chars: int = 500,
    ) -> None:
        """Initialize project analyzer.

        Args:
            max_files: Maximum number of files to list.
            max_depth: Maximum directory depth to scan.
            max_readme_chars: Maximum characters to read from README.
        """
        self.max_files = max_files
        self.max_depth = max_depth
        self.max_readme_chars = max_readme_chars

    async def analyze(self, workspace: Path) -> ProjectInfo:
        """Analyze project structure and configuration.

        Args:
            workspace: Path to the project root directory.

        Returns:
            ProjectInfo with detected project characteristics.
        """
        workspace = Path(workspace).resolve()

        if not workspace.exists():
            return ProjectInfo(structure_summary="Directory does not exist")

        if not workspace.is_dir():
            return ProjectInfo(structure_summary="Path is not a directory")

        # Detect language first
        language = self._detect_language(workspace)

        # Detect build system
        build_system = self._detect_build_system(workspace)

        # Parse dependencies
        dependencies = self._parse_dependencies(workspace, language)

        # Detect framework from dependencies
        framework = self._detect_framework(dependencies, language)

        # Detect test framework
        test_framework = self._detect_test_framework(dependencies, language)

        # List key files
        file_list = self._list_key_files(workspace)

        # Find config files
        config_files = self._find_config_files(workspace)

        # Analyze structure
        structure = self._analyze_structure(workspace, file_list)

        # Find entry points
        entry_points = self._find_entry_points(workspace, language)

        # Summarize README
        readme_summary = self._summarize_readme(workspace)

        return ProjectInfo(
            language=language,
            framework=framework,
            build_system=build_system,
            structure_summary=structure,
            file_list=file_list,
            dependencies=dependencies,
            readme_summary=readme_summary,
            config_files=config_files,
            test_framework=test_framework,
            entry_points=entry_points,
        )

    def _detect_language(self, workspace: Path) -> str:
        """Detect primary programming language."""
        for language, indicators in LANGUAGE_INDICATORS.items():
            for indicator in indicators:
                if "*" in indicator:
                    # Glob pattern
                    if list(workspace.glob(indicator)):
                        return language
                elif (workspace / indicator).exists():
                    return language
        return "unknown"

    def _detect_build_system(self, workspace: Path) -> str:
        """Detect build system from project files."""
        # Check in priority order
        priority_order = [
            "pnpm-lock.yaml",
            "yarn.lock",
            "pyproject.toml",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "pom.xml",
            "build.gradle",
            "setup.py",
            "requirements.txt",
            "Pipfile",
            "Makefile",
            "CMakeLists.txt",
        ]
        for indicator in priority_order:
            if (workspace / indicator).exists():
                return BUILD_SYSTEM_INDICATORS.get(indicator, "unknown")
        return "unknown"

    def _parse_dependencies(self, workspace: Path, language: str) -> list[str]:
        """Parse dependencies from project configuration."""
        deps: list[str] = []

        if language == "python":
            deps.extend(self._parse_python_deps(workspace))
        elif language in ("javascript", "typescript"):
            deps.extend(self._parse_node_deps(workspace))
        elif language == "rust":
            deps.extend(self._parse_cargo_deps(workspace))
        elif language == "go":
            deps.extend(self._parse_go_deps(workspace))

        return deps

    def _parse_python_deps(self, workspace: Path) -> list[str]:
        """Parse Python dependencies from pyproject.toml or requirements.txt."""
        deps: list[str] = []

        # Try pyproject.toml first
        pyproject = workspace / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomllib

                content = pyproject.read_text()
                data = tomllib.loads(content)

                # Poetry dependencies
                if "tool" in data and "poetry" in data["tool"]:
                    poetry_deps = data["tool"]["poetry"].get("dependencies", {})
                    deps.extend(k for k in poetry_deps if k != "python" and not k.startswith("^"))

                # PEP 621 dependencies
                if "project" in data:
                    project_deps = data["project"].get("dependencies", [])
                    for dep in project_deps:
                        # Extract package name from requirement specifier
                        match = re.match(r"^([a-zA-Z0-9_-]+)", dep)
                        if match:
                            deps.append(match.group(1))

            except Exception:
                pass

        # Fall back to requirements.txt
        requirements = workspace / "requirements.txt"
        if requirements.exists() and not deps:
            try:
                for line in requirements.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and not line.startswith("-"):
                        match = re.match(r"^([a-zA-Z0-9_-]+)", line)
                        if match:
                            deps.append(match.group(1))
            except Exception:
                pass

        return deps[:50]  # Limit to top 50

    def _parse_node_deps(self, workspace: Path) -> list[str]:
        """Parse Node.js dependencies from package.json."""
        deps: list[str] = []
        package_json = workspace / "package.json"

        if package_json.exists():
            try:
                data = json.loads(package_json.read_text())
                deps.extend(data.get("dependencies", {}).keys())
                deps.extend(data.get("devDependencies", {}).keys())
            except Exception:
                pass

        return deps[:50]

    def _parse_cargo_deps(self, workspace: Path) -> list[str]:
        """Parse Rust dependencies from Cargo.toml."""
        deps: list[str] = []
        cargo_toml = workspace / "Cargo.toml"

        if cargo_toml.exists():
            try:
                import tomllib

                data = tomllib.loads(cargo_toml.read_text())
                deps.extend(data.get("dependencies", {}).keys())
                deps.extend(data.get("dev-dependencies", {}).keys())
            except Exception:
                pass

        return deps[:50]

    def _parse_go_deps(self, workspace: Path) -> list[str]:
        """Parse Go dependencies from go.mod."""
        deps: list[str] = []
        go_mod = workspace / "go.mod"

        if go_mod.exists():
            try:
                content = go_mod.read_text()
                # Match require blocks
                for match in re.finditer(r"require\s+\(([^)]+)\)", content):
                    block = match.group(1)
                    for line in block.splitlines():
                        parts = line.strip().split()
                        if parts:
                            deps.append(parts[0])
                # Match single requires
                for match in re.finditer(r"require\s+(\S+)\s+v", content):
                    deps.append(match.group(1))
            except Exception:
                pass

        return deps[:50]

    def _detect_framework(self, dependencies: list[str], language: str) -> str | None:
        """Detect framework from dependencies."""
        deps_lower = {d.lower() for d in dependencies}

        for framework, (lang, indicators) in FRAMEWORK_INDICATORS.items():
            if lang == language:
                for indicator in indicators:
                    if indicator.lower() in deps_lower:
                        return framework

        return None

    def _detect_test_framework(self, dependencies: list[str], language: str) -> str | None:
        """Detect test framework from dependencies."""
        deps_lower = {d.lower() for d in dependencies}

        for framework, (lang, indicators) in TEST_FRAMEWORK_INDICATORS.items():
            if lang == language:
                # Rust has built-in testing
                if framework == "cargo-test" and language == "rust":
                    return "cargo-test"
                for indicator in indicators:
                    if indicator.lower() in deps_lower:
                        return framework

        return None

    def _list_key_files(self, workspace: Path) -> list[str]:
        """List key files in the project."""
        files: list[str] = []
        ignore_dirs = {
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            "target",
            "dist",
            "build",
            ".next",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "htmlcov",
        }
        ignore_extensions = {".pyc", ".pyo", ".so", ".o", ".a", ".dylib"}

        def _walk(path: Path, depth: int = 0) -> None:
            if depth > self.max_depth or len(files) >= self.max_files:
                return

            try:
                for item in sorted(path.iterdir()):
                    if len(files) >= self.max_files:
                        return

                    if item.name.startswith(".") and item.name not in (
                        ".github",
                        ".vscode",
                    ):
                        continue

                    if item.is_dir():
                        if item.name not in ignore_dirs:
                            _walk(item, depth + 1)
                    elif item.is_file():
                        if item.suffix not in ignore_extensions:
                            rel_path = str(item.relative_to(workspace))
                            files.append(rel_path)
            except PermissionError:
                pass

        _walk(workspace)
        return files

    def _find_config_files(self, workspace: Path) -> list[str]:
        """Find configuration files in the project."""
        config_patterns = [
            "*.toml",
            "*.yaml",
            "*.yml",
            "*.json",
            "*.ini",
            "*.cfg",
            ".env*",
            "Makefile",
            "Dockerfile*",
            "docker-compose*.yml",
            "*.config.js",
            "*.config.ts",
            "*.config.mjs",
        ]
        configs: list[str] = []

        for pattern in config_patterns:
            for path in workspace.glob(pattern):
                if path.is_file():
                    configs.append(path.name)

        return sorted(set(configs))[:20]

    def _analyze_structure(self, workspace: Path, files: list[str]) -> str:
        """Analyze and summarize project structure."""
        # Check for common patterns
        has_src = (workspace / "src").is_dir()
        has_tests = (workspace / "tests").is_dir() or (workspace / "test").is_dir()
        has_docs = (workspace / "docs").is_dir()
        has_examples = (workspace / "examples").is_dir()

        # Check for monorepo patterns
        has_packages = (workspace / "packages").is_dir()
        has_apps = (workspace / "apps").is_dir()

        if has_packages or has_apps:
            return "monorepo"
        elif has_src and has_tests:
            return "standard (src/ + tests/)"
        elif has_src:
            return "src-based"
        elif has_tests:
            return "flat with tests/"
        else:
            return "flat"

    def _find_entry_points(self, workspace: Path, language: str) -> list[str]:
        """Find likely entry points for the project."""
        entry_points: list[str] = []

        if language == "python":
            candidates = [
                "main.py",
                "app.py",
                "__main__.py",
                "cli.py",
                "src/main.py",
                "src/app.py",
            ]
        elif language in ("javascript", "typescript"):
            candidates = [
                "index.js",
                "index.ts",
                "main.js",
                "main.ts",
                "app.js",
                "app.ts",
                "src/index.js",
                "src/index.ts",
                "src/main.js",
                "src/main.ts",
            ]
        elif language == "rust":
            candidates = ["src/main.rs", "src/lib.rs"]
        elif language == "go":
            candidates = ["main.go", "cmd/main.go"]
        else:
            candidates = []

        for candidate in candidates:
            if (workspace / candidate).exists():
                entry_points.append(candidate)

        return entry_points

    def _summarize_readme(self, workspace: Path) -> str:
        """Summarize README.md content."""
        readme_names = ["README.md", "README.rst", "README.txt", "README"]

        for name in readme_names:
            readme = workspace / name
            if readme.exists():
                try:
                    content = readme.read_text(errors="ignore")

                    # Extract first meaningful paragraph
                    lines = content.splitlines()
                    summary_lines: list[str] = []

                    in_content = False
                    for line in lines:
                        stripped = line.strip()

                        # Skip headers and badges
                        if stripped.startswith("#"):
                            in_content = True
                            continue
                        if stripped.startswith("![") or stripped.startswith("[!"):
                            continue

                        if in_content and stripped:
                            summary_lines.append(stripped)
                            if len(" ".join(summary_lines)) > self.max_readme_chars:
                                break

                    summary = " ".join(summary_lines)
                    return summary[: self.max_readme_chars]

                except Exception:
                    pass

        return ""
