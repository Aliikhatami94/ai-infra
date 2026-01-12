"""Task verification for the Executor module.

Provides multi-level verification to determine if a task succeeded:
1. File existence checks
2. Python syntax validation (ast.parse)
3. Import resolution checks
4. Test execution (pytest)
5. Type checking (mypy)
"""

from __future__ import annotations

import ast
import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ai_infra.executor.models import Task
from ai_infra.logging import get_logger

logger = get_logger("executor.verifier")


class ProjectType(Enum):
    """Detected project type for language-agnostic verification."""

    PYTHON = "python"
    NODEJS = "nodejs"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"
    MAKEFILE = "makefile"
    UNKNOWN = "unknown"


class CheckLevel(Enum):
    """Verification check levels, from fast to thorough."""

    FILES = "files"  # Level 1: File existence
    SYNTAX = "syntax"  # Level 2: Python parses
    IMPORTS = "imports"  # Level 3: Imports resolve (static)
    RUNTIME = "runtime"  # Level 4: Actually import modules (catches circular imports)
    TESTS = "tests"  # Level 5: Tests pass
    TYPES = "types"  # Level 6: Type checking


class CheckStatus(Enum):
    """Status of a single verification check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


def detect_project_type(project_root: Path) -> ProjectType:
    """Detect project type from indicator files.

    Checks for language-specific files to determine what test runner to use.

    Args:
        project_root: Root directory of the project

    Returns:
        The detected ProjectType

    Example:
        >>> detect_project_type(Path("/my/python/project"))
        ProjectType.PYTHON
    """
    project_root = Path(project_root)

    # Python indicators
    if (
        (project_root / "pyproject.toml").exists()
        or (project_root / "setup.py").exists()
        or (project_root / "requirements.txt").exists()
    ):
        return ProjectType.PYTHON

    # Node.js / TypeScript indicators
    if (project_root / "package.json").exists():
        if (project_root / "tsconfig.json").exists():
            return ProjectType.TYPESCRIPT
        return ProjectType.NODEJS

    # Rust indicators
    if (project_root / "Cargo.toml").exists():
        return ProjectType.RUST

    # Go indicators
    if (project_root / "go.mod").exists():
        return ProjectType.GO

    # Makefile with test target
    makefile = project_root / "Makefile"
    if makefile.exists():
        try:
            content = makefile.read_text()
            if "test:" in content or "test :" in content:
                return ProjectType.MAKEFILE
        except Exception:
            pass

    return ProjectType.UNKNOWN


def get_test_command(project_type: ProjectType, project_root: Path) -> list[str] | None:
    """Get the test command for a project type.

    Args:
        project_type: The detected project type
        project_root: Root directory of the project

    Returns:
        List of command arguments, or None if no test runner available

    Example:
        >>> get_test_command(ProjectType.PYTHON, Path("/my/project"))
        ['/usr/bin/python', '-m', 'pytest', '-q', '--tb=short']
    """
    project_root = Path(project_root)

    match project_type:
        case ProjectType.PYTHON:
            return [sys.executable, "-m", "pytest", "-q", "--tb=short"]

        case ProjectType.NODEJS:
            # Check for test script in package.json
            pkg_json = project_root / "package.json"
            if pkg_json.exists():
                try:
                    pkg = json.loads(pkg_json.read_text())
                    if "scripts" in pkg and "test" in pkg["scripts"]:
                        # Use npm test if script exists
                        return ["npm", "test"]
                except Exception:
                    pass
            # Fallback to node --test
            tests_dir = project_root / "tests"
            if tests_dir.exists():
                return ["node", "--test", str(tests_dir)]
            return None

        case ProjectType.TYPESCRIPT:
            return ["npm", "test"]

        case ProjectType.RUST:
            return ["cargo", "test"]

        case ProjectType.GO:
            return ["go", "test", "./..."]

        case ProjectType.MAKEFILE:
            return ["make", "test"]

        case ProjectType.UNKNOWN:
            return None

    return None


@dataclass
class CheckResult:
    """Result of a single verification check.

    Attributes:
        name: Descriptive name (e.g., "file_exists:src/main.py")
        level: The verification level
        status: Whether the check passed, failed, or was skipped
        message: Human-readable description
        error: Error message if the check failed
        duration_ms: How long the check took
        metadata: Additional check-specific data
    """

    name: str
    level: CheckLevel
    status: CheckStatus
    message: str = ""
    error: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether this check passed."""
        return self.status == CheckStatus.PASSED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "level": self.level.value,
            "status": self.status.value,
            "message": self.message,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class VerificationResult:
    """Result of all verification checks for a task.

    Attributes:
        task_id: The task that was verified
        checks: All individual check results
        overall: Whether all checks passed
        levels_run: Which verification levels were executed
        total_duration_ms: Total verification time
    """

    task_id: str
    checks: list[CheckResult] = field(default_factory=list)
    levels_run: list[CheckLevel] = field(default_factory=list)
    total_duration_ms: float = 0.0

    @property
    def overall(self) -> bool:
        """Whether all checks passed."""
        return all(c.passed or c.status == CheckStatus.SKIPPED for c in self.checks)

    @property
    def passed_count(self) -> int:
        """Number of checks that passed."""
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed_count(self) -> int:
        """Number of checks that failed."""
        return sum(1 for c in self.checks if c.status == CheckStatus.FAILED)

    @property
    def error_count(self) -> int:
        """Number of checks that errored."""
        return sum(1 for c in self.checks if c.status == CheckStatus.ERROR)

    def get_failures(self) -> list[CheckResult]:
        """Get all failed checks."""
        return [c for c in self.checks if c.status == CheckStatus.FAILED]

    def get_errors(self) -> list[CheckResult]:
        """Get all errored checks."""
        return [c for c in self.checks if c.status == CheckStatus.ERROR]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "checks": [c.to_dict() for c in self.checks],
            "overall": self.overall,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "error_count": self.error_count,
            "levels_run": [lvl.value for lvl in self.levels_run],
            "total_duration_ms": self.total_duration_ms,
        }

    def summary(self) -> str:
        """Get a human-readable summary."""
        status = "PASSED" if self.overall else "FAILED"
        return (
            f"Verification {status}: {self.passed_count} passed, "
            f"{self.failed_count} failed, {self.error_count} errors "
            f"({self.total_duration_ms:.0f}ms)"
        )


class TaskVerifier:
    """Verify task completion at multiple levels.

    Verification levels (in order):
    1. FILES: Check expected files exist/changed
    2. SYNTAX: Check all Python files parse without syntax errors
    3. IMPORTS: Check imports can be resolved
    4. TESTS: Run pytest if tests directory exists
    5. TYPES: Run mypy if pyproject.toml exists

    Example:
        verifier = TaskVerifier(workspace=Path("/path/to/project"))
        result = await verifier.verify(task)
        if result.overall:
            print("Task completed successfully!")
        else:
            for failure in result.get_failures():
                print(f"Failed: {failure.name} - {failure.error}")
    """

    # Default file patterns to check for syntax
    DEFAULT_PYTHON_PATTERNS = ("*.py",)

    # Directories to exclude from syntax checking
    DEFAULT_EXCLUDE_DIRS = frozenset(
        {
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "env",
            ".env",
            "node_modules",
            "dist",
            "build",
            ".pytest_cache",
            ".mypy_cache",
            "htmlcov",
            ".ruff_cache",
            ".tox",
            "eggs",
            "*.egg-info",
        }
    )

    def __init__(
        self,
        workspace: Path,
        *,
        exclude_dirs: frozenset[str] | None = None,
        pytest_args: list[str] | None = None,
        mypy_args: list[str] | None = None,
        timeout_seconds: float = 300.0,
    ):
        """Initialize the verifier.

        Args:
            workspace: Root directory of the project to verify
            exclude_dirs: Directories to exclude from checks
            pytest_args: Additional arguments for pytest
            mypy_args: Additional arguments for mypy
            timeout_seconds: Timeout for external commands
        """
        self.workspace = Path(workspace).resolve()
        self.exclude_dirs = exclude_dirs or self.DEFAULT_EXCLUDE_DIRS
        self.pytest_args = pytest_args or ["-q", "--tb=short"]
        self.mypy_args = mypy_args or ["--ignore-missing-imports"]
        self.timeout_seconds = timeout_seconds

    def _should_skip_path(self, path: Path) -> bool:
        """Check if a path should be skipped based on exclude patterns."""
        parts = path.parts
        for part in parts:
            if part in self.exclude_dirs:
                return True
            # Handle glob patterns like *.egg-info
            for pattern in self.exclude_dirs:
                if "*" in pattern:
                    import fnmatch

                    if fnmatch.fnmatch(part, pattern):
                        return True
        return False

    async def verify(
        self,
        task: Task,
        *,
        levels: list[CheckLevel] | None = None,
        stop_on_failure: bool = False,
    ) -> VerificationResult:
        """Verify a task at specified levels.

        Args:
            task: The task to verify
            levels: Which levels to check (default: all)
            stop_on_failure: Stop at first failing level

        Returns:
            VerificationResult with all check results
        """
        import time

        start_time = time.perf_counter()

        if levels is None:
            levels = list(CheckLevel)

        result = VerificationResult(task_id=task.id, levels_run=levels)
        checks: list[CheckResult] = []

        for level in levels:
            level_checks = await self._run_level(task, level)
            checks.extend(level_checks)

            # Check if we should stop early
            if stop_on_failure and any(c.status == CheckStatus.FAILED for c in level_checks):
                logger.info(f"Stopping verification at level {level.value} due to failures")
                break

        result.checks = checks
        result.total_duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(f"Verification complete: {result.summary()}")
        return result

    async def _run_level(self, task: Task, level: CheckLevel) -> list[CheckResult]:
        """Run all checks for a specific level."""
        if level == CheckLevel.FILES:
            return await self._check_files(task)
        elif level == CheckLevel.SYNTAX:
            return await self._check_syntax()
        elif level == CheckLevel.IMPORTS:
            return await self._check_imports()
        elif level == CheckLevel.RUNTIME:
            return await self._check_runtime()
        elif level == CheckLevel.TESTS:
            return await self._check_tests()
        elif level == CheckLevel.TYPES:
            return await self._check_types()
        else:
            return []

    async def _check_files(self, task: Task) -> list[CheckResult]:
        """Level 1: Check that expected files exist."""
        checks: list[CheckResult] = []

        # Check file_hints from the task
        expected_files = getattr(task, "expected_files", None) or task.file_hints

        if not expected_files:
            checks.append(
                CheckResult(
                    name="file_check",
                    level=CheckLevel.FILES,
                    status=CheckStatus.SKIPPED,
                    message="No expected files specified",
                )
            )
            return checks

        for file_path in expected_files:
            full_path = self.workspace / file_path
            exists = full_path.exists()

            checks.append(
                CheckResult(
                    name=f"file_exists:{file_path}",
                    level=CheckLevel.FILES,
                    status=CheckStatus.PASSED if exists else CheckStatus.FAILED,
                    message=f"File {'exists' if exists else 'not found'}: {file_path}",
                    error=None if exists else f"Expected file not found: {file_path}",
                    metadata={"path": str(file_path), "absolute_path": str(full_path)},
                )
            )

        return checks

    async def _check_syntax(self) -> list[CheckResult]:
        """Level 2: Check that all Python files parse without syntax errors."""
        checks: list[CheckResult] = []

        def _scan_and_parse() -> list[CheckResult]:
            results: list[CheckResult] = []
            for py_file in self.workspace.rglob("*.py"):
                if self._should_skip_path(py_file):
                    continue

                relative_path = py_file.relative_to(self.workspace)
                try:
                    source = py_file.read_text(encoding="utf-8")
                    ast.parse(source, filename=str(py_file))
                    results.append(
                        CheckResult(
                            name=f"syntax:{relative_path}",
                            level=CheckLevel.SYNTAX,
                            status=CheckStatus.PASSED,
                            message=f"Parses successfully: {relative_path}",
                        )
                    )
                except SyntaxError as e:
                    results.append(
                        CheckResult(
                            name=f"syntax:{relative_path}",
                            level=CheckLevel.SYNTAX,
                            status=CheckStatus.FAILED,
                            message=f"Syntax error in {relative_path}",
                            error=f"Line {e.lineno}: {e.msg}",
                            metadata={
                                "line": e.lineno,
                                "offset": e.offset,
                                "text": e.text,
                            },
                        )
                    )
                except (OSError, UnicodeDecodeError) as e:
                    results.append(
                        CheckResult(
                            name=f"syntax:{relative_path}",
                            level=CheckLevel.SYNTAX,
                            status=CheckStatus.ERROR,
                            message=f"Could not read {relative_path}",
                            error=str(e),
                        )
                    )
            return results

        # Run in thread to avoid blocking
        checks = await asyncio.to_thread(_scan_and_parse)

        if not checks:
            checks.append(
                CheckResult(
                    name="syntax_check",
                    level=CheckLevel.SYNTAX,
                    status=CheckStatus.SKIPPED,
                    message="No Python files found",
                )
            )

        return checks

    async def _check_imports(self) -> list[CheckResult]:
        """Level 3: Check that imports can be resolved.

        This uses a lightweight approach - we parse the AST and check if
        top-level modules are available, without actually importing them.
        """
        checks: list[CheckResult] = []
        seen_imports: set[str] = set()

        def _collect_imports() -> dict[str, list[Path]]:
            """Collect all unique imports from Python files."""
            imports: dict[str, list[Path]] = {}

            for py_file in self.workspace.rglob("*.py"):
                if self._should_skip_path(py_file):
                    continue

                try:
                    source = py_file.read_text(encoding="utf-8")
                    tree = ast.parse(source, filename=str(py_file))

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                module = alias.name.split(".")[0]
                                if module not in imports:
                                    imports[module] = []
                                imports[module].append(py_file)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                module = node.module.split(".")[0]
                                if module not in imports:
                                    imports[module] = []
                                imports[module].append(py_file)
                except (SyntaxError, OSError, UnicodeDecodeError):
                    # Skip files that can't be parsed
                    pass

            return imports

        def _check_module_available(module_name: str) -> bool:
            """Check if a module is available without importing it."""
            import importlib.util

            # Check if it's a stdlib module or installed package
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                return True

            # Check if it's a local module in the workspace
            # Look for module_name.py or module_name/__init__.py
            local_file = self.workspace / f"{module_name}.py"
            local_pkg = self.workspace / module_name / "__init__.py"
            src_file = self.workspace / "src" / f"{module_name}.py"
            src_pkg = self.workspace / "src" / module_name / "__init__.py"

            return any(p.exists() for p in [local_file, local_pkg, src_file, src_pkg])

        imports = await asyncio.to_thread(_collect_imports)

        if not imports:
            checks.append(
                CheckResult(
                    name="import_check",
                    level=CheckLevel.IMPORTS,
                    status=CheckStatus.SKIPPED,
                    message="No imports found",
                )
            )
            return checks

        for module_name, files in imports.items():
            if module_name in seen_imports:
                continue
            seen_imports.add(module_name)

            available = await asyncio.to_thread(_check_module_available, module_name)

            if available:
                checks.append(
                    CheckResult(
                        name=f"import:{module_name}",
                        level=CheckLevel.IMPORTS,
                        status=CheckStatus.PASSED,
                        message=f"Module available: {module_name}",
                        metadata={
                            "used_in": [str(f.relative_to(self.workspace)) for f in files[:3]]
                        },
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        name=f"import:{module_name}",
                        level=CheckLevel.IMPORTS,
                        status=CheckStatus.FAILED,
                        message=f"Module not found: {module_name}",
                        error=f"Cannot resolve import: {module_name}",
                        metadata={
                            "used_in": [str(f.relative_to(self.workspace)) for f in files[:3]]
                        },
                    )
                )

        return checks

    async def _check_runtime(self) -> list[CheckResult]:
        """Level 4: Actually import Python modules to catch runtime errors.

        This runs a subprocess that attempts to import each Python module,
        catching errors like circular imports that static analysis misses.

        For other languages, attempts to run the appropriate syntax check:
        - Node.js: node --check
        - TypeScript: tsc --noEmit (if tsconfig exists)
        - Rust: cargo check
        - Go: go build -n
        """

        checks: list[CheckResult] = []
        project_type = detect_project_type(self.workspace)

        if project_type == ProjectType.PYTHON:
            checks.extend(await self._check_runtime_python())
        elif project_type in (ProjectType.NODEJS, ProjectType.TYPESCRIPT):
            checks.extend(await self._check_runtime_node())
        elif project_type == ProjectType.RUST:
            checks.extend(await self._check_runtime_rust())
        elif project_type == ProjectType.GO:
            checks.extend(await self._check_runtime_go())
        else:
            checks.append(
                CheckResult(
                    name="runtime_check",
                    level=CheckLevel.RUNTIME,
                    status=CheckStatus.SKIPPED,
                    message=f"No runtime check available for {project_type.value}",
                )
            )

        return checks

    async def _check_runtime_python(self) -> list[CheckResult]:
        """Check Python modules by actually importing them in a subprocess."""
        import time

        checks: list[CheckResult] = []

        # Find all Python modules in src/ or at root level
        modules_to_check: list[tuple[str, Path]] = []

        # Check src/ directory structure
        src_dir = self.workspace / "src"
        if src_dir.exists():
            for py_file in src_dir.rglob("*.py"):
                if self._should_skip_path(py_file) or py_file.name.startswith("_"):
                    continue
                relative = py_file.relative_to(self.workspace)
                # Convert path to module: src/foo/bar.py -> src.foo.bar
                module_name = str(relative.with_suffix("")).replace("/", ".")
                modules_to_check.append((module_name, py_file))
        else:
            # Check root level modules
            for py_file in self.workspace.glob("*.py"):
                if self._should_skip_path(py_file) or py_file.name.startswith("_"):
                    continue
                module_name = py_file.stem
                modules_to_check.append((module_name, py_file))

        if not modules_to_check:
            checks.append(
                CheckResult(
                    name="runtime_python",
                    level=CheckLevel.RUNTIME,
                    status=CheckStatus.SKIPPED,
                    message="No Python modules found to check",
                )
            )
            return checks

        # Try importing each module in a subprocess
        for module_name, py_file in modules_to_check:
            start = time.perf_counter()

            # Run import in subprocess with workspace as PYTHONPATH
            import_code = f"import {module_name}"
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        subprocess.run,
                        [sys.executable, "-c", import_code],
                        capture_output=True,
                        text=True,
                        cwd=str(self.workspace),
                        env={**subprocess.os.environ, "PYTHONPATH": str(self.workspace)},
                        timeout=10,
                    ),
                    timeout=15,
                )

                duration = (time.perf_counter() - start) * 1000

                if result.returncode == 0:
                    checks.append(
                        CheckResult(
                            name=f"runtime:{module_name}",
                            level=CheckLevel.RUNTIME,
                            status=CheckStatus.PASSED,
                            message=f"Module imports successfully: {module_name}",
                            duration_ms=duration,
                        )
                    )
                else:
                    # Parse the error to give useful feedback
                    error_msg = result.stderr.strip()
                    # Extract the key error line
                    error_lines = error_msg.split("\n")
                    short_error = error_lines[-1] if error_lines else error_msg

                    checks.append(
                        CheckResult(
                            name=f"runtime:{module_name}",
                            level=CheckLevel.RUNTIME,
                            status=CheckStatus.FAILED,
                            message=f"Import failed: {module_name}",
                            error=short_error,
                            duration_ms=duration,
                            metadata={"full_error": error_msg, "file": str(py_file)},
                        )
                    )
            except TimeoutError:
                checks.append(
                    CheckResult(
                        name=f"runtime:{module_name}",
                        level=CheckLevel.RUNTIME,
                        status=CheckStatus.ERROR,
                        message=f"Timeout importing: {module_name}",
                        error="Import timed out after 15 seconds",
                    )
                )
            except Exception as e:
                checks.append(
                    CheckResult(
                        name=f"runtime:{module_name}",
                        level=CheckLevel.RUNTIME,
                        status=CheckStatus.ERROR,
                        message=f"Error checking: {module_name}",
                        error=str(e),
                    )
                )

        return checks

    async def _check_runtime_node(self) -> list[CheckResult]:
        """Check Node.js/TypeScript files with node --check or tsc --noEmit."""
        checks: list[CheckResult] = []

        # For TypeScript, use tsc --noEmit
        if (self.workspace / "tsconfig.json").exists():
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        subprocess.run,
                        ["npx", "tsc", "--noEmit"],
                        capture_output=True,
                        text=True,
                        cwd=str(self.workspace),
                        timeout=60,
                    ),
                    timeout=90,
                )

                if result.returncode == 0:
                    checks.append(
                        CheckResult(
                            name="runtime:typescript",
                            level=CheckLevel.RUNTIME,
                            status=CheckStatus.PASSED,
                            message="TypeScript compiles successfully",
                        )
                    )
                else:
                    checks.append(
                        CheckResult(
                            name="runtime:typescript",
                            level=CheckLevel.RUNTIME,
                            status=CheckStatus.FAILED,
                            message="TypeScript compilation failed",
                            error=result.stderr or result.stdout,
                        )
                    )
            except Exception as e:
                checks.append(
                    CheckResult(
                        name="runtime:typescript",
                        level=CheckLevel.RUNTIME,
                        status=CheckStatus.ERROR,
                        message="Error running tsc",
                        error=str(e),
                    )
                )
        else:
            # For plain Node.js, check each JS file with node --check
            js_files = list(self.workspace.glob("**/*.js"))
            js_files = [f for f in js_files if not self._should_skip_path(f)]

            if not js_files:
                checks.append(
                    CheckResult(
                        name="runtime:nodejs",
                        level=CheckLevel.RUNTIME,
                        status=CheckStatus.SKIPPED,
                        message="No JavaScript files found",
                    )
                )
                return checks

            for js_file in js_files[:10]:  # Limit to first 10 files
                relative = js_file.relative_to(self.workspace)
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            subprocess.run,
                            ["node", "--check", str(js_file)],
                            capture_output=True,
                            text=True,
                            cwd=str(self.workspace),
                            timeout=10,
                        ),
                        timeout=15,
                    )

                    if result.returncode == 0:
                        checks.append(
                            CheckResult(
                                name=f"runtime:{relative}",
                                level=CheckLevel.RUNTIME,
                                status=CheckStatus.PASSED,
                                message=f"Syntax valid: {relative}",
                            )
                        )
                    else:
                        checks.append(
                            CheckResult(
                                name=f"runtime:{relative}",
                                level=CheckLevel.RUNTIME,
                                status=CheckStatus.FAILED,
                                message=f"Syntax error: {relative}",
                                error=result.stderr,
                            )
                        )
                except Exception as e:
                    checks.append(
                        CheckResult(
                            name=f"runtime:{relative}",
                            level=CheckLevel.RUNTIME,
                            status=CheckStatus.ERROR,
                            message=f"Error checking: {relative}",
                            error=str(e),
                        )
                    )

        return checks

    async def _check_runtime_rust(self) -> list[CheckResult]:
        """Check Rust code with cargo check."""
        checks: list[CheckResult] = []

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    subprocess.run,
                    ["cargo", "check"],
                    capture_output=True,
                    text=True,
                    cwd=str(self.workspace),
                    timeout=120,
                ),
                timeout=150,
            )

            if result.returncode == 0:
                checks.append(
                    CheckResult(
                        name="runtime:rust",
                        level=CheckLevel.RUNTIME,
                        status=CheckStatus.PASSED,
                        message="Rust code compiles successfully",
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        name="runtime:rust",
                        level=CheckLevel.RUNTIME,
                        status=CheckStatus.FAILED,
                        message="Rust compilation failed",
                        error=result.stderr,
                    )
                )
        except Exception as e:
            checks.append(
                CheckResult(
                    name="runtime:rust",
                    level=CheckLevel.RUNTIME,
                    status=CheckStatus.ERROR,
                    message="Error running cargo check",
                    error=str(e),
                )
            )

        return checks

    async def _check_runtime_go(self) -> list[CheckResult]:
        """Check Go code with go build -n (dry run)."""
        checks: list[CheckResult] = []

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    subprocess.run,
                    ["go", "build", "-n", "./..."],
                    capture_output=True,
                    text=True,
                    cwd=str(self.workspace),
                    timeout=60,
                ),
                timeout=90,
            )

            if result.returncode == 0:
                checks.append(
                    CheckResult(
                        name="runtime:go",
                        level=CheckLevel.RUNTIME,
                        status=CheckStatus.PASSED,
                        message="Go code compiles successfully",
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        name="runtime:go",
                        level=CheckLevel.RUNTIME,
                        status=CheckStatus.FAILED,
                        message="Go compilation failed",
                        error=result.stderr,
                    )
                )
        except Exception as e:
            checks.append(
                CheckResult(
                    name="runtime:go",
                    level=CheckLevel.RUNTIME,
                    status=CheckStatus.ERROR,
                    message="Error running go build",
                    error=str(e),
                )
            )

        return checks

    async def _check_tests(self) -> list[CheckResult]:
        """Level 5: Run appropriate test runner based on project type.

        Detects the project type and runs the corresponding test command:
        - Python: pytest
        - Node.js: npm test or node --test
        - TypeScript: npm test
        - Rust: cargo test
        - Go: go test
        - Makefile: make test
        """
        checks: list[CheckResult] = []

        # Detect project type
        project_type = detect_project_type(self.workspace)
        logger.debug(f"Detected project type: {project_type.value}")

        # Get appropriate test command
        test_cmd = get_test_command(project_type, self.workspace)

        if test_cmd is None:
            # Check if tests directory exists for unknown projects
            tests_dir = self.workspace / "tests"
            if not tests_dir.exists():
                checks.append(
                    CheckResult(
                        name="tests",
                        level=CheckLevel.TESTS,
                        status=CheckStatus.SKIPPED,
                        message="No tests directory found",
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        name="tests",
                        level=CheckLevel.TESTS,
                        status=CheckStatus.SKIPPED,
                        message=f"No test runner available for project type: {project_type.value}",
                        metadata={"project_type": project_type.value},
                    )
                )
            return checks

        # Check if there are any test files to run
        tests_dir = self.workspace / "tests"
        has_test_files = False
        if tests_dir.exists():
            for pattern in [
                "test_*.py",
                "*_test.py",
                "*.test.js",
                "*.test.ts",
                "*_test.go",
                "*_test.rs",
            ]:
                if list(tests_dir.glob(f"**/{pattern}")):
                    has_test_files = True
                    break

        if not has_test_files:
            checks.append(
                CheckResult(
                    name="tests",
                    level=CheckLevel.TESTS,
                    status=CheckStatus.SKIPPED,
                    message="No test files found in tests/ directory",
                    metadata={"project_type": project_type.value},
                )
            )
            return checks

        # For Python projects, append any custom pytest args
        if project_type == ProjectType.PYTHON and self.pytest_args:
            test_cmd = test_cmd + self.pytest_args

        try:
            result = await self._run_command(test_cmd)

            # Handle pytest exit codes:
            # 0 = all tests passed
            # 1 = tests failed
            # 5 = no tests collected (treat as skipped, not failure)
            returncode = result["returncode"]

            if returncode == 0:
                status = CheckStatus.PASSED
                message = "Tests passed"
            elif returncode == 5:
                # Pytest: no tests collected - this is okay if tests don't exist yet
                status = CheckStatus.SKIPPED
                message = "No tests collected"
            else:
                status = CheckStatus.FAILED
                message = "Tests failed"

            runner_name = test_cmd[0] if test_cmd else "unknown"
            checks.append(
                CheckResult(
                    name="tests",
                    level=CheckLevel.TESTS,
                    status=status,
                    message=f"{message} ({runner_name})",
                    error=None
                    if status != CheckStatus.FAILED
                    else result["stderr"] or result["stdout"],
                    duration_ms=result["duration_ms"],
                    metadata={
                        "returncode": returncode,
                        "stdout_lines": len(result["stdout"].split("\n"))
                        if result["stdout"]
                        else 0,
                        "project_type": project_type.value,
                        "test_command": " ".join(test_cmd),
                    },
                )
            )
        except TimeoutError:
            checks.append(
                CheckResult(
                    name="tests",
                    level=CheckLevel.TESTS,
                    status=CheckStatus.ERROR,
                    message="Tests timed out",
                    error=f"Test runner exceeded {self.timeout_seconds}s timeout",
                    metadata={"project_type": project_type.value},
                )
            )
        except Exception as e:
            checks.append(
                CheckResult(
                    name="tests",
                    level=CheckLevel.TESTS,
                    status=CheckStatus.ERROR,
                    message="Failed to run tests",
                    error=str(e),
                    metadata={"project_type": project_type.value},
                )
            )

        return checks

    async def _check_types(self) -> list[CheckResult]:
        """Level 5: Run mypy if pyproject.toml exists."""
        checks: list[CheckResult] = []

        pyproject = self.workspace / "pyproject.toml"
        if not pyproject.exists():
            checks.append(
                CheckResult(
                    name="types",
                    level=CheckLevel.TYPES,
                    status=CheckStatus.SKIPPED,
                    message="No pyproject.toml found",
                )
            )
            return checks

        # Check if there's a src directory (common layout)
        src_dir = self.workspace / "src"
        target = str(src_dir) if src_dir.exists() else str(self.workspace)

        try:
            result = await self._run_command(
                [sys.executable, "-m", "mypy", target] + self.mypy_args
            )

            passed = result["returncode"] == 0
            checks.append(
                CheckResult(
                    name="types",
                    level=CheckLevel.TYPES,
                    status=CheckStatus.PASSED if passed else CheckStatus.FAILED,
                    message="Type checking passed" if passed else "Type errors found",
                    error=None if passed else result["stdout"],
                    duration_ms=result["duration_ms"],
                    metadata={
                        "returncode": result["returncode"],
                    },
                )
            )
        except TimeoutError:
            checks.append(
                CheckResult(
                    name="types",
                    level=CheckLevel.TYPES,
                    status=CheckStatus.ERROR,
                    message="Type checking timed out",
                    error=f"Mypy exceeded {self.timeout_seconds}s timeout",
                )
            )
        except Exception as e:
            checks.append(
                CheckResult(
                    name="types",
                    level=CheckLevel.TYPES,
                    status=CheckStatus.ERROR,
                    message="Failed to run type checking",
                    error=str(e),
                )
            )

        return checks

    async def _run_command(self, cmd: list[str]) -> dict[str, Any]:
        """Run an external command and capture output."""
        import time

        start_time = time.perf_counter()

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.workspace,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout_seconds
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise

        duration_ms = (time.perf_counter() - start_time) * 1000

        return {
            "returncode": proc.returncode,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "duration_ms": duration_ms,
        }

    async def quick_verify(self, task: Task) -> VerificationResult:
        """Fast verification: only check files and syntax.

        Use this for quick feedback during development.
        """
        return await self.verify(
            task, levels=[CheckLevel.FILES, CheckLevel.SYNTAX], stop_on_failure=True
        )

    async def full_verify(self, task: Task) -> VerificationResult:
        """Complete verification: all levels.

        Use this for final validation before marking a task complete.
        """
        return await self.verify(task, levels=list(CheckLevel))
