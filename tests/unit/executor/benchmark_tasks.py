"""Benchmark task definitions and harness for executor validation.

This module implements Phase 0.1 of EXECUTOR.md:
- 5 benchmark tasks of increasing complexity (L1-L5)
- Test harness using Agent(deep=True)
- Verification functions for each task level

The goal is to prove that DeepAgents can complete real coding tasks
before building the orchestration infrastructure.
"""

from __future__ import annotations

import ast
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ai_infra.llm import Agent
from ai_infra.llm.workspace import Workspace


class TaskLevel(Enum):
    """Complexity levels for benchmark tasks."""

    L1 = 1  # Create a new empty file
    L2 = 2  # Add a function to existing file
    L3 = 3  # Modify function signature + update callers
    L4 = 4  # Create new module with tests
    L5 = 5  # Refactor: extract class to new file


class VerificationStatus(Enum):
    """Result of task verification."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class VerificationResult:
    """Result of verifying a task completion."""

    status: VerificationStatus
    checks: dict[str, bool] = field(default_factory=dict)
    error: str | None = None
    details: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASSED


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark task."""

    task_id: str
    level: TaskLevel
    passed: bool
    verification: VerificationResult
    duration_seconds: float
    token_count: int = 0
    agent_response: str | None = None
    error: str | None = None
    started_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "level": self.level.name,
            "passed": self.passed,
            "verification": {
                "status": self.verification.status.value,
                "checks": self.verification.checks,
                "error": self.verification.error,
                "details": self.verification.details,
            },
            "duration_seconds": self.duration_seconds,
            "token_count": self.token_count,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
        }


@dataclass
class BenchmarkTask:
    """A benchmark task definition."""

    id: str
    level: TaskLevel
    description: str
    setup: str  # Description of initial state
    success_criteria: list[str]
    verify: type[TaskVerifier]

    def get_prompt(self) -> str:
        """Generate the prompt for the agent."""
        criteria_text = "\n".join(f"- {c}" for c in self.success_criteria)
        return f"""You are an expert software engineer. Complete this task:

## Task
{self.description}

## Success Criteria
{criteria_text}

## Instructions
1. Use ls and read_file to understand the current state
2. Use write_file or edit_file to make changes
3. Verify your changes work before finishing

Begin."""


class TaskVerifier:
    """Base class for task verification."""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir

    def verify(self) -> VerificationResult:
        """Verify the task was completed successfully."""
        raise NotImplementedError


class L1Verifier(TaskVerifier):
    """Verify L1: Create a new empty file."""

    def __init__(self, workspace_dir: Path, expected_path: str):
        super().__init__(workspace_dir)
        self.expected_path = expected_path

    def verify(self) -> VerificationResult:
        checks: dict[str, bool] = {}
        full_path = self.workspace_dir / self.expected_path

        # Check 1: File exists
        checks["file_exists"] = full_path.exists()

        # Check 2: Is a file (not directory)
        checks["is_file"] = full_path.is_file() if full_path.exists() else False

        passed = all(checks.values())
        return VerificationResult(
            status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
            checks=checks,
            details=f"Expected file at: {self.expected_path}",
        )


class L2Verifier(TaskVerifier):
    """Verify L2: Add a function to existing file."""

    def __init__(self, workspace_dir: Path, file_path: str, function_name: str):
        super().__init__(workspace_dir)
        self.file_path = file_path
        self.function_name = function_name

    def verify(self) -> VerificationResult:
        checks: dict[str, bool] = {}
        full_path = self.workspace_dir / self.file_path

        # Check 1: File exists
        checks["file_exists"] = full_path.exists()

        if not full_path.exists():
            return VerificationResult(
                status=VerificationStatus.FAILED,
                checks=checks,
                error=f"File not found: {self.file_path}",
            )

        # Check 2: File parses as valid Python
        content = full_path.read_text()
        try:
            tree = ast.parse(content)
            checks["file_parses"] = True
        except SyntaxError as e:
            checks["file_parses"] = False
            return VerificationResult(
                status=VerificationStatus.FAILED,
                checks=checks,
                error=f"Syntax error: {e}",
            )

        # Check 3: Function exists
        function_names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        checks["function_exists"] = self.function_name in function_names

        passed = all(checks.values())
        return VerificationResult(
            status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
            checks=checks,
            details=f"Found functions: {function_names}",
        )


class L3Verifier(TaskVerifier):
    """Verify L3: Modify function signature + update callers."""

    def __init__(
        self,
        workspace_dir: Path,
        target_file: str,
        function_name: str,
        expected_params: list[str],
        caller_files: list[str],
    ):
        super().__init__(workspace_dir)
        self.target_file = target_file
        self.function_name = function_name
        self.expected_params = expected_params
        self.caller_files = caller_files

    def verify(self) -> VerificationResult:
        checks: dict[str, bool] = {}

        # Check 1: Target file parses
        target_path = self.workspace_dir / self.target_file
        if not target_path.exists():
            return VerificationResult(
                status=VerificationStatus.FAILED,
                checks={"target_exists": False},
                error=f"Target file not found: {self.target_file}",
            )

        try:
            tree = ast.parse(target_path.read_text())
            checks["target_parses"] = True
        except SyntaxError as e:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                checks={"target_parses": False},
                error=f"Syntax error in target: {e}",
            )

        # Check 2: Function has expected parameters
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == self.function_name:
                    actual_params = [arg.arg for arg in node.args.args]
                    checks["params_updated"] = all(p in actual_params for p in self.expected_params)
                    break
        else:
            checks["params_updated"] = False

        # Check 3: All caller files parse (no broken imports/calls)
        for caller in self.caller_files:
            caller_path = self.workspace_dir / caller
            if caller_path.exists():
                try:
                    ast.parse(caller_path.read_text())
                    checks[f"caller_{caller}_parses"] = True
                except SyntaxError:
                    checks[f"caller_{caller}_parses"] = False
            else:
                checks[f"caller_{caller}_exists"] = False

        passed = all(checks.values())
        return VerificationResult(
            status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
            checks=checks,
        )


class L4Verifier(TaskVerifier):
    """Verify L4: Create new module with tests."""

    def __init__(
        self,
        workspace_dir: Path,
        module_path: str,
        test_path: str,
        expected_classes: list[str] | None = None,
        expected_functions: list[str] | None = None,
    ):
        super().__init__(workspace_dir)
        self.module_path = module_path
        self.test_path = test_path
        self.expected_classes = expected_classes or []
        self.expected_functions = expected_functions or []

    def verify(self) -> VerificationResult:
        checks: dict[str, bool] = {}

        # Check 1: Module file exists and parses
        module_full = self.workspace_dir / self.module_path
        if not module_full.exists():
            return VerificationResult(
                status=VerificationStatus.FAILED,
                checks={"module_exists": False},
                error=f"Module not found: {self.module_path}",
            )

        try:
            module_tree = ast.parse(module_full.read_text())
            checks["module_parses"] = True
        except SyntaxError as e:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                checks={"module_parses": False},
                error=f"Module syntax error: {e}",
            )

        # Check 2: Expected classes/functions exist
        found_classes = {
            node.name for node in ast.walk(module_tree) if isinstance(node, ast.ClassDef)
        }
        found_functions = {
            node.name
            for node in ast.walk(module_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        for cls in self.expected_classes:
            checks[f"class_{cls}_exists"] = cls in found_classes

        for func in self.expected_functions:
            checks[f"function_{func}_exists"] = func in found_functions

        # Check 3: Test file exists and parses
        test_full = self.workspace_dir / self.test_path
        if not test_full.exists():
            checks["test_exists"] = False
        else:
            checks["test_exists"] = True
            try:
                test_tree = ast.parse(test_full.read_text())
                checks["test_parses"] = True

                # Check that test file has test functions
                test_functions = [
                    node.name
                    for node in ast.walk(test_tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name.startswith("test_")
                ]
                checks["has_test_functions"] = len(test_functions) > 0
            except SyntaxError:
                checks["test_parses"] = False

        passed = all(checks.values())
        return VerificationResult(
            status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
            checks=checks,
            details=f"Classes: {found_classes}, Functions: {found_functions}",
        )


class L5Verifier(TaskVerifier):
    """Verify L5: Refactor - extract class to new file."""

    def __init__(
        self,
        workspace_dir: Path,
        original_file: str,
        new_file: str,
        class_name: str,
        import_files: list[str],
    ):
        super().__init__(workspace_dir)
        self.original_file = original_file
        self.new_file = new_file
        self.class_name = class_name
        self.import_files = import_files

    def verify(self) -> VerificationResult:
        checks: dict[str, bool] = {}

        # Check 1: New file exists and contains the class
        new_path = self.workspace_dir / self.new_file
        if not new_path.exists():
            return VerificationResult(
                status=VerificationStatus.FAILED,
                checks={"new_file_exists": False},
                error=f"New file not found: {self.new_file}",
            )

        try:
            new_tree = ast.parse(new_path.read_text())
            checks["new_file_parses"] = True
        except SyntaxError as e:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                checks={"new_file_parses": False},
                error=f"New file syntax error: {e}",
            )

        new_classes = {node.name for node in ast.walk(new_tree) if isinstance(node, ast.ClassDef)}
        checks["class_in_new_file"] = self.class_name in new_classes

        # Check 2: Original file parses (class may or may not be removed)
        orig_path = self.workspace_dir / self.original_file
        if orig_path.exists():
            try:
                ast.parse(orig_path.read_text())
                checks["original_parses"] = True
            except SyntaxError:
                checks["original_parses"] = False
        else:
            checks["original_exists"] = False

        # Check 3: All files that need to import the class still parse
        for import_file in self.import_files:
            import_path = self.workspace_dir / import_file
            if import_path.exists():
                try:
                    ast.parse(import_path.read_text())
                    checks[f"import_{import_file}_parses"] = True
                except SyntaxError:
                    checks[f"import_{import_file}_parses"] = False

        passed = all(checks.values())
        return VerificationResult(
            status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
            checks=checks,
        )


# =============================================================================
# Benchmark Task Definitions
# =============================================================================

BENCHMARK_TASKS: list[BenchmarkTask] = [
    BenchmarkTask(
        id="L1-create-file",
        level=TaskLevel.L1,
        description="Create a new Python file at `src/utils/helpers.py` with a docstring.",
        setup="Empty workspace with src/ directory",
        success_criteria=[
            "File exists at src/utils/helpers.py",
            "File is valid Python (can be parsed)",
        ],
        verify=L1Verifier,  # type: ignore[arg-type]
    ),
    BenchmarkTask(
        id="L2-add-function",
        level=TaskLevel.L2,
        description=(
            "Add a function called `format_currency(amount: float, currency: str = 'USD') -> str` "
            "to the existing file `src/formatters.py`. The function should format a number as "
            "currency (e.g., '$1,234.56')."
        ),
        setup="Workspace with src/formatters.py containing existing format_date function",
        success_criteria=[
            "File src/formatters.py exists",
            "File parses as valid Python",
            "Function format_currency exists",
        ],
        verify=L2Verifier,  # type: ignore[arg-type]
    ),
    BenchmarkTask(
        id="L3-update-signature",
        level=TaskLevel.L3,
        description=(
            "Update the function `process_data(data: list)` in `src/processor.py` to accept "
            "an additional optional parameter `validate: bool = True`. Update all callers "
            "in `src/handlers.py` to pass this parameter explicitly."
        ),
        setup="Workspace with processor.py and handlers.py that calls process_data",
        success_criteria=[
            "processor.py parses",
            "process_data has validate parameter",
            "handlers.py parses (callers updated correctly)",
        ],
        verify=L3Verifier,  # type: ignore[arg-type]
    ),
    BenchmarkTask(
        id="L4-new-module",
        level=TaskLevel.L4,
        description=(
            "Create a new module `src/cache.py` with a `Cache` class that has methods "
            "`get(key: str) -> Any`, `set(key: str, value: Any) -> None`, and `clear() -> None`. "
            "Also create `tests/test_cache.py` with tests for each method."
        ),
        setup="Workspace with existing src/ and tests/ directories",
        success_criteria=[
            "src/cache.py exists and parses",
            "Cache class exists with get, set, clear methods",
            "tests/test_cache.py exists with test functions",
        ],
        verify=L4Verifier,  # type: ignore[arg-type]
    ),
    BenchmarkTask(
        id="L5-refactor-extract",
        level=TaskLevel.L5,
        description=(
            "Extract the `Logger` class from `src/app.py` into a new file `src/logger.py`. "
            "Update `src/app.py` to import Logger from the new location. "
            "Ensure `src/main.py` which imports from app.py still works."
        ),
        setup="Workspace with app.py containing Logger class, main.py that uses it",
        success_criteria=[
            "src/logger.py exists with Logger class",
            "src/app.py parses and imports from logger",
            "src/main.py parses",
        ],
        verify=L5Verifier,  # type: ignore[arg-type]
    ),
]


# =============================================================================
# Fixture Setup Functions
# =============================================================================


def setup_l1_fixture(workspace_dir: Path) -> dict[str, Any]:
    """Set up fixture for L1: Create a new empty file."""
    # Create src directory
    (workspace_dir / "src").mkdir(parents=True, exist_ok=True)
    return {"expected_path": "src/utils/helpers.py"}


def setup_l2_fixture(workspace_dir: Path) -> dict[str, Any]:
    """Set up fixture for L2: Add a function to existing file."""
    src_dir = workspace_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    # Create existing file with a function
    formatters_content = '''"""Formatting utilities."""

from datetime import datetime


def format_date(dt: datetime, fmt: str = "%Y-%m-%d") -> str:
    """Format a datetime object as a string."""
    return dt.strftime(fmt)
'''
    (src_dir / "formatters.py").write_text(formatters_content)

    return {
        "file_path": "src/formatters.py",
        "function_name": "format_currency",
    }


def setup_l3_fixture(workspace_dir: Path) -> dict[str, Any]:
    """Set up fixture for L3: Modify function signature + update callers."""
    src_dir = workspace_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    # Create processor.py with the function to modify
    processor_content = '''"""Data processing module."""

from typing import Any


def process_data(data: list) -> list[dict[str, Any]]:
    """Process a list of data items."""
    result = []
    for item in data:
        result.append({"processed": True, "value": item})
    return result
'''
    (src_dir / "processor.py").write_text(processor_content)

    # Create handlers.py that calls process_data
    handlers_content = '''"""Request handlers."""

from src.processor import process_data


def handle_request(items: list) -> dict:
    """Handle a request with items."""
    processed = process_data(items)
    return {"status": "ok", "data": processed}


def handle_batch(batches: list[list]) -> list[dict]:
    """Handle multiple batches."""
    results = []
    for batch in batches:
        processed = process_data(batch)
        results.append({"batch": processed})
    return results
'''
    (src_dir / "handlers.py").write_text(handlers_content)

    return {
        "target_file": "src/processor.py",
        "function_name": "process_data",
        "expected_params": ["data", "validate"],
        "caller_files": ["src/handlers.py"],
    }


def setup_l4_fixture(workspace_dir: Path) -> dict[str, Any]:
    """Set up fixture for L4: Create new module with tests."""
    (workspace_dir / "src").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "tests").mkdir(parents=True, exist_ok=True)

    # Create __init__.py files
    (workspace_dir / "src" / "__init__.py").write_text("")
    (workspace_dir / "tests" / "__init__.py").write_text("")

    return {
        "module_path": "src/cache.py",
        "test_path": "tests/test_cache.py",
        "expected_classes": ["Cache"],
        "expected_functions": [],
    }


def setup_l5_fixture(workspace_dir: Path) -> dict[str, Any]:
    """Set up fixture for L5: Refactor - extract class to new file."""
    src_dir = workspace_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    # Create app.py with Logger class embedded
    app_content = '''"""Main application module."""

import sys
from datetime import datetime


class Logger:
    """Application logger."""

    def __init__(self, name: str):
        self.name = name

    def info(self, message: str) -> None:
        """Log an info message."""
        timestamp = datetime.now().isoformat()
        print(f"[{timestamp}] INFO [{self.name}]: {message}")

    def error(self, message: str) -> None:
        """Log an error message."""
        timestamp = datetime.now().isoformat()
        print(f"[{timestamp}] ERROR [{self.name}]: {message}", file=sys.stderr)


class App:
    """Main application class."""

    def __init__(self, name: str):
        self.name = name
        self.logger = Logger(name)

    def run(self) -> None:
        """Run the application."""
        self.logger.info("Application started")
'''
    (src_dir / "app.py").write_text(app_content)

    # Create main.py that imports from app
    main_content = '''"""Entry point."""

from src.app import App, Logger


def main() -> None:
    """Main entry point."""
    logger = Logger("main")
    logger.info("Starting...")

    app = App("my-app")
    app.run()


if __name__ == "__main__":
    main()
'''
    (src_dir / "main.py").write_text(main_content)

    return {
        "original_file": "src/app.py",
        "new_file": "src/logger.py",
        "class_name": "Logger",
        "import_files": ["src/app.py", "src/main.py"],
    }


FIXTURE_SETUP = {
    TaskLevel.L1: setup_l1_fixture,
    TaskLevel.L2: setup_l2_fixture,
    TaskLevel.L3: setup_l3_fixture,
    TaskLevel.L4: setup_l4_fixture,
    TaskLevel.L5: setup_l5_fixture,
}


# =============================================================================
# Test Harness
# =============================================================================


class BenchmarkHarness:
    """Harness for running benchmark tasks."""

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        workspace_mode: str = "sandboxed",
    ):
        self.model_name = model_name
        self.workspace_mode = workspace_mode

    def create_workspace(self) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
        """Create a temporary workspace directory."""
        temp_dir = tempfile.TemporaryDirectory(prefix="executor_benchmark_")
        return Path(temp_dir.name), temp_dir

    def get_task(self, task_id: str) -> BenchmarkTask | None:
        """Get a task by ID."""
        for task in BENCHMARK_TASKS:
            if task.id == task_id:
                return task
        return None

    def setup_fixture(self, level: TaskLevel, workspace_dir: Path) -> dict[str, Any]:
        """Set up the fixture for a task level."""
        setup_func = FIXTURE_SETUP.get(level)
        if setup_func:
            return setup_func(workspace_dir)
        return {}

    async def run_task(self, task: BenchmarkTask) -> BenchmarkResult:
        """Run a single benchmark task."""
        import time

        workspace_dir, temp_dir = self.create_workspace()

        try:
            # Set up fixture
            fixture_params = self.setup_fixture(task.level, workspace_dir)

            # Create agent with DeepAgents mode
            agent = Agent(
                deep=True,
                workspace=Workspace(workspace_dir, mode=self.workspace_mode),  # type: ignore[arg-type]
                model_name=self.model_name,
            )

            # Run the task
            start_time = time.time()
            prompt = task.get_prompt()

            result = await agent.arun(prompt)
            duration = time.time() - start_time

            # Verify the result
            verifier = self._create_verifier(task, workspace_dir, fixture_params)
            verification = verifier.verify()

            return BenchmarkResult(
                task_id=task.id,
                level=task.level,
                passed=verification.passed,
                verification=verification,
                duration_seconds=duration,
                agent_response=str(result) if result else None,
            )

        except Exception as e:
            return BenchmarkResult(
                task_id=task.id,
                level=task.level,
                passed=False,
                verification=VerificationResult(
                    status=VerificationStatus.ERROR,
                    error=str(e),
                ),
                duration_seconds=0,
                error=str(e),
            )
        finally:
            temp_dir.cleanup()

    def _create_verifier(
        self, task: BenchmarkTask, workspace_dir: Path, fixture_params: dict[str, Any]
    ) -> TaskVerifier:
        """Create the appropriate verifier for a task."""
        if task.level == TaskLevel.L1:
            return L1Verifier(
                workspace_dir,
                expected_path=fixture_params.get("expected_path", ""),
            )
        elif task.level == TaskLevel.L2:
            return L2Verifier(
                workspace_dir,
                file_path=fixture_params.get("file_path", ""),
                function_name=fixture_params.get("function_name", ""),
            )
        elif task.level == TaskLevel.L3:
            return L3Verifier(
                workspace_dir,
                target_file=fixture_params.get("target_file", ""),
                function_name=fixture_params.get("function_name", ""),
                expected_params=fixture_params.get("expected_params", []),
                caller_files=fixture_params.get("caller_files", []),
            )
        elif task.level == TaskLevel.L4:
            return L4Verifier(
                workspace_dir,
                module_path=fixture_params.get("module_path", ""),
                test_path=fixture_params.get("test_path", ""),
                expected_classes=fixture_params.get("expected_classes", []),
                expected_functions=fixture_params.get("expected_functions", []),
            )
        elif task.level == TaskLevel.L5:
            return L5Verifier(
                workspace_dir,
                original_file=fixture_params.get("original_file", ""),
                new_file=fixture_params.get("new_file", ""),
                class_name=fixture_params.get("class_name", ""),
                import_files=fixture_params.get("import_files", []),
            )
        else:
            raise ValueError(f"Unknown task level: {task.level}")

    async def run_all(self) -> list[BenchmarkResult]:
        """Run all benchmark tasks."""
        results = []
        for task in BENCHMARK_TASKS:
            result = await self.run_task(task)
            results.append(result)
        return results

    async def run_level(self, level: TaskLevel) -> list[BenchmarkResult]:
        """Run all tasks at a specific level."""
        results = []
        for task in BENCHMARK_TASKS:
            if task.level == level:
                result = await self.run_task(task)
                results.append(result)
        return results


def get_benchmark_summary(results: list[BenchmarkResult]) -> dict[str, Any]:
    """Generate a summary of benchmark results."""
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    by_level: dict[str, dict[str, int]] = {}
    for result in results:
        level_name = result.level.name
        if level_name not in by_level:
            by_level[level_name] = {"passed": 0, "failed": 0}
        if result.passed:
            by_level[level_name]["passed"] += 1
        else:
            by_level[level_name]["failed"] += 1

    total_duration = sum(r.duration_seconds for r in results)
    total_tokens = sum(r.token_count for r in results)

    return {
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "success_rate": passed / len(results) if results else 0,
        "by_level": by_level,
        "total_duration_seconds": total_duration,
        "total_tokens": total_tokens,
        "results": [r.to_dict() for r in results],
    }
