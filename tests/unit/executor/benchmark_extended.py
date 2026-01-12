"""Extended benchmark suite for executor validation.

This module extends the base benchmark tasks to provide a comprehensive
suite of 20 tasks as specified in Phase 4.4 of EXECUTOR.md.

The tasks cover:
- File operations (create, modify, delete)
- Function/class creation and modification
- Refactoring operations
- Multi-file changes
- Test generation
- Documentation updates
- Error handling patterns
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .benchmark_tasks import (
    BENCHMARK_TASKS,
    BenchmarkHarness,
    BenchmarkResult,
    TaskLevel,
    TaskVerifier,
    VerificationResult,
    VerificationStatus,
    get_benchmark_summary,
)

# =============================================================================
# Extended Task Levels
# =============================================================================


class ExtendedTaskLevel(Enum):
    """Extended complexity levels beyond L1-L5."""

    L6 = 6  # Add error handling to existing code
    L7 = 7  # Create API endpoint
    L8 = 8  # Add logging throughout module
    L9 = 9  # Create configuration system
    L10 = 10  # Implement caching layer


# =============================================================================
# Additional Verifiers
# =============================================================================


class FunctionSignatureVerifier(TaskVerifier):
    """Verify a function has specific signature elements."""

    def __init__(
        self,
        workspace_dir: Path,
        file_path: str,
        function_name: str,
        expected_params: list[str] | None = None,
        expected_return_type: str | None = None,
        should_be_async: bool = False,
    ):
        super().__init__(workspace_dir)
        self.file_path = file_path
        self.function_name = function_name
        self.expected_params = expected_params or []
        self.expected_return_type = expected_return_type
        self.should_be_async = should_be_async

    def verify(self) -> VerificationResult:
        checks: dict[str, bool] = {}
        full_path = self.workspace_dir / self.file_path

        if not full_path.exists():
            return VerificationResult(
                status=VerificationStatus.FAILED,
                checks={"file_exists": False},
                error=f"File not found: {self.file_path}",
            )

        try:
            tree = ast.parse(full_path.read_text())
            checks["file_parses"] = True
        except SyntaxError as e:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                checks={"file_parses": False},
                error=f"Syntax error: {e}",
            )

        # Find the function
        func_node = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == self.function_name:
                    func_node = node
                    break

        if func_node is None:
            checks["function_exists"] = False
            return VerificationResult(
                status=VerificationStatus.FAILED,
                checks=checks,
                error=f"Function {self.function_name} not found",
            )

        checks["function_exists"] = True

        # Check async
        if self.should_be_async:
            checks["is_async"] = isinstance(func_node, ast.AsyncFunctionDef)

        # Check parameters
        actual_params = [arg.arg for arg in func_node.args.args]
        for param in self.expected_params:
            checks[f"has_param_{param}"] = param in actual_params

        passed = all(checks.values())
        return VerificationResult(
            status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
            checks=checks,
            details=f"Found params: {actual_params}",
        )


class ClassMethodVerifier(TaskVerifier):
    """Verify a class has specific methods."""

    def __init__(
        self,
        workspace_dir: Path,
        file_path: str,
        class_name: str,
        expected_methods: list[str],
    ):
        super().__init__(workspace_dir)
        self.file_path = file_path
        self.class_name = class_name
        self.expected_methods = expected_methods

    def verify(self) -> VerificationResult:
        checks: dict[str, bool] = {}
        full_path = self.workspace_dir / self.file_path

        if not full_path.exists():
            return VerificationResult(
                status=VerificationStatus.FAILED,
                checks={"file_exists": False},
                error=f"File not found: {self.file_path}",
            )

        try:
            tree = ast.parse(full_path.read_text())
            checks["file_parses"] = True
        except SyntaxError as e:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                checks={"file_parses": False},
                error=f"Syntax error: {e}",
            )

        # Find the class
        class_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == self.class_name:
                class_node = node
                break

        if class_node is None:
            checks["class_exists"] = False
            return VerificationResult(
                status=VerificationStatus.FAILED,
                checks=checks,
                error=f"Class {self.class_name} not found",
            )

        checks["class_exists"] = True

        # Find methods
        method_names = [
            node.name
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        for method in self.expected_methods:
            checks[f"has_method_{method}"] = method in method_names

        passed = all(checks.values())
        return VerificationResult(
            status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
            checks=checks,
            details=f"Found methods: {method_names}",
        )


class MultiFileVerifier(TaskVerifier):
    """Verify multiple files exist and parse correctly."""

    def __init__(
        self,
        workspace_dir: Path,
        required_files: list[str],
        content_checks: dict[str, list[str]] | None = None,
    ):
        super().__init__(workspace_dir)
        self.required_files = required_files
        self.content_checks = content_checks or {}

    def verify(self) -> VerificationResult:
        checks: dict[str, bool] = {}

        for file_path in self.required_files:
            full_path = self.workspace_dir / file_path
            checks[f"{file_path}_exists"] = full_path.exists()

            if full_path.exists() and file_path.endswith(".py"):
                try:
                    ast.parse(full_path.read_text())
                    checks[f"{file_path}_parses"] = True
                except SyntaxError:
                    checks[f"{file_path}_parses"] = False

            # Check content if specified
            if file_path in self.content_checks and full_path.exists():
                content = full_path.read_text()
                for pattern in self.content_checks[file_path]:
                    checks[f"{file_path}_contains_{pattern[:20]}"] = pattern in content

        passed = all(checks.values())
        return VerificationResult(
            status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
            checks=checks,
        )


class TryExceptVerifier(TaskVerifier):
    """Verify a file contains try-except blocks."""

    def __init__(
        self,
        workspace_dir: Path,
        file_path: str,
        min_try_blocks: int = 1,
    ):
        super().__init__(workspace_dir)
        self.file_path = file_path
        self.min_try_blocks = min_try_blocks

    def verify(self) -> VerificationResult:
        checks: dict[str, bool] = {}
        full_path = self.workspace_dir / self.file_path

        if not full_path.exists():
            return VerificationResult(
                status=VerificationStatus.FAILED,
                checks={"file_exists": False},
            )

        try:
            tree = ast.parse(full_path.read_text())
            checks["file_parses"] = True
        except SyntaxError:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                checks={"file_parses": False},
            )

        # Count try blocks
        try_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Try))
        checks["has_try_blocks"] = try_count >= self.min_try_blocks

        passed = all(checks.values())
        return VerificationResult(
            status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
            checks=checks,
            details=f"Found {try_count} try-except blocks",
        )


# =============================================================================
# Extended Benchmark Tasks (20 total)
# =============================================================================


# Tasks 1-5 are from BENCHMARK_TASKS
# Tasks 6-20 are extensions below


@dataclass
class ExtendedBenchmarkTask:
    """An extended benchmark task definition."""

    id: str
    level: int  # 1-10 for complexity
    category: str  # "file", "function", "class", "refactor", "test", "config"
    description: str
    setup_func: Callable[[Path], dict[str, Any]]
    verify_func: Callable[[Path, dict[str, Any]], VerificationResult]
    success_criteria: list[str]


def setup_error_handling(workspace_dir: Path) -> dict[str, Any]:
    """Set up fixture for adding error handling."""
    src_dir = workspace_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    content = '''"""API client module."""

import httpx


def fetch_data(url: str) -> dict:
    """Fetch data from a URL."""
    response = httpx.get(url)
    return response.json()


def post_data(url: str, data: dict) -> dict:
    """Post data to a URL."""
    response = httpx.post(url, json=data)
    return response.json()
'''
    (src_dir / "client.py").write_text(content)
    return {"file_path": "src/client.py"}


def verify_error_handling(workspace_dir: Path, params: dict[str, Any]) -> VerificationResult:
    """Verify error handling was added."""
    verifier = TryExceptVerifier(
        workspace_dir,
        params["file_path"],
        min_try_blocks=2,
    )
    return verifier.verify()


def setup_api_endpoint(workspace_dir: Path) -> dict[str, Any]:
    """Set up fixture for creating API endpoint."""
    src_dir = workspace_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    (src_dir / "__init__.py").write_text("")
    (src_dir / "models.py").write_text('''"""Data models."""

from dataclasses import dataclass


@dataclass
class User:
    id: int
    name: str
    email: str
''')
    return {
        "endpoint_file": "src/api.py",
        "expected_functions": ["get_users", "create_user"],
    }


def verify_api_endpoint(workspace_dir: Path, params: dict[str, Any]) -> VerificationResult:
    """Verify API endpoint was created."""
    verifier = MultiFileVerifier(
        workspace_dir,
        [params["endpoint_file"]],
        {params["endpoint_file"]: params["expected_functions"]},
    )
    return verifier.verify()


def setup_logging(workspace_dir: Path) -> dict[str, Any]:
    """Set up fixture for adding logging."""
    src_dir = workspace_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    content = '''"""Service module."""


class UserService:
    """User management service."""

    def __init__(self, db):
        self.db = db

    def get_user(self, user_id: int):
        return self.db.find_user(user_id)

    def create_user(self, name: str, email: str):
        return self.db.create_user(name, email)

    def delete_user(self, user_id: int):
        return self.db.delete_user(user_id)
'''
    (src_dir / "service.py").write_text(content)
    return {"file_path": "src/service.py"}


def verify_logging(workspace_dir: Path, params: dict[str, Any]) -> VerificationResult:
    """Verify logging was added."""
    full_path = workspace_dir / params["file_path"]
    checks: dict[str, bool] = {}

    if not full_path.exists():
        return VerificationResult(
            status=VerificationStatus.FAILED,
            checks={"file_exists": False},
        )

    content = full_path.read_text()

    try:
        ast.parse(content)
        checks["file_parses"] = True
    except SyntaxError:
        return VerificationResult(
            status=VerificationStatus.FAILED,
            checks={"file_parses": False},
        )

    # Check for logging import
    checks["has_logging_import"] = "import logging" in content or "from logging" in content

    # Check for logger usage
    checks["uses_logger"] = "logger." in content.lower() or "logging." in content

    passed = all(checks.values())
    return VerificationResult(
        status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
        checks=checks,
    )


def setup_config_system(workspace_dir: Path) -> dict[str, Any]:
    """Set up fixture for configuration system."""
    src_dir = workspace_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "__init__.py").write_text("")

    return {
        "config_file": "src/config.py",
        "expected_classes": ["Config", "Settings"],
    }


def verify_config_system(workspace_dir: Path, params: dict[str, Any]) -> VerificationResult:
    """Verify configuration system was created."""
    verifier = ClassMethodVerifier(
        workspace_dir,
        params["config_file"],
        "Config",
        ["load", "save"],
    )
    result1 = verifier.verify()

    # Also check file exists
    full_path = workspace_dir / params["config_file"]
    if not full_path.exists():
        return VerificationResult(
            status=VerificationStatus.FAILED,
            checks={"file_exists": False},
        )

    return result1


def setup_caching(workspace_dir: Path) -> dict[str, Any]:
    """Set up fixture for caching layer."""
    src_dir = workspace_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    (src_dir / "data.py").write_text('''"""Data access layer."""


def get_user(user_id: int) -> dict:
    """Get user from database."""
    # Simulated database call
    return {"id": user_id, "name": "Test"}


def get_product(product_id: int) -> dict:
    """Get product from database."""
    return {"id": product_id, "name": "Product"}
''')

    return {
        "cache_file": "src/cache.py",
        "data_file": "src/data.py",
    }


def verify_caching(workspace_dir: Path, params: dict[str, Any]) -> VerificationResult:
    """Verify caching layer was implemented."""
    verifier = ClassMethodVerifier(
        workspace_dir,
        params["cache_file"],
        "Cache",
        ["get", "set", "delete", "clear"],
    )
    return verifier.verify()


def setup_data_validation(workspace_dir: Path) -> dict[str, Any]:
    """Set up fixture for data validation."""
    src_dir = workspace_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    return {
        "validator_file": "src/validators.py",
        "expected_functions": ["validate_email", "validate_phone"],
    }


def verify_data_validation(workspace_dir: Path, params: dict[str, Any]) -> VerificationResult:
    """Verify data validation was created."""
    verifier = MultiFileVerifier(
        workspace_dir,
        [params["validator_file"]],
    )
    result = verifier.verify()

    if not result.passed:
        return result

    # Check for validation functions
    full_path = workspace_dir / params["validator_file"]
    try:
        tree = ast.parse(full_path.read_text())
        func_names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        checks = result.checks.copy()
        for func in params["expected_functions"]:
            checks[f"has_{func}"] = func in func_names

        passed = all(checks.values())
        return VerificationResult(
            status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
            checks=checks,
        )
    except SyntaxError:
        return VerificationResult(
            status=VerificationStatus.FAILED,
            checks={"file_parses": False},
        )


def setup_decorator(workspace_dir: Path) -> dict[str, Any]:
    """Set up fixture for creating decorator."""
    src_dir = workspace_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    return {
        "decorator_file": "src/decorators.py",
        "decorator_name": "retry",
    }


def verify_decorator(workspace_dir: Path, params: dict[str, Any]) -> VerificationResult:
    """Verify decorator was created."""
    full_path = workspace_dir / params["decorator_file"]
    checks: dict[str, bool] = {}

    if not full_path.exists():
        return VerificationResult(
            status=VerificationStatus.FAILED,
            checks={"file_exists": False},
        )

    content = full_path.read_text()

    try:
        tree = ast.parse(content)
        checks["file_parses"] = True
    except SyntaxError:
        return VerificationResult(
            status=VerificationStatus.FAILED,
            checks={"file_parses": False},
        )

    # Check for decorator function
    func_names = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    checks["decorator_exists"] = params["decorator_name"] in func_names

    # Check for wrapper pattern
    checks["has_wrapper"] = "wrapper" in content or "functools.wraps" in content

    passed = all(checks.values())
    return VerificationResult(
        status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
        checks=checks,
    )


def setup_context_manager(workspace_dir: Path) -> dict[str, Any]:
    """Set up fixture for context manager."""
    src_dir = workspace_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    return {
        "file_path": "src/context.py",
        "class_name": "Timer",
    }


def verify_context_manager(workspace_dir: Path, params: dict[str, Any]) -> VerificationResult:
    """Verify context manager was created."""
    verifier = ClassMethodVerifier(
        workspace_dir,
        params["file_path"],
        params["class_name"],
        ["__enter__", "__exit__"],
    )
    return verifier.verify()


def setup_async_handler(workspace_dir: Path) -> dict[str, Any]:
    """Set up fixture for async handler."""
    src_dir = workspace_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    return {
        "file_path": "src/async_handlers.py",
        "function_name": "handle_request",
    }


def verify_async_handler(workspace_dir: Path, params: dict[str, Any]) -> VerificationResult:
    """Verify async handler was created."""
    verifier = FunctionSignatureVerifier(
        workspace_dir,
        params["file_path"],
        params["function_name"],
        expected_params=["request"],
        should_be_async=True,
    )
    return verifier.verify()


def setup_cli(workspace_dir: Path) -> dict[str, Any]:
    """Set up fixture for CLI creation."""
    src_dir = workspace_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    return {
        "cli_file": "src/cli.py",
        "expected_commands": ["main", "run", "version"],
    }


def verify_cli(workspace_dir: Path, params: dict[str, Any]) -> VerificationResult:
    """Verify CLI was created."""
    full_path = workspace_dir / params["cli_file"]
    checks: dict[str, bool] = {}

    if not full_path.exists():
        return VerificationResult(
            status=VerificationStatus.FAILED,
            checks={"file_exists": False},
        )

    content = full_path.read_text()

    try:
        tree = ast.parse(content)
        checks["file_parses"] = True
    except SyntaxError:
        return VerificationResult(
            status=VerificationStatus.FAILED,
            checks={"file_parses": False},
        )

    # Check for argparse or click
    checks["has_cli_lib"] = "argparse" in content or "click" in content or "typer" in content

    func_names = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    # At least main function
    checks["has_main"] = "main" in func_names

    passed = all(checks.values())
    return VerificationResult(
        status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
        checks=checks,
        details=f"Found functions: {func_names}",
    )


def setup_type_hints(workspace_dir: Path) -> dict[str, Any]:
    """Set up fixture for adding type hints."""
    src_dir = workspace_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    content = '''"""Utility functions without type hints."""


def add(a, b):
    return a + b


def concat(items):
    return "".join(str(i) for i in items)


def process(data, options=None):
    if options is None:
        options = {}
    return {"data": data, "options": options}
'''
    (src_dir / "utils.py").write_text(content)

    return {"file_path": "src/utils.py"}


def verify_type_hints(workspace_dir: Path, params: dict[str, Any]) -> VerificationResult:
    """Verify type hints were added."""
    full_path = workspace_dir / params["file_path"]
    checks: dict[str, bool] = {}

    if not full_path.exists():
        return VerificationResult(
            status=VerificationStatus.FAILED,
            checks={"file_exists": False},
        )

    content = full_path.read_text()

    try:
        ast.parse(content)
        checks["file_parses"] = True
    except SyntaxError:
        return VerificationResult(
            status=VerificationStatus.FAILED,
            checks={"file_parses": False},
        )

    # Check for type annotations
    checks["has_return_type"] = "->" in content
    checks["has_param_type"] = ": int" in content or ": str" in content or ": list" in content

    passed = all(checks.values())
    return VerificationResult(
        status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
        checks=checks,
    )


def setup_docstrings(workspace_dir: Path) -> dict[str, Any]:
    """Set up fixture for adding docstrings."""
    src_dir = workspace_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    content = """

def helper(x):
    return x * 2


def process(data):
    result = []
    for item in data:
        result.append(helper(item))
    return result


class Handler:
    def __init__(self, name):
        self.name = name

    def handle(self, event):
        return f"{self.name}: {event}"
"""
    (src_dir / "module.py").write_text(content)

    return {"file_path": "src/module.py"}


def verify_docstrings(workspace_dir: Path, params: dict[str, Any]) -> VerificationResult:
    """Verify docstrings were added."""
    full_path = workspace_dir / params["file_path"]
    checks: dict[str, bool] = {}

    if not full_path.exists():
        return VerificationResult(
            status=VerificationStatus.FAILED,
            checks={"file_exists": False},
        )

    content = full_path.read_text()

    try:
        tree = ast.parse(content)
        checks["file_parses"] = True
    except SyntaxError:
        return VerificationResult(
            status=VerificationStatus.FAILED,
            checks={"file_parses": False},
        )

    # Count docstrings
    docstring_count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                docstring_count += 1

    # Should have at least 3 docstrings (2 functions + 1 class + methods)
    checks["has_docstrings"] = docstring_count >= 3

    passed = all(checks.values())
    return VerificationResult(
        status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
        checks=checks,
        details=f"Found {docstring_count} docstrings",
    )


def setup_fixture_project(workspace_dir: Path) -> dict[str, Any]:
    """Set up fixture for creating test fixtures."""
    src_dir = workspace_dir / "src"
    tests_dir = workspace_dir / "tests"
    src_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    (src_dir / "__init__.py").write_text("")
    (tests_dir / "__init__.py").write_text("")
    (tests_dir / "conftest.py").write_text('"""Test configuration."""')

    (src_dir / "models.py").write_text('''"""Data models."""

from dataclasses import dataclass


@dataclass
class User:
    id: int
    name: str
    email: str
''')

    return {
        "conftest_file": "tests/conftest.py",
        "expected_fixtures": ["user", "db"],
    }


def verify_fixture_project(workspace_dir: Path, params: dict[str, Any]) -> VerificationResult:
    """Verify test fixtures were created."""
    full_path = workspace_dir / params["conftest_file"]
    checks: dict[str, bool] = {}

    if not full_path.exists():
        return VerificationResult(
            status=VerificationStatus.FAILED,
            checks={"file_exists": False},
        )

    content = full_path.read_text()

    try:
        ast.parse(content)
        checks["file_parses"] = True
    except SyntaxError:
        return VerificationResult(
            status=VerificationStatus.FAILED,
            checks={"file_parses": False},
        )

    # Check for pytest.fixture decorator
    checks["has_fixture_decorator"] = "@pytest.fixture" in content or "pytest.fixture" in content

    passed = all(checks.values())
    return VerificationResult(
        status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
        checks=checks,
    )


# =============================================================================
# Extended Task Definitions (Tasks 6-20)
# =============================================================================


EXTENDED_TASKS: list[ExtendedBenchmarkTask] = [
    # Task 6: Add error handling
    ExtendedBenchmarkTask(
        id="T06-error-handling",
        level=6,
        category="refactor",
        description="Add try-except error handling to fetch_data and post_data functions in src/client.py",
        setup_func=setup_error_handling,
        verify_func=verify_error_handling,
        success_criteria=[
            "File parses correctly",
            "Has at least 2 try-except blocks",
        ],
    ),
    # Task 7: Create API endpoint
    ExtendedBenchmarkTask(
        id="T07-api-endpoint",
        level=6,
        category="function",
        description="Create src/api.py with get_users and create_user endpoint functions",
        setup_func=setup_api_endpoint,
        verify_func=verify_api_endpoint,
        success_criteria=[
            "File exists and parses",
            "Has get_users function",
            "Has create_user function",
        ],
    ),
    # Task 8: Add logging
    ExtendedBenchmarkTask(
        id="T08-add-logging",
        level=7,
        category="refactor",
        description="Add logging to all methods in UserService class in src/service.py",
        setup_func=setup_logging,
        verify_func=verify_logging,
        success_criteria=[
            "File parses correctly",
            "Has logging import",
            "Uses logger in methods",
        ],
    ),
    # Task 9: Configuration system
    ExtendedBenchmarkTask(
        id="T09-config-system",
        level=8,
        category="class",
        description="Create src/config.py with a Config class that has load and save methods",
        setup_func=setup_config_system,
        verify_func=verify_config_system,
        success_criteria=[
            "File exists and parses",
            "Config class exists",
            "Has load method",
            "Has save method",
        ],
    ),
    # Task 10: Caching layer
    ExtendedBenchmarkTask(
        id="T10-caching-layer",
        level=8,
        category="class",
        description="Create src/cache.py with Cache class having get, set, delete, and clear methods",
        setup_func=setup_caching,
        verify_func=verify_caching,
        success_criteria=[
            "File exists and parses",
            "Cache class exists",
            "Has get, set, delete, clear methods",
        ],
    ),
    # Task 11: Data validation
    ExtendedBenchmarkTask(
        id="T11-data-validation",
        level=5,
        category="function",
        description="Create src/validators.py with validate_email and validate_phone functions",
        setup_func=setup_data_validation,
        verify_func=verify_data_validation,
        success_criteria=[
            "File exists and parses",
            "Has validate_email function",
            "Has validate_phone function",
        ],
    ),
    # Task 12: Decorator
    ExtendedBenchmarkTask(
        id="T12-decorator",
        level=7,
        category="function",
        description="Create src/decorators.py with a retry decorator that retries failed calls",
        setup_func=setup_decorator,
        verify_func=verify_decorator,
        success_criteria=[
            "File exists and parses",
            "retry function exists",
            "Uses wrapper pattern",
        ],
    ),
    # Task 13: Context manager
    ExtendedBenchmarkTask(
        id="T13-context-manager",
        level=6,
        category="class",
        description="Create src/context.py with a Timer context manager class",
        setup_func=setup_context_manager,
        verify_func=verify_context_manager,
        success_criteria=[
            "File exists and parses",
            "Timer class exists",
            "Has __enter__ and __exit__ methods",
        ],
    ),
    # Task 14: Async handler
    ExtendedBenchmarkTask(
        id="T14-async-handler",
        level=7,
        category="function",
        description="Create src/async_handlers.py with an async handle_request function",
        setup_func=setup_async_handler,
        verify_func=verify_async_handler,
        success_criteria=[
            "File exists and parses",
            "handle_request is async",
            "Has request parameter",
        ],
    ),
    # Task 15: CLI
    ExtendedBenchmarkTask(
        id="T15-cli",
        level=8,
        category="file",
        description="Create src/cli.py with command-line interface using argparse or click",
        setup_func=setup_cli,
        verify_func=verify_cli,
        success_criteria=[
            "File exists and parses",
            "Uses argparse or click",
            "Has main function",
        ],
    ),
    # Task 16: Type hints
    ExtendedBenchmarkTask(
        id="T16-type-hints",
        level=4,
        category="refactor",
        description="Add type hints to all functions in src/utils.py",
        setup_func=setup_type_hints,
        verify_func=verify_type_hints,
        success_criteria=[
            "File parses correctly",
            "Has return type annotations",
            "Has parameter type annotations",
        ],
    ),
    # Task 17: Docstrings
    ExtendedBenchmarkTask(
        id="T17-docstrings",
        level=3,
        category="refactor",
        description="Add docstrings to all functions and classes in src/module.py",
        setup_func=setup_docstrings,
        verify_func=verify_docstrings,
        success_criteria=[
            "File parses correctly",
            "Functions have docstrings",
            "Classes have docstrings",
        ],
    ),
    # Task 18: Test fixtures
    ExtendedBenchmarkTask(
        id="T18-test-fixtures",
        level=6,
        category="test",
        description="Add pytest fixtures for user and db to tests/conftest.py",
        setup_func=setup_fixture_project,
        verify_func=verify_fixture_project,
        success_criteria=[
            "conftest.py exists and parses",
            "Has pytest.fixture decorator",
        ],
    ),
    # Task 19: Multi-file refactor (simpler setup)
    ExtendedBenchmarkTask(
        id="T19-split-module",
        level=9,
        category="refactor",
        description="Split src/app.py into src/models.py and src/services.py",
        setup_func=lambda p: (
            (p / "src").mkdir(parents=True, exist_ok=True),
            (p / "src" / "app.py").write_text('''"""Application module."""

class User:
    def __init__(self, name):
        self.name = name

class UserService:
    def get_user(self, id):
        return User("test")
'''),
            {"files": ["src/models.py", "src/services.py"]},
        )[-1],
        verify_func=lambda p, params: MultiFileVerifier(p, params["files"]).verify(),
        success_criteria=[
            "src/models.py exists",
            "src/services.py exists",
            "Both files parse correctly",
        ],
    ),
    # Task 20: Integration test
    ExtendedBenchmarkTask(
        id="T20-integration-test",
        level=9,
        category="test",
        description="Create tests/test_integration.py with integration tests for the API",
        setup_func=lambda p: (
            (p / "tests").mkdir(parents=True, exist_ok=True),
            (p / "tests" / "__init__.py").write_text(""),
            {"test_file": "tests/test_integration.py"},
        )[-1],
        verify_func=lambda p, params: MultiFileVerifier(
            p,
            [params["test_file"]],
            {params["test_file"]: ["def test_", "pytest"]},
        ).verify(),
        success_criteria=[
            "Test file exists",
            "Has test functions",
            "Uses pytest",
        ],
    ),
]


# =============================================================================
# Extended Benchmark Harness
# =============================================================================


class ExtendedBenchmarkHarness(BenchmarkHarness):
    """Extended harness for running all 20 benchmark tasks."""

    def get_all_tasks(self) -> list[tuple[str, int, str]]:
        """Get all tasks (base + extended)."""
        tasks = []

        # Base tasks (1-5)
        for task in BENCHMARK_TASKS:
            tasks.append((task.id, task.level.value, "base"))

        # Extended tasks (6-20)
        for task in EXTENDED_TASKS:
            tasks.append((task.id, task.level, "extended"))

        return tasks

    async def run_extended_task(self, task: ExtendedBenchmarkTask) -> BenchmarkResult:
        """Run an extended benchmark task."""
        import time

        workspace_dir, temp_dir = self.create_workspace()

        try:
            # Set up fixture
            fixture_params = task.setup_func(workspace_dir)

            # Create agent
            from ai_infra.llm import Agent
            from ai_infra.llm.workspace import Workspace

            agent = Agent(
                deep=True,
                workspace=Workspace(workspace_dir, mode=self.workspace_mode),  # type: ignore[arg-type]
                model_name=self.model_name,
            )

            # Build prompt
            criteria_text = "\n".join(f"- {c}" for c in task.success_criteria)
            prompt = f"""You are an expert software engineer. Complete this task:

## Task
{task.description}

## Success Criteria
{criteria_text}

## Instructions
1. Use ls and read_file to understand the current state
2. Use write_file or edit_file to make changes
3. Verify your changes work before finishing

Begin."""

            # Run the task
            start_time = time.time()
            result = await agent.arun(prompt)
            duration = time.time() - start_time

            # Verify
            verification = task.verify_func(workspace_dir, fixture_params)

            return BenchmarkResult(
                task_id=task.id,
                level=TaskLevel(min(task.level, 5)),  # Map to TaskLevel enum
                passed=verification.passed,
                verification=verification,
                duration_seconds=duration,
                agent_response=str(result) if result else None,
            )

        except Exception as e:
            return BenchmarkResult(
                task_id=task.id,
                level=TaskLevel.L5,
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

    async def run_all_extended(self) -> list[BenchmarkResult]:
        """Run all 20 benchmark tasks."""
        results = []

        # Run base tasks (1-5)
        base_results = await self.run_all()
        results.extend(base_results)

        # Run extended tasks (6-20)
        for task in EXTENDED_TASKS:
            result = await self.run_extended_task(task)
            results.append(result)

        return results


def get_extended_benchmark_summary(results: list[BenchmarkResult]) -> dict[str, Any]:
    """Generate summary for extended benchmark results."""
    base_summary = get_benchmark_summary(results)

    # Add category breakdown
    categories: dict[str, dict[str, int]] = {}
    for result in results:
        # Determine category from task ID
        task_id = result.task_id
        if task_id.startswith("L"):
            category = "base"
        else:
            # Find in extended tasks
            for task in EXTENDED_TASKS:
                if task.id == task_id:
                    category = task.category
                    break
            else:
                category = "unknown"

        if category not in categories:
            categories[category] = {"passed": 0, "failed": 0}

        if result.passed:
            categories[category]["passed"] += 1
        else:
            categories[category]["failed"] += 1

    base_summary["by_category"] = categories

    return base_summary
