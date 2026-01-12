"""Unit tests for benchmark task verifiers.

These tests validate that the verifiers correctly detect success/failure
for each task level, without requiring actual LLM calls.
"""

from __future__ import annotations

from pathlib import Path

from tests.unit.executor.benchmark_tasks import (
    BENCHMARK_TASKS,
    BenchmarkHarness,
    L1Verifier,
    L2Verifier,
    L3Verifier,
    L4Verifier,
    L5Verifier,
    TaskLevel,
    VerificationStatus,
    setup_l1_fixture,
    setup_l2_fixture,
    setup_l3_fixture,
    setup_l4_fixture,
    setup_l5_fixture,
)


class TestL1Verifier:
    """Test L1: Create a new empty file."""

    def test_passes_when_file_exists(self, tmp_path: Path):
        """Verifier passes when expected file exists."""
        # Create the expected file
        expected_path = "src/utils/helpers.py"
        full_path = tmp_path / expected_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text('"""Helpers module."""\n')

        verifier = L1Verifier(tmp_path, expected_path)
        result = verifier.verify()

        assert result.passed
        assert result.status == VerificationStatus.PASSED
        assert result.checks["file_exists"] is True
        assert result.checks["is_file"] is True

    def test_fails_when_file_missing(self, tmp_path: Path):
        """Verifier fails when file doesn't exist."""
        verifier = L1Verifier(tmp_path, "src/utils/helpers.py")
        result = verifier.verify()

        assert not result.passed
        assert result.status == VerificationStatus.FAILED
        assert result.checks["file_exists"] is False

    def test_fails_when_path_is_directory(self, tmp_path: Path):
        """Verifier fails when path is a directory, not a file."""
        dir_path = tmp_path / "src" / "utils" / "helpers.py"
        dir_path.mkdir(parents=True, exist_ok=True)

        verifier = L1Verifier(tmp_path, "src/utils/helpers.py")
        result = verifier.verify()

        assert not result.passed
        assert result.checks["is_file"] is False


class TestL2Verifier:
    """Test L2: Add a function to existing file."""

    def test_passes_when_function_exists(self, tmp_path: Path):
        """Verifier passes when function exists in file."""
        file_path = "src/formatters.py"
        full_path = tmp_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text('''"""Formatters."""

def format_date(dt):
    return dt.strftime("%Y-%m-%d")

def format_currency(amount: float, currency: str = "USD") -> str:
    return f"${amount:,.2f}"
''')

        verifier = L2Verifier(tmp_path, file_path, "format_currency")
        result = verifier.verify()

        assert result.passed
        assert result.checks["file_exists"] is True
        assert result.checks["file_parses"] is True
        assert result.checks["function_exists"] is True

    def test_fails_when_function_missing(self, tmp_path: Path):
        """Verifier fails when function doesn't exist."""
        file_path = "src/formatters.py"
        full_path = tmp_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text('''"""Formatters."""

def format_date(dt):
    return dt.strftime("%Y-%m-%d")
''')

        verifier = L2Verifier(tmp_path, file_path, "format_currency")
        result = verifier.verify()

        assert not result.passed
        assert result.checks["function_exists"] is False

    def test_fails_when_syntax_error(self, tmp_path: Path):
        """Verifier fails when file has syntax error."""
        file_path = "src/formatters.py"
        full_path = tmp_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text("""def format_currency(amount:
    # Missing closing paren
""")

        verifier = L2Verifier(tmp_path, file_path, "format_currency")
        result = verifier.verify()

        assert not result.passed
        assert result.checks["file_parses"] is False


class TestL3Verifier:
    """Test L3: Modify function signature + update callers."""

    def test_passes_when_signature_and_callers_updated(self, tmp_path: Path):
        """Verifier passes when function signature updated and callers work."""
        # Create processor.py with updated signature
        (tmp_path / "src").mkdir(parents=True, exist_ok=True)
        (tmp_path / "src" / "processor.py").write_text('''"""Processor."""

def process_data(data: list, validate: bool = True) -> list:
    if validate:
        data = [d for d in data if d is not None]
    return [{"processed": d} for d in data]
''')

        # Create handlers.py that calls with new param
        (tmp_path / "src" / "handlers.py").write_text('''"""Handlers."""

from src.processor import process_data

def handle_request(items: list) -> dict:
    processed = process_data(items, validate=True)
    return {"status": "ok", "data": processed}
''')

        verifier = L3Verifier(
            tmp_path,
            target_file="src/processor.py",
            function_name="process_data",
            expected_params=["data", "validate"],
            caller_files=["src/handlers.py"],
        )
        result = verifier.verify()

        assert result.passed
        assert result.checks["target_parses"] is True
        assert result.checks["params_updated"] is True
        assert result.checks["caller_src/handlers.py_parses"] is True

    def test_fails_when_param_missing(self, tmp_path: Path):
        """Verifier fails when expected parameter is missing."""
        (tmp_path / "src").mkdir(parents=True, exist_ok=True)
        (tmp_path / "src" / "processor.py").write_text('''"""Processor."""

def process_data(data: list) -> list:  # Missing validate param
    return [{"processed": d} for d in data]
''')

        (tmp_path / "src" / "handlers.py").write_text('''"""Handlers."""

from src.processor import process_data

def handle_request(items: list) -> dict:
    return {"data": process_data(items)}
''')

        verifier = L3Verifier(
            tmp_path,
            target_file="src/processor.py",
            function_name="process_data",
            expected_params=["data", "validate"],
            caller_files=["src/handlers.py"],
        )
        result = verifier.verify()

        assert not result.passed
        assert result.checks["params_updated"] is False


class TestL4Verifier:
    """Test L4: Create new module with tests."""

    def test_passes_when_module_and_tests_exist(self, tmp_path: Path):
        """Verifier passes when module and tests are complete."""
        (tmp_path / "src").mkdir(parents=True, exist_ok=True)
        (tmp_path / "tests").mkdir(parents=True, exist_ok=True)

        # Create cache module
        (tmp_path / "src" / "cache.py").write_text('''"""Cache module."""

from typing import Any


class Cache:
    """Simple in-memory cache."""

    def __init__(self):
        self._data = {}

    def get(self, key: str) -> Any:
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def clear(self) -> None:
        self._data.clear()
''')

        # Create test file
        (tmp_path / "tests" / "test_cache.py").write_text('''"""Cache tests."""

from src.cache import Cache


def test_set_and_get():
    cache = Cache()
    cache.set("key", "value")
    assert cache.get("key") == "value"


def test_clear():
    cache = Cache()
    cache.set("key", "value")
    cache.clear()
    assert cache.get("key") is None
''')

        verifier = L4Verifier(
            tmp_path,
            module_path="src/cache.py",
            test_path="tests/test_cache.py",
            expected_classes=["Cache"],
        )
        result = verifier.verify()

        assert result.passed
        assert result.checks["module_parses"] is True
        assert result.checks["class_Cache_exists"] is True
        assert result.checks["test_exists"] is True
        assert result.checks["test_parses"] is True
        assert result.checks["has_test_functions"] is True

    def test_fails_when_no_test_functions(self, tmp_path: Path):
        """Verifier fails when test file has no test_ functions."""
        (tmp_path / "src").mkdir(parents=True, exist_ok=True)
        (tmp_path / "tests").mkdir(parents=True, exist_ok=True)

        (tmp_path / "src" / "cache.py").write_text("""class Cache:
    pass
""")

        (tmp_path / "tests" / "test_cache.py").write_text('''"""Tests - but no test functions!"""

def helper():  # Not a test function
    pass
''')

        verifier = L4Verifier(
            tmp_path,
            module_path="src/cache.py",
            test_path="tests/test_cache.py",
            expected_classes=["Cache"],
        )
        result = verifier.verify()

        assert not result.passed
        assert result.checks["has_test_functions"] is False


class TestL5Verifier:
    """Test L5: Refactor - extract class to new file."""

    def test_passes_when_class_extracted(self, tmp_path: Path):
        """Verifier passes when class extracted and imports updated."""
        (tmp_path / "src").mkdir(parents=True, exist_ok=True)

        # Create new logger.py with extracted class
        (tmp_path / "src" / "logger.py").write_text('''"""Logger module."""


class Logger:
    def __init__(self, name: str):
        self.name = name

    def info(self, msg: str) -> None:
        print(f"[INFO] {self.name}: {msg}")
''')

        # Update app.py to import from logger
        (tmp_path / "src" / "app.py").write_text('''"""App module."""

from src.logger import Logger


class App:
    def __init__(self, name: str):
        self.logger = Logger(name)

    def run(self) -> None:
        self.logger.info("Started")
''')

        # Update main.py
        (tmp_path / "src" / "main.py").write_text('''"""Main entry."""

from src.logger import Logger
from src.app import App


def main():
    logger = Logger("main")
    app = App("test")
''')

        verifier = L5Verifier(
            tmp_path,
            original_file="src/app.py",
            new_file="src/logger.py",
            class_name="Logger",
            import_files=["src/app.py", "src/main.py"],
        )
        result = verifier.verify()

        assert result.passed
        assert result.checks["new_file_parses"] is True
        assert result.checks["class_in_new_file"] is True
        assert result.checks["original_parses"] is True

    def test_fails_when_new_file_missing(self, tmp_path: Path):
        """Verifier fails when extracted file doesn't exist."""
        (tmp_path / "src").mkdir(parents=True, exist_ok=True)

        # Only create app.py, no logger.py
        (tmp_path / "src" / "app.py").write_text("""class App:
    pass
""")

        verifier = L5Verifier(
            tmp_path,
            original_file="src/app.py",
            new_file="src/logger.py",
            class_name="Logger",
            import_files=[],
        )
        result = verifier.verify()

        assert not result.passed
        assert "new_file_exists" in result.checks or result.error is not None


class TestBenchmarkTasks:
    """Test the benchmark task definitions."""

    def test_all_levels_have_tasks(self):
        """Each level should have at least one task."""
        levels_with_tasks = {task.level for task in BENCHMARK_TASKS}
        for level in TaskLevel:
            assert level in levels_with_tasks, f"No task for {level}"

    def test_task_ids_unique(self):
        """All task IDs should be unique."""
        ids = [task.id for task in BENCHMARK_TASKS]
        assert len(ids) == len(set(ids)), "Duplicate task IDs found"

    def test_tasks_have_success_criteria(self):
        """Each task should have at least one success criterion."""
        for task in BENCHMARK_TASKS:
            assert len(task.success_criteria) > 0, f"Task {task.id} has no success criteria"


class TestFixtureSetup:
    """Test fixture setup functions."""

    def test_l1_fixture(self, tmp_path: Path):
        """L1 fixture creates src directory."""
        params = setup_l1_fixture(tmp_path)
        assert (tmp_path / "src").is_dir()
        assert "expected_path" in params

    def test_l2_fixture(self, tmp_path: Path):
        """L2 fixture creates formatters.py with existing function."""
        params = setup_l2_fixture(tmp_path)
        assert (tmp_path / "src" / "formatters.py").exists()
        content = (tmp_path / "src" / "formatters.py").read_text()
        assert "def format_date" in content
        assert "file_path" in params

    def test_l3_fixture(self, tmp_path: Path):
        """L3 fixture creates processor.py and handlers.py."""
        params = setup_l3_fixture(tmp_path)
        assert (tmp_path / "src" / "processor.py").exists()
        assert (tmp_path / "src" / "handlers.py").exists()
        assert "target_file" in params
        assert "caller_files" in params

    def test_l4_fixture(self, tmp_path: Path):
        """L4 fixture creates src and tests directories."""
        params = setup_l4_fixture(tmp_path)
        assert (tmp_path / "src").is_dir()
        assert (tmp_path / "tests").is_dir()
        assert "module_path" in params
        assert "test_path" in params

    def test_l5_fixture(self, tmp_path: Path):
        """L5 fixture creates app.py with Logger class and main.py."""
        params = setup_l5_fixture(tmp_path)
        assert (tmp_path / "src" / "app.py").exists()
        assert (tmp_path / "src" / "main.py").exists()
        content = (tmp_path / "src" / "app.py").read_text()
        assert "class Logger" in content
        assert "class_name" in params


class TestBenchmarkHarness:
    """Test the benchmark harness."""

    def test_get_task_by_id(self):
        """Can retrieve task by ID."""
        harness = BenchmarkHarness()
        task = harness.get_task("L1-create-file")
        assert task is not None
        assert task.level == TaskLevel.L1

    def test_get_nonexistent_task(self):
        """Returns None for unknown task ID."""
        harness = BenchmarkHarness()
        task = harness.get_task("nonexistent")
        assert task is None

    def test_create_workspace(self):
        """Creates a temporary workspace."""
        harness = BenchmarkHarness()
        workspace_dir, temp_dir = harness.create_workspace()
        try:
            assert workspace_dir.exists()
            assert workspace_dir.is_dir()
        finally:
            temp_dir.cleanup()

    def test_setup_fixture_for_each_level(self, tmp_path: Path):
        """Can set up fixture for each level."""
        harness = BenchmarkHarness()
        for level in TaskLevel:
            # Create a fresh subdirectory for each level
            level_dir = tmp_path / level.name
            level_dir.mkdir()
            params = harness.setup_fixture(level, level_dir)
            assert isinstance(params, dict)
