"""Scenario tests for Shell Tool verification across project types.

Phase 5.3 of EXECUTOR_CLI.md - Scenario Tests.

This module tests end-to-end scenarios for:
- 5.3.1 Python project (pytest discovery)
- 5.3.2 Node.js project (npm test discovery)
- 5.3.3 Rust project (cargo test discovery)
- 5.3.4 Makefile project (make test discovery)
- 5.3.5 Multi-language monorepo

These tests create realistic project structures and verify that:
1. Project type is correctly detected
2. Test command is correctly determined
3. TaskVerifier can run verification levels
4. Verification agent heuristics work correctly
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Mark all tests as scenario tests
pytestmark = [
    pytest.mark.scenario,
    pytest.mark.skipif(
        sys.platform.startswith("win"),
        reason="Scenario tests run on Unix only",
    ),
]


# =============================================================================
# Test Fixtures for Project Scaffolding
# =============================================================================


@pytest.fixture
def python_project(tmp_path: Path) -> Path:
    """Create a realistic Python project structure."""
    project = tmp_path / "python_project"
    project.mkdir()

    # pyproject.toml
    (project / "pyproject.toml").write_text(
        """\
[project]
name = "myapp"
version = "0.1.0"
requires-python = ">=3.11"

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
"""
    )

    # src/myapp/__init__.py
    src = project / "src" / "myapp"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text('__version__ = "0.1.0"\n')

    # src/myapp/calculator.py
    (src / "calculator.py").write_text(
        """\
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b

def subtract(a: int, b: int) -> int:
    \"\"\"Subtract b from a.\"\"\"
    return a - b

def multiply(a: int, b: int) -> int:
    \"\"\"Multiply two numbers.\"\"\"
    return a * b

def divide(a: int, b: int) -> float:
    \"\"\"Divide a by b.\"\"\"
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
"""
    )

    # tests/__init__.py
    tests = project / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("")

    # tests/test_calculator.py
    (tests / "test_calculator.py").write_text(
        """\
import pytest
import sys
sys.path.insert(0, str(__file__).rsplit('/tests/', 1)[0] + '/src')

from myapp.calculator import add, subtract, multiply, divide

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0

def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(1, 1) == 0
    assert subtract(0, 5) == -5

def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(-2, 3) == -6
    assert multiply(0, 5) == 0

def test_divide():
    assert divide(6, 2) == 3.0
    assert divide(5, 2) == 2.5

def test_divide_by_zero():
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(5, 0)
"""
    )

    # README.md
    (project / "README.md").write_text(
        """\
# My App

A simple calculator application.

## Testing

Run tests with:
```bash
pytest
```
"""
    )

    return project


@pytest.fixture
def nodejs_project(tmp_path: Path) -> Path:
    """Create a realistic Node.js project structure."""
    project = tmp_path / "nodejs_project"
    project.mkdir()

    # package.json
    (project / "package.json").write_text(
        json.dumps(
            {
                "name": "myapp",
                "version": "1.0.0",
                "main": "src/index.js",
                "scripts": {
                    "test": "node --test tests/",
                    "start": "node src/index.js",
                },
                "devDependencies": {},
            },
            indent=2,
        )
    )

    # src/calculator.js
    src = project / "src"
    src.mkdir()
    (src / "calculator.js").write_text(
        """\
function add(a, b) {
    return a + b;
}

function subtract(a, b) {
    return a - b;
}

function multiply(a, b) {
    return a * b;
}

function divide(a, b) {
    if (b === 0) {
        throw new Error("Cannot divide by zero");
    }
    return a / b;
}

module.exports = { add, subtract, multiply, divide };
"""
    )

    # src/index.js
    (src / "index.js").write_text(
        """\
const { add, subtract, multiply, divide } = require('./calculator');

console.log('Calculator ready');
console.log('2 + 3 =', add(2, 3));
"""
    )

    # tests/calculator.test.js
    tests = project / "tests"
    tests.mkdir()
    (tests / "calculator.test.js").write_text(
        """\
const assert = require('assert');
const { describe, it } = require('node:test');
const { add, subtract, multiply, divide } = require('../src/calculator');

describe('Calculator', () => {
    describe('add', () => {
        it('should add two positive numbers', () => {
            assert.strictEqual(add(2, 3), 5);
        });

        it('should add negative numbers', () => {
            assert.strictEqual(add(-1, 1), 0);
        });
    });

    describe('subtract', () => {
        it('should subtract two numbers', () => {
            assert.strictEqual(subtract(5, 3), 2);
        });
    });

    describe('multiply', () => {
        it('should multiply two numbers', () => {
            assert.strictEqual(multiply(2, 3), 6);
        });
    });

    describe('divide', () => {
        it('should divide two numbers', () => {
            assert.strictEqual(divide(6, 2), 3);
        });

        it('should throw on division by zero', () => {
            assert.throws(() => divide(5, 0), /Cannot divide by zero/);
        });
    });
});
"""
    )

    # README.md
    (project / "README.md").write_text(
        """\
# My App

A simple calculator application in Node.js.

## Testing

Run tests with:
```bash
npm test
```
"""
    )

    return project


@pytest.fixture
def rust_project(tmp_path: Path) -> Path:
    """Create a realistic Rust project structure."""
    project = tmp_path / "rust_project"
    project.mkdir()

    # Cargo.toml
    (project / "Cargo.toml").write_text(
        """\
[package]
name = "myapp"
version = "0.1.0"
edition = "2021"

[dependencies]
"""
    )

    # src/lib.rs
    src = project / "src"
    src.mkdir()
    (src / "lib.rs").write_text(
        """\
/// Add two numbers
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Subtract b from a
pub fn subtract(a: i32, b: i32) -> i32 {
    a - b
}

/// Multiply two numbers
pub fn multiply(a: i32, b: i32) -> i32 {
    a * b
}

/// Divide a by b
pub fn divide(a: i32, b: i32) -> Result<f64, &'static str> {
    if b == 0 {
        Err("Cannot divide by zero")
    } else {
        Ok(a as f64 / b as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
        assert_eq!(add(-1, 1), 0);
    }

    #[test]
    fn test_subtract() {
        assert_eq!(subtract(5, 3), 2);
    }

    #[test]
    fn test_multiply() {
        assert_eq!(multiply(2, 3), 6);
    }

    #[test]
    fn test_divide() {
        assert_eq!(divide(6, 2), Ok(3.0));
    }

    #[test]
    fn test_divide_by_zero() {
        assert_eq!(divide(5, 0), Err("Cannot divide by zero"));
    }
}
"""
    )

    # README.md
    (project / "README.md").write_text(
        """\
# My App

A simple calculator library in Rust.

## Testing

Run tests with:
```bash
cargo test
```
"""
    )

    return project


@pytest.fixture
def go_project(tmp_path: Path) -> Path:
    """Create a realistic Go project structure."""
    project = tmp_path / "go_project"
    project.mkdir()

    # go.mod
    (project / "go.mod").write_text(
        """\
module myapp

go 1.21
"""
    )

    # calculator.go
    (project / "calculator.go").write_text(
        """\
package myapp

import "errors"

// Add returns the sum of a and b
func Add(a, b int) int {
    return a + b
}

// Subtract returns a minus b
func Subtract(a, b int) int {
    return a - b
}

// Multiply returns the product of a and b
func Multiply(a, b int) int {
    return a * b
}

// Divide returns a divided by b
func Divide(a, b int) (float64, error) {
    if b == 0 {
        return 0, errors.New("cannot divide by zero")
    }
    return float64(a) / float64(b), nil
}
"""
    )

    # calculator_test.go
    (project / "calculator_test.go").write_text(
        """\
package myapp

import "testing"

func TestAdd(t *testing.T) {
    result := Add(2, 3)
    if result != 5 {
        t.Errorf("Add(2, 3) = %d; want 5", result)
    }
}

func TestSubtract(t *testing.T) {
    result := Subtract(5, 3)
    if result != 2 {
        t.Errorf("Subtract(5, 3) = %d; want 2", result)
    }
}

func TestMultiply(t *testing.T) {
    result := Multiply(2, 3)
    if result != 6 {
        t.Errorf("Multiply(2, 3) = %d; want 6", result)
    }
}

func TestDivide(t *testing.T) {
    result, err := Divide(6, 2)
    if err != nil {
        t.Errorf("Divide(6, 2) returned error: %v", err)
    }
    if result != 3.0 {
        t.Errorf("Divide(6, 2) = %f; want 3.0", result)
    }
}

func TestDivideByZero(t *testing.T) {
    _, err := Divide(5, 0)
    if err == nil {
        t.Error("Divide(5, 0) should return error")
    }
}
"""
    )

    # README.md
    (project / "README.md").write_text(
        """\
# My App

A simple calculator package in Go.

## Testing

Run tests with:
```bash
go test ./...
```
"""
    )

    return project


@pytest.fixture
def makefile_project(tmp_path: Path) -> Path:
    """Create a project with Makefile-based testing."""
    project = tmp_path / "makefile_project"
    project.mkdir()

    # Makefile
    (project / "Makefile").write_text(
        """\
.PHONY: test build clean

test:
\t@echo "Running tests..."
\t@./run_tests.sh
\t@echo "Tests passed!"

build:
\t@echo "Building..."

clean:
\t@echo "Cleaning..."
"""
    )

    # run_tests.sh (simple test script)
    test_script = project / "run_tests.sh"
    test_script.write_text(
        """\
#!/bin/bash
# Simple test runner

echo "Test 1: Checking file exists"
if [ -f "src/main.sh" ]; then
    echo "  PASS: src/main.sh exists"
else
    echo "  FAIL: src/main.sh not found"
    exit 1
fi

echo "Test 2: Running main script"
output=$(./src/main.sh 2>&1)
if [ $? -eq 0 ]; then
    echo "  PASS: main.sh executed successfully"
else
    echo "  FAIL: main.sh failed"
    exit 1
fi

echo "All tests passed!"
exit 0
"""
    )
    test_script.chmod(0o755)

    # src/main.sh
    src = project / "src"
    src.mkdir()
    main_script = src / "main.sh"
    main_script.write_text(
        """\
#!/bin/bash
echo "Hello, World!"
exit 0
"""
    )
    main_script.chmod(0o755)

    # README.md
    (project / "README.md").write_text(
        """\
# My App

A shell-based application with Makefile.

## Testing

Run tests with:
```bash
make test
```
"""
    )

    return project


@pytest.fixture
def monorepo_project(tmp_path: Path) -> Path:
    """Create a multi-language monorepo structure."""
    project = tmp_path / "monorepo"
    project.mkdir()

    # Root README.md
    (project / "README.md").write_text(
        """\
# Monorepo

A multi-language monorepo with Python, Node.js, and Go services.

## Structure

- `services/api/` - Python FastAPI backend
- `services/web/` - Node.js frontend
- `services/worker/` - Go worker service

## Testing

Run all tests with:
```bash
make test
```
"""
    )

    # Root Makefile
    (project / "Makefile").write_text(
        """\
.PHONY: test test-api test-web test-worker

test: test-api test-web test-worker
\t@echo "All tests passed!"

test-api:
\t@echo "Testing API service..."
\tcd services/api && python -m pytest -q

test-web:
\t@echo "Testing Web service..."
\tcd services/web && npm test

test-worker:
\t@echo "Testing Worker service..."
\tcd services/worker && go test ./...
"""
    )

    # Python API service
    api = project / "services" / "api"
    api.mkdir(parents=True)
    (api / "pyproject.toml").write_text(
        """\
[project]
name = "api"
version = "0.1.0"
"""
    )
    (api / "main.py").write_text(
        """\
def hello():
    return "Hello from API"
"""
    )
    api_tests = api / "tests"
    api_tests.mkdir()
    (api_tests / "__init__.py").write_text("")
    (api_tests / "test_main.py").write_text(
        """\
import sys
sys.path.insert(0, str(__file__).rsplit('/tests/', 1)[0])
from main import hello

def test_hello():
    assert hello() == "Hello from API"
"""
    )

    # Node.js Web service
    web = project / "services" / "web"
    web.mkdir(parents=True)
    (web / "package.json").write_text(
        json.dumps(
            {
                "name": "web",
                "version": "1.0.0",
                "scripts": {"test": "node --test tests/"},
            },
            indent=2,
        )
    )
    (web / "app.js").write_text(
        """\
module.exports = {
    greet: () => "Hello from Web"
};
"""
    )
    web_tests = web / "tests"
    web_tests.mkdir()
    (web_tests / "app.test.js").write_text(
        """\
const assert = require('assert');
const { describe, it } = require('node:test');
const { greet } = require('../app');

describe('App', () => {
    it('should greet', () => {
        assert.strictEqual(greet(), "Hello from Web");
    });
});
"""
    )

    # Go Worker service
    worker = project / "services" / "worker"
    worker.mkdir(parents=True)
    (worker / "go.mod").write_text(
        """\
module worker

go 1.21
"""
    )
    (worker / "worker.go").write_text(
        """\
package worker

func Process() string {
    return "Hello from Worker"
}
"""
    )
    (worker / "worker_test.go").write_text(
        """\
package worker

import "testing"

func TestProcess(t *testing.T) {
    result := Process()
    if result != "Hello from Worker" {
        t.Errorf("Process() = %s; want Hello from Worker", result)
    }
}
"""
    )

    return project


# =============================================================================
# 5.3.1 Scenario: Python Project (pytest discovery)
# =============================================================================


class TestPythonProjectScenario:
    """Scenario tests for Python project with pytest (5.3.1)."""

    def test_detects_python_project_type(self, python_project: Path) -> None:
        """Detect Python project from pyproject.toml."""
        from ai_infra.executor.verifier import ProjectType, detect_project_type

        project_type = detect_project_type(python_project)

        assert project_type == ProjectType.PYTHON

    def test_gets_pytest_command(self, python_project: Path) -> None:
        """Get pytest command for Python project."""
        from ai_infra.executor.verifier import ProjectType, get_test_command

        cmd = get_test_command(ProjectType.PYTHON, python_project)

        assert cmd is not None
        assert "pytest" in cmd

    def test_detects_python_from_setup_py(self, tmp_path: Path) -> None:
        """Detect Python project from setup.py."""
        from ai_infra.executor.verifier import ProjectType, detect_project_type

        (tmp_path / "setup.py").write_text("from setuptools import setup; setup()")

        project_type = detect_project_type(tmp_path)
        assert project_type == ProjectType.PYTHON

    def test_detects_python_from_requirements_txt(self, tmp_path: Path) -> None:
        """Detect Python project from requirements.txt."""
        from ai_infra.executor.verifier import ProjectType, detect_project_type

        (tmp_path / "requirements.txt").write_text("requests>=2.0\n")

        project_type = detect_project_type(tmp_path)
        assert project_type == ProjectType.PYTHON

    @pytest.mark.asyncio
    async def test_verifier_checks_python_syntax(self, python_project: Path) -> None:
        """TaskVerifier checks Python syntax correctly."""
        from ai_infra.executor.models import Task
        from ai_infra.executor.verifier import CheckLevel, CheckStatus, TaskVerifier

        task = Task(id="1.1", title="Test Python project")
        verifier = TaskVerifier(workspace=python_project)

        result = await verifier.verify(task, levels=[CheckLevel.SYNTAX])

        # All Python files should parse correctly
        syntax_checks = [c for c in result.checks if c.level == CheckLevel.SYNTAX]
        assert all(c.status == CheckStatus.PASSED for c in syntax_checks)

    @pytest.mark.asyncio
    async def test_verifier_detects_syntax_error(self, python_project: Path) -> None:
        """TaskVerifier detects Python syntax errors."""
        from ai_infra.executor.models import Task
        from ai_infra.executor.verifier import CheckLevel, CheckStatus, TaskVerifier

        # Introduce syntax error
        bad_file = python_project / "src" / "myapp" / "broken.py"
        bad_file.write_text("def broken(\n")  # Missing closing paren

        task = Task(id="1.1", title="Test broken file")
        verifier = TaskVerifier(workspace=python_project)

        result = await verifier.verify(task, levels=[CheckLevel.SYNTAX])

        # Should have at least one failed check
        failed = [c for c in result.checks if c.status == CheckStatus.FAILED]
        assert len(failed) >= 1
        assert any("broken.py" in c.name for c in failed)

    @pytest.mark.asyncio
    async def test_verifier_checks_file_existence(self, python_project: Path) -> None:
        """TaskVerifier checks expected files exist."""
        from ai_infra.executor.models import Task
        from ai_infra.executor.verifier import CheckLevel, CheckStatus, TaskVerifier

        task = Task(
            id="1.1",
            title="Create calculator",
            file_hints=["src/myapp/calculator.py", "tests/test_calculator.py"],
        )
        verifier = TaskVerifier(workspace=python_project)

        result = await verifier.verify(task, levels=[CheckLevel.FILES])

        # Both files should exist
        file_checks = [c for c in result.checks if c.level == CheckLevel.FILES]
        assert len(file_checks) == 2
        assert all(c.status == CheckStatus.PASSED for c in file_checks)

    def test_task_needs_verification_for_function(self) -> None:
        """Task with 'function' keyword needs deep verification."""
        from ai_infra.executor.agents.verify_agent import task_needs_deep_verification
        from ai_infra.executor.models import Task

        task = Task(id="1.1", title="Add divide function to calculator")

        assert task_needs_deep_verification(task) is True

    def test_task_needs_verification_for_test(self) -> None:
        """Task with 'test' keyword needs deep verification."""
        from ai_infra.executor.agents.verify_agent import task_needs_deep_verification
        from ai_infra.executor.models import Task

        task = Task(id="1.1", title="Add unit tests for calculator")

        assert task_needs_deep_verification(task) is True


# =============================================================================
# 5.3.2 Scenario: Node.js Project (npm test discovery)
# =============================================================================


class TestNodejsProjectScenario:
    """Scenario tests for Node.js project with npm test (5.3.2)."""

    def test_detects_nodejs_project_type(self, nodejs_project: Path) -> None:
        """Detect Node.js project from package.json."""
        from ai_infra.executor.verifier import ProjectType, detect_project_type

        project_type = detect_project_type(nodejs_project)

        assert project_type == ProjectType.NODEJS

    def test_gets_npm_test_command(self, nodejs_project: Path) -> None:
        """Get npm test command for Node.js project."""
        from ai_infra.executor.verifier import ProjectType, get_test_command

        cmd = get_test_command(ProjectType.NODEJS, nodejs_project)

        assert cmd is not None
        assert cmd == ["npm", "test"]

    def test_detects_typescript_project(self, tmp_path: Path) -> None:
        """Detect TypeScript project from package.json + tsconfig.json."""
        from ai_infra.executor.verifier import ProjectType, detect_project_type

        (tmp_path / "package.json").write_text('{"name": "app"}')
        (tmp_path / "tsconfig.json").write_text('{"compilerOptions": {}}')

        project_type = detect_project_type(tmp_path)
        assert project_type == ProjectType.TYPESCRIPT

    def test_gets_npm_test_for_typescript(self, tmp_path: Path) -> None:
        """Get npm test command for TypeScript project."""
        from ai_infra.executor.verifier import ProjectType, get_test_command

        cmd = get_test_command(ProjectType.TYPESCRIPT, tmp_path)

        assert cmd is not None
        assert cmd == ["npm", "test"]

    def test_nodejs_without_test_script(self, tmp_path: Path) -> None:
        """Node.js project without test script uses node --test."""
        from ai_infra.executor.verifier import ProjectType, get_test_command

        (tmp_path / "package.json").write_text('{"name": "app"}')
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test.js").write_text("console.log('test')")

        cmd = get_test_command(ProjectType.NODEJS, tmp_path)

        assert cmd is not None
        assert cmd[0] == "node"
        assert "--test" in cmd

    @pytest.mark.asyncio
    async def test_verifier_checks_js_files_exist(self, nodejs_project: Path) -> None:
        """TaskVerifier checks JavaScript files exist."""
        from ai_infra.executor.models import Task
        from ai_infra.executor.verifier import CheckLevel, CheckStatus, TaskVerifier

        task = Task(
            id="1.1",
            title="Create calculator",
            file_hints=["src/calculator.js", "tests/calculator.test.js"],
        )
        verifier = TaskVerifier(workspace=nodejs_project)

        result = await verifier.verify(task, levels=[CheckLevel.FILES])

        file_checks = [c for c in result.checks if c.level == CheckLevel.FILES]
        assert len(file_checks) == 2
        assert all(c.status == CheckStatus.PASSED for c in file_checks)


# =============================================================================
# 5.3.3 Scenario: Rust Project (cargo test discovery)
# =============================================================================


class TestRustProjectScenario:
    """Scenario tests for Rust project with cargo test (5.3.3)."""

    def test_detects_rust_project_type(self, rust_project: Path) -> None:
        """Detect Rust project from Cargo.toml."""
        from ai_infra.executor.verifier import ProjectType, detect_project_type

        project_type = detect_project_type(rust_project)

        assert project_type == ProjectType.RUST

    def test_gets_cargo_test_command(self, rust_project: Path) -> None:
        """Get cargo test command for Rust project."""
        from ai_infra.executor.verifier import ProjectType, get_test_command

        cmd = get_test_command(ProjectType.RUST, rust_project)

        assert cmd is not None
        assert cmd == ["cargo", "test"]

    @pytest.mark.asyncio
    async def test_verifier_checks_rust_files_exist(self, rust_project: Path) -> None:
        """TaskVerifier checks Rust files exist."""
        from ai_infra.executor.models import Task
        from ai_infra.executor.verifier import CheckLevel, CheckStatus, TaskVerifier

        task = Task(
            id="1.1",
            title="Create calculator",
            file_hints=["src/lib.rs", "Cargo.toml"],
        )
        verifier = TaskVerifier(workspace=rust_project)

        result = await verifier.verify(task, levels=[CheckLevel.FILES])

        file_checks = [c for c in result.checks if c.level == CheckLevel.FILES]
        assert len(file_checks) == 2
        assert all(c.status == CheckStatus.PASSED for c in file_checks)

    def test_rust_project_readme_exists(self, rust_project: Path) -> None:
        """Rust project has README.md with cargo test instructions."""
        readme = rust_project / "README.md"
        assert readme.exists()

        content = readme.read_text()
        assert "cargo test" in content


# =============================================================================
# 5.3.4 Scenario: Makefile Project (make test discovery)
# =============================================================================


class TestMakefileProjectScenario:
    """Scenario tests for Makefile project with make test (5.3.4)."""

    def test_detects_makefile_project_type(self, makefile_project: Path) -> None:
        """Detect Makefile project from Makefile with test target."""
        from ai_infra.executor.verifier import ProjectType, detect_project_type

        project_type = detect_project_type(makefile_project)

        assert project_type == ProjectType.MAKEFILE

    def test_gets_make_test_command(self, makefile_project: Path) -> None:
        """Get make test command for Makefile project."""
        from ai_infra.executor.verifier import ProjectType, get_test_command

        cmd = get_test_command(ProjectType.MAKEFILE, makefile_project)

        assert cmd is not None
        assert cmd == ["make", "test"]

    def test_makefile_without_test_target_not_detected(self, tmp_path: Path) -> None:
        """Makefile without test target is not detected as MAKEFILE type."""
        from ai_infra.executor.verifier import ProjectType, detect_project_type

        (tmp_path / "Makefile").write_text(
            """\
.PHONY: build clean

build:
\t@echo "Building..."

clean:
\t@echo "Cleaning..."
"""
        )

        project_type = detect_project_type(tmp_path)
        assert project_type == ProjectType.UNKNOWN

    @pytest.mark.asyncio
    async def test_verifier_checks_makefile_exists(self, makefile_project: Path) -> None:
        """TaskVerifier checks Makefile and scripts exist."""
        from ai_infra.executor.models import Task
        from ai_infra.executor.verifier import CheckLevel, CheckStatus, TaskVerifier

        task = Task(
            id="1.1",
            title="Setup project",
            file_hints=["Makefile", "src/main.sh"],
        )
        verifier = TaskVerifier(workspace=makefile_project)

        result = await verifier.verify(task, levels=[CheckLevel.FILES])

        file_checks = [c for c in result.checks if c.level == CheckLevel.FILES]
        assert len(file_checks) == 2
        assert all(c.status == CheckStatus.PASSED for c in file_checks)

    @pytest.mark.asyncio
    async def test_shell_session_can_run_make(self, makefile_project: Path) -> None:
        """Shell session can execute make commands."""
        from ai_infra.llm.shell.session import SessionConfig, ShellSession

        config = SessionConfig(workspace_root=makefile_project)

        async with ShellSession(config) as session:
            # Run make test
            result = await session.execute("make test")

            assert result.success
            assert result.exit_code == 0
            assert "Tests passed" in result.stdout or "All tests passed" in result.stdout


# =============================================================================
# 5.3.5 Scenario: Go Project (go test discovery)
# =============================================================================


class TestGoProjectScenario:
    """Scenario tests for Go project with go test (5.3.5 - added for completeness)."""

    def test_detects_go_project_type(self, go_project: Path) -> None:
        """Detect Go project from go.mod."""
        from ai_infra.executor.verifier import ProjectType, detect_project_type

        project_type = detect_project_type(go_project)

        assert project_type == ProjectType.GO

    def test_gets_go_test_command(self, go_project: Path) -> None:
        """Get go test command for Go project."""
        from ai_infra.executor.verifier import ProjectType, get_test_command

        cmd = get_test_command(ProjectType.GO, go_project)

        assert cmd is not None
        assert cmd == ["go", "test", "./..."]

    @pytest.mark.asyncio
    async def test_verifier_checks_go_files_exist(self, go_project: Path) -> None:
        """TaskVerifier checks Go files exist."""
        from ai_infra.executor.models import Task
        from ai_infra.executor.verifier import CheckLevel, CheckStatus, TaskVerifier

        task = Task(
            id="1.1",
            title="Create calculator",
            file_hints=["calculator.go", "calculator_test.go"],
        )
        verifier = TaskVerifier(workspace=go_project)

        result = await verifier.verify(task, levels=[CheckLevel.FILES])

        file_checks = [c for c in result.checks if c.level == CheckLevel.FILES]
        assert len(file_checks) == 2
        assert all(c.status == CheckStatus.PASSED for c in file_checks)


# =============================================================================
# 5.3.5 Scenario: Multi-language Monorepo
# =============================================================================


class TestMonorepoScenario:
    """Scenario tests for multi-language monorepo (5.3.5)."""

    def test_root_detects_makefile_type(self, monorepo_project: Path) -> None:
        """Monorepo root is detected as Makefile project (has test target)."""
        from ai_infra.executor.verifier import ProjectType, detect_project_type

        project_type = detect_project_type(monorepo_project)

        # Root has Makefile with test target
        assert project_type == ProjectType.MAKEFILE

    def test_api_service_detects_as_python(self, monorepo_project: Path) -> None:
        """API service subdirectory is detected as Python."""
        from ai_infra.executor.verifier import ProjectType, detect_project_type

        api_path = monorepo_project / "services" / "api"
        project_type = detect_project_type(api_path)

        assert project_type == ProjectType.PYTHON

    def test_web_service_detects_as_nodejs(self, monorepo_project: Path) -> None:
        """Web service subdirectory is detected as Node.js."""
        from ai_infra.executor.verifier import ProjectType, detect_project_type

        web_path = monorepo_project / "services" / "web"
        project_type = detect_project_type(web_path)

        assert project_type == ProjectType.NODEJS

    def test_worker_service_detects_as_go(self, monorepo_project: Path) -> None:
        """Worker service subdirectory is detected as Go."""
        from ai_infra.executor.verifier import ProjectType, detect_project_type

        worker_path = monorepo_project / "services" / "worker"
        project_type = detect_project_type(worker_path)

        assert project_type == ProjectType.GO

    @pytest.mark.asyncio
    async def test_verifier_checks_monorepo_structure(self, monorepo_project: Path) -> None:
        """TaskVerifier checks monorepo file structure."""
        from ai_infra.executor.models import Task
        from ai_infra.executor.verifier import CheckLevel, CheckStatus, TaskVerifier

        task = Task(
            id="1.1",
            title="Setup monorepo",
            file_hints=[
                "Makefile",
                "services/api/main.py",
                "services/web/app.js",
                "services/worker/worker.go",
            ],
        )
        verifier = TaskVerifier(workspace=monorepo_project)

        result = await verifier.verify(task, levels=[CheckLevel.FILES])

        file_checks = [c for c in result.checks if c.level == CheckLevel.FILES]
        assert len(file_checks) == 4
        assert all(c.status == CheckStatus.PASSED for c in file_checks)

    @pytest.mark.asyncio
    async def test_verifier_checks_python_syntax_in_subdir(self, monorepo_project: Path) -> None:
        """TaskVerifier can check Python syntax in subdirectory."""
        from ai_infra.executor.models import Task
        from ai_infra.executor.verifier import CheckLevel, CheckStatus, TaskVerifier

        api_path = monorepo_project / "services" / "api"
        task = Task(id="1.1", title="Check API")
        verifier = TaskVerifier(workspace=api_path)

        result = await verifier.verify(task, levels=[CheckLevel.SYNTAX])

        syntax_checks = [c for c in result.checks if c.level == CheckLevel.SYNTAX]
        assert len(syntax_checks) >= 1
        assert all(c.status == CheckStatus.PASSED for c in syntax_checks)

    def test_docs_only_change_detection(self) -> None:
        """Docs-only changes skip verification."""
        from ai_infra.executor.agents.verify_agent import is_docs_only_change

        # All docs
        assert is_docs_only_change(["README.md", "docs/guide.md"]) is True
        assert is_docs_only_change(["CHANGELOG.md"]) is True

        # Mixed (not docs-only)
        assert is_docs_only_change(["README.md", "services/api/main.py"]) is False
        assert is_docs_only_change(["services/web/app.js"]) is False

    @pytest.mark.asyncio
    async def test_shell_session_navigates_monorepo(self, monorepo_project: Path) -> None:
        """Shell session can navigate between monorepo services."""
        from ai_infra.llm.shell.session import SessionConfig, ShellSession

        config = SessionConfig(workspace_root=monorepo_project)

        async with ShellSession(config) as session:
            # Navigate to api service
            await session.execute("cd services/api")
            result = await session.execute("ls -la")
            assert "main.py" in result.stdout

            # Navigate to web service
            await session.execute("cd ../web")
            result = await session.execute("ls -la")
            assert "package.json" in result.stdout

            # Navigate to worker service
            await session.execute("cd ../worker")
            result = await session.execute("ls -la")
            assert "go.mod" in result.stdout


# =============================================================================
# Additional Verification Heuristics Tests
# =============================================================================


class TestVerificationHeuristics:
    """Tests for verification agent heuristics across scenarios."""

    def test_api_task_needs_verification(self) -> None:
        """Task modifying API needs deep verification."""
        from ai_infra.executor.agents.verify_agent import task_needs_deep_verification
        from ai_infra.executor.models import Task

        task = Task(id="1.1", title="Add REST API endpoint for users")
        assert task_needs_deep_verification(task) is True

    def test_endpoint_task_needs_verification(self) -> None:
        """Task creating endpoint needs deep verification."""
        from ai_infra.executor.agents.verify_agent import task_needs_deep_verification
        from ai_infra.executor.models import Task

        task = Task(id="1.1", title="Create authentication endpoint")
        assert task_needs_deep_verification(task) is True

    def test_database_task_needs_verification(self) -> None:
        """Task modifying database needs deep verification."""
        from ai_infra.executor.agents.verify_agent import task_needs_deep_verification
        from ai_infra.executor.models import Task

        task = Task(id="1.1", title="Add database migration for user table")
        assert task_needs_deep_verification(task) is True

    def test_readme_task_skips_verification(self) -> None:
        """Task updating README does not need deep verification."""
        from ai_infra.executor.agents.verify_agent import task_needs_deep_verification
        from ai_infra.executor.models import Task

        task = Task(id="1.1", title="Update README with installation instructions")
        assert task_needs_deep_verification(task) is False

    def test_docs_task_skips_verification(self) -> None:
        """Task updating docs does not need deep verification."""
        from ai_infra.executor.agents.verify_agent import task_needs_deep_verification
        from ai_infra.executor.models import Task

        task = Task(id="1.1", title="Add changelog entry")
        assert task_needs_deep_verification(task) is False

    def test_unknown_project_returns_none_command(self, tmp_path: Path) -> None:
        """Unknown project type returns None for test command."""
        from ai_infra.executor.verifier import ProjectType, get_test_command

        cmd = get_test_command(ProjectType.UNKNOWN, tmp_path)
        assert cmd is None

    def test_empty_files_not_docs_only(self) -> None:
        """Empty file list is not docs-only."""
        from ai_infra.executor.agents.verify_agent import is_docs_only_change

        assert is_docs_only_change([]) is False
