# Autonomous Verification

> How AI agents verify task completion through shell-based testing and validation.

## Overview

The Executor uses autonomous verification to ensure tasks are genuinely complete. This guide explains the verification system, how it detects project types, and how agents validate their work.

---

## Quick Start

```python
from ai_infra import Executor

executor = Executor(
    roadmap_path="ROADMAP.md",
    enable_shell=True,
    enable_verification=True,
)

await executor.arun()
# Tasks are automatically verified after implementation
```

---

## How Verification Works

### 1. Task Implementation

The agent implements a task from the roadmap:

```markdown
## Phase 1: Core Features
- [ ] **1.1** Implement user authentication
```

### 2. Verification Trigger

After implementation, the verification agent checks if the task is complete:

```python
# Internal flow
task = "Implement user authentication"
implementation_result = await agent.implement(task)

# Verification is triggered
verification = await verifier.verify(task, implementation_result)
```

### 3. Project-Aware Testing

The verifier detects the project type and runs appropriate tests:

```python
# Python project detected
await run_shell.ainvoke({"command": "pytest tests/test_auth.py -v"})

# Node.js project
await run_shell.ainvoke({"command": "npm test -- --grep 'auth'"})
```

### 4. Verification Result

```python
@dataclass
class VerificationResult:
    success: bool           # Overall verification passed
    tests_passed: int       # Number of tests that passed
    tests_failed: int       # Number of tests that failed
    coverage: float | None  # Code coverage percentage
    issues: list[str]       # List of issues found
    suggestions: list[str]  # Suggestions for improvement
```

---

## Verification Agent

The `VerificationAgent` is a specialized agent for task verification:

```python
from ai_infra.llm.shell import VerificationAgent

verifier = VerificationAgent(
    workspace_root="/project",
    test_command="pytest -v",
    coverage_threshold=80.0,
)

result = await verifier.verify(
    task="Implement user authentication",
    implementation_result="Created auth.py with login/logout functions",
)

if result.success:
    print("Task verified successfully!")
else:
    print(f"Verification failed: {result.issues}")
```

### Configuration

```python
verifier = VerificationAgent(
    # Working directory
    workspace_root="/project",

    # Default test command (can be auto-detected)
    test_command="pytest -v",

    # Minimum coverage required
    coverage_threshold=80.0,

    # Maximum verification attempts
    max_attempts=3,

    # Timeout for test execution
    timeout=300.0,
)
```

---

## Project Type Detection

The verifier automatically detects project types and chooses appropriate test strategies:

### Detection Logic

```python
from ai_infra.llm.shell import detect_project_type

project_type = await detect_project_type("/path/to/project")
# Returns: "python", "nodejs", "rust", "go", "makefile", "unknown"
```

### Detection Rules

| Indicator | Project Type | Test Command |
|-----------|--------------|--------------|
| `pyproject.toml`, `setup.py`, `requirements.txt` | Python | `pytest` |
| `package.json` | Node.js | `npm test` |
| `Cargo.toml` | Rust | `cargo test` |
| `go.mod` | Go | `go test ./...` |
| `Makefile` with `test` target | Makefile | `make test` |

### Customizing Detection

```python
from ai_infra.llm.shell import ProjectDetector

detector = ProjectDetector(
    custom_rules=[
        {
            "indicator": "build.gradle",
            "type": "java",
            "test_command": "./gradlew test",
        },
        {
            "indicator": "mix.exs",
            "type": "elixir",
            "test_command": "mix test",
        },
    ]
)

project_type = await detector.detect("/path/to/project")
```

---

## Task Verifier

The `TaskVerifier` provides high-level verification orchestration:

```python
from ai_infra.llm.shell import TaskVerifier

verifier = TaskVerifier(
    workspace_root="/project",
    verification_depth="deep",  # "shallow", "normal", "deep"
)

result = await verifier.verify_task(
    task_id="1.1",
    task_description="Implement user authentication",
    files_changed=["src/auth.py", "tests/test_auth.py"],
)
```

### Verification Depths

| Depth | Checks Performed |
|-------|------------------|
| `shallow` | Syntax check, import verification |
| `normal` | Unit tests, basic integration |
| `deep` | Full test suite, coverage, linting |

```python
# Deep verification example
verifier = TaskVerifier(
    workspace_root="/project",
    verification_depth="deep",
)

# Runs:
# 1. Syntax check: python -m py_compile src/auth.py
# 2. Import check: python -c "import src.auth"
# 3. Unit tests: pytest tests/test_auth.py -v
# 4. Coverage: pytest --cov=src/auth --cov-fail-under=80
# 5. Linting: ruff check src/auth.py
# 6. Type check: mypy src/auth.py
```

---

## Verification Strategies

### Strategy 1: Test-Based Verification

```python
# Most common: run tests related to the task
await verifier.verify(
    task="Add password hashing",
    strategy="test",
    test_pattern="test_*password*.py",
)
```

### Strategy 2: Build Verification

```python
# Verify the project builds successfully
await verifier.verify(
    task="Fix compilation error",
    strategy="build",
)
# Runs: make build / npm run build / cargo build
```

### Strategy 3: Lint Verification

```python
# Verify code quality standards
await verifier.verify(
    task="Refactor utility functions",
    strategy="lint",
)
# Runs: ruff check / eslint / clippy
```

### Strategy 4: Integration Verification

```python
# Run integration tests
await verifier.verify(
    task="Implement API endpoint",
    strategy="integration",
    test_pattern="tests/integration/",
)
```

### Strategy 5: Custom Verification

```python
# Custom verification commands
await verifier.verify(
    task="Update database schema",
    strategy="custom",
    commands=[
        "alembic upgrade head",
        "python -c 'from app.models import *; print(\"OK\")'",
        "pytest tests/test_models.py -v",
    ],
)
```

---

## Integration with Executor

### Enabling Verification

```python
from ai_infra import Executor

executor = Executor(
    roadmap_path="ROADMAP.md",
    enable_shell=True,
    enable_verification=True,
    verification_depth="normal",
)
```

### CLI Options

```bash
# Enable verification (default)
ai-infra executor run --roadmap ROADMAP.md --verify

# Disable verification
ai-infra executor run --roadmap ROADMAP.md --no-verify

# Deep verification
ai-infra executor run --roadmap ROADMAP.md --verify-depth deep

# Custom coverage threshold
ai-infra executor run --roadmap ROADMAP.md --coverage-threshold 90
```

### Verification in Roadmap

Tasks can specify verification requirements:

```markdown
## Phase 1: Core Features

- [ ] **1.1** Implement user authentication
  - Verify: `pytest tests/test_auth.py --cov=auth --cov-fail-under=90`
  - Lint: `ruff check src/auth.py`
```

---

## Verification Results

### Success Criteria

```python
@dataclass
class VerificationCriteria:
    tests_must_pass: bool = True
    coverage_threshold: float = 80.0
    lint_must_pass: bool = True
    type_check_must_pass: bool = True
```

### Result Handling

```python
result = await verifier.verify(task)

if result.success:
    # Mark task complete in roadmap
    await executor.mark_complete(task_id)
else:
    # Attempt to fix issues
    for issue in result.issues:
        fix = await agent.fix_issue(issue)

    # Re-verify
    result = await verifier.verify(task)
```

### Retry Logic

```python
verifier = TaskVerifier(
    workspace_root="/project",
    max_retries=3,
    retry_delay=5.0,
)

# Verifier will retry up to 3 times on failure
result = await verifier.verify_with_retry(task)
```

---

## Monorepo Support

For monorepos with multiple projects:

```python
from ai_infra.llm.shell import MonorepoVerifier

verifier = MonorepoVerifier(
    workspace_root="/monorepo",
    packages={
        "frontend": {"type": "nodejs", "test": "npm test"},
        "backend": {"type": "python", "test": "pytest"},
        "shared": {"type": "rust", "test": "cargo test"},
    },
)

# Detect which package a task affects
result = await verifier.verify(
    task="Fix shared utility function",
    files_changed=["packages/shared/src/utils.rs"],
)
# Runs: cargo test in packages/shared/
```

---

## Custom Verification Hooks

### Pre-Verification Hook

```python
async def pre_verify(task: str, files: list[str]) -> None:
    """Called before verification starts."""
    print(f"Starting verification for: {task}")
    # Set up test fixtures, databases, etc.

verifier = TaskVerifier(
    workspace_root="/project",
    pre_verify_hook=pre_verify,
)
```

### Post-Verification Hook

```python
async def post_verify(result: VerificationResult) -> None:
    """Called after verification completes."""
    if result.success:
        await notify_success(result)
    else:
        await notify_failure(result)

verifier = TaskVerifier(
    workspace_root="/project",
    post_verify_hook=post_verify,
)
```

---

## Best Practices

### 1. Write Tests First

```markdown
## Phase 1
- [ ] **1.1** Write tests for user authentication
- [ ] **1.2** Implement user authentication
  - Verify: `pytest tests/test_auth.py -v`
```

### 2. Use Appropriate Depth

```python
# Development: fast verification
verifier = TaskVerifier(verification_depth="shallow")

# CI/Production: thorough verification
verifier = TaskVerifier(verification_depth="deep")
```

### 3. Set Realistic Coverage Thresholds

```python
# New code: high coverage
verifier = TaskVerifier(coverage_threshold=90.0)

# Legacy code: lower threshold
verifier = TaskVerifier(coverage_threshold=60.0)
```

### 4. Isolate Test Environments

```python
verifier = TaskVerifier(
    workspace_root="/project",
    pre_verify_hook=setup_test_database,
    post_verify_hook=cleanup_test_database,
)
```

---

## Debugging Verification

### Enable Verbose Output

```python
import logging
logging.getLogger("ai_infra.llm.shell.verifier").setLevel(logging.DEBUG)
```

### Inspect Verification Details

```python
result = await verifier.verify(task)

print(f"Commands executed: {result.commands}")
print(f"Test output: {result.test_output}")
print(f"Coverage report: {result.coverage_report}")
```

### Manual Verification

```python
# Run verification commands manually
from ai_infra.llm.shell import run_shell

result = await run_shell.ainvoke({
    "command": "pytest tests/ -v --tb=long",
    "cwd": "/project"
})
print(result["stdout"])
```

---

## See Also

- [Shell Tool API](../tools/shell-tool.md) — Command execution
- [ShellMiddleware](../tools/shell-middleware.md) — Session management
- [Using Shell Tool in Agents](../guides/shell-tool-guide.md) — Integration guide
- [Executor Architecture](graph-architecture.md) — Executor design
